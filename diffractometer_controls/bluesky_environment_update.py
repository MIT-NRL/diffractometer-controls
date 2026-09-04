"""Safe, GUI-driven updater for the beamline Bluesky environment.

This module intentionally uses only the Python standard library.  The GUI
launches it with the system Python so that it does not depend on the Conda
environment while that environment is being updated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import selectors
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - the updater is deliberately Linux-only
    fcntl = None

try:
    from diffractometer_controls.local_maintenance import (
        LOCAL_MAINTENANCE_DISABLED_MESSAGE,
        local_maintenance_allowed,
    )
except ImportError:
    from local_maintenance import (
        LOCAL_MAINTENANCE_DISABLED_MESSAGE,
        local_maintenance_allowed,
    )


ENVIRONMENT_NAME = "bluesky-server"
DEFAULT_MAMBA = Path("/home/mitr_4dh4/mambaforge/bin/mamba")
DEFAULT_ENV_PREFIX = Path("/home/mitr_4dh4/mambaforge/envs/bluesky-server")
DEFAULT_TILED_ENV = Path("/home/mitr_4dh4/.config/tiled/tiled-server.env")
DEFAULT_CONTROL_ENV = Path("/home/mitr_4dh4/.config/diffractometer-controls/control.env")
DEFAULT_QSERVER_CLIENT_ENV = Path(
    "/home/mitr_4dh4/.config/bluesky-queueserver/client-zmq.env"
)
SERVICE_NAMES = ("queue-server", "tiled-server", "bluesky-proxy")
PLAN_TIMEOUT = 30 * 60
PACKAGE_TIMEOUT = 60 * 60
DATABASE_TIMEOUT = 30 * 60
HEARTBEAT_INTERVAL = 10.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_simple_env(path: Path) -> dict[str, str]:
    """Read the simple KEY=VALUE files used by the beamline services."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def extract_json_value(text: str, expected_type: type = dict):
    decoder = json.JSONDecoder()
    opening = "{" if expected_type is dict else "["
    for match in re.finditer(re.escape(opening), text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, expected_type):
            return value
    kind = "object" if expected_type is dict else "array"
    raise ValueError(f"Command output did not contain a JSON {kind}.")


def extract_json_document(text: str) -> dict:
    return extract_json_value(text, dict)


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def explicit_snapshot_urls(path: Path) -> list[str]:
    """Return package URLs from Conda or Mamba explicit-list output."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as ex:
        raise UpdateFailure(f"Could not read the Conda restore snapshot: {path}") from ex
    stripped = [line.strip() for line in lines if line.strip()]
    has_conda_marker = "@EXPLICIT" in stripped
    has_mamba_header = bool(
        stripped and stripped[0].startswith("List of packages in environment:")
    )
    urls = [line for line in stripped if line.startswith(("https://", "http://"))]
    allowed = {
        line
        for line in stripped
        if line.startswith(("#", "@EXPLICIT", "List of packages in environment:"))
        or line.startswith(("https://", "http://"))
    }
    if not (has_conda_marker or has_mamba_header) or len(allowed) != len(stripped):
        raise UpdateFailure(f"The Conda restore snapshot is not an explicit file: {path}")
    if not urls:
        raise UpdateFailure(f"The Conda restore snapshot contains no packages: {path}")
    return urls


def explicit_snapshot_artifacts(path: Path) -> set[str]:
    """Return decoded artifact filenames from a Conda explicit snapshot."""
    artifacts = {
        urllib.parse.unquote(urllib.parse.urlparse(line).path.rsplit("/", 1)[-1])
        for line in explicit_snapshot_urls(path)
    }
    return artifacts


def installed_conda_artifacts(prefix: Path) -> set[str]:
    """Return artifact filenames recorded in an installed Conda prefix."""
    artifacts: set[str] = set()
    for record_path in (prefix / "conda-meta").glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            raise UpdateFailure(f"Could not read Conda package record: {record_path}") from ex
        artifact = record.get("fn")
        if not artifact and record.get("url"):
            artifact = urllib.parse.unquote(
                urllib.parse.urlparse(str(record["url"])).path.rsplit("/", 1)[-1]
            )
        if not artifact:
            raise UpdateFailure(f"Conda package record has no artifact name: {record_path}")
        artifacts.add(str(artifact))
    return artifacts


def _read_distribution_identity(metadata_path: Path) -> tuple[str, str] | None:
    try:
        lines = metadata_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    values: dict[str, str] = {}
    for line in lines:
        if not line:
            break
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key in {"Name", "Version"}:
            values[key] = value.strip()
    if not values.get("Name") or not values.get("Version"):
        return None
    return values["Name"], values["Version"]


def find_duplicate_distribution_metadata(env_prefix: Path) -> tuple[list[dict], list[dict]]:
    """Find safely repairable and ambiguous duplicate ``.dist-info`` records."""
    prefix = env_prefix.resolve()
    conda_versions: dict[str, set[str]] = {}
    for record_path in (prefix / "conda-meta").glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        name = normalize_package_name(record.get("name", ""))
        version = str(record.get("version", ""))
        if name and version:
            conda_versions.setdefault(name, set()).add(version)

    python_name = (prefix / "bin" / "python").resolve().name
    active_site_packages = prefix / "lib" / python_name / "site-packages"
    if active_site_packages.is_dir():
        metadata_files = active_site_packages.glob("*.dist-info/METADATA")
    else:
        metadata_files = prefix.glob("lib/python*/site-packages/*.dist-info/METADATA")
    distributions: dict[str, list[dict]] = {}
    for metadata_file in metadata_files:
        identity = _read_distribution_identity(metadata_file)
        if identity is None:
            continue
        display_name, version = identity
        metadata_dir = metadata_file.parent.resolve()
        try:
            metadata_dir.relative_to(prefix)
        except ValueError:
            continue
        distributions.setdefault(normalize_package_name(display_name), []).append(
            {"name": display_name, "version": version, "path": str(metadata_dir)}
        )

    repairs: list[dict] = []
    ambiguous: list[dict] = []
    for normalized_name, entries in sorted(distributions.items()):
        if len(entries) < 2:
            continue
        expected_versions = conda_versions.get(normalized_name, set())
        keep = [entry for entry in entries if entry["version"] in expected_versions]
        if len(keep) == 1:
            for entry in entries:
                if entry is keep[0]:
                    continue
                repairs.append(
                    {
                        "name": keep[0]["name"],
                        "keep_version": keep[0]["version"],
                        "remove_version": entry["version"],
                        "path": entry["path"],
                    }
                )
        else:
            ambiguous.append(
                {
                    "name": entries[0]["name"],
                    "versions": sorted(entry["version"] for entry in entries),
                    "conda_versions": sorted(expected_versions),
                }
            )
    return repairs, ambiguous


def _package_artifact(item: dict | None) -> tuple | None:
    if item is None:
        return None
    return tuple(
        str(item.get(field) or "")
        for field in ("name", "version", "build_string", "build", "sha256", "md5", "fn")
    )


def mamba_plan_signature(payload: dict) -> tuple:
    """Return the meaningful package operations from a Mamba JSON plan."""
    actions = payload.get("actions") or {}
    links = {str(item.get("name")): item for item in actions.get("LINK", [])}
    unlinks = {str(item.get("name")): item for item in actions.get("UNLINK", [])}
    operations = []
    for name in sorted(set(links) | set(unlinks), key=str.casefold):
        old = _package_artifact(unlinks.get(name))
        new = _package_artifact(links.get(name))
        if old == new:
            continue
        operations.append((name, old, new))
    return tuple(operations)


def combined_plan_digest(environment_payload: dict, update_payload: dict) -> str:
    signature = {
        "project_environment": mamba_plan_signature(environment_payload),
        "update_all": mamba_plan_signature(update_payload),
    }
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clean_terminal_line(text: str) -> str:
    """Collapse terminal control sequences so progress spinners do not flood the log."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    if "\r" in text:
        text = text.rsplit("\r", 1)[-1]
    cleaned: list[str] = []
    for character in text:
        if character == "\b":
            if cleaned:
                cleaned.pop()
        elif character >= " " or character == "\t":
            cleaned.append(character)
    result = "".join(cleaned).rstrip()
    if len(result) > 4000:
        result = result[:4000] + " … [line truncated]"
    return result


def _version_order_key(version: str) -> tuple:
    """Return a practical ordering key for the version forms used by Conda packages."""
    text = str(version).strip().lower()
    release_match = re.match(r"^(?:(\d+)!)?([0-9]+(?:[._-][0-9]+)*)", text)
    if release_match is None:
        return (0, (), 0, 0, text)
    epoch = int(release_match.group(1) or 0)
    release = tuple(int(part) for part in re.split(r"[._-]", release_match.group(2)))
    release = release + (0,) * (8 - len(release))
    suffix = text[release_match.end() :].lstrip("._-+")
    qualifier_match = re.match(
        r"(?:(dev|a|alpha|b|beta|pre|preview|rc|post|rev|r)(\d*)|(.*))$",
        suffix,
    )
    qualifier = qualifier_match.group(1) if qualifier_match else None
    qualifier_number = int((qualifier_match.group(2) if qualifier_match else "") or 0)
    qualifier_rank = {
        "dev": -4,
        "a": -3,
        "alpha": -3,
        "b": -2,
        "beta": -2,
        "pre": -1,
        "preview": -1,
        "rc": -1,
        None: 0,
        "post": 1,
        "rev": 1,
        "r": 1,
    }.get(qualifier, 0)
    return (epoch, release, qualifier_rank, qualifier_number, suffix)


def _append_plan_section(lines: list[str], title: str, entries: list[str]) -> None:
    if not entries:
        return
    if lines:
        lines.append("")
    lines.append(f"{title} ({len(entries)}):")
    lines.extend(f"  {entry}" for entry in entries)


def summarize_mamba_plan(payload: dict) -> tuple[list[str], bool]:
    """Return a grouped, readable transaction and whether Conda would change."""
    actions = payload.get("actions") or {}
    links = {str(item.get("name")): item for item in actions.get("LINK", [])}
    unlinks = {str(item.get("name")): item for item in actions.get("UNLINK", [])}
    names = []
    for name in sorted(set(links) | set(unlinks), key=str.casefold):
        old = unlinks.get(name)
        new = links.get(name)
        # Libmamba may include identical unlink/link pairs in an update plan.
        # They are transaction noise, not a package change worth presenting.
        if old and new and all(
            old.get(field) == new.get(field)
            for field in ("version", "build_string", "build", "sha256", "md5")
        ):
            continue
        names.append(name)
    installs: list[str] = []
    upgrades: list[str] = []
    downgrades: list[str] = []
    rebuilds: list[str] = []
    removals: list[str] = []
    for name in names:
        old = unlinks.get(name)
        new = links.get(name)
        if old and new:
            old_version = str(old.get("version", "?"))
            new_version = str(new.get("version", "?"))
            old_build = str(old.get("build_string") or old.get("build") or "")
            new_build = str(new.get("build_string") or new.get("build") or "")
            if old_version == new_version:
                build_change = (
                    f" ({old_build} -> {new_build})"
                    if old_build != new_build
                    else " (package relink)"
                )
                rebuilds.append(f"{name} {old_version}{build_change}")
            elif _version_order_key(new_version) < _version_order_key(old_version):
                downgrades.append(f"{name} {old_version} -> {new_version}")
            else:
                upgrades.append(f"{name} {old_version} -> {new_version}")
        elif new:
            installs.append(f"{name} {new.get('version', '?')}")
        elif old:
            removals.append(f"{name} {old.get('version', '?')}")
    lines: list[str] = []
    _append_plan_section(lines, "New packages", installs)
    _append_plan_section(lines, "Upgrades", upgrades)
    _append_plan_section(lines, "Downgrades", downgrades)
    _append_plan_section(lines, "Rebuilds / unchanged version", rebuilds)
    _append_plan_section(lines, "Removals", removals)
    if not lines:
        lines.append("No Conda package changes are currently proposed.")
    return lines, bool(names)


def parse_qserver_response(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("Queue Server response did not contain a result dictionary.")
    value = ast.literal_eval(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Queue Server returned an invalid response.")
    return value


def filter_pip_freeze(freeze_text: str, pip_package_names: set[str]) -> list[str]:
    """Keep only pip-owned distributions from a complete ``pip freeze``."""
    normalized_names = {re.sub(r"[-_.]+", "-", name).lower() for name in pip_package_names}
    selected: list[str] = []
    for raw_line in freeze_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " @ " in line:
            name = line.split(" @ ", 1)[0]
        elif "==" in line:
            name = line.split("==", 1)[0]
        elif line.startswith("-e ") and "#egg=" in line:
            name = line.rsplit("#egg=", 1)[1]
        else:
            continue
        normalized = re.sub(r"[-_.]+", "-", name.strip()).lower()
        if normalized in normalized_names:
            selected.append(line)
    return selected


def parse_pip_check_issues(output: str) -> list[str]:
    """Extract dependency errors while ignoring pip's incidental warnings."""
    issues = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("WARNING:"):
            continue
        if line == "No broken requirements found.":
            continue
        issues.append(line)
    return issues


def pip_check_issue_key(issue: str) -> str:
    """Identify a pip-check issue independently of the requiring package version."""
    match = re.fullmatch(
        r"(?P<package>\S+)\s+\S+\s+has requirement\s+(?P<problem>.+)",
        issue.strip(),
    )
    if match is None:
        return issue.strip()
    return f"{match.group('package').lower()} has requirement {match.group('problem')}"


def pip_check_issue_packages(issue: str) -> tuple[str, str] | None:
    """Return the requiring and required distribution names from a pip-check issue."""
    match = re.fullmatch(
        r"(?P<package>\S+)\s+\S+\s+has requirement\s+"
        r"(?P<dependency>[A-Za-z0-9_.-]+).+",
        issue.strip(),
    )
    if match is None:
        return None
    return (
        normalize_package_name(match.group("package")),
        normalize_package_name(match.group("dependency")),
    )


def installed_conda_package_names(prefix: Path) -> set[str]:
    """Return normalized package names owned by Conda in a prefix."""
    names: set[str] = set()
    for record_path in (prefix / "conda-meta").glob("*.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as ex:
            raise UpdateFailure(f"Could not read Conda package record: {record_path}") from ex
        if record.get("name"):
            names.add(normalize_package_name(str(record["name"])))
    return names


def is_conda_managed_pip_issue(issue: str, conda_names: set[str]) -> bool:
    """Whether pip is reporting metadata solely between Conda-owned packages."""
    packages = pip_check_issue_packages(issue)
    return packages is not None and all(name in conda_names for name in packages)


def redact_command(command: list[str]) -> str:
    redacted: list[str] = []
    for value in command:
        text = str(value)
        if text.startswith("postgresql") and "://" in text:
            parsed = urllib.parse.urlsplit(text)
            host = parsed.hostname or "localhost"
            if parsed.port:
                host = f"{host}:{parsed.port}"
            username = parsed.username or "user"
            text = urllib.parse.urlunsplit(
                (parsed.scheme, f"{username}:***@{host}", parsed.path, parsed.query, parsed.fragment)
            )
        redacted.append(shlex.quote(text))
    return " ".join(redacted)


class UpdateFailure(RuntimeError):
    pass


class UpdateContext:
    def __init__(self, run_dir: Path, repo_root: Path):
        self.run_dir = run_dir.resolve()
        self.repo_root = repo_root.resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.run_dir / "update.log"
        self.state_path = self.run_dir / "status.json"
        self.plan_path = self.run_dir / "plan.json"
        self.summary_path = self.run_dir / "plan_summary.txt"
        self.manifest_path = self.run_dir / "restore_manifest.json"
        self.environment_file = self.repo_root / "environmentBluesky.yml"
        self.pip_requirements = self.repo_root / "requirementsBlueskyPip.txt"
        self.gui_script = self.repo_root / "diffractometer_controls" / "4dh4gui.pl"
        self.env_prefix = DEFAULT_ENV_PREFIX
        self.env_python = self.env_prefix / "bin" / "python"
        self.qserver = self.env_prefix / "bin" / "qserver"
        self.tiled = self.env_prefix / "bin" / "tiled"
        self._log_handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self._manifest: dict = {}
        self._state_payload: dict = {}
        self._state_message = ""

    def close(self) -> None:
        self._log_handle.close()

    def log(self, message: str = "") -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self._log_handle.write(line + "\n")
        self._log_handle.flush()

    def write_json_atomic(self, path: Path, payload: dict) -> None:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)

    def set_state(
        self,
        phase: str,
        status: str,
        message: str,
        progress: int,
        **extra,
    ) -> None:
        payload = {
            "phase": phase,
            "status": status,
            "message": message,
            "progress": max(0, min(100, int(progress))),
            "updated_at": utc_now(),
            **extra,
        }
        self._state_payload = payload
        self._state_message = message
        self.write_json_atomic(self.state_path, payload)

    def _command_heartbeat(self, elapsed: float) -> None:
        if not self._state_payload:
            return
        seconds = max(0, int(elapsed))
        minutes, remainder = divmod(seconds, 60)
        shown = f"{minutes}m {remainder:02d}s" if minutes else f"{remainder}s"
        payload = {
            **self._state_payload,
            "message": f"{self._state_message} — still running ({shown})",
            "updated_at": utc_now(),
            "command_elapsed_seconds": seconds,
        }
        self._state_payload = payload
        self.write_json_atomic(self.state_path, payload)

    def _finish_command_heartbeat(self) -> None:
        if not self._state_payload or "command_elapsed_seconds" not in self._state_payload:
            return
        payload = {
            key: value
            for key, value in self._state_payload.items()
            if key != "command_elapsed_seconds"
        }
        payload["message"] = self._state_message
        payload["updated_at"] = utc_now()
        self._state_payload = payload
        self.write_json_atomic(self.state_path, payload)

    def save_manifest(self) -> None:
        self.write_json_atomic(self.manifest_path, self._manifest)

    def command_env(self, *env_files: Path) -> dict[str, str]:
        env = dict(os.environ)
        for env_file in env_files:
            env.update(read_simple_env(env_file))
        return env

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        capture: bool = False,
        timeout: float | None = 10 * 60,
        display_command: str | None = None,
        log_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        shown = display_command or redact_command(command)
        self.log(f"$ {shown}")
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        chunks: list[bytes] = []
        live_buffer = b""
        started = time.monotonic()
        next_heartbeat = started + HEARTBEAT_INTERVAL
        timed_out = False
        try:
            while selector.get_map():
                now = time.monotonic()
                if timeout is not None and now - started >= timeout:
                    timed_out = True
                    break
                wait = min(1.0, max(0.0, next_heartbeat - now))
                if timeout is not None:
                    wait = min(wait, max(0.0, timeout - (now - started)))
                events = selector.select(wait)
                for key, _mask in events:
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    chunks.append(chunk)
                    if not capture:
                        live_buffer += chunk
                        complete = live_buffer.split(b"\n")
                        live_buffer = complete.pop()
                        for raw_line in complete:
                            self.log(_clean_terminal_line(raw_line.decode(errors="replace")))
                now = time.monotonic()
                if now >= next_heartbeat and process.poll() is None:
                    elapsed = now - started
                    self._command_heartbeat(elapsed)
                    self.log(f"Still running ({int(elapsed)} seconds): {shown}")
                    next_heartbeat = now + HEARTBEAT_INTERVAL
        finally:
            selector.close()

        if timed_out:
            self.log(f"Command timed out after {int(timeout or 0)} seconds; terminating it.")
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            process.stdout.close()
            self._finish_command_heartbeat()
            raise UpdateFailure(f"Command timed out after {int(timeout or 0)} seconds: {shown}")

        returncode = process.wait()
        process.stdout.close()
        if not capture and live_buffer:
            self.log(_clean_terminal_line(live_buffer.decode(errors="replace")))
        output = b"".join(chunks).decode(errors="replace")
        if capture and output.strip() and (log_output or returncode != 0):
            for line in output.rstrip().splitlines():
                self.log(_clean_terminal_line(line))
        self._finish_command_heartbeat()
        result = subprocess.CompletedProcess(command, returncode, output, None)
        if check and result.returncode != 0:
            raise UpdateFailure(f"Command failed with exit code {result.returncode}: {shown}")
        return result

    def qserver_env(self) -> dict[str, str]:
        return self.command_env(DEFAULT_QSERVER_CLIENT_ENV)

    def qserver_request(self, *arguments: str, check: bool = True) -> dict:
        result = self.run(
            [str(self.qserver), *arguments],
            env=self.qserver_env(),
            capture=True,
            timeout=15,
            check=check,
        )
        if result.returncode != 0:
            return {}
        return parse_qserver_response(result.stdout or "")

    def qserver_status(self, *, required: bool) -> dict:
        try:
            return self.qserver_request("status", check=True)
        except Exception:
            if required:
                raise
            return {}

    def ensure_no_active_run(self) -> dict:
        status = self.qserver_status(required=False)
        if not status:
            self.log("Queue Server is not currently reachable; continuing with service-level checks.")
            return {}
        running_uid = status.get("running_item_uid")
        manager_state = str(status.get("manager_state", "")).lower()
        re_state = str(status.get("re_state", "")).lower()
        unsafe_states = {"executing_queue", "executing_task", "paused", "running"}
        if running_uid or manager_state in unsafe_states or re_state in {"running", "paused"}:
            raise UpdateFailure(
                "A Queue Server plan or task is active. Stop or finish it before updating."
            )
        return status

    def _resolve_mamba_plan(
        self, first_progress: int, second_progress: int
    ) -> tuple[dict, dict]:
        self.set_state(
            "resolving", "running", "Checking project environment requirements", first_progress
        )
        environment_command = [
            str(DEFAULT_MAMBA),
            "env",
            "update",
            "--name",
            ENVIRONMENT_NAME,
            "--file",
            str(self.environment_file),
            "--dry-run",
            "--json",
        ]
        environment_result = self.run(
            environment_command,
            capture=True,
            timeout=PLAN_TIMEOUT,
            log_output=False,
        )
        environment_payload = extract_json_document(environment_result.stdout or "")
        if environment_payload.get("success") is False:
            raise UpdateFailure(
                str(
                    environment_payload.get("error")
                    or "Mamba could not resolve project requirements."
                )
            )

        self.set_state(
            "resolving",
            "running",
            "Resolving available updates with Mamba",
            second_progress,
        )
        update_command = [
            str(DEFAULT_MAMBA),
            "update",
            "--name",
            ENVIRONMENT_NAME,
            "--all",
            "--dry-run",
            "--json",
        ]
        update_result = self.run(
            update_command,
            capture=True,
            timeout=PLAN_TIMEOUT,
            log_output=False,
        )
        update_payload = extract_json_document(update_result.stdout or "")
        if update_payload.get("success") is False:
            raise UpdateFailure(
                str(update_payload.get("error") or "Mamba could not resolve the update.")
            )
        return environment_payload, update_payload

    def _metadata_repairs(self) -> list[dict]:
        repairs, ambiguous = find_duplicate_distribution_metadata(self.env_prefix)
        if ambiguous:
            details = "; ".join(
                f"{item['name']}: installed={','.join(item['versions'])}, "
                f"Conda={','.join(item['conda_versions']) or 'unknown'}"
                for item in ambiguous
            )
            raise UpdateFailure(
                "Duplicate Python package metadata could not be repaired safely: " + details
            )
        return repairs

    @staticmethod
    def _repair_signature(repairs: list[dict]) -> list[tuple[str, str, str]]:
        return sorted(
            (
                normalize_package_name(item.get("name", "")),
                str(item.get("keep_version", "")),
                str(item.get("remove_version", "")),
            )
            for item in repairs
        )

    def _revalidate_approved_plan(self) -> list[dict]:
        if not self.plan_path.exists():
            raise UpdateFailure("The approved update plan is missing. Run the preview again.")
        try:
            approved = json.loads(self.plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            raise UpdateFailure("The approved update plan is unreadable. Run the preview again.") from ex
        approved_digest = str(approved.get("plan_digest", ""))
        if not approved_digest:
            raise UpdateFailure("The approved update plan is outdated. Run the preview again.")

        current_environment, current_update = self._resolve_mamba_plan(5, 8)
        current_digest = combined_plan_digest(current_environment, current_update)
        if current_digest != approved_digest:
            raise UpdateFailure(
                "Available packages changed after the preview. No changes were made; "
                "review and approve a new update plan."
            )
        current_repairs = self._metadata_repairs()
        approved_repairs = list(approved.get("metadata_repairs") or [])
        if self._repair_signature(current_repairs) != self._repair_signature(approved_repairs):
            raise UpdateFailure(
                "Installed package metadata changed after the preview. No changes were made; "
                "review and approve a new update plan."
            )
        self.log(f"Approved package plan revalidated ({current_digest[:12]}).")
        return current_repairs

    def check(self) -> None:
        self.set_state("preflight", "running", "Checking Queue Server state", 5)
        self.log("Starting Bluesky environment update preview.")
        self.ensure_no_active_run()
        if not self.environment_file.exists():
            raise UpdateFailure(f"Environment file is missing: {self.environment_file}")
        if not DEFAULT_MAMBA.exists():
            raise UpdateFailure(f"Mamba executable is missing: {DEFAULT_MAMBA}")
        environment_payload, update_payload = self._resolve_mamba_plan(20, 45)
        metadata_repairs = self._metadata_repairs()
        plan_digest = combined_plan_digest(environment_payload, update_payload)
        self.write_json_atomic(
            self.plan_path,
            {
                "project_environment": environment_payload,
                "update_all": update_payload,
                "metadata_repairs": metadata_repairs,
                "plan_digest": plan_digest,
            },
        )
        environment_summary, has_environment_changes = summarize_mamba_plan(
            environment_payload
        )
        update_summary, has_update_changes = summarize_mamba_plan(update_payload)
        summary = [
            "Project environment requirements:",
            *[f"  {line}" for line in environment_summary],
            "",
            "Available Mamba package updates:",
            *[f"  {line}" for line in update_summary],
        ]
        has_conda_changes = has_environment_changes or has_update_changes
        if metadata_repairs:
            summary.extend(
                [
                    "",
                    f"Safe metadata repairs ({len(metadata_repairs)}):",
                    *[
                        f"  Remove stale {item['name']} {item['remove_version']} metadata; "
                        f"keep Conda version {item['keep_version']}"
                        for item in metadata_repairs
                    ],
                ]
            )
        if self.pip_requirements.exists():
            summary.extend(
                [
                    "",
                    "Pip-only packages will be checked after Mamba and installed with --no-deps:",
                    *[
                        f"  {line.strip()}"
                        for line in self.pip_requirements.read_text(encoding="utf-8").splitlines()
                        if line.strip() and not line.lstrip().startswith("#")
                    ],
                ]
            )
        self.summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
        for line in summary:
            self.log(line)
        self.set_state(
            "ready",
            "ready",
            "Update preview is ready for confirmation",
            100,
            has_conda_changes=has_conda_changes,
            has_metadata_repairs=bool(metadata_repairs),
            plan_digest=plan_digest,
            summary_file=str(self.summary_path),
        )

    def _record_restore_point(self) -> None:
        self.set_state("snapshot", "running", "Recording environment restore point", 12)
        revisions = self.run(
            [str(DEFAULT_MAMBA), "list", "--name", ENVIRONMENT_NAME, "--revisions"],
            capture=True,
            timeout=PLAN_TIMEOUT,
        ).stdout or ""
        (self.run_dir / "conda-revisions.txt").write_text(revisions, encoding="utf-8")
        revision_matches = re.findall(r"\(rev\s+(\d+)\)", revisions)
        revision = int(revision_matches[-1]) if revision_matches else None

        explicit = self.run(
            [str(DEFAULT_MAMBA), "list", "--name", ENVIRONMENT_NAME, "--explicit"],
            capture=True,
            timeout=PLAN_TIMEOUT,
            log_output=False,
        ).stdout or ""
        explicit_urls = [
            line.strip()
            for line in explicit.splitlines()
            if line.strip().startswith(("https://", "http://"))
        ]
        if not explicit_urls:
            raise UpdateFailure("Mamba did not return an explicit environment snapshot.")
        (self.run_dir / "conda-explicit.txt").write_text(
            "# Generated by the Bluesky environment updater\n@EXPLICIT\n"
            + "\n".join(explicit_urls)
            + "\n",
            encoding="utf-8",
        )
        conda_packages_text = self.run(
            [str(DEFAULT_MAMBA), "list", "--name", ENVIRONMENT_NAME, "--json"],
            capture=True,
            timeout=PLAN_TIMEOUT,
            log_output=False,
        ).stdout or "[]"
        conda_packages = extract_json_value(conda_packages_text, list)
        pip_package_names = {
            str(item.get("name", ""))
            for item in conda_packages
            if str(item.get("channel", "")).lower() == "pypi"
        }
        pip_freeze_all = self.run(
            [str(self.env_python), "-m", "pip", "freeze", "--all"],
            capture=True,
            timeout=PLAN_TIMEOUT,
            log_output=False,
        ).stdout or ""
        pip_freeze = filter_pip_freeze(pip_freeze_all, pip_package_names)
        (self.run_dir / "pip-freeze.txt").write_text(
            "\n".join(pip_freeze) + ("\n" if pip_freeze else ""), encoding="utf-8"
        )
        pip_check = self.run(
            [str(self.env_python), "-m", "pip", "check"],
            capture=True,
            timeout=PLAN_TIMEOUT,
            check=False,
        )
        pip_check_issues = parse_pip_check_issues(pip_check.stdout or "")
        self._manifest.update(
            {
                "created_at": utc_now(),
                "environment_name": ENVIRONMENT_NAME,
                "conda_revision": revision,
                "conda_explicit": str(self.run_dir / "conda-explicit.txt"),
                "pip_freeze": str(self.run_dir / "pip-freeze.txt"),
                "pip_check_issues": pip_check_issues,
                "database_backups": {},
                "metadata_backups": [],
                "migrated_databases": [],
                "databases_requiring_restore": [],
            }
        )
        self.save_manifest()

    def _repair_duplicate_metadata(self, repairs: list[dict]) -> None:
        if not repairs:
            self.log("No duplicate Python distribution metadata requires repair.")
            return
        self.set_state("repair", "running", "Repairing duplicate package metadata", 35)
        backup_root = self.run_dir / "metadata-backups"
        backup_root.mkdir(parents=True, exist_ok=True)
        prefix = self.env_prefix.resolve()
        for index, repair in enumerate(repairs, start=1):
            metadata_dir = Path(str(repair.get("path", ""))).resolve()
            try:
                metadata_dir.relative_to(prefix)
            except ValueError as ex:
                raise UpdateFailure(
                    f"Refusing metadata repair outside the environment: {metadata_dir}"
                ) from ex
            if (
                metadata_dir.suffix != ".dist-info"
                or metadata_dir.is_symlink()
                or not metadata_dir.is_dir()
            ):
                raise UpdateFailure(f"Unsafe or missing metadata repair target: {metadata_dir}")
            identity = _read_distribution_identity(metadata_dir / "METADATA")
            expected_name = normalize_package_name(repair.get("name", ""))
            expected_version = str(repair.get("remove_version"))
            if (
                identity is None
                or normalize_package_name(identity[0]) != expected_name
                or identity[1] != expected_version
            ):
                raise UpdateFailure(
                    f"Metadata changed before repair: expected {repair['name']} "
                    f"{expected_version}, found {identity}."
                )
            backup = backup_root / f"{index:02d}-{metadata_dir.name}"
            shutil.copytree(metadata_dir, backup)
            shutil.rmtree(metadata_dir)
            record = {**repair, "backup": str(backup), "removed_at": utc_now()}
            self._manifest.setdefault("metadata_backups", []).append(record)
            self.save_manifest()
            self.log(
                f"Removed stale {repair['name']} {repair['remove_version']} metadata; "
                f"kept {repair['keep_version']} (backup: {backup})."
            )

    def _verify_pip_consistency(self) -> None:
        result = self.run(
            [str(self.env_python), "-m", "pip", "check"],
            capture=True,
            check=False,
        )
        all_current = parse_pip_check_issues(result.stdout or "")
        conda_names = installed_conda_package_names(self.env_prefix)
        conda_managed = [
            issue
            for issue in all_current
            if is_conda_managed_pip_issue(issue, conda_names)
        ]
        current = [issue for issue in all_current if issue not in conda_managed]
        baseline = {
            pip_check_issue_key(issue)
            for issue in (self._manifest.get("pip_check_issues") or [])
            if not is_conda_managed_pip_issue(issue, conda_names)
        }
        new_issues = [
            issue for issue in current if pip_check_issue_key(issue) not in baseline
        ]
        if new_issues:
            raise UpdateFailure(
                "The update introduced package dependency problems: " + "; ".join(new_issues)
            )
        if current:
            self.log("Pre-existing pip dependency issue(s) remain:")
            for issue in current:
                self.log(f"  {issue}")
        else:
            self.log("Pip-owned package dependency check passed.")
        if conda_managed:
            self.log(
                "Ignoring pip metadata warning(s) between Conda-managed packages; "
                "Mamba resolved their Conda dependencies:"
            )
            for issue in conda_managed:
                self.log(f"  {issue}")

    def _postgres_env(self) -> dict[str, str]:
        env = self.command_env(DEFAULT_TILED_ENV)
        if env.get("TILED_PGPASSWORD"):
            env["PGPASSWORD"] = env["TILED_PGPASSWORD"]
        return env

    def _database_names(self) -> list[tuple[str, str]]:
        env = self._postgres_env()
        requested = [
            ("auth", env.get("TILED_AUTH_DB", "")),
            ("catalog", env.get("TILED_PG_CATALOG_DB", "")),
            ("test_catalog", env.get("TILED_PG_TEST_CATALOG_DB", "")),
        ]
        result: list[tuple[str, str]] = []
        seen: set[str] = set()
        for role, name in requested:
            if name and name not in seen:
                result.append((role, name))
                seen.add(name)
        if not result:
            raise UpdateFailure("No Tiled databases are configured for backup.")
        return result

    def _backup_databases(self) -> None:
        self.set_state("backup", "running", "Backing up Tiled databases", 20)
        env = self._postgres_env()
        host = env.get("TILED_PGHOST", "localhost")
        port = env.get("TILED_PGPORT", "5432")
        user = env.get("TILED_PGUSER", "tiled")
        backup_dir = self.run_dir / "database-backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        pg_dump = shutil.which("pg_dump") or "/usr/bin/pg_dump"
        pg_restore = shutil.which("pg_restore") or "/usr/bin/pg_restore"
        for role, database in self._database_names():
            destination = backup_dir / f"{database}.dump"
            self.run(
                [
                    pg_dump,
                    "-h",
                    host,
                    "-p",
                    port,
                    "-U",
                    user,
                    "--format=custom",
                    "--no-owner",
                    "--no-privileges",
                    f"--file={destination}",
                    database,
                ],
                env=env,
            )
            self.run(
                [pg_restore, "--list", str(destination)],
                capture=True,
                timeout=DATABASE_TIMEOUT,
                log_output=False,
            )
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            self.log(f"Verified {database}: {destination.stat().st_size} bytes, sha256={digest}")
            self._manifest["database_backups"][database] = {
                "role": role,
                "path": str(destination),
                "sha256": digest,
            }
            self.save_manifest()

    def _close_worker(self) -> None:
        status = self.ensure_no_active_run()
        if not status or not status.get("worker_environment_exists"):
            return
        self.set_state("stopping", "running", "Closing Queue Server worker", 28)
        response = self.qserver_request("environment", "close")
        if response.get("success") is False:
            raise UpdateFailure(str(response.get("msg") or "Queue Server worker would not close."))
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            current = self.qserver_status(required=False)
            if not current or not current.get("worker_environment_exists"):
                return
            time.sleep(1)
        raise UpdateFailure("Timed out while closing the Queue Server worker.")

    def _systemctl(self, action: str, service: str, *, check: bool = True) -> None:
        self.run(
            ["systemctl", "--user", action, f"{service}.service"],
            capture=True,
            timeout=60,
            check=check,
        )

    def _stop_services(self) -> None:
        self.set_state("stopping", "running", "Stopping Bluesky services", 32)
        for service in SERVICE_NAMES:
            self._systemctl("stop", service, check=False)
        for service in SERVICE_NAMES:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                result = self.run(
                    ["systemctl", "--user", "is-active", f"{service}.service"],
                    capture=True,
                    timeout=10,
                    check=False,
                )
                states = [
                    line.strip()
                    for line in (result.stdout or "").splitlines()
                    if line.strip()
                ]
                if states and states[-1] in {"inactive", "failed", "unknown"}:
                    break
                time.sleep(1)
            else:
                raise UpdateFailure(f"Service did not stop: {service}")
        self._manifest["services_stopped"] = True
        self.save_manifest()

    def _update_environment(self) -> None:
        self.set_state("mamba", "running", "Applying project requirements with Mamba", 38)
        self.run(
            [
                str(DEFAULT_MAMBA),
                "env",
                "update",
                "--name",
                ENVIRONMENT_NAME,
                "--file",
                str(self.environment_file),
                "--yes",
            ],
            timeout=PACKAGE_TIMEOUT,
        )
        self.set_state("mamba", "running", "Updating environment with Mamba", 46)
        self.run(
            [
                str(DEFAULT_MAMBA),
                "update",
                "--name",
                ENVIRONMENT_NAME,
                "--all",
                "--yes",
            ],
            timeout=PACKAGE_TIMEOUT,
        )
        if self.pip_requirements.exists():
            self.set_state(
                "pip", "running", "Updating pip-only packages without dependencies", 54
            )
            self.run(
                [
                    str(self.env_python),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--no-deps",
                    "--requirement",
                    str(self.pip_requirements),
                ],
                timeout=PACKAGE_TIMEOUT,
            )
        self._verify_pip_consistency()

    def _smoke_test_environment(self) -> None:
        self.set_state("testing", "running", "Running Python import checks", 62)
        imports = (
            "import bluesky, bluesky_queueserver, tiled, ophyd, pydm, cv2, "
            "neutron_imaging_tools; print('Core imports OK')"
        )
        smoke_env = dict(os.environ)
        smoke_env["OPHYD_CONTROL_LAYER"] = "dummy"
        self.run([str(self.env_python), "-c", imports], env=smoke_env)
        config_path = Path("/home/mitr_4dh4/.config/tiled/profiles/config.yml")
        if config_path.exists():
            self.run([str(self.tiled), "admin", "check-config", str(config_path)])

    def _required_tiled_revisions(self) -> dict[str, str]:
        code = (
            "import json; "
            "from tiled.catalog.core import REQUIRED_REVISION as catalog; "
            "from tiled.authn_database.core import REQUIRED_REVISION as auth; "
            "print(json.dumps({'catalog': catalog, 'auth': auth}))"
        )
        output = self.run([str(self.env_python), "-c", code], capture=True).stdout or ""
        return extract_json_document(output)

    def _database_revision(self, database: str) -> str:
        env = self._postgres_env()
        result = self.run(
            [
                shutil.which("psql") or "/usr/bin/psql",
                "-h",
                env.get("TILED_PGHOST", "localhost"),
                "-p",
                env.get("TILED_PGPORT", "5432"),
                "-U",
                env.get("TILED_PGUSER", "tiled"),
                "-d",
                database,
                "-Atc",
                "select version_num from alembic_version limit 1",
            ],
            env=env,
            capture=True,
        )
        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        revisions = [line for line in lines if re.fullmatch(r"[0-9a-f]+", line)]
        if not revisions:
            raise UpdateFailure(f"Could not determine the schema revision for {database}.")
        return revisions[-1]

    def _database_uri(self, database: str) -> str:
        env = self._postgres_env()
        user = urllib.parse.quote(env.get("TILED_PGUSER", "tiled"), safe="")
        password = urllib.parse.quote(env.get("TILED_PGPASSWORD", ""), safe="")
        host = env.get("TILED_PGHOST", "localhost")
        port = env.get("TILED_PGPORT", "5432")
        db = urllib.parse.quote(database, safe="")
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db}"

    def _migrate_databases(self) -> None:
        self.set_state("migration", "running", "Checking Tiled database schemas", 70)
        required = self._required_tiled_revisions()
        for role, database in self._database_names():
            expected = required["auth" if role == "auth" else "catalog"]
            current = self._database_revision(database)
            if current == expected:
                self.log(f"{database}: schema is current ({current}).")
                continue
            if database not in self._manifest.get("database_backups", {}):
                raise UpdateFailure(f"Refusing to migrate {database}: verified backup is missing.")
            self.log(f"{database}: migrating {current} -> {expected}.")
            uri = self._database_uri(database)
            if role == "auth":
                command = [str(self.tiled), "admin", "upgrade-database", uri, expected]
            else:
                command = [str(self.tiled), "catalog", "upgrade-database", uri, expected]
            if database not in self._manifest["databases_requiring_restore"]:
                self._manifest["databases_requiring_restore"].append(database)
                self.save_manifest()
            self.run(command, display_command=redact_command(command))
            migrated = self._database_revision(database)
            if migrated != expected:
                raise UpdateFailure(
                    f"Migration of {database} ended at {migrated}, expected {expected}."
                )
            self._manifest["migrated_databases"].append(database)
            self.save_manifest()

    def _wait_service_active(self, service: str, timeout: float = 60) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.run(
                ["systemctl", "--user", "is-active", f"{service}.service"],
                capture=True,
                timeout=10,
                check=False,
            )
            output_lines = [
                line.strip() for line in (result.stdout or "").splitlines() if line.strip()
            ]
            if output_lines and output_lines[-1] == "active":
                self.log(f"{service}.service is active.")
                return
            time.sleep(1)
        raise UpdateFailure(f"Service did not become active: {service}")

    def _tiled_url(self) -> str:
        env = self.command_env(DEFAULT_CONTROL_ENV, DEFAULT_TILED_ENV)
        configured = env.get("TILED_URI") or env.get("MITR_TILED_URI")
        if configured:
            return configured.rstrip("/") + "/"
        host = env.get("MITR_CONTROL_HOST") or env.get("MITR_CONTROL_IP") or "localhost"
        port = env.get("MITR_TILED_PORT", "8000")
        return f"http://{host}:{port}/"

    def _wait_tiled_http(self, timeout: float = 60) -> None:
        url = self._tiled_url()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    if 100 <= int(response.status) < 500:
                        self.log(f"Tiled health check returned HTTP {response.status}.")
                        return
            except urllib.error.HTTPError as ex:
                if 100 <= int(ex.code) < 500:
                    self.log(f"Tiled health check returned HTTP {ex.code}.")
                    return
            except Exception:
                pass
            time.sleep(1)
        raise UpdateFailure(f"Tiled did not respond at {url}.")

    def _start_services(self) -> None:
        self.set_state("restarting", "running", "Starting Tiled", 78)
        self._systemctl("start", "tiled-server")
        self._wait_service_active("tiled-server")
        self._wait_tiled_http()

        self.set_state("restarting", "running", "Starting Bluesky proxy", 84)
        self._systemctl("start", "bluesky-proxy")
        self._wait_service_active("bluesky-proxy")

        self.set_state("restarting", "running", "Starting Queue Server", 88)
        self._systemctl("start", "queue-server")
        self._wait_service_active("queue-server")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.qserver_status(required=False):
                break
            time.sleep(1)
        else:
            raise UpdateFailure("Queue Server manager did not become reachable.")

    def _open_worker(self) -> None:
        self.set_state("worker", "running", "Opening Queue Server environment", 92)
        status = self.qserver_status(required=True)
        if not status.get("worker_environment_exists"):
            response = self.qserver_request("environment", "open")
            if response.get("success") is False:
                raise UpdateFailure(str(response.get("msg") or "Worker environment did not open."))
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            status = self.qserver_status(required=False)
            if (
                status.get("worker_environment_exists")
                and status.get("worker_environment_state") == "idle"
                and status.get("re_state") == "idle"
            ):
                self.log("Queue Server worker is open and idle.")
                return
            if (
                status.get("manager_state") == "idle"
                and status.get("worker_environment_state") == "closed"
            ):
                raise UpdateFailure("Queue Server worker closed during startup.")
            time.sleep(1)
        raise UpdateFailure("Timed out while opening the Queue Server worker.")

    def _restart_gui(self) -> None:
        if not self.gui_script.exists():
            self.log(f"GUI restart script is missing: {self.gui_script}")
            return
        # The updater itself runs in a transient systemd service.  A detached
        # screen process started directly here would remain in that service's
        # cgroup and be killed when the updater exits.  Give the GUI its own
        # scope so that its lifetime is independent of the update job.
        self.run(
            [
                "/usr/bin/systemd-run",
                "--user",
                "--scope",
                "--quiet",
                "/usr/bin/perl",
                str(self.gui_script),
                "restart",
            ],
            timeout=60,
        )

    def apply(self) -> None:
        self.log("Starting confirmed Bluesky environment update.")
        self.set_state("preflight", "running", "Running final safety checks", 3)
        self.ensure_no_active_run()
        metadata_repairs = self._revalidate_approved_plan()
        self._record_restore_point()
        self._backup_databases()
        self._manifest["services_stop_started"] = True
        self.save_manifest()
        self._close_worker()
        self._stop_services()
        self._repair_duplicate_metadata(metadata_repairs)
        self._manifest["environment_update_started"] = True
        self.save_manifest()
        self._update_environment()
        self._smoke_test_environment()
        self._migrate_databases()
        self._start_services()
        self._open_worker()
        self.log("All Bluesky services restarted successfully; the worker is healthy.")
        self._manifest["deployment_healthy"] = True
        self._manifest["completed_at"] = utc_now()
        self._manifest["success"] = True
        self.save_manifest()
        self.set_state(
            "complete",
            "success",
            "Environment update and service restart completed. Review the log, then restart the GUI.",
            100,
            restore_manifest=str(self.manifest_path),
            gui_restart_required=True,
        )

    def recover_failed_apply(self) -> str:
        """Best-effort rollback/restart after a confirmed update fails."""
        if not self.manifest_path.exists():
            return "not-required"
        self._load_restore_manifest()
        stop_started = bool(self._manifest.get("services_stop_started"))
        update_started = bool(self._manifest.get("environment_update_started"))
        database_changed = bool(self._manifest.get("databases_requiring_restore"))
        deployment_healthy = bool(self._manifest.get("deployment_healthy"))
        if not stop_started and not update_started and not database_changed:
            return "not-required"

        self.log("Starting automatic recovery after the failed update.")
        self.set_state("recovery", "running", "Automatically recovering the previous environment", 5)
        if (update_started or database_changed) and not deployment_healthy:
            self._stop_services()
            self._restore_environment()
            self._restore_databases()
            self._smoke_test_environment()
        self._start_services()
        self._open_worker()
        self._manifest["automatic_recovery_at"] = utc_now()
        self._manifest["automatic_recovery_success"] = True
        self.save_manifest()
        self.log(
            "Automatic recovery completed; services and the worker are healthy. "
            "The GUI was left running for review."
        )
        return "success"

    def _load_restore_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise UpdateFailure(f"Restore manifest is missing: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise UpdateFailure("Restore manifest is invalid.")
        self._manifest = payload

    def _restore_environment(self) -> None:
        explicit_path = Path(str(self._manifest.get("conda_explicit", "")))
        explicit_urls = explicit_snapshot_urls(explicit_path)
        expected_artifacts = {
            urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit("/", 1)[-1])
            for url in explicit_urls
        }
        restore_explicit_path = self.run_dir / "conda-restore-explicit.txt"
        restore_explicit_path.write_text(
            "# Normalized restore snapshot\n@EXPLICIT\n"
            + "\n".join(explicit_urls)
            + "\n",
            encoding="utf-8",
        )
        backup_prefix = Path(
            str(
                self._manifest.get("environment_backup_prefix")
                or self.run_dir / "environment-before-restore"
            )
        )
        failed_prefix = self.run_dir / "environment-from-failed-restore"

        installed_artifacts = installed_conda_artifacts(self.env_prefix)
        if installed_artifacts != expected_artifacts:
            if backup_prefix.exists():
                # A previous attempt was interrupted after preserving the source
                # environment. Preserve any partial recreation before retrying.
                if self.env_prefix.exists():
                    if failed_prefix.exists():
                        raise UpdateFailure(
                            "Both a restore backup and a failed restore prefix already exist; "
                            "refusing to overwrite either one."
                        )
                    self.env_prefix.rename(failed_prefix)
            else:
                if not self.env_prefix.exists():
                    raise UpdateFailure(
                        f"The environment to restore is missing: {self.env_prefix}"
                    )
                if self.env_prefix.stat().st_dev != self.run_dir.stat().st_dev:
                    raise UpdateFailure(
                        "The environment and restore directory are on different filesystems; "
                        "a fast, atomic backup is not possible."
                    )
                self.env_prefix.rename(backup_prefix)
                self._manifest["environment_backup_prefix"] = str(backup_prefix)
                self._manifest["environment_backup_created_at"] = utc_now()
                self.save_manifest()

            self.set_state(
                "restore",
                "running",
                f"Recreating {ENVIRONMENT_NAME} from the exact Mamba snapshot",
                30,
            )
            try:
                self.run(
                    [
                        str(DEFAULT_MAMBA),
                        "create",
                        "--prefix",
                        str(self.env_prefix),
                        "--file",
                        str(restore_explicit_path),
                        "--yes",
                    ],
                    timeout=PACKAGE_TIMEOUT,
                )
                installed_artifacts = installed_conda_artifacts(self.env_prefix)
                missing = expected_artifacts - installed_artifacts
                unexpected = installed_artifacts - expected_artifacts
                if missing or unexpected:
                    raise UpdateFailure(
                        "The recreated environment does not exactly match its snapshot "
                        f"({len(missing)} missing, {len(unexpected)} unexpected artifacts)."
                    )
            except Exception:
                # Creation did not produce a verified environment. Keep the
                # partial result for diagnosis and put the original prefix back.
                if self.env_prefix.exists() and not failed_prefix.exists():
                    self.env_prefix.rename(failed_prefix)
                if backup_prefix.exists() and not self.env_prefix.exists():
                    backup_prefix.rename(self.env_prefix)
                    self._manifest["environment_backup_rolled_back_at"] = utc_now()
                    self.save_manifest()
                raise

            self._manifest["environment_snapshot_restored_at"] = utc_now()
            self._manifest["environment_snapshot_package_count"] = len(expected_artifacts)
            self.save_manifest()
        else:
            self.log(
                f"The Conda environment already exactly matches all "
                f"{len(expected_artifacts)} snapshot artifacts."
            )

        pip_restore_value = self._manifest.get("pip_freeze")
        pip_restore = Path(str(pip_restore_value)) if pip_restore_value else None
        if pip_restore is not None and pip_restore.exists() and pip_restore.stat().st_size:
            self.set_state("restore", "running", "Restoring pip-only packages", 48)
            self.run(
                [
                    str(self.env_python),
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-deps",
                    "--requirement",
                    str(pip_restore),
                ],
                timeout=PACKAGE_TIMEOUT,
            )
        self._verify_pip_consistency()

    def _restore_databases(self) -> None:
        databases = list(self._manifest.get("databases_requiring_restore") or [])
        if not databases:
            self.log("No database migrations were attempted; database restore is unnecessary.")
            return
        env = self._postgres_env()
        backups = self._manifest.get("database_backups") or {}
        pg_restore = shutil.which("pg_restore") or "/usr/bin/pg_restore"
        for database in databases:
            backup = backups.get(database) or {}
            backup_path = Path(str(backup.get("path", "")))
            if not backup_path.exists():
                raise UpdateFailure(f"Database backup is missing for {database}: {backup_path}")
            digest = hashlib.sha256(backup_path.read_bytes()).hexdigest()
            if digest != backup.get("sha256"):
                raise UpdateFailure(f"Database backup checksum failed for {database}.")
            self.set_state("restore", "running", f"Restoring database {database}", 62)
            self.run(
                [
                    pg_restore,
                    "--clean",
                    "--if-exists",
                    "--no-owner",
                    "--no-privileges",
                    "--host",
                    env.get("TILED_PGHOST", "localhost"),
                    "--port",
                    env.get("TILED_PGPORT", "5432"),
                    "--username",
                    env.get("TILED_PGUSER", "tiled"),
                    "--dbname",
                    database,
                    str(backup_path),
                ],
                env=env,
                timeout=DATABASE_TIMEOUT,
            )

    def restore(self) -> None:
        self.log("Starting confirmed restore of the previous Bluesky environment.")
        self._load_restore_manifest()
        self.set_state("restore", "running", "Stopping services for restore", 10)
        self._stop_services()
        self._restore_environment()
        self._restore_databases()
        self._smoke_test_environment()
        self._start_services()
        self._open_worker()
        self._manifest["restored_at"] = utc_now()
        self._manifest["restore_success"] = True
        self.save_manifest()
        self.set_state(
            "complete",
            "success",
            "Previous environment and services restored. Review the log, then restart the GUI.",
            100,
            restore_manifest=str(self.manifest_path),
            gui_restart_required=True,
        )

    def restart_gui(self) -> None:
        """Restart the GUI only after the operator accepts the completed update."""
        self.set_state("gui", "running", "Restarting the control GUI", 100)
        self._restart_gui()
        if self.manifest_path.exists():
            self._load_restore_manifest()
            self._manifest["gui_restarted_at"] = utc_now()
            self.save_manifest()
        self.log("GUI restart completed successfully.")
        self.set_state(
            "complete",
            "success",
            "Environment update complete; GUI restarted successfully.",
            100,
            restore_manifest=str(self.manifest_path),
            gui_restart_required=False,
        )


def acquire_update_lock(run_dir: Path):
    if fcntl is None:
        raise UpdateFailure("The Bluesky environment updater requires Linux.")
    lock_dir = Path.home() / ".local" / "state" / "diffractometer-controls"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (lock_dir / "bluesky-environment-update.lock").open("w")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as ex:
        raise UpdateFailure("Another Bluesky environment update is already running.") from ex
    lock_handle.write(str(run_dir) + "\n")
    lock_handle.flush()
    return lock_handle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "apply", "restore", "restart-gui"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = UpdateContext(args.run_dir, args.repo_root)
    lock_handle = None
    try:
        if not local_maintenance_allowed():
            raise UpdateFailure(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
        lock_handle = acquire_update_lock(context.run_dir)
        if args.mode == "check":
            context.check()
        elif args.mode == "apply":
            context.apply()
        elif args.mode == "restore":
            context.restore()
        else:
            context.restart_gui()
        return 0
    except Exception as ex:
        context.log(f"ERROR: {ex}")
        original_traceback = traceback.format_exc()
        for line in original_traceback.rstrip().splitlines():
            context.log(line)
        recovery_status = "not-attempted"
        recovery_error = ""
        if args.mode == "apply" and context.manifest_path.exists():
            try:
                recovery_status = context.recover_failed_apply()
            except Exception as recovery_ex:
                recovery_status = "failed"
                recovery_error = str(recovery_ex)
                context.log(f"AUTOMATIC RECOVERY ERROR: {recovery_ex}")
                for line in traceback.format_exc().rstrip().splitlines():
                    context.log(line)
        restore_available = context.manifest_path.exists() and recovery_status not in {
            "success",
            "not-required",
        }
        message = str(ex)
        if recovery_status == "success":
            message += " Automatic recovery restored the previous environment and services."
        elif recovery_status == "failed":
            message += f" Automatic recovery also failed: {recovery_error}"
        context.set_state(
            "failed",
            "failed",
            message,
            100,
            restore_available=restore_available,
            restore_manifest=str(context.manifest_path) if restore_available else "",
            automatic_recovery=recovery_status,
            gui_restart_required=(recovery_status == "success"),
        )
        return 1
    finally:
        if lock_handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
            except Exception:
                pass
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())
