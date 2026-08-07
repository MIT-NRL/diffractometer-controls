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
import shlex
import shutil
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
DEFAULT_CONDA = Path("/home/mitr_4dh4/mambaforge/bin/conda")
DEFAULT_ENV_PREFIX = Path("/home/mitr_4dh4/mambaforge/envs/bluesky-server")
DEFAULT_TILED_ENV = Path("/home/mitr_4dh4/.config/tiled/tiled-server.env")
DEFAULT_CONTROL_ENV = Path("/home/mitr_4dh4/.config/diffractometer-controls/control.env")
DEFAULT_QSERVER_CLIENT_ENV = Path(
    "/home/mitr_4dh4/.config/bluesky-queueserver/client-zmq.env"
)
SERVICE_NAMES = ("queue-server", "tiled-server", "bluesky-proxy")


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


def summarize_mamba_plan(payload: dict) -> tuple[list[str], bool]:
    """Return readable transaction lines and whether Conda would change."""
    actions = payload.get("actions") or {}
    links = {str(item.get("name")): item for item in actions.get("LINK", [])}
    unlinks = {str(item.get("name")): item for item in actions.get("UNLINK", [])}
    names = sorted(set(links) | set(unlinks), key=str.casefold)
    lines: list[str] = []
    for name in names:
        old = unlinks.get(name)
        new = links.get(name)
        if old and new:
            old_version = str(old.get("version", "?"))
            new_version = str(new.get("version", "?"))
            old_build = str(old.get("build_string") or old.get("build") or "")
            new_build = str(new.get("build_string") or new.get("build") or "")
            if old_version == new_version and old_build != new_build:
                lines.append(f"Rebuild {name}: {old_version} ({old_build} -> {new_build})")
            else:
                lines.append(f"Update {name}: {old_version} -> {new_version}")
        elif new:
            lines.append(f"Install {name}: {new.get('version', '?')}")
        elif old:
            lines.append(f"Remove {name}: {old.get('version', '?')}")
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
        timeout: float | None = None,
        display_command: str | None = None,
        log_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        shown = display_command or redact_command(command)
        self.log(f"$ {shown}")
        if capture:
            result = subprocess.run(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            output = result.stdout or ""
            if output.strip() and (log_output or result.returncode != 0):
                for line in output.rstrip().splitlines():
                    self.log(line)
        else:
            process = subprocess.Popen(
                command,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            output_lines: list[str] = []
            assert process.stdout is not None
            for line in process.stdout:
                clean = line.rstrip("\n")
                output_lines.append(clean)
                self.log(clean)
            returncode = process.wait(timeout=timeout)
            result = subprocess.CompletedProcess(command, returncode, "\n".join(output_lines), None)
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

    def check(self) -> None:
        self.set_state("preflight", "running", "Checking Queue Server state", 5)
        self.log("Starting Bluesky environment update preview.")
        self.ensure_no_active_run()
        if not self.environment_file.exists():
            raise UpdateFailure(f"Environment file is missing: {self.environment_file}")
        if not DEFAULT_MAMBA.exists():
            raise UpdateFailure(f"Mamba executable is missing: {DEFAULT_MAMBA}")

        self.set_state("resolving", "running", "Checking project environment requirements", 20)
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
            environment_command, capture=True, timeout=None, log_output=False
        )
        environment_payload = extract_json_document(environment_result.stdout or "")
        if environment_payload.get("success") is False:
            raise UpdateFailure(
                str(
                    environment_payload.get("error")
                    or "Mamba could not resolve project requirements."
                )
            )

        self.set_state("resolving", "running", "Resolving available Conda updates", 45)
        update_command = [
            str(DEFAULT_MAMBA),
            "update",
            "--name",
            ENVIRONMENT_NAME,
            "--all",
            "--dry-run",
            "--json",
        ]
        update_result = self.run(update_command, capture=True, timeout=None, log_output=False)
        update_payload = extract_json_document(update_result.stdout or "")
        if update_payload.get("success") is False:
            raise UpdateFailure(
                str(update_payload.get("error") or "Mamba could not resolve the update.")
            )
        self.write_json_atomic(
            self.plan_path,
            {
                "project_environment": environment_payload,
                "update_all": update_payload,
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
            "Available Conda updates:",
            *[f"  {line}" for line in update_summary],
        ]
        has_conda_changes = has_environment_changes or has_update_changes
        if self.pip_requirements.exists():
            summary.extend(
                [
                    "",
                    "Pip-only packages will be checked after Conda and installed with --no-deps:",
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
            summary_file=str(self.summary_path),
        )

    def _record_restore_point(self) -> None:
        self.set_state("snapshot", "running", "Recording environment restore point", 12)
        revisions = self.run(
            [str(DEFAULT_CONDA), "list", "--name", ENVIRONMENT_NAME, "--revisions"],
            capture=True,
        ).stdout or ""
        (self.run_dir / "conda-revisions.txt").write_text(revisions, encoding="utf-8")
        revision_matches = re.findall(r"\(rev\s+(\d+)\)", revisions)
        revision = int(revision_matches[-1]) if revision_matches else None

        explicit = self.run(
            [str(DEFAULT_CONDA), "list", "--name", ENVIRONMENT_NAME, "--explicit"],
            capture=True,
            log_output=False,
        ).stdout or ""
        (self.run_dir / "conda-explicit.txt").write_text(explicit, encoding="utf-8")
        conda_packages_text = self.run(
            [str(DEFAULT_CONDA), "list", "--name", ENVIRONMENT_NAME, "--json"],
            capture=True,
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
            log_output=False,
        ).stdout or ""
        pip_freeze = filter_pip_freeze(pip_freeze_all, pip_package_names)
        (self.run_dir / "pip-freeze.txt").write_text(
            "\n".join(pip_freeze) + ("\n" if pip_freeze else ""), encoding="utf-8"
        )
        pip_check = self.run(
            [str(self.env_python), "-m", "pip", "check"],
            capture=True,
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
                "migrated_databases": [],
                "databases_requiring_restore": [],
            }
        )
        self.save_manifest()

    def _verify_pip_consistency(self) -> None:
        result = self.run(
            [str(self.env_python), "-m", "pip", "check"],
            capture=True,
            check=False,
        )
        current = parse_pip_check_issues(result.stdout or "")
        baseline = set(self._manifest.get("pip_check_issues") or [])
        new_issues = [issue for issue in current if issue not in baseline]
        if new_issues:
            raise UpdateFailure(
                "The update introduced package dependency problems: " + "; ".join(new_issues)
            )
        if current:
            self.log("Pre-existing pip dependency issue(s) remain:")
            for issue in current:
                self.log(f"  {issue}")
        else:
            self.log("Pip dependency check passed.")

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
            self.run([pg_restore, "--list", str(destination)], capture=True)
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
        self.set_state("conda", "running", "Applying project environment requirements", 38)
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
            ]
        )
        self.set_state("conda", "running", "Updating Conda environment", 46)
        self.run(
            [
                str(DEFAULT_MAMBA),
                "update",
                "--name",
                ENVIRONMENT_NAME,
                "--all",
                "--yes",
            ]
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
                ]
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
        self.run(["/usr/bin/perl", str(self.gui_script), "restart"], timeout=60)

    def apply(self) -> None:
        self.log("Starting confirmed Bluesky environment update.")
        self.set_state("preflight", "running", "Running final safety checks", 3)
        self.ensure_no_active_run()
        self._record_restore_point()
        self._backup_databases()
        self._close_worker()
        self._stop_services()
        self._update_environment()
        self._smoke_test_environment()
        self._migrate_databases()
        self._start_services()
        self._open_worker()
        self._manifest["completed_at"] = utc_now()
        self._manifest["success"] = True
        self.save_manifest()
        self.set_state("gui", "running", "Restarting the control GUI", 98)
        self._restart_gui()
        self.set_state(
            "complete",
            "success",
            "Environment update completed successfully",
            100,
            restore_manifest=str(self.manifest_path),
        )

    def _load_restore_manifest(self) -> None:
        if not self.manifest_path.exists():
            raise UpdateFailure(f"Restore manifest is missing: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise UpdateFailure("Restore manifest is invalid.")
        self._manifest = payload

    def _restore_environment(self) -> None:
        revision = self._manifest.get("conda_revision")
        if revision is None:
            raise UpdateFailure("The restore point does not contain a Conda revision.")
        self.set_state("restore", "running", f"Restoring Conda revision {revision}", 30)
        self.run(
            [
                str(DEFAULT_CONDA),
                "install",
                "--name",
                ENVIRONMENT_NAME,
                "--revision",
                str(revision),
                "--yes",
            ]
        )
        pip_restore = Path(str(self._manifest.get("pip_freeze", "")))
        if pip_restore.exists() and pip_restore.stat().st_size:
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
                ]
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
        self.set_state("gui", "running", "Restarting the control GUI", 98)
        self._restart_gui()
        self.set_state(
            "complete",
            "success",
            "Previous environment restored successfully",
            100,
            restore_manifest=str(self.manifest_path),
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
    parser.add_argument("mode", choices=("check", "apply", "restore"))
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
        else:
            context.restore()
        return 0
    except Exception as ex:
        context.log(f"ERROR: {ex}")
        for line in traceback.format_exc().rstrip().splitlines():
            context.log(line)
        restore_available = context.manifest_path.exists()
        context.set_state(
            "failed",
            "failed",
            str(ex),
            100,
            restore_available=restore_available,
            restore_manifest=str(context.manifest_path) if restore_available else "",
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
