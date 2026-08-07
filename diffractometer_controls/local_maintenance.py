"""Fail-closed permission check for control-host maintenance operations."""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping
from pathlib import Path


LOCAL_MAINTENANCE_ENV = "MITR_ALLOW_LOCAL_MAINTENANCE"
DEFAULT_CONTROL_ENV = Path.home() / ".config" / "diffractometer-controls" / "control.env"
LOCAL_MAINTENANCE_DISABLED_MESSAGE = (
    "Local maintenance is disabled on this computer. These controls are only "
    "available on the configured Linux control host."
)
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def _read_env_value(path: Path, key: str) -> str | None:
    """Read one value from the simple KEY=VALUE control configuration."""
    try:
        lines = path.expanduser().read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return None

    value = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        candidate_key, candidate_value = line.split("=", 1)
        if candidate_key.strip() != key:
            continue
        candidate_value = candidate_value.strip()
        if (
            len(candidate_value) >= 2
            and candidate_value[0] == candidate_value[-1]
            and candidate_value[0] in {"'", '"'}
        ):
            candidate_value = candidate_value[1:-1]
        value = candidate_value
    return value


def local_maintenance_allowed(
    *,
    environ: Mapping[str, str] | None = None,
    control_env: Path | None = None,
    system_name: str | None = None,
) -> bool:
    """Return whether this process is permitted to manage local control services.

    Permission is Linux-only and opt-in. The process environment takes precedence;
    detached helpers fall back to the per-machine ``control.env`` file.
    """
    if (system_name or platform.system()).casefold() != "linux":
        return False

    values = os.environ if environ is None else environ
    raw_value = values.get(LOCAL_MAINTENANCE_ENV)
    if raw_value is None:
        raw_value = _read_env_value(control_env or DEFAULT_CONTROL_ENV, LOCAL_MAINTENANCE_ENV)
    return str(raw_value or "").strip().casefold() in _TRUE_VALUES
