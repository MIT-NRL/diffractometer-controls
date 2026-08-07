"""Helpers for the adaptive imaging focus workflow.

This module intentionally stays independent of Bluesky/Ophyd so the command
protocol and session bookkeeping can be tested without loading beamline startup.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Mapping, Sequence
import uuid


ACCEPTED_FOCUS_COMMANDS = (
    "go_to_focus",
    "scan_around_focus",
    "extend_left",
    "extend_right",
    "complete",
    "abort",
)


def focus_adaptive_now() -> float:
    return float(time.time())


def to_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def to_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _linspace(start: float, stop: float, num: int) -> tuple[float, ...]:
    if num <= 1:
        return (float(start),)
    step_size = (float(stop) - float(start)) / float(num - 1)
    return tuple(float(start) + step_size * i for i in range(int(num)))


@dataclass(frozen=True)
class FocusScanSpec:
    """Validated coarse focus-scan parameters."""

    start_pos: float
    stop_pos: float
    num_steps: int
    positions: tuple[float, ...]
    step_size: float
    scan_mode: str
    focus_guess: float | None = None
    scan_half_range: float | None = None

    @classmethod
    def from_inputs(
        cls,
        *,
        focus_guess: float | None = None,
        scan_half_range: float | None = None,
        num_steps: int = 15,
        start_pos: float | None = None,
        stop_pos: float | None = None,
    ) -> "FocusScanSpec":
        num_steps_calc = int(max(2, int(num_steps)))
        has_explicit_bounds = (start_pos is not None) and (stop_pos is not None)
        has_guess_range = (focus_guess is not None) and (scan_half_range is not None)

        guess = to_float(focus_guess, None)
        half = to_float(scan_half_range, None)
        if has_explicit_bounds:
            start_pos_calc = float(start_pos)
            stop_pos_calc = float(stop_pos)
            scan_mode = "explicit_bounds"
        elif has_guess_range:
            if half is None or half <= 0:
                raise ValueError("'scan_half_range' must be positive.")
            if guess is None:
                raise ValueError("'focus_guess' must be finite.")
            start_pos_calc = guess - half
            stop_pos_calc = guess + half
            scan_mode = "guess_range"
        else:
            raise ValueError(
                "Provide either both 'start_pos' and 'stop_pos', "
                "or both 'focus_guess' and 'scan_half_range'."
            )

        if stop_pos_calc <= start_pos_calc:
            raise ValueError("'stop_pos' must be greater than 'start_pos'.")

        positions = _linspace(start_pos_calc, stop_pos_calc, num_steps_calc)
        step_size = float(positions[1] - positions[0]) if num_steps_calc > 1 else float("nan")
        return cls(
            start_pos=float(start_pos_calc),
            stop_pos=float(stop_pos_calc),
            num_steps=int(num_steps_calc),
            positions=positions,
            step_size=float(step_size),
            scan_mode=scan_mode,
            focus_guess=guess,
            scan_half_range=half,
        )

    def initial_state(self, *, plan: str, motor: str, file_name: str, file_dir: str, total_units: int) -> dict[str, Any]:
        return {
            "plan": str(plan),
            "motor": str(motor),
            "file_name": str(file_name),
            "file_dir": str(file_dir),
            "start_pos": float(self.start_pos),
            "stop_pos": float(self.stop_pos),
            "step_size": float(self.step_size),
            "num_steps": int(self.num_steps),
            "done_units": 0,
            "total_units": int(total_units),
        }

    def plan_pattern_args(self, *, motor: str) -> dict[str, Any]:
        return {
            "motor": str(motor),
            "start_pos": float(self.start_pos),
            "stop_pos": float(self.stop_pos),
            "stop_pos_calc": float(self.stop_pos),
            "focus_guess": self.focus_guess,
            "scan_half_range": self.scan_half_range,
            "step_size": float(self.step_size),
            "num_steps": int(self.num_steps),
        }


class FocusAdaptiveSessionStore:
    """Thread-safe in-memory session registry used by Queue Server functions."""

    def __init__(self, *, history_limit: int = 500, now_func=focus_adaptive_now):
        self.history_limit = int(max(1, history_limit))
        self._now = now_func
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, *, initial_state: Mapping[str, Any] | None = None) -> str:
        session_id = str(uuid.uuid4())
        now = float(self._now())
        with self._lock:
            self._sessions[session_id] = {
                "session_id": session_id,
                "created": now,
                "updated": now,
                "status": "created",
                "state": dict(initial_state or {}),
                "commands": deque(),
                "history": [],
            }
        return session_id

    def update(self, session_id: str, *, status: str | None = None, state_update: Mapping[str, Any] | None = None):
        with self._lock:
            entry = self._sessions.get(str(session_id), None)
            if entry is None:
                return
            if status is not None:
                entry["status"] = str(status)
            if state_update:
                entry["state"].update(dict(state_update))
            entry["updated"] = float(self._now())

    def add_history(self, session_id: str, event: str, payload: Mapping[str, Any] | None = None):
        with self._lock:
            entry = self._sessions.get(str(session_id), None)
            if entry is None:
                return
            rec = {
                "ts": float(self._now()),
                "event": str(event),
                "payload": dict(payload or {}),
            }
            entry["history"].append(rec)
            if len(entry["history"]) > self.history_limit:
                entry["history"] = entry["history"][-self.history_limit :]
            entry["updated"] = rec["ts"]

    def submit_command(self, session_id: str, command: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        cmd = {
            "ts": float(self._now()),
            "command": str(command),
            "payload": dict(payload or {}),
        }
        with self._lock:
            entry = self._sessions.get(str(session_id), None)
            if entry is None:
                return {
                    "ok": False,
                    "error": "session_not_found",
                    "session_id": str(session_id),
                }
            entry["commands"].append(cmd)
            entry["updated"] = cmd["ts"]
            queued = int(len(entry["commands"]))
        return {
            "ok": True,
            "session_id": str(session_id),
            "queued_commands": queued,
            "accepted_command": str(command),
        }

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._sessions.get(str(session_id), None)
            if entry is None:
                return {
                    "ok": False,
                    "error": "session_not_found",
                    "session_id": str(session_id),
                }
            return {
                "ok": True,
                "session_id": str(session_id),
                "created": float(entry["created"]),
                "updated": float(entry["updated"]),
                "status": str(entry["status"]),
                "queued_commands": int(len(entry["commands"])),
                "state": dict(entry["state"]),
                "history_count": int(len(entry["history"])),
            }

    def pop_command(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._sessions.get(str(session_id), None)
            if entry is None or not entry["commands"]:
                return None
            cmd = entry["commands"].popleft()
            entry["updated"] = float(self._now())
            return cmd

    def delete(self, session_id: str):
        with self._lock:
            self._sessions.pop(str(session_id), None)


@dataclass(frozen=True)
class FocusCommandAction:
    """Normalized result of one adaptive command."""

    kind: str
    command: str
    positions: tuple[float, ...] = field(default_factory=tuple)
    source: str = ""
    status: str = ""
    reason: str = ""
    state_update: dict[str, Any] = field(default_factory=dict)
    history_event: str = ""
    history_payload: dict[str, Any] = field(default_factory=dict)


def _payload_value(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload.get(name)
    return None


def normalize_focus_command(
    command: str,
    payload: Mapping[str, Any] | None = None,
    *,
    current_position: float | None = None,
    coarse_step_size: float | None = None,
    left_bound: float | None = None,
    right_bound: float | None = None,
) -> FocusCommandAction:
    """Convert an external command payload into an explicit plan action."""
    cmd = str(command or "").strip().lower()
    payload = dict(payload or {})

    if cmd in {"complete", "abort"}:
        end_status = "completed" if cmd == "complete" else "aborted"
        return FocusCommandAction(
            kind="terminal",
            command=cmd,
            status=end_status,
            history_event="session_end",
            history_payload={"status": end_status, "command": cmd, "payload": dict(payload)},
        )

    if cmd == "go_to_focus":
        target = to_float(
            _payload_value(payload, "target_position", "position", "focus_position"),
            default=to_float(current_position, default=None),
        )
        if target is None:
            return FocusCommandAction(kind="ignore", command=cmd, reason="no_target")
        return FocusCommandAction(
            kind="acquire",
            command=cmd,
            positions=(float(target),),
            source=cmd,
            state_update={"last_position": float(target)},
            history_event="go_to_focus_done",
            history_payload={"target_position": float(target)},
        )

    if cmd == "scan_around_focus":
        center = to_float(
            _payload_value(payload, "center", "position", "target_position"),
            default=to_float(current_position, default=0.0),
        )
        default_step_size = (
            float(coarse_step_size) / 2.0
            if coarse_step_size is not None and coarse_step_size > 0
            else 0.1
        )
        local_step_size = to_float(_payload_value(payload, "step_size", "step"), default=default_step_size)
        pts = to_int(_payload_value(payload, "num_points", "points"), default=7)
        pts = int(max(3, pts if pts is not None else 7))
        if pts % 2 == 0:
            pts += 1
        if local_step_size is None or local_step_size <= 0:
            return FocusCommandAction(kind="ignore", command=cmd, reason="bad_step")
        half_n = pts // 2
        positions = tuple(float(center) + float(i * local_step_size) for i in range(-half_n, half_n + 1))
        return FocusCommandAction(kind="acquire", command=cmd, positions=positions, source=cmd)

    if cmd in {"extend_left", "extend_right"}:
        step_size = to_float(coarse_step_size, default=None)
        if step_size is None or step_size <= 0:
            return FocusCommandAction(kind="ignore", command=cmd, reason="no_coarse_step")
        ext_n = to_int(_payload_value(payload, "num_points", "points"), default=3)
        ext_n = int(max(1, ext_n if ext_n is not None else 3))
        if cmd == "extend_left":
            start = to_float(left_bound, default=0.0)
            positions = tuple(float(start) - float(step_size) * i for i in range(1, ext_n + 1))
        else:
            start = to_float(right_bound, default=0.0)
            positions = tuple(float(start) + float(step_size) * i for i in range(1, ext_n + 1))
        return FocusCommandAction(kind="acquire", command=cmd, positions=positions, source=cmd)

    return FocusCommandAction(kind="ignore", command=cmd, reason="unknown_command")


def build_focus_adaptive_metadata(
    *,
    file_name: str,
    file_dir: str,
    detector_names: Sequence[str],
    detector_config: Mapping[str, Any],
    motor_name: str,
    scan_spec: FocusScanSpec,
    session_id: str,
    total_time: float,
    total_units: int,
    plan_patterns_module: str,
    md: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the start-document metadata used by the adaptive focus plan."""
    out = {
        "file_name": str(file_name),
        "file_dir": str(file_dir),
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": list(detector_names),
        "plan_args": {
            "detectors": list(detector_names),
        },
        "det_config": dict(detector_config),
        "plan_name": "adaptive_imaging_focus_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": str(plan_patterns_module),
        "plan_pattern_args": scan_spec.plan_pattern_args(motor=str(motor_name)),
        "motors": str(motor_name),
        "focus_adaptive": {
            "status": "coarse_scan_then_wait",
            "notes": "Runs coarse scan, then executes queued adaptive commands",
            "scan_mode": str(scan_spec.scan_mode),
            "session_id": str(session_id),
            "command_submit_fn": "adaptive_focus_submit_command",
            "command_state_fn": "adaptive_focus_get_session",
            "accepted_commands": list(ACCEPTED_FOCUS_COMMANDS),
        },
    }
    out.update(dict(md or {}))
    return out
