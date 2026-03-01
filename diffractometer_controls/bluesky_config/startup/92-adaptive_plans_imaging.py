import numpy as np

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import plan_patterns, utils
from bluesky_queueserver import parameter_annotation_decorator
from epics import caget
from collections import deque
import threading
import time
import uuid


def _collect_focus_motor_names():
    # Reuse the motor collection helper from 91-plans_imaging when available.
    fn = globals().get("_collect_movable_names", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            pass
    return []


_focus_adaptive_sessions = {}
_focus_adaptive_lock = threading.Lock()


def _focus_adaptive_now() -> float:
    return float(time.time())


def _focus_adaptive_create_session(*, initial_state=None) -> str:
    session_id = str(uuid.uuid4())
    now = _focus_adaptive_now()
    state = dict(initial_state or {})
    with _focus_adaptive_lock:
        _focus_adaptive_sessions[session_id] = {
            "session_id": session_id,
            "created": now,
            "updated": now,
            "status": "created",
            "state": state,
            "commands": deque(),
            "history": [],
        }
    return session_id


def _focus_adaptive_session_update(session_id: str, *, status=None, state_update=None):
    with _focus_adaptive_lock:
        entry = _focus_adaptive_sessions.get(str(session_id), None)
        if entry is None:
            return
        if status is not None:
            entry["status"] = str(status)
        if state_update:
            entry["state"].update(dict(state_update))
        entry["updated"] = _focus_adaptive_now()


def _focus_adaptive_session_add_history(session_id: str, event: str, payload=None):
    with _focus_adaptive_lock:
        entry = _focus_adaptive_sessions.get(str(session_id), None)
        if entry is None:
            return
        rec = {
            "ts": _focus_adaptive_now(),
            "event": str(event),
            "payload": dict(payload or {}),
        }
        entry["history"].append(rec)
        if len(entry["history"]) > 500:
            entry["history"] = entry["history"][-500:]
        entry["updated"] = rec["ts"]


def adaptive_focus_submit_command(session_id: str, command: str, payload: dict = None):
    """
    Submit a command to a running adaptive focus session.

    Intended for external GUI/process integration (e.g. function_execute).
    """
    cmd = {
        "ts": _focus_adaptive_now(),
        "command": str(command),
        "payload": dict(payload or {}),
    }
    with _focus_adaptive_lock:
        entry = _focus_adaptive_sessions.get(str(session_id), None)
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


def adaptive_focus_get_session(session_id: str):
    """Return snapshot state for one adaptive focus session."""
    with _focus_adaptive_lock:
        entry = _focus_adaptive_sessions.get(str(session_id), None)
        if entry is None:
            return {
                "ok": False,
                "error": "session_not_found",
                "session_id": str(session_id),
            }
        state = dict(entry["state"])
        return {
            "ok": True,
            "session_id": str(session_id),
            "created": float(entry["created"]),
            "updated": float(entry["updated"]),
            "status": str(entry["status"]),
            "queued_commands": int(len(entry["commands"])),
            "state": state,
            "history_count": int(len(entry["history"])),
        }


def _adaptive_focus_pop_command(session_id: str):
    with _focus_adaptive_lock:
        entry = _focus_adaptive_sessions.get(str(session_id), None)
        if entry is None or not entry["commands"]:
            return None
        cmd = entry["commands"].popleft()
        entry["updated"] = _focus_adaptive_now()
        return cmd


def _focus_adaptive_delete_session(session_id: str):
    with _focus_adaptive_lock:
        _focus_adaptive_sessions.pop(str(session_id), None)


def _adaptive_focus_to_float(value, default=None):
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _adaptive_focus_to_int(value, default=None):
    try:
        return int(value)
    except Exception:
        return default


@parameter_annotation_decorator(
    {
        "parameters": {
            "detector": {
                "annotation": "__READABLE__",
                "default": "cam1",
                "description": "Imaging detector (must be readable, default: cam1)",
            },
            "motor": {
                "annotation": "typing.Union[str, Motors]",
                "default": "cam_focus",
                "description": "Focus motor to scan (must be movable)",
                "devices": {"Motors": _collect_focus_motor_names()},
                "convert_device_names": True,
            },
        }
    }
)
def adaptive_imaging_focus_scan(
    file_name: str,
    file_dir: str,
    motor=cam_focus,
    focus_guess: float = None,
    scan_half_range: float = None,
    num_steps: int = 15,
    detector=cam1,
    start_pos: float = None,
    stop_pos: float = None,
    exposure_time: float = None,
    md: dict = None,
):
    """
    Skeleton adaptive focus plan.

    Current scope:
      - validates coarse scan parameters
      - computes coarse positions and estimated timing
      - builds metadata and hints in the same pattern as imaging_scan
      - creates a placeholder run body

    Adaptive behavior and GUI command loop will be added next.
    """
    file_name = str(file_name).strip().replace(" ", "_").replace("__", "_")
    file_dir = str(file_dir).strip().replace(" ", "_").replace("__", "_")
    detector = [detector]

    old_exposure_time = detector[0].cam.acquire_time.get()
    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    num_steps_calc = int(max(2, int(num_steps)))
    has_explicit_bounds = (start_pos is not None) and (stop_pos is not None)
    has_guess_range = (focus_guess is not None) and (scan_half_range is not None)
    if has_explicit_bounds:
        start_pos_calc = float(start_pos)
        stop_pos_calc = float(stop_pos)
    elif has_guess_range:
        guess = float(focus_guess)
        half = float(scan_half_range)
        if half <= 0:
            raise ValueError("'scan_half_range' must be positive.")
        start_pos_calc = guess - half
        stop_pos_calc = guess + half
    else:
        raise ValueError(
            "Provide either both 'start_pos' and 'stop_pos', "
            "or both 'focus_guess' and 'scan_half_range'."
        )
    if stop_pos_calc <= start_pos_calc:
        raise ValueError("'stop_pos' must be greater than 'start_pos'.")
    positions = np.linspace(
        start=float(start_pos_calc),
        stop=float(stop_pos_calc),
        num=num_steps_calc,
        endpoint=True,
    )
    step_calc = float(positions[1] - positions[0]) if num_steps_calc > 1 else np.nan

    # Keep timing estimate pattern consistent with imaging_scan.
    transfer_rate = float(globals().get("transfer_time_per_bytes", 4.1203007518796994e-08))
    image_bytes = caget("4dh4:cam1:ArraySize_RBV")
    try:
        image_bytes = float(image_bytes)
    except Exception:
        image_bytes = 0.0
    transfer_time_per_image = float(image_bytes) * transfer_rate
    total_units = int(num_steps_calc)
    total_time = (
        float(num_steps_calc) * detector[0].cam.acquire_time.get()
        + float(num_steps_calc) * transfer_time_per_image
    )
    session_id = _focus_adaptive_create_session(
        initial_state={
            "plan": "adaptive_imaging_focus_scan",
            "motor": motor.name,
            "file_name": file_name,
            "file_dir": file_dir,
            "start_pos": float(start_pos_calc),
            "stop_pos": float(stop_pos_calc),
            "step": float(step_calc),
            "num_steps": int(num_steps_calc),
            "done_units": 0,
            "total_units": int(total_units),
        }
    )

    md = md or {}
    _md = {
        "file_name": file_name,
        "file_dir": file_dir,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "plan_args": {},
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "plan_name": "adaptive_imaging_focus_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(
            motor=motor.name,
            start_pos=float(start_pos_calc),
            stop_pos=float(stop_pos_calc),
            stop_pos_calc=float(stop_pos_calc),
            focus_guess=float(focus_guess) if focus_guess is not None else None,
            scan_half_range=float(scan_half_range) if scan_half_range is not None else None,
            step=float(step_calc),
            num_steps=int(num_steps_calc),
        ),
        "motors": motor.name,
        "focus_adaptive": {
            "status": "coarse_scan_then_wait",
            "notes": "Runs coarse scan, then executes queued adaptive commands",
            "scan_mode": "guess_range" if has_guess_range and (not has_explicit_bounds) else "explicit_bounds",
            "session_id": str(session_id),
            "command_submit_fn": "adaptive_focus_submit_command",
            "command_state_fn": "adaptive_focus_get_session",
            "accepted_commands": [
                "go_to_focus",
                "scan_around_focus",
                "extend_left",
                "extend_right",
                "complete",
                "abort",
            ],
        },
    }
    _md.update(md)

    x_fields = []
    x_fields.extend(utils.get_hinted_fields(motor))
    default_hints = {}
    if len(x_fields) > 0:
        default_hints.update(dimensions=[(x_fields, "primary")])
    _md["hints"] = default_hints
    _md["hints"].update(md.get("hints", {}) or {})

    @bpp.run_decorator(md=_md)
    def _main_plan():
        progress = None
        _progress_cls = globals().get("_ProgressEstimator", None)
        if _progress_cls is not None:
            try:
                progress = _progress_cls(
                    total_units=total_units,
                    initial_total_time_s=total_time,
                )
            except Exception:
                progress = None
        _focus_adaptive_session_update(
            session_id,
            status="running_coarse_scan",
            state_update={
                "done_units": 0,
                "total_units": int(total_units),
                "queued_commands": 0,
            },
        )
        _focus_adaptive_session_add_history(
            session_id,
            "coarse_scan_started",
            {
                "start_pos": float(start_pos_calc),
                "stop_pos": float(stop_pos_calc),
                "num_steps": int(num_steps_calc),
            },
        )

        _reset_array_counter = globals().get("_reset_detector_array_counter", None)
        if callable(_reset_array_counter):
            yield from _reset_array_counter(detector)

        for det in detector:
            if hasattr(det, "tiff1"):
                if hasattr(det.tiff1, "file_name"):
                    det.tiff1.file_name.put(file_name)
                if hasattr(det.tiff1, "folder_name"):
                    det.tiff1.folder_name.put(file_dir)
            yield from bps.stage(det)
        yield from bps.stage(motor)

        done_units = 0
        total_units_runtime = int(total_units)
        left_bound = float(start_pos_calc)
        right_bound = float(stop_pos_calc)

        def _set_total_units_runtime(new_total):
            nonlocal total_units_runtime
            total_units_runtime = int(max(1, int(new_total)))
            if progress is not None:
                try:
                    progress.total_units = int(total_units_runtime)
                except Exception:
                    pass
            _focus_adaptive_session_update(
                session_id,
                state_update={"total_units": int(total_units_runtime)},
            )

        def _acquire_positions(pos_list, *, source="coarse"):
            nonlocal done_units, left_bound, right_bound
            for pos in list(pos_list):
                p = float(pos)
                yield from bps.checkpoint()
                if progress is not None and hasattr(progress, "on_unit_start"):
                    yield from progress.on_unit_start(done_units)
                yield from bps.mv(motor, p)
                yield from bps.trigger_and_read(detector + [motor])
                done_units += 1
                if progress is not None and hasattr(progress, "on_unit_success"):
                    yield from progress.on_unit_success(done_units - 1)
                left_bound = float(min(left_bound, p))
                right_bound = float(max(right_bound, p))
                _focus_adaptive_session_update(
                    session_id,
                    state_update={
                        "done_units": int(done_units),
                        "last_position": p,
                        "left_bound": float(left_bound),
                        "right_bound": float(right_bound),
                    },
                )
            _focus_adaptive_session_add_history(
                session_id,
                "acquire_positions",
                {"source": str(source), "count": int(len(pos_list))},
            )

        yield from _acquire_positions(positions, source="coarse")

        _focus_adaptive_session_update(
            session_id,
            status="awaiting_command",
            state_update={
                "done_units": int(done_units),
                "total_units": int(total_units_runtime),
                "coarse_scan_complete": True,
                "left_bound": float(left_bound),
                "right_bound": float(right_bound),
            },
        )
        _focus_adaptive_session_add_history(
            session_id,
            "coarse_scan_complete",
            {"done_units": int(done_units)},
        )

        # Adaptive command loop: execute external GUI/agent commands until completion.
        is_complete = False
        while not is_complete:
            cmd = _adaptive_focus_pop_command(session_id)
            if cmd is None:
                yield from bps.checkpoint()
                yield from bps.sleep(0.2)
                continue
            cmd_name = str(cmd.get("command", "")).strip().lower()
            payload = dict(cmd.get("payload", {}) or {})
            _focus_adaptive_session_update(
                session_id,
                status="processing_command",
                state_update={"last_command": cmd_name},
            )
            _focus_adaptive_session_add_history(
                session_id,
                "command_received",
                {"command": cmd_name, "payload": payload},
            )

            if cmd_name in {"complete", "abort"}:
                end_status = "completed" if cmd_name == "complete" else "aborted"
                _focus_adaptive_session_update(session_id, status=end_status)
                _focus_adaptive_session_add_history(
                    session_id, "session_end", {"status": end_status}
                )
                is_complete = True
                continue

            if cmd_name == "go_to_focus":
                target = _adaptive_focus_to_float(
                    payload.get("target_position", payload.get("position", payload.get("focus_position", None))),
                    default=_adaptive_focus_to_float(motor.position, default=None),
                )
                if target is None:
                    _focus_adaptive_session_add_history(
                        session_id, "command_ignored", {"command": cmd_name, "reason": "no_target"}
                    )
                    _focus_adaptive_session_update(session_id, status="awaiting_command")
                    continue
                yield from bps.checkpoint()
                yield from bps.mv(motor, float(target))
                _focus_adaptive_session_update(
                    session_id,
                    status="awaiting_command",
                    state_update={"last_position": float(target)},
                )
                _focus_adaptive_session_add_history(
                    session_id, "go_to_focus_done", {"target_position": float(target)}
                )
                continue

            if cmd_name == "scan_around_focus":
                center = _adaptive_focus_to_float(
                    payload.get("center", payload.get("position", payload.get("target_position", None))),
                    default=_adaptive_focus_to_float(motor.position, default=0.0),
                )
                local_step = _adaptive_focus_to_float(
                    payload.get("step_size", payload.get("step", None)),
                    default=float(step_calc / 2.0) if np.isfinite(step_calc) and (step_calc > 0) else 0.1,
                )
                pts = _adaptive_focus_to_int(
                    payload.get("num_points", payload.get("points", 7)), default=7
                )
                pts = int(max(3, pts))
                if pts % 2 == 0:
                    pts += 1
                if local_step is None or local_step <= 0:
                    _focus_adaptive_session_add_history(
                        session_id, "command_ignored", {"command": cmd_name, "reason": "bad_step"}
                    )
                    _focus_adaptive_session_update(session_id, status="awaiting_command")
                    continue
                half_n = pts // 2
                offsets = np.arange(-half_n, half_n + 1, 1, dtype=float) * float(local_step)
                local_positions = np.asarray(center, dtype=float) + offsets
                _set_total_units_runtime(int(total_units_runtime + len(local_positions)))
                yield from _acquire_positions(local_positions, source="scan_around_focus")
                _focus_adaptive_session_update(session_id, status="awaiting_command")
                continue

            if cmd_name in {"extend_left", "extend_right"}:
                ext_n = _adaptive_focus_to_int(
                    payload.get("num_points", payload.get("points", 3)), default=3
                )
                ext_n = int(max(1, ext_n))
                coarse_step = float(step_calc) if np.isfinite(step_calc) and (step_calc > 0) else None
                if coarse_step is None:
                    _focus_adaptive_session_add_history(
                        session_id, "command_ignored", {"command": cmd_name, "reason": "no_coarse_step"}
                    )
                    _focus_adaptive_session_update(session_id, status="awaiting_command")
                    continue
                if cmd_name == "extend_left":
                    ext_positions = [float(left_bound - coarse_step * i) for i in range(1, ext_n + 1)]
                else:
                    ext_positions = [float(right_bound + coarse_step * i) for i in range(1, ext_n + 1)]
                _set_total_units_runtime(int(total_units_runtime + len(ext_positions)))
                yield from _acquire_positions(ext_positions, source=cmd_name)
                _focus_adaptive_session_update(session_id, status="awaiting_command")
                continue

            _focus_adaptive_session_add_history(
                session_id, "command_ignored", {"command": cmd_name, "reason": "unknown_command"}
            )
            _focus_adaptive_session_update(session_id, status="awaiting_command")

        if exposure_time is not None:
            yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)
        final_state = adaptive_focus_get_session(session_id)
        return {
            "session_id": str(session_id),
            "status": str(final_state.get("status", "unknown")),
            "coarse_points": int(num_steps_calc),
            "done_units": int(final_state.get("state", {}).get("done_units", done_units)),
            "total_units": int(final_state.get("state", {}).get("total_units", total_units_runtime)),
        }

    try:
        return (yield from _main_plan())
    finally:
        _focus_adaptive_delete_session(session_id)
