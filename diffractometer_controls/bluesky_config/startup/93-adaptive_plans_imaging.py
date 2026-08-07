import numpy as np

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import plan_patterns, utils
from bluesky_queueserver import parameter_annotation_decorator
from epics import caget
from ophyd import Signal
from pathlib import Path
import sys

try:
    from diffractometer_controls.adaptive_focus import (
        FocusAdaptiveSessionStore,
        FocusScanSpec,
        build_focus_adaptive_metadata,
        normalize_focus_command,
        to_float,
    )
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
    )
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.adaptive_focus import (
        FocusAdaptiveSessionStore,
        FocusScanSpec,
        build_focus_adaptive_metadata,
        normalize_focus_command,
        to_float,
    )
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
    )


def _collect_focus_motor_names():
    # Reuse the shared motor collection helper from startup globals.
    fn = globals().get("_collect_movable_names", None)
    if callable(fn):
        try:
            return list(fn())
        except Exception:
            pass
    return []


_focus_adaptive_store = FocusAdaptiveSessionStore()


def adaptive_focus_submit_command(session_id: str, command: str, payload: dict = None):
    """
    Submit a command to a running adaptive focus session.

    Intended for external GUI/process integration (e.g. function_execute).
    """
    return _focus_adaptive_store.submit_command(session_id, command, payload)


def adaptive_focus_get_session(session_id: str):
    """Return snapshot state for one adaptive focus session."""
    return _focus_adaptive_store.get_session(session_id)


def _plan_estimation_context():
    transfer_rate = float(globals().get("transfer_time_per_bytes", 4.1203007518796994e-08))
    return build_estimation_context(
        caget_func=caget,
        transfer_time_per_bytes=transfer_rate,
    )


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
                "default": "cam1.focus",
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
    motor=cam1.focus,
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
    Adaptive imaging focus plan.

    Runs an initial coarse focus scan, then waits for external GUI/analysis
    commands that request additional focus measurements or terminate the run.
    """
    file_name = str(file_name).strip().replace(" ", "_").replace("__", "_")
    file_dir = str(file_dir).strip().replace(" ", "_").replace("__", "_")
    detector = [detector]
    motor_position_signal = Signal(
        name=f"{motor.name}_position",
        value=np.nan,
        kind="hinted",
    )

    scan_spec = FocusScanSpec.from_inputs(
        focus_guess=focus_guess,
        scan_half_range=scan_half_range,
        num_steps=num_steps,
        start_pos=start_pos,
        stop_pos=stop_pos,
    )

    estimate = estimate_plan_runtime(
        "adaptive_imaging_focus_scan",
        kwargs={
            "focus_guess": focus_guess,
            "scan_half_range": scan_half_range,
            "num_steps": scan_spec.num_steps,
            "start_pos": scan_spec.start_pos,
            "stop_pos": scan_spec.stop_pos,
            "exposure_time": exposure_time,
        },
        context=_plan_estimation_context(),
    )
    total_units = int(estimate.get("estimated_total_units") or scan_spec.num_steps)
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    session_id = _focus_adaptive_store.create(
        initial_state=scan_spec.initial_state(
            plan="adaptive_imaging_focus_scan",
            motor=motor.name,
            file_name=file_name,
            file_dir=file_dir,
            total_units=total_units,
        )
    )

    md = md or {}
    _md = build_focus_adaptive_metadata(
        file_name=file_name,
        file_dir=file_dir,
        detector_names=[det.name for det in detector],
        detector_config={
            "exposure_time": float(exposure_time) if exposure_time is not None else detector[0].cam.acquire_time.get(),
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        motor_name=motor.name,
        scan_spec=scan_spec,
        session_id=session_id,
        total_time=total_time,
        total_units=total_units,
        plan_patterns_module=plan_patterns.__name__,
        md=md,
    )

    x_fields = _set_scan_motor_metadata(_md, [motor])

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
        _focus_adaptive_store.update(
            session_id,
            status="running_coarse_scan",
            state_update={
                "done_units": 0,
                "total_units": int(total_units),
                "queued_commands": 0,
            },
        )
        _focus_adaptive_store.add_history(
            session_id,
            "coarse_scan_started",
            {
                "start_pos": float(scan_spec.start_pos),
                "stop_pos": float(scan_spec.stop_pos),
                "num_steps": int(scan_spec.num_steps),
            },
        )

        done_units = 0
        total_units_runtime = int(total_units)
        left_bound = float(scan_spec.start_pos)
        right_bound = float(scan_spec.stop_pos)

        def _set_total_units_runtime(new_total):
            nonlocal total_units_runtime
            total_units_runtime = int(max(1, int(new_total)))
            if progress is not None:
                try:
                    progress.total_units = int(total_units_runtime)
                except Exception:
                    pass
            _focus_adaptive_store.update(
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
                yield from bps.mv(motor_position_signal, p)
                yield from bps.trigger_and_read(detector + [motor, motor_position_signal])
                done_units += 1
                if progress is not None and hasattr(progress, "on_unit_success"):
                    yield from progress.on_unit_success(done_units - 1)
                left_bound = float(min(left_bound, p))
                right_bound = float(max(right_bound, p))
                _focus_adaptive_store.update(
                    session_id,
                    state_update={
                        "done_units": int(done_units),
                        "last_position": p,
                        "left_bound": float(left_bound),
                        "right_bound": float(right_bound),
                    },
                )
            _focus_adaptive_store.add_history(
                session_id,
                "acquire_positions",
                {"source": str(source), "count": int(len(pos_list))},
            )

        def _adaptive_body():
            nonlocal done_units, total_units_runtime, left_bound, right_bound
            yield from _acquire_positions(scan_spec.positions, source="coarse")

            _focus_adaptive_store.update(
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
            _focus_adaptive_store.add_history(
                session_id,
                "coarse_scan_complete",
                {"done_units": int(done_units)},
            )

            # Adaptive command loop: execute external GUI/agent commands until completion.
            is_complete = False
            while not is_complete:
                cmd = _focus_adaptive_store.pop_command(session_id)
                if cmd is None:
                    yield from bps.checkpoint()
                    yield from bps.sleep(0.2)
                    continue
                cmd_name = str(cmd.get("command", "")).strip().lower()
                payload = dict(cmd.get("payload", {}) or {})
                _focus_adaptive_store.update(
                    session_id,
                    status="processing_command",
                    state_update={"last_command": cmd_name},
                )
                _focus_adaptive_store.add_history(
                    session_id,
                    "command_received",
                    {"command": cmd_name, "payload": payload},
                )

                action = normalize_focus_command(
                    cmd_name,
                    payload,
                    current_position=to_float(getattr(motor, "position", None), default=None),
                    coarse_step_size=float(scan_spec.step_size) if np.isfinite(scan_spec.step_size) else None,
                    left_bound=left_bound,
                    right_bound=right_bound,
                )
                if action.kind == "terminal":
                    _focus_adaptive_store.update(session_id, status=action.status)
                    _focus_adaptive_store.add_history(
                        session_id, action.history_event, action.history_payload
                    )
                    is_complete = True
                    continue

                if action.kind == "acquire":
                    _set_total_units_runtime(int(total_units_runtime + len(action.positions)))
                    yield from _acquire_positions(action.positions, source=action.source)
                    _focus_adaptive_store.update(
                        session_id,
                        status="awaiting_command",
                        state_update=action.state_update,
                    )
                    if action.history_event:
                        _focus_adaptive_store.add_history(
                            session_id, action.history_event, action.history_payload
                        )
                    continue

                _focus_adaptive_store.add_history(
                    session_id,
                    "command_ignored",
                    {"command": cmd_name, "reason": action.reason},
                )
                _focus_adaptive_store.update(session_id, status="awaiting_command")

        _reset_array_counter = globals().get("_reset_detector_array_counter", None)
        if callable(_reset_array_counter):
            yield from _reset_array_counter(detector)

        for det in detector:
            if hasattr(det, "tiff1"):
                if hasattr(det.tiff1, "file_name"):
                    det.tiff1.file_name.put(file_name)
                if hasattr(det.tiff1, "folder_name"):
                    det.tiff1.folder_name.put(file_dir)
        yield from bpp.stage_wrapper(
            _adaptive_body(),
            _deduplicate_stage_devices(detector + [motor]),
        )
        final_state = adaptive_focus_get_session(session_id)
        return {
            "session_id": str(session_id),
            "status": str(final_state.get("status", "unknown")),
            "coarse_points": int(scan_spec.num_steps),
            "done_units": int(final_state.get("state", {}).get("done_units", done_units)),
            "total_units": int(final_state.get("state", {}).get("total_units", total_units_runtime)),
        }

    old_exposure_times = [det.cam.acquire_time.get() for det in detector]

    def _restore_exposure_time():
        if exposure_time is not None:
            for det, old in zip(detector, old_exposure_times):
                yield from bps.mov(det.cam.acquire_time, old)

    def _run_with_exposure_time():
        if exposure_time is not None:
            for det in detector:
                yield from bps.mov(det.cam.acquire_time, exposure_time)
        return (yield from bpp.finalize_wrapper(_main_plan(), _restore_exposure_time()))

    try:
        return (yield from _run_with_exposure_time())
    finally:
        _focus_adaptive_store.delete(session_id)
