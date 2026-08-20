# import bluesky.plans
import bluesky.plans as bp
from bluesky.plans import scan, count, grid_scan, rel_scan, rel_grid_scan
from bluesky.protocols import Readable, Movable

from bluesky_queueserver import parameter_annotation_decorator


# import bluesky.plan_stubs
import bluesky.plan_stubs as bps
# from bluesky.plan_stubs import *
from bluesky import plan_patterns, utils
from typing import Annotated
from collections import defaultdict
import time
from datetime import datetime, timedelta

import bluesky.preprocessors
import bluesky.preprocessors as bpp
import numpy as np
from ophyd import (Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor)
from ophyd.positioner import PositionerBase
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from epics import caput, caget, cainfo
from functools import partial
from pathlib import Path
import sys
from cycler import cycler

try:
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
    )
    from diffractometer_controls.tomography_scan_parameters import (
        extend_tomography_angles_deg,
    )
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
    )
    from diffractometer_controls.tomography_scan_parameters import (
        extend_tomography_angles_deg,
    )

transfer_time_per_bytes = 4.1203007518796994e-08 # transfer speed in seconds per byte testing on the ASI294MM Pro


def _plan_estimation_context():
    return build_estimation_context(
        caget_func=caget,
        transfer_time_per_bytes=transfer_time_per_bytes,
    )


def _collect_tomo_motor_names():
    """Collect only tomography motors for the tomo_scan dropdown.

    Restrict the list to the two tomography theta motors while keeping their
    normal dotted device names.
    """
    g = globals()
    names = []
    candidates = [
        ("stage1", "theta"),
        ("stage2", "theta"),
    ]
    for parent_name, attr in candidates:
        parent = g.get(parent_name)
        if parent is None or not hasattr(parent, attr):
            continue
        motor = getattr(parent, attr)
        if isinstance(motor, PositionerBase):
            names.append(f"{parent_name}.{attr}")
    return names


def _collect_imaging_detector_names():
    """Collect supported imaging detectors for Queue Server dropdowns."""
    names = []
    for name in ("cam1", "sim_focus_cam"):
        detector = globals().get(name, None)
        if detector is None:
            continue
        if not callable(getattr(detector, "read", None)):
            continue
        if not callable(getattr(detector, "trigger", None)):
            continue
        if not hasattr(detector, "cam") or not hasattr(detector, "tiff1"):
            continue
        names.append(name)
    return names


_CAMERA_ADVANCED_SCAN_ATTRS = ("acquire_time", "gain", "offset")


def _camera_advanced_axis_alias(detector_name, attr):
    """Return the Queue Server-safe top-level alias for a camera control."""
    return f"{detector_name}_{attr}"


def _collect_imaging_advanced_scan_names():
    """Collect explicitly registered camera configuration scan axes.

    These aliases intentionally avoid resolving ``detector.cam`` here.  The
    live camera has optional AreaDetector records that must not be eagerly
    instantiated while Queue Server builds its startup inventory.
    """
    names = []
    g = globals()
    for detector_name in _collect_imaging_detector_names():
        for attr in _CAMERA_ADVANCED_SCAN_ATTRS:
            axis_name = _camera_advanced_axis_alias(detector_name, attr)
            signal = g.get(axis_name)
            if (
                callable(getattr(signal, "set", None))
                and callable(getattr(signal, "read", None))
            ):
                names.append(axis_name)
    return names


def _scan_axis_position(axis):
    """Return a scan axis value for motors and signal-like configuration axes."""
    if hasattr(axis, "position"):
        return axis.position
    getter = getattr(axis, "get", None)
    if callable(getter):
        return getter()
    raise TypeError(f"Scan axis {axis!r} has neither 'position' nor 'get()'")


def _scan_axis_units(axis):
    """Return display units without requiring a motor-specific ``egu`` attribute."""
    units = getattr(axis, "egu", None)
    if units:
        return str(units)
    try:
        units = (axis.metadata or {}).get("units", "")
    except Exception:
        units = ""
    return str(units or "")


def _camera_scan_axis_name(detector, axis):
    """Return the camera attribute scanned by ``axis``, or ``None``."""
    camera = getattr(detector, "cam", None)
    for attr in _CAMERA_ADVANCED_SCAN_ATTRS:
        if axis is getattr(camera, attr, None):
            return attr
    return None

def _one_nd_step_repeat(
    detectors,
    step,
    pos_cache,
    take_reading=None,
    num_exposures=1,
    on_exposure_start=None,
    on_exposure_success=None,
    on_step_start=None,
    on_step_success=None,
):
    """
    Inner loop of an N-dimensional step scan

    This is the default function for ``per_step`` param`` in ND plans.

    Parameters
    ----------
    detectors : list or tuple
        devices to read
    step : dict
        mapping motors to positions in this step
    pos_cache : dict
        mapping motors to their last-set positions
    take_reading : plan, optional
        function to do the actual acquisition ::

           def take_reading(dets, name='primary'):
                yield from ...

        Callable[List[OphydObj], Optional[str]] -> Generator[Msg], optional

        Defaults to `trigger_and_read`

    Yields
    ------
    msg : Msg
    """

    def exposure():
        yield from bps.trigger_and_read(list(detectors) + list(motors))


    # take_reading = trigger_and_read if take_reading is None else take_reading
    motors = step.keys()
    step_t0 = time.monotonic()
    if callable(on_step_start):
        cb = on_step_start()
        if cb is not None:
            yield from cb
    yield from bps.move_per_step(step, pos_cache)
    yield from _repeater_with_checkpoints(
        num_exposures,
        exposure,
        on_iteration_start=on_exposure_start,
        on_iteration_success=on_exposure_success,
    )  # type: ignore  # Movable issue
    if callable(on_step_success):
        cb = on_step_success(num_exposures, max(0.0, time.monotonic() - step_t0))
        if cb is not None:
            yield from cb


def _repeater_with_checkpoints(
    n,
    gen_func,
    *args,
    on_iteration_start=None,
    on_iteration_success=None,
    **kwargs,
):
    """Repeat a plan with a checkpoint before each repetition.

    A deferred pause is processed at checkpoints, so this creates one pause
    boundary per exposure.
    """
    if n is None:
        i = 0
        while True:
            yield from bps.checkpoint()
            if callable(on_iteration_start):
                cb = on_iteration_start(i)
                if cb is not None:
                    yield from cb
            yield from gen_func(*args, **kwargs)
            if callable(on_iteration_success):
                cb = on_iteration_success(i)
                if cb is not None:
                    yield from cb
            i += 1
    else:
        for i in range(n):
            yield from bps.checkpoint()
            if callable(on_iteration_start):
                cb = on_iteration_start(i)
                if cb is not None:
                    yield from cb
            yield from gen_func(*args, **kwargs)
            if callable(on_iteration_success):
                cb = on_iteration_success(i)
                if cb is not None:
                    yield from cb

def _inner_product_custom(args, num:int = None, step_size:float = None, offset:float = 0, endpoint=True):
    """Scan over one multi-motor trajectory.

    Parameters
    ----------
    num : integer
        number of steps
    args : list of {Positioner, Positioner, int}
        patterned like (``motor1, start1, stop1, ..., motorN, startN, stopN``)
        Motors can be any 'setable' object (motor, temp controller, etc.)

    Returns
    -------
    cyc : cycler
    """
    if len(args) % 3 != 0:
        raise ValueError("Wrong number of positional arguments for 'inner_product'")

    cyclers = []
    for (
        motor,
        start,
        stop,
    ) in partition(3, args):
        if num is not None:
            steps = np.linspace(start + offset, stop, num=num, endpoint=endpoint)
        elif step_size is not None:
            steps = np.arange(start + offset, stop + step_size/2*endpoint, step_size)
        else:
            raise ValueError("Must provide either 'num' or 'step_size'")
        c = cycler(motor, steps)
        cyclers.append(c)
    return functools.reduce(operator.add, cyclers)


def _ensure_detector_temperature(detectors, target_temperature=-15, threshold=-10, poll_interval=0.5):
    """
    Ensure all detectors are at or below the target temperature.

    Parameters
    ----------
    detectors : list
        List of detector objects to check and adjust.
    target_temperature : float
        The temperature to set for detectors that are above the threshold.
    threshold : float
        The temperature below which the detectors are considered ready.
    poll_interval : float
        Time (in seconds) to wait between temperature checks.

    Yields
    ------
    Bluesky plan messages to set and wait for detector temperatures.
    """
    # First loop: Set the target temperature for all detectors
    for det in detectors:
        if det.cam.temperature_actual.get() > threshold:
            print("****************************************")
            print(f"{det.name} temperature is currently {det.cam.temperature_actual.get()} C and above {threshold} C")         
            if det.cam.temperature.get() > threshold:
                print(f"Setting {det.name} target temperature to {target_temperature} C.")
                yield from bps.mov(det.cam.temperature, target_temperature)
            else:
                print(f"{det.name} temperature set below threshold, will wait for it to cool down.")
        print("****************************************")

    # Second loop: Wait for all detectors to reach the desired temperature
    print("Waiting for all detectors to cool down to the threshold...")
    all_reached = False
    while not all_reached:
        all_reached = True
        for det in detectors:
            current_temp = det.cam.temperature_actual.get()
            if current_temp > threshold:
                all_reached = False
        yield from bps.sleep(poll_interval)

    print("All detectors have reached the desired temperature.")


def _reset_detector_array_counter(detectors):
    """Reset AreaDetector ArrayCounter to zero for all listed detectors."""
    for det in detectors:
        try:
            sig = getattr(det.cam, "array_counter", None)
            if sig is None:
                continue
            yield from bps.mov(sig, 0)
        except Exception:
            # Keep plan execution robust if a detector does not expose this PV.
            continue

@parameter_annotation_decorator({
    "parameters": {
        "detector": {
            "annotation": "ImagingDetectors",
            "default": "cam1",
            "description": "Imaging detector (default: cam1)",
            "devices": {"ImagingDetectors": _collect_imaging_detector_names()},
            "convert_device_names": True,
        },
        "motor": {
            "annotation": "typing.Union[str, Motors]",
            "description": "Motor to scan (must be movable)",
            "devices": {"Motors": _collect_tomo_motor_names()},
            "convert_device_names": True,
        }
    }
})
def tomo_scan(file_name:str, 
              file_dir:str, 
              motor, 
              detector=cam1,
              exposure_time:float = None,
              num_projections:int = None,
              angle_step_size:float = None,
              start_angle:float = 0, 
              stop_angle:float = 180,
              num_exposures:int = 1,
              include_stop_angle:bool = True,
              tilt_correction_projections:int = 0,
              full_360_scan:bool = False,
              return_to_start:bool = True, 
              check_temperature:bool = True,
              md:dict = None):
    """Acquire an inclusive 0–180° tomography base set by default.

    ``num_projections`` describes the base set, including both endpoints.
    ``tilt_correction_projections`` adds a sparse set of exact 180° pairs in
    the second half and always includes 360°. ``full_360_scan`` instead adds
    every second-half pair at the same angular density as the base set.
    """
    file_name = str(file_name).strip().replace(" ","_").replace("__","_")
    file_dir = str(file_dir).strip().replace(" ","_").replace("__","_")

    # Resolve and validate the complete trajectory before changing detector or
    # motor state.
    tilt_correction_projections = int(tilt_correction_projections)
    if tilt_correction_projections < 0:
        raise ValueError("tilt_correction_projections must be non-negative")
    base_positions, base_num_projections_calc, angle_step_size_calc, base_stop_angle = (
        _tomo_positions_from_num_or_step_size(
            start_angle,
            stop_angle,
            num_projections=num_projections,
            angle_step_size=angle_step_size,
            include_stop=include_stop_angle,
        )
    )
    if bool(full_360_scan) or tilt_correction_projections > 0:
        positions = np.asarray(
            extend_tomography_angles_deg(
                base_positions,
                tilt_correction_projections=tilt_correction_projections,
                full_360_scan=bool(full_360_scan),
            ),
            dtype=float,
        )
    else:
        positions = np.asarray(base_positions, dtype=float)
    num_projections_calc = int(len(positions))
    actual_stop_angle = float(positions[-1])
    actual_tilt_correction_projections = (
        base_num_projections_calc - 1
        if bool(full_360_scan)
        else tilt_correction_projections
    )
    detector = [detector]

    # Ensure temperature is checked within the main plan
    if check_temperature:
        yield from _ensure_detector_temperature(detectors=detector, target_temperature=-15, threshold=-10)

    old_exposure_time = detector[0].cam.acquire_time.get()

    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    estimate = estimate_plan_runtime(
        "tomo_scan",
        kwargs={
            "exposure_time": exposure_time,
            "num_projections": base_num_projections_calc,
            "angle_step_size": angle_step_size_calc,
            "start_angle": start_angle,
            "stop_angle": base_stop_angle,
            "num_exposures": num_exposures,
            "include_stop_angle": include_stop_angle,
            "tilt_correction_projections": tilt_correction_projections,
            "full_360_scan": full_360_scan,
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or (num_exposures * num_projections_calc))

    

    print("#===============#")
    print(
        f"Starting tomography scan with {base_num_projections_calc} base projections "
        f"from {start_angle} to {base_stop_angle}, plus "
        f"{actual_tilt_correction_projections} paired second-half projections "
        f"through {actual_stop_angle}, for {num_projections_calc} total positions "
        f"at {num_exposures} exposures per position."
    )
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds")
    print(f"and finish at {datetime.now() + timedelta(seconds=total_time)}.")
    print("#===============#")

    caput("4dh4:TS:RotationStart",start_angle)
    caput("4dh4:TS:RotationStop",actual_stop_angle)
    caput("4dh4:TS:NumAngles",num_projections_calc)
    caput("4dh4:TS:RotationStep", angle_step_size_calc)


    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    md = md or {}
    _md = {
        "file_name": file_name,
        "file_dir": file_dir,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detector],
        "plan_args": {
            "detectors": [det.name for det in detector],
            # "num": num,
            # "args": md_args,
        },
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "num_exposures": num_exposures,
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "experiment_type": "imaging",
        "plan_name": "tomo_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(
            motor=motor.name,
            start_angle=start_angle,
            stop_angle=base_stop_angle,
            num_projections=base_num_projections_calc,
            total_angular_positions=num_projections_calc,
            angle_step_size=angle_step_size_calc,
            include_stop_angle=include_stop_angle,
            tilt_correction_projections=actual_tilt_correction_projections,
            full_360_scan=bool(full_360_scan),
        ),
        "motors": [motor.name],
    }
    _md.update(md)
    x_fields = _set_scan_motor_metadata(_md, [motor])
    

    # def background_exposure(frame_type: str="dark"):
    #     if frame_type == "dark":
    #         for det in detector:
    #             yield from bps.mov(det.cam.frame_type, 1)
    #     elif frame_type == "flat":
    #         for det in detector:
    #             yield from bps.mov(det.cam.frame_type, 2)
    #     yield from bps.trigger_and_read(detector, name=frame_type)

    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )
        def _on_step_start():
            yield from progress.mark_started()

        def _on_step_success(step_units, step_elapsed_s):
            yield from progress.on_units_success(unit_count=step_units, elapsed_s=step_elapsed_s)

        yield from _reset_detector_array_counter(detector)

        for det in detector:
            det.tiff1.file_name.put(file_name)
            det.tiff1.folder_name.put(file_dir)
        yield from _stage_devices_once(detector + [motor])

        # print("Close shutter then press Resume to take the dark field")
        # yield from bps.checkpoint()
        # yield from bps.pause()
        # yield from bps.repeater(num_dark,background_exposure, frame_type="dark")

        # print("Open shutter and remove the sample then press Resume to take the flat field")
        # yield from bps.checkpoint()
        # yield from bps.pause()
        # yield from bps.repeater(num_white,background_exposure, frame_type="flat")

        # for det in detector:
        #     yield from bps.mov(det.cam.frame_type, 0)
        
        pos_cache = defaultdict(lambda: None)
        # if num_projections is not None:
        scan_cycler = cycler(motor, positions)
                        

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(scan_cycler):
                yield from _one_nd_step_repeat(
                    detector,
                    step,
                    pos_cache,
                    num_exposures=num_exposures,
                    on_step_start=_on_step_start,
                    on_step_success=_on_step_success,
                )

        # print("Replace the sample, open the shutter, and press Resume to start the scan")
        # yield from bps.checkpoint()
        # yield from bps.pause()
        yield from inner_scan_nd()

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)

        if return_to_start:
            yield from bps.mv(motor, start_angle)

    return(yield from main_plan())


@parameter_annotation_decorator({
    "parameters": {
        "detector": {
            "annotation": "ImagingDetectors",
            "default": "cam1",
            "description": "Imaging detector (default: cam1)",
            "devices": {"ImagingDetectors": _collect_imaging_detector_names()},
            "convert_device_names": True,
        }
    }
})
def imaging(
            file_name:str, 
            file_dir:str,
            detector=cam1, 
            exposure_time:float = None,
            num_exposures:int = 1,
            gain:int = None,
            offset:int = None,
            check_temperature:bool = True,
            md:dict = None
            ):
    '''
    Tomography scan that performs dark field scans, flat field scans, and then the actual tomography scan.
    '''

    file_name = str(file_name).strip().replace(" ","_").replace("__","_")
    file_dir = str(file_dir).strip().replace(" ","_").replace("__","_")

    detector = [detector]

    # Ensure temperature is checked within the main plan
    if check_temperature:
        yield from _ensure_detector_temperature(detectors=detector, target_temperature=-15, threshold=-10)


    old_exposure_time = detector[0].cam.acquire_time.get()
    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    estimate = estimate_plan_runtime(
        "imaging",
        kwargs={
            "exposure_time": exposure_time,
            "num_exposures": num_exposures,
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or num_exposures)

    print("#===============#")
    print(f"Starting imaging with {num_exposures} exposures.")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The measurement time is estimated to take {hours} hours, {minutes} minutes, and {seconds} seconds")
    print(f"and finish at {datetime.now() + timedelta(seconds=total_time)}.")
    print("#===============#")
    
    old_gain = detector[0].cam.gain.get()
    old_offset = detector[0].cam.offset.get()
    if gain is not None:
        for det in detector:
            yield from bps.mov(det.cam.gain, gain)
    if offset is not None:
        for det in detector:
            yield from bps.mov(det.cam.offset, offset)

    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    md = md or {}
    _md = {
        "file_name": file_name,
        "file_dir": file_dir,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detector],
        "plan_args": {
            "detectors": [det.name for det in detector],
            # "num": num,
            # "args": md_args,
        },
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "num_exposures": num_exposures,
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "experiment_type": "imaging",
        "plan_name": "imaging",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
    }
    _md.update(md)
    

    def exposure():
        yield from bps.trigger_and_read(detector)

    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )
        yield from _reset_detector_array_counter(detector)

        for det in detector:
            det.tiff1.file_name.put(file_name)
            det.tiff1.folder_name.put(file_dir)
        yield from _stage_devices_once(detector)

        yield from _repeater_with_checkpoints(
            num_exposures,
            exposure,
            on_iteration_start=progress.on_unit_start,
            on_iteration_success=progress.on_unit_success,
        )

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)
        yield from bps.mov(detector[0].cam.gain, old_gain)
        yield from bps.mov(detector[0].cam.offset, old_offset)

    return(yield from main_plan())



def _imaging_scan_parameter_annotation():
    """Return the shared Queue Server annotations for imaging scan axes."""
    return {
        "parameters": {
            "detector": {
                "annotation": "ImagingDetectors",
                "default": "cam1",
                "description": "Imaging detector (default: cam1)",
                "devices": {"ImagingDetectors": _collect_imaging_detector_names()},
                "convert_device_names": True,
            },
            "motor": {
                "annotation": "typing.Union[str, Motors, Cam_advanced]",
                "description": "Motor or supported camera setting to scan",
                "devices": {
                    "Motors": _collect_movable_names(),
                    # Queue Server group names must be valid Python identifiers.
                    "Cam_advanced": _collect_imaging_advanced_scan_names(),
                },
                "convert_device_names": True,
            },
        },
    }


def _normalise_discrete_scan_positions(positions):
    """Validate and normalize an explicit list of imaging scan positions."""
    try:
        values = np.asarray(list(positions), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a non-empty sequence of numbers") from exc
    if values.ndim != 1 or len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("positions must be a non-empty sequence of finite numbers")
    return values


def _resolve_imaging_scan_positions(
    scan_kind,
    *,
    original_position,
    start_pos=None,
    stop_pos=None,
    start_relative=None,
    stop_relative=None,
    positions=None,
    step_size=None,
    num_steps=None,
):
    """Resolve each public imaging scan form to absolute axis positions."""
    if scan_kind == "linear":
        scan_positions, num_steps_calc, step_size_calc, stop_pos_calc = (
            _scan_positions_from_num_or_step_size(
                start_pos,
                stop_pos,
                num_steps=num_steps,
                step_size=step_size,
            )
        )
        return scan_positions, {
            "start_pos": float(start_pos),
            "stop_pos": float(stop_pos_calc),
            "step_size": float(step_size_calc),
            "num_steps": int(num_steps_calc),
        }

    if scan_kind == "discrete":
        scan_positions = _normalise_discrete_scan_positions(positions)
        step_size_calc = (
            float(scan_positions[1] - scan_positions[0])
            if len(scan_positions) > 1
            else None
        )
        return scan_positions, {
            "positions": scan_positions.tolist(),
            "start_pos": float(scan_positions[0]),
            "stop_pos": float(scan_positions[-1]),
            "step_size": step_size_calc,
            "num_steps": int(len(scan_positions)),
        }

    if scan_kind == "relative":
        try:
            absolute_start = float(original_position) + float(start_relative)
            absolute_stop = float(original_position) + float(stop_relative)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_relative and stop_relative must be numeric") from exc
        scan_positions, num_steps_calc, step_size_calc, stop_pos_calc = (
            _scan_positions_from_num_or_step_size(
                absolute_start,
                absolute_stop,
                num_steps=num_steps,
                step_size=step_size,
            )
        )
        return scan_positions, {
            "start_relative": float(start_relative),
            "stop_relative": float(stop_relative),
            "start_pos": float(absolute_start),
            "stop_pos": float(stop_pos_calc),
            "step_size": float(step_size_calc),
            "num_steps": int(num_steps_calc),
        }

    raise ValueError(f"Unsupported imaging scan kind: {scan_kind!r}")


def _run_imaging_scan_impl(
            file_name:str, 
            file_dir:str, 
            motor, 
            *,
            plan_name,
            scan_kind,
            start_pos:float = None,
            stop_pos:float = None,
            start_relative:float = None,
            stop_relative:float = None,
            positions=None,
            step_size:float = None,
            num_steps:int = None,
            detector=None,
            exposure_time:float = None,
            num_exposures:int = 1,
            gain:int = None,
            offset:int = None,
            return_to_original_position:bool = True,
            check_temperature:bool = True,
            md:dict = None):
    '''
    General scan for the imaging detector system.
    '''

    file_name = str(file_name).strip().replace(" ","_").replace("__","_")
    file_dir = str(file_dir).strip().replace(" ","_").replace("__","_")

    detector = [detector]
    original_pos = _scan_axis_position(motor)
    camera_axis = _camera_scan_axis_name(detector[0], motor)
    if camera_axis == "acquire_time" and exposure_time is not None:
        raise ValueError("Do not pass exposure_time when scanning detector.cam.acquire_time")
    if camera_axis == "gain" and gain is not None:
        raise ValueError("Do not pass gain when scanning detector.cam.gain")
    if camera_axis == "offset" and offset is not None:
        raise ValueError("Do not pass offset when scanning detector.cam.offset")

    # Ensure temperature is checked within the main plan
    if check_temperature:
        yield from _ensure_detector_temperature(detectors=detector, target_temperature=-15, threshold=-10)

    old_exposure_time = detector[0].cam.acquire_time.get()

    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    old_gain = detector[0].cam.gain.get()
    old_offset = detector[0].cam.offset.get()
    if gain is not None:
        for det in detector:
            yield from bps.mov(det.cam.gain, gain)
    if offset is not None:
        for det in detector:
            yield from bps.mov(det.cam.offset, offset)

    positions, scan_details = _resolve_imaging_scan_positions(
        scan_kind,
        original_position=original_pos,
        start_pos=start_pos,
        stop_pos=stop_pos,
        start_relative=start_relative,
        stop_relative=stop_relative,
        positions=positions,
        step_size=step_size,
        num_steps=num_steps,
    )
    num_steps_calc = scan_details["num_steps"]
    step_size_calc = scan_details["step_size"]
    stop_pos_calc = scan_details["stop_pos"]

    estimate_kwargs = {
        "motor": getattr(motor, "name", motor),
        "scan_axis": camera_axis,
        "exposure_time": exposure_time,
        "num_exposures": num_exposures,
    }
    if scan_kind == "discrete":
        estimate_kwargs["positions"] = scan_details["positions"]
    elif scan_kind == "relative":
        estimate_kwargs.update(
            start_relative=scan_details["start_relative"],
            stop_relative=scan_details["stop_relative"],
            step_size=step_size_calc,
            num_steps=num_steps_calc,
            # The core has read the axis at run start, so this gives the
            # worker an exact estimate even when the queued relative plan
            # could not know the starting value yet.
            resolved_positions=positions.tolist(),
        )
    else:
        estimate_kwargs.update(
            start_pos=scan_details["start_pos"],
            stop_pos=stop_pos_calc,
            step_size=step_size_calc,
            num_steps=num_steps_calc,
        )
    estimate = estimate_plan_runtime(
        plan_name,
        kwargs=estimate_kwargs,
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or (num_exposures * num_steps_calc))
    progress_unit_durations_s = None
    if camera_axis == "acquire_time" and total_units > 0:
        # The scan visits each exposure value in order.  Preserve that known
        # schedule for the live ETA instead of inferring later, longer
        # exposures from the shorter points that have already completed.
        exposure_sum_s = float(sum(max(0.0, float(position)) for position in positions))
        exposure_total_s = exposure_sum_s * int(num_exposures)
        transfer_per_frame_s = max(
            0.0,
            (total_time - exposure_total_s) / float(total_units),
        )
        progress_unit_durations_s = tuple(
            max(0.0, float(position)) + transfer_per_frame_s
            for position in positions
            for _ in range(int(num_exposures))
        )

    print("#===============#")
    axis_name = getattr(motor, "name", type(motor).__name__)
    axis_units = _scan_axis_units(motor)
    units_suffix = f" {axis_units}" if axis_units else ""
    if scan_kind == "discrete":
        scan_description = f"at {num_steps_calc} specified positions"
    elif scan_kind == "relative":
        scan_description = (
            f"from {scan_details['start_relative']} to {scan_details['stop_relative']} "
            f"relative to {original_pos}"
        )
    else:
        scan_description = f"from {scan_details['start_pos']} to {stop_pos_calc}"
    step_description = (
        f" of {step_size_calc}{units_suffix}" if step_size_calc is not None else ""
    )
    print(
        f"Starting scan of {axis_name} {scan_description} \n"
        f"in {num_steps_calc} steps{step_description} with {num_exposures} "
        "exposures at each position."
    )
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds")
    print(f"and finish at {datetime.now() + timedelta(seconds=total_time)}.")
    print("#===============#")


    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    md = md or {}
    _md = {
        "file_name": file_name,
        "file_dir": file_dir,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detector],
        "plan_args": {
            "detectors": [det.name for det in detector],
            # "num": num,
            # "args": md_args,
        },
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "num_exposures": num_exposures,
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "experiment_type": "imaging",
        "plan_name": plan_name,
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(
            motor=motor.name,
            **scan_details,
            num_exposures=num_exposures,
        ),
        "motors": [motor.name],
    }
    _md.update(md)
    x_fields = _set_scan_motor_metadata(_md, [motor])
    

    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
            planned_unit_durations_s=progress_unit_durations_s,
        )
        def _on_step_start():
            yield from progress.mark_started()

        def _on_step_success(step_units, step_elapsed_s):
            yield from progress.on_units_success(unit_count=step_units, elapsed_s=step_elapsed_s)

        yield from _reset_detector_array_counter(detector)

        for det in detector:
            det.tiff1.file_name.put(file_name)
            det.tiff1.folder_name.put(file_dir)
        yield from _stage_devices_once(detector + [motor])
        
        pos_cache = defaultdict(lambda: None)
        scan_cycler = cycler(motor, positions)

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(scan_cycler):
                yield from _one_nd_step_repeat(
                    detector,
                    step,
                    pos_cache,
                    num_exposures=num_exposures,
                    on_step_start=_on_step_start,
                    on_step_success=_on_step_success,
                )

        # print("Replace the sample, open the shutter, and press Resume to start the scan")
        # yield from bps.checkpoint()
        # yield from bps.pause()
        yield from inner_scan_nd()

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)
        yield from bps.mov(detector[0].cam.gain, old_gain)
        yield from bps.mov(detector[0].cam.offset, old_offset)
        
        if return_to_original_position:
            yield from bps.mv(motor, original_pos)

    return(yield from main_plan())


def _make_imaging_scan_runner(implementation):
    """Hide the generator implementation from Queue Server plan discovery."""
    def runner(*args, **kwargs):
        return implementation(*args, **kwargs)

    return runner


# Queue Server treats every generator function in the startup namespace as a
# public plan. Keep the implementation in a closure so only the three
# explicitly annotated public wrappers below are discovered.
_run_imaging_scan = _make_imaging_scan_runner(_run_imaging_scan_impl)
del _run_imaging_scan_impl


@parameter_annotation_decorator(_imaging_scan_parameter_annotation())
def imaging_scan(
    file_name: str,
    file_dir: str,
    motor,
    start_pos: float,
    stop_pos: float,
    step_size: float = None,
    num_steps: int = None,
    detector=cam1,
    exposure_time: float = None,
    num_exposures: int = 1,
    gain: int = None,
    offset: int = None,
    return_to_original_position: bool = True,
    check_temperature: bool = True,
    md: dict = None,
):
    """Image at inclusive absolute positions along a motor or camera setting."""
    return (
        yield from _run_imaging_scan(
            file_name, file_dir, motor, plan_name="imaging_scan", scan_kind="linear",
            start_pos=start_pos, stop_pos=stop_pos, step_size=step_size,
            num_steps=num_steps, detector=detector, exposure_time=exposure_time,
            num_exposures=num_exposures, gain=gain, offset=offset,
            return_to_original_position=return_to_original_position,
            check_temperature=check_temperature, md=md,
        )
    )


@parameter_annotation_decorator(_imaging_scan_parameter_annotation())
def imaging_scan_discrete(
    file_name: str,
    file_dir: str,
    motor,
    positions: list[float],
    detector=cam1,
    exposure_time: float = None,
    num_exposures: int = 1,
    gain: int = None,
    offset: int = None,
    return_to_original_position: bool = True,
    check_temperature: bool = True,
    md: dict = None,
):
    """Image at the exact, ordered values in ``positions``."""
    return (
        yield from _run_imaging_scan(
            file_name, file_dir, motor, plan_name="imaging_scan_discrete",
            scan_kind="discrete", positions=positions, detector=detector,
            exposure_time=exposure_time, num_exposures=num_exposures, gain=gain,
            offset=offset, return_to_original_position=return_to_original_position,
            check_temperature=check_temperature, md=md,
        )
    )


@parameter_annotation_decorator(_imaging_scan_parameter_annotation())
def imaging_scan_rel(
    file_name: str,
    file_dir: str,
    motor,
    start_relative: float,
    stop_relative: float,
    step_size: float = None,
    num_steps: int = None,
    detector=cam1,
    exposure_time: float = None,
    num_exposures: int = 1,
    gain: int = None,
    offset: int = None,
    return_to_original_position: bool = True,
    check_temperature: bool = True,
    md: dict = None,
):
    """Image over offsets from the axis value measured when the run begins."""
    return (
        yield from _run_imaging_scan(
            file_name, file_dir, motor, plan_name="imaging_scan_rel",
            scan_kind="relative", start_relative=start_relative,
            stop_relative=stop_relative, step_size=step_size, num_steps=num_steps,
            detector=detector, exposure_time=exposure_time,
            num_exposures=num_exposures, gain=gain, offset=offset,
            return_to_original_position=return_to_original_position,
            check_temperature=check_temperature, md=md,
        )
    )
