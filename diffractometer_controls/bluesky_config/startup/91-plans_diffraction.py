# import bluesky.plans
import bluesky.plan_patterns
import bluesky.plans as bp
from bluesky.plans import scan, count, grid_scan, rel_scan, rel_grid_scan
from bluesky_queueserver import parameter_annotation_decorator

# import bluesky.plan_stubs
import bluesky.plan_stubs as bps
# from bluesky.plan_stubs import *
from bluesky import plan_patterns, utils
from collections import defaultdict

import bluesky.preprocessors
import bluesky.preprocessors as bpp
import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from epics import caput, caget, cainfo
from functools import partial

import functools
import operator
from cycler import cycler
from pathlib import Path
import sys
from datetime import datetime, timedelta
import time

try:
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
    )
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
    )

try:
    # cytools is a drop-in replacement for toolz, implemented in Cython
    from cytools import partition
except ImportError:
    from toolz import partition


frame_type_sig = EpicsSignal("4dh4:TS:FrameType", name="frame_type_sig")

# monitor_and_count = bpp.monitor_during_decorator([he3psd0.counts])(bp.count)


def _collect_diffraction_detector_names():
    """Collect HE3 diffraction detector names for Queue Server dropdowns."""
    required_components = {"acquire", "acquire_time", "nbins", "soft_lld", "counts", "total_counts"}
    names = []
    g = globals()
    for var, obj in list(g.items()):
        if var.startswith("_"):
            continue
        try:
            if not isinstance(obj, Device):
                continue
            component_names = set(getattr(obj, "component_names", ()))
            if required_components.issubset(component_names):
                names.append(var)
        except Exception:
            continue

    seen = set()
    out = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _plan_estimation_context():
    return build_estimation_context(caget_func=caget)


@parameter_annotation_decorator(
    {
        "parameters": {
            "detectors": {
                "annotation": "typing.Union[typing.List[DiffractionDetectors], DiffractionDetectors]",
                "description": "Diffraction detector or detectors to read",
                "devices": {"DiffractionDetectors": _collect_diffraction_detector_names()},
                "convert_device_names": True,
            }
        }
    }
)
def count_he3(
                title:str,
                sample:str = "",
                gauge_volume:str = "",
                *,
                detectors, 
                acquire_time:float = None,
                num = 1, 
                md = None
                
            ):
    """
    Take one or more readings from detectors.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    num : integer, optional
        number of readings to take; default is 1

        If None, capture data until canceled

    md : dict, optional
        metadata
    """
    if not isinstance(detectors,list):
        detectors = [detectors]
    if num is None:
        num_intervals = None
    else:
        num_intervals = num - 1

    old_acquire_time = detectors[0].acquire_time.get()
    if acquire_time is not None:
        for det in detectors:
            yield from bps.mov(det.acquire_time, acquire_time)

    estimate = estimate_plan_runtime(
        "count_he3",
        kwargs={
            "acquire_time": detectors[0].acquire_time.get(),
            "num": num,
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or (num or 0))

    _md = {
        "title": title,
        "sample": sample,
        "gauge_volume": gauge_volume,
        "detectors": [det.name for det in detectors],
        "plan_args": {
            "detectors": [det.name for det in detectors],
            "acquire_time": detectors[0].acquire_time.get(),
            "num": num, 
        },
        "det_config": {
            "ophyd_defs":list(map(repr, detectors)),
            "acquire_time": detectors[0].acquire_time.get(),
            "nbins": detectors[0].nbins.get(),
            "soft_lld": detectors[0].soft_lld.get()
        },
        "num_points": num,
        "num_intervals": num_intervals,
        "estimated_total_time_s": total_time,
        "estimated_total_units": total_units,
        "experiment_type": "diffraction",
        "plan_name": "count_he3",
        "hints": {},
    }
    _md.update(md or {})
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])

    predeclare = os.environ.get("BLUESKY_PREDECLARE", False)

    # @bpp.monitor_during_decorator([detectors[0].counts])
    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner_count():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )
        try:
            if predeclare:
                yield from bps.declare_stream(*detectors, name="primary")
            if num is None:
                i = 0
                while True:
                    yield from bps.checkpoint()
                    yield from progress.on_unit_start(i)
                    yield from bps.one_shot(detectors)
                    yield from progress.on_unit_success(i)
                    i += 1
            else:
                for i in range(num):
                    yield from bps.checkpoint()
                    yield from progress.on_unit_start(i)
                    yield from bps.one_shot(detectors)
                    yield from progress.on_unit_success(i)
        finally:
            if acquire_time is not None:
                for det in detectors:
                    yield from bps.mov(det.acquire_time, old_acquire_time)

    return (yield from inner_count())





@parameter_annotation_decorator(
    {
        "parameters": {
            "detectors": {
                "annotation": "typing.Union[typing.List[DiffractionDetectors], DiffractionDetectors]",
                "description": "Diffraction detector or detectors to read",
                "devices": {"DiffractionDetectors": _collect_diffraction_detector_names()},
                "convert_device_names": True,
            },
            "motor": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Motor to scan (must be movable)",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
        }
    }
)
def scan_he3( 
            title:str,
            sample:str = "",
            gauge_volume:str = "",
            *,
            detectors, 
            motor, 
            start_pos:float, 
            stop_pos:float, 
            step_size:float = None,
            num_steps:int = None,
            acquire_time:float = None,
            return_to_original_position:bool = True,
            md:dict = None
              ):
    '''
    General scan for the imaging detector system.
    '''
    original_pos = motor.position

    if not isinstance(detectors,list):
        detectors = [detectors]

    old_acquire_time = detectors[0].acquire_time.get()
    if acquire_time is not None:
        for det in detectors:
            yield from bps.mov(det.acquire_time, acquire_time)

    positions, num_steps_calc, step_size_calc, stop_pos_calc = _scan_positions_from_num_or_step_size(
        start_pos,
        stop_pos,
        num_steps=num_steps,
        step_size=step_size,
    )

    estimate = estimate_plan_runtime(
        "scan_he3",
        kwargs={
            "start_pos": start_pos,
            "stop_pos": stop_pos_calc,
            "step_size": step_size_calc,
            "num_steps": num_steps_calc,
            "acquire_time": detectors[0].acquire_time.get(),
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or num_steps_calc)

    print("#===============#")
    if title:
        print(f"Title: {title}")
    if sample:
        print(f"Sample: {sample}")
    if gauge_volume:
        print(f"Gauge volume: {gauge_volume}")
    print(
        f"Starting scan of {motor.name} from {start_pos} to {stop_pos_calc} "
        f"\nin {num_steps_calc} steps of {step_size_calc} {motor.egu}."
    )
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds.")
    print(f"and finish at {datetime.now() + timedelta(seconds=total_time)}.")
    print("#===============#")


    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    md = md or {}
    _md = {
        "title": title,
        "sample": sample,
        "gauge_volume": gauge_volume,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detectors],
        "plan_args": {
            "detectors": [det.name for det in detectors],
            "acquire_time": detectors[0].acquire_time.get(),
            "start_pos": start_pos,
            "stop_pos": stop_pos_calc,
            "step_size": step_size_calc,
            "num_steps": num_steps_calc,
        },
        "det_config": {
            "ophyd_defs":list(map(repr, detectors)),
            "acquire_time": detectors[0].acquire_time.get(),
            "nbins": detectors[0].nbins.get(),
            "soft_lld": detectors[0].soft_lld.get()
        },
        "experiment_type": "diffraction",
        "plan_name": "scan_he3",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, start_pos=start_pos, stop_pos=stop_pos_calc, step_size=step_size_calc, num_steps=num_steps_calc),  # noqa: C408
        "motors": [motor.name],
    }
    _md.update(md)
    x_fields = _set_scan_motor_metadata(_md, [motor])
    
    # @bpp.monitor_during_decorator([detector[0]])
    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )

        yield from _stage_devices_once(detectors + [motor])
        
        pos_cache = defaultdict(lambda: None)
        scan_cycler = cycler(motor, positions)

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(scan_cycler):
                step_t0 = time.monotonic()
                yield from progress.mark_started()
                yield from bps.one_nd_step(detectors, step, pos_cache)
                yield from progress.on_units_success(
                    unit_count=1,
                    elapsed_s=max(0.0, time.monotonic() - step_t0),
                )

        yield from inner_scan_nd()

        if return_to_original_position:
            yield from bps.mv(motor, original_pos)
        yield from bps.mov(detectors[0].acquire_time, old_acquire_time)

    return(yield from main_plan())


@parameter_annotation_decorator(
    {
        "parameters": {
            "detectors": {
                "annotation": "typing.Union[typing.List[DiffractionDetectors], DiffractionDetectors]",
                "description": "Diffraction detector or detectors to read",
                "devices": {"DiffractionDetectors": _collect_diffraction_detector_names()},
                "convert_device_names": True,
            },
            "motor1": {
                "annotation": "typing.Union[str, Motors]",
                "description": "First motor to scan in parallel",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
            "motor2": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Second motor to scan in parallel",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
        }
    }
)
def scan_parallel_he3( 
            title:str,
            sample:str = "",
            gauge_volume:str = "",
            *,
            detectors, 
            motor1, 
            start_pos1:float, 
            stop_pos1:float, 
            motor2, 
            start_pos2:float, 
            stop_pos2:float, 
            num_steps:int,
            acquire_time:float = None,
            return_to_original_position:bool = True,
            md:dict = None
              ):
    '''
    General scan for the imaging detector system.
    '''
    original_pos1 = motor1.position
    original_pos2 = motor2.position

    if not isinstance(detectors,list):
        detectors = [detectors]

    old_acquire_time = detectors[0].acquire_time.get()
    if acquire_time is not None:
        for det in detectors:
            yield from bps.mov(det.acquire_time, acquire_time)
    
    num_steps_calc = int(num_steps)
    if num_steps_calc < 2:
        raise ValueError("num_steps must be at least 2")
    step_size1 = (stop_pos1-start_pos1)/(num_steps_calc-1)
    step_size2 = (stop_pos2-start_pos2)/(num_steps_calc-1)
    estimate = estimate_plan_runtime(
        "scan_parallel_he3",
        kwargs={
            "num_steps": num_steps_calc,
            "acquire_time": acquire_time,
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or num_steps_calc)

    print("#===============#")
    print(f"Starting scan of {motor1.name} from {start_pos1} to {stop_pos1} \nin {num_steps_calc} steps of {step_size1} {motor1.egu}.")
    print(f"In parallel scanning {motor2.name} from {start_pos2} to {stop_pos2} \nin {num_steps_calc} steps of {step_size2} {motor2.egu}.")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds.")
    print("#===============#")


    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    # motor_names = motor.name
    md = md or {}
    _md = {
        "title": title,
        "sample": sample,
        "gauge_volume": gauge_volume,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detectors],
        "plan_args": {
            "detectors": [det.name for det in detectors],
            "acquire_time": detectors[0].acquire_time.get(),
        },
        "det_config": {
            "ophyd_defs":list(map(repr, detectors)),
            "acquire_time": detectors[0].acquire_time.get(),
            "nbins": detectors[0].nbins.get(),
            "soft_lld": detectors[0].soft_lld.get()
        },
        "experiment_type": "diffraction",
        "plan_name": "scan_parallel_he3",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor1=motor1.name, start_pos1=start_pos1, stop_pos1=stop_pos1, motor2=motor2.name, start_pos2=start_pos2, stop_pos2=stop_pos2, step_size1=step_size1, step_size2=step_size2, num_steps=num_steps_calc),  # noqa: C408
        "motors": [motor1.name, motor2.name],
    }
    _md.update(md)

    x_fields = _set_scan_motor_metadata(_md, [motor1, motor2])
    
    # @bpp.monitor_during_decorator([detectors[0]])
    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )

        yield from _stage_devices_once(detectors + [motor1, motor2])
        
        pos_cache = defaultdict(lambda: None)
        cycler1 = plan_patterns.inner_product(num=num_steps_calc, args=[motor1, start_pos1, stop_pos1])
        cycler2 = plan_patterns.inner_product(num=num_steps_calc, args=[motor2, start_pos2, stop_pos2])

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detectors, name="primary")
            for step1, step2 in list(zip(cycler1,cycler2)):
                # yield from bps.one_nd_step(detectors, step, pos_cache)
                step_t0 = time.monotonic()
                yield from progress.mark_started()
                yield from bps.move_per_step(step1, pos_cache)
                yield from bps.move_per_step(step2, pos_cache)
                yield from bps.trigger_and_read(list(detectors) + list([motor1, motor2]))
                yield from progress.on_units_success(
                    unit_count=1,
                    elapsed_s=max(0.0, time.monotonic() - step_t0),
                )

        yield from inner_scan_nd()

        if return_to_original_position:
            yield from bps.mv(motor1, original_pos1)
            yield from bps.mv(motor2, original_pos2)
        yield from bps.mov(detectors[0].acquire_time, old_acquire_time)

    return(yield from main_plan())


@parameter_annotation_decorator(
    {
        "parameters": {
            "detectors": {
                "annotation": "typing.Union[typing.List[DiffractionDetectors], DiffractionDetectors]",
                "description": "Diffraction detector or detectors to read",
                "devices": {"DiffractionDetectors": _collect_diffraction_detector_names()},
                "convert_device_names": True,
            },
            "motor": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Motor to scan through the requested positions",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
        }
    }
)
def scan_list_he3( 
            title:str,
            sample:str = "",
            gauge_volume:str = "",
            *,
            detectors, 
            motor, 
            position_list:list,
            acquire_time:float = None,
            return_to_original_position:bool = True,
            md:dict = None
              ):
    '''
    General scan for the imaging detector system.
    '''


    if not isinstance(detectors,list):
        detectors = [detectors]

    old_acquire_time = detectors[0].acquire_time.get()
    if acquire_time is not None:
        for det in detectors:
            yield from bps.mov(det.acquire_time, acquire_time)

    num_steps = len(position_list)
    estimate = estimate_plan_runtime(
        "scan_list_he3",
        kwargs={
            "position_list": position_list,
            "acquire_time": acquire_time,
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or num_steps)

    print("#===============#")
    print(f"Starting scan of {motor.name} through the following positions:")
    print(f"{position_list}")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds.")
    print("#===============#")

    original_pos = motor.position

    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    md = md or {}
    _md = {
        "title": title,
        "sample": sample,
        "gauge_volume": gauge_volume,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detectors],
        "plan_args": {
            "detectors": [det.name for det in detectors],
            "acquire_time": detectors[0].acquire_time.get(),
        },
        "det_config": {
            "ophyd_defs":list(map(repr, detectors)),
            "acquire_time": detectors[0].acquire_time.get(),
            "nbins": detectors[0].nbins.get(),
            "soft_lld": detectors[0].soft_lld.get()
        },
        "experiment_type": "diffraction",
        "plan_name": "scan_list_he3",
        "plan_pattern": "inner_list_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, position_list=position_list, num_steps=num_steps),  # noqa: C408
        "motors": [motor.name],
    }
    _md.update(md)
    x_fields = _set_scan_motor_metadata(_md, [motor])
    
    # @bpp.monitor_during_decorator([detector[0].counts])
    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )

        yield from _stage_devices_once(detectors + [motor])
        
        pos_cache = defaultdict(lambda: None)
        cycler = plan_patterns.inner_list_product(args=[motor, position_list])

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(cycler):
                step_t0 = time.monotonic()
                yield from progress.mark_started()
                yield from bps.one_nd_step(detectors, step, pos_cache)
                yield from progress.on_units_success(
                    unit_count=1,
                    elapsed_s=max(0.0, time.monotonic() - step_t0),
                )

        yield from inner_scan_nd()

        if return_to_original_position:
            yield from bps.mv(motor, original_pos)
        yield from bps.mov(detectors[0].acquire_time, old_acquire_time)

    return(yield from main_plan())


@parameter_annotation_decorator(
    {
        "parameters": {
            "detectors": {
                "annotation": "typing.Union[typing.List[DiffractionDetectors], DiffractionDetectors]",
                "description": "Diffraction detector or detectors to read",
                "devices": {"DiffractionDetectors": _collect_diffraction_detector_names()},
                "convert_device_names": True,
            },
            "motor_outer": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Outer motor for the 2D scan",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
            "motor_inner": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Inner motor for the 2D scan",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
        }
    }
)
def scan2D_he3( 
            title:str,
            sample:str = "",
            gauge_volume:str = "",
            *,
            detectors, 
            motor_outer, 
            start_pos_outer:float, 
            stop_pos_outer:float, 
            step_size_outer:float,
            motor_inner,
            start_pos_inner:float,
            stop_pos_inner:float,
            step_size_inner:float,
            acquire_time:float = None,
            return_to_original_positions:bool = True,
            md:dict = None
              ):
    '''
    General scan for the imaging detector system.
    '''
    motors = [motor_outer, motor_inner]

    if not isinstance(detectors,list):
        detectors = [detectors]

    old_acquire_time = detectors[0].acquire_time.get()
    if acquire_time is not None:
        for det in detectors:
            yield from bps.mov(det.acquire_time, acquire_time)
    
    positions_outer = _step_size_positions(
        start_pos_outer,
        stop_pos_outer,
        step_size_outer,
        include_stop=True,
    )
    if len(positions_outer) < 2:
        raise ValueError("step_size_outer must produce at least 2 scan positions")
    num_steps_outer = int(len(positions_outer))
    step_size_outer_calc = float(positions_outer[1] - positions_outer[0])
    stop_pos_outer_calc = float(positions_outer[-1])

    positions_inner = _step_size_positions(
        start_pos_inner,
        stop_pos_inner,
        step_size_inner,
        include_stop=True,
    )
    if len(positions_inner) < 2:
        raise ValueError("step_size_inner must produce at least 2 scan positions")
    num_steps_inner = int(len(positions_inner))
    step_size_inner_calc = float(positions_inner[1] - positions_inner[0])
    stop_pos_inner_calc = float(positions_inner[-1])

    total_steps = num_steps_outer*num_steps_inner
    estimate = estimate_plan_runtime(
        "scan2D_he3",
        kwargs={
            "start_pos_outer": start_pos_outer,
            "stop_pos_outer": stop_pos_outer_calc,
            "step_size_outer": step_size_outer_calc,
            "start_pos_inner": start_pos_inner,
            "stop_pos_inner": stop_pos_inner_calc,
            "step_size_inner": step_size_inner_calc,
            "acquire_time": acquire_time,
        },
        context=_plan_estimation_context(),
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or total_steps)

    print("#===============#")
    print(f"Starting 2D outer scan of with \n{motor_outer.name} from {start_pos_outer} to {stop_pos_outer_calc} \nin {num_steps_outer} steps of {step_size_outer_calc} {motor_outer.egu}.")
    print(f"with the inner scan of \n{motor_inner.name} from {start_pos_inner} to {stop_pos_inner_calc} \nin {num_steps_inner} steps of {step_size_inner_calc} {motor_inner.egu}.")
    print(f"Total of {total_steps} steps with an acquire time of {acquire_time} seconds each.")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds.")
    print("#===============#")

    original_pos_inner = motor_inner.position
    original_pos_outer = motor_outer.position



    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    motor_names = [motor.name for motor in motors]
    md = md or {}
    _md = {
        "title": title,
        "sample": sample,
        "gauge_volume": gauge_volume,
        "estimated_total_time_s": float(total_time),
        "estimated_total_units": int(total_units),
        "detectors": [det.name for det in detectors],
        "plan_args": {
            "detectors": [det.name for det in detectors],
            "acquire_time": detectors[0].acquire_time.get(),
        },
        "det_config": {
            "ophyd_defs":list(map(repr, detectors)),
            "acquire_time": detectors[0].acquire_time.get(),
            "nbins": detectors[0].nbins.get(),
            "soft_lld": detectors[0].soft_lld.get()
        },
        "experiment_type": "diffraction",
        "plan_name": "scan2D_he3",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor_outer=motor_outer.name, start_pos_outer=start_pos_outer, stop_pos_outer=stop_pos_outer_calc, step_size_outer=step_size_outer_calc, num_steps_outer=num_steps_outer,
                                  motor_inner=motor_inner.name, start_pos_inner=start_pos_inner, stop_pos_inner=stop_pos_inner_calc, step_size_inner=step_size_inner_calc, num_steps_inner=num_steps_inner),
        "motors": motor_names,
    }
    _md.update(md)

    x_fields = _set_scan_motor_metadata(_md, motors)
    
    # @bpp.monitor_during_decorator([detector[0].counts])
    @bpp.run_decorator(md=_md)
    def main_plan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )

        yield from _stage_devices_once(detectors + [motor_outer, motor_inner])
        
        pos_cache = defaultdict(lambda: None)
        cycler = plan_patterns.outer_product(args=[motor_outer, start_pos_outer, stop_pos_outer_calc, num_steps_outer, motor_inner, start_pos_inner, stop_pos_inner_calc, num_steps_inner])

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(cycler):
                step_t0 = time.monotonic()
                yield from progress.mark_started()
                yield from bps.one_nd_step(detectors, step, pos_cache)
                yield from progress.on_units_success(
                    unit_count=1,
                    elapsed_s=max(0.0, time.monotonic() - step_t0),
                )

        yield from inner_scan_nd()

        if return_to_original_positions:
            yield from bps.mv(motor_inner, original_pos_inner)
            yield from bps.mv(motor_outer, original_pos_outer)
        yield from bps.mov(detectors[0].acquire_time, old_acquire_time)

    return(yield from main_plan())
