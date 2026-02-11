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

import bluesky.preprocessors
import bluesky.preprocessors as bpp
import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.positioner import PositionerBase
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from epics import caput, caget, cainfo
from functools import partial

transfer_time_per_bytes = 4.1203007518796994e-08 # transfer speed in seconds per byte testing on the ASI294MM Pro


def _collect_movable_names():
    """Collect motor variable names from module globals.

    The returned names are used with `convert_device_names=True`, so they
    must be resolvable in the namespace (variable names, not device.name).
    """
    names = []
    g = globals()
    for var, obj in list(g.items()):
        if var.startswith("_"):
            continue
        try:
            # Prefer actual positioners/motors
            if isinstance(obj, PositionerBase):
                names.append(var)
                continue
            # If the global is a Device, inspect its public attributes
            # and collect any subdevices that are PositionerBase instances
            if isinstance(obj, Device):
                # Prefer using ophyd component introspection to capture nested subdevices
                try:
                    for comp_name, comp in obj.walk_components():
                        try:
                            comp_cls = getattr(comp, "cls", None)
                            if comp_cls and issubclass(comp_cls, PositionerBase):
                                names.append(f"{var}.{comp_name}")
                        except Exception:
                            continue
                except Exception:
                    for attr in dir(obj):
                        if attr.startswith("_"):
                            continue
                        try:
                            sub = getattr(obj, attr)
                            if isinstance(sub, PositionerBase):
                                names.append(f"{var}.{attr}")
                        except Exception:
                            continue
        except Exception:
            continue
    # Deduplicate while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def _collect_tomo_motor_names():
    """Collect only tomography motors for the tomo_scan dropdown.

    We create stable aliases in this module's globals so
    `convert_device_names=True` can resolve them reliably.
    """
    g = globals()
    aliases = []
    # Prefer stage theta motors for tomography
    candidates = [
        ("tomo_stage1_theta", "stage1", "theta"),
        ("tomo_stage2_theta", "stage2", "theta"),
    ]
    for alias, parent_name, attr in candidates:
        parent = g.get(parent_name)
        if parent is None or not hasattr(parent, attr):
            continue
        motor = getattr(parent, attr)
        if isinstance(motor, PositionerBase):
            # Register several name variants so the motor can be found by
            # different naming conventions used in RE Worker namespaces.
            names_to_add = [alias, f"{parent_name}.{attr}", f"{parent_name}_{attr}"]
            for name in names_to_add:
                # Avoid overwriting existing globals unintentionally
                if name not in g:
                    g[name] = motor
                if name not in aliases:
                    aliases.append(name)
    return aliases

def _one_nd_step_repeat(
    detectors,
    step,
    pos_cache,
    take_reading=None,
    num_exposures=1,
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
    yield from bps.move_per_step(step, pos_cache)
    yield from _repeater_with_checkpoints(num_exposures, exposure)  # type: ignore  # Movable issue


def _repeater_with_checkpoints(n, gen_func, *args, **kwargs):
    """Repeat a plan with a checkpoint before each repetition.

    A deferred pause is processed at checkpoints, so this creates one pause
    boundary per exposure.
    """
    if n is None:
        while True:
            yield from bps.checkpoint()
            yield from gen_func(*args, **kwargs)
    else:
        for _ in range(n):
            yield from bps.checkpoint()
            yield from gen_func(*args, **kwargs)

def _inner_product_custom(args, num:int = None, step:float = None, offset:float = 0, endpoint=True):
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
        elif step is not None:
            steps = np.arange(start + offset, stop + step/2*endpoint, step)
        else:
            raise ValueError("Must provide either 'num' or 'step'")
        c = cycler(motor, steps)
        cyclers.append(c)
    return functools.reduce(operator.add, cyclers)


def _ensure_detector_temperature(detectors, target_temperature=-15, threshold=-10, poll_interval=0.5):
    """
    Ensure all detectors are at or below the target temperature.

    Parameters:
    -----------
    detectors : list
        List of detector objects to check and adjust.
    target_temperature : float
        The temperature to set for detectors that are above the threshold.
    threshold : float
        The temperature below which the detectors are considered ready.
    poll_interval : float
        Time (in seconds) to wait between temperature checks.

    Yields:
    -------
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

@parameter_annotation_decorator({
    "parameters": {
        "detector": {
            "annotation": "__READABLE__",
            "default": "cam1",
            "description": "Imaging detector (must be readable, default: cam1)"
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
              angle_step:float = None,
              start_angle:float = 0, 
              stop_angle:float = 360,
              num_exposures:int = 1,
              include_stop_angle:bool = False,
              return_to_start:bool = True, 
              check_temperature:bool = True,
              check_180_deg:bool = True,
              md:dict = None):
    '''
    Tomography scan that defaults to 360-step degrees.
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

    if num_projections is not None:
        num_projections_calc = num_projections
        if include_stop_angle:
            angle_step_calc = (stop_angle-start_angle)/(num_projections-1)
            actual_stop_angle = stop_angle
        else:
            angle_step_calc = (stop_angle-start_angle)/num_projections
            actual_stop_angle = stop_angle - angle_step_calc
    elif angle_step is not None:
        angle_step_calc = angle_step
        if include_stop_angle:
            num_projections_calc = int((stop_angle-start_angle)/angle_step) + 1
            actual_stop_angle = stop_angle
        else:
            num_projections_calc = int((stop_angle-start_angle)/angle_step)
            actual_stop_angle = stop_angle - angle_step

    image_bytes = caget("4dh4:cam1:ArraySize_RBV")
    transfer_time_per_image = image_bytes * transfer_time_per_bytes

    total_time = (
        num_exposures * num_projections_calc * detector[0].cam.acquire_time.get()
        + num_exposures * num_projections_calc * transfer_time_per_image
    )  # in seconds

    

    print("#===============#")
    print(f"Starting tomography scan from {start_angle} to {actual_stop_angle} \nin {num_projections_calc} steps of {angle_step_calc} degrees with {num_exposures} exposured per step.")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds")
    print(f"and finish at {datetime.now() + timedelta(seconds=total_time)}.")
    print("#===============#")

    caput("4dh4:TS:RotationStart",start_angle)
    caput("4dh4:TS:RotationStop",actual_stop_angle)
    caput("4dh4:TS:NumAngles",num_projections_calc)
    caput("4dh4:TS:RotationStep", angle_step_calc)


    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    motor_names = motor.name
    md = md or {}
    _md = {
        "file_name": file_name,
        "file_dir": file_dir,
        "plan_args": {
            "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
        },
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "num_exposures": num_exposures,
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "plan_name": "tomo_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, start_angle=start_angle, stop_angle=actual_stop_angle, num_projections=num_projections, angle_step=angle_step, include_stop_angle=include_stop_angle),  # noqa: C408
        "motors": motor_names,
    }
    _md.update(md)

    x_fields = []
    x_fields.extend(utils.get_hinted_fields(motor))

    default_dimensions = [(x_fields, "primary")]

    default_hints = {}
    if len(x_fields) > 0:
        default_hints.update(dimensions=default_dimensions)

    # now add default_hints and override any hints from the original md (if
    # exists)
    _md["hints"] = default_hints
    _md["hints"].update(md.get("hints", {}) or {})
    

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

        for det in detector:
            det.tiff1.file_name.put(file_name)
            det.tiff1.folder_name.put(file_dir)
            yield from bps.stage(det)
        yield from bps.stage(motor)

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
        cycler = _inner_product_custom(num=num_projections, step=angle_step, endpoint=include_stop_angle, args=[motor, start_angle, stop_angle])
        positions = [step[motor] for step in list(cycler)]
        if 180 not in positions and check_180_deg:
            raise RuntimeError("\n\n****Warning****: 180 degrees not in the scan positions. Ending the scan. Please check the scan parameters or disable the check_180_deg parameter.\n\n")
                        

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(cycler):
                yield from _one_nd_step_repeat(detector, step, pos_cache,num_exposures=num_exposures)

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
            "annotation": "__READABLE__",
            "default": "cam1",
            "description": "Imaging detector (must be readable, default: cam1)"
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

    image_bytes = caget("4dh4:cam1:ArraySize_RBV")
    transfer_time_per_image = image_bytes * transfer_time_per_bytes

    total_time = (
        num_exposures * detector[0].cam.acquire_time.get()
        + num_exposures * transfer_time_per_image
    )  # in seconds

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
        "plan_args": {
            "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
        },
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "num_exposures": num_exposures,
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "plan_name": "imaging",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
    }
    _md.update(md)
    

    def exposure():
        yield from bps.trigger_and_read(detector)

    @bpp.run_decorator(md=_md)
    def main_plan():

        for det in detector:
            det.tiff1.file_name.put(file_name)
            det.tiff1.folder_name.put(file_dir)
            yield from bps.stage(det)                 

        yield from _repeater_with_checkpoints(num_exposures, exposure)

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)
        yield from bps.mov(detector[0].cam.gain, old_gain)
        yield from bps.mov(detector[0].cam.offset, old_offset)

    return(yield from main_plan())



@parameter_annotation_decorator({
    "parameters": {
        "detector": {
            "annotation": "__READABLE__",
            "default": "cam1",
            "description": "Imaging detector (must be readable, default: cam1)"
        },
        "motor": {
            "annotation": "typing.Union[str, Motors]",
            "description": "Motor to scan (must be movable)",
            "devices": {"Motors": _collect_movable_names()},
            "convert_device_names": True,
        }
    }
})
def imaging_scan(
            file_name:str, 
            file_dir:str, 
            motor, 
            start_pos:float, 
            stop_pos:float,
            step:float = None,
            num_steps:int = None,
            detector=cam1, 
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

    original_pos = motor.position

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

    old_gain = detector[0].cam.gain.get()
    old_offset = detector[0].cam.offset.get()
    if gain is not None:
        for det in detector:
            yield from bps.mov(det.cam.gain, gain)
    if offset is not None:
        for det in detector:
            yield from bps.mov(det.cam.offset, offset)

    if num_steps is not None:
        positions = np.linspace(start=start_pos, stop=stop_pos, num=num_steps, endpoint=True)
        num_steps_calc = num_steps
        step_cal = positions[1] - positions[0]
        stop_pos_calc = positions[-1]
    elif step is not None:
        positions = np.arange(start=start_pos, stop=stop_pos + step/2, step=step)
        num_steps_calc = len(positions)
        step_cal = step
        stop_pos_calc = positions[-1]

    image_bytes = caget("4dh4:cam1:ArraySize_RBV")
    transfer_time_per_image = image_bytes * transfer_time_per_bytes

    total_time = (
        num_exposures * num_steps_calc * detector[0].cam.acquire_time.get()
        + num_exposures * num_steps_calc * transfer_time_per_image
    )  # in seconds

    print("#===============#")
    print(f"Starting scan of {motor.name} from {start_pos} to {stop_pos_calc} \nin {num_steps_calc} steps of {step_cal} {motor.egu} with {num_exposures} exposures at each position.")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds")
    print(f"and finish at {datetime.now() + timedelta(seconds=total_time)}.")
    print("#===============#")


    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    motor_names = motor.name
    md = md or {}
    _md = {
        "file_name": file_name,
        "file_dir": file_dir,
        "plan_args": {
            # "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
        },
        "det_config": {
            "exposure_time": detector[0].cam.acquire_time.get(),
            "num_exposures": num_exposures,
            "gain": detector[0].cam.gain.get(),
            "offset": detector[0].cam.offset.get(),
        },
        "plan_name": "imaging_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, start_pos=start_pos, stop_pos=stop_pos, step=step, num_steps=num_steps, num_exposures=num_exposures),  # noqa: C408
        "motors": motor_names,
    }
    _md.update(md)

    x_fields = []
    x_fields.extend(utils.get_hinted_fields(motor))

    default_dimensions = [(x_fields, "primary")]

    default_hints = {}
    if len(x_fields) > 0:
        default_hints.update(dimensions=default_dimensions)

    # now add default_hints and override any hints from the original md (if
    # exists)
    _md["hints"] = default_hints
    _md["hints"].update(md.get("hints", {}) or {})
    

    @bpp.run_decorator(md=_md)
    def main_plan():

        for det in detector:
            det.tiff1.file_name.put(file_name)
            det.tiff1.folder_name.put(file_dir)
            yield from bps.stage(det)
        yield from bps.stage(motor)
        
        pos_cache = defaultdict(lambda: None)
        # cycler = plan_patterns.inner_product(num=num_steps, args=[motor, start_pos, stop_pos])
        cycler = _inner_product_custom(step=step, num=num_steps, args=[motor, start_pos, stop_pos])
        print(cycler)

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(cycler):
                yield from _one_nd_step_repeat(detector, step, pos_cache,num_exposures=num_exposures)

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
