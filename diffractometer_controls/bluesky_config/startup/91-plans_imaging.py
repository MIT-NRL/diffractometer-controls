# import bluesky.plans
import bluesky.plans as bp
from bluesky.plans import scan, count, grid_scan, rel_scan, rel_grid_scan

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




def tomo_scan(file_name:str, 
              detector, 
              motor, 
              exposure_time:float = None,
              angle_step:float = 1,
              start_angle:float = None, 
              stop_angle:float = None,
              return_to_start:bool = True, 
              md:dict = None):
    '''
    Tomography scan that defaults to 360-step degrees.
    '''
    if (start_angle is None) and (stop_angle is None):
        start_angle, stop_angle = 0, 360-angle_step
    elif (start_angle is not None) and (stop_angle is None):
        stop_angle = 360 + start_angle - angle_step
    num_angles = int((stop_angle - start_angle + angle_step) / angle_step)
    while num_angles*angle_step > 360:
        num_angles -= 1
    if ((stop_angle-start_angle+angle_step) % angle_step) != 0:
        angle_step_new = (stop_angle-start_angle+angle_step) / num_angles
        print(f"\n#===============#\n360 not divisible by {angle_step}.\nUsing a step size of {angle_step_new} instead.\n#===============#\n")
        angle_step = angle_step_new

    print("#===============#")
    print(f"Starting tomography scan from {start_angle} to {stop_angle} \nin {num_angles} steps of {angle_step} degrees.")
    print("#===============#")

    caput("4dh4:TS:RotationStart",start_angle)
    caput("4dh4:TS:RotationStop",stop_angle)
    caput("4dh4:TS:NumAngles",num_angles)
    caput("4dh4:TS:RotationStep", angle_step)

    file_name = str(file_name).strip().replace(" ","_").replace("__","_")

    detector = [detector]

    old_exposure_time = detector[0].cam.acquire_time.get()

    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    motor_names = motor.name
    md = md or {}
    _md = {
        "plan_args": {
            "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
        },
        "plan_name": "tomo_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, start_angle=start_angle, stop_angle=stop_angle, num_angles=num_angles),  # noqa: C408
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
        cycler = plan_patterns.inner_product(num=num_angles, args=[motor, start_angle, stop_angle])

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(cycler):
                yield from bps.one_nd_step(detector, step, pos_cache)

        # print("Replace the sample, open the shutter, and press Resume to start the scan")
        # yield from bps.checkpoint()
        # yield from bps.pause()
        yield from inner_scan_nd()

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)

        if return_to_start:
            yield from bps.mv(motor, start_angle)

    return(yield from main_plan())




def imaging(file_name:str, 
              detector, 
              exposure_time:float = None,
              num_exposures:int = 1,
              md:dict = None):
    '''
    Tomography scan that performs dark field scans, flat field scans, and then the actual tomography scan.
    '''

    file_name = str(file_name).strip().replace(" ","_").replace("__","_")

    detector = [detector]

    old_exposure_time = detector[0].cam.acquire_time.get()
    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    md = md or {}
    _md = {
        "plan_args": {
            "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
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
            yield from bps.stage(det)     

        yield from bps.repeater(num_exposures,exposure)

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)

    return(yield from main_plan())




def imaging_scan(file_name:str, 
              detector, 
              motor, 
              start_pos:float, 
              stop_pos:float, 
              step:float,
              exposure_time:float = None,
              return_to_original_position:bool = True,
              md:dict = None):
    '''
    General scan for the imaging detector system.
    '''

    original_pos = motor.position

    num_steps = int(round((stop_pos-start_pos)/step) + 1)
    step = (stop_pos-start_pos)/(num_steps-1)
    total_time = num_steps*detector.cam.acquire_time.get() # in seconds

    print("#===============#")
    print(f"Starting scan of {motor.name} from {start_pos} to {stop_pos} \nin {num_steps} steps of {step} {motor.egu}.")
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60
    print(f"The scan time is estimated to be {hours} hours, {minutes} minutes, and {seconds} seconds.")
    print("#===============#")


    file_name = str(file_name).strip().replace(" ","_").replace("__","_")


    detector = [detector]

    old_exposure_time = detector[0].cam.acquire_time.get()

    if exposure_time is not None:
        for det in detector:
            yield from bps.mov(det.cam.acquire_time, exposure_time)

    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    motor_names = motor.name
    md = md or {}
    _md = {
        "plan_args": {
            "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
        },
        "plan_name": "imaging_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, start_pos=start_pos, stop_pos=stop_pos, step=step),  # noqa: C408
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
            yield from bps.stage(det)
        yield from bps.stage(motor)
        
        pos_cache = defaultdict(lambda: None)
        cycler = plan_patterns.inner_product(num=num_steps, args=[motor, start_pos, stop_pos])

        def inner_scan_nd():
            # yield from bps.declare_stream(motor, *detector, name="primary")
            for step in list(cycler):
                yield from bps.one_nd_step(detector, step, pos_cache)

        # print("Replace the sample, open the shutter, and press Resume to start the scan")
        # yield from bps.checkpoint()
        # yield from bps.pause()
        yield from inner_scan_nd()

        yield from bps.mov(detector[0].cam.acquire_time, old_exposure_time)

        if return_to_original_position:
            yield from bps.mv(motor, original_pos)

    return(yield from main_plan())