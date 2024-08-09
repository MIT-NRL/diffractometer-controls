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

frame_type_sig = EpicsSignal("4dh4:TS:FrameType", name="frame_type_sig")

monitor_and_count = bpp.monitor_during_decorator([he3psd.counts])(bp.count)



# def tomo_scan(detectors, *args, num=None, md=None):
#     """
#     Scan over one multi-motor trajectory.

#     Parameters
#     ----------
#     detectors : list
#         list of 'readable' objects
#     *args :
#         For one dimension, ``motor, start, stop``.
#         In general:

#         .. code-block:: python

#             motor1, start1, stop1,
#             motor2, start2, stop2,
#             ...,
#             motorN, startN, stopN

#         Motors can be any 'settable' object (motor, temp controller, etc.)
#     num : integer
#         number of points
#     per_step : callable, optional
#         hook for customizing action of inner loop (messages per step).
#         See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
#         for details.
#     md : dict, optional
#         metadata

#     See Also
#     --------
#     :func:`bluesky.plans.relative_inner_product_scan`
#     :func:`bluesky.plans.grid_scan`
#     :func:`bluesky.plans.scan_nd`
#     """
#     # For back-compat reasons, we accept 'num' as the last positional argument:
#     # scan(detectors, motor, -1, 1, 3)
#     # or by keyword:
#     # scan(detectors, motor, -1, 1, num=3)
#     # ... which requires some special processing.
#     if num is None:
#         if len(args) % 3 != 1:
#             raise ValueError(
#                 "The number of points to scan must be provided "
#                 "as the last positional argument or as keyword "
#                 "argument 'num'."
#             )
#         num = args[-1]
#         args = args[:-1]

#     if not (float(num).is_integer() and num > 0.0):
#         raise ValueError(
#             f"The parameter `num` is expected to be a number of "
#             f"steps (not step size!) It must therefore be a "
#             f"whole number. The given value was {num}."
#         )
#     num = int(num)

#     md_args = list(chain(*((repr(motor), start, stop) for motor, start, stop in partition(3, args))))
#     motor_names = tuple(motor.name for motor, start, stop in partition(3, args))
#     md = md or {}
#     _md = {
#         "plan_args": {
#             "detectors": list(map(repr, detectors)),
#             "num": num,
#             "args": md_args,
#         },
#         "plan_name": "scan",
#         "plan_pattern": "inner_product",
#         "plan_pattern_module": plan_patterns.__name__,
#         "plan_pattern_args": dict(num=num, args=md_args),  # noqa: C408
#         "motors": motor_names,
#     }
#     _md.update(md)

#     # get hints for best effort callback
#     motors = [motor for motor, start, stop in partition(3, args)]

#     # Give a hint that the motors all lie along the same axis
#     # [(['motor1', 'motor2', ...], 'primary'), ] is 1D (this case)
#     # [ ('motor1', 'primary'), ('motor2', 'primary'), ... ] is 2D for example
#     # call x_fields because these are meant to be the x (independent) axis
#     x_fields = []
#     for motor in motors:
#         x_fields.extend(get_hinted_fields(motor))

#     default_dimensions = [(x_fields, "primary")]

#     default_hints = {}
#     if len(x_fields) > 0:
#         default_hints.update(dimensions=default_dimensions)

#     # now add default_hints and override any hints from the original md (if
#     # exists)
#     _md["hints"] = default_hints
#     _md["hints"].update(md.get("hints", {}) or {})

#     # At the start of the plan, perform dark field
#     def background_exposure(frame_type: str="dark"):
#         if frame_type == "dark":
#             for obj in detectors:
#                 yield from bps.mov(obj.cam.frame_type, 1)
#         elif frame_type == "flat":
#             for obj in detectors:
#                 yield from bps.mov(obj.cam.frame_type, 2)
#         yield from bps.trigger_and_read(detectors, name=frame_type)


#     @bpp.run_decorator(md=_md)
#     @bpp.stage_decorator(list(detectors) + motors)
#     def full_plan():

#         print("Close shutter then press Resume to take the dark field")
#         yield from bps.pause()
#         yield from background_exposure("dark")

#         print("Open shutter and remove the sample then press Resume to take the flat field")
#         yield from bps.pause()
#         yield from background_exposure("flat")

#         for obj in detectors:
#             yield from bps.mov(obj.cam.frame_type, 0)
#         full_cycler = plan_patterns.inner_product(num=num, args=args)

#         pos_cache = defaultdict(lambda: None)
#         cycler = utils.merge_cycler(full_cycler)
#         motors = list(cycler.keys)
#         print(list(cycler))
        
#         print("Replace the sample, open the shutter, and press Resume to start the scan")
#         yield from bps.pause()
#         def inner_scan_nd():
#             yield from bps.declare_stream(*motors, *detectors, name="primary")
#             for step in list(cycler):
#                 yield from bps.one_nd_step(detectors, step, pos_cache)

#         yield from inner_scan_nd()

#     return (yield from full_plan())



def tomo_scan(file_name:str, 
              detector, 
              motor, 
              angle_step:float = 1,
              start_angle:float = None, 
              stop_angle:float = None,
              move_to_start:bool = True, 
              md:dict = None):
    '''
    Tomography scan that performs dark field scans, flat field scans, and then the actual tomography scan.
    '''
    if (start_angle is None) and (stop_angle is None):
        start_angle, stop_angle = 0, 360-angle_step
    elif (start_angle is not None) and (stop_angle is None):
        stop_angle = 360 + start_angle - angle_step
    num_angles = int((stop_angle - start_angle + angle_step) // angle_step)
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

        if move_to_start:
            yield from bps.mv(motor, start_angle)

    return(yield from main_plan())




def imaging(file_name:str, 
              detector, 
              num_exposures:int = 1,
              md:dict = None):
    '''
    Tomography scan that performs dark field scans, flat field scans, and then the actual tomography scan.
    '''

    file_name = str(file_name).strip().replace(" ","_").replace("__","_")

    detector = [detector]
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

    return(yield from main_plan())




def imaging_scan(file_name:str, 
              detector, 
              motor, 
              start_pos:float = 0, 
              stop_pos:float = 30, 
              step:float = 5,
              md:dict = None):
    '''
    General scan for the imaging detector system.
    '''

    file_name = str(file_name).strip().replace(" ","_").replace("__","_")

    num_steps = (stop_pos-start_pos+1)/step

    detector = [detector]
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

    return(yield from main_plan())