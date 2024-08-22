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

frame_type_sig = EpicsSignal("4dh4:TS:FrameType", name="frame_type_sig")

# monitor_and_count = bpp.monitor_during_decorator([he3psd0.counts])(bp.count)


def count_he3(
                title:str,
                sample:str,
                detectors, 
                num=1, 
                delay=None, 
                *, 
                per_shot=None, 
                md=None
                
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
    delay : iterable or scalar, optional
        Time delay in seconds between successive readings; default is 0.
    per_shot : callable, optional
        hook for customizing action of inner loop (messages per step)
        Expected signature ::

           def f(detectors: Iterable[OphydObj]) -> Generator[Msg]:
               ...

    md : dict, optional
        metadata

    Notes
    -----
    If ``delay`` is an iterable, it must have at least ``num - 1`` entries or
    the plan will raise a ``ValueError`` during iteration.
    """
    if not isinstance(detectors,list):
        detectors = [detectors]
    if num is None:
        num_intervals = None
    else:
        num_intervals = num - 1
    _md = {
        "title": title,
        "sample": sample,
        "detectors": [det.name for det in detectors],
        "num_points": num,
        "num_intervals": num_intervals,
        "plan_args": {"detectors": list(map(repr, detectors)), "num": num, "delay": delay},
        "plan_name": "count_he3",
        "hints": {},
    }
    _md.update(md or {})
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])
    

    # per_shot might define a different stream, so do not predeclare primary
    predeclare = per_shot is None and os.environ.get("BLUESKY_PREDECLARE", False)
    if per_shot is None:
        per_shot = bps.one_shot

    @bpp.monitor_during_decorator([detectors[0].counts])
    @bpp.stage_decorator(detectors)
    @bpp.run_decorator(md=_md)
    def inner_count():
        if predeclare:
            yield from bps.declare_stream(*detectors, name="primary")
        return (yield from bps.repeat(partial(per_shot, detectors), num=num, delay=delay))

    return (yield from inner_count())





def scan_he3( 
            title:str,
            sample:str,
            detector, 
            motor, 
            start_pos:float, 
            stop_pos:float, 
            step:float,
            return_to_start:bool = True,
            md:dict = None
              ):
    '''
    General scan for the imaging detector system.
    '''

    
    num_steps = int(round((stop_pos-start_pos)/step) + 1)
    step = (stop_pos-start_pos)/(num_steps-1)

    print("#===============#")
    print(f"Starting tomography scan from {start_pos} to {stop_pos} \nin {num_steps} steps of {step} {motor.egu}.")
    print("#===============#")

    if not isinstance(detector,list):
        detector = [detector]
    # md_args = list(chain(*((repr(motor), start, stop) for motor, start_angle, stop_angle)))
    motor_names = motor.name
    md = md or {}
    _md = {
        "title": title,
        "sample": sample,
        "plan_args": {
            "detectors": list(map(repr, detector)),
            # "num": num,
            # "args": md_args,
        },
        "plan_name": "scan_he3",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(motor=motor.name, start_pos=start_pos, stop_pos=stop_pos, step=step, num_steps=num_steps),  # noqa: C408
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
    
    @bpp.monitor_during_decorator([detector[0].counts])
    @bpp.run_decorator(md=_md)
    def main_plan():

        for det in detector:
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

        if return_to_start:
            yield from bps.mv(motor, start_pos)

    return(yield from main_plan())