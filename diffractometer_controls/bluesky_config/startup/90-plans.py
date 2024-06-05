# import bluesky.plans
import bluesky.plans as bp
from bluesky.plans import *

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

frame_type_sig = EpicsSignal("4dh4:TS:FrameType", name="frame_type_sig")

monitor_and_count = bpp.monitor_during_decorator([he3psd.counts])(bp.count)



def tomo_scan(detectors, *args, num=None, md=None):
    """
    Scan over one multi-motor trajectory.

    Parameters
    ----------
    detectors : list
        list of 'readable' objects
    *args :
        For one dimension, ``motor, start, stop``.
        In general:

        .. code-block:: python

            motor1, start1, stop1,
            motor2, start2, stop2,
            ...,
            motorN, startN, stopN

        Motors can be any 'settable' object (motor, temp controller, etc.)
    num : integer
        number of points
    per_step : callable, optional
        hook for customizing action of inner loop (messages per step).
        See docstring of :func:`bluesky.plan_stubs.one_nd_step` (the default)
        for details.
    md : dict, optional
        metadata

    See Also
    --------
    :func:`bluesky.plans.relative_inner_product_scan`
    :func:`bluesky.plans.grid_scan`
    :func:`bluesky.plans.scan_nd`
    """
    # For back-compat reasons, we accept 'num' as the last positional argument:
    # scan(detectors, motor, -1, 1, 3)
    # or by keyword:
    # scan(detectors, motor, -1, 1, num=3)
    # ... which requires some special processing.
    if num is None:
        if len(args) % 3 != 1:
            raise ValueError(
                "The number of points to scan must be provided "
                "as the last positional argument or as keyword "
                "argument 'num'."
            )
        num = args[-1]
        args = args[:-1]

    if not (float(num).is_integer() and num > 0.0):
        raise ValueError(
            f"The parameter `num` is expected to be a number of "
            f"steps (not step size!) It must therefore be a "
            f"whole number. The given value was {num}."
        )
    num = int(num)

    md_args = list(chain(*((repr(motor), start, stop) for motor, start, stop in partition(3, args))))
    motor_names = tuple(motor.name for motor, start, stop in partition(3, args))
    md = md or {}
    _md = {
        "plan_args": {
            "detectors": list(map(repr, detectors)),
            "num": num,
            "args": md_args,
        },
        "plan_name": "scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": dict(num=num, args=md_args),  # noqa: C408
        "motors": motor_names,
    }
    _md.update(md)

    # get hints for best effort callback
    motors = [motor for motor, start, stop in partition(3, args)]

    # Give a hint that the motors all lie along the same axis
    # [(['motor1', 'motor2', ...], 'primary'), ] is 1D (this case)
    # [ ('motor1', 'primary'), ('motor2', 'primary'), ... ] is 2D for example
    # call x_fields because these are meant to be the x (independent) axis
    x_fields = []
    for motor in motors:
        x_fields.extend(get_hinted_fields(motor))

    default_dimensions = [(x_fields, "primary")]

    default_hints = {}
    if len(x_fields) > 0:
        default_hints.update(dimensions=default_dimensions)

    # now add default_hints and override any hints from the original md (if
    # exists)
    _md["hints"] = default_hints
    _md["hints"].update(md.get("hints", {}) or {})

    # At the start of the plan, perform dark field
    def background_exposure(frame_type: str="dark"):
        if frame_type == "dark":
            for obj in detectors:
                obj.cam.frame_type.put("Background")
        elif frame_type == "flat":
            for obj in detectors:
                obj.cam.frame_type.put("FlatField")
        yield from bps.trigger_and_read(detectors, name=frame_type)


    @bpp.run_decorator(md=_md)
    @bpp.stage_decorator(list(detectors) + motors)
    def full_plan():
        print(args)
        yield from background_exposure("dark")

        yield from background_exposure("flat")

        for obj in detectors:
            obj.cam.frame_type.put("Normal")
        full_cycler = plan_patterns.inner_product(num=num, args=args)

        pos_cache = defaultdict(lambda: None)
        cycler = utils.merge_cycler(full_cycler)
        motors = list(cycler.keys)
                
        def inner_scan_nd():
            yield from bps.declare_stream(*motors, *detectors, name="primary")
            for step in list(cycler):
                yield from bps.one_nd_step(detectors, step, pos_cache)

        yield from inner_scan_nd()

    return (yield from full_plan())