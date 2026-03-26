import bluesky.plan_stubs as bps
from bluesky_queueserver import parameter_annotation_decorator
from ophyd import Device
from ophyd.positioner import PositionerBase


def _component_walk_name_and_cls(comp_walk):
    """
    Normalize ``Device.walk_components()`` output across ophyd versions.

    Recent ophyd returns ``ComponentWalk`` objects with ``dotted_name`` and ``item``.
    Older versions yielded ``(name, component)`` tuples.
    """
    comp_name = getattr(comp_walk, "dotted_name", None)
    comp = getattr(comp_walk, "item", comp_walk)
    if comp_name is None:
        comp_name, comp = comp_walk
    return comp_name, getattr(comp, "cls", None)


def _collect_movable_names():
    """Collect movable device names for Queue Server dropdowns.

    Prefer addressable nested movable components when a movable device exposes
    them. This keeps pseudo-positioners such as ``analyzer1`` from appearing
    twice when ``analyzer1`` and ``analyzer1.counts`` target the same motion.
    """
    names = []
    for var, obj in list(globals().items()):
        if var.startswith("_"):
            continue
        try:
            nested_names = []
            if isinstance(obj, Device):
                try:
                    for comp_walk in obj.walk_components():
                        try:
                            comp_name, comp_cls = _component_walk_name_and_cls(comp_walk)
                            if comp_cls and issubclass(comp_cls, PositionerBase):
                                nested_names.append(f"{var}.{comp_name}")
                        except Exception:
                            continue
                except Exception:
                    for attr in getattr(obj, "component_names", ()):
                        try:
                            sub = getattr(obj, attr)
                            if isinstance(sub, PositionerBase):
                                nested_names.append(f"{var}.{attr}")
                        except Exception:
                            continue
            if nested_names:
                names.extend(nested_names)
            elif isinstance(obj, PositionerBase):
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


@parameter_annotation_decorator(
    {
        "parameters": {
            "motor": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Motor to move (must be movable)",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            }
        }
    }
)
def move_motor(
    motor,
    position: float,
):
    """Move a motor to a specified position."""
    yield from bps.stage(motor)
    yield from bps.mv(motor, position)


@parameter_annotation_decorator(
    {
        "parameters": {
            "seconds": {
                "annotation": "float",
                "description": "Time to wait in seconds (must be non-negative)",
            }
        }
    }
)
def wait_seconds(seconds: float):
    """Pause plan execution for a fixed number of seconds."""
    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("'seconds' must be non-negative.")
    yield from bps.sleep(seconds)
