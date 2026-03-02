import bluesky.plan_stubs as bps
from bluesky_queueserver import parameter_annotation_decorator
from ophyd import Device
from ophyd.positioner import PositionerBase


def _collect_movable_names():
    """Collect movable device names for Queue Server dropdowns."""
    names = []
    g = globals()
    for var, obj in list(g.items()):
        if var.startswith("_"):
            continue
        try:
            if isinstance(obj, PositionerBase):
                names.append(var)
                continue
            if isinstance(obj, Device):
                try:
                    for comp_name, comp in obj.walk_components():
                        try:
                            comp_cls = getattr(comp, "cls", None)
                            if comp_cls and issubclass(comp_cls, PositionerBase):
                                names.append(f"{var}.{comp_name}")
                        except Exception:
                            continue
                except Exception:
                    for attr in getattr(obj, "component_names", ()):
                        try:
                            sub = getattr(obj, attr)
                            if isinstance(sub, PositionerBase):
                                names.append(f"{var}.{attr}")
                        except Exception:
                            continue
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
