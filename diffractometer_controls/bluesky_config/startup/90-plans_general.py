import bluesky.plan_stubs as bps
from bluesky_queueserver import parameter_annotation_decorator
from ophyd import Device
from ophyd.positioner import PositionerBase
import time


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


class _ProgressEstimator:
    """Refine ETA from observed successful plan-unit durations."""

    def __init__(self, total_units, initial_total_time_s, alpha=0.2, outlier_factor=4.0):
        self.total_units = max(0, int(total_units))
        self.done_units = 0
        self.alpha = float(alpha)
        self.outlier_factor = float(outlier_factor)
        self.avg_unit_s = (
            float(initial_total_time_s) / float(self.total_units)
            if self.total_units > 0 and float(initial_total_time_s) > 0
            else 0.0
        )
        self._unit_t0 = None
        self._started = False

    def _compute_finish_epoch(self):
        if self.total_units <= 0 or self.avg_unit_s <= 0:
            return 0.0
        remaining = max(0, self.total_units - self.done_units)
        return float(time.time() + remaining * self.avg_unit_s)

    def _publish_progress(self):
        publish = globals().get("_publish_run_progress", None)
        if callable(publish):
            publish(
                done_units=int(self.done_units),
                total_units=int(self.total_units),
                finish_epoch=float(self._compute_finish_epoch()),
                now=time.time(),
            )
        if False:
            yield None

    def mark_started(self):
        if not self._started:
            self._started = True
            yield from self._publish_progress()

    def _update_avg_from_sample(self, sample_s):
        if sample_s is None or sample_s <= 0:
            return
        if self.avg_unit_s <= 0:
            self.avg_unit_s = float(sample_s)
            return
        if sample_s <= self.avg_unit_s * self.outlier_factor:
            self.avg_unit_s = self.alpha * float(sample_s) + (1.0 - self.alpha) * self.avg_unit_s

    def on_units_success(self, unit_count=1, elapsed_s=None):
        units = max(1, int(unit_count))
        self.done_units = min(self.total_units, self.done_units + units)

        sample_per_unit_s = None
        if elapsed_s is not None and units > 0:
            sample_per_unit_s = max(0.0, float(elapsed_s)) / float(units)
        self._update_avg_from_sample(sample_per_unit_s)
        yield from self._publish_progress()

    def on_unit_start(self, _i=None):
        self._unit_t0 = time.monotonic()
        yield from self.mark_started()

    def on_unit_success(self, _i=None):
        dt = None
        if self._unit_t0 is not None:
            dt = max(0.0, time.monotonic() - self._unit_t0)
        self._unit_t0 = None
        yield from self.on_units_success(unit_count=1, elapsed_s=dt)


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
