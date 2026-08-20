import bluesky.plan_stubs as bps
from bluesky_queueserver import parameter_annotation_decorator
import numpy as np
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


def _object_is_movable(obj):
    """Recognize both PositionerBase motors and protocol-based soft motors."""
    if isinstance(obj, PositionerBase):
        return True
    # ophyd.sim.SynAxis intentionally implements the Bluesky movable protocol
    # without inheriting PositionerBase.
    return bool(
        callable(getattr(obj, "set", None))
        and hasattr(obj, "position")
        and callable(getattr(obj, "read", None))
    )


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
            elif _object_is_movable(obj):
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


def _device_is_descendant_of(device, ancestor):
    """Return True if ``device`` is a child component of ``ancestor``."""
    parent = getattr(device, "parent", None)
    while parent is not None:
        if parent is ancestor:
            return True
        parent = getattr(parent, "parent", None)
    return False


def _deduplicate_stage_devices(devices):
    """Keep stage devices unique, preferring parents over their child devices."""
    out = []
    for device in devices:
        if device is None:
            continue
        if any(device is existing for existing in out):
            continue
        if any(_device_is_descendant_of(device, existing) for existing in out):
            continue
        out = [
            existing
            for existing in out
            if not _device_is_descendant_of(existing, device)
        ]
        out.append(device)
    return out


def _stage_devices_once(devices):
    """Stage devices without double-staging child components."""
    for device in _deduplicate_stage_devices(devices):
        yield from bps.stage(device)


def _hinted_fields_or_readback(device):
    """Return display fields for a motor-like device, with readback fallback."""
    try:
        fields = list(device.hints.get("fields", []))
    except Exception:
        fields = []
    if fields:
        return fields

    for attr in ("user_readback", "readback"):
        try:
            field = getattr(getattr(device, attr), "name", None)
        except Exception:
            field = None
        if field:
            return [field]

    try:
        name = device.name
    except Exception:
        name = None
    return [name] if name else []


def _set_scan_motor_metadata(md, motors, stream_name="primary"):
    """Populate Bluesky scan metadata so LiveTable shows motor readbacks."""
    motor_list = list(motors if isinstance(motors, (list, tuple)) else [motors])
    motor_names = [
        getattr(motor, "name", None)
        for motor in motor_list
        if getattr(motor, "name", None)
    ]
    if motor_names:
        md["motors"] = motor_names

    x_fields = []
    for motor in motor_list:
        x_fields.extend(_hinted_fields_or_readback(motor))

    hints = dict(md.get("hints", {}) or {})
    if x_fields and "dimensions" not in hints:
        hints["dimensions"] = [(x_fields, stream_name)]
    md["hints"] = hints
    return x_fields


def _step_size_positions(start, stop, step_size, *, include_stop=True):
    """Return monotonic positions from start toward stop without overshooting."""
    start = float(start)
    stop = float(stop)
    step_size = float(step_size)
    if step_size == 0:
        raise ValueError("step_size must be non-zero")
    if (stop - start) * step_size < 0:
        raise ValueError("step_size sign must move from start toward stop")

    raw_stop = stop + (0.5 * step_size if include_stop else 0.0)
    positions = np.asarray(np.arange(start=start, stop=raw_stop, step=step_size), dtype=float)
    tol = max(abs(step_size), abs(stop - start), 1.0) * 1e-12
    if step_size > 0:
        positions = positions[positions <= stop + tol]
    else:
        positions = positions[positions >= stop - tol]
    if positions.size and abs(float(positions[-1]) - stop) <= tol:
        positions[-1] = stop
    return positions


def _scan_positions_from_num_or_step_size(start, stop, *, num_steps=None, step_size=None):
    """Normalize 1D scan inputs into positions, count, actual step size, and stop."""
    if num_steps is not None:
        num_steps_calc = int(num_steps)
        if num_steps_calc < 2:
            raise ValueError("num_steps must be at least 2")
        positions = np.linspace(
            start=float(start),
            stop=float(stop),
            num=num_steps_calc,
            endpoint=True,
        )
    elif step_size is not None:
        positions = _step_size_positions(start, stop, step_size, include_stop=True)
        if len(positions) < 2:
            raise ValueError("step_size must produce at least 2 scan positions")
        num_steps_calc = int(len(positions))
    else:
        raise ValueError("Either step_size or num_steps must be provided")

    step_size_calc = float(positions[1] - positions[0])
    stop_calc = float(positions[-1])
    return positions, num_steps_calc, step_size_calc, stop_calc


def _tomo_positions_from_num_or_step_size(
    start,
    stop,
    *,
    num_projections=None,
    angle_step_size=None,
    include_stop=False,
):
    """Normalize tomography inputs into positions, count, actual step size, and stop."""
    if num_projections is not None:
        num_calc = int(num_projections)
        if num_calc < 1:
            raise ValueError("num_projections must be at least 1")
        if include_stop and num_calc < 2:
            raise ValueError("num_projections must be at least 2 when include_stop_angle=True")
        positions = np.linspace(
            start=float(start),
            stop=float(stop),
            num=num_calc,
            endpoint=bool(include_stop),
        )
    elif angle_step_size is not None:
        positions = _step_size_positions(
            start,
            stop,
            angle_step_size,
            include_stop=bool(include_stop),
        )
        if len(positions) < 1:
            raise ValueError("angle_step_size must produce at least 1 scan position")
        num_calc = int(len(positions))
    else:
        raise ValueError("Either angle_step_size or num_projections must be provided")

    if len(positions) > 1:
        step_size_calc = float(positions[1] - positions[0])
    else:
        step_size_calc = float(angle_step_size if angle_step_size is not None else 0.0)
    stop_calc = float(positions[-1])
    return positions, num_calc, step_size_calc, stop_calc


class _ProgressEstimator:
    """Publish a plan ETA from a known schedule or observed unit durations."""

    def __init__(
        self,
        total_units,
        initial_total_time_s,
        alpha=0.2,
        outlier_factor=4.0,
        planned_unit_durations_s=None,
    ):
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
        self._planned_unit_durations_s = None
        if planned_unit_durations_s is not None:
            try:
                planned = tuple(
                    max(0.0, float(duration))
                    for duration in planned_unit_durations_s
                )
            except Exception:
                planned = ()
            if len(planned) == self.total_units:
                self._planned_unit_durations_s = planned

    def _compute_finish_epoch(self):
        if self._planned_unit_durations_s is not None:
            remaining_s = sum(self._planned_unit_durations_s[self.done_units:])
            return float(time.time() + remaining_s)
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
    yield from _stage_devices_once([motor])
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
