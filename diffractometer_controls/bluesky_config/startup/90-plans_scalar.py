import os
import time
from collections import defaultdict

import bluesky.plan_patterns as plan_patterns
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky_queueserver import parameter_annotation_decorator
from cycler import cycler
from ophyd import Device, Signal
from ophyd.ophydobj import Kind

try:
    from diffractometer_controls.plan_time_estimation import estimate_plan_runtime
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.plan_time_estimation import estimate_plan_runtime


def _collect_scalar_readable_names():
    """Collect scalar Signals and selectable fields of compatible Devices."""
    names = []
    for var, obj in list(globals().items()):
        if var.startswith("_"):
            continue
        try:
            if bool(getattr(obj, "scalar_plan_hidden", False)):
                continue
            is_scalar_signal = isinstance(obj, Signal) and bool(obj.kind & Kind.normal)
            is_scalar_device = isinstance(obj, Device) and bool(
                getattr(obj, "scalar_plan_compatible", False)
            )
            is_scalar_readout_device = isinstance(obj, Device) and bool(
                getattr(obj, "scalar_readout_channels", ())
            )
            if is_scalar_signal:
                names.append(var)
            elif is_scalar_device or is_scalar_readout_device:
                signal_fields = list(
                    getattr(obj, "scalar_readout_channels", ())
                    or getattr(obj, "scalar_channels", ())
                    or ()
                )
                if not signal_fields:
                    declarations = dict(getattr(obj, "live_plot_signals", {}) or {})
                    signal_fields = [
                        attribute
                        for attribute, spec in declarations.items()
                        if str(dict(spec or {}).get("role", "signal") or "signal") == "signal"
                    ]
                if signal_fields:
                    names.extend(f"{var}.{attribute}" for attribute in signal_fields)
                else:
                    names.append(var)
        except Exception:
            continue
    return list(dict.fromkeys(names))


def _as_scalar_readable_list(detectors):
    if isinstance(detectors, (list, tuple)):
        detector_list = list(detectors)
    else:
        detector_list = [detectors]
    if not detector_list:
        raise ValueError("At least one scalar detector or signal is required")
    for detector in detector_list:
        if not callable(getattr(detector, "read", None)):
            raise TypeError(f"{detector!r} is not a readable Bluesky object")
    return detector_list


def _scalar_acquisition_devices(readables):
    """Map selected child signals to the parent device that acquires them."""
    acquisition_devices = []
    for readable in readables:
        acquisition_device = readable
        parent = getattr(readable, "parent", None)
        while parent is not None:
            if bool(getattr(parent, "scalar_plan_compatible", False)):
                acquisition_device = parent
                break
            parent = getattr(parent, "parent", None)
        if not any(acquisition_device is existing for existing in acquisition_devices):
            acquisition_devices.append(acquisition_device)
    return acquisition_devices


def _scalar_acquisition_layout(readables, acquisition_devices):
    """Split hardware-triggered devices from passive scalar Signals."""
    passive_signals = []
    for readable in readables:
        if not isinstance(readable, Signal):
            continue
        parent = getattr(readable, "parent", None)
        is_hardware_child = False
        while parent is not None:
            if bool(getattr(parent, "scalar_plan_compatible", False)):
                is_hardware_child = True
                break
            parent = getattr(parent, "parent", None)
        if not is_hardware_child:
            passive_signals.append(readable)

    triggered_devices = [
        device
        for device in acquisition_devices
        if not any(device is signal for signal in passive_signals)
    ]
    return triggered_devices, passive_signals


def _software_average_signals(passive_signals):
    """Create live running-average Signals with the passive input data keys."""
    return [
        (source, Signal(name=source.name, value=float("nan"), kind=source.kind))
        for source in passive_signals
    ]


def _selected_hardware_readables(readables, passive_signals):
    """Return selected fields to record, excluding software-averaged inputs."""
    selected = []
    for readable in readables:
        if any(readable is passive for passive in passive_signals):
            continue
        if not any(readable is existing for existing in selected):
            selected.append(readable)
    return selected


def _configure_passive_live_fields(live_plot_fields, live_monitor_signals, signal_pairs):
    """Route passive inputs through their software running-average streams."""
    for source, output in signal_pairs:
        data_key = str(source.name)
        field = dict(live_plot_fields.get(data_key, {}) or {})
        field.pop("pv", None)
        field["transport"] = "document"
        field["stream"] = f"{data_key}_monitor"
        live_plot_fields[data_key] = field
        if not any(output is existing for existing in live_monitor_signals):
            live_monitor_signals.append(output)


def _sample_passive_signals(signal_pairs, duration, *, sample_period=0.1):
    """Sample passive signals and publish each updated running average."""
    duration = max(0.0, float(duration or 0.0))
    sample_period = max(0.01, float(sample_period))
    samples = {source.name: [] for source, _output in signal_pairs}
    last_values = {}
    deadline = time.monotonic() + duration

    while True:
        for source, output in signal_pairs:
            reading = yield from bps.read(source)
            item = dict(reading or {}).get(source.name, {})
            value = dict(item or {}).get("value")
            last_values[source.name] = value
            try:
                samples[source.name].append(float(value))
            except (TypeError, ValueError):
                pass

            numeric = samples[source.name]
            running_value = (
                sum(numeric) / len(numeric)
                if numeric
                else last_values.get(source.name)
            )
            yield from bps.mv(output, running_value)

        remaining = deadline - time.monotonic()
        if duration <= 0 or remaining <= 0:
            break
        yield from bps.sleep(min(sample_period, remaining))


def _acquire_scalar_point(
    triggered_devices,
    hardware_readables,
    passive_signal_pairs,
    *,
    software_dwell_time,
    extra_readables=(),
):
    """Acquire one event, averaging passive inputs during the exposure."""
    trigger_group = f"scalar-trigger-{time.monotonic_ns()}"
    for device in triggered_devices:
        yield from bps.trigger(device, group=trigger_group)

    if passive_signal_pairs:
        yield from _sample_passive_signals(
            passive_signal_pairs,
            software_dwell_time,
        )

    if triggered_devices:
        yield from bps.wait(group=trigger_group)

    yield from bps.create(name="primary")
    readables = (
        list(hardware_readables)
        + [output for _source, output in passive_signal_pairs]
        + list(extra_readables or ())
    )
    for readable in readables:
        yield from bps.read(readable)
    yield from bps.save()


def _scalar_acquire_time_signals(detectors):
    signals = []
    for detector in detectors:
        signal = getattr(detector, "acquire_time", None)
        if not callable(getattr(signal, "get", None)):
            continue
        if not callable(getattr(signal, "set", None)):
            continue
        if not any(signal is existing for existing in signals):
            signals.append(signal)
    return signals


def _configure_scalar_acquire_time(detectors, acquire_time):
    signals = _scalar_acquire_time_signals(detectors)
    originals = []
    if acquire_time is not None:
        acquire_time = float(acquire_time)
        if acquire_time < 0:
            raise ValueError("acquire_time must be non-negative")
        originals = [(signal, signal.get()) for signal in signals]
        move_args = []
        for signal in signals:
            move_args.extend((signal, acquire_time))
        if move_args:
            yield from bps.mv(*move_args)

    current_times = [float(signal.get()) for signal in signals]
    effective_time = max(
        current_times,
        default=float(acquire_time) if acquire_time is not None else 0.0,
    )
    return signals, originals, effective_time


def _restore_scalar_acquire_times(originals):
    if originals:
        move_args = []
        for signal, value in originals:
            move_args.extend((signal, value))
        yield from bps.mv(*move_args)


def _scalar_detector_config(
    detectors,
    acquire_time_signals,
    *,
    software_dwell_time=None,
):
    acquire_times = {
        signal.name: float(signal.get())
        for signal in acquire_time_signals
    }
    config = {
        "ophyd_defs": list(map(repr, detectors)),
        "acquire_times": acquire_times,
    }
    if software_dwell_time is not None:
        config["software_dwell_time"] = float(software_dwell_time)
    return config


def _scalar_detector_type(detectors):
    detector_types = list(
        dict.fromkeys(
            str(getattr(detector, "detector_type", "") or "scalar")
            for detector in detectors
        )
    )
    return detector_types[0] if len(detector_types) == 1 else "mixed"


_SCALAR_DEVICE_ANNOTATION = {
    "parameters": {
        "detectors": {
            "annotation": "typing.Union[typing.List[ScalarReadables], ScalarReadables]",
            "description": "Scalar detector channels or readable scalar signals",
            "devices": {"ScalarReadables": _collect_scalar_readable_names()},
            "convert_device_names": True,
        }
    }
}

_SCALAR_FILE_TYPE_ANNOTATION = {
    "annotation": "ScalarFileType",
    "description": "Additional scalar data file written after the run",
    "enums": {"ScalarFileType": ["csv", "nexus"]},
}


def _normalize_scalar_file_type(file_type):
    value = str(file_type or "").strip().lower()
    if value not in {"csv", "nexus"}:
        raise ValueError("file_type must be 'csv' or 'nexus'")
    return value


@parameter_annotation_decorator(
    {
        "parameters": {
            **_SCALAR_DEVICE_ANNOTATION["parameters"],
            "file_type": _SCALAR_FILE_TYPE_ANNOTATION,
        }
    }
)
def count_scalar(
    title: str,
    sample: str = "",
    *,
    detectors,
    acquire_time: float = None,
    num: int = 1,
    delay: float = 0.0,
    file_type: str = "csv",
    md: dict = None,
):
    """Count scalar readables with optional hardware or software timing.

    For passive Signals, ``acquire_time`` is a software dwell: values are
    sampled about every 0.1 seconds and their arithmetic mean is recorded.
    A value of zero performs one immediate read.
    """
    detectors = _as_scalar_readable_list(detectors)
    acquisition_devices = _scalar_acquisition_devices(detectors)
    triggered_devices, passive_signals = _scalar_acquisition_layout(
        detectors,
        acquisition_devices,
    )
    hardware_readables = _selected_hardware_readables(detectors, passive_signals)
    passive_signal_pairs = _software_average_signals(passive_signals)
    file_type = _normalize_scalar_file_type(file_type)
    num = int(num)
    delay = float(delay)
    if num < 1:
        raise ValueError("num must be at least 1")
    if delay < 0:
        raise ValueError("delay must be non-negative")

    acquire_signals, original_times, effective_time = yield from _configure_scalar_acquire_time(
        acquisition_devices, acquire_time
    )
    estimate = estimate_plan_runtime(
        "count_scalar",
        kwargs={"num": num, "delay": delay, "acquire_time": effective_time},
        context={},
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or num)

    detector_names = [detector.name for detector in detectors]
    live_plot_fields, live_monitor_signals = _build_live_plot_fields(detectors)
    _configure_passive_live_fields(
        live_plot_fields,
        live_monitor_signals,
        passive_signal_pairs,
    )
    acquisition_monitor = _build_acquisition_monitor(
        acquisition_devices,
        duration=effective_time if acquire_signals else None,
    )
    _md = {
        "title": title,
        "sample": sample,
        "detectors": detector_names,
        "plan_args": {
            "detectors": detector_names,
            "acquire_time": (
                effective_time
                if acquire_signals or acquire_time is not None
                else None
            ),
            "num": num,
            "delay": delay,
            "file_type": file_type,
        },
        "det_config": _scalar_detector_config(
            acquisition_devices,
            acquire_signals,
            software_dwell_time=effective_time if passive_signals else None,
        ),
        "num_points": num,
        "num_intervals": num - 1,
        "estimated_total_time_s": total_time,
        "estimated_total_units": total_units,
        "experiment_type": "diffraction",
        "data_type": "scalar",
        "detector_type": _scalar_detector_type(acquisition_devices),
        "file_type": file_type,
        "live_plot_fields": live_plot_fields,
        "plan_name": "count_scalar",
        "hints": {},
    }
    if acquisition_monitor:
        _md["acquisition_monitor"] = acquisition_monitor
    _md.update(md or {})
    _md["file_type"] = file_type
    _md["hints"].setdefault("dimensions", [(("time",), "primary")])

    predeclare = os.environ.get("BLUESKY_PREDECLARE", False)

    @bpp.stage_decorator(acquisition_devices)
    @bpp.run_decorator(md=_md)
    def inner_count():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )
        if predeclare:
            yield from bps.declare_stream(
                *hardware_readables,
                *(output for _source, output in passive_signal_pairs),
                name="primary",
            )
        for index in range(num):
            yield from bps.checkpoint()
            yield from progress.on_unit_start(index)
            yield from _acquire_scalar_point(
                triggered_devices,
                hardware_readables,
                passive_signal_pairs,
                software_dwell_time=effective_time,
            )
            yield from progress.on_unit_success(index)
            if delay and index < num - 1:
                yield from bps.sleep(delay)

    acquisition_plan = inner_count()
    if live_monitor_signals:
        acquisition_plan = bpp.monitor_during_wrapper(
            acquisition_plan,
            live_monitor_signals,
        )
    return (
        yield from bpp.finalize_wrapper(
            acquisition_plan,
            _restore_scalar_acquire_times(original_times),
        )
    )


@parameter_annotation_decorator(
    {
        "parameters": {
            **_SCALAR_DEVICE_ANNOTATION["parameters"],
            "file_type": _SCALAR_FILE_TYPE_ANNOTATION,
            "motor": {
                "annotation": "typing.Union[str, Motors]",
                "description": "Motor to scan (must be movable)",
                "devices": {"Motors": _collect_movable_names()},
                "convert_device_names": True,
            },
        }
    }
)
def scan_scalar(
    title: str,
    sample: str = "",
    *,
    detectors,
    motor,
    start_pos: float,
    stop_pos: float,
    step_size: float = None,
    num_steps: int = None,
    acquire_time: float = None,
    return_to_original_position: bool = True,
    file_type: str = "csv",
    md: dict = None,
):
    """Scan a motor while recording hardware-timed or averaged scalar data.

    Passive Signals are sampled and averaged at each motor position for
    ``acquire_time`` seconds. A zero-second dwell performs one immediate read.
    """
    detectors = _as_scalar_readable_list(detectors)
    acquisition_devices = _scalar_acquisition_devices(detectors)
    triggered_devices, passive_signals = _scalar_acquisition_layout(
        detectors,
        acquisition_devices,
    )
    hardware_readables = _selected_hardware_readables(detectors, passive_signals)
    passive_signal_pairs = _software_average_signals(passive_signals)
    file_type = _normalize_scalar_file_type(file_type)
    original_position = motor.position
    positions, num_steps_calc, step_size_calc, stop_pos_calc = (
        _scan_positions_from_num_or_step_size(
            start_pos,
            stop_pos,
            num_steps=num_steps,
            step_size=step_size,
        )
    )

    acquire_signals, original_times, effective_time = yield from _configure_scalar_acquire_time(
        acquisition_devices, acquire_time
    )
    estimate = estimate_plan_runtime(
        "scan_scalar",
        kwargs={
            "start_pos": start_pos,
            "stop_pos": stop_pos_calc,
            "step_size": step_size_calc,
            "num_steps": num_steps_calc,
            "acquire_time": effective_time,
        },
        context={},
    )
    total_time = float(estimate.get("estimated_total_time_s") or 0.0)
    total_units = int(estimate.get("estimated_total_units") or num_steps_calc)

    detector_names = [detector.name for detector in detectors]
    live_plot_fields, live_monitor_signals = _build_live_plot_fields(detectors)
    _configure_passive_live_fields(
        live_plot_fields,
        live_monitor_signals,
        passive_signal_pairs,
    )
    acquisition_monitor = _build_acquisition_monitor(
        acquisition_devices,
        duration=effective_time if acquire_signals else None,
    )
    _md = {
        "title": title,
        "sample": sample,
        "estimated_total_time_s": total_time,
        "estimated_total_units": total_units,
        "detectors": detector_names,
        "plan_args": {
            "detectors": detector_names,
            "motor": motor.name,
            "acquire_time": (
                effective_time
                if acquire_signals or acquire_time is not None
                else None
            ),
            "start_pos": start_pos,
            "stop_pos": stop_pos_calc,
            "step_size": step_size_calc,
            "num_steps": num_steps_calc,
            "file_type": file_type,
        },
        "det_config": _scalar_detector_config(
            acquisition_devices,
            acquire_signals,
            software_dwell_time=effective_time if passive_signals else None,
        ),
        "num_points": num_steps_calc,
        "num_intervals": num_steps_calc - 1,
        "experiment_type": "diffraction",
        "data_type": "scalar",
        "detector_type": _scalar_detector_type(acquisition_devices),
        "file_type": file_type,
        "live_plot_fields": live_plot_fields,
        "plan_name": "scan_scalar",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": {
            "motor": motor.name,
            "start_pos": start_pos,
            "stop_pos": stop_pos_calc,
            "step_size": step_size_calc,
            "num_steps": num_steps_calc,
        },
        "motors": [motor.name],
    }
    if acquisition_monitor:
        _md["acquisition_monitor"] = acquisition_monitor
    _md.update(md or {})
    _md["file_type"] = file_type
    _set_scan_motor_metadata(_md, [motor])

    @bpp.stage_decorator(acquisition_devices + [motor])
    @bpp.run_decorator(md=_md)
    def inner_scan():
        progress = _ProgressEstimator(
            total_units=total_units,
            initial_total_time_s=total_time,
        )
        position_cache = defaultdict(lambda: None)
        scan_cycler = cycler(motor, positions)
        for step in scan_cycler:
            step_t0 = time.monotonic()
            yield from progress.mark_started()
            yield from bps.move_per_step(step, position_cache)
            yield from _acquire_scalar_point(
                triggered_devices,
                hardware_readables,
                passive_signal_pairs,
                software_dwell_time=effective_time,
                extra_readables=step.keys(),
            )
            yield from progress.on_units_success(
                unit_count=1,
                elapsed_s=max(0.0, time.monotonic() - step_t0),
            )

    def cleanup():
        if return_to_original_position:
            yield from bps.mv(motor, original_position)
        yield from _restore_scalar_acquire_times(original_times)

    acquisition_plan = inner_scan()
    if live_monitor_signals:
        acquisition_plan = bpp.monitor_during_wrapper(
            acquisition_plan,
            live_monitor_signals,
        )
    return (yield from bpp.finalize_wrapper(acquisition_plan, cleanup()))
