"""Read diffraction runs from Tiled without Qt, QueueServer, or EPICS.

Snapshots contain the original run metadata, descriptors, rows, and event
sequence numbers consumed by the existing diffraction plot controller.
HDF5 references are read through Tiled; client machines need no file mounts.
"""

import ast
import datetime as dt
from collections.abc import Mapping

import numpy as np

DIFFRACTION_EXPERIMENT_TYPE = "diffraction"


def _coerce_number(value):
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if np.isfinite(number) else None


def _catalog_exists(container, name):
    try:
        if name in container:
            return True
    except Exception:
        pass
    try:
        container[name]
    except Exception:
        return False
    return True


def _resolve_catalog_name(container, preferred_names):
    for name in preferred_names:
        if _catalog_exists(container, name):
            return name
    return preferred_names[0]




def _run_metadata(run):
    item = getattr(run, "item", None)
    if isinstance(item, Mapping):
        try:
            metadata = item["attributes"]["metadata"]
        except Exception:
            metadata = None
        if isinstance(metadata, Mapping):
            return dict(metadata)

    metadata = {}
    getter = getattr(run, "metadata", None)
    if callable(getter):
        try:
            metadata = getter()
        except Exception:
            metadata = {}
    elif isinstance(getter, Mapping):
        metadata = getter
    elif isinstance(getattr(run, "_metadata", None), Mapping):
        metadata = run._metadata

    if isinstance(metadata, Mapping):
        return dict(metadata)

    v1 = getattr(run, "v1", None)
    getter = getattr(v1, "metadata", None)
    if callable(getter):
        try:
            metadata = getter()
        except Exception:
            metadata = {}
    elif isinstance(getter, Mapping):
        metadata = getter

    return dict(metadata or {})


def _run_start_stop_docs(run):
    metadata = _run_metadata(run)
    start_doc = metadata.get("start")
    stop_doc = metadata.get("stop")
    if isinstance(start_doc, dict):
        return dict(start_doc), dict(stop_doc or {}) if stop_doc is not None else None

    start_attr = getattr(run, "start", None)
    stop_attr = getattr(run, "stop", None)
    if isinstance(start_attr, Mapping):
        return dict(start_attr), dict(stop_attr or {}) if isinstance(stop_attr, Mapping) else None

    v1 = getattr(run, "v1", None)
    start_attr = getattr(v1, "start", None)
    stop_attr = getattr(v1, "stop", None)
    if isinstance(start_attr, Mapping):
        return dict(start_attr), dict(stop_attr or {}) if isinstance(stop_attr, Mapping) else None

    return dict(metadata or {}), None


def _is_diffraction_start_doc(start_doc):
    if not isinstance(start_doc, dict):
        return False
    return str(start_doc.get("experiment_type", "") or "").strip().lower() == DIFFRACTION_EXPERIMENT_TYPE


def _run_data_type(start_doc):
    """Return the detector-output shape, retaining compatibility with old runs."""
    start_doc = dict(start_doc or {})
    data_type = str(start_doc.get("data_type", "") or "").strip().lower()
    if data_type in {"scalar", "1d", "2d"}:
        return data_type
    plan_name = str(start_doc.get("plan_name", "") or "").strip().lower()
    if plan_name in {"count_scalar", "scan_scalar"}:
        return "scalar"
    return "1d"


def _start_live_plot_fields(start_doc):
    fields = dict(dict(start_doc or {}).get("live_plot_fields", {}) or {})
    return {
        str(data_key): dict(spec or {})
        for data_key, spec in fields.items()
        if str(data_key)
    }


def _data_keys_have_plot_payload(data_keys, start_doc):
    data_keys = _plain_mapping(data_keys)
    declared_fields = set(_start_live_plot_fields(start_doc))
    if declared_fields.intersection(str(key) for key in data_keys):
        return True
    if _run_data_type(start_doc) == "scalar":
        return bool(_legacy_scalar_fields(data_keys, start_doc)) if not declared_fields else False
    if _run_data_type(start_doc) == "1d" and any(
        isinstance(spec, Mapping) and spec.get("dtype") == "array" and len(spec.get("shape", [])) == 1
        for spec in data_keys.values()
    ):
        return True
    return any(
        str(key).endswith("_counts")
        or str(key).endswith("_position_x")
        or str(key).endswith("_total_counts")
        for key in data_keys
    )


def _legacy_scalar_fields(data_keys, start_doc):
    """Recover field labels from recorded descriptors in older scalar runs."""
    detectors = set(start_doc.get("detectors", []) or [])
    motors = set(start_doc.get("motors", []) or [])
    fields = {}
    for key, spec in data_keys.items():
        spec = _plain_mapping(spec)
        source = spec.get("object_name", "")
        if key in motors or source in motors:
            continue
        if detectors and key not in detectors and source not in detectors:
            continue
        if spec.get("shape") or spec.get("dtype") not in {"number", "integer"}:
            continue
        fields[key] = {"label": key, "role": "signal", "units": spec.get("units", ""),
                       "source": source, "transport": "document"}
    return fields


def _format_run_timestamp(epoch_seconds):
    value = _coerce_number(epoch_seconds)
    if value is None:
        return ""
    try:
        dt_value = dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc).astimezone()
    except Exception:
        return ""
    return dt_value.strftime("%Y-%m-%d %H:%M:%S")


def _make_history_entry(uid, start_doc, stop_doc=None):
    from .metadata import run_outcome
    start_doc = dict(start_doc or {})
    title = str(start_doc.get("title", "") or "").strip()
    plan_name = str(start_doc.get("plan_name", "") or "").strip()
    time_value = _coerce_number(start_doc.get("time"))
    timestamp_text = _format_run_timestamp(time_value)
    base_label = title or plan_name or str(uid or "")[:8] or "Diffraction Run"
    outcome = run_outcome(start_doc, stop_doc)
    label = base_label if not timestamp_text else f"{base_label} | {timestamp_text}"
    label += f" | {outcome['status']}"
    return {
        "uid": str(uid or start_doc.get("uid", "") or "").strip(),
        "label": label,
        "title": title,
        "plan_name": plan_name,
        "scan_id": start_doc.get("scan_id"),
        "sample": str(start_doc.get("sample", "") or ""),
        "data_type": _run_data_type(start_doc),
        "time": float(time_value) if time_value is not None else None,
        **outcome,
    }


def _plain_mapping(value):
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value or {})
    except Exception:
        return {}


def _to_plain_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _to_plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    return value


def _stream_descriptor_docs(run_uid, stream_name, stream, start_doc):
    descriptor_docs = []
    descriptors = getattr(stream, "descriptors", None)
    if descriptors is None:
        descriptors = []

    for index, descriptor in enumerate(list(descriptors or [])):
        descriptor_doc = _plain_mapping(descriptor)
        if not descriptor_doc:
            continue
        descriptor_doc = {key: _to_plain_value(value) for key, value in descriptor_doc.items()}
        descriptor_doc.setdefault("uid", f"{run_uid}-{stream_name}-descriptor-{index}")
        descriptor_doc.setdefault("name", str(stream_name or "primary"))
        descriptor_doc.setdefault("run_start", str(run_uid or ""))
        descriptor_docs.append(descriptor_doc)

    if descriptor_docs:
        return descriptor_docs

    stream_metadata = _plain_mapping(getattr(stream, "metadata", {}) or {})
    data_keys = _plain_mapping(stream_metadata.get("data_keys"))
    if not data_keys:
        return []

    return [
        {
            "uid": f"{run_uid}-{stream_name}-descriptor-0",
            "name": str(stream_name or "primary"),
            "run_start": str(run_uid or ""),
            "time": _coerce_number(stream_metadata.get("time"))
            or _coerce_number(start_doc.get("time"))
            or 0.0,
            "configuration": _to_plain_value(stream_metadata.get("configuration", {}) or {}),
            "object_keys": _to_plain_value(stream_metadata.get("object_keys", {}) or {}),
            "data_keys": _to_plain_value(data_keys),
        }
    ]


def _stream_row_count(dataset, data_keys):
    row_counts = []
    saw_constant_like = False
    data_vars = getattr(dataset, "data_vars", {}) or {}
    for var_name in list(getattr(data_vars, "keys", lambda: [])()):
        variable = dataset[var_name]
        values = getattr(variable, "values", variable)
        shape = tuple(getattr(values, "shape", ()) or ())
        data_key = _plain_mapping(dict(data_keys or {}).get(var_name))
        target_shape = tuple(data_key.get("shape", []) or ())
        if not shape:
            continue
        if target_shape and shape == target_shape:
            saw_constant_like = True
            continue
        row_counts.append(int(shape[0]))

    if row_counts:
        return max(row_counts)
    if saw_constant_like:
        return 1

    sizes = getattr(dataset, "sizes", {}) or {}
    try:
        return max(int(value) for value in sizes.values())
    except Exception:
        return 0


def _dataset_column_to_event_list(values, *, row_count, data_key):
    raw_values = getattr(values, "values", values)
    if isinstance(raw_values, np.ndarray):
        shape = tuple(raw_values.shape)
        python_value = raw_values.tolist()
    else:
        shape = tuple(getattr(raw_values, "shape", ()) or ())
        python_value = _to_plain_value(raw_values)

    target_shape = tuple(_plain_mapping(data_key).get("shape", []) or ())

    if not shape:
        item = _to_plain_value(
            raw_values.item() if hasattr(raw_values, "item") else raw_values
        )
        return [item for _ in range(max(1, int(row_count or 1)))]

    if shape and int(shape[0]) == int(max(1, row_count or 1)) and not (
        target_shape and shape == target_shape
    ):
        items = python_value if isinstance(python_value, list) else list(python_value)
        return [_to_plain_value(item) for item in items]

    if target_shape and shape == target_shape:
        item = _to_plain_value(python_value)
        return [item for _ in range(max(1, int(row_count or 1)))]

    if int(max(1, row_count or 1)) == 1:
        return [_to_plain_value(python_value)]

    try:
        items = list(raw_values)
    except Exception:
        return None
    if len(items) != int(max(1, row_count or 1)):
        return None
    return [_to_plain_value(item) for item in items]


def _run_streams(run):
    """Support both historical direct streams and the newer streams namespace."""
    if "streams" in run:
        return run["streams"]
    return run


def _restore_array(value, spec):
    """Handle older SQL columns that returned numeric arrays as strings."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            if not (value.startswith("[") and value.endswith("]")):
                raise ValueError("Stored array is not a numeric array.")
            # NumPy's whitespace representation is not a Python literal.
            # float() rejects truncation markers and nonnumeric content.
            value = [float(token) for token in value[1:-1].split()]
    array = np.asarray(value)
    expected = tuple(spec.get("shape", []))
    if array.shape != expected:
        raise ValueError(f"Stored array shape {array.shape} differs from recorded shape {expected}.")
    return _to_plain_value(array)


def select_history_stream(run, start_doc, *, stream_name="primary"):
    run_uid = str(start_doc.get("uid", ""))
    streams = _run_streams(run)
    ranked = []
    for name in streams:
        name = str(name)
        if name == "artifacts":
            continue
        stream = streams[name]
        descriptors = _stream_descriptor_docs(run_uid, name, stream, start_doc)
        if not descriptors:
            continue
        data_keys = _plain_mapping(descriptors[0].get("data_keys"))
        if not _data_keys_have_plot_payload(data_keys, start_doc):
            continue
        priority = 0 if name == stream_name else (2 if name == "baseline" else 1)
        ranked.append((priority, name, stream, data_keys))
    if not ranked:
        return None, None, None
    _, name, stream, data_keys = min(ranked, key=lambda item: item[:2])
    return name, stream, data_keys


def suggest_diffraction_start_updates(run):
    """Recover absent 1D fields from explicitly tagged diffraction descriptors.

    No network writes or array reads. Explicit experiment types and existing
    values are preserved; untagged runs and imaging are excluded.
    """
    start, _ = _run_start_stop_docs(run)
    if not _is_diffraction_start_doc(start):
        return {}
    streams = _run_streams(run)
    if "primary" not in streams:
        return {}
    docs = _stream_descriptor_docs(start.get("uid", ""), "primary", streams["primary"], start)
    detectors = set()
    for doc in docs:
        keys = doc.get("data_keys", {})
        for key, spec in keys.items():
            if not key.endswith("_counts") or spec.get("dtype") != "array" or len(spec.get("shape", [])) != 1:
                continue
            name = key.removesuffix("_counts")
            axis = keys.get(f"{name}_position_x", {})
            if axis.get("dtype") == "array" and axis.get("shape") == spec.get("shape"):
                detectors.add(name)
    if not detectors:
        return {}
    updates = {}
    if "data_type" not in start:
        updates["data_type"] = "1d"
    if "detectors" not in start:
        updates["detectors"] = sorted(detectors)
    return updates


def build_history_snapshot(
    run, start_doc=None, stop_doc=None, *, stream_name="primary", allow_empty=False,
):
    """Load one PSD or scalar run using its recorded configuration.

    Missing arrays and failed exports raise useful errors rather than producing
    a partial-looking plot. ``allow_empty`` is for live startup backfill only;
    arrays in new runs become readable after HDF5 finalization.
    """
    if start_doc is None:
        start_doc, stop_doc = _run_start_stop_docs(run)
    if not _is_diffraction_start_doc(start_doc):
        raise ValueError("Selected run is not tagged as diffraction data.")
    inferred = suggest_diffraction_start_updates(run)
    start_doc = {**inferred, **start_doc}
    snapshot = {
        "uid": str(start_doc.get("uid", "")),
        "stream_name": "",
        "start": _to_plain_value(start_doc),
        "stop": _to_plain_value(stop_doc),
        "descriptor_docs": [],
        "rows": [],
        "seq_nums": [],
        "pending_reason": "",
        "file_export": _to_plain_value(_run_metadata(run).get("file_export", {})),
        "inferred_metadata": inferred,
    }

    def unavailable(message):
        if allow_empty:
            snapshot["pending_reason"] = message
            return snapshot
        raise RuntimeError(message)

    export = snapshot["file_export"]
    if export and export.get("status") != "ready":
        status = str(export.get("status", "unknown"))
        detail = str(export.get("error", "") or "")
        return unavailable(
            f"HDF5 export is {status}; stored arrays are available after the run finishes "
            f"and registration succeeds.{(' ' + detail) if detail else ''}"
        )
    name, stream, data_keys = select_history_stream(run, start_doc, stream_name=stream_name)
    if stream is None:
        return unavailable("No diffraction-compatible stream was found for this run.")
    snapshot["stream_name"] = name
    snapshot["descriptor_docs"] = _stream_descriptor_docs(snapshot["uid"], name, stream, start_doc)
    if _run_data_type(start_doc) == "scalar" and not _start_live_plot_fields(start_doc):
        snapshot["start"]["live_plot_fields"] = _legacy_scalar_fields(data_keys, start_doc)
    try:
        dataset = stream.read()
    except Exception as exc:
        return unavailable(f"Unable to read Tiled stream {name!r}: {exc}")
    row_count = _stream_row_count(dataset, data_keys)
    if row_count <= 0:
        return unavailable(f"Tiled stream {name!r} does not contain stored rows.")
    columns = {}
    for key in dataset.data_vars:
        values = _dataset_column_to_event_list(
            dataset[key], row_count=row_count, data_key=data_keys.get(key),
        )
        if values is None or len(values) != row_count:
            return unavailable(f"Stored field {key!r} does not match the {row_count} event rows.")
        spec = _plain_mapping(data_keys.get(key))
        if spec.get("dtype") == "array":
            try:
                values = [_restore_array(value, spec) for value in values]
            except (ValueError, TypeError) as exc:
                return unavailable(f"Unable to restore stored array {key!r}: {exc}")
        columns[str(key)] = values
    missing = set(data_keys) - set(columns)
    if missing:
        return unavailable(f"Tiled stream {name!r} is missing stored fields: {', '.join(sorted(missing))}.")
    if not _data_keys_have_plot_payload({key: data_keys.get(key, {}) for key in columns}, snapshot["start"]):
        return unavailable(f"Tiled stream {name!r} has no stored diffraction values.")
    # Event sequence numbers need not be contiguous (e.g. interrupted streams).
    # V3 excludes seq_num from its default data read, while V2 often exposes it
    # as an xarray coordinate. Read the small sequence column if necessary.
    if "seq_num" in dataset:
        seq_nums = np.asarray(dataset["seq_num"]).tolist()
    elif "seq_num" in stream:
        seq_nums = np.asarray(stream["seq_num"].read()).tolist()
    else:
        seq_nums = list(range(1, row_count + 1))
    if not isinstance(seq_nums, list) or len(seq_nums) != row_count:
        return unavailable("Stored sequence numbers do not match the event rows.")
    snapshot["rows"] = [
        {key: values[index] for key, values in columns.items()}
        for index in range(row_count)
    ]
    snapshot["seq_nums"] = seq_nums
    return snapshot


def fetch_recent_history_entries(catalog, *, limit=100):
    """Return newest diffraction runs, filtering and sorting on the server."""
    from tiled.queries import Key

    limit = max(1, int(limit))
    # Explicit start paths also work when the Bluesky client specialization
    # is unavailable. Older catalogs expose the start fields at the top level.
    for prefix in ("start.", ""):
        tree = catalog.search(
            Key(f"{prefix}experiment_type") == DIFFRACTION_EXPERIMENT_TYPE
        )
        entries = []
        for uid, run in tree.sort((f"{prefix}time", -1)).items():
            start, stop = _run_start_stop_docs(run)
            if _is_diffraction_start_doc(start):
                entry = _make_history_entry(uid, start, stop)
                export = _run_metadata(run).get('file_export') or {}
                entry.update(file_status=export.get('status', ''), storage_error=export.get('error', ''))
                entries.append(entry)
            if len(entries) >= limit:
                break
        if entries:
            return entries
    return []
