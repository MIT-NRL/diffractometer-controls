import os
import re

import numpy as np


def _push_progress(progress_queue, stage, **kwargs):
    if progress_queue is None:
        return
    payload = {"stage": str(stage)}
    payload.update(kwargs)
    try:
        progress_queue.put_nowait(payload)
    except Exception:
        pass


def _load_wf_image(path):
    ext = os.path.splitext(str(path))[1].lower()
    exposure_s = None
    if ext in {".tif", ".tiff"}:
        exposure_s = _read_tiff_exposure_seconds(path)
    data = None
    if ext == ".npy":
        data = np.load(path)
    else:
        try:
            import tifffile
            data = tifffile.imread(path)
        except Exception:
            from PIL import Image
            with Image.open(path) as img:
                data = np.array(img)

    arr = np.asarray(data)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.median(arr[..., :3], axis=-1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image data, got shape {arr.shape} for '{path}'.")
    return np.asarray(arr, dtype=np.float32), exposure_s


_EXPOSURE_TIME_KEY_STRONG_HINTS = (
    "acquiretime",
    "exposuretime",
)
_EXPOSURE_TIME_KEY_WEAK_HINTS = (
    "exposure",
)
_EXPOSURE_PAIR_RE = re.compile(
    r"([A-Za-z0-9_:. /()-]{1,96})\s*[:=]\s*([^\n\r;|,]+)"
)
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_RATIO_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*/\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z\u00b5\u03bc]*)\s*$"
)
_VALUE_WITH_UNIT_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([A-Za-z\u00b5\u03bc]*)\s*$"
)
_KEY_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _normalize_key_text(key) -> str:
    text = str(key).strip().lower()
    return _KEY_NORMALIZE_RE.sub("", text)


def _exposure_key_score(key) -> int:
    text = _normalize_key_text(key)
    if not text:
        return 0
    if any(h in text for h in _EXPOSURE_TIME_KEY_STRONG_HINTS):
        return 2
    if text in _EXPOSURE_TIME_KEY_WEAK_HINTS:
        return 1
    return 0


def _unit_to_seconds(unit: str):
    u = str(unit or "").strip().lower()
    u = u.replace("\u03bc", "u").replace("\u00b5", "u")
    if not u:
        return 1.0
    unit_map = {
        "s": 1.0,
        "sec": 1.0,
        "secs": 1.0,
        "second": 1.0,
        "seconds": 1.0,
        "ms": 1e-3,
        "msec": 1e-3,
        "msecs": 1e-3,
        "millisecond": 1e-3,
        "milliseconds": 1e-3,
        "us": 1e-6,
        "usec": 1e-6,
        "usecs": 1e-6,
        "microsecond": 1e-6,
        "microseconds": 1e-6,
        "ns": 1e-9,
        "nsec": 1e-9,
        "nsecs": 1e-9,
        "nanosecond": 1e-9,
        "nanoseconds": 1e-9,
    }
    return unit_map.get(u, None)


def _parse_time_value_text(text: str):
    s = str(text).strip()
    if not s:
        return None

    m = _RATIO_RE.match(s)
    if m is not None:
        try:
            num = float(m.group(1))
            den = float(m.group(2))
        except Exception:
            return None
        if not np.isfinite(num) or not np.isfinite(den) or den == 0:
            return None
        ratio = num / den
        factor = _unit_to_seconds(m.group(3))
        if factor is None:
            return None
        return _to_finite_positive_float(ratio * factor)

    m = _VALUE_WITH_UNIT_RE.match(s)
    if m is None:
        return None
    try:
        val = float(m.group(1))
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    factor = _unit_to_seconds(m.group(2))
    if factor is None:
        return None
    return _to_finite_positive_float(val * factor)


def _looks_like_exposure_key(key) -> bool:
    return _exposure_key_score(key) > 0


def _to_finite_positive_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out) or out <= 0:
        return None
    return float(out)


def _parse_exposure_from_text(text: str):
    s = str(text).strip()
    if not s:
        return None
    best_score = -1
    best_value = None
    for m in _EXPOSURE_PAIR_RE.finditer(s):
        key = m.group(1).strip()
        score = _exposure_key_score(key)
        if score <= 0:
            continue
        out = _parse_time_value_text(m.group(2))
        if out is not None:
            if score > best_score:
                best_score = score
                best_value = out
    return best_value


def _parse_exposure_from_tag_value(value, *, allow_plain_numeric=False):
    if value is None:
        return None
    if isinstance(value, dict):
        for k, v in value.items():
            if _looks_like_exposure_key(k):
                out = _parse_exposure_from_tag_value(v, allow_plain_numeric=True)
                if out is not None:
                    return out
            out = _parse_exposure_from_tag_value(v, allow_plain_numeric=False)
            if out is not None:
                return out
        return None
    if isinstance(value, (list, tuple)):
        if allow_plain_numeric and len(value) == 2:
            try:
                num = float(value[0])
                den = float(value[1])
                if np.isfinite(num) and np.isfinite(den) and den != 0:
                    out = _to_finite_positive_float(num / den)
                    if out is not None:
                        return out
            except Exception:
                pass
        for item in value:
            out = _parse_exposure_from_tag_value(item, allow_plain_numeric=allow_plain_numeric)
            if out is not None:
                return out
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if allow_plain_numeric:
            return _to_finite_positive_float(value)
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    out = _parse_exposure_from_text(text)
    if out is not None:
        return out
    if allow_plain_numeric:
        out = _parse_time_value_text(text)
        if out is not None:
            return out
        float_match = _FLOAT_RE.search(text)
        if float_match is not None:
            return _to_finite_positive_float(float_match.group(0))
    return None


def _read_tiff_exposure_seconds(path):
    exposure_tag_code = 33434  # EXIF ExposureTime
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            if int(exposure_tag_code) in tags:
                out = _parse_exposure_from_tag_value(
                    tags[int(exposure_tag_code)].value, allow_plain_numeric=True
                )
                if out is not None:
                    return out
            for tag in tags.values():
                name = str(getattr(tag, "name", "")).strip().lower()
                allow_plain = _looks_like_exposure_key(name)
                out = _parse_exposure_from_tag_value(
                    getattr(tag, "value", None), allow_plain_numeric=allow_plain
                )
                if out is not None:
                    return out
    except Exception:
        pass

    try:
        from PIL import Image

        with Image.open(path) as img:
            tags = getattr(img, "tag_v2", None)
            if tags is None:
                return None
            if int(exposure_tag_code) in tags:
                out = _parse_exposure_from_tag_value(
                    tags.get(int(exposure_tag_code)), allow_plain_numeric=True
                )
                if out is not None:
                    return out
            for code, value in tags.items():
                allow_plain = int(code) == int(exposure_tag_code)
                out = _parse_exposure_from_tag_value(value, allow_plain_numeric=allow_plain)
                if out is not None:
                    return out
    except Exception:
        pass
    return None


def _median_filter_fallback(image, size):
    arr = np.asarray(image, dtype=np.float32)
    k = int(size)
    if k <= 1:
        return np.asarray(arr, dtype=np.float32)

    try:
        from scipy import ndimage as scipy_ndimage
    except Exception:
        scipy_ndimage = None
    if scipy_ndimage is not None:
        return np.asarray(scipy_ndimage.median_filter(arr, size=k, mode="nearest"), dtype=np.float32)

    # SciPy-free fallback for environments that do not ship scipy.
    try:
        sliding_window_view = np.lib.stride_tricks.sliding_window_view
    except Exception as ex:
        raise RuntimeError("No median filter backend available (requires scipy or numpy sliding_window_view).") from ex

    if arr.ndim != 2:
        raise ValueError(f"Median filter fallback expects 2D array, got shape {arr.shape}.")

    before = k // 2
    after = k - 1 - before
    padded = np.pad(arr, ((before, after), (before, after)), mode="edge")
    windows = sliding_window_view(padded, (k, k))
    filtered = np.median(windows, axis=(-1, -2))
    return np.asarray(filtered, dtype=np.float32)


def _remove_outliers_tomopy(image, size, ncore=None):
    import tomopy as tp
    arr = np.asarray(image, dtype=np.float32)
    input_ndim = arr.ndim
    if input_ndim == 2:
        arr = arr[np.newaxis, ...]
    elif input_ndim != 3:
        raise ValueError(f"remove_outlier expects 2D/3D input, got shape {arr.shape}.")

    # Use scalar window size for broad tomopy compatibility; tuple kernels can
    # produce unstable results on some builds.
    kernel = int(size)
    result = np.asarray(tp.remove_outlier(arr, dif=-1, size=kernel, axis=0, ncore=ncore), dtype=np.float32)
    if result.ndim == 2:
        result = result[np.newaxis, ...]
    elif result.ndim != 3:
        raise ValueError(f"remove_outlier returned unexpected shape {result.shape}.")

    result = np.asarray(
        -tp.remove_outlier(-result, dif=-1, size=kernel, axis=0, ncore=ncore),
        dtype=np.float32,
    )
    if input_ndim == 2:
        if result.ndim == 3:
            if result.shape[0] < 1:
                raise ValueError("remove_outlier returned empty first axis.")
            result = result[0]
        elif result.ndim != 2:
            raise ValueError(f"remove_outlier returned unexpected shape {result.shape}.")
    elif input_ndim == 3 and result.ndim == 2:
        result = result[np.newaxis, ...]

    return np.asarray(result, dtype=np.float32)


def _is_effectively_zero(image, eps=0.0):
    arr = np.asarray(image)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return True
    nz = np.count_nonzero(finite & (np.abs(arr) > float(eps)))
    return int(nz) == 0


def build_wf_norm_array_worker(
    file_paths,
    filter_method="outlier",
    outlier_size=7,
    median_size=6,
    progress_queue=None,
):
    if not file_paths:
        return {"ok": False, "error": "No white-field images were selected."}
    try:
        total = int(len(file_paths))
        _push_progress(progress_queue, "loading", current=0, total=total)
        images = []
        exposures = []
        base_shape = None
        for idx, path in enumerate(file_paths, start=1):
            img, exposure_s = _load_wf_image(path)
            if base_shape is None:
                base_shape = img.shape
            elif img.shape != base_shape:
                return {
                    "ok": False,
                    "error": f"White-field image size mismatch: expected {base_shape}, got {img.shape}.",
                }
            images.append(img)
            exp = _to_finite_positive_float(exposure_s)
            if exp is not None:
                exposures.append(exp)
            _push_progress(progress_queue, "loading", current=int(idx), total=total)

        if not images:
            return {"ok": False, "error": "No readable white-field images were loaded."}

        _push_progress(progress_queue, "combining", total=total)
        stack = np.stack(images, axis=0)
        combined = np.median(stack, axis=0)
        requested = str(filter_method).strip().lower() if filter_method is not None else "outlier"
        method_used = requested
        note = ""
        _push_progress(progress_queue, "filtering", method=requested)
        if requested == "median":
            filtered = _median_filter_fallback(combined, size=median_size)
            method_used = "median"
        else:
            try:
                filtered = _remove_outliers_tomopy(combined, size=outlier_size, ncore=None)
                method_used = "outlier"
                if _is_effectively_zero(filtered, eps=0.0):
                    filtered = _median_filter_fallback(combined, size=median_size)
                    method_used = "median"
                    note = "tomopy outlier result degenerate; used median fallback."
            except Exception:
                filtered = _median_filter_fallback(combined, size=median_size)
                method_used = "median"
                note = "tomopy unavailable; used median filter."
        _push_progress(progress_queue, "done")
        wf_exposure_s = float(np.mean(np.asarray(exposures, dtype=float))) if exposures else None
        return {
            "ok": True,
            "norm_array": np.asarray(filtered, dtype=np.float32),
            "shape": tuple(filtered.shape),
            "count": int(len(images)),
            "filter_requested": requested,
            "filter_used": method_used,
            "filter_note": note,
            "wf_exposure_s": wf_exposure_s,
            "wf_exposure_count": int(len(exposures)),
            "wf_exposure_min_s": float(np.min(exposures)) if exposures else None,
            "wf_exposure_max_s": float(np.max(exposures)) if exposures else None,
        }
    except Exception as ex:
        return {"ok": False, "error": str(ex)}
