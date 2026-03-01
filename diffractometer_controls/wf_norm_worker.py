import os

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
    return np.asarray(arr, dtype=np.float32)


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
        base_shape = None
        for idx, path in enumerate(file_paths, start=1):
            img = _load_wf_image(path)
            if base_shape is None:
                base_shape = img.shape
            elif img.shape != base_shape:
                return {
                    "ok": False,
                    "error": f"White-field image size mismatch: expected {base_shape}, got {img.shape}.",
                }
            images.append(img)
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
        return {
            "ok": True,
            "norm_array": np.asarray(filtered, dtype=np.float32),
            "shape": tuple(filtered.shape),
            "count": int(len(images)),
            "filter_requested": requested,
            "filter_used": method_used,
            "filter_note": note,
        }
    except Exception as ex:
        return {"ok": False, "error": str(ex)}
