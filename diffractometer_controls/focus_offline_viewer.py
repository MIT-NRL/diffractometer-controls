#!/usr/bin/env python3
"""Offline focus analysis viewer for TIFF image sequences.

This tool is designed for testing focus workflows without live motors/detectors.
It loads images from a directory, assigns fake motor positions, lets the user
set a manual ROI, fits an erf step+line model, and plots step sigma against
motor position.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import re
import sys
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# Keep per-task CPU usage predictable on shared control PCs.
# Users can override these before launch if they want higher throughput.
for _env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import numpy as np

try:
    from qtpy import QtCore, QtGui, QtWidgets
except Exception as ex:  # pragma: no cover
    raise SystemExit(f"qtpy is required: {ex}")

try:
    import pyqtgraph as pg
except Exception as ex:  # pragma: no cover
    raise SystemExit(f"pyqtgraph is required: {ex}")

try:
    from scipy import ndimage as scipy_ndimage
except Exception:
    scipy_ndimage = None

try:
    import lmfit as lm
except Exception:
    lm = None


def _read_image(path: Path) -> np.ndarray:
    """Read an image file into a 2D float array."""
    arr = None
    try:
        import tifffile

        arr = tifffile.imread(str(path))
    except Exception:
        pass

    if arr is None:
        try:
            import imageio.v3 as iio

            arr = iio.imread(str(path))
        except Exception:
            pass

    if arr is None:
        try:
            from PIL import Image

            arr = np.array(Image.open(path))
        except Exception as ex:
            raise RuntimeError(f"Could not read image '{path}': {ex}") from ex

    arr = np.asarray(arr)
    if arr.ndim == 3:
        # Use first channel if RGB-style data appears.
        arr = arr[..., 0]
    elif arr.ndim > 3:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])

    return np.asarray(arr, dtype=np.float64)


@dataclass(frozen=True)
class FrameInfo:
    index: int
    path: Path
    position: float


@dataclass
class FitResult:
    ok: bool
    step_sigma: float = np.nan
    step_sigma_stderr: float = np.nan
    step_center: float = np.nan
    psf_sigma: float = np.nan
    psf_sigma_stderr: float = np.nan
    mtf50: float = np.nan
    profile_x: Optional[np.ndarray] = None
    profile_y: Optional[np.ndarray] = None
    profile_fit: Optional[np.ndarray] = None
    lsf_x: Optional[np.ndarray] = None
    lsf_y: Optional[np.ndarray] = None
    mtf_f: Optional[np.ndarray] = None
    mtf_y: Optional[np.ndarray] = None
    edge_m: float = np.nan
    edge_b: float = np.nan
    edge_ymin: float = np.nan
    edge_ymax: float = np.nan
    edge_m_dynamic: float = np.nan
    edge_b_dynamic: float = np.nan


def _natural_key(path: Path):
    tokens = re.split(r"(\d+)", path.name)
    out = []
    for token in tokens:
        out.append(int(token) if token.isdigit() else token.lower())
    return out


_SERIES_INDEX_RE = re.compile(r"^(?P<series>.+)_(?P<idx>\d+)$")


def _extract_series_and_index(path: Path) -> Tuple[str, Optional[int]]:
    stem = path.stem
    m = _SERIES_INDEX_RE.match(stem)
    if not m:
        return stem, None
    series = m.group("series")
    try:
        idx = int(m.group("idx"))
    except Exception:
        idx = None
    return series, idx


def _load_positions_from_csv(csv_path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("filename") or row.get("file") or "").strip()
            pos_str = (row.get("position") or row.get("motor_position") or "").strip()
            if not name or not pos_str:
                continue
            try:
                out[name] = float(pos_str)
            except Exception:
                continue
    return out


def _parse_position_from_tag_value(value, key_name: str = "DetCameraFocusDist") -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            out = _parse_position_from_tag_value(item, key_name=key_name)
            if out is not None:
                return out
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            out = float(value)
            return out if np.isfinite(out) else None
        except Exception:
            return None
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    # Expected format from areaDetector TIFF tags: "DetCameraFocusDist:25.400000"
    if ":" in text:
        lhs, rhs = text.split(":", 1)
        if (not key_name) or (str(lhs).strip() == str(key_name).strip()):
            try:
                out = float(str(rhs).strip())
                return out if np.isfinite(out) else None
            except Exception:
                return None
    try:
        out = float(text)
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _read_tiff_focus_position(
    path: Path, *, tag_code: int = 65064, key_name: str = "DetCameraFocusDist"
) -> Optional[float]:
    def _can_contain_named_text(v) -> bool:
        if isinstance(v, (str, bytes, bytearray)):
            return True
        if isinstance(v, (list, tuple)):
            return any(isinstance(item, (str, bytes, bytearray)) for item in v)
        return False

    # Try Pillow first; lightweight and already a dependency fallback for image reads.
    try:
        from PIL import Image

        with Image.open(path) as img:
            tags = getattr(img, "tag_v2", None)
            if tags is not None:
                if int(tag_code) in tags:
                    out = _parse_position_from_tag_value(tags.get(int(tag_code)), key_name=key_name)
                    if out is not None:
                        return out
                # Fallback: scan all custom values for key_name prefix.
                for _k, v in tags.items():
                    if not _can_contain_named_text(v):
                        continue
                    out = _parse_position_from_tag_value(v, key_name=key_name)
                    if out is not None:
                        return out
    except Exception:
        pass

    # Fallback: tifffile if available.
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            if int(tag_code) in tags:
                out = _parse_position_from_tag_value(
                    tags[int(tag_code)].value, key_name=key_name
                )
                if out is not None:
                    return out
            for t in tags.values():
                v = getattr(t, "value", None)
                if not _can_contain_named_text(v):
                    continue
                out = _parse_position_from_tag_value(v, key_name=key_name)
                if out is not None:
                    return out
    except Exception:
        pass
    return None


def discover_frames(
    image_dir: Path,
    start_position: float,
    step_size: float,
    positions_csv: Optional[Path] = None,
    series_key: Optional[str] = None,
) -> List[FrameInfo]:
    # On Windows, glob is case-insensitive, so separate *.tif/*.TIF patterns
    # can return duplicates. Use a single suffix-based pass instead.
    all_files = [
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    ]
    if not all_files:
        raise RuntimeError(f"No TIFF files found in {image_dir}")

    parsed = []
    for path in all_files:
        series, idx = _extract_series_and_index(path)
        parsed.append((path, series, idx))

    series_counts: Dict[str, int] = {}
    for _, series, _ in parsed:
        series_counts[series] = series_counts.get(series, 0) + 1

    if series_key:
        parsed = [item for item in parsed if item[1] == series_key]
        if not parsed:
            available = ", ".join(sorted(series_counts))
            raise RuntimeError(
                f"Requested --series-key '{series_key}' not found. Available series: {available}"
            )
    elif len(series_counts) > 1:
        # Auto-select the most populated sequence to avoid mixing scans.
        series_key = max(series_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        parsed = [item for item in parsed if item[1] == series_key]
        print(
            f"Multiple TIFF series found ({len(series_counts)}). "
            f"Auto-selected '{series_key}' with {len(parsed)} frames. "
            "Use --series-key to override."
        )

    files = [
        p
        for p, _, _ in sorted(
            parsed,
            key=lambda item: (
                item[2] if item[2] is not None else 10**12,
                _natural_key(item[0]),
            ),
        )
    ]

    csv_positions: Dict[str, float] = {}
    if positions_csv is not None:
        csv_positions = _load_positions_from_csv(positions_csv)

    frames: List[FrameInfo] = []
    for i, path in enumerate(files):
        if path.name in csv_positions:
            pos = float(csv_positions[path.name])
        else:
            tag_pos = _read_tiff_focus_position(path)
            if tag_pos is not None:
                pos = float(tag_pos)
            else:
                pos = float(start_position + i * step_size)
        frames.append(FrameInfo(index=i, path=path, position=float(pos)))
    return frames


def build_frames_from_files(
    file_paths: Sequence[Path],
    start_position: float = 0.0,
    step_size: float = 1.0,
) -> List[FrameInfo]:
    files = []
    seen = set()
    for path in file_paths:
        p = Path(path).expanduser()
        try:
            p = p.resolve()
        except Exception:
            p = p.absolute()
        if (not p.exists()) or (not p.is_file()):
            continue
        if p.suffix.lower() not in {".tif", ".tiff"}:
            continue
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        files.append(p)
    if not files:
        raise RuntimeError("No valid TIFF files selected.")
    parsed = []
    for path in files:
        _series, idx = _extract_series_and_index(path)
        parsed.append((path, idx))
    files_sorted = [
        p
        for p, _ in sorted(
            parsed,
            key=lambda item: (
                item[1] if item[1] is not None else 10**12,
                _natural_key(item[0]),
            ),
        )
    ]
    out: List[FrameInfo] = []
    for i, path in enumerate(files_sorted):
        tag_pos = _read_tiff_focus_position(path)
        if tag_pos is not None:
            pos = float(tag_pos)
        else:
            pos = float(start_position + i * step_size)
        out.append(FrameInfo(index=i, path=path, position=pos))
    return out


def preprocess_image(image: np.ndarray, median_size: int = 6) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)

    if scipy_ndimage is not None:
        if median_size > 1:
            arr = scipy_ndimage.median_filter(arr, size=median_size)

    finite = np.isfinite(arr)
    if finite.any():
        low, high = np.nanpercentile(arr[finite], [0.5, 99.5])
        if np.isfinite(low) and np.isfinite(high) and high > low:
            arr = np.clip(arr, low, high)
    return arr


def preprocess_image_quick(
    image: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
    sample_step: int = 8,
    smooth_size: int = 3,
    smooth_passes: int = 1,
) -> np.ndarray:
    """
    Fast first-pass preprocess:
      - no median filter
      - coarse percentile clipping from a subsampled grid
    """
    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0:
        return arr

    step = int(max(1, sample_step))
    sample = arr[::step, ::step]
    finite_sample = np.isfinite(sample)
    if finite_sample.any():
        try:
            low, high = np.nanpercentile(sample[finite_sample], [float(low_pct), float(high_pct)])
        except Exception:
            low, high = np.nan, np.nan
        if np.isfinite(low) and np.isfinite(high) and high > low:
            arr = np.clip(arr, low, high)

    # Fast denoising pass to stabilize row-wise gradient picks without heavy blur.
    if scipy_ndimage is not None:
        k = int(max(1, smooth_size))
        n_pass = int(max(0, smooth_passes))
        if k > 1 and n_pass > 0:
            for _ in range(n_pass):
                arr = scipy_ndimage.uniform_filter(arr, size=k, mode="nearest")
    return arr


def _bounds_from_roi_rect(
    shape: Tuple[int, int], roi_rect: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    x, y, w, h = roi_rect
    x0 = int(max(0, math.floor(float(x))))
    y0 = int(max(0, math.floor(float(y))))
    x1 = int(min(shape[1], math.ceil(float(x + w))))
    y1 = int(min(shape[0], math.ceil(float(y + h))))
    if x1 <= x0:
        x1 = min(shape[1], x0 + 1)
    if y1 <= y0:
        y1 = min(shape[0], y0 + 1)
    return x0, x1, y0, y1


def _slanted_edge_esf(
    image: np.ndarray,
    x0_abs: float,
    y0_abs: float,
    distance_bin_px: float = 0.25,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float], Optional[float]]:
    """
    Build ESF using slanted-edge geometry:
      1) horizontal gradient
      2) strongest edge point per row
      3) fit edge line x(y)
      4) signed perpendicular distance for each pixel
      5) bin intensity vs distance
    """
    h, w = image.shape
    if h < 4 or w < 4:
        return None, None, None, None

    grad_x = np.gradient(image, axis=1)
    ys = []
    xs = []
    for y in range(h):
        row = grad_x[y, :]
        if not np.isfinite(row).any():
            continue
        idx = int(np.nanargmax(np.abs(row)))
        # avoid unstable edge picks at boundaries
        if idx <= 0 or idx >= (w - 1):
            continue
        ys.append(float(y0_abs + y))
        xs.append(float(x0_abs + idx))

    # Be tolerant for noisier quick-pass inputs and smaller ROIs.
    if len(xs) < max(6, int(0.15 * h)):
        return None, None, None, None

    y_arr = np.asarray(ys, dtype=np.float64)
    x_arr = np.asarray(xs, dtype=np.float64)

    try:
        # fit x = m*y + b
        m, b = np.polyfit(y_arr, x_arr, 1)
    except Exception:
        return None, None, None, None

    # one-pass robust outlier rejection and refit
    pred = m * y_arr + b
    resid = x_arr - pred
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if np.isfinite(mad) and mad > 0:
        keep = np.abs(resid) <= (3.5 * 1.4826 * mad)
        if int(np.count_nonzero(keep)) >= max(6, int(0.15 * h)):
            try:
                m, b = np.polyfit(y_arr[keep], x_arr[keep], 1)
            except Exception:
                pass

    x_esf, esf = _esf_from_edge_line(
        image=image,
        x0_abs=x0_abs,
        y0_abs=y0_abs,
        edge_m=float(m),
        edge_b=float(b),
        distance_bin_px=distance_bin_px,
    )
    if x_esf is None or esf is None:
        return None, None, None, None
    return x_esf, esf, float(m), float(b)


def _esf_from_edge_line(
    image: np.ndarray,
    x0_abs: float,
    y0_abs: float,
    edge_m: float,
    edge_b: float,
    distance_bin_px: float = 0.25,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    h, w = image.shape
    if h < 2 or w < 2:
        return None, None

    m = float(edge_m)
    b = float(edge_b)
    if not (np.isfinite(m) and np.isfinite(b)):
        return None, None

    # signed perpendicular distance to edge line x - m*y - b = 0
    x_coords = np.arange(w, dtype=np.float64) + float(x0_abs)
    y_coords = np.arange(h, dtype=np.float64) + float(y0_abs)
    xx, yy = np.meshgrid(x_coords, y_coords)
    denom = math.sqrt(1.0 + m * m)
    if not np.isfinite(denom) or denom <= 0:
        return None, None
    dist = (xx - (m * yy + b)) / denom

    intens = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(dist) & np.isfinite(intens)
    if int(np.count_nonzero(finite)) < 20:
        return None, None

    d = dist[finite].ravel()
    v = intens[finite].ravel()
    dmin = float(np.nanmin(d))
    dmax = float(np.nanmax(d))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        return None, None

    bw = max(0.05, float(distance_bin_px))
    edges = np.arange(dmin, dmax + bw, bw, dtype=np.float64)
    if edges.size < 4:
        return None, None

    idx = np.digitize(d, edges) - 1
    nb = edges.size - 1
    good = (idx >= 0) & (idx < nb)
    if int(np.count_nonzero(good)) < 20:
        return None, None

    sums = np.bincount(idx[good], weights=v[good], minlength=nb).astype(np.float64)
    cnts = np.bincount(idx[good], minlength=nb).astype(np.float64)
    keep = cnts > 0
    if int(np.count_nonzero(keep)) < 8:
        return None, None

    centers = 0.5 * (edges[:-1] + edges[1:])
    esf = sums[keep] / cnts[keep]
    x_esf = centers[keep]
    return x_esf, esf


def analyze_roi(
    image: np.ndarray,
    x0_abs: float,
    x1_abs: float,
    y0_abs: float,
    y1_abs: float,
    fixed_edge_line: Optional[Tuple[float, float]] = None,
    edge_median_size: int = 5,
) -> FitResult:
    """
    Fit slanted-edge ESF with StepModel(form='erf') + LinearModel.
    """
    result = FitResult(ok=False)
    if lm is None:
        return result
    if image.size == 0:
        return result

    # Use a median-smoothed ROI only for edge-line detection; keep ESF sampling on
    # the original ROI to avoid extra blur in the fitted step width.
    edge_image = np.asarray(image, dtype=np.float64)
    if scipy_ndimage is not None:
        k = int(max(1, edge_median_size))
        if k > 1:
            try:
                edge_image = scipy_ndimage.median_filter(edge_image, size=k)
            except Exception:
                edge_image = np.asarray(image, dtype=np.float64)

    x_data = None
    profile = None
    dyn_m = np.nan
    dyn_b = np.nan
    if fixed_edge_line is not None and len(fixed_edge_line) == 2:
        fixed_m = float(fixed_edge_line[0])
        fixed_b = float(fixed_edge_line[1])
        x_data, profile = _esf_from_edge_line(
            image=image,
            x0_abs=float(x0_abs),
            y0_abs=float(y0_abs),
            edge_m=fixed_m,
            edge_b=fixed_b,
            distance_bin_px=0.25,
        )
        if x_data is not None and profile is not None:
            result.edge_m = fixed_m
            result.edge_b = fixed_b
            result.edge_ymin = float(y0_abs)
            result.edge_ymax = float(max(y0_abs, y1_abs - 1))
        # Even in fixed-line mode, estimate dynamic edge geometry from the image.
        # This enables a true post-full-pass refinement of the fixed global line.
        _xd, _yd, dm, db = _slanted_edge_esf(
            edge_image,
            x0_abs=float(x0_abs),
            y0_abs=float(y0_abs),
            distance_bin_px=0.25,
        )
        if dm is not None and db is not None:
            dyn_m = float(dm)
            dyn_b = float(db)
            result.edge_m_dynamic = dyn_m
            result.edge_b_dynamic = dyn_b

    # Primary dynamic path: slanted-edge ESF from line fit + distance binning.
    if x_data is None or profile is None:
        _x_tmp, _y_tmp, edge_m, edge_b = _slanted_edge_esf(
            edge_image,
            x0_abs=float(x0_abs),
            y0_abs=float(y0_abs),
            distance_bin_px=0.25,
        )
        if edge_m is not None and edge_b is not None:
            x_data, profile = _esf_from_edge_line(
                image=image,
                x0_abs=float(x0_abs),
                y0_abs=float(y0_abs),
                edge_m=float(edge_m),
                edge_b=float(edge_b),
                distance_bin_px=0.25,
            )
            if (x_data is None or profile is None) and (_x_tmp is not None and _y_tmp is not None):
                x_data = _x_tmp
                profile = _y_tmp
        if x_data is not None and profile is not None and edge_m is not None and edge_b is not None:
            result.edge_m = float(edge_m)
            result.edge_b = float(edge_b)
            result.edge_ymin = float(y0_abs)
            result.edge_ymax = float(max(y0_abs, y1_abs - 1))
            result.edge_m_dynamic = float(edge_m)
            result.edge_b_dynamic = float(edge_b)
        else:
            x_data = None
            profile = None

    if x_data is None or profile is None:
        # Fallback: simple integrated profile (legacy behavior) for robustness.
        profile = np.nansum(image, axis=0)
        if profile.size < 8:
            return result
        x_data = np.linspace(float(x0_abs), float(x1_abs), num=profile.size)

    finite = np.isfinite(profile) & np.isfinite(x_data)
    if finite.sum() < max(8, int(0.3 * profile.size)):
        return result

    y = profile[finite]
    x = x_data[finite]
    if y.size < 8:
        return result

    result.profile_x = x
    result.profile_y = y

    step_mod = lm.models.StepModel(form="erf", prefix="step_")
    line_mod = lm.models.LinearModel(prefix="line_")
    mod = step_mod + line_mod

    try:
        params = line_mod.make_params(intercept=float(np.nanmin(y)), slope=0.0)
        params += step_mod.guess(y, x=x)
        fit = mod.fit(y, params=params, x=x, nan_policy="omit")
    except Exception:
        return result

    sigma_param = fit.params.get("step_sigma", None)
    center_param = fit.params.get("step_center", None)
    if sigma_param is None:
        return result

    sigma_val = abs(float(sigma_param.value)) if sigma_param.value is not None else np.nan
    sigma_err = (
        float(sigma_param.stderr)
        if getattr(sigma_param, "stderr", None) is not None
        else np.nan
    )
    center_val = (
        float(center_param.value)
        if (center_param is not None and center_param.value is not None)
        else np.nan
    )
    if not np.isfinite(sigma_val):
        return result

    result.ok = bool(getattr(fit, "success", True))
    result.step_sigma = sigma_val
    result.step_sigma_stderr = sigma_err
    result.step_center = center_val
    result.profile_fit = np.asarray(fit.best_fit, dtype=np.float64)

    # ESF -> LSF -> MTF pipeline
    if result.profile_x is not None and result.profile_fit is not None:
        x_esf = np.asarray(result.profile_x, dtype=np.float64)
        y_esf = np.asarray(result.profile_fit, dtype=np.float64)
        if x_esf.size >= 8 and np.all(np.isfinite(x_esf)) and np.all(np.isfinite(y_esf)):
            dx = float(np.nanmedian(np.diff(x_esf)))
            if np.isfinite(dx) and dx > 0:
                center_ref = center_val if np.isfinite(center_val) else float(np.nanmedian(x_esf))
                x_centered = x_esf - center_ref
                lsf = np.gradient(y_esf, dx)
                window = np.hanning(lsf.size) if lsf.size >= 8 else np.ones(lsf.size, dtype=np.float64)
                lsf_w = lsf * window
                mtf = np.abs(np.fft.rfft(lsf_w))
                freq = np.fft.rfftfreq(lsf_w.size, d=dx)
                if mtf.size > 0 and np.isfinite(mtf[0]) and mtf[0] > 0:
                    mtf = mtf / mtf[0]
                result.lsf_x = x_centered
                result.lsf_y = lsf
                result.mtf_f = freq
                result.mtf_y = mtf

                # LSF width estimate from Gaussian fit to derivative trace (PFT/LSF).
                try:
                    g_mod = lm.models.GaussianModel(prefix="psf_")
                    c_mod = lm.models.ConstantModel(prefix="base_")
                    g_params = g_mod.guess(lsf, x=x_centered)
                    g_params += c_mod.make_params(c=float(np.nanmedian(lsf)))
                    g_fit = (g_mod + c_mod).fit(lsf, params=g_params, x=x_centered, nan_policy="omit")
                    psf_sigma_param = g_fit.params.get("psf_sigma", None)
                    if psf_sigma_param is not None and psf_sigma_param.value is not None:
                        psf_sigma_val = abs(float(psf_sigma_param.value))
                        if np.isfinite(psf_sigma_val):
                            result.psf_sigma = psf_sigma_val
                            if getattr(psf_sigma_param, "stderr", None) is not None:
                                result.psf_sigma_stderr = abs(float(psf_sigma_param.stderr))
                except Exception:
                    pass

                # MTF50 estimate by linear interpolation at first crossing.
                if mtf.size >= 2:
                    below = np.where(mtf <= 0.5)[0]
                    if below.size > 0:
                        i1 = int(below[0])
                        if i1 == 0:
                            result.mtf50 = float(freq[0])
                        else:
                            i0 = i1 - 1
                            y0 = float(mtf[i0])
                            y1 = float(mtf[i1])
                            x0 = float(freq[i0])
                            x1 = float(freq[i1])
                            if np.isfinite(y0) and np.isfinite(y1) and abs(y1 - y0) > 1e-12:
                                frac = (0.5 - y0) / (y1 - y0)
                                result.mtf50 = x0 + frac * (x1 - x0)
    return result


class _TaskSignals(QtCore.QObject):
    done = QtCore.Signal(str, int, int, object, object)


class _TaskRunner(QtCore.QRunnable):
    def __init__(self, kind: str, token: int, frame_index: int, fn, *args, **kwargs):
        super().__init__()
        self.kind = kind
        self.token = int(token)
        self.frame_index = int(frame_index)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _TaskSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.done.emit(self.kind, self.token, self.frame_index, result, None)
        except Exception as ex:
            self.signals.done.emit(self.kind, self.token, self.frame_index, None, ex)


def _load_and_filter_worker(path: Path) -> np.ndarray:
    return preprocess_image(_read_image(path))


def _load_and_quick_filter_worker(path: Path) -> np.ndarray:
    return preprocess_image_quick(_read_image(path))


def _analyze_filtered_worker(
    filtered: np.ndarray,
    roi_bounds: Tuple[int, int, int, int],
    fixed_edge_line: Optional[Tuple[float, float]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> FitResult:
    if cancel_event is not None and cancel_event.is_set():
        return None
    x0, x1, y0, y1 = roi_bounds
    roi_img = filtered[y0:y1, x0:x1]
    if cancel_event is not None and cancel_event.is_set():
        return None
    return analyze_roi(
        roi_img,
        x0_abs=float(x0),
        x1_abs=float(x1),
        y0_abs=float(y0),
        y1_abs=float(y1),
        fixed_edge_line=fixed_edge_line,
    )


def _bulk_reprocess_worker(
    path: Path,
    roi_rect: Tuple[float, float, float, float],
    fixed_edge_line: Optional[Tuple[float, float]],
) -> Tuple[np.ndarray, FitResult]:
    filtered = preprocess_image(_read_image(path))
    bounds = _bounds_from_roi_rect(filtered.shape, roi_rect)
    fit = _analyze_filtered_worker(filtered, bounds, fixed_edge_line=fixed_edge_line)
    return filtered, fit


def _bulk_reprocess_roi_worker(
    path: Path,
    roi_rect: Tuple[float, float, float, float],
    fixed_edge_line: Optional[Tuple[float, float]],
) -> Tuple[None, FitResult]:
    # Fast full-quality reprocess for non-displayed frames: process only ROI.
    raw = _read_image(path)
    bounds = _bounds_from_roi_rect(raw.shape, roi_rect)
    x0, x1, y0, y1 = bounds
    roi_raw = raw[y0:y1, x0:x1]
    # Clip-only preprocessing here; analyze_roi applies ROI median for edge detect.
    roi_filtered = preprocess_image(roi_raw, median_size=1)
    fit = analyze_roi(
        roi_filtered,
        x0_abs=float(x0),
        x1_abs=float(x1),
        y0_abs=float(y0),
        y1_abs=float(y1),
        fixed_edge_line=fixed_edge_line,
    )
    return None, fit


def _bulk_reprocess_cached_worker(
    filtered: np.ndarray,
    roi_rect: Tuple[float, float, float, float],
    fixed_edge_line: Optional[Tuple[float, float]],
) -> Tuple[None, FitResult]:
    bounds = _bounds_from_roi_rect(filtered.shape, roi_rect)
    fit = _analyze_filtered_worker(filtered, bounds, fixed_edge_line=fixed_edge_line)
    return None, fit


class FocusOfflineWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        frames: List[FrameInfo],
        interval_ms: int = 500,
        max_workers_total: int = 8,
        bulk_workers: Optional[int] = None,
        full_workers: Optional[int] = None,
        full_cache_gb: float = 10.0,
        allow_file_open: bool = True,
        parent=None,
    ):
        super().__init__(parent=parent)
        # Ensure consistent image orientation no matter who instantiates this window.
        pg.setConfigOption("imageAxisOrder", "row-major")
        self._apply_pyqtgraph_theme_from_palette()
        self.setWindowTitle("Offline Focus Scan Viewer")
        self.resize(1500, 900)

        self.frames = frames
        self.current_index = 0
        self.playing = False
        self.interval_ms = int(max(50, interval_ms))
        self._allow_file_open = bool(allow_file_open)
        self._last_open_dir = (
            str(frames[0].path.parent) if frames else str(Path.cwd())
        )

        self._quick_filtered_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max_quick_cache_bytes = int(2 * (1024**3))
        self._quick_cache_bytes = 0
        self._full_filtered_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._max_full_cache_bytes = int(max(0.25, float(full_cache_gb)) * (1024**3))
        self._full_cache_bytes = 0
        self._results: Dict[int, FitResult] = {}
        self._result_is_full: Dict[int, bool] = {}
        self._current_filtered: Optional[np.ndarray] = None

        # Total worker budget across all pools:
        #   task queue (single-thread) + bulk reprocess + full prepare.
        self._max_workers_total = int(max(3, max_workers_total))
        _task_workers = 1
        _remaining_workers = int(max(2, self._max_workers_total - _task_workers))
        req_bulk = int(max(1, bulk_workers)) if bulk_workers is not None else None
        req_full = int(max(1, full_workers)) if full_workers is not None else None
        if req_bulk is None and req_full is None:
            _bulk_workers = 1
            _full_workers = int(max(1, _remaining_workers - _bulk_workers))
        elif req_bulk is not None and req_full is None:
            _bulk_workers = int(min(req_bulk, max(1, _remaining_workers - 1)))
            _full_workers = int(max(1, _remaining_workers - _bulk_workers))
        elif req_bulk is None and req_full is not None:
            _full_workers = int(min(req_full, max(1, _remaining_workers - 1)))
            _bulk_workers = int(max(1, _remaining_workers - _full_workers))
        else:
            assert req_bulk is not None and req_full is not None
            if (req_bulk + req_full) <= _remaining_workers:
                _bulk_workers = int(req_bulk)
                _full_workers = int(req_full)
            else:
                _bulk_workers = int(min(req_bulk, max(1, _remaining_workers - 1)))
                _full_workers = int(max(1, _remaining_workers - _bulk_workers))

        self._thread_pool = QtCore.QThreadPool(self)
        self._thread_pool.setMaxThreadCount(_task_workers)
        self._bulk_thread_pool = QtCore.QThreadPool(self)
        self._bulk_worker_count = _bulk_workers
        self._bulk_thread_pool.setMaxThreadCount(self._bulk_worker_count)
        self._task_queue: deque = deque()
        self._task_queue_max = 32
        self._task_worker_busy = False
        self._queued_load_indices: Set[int] = set()
        self._active_load_index: Optional[int] = None
        self._is_frame_loading = False
        self._stream_next_index: Optional[int] = None
        self._analysis_token = 0
        self._analysis_inflight = False
        self._analysis_cancel_event: Optional[threading.Event] = None
        self._pending_analysis_request: Optional[
            Tuple[np.ndarray, Tuple[int, int, int, int], int, Optional[Tuple[float, float]]]
        ] = None
        self._last_committed_roi_rect: Optional[Tuple[float, float, float, float]] = None
        self._roi_initialized_from_first_frame = False
        self._suppress_roi_callbacks = False
        # Keep ROI dragging responsive by default: recompute on release.
        self._live_roi_preview = False
        self._full_worker_count = _full_workers
        self._full_queue: deque = deque()
        self._full_queue_max = 64
        self._full_running_count = 0
        self._full_queued_indices: Set[int] = set()
        self._full_active_indices: Set[int] = set()
        self._full_refresh_active_indices: Set[int] = set()
        self._full_prepared_indices: Set[int] = set()
        self._full_future_to_index: Dict[object, int] = {}
        self._full_process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=self._full_worker_count
        )
        self._pause_full_prepare = False
        self._bulk_reprocess_active = False
        self._bulk_reprocess_token = 0
        self._bulk_reprocess_total = 0
        self._bulk_reprocess_done = 0
        self._bulk_reprocess_uses_disk = False
        self._current_bulk_used_fixed = False
        self._fixed_edge_line: Optional[Tuple[float, float]] = None
        self._fixed_edge_reference_index: Optional[int] = None
        self._full_edge_refined = False
        self._full_dynamic_results_ready = False
        self._full_dynamic_pass_requested = False
        self._full_prepare_refresh_requested = False
        self._last_full_prepare_logged_count = -1
        self._roi_generation = 0
        self._roi_reprocess_after_scan = False
        self._pending_bulk_reprocess: Optional[
            Tuple[bool, Optional[Tuple[int, ...]], bool, bool, str]
        ] = None
        self._bulk_reprocess_all_frames = False
        self._seen_frame_indices: Set[int] = set()
        self._playback_summary_pending = False
        self._load_generation = 0
        self._edge_near_fraction = 0.10
        self._edge_min_candidates = 3
        self._local_fit_points = 7
        self._optimal_focus_position: float = np.nan
        self._optimal_focus_sigma: float = np.nan
        self._optimal_psf_position: float = np.nan
        self._optimal_psf_sigma: float = np.nan
        self._optimal_mtf50_position: float = np.nan
        self._optimal_mtf50_value: float = np.nan
        self._shutting_down = False

        self._build_ui()
        self._last_committed_roi_rect = self._roi_rect()
        self._log(
            f"Worker limits: total={self._max_workers_total}, "
            f"task=1, bulk={self._bulk_worker_count}, full={self._full_worker_count} (processes)"
        )
        self._log(
            "Math thread caps: "
            f"OMP={os.environ.get('OMP_NUM_THREADS')}, "
            f"MKL={os.environ.get('MKL_NUM_THREADS')}, "
            f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
            f"NUMEXPR={os.environ.get('NUMEXPR_NUM_THREADS')}"
        )
        self._log(
            f"Cache budgets: quick={self._max_quick_cache_bytes / (1024**3):.1f} GB, "
            f"full={self._max_full_cache_bytes / (1024**3):.1f} GB"
        )
        self._log(f"Viewer initialized with {len(self.frames)} frames")
        if self.frames:
            self._load_frame(0)
        else:
            self.frame_label.setText("No files loaded. Use Open Files... to select TIFF images.")
            self.statusBar().showMessage("No files loaded. Click Open Files... to begin.")
            self._log("No startup image set. Use Open Files... to load TIFF frames.")

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        control_row = QtWidgets.QHBoxLayout()
        root.addLayout(control_row)

        self.btn_prev = QtWidgets.QPushButton("Prev")
        self.btn_next = QtWidgets.QPushButton("Next")
        self.btn_open_files = (
            QtWidgets.QPushButton("Open Files...") if self._allow_file_open else None
        )
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_play.hide()
        self.btn_recompute = QtWidgets.QPushButton("Recompute ROI")
        self.local_fit_points_spin = QtWidgets.QSpinBox()
        self.local_fit_points_spin.setRange(3, 31)
        self.local_fit_points_spin.setSingleStep(2)
        self.local_fit_points_spin.setValue(self._local_fit_points)

        control_row.addWidget(self.btn_prev)
        control_row.addWidget(self.btn_next)
        if self.btn_open_files is not None:
            control_row.addWidget(self.btn_open_files)
        control_row.addWidget(self.btn_recompute)
        control_row.addWidget(QtWidgets.QLabel("Local fit pts:"))
        control_row.addWidget(self.local_fit_points_spin)
        self.optimal_focus_label = QtWidgets.QLabel("Optimal motor position: --")
        control_row.addWidget(self.optimal_focus_label)
        control_row.addStretch(1)

        body = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(body, 1)

        # Left: image panel
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        self.image_view = pg.ImageView(view=pg.PlotItem())
        # Hide histogram/intensity controls to keep the image panel clean.
        if hasattr(self.image_view, "ui"):
            if hasattr(self.image_view.ui, "histogram"):
                self.image_view.ui.histogram.hide()
            if hasattr(self.image_view.ui, "roiBtn"):
                self.image_view.ui.roiBtn.hide()
            if hasattr(self.image_view.ui, "menuBtn"):
                self.image_view.ui.menuBtn.hide()
        left_layout.addWidget(self.image_view, 1)
        self.frame_label = QtWidgets.QLabel("")
        left_layout.addWidget(self.frame_label)
        body.addWidget(left)

        # Right: analysis panel (top row: ESF/PFT/MTF, bottom row: params vs motor)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        top_row = QtWidgets.QHBoxLayout()
        bottom_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(top_row, 1)
        right_layout.addLayout(bottom_row, 1)

        self.esf_plot = pg.PlotWidget(title="ESF")
        self.esf_plot.addLegend()
        self.esf_raw_curve = self.esf_plot.plot(
            pen=pg.mkPen(50, 120, 220, width=2), name="raw ESF"
        )
        self.esf_fit_curve = self.esf_plot.plot(
            pen=pg.mkPen(220, 80, 60, width=2), name="fit ESF"
        )
        self.esf_plot.setLabel("left", "Intensity")
        self.esf_plot.setLabel("bottom", "Perpendicular distance (px)")

        self.pft_plot = pg.PlotWidget(title="LSF")
        self.pft_curve = self.pft_plot.plot(pen=pg.mkPen(30, 160, 120, width=2))
        self.pft_plot.setLabel("left", "dI/dx")
        self.pft_plot.setLabel("bottom", "Perpendicular distance (px)")

        self.mtf_plot = pg.PlotWidget(title="MTF")
        self.mtf_curve = self.mtf_plot.plot(pen=pg.mkPen(100, 90, 220, width=2))
        self.mtf50_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(200, 80, 80, width=1))
        self.mtf_plot.addItem(self.mtf50_line)
        self.mtf50_line.hide()
        self.mtf_half_line = pg.InfiniteLine(pos=0.5, angle=0, movable=False, pen=pg.mkPen(140, 140, 140, width=1))
        self.mtf_plot.addItem(self.mtf_half_line)
        self.mtf_plot.setYRange(0.0, 1.05)
        self.mtf_plot.setLabel("left", "MTF")
        self.mtf_plot.setLabel("bottom", "Spatial freq (px^-1)")

        top_row.addWidget(self.esf_plot, 1)
        top_row.addWidget(self.pft_plot, 1)
        top_row.addWidget(self.mtf_plot, 1)
        self.pft_plot.setXRange(-30.0, 30.0, padding=0.0)
        self.mtf_plot.setXRange(0.0, 0.4, padding=0.0)
        self.pft_plot.getPlotItem().vb.enableAutoRange(x=False)
        self.mtf_plot.getPlotItem().vb.enableAutoRange(x=False)

        self.sigma_metric_plot = pg.PlotWidget(title="ESF Sigma vs Motor")
        self.curve_sigma_quick = self.sigma_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(170, 170, 170),
            symbolPen=pg.mkPen(120, 120, 120, width=1),
            name="quick",
        )
        self.curve_sigma_full = self.sigma_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush(40, 160, 90),
            symbolPen=pg.mkPen(30, 120, 70, width=1),
            name="full",
        )
        self.curve_sigma_highlight = self.sigma_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=13,
            symbolBrush=pg.mkBrush(255, 220, 0),
            symbolPen=pg.mkPen(0, 0, 0, width=2),
        )
        self.curve_quad_fit = self.sigma_metric_plot.plot(
            pen=pg.mkPen(180, 90, 200, width=2, style=QtCore.Qt.DashLine), name="local quadratic fit"
        )
        self.sigma_error_bars = pg.ErrorBarItem()
        self.sigma_metric_plot.addItem(self.sigma_error_bars)
        self.sigma_metric_plot.setLabel("bottom", "Motor position")
        self.sigma_metric_plot.setLabel("left", "step_sigma")

        self.psf_metric_plot = pg.PlotWidget(title="LSF Width vs Motor")
        self.curve_psf_quick = self.psf_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(175, 175, 175),
            symbolPen=pg.mkPen(120, 120, 120, width=1),
            name="quick",
        )
        self.curve_psf_full = self.psf_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush(210, 120, 30),
            symbolPen=pg.mkPen(160, 90, 20, width=1),
            name="full",
        )
        self.curve_psf_highlight = self.psf_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=13,
            symbolBrush=pg.mkBrush(255, 220, 0),
            symbolPen=pg.mkPen(0, 0, 0, width=2),
        )
        self.curve_psf_fit = self.psf_metric_plot.plot(
            pen=pg.mkPen(180, 90, 200, width=2, style=QtCore.Qt.DashLine)
        )
        self.psf_metric_plot.setLabel("bottom", "Motor position")
        self.psf_metric_plot.setLabel("left", "lsf_sigma")

        self.mtf_metric_plot = pg.PlotWidget(title="MTF50 vs Motor")
        self.curve_mtf50_quick = self.mtf_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=5,
            symbolBrush=pg.mkBrush(170, 170, 170),
            symbolPen=pg.mkPen(120, 120, 120, width=1),
            name="quick",
        )
        self.curve_mtf50_full = self.mtf_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush(100, 90, 220),
            symbolPen=pg.mkPen(75, 70, 170, width=1),
            name="full",
        )
        self.curve_mtf50_highlight = self.mtf_metric_plot.plot(
            pen=None,
            symbol="o",
            symbolSize=13,
            symbolBrush=pg.mkBrush(255, 220, 0),
            symbolPen=pg.mkPen(0, 0, 0, width=2),
        )
        self.curve_mtf50_fit = self.mtf_metric_plot.plot(
            pen=pg.mkPen(180, 90, 200, width=2, style=QtCore.Qt.DashLine)
        )
        self.mtf_metric_plot.setLabel("bottom", "Motor position")
        self.mtf_metric_plot.setLabel("left", "MTF50 (px^-1)")

        bottom_row.addWidget(self.sigma_metric_plot, 1)
        bottom_row.addWidget(self.psf_metric_plot, 1)
        bottom_row.addWidget(self.mtf_metric_plot, 1)
        body.addWidget(right)

        body.setSizes([900, 1400])

        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(2000)
        self.console.setFixedHeight(170)
        root.addWidget(self.console, 0)

        # ROI setup
        self.image_item = self.image_view.getImageItem()
        # Force a stable, non-inverted grayscale mapping independent of app theme.
        try:
            self.image_view.setPredefinedGradient("grey")
        except Exception:
            try:
                hist = getattr(self.image_view.ui, "histogram", None)
                if hist is not None and hasattr(hist, "gradient"):
                    hist.gradient.loadPreset("grey")
            except Exception:
                pass
        try:
            lut = np.arange(256, dtype=np.uint8)
            lut = np.column_stack((lut, lut, lut, np.full_like(lut, 255)))
            self.image_item.setLookupTable(lut)
        except Exception:
            pass
        self.roi = pg.RectROI([100, 100], [200, 400], pen=pg.mkPen(255, 200, 0, width=2))
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.image_view.getView().addItem(self.roi)
        self.edge_line_item = pg.PlotCurveItem(
            pen=pg.mkPen(220, 40, 40, width=2)
        )
        self.edge_line_item.hide()
        self.image_view.getView().addItem(self.edge_line_item)

        self.filter_queue_bar = QtWidgets.QProgressBar(self)
        self.filter_queue_bar.setRange(0, max(1, len(self.frames)))
        self.filter_queue_bar.setValue(0)
        self.filter_queue_bar.setFixedWidth(230)
        self.filter_queue_bar.setFixedHeight(18)
        self.filter_queue_bar.setTextVisible(True)
        self.filter_queue_bar.setAlignment(QtCore.Qt.AlignCenter)
        self._apply_filter_queue_theme()
        self.filter_queue_bar.setFormat("Full filter queue: 0")
        self.filter_queue_label = QtWidgets.QLabel("full 0/0")
        self.statusBar().addPermanentWidget(self.filter_queue_bar)
        self.statusBar().addPermanentWidget(self.filter_queue_label)
        self.statusBar().showMessage("Set ROI around foil edge, then step frames.")

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.interval_ms)
        self._roi_analysis_timer = QtCore.QTimer(self)
        self._roi_analysis_timer.setSingleShot(True)
        self._roi_analysis_timer.setInterval(180)
        self._full_future_timer = QtCore.QTimer(self)
        self._full_future_timer.setInterval(40)

        self.btn_prev.clicked.connect(self._prev_frame)
        self.btn_next.clicked.connect(self._next_frame)
        if self.btn_open_files is not None:
            self.btn_open_files.clicked.connect(self._open_files_dialog)
        self.btn_play.toggled.connect(self._toggle_play)
        self.btn_recompute.clicked.connect(self._recompute_current)
        self.local_fit_points_spin.valueChanged.connect(self._on_local_fit_points_changed)
        self.timer.timeout.connect(self._tick)
        self._roi_analysis_timer.timeout.connect(self._recompute_current)
        self._full_future_timer.timeout.connect(self._drain_full_prepare_futures)
        self.roi.sigRegionChanged.connect(self._on_roi_region_changed)
        self.roi.sigRegionChangeFinished.connect(self._on_roi_region_change_finished)
        self._full_future_timer.start()
        self._update_filter_queue_indicator()

    def _apply_pyqtgraph_theme_from_palette(self):
        pal = self.palette()
        bg = pal.color(QtGui.QPalette.Window)
        fg = pal.color(QtGui.QPalette.WindowText)
        pg.setConfigOption("background", (bg.red(), bg.green(), bg.blue()))
        pg.setConfigOption("foreground", (fg.red(), fg.green(), fg.blue()))

    def _apply_filter_queue_theme(self):
        pal = self.palette()
        border = pal.color(QtGui.QPalette.Mid)
        bg = pal.color(QtGui.QPalette.Base)
        text = pal.color(QtGui.QPalette.Text)
        chunk = pal.color(QtGui.QPalette.Highlight)
        self.filter_queue_bar.setStyleSheet(
            "QProgressBar {"
            f" border: 1px solid {border.name()};"
            " border-radius: 4px;"
            f" background-color: {bg.name()};"
            f" color: {text.name()};"
            " text-align: center;"
            "}"
            "QProgressBar::chunk {"
            f" background-color: {chunk.name()};"
            " border-radius: 3px;"
            " margin: 1px;"
            "}"
        )

    def _log(self, message: str):
        ts = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
        line = f"[{ts}] {message}"
        if hasattr(self, "console"):
            self.console.appendPlainText(line)
            sb = self.console.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _log_final_summary(self, reason: str):
        if not self._results:
            return
        valid_rows = []
        for idx in sorted(self._results):
            r = self._results[idx]
            if np.isfinite(r.step_sigma):
                valid_rows.append((idx, r))
        if not valid_rows:
            return

        best_idx, best_r = min(valid_rows, key=lambda item: float(item[1].step_sigma))
        best_pos = self.frames[best_idx].position
        self._log(f"{reason}")
        self._log(f"Final params: analyzed_frames={len(self._results)}/{len(self.frames)}")
        if np.isfinite(self._optimal_focus_position) and np.isfinite(self._optimal_focus_sigma):
            self._log(
                "Optimal step_sigma (local quadratic): "
                f"motor={self._optimal_focus_position:.5f}, sigma={self._optimal_focus_sigma:.5f}"
            )
        else:
            self._log("Optimal step_sigma (local quadratic): unavailable")
        if np.isfinite(self._optimal_psf_position) and np.isfinite(self._optimal_psf_sigma):
            self._log(
                "Optimal lsf_sigma (local quadratic): "
                f"motor={self._optimal_psf_position:.5f}, lsf_sigma={self._optimal_psf_sigma:.5f}"
            )
        else:
            self._log("Optimal lsf_sigma (local quadratic): unavailable")
        if np.isfinite(self._optimal_mtf50_position) and np.isfinite(self._optimal_mtf50_value):
            self._log(
                "Optimal mtf50 (local quadratic): "
                f"motor={self._optimal_mtf50_position:.5f}, mtf50={self._optimal_mtf50_value:.5f}"
            )
        else:
            self._log("Optimal mtf50 (local quadratic): unavailable")
        self._log(
            "Best observed frame: "
            f"frame={best_idx + 1}, motor={best_pos:.5f}, "
            f"sigma={best_r.step_sigma:.5f}, "
            f"lsf_sigma={best_r.psf_sigma:.5f}, "
            f"mtf50={best_r.mtf50:.5f}"
        )
        if self._fixed_edge_line is not None:
            self._log(
                "Fixed edge line: "
                f"m={self._fixed_edge_line[0]:.6f}, b={self._fixed_edge_line[1]:.6f}"
            )

    def _full_prepared_count(self) -> int:
        n_frames = int(len(self.frames))
        if n_frames <= 0:
            return 0
        return int(sum(1 for idx in self._full_prepared_indices if 0 <= int(idx) < n_frames))

    def _full_prepared_seen_count(self) -> int:
        if not self._seen_frame_indices:
            return 0
        return int(sum(1 for idx in self._seen_frame_indices if int(idx) in self._full_prepared_indices))

    def _all_frames_full_prepared(self) -> bool:
        if not self.frames:
            return True
        return bool(self._full_prepared_count() >= len(self.frames))

    def _needs_disk_fallback_for_all_frames(self) -> bool:
        # Full filtered images are cached with bounded capacity. Once frame count
        # exceeds cache size, all-frame reruns must use disk+cache.
        return bool(len(self._full_filtered_cache) < len(self.frames))

    def _bulk_reprocess_queue_count(self) -> int:
        if not self._bulk_reprocess_active:
            return 0
        if not self._bulk_reprocess_uses_disk:
            return 0
        remaining = int(self._bulk_reprocess_total - self._bulk_reprocess_done)
        return max(0, remaining)

    def _filter_queue_count(self) -> int:
        full_pending = int(len(self._full_queue) + self._full_running_count)
        # Only track heavy full-image filtering backlog here.
        full_backlog = max(0, int(len(self._seen_frame_indices) - self._full_prepared_seen_count()))
        bulk_pending = self._bulk_reprocess_queue_count()
        return max(0, full_pending, full_backlog, bulk_pending)

    def _update_filter_queue_indicator(self):
        if not hasattr(self, "filter_queue_bar"):
            return
        total = max(1, int(len(self.frames)))
        full_ready = int(min(total, self._full_prepared_count()))
        queue_count = self._filter_queue_count()
        bulk_pending = self._bulk_reprocess_queue_count()
        self.filter_queue_bar.setRange(0, total)
        self.filter_queue_bar.setValue(int(min(total, max(0, queue_count))))
        if bulk_pending > 0:
            self.filter_queue_bar.setFormat(f"Reprocess queue: {queue_count}")
        else:
            self.filter_queue_bar.setFormat(f"Full filter queue: {queue_count}")
        self.filter_queue_label.setText(f"full {full_ready}/{len(self.frames)}")

    @staticmethod
    def _array_nbytes(arr: np.ndarray) -> int:
        try:
            return int(arr.nbytes)
        except Exception:
            return int(np.asarray(arr).nbytes)

    def _cache_filtered(self, idx: int, filtered: np.ndarray):
        # Backward-compatible alias: this now means quick-pass cache.
        idx = int(idx)
        new_bytes = self._array_nbytes(filtered)
        if idx in self._quick_filtered_cache:
            old = self._quick_filtered_cache.pop(idx)
            self._quick_cache_bytes = max(0, int(self._quick_cache_bytes - self._array_nbytes(old)))
        self._quick_filtered_cache[idx] = filtered
        self._quick_cache_bytes += int(new_bytes)
        while (
            self._quick_cache_bytes > self._max_quick_cache_bytes
            and len(self._quick_filtered_cache) > 1
        ):
            _old_idx, old_arr = self._quick_filtered_cache.popitem(last=False)
            self._quick_cache_bytes = max(
                0, int(self._quick_cache_bytes - self._array_nbytes(old_arr))
            )

    def _get_filtered_image(self, idx: int) -> Optional[np.ndarray]:
        # Prefer fully processed frames when available.
        full = self._get_full_filtered_image(idx)
        if full is not None:
            return full
        if idx in self._quick_filtered_cache:
            arr = self._quick_filtered_cache.pop(idx)
            self._quick_filtered_cache[idx] = arr
            return arr
        return None

    def _cache_full_filtered(self, idx: int, filtered: np.ndarray):
        idx = int(idx)
        self._full_prepared_indices.add(idx)
        new_bytes = self._array_nbytes(filtered)
        if idx in self._full_filtered_cache:
            old = self._full_filtered_cache.pop(idx)
            self._full_cache_bytes = max(0, int(self._full_cache_bytes - self._array_nbytes(old)))
        self._full_filtered_cache[idx] = filtered
        self._full_cache_bytes += int(new_bytes)
        while (
            self._full_cache_bytes > self._max_full_cache_bytes
            and len(self._full_filtered_cache) > 1
        ):
            _old_idx, old_arr = self._full_filtered_cache.popitem(last=False)
            self._full_cache_bytes = max(
                0, int(self._full_cache_bytes - self._array_nbytes(old_arr))
            )
        # Keep display/navigation cache aligned with full-quality data.
        self._cache_filtered(idx, filtered)

    def _get_full_filtered_image(self, idx: int) -> Optional[np.ndarray]:
        if idx in self._full_filtered_cache:
            arr = self._full_filtered_cache.pop(idx)
            self._full_filtered_cache[idx] = arr
            return arr
        return None

    def _current_roi_bounds(self, shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        pos = self.roi.pos()
        size = self.roi.size()
        return _bounds_from_roi_rect(
            shape,
            (
                float(pos.x()),
                float(pos.y()),
                float(size.x()),
                float(size.y()),
            ),
        )

    def _set_display_image(self, image: np.ndarray):
        prev_suppress = self._suppress_roi_callbacks
        self._suppress_roi_callbacks = True
        roi_prev_block = self.roi.blockSignals(True)
        try:
            self.image_view.setImage(image, autoRange=False, autoLevels=True)
        finally:
            self.roi.blockSignals(roi_prev_block)
            self._suppress_roi_callbacks = prev_suppress

    def _roi_rect(self) -> Tuple[float, float, float, float]:
        pos = self.roi.pos()
        size = self.roi.size()
        return (float(pos.x()), float(pos.y()), float(size.x()), float(size.y()))

    def _initialize_roi_from_shape(self, shape: Tuple[int, int]):
        if self._roi_initialized_from_first_frame:
            return
        h, w = int(shape[0]), int(shape[1])
        if h <= 2 or w <= 2:
            return
        max_w = max(8, w - 2)
        max_h = max(8, h - 2)
        # Default ROI aspect ratio: height:width = 2:1.
        roi_w = int(max(8, round(w * 0.12)))
        roi_h = int(max(8, round(roi_w * 2.0)))
        if roi_h > max_h:
            roi_h = int(max_h)
            roi_w = int(max(8, round(roi_h / 2.0)))
        if roi_w > max_w:
            roi_w = int(max_w)
            roi_h = int(max(8, round(roi_w * 2.0)))
        roi_w = int(min(max_w, max(8, roi_w)))
        roi_h = int(min(max_h, max(8, roi_h)))
        x0 = float((w - roi_w) / 2.0)
        y0 = float((h - roi_h) / 2.0)

        prev_suppress = self._suppress_roi_callbacks
        self._suppress_roi_callbacks = True
        roi_prev_block = self.roi.blockSignals(True)
        try:
            self.roi.setPos([x0, y0], finish=False)
            self.roi.setSize([float(roi_w), float(roi_h)], finish=False)
        finally:
            self.roi.blockSignals(roi_prev_block)
            self._suppress_roi_callbacks = prev_suppress
        self._last_committed_roi_rect = self._roi_rect()
        self._roi_initialized_from_first_frame = True

    def _is_scan_finished(self) -> bool:
        return (
            self.current_index >= (len(self.frames) - 1)
            and (not self.playing)
            and (not self._is_frame_loading)
            and (not self._task_worker_busy)
            and (len(self._full_refresh_active_indices) == 0)
            and (len(self._task_queue) == 0)
        )

    def _is_processing_idle(self) -> bool:
        return (
            (not self.playing)
            and (not self._is_frame_loading)
            and (not self._task_worker_busy)
            and (not self._bulk_reprocess_active)
            and (len(self._full_refresh_active_indices) == 0)
            and (len(self._task_queue) == 0)
        )

    def _is_dataset_complete(self) -> bool:
        if not self.frames:
            return True
        last_idx = len(self.frames) - 1
        return bool(
            (last_idx in self._seen_frame_indices)
            or (len(self._seen_frame_indices) >= len(self.frames))
            or self._all_frames_full_prepared()
            or (len(self._results) >= len(self.frames))
        )

    def _clear_queued_bulk_tasks(self):
        if not self._task_queue:
            return
        kept = deque()
        for task_spec in self._task_queue:
            if task_spec[0] != "bulk_reprocess":
                kept.append(task_spec)
        self._task_queue = kept

    def _cached_frame_indices_upto_current(self) -> List[int]:
        max_idx = int(max(0, self.current_index))
        indices: Set[int] = set()
        for idx in self._seen_frame_indices:
            if idx <= max_idx:
                indices.add(int(idx))
        for idx in self._quick_filtered_cache.keys():
            if idx <= max_idx:
                indices.add(int(idx))
        for idx in self._full_filtered_cache.keys():
            if idx <= max_idx:
                indices.add(int(idx))
        if self._current_filtered is not None:
            indices.add(int(self.current_index))
        return sorted(indices)

    def _get_cached_for_bulk(self, idx: int, use_quick_cache: bool) -> Optional[np.ndarray]:
        if use_quick_cache and idx == self.current_index and self._current_filtered is not None:
            return self._current_filtered
        full = self._get_full_filtered_image(idx)
        if full is not None:
            return full
        if use_quick_cache and idx in self._quick_filtered_cache:
            arr = self._quick_filtered_cache.pop(idx)
            self._quick_filtered_cache[idx] = arr
            return arr
        return None

    def _start_bulk_reprocess(
        self,
        preserve_existing_results: bool = True,
        frame_indices: Optional[Sequence[int]] = None,
        allow_disk_fallback: bool = True,
        clear_queues: bool = True,
        reason: str = "bulk",
        defer_if_busy: bool = True,
    ) -> bool:
        if self._shutting_down:
            return False
        if self._bulk_reprocess_active or self._is_frame_loading or self._task_worker_busy:
            if defer_if_busy:
                idx_tuple = tuple(int(i) for i in frame_indices) if frame_indices is not None else None
                pending_spec = (
                    bool(preserve_existing_results),
                    idx_tuple,
                    bool(allow_disk_fallback),
                    bool(clear_queues),
                    str(reason),
                )
                if self._pending_bulk_reprocess != pending_spec:
                    self._pending_bulk_reprocess = pending_spec
                    self._log(f"Deferred reprocess request ({reason}) until worker is idle.")
                self._update_filter_queue_indicator()
            return False
        if clear_queues:
            self._clear_task_queue()
        else:
            self._clear_queued_bulk_tasks()
        self._bulk_reprocess_active = True
        self._bulk_reprocess_token += 1
        token = self._bulk_reprocess_token
        if frame_indices is None:
            requested_indices = list(range(len(self.frames)))
        else:
            requested_indices = sorted(
                {
                    int(i)
                    for i in frame_indices
                    if 0 <= int(i) < len(self.frames)
                }
            )
        # Keep currently displayed frame responsive: process it first in batch passes.
        cur_idx = int(self.current_index)
        if cur_idx in requested_indices:
            requested_indices = [cur_idx] + [i for i in requested_indices if i != cur_idx]
        self._bulk_reprocess_all_frames = len(requested_indices) == len(self.frames)
        # Keep full filtering independent from reprocessing. Reprocess should
        # consume filtered cache, not pause or flush the heavy filter queue.
        self._pause_full_prepare = False
        self._bulk_reprocess_total = 0
        self._bulk_reprocess_done = 0
        self._bulk_reprocess_uses_disk = bool(allow_disk_fallback)
        self._current_bulk_used_fixed = self._fixed_edge_line is not None
        if not preserve_existing_results:
            self._results.clear()
            self._result_is_full.clear()

        roi_rect = self._roi_rect()
        mode = "full" if self._bulk_reprocess_all_frames else "partial"
        source_mode = "disk+cache" if allow_disk_fallback else "cache-only"
        line_mode = "fixed-edge" if self._fixed_edge_line is not None else "dynamic-edge"
        # Prefer cached filtered frames (full first, then quick) to avoid
        # unnecessary disk rereads during ROI reprocess.
        use_quick_cache = True
        self.statusBar().showMessage(
            f"Reprocessing ({mode}) {len(requested_indices)} frames ({line_mode}, {source_mode})..."
        )
        self._log(
            f"Reprocessing started [{reason}]: requested={len(requested_indices)}, mode={mode}, "
            f"line={line_mode}, source={source_mode}, "
            f"full_cache_ready={self._full_prepared_count()}/{len(self.frames)}, "
            f"workers={self._bulk_worker_count}"
        )
        cached_jobs = 0
        disk_jobs = 0
        for idx in requested_indices:
            frame = self.frames[idx]
            cached = self._get_cached_for_bulk(idx, use_quick_cache=use_quick_cache)
            if cached is not None:
                ok = self._start_bulk_task(
                    token,
                    idx,
                    _bulk_reprocess_cached_worker,
                    cached,
                    roi_rect,
                    self._fixed_edge_line,
                )
                if ok:
                    cached_jobs += 1
            else:
                if not allow_disk_fallback:
                    continue
                # Always run full-frame worker for disk fallbacks so all-frame
                # reprocess fills the full-quality cache consistently.
                worker_fn = _bulk_reprocess_worker
                ok = self._start_bulk_task(
                    token,
                    idx,
                    worker_fn,
                    frame.path,
                    roi_rect,
                    self._fixed_edge_line,
                )
                if ok:
                    disk_jobs += 1
            if ok:
                self._bulk_reprocess_total += 1

        if self._bulk_reprocess_total <= 0:
            self._bulk_reprocess_active = False
            self._bulk_reprocess_uses_disk = False
            self._bulk_reprocess_all_frames = False
            self.statusBar().showMessage("No frames available for ROI reprocess yet.")
            self._log(f"Reprocess skipped [{reason}]: no frames could be scheduled.")
            self._update_filter_queue_indicator()
            return False
        self._log(
            f"Reprocessing scheduling: cached={cached_jobs}, disk={disk_jobs}, "
            f"total={self._bulk_reprocess_total}"
        )
        self._update_filter_queue_indicator()
        return True

    def _start_task(self, kind: str, token: int, frame_index: int, fn, *args, priority: int = 0):
        if self._shutting_down:
            return False
        # Single-worker FIFO queue to mirror sequential event stream processing.
        if len(self._task_queue) >= self._task_queue_max:
            self._log(
                f"Backpressure: queue full ({len(self._task_queue)}/{self._task_queue_max}); "
                f"could not enqueue {kind} for frame {frame_index + 1}"
            )
            self._update_filter_queue_indicator()
            return False
        task_spec = (kind, token, frame_index, fn, args)
        if priority >= 2:
            self._task_queue.appendleft(task_spec)
        else:
            self._task_queue.append(task_spec)
        self._pump_task_queue()
        self._update_filter_queue_indicator()
        return True

    def _start_bulk_task(self, token: int, frame_index: int, fn, *args):
        if self._shutting_down:
            return False
        task = _TaskRunner("bulk_reprocess", token, frame_index, fn, *args)
        task.signals.done.connect(self._on_task_done)
        self._bulk_thread_pool.start(task, priority=0)
        self._update_filter_queue_indicator()
        return True

    def _start_full_frame_refresh(self, frame_index: int) -> bool:
        if self._shutting_down or self._bulk_reprocess_active:
            return False
        idx = int(frame_index)
        if idx in self._full_refresh_active_indices:
            return False
        filtered = self._get_full_filtered_image(idx)
        if filtered is None:
            return False
        self._full_refresh_active_indices.add(idx)
        task = _TaskRunner(
            "refresh_full_frame",
            int(self._roi_generation),
            idx,
            _bulk_reprocess_cached_worker,
            filtered,
            self._roi_rect(),
            self._fixed_edge_line,
        )
        task.signals.done.connect(self._on_task_done)
        self._bulk_thread_pool.start(task, priority=0)
        return True

    @staticmethod
    def _result_has_valid_edge(result: Optional[FitResult]) -> bool:
        if result is None:
            return False
        return bool(
            np.isfinite(result.edge_m)
            and np.isfinite(result.edge_b)
            and np.isfinite(result.edge_ymin)
            and np.isfinite(result.edge_ymax)
        )

    def _preserve_previous_edge_geometry(self, frame_index: int, result: Optional[FitResult]) -> Optional[FitResult]:
        if result is None:
            return result
        if self._result_has_valid_edge(result):
            return result
        prev = self._results.get(int(frame_index), None)
        if not self._result_has_valid_edge(prev):
            return result
        # Quick-pass can occasionally miss edge geometry on noisy frames; keep last
        # good per-frame line so overlay does not disappear while metrics update.
        result.edge_m = float(prev.edge_m)
        result.edge_b = float(prev.edge_b)
        result.edge_ymin = float(prev.edge_ymin)
        result.edge_ymax = float(prev.edge_ymax)
        if np.isfinite(prev.edge_m_dynamic):
            result.edge_m_dynamic = float(prev.edge_m_dynamic)
        if np.isfinite(prev.edge_b_dynamic):
            result.edge_b_dynamic = float(prev.edge_b_dynamic)
        return result

    def _schedule_prefetch(self, center_idx: int):
        # Disabled in sequential stream mode.
        _ = center_idx

    def _clear_task_queue(self):
        self._task_queue.clear()
        self._queued_load_indices.clear()
        self._pending_analysis_request = None
        self._update_filter_queue_indicator()

    def _clear_full_queue(self):
        self._full_queue.clear()
        self._full_queued_indices.clear()
        self._full_active_indices.clear()
        self._full_running_count = 0
        # Best effort cancel of not-yet-started futures.
        for fut in list(self._full_future_to_index.keys()):
            try:
                fut.cancel()
            except Exception:
                pass
        self._full_future_to_index.clear()
        self._update_filter_queue_indicator()

    def _enqueue_full_prepare(self, idx: int):
        if self._shutting_down:
            return
        idx = int(max(0, min(len(self.frames) - 1, idx)))
        if idx in self._full_queued_indices or idx in self._full_active_indices:
            return
        if idx in self._full_filtered_cache:
            return
        if (len(self._full_queue) + self._full_running_count) >= self._full_queue_max:
            self._log(
                f"Backpressure(full): queue full ({len(self._full_queue) + self._full_running_count}/{self._full_queue_max}); "
                f"could not enqueue prepare_full for frame {idx + 1}"
            )
            return
        self._full_queue.append(idx)
        self._full_queued_indices.add(idx)
        self._pump_full_queue()
        self._update_filter_queue_indicator()

    def _pump_full_queue(self):
        if self._shutting_down:
            return
        if self._pause_full_prepare:
            return
        while self._full_running_count < self._full_worker_count and self._full_queue:
            idx = int(self._full_queue.popleft())
            self._full_queued_indices.discard(idx)
            self._full_running_count += 1
            self._full_active_indices.add(idx)
            try:
                fut = self._full_process_pool.submit(_load_and_filter_worker, self.frames[idx].path)
            except Exception as ex:
                self._full_running_count = max(0, int(self._full_running_count) - 1)
                self._full_active_indices.discard(idx)
                self._log(f"Full prepare submit failed for frame {idx + 1}: {ex}")
                break
            self._full_future_to_index[fut] = idx
        self._update_filter_queue_indicator()

    def _handle_full_prepare_result(self, frame_index: int, result, error):
        self._full_running_count = max(0, int(self._full_running_count) - 1)
        self._full_queued_indices.discard(frame_index)
        self._full_active_indices.discard(frame_index)
        if error is not None:
            self._log(f"Full prepare failed for frame {frame_index + 1}: {error}")
        elif result is not None:
            self._cache_full_filtered(frame_index, result)
            # If user is currently viewing this frame, refresh image/analysis with full-quality data.
            if frame_index == self.current_index and not self._bulk_reprocess_active:
                self._current_filtered = result
                self._set_display_image(result)
                self._request_analysis_current()
            elif not self._bulk_reprocess_active:
                # Promote this frame's plotted point from quick(gray) to full(color)
                # as soon as full filtering completes.
                self._start_full_frame_refresh(frame_index)
            if (
                (self._fixed_edge_line is None)
                and (not self._bulk_reprocess_active)
                and self._all_frames_full_prepared()
            ):
                self._try_lock_edge_from_focus_minimum()
            full_ready = self._full_prepared_count()
            if (
                (full_ready != int(self._last_full_prepare_logged_count))
                and (
                    (full_ready <= 3)
                    or (full_ready == len(self.frames))
                    or (full_ready % max(1, len(self.frames) // 10) == 0)
                )
            ):
                self._last_full_prepare_logged_count = int(full_ready)
                self._log(
                    f"Full prepare progress: {full_ready}/{len(self.frames)} cached"
                )
        if (
            self._all_frames_full_prepared()
            and (not self._bulk_reprocess_active)
            and (not self._full_prepare_refresh_requested)
        ):
            self._full_prepare_refresh_requested = True
            needs_full_refresh = any(
                (i not in self._results) or (not bool(self._result_is_full.get(i, False)))
                for i in range(len(self.frames))
            )
            if needs_full_refresh:
                self._log(
                    "Full filtering queue complete. Recomputing ROI metrics on fully filtered frames..."
                )
                self._start_bulk_reprocess(
                    preserve_existing_results=True,
                    frame_indices=None,
                    allow_disk_fallback=self._needs_disk_fallback_for_all_frames(),
                    clear_queues=False,
                    reason="full-prepare-complete-refresh",
                )
            else:
                self._log(
                    "Full filtering queue complete. Per-frame ROI metrics are already current."
                )
        self._pump_full_queue()
        self._maybe_start_full_reprocess_after_scan()
        self._update_filter_queue_indicator()

    def _drain_full_prepare_futures(self):
        if self._shutting_down:
            return
        if not self._full_future_to_index:
            self._pump_full_queue()
            return
        done_futures = [f for f in list(self._full_future_to_index.keys()) if f.done()]
        if not done_futures:
            return
        for fut in done_futures:
            idx = int(self._full_future_to_index.pop(fut))
            err = None
            result = None
            try:
                result = fut.result()
            except Exception as ex:
                err = ex
            self._handle_full_prepare_result(idx, result, err)
        self._update_filter_queue_indicator()

    def _pump_task_queue(self):
        if self._shutting_down:
            self._task_queue.clear()
            self._queued_load_indices.clear()
            self._task_worker_busy = False
            self._is_frame_loading = False
            self._active_load_index = None
            self._update_filter_queue_indicator()
            return
        if self._task_worker_busy:
            return
        if not self._task_queue:
            self._is_frame_loading = False
            if self._pending_bulk_reprocess is not None and not self._bulk_reprocess_active:
                pending = self._pending_bulk_reprocess
                self._pending_bulk_reprocess = None
                preserve, indices, allow_disk, clear_queues, reason = pending
                started = self._start_bulk_reprocess(
                    preserve_existing_results=preserve,
                    frame_indices=indices,
                    allow_disk_fallback=allow_disk,
                    clear_queues=clear_queues,
                    reason=reason,
                    defer_if_busy=False,
                )
                if started:
                    return
                self._pending_bulk_reprocess = pending
            self._maybe_start_full_reprocess_after_scan()
            self._maybe_finalize_playback_summary()
            self._update_filter_queue_indicator()
            return
        kind, token, frame_index, fn, args = self._task_queue.popleft()
        self._task_worker_busy = True
        self._is_frame_loading = bool(kind == "load_filter_display")
        if kind == "load_filter_display":
            self._active_load_index = int(frame_index)
        task = _TaskRunner(kind, token, frame_index, fn, *args)
        task.signals.done.connect(self._on_task_done)
        self._thread_pool.start(task, priority=0)
        self._update_filter_queue_indicator()

    def _request_analysis_current(self):
        filt = self._current_filtered
        if filt is None:
            return
        x0, x1, y0, y1 = self._current_roi_bounds(filt.shape)
        bounds = (x0, x1, y0, y1)
        frame_idx = self.current_index
        fixed_edge_line = self._fixed_edge_line
        if self._analysis_inflight:
            # Coalesce rapid requests: keep only the latest pending request.
            # Do not cancel current analysis here, so frame-to-frame playback
            # can still record metrics for frames that just moved off-screen.
            self._pending_analysis_request = (filt, bounds, frame_idx, fixed_edge_line)
            return
        self._launch_analysis_request(filt, bounds, frame_idx, fixed_edge_line=fixed_edge_line)

    def _launch_analysis_request(
        self,
        filtered: np.ndarray,
        bounds: Tuple[int, int, int, int],
        frame_idx: int,
        fixed_edge_line: Optional[Tuple[float, float]] = None,
    ):
        self._analysis_inflight = True
        self._analysis_token += 1
        token = self._analysis_token
        if self._analysis_cancel_event is not None:
            self._analysis_cancel_event.set()
        cancel_event = threading.Event()
        self._analysis_cancel_event = cancel_event
        ok = self._start_task(
            "analyze_current",
            token,
            frame_idx,
            _analyze_filtered_worker,
            filtered,
            bounds,
            fixed_edge_line,
            cancel_event,
            priority=2,
        )
        if not ok:
            self._analysis_inflight = False
            self._analysis_cancel_event = None

    def _try_lock_edge_from_focus_minimum(
        self, allow_existing: bool = False, rerun_with_fixed_edge: bool = True
    ) -> bool:
        if self._fixed_edge_line is not None and not allow_existing:
            return False
        if self._fixed_edge_line is None and not self._full_dynamic_results_ready:
            # Do not lock edge from quick-pass metrics; wait for dynamic full-pass results.
            if not self._all_frames_full_prepared():
                return False
            if (
                (not self._full_dynamic_pass_requested)
                and (not self._bulk_reprocess_active)
                and (not self._is_frame_loading)
                and (not self._task_worker_busy)
            ):
                self._full_dynamic_pass_requested = True
                self._log("Full cache ready. Running dynamic full-pass before fixed edge lock...")
                self._start_bulk_reprocess(
                    preserve_existing_results=True,
                    allow_disk_fallback=self._needs_disk_fallback_for_all_frames(),
                    reason="dynamic-full-pass-before-lock",
                )
            return False
        if self._bulk_reprocess_active:
            return False
        if len(self._results) < len(self.frames):
            return False

        rows = []
        for idx in sorted(self._results):
            r = self._results[idx]
            m_for_lock = (
                float(r.edge_m_dynamic)
                if np.isfinite(r.edge_m_dynamic)
                else (float(r.edge_m) if np.isfinite(r.edge_m) else np.nan)
            )
            b_for_lock = (
                float(r.edge_b_dynamic)
                if np.isfinite(r.edge_b_dynamic)
                else (float(r.edge_b) if np.isfinite(r.edge_b) else np.nan)
            )
            if (
                np.isfinite(r.step_sigma)
                and np.isfinite(m_for_lock)
                and np.isfinite(b_for_lock)
            ):
                rows.append((idx, float(r.step_sigma), m_for_lock, b_for_lock))
        if len(rows) < self._edge_min_candidates:
            return False

        sigmas = np.asarray([row[1] for row in rows], dtype=np.float64)
        min_sigma = float(np.nanmin(sigmas))
        sigma_std = float(np.nanstd(sigmas)) if np.isfinite(sigmas).any() else 0.0
        tol = max(abs(min_sigma) * self._edge_near_fraction, 0.05 * sigma_std, 1e-12)
        near_rows = [row for row in rows if row[1] <= (min_sigma + tol)]
        if len(near_rows) < self._edge_min_candidates:
            order = np.argsort(sigmas)
            top_n = int(min(max(self._edge_min_candidates, 1), len(rows)))
            near_rows = [rows[int(i)] for i in order[:top_n]]
        if len(near_rows) < 1:
            return False

        m_med = float(np.nanmedian(np.asarray([row[2] for row in near_rows], dtype=np.float64)))
        b_med = float(np.nanmedian(np.asarray([row[3] for row in near_rows], dtype=np.float64)))
        if not (np.isfinite(m_med) and np.isfinite(b_med)):
            return False

        ref_idx = int(min(near_rows, key=lambda row: row[1])[0])
        prev_line = self._fixed_edge_line
        self._fixed_edge_line = (m_med, b_med)
        self._fixed_edge_reference_index = ref_idx
        if prev_line is None:
            self._full_edge_refined = False
        should_rerun = True
        if prev_line is not None:
            y1 = float((self._current_filtered.shape[0] - 1) if self._current_filtered is not None else 4095)
            old_x0 = float(prev_line[1])
            old_x1 = float(prev_line[0] * y1 + prev_line[1])
            new_x0 = float(b_med)
            new_x1 = float(m_med * y1 + b_med)
            max_shift_px = max(abs(new_x0 - old_x0), abs(new_x1 - old_x1))
            # Avoid repeated reruns for tiny line changes.
            should_rerun = bool(max_shift_px > 0.5)
            if not should_rerun:
                self._log(
                    "Fixed edge refinement change below threshold; "
                    f"max_shift={max_shift_px:.3f}px. Keeping current reprocess outputs."
                )
                if self._current_filtered is not None:
                    # Refit current frame once with refined fixed line so overlay remains consistent.
                    self._request_analysis_current()
                elif self.current_index in self._results:
                    self._update_profile_plot(self._results[self.current_index])
                return False
            self._log(
                "Fixed edge refinement detected meaningful change: "
                f"max_shift={max_shift_px:.3f}px; rerunning full reprocess."
            )

        self._log(
            "Locked global edge from near-min sigma: "
            f"frame={ref_idx + 1}, m={m_med:.6f}, b={b_med:.6f}"
        )
        if not bool(rerun_with_fixed_edge):
            self.statusBar().showMessage(
                f"Locked edge from near-min sigma at frame {ref_idx + 1}; "
                "skipping fixed-edge rerun (disk-backed all-frame mode)."
            )
            self._log(
                "Skipping fixed-edge rerun for this cycle to avoid repeated disk rereads."
            )
            if self._current_filtered is not None:
                self._request_analysis_current()
            elif self.current_index in self._results:
                self._update_profile_plot(self._results[self.current_index])
            return False
        self.statusBar().showMessage(
            f"Locked edge from near-min sigma at frame {ref_idx + 1}; reprocessing all frames with fixed edge."
        )
        # On initial lock, jump the viewer to the selected near-minimum frame
        # without triggering a new quick-pass load.
        if prev_line is None and ref_idx != int(self.current_index):
            self._load_frame(ref_idx, cached_only=True)
        # Preserve existing points and overwrite frame-by-frame as refreshed results arrive.
        self._start_bulk_reprocess(
            preserve_existing_results=True,
            allow_disk_fallback=self._needs_disk_fallback_for_all_frames(),
            reason="fixed-edge-rerun",
        )
        return True

    def _maybe_start_full_reprocess_after_scan(self):
        if not self._roi_reprocess_after_scan:
            return
        if self._bulk_reprocess_active:
            return
        if not self._all_frames_full_prepared():
            # Wait until heavy full filtering is complete; then rerun all-frame ROI metrics.
            return
        if not self._is_dataset_complete():
            return
        if not self._is_processing_idle():
            return
        self._roi_reprocess_after_scan = False
        self._log(
            "Scan complete after ROI update. Running full dynamic pass before fixed-edge lock."
        )
        self._start_bulk_reprocess(
            preserve_existing_results=True,
            frame_indices=None,
            allow_disk_fallback=self._needs_disk_fallback_for_all_frames(),
            clear_queues=True,
            reason="scan-complete-full-reprocess",
        )

    def _maybe_finalize_playback_summary(self):
        if not self._playback_summary_pending:
            return
        if not self._is_scan_finished():
            return
        self._playback_summary_pending = False
        self._log("Playback reached final frame")
        self._log_final_summary("Final summary after playback")

    def _on_task_done(self, kind: str, token: int, frame_index: int, result, error):
        if self._shutting_down:
            if kind == "prepare_full":
                self._full_running_count = max(0, int(self._full_running_count) - 1)
                self._full_queued_indices.discard(frame_index)
                self._full_active_indices.discard(frame_index)
            elif kind == "refresh_full_frame":
                self._full_refresh_active_indices.discard(frame_index)
            elif kind == "load_filter_display":
                self._task_worker_busy = False
                self._is_frame_loading = False
                self._queued_load_indices.discard(frame_index)
                if self._active_load_index == frame_index:
                    self._active_load_index = None
            elif kind == "analyze_current":
                self._task_worker_busy = False
                self._analysis_inflight = False
            self._update_filter_queue_indicator()
            return
        if kind == "prepare_full":
            self._handle_full_prepare_result(frame_index, result, error)
            return

        if kind == "bulk_reprocess":
            if token != self._bulk_reprocess_token:
                self._pump_task_queue()
                return

            self._bulk_reprocess_done += 1
            if error is None and result is not None:
                filtered_opt, fit_result = result
                fit_result = self._preserve_previous_edge_geometry(frame_index, fit_result)
                if filtered_opt is not None:
                    self._cache_full_filtered(frame_index, filtered_opt)
                    self._cache_filtered(frame_index, filtered_opt)
                    if frame_index == self.current_index:
                        self._current_filtered = filtered_opt
                        self._set_display_image(filtered_opt)
                self._results[frame_index] = fit_result
                self._result_is_full[frame_index] = bool(
                    (filtered_opt is not None) or (frame_index in self._full_prepared_indices)
                )

                if frame_index == self.current_index:
                    self._update_profile_plot(fit_result)
            self._update_metric_plot()

            if self._bulk_reprocess_done >= self._bulk_reprocess_total:
                self._bulk_reprocess_active = False
                self._bulk_reprocess_uses_disk = False
                if (
                    self._bulk_reprocess_all_frames
                    and self._fixed_edge_line is None
                    and not self._current_bulk_used_fixed
                ):
                    self._full_dynamic_results_ready = True
                    self._full_dynamic_pass_requested = True
                    self._log("Dynamic full-pass complete. Fixed edge lock will use full-pass results.")
                # Ensure currently displayed frame is fully refreshed after batch completion.
                cur_full = self._get_full_filtered_image(self.current_index)
                if cur_full is not None:
                    self._current_filtered = cur_full
                    self._set_display_image(cur_full)
                cur_result = self._results.get(self.current_index, None)
                if cur_result is not None:
                    self._update_profile_plot(cur_result)
                self._update_metric_plot()
                self.statusBar().showMessage(
                    f"ROI reprocess complete: {self._bulk_reprocess_done} frames updated."
                )
                self._log(f"Reprocessing complete: {self._bulk_reprocess_done}/{self._bulk_reprocess_total}")
                reran = False
                rerun_with_fixed = not self._needs_disk_fallback_for_all_frames()
                if self._fixed_edge_line is None:
                    reran = self._try_lock_edge_from_focus_minimum(
                        rerun_with_fixed_edge=rerun_with_fixed
                    )
                elif not self._full_edge_refined:
                    self._full_edge_refined = True
                    if rerun_with_fixed:
                        self._log("Refining fixed edge from full-pass results...")
                    else:
                        self._log(
                            "Refining fixed edge from full-pass results "
                            "(no additional rerun in disk-backed mode)..."
                        )
                    reran = self._try_lock_edge_from_focus_minimum(
                        allow_existing=True,
                        rerun_with_fixed_edge=rerun_with_fixed,
                    )
                if reran:
                    self._pump_task_queue()
                    return
                if self._fixed_edge_line is not None:
                    self._log_final_summary("Final summary after fixed-edge reprocess")
                self._bulk_reprocess_all_frames = False
            else:
                self.statusBar().showMessage(
                    f"ROI reprocessing... {self._bulk_reprocess_done}/{self._bulk_reprocess_total}"
                )
                log_step = (
                    1
                    if int(self._bulk_reprocess_total) <= 40
                    else max(1, int(self._bulk_reprocess_total) // 10)
                )
                if (
                    self._bulk_reprocess_done == 1
                    or (self._bulk_reprocess_done % log_step == 0)
                ):
                    self._log(
                        f"Reprocessing progress: {self._bulk_reprocess_done}/{self._bulk_reprocess_total}"
                    )
            self._pump_task_queue()
            self._update_filter_queue_indicator()
            return

        if kind == "refresh_full_frame":
            self._full_refresh_active_indices.discard(frame_index)
            if token == int(self._roi_generation):
                if error is None and result is not None:
                    _, fit_result = result
                    fit_result = self._preserve_previous_edge_geometry(frame_index, fit_result)
                    self._results[frame_index] = fit_result
                    self._result_is_full[frame_index] = True
                    if frame_index == self.current_index:
                        self._update_profile_plot(fit_result)
                        self._try_lock_edge_from_focus_minimum()
                    self._update_metric_plot()
                elif error is not None:
                    self._log(f"Full-frame ROI refresh failed for frame {frame_index + 1}: {error}")
            self._maybe_start_full_reprocess_after_scan()
            self._update_filter_queue_indicator()
            return

        self._task_worker_busy = False
        if kind == "load_filter_display":
            self._is_frame_loading = False
            self._queued_load_indices.discard(frame_index)
            if self._active_load_index == frame_index:
                self._active_load_index = None

        if kind == "load_filter_display":
            if token != int(self._load_generation):
                self._pump_task_queue()
                self._update_filter_queue_indicator()
                return
            if error is not None:
                self.statusBar().showMessage(f"Failed to load/filter frame {frame_index + 1}: {error}")
                self._log(f"Load/filter failed for frame {frame_index + 1}: {error}")
            elif result is not None:
                if not self._roi_initialized_from_first_frame:
                    self._initialize_roi_from_shape(result.shape)
                self._cache_filtered(frame_index, result)
                self._seen_frame_indices.add(int(frame_index))
                # Immediately queue full-quality filtering for this frame.
                self._enqueue_full_prepare(frame_index)
                self.current_index = frame_index
                self._current_filtered = result
                self._set_display_image(result)
                # Apply best-available result immediately so overlay updates with frame changes.
                existing_result = self._results.get(frame_index, None)
                if existing_result is not None:
                    self._update_profile_plot(existing_result)
                else:
                    self.edge_line_item.hide()
                self._request_analysis_current()
                self._schedule_prefetch(frame_index)
                self._update_metric_plot()
                self._log(
                    f"Quick frame ready: {frame_index + 1}/{len(self.frames)} "
                    f"({self.frames[frame_index].path.name})"
                )
            self._pump_task_queue()
            self._update_filter_queue_indicator()
            return

        if kind == "analyze_current":
            self._analysis_inflight = False
            is_current_request = token == self._analysis_token
            if (error is not None) and is_current_request:
                self.statusBar().showMessage(f"Fit failed for frame {frame_index + 1}: {error}")
            elif (result is not None) and is_current_request:
                result = self._preserve_previous_edge_geometry(frame_index, result)
                self._results[frame_index] = result
                self._result_is_full[frame_index] = bool(frame_index in self._full_prepared_indices)
                self._update_metric_plot()
                if frame_index == self.current_index:
                    self._update_profile_plot(result)
                    self._try_lock_edge_from_focus_minimum()

            if self._pending_analysis_request is not None:
                pending = self._pending_analysis_request
                self._pending_analysis_request = None
                _pfilt, pbounds, pframe, pfixed = pending
                # Use latest currently displayed image data for final recompute.
                if self._current_filtered is not None and pframe == self.current_index:
                    self._launch_analysis_request(
                        self._current_filtered,
                        pbounds,
                        pframe,
                        fixed_edge_line=pfixed,
                    )
            self._pump_task_queue()
            self._update_filter_queue_indicator()
            return

        self._pump_task_queue()
        self._maybe_finalize_playback_summary()
        self._update_filter_queue_indicator()

    def _load_frame(self, idx: int, *, cached_only: bool = False):
        if self._shutting_down:
            return False
        if not self.frames:
            return False
        idx = max(0, min(len(self.frames) - 1, int(idx)))
        if idx in self._queued_load_indices or idx == self._active_load_index:
            return False

        frame = self.frames[idx]
        cached = self._get_filtered_image(idx)
        self.frame_label.setText(
            f"Frame {idx + 1}/{len(self.frames)} | file={frame.path.name} | position={frame.position:.4f}"
        )
        if cached is not None:
            if not self._roi_initialized_from_first_frame:
                self._initialize_roi_from_shape(cached.shape)
            self._seen_frame_indices.add(int(idx))
            # Ensure revisited quick-cached frames still enter the full queue.
            self._enqueue_full_prepare(idx)
            self.current_index = idx
            self._current_filtered = cached
            self._set_display_image(cached)
            # Apply best-available result immediately so ROI edge overlay tracks frame navigation.
            cached_result = self._results.get(idx, None)
            if cached_result is not None:
                self._update_profile_plot(cached_result)
            else:
                self.edge_line_item.hide()
            self._request_analysis_current()
            self._schedule_prefetch(idx)
            self._update_metric_plot()
            self._update_filter_queue_indicator()
            return True
        if cached_only:
            return False

        self._is_frame_loading = True
        self._current_filtered = None
        self.edge_line_item.hide()
        self._pending_analysis_request = None
        self.statusBar().showMessage(
            f"Loading quick pass for frame {idx + 1}/{len(self.frames)} in background..."
        )
        self._log(f"Loading quick pass for frame {idx + 1}/{len(self.frames)}")
        token = int(self._load_generation)
        ok = self._start_task(
            "load_filter_display",
            token,
            idx,
            _load_and_quick_filter_worker,
            frame.path,
            priority=0,
        )
        if ok:
            self._queued_load_indices.add(idx)
            self._update_filter_queue_indicator()
            return True

        self._is_frame_loading = False
        self.statusBar().showMessage(
            f"Backpressure: queue full, could not enqueue frame {idx + 1}/{len(self.frames)}"
        )
        self._update_filter_queue_indicator()
        return False

    def _update_profile_plot(self, result: FitResult):
        if result.profile_x is None or result.profile_y is None:
            self.esf_raw_curve.setData([], [])
            self.esf_fit_curve.setData([], [])
            self.pft_curve.setData([], [])
            self.mtf_curve.setData([], [])
            self.mtf50_line.hide()
            self.pft_plot.setXRange(-30.0, 30.0, padding=0.0)
            self.mtf_plot.setXRange(0.0, 0.4, padding=0.0)
            self.edge_line_item.hide()
            return

        self.esf_raw_curve.setData(result.profile_x, result.profile_y)
        if result.profile_fit is not None:
            self.esf_fit_curve.setData(result.profile_x, result.profile_fit)
        else:
            self.esf_fit_curve.setData([], [])

        if result.lsf_x is not None and result.lsf_y is not None:
            self.pft_curve.setData(result.lsf_x, result.lsf_y)
        else:
            self.pft_curve.setData([], [])

        if result.mtf_f is not None and result.mtf_y is not None:
            self.mtf_curve.setData(result.mtf_f, result.mtf_y)
        else:
            self.mtf_curve.setData([], [])

        if np.isfinite(result.mtf50):
            self.mtf50_line.setPos(float(result.mtf50))
            self.mtf50_line.show()
        else:
            self.mtf50_line.hide()

        self._update_edge_overlay(result)

    def _update_edge_overlay(self, result: FitResult):
        if self._current_filtered is None:
            self.edge_line_item.hide()
            return
        if not (
            np.isfinite(result.edge_m)
            and np.isfinite(result.edge_b)
            and np.isfinite(result.edge_ymin)
            and np.isfinite(result.edge_ymax)
        ):
            self.edge_line_item.hide()
            return

        h, w = self._current_filtered.shape
        y0 = float(max(0.0, min(float(h - 1), result.edge_ymin)))
        y1 = float(max(0.0, min(float(h - 1), result.edge_ymax)))
        if y1 <= y0:
            self.edge_line_item.hide()
            return

        x0 = float(result.edge_m * y0 + result.edge_b)
        x1 = float(result.edge_m * y1 + result.edge_b)
        x0 = float(max(0.0, min(float(w - 1), x0)))
        x1 = float(max(0.0, min(float(w - 1), x1)))
        self.edge_line_item.setData(
            np.asarray([x0, x1], dtype=np.float64),
            np.asarray([y0, y1], dtype=np.float64),
        )
        self.edge_line_item.show()

    @staticmethod
    def _local_parabola_fit(
        x: np.ndarray,
        y: np.ndarray,
        local_points: int = 7,
        find: str = "min",
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
        finite = np.isfinite(x) & np.isfinite(y)
        if int(np.count_nonzero(finite)) < 3:
            return None
        xf = np.asarray(x[finite], dtype=np.float64)
        yf = np.asarray(y[finite], dtype=np.float64)
        order = np.argsort(xf)
        xf = xf[order]
        yf = yf[order]
        if xf.size < 3:
            return None
        if find == "max":
            center_idx = int(np.nanargmax(yf))
        else:
            center_idx = int(np.nanargmin(yf))
        center_x = float(xf[center_idx])
        local_n = int(max(3, min(local_points, xf.size)))
        nearest = np.argsort(np.abs(xf - center_x))[:local_n]
        nearest = np.sort(nearest)
        x_local = xf[nearest]
        y_local = yf[nearest]
        if x_local.size < 3:
            return None
        coeff = np.polyfit(x_local, y_local, 2)
        a, b, c = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
        if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)) or abs(a) < 1e-15:
            return None
        if find == "max" and a >= 0.0:
            return None
        if find != "max" and a <= 0.0:
            return None
        x_line = np.linspace(float(np.nanmin(x_local)), float(np.nanmax(x_local)), 200)
        y_line = np.polyval(coeff, x_line)
        x_vertex = -b / (2.0 * a)
        y_vertex = float(np.polyval(coeff, x_vertex))
        return x_line, y_line, float(x_vertex), y_vertex

    def _update_metric_plot(self):
        if not self.frames:
            self.curve_sigma_quick.setData([], [])
            self.curve_sigma_full.setData([], [])
            self.curve_sigma_highlight.setData([], [])
            self.curve_quad_fit.setData([], [])
            self.curve_psf_quick.setData([], [])
            self.curve_psf_full.setData([], [])
            self.curve_psf_highlight.setData([], [])
            self.curve_psf_fit.setData([], [])
            self.curve_mtf50_quick.setData([], [])
            self.curve_mtf50_full.setData([], [])
            self.curve_mtf50_highlight.setData([], [])
            self.curve_mtf50_fit.setData([], [])
            empty = np.empty(0, dtype=np.float64)
            self.sigma_error_bars.setData(
                x=empty,
                y=empty,
                top=empty,
                bottom=empty,
                beam=0.0,
            )
            self._optimal_focus_position = np.nan
            self._optimal_focus_sigma = np.nan
            self._optimal_psf_position = np.nan
            self._optimal_psf_sigma = np.nan
            self._optimal_mtf50_position = np.nan
            self._optimal_mtf50_value = np.nan
            self.optimal_focus_label.setText("Optimal motor position: --")
            return

        xs: List[float] = []
        sigmas: List[float] = []
        sigma_errs: List[float] = []
        psf_sigmas: List[float] = []
        mtf50s: List[float] = []
        xs_quick: List[float] = []
        sigma_quick: List[float] = []
        psf_quick: List[float] = []
        mtf_quick: List[float] = []
        xs_full: List[float] = []
        sigma_full: List[float] = []
        psf_full: List[float] = []
        mtf_full: List[float] = []
        for idx in sorted(self._results):
            r = self._results[idx]
            pos = self.frames[idx].position
            is_full = bool(self._result_is_full.get(idx, idx in self._full_prepared_indices))
            xs.append(pos)
            sigmas.append(r.step_sigma if np.isfinite(r.step_sigma) else np.nan)
            sigma_errs.append(
                r.step_sigma_stderr if np.isfinite(r.step_sigma_stderr) else np.nan
            )
            psf_sigmas.append(r.psf_sigma if np.isfinite(r.psf_sigma) else np.nan)
            mtf50s.append(r.mtf50 if np.isfinite(r.mtf50) else np.nan)
            if is_full:
                xs_full.append(pos)
                sigma_full.append(r.step_sigma if np.isfinite(r.step_sigma) else np.nan)
                psf_full.append(r.psf_sigma if np.isfinite(r.psf_sigma) else np.nan)
                mtf_full.append(r.mtf50 if np.isfinite(r.mtf50) else np.nan)
            else:
                xs_quick.append(pos)
                sigma_quick.append(r.step_sigma if np.isfinite(r.step_sigma) else np.nan)
                psf_quick.append(r.psf_sigma if np.isfinite(r.psf_sigma) else np.nan)
                mtf_quick.append(r.mtf50 if np.isfinite(r.mtf50) else np.nan)

        x_arr = np.asarray(xs, dtype=np.float64)
        sigma_arr = np.asarray(sigmas, dtype=np.float64)
        sigma_err_arr = np.asarray(sigma_errs, dtype=np.float64)
        psf_sigma_arr = np.asarray(psf_sigmas, dtype=np.float64)
        mtf50_arr = np.asarray(mtf50s, dtype=np.float64)

        self.curve_sigma_quick.setData(
            np.asarray(xs_quick, dtype=np.float64),
            np.asarray(sigma_quick, dtype=np.float64),
        )
        self.curve_sigma_full.setData(
            np.asarray(xs_full, dtype=np.float64),
            np.asarray(sigma_full, dtype=np.float64),
        )
        self.curve_psf_quick.setData(
            np.asarray(xs_quick, dtype=np.float64),
            np.asarray(psf_quick, dtype=np.float64),
        )
        self.curve_psf_full.setData(
            np.asarray(xs_full, dtype=np.float64),
            np.asarray(psf_full, dtype=np.float64),
        )
        self.curve_mtf50_quick.setData(
            np.asarray(xs_quick, dtype=np.float64),
            np.asarray(mtf_quick, dtype=np.float64),
        )
        self.curve_mtf50_full.setData(
            np.asarray(xs_full, dtype=np.float64),
            np.asarray(mtf_full, dtype=np.float64),
        )

        # Highlight current frame across all lower metric plots.
        current_idx = int(self.current_index)
        current_x = float(self.frames[current_idx].position)
        current_result = self._results.get(current_idx, None)
        if current_result is None or not np.isfinite(current_x):
            self.curve_sigma_highlight.setData([], [])
            self.curve_psf_highlight.setData([], [])
            self.curve_mtf50_highlight.setData([], [])
        else:
            sigma_y = float(current_result.step_sigma) if np.isfinite(current_result.step_sigma) else np.nan
            psf_y = float(current_result.psf_sigma) if np.isfinite(current_result.psf_sigma) else np.nan
            mtf_y = float(current_result.mtf50) if np.isfinite(current_result.mtf50) else np.nan

            if np.isfinite(sigma_y):
                self.curve_sigma_highlight.setData(
                    np.asarray([current_x], dtype=np.float64),
                    np.asarray([sigma_y], dtype=np.float64),
                )
            else:
                self.curve_sigma_highlight.setData([], [])

            if np.isfinite(psf_y):
                self.curve_psf_highlight.setData(
                    np.asarray([current_x], dtype=np.float64),
                    np.asarray([psf_y], dtype=np.float64),
                )
            else:
                self.curve_psf_highlight.setData([], [])

            if np.isfinite(mtf_y):
                self.curve_mtf50_highlight.setData(
                    np.asarray([current_x], dtype=np.float64),
                    np.asarray([mtf_y], dtype=np.float64),
                )
            else:
                self.curve_mtf50_highlight.setData([], [])

        finite_err = (
            np.isfinite(x_arr)
            & np.isfinite(sigma_arr)
            & np.isfinite(sigma_err_arr)
            & (sigma_err_arr > 0)
        )
        if finite_err.any():
            self.sigma_error_bars.setData(
                x=x_arr[finite_err],
                y=sigma_arr[finite_err],
                top=sigma_err_arr[finite_err],
                bottom=sigma_err_arr[finite_err],
                beam=0.0,
            )
        else:
            empty = np.empty(0, dtype=np.float64)
            self.sigma_error_bars.setData(
                x=empty,
                y=empty,
                top=empty,
                bottom=empty,
                beam=0.0,
            )

        # Fit locally around current extrema using all currently available points
        # (quick + full), then update smoothly as reprocessed points overwrite.
        local_points = int(max(3, self._local_fit_points))
        sigma_fit = self._local_parabola_fit(
            x_arr, sigma_arr, local_points=local_points, find="min"
        )
        if sigma_fit is not None:
            x_line, y_line, x_min, y_min = sigma_fit
            self.curve_quad_fit.setData(x_line, y_line)
            self._optimal_focus_position = float(x_min)
            self._optimal_focus_sigma = float(y_min)
        else:
            self.curve_quad_fit.setData([], [])
            self._optimal_focus_position = np.nan
            self._optimal_focus_sigma = np.nan

        psf_fit = self._local_parabola_fit(
            x_arr, psf_sigma_arr, local_points=local_points, find="min"
        )
        if psf_fit is not None:
            self.curve_psf_fit.setData(psf_fit[0], psf_fit[1])
            self._optimal_psf_position = float(psf_fit[2])
            self._optimal_psf_sigma = float(psf_fit[3])
        else:
            self.curve_psf_fit.setData([], [])
            self._optimal_psf_position = np.nan
            self._optimal_psf_sigma = np.nan

        mtf_fit = self._local_parabola_fit(
            x_arr, mtf50_arr, local_points=local_points, find="max"
        )
        if mtf_fit is not None:
            self.curve_mtf50_fit.setData(mtf_fit[0], mtf_fit[1])
            self._optimal_mtf50_position = float(mtf_fit[2])
            self._optimal_mtf50_value = float(mtf_fit[3])
        else:
            self.curve_mtf50_fit.setData([], [])
            self._optimal_mtf50_position = np.nan
            self._optimal_mtf50_value = np.nan

        if np.isfinite(self._optimal_mtf50_position):
            self.optimal_focus_label.setText(
                f"Optimal motor position (MTF50): {self._optimal_mtf50_position:.5f}"
            )
        elif np.isfinite(self._optimal_focus_position):
            self.optimal_focus_label.setText(
                f"Optimal motor position (sigma): {self._optimal_focus_position:.5f}"
            )
        else:
            self.optimal_focus_label.setText("Optimal motor position: --")

        if not self._bulk_reprocess_active and not self._is_frame_loading:
            sigma_txt = (
                f"sigma: m={self._optimal_focus_position:.5f}, v={self._optimal_focus_sigma:.5f}"
                if np.isfinite(self._optimal_focus_position) and np.isfinite(self._optimal_focus_sigma)
                else "sigma: n/a"
            )
            psf_txt = (
                f"lsf: m={self._optimal_psf_position:.5f}, v={self._optimal_psf_sigma:.5f}"
                if np.isfinite(self._optimal_psf_position) and np.isfinite(self._optimal_psf_sigma)
                else "lsf: n/a"
            )
            mtf_txt = (
                f"mtf50: m={self._optimal_mtf50_position:.5f}, v={self._optimal_mtf50_value:.5f}"
                if np.isfinite(self._optimal_mtf50_position) and np.isfinite(self._optimal_mtf50_value)
                else "mtf50: n/a"
            )
            self.statusBar().showMessage(
                f"Best focus (local quadratic) | {sigma_txt} | {psf_txt} | {mtf_txt}"
            )

    def _recompute_current(self):
        if self._current_filtered is None:
            self._load_frame(self.current_index)
            return
        self._request_analysis_current()

    def _on_roi_region_changed(self):
        if self._suppress_roi_callbacks:
            return
        if self._current_filtered is None:
            return
        # Drop in-flight ROI analysis while dragging so old ROI positions
        # do not race back into the UI.
        if self._analysis_cancel_event is not None:
            self._analysis_cancel_event.set()
        self._analysis_token += 1
        self._pending_analysis_request = None
        x0, x1, y0, y1 = self._current_roi_bounds(self._current_filtered.shape)
        if self._is_dataset_complete():
            self.statusBar().showMessage(
                f"ROI moving: x=[{x0},{x1}) y=[{y0},{y1}) | release mouse to reprocess all frames"
            )
        else:
            self.statusBar().showMessage(
                f"ROI moving: x=[{x0},{x1}) y=[{y0},{y1}) | release mouse to refit"
            )
        if self._live_roi_preview:
            self._roi_analysis_timer.start()

    def _open_files_dialog(self):
        start_dir = self._last_open_dir or str(Path.cwd())
        files, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select TIFF files",
            start_dir,
            "TIFF Images (*.tif *.tiff);;All Files (*)",
        )
        if not files:
            return
        first_path = Path(files[0]).expanduser()
        self._last_open_dir = str(first_path.parent)
        try:
            frames = build_frames_from_files(files, start_position=0.0, step_size=1.0)
        except Exception as ex:
            self.statusBar().showMessage(f"Open files failed: {ex}")
            self._log(f"Open files failed: {ex}")
            return
        self._replace_dataset(frames, source="file dialog")

    def _replace_dataset(self, frames: Sequence[FrameInfo], source: str = "manual"):
        new_frames = [
            FrameInfo(index=i, path=Path(f.path), position=float(f.position))
            for i, f in enumerate(frames)
        ]
        if not new_frames:
            return

        self.playing = False
        self.timer.stop()
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(False)
        self.btn_play.blockSignals(False)
        self.btn_play.setText("Play")

        if self._analysis_cancel_event is not None:
            self._analysis_cancel_event.set()
        self._analysis_cancel_event = None
        self._analysis_inflight = False
        self._pending_analysis_request = None
        self._analysis_token += 1
        self._load_generation += 1
        self._roi_generation += 1
        self._bulk_reprocess_token += 1

        self._pending_bulk_reprocess = None
        self._roi_reprocess_after_scan = False
        self._pause_full_prepare = False
        self._bulk_reprocess_active = False
        self._bulk_reprocess_total = 0
        self._bulk_reprocess_done = 0
        self._bulk_reprocess_uses_disk = False
        self._bulk_reprocess_all_frames = False
        self._current_bulk_used_fixed = False
        self._full_dynamic_results_ready = False
        self._full_dynamic_pass_requested = False
        self._full_prepare_refresh_requested = False
        self._last_full_prepare_logged_count = -1
        self._full_edge_refined = False
        self._fixed_edge_line = None
        self._fixed_edge_reference_index = None
        self._playback_summary_pending = False
        self._stream_next_index = None

        self._clear_task_queue()
        self._clear_full_queue()

        self._quick_filtered_cache.clear()
        self._quick_cache_bytes = 0
        self._full_filtered_cache.clear()
        self._full_cache_bytes = 0
        self._full_prepared_indices.clear()
        self._results.clear()
        self._result_is_full.clear()
        self._seen_frame_indices.clear()
        self._current_filtered = None
        self._roi_initialized_from_first_frame = False
        self._last_committed_roi_rect = self._roi_rect()

        self._optimal_focus_position = np.nan
        self._optimal_focus_sigma = np.nan
        self._optimal_psf_position = np.nan
        self._optimal_psf_sigma = np.nan
        self._optimal_mtf50_position = np.nan
        self._optimal_mtf50_value = np.nan

        self.frames = new_frames
        self.current_index = 0
        self.edge_line_item.hide()
        self.esf_raw_curve.setData([], [])
        self.esf_fit_curve.setData([], [])
        self.pft_curve.setData([], [])
        self.mtf_curve.setData([], [])
        self.mtf50_line.hide()
        self.frame_label.setText("")
        self._update_metric_plot()
        self._update_filter_queue_indicator()

        self.statusBar().showMessage(
            f"Loaded {len(self.frames)} files from {source}; starting full filtering queue..."
        )
        self._log(
            f"Dataset replaced ({source}): {len(self.frames)} frames. Starting processing."
        )
        self._load_frame(0)
        for idx in range(len(self.frames)):
            self._enqueue_full_prepare(idx)

    def _on_roi_region_change_finished(self):
        if self._suppress_roi_callbacks:
            return
        self._roi_analysis_timer.stop()
        if self._analysis_cancel_event is not None:
            self._analysis_cancel_event.set()
        self._analysis_token += 1
        self._pending_analysis_request = None
        roi_rect = self._roi_rect()
        if self._last_committed_roi_rect is not None:
            max_diff = max(abs(a - b) for a, b in zip(roi_rect, self._last_committed_roi_rect))
            if max_diff < 0.5:
                return
        self._last_committed_roi_rect = roi_rect
        self._roi_generation += 1
        # ROI change invalidates any previously locked global edge geometry.
        self._fixed_edge_line = None
        self._fixed_edge_reference_index = None
        self._full_edge_refined = False
        self._full_dynamic_results_ready = False
        self._full_dynamic_pass_requested = False
        self._full_prepare_refresh_requested = False
        self._pending_bulk_reprocess = None
        self._clear_queued_bulk_tasks()
        self._stream_next_index = int(self.current_index + 1)
        if self._is_dataset_complete():
            # Dataset done: rerun all-frame ROI metrics after full filtering finishes.
            if self._all_frames_full_prepared():
                self._roi_reprocess_after_scan = False
                self._start_bulk_reprocess(
                    preserve_existing_results=True,
                    frame_indices=None,
                    allow_disk_fallback=self._needs_disk_fallback_for_all_frames(),
                    clear_queues=True,
                    reason="roi-release-scan-finished",
                )
            else:
                self._roi_reprocess_after_scan = True
                self._log(
                    "ROI updated after scan; waiting for full filtering queue "
                    "to complete before all-frame ROI reprocess."
                )
            return
        # During scanning: clear stale metrics, quickly reanalyze already-collected cached frames,
        # and complete a full all-frame pass when the scan reaches the end.
        self._roi_reprocess_after_scan = True
        # Always refresh current-frame peak-shape plots immediately after ROI change.
        self._recompute_current()
        cached_indices = self._cached_frame_indices_upto_current()
        cached_indices = [i for i in cached_indices if int(i) != int(self.current_index)]
        if cached_indices:
            self._start_bulk_reprocess(
                preserve_existing_results=True,
                frame_indices=cached_indices,
                allow_disk_fallback=False,
                clear_queues=False,
                reason="roi-release-partial-during-scan",
            )
        self._log(
            "ROI updated during scan: rerunning line/peak metrics on cached frames now; "
            "full all-frame reprocess will run at scan completion."
        )

    def _prev_frame(self):
        self._load_frame(self._neighbor_index_by_position(-1))

    def _next_frame(self):
        self._load_frame(self._neighbor_index_by_position(+1))

    def _neighbor_index_by_position(self, step: int) -> int:
        if not self.frames:
            return int(self.current_index)
        ordered = sorted(
            range(len(self.frames)),
            key=lambda i: (float(self.frames[i].position), int(i)),
        )
        cur_idx = int(max(0, min(len(self.frames) - 1, int(self.current_index))))
        try:
            cur_rank = int(ordered.index(cur_idx))
        except ValueError:
            cur_rank = 0
        next_rank = int(max(0, min(len(ordered) - 1, cur_rank + int(step))))
        return int(ordered[next_rank])

    def _on_interval_changed(self, value: int):
        self.interval_ms = int(value)
        self.timer.setInterval(self.interval_ms)

    def _on_local_fit_points_changed(self, value: int):
        v = int(value)
        if v % 2 == 0:
            v += 1
        v = int(max(3, min(31, v)))
        if v != int(self.local_fit_points_spin.value()):
            prev = self.local_fit_points_spin.blockSignals(True)
            self.local_fit_points_spin.setValue(v)
            self.local_fit_points_spin.blockSignals(prev)
        self._local_fit_points = v
        self._update_metric_plot()

    def _toggle_play(self, checked: bool):
        self.playing = bool(checked)
        self.btn_play.setText("Pause" if self.playing else "Play")
        if self.playing:
            self._stream_next_index = int(self.current_index + 1)
            self.timer.start()
            self._log("Playback started")
        else:
            self.timer.stop()
            self._log("Playback paused")

    def _tick(self):
        if self._stream_next_index is None:
            self._stream_next_index = int(self.current_index + 1)
        nxt = int(self._stream_next_index)
        if nxt >= len(self.frames):
            self.timer.stop()
            self.btn_play.setChecked(False)
            if self._is_scan_finished():
                self._log("Playback reached final frame")
                self._log_final_summary("Final summary after playback")
            else:
                self._playback_summary_pending = True
                self._log("Playback reached final frame; waiting for last queued frame processing.")
            return
        enqueued = self._load_frame(nxt)
        if enqueued:
            self._stream_next_index = int(nxt + 1)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
            QtCore.QEvent.StyleChange,
        ):
            self._apply_pyqtgraph_theme_from_palette()
            if hasattr(self, "filter_queue_bar"):
                self._apply_filter_queue_theme()

    def closeEvent(self, event):
        self._shutting_down = True
        self.playing = False
        self.timer.stop()
        self._roi_analysis_timer.stop()
        self._full_future_timer.stop()
        if self._analysis_cancel_event is not None:
            self._analysis_cancel_event.set()
        self._analysis_inflight = False
        self._pending_analysis_request = None
        self._pending_bulk_reprocess = None
        self._roi_reprocess_after_scan = False
        self._pause_full_prepare = True
        self._bulk_reprocess_active = False
        self._bulk_reprocess_total = 0
        self._bulk_reprocess_done = 0
        self._clear_task_queue()
        self._clear_full_queue()
        self._full_active_indices.clear()
        self._full_refresh_active_indices.clear()

        for pool in (self._thread_pool, self._bulk_thread_pool):
            try:
                pool.clear()
            except Exception:
                pass
        for pool in (self._thread_pool, self._bulk_thread_pool):
            try:
                pool.waitForDone(3000)
            except TypeError:
                pool.waitForDone()
            except Exception:
                pass
        try:
            self._full_process_pool.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            self._full_process_pool.shutdown(wait=True)
        except Exception:
            pass

        self._update_filter_queue_indicator()
        super().closeEvent(event)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline focus scan viewer")
    p.add_argument(
        "image_dir",
        nargs="?",
        default=None,
        help="Optional directory containing TIFF images (if omitted, use Open Files... in the UI)",
    )
    p.add_argument("--start-position", type=float, default=0.0, help="Fake motor start position")
    p.add_argument("--step-size", type=float, default=1.0, help="Fake motor step size")
    p.add_argument(
        "--series-key",
        type=str,
        default=None,
        help="Optional series key to select files named like '<series-key>_####.tif'",
    )
    p.add_argument(
        "--positions-csv",
        type=str,
        default=None,
        help="Optional CSV mapping filename->position (columns: filename,position)",
    )
    p.add_argument("--interval-ms", type=int, default=500, help="Playback interval in milliseconds")
    p.add_argument(
        "--max-workers-total",
        type=int,
        default=8,
        help=(
            "Maximum total concurrent workers across task, bulk, and full pools "
            "(minimum effective value is 3). "
            "Auto-raised if needed to satisfy requested --full-workers/--c."
        ),
    )
    p.add_argument(
        "--bulk-workers",
        type=int,
        default=1,
        help="Optional cap for ROI/bulk reprocess workers (recommended: 1 on beamline PCs).",
    )
    p.add_argument(
        "--full-workers",
        "--filter-workers",
        "--max-processes",
        "--c",
        dest="full_workers",
        type=int,
        default=6,
        help=(
            "Optional cap for full-filter process workers (primary CPU load). "
            "Aliases: --max-processes, --c. "
            "Total worker budget is auto-adjusted to honor this value."
        ),
    )
    p.add_argument(
        "--full-cache-gb",
        "--max-ram-gb",
        "--mem",
        type=float,
        default=10.0,
        help=(
            "Full filtered image cache budget in GB (LRU, minimum 0.25 GB). "
            "Aliases: --max-ram-gb, --mem"
        ),
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if lm is None:
        raise SystemExit("lmfit is required for step-model fitting. Install with: pip install lmfit")

    positions_csv = Path(args.positions_csv).expanduser().resolve() if args.positions_csv else None
    if positions_csv is not None and not positions_csv.exists():
        raise SystemExit(f"Positions CSV does not exist: {positions_csv}")

    frames: List[FrameInfo] = []
    if args.image_dir:
        image_dir = Path(args.image_dir).expanduser().resolve()
        if not image_dir.exists():
            raise SystemExit(f"Image directory does not exist: {image_dir}")
        frames = discover_frames(
            image_dir=image_dir,
            start_position=args.start_position,
            step_size=args.step_size,
            positions_csv=positions_csv,
            series_key=args.series_key,
        )

    requested_full_workers = int(max(1, int(args.full_workers)))
    requested_bulk_workers = int(max(1, int(args.bulk_workers)))
    effective_max_workers_total = int(
        max(
            int(args.max_workers_total),
            1 + requested_bulk_workers + requested_full_workers,
        )
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOption("imageAxisOrder", "row-major")

    w = FocusOfflineWindow(
        frames=frames,
        interval_ms=args.interval_ms,
        max_workers_total=effective_max_workers_total,
        bulk_workers=requested_bulk_workers,
        full_workers=requested_full_workers,
        full_cache_gb=args.full_cache_gb,
    )
    w.show()
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
