import math
from typing import Any, Mapping

import numpy as np


DEFAULT_TRANSFER_TIME_PER_BYTES = 4.1203007518796994e-08
DEFAULT_IMAGE_BYTES_PV = "4dh4:cam1:ArraySize_RBV"
DEFAULT_IMAGING_EXPOSURE_PV = "4dh4:cam1:AcquireTime_RBV"
DEFAULT_DIFFRACTION_ACQUIRE_PV = "4dh4:he3PSD:AcquireTime_RBV"

def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default

def _caget_float(caget_func, pv: str) -> float | None:
    if not callable(caget_func):
        return None
    try:
        return _as_float(caget_func(str(pv)), None)
    except Exception:
        return None


def build_estimation_context(
    *,
    caget_func=None,
    image_bytes: float | None = None,
    imaging_exposure_time_s: float | None = None,
    diffraction_acquire_time_s: float | None = None,
    transfer_time_per_bytes: float | None = None,
) -> dict[str, float]:
    if image_bytes is None:
        image_bytes = _caget_float(caget_func, DEFAULT_IMAGE_BYTES_PV)
    if imaging_exposure_time_s is None:
        imaging_exposure_time_s = _caget_float(caget_func, DEFAULT_IMAGING_EXPOSURE_PV)
    if diffraction_acquire_time_s is None:
        diffraction_acquire_time_s = _caget_float(caget_func, DEFAULT_DIFFRACTION_ACQUIRE_PV)
    transfer_time_per_bytes = _as_float(transfer_time_per_bytes, DEFAULT_TRANSFER_TIME_PER_BYTES)
    return {
        "image_bytes": float(image_bytes) if image_bytes is not None else 0.0,
        "imaging_exposure_time_s": float(imaging_exposure_time_s) if imaging_exposure_time_s is not None else 0.0,
        "diffraction_acquire_time_s": float(diffraction_acquire_time_s) if diffraction_acquire_time_s is not None else 0.0,
        "transfer_time_per_bytes": float(transfer_time_per_bytes),
    }


def _unknown_estimate(units: int | None = None) -> dict[str, float | int | None]:
    return {
        "estimated_total_time_s": None,
        "estimated_total_units": int(units) if units is not None else None,
    }


def _known_estimate(total_time_s: float, total_units: int | None = None) -> dict[str, float | int | None]:
    return {
        "estimated_total_time_s": max(0.0, float(total_time_s)),
        "estimated_total_units": int(total_units) if total_units is not None else None,
    }


def _num_steps_from_step_size_scan(
    start: float,
    stop: float,
    step_size: float,
    *,
    include_stop: bool = True,
) -> int | None:
    if step_size == 0:
        return None
    if (stop - start) * step_size < 0:
        return None
    try:
        raw_stop = stop + (0.5 * step_size if include_stop else 0.0)
        positions = np.arange(start=start, stop=raw_stop, step=step_size)
    except Exception:
        return None
    tol = max(abs(step_size), abs(stop - start), 1.0) * 1e-12
    if step_size > 0:
        positions = positions[positions <= stop + tol]
    else:
        positions = positions[positions >= stop - tol]
    return int(len(positions))


def _transfer_time_per_image(context: Mapping[str, Any]) -> float:
    image_bytes = _as_float(context.get("image_bytes"), 0.0) or 0.0
    transfer_time_per_bytes = _as_float(context.get("transfer_time_per_bytes"), DEFAULT_TRANSFER_TIME_PER_BYTES)
    return float(image_bytes) * float(transfer_time_per_bytes)


def _estimate_imaging(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    num_exposures = max(1, _as_int(params.get("num_exposures"), 1) or 1)
    exposure_time = _as_float(params.get("exposure_time"), None)
    if exposure_time is None:
        exposure_time = _as_float(context.get("imaging_exposure_time_s"), None)
    if exposure_time is None:
        return _unknown_estimate(num_exposures)
    total_time = float(num_exposures) * (float(exposure_time) + _transfer_time_per_image(context))
    return _known_estimate(total_time, num_exposures)


def _estimate_imaging_scan(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    num_steps = _as_int(params.get("num_steps"), None)
    if num_steps is None:
        step_size = _as_float(params.get("step_size", params.get("step")), None)
        start_pos = _as_float(params.get("start_pos"), None)
        stop_pos = _as_float(params.get("stop_pos"), None)
        if step_size is None or start_pos is None or stop_pos is None:
            return _unknown_estimate()
        num_steps = _num_steps_from_step_size_scan(start_pos, stop_pos, step_size)
        if num_steps is None:
            return _unknown_estimate()
    num_steps = max(1, int(num_steps))
    num_exposures = max(1, _as_int(params.get("num_exposures"), 1) or 1)
    exposure_time = _as_float(params.get("exposure_time"), None)
    if exposure_time is None:
        exposure_time = _as_float(context.get("imaging_exposure_time_s"), None)
    if exposure_time is None:
        return _unknown_estimate(num_exposures * num_steps)
    total_units = num_exposures * num_steps
    total_time = float(total_units) * (float(exposure_time) + _transfer_time_per_image(context))
    return _known_estimate(total_time, total_units)


def _estimate_tomo_scan(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    start_angle = _as_float(params.get("start_angle"), 0.0) or 0.0
    stop_angle = _as_float(params.get("stop_angle"), 360.0) or 360.0
    include_stop_angle = bool(params.get("include_stop_angle", False))
    num_projections = _as_int(params.get("num_projections"), None)
    angle_step_size = _as_float(params.get("angle_step_size", params.get("angle_step")), None)
    if num_projections is not None:
        num_projections_calc = int(num_projections)
    elif angle_step_size is not None:
        num_projections_calc = _num_steps_from_step_size_scan(
            start_angle,
            stop_angle,
            angle_step_size,
            include_stop=include_stop_angle,
        )
        if num_projections_calc is None:
            return _unknown_estimate()
    else:
        return _unknown_estimate()
    if num_projections_calc <= 0:
        return _unknown_estimate()
    num_exposures = max(1, _as_int(params.get("num_exposures"), 1) or 1)
    exposure_time = _as_float(params.get("exposure_time"), None)
    if exposure_time is None:
        exposure_time = _as_float(context.get("imaging_exposure_time_s"), None)
    if exposure_time is None:
        return _unknown_estimate(num_exposures * num_projections_calc)
    total_units = num_exposures * num_projections_calc
    total_time = float(total_units) * (float(exposure_time) + _transfer_time_per_image(context))
    return _known_estimate(total_time, total_units)


def _estimate_adaptive_imaging_focus_scan(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    num_steps = max(2, _as_int(params.get("num_steps"), 15) or 15)
    start_pos = _as_float(params.get("start_pos"), None)
    stop_pos = _as_float(params.get("stop_pos"), None)
    focus_guess = _as_float(params.get("focus_guess"), None)
    scan_half_range = _as_float(params.get("scan_half_range"), None)
    has_explicit_bounds = (start_pos is not None) and (stop_pos is not None)
    has_guess_range = (focus_guess is not None) and (scan_half_range is not None)
    if not has_explicit_bounds and not has_guess_range:
        return _unknown_estimate(num_steps)
    exposure_time = _as_float(params.get("exposure_time"), None)
    if exposure_time is None:
        exposure_time = _as_float(context.get("imaging_exposure_time_s"), None)
    if exposure_time is None:
        return _unknown_estimate(num_steps)
    total_time = float(num_steps) * (float(exposure_time) + _transfer_time_per_image(context))
    return _known_estimate(total_time, num_steps)


def _estimate_scan_he3(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    num_steps = _as_int(params.get("num_steps"), None)
    if num_steps is None:
        start_pos = _as_float(params.get("start_pos"), None)
        stop_pos = _as_float(params.get("stop_pos"), None)
        step_size = _as_float(params.get("step_size", params.get("step")), None)
        if start_pos is None or stop_pos is None or step_size is None:
            return _unknown_estimate()
        num_steps = _num_steps_from_step_size_scan(start_pos, stop_pos, step_size)
        if num_steps is None or num_steps <= 0:
            return _unknown_estimate()
    else:
        if num_steps <= 0:
            return _unknown_estimate()
    acquire_time = _as_float(params.get("acquire_time"), None)
    if acquire_time is None:
        acquire_time = _as_float(context.get("diffraction_acquire_time_s"), None)
    if acquire_time is None:
        return _unknown_estimate(num_steps)
    return _known_estimate(float(num_steps) * float(acquire_time), num_steps)


def _estimate_count_he3(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    num = _as_int(params.get("num"), 1)
    if num is None or num <= 0:
        return _unknown_estimate()
    acquire_time = _as_float(params.get("acquire_time"), None)
    if acquire_time is None:
        acquire_time = _as_float(context.get("diffraction_acquire_time_s"), None)
    if acquire_time is None:
        return _unknown_estimate(num)
    return _known_estimate(float(num) * float(acquire_time), num)


def _estimate_scan_parallel_he3(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    num_steps = _as_int(params.get("num_steps"), None)
    if num_steps is None or num_steps <= 0:
        return _unknown_estimate()
    acquire_time = _as_float(params.get("acquire_time"), None)
    if acquire_time is None:
        acquire_time = _as_float(context.get("diffraction_acquire_time_s"), None)
    if acquire_time is None:
        return _unknown_estimate(num_steps)
    return _known_estimate(float(num_steps) * float(acquire_time), num_steps)


def _estimate_scan_list_he3(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    position_list = params.get("position_list", ())
    try:
        num_steps = len(position_list)
    except Exception:
        return _unknown_estimate()
    acquire_time = _as_float(params.get("acquire_time"), None)
    if acquire_time is None:
        acquire_time = _as_float(context.get("diffraction_acquire_time_s"), None)
    if acquire_time is None:
        return _unknown_estimate(num_steps)
    return _known_estimate(float(num_steps) * float(acquire_time), num_steps)


def _estimate_scan2d_he3(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    start_outer = _as_float(params.get("start_pos_outer"), None)
    stop_outer = _as_float(params.get("stop_pos_outer"), None)
    step_size_outer = _as_float(params.get("step_size_outer", params.get("step_outer")), None)
    start_inner = _as_float(params.get("start_pos_inner"), None)
    stop_inner = _as_float(params.get("stop_pos_inner"), None)
    step_size_inner = _as_float(params.get("step_size_inner", params.get("step_inner")), None)
    if None in (start_outer, stop_outer, step_size_outer, start_inner, stop_inner, step_size_inner):
        return _unknown_estimate()
    num_steps_outer = _num_steps_from_step_size_scan(float(start_outer), float(stop_outer), float(step_size_outer))
    num_steps_inner = _num_steps_from_step_size_scan(float(start_inner), float(stop_inner), float(step_size_inner))
    if (
        num_steps_outer is None
        or num_steps_inner is None
        or num_steps_outer <= 0
        or num_steps_inner <= 0
    ):
        return _unknown_estimate()
    total_steps = int(num_steps_outer) * int(num_steps_inner)
    acquire_time = _as_float(params.get("acquire_time"), None)
    if acquire_time is None:
        acquire_time = _as_float(context.get("diffraction_acquire_time_s"), None)
    if acquire_time is None:
        return _unknown_estimate(total_steps)
    return _known_estimate(float(total_steps) * float(acquire_time), total_steps)


def _estimate_wait_seconds(params: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, float | int | None]:
    seconds = _as_float(params.get("seconds"), None)
    if seconds is None:
        return _unknown_estimate(1)
    return _known_estimate(max(0.0, float(seconds)), 1)


_ESTIMATORS = {
    "count_he3": _estimate_count_he3,
    "imaging": _estimate_imaging,
    "imaging_scan": _estimate_imaging_scan,
    "tomo_scan": _estimate_tomo_scan,
    "adaptive_imaging_focus_scan": _estimate_adaptive_imaging_focus_scan,
    "scan_he3": _estimate_scan_he3,
    "scan_parallel_he3": _estimate_scan_parallel_he3,
    "scan_list_he3": _estimate_scan_list_he3,
    "scan2D_he3": _estimate_scan2d_he3,
    "wait_seconds": _estimate_wait_seconds,
}


def estimate_plan_runtime(
    plan_name: str,
    *,
    kwargs: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, float | int | None]:
    estimator = _ESTIMATORS.get(str(plan_name))
    if estimator is None:
        return _unknown_estimate()
    params = dict(kwargs or {})
    try:
        result = estimator(params, context or {})
    except Exception:
        result = _unknown_estimate()
    result.setdefault("estimated_total_time_s", None)
    result.setdefault("estimated_total_units", None)
    return result


def format_estimated_time(seconds: float | None) -> str:
    value = _as_float(seconds, None)
    if value is None:
        return "--"
    if value < 60.0:
        return f"{value:.1f}s"
    if value < 3600.0:
        return f"{value / 60.0:.1f}m"
    return f"{value / 3600.0:.1f}h"
