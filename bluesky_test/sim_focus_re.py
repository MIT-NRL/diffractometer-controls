"""
Prototype Bluesky focus simulation for Queue Server integration testing.

This file is intentionally outside startup config so it can be iterated safely.
When validated, move relevant pieces into bluesky_config/startup.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
import numpy as np
from bluesky import plan_patterns, utils
from bluesky.run_engine import RunEngine
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.positioner import PositionerBase
from ophyd.sim import SynAxis
from ophyd.status import Status

try:
    from bluesky_queueserver import parameter_annotation_decorator
except Exception:
    # Keep this prototype runnable even without Queue Server installed.
    def parameter_annotation_decorator(_schema):
        def _decorator(func):
            return func

        return _decorator


def _write_image(path: Path, image: np.ndarray):
    """Write image data with backend fallbacks."""
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import tifffile

        tifffile.imwrite(str(path), image)
        return
    except Exception:
        pass

    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), image)
        return
    except Exception:
        pass

    try:
        from PIL import Image

        Image.fromarray(image).save(str(path))
        return
    except Exception as ex:
        raise RuntimeError(f"Failed to write simulated image '{path}': {ex}") from ex


def _local_quadratic_extremum(x_vals, y_vals, *, mode="min", window_points=7):
    """Estimate local extremum from a small neighborhood fit."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(finite) < 3:
        return float("nan")

    x_arr = x_arr[finite]
    y_arr = y_arr[finite]
    anchor = int(np.argmax(y_arr)) if mode == "max" else int(np.argmin(y_arr))

    n = int(x_arr.size)
    win = int(max(3, window_points))
    if win % 2 == 0:
        win += 1
    win = min(win, n)

    left = max(0, anchor - win // 2)
    right = min(n, left + win)
    left = max(0, right - win)

    x_fit = x_arr[left:right]
    y_fit = y_arr[left:right]
    if x_fit.size < 3 or np.ptp(x_fit) <= 0:
        return float(x_arr[anchor])

    a, b, _c = np.polyfit(x_fit, y_fit, 2)
    if abs(a) <= 1e-12:
        return float(x_arr[anchor])
    if (mode == "min" and a <= 0) or (mode == "max" and a >= 0):
        return float(x_arr[anchor])

    x_star = float(-b / (2.0 * a))
    return float(np.clip(x_star, np.min(x_fit), np.max(x_fit)))


class SimFocusDetector(Device):
    """
    Synthetic detector that writes one slanted-edge image per trigger.

    Signals expose both image file path and simple focus metrics.
    """

    frame_index = Cpt(Signal, value=-1, kind="normal")
    image_path = Cpt(Signal, value="", kind="normal")
    focus_motor = Cpt(Signal, value=np.nan, kind="normal")
    lsf_sigma = Cpt(Signal, value=np.nan, kind="hinted")
    mtf50 = Cpt(Signal, value=np.nan, kind="hinted")

    def __init__(
        self,
        *args,
        motor,
        output_root="./sim_focus_data",
        image_shape=(4500, 4500),
        motor_optimum=0.0,
        sigma_min=2.0,
        sigma_curve=0.9,
        sigma_jitter=0.0,
        edge_angle_deg=7.5,
        edge_offset_px=0.0,
        dark_level=900.0,
        bright_level=48000.0,
        gradient_amp=1500.0,
        noise_std=0.0,
        random_seed=7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._motor = motor
        self._output_root = Path(output_root)
        self._image_shape = tuple(int(v) for v in image_shape)
        self._motor_optimum = float(motor_optimum)
        self._sigma_min = float(sigma_min)
        self._sigma_curve = float(sigma_curve)
        self._sigma_jitter = float(sigma_jitter)
        self._edge_angle_deg = float(edge_angle_deg)
        self._edge_offset_px = float(edge_offset_px)
        self._dark_level = float(dark_level)
        self._bright_level = float(bright_level)
        self._gradient_amp = float(gradient_amp)
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(int(random_seed))
        self._frame_counter = 0
        self._run_dir = None
        self._file_stem = self.name

    def configure_output(self, *, output_dir=None, file_stem=None, reset_counter=True):
        if output_dir:
            self._output_root = Path(str(output_dir))
        if file_stem:
            self._file_stem = str(file_stem).strip().replace(" ", "_")
        if reset_counter:
            self._frame_counter = 0

    def stage(self):
        if self._run_dir is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            self._run_dir = self._output_root / f"{self._file_stem}_{stamp}"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        return super().stage()

    def unstage(self):
        self._run_dir = None
        return super().unstage()

    def _compute_sigma(self, motor_pos):
        sigma = self._sigma_min + self._sigma_curve * (float(motor_pos) - self._motor_optimum) ** 2
        sigma += float(self._rng.normal(0.0, self._sigma_jitter))
        return max(0.4, float(sigma))

    def _generate_image(self, sigma):
        h, w = self._image_shape
        yy, xx = np.indices((h, w), dtype=np.float64)

        x_center = 0.5 * w + self._edge_offset_px
        y_center = 0.5 * h
        slope = math.tan(math.radians(self._edge_angle_deg))
        edge_x = x_center + slope * (yy - y_center)

        profile = 0.5 * (1.0 + np.tanh((xx - edge_x) / max(0.2, sigma)))
        image = self._dark_level + (self._bright_level - self._dark_level) * profile

        if self._gradient_amp != 0:
            image = image + self._gradient_amp * ((yy / max(1.0, h - 1.0)) - 0.5)
        if self._noise_std > 0:
            image = image + self._rng.normal(0.0, self._noise_std, size=image.shape)

        np.clip(image, 0, 65535, out=image)
        return image.astype(np.uint16, copy=False)

    def trigger(self):
        st = Status()
        try:
            motor_pos = float(self._motor.position)
            sigma = self._compute_sigma(motor_pos)
            mtf50 = float(0.44 / max(1e-6, sigma))

            image = self._generate_image(sigma=sigma)
            if self._run_dir is None:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                self._run_dir = self._output_root / f"{self._file_stem}_{stamp}"
                self._run_dir.mkdir(parents=True, exist_ok=True)

            path = self._run_dir / f"{self._file_stem}_{self._frame_counter:04d}.tif"
            _write_image(path, image)

            self.frame_index.put(int(self._frame_counter))
            self.image_path.put(str(path))
            self.focus_motor.put(motor_pos)
            self.lsf_sigma.put(sigma)
            self.mtf50.put(mtf50)
            self._frame_counter += 1
        except Exception as ex:
            st.set_exception(ex)
        else:
            st.set_finished()
        return st


def _collect_tiff_files(source_dir: Path, *, file_glob: str = "*.tif", recursive: bool = True):
    src = Path(source_dir).expanduser()
    if recursive:
        paths = list(src.rglob(file_glob))
    else:
        paths = list(src.glob(file_glob))
    paths = [p.resolve() for p in paths if p.is_file()]
    paths.sort(key=lambda p: str(p).lower())
    return paths


class ReplayFocusDetector(Device):
    """
    Replay detector that emits existing TIFF file paths in a repeating cycle.

    It does not write new images; each trigger advances to the next file path.
    """

    frame_index = Cpt(Signal, value=-1, kind="normal")
    image_path = Cpt(Signal, value="", kind="normal")
    focus_motor = Cpt(Signal, value=np.nan, kind="normal")
    lsf_sigma = Cpt(Signal, value=np.nan, kind="hinted")
    mtf50 = Cpt(Signal, value=np.nan, kind="hinted")

    def __init__(
        self,
        *args,
        motor,
        source_dir="./sim_focus_data",
        file_glob="*.tif",
        recursive=True,
        motor_optimum=0.0,
        sigma_min=2.0,
        sigma_curve=0.9,
        sigma_jitter=0.0,
        random_seed=13,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._motor = motor
        self._source_dir = Path(source_dir)
        self._file_glob = str(file_glob)
        self._recursive = bool(recursive)
        self._source_files = []
        self._frame_counter = 0
        self._motor_optimum = float(motor_optimum)
        self._sigma_min = float(sigma_min)
        self._sigma_curve = float(sigma_curve)
        self._sigma_jitter = float(sigma_jitter)
        self._rng = np.random.default_rng(int(random_seed))

    def configure_source(
        self,
        *,
        source_dir=None,
        file_glob=None,
        recursive=None,
        reset_counter=False,
    ):
        if source_dir is not None:
            self._source_dir = Path(str(source_dir))
        if file_glob is not None:
            self._file_glob = str(file_glob)
        if recursive is not None:
            self._recursive = bool(recursive)
        if reset_counter:
            self._frame_counter = 0

    def _refresh_source_files(self):
        self._source_files = _collect_tiff_files(
            self._source_dir,
            file_glob=self._file_glob,
            recursive=self._recursive,
        )
        return self._source_files

    def stage(self):
        files = self._refresh_source_files()
        if not files:
            src = str(self._source_dir.expanduser())
            raise RuntimeError(
                f"ReplayFocusDetector found no TIFF files in '{src}' (glob='{self._file_glob}', recursive={self._recursive})."
            )
        return super().stage()

    def _compute_sigma(self, motor_pos):
        sigma = self._sigma_min + self._sigma_curve * (float(motor_pos) - self._motor_optimum) ** 2
        sigma += float(self._rng.normal(0.0, self._sigma_jitter))
        return max(0.4, float(sigma))

    def trigger(self):
        st = Status()
        try:
            if not self._source_files:
                self._refresh_source_files()
            if not self._source_files:
                raise RuntimeError("ReplayFocusDetector has no source TIFF files to emit.")

            src_idx = int(self._frame_counter % len(self._source_files))
            path = self._source_files[src_idx]
            motor_pos = float(self._motor.position)
            sigma = self._compute_sigma(motor_pos)
            mtf50 = float(0.44 / max(1e-6, sigma))

            self.frame_index.put(int(self._frame_counter))
            self.image_path.put(str(path))
            self.focus_motor.put(motor_pos)
            self.lsf_sigma.put(sigma)
            self.mtf50.put(mtf50)
            self._frame_counter += 1
        except Exception as ex:
            st.set_exception(ex)
        else:
            st.set_finished()
        return st


focus_sim_motor = SynAxis(name="focus_sim_motor")
focus_sim_detector = SimFocusDetector(
    name="focus_sim_detector",
    motor=focus_sim_motor,
)
focus_replay_detector = ReplayFocusDetector(
    name="focus_replay_detector",
    motor=focus_sim_motor,
)


def _collect_sim_focus_motor_names():
    names = []
    for var, obj in globals().items():
        if var.startswith("_"):
            continue
        try:
            if isinstance(obj, PositionerBase):
                names.append(var)
        except Exception:
            continue
    return list(dict.fromkeys(names))


def _collect_sim_focus_detector_names():
    names = []
    for var, obj in globals().items():
        if var.startswith("_"):
            continue
        try:
            if isinstance(obj, (SimFocusDetector, ReplayFocusDetector)):
                names.append(var)
        except Exception:
            continue
    return list(dict.fromkeys(names))


@parameter_annotation_decorator(
    {
        "parameters": {
            "motor": {
                "annotation": "typing.Union[str, FocusMotors]",
                "description": "Motor to scan for focus simulation",
                "devices": {"FocusMotors": _collect_sim_focus_motor_names()},
                "convert_device_names": True,
            },
            "detector": {
                "annotation": "typing.Union[str, FocusDetectors]",
                "description": "Simulated focus detector",
                "devices": {"FocusDetectors": _collect_sim_focus_detector_names()},
                "convert_device_names": True,
            },
        }
    }
)
def sim_focus_scan(
    start_pos=-3.0,
    stop_pos=3.0,
    num_steps=15,
    exposure_time_s=0.5,
    motor=focus_sim_motor,
    detector=focus_sim_detector,
    output_dir="./sim_focus_data",
    file_stem="FocusScanSim",
    local_fit_points=7,
    focus_target="lsf_sigma",
    move_to_best=False,
    md=None,
):
    """
    Synthetic focus scan that emits file paths and focus metrics.

    Primary stream includes:
      - motor position
      - detector.image_path
      - detector.lsf_sigma
      - detector.mtf50

    Secondary stream "focus_summary" includes:
      - best_focus_lsf (local quadratic minimum)
      - best_focus_mtf50 (local quadratic maximum)
    """
    num_steps = int(max(3, num_steps))
    exposure_time_s = float(max(0.0, exposure_time_s))
    positions = np.linspace(float(start_pos), float(stop_pos), num=num_steps, endpoint=True)

    output_dir = str(output_dir).strip()
    file_stem = str(file_stem).strip().replace(" ", "_")

    md = md or {}
    _md = {
        "plan_name": "sim_focus_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": {
            "motor": getattr(motor, "name", "motor"),
            "start_pos": float(start_pos),
            "stop_pos": float(stop_pos),
            "num_steps": int(num_steps),
        },
        "estimated_total_units": int(num_steps),
        "estimated_total_time_s": float(num_steps * exposure_time_s),
        "focus_simulation": {
            "output_dir": output_dir,
            "file_stem": file_stem,
            "focus_target": str(focus_target),
            "local_fit_points": int(local_fit_points),
            "exposure_time_s": float(exposure_time_s),
        },
    }
    _md.update(md)

    x_fields = []
    x_fields.extend(utils.get_hinted_fields(motor))
    _md["hints"] = {"dimensions": [(x_fields, "primary")]} if x_fields else {}
    _md["hints"].update(md.get("hints", {}) or {})

    xs = []
    sigma_vals = []
    mtf_vals = []

    @bpp.run_decorator(md=_md)
    def _main():
        detector.configure_output(output_dir=output_dir, file_stem=file_stem, reset_counter=True)
        yield from bps.stage(detector)
        yield from bps.stage(motor)
        try:
            for pos in positions:
                yield from bps.checkpoint()
                yield from bps.mv(motor, float(pos))
                if exposure_time_s > 0:
                    yield from bps.sleep(exposure_time_s)
                yield from bps.trigger_and_read([detector, motor])
                xs.append(float(pos))
                sigma_vals.append(float(detector.lsf_sigma.get()))
                mtf_vals.append(float(detector.mtf50.get()))
        finally:
            yield from bps.unstage(detector)

        best_focus_lsf = _local_quadratic_extremum(
            xs,
            sigma_vals,
            mode="min",
            window_points=int(local_fit_points),
        )
        best_focus_mtf = _local_quadratic_extremum(
            xs,
            mtf_vals,
            mode="max",
            window_points=int(local_fit_points),
        )

        sig_best_lsf = Signal(name="best_focus_lsf", value=float(best_focus_lsf))
        sig_best_mtf = Signal(name="best_focus_mtf50", value=float(best_focus_mtf))
        sig_fit_pts = Signal(name="fit_points", value=int(local_fit_points))

        yield from bps.create(name="focus_summary")
        yield from bps.read(sig_best_lsf)
        yield from bps.read(sig_best_mtf)
        yield from bps.read(sig_fit_pts)
        yield from bps.save()

        target = str(focus_target).strip().lower()
        if move_to_best:
            best = float(best_focus_mtf) if target == "mtf50" else float(best_focus_lsf)
            if math.isfinite(best):
                yield from bps.mv(motor, best)

    return (yield from _main())


@parameter_annotation_decorator(
    {
        "parameters": {
            "motor": {
                "annotation": "typing.Union[str, FocusMotors]",
                "description": "Motor to scan for focus replay",
                "devices": {"FocusMotors": _collect_sim_focus_motor_names()},
                "convert_device_names": True,
            },
            "detector": {
                "annotation": "typing.Union[str, FocusDetectors]",
                "description": "Replay detector (reads existing TIFF file paths)",
                "devices": {"FocusDetectors": _collect_sim_focus_detector_names()},
                "convert_device_names": True,
            },
        }
    }
)
def replay_focus_scan(
    *,
    source_dir,
    start_pos=-3.0,
    stop_pos=3.0,
    num_steps=15,
    exposure_time_s=2.0,
    file_glob="*.tif",
    recursive=True,
    motor=focus_sim_motor,
    detector=focus_replay_detector,
    local_fit_points=7,
    focus_target="lsf_sigma",
    move_to_best=False,
    md=None,
):
    """
    Replay focus scan that cycles through existing TIFF paths.

    On each trigger, detector.image_path points to the next file in source_dir.
    If num_steps exceeds number of files, files repeat from the beginning.
    """
    num_steps = int(max(3, num_steps))
    exposure_time_s = float(max(0.0, exposure_time_s))
    positions = np.linspace(float(start_pos), float(stop_pos), num=num_steps, endpoint=True)
    src = Path(str(source_dir)).expanduser().resolve()
    detector.configure_source(
        source_dir=str(src),
        file_glob=str(file_glob),
        recursive=bool(recursive),
        reset_counter=True,
    )

    md = md or {}
    _md = {
        "plan_name": "replay_focus_scan",
        "plan_pattern": "inner_product",
        "plan_pattern_module": plan_patterns.__name__,
        "plan_pattern_args": {
            "motor": getattr(motor, "name", "motor"),
            "start_pos": float(start_pos),
            "stop_pos": float(stop_pos),
            "num_steps": int(num_steps),
        },
        "estimated_total_units": int(num_steps),
        "estimated_total_time_s": float(num_steps * exposure_time_s),
        "focus_replay": {
            "source_dir": str(src),
            "file_glob": str(file_glob),
            "recursive": bool(recursive),
            "focus_target": str(focus_target),
            "local_fit_points": int(local_fit_points),
            "exposure_time_s": float(exposure_time_s),
        },
    }
    _md.update(md)

    x_fields = []
    x_fields.extend(utils.get_hinted_fields(motor))
    _md["hints"] = {"dimensions": [(x_fields, "primary")]} if x_fields else {}
    _md["hints"].update(md.get("hints", {}) or {})

    xs = []
    sigma_vals = []
    mtf_vals = []

    @bpp.run_decorator(md=_md)
    def _main():
        yield from bps.stage(detector)
        yield from bps.stage(motor)
        try:
            for pos in positions:
                yield from bps.checkpoint()
                yield from bps.mv(motor, float(pos))
                if exposure_time_s > 0:
                    yield from bps.sleep(exposure_time_s)
                yield from bps.trigger_and_read([detector, motor])
                xs.append(float(pos))
                sigma_vals.append(float(detector.lsf_sigma.get()))
                mtf_vals.append(float(detector.mtf50.get()))
        finally:
            yield from bps.unstage(detector)

        best_focus_lsf = _local_quadratic_extremum(
            xs,
            sigma_vals,
            mode="min",
            window_points=int(local_fit_points),
        )
        best_focus_mtf = _local_quadratic_extremum(
            xs,
            mtf_vals,
            mode="max",
            window_points=int(local_fit_points),
        )

        sig_best_lsf = Signal(name="best_focus_lsf", value=float(best_focus_lsf))
        sig_best_mtf = Signal(name="best_focus_mtf50", value=float(best_focus_mtf))
        sig_fit_pts = Signal(name="fit_points", value=int(local_fit_points))

        yield from bps.create(name="focus_summary")
        yield from bps.read(sig_best_lsf)
        yield from bps.read(sig_best_mtf)
        yield from bps.read(sig_fit_pts)
        yield from bps.save()

        target = str(focus_target).strip().lower()
        if move_to_best:
            best = float(best_focus_mtf) if target == "mtf50" else float(best_focus_lsf)
            if math.isfinite(best):
                yield from bps.mv(motor, best)

    return (yield from _main())


class FocusRecommendationAgent:
    """Stream consumer that estimates best focus from Bluesky event documents."""

    def __init__(
        self,
        *,
        stream_name="primary",
        motor_key="focus_sim_motor",
        sigma_key="focus_sim_detector_lsf_sigma",
        mtf50_key="focus_sim_detector_mtf50",
    ):
        self.stream_name = str(stream_name)
        self.motor_key = str(motor_key)
        self.sigma_key = str(sigma_key)
        self.mtf50_key = str(mtf50_key)

        self._active_run_uid: Optional[str] = None
        self._descriptor_stream: Dict[str, str] = {}
        self._run_points: Dict[str, Dict[str, list]] = {}

    @staticmethod
    def _to_float_or_nan(value):
        try:
            return float(value)
        except Exception:
            return float("nan")

    def on_document(self, name, doc):
        if name == "start":
            uid = str(doc.get("uid", ""))
            self._active_run_uid = uid
            self._descriptor_stream.clear()
            self._run_points[uid] = {"x": [], "sigma": [], "mtf50": []}
            return
        if name == "descriptor":
            self._descriptor_stream[str(doc.get("uid", ""))] = str(doc.get("name", ""))
            return
        if name != "event":
            return

        descriptor_uid = str(doc.get("descriptor", ""))
        stream = self._descriptor_stream.get(descriptor_uid, "")
        if self.stream_name and stream != self.stream_name:
            return
        uid = self._active_run_uid
        if uid is None or uid not in self._run_points:
            return

        data = doc.get("data", {}) or {}
        x = self._to_float_or_nan(data.get(self.motor_key))
        sigma = self._to_float_or_nan(data.get(self.sigma_key))
        mtf50 = self._to_float_or_nan(data.get(self.mtf50_key))
        if not np.isfinite(x):
            return
        self._run_points[uid]["x"].append(x)
        self._run_points[uid]["sigma"].append(sigma)
        self._run_points[uid]["mtf50"].append(mtf50)

    def _resolve_uid(self, run_uid: Optional[str]) -> Optional[str]:
        if run_uid:
            return str(run_uid)
        if not self._run_points:
            return None
        return list(self._run_points.keys())[-1]

    def recommendation(self, *, metric="lsf_sigma", run_uid=None, window_points=7) -> float:
        uid = self._resolve_uid(run_uid)
        if uid is None or uid not in self._run_points:
            return float("nan")
        points = self._run_points[uid]
        x = points["x"]
        metric_l = str(metric).strip().lower()
        if metric_l == "mtf50":
            return _local_quadratic_extremum(
                x,
                points["mtf50"],
                mode="max",
                window_points=int(window_points),
            )
        return _local_quadratic_extremum(
            x,
            points["sigma"],
            mode="min",
            window_points=int(window_points),
        )

    def summary(self, *, run_uid=None, window_points=7):
        uid = self._resolve_uid(run_uid)
        if uid is None or uid not in self._run_points:
            return {
                "run_uid": None,
                "n_points": 0,
                "x_min": np.nan,
                "x_max": np.nan,
                "best_lsf": np.nan,
                "best_mtf50": np.nan,
            }
        points = self._run_points[uid]
        x_vals = np.asarray(points["x"], dtype=float)
        finite_x = x_vals[np.isfinite(x_vals)]
        return {
            "run_uid": uid,
            "n_points": int(len(points["x"])),
            "x_min": float(np.nanmin(finite_x)) if finite_x.size else np.nan,
            "x_max": float(np.nanmax(finite_x)) if finite_x.size else np.nan,
            "best_lsf": float(
                self.recommendation(
                    metric="lsf_sigma", run_uid=uid, window_points=window_points
                )
            ),
            "best_mtf50": float(
                self.recommendation(metric="mtf50", run_uid=uid, window_points=window_points)
            ),
        }


def build_test_re():
    """Create a minimal local RunEngine for quick testing."""
    return RunEngine({})


def run_demo():
    """Run one local demo scan and print where images were written."""
    re = build_test_re()
    re(sim_focus_scan(start_pos=-2, stop_pos=2, num_steps=11, move_to_best=True))
    print(f"Demo complete. Last image: {focus_sim_detector.image_path.get()}")


def run_replay_demo(
    *,
    source_dir,
    start_pos=-3.0,
    stop_pos=3.0,
    num_steps=None,
    exposure_time_s=2.0,
    file_glob="*.tif",
    recursive=True,
):
    """
    Non-interactive replay demo:
      - scans motor positions
      - emits existing TIFF paths from source_dir in a repeating cycle
    """
    re = build_test_re()
    src = Path(str(source_dir)).expanduser().resolve()
    files = _collect_tiff_files(src, file_glob=str(file_glob), recursive=bool(recursive))
    if num_steps is None:
        num_steps = len(files)
    num_steps = int(max(1, int(num_steps)))
    uid_t = re(
        replay_focus_scan(
            source_dir=source_dir,
            start_pos=float(start_pos),
            stop_pos=float(stop_pos),
            num_steps=int(num_steps),
            exposure_time_s=float(exposure_time_s),
            file_glob=str(file_glob),
            recursive=bool(recursive),
            detector=focus_replay_detector,
            motor=focus_sim_motor,
            move_to_best=False,
        )
    )
    uid = uid_t[0] if uid_t else None
    print(
        f"Replay demo complete. run_uid={uid} source='{src}' "
        f"files={len(files)} steps={int(num_steps)} last_path={focus_replay_detector.image_path.get()}"
    )
    return {"run_uid": uid, "source_dir": str(src), "n_files": len(files), "n_steps": int(num_steps)}


def run_replay_online_demo(
    *,
    source_dir,
    start_pos=-3.0,
    stop_pos=3.0,
    num_steps=None,
    exposure_time_s=2.0,
    file_glob="*.tif",
    recursive=True,
):
    """
    Noninteractive replay scan with online viewer attached.

    The viewer pops up for ROI/plot interaction, but the Bluesky plan itself
    is a finite replay scan (no adaptive/wait-for-button loop).
    """
    re = build_test_re()
    src = Path(str(source_dir)).expanduser().resolve()
    files = _collect_tiff_files(src, file_glob=str(file_glob), recursive=bool(recursive))
    if not files:
        raise RuntimeError(
            f"No TIFF files found in '{src}' (glob='{file_glob}', recursive={bool(recursive)})."
        )
    if num_steps is None:
        num_steps = len(files)
    num_steps = int(max(1, int(num_steps)))
    try:
        from diffractometer_controls.focus_online_viewer import attach_to_run_engine
    except Exception:
        from focus_online_viewer import attach_to_run_engine
    bridge, viewer_token = attach_to_run_engine(
        re,
        image_key="focus_replay_detector_image_path",
        motor_key="focus_sim_motor",
        stream_name="primary",
        follow_latest=True,
        reset_viewer_on_new_run=False,
        interval_ms=200,
        max_workers_total=8,
        bulk_workers=1,
        full_workers=6,
    )
    uid_t = re(
        replay_focus_scan(
            source_dir=str(src),
            start_pos=float(start_pos),
            stop_pos=float(stop_pos),
            num_steps=int(num_steps),
            exposure_time_s=float(exposure_time_s),
            file_glob=str(file_glob),
            recursive=bool(recursive),
            detector=focus_replay_detector,
            motor=focus_sim_motor,
            move_to_best=False,
            md={"replay_mode": "online_viewer_noninteractive"},
        )
    )
    uid = uid_t[0] if uid_t else None
    print(
        f"Replay online demo complete. run_uid={uid} source='{src}' "
        f"files={len(files)} steps={int(num_steps)} viewer_token={viewer_token}"
    )
    return {
        "re": re,
        "bridge": bridge,
        "viewer_token": viewer_token,
        "run_uid": uid,
        "source_dir": str(src),
        "n_files": len(files),
        "n_steps": int(num_steps),
    }


def run_demo_online():
    """
    Run a local focus session with online viewer controls.

    Viewer controls:
      - Focus metric selector (default mtf50)
      - Go to Focus: move + one image at latest recommendation
      - Scan Around Focus: local fine scan near recommendation
      - Extend Left +3 / Extend Right +3: append points beyond current limits
        using the initial coarse step size
      - Complete: marks session complete and disables those controls
    """
    re = build_test_re()
    agent = FocusRecommendationAgent()
    agent_token = int(re.subscribe(agent.on_document))
    try:
        from qtpy import QtCore, QtWidgets
    except Exception as ex:
        raise RuntimeError(f"Qt is required for run_demo_online: {ex}") from ex
    try:
        from diffractometer_controls.focus_online_viewer import attach_to_run_engine
    except Exception:
        from focus_online_viewer import attach_to_run_engine

    session = {"active": True}
    pending_plans = deque()
    runner_state = {"running": False}
    command_queue = deque()
    command_lock = threading.Lock()
    app = QtWidgets.QApplication.instance()
    coarse_start = float(-3.0)
    coarse_stop = float(3.0)
    coarse_steps = int(15)
    default_exp_s = float(0.5)
    coarse_step = abs((coarse_stop - coarse_start) / max(1, (coarse_steps - 1)))
    scan_points = int(7)
    default_scan_step = float(0.5 / max(1, (scan_points - 1) / 2.0))

    def _metric_norm(metric: str) -> str:
        m = str(metric).strip().lower()
        if m in ("mtf50", "lsf_sigma", "step_sigma"):
            return m
        return "mtf50"

    def _submit_plan(plan, *, label: str):
        if not bool(session["active"]):
            print(f"{label}: ignored (session complete).")
            return
        pending_plans.append((plan, str(label)))
        try:
            QtCore.QTimer.singleShot(0, _drain_plan_queue)
        except Exception:
            _drain_plan_queue()

    def _drain_plan_queue():
        if runner_state["running"]:
            return
        while pending_plans and bool(session["active"]):
            # Respect RE state in case a plan was queued while another was running.
            if str(getattr(re, "state", "idle")).lower() != "idle":
                return
            plan, label = pending_plans.popleft()
            runner_state["running"] = True
            try:
                uid_t = re(plan)
                run_uid = uid_t[0] if uid_t else "unknown"
                print(f"{label}: complete run_uid={run_uid}")
            except Exception as ex:
                print(f"{label}: failed: {ex}")
            finally:
                runner_state["running"] = False
        # Keep trying in case RE transitions to idle slightly after callback return.
        if pending_plans and bool(session["active"]):
            try:
                QtCore.QTimer.singleShot(50, _drain_plan_queue)
            except Exception:
                pass

    def _on_go_to_focus(metric: str):
        if not bool(session["active"]):
            print("Go to Focus ignored: session is marked complete.")
            return
        metric_n = _metric_norm(metric)
        print(f"Go to Focus requested: metric={metric_n}")
        with command_lock:
            command_queue.append(
                {
                    "action": "move",
                    "metric": metric_n,
                    "acquire_image": True,
                    "exposure_time_s": float(default_exp_s),
                }
            )

    def _on_scan_around_focus(metric: str, step_size: float = default_scan_step):
        if not bool(session["active"]):
            print("Scan Around Focus ignored: session is marked complete.")
            return
        metric_n = _metric_norm(metric)
        step_size = float(max(1e-4, step_size))
        half_range = float(step_size * max(1, (scan_points - 1) / 2.0))
        print(
            f"Scan Around Focus requested: metric={metric_n} "
            f"step={step_size:.5f} half_range={half_range:.5f}"
        )
        with command_lock:
            command_queue.append(
                {
                    "action": "scan",
                    "metric": metric_n,
                    "half_range": float(half_range),
                    "num_steps": int(scan_points),
                    "exposure_time_s": float(default_exp_s),
                }
            )

    def _on_extend_left():
        if not bool(session["active"]):
            print("Extend Left +3 ignored: session is marked complete.")
            return
        print(f"Extend Left +3 requested: step={coarse_step:.5f}")
        with command_lock:
            command_queue.append(
                {
                    "action": "extend_left",
                    "num_points": int(3),
                    "step_size": float(coarse_step),
                    "exposure_time_s": float(default_exp_s),
                }
            )

    def _on_extend_right():
        if not bool(session["active"]):
            print("Extend Right +3 ignored: session is marked complete.")
            return
        print(f"Extend Right +3 requested: step={coarse_step:.5f}")
        with command_lock:
            command_queue.append(
                {
                    "action": "extend_right",
                    "num_points": int(3),
                    "step_size": float(coarse_step),
                    "exposure_time_s": float(default_exp_s),
                }
            )

    def _on_mark_complete():
        with command_lock:
            command_queue.append({"action": "complete"})
        try:
            drain_timer.stop()
        except Exception:
            pass
        print("Focus session marked complete (queued).")

    bridge, token = attach_to_run_engine(
        re,
        image_key="focus_sim_detector_image_path",
        motor_key="focus_sim_motor",
        stream_name="primary",
        follow_latest=True,
        reset_viewer_on_new_run=False,
        on_go_to_focus=_on_go_to_focus,
        on_scan_around_focus=_on_scan_around_focus,
        on_extend_left=_on_extend_left,
        on_extend_right=_on_extend_right,
        on_mark_complete=_on_mark_complete,
        focus_metric_options=("mtf50", "lsf_sigma", "step_sigma"),
        default_focus_metric="mtf50",
        default_scan_step=float(default_scan_step),
        interval_ms=200,
        max_workers_total=8,
        bulk_workers=1,
        full_workers=6,
    )
    @bpp.run_decorator(
        md={
            "plan_name": "sim_focus_adaptive_session",
            "autofocus_action": "adaptive_session",
        }
    )
    def _adaptive_focus_session_plan():
        detector = focus_sim_detector
        motor = focus_sim_motor
        detector.configure_output(
            output_dir="./sim_focus_data",
            file_stem="FocusScanOnlineInitial",
            reset_counter=True,
        )
        yield from bps.stage(detector)
        yield from bps.stage(motor)
        try:
            # Initial auto-start coarse sweep.
            coarse_positions = np.linspace(
                coarse_start,
                coarse_stop,
                num=coarse_steps,
                endpoint=True,
            )
            for pos in coarse_positions:
                yield from bps.checkpoint()
                yield from bps.mv(motor, float(pos))
                yield from bps.sleep(default_exp_s)
                yield from bps.trigger_and_read([detector, motor])

            # Adaptive loop: wait for GUI commands and append points in the same run.
            while True:
                cmd = None
                with command_lock:
                    if command_queue:
                        cmd = command_queue.popleft()
                if cmd is None:
                    yield from bps.sleep(0.1)
                    continue

                action = str(cmd.get("action", "")).lower()
                if action == "complete":
                    session["active"] = False
                    break
                if action == "move":
                    metric_n = _metric_norm(str(cmd.get("metric", "mtf50")))
                    target = agent.recommendation(metric=metric_n, window_points=7)
                    if not np.isfinite(target):
                        target = float(motor.position)
                    current = float(motor.position)
                    print(
                        f"Executing Go to Focus: metric={metric_n} "
                        f"current={current:.5f} target={float(target):.5f}"
                    )
                    yield from bps.checkpoint()
                    yield from bps.mv(motor, float(target))
                    if bool(cmd.get("acquire_image", True)):
                        exp_s = float(max(0.0, cmd.get("exposure_time_s", 0.0)))
                        if exp_s > 0:
                            yield from bps.sleep(exp_s)
                        yield from bps.trigger_and_read([detector, motor])
                        print(
                            "Go to Focus acquired frame: "
                            f"frame_index={int(detector.frame_index.get())} "
                            f"motor={float(motor.position):.5f}"
                        )
                    print(f"Go to Focus complete: motor={float(motor.position):.5f}")
                    continue
                if action == "scan":
                    metric_n = _metric_norm(str(cmd.get("metric", "mtf50")))
                    center = agent.recommendation(metric=metric_n, window_points=7)
                    if not np.isfinite(center):
                        center = float(motor.position)
                    half_range = abs(float(cmd.get("half_range", 0.5)))
                    n_steps = int(max(3, cmd.get("num_steps", 7)))
                    exp_s = float(max(0.0, cmd.get("exposure_time_s", 0.5)))
                    print(
                        f"Executing Scan Around Focus: metric={metric_n} "
                        f"center={float(center):.5f} half_range={half_range:.5f} "
                        f"steps={n_steps}"
                    )
                    scan_positions = np.linspace(
                        center - half_range,
                        center + half_range,
                        num=n_steps,
                        endpoint=True,
                    )
                    for pos in scan_positions:
                        yield from bps.checkpoint()
                        yield from bps.mv(motor, float(pos))
                        if exp_s > 0:
                            yield from bps.sleep(exp_s)
                        yield from bps.trigger_and_read([detector, motor])
                    continue
                if action in ("extend_left", "extend_right"):
                    n_points = int(max(1, cmd.get("num_points", 3)))
                    step_size = abs(float(cmd.get("step_size", coarse_step)))
                    exp_s = float(max(0.0, cmd.get("exposure_time_s", default_exp_s)))
                    summary = agent.summary(window_points=7)
                    x_min = float(summary.get("x_min", np.nan))
                    x_max = float(summary.get("x_max", np.nan))
                    if not np.isfinite(x_min):
                        x_min = float(motor.position)
                    if not np.isfinite(x_max):
                        x_max = float(motor.position)
                    if action == "extend_left":
                        positions = [float(x_min - step_size * (k + 1)) for k in range(n_points)]
                        print(
                            f"Executing Extend Left +{n_points}: "
                            f"limit={x_min:.5f} step={step_size:.5f}"
                        )
                    else:
                        positions = [float(x_max + step_size * (k + 1)) for k in range(n_points)]
                        print(
                            f"Executing Extend Right +{n_points}: "
                            f"limit={x_max:.5f} step={step_size:.5f}"
                        )
                    for pos in positions:
                        yield from bps.checkpoint()
                        yield from bps.mv(motor, float(pos))
                        if exp_s > 0:
                            yield from bps.sleep(exp_s)
                        yield from bps.trigger_and_read([detector, motor])
                    continue
        finally:
            yield from bps.unstage(detector)
            yield from bps.unstage(motor)

    _submit_plan(_adaptive_focus_session_plan(), label="Adaptive focus session")
    drain_timer = QtCore.QTimer(app)
    drain_timer.setInterval(200)
    drain_timer.timeout.connect(_drain_plan_queue)
    drain_timer.start()
    print("Online adaptive focus session started. Initial scan queued/running. " f"viewer_token={token}")
    return {
        "re": re,
        "bridge": bridge,
        "viewer_token": token,
        "agent": agent,
        "agent_token": agent_token,
        "initial_uid": None,
        "session": session,
        "pending_plans": pending_plans,
        "runner_state": runner_state,
        "command_queue": command_queue,
        "drain_timer": drain_timer,
    }


def run_two_stage_autofocus_demo(
    *,
    coarse_start=-3.0,
    coarse_stop=3.0,
    coarse_steps=15,
    fine_half_range=0.8,
    fine_steps=9,
    exposure_time_s=0.5,
    local_fit_points=7,
    focus_target="lsf_sigma",
    output_dir="./sim_focus_data",
    file_stem="FocusAutoSim",
    use_online_viewer=True,
):
    """
    Two-stage autofocus demo:
      1) coarse scan -> move to coarse recommendation
      2) fine scan around coarse recommendation -> move to fine recommendation
      3) report recommendation error vs known simulator optimum
    """
    re = build_test_re()
    agent = FocusRecommendationAgent()
    agent_token = int(re.subscribe(agent.on_document))

    bridge = None
    viewer_token = None
    if use_online_viewer:
        try:
            from diffractometer_controls.focus_online_viewer import attach_to_run_engine
        except Exception:
            from focus_online_viewer import attach_to_run_engine
        bridge, viewer_token = attach_to_run_engine(
            re,
            image_key="focus_sim_detector_image_path",
            motor_key="focus_sim_motor",
            stream_name="primary",
            follow_latest=True,
            reset_viewer_on_new_run=False,
            interval_ms=200,
            max_workers_total=8,
            bulk_workers=1,
            full_workers=6,
        )

    coarse_uid_t = re(
        sim_focus_scan(
            start_pos=coarse_start,
            stop_pos=coarse_stop,
            num_steps=int(coarse_steps),
            exposure_time_s=float(exposure_time_s),
            local_fit_points=int(local_fit_points),
            focus_target=str(focus_target),
            move_to_best=True,
            output_dir=output_dir,
            file_stem=f"{file_stem}_coarse",
            md={"autofocus_stage": "coarse"},
        )
    )
    coarse_uid = coarse_uid_t[0] if coarse_uid_t else None
    coarse_center = float(focus_sim_motor.position)

    fine_start = float(coarse_center - abs(float(fine_half_range)))
    fine_stop = float(coarse_center + abs(float(fine_half_range)))
    fine_uid_t = re(
        sim_focus_scan(
            start_pos=fine_start,
            stop_pos=fine_stop,
            num_steps=int(fine_steps),
            exposure_time_s=float(exposure_time_s),
            local_fit_points=int(local_fit_points),
            focus_target=str(focus_target),
            move_to_best=True,
            output_dir=output_dir,
            file_stem=f"{file_stem}_fine",
            md={"autofocus_stage": "fine", "coarse_center": coarse_center},
        )
    )
    fine_uid = fine_uid_t[0] if fine_uid_t else None

    optimum = float(getattr(focus_sim_detector, "_motor_optimum", np.nan))
    coarse_summary = agent.summary(run_uid=coarse_uid, window_points=local_fit_points)
    fine_summary = agent.summary(run_uid=fine_uid, window_points=local_fit_points)
    coarse_best = (
        float(coarse_summary["best_mtf50"])
        if str(focus_target).strip().lower() == "mtf50"
        else float(coarse_summary["best_lsf"])
    )
    fine_best = (
        float(fine_summary["best_mtf50"])
        if str(focus_target).strip().lower() == "mtf50"
        else float(fine_summary["best_lsf"])
    )
    final_motor = float(focus_sim_motor.position)
    coarse_err = abs(coarse_best - optimum) if (np.isfinite(coarse_best) and np.isfinite(optimum)) else np.nan
    fine_err = abs(fine_best - optimum) if (np.isfinite(fine_best) and np.isfinite(optimum)) else np.nan
    final_err = abs(final_motor - optimum) if (np.isfinite(final_motor) and np.isfinite(optimum)) else np.nan

    print("Two-stage autofocus complete.")
    print(
        f"Coarse request range: [{float(coarse_start):.5f}, {float(coarse_stop):.5f}] "
        f"observed=[{float(coarse_summary.get('x_min', np.nan)):.5f}, {float(coarse_summary.get('x_max', np.nan)):.5f}]"
    )
    print(
        f"Fine request range:   [{fine_start:.5f}, {fine_stop:.5f}] "
        f"observed=[{float(fine_summary.get('x_min', np.nan)):.5f}, {float(fine_summary.get('x_max', np.nan)):.5f}]"
    )
    print(
        f"Coarse run={coarse_uid} center={coarse_center:.5f} "
        f"recommended={coarse_best:.5f} abs_err={coarse_err:.5f}"
    )
    print(
        f"Fine   run={fine_uid} final_motor={final_motor:.5f} "
        f"recommended={fine_best:.5f} abs_err={fine_err:.5f}"
    )
    print(f"Known simulator optimum={optimum:.5f} final_abs_err={final_err:.5f}")

    return {
        "re": re,
        "agent": agent,
        "agent_token": agent_token,
        "bridge": bridge,
        "viewer_token": viewer_token,
        "coarse_uid": coarse_uid,
        "fine_uid": fine_uid,
        "coarse_center": coarse_center,
        "coarse_best": coarse_best,
        "fine_best": fine_best,
        "optimum": optimum,
        "final_motor": final_motor,
        "coarse_error_abs": coarse_err,
        "fine_error_abs": fine_err,
        "final_error_abs": final_err,
    }


if __name__ == "__main__":
    print("sim_focus_re loaded. No demo is auto-run.")
    print(
        "Use run_demo_online(), run_replay_demo(...), run_replay_online_demo(...), "
        "or run_two_stage_autofocus_demo() from IPython."
    )
