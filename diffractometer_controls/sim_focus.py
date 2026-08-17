"""Ophyd simulation devices for end-to-end adaptive focus testing."""

from __future__ import annotations

import os
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from ophyd import Component as Cpt
from ophyd import Device, Signal
from ophyd.sim import SynAxis
from ophyd.status import DeviceStatus
from scipy.special import erf


DEFAULT_SIM_FOCUS_DATA_ROOT = "/home/mitr_4dh4/Data/TestData/Focus"
DEFAULT_SIM_FOCUS_BLUR_PER_DISTANCE_SQUARED = 4.0


class SimulatedFocusMotor(SynAxis):
    """Soft Ophyd motor used as the focus coordinate for the simulated camera."""

    def __init__(self, *, name: str = "sim_focus_motor", value: float = 0.0, delay: float = 0.02):
        super().__init__(
            name=name,
            value=float(value),
            delay=float(max(0.0, delay)),
            labels={"positioner", "simulated"},
            egu="mm",
        )


class SimulatedFocusCamera(Device):
    """Small camera-like configuration surface used by imaging plans."""

    acquire = Cpt(Signal, value=0, kind="omitted")
    acquire_time = Cpt(Signal, value=0.05, kind="config")
    gain = Cpt(Signal, value=1.0, kind="config")
    offset = Cpt(Signal, value=0.0, kind="config")
    array_counter = Cpt(Signal, value=0, kind="config")


class SimulatedFocusTIFF(Device):
    """Plan-compatible file naming controls matching ``cam1.tiff1``."""

    file_name = Cpt(Signal, value="sim_focus", kind="config")
    folder_name = Cpt(Signal, value="adaptive_focus_sim", kind="config")


class SimulatedFocusStats(Device):
    total = Cpt(Signal, value=0.0, kind="normal")


class SimulatedFocusDetector(Device):
    """File-writing slanted-edge detector coupled to an Ophyd motor.

    Each trigger creates a uint16 TIFF. The edge's Gaussian blur is smallest at
    ``best_focus`` and grows quadratically with distance from that position.
    The path-valued ``image_path`` signal is emitted in the primary Bluesky
    event, which is the same interface consumed by the online focus viewer.
    """

    cam = Cpt(SimulatedFocusCamera, "")
    tiff1 = Cpt(SimulatedFocusTIFF, "")
    stats1 = Cpt(SimulatedFocusStats, "")
    image_path = Cpt(Signal, value="", kind="hinted")
    blur_sigma = Cpt(Signal, value=np.nan, kind="normal")

    def __init__(
        self,
        *,
        motor: SimulatedFocusMotor,
        name: str = "sim_focus_cam",
        data_root: Optional[os.PathLike | str] = None,
        image_shape: Sequence[int] = (512, 512),
        best_focus: float = 0.0,
        minimum_blur_sigma: float = 1.2,
        blur_per_distance_squared: float = DEFAULT_SIM_FOCUS_BLUR_PER_DISTANCE_SQUARED,
        edge_angle_degrees: float = 5.0,
        dark_level: float = 3500.0,
        bright_level: float = 48000.0,
        read_noise: float = 35.0,
        random_seed: int = 2026,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        image_shape = tuple(image_shape)
        if len(image_shape) != 2:
            raise ValueError("image_shape must contain exactly two dimensions")
        height, width = (int(image_shape[0]), int(image_shape[1]))
        if height < 32 or width < 32:
            raise ValueError("simulated focus images must be at least 32x32 pixels")
        if minimum_blur_sigma <= 0 or blur_per_distance_squared < 0:
            raise ValueError("blur parameters must be positive")

        self.focus_motor = motor
        self.image_shape = (height, width)
        self.best_focus = float(best_focus)
        self.minimum_blur_sigma = float(minimum_blur_sigma)
        self.blur_per_distance_squared = float(blur_per_distance_squared)
        self.edge_angle_degrees = float(edge_angle_degrees)
        self.dark_level = float(dark_level)
        self.bright_level = float(bright_level)
        self.read_noise = float(max(0.0, read_noise))
        self.random_seed = int(random_seed)
        configured_root = data_root or os.environ.get(
            "MITR_SIM_FOCUS_DATA_ROOT",
            DEFAULT_SIM_FOCUS_DATA_ROOT,
        )
        self.data_root = Path(configured_root).expanduser()
        # This attribute is used by the imaging-directory discovery service.
        self.tiff1.write_path_template = str(self.data_root / "%Y")
        self.tiff1.read_path_template = self.tiff1.write_path_template

        self._stage_uid = "unstaged"
        self._trigger_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_status: Optional[DeviceStatus] = None
        self._trigger_thread: Optional[threading.Thread] = None
        self._active_temp_path: Optional[Path] = None

    def focus_blur_sigma(self, position: Optional[float] = None) -> float:
        """Return the image-plane Gaussian sigma at a motor position."""
        if position is None:
            position = self._motor_position()
        distance = float(position) - self.best_focus
        return float(
            self.minimum_blur_sigma
            + self.blur_per_distance_squared * distance * distance
        )

    def _motor_position(self) -> float:
        try:
            return float(self.focus_motor.position)
        except Exception:
            return float(self.focus_motor.get())

    def generate_image(self, position: Optional[float] = None, *, frame_number: int = 1) -> np.ndarray:
        """Generate one deterministic noisy slanted-edge image."""
        motor_position = self._motor_position() if position is None else float(position)
        sigma = self.focus_blur_sigma(motor_position)
        height, width = self.image_shape
        yy, xx = np.indices((height, width), dtype=np.float64)
        slope = float(np.tan(np.deg2rad(self.edge_angle_degrees)))
        edge_x = (0.5 * (width - 1)) + slope * (yy - 0.5 * (height - 1))
        signed_distance = (xx - edge_x) / np.sqrt(1.0 + slope * slope)
        edge_fraction = 0.5 * (
            1.0 + erf(signed_distance / (np.sqrt(2.0) * sigma))
        )

        # Gentle illumination structure makes the data realistic without
        # obscuring the well-defined edge-spread width.
        illumination = 1.0 + 0.025 * ((yy / max(1.0, height - 1.0)) - 0.5)
        image = (
            self.dark_level
            + (self.bright_level - self.dark_level) * edge_fraction
        ) * illumination
        image = image * float(self.cam.gain.get()) + float(self.cam.offset.get())

        if self.read_noise > 0:
            exposure = max(1e-6, float(self.cam.acquire_time.get()))
            noise_scale = self.read_noise / np.sqrt(max(exposure, 0.01) / 0.05)
            rng = np.random.default_rng(self.random_seed + int(frame_number))
            image = image + rng.normal(0.0, noise_scale, size=image.shape)

        return np.clip(np.rint(image), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    @staticmethod
    def _safe_stem(value: object) -> str:
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "sim_focus").strip())
        return stem.strip("._") or "sim_focus"

    @staticmethod
    def _safe_relative_folder(value: object) -> Path:
        raw = str(value or "adaptive_focus_sim").strip().replace("\\", "/")
        candidate = Path(raw)
        if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
            return Path("adaptive_focus_sim")
        safe_parts = [SimulatedFocusDetector._safe_stem(part) for part in candidate.parts]
        return Path(*safe_parts) if safe_parts else Path("adaptive_focus_sim")

    def _next_output_path(self, frame_number: int) -> Path:
        root = Path(datetime.now().strftime(str(self.data_root / "%Y")))
        folder = self._safe_relative_folder(self.tiff1.folder_name.get())
        stem = self._safe_stem(self.tiff1.file_name.get())
        return root / folder / f"{stem}_{self._stage_uid}_{int(frame_number):04d}.tif"

    def stage(self):
        self._stop_event.clear()
        self._stage_uid = uuid.uuid4().hex[:8]
        return super().stage()

    def unstage(self):
        self.stop(success=False)
        return super().unstage()

    def trigger(self):
        with self._trigger_lock:
            if self._active_status is not None and not self._active_status.done:
                raise RuntimeError(f"{self.name} already has an acquisition in progress")
            self._stop_event.clear()
            status = DeviceStatus(self)
            self._active_status = status
            self.cam.acquire.put(1)
            thread = threading.Thread(
                target=self._acquire_once,
                args=(status,),
                name=f"{self.name}-trigger",
                daemon=True,
            )
            self._trigger_thread = thread
            thread.start()
            return status

    def _acquire_once(self, status: DeviceStatus):
        temp_path = None
        try:
            delay = max(0.0, float(self.cam.acquire_time.get()))
            if self._stop_event.wait(delay):
                raise RuntimeError("simulated focus acquisition was stopped")

            frame_number = int(self.cam.array_counter.get()) + 1
            motor_position = self._motor_position()
            image = self.generate_image(motor_position, frame_number=frame_number)
            output_path = self._next_output_path(frame_number)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp.tif")
            self._active_temp_path = temp_path

            import tifffile

            tifffile.imwrite(str(temp_path), image, photometric="minisblack")
            if self._stop_event.is_set():
                raise RuntimeError("simulated focus acquisition was stopped")
            os.replace(temp_path, output_path)
            temp_path = None
            self._active_temp_path = None

            self.cam.array_counter.put(frame_number)
            self.image_path.put(str(output_path.resolve()))
            self.blur_sigma.put(self.focus_blur_sigma(motor_position))
            self.stats1.total.put(float(np.sum(image, dtype=np.float64)))
            if not status.done:
                status.set_finished()
        except Exception as ex:
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            if not status.done:
                status.set_exception(ex)
        finally:
            self._active_temp_path = None
            self.cam.acquire.put(0)
            with self._trigger_lock:
                if self._active_status is status:
                    self._active_status = None

    def stop(self, *, success: bool = False):
        self._stop_event.set()
        with self._trigger_lock:
            status = self._active_status
        if status is not None and not status.done:
            status.set_exception(RuntimeError("simulated focus acquisition was stopped"))
        temp_path = self._active_temp_path
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        self.cam.acquire.put(0)
        return super().stop(success=success)
