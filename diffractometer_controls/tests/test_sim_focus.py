import os
import runpy
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPHYD_CONTROL_LAYER", "dummy")

import numpy as np
import tifffile

from diffractometer_controls.sim_focus import (
    SimulatedFocusDetector,
    SimulatedFocusMotor,
)
from diffractometer_controls.focus_offline_viewer import (
    FocusOfflineWindow,
    analyze_roi,
    preprocess_image_with_mode,
)


def _edge_width_from_center_row(image):
    row = np.asarray(image[image.shape[0] // 2], dtype=np.float64)
    derivative = np.clip(np.diff(row), 0.0, None)
    x = np.arange(derivative.size, dtype=np.float64) + 0.5
    weight = derivative / np.sum(derivative)
    center = float(np.sum(x * weight))
    return float(np.sqrt(np.sum(weight * (x - center) ** 2)))


class SimulatedFocusImageTests(unittest.TestCase):
    def test_imaging_plan_detector_fields_are_dropdowns(self):
        startup_dir = (
            Path(__file__).resolve().parents[1]
            / "bluesky_config"
            / "startup"
        )
        motor = SimulatedFocusMotor(name="sim_focus_motor", delay=0)
        cam1 = SimulatedFocusDetector(
            name="cam1",
            motor=motor,
            image_shape=(32, 32),
            read_noise=0,
        )
        sim_cam = SimulatedFocusDetector(
            name="sim_focus_cam",
            motor=motor,
            image_shape=(32, 32),
            read_noise=0,
        )
        imaging_namespace = runpy.run_path(
            str(startup_dir / "92-plans_imaging.py"),
            init_globals={
                "cam1": cam1,
                "sim_focus_cam": sim_cam,
                "_collect_movable_names": lambda: ["sim_focus_motor"],
            },
        )
        adaptive_namespace = runpy.run_path(
            str(startup_dir / "93-adaptive_plans_imaging.py"),
            init_globals={
                "cam1": SimpleNamespace(focus=motor),
                "sim_focus_cam": sim_cam,
                "_collect_movable_names": lambda: ["sim_focus_motor"],
                "_collect_imaging_detector_names": lambda: [
                    "cam1",
                    "sim_focus_cam",
                ],
            },
        )

        plans = [
            imaging_namespace["tomo_scan"],
            imaging_namespace["imaging"],
            imaging_namespace["imaging_scan"],
            adaptive_namespace["adaptive_imaging_focus_scan"],
        ]
        for plan in plans:
            with self.subTest(plan=plan.__name__):
                annotation = plan._custom_parameter_annotation_["parameters"]["detector"]
                self.assertEqual(annotation["annotation"], "ImagingDetectors")
                self.assertEqual(annotation["default"], "cam1")
                self.assertEqual(
                    annotation["devices"]["ImagingDetectors"],
                    ["cam1", "sim_focus_cam"],
                )
                self.assertTrue(annotation["convert_device_names"])

    def test_sim_motor_is_in_adaptive_plan_motor_choices(self):
        startup_file = (
            Path(__file__).resolve().parents[1]
            / "bluesky_config"
            / "startup"
            / "90-plans_general.py"
        )
        namespace = runpy.run_path(str(startup_file))
        collector = namespace["_collect_movable_names"]
        collector.__globals__["sim_focus_motor"] = SimulatedFocusMotor(
            name="sim_focus_motor",
            delay=0,
        )

        choices = collector()

        self.assertIn("sim_focus_motor", choices)

    def test_motor_position_changes_slanted_edge_blur(self):
        motor = SimulatedFocusMotor(name="test_focus_motor", delay=0)
        detector = SimulatedFocusDetector(
            name="test_focus_cam",
            motor=motor,
            image_shape=(256, 256),
            read_noise=0,
        )

        in_focus = detector.generate_image(0.0)
        out_of_focus = detector.generate_image(2.0)

        self.assertEqual(in_focus.dtype, np.uint16)
        self.assertEqual(in_focus.shape, (256, 256))
        self.assertGreater(
            _edge_width_from_center_row(out_of_focus),
            2.5 * _edge_width_from_center_row(in_focus),
        )
        self.assertEqual(detector.focus_blur_sigma(0.0), 1.2)
        self.assertEqual(detector.focus_blur_sigma(2.0), 17.2)

    def test_existing_focus_pipeline_finds_the_sharpest_simulated_frame(self):
        motor = SimulatedFocusMotor(name="test_focus_motor", delay=0)
        detector = SimulatedFocusDetector(
            name="test_focus_cam",
            motor=motor,
            image_shape=(256, 256),
            random_seed=4,
        )
        metrics = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            for position in (-2.0, -2.0 / 7.0, 0.0, 2.0 / 7.0, 2.0):
                filtered = preprocess_image_with_mode(
                    detector.generate_image(position),
                    filter_mode="gamma",
                    filter_size=5,
                )
                metrics[position] = analyze_roi(
                    filtered,
                    x0_abs=96,
                    x1_abs=160,
                    y0_abs=64,
                    y1_abs=192,
                )

        self.assertTrue(all(result.ok for result in metrics.values()))
        self.assertLess(metrics[0.0].psf_sigma, metrics[-2.0].psf_sigma)
        self.assertLess(metrics[0.0].psf_sigma, metrics[2.0].psf_sigma)
        self.assertGreater(metrics[0.0].mtf50, metrics[-2.0].mtf50)
        self.assertGreater(metrics[0.0].mtf50, metrics[2.0].mtf50)

        local_positions = np.asarray([-2.0 / 7.0, 0.0, 2.0 / 7.0])
        local_mtf50 = np.asarray([metrics[position].mtf50 for position in local_positions])
        local_fit = FocusOfflineWindow._local_parabola_fit(
            local_positions,
            local_mtf50,
            local_radius=0.5,
            find="max",
        )
        self.assertIsNotNone(local_fit)
        self.assertAlmostEqual(local_fit[2], 0.0, places=3)

    def test_trigger_writes_tiff_and_publishes_path_signal(self):
        with tempfile.TemporaryDirectory() as tmp:
            motor = SimulatedFocusMotor(name="test_focus_motor", delay=0)
            detector = SimulatedFocusDetector(
                name="test_focus_cam",
                motor=motor,
                data_root=tmp,
                image_shape=(96, 128),
                read_noise=0,
            )
            detector.cam.acquire_time.put(0)
            detector.tiff1.file_name.put("focus test")
            detector.tiff1.folder_name.put("session/a")
            detector.stage()
            try:
                motor.set(1.5).wait(timeout=2)
                detector.trigger().wait(timeout=5)

                image_key = f"{detector.name}_image_path"
                path = Path(detector.read()[image_key]["value"])
                self.assertTrue(path.is_file())
                self.assertEqual(path.suffix, ".tif")
                self.assertEqual(tifffile.imread(path).shape, (96, 128))
                self.assertEqual(detector.cam.array_counter.get(), 1)
                self.assertAlmostEqual(detector.blur_sigma.get(), 10.2)
                self.assertEqual(detector.describe()[image_key]["dtype"], "string")

                detector.trigger().wait(timeout=5)
                second_path = Path(detector.read()[image_key]["value"])
                self.assertNotEqual(second_path, path)
                self.assertTrue(second_path.is_file())
                self.assertEqual(detector.cam.array_counter.get(), 2)
            finally:
                detector.unstage()

    def test_stop_cancels_an_inflight_file_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            motor = SimulatedFocusMotor(name="test_focus_motor", delay=0)
            detector = SimulatedFocusDetector(
                name="test_focus_cam",
                motor=motor,
                data_root=tmp,
                image_shape=(64, 64),
            )
            detector.cam.acquire_time.put(2.0)
            detector.stage()
            status = detector.trigger()
            time.sleep(0.02)
            detector.stop(success=False)

            self.assertTrue(status.done)
            self.assertFalse(status.success)
            self.assertEqual(detector.cam.acquire.get(), 0)
            self.assertEqual(list(Path(tmp).rglob("*.tif")), [])
            detector.unstage()


if __name__ == "__main__":
    unittest.main()
