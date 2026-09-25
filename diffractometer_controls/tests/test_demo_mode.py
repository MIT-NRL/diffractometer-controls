import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from diffractometer_controls.demo_runtime import (
    DEFAULT_PORTS,
    DemoRuntime,
    DemoRuntimeError,
    demo_environment,
)
from diffractometer_controls.demo_simulation import (
    diffraction_spectrum,
    edge_blur_width,
    gaussian_image,
    slanted_edge_image,
)


class DemoSimulationTests(unittest.TestCase):
    @staticmethod
    def _width(axis, values):
        weights = np.asarray(values, dtype=float)
        weights = np.clip(weights - np.percentile(weights, 10), 0, None)
        center = np.average(axis, weights=weights)
        return np.sqrt(np.average((axis - center) ** 2, weights=weights))

    def test_diffraction_optimum_is_stronger_and_narrower(self):
        axis_best, best = diffraction_spectrum(3.0, exposure=1.0, seed=5)
        axis_off, off = diffraction_spectrum(5.0, exposure=1.0, seed=5)
        self.assertGreater(best.sum(), off.sum())
        self.assertLess(self._width(axis_best, best), self._width(axis_off, off))

    def test_diffraction_is_seeded_and_scales_with_exposure(self):
        _, first = diffraction_spectrum(3.0, exposure=0.2, seed=19)
        _, repeat = diffraction_spectrum(3.0, exposure=0.2, seed=19)
        _, longer = diffraction_spectrum(3.0, exposure=1.0, seed=19)
        np.testing.assert_array_equal(first, repeat)
        self.assertGreater(longer.sum(), first.sum() * 4)

    def test_diffraction_bin_count_and_detector_offset(self):
        axis0, _ = diffraction_spectrum(3.0, nbins=127, detector_offset=0.0, seed=3)
        axis1, shifted = diffraction_spectrum(3.0, nbins=83, detector_offset=0.2, seed=3)
        _, unshifted = diffraction_spectrum(3.0, nbins=83, detector_offset=0.0, seed=3)
        self.assertEqual(axis0.shape, (127,))
        self.assertEqual(shifted.shape, (83,))
        self.assertFalse(np.array_equal(shifted, unshifted))

    def test_gaussian_image_follows_xy_motors(self):
        image0 = gaussian_image(0, 0, exposure=2.0, seed=7)
        image1 = gaussian_image(2, -2, exposure=2.0, seed=7)
        y0, x0 = np.unravel_index(np.argmax(image0), image0.shape)
        y1, x1 = np.unravel_index(np.argmax(image1), image1.shape)
        self.assertGreater(x1, x0)
        self.assertLess(y1, y0)

    def test_slanted_edge_blur_has_known_optimum(self):
        self.assertLess(edge_blur_width(0), edge_blur_width(2))
        sharp = slanted_edge_image(0, exposure=5.0, seed=2).astype(float)
        blurred = slanted_edge_image(2, exposure=5.0, seed=2).astype(float)
        self.assertGreater(np.max(np.abs(np.diff(sharp, axis=1))), np.max(np.abs(np.diff(blurred, axis=1))))


class DemoEnvironmentTests(unittest.TestCase):
    def test_environment_is_loopback_only_and_uses_dedicated_prefix(self):
        env = demo_environment()
        self.assertEqual(env["MITR_DEMO_ACTIVE"], "1")
        self.assertEqual(env["MITR_EPICS_PREFIX"], "demo4dh4:")
        self.assertEqual(env["EPICS_CA_AUTO_ADDR_LIST"], "NO")
        self.assertEqual(env["EPICS_CA_ADDR_LIST"], "127.0.0.1")
        self.assertIn(str(DEFAULT_PORTS["qserver_control"]), env["MITR_QSERVER_CONTROL_ADDR"])

    def test_occupied_port_error_names_the_service(self):
        runtime = DemoRuntime()
        with mock.patch.object(runtime, "_port_available", return_value=False):
            with self.assertRaisesRegex(DemoRuntimeError, r"redis \(6380\)"):
                runtime._check_ports()

    def test_demo_startup_is_tiled_free_and_excludes_adaptive_profile(self):
        startup = Path(__file__).resolve().parents[1] / "bluesky_config" / "demo" / "startup"
        names = {path.name for path in startup.glob("*.py")}
        source = "\n".join(path.read_text().lower() for path in startup.glob("*.py"))
        self.assertNotIn("tiled.client", source)
        self.assertFalse(any("adaptive" in name for name in names))


if __name__ == "__main__":
    unittest.main()
