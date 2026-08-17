import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from diffractometer_controls import focus_offline_viewer
from diffractometer_controls import wf_norm_worker
from diffractometer_controls.live_image_pipeline import LiveImageState


EXPECTED_GAMMA_SETTINGS = {
    "size": 5,
    "dif": "auto",
    "sigma_multiplier": 8.0,
    "backend": "auto",
    "threshold_mode": "shared",
    "calibration_frames": 8,
}


class GammaFilterTests(unittest.TestCase):
    def test_beamline_settings_are_explicit(self):
        self.assertEqual(focus_offline_viewer.NIT_GAMMA_FILTER_SETTINGS, EXPECTED_GAMMA_SETTINGS)
        self.assertEqual(wf_norm_worker.NIT_GAMMA_FILTER_SETTINGS, EXPECTED_GAMMA_SETTINGS)

    def test_focus_filter_passes_beamline_settings(self):
        captured = {}

        def fake_remove_gammas(image, **kwargs):
            captured.update(kwargs)
            return image

        with mock.patch.object(
            focus_offline_viewer, "_resolve_nit_remove_gammas", return_value=fake_remove_gammas
        ):
            result = focus_offline_viewer._nit_gamma_filter(np.ones((9, 9)), size=5)

        self.assertEqual(result.shape, (9, 9))
        for key, value in EXPECTED_GAMMA_SETTINGS.items():
            self.assertEqual(captured[key], value)
        self.assertEqual(captured["axis"], 0)
        self.assertTrue(captured["symmetric"])

    def test_focus_defaults_to_gamma(self):
        args = focus_offline_viewer.build_arg_parser().parse_args([])
        self.assertEqual(args.preprocess_mode, "gamma")
        self.assertEqual(args.preprocess_size, 5)


class LiveImageStateTests(unittest.TestCase):
    def test_raw_frame_is_an_owned_exact_copy(self):
        source = np.arange(25, dtype=np.uint16).reshape(5, 5)
        expected = source.copy()
        state = LiveImageState()

        state.ingest_raw(source, exposure_s=1.5)
        source[:] = 0

        self.assertTrue(np.array_equal(state.raw_frame, expected))
        self.assertEqual(state.raw_frame.dtype, np.uint16)
        self.assertEqual(state.raw_exposure_s, 1.5)

    def test_stale_filter_results_cannot_replace_new_frames(self):
        state = LiveImageState()
        state.ingest_raw(np.full((5, 5), 1, dtype=np.uint16))
        state.set_filter_enabled(True)
        stale_request = state.make_filter_request()

        state.ingest_raw(np.full((5, 5), 2, dtype=np.uint16))
        accepted = state.accept_filtered(
            np.full((5, 5), 10, dtype=np.float32),
            frame_id=stale_request.frame_id,
            generation=stale_request.generation,
        )

        self.assertFalse(accepted)
        self.assertTrue(np.array_equal(state.display_source(), state.raw_frame))
        self.assertTrue(np.all(state.display_source() == 2))

    def test_toggle_generation_rejects_inflight_results(self):
        raw = np.full((5, 5), 3, dtype=np.uint16)
        state = LiveImageState()
        state.ingest_raw(raw)
        state.set_filter_enabled(True)
        stale_request = state.make_filter_request()

        state.set_filter_enabled(False)
        self.assertTrue(np.array_equal(state.display_source(), raw))
        state.set_filter_enabled(True)

        self.assertFalse(
            state.accept_filtered(
                np.full((5, 5), 30, dtype=np.float32),
                frame_id=stale_request.frame_id,
                generation=stale_request.generation,
            )
        )
        self.assertTrue(np.array_equal(state.display_source(), raw))

    def test_filter_and_normalization_are_rebuilt_from_raw(self):
        raw = np.full((5, 5), 100, dtype=np.uint16)
        filtered = np.full((5, 5), 90, dtype=np.float32)
        state = LiveImageState()
        state.ingest_raw(raw)

        self.assertTrue(np.array_equal(state.compose_display(), raw))
        self.assertTrue(np.array_equal(state.compose_display(lambda image: image / 10.0), raw / 10.0))

        state.set_filter_enabled(True)
        request = state.make_filter_request()
        self.assertTrue(
            state.accept_filtered(
                filtered,
                frame_id=request.frame_id,
                generation=request.generation,
            )
        )
        self.assertTrue(np.array_equal(state.compose_display(), filtered))
        self.assertTrue(
            np.array_equal(state.compose_display(lambda image: image / 10.0), filtered / 10.0)
        )

        state.set_filter_enabled(False)
        restored = state.compose_display()
        self.assertTrue(np.array_equal(restored, raw))
        self.assertEqual(restored.dtype, raw.dtype)


class WhiteFieldWorkerTests(unittest.TestCase):
    def _write_stack(self, directory, count=5):
        paths = []
        for index in range(count):
            image = np.full((16, 16), 100.0, dtype=np.float32)
            image[8, 8] = 1000.0 if index == 0 else 100.0
            path = Path(directory) / f"wf_{index}.npy"
            np.save(path, image)
            paths.append(str(path))
        return paths

    def test_worker_uses_mad_adaptive_then_gamma(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_stack(directory)
            with mock.patch.object(
                wf_norm_worker,
                "_combine_images_nit",
                side_effect=lambda stack, method="mad_adaptive", return_diagnostics=False: (
                    np.median(stack, axis=0),
                    {"rejected_fraction": 0.125, "accepted_count": np.ones((16, 16))},
                ),
            ) as merge_mock, mock.patch.object(
                wf_norm_worker,
                "_remove_gammas_nit",
                side_effect=lambda image, size, ncore=None, return_diagnostics=False: (
                    image,
                    {
                        "backend": "opencv",
                        "changed_fraction": 0.025,
                        "dif": 12.5,
                        "changed_fraction_per_image": np.array([0.025]),
                    },
                ),
            ) as gamma_mock:
                payload = wf_norm_worker.build_wf_norm_array_worker(paths)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["merge_used"], "mad_adaptive")
        self.assertEqual(payload["filter_used"], "gamma")
        self.assertEqual(payload["merge_diagnostics"], {"rejected_fraction": 0.125})
        self.assertEqual(
            payload["filter_diagnostics"],
            {"backend": "opencv", "changed_fraction": 0.025, "threshold": 12.5},
        )
        merge_mock.assert_called_once()
        self.assertEqual(merge_mock.call_args.kwargs["method"], "mad_adaptive")
        gamma_mock.assert_called_once()
        self.assertEqual(gamma_mock.call_args.kwargs["size"], 5)
        self.assertTrue(gamma_mock.call_args.kwargs["return_diagnostics"])

    def test_live_filter_worker_returns_small_diagnostics(self):
        image = np.full((21, 21), 100.0, dtype=np.float32)
        diagnostics = {
            "backend": "opencv",
            "changed_fraction": 0.01,
            "dif": 8.0,
            "changed_fraction_per_image": np.ones(1),
        }
        with mock.patch.object(
            wf_norm_worker,
            "_remove_gammas_nit",
            return_value=(image, diagnostics),
        ):
            payload = wf_norm_worker.filter_live_image_worker(image)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["filtered_image"].shape, image.shape)
        self.assertEqual(
            payload["filter_diagnostics"],
            {"backend": "opencv", "changed_fraction": 0.01, "threshold": 8.0},
        )
        self.assertEqual(payload["filter_requested"], "gamma")
        self.assertEqual(payload["filter_used"], "gamma")

    def test_worker_honors_mean_merge_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_stack(directory)
            payload = wf_norm_worker.build_wf_norm_array_worker(
                paths,
                filter_method="median",
                merge_method="mean",
            )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["merge_requested"], "mean")
        self.assertEqual(payload["merge_used"], "mean")
        self.assertEqual(payload["filter_used"], "median")

    def test_live_filter_worker_honors_median_selection(self):
        image = np.full((21, 21), 100.0, dtype=np.float32)
        image[10, 10] = 1000.0

        payload = wf_norm_worker.filter_live_image_worker(
            image,
            filter_method="median",
        )

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["filter_requested"], "median")
        self.assertEqual(payload["filter_used"], "median")
        self.assertAlmostEqual(float(payload["filtered_image"][10, 10]), 100.0)

    @unittest.skipUnless(
        importlib.util.find_spec("neutron_imaging_tools") is not None,
        "neutron-imaging-tools is not installed",
    )
    def test_installed_gamma_filter_removes_bright_and_dark_spikes(self):
        image = np.full((21, 21), 100.0, dtype=np.float32)
        image[10, 10] = 1000.0
        image[5, 5] = -500.0

        filtered = wf_norm_worker._remove_gammas_nit(image, size=5)

        self.assertAlmostEqual(float(filtered[10, 10]), 100.0)
        self.assertAlmostEqual(float(filtered[5, 5]), 100.0)

        payload = wf_norm_worker.filter_live_image_worker(image)
        self.assertTrue(payload["ok"], payload.get("error", ""))
        self.assertAlmostEqual(float(payload["filtered_image"][10, 10]), 100.0)
        self.assertAlmostEqual(float(payload["filtered_image"][5, 5]), 100.0)
        self.assertIn(
            payload["filter_diagnostics"]["backend"],
            {"opencv", "scipy", "tomopy"},
        )

    @unittest.skipUnless(
        importlib.util.find_spec("neutron_imaging_tools") is not None,
        "neutron-imaging-tools is not installed",
    )
    def test_installed_nit_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = wf_norm_worker.build_wf_norm_array_worker(self._write_stack(directory))

        self.assertTrue(payload["ok"], payload.get("error", ""))
        self.assertEqual(payload["merge_used"], "mad_adaptive")
        self.assertEqual(payload["filter_used"], "gamma")
        self.assertIn(payload["filter_diagnostics"]["backend"], {"opencv", "scipy", "tomopy"})
        self.assertGreaterEqual(payload["filter_diagnostics"]["changed_fraction"], 0.0)
        self.assertGreaterEqual(payload["filter_diagnostics"]["threshold"], 0.0)
        self.assertGreaterEqual(payload["merge_diagnostics"]["rejected_fraction"], 0.0)
        self.assertEqual(payload["norm_array"].shape, (16, 16))
        self.assertTrue(np.all(np.isfinite(payload["norm_array"])))


if __name__ == "__main__":
    unittest.main()
