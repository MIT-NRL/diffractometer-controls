"""History normalization tests, independent of hardware and Qt."""

import unittest

import numpy as np
import xarray as xr

from diffractometer_controls.analysis.diffraction_history import build_history_snapshot


class Stream(dict):
    def __init__(self, dataset, data_keys):
        super().__init__()
        self.dataset = dataset
        self.metadata = {"data_keys": data_keys}

    def read(self):
        return self.dataset


class Run(dict):
    def __init__(self, stream, *, namespace=False, export=None, scalar=False):
        super().__init__(streams={"primary": stream}) if namespace else super().__init__(primary=stream)
        self["artifacts"] = object()  # Must never try to read the whole-file view.
        self.metadata = {"start": {
            "uid": "history-run", "time": 1, "experiment_type": "diffraction",
            "data_type": "scalar" if scalar else "1d", "detectors": ["temperature" if scalar else "psd"],
        }, "stop": {"exit_status": "success"}}
        if export is not None:
            self.metadata["file_export"] = export


def psd_stream():
    return Stream(xr.Dataset({
        "psd_counts": (("event", "bin"), [[1, 2, 3], [4, 5, 6]]),
        "psd_position_x": (("bin",), [-1, 0, 1]),
    }, coords={"seq_num": ("event", [2, 4])}), {
        "psd_counts": {"dtype": "array", "shape": [3]},
        "psd_position_x": {"dtype": "array", "shape": [3]},
    })


class DiffractionHistoryTests(unittest.TestCase):
    def test_both_stream_layouts_keep_constant_axes_and_sequence_gaps(self):
        for namespace in (False, True):
            with self.subTest(namespace=namespace):
                snapshot = build_history_snapshot(Run(psd_stream(), namespace=namespace))
                self.assertEqual(snapshot["seq_nums"], [2, 4])
                self.assertEqual(snapshot["rows"][1]["psd_counts"], [4, 5, 6])
                self.assertEqual(snapshot["rows"][1]["psd_position_x"], [-1, 0, 1])

    def test_old_scalar_metadata_recovers_only_recorded_detector_fields(self):
        stream = Stream(xr.Dataset({"temperature": ("event", [295, 296]), "theta": ("event", [1, 2])}), {
            "temperature": {"dtype": "number", "shape": [], "object_name": "temperature", "units": "C"},
            "theta": {"dtype": "number", "shape": [], "object_name": "theta", "units": "deg"},
        })
        run = Run(stream, scalar=True)
        run.metadata["start"]["motors"] = ["theta"]
        snapshot = build_history_snapshot(run)
        self.assertEqual(list(snapshot["start"]["live_plot_fields"]), ["temperature"])
        self.assertNotIn("live_plot_fields", run.metadata["start"])
        self.assertEqual(snapshot["rows"][1]["theta"], 2)

    def test_unfinished_or_failed_export_is_explicit_and_does_not_read_arrays(self):
        for status in ("collecting", "writing", "failed"):
            with self.subTest(status=status):
                run = Run(psd_stream(), export={"status": status, "error": "disk offline"})
                with self.assertRaisesRegex(RuntimeError, status):
                    build_history_snapshot(run)
                snapshot = build_history_snapshot(run, allow_empty=True)
                self.assertEqual(snapshot["rows"], [])
                self.assertIn(status, snapshot["pending_reason"])

    def test_missing_array_is_an_error_instead_of_a_partial_plot(self):
        stream = psd_stream()
        stream.dataset = stream.dataset.drop_vars("psd_counts")
        with self.assertRaisesRegex(RuntimeError, "missing stored fields.*psd_counts"):
            build_history_snapshot(Run(stream))

    def test_non_diffraction_run_is_rejected(self):
        run = Run(psd_stream())
        run.metadata["start"]["experiment_type"] = "imaging"
        with self.assertRaisesRegex(ValueError, "diffraction"):
            build_history_snapshot(run)

    def test_one_bin_one_event_keeps_profile_shape(self):
        stream = Stream(xr.Dataset({"psd_counts": (("event", "bin"), np.array([[9]]))}),
                        {"psd_counts": {"dtype": "array", "shape": [1]}})
        self.assertEqual(build_history_snapshot(Run(stream))["rows"], [{"psd_counts": [9]}])
