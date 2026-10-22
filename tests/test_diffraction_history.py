"""History normalization tests, independent of hardware and Qt."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
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
    def test_startup_recovers_monitor_descriptors_even_before_first_committed_point(self):
        run = Run(psd_stream(), export={"status": "collecting", "incremental": True, "committed_points": 0})
        run.metadata["stop"] = None
        run.metadata["start"]["live_plot_fields"] = {
            "psd_counts": {"role": "profile", "transport": "document", "stream": "psd_counts_monitor"}}
        monitor = psd_stream()
        monitor.descriptors = [{"uid": "actual-monitor-uid", "name": "psd_counts_monitor",
                                "run_start": "history-run", "data_keys": monitor.metadata["data_keys"]}]
        monitor.read = Mock(side_effect=AssertionError("Read a monitor array"))
        run["psd_counts_monitor"] = monitor
        snapshot = build_history_snapshot(run, allow_empty=True)
        self.assertEqual(snapshot["rows"], [])
        self.assertEqual(snapshot["descriptor_docs"][0]["uid"], "actual-monitor-uid")
        monitor.read.assert_not_called()

    def scalar_stream(self, *, empty=False):
        values = [] if empty else [295, 296]
        return Stream(xr.Dataset({'temperature': ('event', values), 'theta': ('event', values)},
                                 coords={'seq_num': ('event', [] if empty else [2, 4])}), {
            'temperature': {'dtype': 'number', 'shape': [], 'object_name': 'temperature', 'units': 'C'},
            'theta': {'dtype': 'number', 'shape': [], 'object_name': 'theta', 'units': 'deg'},
        })

    def test_scalar_readings_ignore_file_export_status_and_hdf5_point_count(self):
        for status in ('collecting', 'writing', 'failed', 'ready'):
            for incremental in (False, True):
                with self.subTest(status=status, incremental=incremental):
                    export = {'status': status, 'incremental': incremental,
                              'committed_points': 0, 'error': 'file unavailable'}
                    run = Run(self.scalar_stream(), scalar=True, export=export)
                    run.metadata['start']['motors'] = ['theta']
                    run.metadata['stop'] = None
                    snapshot = build_history_snapshot(run)
                    self.assertEqual(snapshot['seq_nums'], [2, 4])
                    self.assertEqual([row['temperature'] for row in snapshot['rows']], [295, 296])
                    self.assertEqual(snapshot['file_export'], export)
                    self.assertIsNone(snapshot['stop'])
                    self.assertEqual(run.metadata['file_export'], export)

    def test_native_scalar_table_reads_values_and_sequences_in_one_snapshot(self):
        stream = self.scalar_stream()
        table = Mock()
        table.read.return_value = pd.DataFrame({'temperature': [295, 296], 'theta': [1, 2],
                                               'seq_num': [2, 4], 'ts_temperature': [10, 20]})
        stream.base = {'internal': table}
        stream.read = Mock(side_effect=AssertionError('Used separate field reads'))
        run = Run(stream, scalar=True, export={'status': 'collecting'})
        snapshot = build_history_snapshot(run)
        self.assertEqual(snapshot['seq_nums'], [2, 4])
        self.assertEqual(snapshot['rows'], [{'temperature': 295, 'theta': 1}, {'temperature': 296, 'theta': 2}])
        table.read.assert_called_once_with()
        stream.read.assert_not_called()

    def test_empty_scalar_acquisition_waits_instead_of_loading_monitor_samples(self):
        run = Run(self.scalar_stream(empty=True), scalar=True, export={'status': 'collecting'})
        run.metadata['stop'] = None
        run['temperature_monitor'] = self.scalar_stream()
        snapshot = build_history_snapshot(run, allow_empty=True)
        self.assertEqual(snapshot['rows'], [])
        self.assertTrue(snapshot['pending_reason'])
        del run['primary']
        snapshot = build_history_snapshot(run, allow_empty=True)
        self.assertEqual(snapshot['rows'], [])
        self.assertIn('acquisition stream', snapshot['pending_reason'])

    def test_missing_native_scalar_column_is_an_error(self):
        stream = self.scalar_stream()
        stream.base = {'internal': SimpleNamespace(read=lambda: pd.DataFrame({'seq_num': [1], 'theta': [2]}))}
        with self.assertRaisesRegex(RuntimeError, 'Missing stored scalar fields: temperature'):
            build_history_snapshot(Run(stream, scalar=True))

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
