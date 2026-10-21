"""Outcome evidence, searchable metadata and full-data fit viewport regressions."""

import copy
import time
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from qtpy import QtCore, QtTest, QtWidgets

from diffraction_analysis.history import build_history_snapshot, fetch_recent_history_entries
from diffraction_analysis.metadata import acquisition_details, environment_readings, read_metadata_details, run_outcome, run_overview
from diffraction_analysis.panel import AnalysisPanel
from diffraction_analysis.fitting import FitConfig, fit_curve
from diffraction_analysis.viewer import RunDataPanel
from diffraction_analysis.trends import FitTrends
from diffraction_analysis.window import DiffractionAnalysisWindow
from tests.test_diffraction_batch import scan_run
from tests.test_diffraction_history import Stream
from tests.test_diffraction_history_window import Catalog


class OutcomeTests(unittest.TestCase):
    def test_recorded_outcomes_and_missing_stop_are_distinct(self):
        for raw, expected in [('success', 'Complete'), ('fail', 'Failed'), ('abort', 'Aborted'),
                              ('stopped', 'Stopped'), ('custom', 'Custom'), ('', 'Unknown')]:
            with self.subTest(raw=raw):
                outcome = run_outcome({}, {'exit_status': raw, 'reason': 'recorded explanation'})
                self.assertEqual(outcome['status'], expected)
                self.assertEqual(outcome['reason'], 'recorded explanation')
        self.assertEqual(run_outcome({}, None)['status'], 'Unfinished')
        self.assertIn('does not establish', run_outcome({}, None)['note'])

    def test_success_with_known_shortfall_is_partial_without_claiming_operator_stop(self):
        start = {'num_points': 20}
        stop = {'exit_status': 'success', 'num_events': {'primary': 7, 'baseline': 2}}
        outcome = run_outcome(start, stop)
        self.assertEqual((outcome['status'], outcome['acquired'], outcome['planned']), ('Partial', 7, 20))
        self.assertIn('may have', outcome['note'])
        self.assertEqual(run_outcome({}, stop)['status'], 'Complete')
        self.assertEqual(run_outcome(start, {'exit_status': 'fail', 'num_events': {'primary': 7}})['status'], 'Failed')
        self.assertEqual(run_outcome({'num_points': None}, {'exit_status': 'success'})['status'], 'Complete')
        self.assertEqual(run_outcome({'plan_pattern_args': {'num_steps': 20}}, stop)['status'], 'Partial')

    def test_history_entries_keep_outcomes_and_reasons_without_data_reads(self):
        run = scan_run()
        run.metadata['stop'].update(exit_status='fail', reason='motor timeout')
        with patch.object(type(run['primary']), 'read', side_effect=AssertionError('Read array in listing')):
            entry, = fetch_recent_history_entries(Catalog(scan=run))
        self.assertEqual(entry['status'], 'Failed')
        self.assertEqual(entry['reason'], 'motor timeout')
        self.assertIn('Failed', entry['label'])

    def test_overview_keeps_acquisition_and_storage_outcomes_separate(self):
        metadata = {'start': {'uid': 'run', 'time': 100, 'num_points': 5},
                    'stop': {'exit_status': 'success', 'time': 112, 'num_events': {'primary': 5}},
                    'file_export': {'status': 'failed', 'error': 'disk offline', 'file_uri': 'file:///data/run.h5'}}
        original = copy.deepcopy(metadata)
        overview = run_overview(metadata, point_count=5)
        self.assertEqual(overview['Run']['Status'], 'Complete')
        self.assertEqual(overview['Run']['Duration'], '12 s')
        self.assertEqual(overview['Storage']['Export status'], 'failed')
        self.assertEqual(overview['Storage']['Export error'], 'disk offline')
        self.assertEqual(overview['Acquisition']['Points loaded'], 5)
        self.assertEqual(metadata, original)

    def test_scan_geometry_exposure_and_detector_settings_keep_units_and_zeros(self):
        metadata = {'start': {'motors': ['theta'], 'plan_args': {'acquire_time': 0, 'delay': 0},
                    'plan_pattern_args': {'motor': 'theta', 'start_pos': 0, 'stop_pos': 2, 'step_size': 0.5, 'num_steps': 5},
                    'detector_configurations': {'psd0': {'acquire_time': 0, 'nbins': 350, 'soft_lld': 0},
                                                'psd1': {'acquire_time': 2, 'nbins': 128}},
                    'gauge_volume': '2 mm slit', 'bluesky_mode': 'production'},
                    'stream_descriptors': [{'data_keys': {'theta': {'units': 'deg'}}}]}
        detail = acquisition_details(metadata)
        self.assertEqual(detail['Timing']['Exposure / point (set)'], '0 s')
        self.assertEqual(detail['Timing']['Delay between points'], '0 s')
        self.assertEqual(detail['Scan geometry']['theta']['Start (planned)'], '0 deg')
        self.assertEqual(detail['Scan geometry']['theta']['Step size'], '0.5 deg')
        self.assertEqual(detail['Scan geometry']['theta']['Number of points'], 5)
        self.assertEqual(detail['Detector settings']['psd0']['Lower threshold'], 0)
        self.assertEqual(detail['Detector settings']['psd1']['Exposure / point'], '2 s')
        self.assertEqual(run_overview(metadata)['Sample / experiment']['Gauge volume'], '2 mm slit')

    def test_parallel_grid_list_scans_and_scalar_software_dwell(self):
        parallel = {'start': {'motors': ['a', 'b'], 'plan_pattern_args': {'motor1': 'a', 'motor2': 'b',
                     'start_pos1': -2, 'stop_pos1': 2, 'step_size1': 1, 'start_pos2': 3, 'stop_pos2': 7,
                     'step_size2': 1, 'num_steps': 5}}}
        self.assertEqual(set(acquisition_details(parallel)['Scan geometry']), {'a', 'b'})
        grid = {'start': {'motors': ['a', 'b'], 'plan_pattern': 'outer_product',
                         'plan_pattern_args': {'args': ['a', 0, 1, 3, 'b', 2, 3, 4]}}}
        self.assertEqual(acquisition_details(grid)['Scan geometry']['b']['Number of points'], 4)
        listed = {'start': {'plan_pattern_args': {'motor': 'a', 'position_list': [1, 4, 9], 'num_steps': 3}}}
        self.assertEqual(acquisition_details(listed)['Scan geometry']['a']['Positions (planned)'], [1, 4, 9])
        passive = {'start': {'plan_args': {'acquire_time': None}, 'det_config': {'software_dwell_time': 0.5}}}
        self.assertEqual(acquisition_details(passive)['Timing']['Software dwell / point'], '0.5 s')
        self.assertNotIn('Exposure / point (set)', acquisition_details(passive)['Timing'])

    def test_old_baseline_is_read_without_touching_primary_arrays_or_catalog_metadata(self):
        run = scan_run()
        baseline = Stream(xr.Dataset({'reactor_power_6': ('event', [5.2, 5.3]),
                                     'theta': ('event', [1, 2]), 'theta_setpoint': ('event', [1, 2])}),
                          {'reactor_power_6': {'dtype': 'number', 'shape': [], 'units': 'MW'},
                           'theta': {'dtype': 'number', 'shape': [], 'units': 'deg'}})
        run['baseline'] = baseline
        run['reactor_power_6_monitor'] = object()
        original = copy.deepcopy(run.metadata)
        with patch.object(run['primary'], 'read', side_effect=AssertionError('Read primary while loading metadata')):
            metadata = read_metadata_details(run, run.metadata)
        readings = environment_readings(metadata)
        self.assertEqual([r['signal'] for r in readings], ['reactor_power_6', 'theta'])
        self.assertEqual(readings[0]['units'], 'MW')
        self.assertEqual(readings[0]['monitor'], 'reactor_power_6_monitor')
        self.assertEqual((readings[0]['start'], readings[0]['end']), (5.2, 5.3))
        self.assertNotIn('baseline_readings', metadata)
        self.assertEqual(run.metadata, original)

    def test_baseline_failure_is_optional_and_single_reading_is_not_an_end_readback(self):
        run = scan_run()
        baseline = Stream(xr.Dataset({'reactor_power_6': ('event', [0.0])}),
                          {'reactor_power_6': {'dtype': 'number', 'shape': []}})
        run['baseline'] = baseline
        detail = read_metadata_details(run, run.metadata)
        reading, = environment_readings(detail)
        self.assertEqual(reading['start'], 0)
        self.assertIsNone(reading['end'])
        self.assertEqual(reading['units'], '')
        with patch.object(baseline, 'read', side_effect=RuntimeError('Baseline offline')):
            detail = read_metadata_details(run, run.metadata)
        self.assertIn('Baseline offline', detail['viewer_details']['notes'][0])
        self.assertEqual(environment_readings(detail), [])


class MetadataViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return
            QtTest.QTest.qWait(10)
        self.fail('Viewer did not finish loading')

    def close_window(self, window):
        window.close()
        self.app.processEvents()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_status_search_overview_and_nested_metadata_search(self):
        run = scan_run()
        run.metadata['stop'].update(exit_status='fail', reason='motor timeout')
        run.metadata['start']['plan_args'] = {'motor': {'name': 'theta', 'limit': 42}}
        window = DiffractionAnalysisWindow({'r': Catalog(scan=run)}, catalogs={'Runs': 'r'})
        try:
            self.wait_for(lambda: window.data_view.snapshot is not None)
            self.assertIn('Failed', window.run_list.item(0).text())
            self.assertIn('motor timeout', window.run_list.item(0).toolTip())
            self.assertEqual(window.details_tabs.tabText(1), 'Overview')
            self.assertEqual(window._overview['Run']['Reason'], 'motor timeout')
            window.copy_overview_button.click()
            self.assertIn('Status: Failed', self.app.clipboard().text())
            window.search_edit.setText('failed')
            self.assertFalse(window.run_list.item(0).isHidden())
            window.search_edit.setText('complete')
            self.assertTrue(window.run_list.item(0).isHidden())
            window.metadata_filter.setText('42')
            start = window.metadata_tree.topLevelItem(0)
            self.assertEqual(start.text(0), 'start')
            self.assertFalse(start.isHidden())
            children = {start.child(i).text(0): start.child(i) for i in range(start.childCount())}
            self.assertFalse(children['plan_args'].isHidden())
            self.assertTrue(children['uid'].isHidden())
            window.metadata_filter.clear()
            self.assertFalse(children['uid'].isHidden())
        finally:
            self.close_window(window)

    def test_reset_view_and_hidden_initial_load_use_full_data_preserving_window(self):
        panel = AnalysisPanel()
        try:
            snapshot = build_history_snapshot(scan_run())
            panel.set_snapshot(snapshot)
            # Reproduce the old bug: clipping retains a tiny displayed subset,
            # and autoRange on the next load used that subset as its bounds.
            panel.plot.setRange(xRange=(-0.01, 0.01), yRange=(70, 80), padding=0)
            self.app.processEvents()
            panel.set_snapshot(snapshot)
            panel.show()
            QtTest.QTest.qWait(50)
            self.app.processEvents()
            x, y = panel.curve['x'], panel.curve['y']
            bounds = panel.plot.viewRange()
            self.assertLessEqual(bounds[0][0], x.min())
            self.assertGreaterEqual(bounds[0][1], x.max())
            self.assertLessEqual(bounds[1][0], y.min())
            self.assertGreaterEqual(bounds[1][1], y.max())
            panel.min_spin.setValue(-1)
            panel.max_spin.setValue(1)
            fit = fit_curve(panel.curve['x'], panel.curve['y'], FitConfig(xmin=-1, xmax=1))
            panel.show_fit(fit)
            batch = panel.batch
            panel.plot.setRange(xRange=(-0.01, 0.01), yRange=(70, 80), padding=0)
            changed = QtTest.QSignalSpy(panel.context_changed)
            QtTest.QTest.mouseDClick(panel.plot.viewport(), QtCore.Qt.LeftButton)
            np.testing.assert_allclose(panel.region.getRegion(), [-1, 1])
            self.assertEqual(len(changed), 0)
            self.assertIs(panel.fit_result, fit)
            self.assertIs(panel.batch, batch)
            self.assertLessEqual(panel.plot.viewRange()[0][0], -5)
            self.assertGreaterEqual(panel.plot.viewRange()[0][1], 5)
            panel.plot.setXRange(-0.01, 0.01, padding=0)
            QtTest.QTest.mouseDClick(panel.residual_plot.viewport(), QtCore.Qt.LeftButton)
            self.assertLessEqual(panel.plot.viewRange()[0][0], -5)
            self.assertEqual(len(changed), 0)
            self.assertIs(panel.fit_result, fit)
        finally:
            panel.close()
            panel.results_widget.close()
            panel.deleteLater()
            panel.results_widget.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_run_overlay_and_parameter_plot_double_click_restore_both_axes(self):
        viewer = RunDataPanel()
        trends = FitTrends()
        try:
            viewer.set_snapshot(build_history_snapshot(scan_run()))
            viewer.show()
            trends.show()
            self.app.processEvents()
            viewer.plot.setRange(xRange=(-0.01, 0.01), yRange=(70, 80), padding=0)
            QtTest.QTest.mouseDClick(viewer.plot.viewport(), QtCore.Qt.LeftButton)
            limits = viewer.plot.viewRange()
            self.assertLessEqual(limits[0][0], -5)
            self.assertGreaterEqual(limits[0][1], 5)
            self.assertLessEqual(limits[1][0], min(curve['y'].min() for curve in viewer.curves))
            self.assertGreaterEqual(limits[1][1], max(curve['y'].max() for curve in viewer.curves))
            self.assertTrue(viewer.show_all.isChecked())
            trends.line.setData([-2, 2], [-3, 3])
            trends.plot.setRange(xRange=(-0.01, 0.01), yRange=(-0.01, 0.01), padding=0)
            QtTest.QTest.mouseDClick(trends.plot.viewport(), QtCore.Qt.LeftButton)
            limits = trends.plot.viewRange()
            self.assertLessEqual(limits[0][0], -2)
            self.assertGreaterEqual(limits[0][1], 2)
            self.assertLessEqual(limits[1][0], -3)
            self.assertGreaterEqual(limits[1][1], 3)
        finally:
            for widget in (viewer, trends, trends.table):
                widget.close()
                widget.deleteLater()
            self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
