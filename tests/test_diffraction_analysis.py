import json
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from qtpy import QtCore, QtTest, QtWidgets

from diffraction_analysis.curves import select_curve, signal_fields
from diffraction_analysis.history import build_history_snapshot
from diffraction_analysis.window import DiffractionAnalysisWindow
from tests.test_diffraction_history import Run, Stream
from tests.test_diffraction_history_window import Catalog


def generic_run(uid='generic'):
    x = np.linspace(-6, 6, 151)
    y = 8 + 70 * np.exp(-0.5 * ((x - 0.8) / 0.6) ** 2)
    stream = Stream(xr.Dataset({'intensity': (('event','bin'), [y, y * 2]), 'angle': (('event','bin'), [x, x])}), {
        'intensity': {'dtype': 'array', 'shape': [len(x)], 'units': 'counts'},
        'angle': {'dtype': 'array', 'shape': [len(x)], 'units': 'deg'},
    })
    run = Run(stream)
    run.metadata['start'].update(uid=uid, data_type='1d', plan_name='generic_measurement',
        live_plot_fields={'intensity': {'role':'signal', 'label':'Intensity'}, 'angle': {'role':'coordinate'}})
    return run


class DiffractionAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait_for(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                self.app.processEvents()
                return
            QtTest.QTest.qWait(10)
        self.fail('Analysis operation timed out')

    def window(self):
        window = DiffractionAnalysisWindow({'measurement-runs': Catalog(generic=generic_run())}, catalogs={'Beamline': 'measurement-runs'})
        self.wait_for(lambda: window.analysis.curve is not None)
        window.workspace_tabs.setCurrentIndex(1)
        window.analysis.axis_combo.setCurrentIndex(window.analysis.axis_combo.findData('angle'))
        return window

    def close(self, window):
        window.close()
        self.app.processEvents()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_generic_view_has_no_gauges_or_autofit_and_fits_on_request(self):
        window = self.window()
        try:
            panel = window.analysis
            self.assertIsNone(panel.fit_result)
            self.assertEqual(panel.curve['field'], 'intensity')
            self.assertEqual(panel.curve['x_units'], 'deg')
            self.assertFalse(any('Gauge' in type(widget).__name__ or 'ScalarCountReadout' == type(widget).__name__ for widget in window.findChildren(QtWidgets.QWidget)))
            panel.fit_button.click()
            self.wait_for(lambda: panel.fit_result is not None)
            self.assertAlmostEqual(panel.fit_result['parameters']['p1_center']['value'], 0.8, places=4)
            self.assertTrue(panel.results.rowCount() > 3)
            self.assertEqual(window.details_tabs.currentIndex(), 0)
            panel.copy_button.click()
            copied = json.loads(self.app.clipboard().text())
            self.assertEqual(copied['uid'], 'generic')
            self.assertEqual(copied['x_field'], 'angle')
            panel.min_spin.setValue(-2)
            self.assertIsNone(panel.fit_result)
            self.assertIsNone(panel.fit_line.xData)
            self.assertEqual(panel.region.getRegion()[0], -2)
        finally:
            self.close(window)

    def test_parameter_or_point_change_invalidates_inflight_fit(self):
        release, started = threading.Event(), threading.Event()
        original = DiffractionAnalysisWindow._fit_task

        def slow_fit(curve, config):
            started.set()
            release.wait(5)
            return original(curve, config)

        with patch.object(DiffractionAnalysisWindow, '_fit_task', new=staticmethod(slow_fit)):
            window = self.window()
            try:
                panel = window.analysis
                panel.fit_button.click()
                self.wait_for(started.is_set)
                panel.point_combo.setCurrentIndex(1)
                release.set()
                self.wait_for(lambda: all(future.done() for future in window._futures))
                self.assertIsNone(panel.fit_result)
                self.assertEqual(panel.curve['point'], 1)
            finally:
                release.set()
                self.close(window)

    def test_invalid_fit_window_keeps_data_and_reports_error(self):
        window = self.window()
        try:
            panel = window.analysis
            panel.min_spin.setValue(4)
            panel.max_spin.setValue(-4)
            panel.fit_button.click()
            self.wait_for(lambda: 'Fit failed:' in panel.message.text())
            self.assertIsNone(panel.fit_result)
            self.assertIsNotNone(panel.curve)
        finally:
            self.close(window)

    def test_curves_support_scalar_axes_and_missing_values(self):
        snapshot = {'uid': 'scalar', 'start': {'data_type':'scalar', 'motors':['temperature'],
                     'live_plot_fields': {'intensity':{'role':'signal'}}},
                    'rows':[{'intensity': 5, 'temperature': 1}, {'intensity': None, 'temperature': 2}, {'intensity':9,'temperature':3}],
                    'seq_nums':[1,2,3], 'descriptor_docs':[]}
        self.assertEqual(signal_fields(snapshot), ['intensity'])
        curve = select_curve(snapshot, 'intensity', x_field='temperature')
        np.testing.assert_array_equal(curve['x'], [1,3])
        np.testing.assert_array_equal(curve['y'], [5,9])

    def test_generic_1d_loader_does_not_require_instrument_field_names(self):
        run = generic_run()
        run.metadata['start'].pop('live_plot_fields')
        snapshot = build_history_snapshot(run)
        self.assertEqual(snapshot['rows'][0]['intensity'][0], run['primary'].dataset['intensity'].values[0, 0])

    def test_generic_scalar_loader_can_infer_numeric_signals_without_detector_names(self):
        run = Run(Stream(xr.Dataset({'voltage': ('event', [0.1,0.2]), 'angle': ('event',[1,2])}), {
            'voltage': {'dtype':'number','shape':[], 'units':'V'},
            'angle': {'dtype':'number','shape':[], 'units':'deg'},
        }), scalar=True)
        run.metadata['start'].pop('detectors')
        run.metadata['start']['motors']=['angle']
        snapshot=build_history_snapshot(run)
        self.assertEqual(signal_fields(snapshot), ['voltage'])
