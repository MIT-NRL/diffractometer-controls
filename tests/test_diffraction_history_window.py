import json
import threading
import time
import unittest
from unittest.mock import patch

import xarray as xr
from qtpy import QtCore, QtTest, QtWidgets

from diffractometer_controls.screens.diffraction import diffraction_live_plot as live
from diffractometer_controls.screens.diffraction.diffraction_history_window import DiffractionHistoryWindow
from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph
from tests.test_diffraction_history import Run, Stream, psd_stream


class Catalog(dict):
    def search(self, query):
        return Catalog((key, run) for key, run in self.items()
                       if run.metadata['start'].get('experiment_type') == 'diffraction')

    def sort(self, sorting):
        return Catalog(sorted(self.items(), key=lambda pair: pair[1].metadata['start']['time'], reverse=True))


def make_psd(uid, timestamp=1):
    run = Run(psd_stream())
    run.metadata['start'].update(uid=uid, time=timestamp, plan_name='count_he3', title='PSD scan', scan_id=timestamp)
    return run


def make_scalar(uid):
    run = Run(Stream(xr.Dataset({'temperature': ('event', [295, 296]), 'theta': ('event', [0.5, 1.5])}), {
        'temperature': {'dtype': 'number', 'shape': [], 'object_name': 'temperature', 'units': 'C'},
        'theta': {'dtype': 'number', 'shape': [], 'object_name': 'theta', 'units': 'deg'},
    }), scalar=True)
    run.metadata['start'].update(uid=uid, time=0, plan_name='scan_scalar', motors=['theta'], title='Temperature', sample='copper')
    return run


class DiffractionHistoryWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                self.app.processEvents()
                return
            QtTest.QTest.qWait(10)
        self.fail('Timed out waiting for analysis viewer')

    def close_window(self, window):
        window.close()
        self.app.processEvents()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_window_loads_psd_and_scalar_and_filters_out_imaging(self):
        imaging = make_psd('imaging', 5)
        imaging.metadata['start']['experiment_type'] = 'imaging'
        client = {'4dh4': Catalog(psd=make_psd('psd', 2), scalar=make_scalar('scalar'), imaging=imaging)}
        with patch.object(live, 'PV', side_effect=AssertionError('Historical view connected to hardware')):
            window = DiffractionHistoryWindow(client)
            try:
                self.wait_for(lambda: window.analysis.curve is not None and window.analysis.curve['field'] == 'psd_counts')
                self.assertEqual(window.run_list.count(), 2)
                self.assertEqual(window.analysis.curve['y'].tolist(), [1, 2, 3])
                window.analysis.point_combo.setCurrentIndex(1)
                self.assertEqual(window.analysis.curve['y'].tolist(), [4, 5, 6])
                self.assertEqual(json.loads(window.metadata_json.toPlainText())['start']['uid'], 'psd')
                self.assertIs(window.details_splitter.widget(0), window.workspace_tabs)
                window.run_list.setCurrentRow(1)
                self.wait_for(lambda: window.analysis.curve is not None and window.analysis.curve['field'] == 'temperature')
                self.assertEqual(window.analysis.curve['y'].tolist(), [295, 296])
                self.assertEqual(window.analysis.curve['x'].tolist(), [0.5, 1.5])
                window.search_edit.setText('copper')
                self.assertTrue(window.run_list.item(0).isHidden())
                self.assertFalse(window.run_list.item(1).isHidden())
                self.assertIn('1 shown / 2 loaded', window.count_label.text())
            finally:
                self.close_window(window)

    def test_loading_failure_keeps_metadata_and_clears_previous_plot(self):
        run = make_psd('broken', 2)
        run.metadata['file_export'] = {'status': 'failed', 'error': 'disk offline'}
        window = DiffractionHistoryWindow({'4dh4': Catalog(broken=run)})
        try:
            self.wait_for(lambda: 'failed' in window.statusBar().currentMessage())
            self.assertIsNone(window.analysis.curve)
            self.assertEqual(json.loads(window.metadata_json.toPlainText())['file_export']['error'], 'disk offline')
        finally:
            self.close_window(window)

    def test_scalar_export_failure_keeps_the_data_plot_and_error_metadata(self):
        run = make_scalar('scalar-export-failed')
        run.metadata['file_export'] = {'status': 'failed', 'error': 'CSV disk offline'}
        window = DiffractionHistoryWindow({'4dh4': Catalog(scalar=run)})
        try:
            self.wait_for(lambda: window.analysis.curve is not None)
            self.assertEqual(window.analysis.curve['y'].tolist(), [295, 296])
            self.assertEqual(window.analysis.curve['x'].tolist(), [0.5, 1.5])
            self.assertEqual(json.loads(window.metadata_json.toPlainText())['file_export']['error'], 'CSV disk offline')
            self.assertTrue(window.data_view.plot._message.isHidden())
        finally:
            self.close_window(window)

    def test_load_more_keeps_initial_limit_at_100(self):
        catalog = Catalog((str(i), make_psd(str(i), i)) for i in range(101))
        window = DiffractionHistoryWindow({'4dh4': catalog})
        try:
            self.wait_for(lambda: window.run_list.count() == 100)
            self.assertTrue(window.load_more_button.isEnabled())
            window._load_more()
            self.wait_for(lambda: window.run_list.count() == 101)
            self.assertFalse(window.load_more_button.isEnabled())
        finally:
            self.close_window(window)

    def test_run_read_failure_without_metadata_keeps_error_overlay_and_selection(self):
        with patch.object(DiffractionHistoryWindow, '_run_task', side_effect=ConnectionError('Catalog offline')):
            window = DiffractionHistoryWindow({'4dh4': Catalog(unavailable=make_psd('unavailable'))})
            try:
                self.wait_for(lambda: 'Catalog offline' in window.statusBar().currentMessage())
                self.assertIsNone(window.analysis.curve)
                self.assertEqual(window.run_list.currentItem().data(QtCore.Qt.UserRole), 'unavailable')
                self.assertIn('Catalog offline', window.data_view.plot._message.text())
                self.assertFalse(window.data_view.plot._message.isHidden())
            finally:
                self.close_window(window)

    def test_catalog_switch_discards_inflight_old_run(self):
        started = threading.Event()
        release = threading.Event()
        original = DiffractionHistoryWindow._run_task

        def slow_load(self, catalog, uid):
            if uid == 'slow':
                started.set()
                release.wait(5)
            return original(self, catalog, uid)

        with patch.object(DiffractionHistoryWindow, '_run_task', new=slow_load):
            window = DiffractionHistoryWindow({'4dh4': Catalog(slow=make_psd('slow')), 'testdb': Catalog(fast=make_scalar('fast'))})
            try:
                self.wait_for(started.is_set)
                window.catalog_combo.setCurrentIndex(1)
                self.wait_for(lambda: window.analysis.curve is not None and window.analysis.curve['field'] == 'temperature')
                release.set()
                self.wait_for(lambda: not window._futures)
                self.assertEqual(json.loads(window.metadata_json.toPlainText())['start']['uid'], 'fast')
                self.assertEqual(window.analysis.curve['field'], 'temperature')
            finally:
                release.set()
                self.close_window(window)

    def test_close_during_load_shuts_down_and_ignores_completion(self):
        release = threading.Event()
        started = threading.Event()
        original = DiffractionHistoryWindow._run_task

        def slow_load(self, catalog, uid):
            started.set()
            release.wait(5)
            return original(self, catalog, uid)

        with patch.object(DiffractionHistoryWindow, '_run_task', new=slow_load):
            window = DiffractionHistoryWindow({'4dh4': Catalog(slow=make_psd('slow'))})
            try:
                self.wait_for(started.is_set)
                window.shutdown()
                release.set()
                self.wait_for(lambda: all(future.done() for future in window._futures))
                self.assertTrue(window._closed)
                self.assertTrue(window._executor._shutdown)
            finally:
                release.set()
                self.close_window(window)

    def test_embedded_button_opens_one_independent_popout(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        viewer = live.DiffractionHistoryViewer(plot)
        controller = live.DiffractionLivePlot(viewer)
        controller._history_client = {'4dh4': Catalog(psd=make_psd('psd'))}
        try:
            with patch.object(live, 'get_bluesky_active_mode', return_value='production'):
                viewer._analysis_button.click()
                window = controller._analysis_viewer
                self.wait_for(lambda: window.analysis.curve is not None)
                viewer._analysis_button.click()
                self.assertIs(controller._analysis_viewer, window)
                self.assertTrue(controller._live_follow_enabled)
                self.assertEqual(plot._summary_y, {})
                self.assertEqual(window.analysis.curve['y'].tolist(), [1, 2, 3])
                window.analysis.point_combo.setCurrentIndex(1)
                self.assertEqual(window.analysis.curve['y'].tolist(), [4, 5, 6])
        finally:
            controller.shutdown()
            viewer.close()
            self.app.processEvents()
