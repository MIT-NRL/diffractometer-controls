"""A missing or failed run must remain a navigable history selection."""

import concurrent.futures
import time
import unittest
from unittest.mock import patch

from qtpy import QtCore, QtTest, QtWidgets

from diffraction_analysis.history import build_history_snapshot, fetch_recent_history_entries
from diffractometer_controls.screens.diffraction import diffraction_live_plot as live
from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph
from tests.test_diffraction_history_window import Catalog, make_psd


class HistoryFailureNavigationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def wait_for(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.app.processEvents()
            self.app.processEvents()
            if predicate():
                return
            QtTest.QTest.qWait(10)
        self.fail('History selection timed out')

    def test_arrows_cross_acquisition_and_storage_failures_and_restore_plot(self):
        for failure in ('acquisition', 'storage'):
            with self.subTest(failure=failure):
                bad = make_psd('bad', 2)
                if failure == 'acquisition':
                    bad.metadata['stop'].update(exit_status='fail', reason='detector timeout')
                else:
                    bad.metadata['file_export'] = {'status': 'failed', 'error': 'Tiled delivery failed'}
                catalog = Catalog(new=make_psd('new', 3), bad=bad, old=make_psd('old', 1))
                plot = DiffractionPlotWidgetPyQtGraph()
                widget = live.DiffractionHistoryViewer(plot)
                controller = live.DiffractionLivePlot(widget)
                controller._tiled_enabled = True
                controller._history_catalog = catalog
                controller._history_entries = fetch_recent_history_entries(catalog)
                controller._history_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                try:
                    with patch.object(controller, '_ensure_history_catalog', return_value=(catalog, 'runs', 'test')):
                        controller.show_history_snapshot(build_history_snapshot(catalog['new']))
                        self.wait_for(lambda: widget._run_combo.currentData() == 'new')
                        self.assertTrue(plot._summary_y)
                        widget._previous_button.click()
                        self.wait_for(lambda: bool(widget._history_message.text()))
                        self.assertEqual(widget._run_combo.currentData(), 'bad')
                        self.assertEqual(controller._current_selected_history_uid(), 'bad')
                        self.assertEqual(plot._summary_y, {})
                        self.assertFalse(widget._run_combo.itemIcon(1).isNull())
                        self.assertIn('unavailable', widget._history_message.text())
                        self.assertTrue(widget._previous_button.isEnabled())
                        self.assertTrue(widget._next_button.isEnabled())
                        widget._previous_button.click()
                        self.wait_for(lambda: controller._displayed_run_uid == 'old' and bool(plot._summary_y))
                        self.assertEqual(widget._run_combo.currentData(), 'old')
                        self.assertEqual(widget._history_message.text(), '')
                        self.assertEqual(controller._history_entry_for_uid('old')['status'], 'Complete')
                        widget._next_button.click()
                        self.wait_for(lambda: bool(widget._history_message.text()))
                        widget._next_button.click()
                        self.wait_for(lambda: controller._displayed_run_uid == 'new' and bool(plot._summary_y))
                        self.assertEqual(widget._run_combo.currentData(), 'new')
                finally:
                    controller.shutdown()
                    widget.close()
                    widget.deleteLater()
                    self.app.processEvents()
                    self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_exception_callback_preserves_uid_and_request_kind(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        widget = live.DiffractionHistoryViewer(plot)
        controller = live.DiffractionLivePlot(widget)
        try:
            future = concurrent.futures.Future()
            future.set_exception(ConnectionError('Connection refused'))
            result = QtTest.QSignalSpy(controller._history_load_completed)
            controller._history_load_token = 9
            controller._emit_history_load_result(8, future, uid='requested', request_kind='latest')
            self.assertEqual(result[0][1]['uid'], 'requested')
            self.assertEqual(result[0][1]['request_kind'], 'latest')
            self.assertIn('Connection refused', result[0][1]['error'])
        finally:
            controller.shutdown()
            widget.close()
