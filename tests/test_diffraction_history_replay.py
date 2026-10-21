import unittest
from unittest.mock import patch

import numpy as np
from qtpy.QtWidgets import QApplication

from diffractometer_controls.screens.diffraction import diffraction_live_plot as live
from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph


class DiffractionHistoryReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def flush(self):
        self.app.processEvents()
        self.app.processEvents()

    def test_both_plotters_replay_psd_then_scalar_without_hardware(self):
        for widget_class in (DiffractionPlotWidgetPyQtGraph, live.DiffractionPlotWidget):
            with self.subTest(plotter=widget_class.__name__), \
                 patch.object(live, "PV", side_effect=AssertionError("history accessed a PV")), \
                 patch.object(live, "caget", side_effect=AssertionError("history accessed a PV")):
                widget = widget_class()
                controller = live.DiffractionLivePlot(widget)
                try:
                    snapshot = {
                        "uid": "psd", "start": {"uid": "psd", "experiment_type": "diffraction", "plan_name": "count_he3"},
                        "stop": {"exit_status": "success"}, "descriptor_docs": [], "seq_nums": [2, 4],
                        "rows": [{"psd_counts": [1, 2, 3], "psd_position_x": [-1, 0, 1]},
                                 {"psd_counts": [4, 5, 6], "psd_position_x": [-1, 0, 1]}],
                    }
                    controller.show_history_snapshot(snapshot)
                    self.flush()
                    self.assertEqual(widget._summary_y["psd"], [6, 15])
                    self.assertEqual(len(widget._profile_history["psd"]), 2)
                    self.assertFalse(controller._live_subscriptions)
                    snapshot = {
                        "uid": "scalar", "start": {"uid": "scalar", "data_type": "scalar", "plan_name": "scan_scalar",
                            "experiment_type": "diffraction", "motors": ["theta"],
                            "live_plot_fields": {"temperature": {"label": "Temperature", "units": "C", "transport": "ca", "pv": "DO:NOT:CONNECT"}}},
                        "stop": {"exit_status": "abort", "reason": "operator stopped"}, "descriptor_docs": [],
                        "seq_nums": [1, 2], "rows": [{"temperature": 295, "theta": 0.5}, {"temperature": 296, "theta": 1.5}],
                    }
                    controller.show_history_snapshot(snapshot)
                    self.flush()
                    self.assertEqual(widget._summary_y["Temperature"], [295, 296])
                    np.testing.assert_array_equal(widget._summary_x["Temperature"], [0.5, 1.5])
                    self.assertEqual(widget._plot_mode, "scalar")
                    self.assertIn("abort", controller._history_status_text)
                    self.assertFalse(controller._live_subscriptions)
                finally:
                    controller.shutdown()
                    widget.close()
                    widget.deleteLater()
                    self.flush()

    def test_gui_uses_login_client_without_reopening_cached_tokens(self):
        widget = DiffractionPlotWidgetPyQtGraph()
        controller = live.DiffractionLivePlot(widget)
        client = {"testdb": object()}
        try:
            with patch.object(live, "ensure_tiled_login", return_value=client), \
                 patch.object(controller, "_refresh_history_async"), \
                 patch.object(live, "get_bluesky_active_mode", return_value="test"), \
                 patch.object(live, "_build_tiled_client", side_effect=AssertionError("discarded login")):
                controller._on_tiled_enabled_toggled(True)
                catalog, name, mode = controller._ensure_history_catalog()
                self.assertIs(catalog, client["testdb"])
                self.assertEqual((name, mode), ("testdb", "test"))
        finally:
            controller.shutdown()
            widget.close()

    def test_previous_and_next_buttons_follow_newest_first_order(self):
        plot = DiffractionPlotWidgetPyQtGraph()
        widget = live.DiffractionHistoryViewer(plot)
        try:
            state = {"entries": [{"uid": "new", "label": "New"}, {"uid": "old", "label": "Old"}],
                     "selected_uid": "new", "history_enabled": True, "tiled_enabled": True}
            widget.set_history_state(state)
            self.assertTrue(widget._previous_button.isEnabled())
            self.assertFalse(widget._next_button.isEnabled())
            widget.set_history_state({**state, "selected_uid": "old"})
            self.assertFalse(widget._previous_button.isEnabled())
            self.assertTrue(widget._next_button.isEnabled())
        finally:
            widget.close()
