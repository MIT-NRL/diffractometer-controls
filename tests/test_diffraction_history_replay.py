import unittest
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qtpy.QtWidgets import QApplication
from qtpy import QtTest

from diffractometer_controls.screens.diffraction import diffraction_live_plot as live
from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph


class DiffractionHistoryReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def flush(self):
        self.app.processEvents()

    def test_finished_scalar_plot_survives_late_running_item_and_navigation(self):
        for widget_class in (DiffractionPlotWidgetPyQtGraph, live.DiffractionPlotWidget):
            with self.subTest(plotter=widget_class.__name__), patch.object(live, "PV", None), \
                 patch.object(live, "_safe_caget", return_value=None):
                widget = widget_class()
                item = {"item_uid": "queued-plan", "name": "count_scalar", "kwargs": {"detectors": ["counter"]}}
                client = SimpleNamespace(_running_item=item)
                controller = live.DiffractionLivePlot(widget, re_client=client)
                try:
                    start = {"uid": "finished", "experiment_type": "diffraction",
                             "plan_name": "count_scalar", "data_type": "scalar", "detectors": ["counter"]}
                    descriptor = {"uid": "primary", "run_start": "finished", "name": "primary",
                                  "data_keys": {"counter": {"dtype": "number", "shape": []}}}
                    controller.on_document("start", start)
                    controller.on_document("descriptor", descriptor)
                    controller.on_document("event", {"descriptor": "primary", "seq_num": 2, "data": {"counter": 20}})
                    controller._apply_live_backfill_snapshot({"start": start, "descriptor_docs": [descriptor],
                        "seq_nums": [1], "rows": [{"counter": 10}]})
                    controller.on_document("stop", {"run_start": "finished", "exit_status": "success"})
                    self.flush()
                    self.assertEqual(widget._summary_y["counter"], [10, 20])

                    # Run-list and queue polls can still report the old plan
                    # after its Stop and the status PV's transition to idle.
                    controller._on_running_item_changed(SimpleNamespace(running_item=item))
                    controller.deactivate()
                    controller.activate()
                    self.flush()
                    self.assertEqual(controller._run_uid, "finished")
                    self.assertFalse(controller._provisional_live_context)
                    self.assertEqual(widget._summary_y["counter"], [10, 20])
                    self.assertFalse(controller._live_subscriptions)

                    # The status PV can also lag behind the Stop document.
                    with patch.object(live, "_safe_caget", side_effect=lambda pv, **kw:
                                      "RUNNING" if pv == live.RUN_STATE_STATUS_PV else "finished"):
                        controller._on_running_item_changed(SimpleNamespace(running_item=item))
                        self.flush()
                        self.assertFalse(controller._live_run_active)
                        self.assertEqual(controller._current_live_run_uid, "")
                        self.assertEqual(widget._summary_y["counter"], [10, 20])

                    # The next actual run, even of the same plan, still resets.
                    controller.on_document("start", {**start, "uid": "next"})
                    controller.on_document("descriptor", {**descriptor, "uid": "next-primary", "run_start": "next"})
                    controller.on_document("event", {"descriptor": "next-primary", "seq_num": 1, "data": {"counter": 30}})
                    self.flush()
                    self.assertEqual(widget._summary_y["counter"], [30])
                finally:
                    controller.shutdown()
                    widget.close()

    def test_delayed_automatic_history_refresh_keeps_finished_scalar_run(self):
        with patch.object(live, "PV", None), patch.object(live, "_safe_caget", return_value=None):
            widget = DiffractionPlotWidgetPyQtGraph()
            controller = live.DiffractionLivePlot(widget)
            try:
                start = {"uid": "finished", "experiment_type": "diffraction",
                         "plan_name": "count_scalar", "data_type": "scalar", "detectors": ["counter"]}
                controller.on_document("start", start)
                controller._apply_live_backfill_snapshot({"start": start, "seq_nums": [1, 2],
                                                         "rows": [{"counter": 10}, {"counter": 20}]})
                controller.on_document("stop", {"run_start": "finished", "exit_status": "success"})
                controller._tiled_enabled = True
                # This automatic read began while the run was active, but
                # arrives after Stop; Tiled may still list a previous run.
                with patch.object(controller, "_request_history_load") as load:
                    controller._on_history_refresh_completed(controller._history_refresh_token,
                        {"autoload_live": True, "entries": [{"uid": "previous"}]})
                    self.flush()
                    load.assert_not_called()
                self.assertEqual(controller._displayed_run_uid, "finished")
                self.assertEqual(widget._summary_y["counter"], [10, 20])
            finally:
                controller.shutdown()
                widget.close()

    def test_live_update_during_history_redraw_remains_the_newest_preview(self):
        for scalar in (False, True):
            with self.subTest(scalar=scalar), patch.object(live, "PV", None):
                widget = DiffractionPlotWidgetPyQtGraph()
                controller = live.DiffractionLivePlot(widget)
                try:
                    start = {"uid": "running", "experiment_type": "diffraction",
                             "plan_name": "count_scalar" if scalar else "count_he3psd",
                             "detectors": ["counter" if scalar else "psd"]}
                    controller.on_document("start", start)
                    def update(value):
                        if scalar:
                            controller._handle_live_scalar_update("counter", value)
                        else:
                            controller._handle_live_counts_update("psd", [value, value + 1])
                    update(10)
                    render_row = controller._render_history_row
                    def render_with_new_live_value(**kwargs):
                        render_row(**kwargs)
                        update(30)
                    with patch.object(controller, "_render_history_row", side_effect=render_with_new_live_value):
                        controller._apply_live_backfill_snapshot({"start": start, "seq_nums": [1],
                            "rows": [{"counter": 5} if scalar else {"psd_counts": [5, 6]}]})
                    self.flush()
                    if scalar:
                        self.assertEqual(controller._live_preview_values["counter"][1], 30)
                        self.assertEqual(widget._summary_live_lines["counter"].getData()[1][-1], 30)
                    else:
                        np.testing.assert_array_equal(controller._live_preview_profiles["psd"][1], [30, 31])
                        np.testing.assert_array_equal(widget._live_profile_lines["psd"].getData()[1], [30, 31])
                finally:
                    controller.shutdown()
                    widget.close()

    def test_midrun_psd_survives_history_failure_and_keeps_monitor_updates_after_backfill(self):
        for widget_class in (DiffractionPlotWidgetPyQtGraph, live.DiffractionPlotWidget):
            with self.subTest(plotter=widget_class.__name__), patch.object(live, "PV", None), \
                 patch.object(live, "_safe_caget", side_effect=lambda pv, **kw: "RUNNING" if pv == live.RUN_STATE_STATUS_PV else "running"):
                widget = widget_class()
                client = SimpleNamespace(_running_item={"name": "count_he3psd", "kwargs": {"detectors": ["psd"], "num_exposures": 10}})
                controller = live.DiffractionLivePlot(widget, re_client=client)
                try:
                    controller._arm_current_running_item()
                    self.assertEqual(controller._run_uid, "running")
                    controller._live_backfill_pending = True
                    # No Start or descriptor was received by this GUI.
                    monitor = {"descriptor": "missed-monitor", "seq_num": 1,
                               "data": {"psd_counts": [8, 9, 10]}}
                    controller.on_document("event", monitor)
                    controller._on_history_load_completed(controller._history_load_token,
                        {"uid": "running", "request_kind": "startup", "error": "Tiled offline"})
                    self.flush()
                    self.assertIn("psd", widget._live_profile_lines)
                    self.assertEqual(controller._rendered_primary_seq_nums, set())

                    start = {"uid": "running", "plan_name": "count_he3psd", "experiment_type": "diffraction", "detectors": ["psd"]}
                    descriptor = {"uid": "primary", "run_start": "running", "name": "primary",
                                  "data_keys": {"psd_counts": {"dtype": "array", "shape": [3]}}}
                    snapshot = {"start": start, "descriptor_docs": [descriptor], "seq_nums": [1],
                                "rows": [{"psd_counts": [1, 2, 3]}]}
                    controller._on_history_load_completed(controller._history_load_token,
                        {"uid": "running", "request_kind": "startup", "snapshot": snapshot})
                    self.flush()
                    self.assertEqual(widget._summary_y["psd"], [6])
                    self.assertIn("psd", widget._live_profile_lines)
                    np.testing.assert_array_equal(controller._live_preview_profiles["psd"][1], [8, 9, 10])

                    # The monitor's sequence 1 must not collide with primary 1.
                    controller.on_document("event", {**monitor, "data": {"psd_counts": [20, 21, 22]}})
                    self.flush()
                    np.testing.assert_array_equal(controller._live_preview_profiles["psd"][1], [20, 21, 22])
                    controller.on_document("event", {"descriptor": "primary", "seq_num": 2,
                                                       "data": {"psd_counts": [30, 31, 32]}})
                    self.flush()
                    self.assertEqual(widget._summary_y["psd"], [6, 93])
                    self.assertFalse(controller._live_preview_profiles)
                    controller.on_document("event", {**monitor, "data": {"psd_counts": [40, 41, 42]}})
                    controller._apply_live_backfill_snapshot(snapshot)
                    self.flush()
                    self.assertEqual(widget._summary_y["psd"], [6, 93])
                    line = widget._live_profile_lines["psd"]
                    plotted_y = line.getData()[1] if isinstance(widget, DiffractionPlotWidgetPyQtGraph) else line.get_ydata()
                    np.testing.assert_array_equal(plotted_y, [40, 41, 42])
                finally:
                    controller.shutdown()
                    widget.close()

    def test_midrun_scalar_preview_is_independent_of_tiled_and_preserved_on_replay(self):
        with patch.object(live, "PV", None), patch.object(live, "_safe_caget", return_value=None):
            widget = DiffractionPlotWidgetPyQtGraph()
            client = SimpleNamespace(_running_item={"name": "count_scalar", "kwargs": {"detectors": ["counter"]}})
            controller = live.DiffractionLivePlot(widget, re_client=client)
            try:
                controller._arm_current_running_item()
                controller.on_document("event", {"descriptor": "missed-monitor", "seq_num": 1, "data": {"counter": 12}})
                self.flush()
                self.assertEqual(controller._live_preview_values["counter"], (1, 12))
                self.assertIn("counter", widget._summary_live_lines)
                start = {"uid": "running", "experiment_type": "diffraction", "plan_name": "count_scalar", "detectors": ["counter"],
                         "live_plot_fields": {"counter": {"label": "Beam monitor", "transport": "document", "role": "signal"}}}
                controller._current_live_run_uid = "running"
                controller._on_history_load_completed(controller._history_load_token,
                    {"uid": "running", "request_kind": "startup", "snapshot": {"start": start, "rows": []}})
                self.flush()
                self.assertIn("Beam monitor", widget._summary_live_lines)
                self.assertNotIn("counter", widget._summary_live_lines)
                controller._apply_live_backfill_snapshot({"start": start, "seq_nums": [1], "rows": [{"counter": 5}]})
                self.flush()
                np.testing.assert_array_equal(widget._summary_live_lines["Beam monitor"].getData()[1], [5, 12])
                controller.on_document("event", {"descriptor": "missed-monitor", "seq_num": 1, "data": {"counter": 25}})
                self.flush()
                np.testing.assert_array_equal(widget._summary_live_lines["Beam monitor"].getData()[1], [5, 25])
                self.assertEqual(controller._rendered_primary_seq_nums, {1})
            finally:
                controller.shutdown()
                widget.close()

    def test_late_latest_history_success_or_failure_cannot_replace_provisional_live_view(self):
        for error in ("", "Tiled offline"):
            with self.subTest(error=error), patch.object(live, "PV", None), patch.object(live, "_safe_caget", return_value=None):
                widget = DiffractionPlotWidgetPyQtGraph()
                client = SimpleNamespace(_running_item={"name": "count_he3psd", "kwargs": {"detectors": ["psd"]}})
                controller = live.DiffractionLivePlot(widget, re_client=client)
                try:
                    token = controller._history_load_token
                    controller._arm_current_running_item()
                    controller.on_document("event", {"descriptor": "missed", "seq_num": 1, "data": {"psd_counts": [9, 10]}})
                    controller._on_history_load_completed(token, {"uid": "old", "request_kind": "latest", "error": error,
                        "snapshot": {"start": {"uid": "old", "experiment_type": "diffraction"}, "rows": [{"psd_counts": [1, 2]}]}})
                    self.flush()
                    self.assertEqual(controller._run_uid, "__live_running__")
                    self.assertTrue(controller._live_follow_enabled)
                    self.assertIn("psd", widget._live_profile_lines)
                    self.assertEqual(widget._summary_y, {})
                finally:
                    controller.shutdown()
                    widget.close()

    def test_live_backfill_merges_new_documents_in_order_without_duplicates(self):
        for widget_class in (DiffractionPlotWidgetPyQtGraph, live.DiffractionPlotWidget):
            with self.subTest(plotter=widget_class.__name__), patch.object(live, "PV", None):
                widget = widget_class()
                controller = live.DiffractionLivePlot(widget)
                try:
                    start = {"uid": "running", "experiment_type": "diffraction", "data_type": "1d"}
                    descriptor = {"uid": "descriptor", "run_start": "running", "name": "primary",
                                  "data_keys": {"psd_counts": {"dtype": "array", "shape": [3]}}}
                    controller.on_document("start", start)
                    controller.on_document("descriptor", descriptor)
                    controller.on_document("event", {"descriptor": "descriptor", "seq_num": 3,
                                                       "data": {"psd_counts": [10, 10, 10]}})
                    controller._apply_live_backfill_snapshot({"start": start, "descriptor_docs": [descriptor],
                        "seq_nums": [1, 2, 3], "rows": [{"psd_counts": [1, 2, 3]},
                            {"psd_counts": [4, 5, 6]}, {"psd_counts": [10, 10, 10]}]})
                    self.flush()
                    self.assertEqual(widget._summary_y["psd"], [6, 15, 30])
                    with patch.object(controller, "_extract_summary_x", side_effect=AssertionError("Duplicate changed scan coordinates")):
                        controller.on_document("event", {"descriptor": "descriptor", "seq_num": 3,
                                                           "data": {"psd_counts": [10, 10, 10]}})
                    controller.on_document("event", {"descriptor": "descriptor", "seq_num": 4,
                                                       "data": {"psd_counts": [20, 20, 20]}})
                    self.flush()
                    self.assertEqual(widget._summary_y["psd"], [6, 15, 30, 60])
                    self.assertEqual(controller._rendered_primary_seq_nums, {1, 2, 3, 4})
                finally:
                    controller.shutdown()
                    widget.close()

    def test_local_reader_can_auto_connect_without_remote_password_dialog(self):
        for client in ({"4dh4": object()}, None):
            with self.subTest(local_reader=client is not None), \
                 patch("diffractometer_controls.analysis.tiled_auth.build_tiled_client_from_local_reader", return_value=client), \
                 patch.object(live, "ensure_tiled_login", side_effect=AssertionError("Unexpected login dialog")), \
                 patch.object(live.DiffractionLivePlot, "_refresh_history_async") as refresh:
                widget = DiffractionPlotWidgetPyQtGraph()
                controller = live.DiffractionLivePlot(widget, auto_history=True)
                try:
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        self.flush()
                        if refresh.called or client is None:
                            QtTest.QTest.qWait(50)
                            self.flush()
                            break
                        QtTest.QTest.qWait(10)
                    self.assertEqual(controller._tiled_enabled, client is not None)
                    if client is not None:
                        refresh.assert_called_with(autoload_live=True)
                finally:
                    controller.shutdown()
                    widget.close()

    def test_waiting_for_first_long_exposure_retries_beyond_old_limit(self):
        widget = DiffractionPlotWidgetPyQtGraph()
        controller = live.DiffractionLivePlot(widget)
        try:
            controller._tiled_enabled = True
            controller._current_live_run_uid = "long-exposure"
            controller._startup_history_retry_count = 10
            controller._backfill_retry_timer.start(0)
            controller._on_history_load_completed(controller._history_load_token, {
                "uid": "long-exposure", "request_kind": "startup",
                "snapshot": {"start": {"uid": "long-exposure", "experiment_type": "diffraction"},
                             "rows": [], "pending_reason": "Waiting for a committed point"},
            })
            self.assertTrue(controller._backfill_retry_timer.isActive())
            self.assertEqual(controller._backfill_retry_timer.interval(), 5000)
            self.assertFalse(controller._live_backfill_pending)
            controller._on_live_follow_toggled(False)
            self.assertFalse(controller._backfill_retry_timer.isActive())
        finally:
            controller.shutdown()
            widget.close()
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
