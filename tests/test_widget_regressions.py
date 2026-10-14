"""Plan fidelity, validation and lifecycle regressions with real disconnected Qt widgets."""
import copy
import os
import sys
import threading
import unittest
from unittest.mock import patch
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from qtpy import QtCore, QtTest, QtWidgets
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt import run_engine_client as rec
from control_ui.core.compatibility import install_bluesky_compatibility
from control_ui.core.lifecycle import DisplayOwner
from control_ui.core.options import PlanEditorOptions
from control_ui.core.services import ControlServices
from control_ui.widgets.re_plan_editor_widget import RePlanEditorTable, RePlanEditorWidget, _PersistentCheckableMenu
from control_ui.widgets.re_queue_widget import QtRePlanQueueEstimated
from control_ui.widgets.re_control_panel import REControlPanel, _NonBlockingQtReEnvironmentControls
from control_ui.widgets.re_plans import REPlans


class WidgetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        install_bluesky_compatibility()

    def setUp(self):
        self.model = RunEngineClient(zmq_control_addr="tcp://127.0.0.1:1", zmq_info_addr="tcp://127.0.0.1:2")
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            shutdown = getattr(widget, "shutdown", None)
            if shutdown:
                shutdown()
            widget.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()

    def table(self, annotation, value, name="detectors"):
        self.model._allowed_plans["test_plan"] = {"name":"test_plan", "parameters":[
            {"name":name,"kind":{"name":"POSITIONAL_OR_KEYWORD","value":1},"annotation":annotation}]}
        table = RePlanEditorTable(self.model, options=PlanEditorOptions())
        self.widgets.append(table)
        table.show_item(item={"item_type":"plan","name":"test_plan","kwargs":{name:value}},editable=True)
        return table

    def test_loading_multi_selection_preserves_names_and_order(self):
        names = ["second", "union_detector", "x", "unavailable"]
        table = self.table({"type":"typing.List[Detectors]","devices":{"Detectors":["x","union_detector","second"]}}, names)
        self.assertEqual(table.get_modified_item()["kwargs"]["detectors"], names)
        self.assertEqual(table.queue_item["kwargs"]["detectors"], names)

    def test_unavailable_choice_can_be_removed_from_grouped_menu(self):
        table = self.table({"type":"typing.List[Detectors]","devices":{"Detectors":["camera.signal"]}}, ["camera.signal", "unavailable"])
        combo = table.cellWidget(0, 2)
        def choose(menu, _position):
            action = next(action for action in menu.actions() if action.data() == "unavailable")
            action.setChecked(False)
        with patch.object(_PersistentCheckableMenu, "exec_", choose):
            combo.showPopup()
        self.assertEqual(table.get_modified_item()["kwargs"]["detectors"], ["camera.signal"])

    def test_clearing_required_selector_disables_add(self):
        self.table({"type":"Motors","devices":{"Motors":["motor_one"]}}, "motor_one", "motor")
        self.model._allowed_devices = {"motor_one":{"is_movable":True}}
        self.model._re_manager_connected = True
        editor = RePlanEditorWidget(self.model)
        self.widgets.append(editor)
        editor.load_new_plan_item({"name":"test_plan","item_type":"plan","kwargs":{"motor":"motor_one"}})
        self.app.processEvents()
        editor._plan_editor._wd_editor.cellWidget(0,2).setCurrentIndex(0)
        self.app.processEvents()
        self.assertFalse(editor._plan_editor._pb_add_to_queue.isEnabled())

    def test_empty_directory_success_clears_suggestions(self):
        table = self.table({"type":"str"}, "manual", "file_dir")
        table._on_file_dir_choices_ready(["old"], None)
        table._on_file_dir_choices_ready([], None)
        self.assertEqual(table._file_dir_cached_choices, [])
        self.assertEqual(table.cellWidget(0,2).currentText(), "manual")

    def test_explicit_empty_file_directory_is_a_valid_string_argument(self):
        table = self.table({"type":"str"}, "", "file_dir")
        validity = []
        table.signal_parameters_valid.connect(validity.append)
        table._validate_cell_values()
        self.assertTrue(validity[-1])
        self.assertEqual(table.get_modified_item()["kwargs"]["file_dir"], "")

    def test_delayed_initialization_is_cancelled_on_destruction(self):
        services = ControlServices(self.model, None, None)
        owner = DisplayOwner(lambda: REPlans(services=services), services)
        errors = []
        with patch.object(sys,"excepthook",lambda *args: errors.append(args)):
            owner.shutdown()
            owner.widget.deleteLater()
            self.app.sendPostedEvents(None,QtCore.QEvent.DeferredDelete)
            QtTest.QTest.qWait(90)
        self.assertEqual(errors, [])

    def test_console_preserves_chunks_carriage_return_and_literal_markup(self):
        self.model.start_console_output_monitoring = lambda: None
        with patch.object(rec.QtReConsoleMonitor,"_start_thread",lambda self:None), patch.object(rec.QtReConsoleMonitor,"_start_timer",lambda self:None):
            console = rec.QtReConsoleMonitor(self.model)
        for text in ("Hello ", "world\n", "<b>literal</b>\n", "10%", "\r20%"):
            console._process_new_console_output((1.0,text))
        console._display_text()
        self.assertEqual(console._text_edit.toPlainText(), "Hello world\n<b>literal</b>\n20%")
        console._dc_shutdown_console()
        self.widgets.append(console)

    def test_finished_status_worker_restarts_after_navigation(self):
        calls=[]
        self.model.load_re_manager_status=lambda: calls.append(1)
        services=ControlServices(self.model,None,None)
        owner=DisplayOwner(lambda:REControlPanel(services=services),services)
        manager=owner.widget._re_manager
        self.model._re_manager_connected=True
        manager.update_period=0.05
        manager.updates_activated=True
        manager._start_thread()
        owner.deactivate()
        QtTest.QTest.qWait(150)
        before=len(calls)
        owner.activate()
        QtTest.QTest.qWait(150)
        self.assertGreater(len(calls),before)
        owner.shutdown()
        QtTest.QTest.qWait(150)
        owner.widget.deleteLater()

    def test_environment_request_does_not_block_qt(self):
        release=threading.Event()
        started=threading.Event()
        self.model._client.environment_open=lambda: (started.set(),release.wait(1))[-1]
        control=_NonBlockingQtReEnvironmentControls(self.model)
        self.widgets.append(control)
        control._pb_env_open_clicked()
        heartbeats=[]
        QtCore.QTimer.singleShot(0,lambda:heartbeats.append(1))
        QtTest.QTest.qWait(50)
        self.assertTrue(started.is_set())
        self.assertEqual(heartbeats,[1])
        release.set()
        QtTest.QTest.qWait(100)

    def test_queue_metadata_and_unknown_totals(self):
        queue=QtRePlanQueueEstimated(self.model)
        self.widgets.append(queue)
        self.model._allowed_plans["wait"]={"parameters":[{"name":"seconds"}]}
        queue._normalize_plan_kwargs({"name":"wait","args":[2]})
        self.model._allowed_plans["wait"]={"parameters":[{"name":"num"},{"name":"seconds"}]}
        self.assertEqual(queue._normalize_plan_kwargs({"name":"wait","args":[3,10]}),{"num":3,"seconds":10})
        queue._plan_queue_items=[{"item_uid":"known"},{"item_uid":"unknown"}]
        queue._table.setRowCount(2)
        queue._estimate_cache={"known":{"estimated_total_time_s":10}}
        queue._apply_estimates_to_table()
        self.assertIn("incomplete",queue._total_time_label.text())
        self.assertIsNone(queue._completion_total_s)


if __name__ == "__main__":
    unittest.main()
