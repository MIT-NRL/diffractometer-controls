"""Real Qt coverage for shared-widget workers, injected APIs and destruction."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import threading
import unittest
from unittest.mock import Mock, patch
from qtpy import QtCore, QtTest, QtWidgets
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt import run_engine_client as rec
from control_ui.core.compatibility import install_bluesky_compatibility
from control_ui.core.options import PlanEditorOptions
from control_ui.widgets.re_plan_editor_widget import RePlanEditorWidget
from control_ui.widgets.re_queue_widget import QtRePlanQueueEstimated


class SharedWidgetOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        install_bluesky_compatibility()

    def setUp(self):
        self.model = RunEngineClient(zmq_control_addr="tcp://127.0.0.1:1", zmq_info_addr="tcp://127.0.0.1:2")

    def test_editor_uses_injected_api_and_releases_model_callbacks_on_destruction(self):
        api = Mock()
        before = {name: len(getattr(self.model.events, name).callbacks) for name in self.model.events}
        editor = RePlanEditorWidget(self.model, options=PlanEditorOptions(), re_manager_api=api)
        tables = [editor._plan_editor._wd_editor, editor._plan_viewer._wd_editor]
        for table in tables:
            self.assertIs(table._get_file_dir_api(), api)
            self.assertEqual(table.options.local_roots, ())
        editor.shutdown(wait=True, timeout=0.5)
        self.assertTrue(all(not table._file_dir_query_thread.is_alive() for table in tables))
        editor.deleteLater()
        QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        after = {name: len(getattr(self.model.events, name).callbacks) for name in self.model.events}
        self.assertEqual(after, before)

    def test_queue_coalesces_updates_and_cancels_a_finishing_estimate(self):
        started, release = threading.Event(), threading.Event()
        calls = []

        def context():
            calls.append(1)
            started.set()
            release.wait(1)
            return {}

        queue = QtRePlanQueueEstimated(self.model, estimation_context=context)
        item = {"item_type": "plan", "item_uid": "one", "name": "wait_seconds", "kwargs": {"seconds": 2}}
        queue.slot_plan_queue_changed([item], [])
        QtTest.QTest.qWait(110)
        self.assertTrue(started.wait(0.5))
        for index in range(4):
            queue.slot_plan_queue_changed([{**item, "item_uid": str(index)}], [])
        QtTest.QTest.qWait(180)
        self.assertEqual(len(calls), 1)
        queue.shutdown()
        release.set()
        queue._estimate_thread.join(timeout=0.5)
        self.assertFalse(queue._estimate_thread.is_alive())
        QtTest.QTest.qWait(50)
        self.assertFalse(queue._estimate_timer.isActive())
        queue.deleteLater()

    def test_console_delayed_scroll_is_cancelled_when_destroyed(self):
        self.model.start_console_output_monitoring = lambda: None
        errors = []
        with patch.object(rec.QtReConsoleMonitor, "_start_thread", lambda self: None), \
             patch.object(rec.QtReConsoleMonitor, "_start_timer", lambda self: None), \
             patch.object(sys, "excepthook", lambda *args: errors.append(args)):
            console = rec.QtReConsoleMonitor(self.model)
            console._text_list = ["Reference console output", ""]
            console._display_text()
            console.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            QtTest.QTest.qWait(90)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
