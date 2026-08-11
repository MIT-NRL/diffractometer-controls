import copy
import os
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MITR_FILE_DIR_QUERY_MODE", "local")

try:
    from qtpy import QtCore, QtTest, QtWidgets
    from bluesky_widgets.models.run_engine_client import RunEngineClient

    from diffractometer_controls.re_plan_editor_widget import RePlanEditorWidget
    from diffractometer_controls.re_plans import REPlans
    from diffractometer_controls.re_queue_widget import QtRePlanQueueEstimated

    _GUI_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only without GUI extras
    _GUI_IMPORT_ERROR = exc


@unittest.skipIf(_GUI_IMPORT_ERROR is not None, f"GUI dependencies unavailable: {_GUI_IMPORT_ERROR}")
class RePlanWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.model = RunEngineClient(
            zmq_control_addr="tcp://127.0.0.1:60615",
            zmq_info_addr="tcp://127.0.0.1:60625",
        )
        self.model._re_manager_connected = True
        self.model._allowed_plans = {
            "empty_plan": {"name": "empty_plan", "parameters": []},
            "second_plan": {"name": "second_plan", "parameters": []},
        }
        self.items = [
            {
                "item_uid": "uid-1",
                "item_type": "plan",
                "name": "empty_plan",
                "kwargs": {},
                "user": "operator",
                "user_group": "primary",
            },
            {
                "item_uid": "uid-2",
                "item_type": "plan",
                "name": "second_plan",
                "kwargs": {},
                "user": "operator",
                "user_group": "primary",
            },
        ]
        self.model._plan_queue_items = copy.deepcopy(self.items)
        self.model._plan_queue_items_pos = {"uid-1": 0, "uid-2": 1}

        self.queue = QtRePlanQueueEstimated(self.model)
        # Runtime estimates are independently tested; keep these interaction
        # tests deterministic and free of worker threads or EPICS reads.
        self.queue._schedule_estimate_update = lambda _items: None
        self.queue._estimate_timer.stop()
        self.queue._completion_refresh_timer.stop()
        self.editor = RePlanEditorWidget(self.model)
        REPlans._connect_queue_editor_workflows(self.queue, self.editor)
        self.queue.slot_plan_queue_changed(
            self.model._plan_queue_items,
            self.model.selected_queue_item_uids,
        )
        self._process_events()

    def tearDown(self):
        self.queue._estimate_timer.stop()
        self.queue._completion_refresh_timer.stop()
        self.editor.shutdown(wait=True)
        self.queue.deleteLater()
        self.editor.deleteLater()
        self._process_events()

    def _process_events(self):
        self.app.processEvents()

    def _select_row(self, row):
        self.queue._table.selectRow(row)
        self._process_events()

    def test_queue_selection_opens_and_clears_viewer(self):
        self.editor._switch_tab("edit")

        self._select_row(0)

        self.assertEqual(self.model.selected_queue_item_uids, ["uid-1"])
        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_viewer,
        )
        self.assertEqual(self.editor._plan_viewer._lb_item_name.text(), "empty_plan")

        self.queue._table.clearSelection()
        self._process_events()

        self.assertEqual(self.model.selected_queue_item_uids, [])
        self.assertEqual(self.editor._plan_viewer._lb_item_name.text(), "-")

    def test_queue_selection_hides_but_preserves_new_item_edit(self):
        self.editor.load_new_plan_item(
            {"item_type": "plan", "name": "empty_plan", "kwargs": {}},
            preserve_existing=False,
        )
        self.assertEqual(self.editor._plan_editor._current_item_source, "NEW ITEM")

        self._select_row(1)

        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_viewer,
        )
        self.assertEqual(self.editor._plan_viewer._lb_item_name.text(), "second_plan")
        self.assertEqual(self.editor._plan_editor._current_item_source, "NEW ITEM")
        self.assertEqual(
            self.editor._plan_editor._wd_editor.queue_item["name"],
            "empty_plan",
        )

    def test_existing_selection_is_loaded_when_editor_is_created(self):
        self.editor.shutdown(wait=True)
        self.editor.deleteLater()
        self._process_events()

        # Simulate opening the screen after a queue item was already selected.
        self.model._selected_queue_item_uids = ["uid-2"]
        self.editor = RePlanEditorWidget(self.model)
        REPlans._connect_queue_editor_workflows(self.queue, self.editor)
        self._process_events()

        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_viewer,
        )
        self.assertEqual(self.editor._plan_viewer._lb_item_name.text(), "second_plan")

    def test_double_click_edit_and_cancel_clear_editor_in_place(self):
        self._select_row(0)
        self.assertEqual(len(self.queue.registered_item_editors), 1)

        self.queue._on_table_cell_double_clicked(0, 0)
        self._process_events()

        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_editor,
        )
        self.assertEqual(self.editor._plan_editor._current_item_source, "QUEUE ITEM")

        QtTest.QTest.mouseClick(
            self.editor._plan_editor._pb_cancel,
            QtCore.Qt.LeftButton,
        )
        self._process_events()

        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_editor,
        )
        self.assertEqual(self.editor._plan_editor._current_item_source, "")
        self.assertFalse(self.editor._plan_editor._edit_mode_enabled)
        self.assertFalse(self.editor._plan_editor._pb_cancel.isEnabled())

    def test_viewer_edit_copy_reset_save_and_add_transitions(self):
        copied = []
        updated = []
        added = []
        self.model.queue_item_copy_to_queue = lambda: copied.append(True)
        self.model.queue_item_update = lambda *, item: updated.append(copy.deepcopy(item))
        self.model.queue_item_add = lambda *, item: added.append(copy.deepcopy(item))

        self._select_row(0)
        QtTest.QTest.mouseClick(
            self.editor._plan_viewer._pb_copy_to_queue,
            QtCore.Qt.LeftButton,
        )
        self._process_events()
        self.assertEqual(copied, [True])

        QtTest.QTest.mouseClick(
            self.editor._plan_viewer._pb_edit,
            QtCore.Qt.LeftButton,
        )
        self._process_events()
        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_editor,
        )

        reset_calls = []
        original_reset = self.editor._plan_editor._wd_editor.reset_item
        self.editor._plan_editor._wd_editor.reset_item = lambda: reset_calls.append(True)
        QtTest.QTest.mouseClick(
            self.editor._plan_editor._pb_reset,
            QtCore.Qt.LeftButton,
        )
        self._process_events()
        self.assertEqual(reset_calls, [True])
        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_editor,
        )
        self.editor._plan_editor._wd_editor.reset_item = original_reset

        self.editor._plan_editor._editor_state_valid = True
        self.editor._plan_editor._update_widget_state()
        self.assertTrue(self.editor._plan_editor._pb_save_item.isEnabled())
        QtTest.QTest.mouseClick(
            self.editor._plan_editor._pb_save_item,
            QtCore.Qt.LeftButton,
        )
        self._process_events()
        self.assertEqual(updated[0]["item_uid"], "uid-1")
        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_viewer,
        )

        self.editor.load_new_plan_item(
            {"item_type": "plan", "name": "empty_plan", "kwargs": {}},
            preserve_existing=False,
        )
        self.editor._plan_editor._editor_state_valid = True
        self.editor._plan_editor._update_widget_state()
        self.assertTrue(self.editor._plan_editor._pb_add_to_queue.isEnabled())
        QtTest.QTest.mouseClick(
            self.editor._plan_editor._pb_add_to_queue,
            QtCore.Qt.LeftButton,
        )
        self._process_events()
        self.assertEqual(added[0]["name"], "empty_plan")
        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_viewer,
        )

    def test_queue_refresh_updates_viewer_without_interrupting_edit(self):
        self._select_row(0)
        QtTest.QTest.mouseClick(
            self.editor._plan_viewer._pb_edit,
            QtCore.Qt.LeftButton,
        )
        self._process_events()

        self.model._plan_queue_items[0]["name"] = "second_plan"
        self.model.events.plan_queue_changed(
            plan_queue_items=self.model._plan_queue_items,
            selected_item_uids=["uid-1"],
        )
        self._process_events()

        self.assertEqual(self.editor._plan_viewer._lb_item_name.text(), "second_plan")
        self.assertIs(
            self.editor._tab_widget.currentWidget(),
            self.editor._plan_editor,
        )

    def test_reorganized_queue_toolbar_routes_all_actions(self):
        calls = []
        self.model.queue_items_move_up = lambda: calls.append("up")
        self.model.queue_items_move_down = lambda: calls.append("down")
        self.model.queue_items_move_to_top = lambda: calls.append("top")
        self.model.queue_items_move_to_bottom = lambda: calls.append("bottom")
        self.model.queue_items_remove = lambda: calls.append("delete")
        self.model.queue_item_copy_to_queue = lambda: calls.append("duplicate")
        self.model.queue_clear = lambda: calls.append("clear")
        self.model.queue_mode_loop_enable = lambda enabled: calls.append(
            ("loop", enabled)
        )
        REPlans._reorganize_queue_toolbar(self.queue)

        self._select_row(1)
        for button in (
            self.queue._pb_move_up,
            self.queue._pb_move_to_top,
            self.queue._pb_delete_plan,
            self.queue._pb_duplicate_plan,
        ):
            self.assertTrue(button.isEnabled())
            QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)

        self._select_row(0)
        for button in (
            self.queue._pb_move_down,
            self.queue._pb_move_to_bottom,
        ):
            self.assertTrue(button.isEnabled())
            QtTest.QTest.mouseClick(button, QtCore.Qt.LeftButton)

        QtTest.QTest.mouseClick(self.queue._pb_loop_on, QtCore.Qt.LeftButton)
        QtTest.QTest.mouseClick(self.queue._pb_clear_queue, QtCore.Qt.LeftButton)
        self._process_events()

        self.assertEqual(
            calls,
            [
                "up",
                "top",
                "delete",
                "duplicate",
                "down",
                "bottom",
                ("loop", True),
                "clear",
            ],
        )

        QtTest.QTest.mouseClick(self.queue._pb_deselect, QtCore.Qt.LeftButton)
        self._process_events()
        self.assertEqual(self.model.selected_queue_item_uids, [])


if __name__ == "__main__":
    unittest.main()
