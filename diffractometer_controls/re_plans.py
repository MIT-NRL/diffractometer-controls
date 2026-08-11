import sys
from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QTableWidget, QHeaderView)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanHistory,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)
try:
    from diffractometer_controls.re_plan_editor_widget import RePlanEditorWidget
    from diffractometer_controls.re_queue_widget import QtRePlanQueueEstimated
except Exception:
    from re_plan_editor_widget import RePlanEditorWidget
    from re_queue_widget import QtRePlanQueueEstimated

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

class REPlans(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='re_plans.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("REScreen here")
        # self.customize_ui()

    def ui_filename(self):
        return 're_plans.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def prepare_for_detach(self):
        for editor in self.findChildren(RePlanEditorWidget):
            shutdown = getattr(editor, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown(wait=True, timeout=0.25)
                except Exception:
                    pass

    @staticmethod
    def _refresh_queue_table_layout(queue_widget):
        table = getattr(queue_widget, "_table", None)
        labels = getattr(queue_widget, "_table_column_labels", ())
        if table is None or not labels:
            return

        try:
            type_col = labels.index("")
            name_col = labels.index("Name")
            parameters_col = labels.index("Parameters")
            estimate_col = labels.index("Est. Time")
            user_col = labels.index("USER")
            group_col = labels.index("GROUP")
        except ValueError:
            return

        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(type_col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(name_col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(parameters_col, QHeaderView.Interactive)
        header.setSectionResizeMode(estimate_col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(user_col, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(group_col, QHeaderView.ResizeToContents)

        vertical_header = table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeToContents)
        vertical_header.setMinimumSectionSize(max(table.fontMetrics().height() + 8, 28))

        fm = table.fontMetrics()
        params_header_width = fm.horizontalAdvance("Parameters") + 28
        current_width = table.columnWidth(parameters_col)
        table.setColumnWidth(parameters_col, max(params_header_width, current_width))
        est_header_width = fm.horizontalAdvance("Est. Time") + 28
        table.setColumnWidth(estimate_col, est_header_width)
        table.resizeRowsToContents()

    @staticmethod
    def _reorganize_queue_toolbar(queue_widget):
        """Rebuild Plan Queue toolbar into two rows grouped by function."""
        layout = queue_widget.layout()
        if layout is None or layout.count() == 0:
            return

        buttons = {
            "up": getattr(queue_widget, "_pb_move_up", None),
            "down": getattr(queue_widget, "_pb_move_down", None),
            "top": getattr(queue_widget, "_pb_move_to_top", None),
            "bottom": getattr(queue_widget, "_pb_move_to_bottom", None),
            "deselect": getattr(queue_widget, "_pb_deselect", None),
            "clear": getattr(queue_widget, "_pb_clear_queue", None),
            "loop": getattr(queue_widget, "_pb_loop_on", None),
            "delete": getattr(queue_widget, "_pb_delete_plan", None),
            "duplicate": getattr(queue_widget, "_pb_duplicate_plan", None),
        }
        if any(v is None for v in buttons.values()):
            return

        old_toolbar_item = layout.itemAt(0)
        if old_toolbar_item is None or old_toolbar_item.layout() is None:
            return
        old_toolbar = old_toolbar_item.layout()

        # Remove old top toolbar row from the queue layout.
        layout.removeItem(old_toolbar)
        # Remove orphan labels from the original toolbar (e.g. "QUEUE").
        for n in reversed(range(old_toolbar.count())):
            item = old_toolbar.takeAt(n)
            widget = item.widget()
            if isinstance(widget, QLabel):
                widget.deleteLater()

        toolbar_layout = QVBoxLayout()
        toolbar_layout.setContentsMargins(0, 0, 0, 0)
        toolbar_layout.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(4)
        row1.addWidget(QLabel("Move"))
        row1.addWidget(buttons["up"])
        row1.addWidget(buttons["down"])
        row1.addWidget(buttons["top"])
        row1.addWidget(buttons["bottom"])
        row1.addSpacing(8)
        row1.addWidget(QLabel("Mode"))
        row1.addWidget(buttons["loop"])

        row2 = QHBoxLayout()
        row2.setContentsMargins(0, 0, 0, 0)
        row2.setSpacing(4)
        row2.addWidget(QLabel("Selection"))
        row2.addWidget(buttons["deselect"])
        row2.addWidget(buttons["clear"])
        row2.addSpacing(8)
        row2.addWidget(QLabel("Edit"))
        row2.addWidget(buttons["delete"])
        row2.addWidget(buttons["duplicate"])

        queue_label = QLabel("QUEUE")
        queue_label.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)

        rows_layout = QGridLayout()
        rows_layout.setContentsMargins(0, 0, 0, 0)
        rows_layout.setHorizontalSpacing(10)
        rows_layout.setVerticalSpacing(2)
        rows_layout.addWidget(queue_label, 0, 0, 2, 1)
        rows_layout.addLayout(row1, 0, 1)
        rows_layout.addLayout(row2, 1, 1)
        rows_layout.setColumnStretch(1, 1)

        toolbar_layout.addLayout(rows_layout)
        layout.insertLayout(0, toolbar_layout)

    @staticmethod
    def _style_queue_widget(queue_widget):
        """Apply clearer styling and larger text to Plan Queue panel."""
        pal = queue_widget.palette()
        title_color = pal.color(QtGui.QPalette.Active, QtGui.QPalette.WindowText).name()
        muted_color = pal.color(QtGui.QPalette.Active, QtGui.QPalette.Mid).name()

        # Match the centered, larger title style used on other RE widgets.
        for group_box in queue_widget.findChildren(QGroupBox):
            group_box.setStyleSheet(
                "QGroupBox { font-size: 14px; font-weight: 700; margin-top: 10px; } "
                "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 6px; }"
            )

        # Toolbar labels and controls: slightly larger, consistent spacing.
        for label in queue_widget.findChildren(QLabel):
            text = label.text().strip()
            if text == "QUEUE":
                label.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {title_color};")
            if text.startswith("Total Est. Time:"):
                label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {title_color};")
            if text.startswith("Est. Completion:"):
                label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {title_color};")
            if text in {"Move", "Mode", "Selection", "Edit"}:
                label.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {muted_color};")

        for button in queue_widget.findChildren(QPushButton):
            # Match default button styling used in the rest of bluesky widgets.
            button.setStyleSheet("")

        # Queue table readability: larger text and wrapped parameters.
        for table in queue_widget.findChildren(QTableWidget):
            table.setStyleSheet(
                "QTableWidget { font-size: 13px; } "
                "QTableWidget::item { padding: 3px 4px 1px 4px; } "
                "QHeaderView::section { font-size: 13px; font-weight: 700; padding: 4px; }"
            )
            table.setWordWrap(False)
            table.setTextElideMode(QtCore.Qt.ElideNone)
            table.verticalHeader().setDefaultSectionSize(max(table.fontMetrics().height() + 8, 28))

    def customize_ui(self):
        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client

        re_queue = QtRePlanQueueEstimated(re_client)
        re_plan_editor = RePlanEditorWidget(re_client)
        re_plan_editor.setObjectName("REPlanEditorWidget")
        self.ui.RE_Queue.layout().addWidget(re_queue)
        self.ui.RE_Plan_Editor.layout().addWidget(re_plan_editor)

        # Rebuild queue toolbar into grouped two-row controls.
        QtCore.QTimer.singleShot(0, lambda: self._reorganize_queue_toolbar(re_queue))
        QtCore.QTimer.singleShot(0, lambda: self._style_queue_widget(re_queue))
        re_queue.signal_plan_queue_changed.connect(lambda *_: self._refresh_queue_table_layout(re_queue))
        QtCore.QTimer.singleShot(0, lambda: re_queue.slot_update_widgets(bool(re_client.re_manager_connected)))
        QtCore.QTimer.singleShot(
            0,
            lambda: re_queue.slot_plan_queue_changed(
                list(getattr(re_client, "_plan_queue_items", []) or []),
                list(getattr(re_client, "selected_queue_item_uids", []) or []),
            ),
        )
        QtCore.QTimer.singleShot(0, lambda: self._refresh_queue_table_layout(re_queue))
