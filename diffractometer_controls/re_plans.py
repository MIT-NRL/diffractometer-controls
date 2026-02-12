import sys

from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QTableWidget)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)
try:
    from diffractometer_controls.re_plan_editor_widget import RePlanEditorWidget
except Exception:
    from re_plan_editor_widget import RePlanEditorWidget

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
                label.setStyleSheet("font-size: 14px; font-weight: 700; color: #111827;")
            if text in {"Move", "Mode", "Selection", "Edit"}:
                label.setStyleSheet("font-size: 13px; font-weight: 700; color: #334155;")

        for button in queue_widget.findChildren(QPushButton):
            # Match default button styling used in the rest of bluesky widgets.
            button.setStyleSheet("")

        # Queue table readability: larger text and wrapped parameters.
        for table in queue_widget.findChildren(QTableWidget):
            table.setStyleSheet(
                "QTableWidget { font-size: 13px; } "
                "QHeaderView::section { font-size: 13px; font-weight: 700; padding: 4px; }"
            )
            table.setWordWrap(True)
            table.setTextElideMode(QtCore.Qt.ElideNone)
            # Approximate two lines of text for long parameter cells.
            row_height = max(table.fontMetrics().height() * 2 + 8, 34)
            table.verticalHeader().setDefaultSectionSize(row_height)

    def customize_ui(self):
        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client

        re_queue = QtRePlanQueue(re_client)
        re_plan_editor = RePlanEditorWidget(re_client)
        self.ui.RE_Queue.layout().addWidget(re_queue)
        self.ui.RE_Plan_Editor.layout().addWidget(re_plan_editor)

        # Rebuild queue toolbar into grouped two-row controls.
        QtCore.QTimer.singleShot(0, lambda: self._reorganize_queue_toolbar(re_queue))
        QtCore.QTimer.singleShot(0, lambda: self._style_queue_widget(re_queue))
