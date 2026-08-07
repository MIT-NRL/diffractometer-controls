import sys

from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QTextEdit)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanEditor,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

class REPlans(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='re_extras.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("REScreen here")
        # self.customize_ui()

    def ui_filename(self):
        return 're_extras.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def prepare_for_detach(self):
        console = getattr(self, "_re_console", None)
        if console is not None:
            console._dc_console_stop_requested = True

    def _console_text_edit(self):
        console = getattr(self, "_re_console", None)
        if console is None:
            return None

        text_edit = getattr(console, "_text_edit", None)
        if text_edit is None:
            text_edits = console.findChildren(QTextEdit)
            text_edit = text_edits[0] if text_edits else None
        return text_edit

    def _apply_console_theme(self):
        text_edit = self._console_text_edit()
        if text_edit is None:
            return

        app = QApplication.instance()
        palette = QtGui.QPalette(app.palette() if app is not None else self.palette())
        base_color = palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Base)
        text_color = palette.color(QtGui.QPalette.Active, QtGui.QPalette.Text)
        highlight_color = palette.color(QtGui.QPalette.Active, QtGui.QPalette.Highlight)
        highlighted_text_color = palette.color(
            QtGui.QPalette.Active, QtGui.QPalette.HighlightedText
        )

        for group in (QtGui.QPalette.Active, QtGui.QPalette.Inactive, QtGui.QPalette.Disabled):
            palette.setColor(group, QtGui.QPalette.Base, base_color)
            palette.setColor(group, QtGui.QPalette.Text, text_color)

        text_edit.setPalette(palette)
        text_edit.setAutoFillBackground(True)
        text_edit.setStyleSheet(
            "QTextEdit {"
            f" background-color: {base_color.name()};"
            f" color: {text_color.name()};"
            f" selection-background-color: {highlight_color.name()};"
            f" selection-color: {highlighted_text_color.name()};"
            " }"
        )
        viewport = text_edit.viewport()
        if viewport is not None:
            viewport.setPalette(palette)
            viewport.setAutoFillBackground(True)
            viewport.update()
        text_edit.update()

    def _scroll_console_to_bottom(self):
        console = getattr(self, "_re_console", None)
        text_edit = self._console_text_edit()
        if console is None or text_edit is None:
            return

        scrollbar = text_edit.verticalScrollBar()
        if scrollbar is None:
            return

        console._te_scrolled_to_bottom = True
        scrollbar.setValue(scrollbar.maximum())

    def _sync_console_autoscroll_state(self, *_args):
        console = getattr(self, "_re_console", None)
        text_edit = self._console_text_edit()
        if console is None or text_edit is None:
            return

        if not getattr(console, "_autoscroll_enabled", False):
            console._te_scrolled_to_bottom = False
            return

        if getattr(console, "_is_slider_pressed", False) or getattr(console, "_updating_text", False):
            return

        scrollbar = text_edit.verticalScrollBar()
        if scrollbar is None:
            return

        console._te_scrolled_to_bottom = scrollbar.value() >= max(0, scrollbar.maximum() - 1)

    def _handle_console_autoscroll_toggled(self, state):
        console = getattr(self, "_re_console", None)
        if console is None:
            return

        enabled = state == QtCore.Qt.Checked
        console._autoscroll_enabled = enabled
        console._te_scrolled_to_bottom = enabled
        if enabled:
            QtCore.QTimer.singleShot(0, self._scroll_console_to_bottom)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
        ):
            self._apply_console_theme()

    def customize_ui(self):
        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client

        re_console = QtReConsoleMonitor(re_client)
        re_queue_history = QtRePlanHistory(re_client)
        self.ui.RE_Console.layout().addWidget(re_console)
        self.ui.RE_Queue_History.layout().addWidget(re_queue_history)
        self._re_console = re_console

        text_edit = self._console_text_edit()
        if text_edit is not None:
            scrollbar = text_edit.verticalScrollBar()
            if scrollbar is not None:
                scrollbar.valueChanged.connect(self._sync_console_autoscroll_state)

        autoscroll_checkbox = getattr(re_console, "_cb_autoscroll", None)
        if autoscroll_checkbox is not None:
            autoscroll_checkbox.stateChanged.connect(self._handle_console_autoscroll_toggled)

        re_queue_history.slot_update_widgets()
        re_queue_history.slot_plan_history_changed(
            list(getattr(re_client, "_plan_history_items", []) or []),
            list(getattr(re_client, "selected_history_item_pos", []) or []),
        )
        self._apply_console_theme()
