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

    def _apply_console_theme(self):
        console = getattr(self, "_re_console", None)
        if console is None:
            return

        text_edit = getattr(console, "_text_edit", None)
        if text_edit is None:
            text_edits = console.findChildren(QTextEdit)
            text_edit = text_edits[0] if text_edits else None
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
        self._apply_console_theme()
