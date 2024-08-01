import sys

from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel)
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

    def customize_ui(self):
        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client

        re_console = QtReConsoleMonitor(re_client)
        re_queue_history = QtRePlanHistory(re_client)
        self.ui.RE_Console.layout().addWidget(re_console)
        self.ui.RE_Queue_History.layout().addWidget(re_queue_history)
