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

class REScreen(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='re_screen.ui'):
        super().__init__(parent, args, macros, ui_filename)
        print("REScreen here")
        # self.customize_ui()

    def ui_filename(self):
        return 're_screen.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        # button = self.ui.pushButton
        # print('Here')
        # button.clicked.connect(self.printstuff)

        

        layout = QVBoxLayout()
        

        button2 = QPushButton('here')

        re_client = RunEngineClient()
        re_manager = QtReManagerConnection(re_client)
        re_environment = QtReEnvironmentControls(re_client)
        re_console = QtReStatusMonitor(re_client)

        self.ui.RE_Connection.layout().addWidget(re_manager)
        self.ui.RE_Worker.layout().addWidget(re_environment)
        self.ui.RE_Console.layout().addWidget(re_console)


    # def printstuff():
    #     print("button pressed")