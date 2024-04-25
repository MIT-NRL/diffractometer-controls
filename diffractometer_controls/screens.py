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

class Screen(Display):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='screens.ui'):
        super().__init__(parent, args, macros, ui_filename)

        self.customize_ui()

    def ui_filename(self):
        return 'screens.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        button = self.ui.pushButton
        print('Here')
        button.clicked.connect(self.printstuff)

        frame = self.ui.Frame1

        layout = QVBoxLayout()
        

        button2 = QPushButton('here')

        re_client = RunEngineClient()
        re_manager = QtReManagerConnection(re_client)

        frame = self.ui.Frame1
        frame.layout().addWidget(re_manager)


    def printstuff():
        print("button pressed")