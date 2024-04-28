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
        # print("REScreen here")
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

        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client
        # re_client = RunEngineClient(zmq_control_addr='tcp://192.168.0.14:60615')
        re_manager = QtReManagerConnection(re_client)
        re_environment = QtReEnvironmentControls(re_client)
        re_status = QtReStatusMonitor(re_client)
        re_running_plan = QtReRunningPlan(re_client)
        re_queue_controls = QtReQueueControls(re_client)
        re_plan_execution = QtReExecutionControls(re_client)

        self.ui.RE_Connection.layout().addWidget(re_manager)
        self.ui.RE_Worker.layout().addWidget(re_environment)
        self.ui.RE_Status.layout().addWidget(re_status)
        self.ui.RE_Running.layout().addWidget(re_running_plan)
        self.ui.RE_Queue_Controls.layout().addWidget(re_queue_controls)
        self.ui.RE_Plan_Execution.layout().addWidget(re_plan_execution)


    # def printstuff():
    #     print("button pressed")