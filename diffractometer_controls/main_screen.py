import sys

# from pydm.display import Display
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

from bluesky_widgets.qt.figures import QtFigure, QtFigures
from bluesky_widgets.models.auto_plot_builders import AutoLines, AutoPlotter
from bluesky_widgets.models.plot_builders import Lines

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

class MainScreen(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='main_screen.ui'):
        super().__init__(parent, args, macros, ui_filename)
        print("MainScreen here")

    def ui_filename(self):
        return 'main_screen.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
            # re_client = RunEngineClient(zmq_control_addr='tcp://192.168.0.14:60615')
            from application import MITRApplication
            from bluesky_widgets.utils.streaming import stream_documents_into_runs

            app = MITRApplication.instance()
            re_client = app.re_client
            # print(app.test)

            re_queue = QtRePlanQueue(re_client)
            re_plan_editor = QtRePlanEditor(re_client)
            self.ui.RE_Queue.layout().addWidget(re_queue)
            self.ui.RE_Plan_Editor.layout().addWidget(re_plan_editor)

            figModel = AutoLines(max_runs=3)
            viewer = QtFigures(figModel.figures)
            app.re_dispatcher.subscribe(stream_documents_into_runs(figModel.add_run))
            app.re_dispatcher.start()

            self.ui.Data_Viewer.layout().addWidget(viewer)