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
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.utils import install_remote_qt_kicker

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

class MainScreen(display.MITRDisplay):
    re_dispatcher: RemoteDispatcher
    re_client: RunEngineClient

    def __init__(self, parent=None, args=None, macros=None, ui_filename='main_screen.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("MainScreen here")

    def ui_filename(self):
        return 'main_screen.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
            from application import MITRApplication
            from bluesky_widgets.utils.streaming import stream_documents_into_runs

            app = MITRApplication.instance()
            re_client = app.re_client

            re_queue = QtRePlanQueue(re_client)
            re_plan_editor = QtRePlanEditor(re_client)
            self.ui.RE_Queue.layout().addWidget(re_queue)
            self.ui.RE_Plan_Editor.layout().addWidget(re_plan_editor)

            # figModel = Lines('motor',['det1','det2'],max_runs=3)
            figModel = AutoLines(max_runs=3)
            viewer = QtFigures(figModel.figures)
            self.runs = []
            app.re_dispatcher.subscribe(stream_documents_into_runs(figModel.add_run))
            # app.re_dispatcher.subscribe(print)
            app.re_dispatcher.start()
            # install_remote_qt_kicker()
            
            re_console = QtReConsoleMonitor(re_client)
            re_queue_history = QtRePlanHistory(re_client)
            self.ui.RE_Console.layout().addWidget(re_console)
            self.ui.RE_Queue_History.layout().addWidget(re_queue_history)

            self.ui.Data_Viewer.layout().addWidget(viewer)

            self.ui.pushButton.clicked.connect(self.printstuff)

    def printstuff(self):
         print(self.runs)