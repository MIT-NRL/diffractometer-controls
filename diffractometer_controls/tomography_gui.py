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

# from bluesky_widgets.qt.figures import QtFigure, QtFigures
# from bluesky_widgets.models.auto_plot_builders import AutoLines, AutoPlotter, AutoImages
from bluesky_widgets.models.plot_builders import Lines, Images
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.utils import install_remote_qt_kicker

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

from figures import QtFigure, NewLivePlot

class MainScreen(display.MITRDisplay):
    re_dispatcher: RemoteDispatcher
    re_client: RunEngineClient

    def __init__(self, parent=None, args=None, macros=None, ui_filename='tomography_gui.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("MainScreen here")

    def ui_filename(self):
        return 'tomography_gui.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
            from application import MITRApplication
            from bluesky_widgets.utils.streaming import stream_documents_into_runs
            