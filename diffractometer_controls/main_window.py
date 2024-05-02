import logging
import warnings
from pathlib import Path
from icecream import ic

import qtawesome as qta
from pydm import data_plugins
from pydm.main_window import PyDMMainWindow
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
from pydm.widgets import PyDMByteIndicator


class MITRMainWindow(PyDMMainWindow):
        def __init__(self, re_client: RunEngineClient = None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.customize_ui()
            # self.export_actions()
        #     self.ui.setupUi(self)
            self.customize_ui()
            # self.re_client = re_client
            # print(self.re_client.re_manager_status)

        def customize_ui(self):
            from application import MITRApplication
            app = MITRApplication.instance()
            icon_path = str(Path("./NRL_Logo.png").resolve())
            self.setWindowIcon(QtGui.QIcon(icon_path))

            bar = self.statusBar()
            heartbeat_indicator = PyDMByteIndicator(init_channel='ca://4dh4:HEARTBEAT')
            heartbeat_indicator.labels = ['IOC Heartbeat']
            heartbeat_indicator.labelPosition = 2

            bar.addPermanentWidget(heartbeat_indicator)

        def update_window_title(self):
            if self.showing_file_path_in_title_bar:
                title = self.current_file()
            else:
                title = self.display_widget().windowTitle()
            title += " - MITR 4DH4 Beamline Controls"
            if data_plugins.is_read_only():
                title += " [Read Only Mode]"
            self.setWindowTitle(title)


