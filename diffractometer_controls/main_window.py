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


class MITRMainWindow(PyDMMainWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # self.customize_ui()
            # self.export_actions()
        #     self.ui.setupUi(self)
            self.customize_ui()

        def customize_ui(self):
            from application import MITRApplication
            app = MITRApplication.instance()
            icon_path = str(Path("./NRL_Logo.png").resolve())
            self.setWindowIcon(QtGui.QIcon(icon_path))

            bar = self.statusBar()
            _label = QLabel()
            _label.setText("Queue:")
            bar.addPermanentWidget(_label)
            self.ui.environment_label = QLabel()
            self.ui.environment_label.setText("N/A")
            bar.addPermanentWidget(self.ui.environment_label)
            # re_connect_frame = self.ui
            # button = self.ui.pushButton1
            # self.ui.
            # # layout = self.ui.verticalLayout.addWidget(button)
            # layout = QtWidgets.QVBoxLayout()
            # layout.addWidget(button)
            # self.setLayout(layout)
            # layout.addWidget(button)
            ...
        #     re_connect_frame = self.ui.REConnectWidget
        #     layout = QVBoxLayout()
        #     re_connect_frame.setLayout(layout)
        #     layout.addWidget(QtReManagerConnection(self))

