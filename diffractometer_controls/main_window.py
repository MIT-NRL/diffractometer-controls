import logging
import warnings
from pathlib import Path
# from icecream import ic

import qtawesome as qta
from pydm import data_plugins
from pydm.main_window import PyDMMainWindow
from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt, QTimer, Slot, QSize, QLibraryInfo
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QAction)
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
from pydm.widgets import PyDMByteIndicator, PyDMRelatedDisplayButton


class MITRMainWindow(PyDMMainWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.macros = kwargs.get('macros', {})
            self.customize_ui()


        def customize_ui(self):
            from application import MITRApplication
            app = MITRApplication.instance()
            icon_path = str(Path("./NRL_Logo.png").resolve())
            self.setWindowIcon(QtGui.QIcon(icon_path))

            bar = self.statusBar()
            heartbeat_indicator = PyDMByteIndicator(init_channel=f"ca://{self.macros['P']}HEARTBEAT")
            heartbeat_indicator.labels = ['IOC Heartbeat']
            heartbeat_indicator.labelPosition = 2

            bar.addPermanentWidget(heartbeat_indicator)

            gear_icon = qta.icon('fa6s.gear')
            # action = QAction(gear_icon, 'Controls', self)
            controls = PyDMRelatedDisplayButton(filename="/home/mitr_4dh4/EPICS/IOCs/4dh4/4dh4App/op/adl/ioc_motors.adl")
            controls.macros = ','.join(['='.join(items) for items in self.macros.items()])
            controls.setText("Controls")
            controls.setIcon(gear_icon)
            controls.openInNewWindow = True
            #move the label to below the icon
            controls.iconPosition = 0
            # set the size of the icon
            controls.setIconSize(QSize(25, 25))

            controlsAll = PyDMRelatedDisplayButton(filename="extra_ui/4dh4All.ui")
            controlsAll.macros = ','.join(['='.join(items) for items in self.macros.items()])
            controlsAll.setText("Controls")
            controlsAll.setIcon(gear_icon)
            controlsAll.openInNewWindow = True
            controlsAll.setIconSize(QSize(25, 25))

            self.ui.navbar.addWidget(controls)
            self.ui.navbar.addWidget(controlsAll)


        def update_window_title(self):
            if self.showing_file_path_in_title_bar:
                title = self.current_file()
            else:
                title = self.display_widget().windowTitle()
            title += " - MITR 4DH4 Beamline Controls"
            if data_plugins.is_read_only():
                title += " [Read Only Mode]"
            self.setWindowTitle(title)


