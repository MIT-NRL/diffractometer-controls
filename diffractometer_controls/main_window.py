import logging
import warnings
import subprocess
from pathlib import Path

import qtawesome as qta
from pydm import data_plugins
from pydm.display import load_file, ScreenTarget
from pydm.main_window import PyDMMainWindow
from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt, QTimer, Slot, QSize, QLibraryInfo
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QAction, QToolButton, QMessageBox)
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
from bluesky_queueserver_api.zmq import REManagerAPI

class MITRMainWindow(PyDMMainWindow):
    re_manager_api: REManagerAPI

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macros = kwargs.get('macros', {})
        self.macros_str = ','.join(['='.join(items) for items in self.macros.items()])
        from application import MITRApplication
        app = MITRApplication.instance()
        self.re_manager_api = app.re_manager_api
        self.customize_ui()


    def customize_ui(self):
        # from application import MITRApplication
        # app = MITRApplication.instance()
        icon_path = str(Path("./NRL_Logo.png").resolve())
        self.setWindowIcon(QtGui.QIcon(icon_path))
        re_manager_api = self.re_manager_api

        bar = self.statusBar()
        heartbeat_indicator = PyDMByteIndicator(init_channel=f"ca://{self.macros['P']}HEARTBEAT")
        heartbeat_indicator.labels = ['IOC Heartbeat']
        heartbeat_indicator.labelPosition = 2

        bar.addPermanentWidget(heartbeat_indicator)

        gear_icon = qta.icon('fa6s.gear')
        # controls = PyDMRelatedDisplayButton(filename="/home/mitr_4dh4/EPICS/IOCs/4dh4/4dh4App/op/adl/ioc_motors.adl")
        # controls.macros = self.macros_str
        # controls.setText("Controls")
        # controls.setIcon(gear_icon)
        # controls.openInNewWindow = True
        # #move the label to below the icon
        # controls.iconPosition = 0
        # # set the size of the icon
        # controls.setIconSize(QSize(25, 25))
        # Create a QToolButton

        controlsAll = QToolButton(self)
        controlsAll.setIcon(qta.icon('fa6s.gear'))  # Set an appropriate icon
        controlsAll.setText("All Controls")  # Set the text for the button
        controlsAll.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)  # Text below the icon
        controlsAll.setIconSize(QSize(24, 24))  # Match the icon size of the home button


        # Connect the button to the load_file function
        controlsAll.clicked.connect(lambda: load_file(
            file="extra_ui/4dh4All.ui",
            macros=self.macros,
        ))

        # Add the button to the navbar
        self.ui.navbar.addWidget(controlsAll)

        # controlsAll = CustomRelatedDisplayButtonWrapper(
        #     parent=self,
        #     filename="extra_ui/4dh4All.ui",
        #     macros=self.macros_str,
        #     icon=gear_icon,
        #     text="All Controls",
        # )

        # self.ui.navbar.addWidget(controls)
        # self.ui.navbar.addWidget(controlsAll)

        # Add a "Control System" menu to the menu bar
        control_system_menu = self.menuBar().addMenu("Control System")

        # Add a "Bluesky Controls" submenu
        bluesky_menu = control_system_menu.addMenu("Bluesky Controls")

        # Add vscode editor of the bluesky directory
        bluesky_vscode = bluesky_menu.addAction("Edit Bluesky Files")
        bluesky_vscode.triggered.connect(lambda: subprocess.Popen(["code", "-n", "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls"]))
        bluesky_vscode.setIcon(qta.icon('fa6.file-code'))
        bluesky_vscode.setToolTip("Edit the Bluesky files in VSCode")
        bluesky_menu.addAction(bluesky_vscode)

        # add line to the menu
        bluesky_menu.addSeparator()

        # Add actions to the "Bluesky Controls" submenu
        bluesky_RE_reset = bluesky_menu.addAction("RE Manager Reset")
        bluesky_RE_reset.triggered.connect(lambda: self.reset_process("queue-server"))
        bluesky_RE_reset.setIcon(qta.icon('fa5s.redo'))
        bluesky_RE_reset.setToolTip("Reset the Bluesky Run Engine Manager")
        bluesky_menu.addAction(bluesky_RE_reset)

        # Add actions to the "Bluesky Controls" submenu
        bluesky_proxy_reset = bluesky_menu.addAction("RE Proxy Reset")
        bluesky_proxy_reset.triggered.connect(lambda: self.reset_process("bluesky-proxy"))
        bluesky_proxy_reset.setIcon(qta.icon('fa5s.redo'))
        bluesky_proxy_reset.setToolTip("Reset the Bluesky Run Engine Proxy")
        bluesky_menu.addAction(bluesky_proxy_reset)

        # Add Bluesky GUI reset action to the "Bluesky Controls" submenu
        bluesky_gui_reset = bluesky_menu.addAction("GUI Reset")
        bluesky_gui_reset.triggered.connect(lambda: self.control_servers("4dh4gui", "restart"))
        bluesky_gui_reset.setIcon(qta.icon('fa5s.redo'))
        bluesky_gui_reset.setToolTip("Reset the Bluesky GUI")
        bluesky_menu.addAction(bluesky_gui_reset)

        # Add separator before suspender actions
        bluesky_menu.addSeparator()

        # Add a "Suspender" submenu under "Bluesky Controls"
        suspender_menu = bluesky_menu.addMenu("Suspender")

        # Remove Reactor Power Suspender action
        remove_suspender_action = suspender_menu.addAction("Remove Reactor Power Suspender")
        remove_suspender_action.setIcon(qta.icon('fa5s.trash'))
        remove_suspender_action.setToolTip("Remove the reactor power suspender from the Run Engine")
        remove_suspender_action.triggered.connect(
            lambda: re_manager_api.script_upload("RE.remove_suspender(reactor_power_suspender)")
        )

        # Install Reactor Power Suspender action
        install_suspender_action = suspender_menu.addAction("Install Reactor Power Suspender")
        install_suspender_action.setIcon(qta.icon('fa5s.plus'))
        install_suspender_action.setToolTip("Install the reactor power suspender to the Run Engine")
        install_suspender_action.triggered.connect(
            lambda: re_manager_api.script_upload("RE.install_suspender(reactor_power_suspender)")
        )
        

        # Add a "EPICS Controls" submenu
        epics_menu = control_system_menu.addMenu("EPICS Controls")

        # Add vscode editor of the EPICS directory
        epics_vscode = epics_menu.addAction("Edit EPICS IOC Files")
        epics_vscode.triggered.connect(lambda: subprocess.Popen(["code", "-n", "/home/mitr_4dh4/EPICS/IOCs/4dh4"]))
        epics_vscode.setIcon(qta.icon('fa6.file-code'))
        epics_vscode.setToolTip("Edit the EPICS IOC files in VSCode")
        epics_menu.addAction(epics_vscode)

        epics_top_vscode = epics_menu.addAction("Edit EPICS Files")
        epics_top_vscode.triggered.connect(lambda: subprocess.Popen(["code", "-n", "/home/mitr_4dh4/EPICS"]))
        epics_top_vscode.setIcon(qta.icon('fa6.file-code'))
        epics_top_vscode.setToolTip("Edit the EPICS files in VSCode")
        epics_menu.addAction(epics_top_vscode)

        # add line to the menu
        epics_menu.addSeparator()

        # Add start action to the "EPICS Controls" submenu
        epics_ioc_start = epics_menu.addAction("IOC Start")
        epics_ioc_start.triggered.connect(lambda: self.control_servers("4dh4ioc", "start"))
        epics_ioc_start.setIcon(qta.icon('fa5s.play'))
        epics_ioc_start.setToolTip("Start the EPICS IOC")
        epics_menu.addAction(epics_ioc_start)

        # Add actions to the "EPICS Controls" submenu
        epics_ioc_reset = epics_menu.addAction("IOC Reset")
        epics_ioc_reset.triggered.connect(lambda: self.control_servers("4dh4ioc", "restart"))
        epics_ioc_reset.setIcon(qta.icon('fa5s.redo'))
        epics_ioc_reset.setToolTip("Reset the EPICS IOC")
        epics_menu.addAction(epics_ioc_reset)

        # Add stop action to the "EPICS Controls" submenu
        epics_ioc_stop = epics_menu.addAction("IOC Stop")
        epics_ioc_stop.triggered.connect(lambda: self.control_servers("4dh4ioc", "stop"))
        epics_ioc_stop.setIcon(qta.icon('fa5s.stop'))
        epics_ioc_stop.setToolTip("Stop the EPICS IOC")
        epics_menu.addAction(epics_ioc_stop)

        # Add the "Controls" action to the menu bar
        controls_action = QAction(gear_icon, "Old Control Menu", self)
        controls_action.triggered.connect(lambda: load_file(
            file="/home/mitr_4dh4/EPICS/IOCs/4dh4/4dh4App/op/adl/ioc_motors.adl",
            macros=self.macros,
            # open_in_new_window=True
        ))
        control_system_menu.addAction(controls_action)


    def update_window_title(self):
        if self.showing_file_path_in_title_bar:
            title = self.current_file()
        else:
            title = self.display_widget().windowTitle()
        title += " - MITR 4DH4 Beamline Controls"
        if data_plugins.is_read_only():
            title += " [Read Only Mode]"
        self.setWindowTitle(title)

    def reset_process(self, process_name):
        """
        Resets a given process using systemctl.

        Parameters:
            process_name (str): The name of the process to reset.
        """
        try:
            # Run the systemctl command to restart the service
            subprocess.run(
                ["systemctl", "--user", "restart", f"{process_name}.service"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Show a success message
            QMessageBox.information(self, "Success", f"{process_name} restarted successfully.")
        except subprocess.CalledProcessError as e:
            # Show an error message if the command fails
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to restart {process_name}.\n\nError: {e.stderr.decode('utf-8')}",
            )
        except Exception as e:
            # Catch any other exceptions
            QMessageBox.critical(
                self,
                "Error",
                f"An unexpected error occurred while restarting {process_name}:\n\n{str(e)}",
            )

    def control_servers(self, server_name, command):
        """
        Controls a server by running the specified command in an interactive Bash shell.

        Parameters:
            server_name (str): The name of the server to control.
            command (str): The command to execute (e.g., "restart", "start", "stop").
        """
        try:
            # Construct the full command
            full_command = f"{server_name} {command}"
            
            # Run the command in an interactive Bash shell
            subprocess.run(
                ["bash", "-i", "-c", full_command],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Show a success message
            QMessageBox.information(None, "Success", f"{server_name} {command} executed successfully.")
        except subprocess.CalledProcessError as e:
            # Show an error message if the command fails
            QMessageBox.critical(
                None,
                "Error",
                f"Failed to execute {server_name} {command}.\n\nError: {e.stderr.decode('utf-8')}",
            )
        except Exception as e:
            # Catch any other exceptions
            QMessageBox.critical(
                None,
                "Error",
                f"An unexpected error occurred while executing {server_name} {command}:\n\n{str(e)}",
            )




# from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
# from pydm.widgets.related_display_button import PyDMRelatedDisplayButton
# from qtpy import QtWidgets

# class CustomRelatedDisplayButtonWrapper(QWidget):
#     def __init__(self, parent=None, filename=None, macros=None, icon=None, text=""):
#         super().__init__(parent)

#         # Create a vertical layout for the wrapper
#         layout = QVBoxLayout(self)
#         layout.setContentsMargins(0, 0, 0, 0)  # Remove margins
#         layout.setSpacing(0)  # Set small positive spacing between icon and text

#         # Create the PyDMRelatedDisplayButton
#         self.button = PyDMRelatedDisplayButton(parent, filename=filename)
#         self.button.macros = macros
#         self.button.setIcon(icon)
#         self.button.setText("")  # Remove default text from the button
#         self.button.setIconSize(QSize(24, 24))  # Set icon size
#         self.button.openInNewWindow = True
#         self.button.setStyleSheet('''
#             QPushButton {
#                 background-color: transparent;  /* Transparent button */
#                 border: none;  /* Remove border */
#                 padding: -1px;  /* Remove internal padding */
#                 margin: 0px;  /* Remove internal margins */
#             }
#         ''')
#         self.button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

#         # Add a QLabel for the text below the button
#         self.text_label = QLabel(text, self)
#         self.text_label.setAlignment(Qt.AlignCenter)
#         self.text_label.setStyleSheet('''
#             QLabel {
#                 font-size: 10px;  /* Match the correct font size */
#                 color: black;  /* Match the text color */
#                 padding: 0px;  /* Remove padding */
#                 margin: 0px;  /* Remove margins */
#             }
#         ''')
#         self.text_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

#         # Add the button and label to the layout
#         layout.addWidget(self.button, alignment=Qt.AlignHCenter)
#         layout.addWidget(self.text_label, alignment=Qt.AlignHCenter)

#         self.setLayout(layout)

#     def setIcon(self, icon):
#         self.button.setIcon(icon)

#     def setText(self, text):
#         self.text_label.setText(text)