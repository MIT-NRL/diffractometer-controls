import logging
import warnings
import subprocess
import time
import math
from pathlib import Path

import qtawesome as qta
from epics import caput
from pydm import data_plugins
from pydm.display import load_file, ScreenTarget
from pydm.main_window import PyDMMainWindow
from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt, QTimer, Slot, QSize, QLibraryInfo
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel, QAction, QToolButton, QMessageBox, QProgressBar, QSizePolicy)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)
try:
    from diffractometer_controls.re_plan_editor_widget import RePlanEditorWidget
except Exception:
    from re_plan_editor_widget import RePlanEditorWidget
from bluesky_widgets.models.run_engine_client import RunEngineClient
from pydm.widgets import PyDMByteIndicator, PyDMRelatedDisplayButton
from bluesky_queueserver_api.zmq import REManagerAPI
from pydm.widgets.channel import PyDMChannel

class MITRMainWindow(PyDMMainWindow):
    re_manager_api: REManagerAPI
    _RUN_STATE_INDEX_MAP = {
        0: "IDLE",
        1: "RUNNING",
        2: "PAUSED",
        3: "DONE",
        4: "ABORTED",
        5: "FAILED",
        6: "CLOSED",
        7: "STALE",
        8: "SUSPENDED",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macros = kwargs.get('macros', {})
        self.macros_str = ','.join(['='.join(items) for items in self.macros.items()])
        self._run_state = "IDLE"
        self._run_start_epoch = 0.0
        self._run_finish_epoch = 0.0
        self._run_last_update_epoch = 0.0
        self._run_done_units = 0
        self._run_total_units = 0
        self._run_plan_name = ""
        self._run_initial_remaining_s = 0.0
        self._run_suspended = False
        self._run_progress_frozen_value = 0
        self._run_is_suspended_display = False
        self._run_channels = []
        self._run_display_state = "IDLE"
        self._run_pulse_period_s = 2.8
        self._run_state_font_px = 16
        self._run_finish_font_px = 16
        self._run_progress_font_px = 15
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
        self._setup_run_status_widget(self.ui.navbar)

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
            lambda: self._set_reactor_power_suspender_enabled(False)
        )

        # Install Reactor Power Suspender action
        install_suspender_action = suspender_menu.addAction("Install Reactor Power Suspender")
        install_suspender_action.setIcon(qta.icon('fa5s.plus'))
        install_suspender_action.setToolTip("Install the reactor power suspender to the Run Engine")
        install_suspender_action.triggered.connect(
            lambda: self._set_reactor_power_suspender_enabled(True)
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

    def _setup_run_status_widget(self, toolbar):
        prefix = f"{self.macros.get('P', '')}Bluesky:Run:"
        panel = QWidget(self)
        self._run_status_panel = panel
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(8)

        self._run_state_label = QLabel("Run: IDLE", panel)
        self._run_state_label.setMinimumWidth(0)
        self._run_state_label.setAlignment(Qt.AlignCenter)
        self._run_state_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._run_state_label.setStyleSheet(self._run_state_style("IDLE"))

        self._run_finish_label = QLabel("Finish: --", panel)
        self._run_finish_label.setMinimumWidth(250)
        self._run_finish_label.setMaximumWidth(250)
        self._run_finish_label.setAlignment(Qt.AlignCenter)
        self._run_finish_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        self._run_progress = QProgressBar(panel)
        self._run_progress.setMinimumWidth(0)
        self._run_progress.setMaximumWidth(16777215)
        self._run_progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._run_progress.setRange(0, 1000)
        self._run_progress.setValue(0)
        self._run_progress.setTextVisible(True)
        self._run_progress.setFormat("No active run")
        self._run_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #9ca3af; border-radius: 5px; "
            "background-color: #f3f4f6; color: #111827; text-align: center; } "
            "QProgressBar::chunk { background-color: #60a5fa; }"
        )

        layout.addWidget(self._run_state_label, 1)
        layout.addWidget(self._run_progress, 1)
        layout.addWidget(self._run_finish_label, 0)
        toolbar.addWidget(panel)

        self._run_channels = [
            PyDMChannel(address=f"ca://{prefix}State", value_slot=self._on_run_state_changed),
            PyDMChannel(address=f"ca://{prefix}Suspended", value_slot=self._on_run_suspended_changed),
            PyDMChannel(address=f"ca://{prefix}StartEpoch", value_slot=self._on_run_start_epoch_changed),
            PyDMChannel(address=f"ca://{prefix}FinishEpoch", value_slot=self._on_run_finish_epoch_changed),
            PyDMChannel(address=f"ca://{prefix}LastUpdateEpoch", value_slot=self._on_run_last_update_epoch_changed),
            PyDMChannel(address=f"ca://{prefix}DoneUnits", value_slot=self._on_run_done_units_changed),
            PyDMChannel(address=f"ca://{prefix}TotalUnits", value_slot=self._on_run_total_units_changed),
            PyDMChannel(address=f"ca://{prefix}PlanName", value_slot=self._on_run_plan_name_changed),
        ]
        for ch in self._run_channels:
            ch.connect()

        self._run_eta_timer = QTimer(self)
        self._run_eta_timer.timeout.connect(self._update_run_status_widget)
        self._run_eta_timer.start(1000)

        self._run_anim_timer = QTimer(self)
        self._run_anim_timer.timeout.connect(self._tick_run_state_animation)
        self._run_anim_timer.start(50)
        QtCore.QTimer.singleShot(0, self._apply_run_widget_scale)

    def _apply_run_widget_scale(self):
        toolbar = getattr(getattr(self, "ui", None), "navbar", None)
        if toolbar is None or not hasattr(self, "_run_state_label"):
            return
        h = max(18, int(toolbar.height()))
        self._run_state_font_px = max(10, min(int(h * 0.33), 20))
        self._run_finish_font_px = max(10, min(int(h * 0.33), 20))
        self._run_progress_font_px = max(10, min(int(h * 0.30), 18))

        self._run_finish_label.setStyleSheet(
            f"font-size: {self._run_finish_font_px}px; font-weight: 700; color: #1f2937;"
        )
        self._run_progress.setStyleSheet(
            "QProgressBar { border: 1px solid #9ca3af; border-radius: 5px; "
            "background-color: #f3f4f6; color: #111827; text-align: center; "
            f"font-size: {self._run_progress_font_px}px; font-weight: 700; }} "
            "QProgressBar::chunk { background-color: #60a5fa; }"
        )
        self._run_state_label.setStyleSheet(self._run_state_style(self._run_display_state))

    @staticmethod
    def _blend_hex_rgb(color_a, color_b, t):
        t = max(0.0, min(1.0, float(t)))
        ra, ga, ba = color_a
        rb, gb, bb = color_b
        r = int(round(ra + (rb - ra) * t))
        g = int(round(ga + (gb - ga) * t))
        b = int(round(ba + (bb - ba) * t))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _run_state_style(self, state, pulse_t=0.5):
        if state == "RUNNING":
            # Pulse the red badge brightness with a sine profile.
            bg = self._blend_hex_rgb((252, 165, 165), (254, 226, 226), pulse_t)
            border = self._blend_hex_rgb((248, 113, 113), (254, 202, 202), pulse_t)
            return (
                f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #7f1d1d; "
                f"background-color: {bg}; border: 1px solid {border}; "
                "border-radius: 6px; padding: 3px 8px;"
            )

        palette = {
            "PAUSED": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #92400e; background-color: #fef3c7; border: 1px solid #fcd34d; border-radius: 6px; padding: 3px 8px;",
            "SUSPENDED": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #7c2d12; background-color: #ffedd5; border: 1px solid #fdba74; border-radius: 6px; padding: 3px 8px;",
            "DONE": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #065f46; background-color: #d1fae5; border: 1px solid #86efac; border-radius: 6px; padding: 3px 8px;",
            "ABORTED": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #991b1b; background-color: #fee2e2; border: 1px solid #fca5a5; border-radius: 6px; padding: 3px 8px;",
            "FAILED": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #991b1b; background-color: #fee2e2; border: 1px solid #fca5a5; border-radius: 6px; padding: 3px 8px;",
            "CLOSED": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #6b7280; background-color: #f3f4f6; border: 1px solid #d1d5db; border-radius: 6px; padding: 3px 8px;",
            "STALE": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #92400e; background-color: #fef3c7; border: 1px solid #fcd34d; border-radius: 6px; padding: 3px 8px;",
            "IDLE": f"font-size: {self._run_state_font_px}px; font-weight: 800; color: #334155; background-color: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 6px; padding: 3px 8px;",
        }
        return palette.get(state, palette["IDLE"])

    def _tick_run_state_animation(self):
        if self._run_display_state != "RUNNING":
            return
        now = time.monotonic()
        phase = (2.0 * math.pi * now) / max(self._run_pulse_period_s, 0.2)
        pulse_t = 0.5 + 0.5 * math.sin(phase)
        self._run_state_label.setStyleSheet(self._run_state_style("RUNNING", pulse_t=pulse_t))

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return int(default)

    def _on_run_state_changed(self, value):
        state = "IDLE"
        if isinstance(value, (int, float)):
            state = self._RUN_STATE_INDEX_MAP.get(int(value), str(value).strip().upper())
        elif value is not None:
            raw = str(value).strip()
            if raw.isdigit():
                state = self._RUN_STATE_INDEX_MAP.get(int(raw), raw.upper())
            else:
                state = raw.upper()
        if state in ("DONE", "ABORTED", "FAILED"):
            # Present a simplified lifecycle in the toolbar: active vs idle.
            state = "IDLE"
            self._run_suspended = False
            self._run_start_epoch = 0.0
            self._run_finish_epoch = 0.0
            self._run_done_units = 0
            self._run_total_units = 0
            self._run_initial_remaining_s = 0.0
        if state == "RUNNING" and self._run_state != "RUNNING":
            self._run_initial_remaining_s = 0.0
        self._run_state = state or "IDLE"
        self._update_run_status_widget()

    def _on_run_suspended_changed(self, value):
        if isinstance(value, str):
            raw = value.strip().lower()
            self._run_suspended = raw in ("1", "true", "yes", "on")
        else:
            try:
                self._run_suspended = bool(int(float(value)))
            except Exception:
                self._run_suspended = bool(value)
        self._update_run_status_widget()

    def _on_run_finish_epoch_changed(self, value):
        self._run_finish_epoch = self._as_float(value, default=0.0)
        if self._run_state == "RUNNING" and self._run_finish_epoch > 0:
            remaining = max(0.0, self._run_finish_epoch - time.time())
            if self._run_initial_remaining_s <= 0.0 or remaining > self._run_initial_remaining_s:
                self._run_initial_remaining_s = remaining
        self._update_run_status_widget()

    def _on_run_start_epoch_changed(self, value):
        self._run_start_epoch = self._as_float(value, default=0.0)
        self._update_run_status_widget()

    def _on_run_last_update_epoch_changed(self, value):
        self._run_last_update_epoch = self._as_float(value, default=0.0)
        self._update_run_status_widget()

    def _on_run_done_units_changed(self, value):
        self._run_done_units = max(0, self._as_int(value, default=0))
        self._update_run_status_widget()

    def _on_run_total_units_changed(self, value):
        self._run_total_units = max(0, self._as_int(value, default=0))
        self._update_run_status_widget()

    def _on_run_plan_name_changed(self, value):
        self._run_plan_name = str(value).strip() if value is not None else ""
        self._update_run_status_widget()

    @staticmethod
    def _format_hms(seconds):
        sec = max(0, int(round(seconds)))
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _update_run_status_widget(self):
        now = time.time()
        state = self._run_state
        if self._run_suspended and state in ("RUNNING", "PAUSED", "STALE", "SUSPENDED"):
            state = "SUSPENDED"

        stale_timeout_s = 20.0
        if state == "RUNNING" and self._run_last_update_epoch > 0:
            # If finish_epoch is known, avoid marking long exposures as stale just
            # because there are no mid-exposure updates. Only mark stale after ETA
            # has passed and updates are still missing.
            if self._run_finish_epoch > 0:
                finish_grace_s = 30.0
                if (now > (self._run_finish_epoch + finish_grace_s)) and (
                    (now - self._run_last_update_epoch) > stale_timeout_s
                ):
                    state = "STALE"
            elif (now - self._run_last_update_epoch) > stale_timeout_s:
                state = "STALE"

        self._run_display_state = state
        self._run_state_label.setText(f"Run: {state}")
        self._run_state_label.setStyleSheet(self._run_state_style(state))

        if state in ("SUSPENDED", "PAUSED"):
            self._run_finish_label.setText("Finish: pending")
        elif self._run_finish_epoch > 0:
            finish_local = QtCore.QDateTime.fromSecsSinceEpoch(
                int(self._run_finish_epoch), QtCore.Qt.LocalTime
            ).toString("yyyy-MM-dd HH:mm:ss")
            self._run_finish_label.setText(f"Finish: {finish_local}")
        else:
            self._run_finish_label.setText("Finish: --")

        if state in ("RUNNING", "STALE"):
            self._run_is_suspended_display = False
            remaining = max(0.0, self._run_finish_epoch - now) if self._run_finish_epoch > 0 else 0.0

            if self._run_initial_remaining_s <= 0.0 and self._run_finish_epoch > 0:
                self._run_initial_remaining_s = max(remaining, 1.0)

            frac_time = 0.0
            if self._run_start_epoch > 0 and self._run_finish_epoch > self._run_start_epoch:
                total = self._run_finish_epoch - self._run_start_epoch
                elapsed = now - self._run_start_epoch
                frac_time = min(1.0, max(0.0, elapsed / total))
            elif self._run_initial_remaining_s > 0:
                frac_time = 1.0 - min(1.0, remaining / self._run_initial_remaining_s)
                frac_time = max(0.0, frac_time)

            frac_units = 0.0
            if self._run_total_units > 0:
                frac_units = min(1.0, max(0.0, float(self._run_done_units) / float(self._run_total_units)))

            # Use whichever signal indicates more progress to keep the bar active
            # during long steps even when done_units are only updated at boundaries.
            frac = max(frac_units, frac_time)
            value = int(round(frac * 1000))

            self._run_progress.setRange(0, 1000)
            self._run_progress.setValue(value)
            plan_txt = f" [{self._run_plan_name}]" if self._run_plan_name else ""
            self._run_progress.setFormat(f"{self._format_hms(remaining)} remaining{plan_txt}")
        elif state in ("SUSPENDED", "PAUSED"):
            if not self._run_is_suspended_display:
                frozen = self._run_progress.value()
                if self._run_total_units > 0:
                    frac_units = min(1.0, max(0.0, float(self._run_done_units) / float(self._run_total_units)))
                    frozen = int(round(frac_units * 1000))
                self._run_progress_frozen_value = max(0, min(1000, int(frozen)))
                self._run_is_suspended_display = True
            self._run_progress.setRange(0, 1000)
            self._run_progress.setValue(self._run_progress_frozen_value)
            plan_txt = f" [{self._run_plan_name}]" if self._run_plan_name else ""
            if state == "PAUSED":
                self._run_progress.setFormat(f"Paused{plan_txt}")
            else:
                self._run_progress.setFormat(f"Suspended{plan_txt}")
        elif state == "DONE":
            self._run_is_suspended_display = False
            self._run_progress.setRange(0, 1000)
            self._run_progress.setValue(1000)
            self._run_progress.setFormat("Completed")
        elif state in ("ABORTED", "FAILED"):
            self._run_is_suspended_display = False
            self._run_progress.setRange(0, 1000)
            self._run_progress.setValue(0)
            self._run_progress.setFormat(state.title())
        else:
            self._run_is_suspended_display = False
            self._run_progress.setRange(0, 1000)
            self._run_progress.setValue(0)
            self._run_progress.setFormat("No active run")


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

    def _set_reactor_power_suspender_enabled(self, enabled):
        prefix = f"{self.macros.get('P', '')}Bluesky:SuspenderEnable"
        try:
            caput(prefix, 1 if enabled else 0, wait=False)
            return
        except Exception:
            pass

        # Fallback: direct RE script command if CA path is unavailable.
        cmd = (
            "RE.install_suspender(reactor_power_suspender)"
            if enabled
            else "RE.remove_suspender(reactor_power_suspender)"
        )
        self.re_manager_api.script_upload(cmd)

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
