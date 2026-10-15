"""Local process controls and EPICS health for the diffractometer IOCs."""

from __future__ import annotations

import codecs
import re
import time

from pydm import Display, data_plugins
from pydm.widgets import PyDMLabel
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore, QtGui, QtWidgets
from control_ui.widgets.service_status import ResourceGrid, ServiceCard, StatusBadge, configure_console_font, overview_page, status_legend

from diffractometer_controls.maintenance.local_maintenance import LOCAL_MAINTENANCE_DISABLED_MESSAGE, local_maintenance_allowed


from diffractometer_controls.site.mitr.ioc_specs import IOCSpec, IOC_SPECS, ioc_spec


def process_command(key: str, action: str) -> tuple[str, list[str]]:
    """Return a service action as argv, with no shell or implicit group startup."""
    if action not in {"start", "stop", "restart"}:
        raise ValueError(f"Unsupported IOC action: {action}")
    return "systemctl", ["--user", action, ioc_spec(key).service]


def status_command(key: str) -> tuple[str, list[str]]:
    return "systemctl", ["--user", "show", ioc_spec(key).service,
                         "--property=LoadState,ActiveState,SubState,MainPID", "--no-pager"]


def journal_command(key: str) -> tuple[str, list[str]]:
    return "journalctl", ["--user", "--unit=" + ioc_spec(key).service,
                          "--lines=2000", "--follow", "--output=cat", "--no-pager"]


_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_SCREEN_CARET_ESCAPE = re.compile(r"\^\[\[[0-?]*[ -/]*[@-~]")


def clean_console_text(value) -> str:
    """Remove terminal cursor/control sequences from a persisted console log."""
    text = str(value or "")
    text = _ANSI_ESCAPE.sub("", text)
    text = _SCREEN_CARET_ESCAPE.sub("", text)
    while "\b" in text:
        updated = re.sub(r"[^\n]\x08", "", text)
        if updated == text:
            break
        text = updated
    text = text.replace("\b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _dialog_button(text: str) -> QtWidgets.QPushButton:
    """Create a dialog button that Enter cannot trigger implicitly."""
    button = QtWidgets.QPushButton(text)
    button.setAutoDefault(False)
    button.setDefault(False)
    return button


def parse_process_status(key: str, output: str, exit_code: int) -> tuple[bool, str]:
    text = str(output or "").strip()
    fields = {}
    for line in text.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            fields[name.strip()] = value.strip()
    running = fields.get("ActiveState") == "active"
    pid = fields.get("MainPID", "0")
    state = "/".join(
        value for value in (fields.get("ActiveState"), fields.get("SubState")) if value
    ) or "unknown"
    if fields.get("LoadState") == "not-found":
        return False, "service not installed"
    return running, f"{state}  •  PID {pid}" if pid != "0" else state


def _value_slot(method):
    for value_type in (int, float, str, bool, object):
        method = QtCore.Slot(value_type)(method)
    return method


def format_percent(value) -> str:
    """Format a devIocStats percentage value."""
    try:
        return f"{float(value):.1f}%"
    except (TypeError, ValueError):
        return "—"


def format_bytes(value) -> str:
    """Format a byte count using compact binary units."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "—"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit = units[0]
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    precision = 0 if unit in {"B", "KiB"} else 1
    return f"{amount:.{precision}f} {unit}"


def format_count(value) -> str:
    """Format a numeric IOC counter as an integer."""
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


RESOURCE_METRICS = (
    ("IOC CPU", "IOC_CPU_LOAD", format_percent, "CPU used by this IOC process."),
    (
        "IOC memory",
        "MEM_USED",
        format_bytes,
        "Allocated memory reported for this IOC process.",
    ),
    ("Host CPU", "SYS_CPU_LOAD", format_percent, "Total CPU load on this computer."),
    ("Host RAM free", "MEM_FREE", format_bytes, "Free physical memory on this computer."),
    ("Open FDs", "FD_CNT", format_count, "File descriptors allocated by this IOC."),
    ("CA clients", "CA_CLNT_CNT", format_count, "Channel Access clients connected to this IOC."),
    (
        "CA connections",
        "CA_CONN_CNT",
        format_count,
        "Channel Access circuits and channels reported by this IOC.",
    ),
    (
        "Suspended tasks",
        "SUSP_TASK_CNT",
        format_count,
        "Suspended IOC tasks; this should normally remain zero.",
    ),
)


class _FormattedPVLabel(QtWidgets.QLabel):
    """Small read-only PV label with display-specific value formatting."""

    def __init__(self, pv: str, formatter, parent=None):
        super().__init__("—", parent)
        self._formatter = formatter
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._channel = PyDMChannel(
            address=f"ca://{pv}",
            connection_slot=self._connection_changed,
            value_slot=self._value_changed,
        )
        self._channel.connect()

    @QtCore.Slot(bool)
    def _connection_changed(self, connected):
        if not connected:
            self.setText("—")

    @_value_slot
    def _value_changed(self, value):
        self.setText(self._formatter(value))

    def close(self):
        self._channel.disconnect()


class _Heartbeat(QtCore.QObject):
    changed = QtCore.Signal(bool, str)

    def __init__(self, pv: str, parent=None):
        super().__init__(parent)
        self._connected = False
        self._last_value = None
        self._last_change = 0.0
        self._channel = PyDMChannel(
            address=f"ca://{pv}",
            connection_slot=self._connection_changed,
            value_slot=self._value_changed,
        )
        self._channel.connect()

    @QtCore.Slot(bool)
    def _connection_changed(self, connected):
        self._connected = bool(connected)
        if not connected:
            self.changed.emit(False, "EPICS disconnected")

    @_value_slot
    def _value_changed(self, value):
        now = time.monotonic()
        if value != self._last_value:
            self._last_change = now
            self._last_value = value
        self.changed.emit(True, f"EPICS online  •  heartbeat {value}")

    def poll(self):
        if self._connected and time.monotonic() - self._last_change > 4.0:
            self.changed.emit(False, "EPICS heartbeat stale")

    def close(self):
        self._channel.disconnect()


class IOCConsoleWindow(QtWidgets.QDialog):
    """Read-only systemd journal with independent find, copy and follow controls."""

    def __init__(self, spec: IOCSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._journal_process = None
        self._closed = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(f"{spec.title} Console")
        self.resize(1050, 720)
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(f"{spec.title} Console")
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(heading)
        layout.addWidget(QtWidgets.QLabel(f"Live journal for {spec.service}. Console output is read-only."))
        toolbar = QtWidgets.QHBoxLayout()
        self.clear_button = _dialog_button("Clear view")
        self.clear_button.clicked.connect(self._clear_view)
        toolbar.addWidget(self.clear_button)
        self.copy_button = _dialog_button("Copy all")
        self.copy_button.clicked.connect(self._copy_all)
        toolbar.addWidget(self.copy_button)
        self.bottom_button = _dialog_button("Jump to bottom")
        self.bottom_button.clicked.connect(self._jump_to_bottom)
        toolbar.addWidget(self.bottom_button)
        self.follow_checkbox = QtWidgets.QCheckBox("Follow output")
        self.follow_checkbox.setChecked(True)
        toolbar.addWidget(self.follow_checkbox)
        self.find_edit = QtWidgets.QLineEdit()
        self.find_edit.setPlaceholderText("Find in output")
        self.find_edit.setClearButtonEnabled(True)
        self.find_edit.returnPressed.connect(self._find_next)
        toolbar.addWidget(self.find_edit, 1)
        self.find_button = _dialog_button("Find next")
        self.find_button.clicked.connect(self._find_next)
        toolbar.addWidget(self.find_button)
        layout.addLayout(toolbar)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.output.document().setMaximumBlockCount(30000)
        configure_console_font(self.output)
        layout.addWidget(self.output, 1)
        self.status_label = QtWidgets.QLabel("Loading and following IOC journal…")
        layout.addWidget(self.status_label)
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_journal)
        process.finished.connect(self._journal_finished)
        process.errorOccurred.connect(self._journal_error)
        self._journal_process = process
        program, arguments = journal_command(spec.key)
        process.start(program, arguments)

    def _append_output(self, text):
        value = clean_console_text(text)
        if not value:
            return
        bar = self.output.verticalScrollBar()
        old = bar.value()
        cursor = QtGui.QTextCursor(self.output.document())
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(value)
        bar.setValue(bar.maximum() if self.follow_checkbox.isChecked() else old)

    @QtCore.Slot()
    def _read_journal(self):
        if self._journal_process is not None:
            self._append_output(self._decoder.decode(bytes(self._journal_process.readAllStandardOutput())))
            self.status_label.setText(f"Following {self.spec.service} journal (read only)")

    def _journal_finished(self, code, state):
        self._read_journal()
        if not self._closed:
            self.status_label.setText(f"Journal stopped (exit {code}). Reopen the console to reconnect.")

    def _journal_error(self, error):
        if not self._closed:
            self.status_label.setText(f"Unable to read journal: {self._journal_process.errorString()}")

    def _clear_view(self):
        self.output.clear()

    def _copy_all(self):
        QtWidgets.QApplication.clipboard().setText(self.output.toPlainText())

    def _jump_to_bottom(self):
        bar = self.output.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _find_next(self):
        text = self.find_edit.text()
        if text and not self.output.find(text):
            self.output.moveCursor(QtGui.QTextCursor.Start)
            if not self.output.find(text):
                self.status_label.setText(f"Not found: {text}")

    def closeEvent(self, event):
        self._closed = True
        if self._journal_process is not None and self._journal_process.state() != QtCore.QProcess.NotRunning:
            self._journal_process.terminate()
            self._journal_process.waitForFinished(1000)
        super().closeEvent(event)


class _IOCRow(ServiceCard):
    def __init__(self, spec: IOCSpec, maintenance: bool, parent=None):
        super().__init__(spec.title, spec.transport, parent)
        self.spec = spec
        self._maintenance = maintenance
        self._status_process = None
        self._control_process = None
        self._console_window = None
        self._resource_labels = []
        self.process_label = QtWidgets.QLabel("Checking process…")
        self.process_label.setTextFormat(QtCore.Qt.PlainText)
        self.process_label.setWordWrap(True)
        self.epics_label = StatusBadge("Checking EPICS heartbeat…", parent=self)
        self.content.addWidget(self.process_label)
        self.content.addWidget(self.epics_label)

        self.resources = ResourceGrid("IOC resource usage", self)
        self.diagnostics = ResourceGrid("Host and connection diagnostics", self)
        core_suffixes = ("IOC_CPU_LOAD", "MEM_USED", "CA_CLNT_CNT")
        for title_text, suffix, formatter, tooltip in RESOURCE_METRICS:
            target = self.resources if suffix in core_suffixes else self.diagnostics
            metric_value = _FormattedPVLabel(f"{spec.stats_prefix}:{suffix}", formatter, target)
            target.add_metric(suffix, title_text, metric_value, tooltip)
            self._resource_labels.append(metric_value)
        uptime = PyDMLabel(init_channel=f"ca://{spec.uptime}")
        uptime.setText("—")
        self.resources.add_metric("uptime", "Uptime", uptime)
        self.content.addWidget(self.resources)
        self.details_button = QtWidgets.QToolButton()
        self.details_button.setText("Host and connection details")
        self.details_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.details_button.setArrowType(QtCore.Qt.RightArrow)
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._show_diagnostics)
        self.content.addWidget(self.details_button, alignment=QtCore.Qt.AlignLeft)
        self.content.addWidget(self.diagnostics)
        self.diagnostics.hide()

        self.buttons = {}
        for action, label in (("start", "Start"), ("restart", "Restart"), ("stop", "Stop")):
            button = self.add_action(label, lambda checked=False, value=action: self.control(value))
            button.setEnabled(maintenance and not data_plugins.is_read_only())
            if not maintenance:
                button.setToolTip(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
            elif data_plugins.is_read_only():
                button.setToolTip("Service controls are disabled in read-only mode.")
            self.buttons[action] = button
        console_button = self.add_action("Console / logs", self.open_console)
        console_button.setEnabled(maintenance)
        if not maintenance:
            console_button.setToolTip(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
        else:
            console_button.setToolTip("Open the live systemd journal (read only).")
        self.actions.addStretch(1)

        self._heartbeat = _Heartbeat(spec.heartbeat, self)
        self._heartbeat.changed.connect(self._set_epics_status)

    def _show_diagnostics(self, visible):
        self.diagnostics.setVisible(visible)
        self.details_button.setArrowType(QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow)

    @QtCore.Slot(bool, str)
    def _set_epics_status(self, ok, text):
        level = "healthy" if ok else ("attention" if "stale" in text.casefold() else "error")
        self.epics_label.set_status(text, level)

    def refresh(self):
        self._heartbeat.poll()
        if self._status_process is not None:
            return
        program, arguments = status_command(self.spec.key)
        process = QtCore.QProcess(self)
        self._status_process = process
        process.finished.connect(self._status_finished)
        process.start(program, arguments)

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _status_finished(self, exit_code, exit_status):
        process = self._status_process
        if process is None:
            return
        output = bytes(process.readAllStandardOutput()).decode(errors="replace")
        error = bytes(process.readAllStandardError()).decode(errors="replace")
        ok, text = parse_process_status(self.spec.key, output or error, int(exit_code))
        self.process_label.setText(f"Process: {text}")
        self.status_badge.set_status("Running" if ok else "Stopped / unavailable", "healthy" if ok else "error")
        process.deleteLater()
        self._status_process = None

    def control(self, action: str):
        if not self._maintenance or self._control_process is not None or data_plugins.is_read_only():
            return
        if action in {"stop", "restart"}:
            answer = QtWidgets.QMessageBox.question(
                self,
                f"{action.title()} {self.spec.title}?",
                f"Are you sure you want to {action} the {self.spec.title.lower()}?",
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        program, arguments = process_command(self.spec.key, action)
        process = QtCore.QProcess(self)
        process.setProperty("ioc_action", action)
        self._control_process = process
        self.process_label.setText(f"Process: {action} requested…")
        self.status_badge.set_status(action.title() + " requested", "active")
        process.finished.connect(self._control_finished)
        process.start(program, arguments)

    def open_console(self):
        if not self._maintenance:
            return
        window = self._console_window
        if window is not None:
            window.show()
            window.raise_()
            window.activateWindow()
            return
        window = IOCConsoleWindow(self.spec, self)
        self._console_window = window
        window.destroyed.connect(self._console_destroyed)
        window.show()

    @QtCore.Slot()
    def _console_destroyed(self):
        self._console_window = None

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _control_finished(self, exit_code, exit_status):
        process = self._control_process
        if process is None:
            return
        action = process.property("ioc_action")
        error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        process.deleteLater()
        self._control_process = None
        if exit_code != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "IOC control failed",
                error or f"{self.spec.title}: {action} failed with exit code {exit_code}.",
            )
        QtCore.QTimer.singleShot(500, self.refresh)

    def close(self):
        if self._console_window is not None:
            self._console_window.close()
        self._heartbeat.close()
        for label in self._resource_labels:
            label.close()
        for process in (self._status_process, self._control_process):
            if process is not None and process.state() != QtCore.QProcess.NotRunning:
                process.terminate()


class IOCControlPanel(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super().__init__(parent=parent, args=args, macros=macros)
        self.setWindowTitle("EPICS IOC Control and Status")
        self.resize(1040, 880)
        layout = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel("EPICS IOC Control and Status")
        heading.setStyleSheet("font-size: 21px; font-weight: 600;")
        layout.addWidget(heading)

        note = QtWidgets.QLabel(
            "The IOCs use independent, manually started services. Process state and "
            "EPICS heartbeat are checked separately."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(status_legend(self))

        self.sections = QtWidgets.QTabWidget()
        layout.addWidget(self.sections, 1)
        overview, cards = overview_page(self)
        self.sections.addTab(overview, "Overview")

        maintenance = local_maintenance_allowed()
        if not maintenance:
            warning = QtWidgets.QLabel(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b3261e; font-weight: 600;")
            layout.addWidget(warning)

        self._rows = [_IOCRow(spec, maintenance, self) for spec in IOC_SPECS]
        for row in self._rows:
            cards.addWidget(row)
        cards.addStretch(1)

        diagnostics = QtWidgets.QWidget()
        diagnostic_layout = QtWidgets.QVBoxLayout(diagnostics)
        cameras = QtWidgets.QGroupBox("Camera IOC acquisition health")
        camera_layout = QtWidgets.QGridLayout(cameras)
        headers = ("Camera", "RTSP", "Control", "Published FPS", "Frame age (s)")
        for column, text in enumerate(headers):
            camera_layout.addWidget(QtWidgets.QLabel(text), 0, column)
        for row_number, camera in enumerate(("Reolink1", "Reolink2"), start=1):
            camera_layout.addWidget(QtWidgets.QLabel(camera), row_number, 0)
            for column, suffix in enumerate(
                ("RTSPConnected_RBV", "ControlConnected_RBV", "PublishedFPS_RBV", "FrameAge_RBV"),
                start=1,
            ):
                camera_layout.addWidget(
                    PyDMLabel(init_channel=f"ca://4dh4:{camera}:{suffix}"),
                    row_number,
                    column,
                )
        diagnostic_layout.addWidget(cameras)
        diagnostic_layout.addStretch(1)
        self.sections.addTab(diagnostics, "Camera health")

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(2500)
        self.refresh()

    def refresh(self):
        for row in self._rows:
            row.refresh()

    def closeEvent(self, event):
        self._timer.stop()
        for row in self._rows:
            row.close()
        super().closeEvent(event)
