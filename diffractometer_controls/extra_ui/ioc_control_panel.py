"""Local process controls and EPICS health for the diffractometer IOCs."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import NamedTuple

from pydm import Display
from pydm.widgets import PyDMLabel
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore, QtGui, QtWidgets

try:
    from diffractometer_controls.local_maintenance import (
        LOCAL_MAINTENANCE_DISABLED_MESSAGE,
        local_maintenance_allowed,
    )
except Exception:
    from local_maintenance import (
        LOCAL_MAINTENANCE_DISABLED_MESSAGE,
        local_maintenance_allowed,
    )


MAIN_LAUNCHER = Path(
    "/home/mitr_4dh4/EPICS/IOCs/4dh4/iocBoot/ioc4dh4/softioc/4dh4.pl"
)
CAMERA_SERVICE = "reolinkioc.service"
MAIN_SCREEN_NAME = "4dh4ioc"
MAIN_CONSOLE_LOG_DIR = MAIN_LAUNCHER.parent / "logs" / "iocConsole"


class IOCSpec(NamedTuple):
    """Static IOC identity safe for PyDM's dynamic display loader."""

    key: str
    title: str
    heartbeat: str
    uptime: str
    stats_prefix: str
    transport: str


IOC_SPECS = (
    IOCSpec(
        "main",
        "Main control IOC",
        "4dh4:HEARTBEAT",
        "4dh4:UPTIME",
        "4dh4",
        "screen 4dh4ioc  •  private CA/PVA endpoints",
    ),
    IOCSpec(
        "camera",
        "Reolink camera IOC",
        "4dh4:ReolinkIOC:HEARTBEAT",
        "4dh4:ReolinkIOC:UPTIME",
        "4dh4:ReolinkIOC",
        "systemd reolinkioc.service  •  private CA/PVA endpoints",
    ),
)


def process_command(key: str, action: str) -> tuple[str, list[str]]:
    """Return an argv-only control command; never invoke a shell."""
    if action not in {"start", "stop", "restart"}:
        raise ValueError(f"Unsupported IOC action: {action}")
    if key == "main":
        return str(MAIN_LAUNCHER), [action]
    if key == "camera":
        return "systemctl", ["--user", action, CAMERA_SERVICE]
    raise ValueError(f"Unknown IOC: {key}")


def status_command(key: str) -> tuple[str, list[str]]:
    if key == "main":
        return str(MAIN_LAUNCHER), ["status"]
    if key == "camera":
        return "systemctl", [
            "--user",
            "show",
            CAMERA_SERVICE,
            "--property=LoadState",
            "--property=ActiveState",
            "--property=SubState",
            "--property=MainPID",
            "--no-pager",
        ]
    raise ValueError(f"Unknown IOC: {key}")


def camera_log_command() -> tuple[str, list[str]]:
    """Return the live systemd journal command for the camera IOC."""
    return (
        "journalctl",
        [
            "--user",
            f"--unit={CAMERA_SERVICE}",
            "--lines=2000",
            "--follow",
            "--output=cat",
            "--no-hostname",
        ],
    )


def screen_input_commands(command: str) -> tuple[tuple[str, list[str]], ...]:
    """Return shell-free Screen operations for text followed by the Enter key."""
    value = str(command).replace("\r", " ").replace("\n", " ").strip()
    base_arguments = ["-S", MAIN_SCREEN_NAME, "-p", "0", "-X", "stuff"]
    operations = []
    if value:
        operations.append(("screen", [*base_arguments, value]))
    operations.append(("screen", [*base_arguments, "\r"]))
    return tuple(operations)


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
    if key == "main":
        running = exit_code == 0 and " is running " in f" {text} "
        return running, text or "No launcher status returned"

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
    """Scrollable GUI output viewer with optional main-IOC shell input."""

    def __init__(self, spec: IOCSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._main_log_path = None
        self._main_log_position = 0
        self._journal_process = None
        self._command_process = None
        self._command_queue = []
        self._pending_command = ""
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        self.setWindowTitle(f"{spec.title} Console")
        self.resize(1050, 720)

        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel(f"{spec.title} Console")
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(heading)

        if spec.key == "main":
            explanation = (
                "Live Screen console log. Commands entered below are sent to the "
                "running EPICS IOC shell."
            )
        else:
            explanation = (
                "Live systemd journal with recent startup output. The camera IOC "
                "console is read-only."
            )
        note = QtWidgets.QLabel(explanation)
        note.setWordWrap(True)
        layout.addWidget(note)

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
        toolbar.addSpacing(20)
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
        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.output.setFont(fixed_font)
        layout.addWidget(self.output, 1)

        self.status_label = QtWidgets.QLabel("Opening console output…")
        layout.addWidget(self.status_label)

        if spec.key == "main":
            command_layout = QtWidgets.QHBoxLayout()
            command_layout.addWidget(QtWidgets.QLabel("IOC command:"))
            self.command_edit = QtWidgets.QLineEdit()
            self.command_edit.setPlaceholderText(
                "Enter an IOC shell command, for example: dbl"
            )
            self.command_edit.setMaxLength(2048)
            self.command_edit.returnPressed.connect(self._send_main_command)
            command_layout.addWidget(self.command_edit, 1)
            self.send_button = _dialog_button("Send")
            self.send_button.clicked.connect(self._send_main_command)
            command_layout.addWidget(self.send_button)
            layout.addLayout(command_layout)
        else:
            self.command_edit = None
            self.send_button = None

        if spec.key == "main":
            self._poll_timer = QtCore.QTimer(self)
            self._poll_timer.timeout.connect(self._poll_main_log)
            self._poll_timer.start(500)
            self._poll_main_log()
        else:
            self._poll_timer = None
            self._start_camera_journal()

    def _append_output(self, text):
        value = clean_console_text(text)
        if not value:
            return
        scroll_bar = self.output.verticalScrollBar()
        old_scroll = scroll_bar.value()
        cursor = QtGui.QTextCursor(self.output.document())
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(value)
        if self.follow_checkbox.isChecked():
            self._jump_to_bottom()
        else:
            scroll_bar.setValue(old_scroll)

    def _active_main_log(self):
        try:
            files = list(MAIN_CONSOLE_LOG_DIR.glob("ioc4dh4-console.log_*"))
            return max(files, key=lambda path: path.stat().st_mtime) if files else None
        except OSError:
            return None

    def _poll_main_log(self):
        path = self._active_main_log()
        if path is None:
            self.status_label.setText(f"No console log found in {MAIN_CONSOLE_LOG_DIR}")
            return
        if path != self._main_log_path:
            self._main_log_path = path
            self._main_log_position = 0
            self._append_output(f"\n===== {path.name} =====\n")
        try:
            size = path.stat().st_size
            if size < self._main_log_position:
                self._main_log_position = 0
            with path.open("rb") as stream:
                stream.seek(self._main_log_position)
                data = stream.read()
                self._main_log_position = stream.tell()
        except OSError as ex:
            self.status_label.setText(f"Unable to read console log: {ex}")
            return
        if data:
            self._append_output(data.decode("utf-8", errors="replace"))
        self.status_label.setText(f"Following {path.name}")

    def _start_camera_journal(self):
        program, arguments = camera_log_command()
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_camera_journal)
        process.finished.connect(self._camera_journal_finished)
        process.errorOccurred.connect(self._camera_journal_error)
        self._journal_process = process
        process.start(program, arguments)
        self.status_label.setText("Loading and following camera IOC journal…")

    @QtCore.Slot()
    def _read_camera_journal(self):
        process = self._journal_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode(errors="replace")
        self._append_output(data)
        self.status_label.setText("Following reolinkioc.service journal (read only)")

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _camera_journal_finished(self, exit_code, exit_status):
        self._read_camera_journal()
        if self.isVisible():
            self.status_label.setText(
                f"Camera journal stopped with exit code {int(exit_code)}"
            )

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _camera_journal_error(self, error):
        process = self._journal_process
        detail = process.errorString() if process is not None else str(error)
        self.status_label.setText(f"Unable to read camera journal: {detail}")

    def _send_main_command(self):
        if self.command_edit is None or self._command_process is not None:
            return
        command = self.command_edit.text().strip()
        self._command_queue = list(screen_input_commands(command))
        self._pending_command = command
        self.send_button.setEnabled(False)
        self.command_edit.clear()
        display_command = command or "<Return>"
        self.status_label.setText(f"Sending IOC command: {display_command}")
        self._start_next_main_command_step()

    def _start_next_main_command_step(self):
        if not self._command_queue:
            self._finish_main_command(True)
            return
        program, arguments = self._command_queue.pop(0)
        process = QtCore.QProcess(self)
        self._command_process = process
        process.finished.connect(self._main_command_finished)
        process.errorOccurred.connect(self._main_command_error)
        process.start(program, arguments)

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _main_command_finished(self, exit_code, exit_status):
        process = self._command_process
        if process is None:
            return
        error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        process.deleteLater()
        self._command_process = None
        if exit_code != 0:
            self._command_queue.clear()
            self._finish_main_command(False, error or "Unable to send IOC command")
            return
        self._start_next_main_command_step()

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _main_command_error(self, error):
        process = self._command_process
        detail = process.errorString() if process is not None else str(error)
        if process is not None:
            process.deleteLater()
        self._command_process = None
        self._command_queue.clear()
        self._finish_main_command(False, f"Unable to send IOC command: {detail}")

    def _finish_main_command(self, success, error=""):
        command = self._pending_command
        self._pending_command = ""
        self.send_button.setEnabled(True)
        display_command = command or "<Return>"
        self.status_label.setText(
            f"IOC command sent: {display_command}" if success else error
        )
        self.command_edit.setFocus()

    def _clear_view(self):
        self.output.clear()

    def _copy_all(self):
        QtWidgets.QApplication.clipboard().setText(self.output.toPlainText())
        self.status_label.setText("Console output copied to clipboard")

    def _jump_to_bottom(self):
        scroll_bar = self.output.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def _find_next(self):
        text = self.find_edit.text()
        if not text:
            return
        if self.output.find(text):
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.Start)
        self.output.setTextCursor(cursor)
        if not self.output.find(text):
            self.status_label.setText(f'Not found: "{text}"')

    def closeEvent(self, event):
        if self._poll_timer is not None:
            self._poll_timer.stop()
        for process in (self._journal_process, self._command_process):
            if process is not None and process.state() != QtCore.QProcess.NotRunning:
                process.terminate()
                process.waitForFinished(500)
        super().closeEvent(event)


class _IOCRow(QtWidgets.QFrame):
    def __init__(self, spec: IOCSpec, maintenance: bool, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._maintenance = maintenance
        self._status_process = None
        self._control_process = None
        self._console_window = None
        self._resource_labels = []
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)

        layout = QtWidgets.QGridLayout(self)
        title = QtWidgets.QLabel(spec.title)
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        layout.addWidget(title, 0, 0, 1, 3)

        identity = QtWidgets.QLabel(spec.transport)
        identity.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(identity, 1, 0, 1, 3)

        self.process_label = QtWidgets.QLabel("Checking process…")
        self.epics_label = QtWidgets.QLabel("Checking EPICS heartbeat…")
        layout.addWidget(self.process_label, 2, 0, 1, 3)
        layout.addWidget(self.epics_label, 3, 0, 1, 3)

        layout.addWidget(QtWidgets.QLabel("Uptime:"), 4, 0)
        uptime = PyDMLabel(init_channel=f"ca://{spec.uptime}")
        uptime.setText("—")
        layout.addWidget(uptime, 4, 1, 1, 2)

        resources = QtWidgets.QGroupBox("Resource usage")
        resource_layout = QtWidgets.QGridLayout(resources)
        resource_layout.setHorizontalSpacing(12)
        for index, (title_text, suffix, formatter, tooltip) in enumerate(
            RESOURCE_METRICS
        ):
            metric_row = index // 4
            metric_column = (index % 4) * 2
            metric_title = QtWidgets.QLabel(f"{title_text}:")
            metric_title.setToolTip(tooltip)
            metric_value = _FormattedPVLabel(
                f"{spec.stats_prefix}:{suffix}", formatter, resources
            )
            metric_value.setToolTip(tooltip)
            metric_value.setMinimumWidth(62)
            resource_layout.addWidget(metric_title, metric_row, metric_column)
            resource_layout.addWidget(metric_value, metric_row, metric_column + 1)
            self._resource_labels.append(metric_value)
        layout.addWidget(resources, 5, 0, 1, 3)

        controls = QtWidgets.QHBoxLayout()
        for action, label in (("start", "Start"), ("restart", "Restart"), ("stop", "Stop")):
            button = QtWidgets.QPushButton(label)
            button.setEnabled(maintenance)
            if not maintenance:
                button.setToolTip(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
            button.clicked.connect(lambda checked=False, value=action: self.control(value))
            controls.addWidget(button)
        console_button = QtWidgets.QPushButton("Console")
        console_button.setEnabled(maintenance)
        if not maintenance:
            console_button.setToolTip(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
        elif spec.key == "main":
            console_button.setToolTip(
                "Open a scrollable GUI console with interactive IOC shell input."
            )
        else:
            console_button.setToolTip(
                "Open a scrollable live systemd journal for the camera IOC "
                "(read only)."
            )
        console_button.clicked.connect(self.open_console)
        controls.addWidget(console_button)
        controls.addStretch(1)
        layout.addLayout(controls, 6, 0, 1, 3)

        self._heartbeat = _Heartbeat(spec.heartbeat, self)
        self._heartbeat.changed.connect(self._set_epics_status)

    @staticmethod
    def _status_style(ok: bool) -> str:
        color = "#2e7d32" if ok else "#b3261e"
        return f"font-weight: 600; color: {color};"

    @QtCore.Slot(bool, str)
    def _set_epics_status(self, ok, text):
        self.epics_label.setText(text)
        self.epics_label.setStyleSheet(self._status_style(bool(ok)))

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
        self.process_label.setStyleSheet(self._status_style(ok))
        process.deleteLater()
        self._status_process = None

    def control(self, action: str):
        if not self._maintenance or self._control_process is not None:
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
        self.resize(980, 700)
        layout = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel("EPICS IOC Control and Status")
        heading.setStyleSheet("font-size: 21px; font-weight: 600;")
        layout.addWidget(heading)

        note = QtWidgets.QLabel(
            "The two IOCs run independently on this computer. Process state and "
            "EPICS heartbeat are checked separately."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        maintenance = local_maintenance_allowed()
        if not maintenance:
            warning = QtWidgets.QLabel(LOCAL_MAINTENANCE_DISABLED_MESSAGE)
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #b3261e; font-weight: 600;")
            layout.addWidget(warning)

        self._rows = [_IOCRow(spec, maintenance, self) for spec in IOC_SPECS]
        for row in self._rows:
            layout.addWidget(row)

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
        layout.addWidget(cameras)
        layout.addStretch(1)

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
