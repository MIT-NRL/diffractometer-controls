"""Local Bluesky service health, resource accounting and independent consoles."""

from __future__ import annotations

import codecs
import ipaddress
import os
import platform
import socket
import threading
import time
from typing import NamedTuple

from bluesky_queueserver_api.zmq import REManagerAPI
from pydm import data_plugins
from qtpy import QtCore, QtGui, QtWidgets
from control_ui.widgets.service_status import ResourceGrid, ServiceCard, StatusBadge, configure_console_font, overview_page, status_legend

from diffractometer_controls.maintenance.local_maintenance import local_maintenance_allowed
from diffractometer_controls.site.mitr import profile
from diffractometer_controls.site.mitr.ioc_control_panel import clean_console_text, format_bytes


class ServiceSpec(NamedTuple):
    title: str
    unit: str
    description: str


SERVICES = (
    ServiceSpec("Queue Server", "queue-server.service", "RunEngine manager and worker processes"),
    ServiceSpec("Document proxy", "bluesky-proxy.service", "Bluesky document stream on ports 5567 / 5568"),
    ServiceSpec("Tiled", "tiled-server.service", "Run catalog and data access"),
)
PROPERTIES = (
    "LoadState", "ActiveState", "SubState", "MainPID", "MemoryCurrent",
    "CPUUsageNSec", "TasksCurrent", "ActiveEnterTimestampMonotonic",
)


def is_local_host(host):
    value = str(host).strip().lower().strip("[]")
    if value in {"localhost", socket.gethostname().lower(), socket.getfqdn().lower()}:
        return True
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    try:
        return any(value == entry[4][0] for entry in socket.getaddrinfo(socket.gethostname(), None))
    except OSError:
        return False


def parse_properties(output):
    return dict(line.split("=", 1) for line in output.splitlines() if "=" in line)


def idle_status(status):
    """A paused/suspended plan and background work also preclude service stops."""
    return bool(status) and (
        status.get("manager_state") == "idle"
        and status.get("re_state") in {None, "idle"}
        and status.get("worker_environment_state") in {"idle", "closed"}
        and not status.get("running_item_uid")
        and not status.get("worker_background_tasks", 0)
    )


class QueueMonitor(QtCore.QThread):
    """Own a separate API/subscriber so the experiment console keeps its messages."""

    status_received = QtCore.Signal(object, str)
    console_received = QtCore.Signal(str)

    def __init__(self, host, parent=None):
        super().__init__(parent)
        self.host = host
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        client = None
        try:
            endpoints = profile.endpoints(self.host)
            client = REManagerAPI(
                zmq_control_addr=endpoints.control, zmq_info_addr=endpoints.info,
                zmq_public_key=os.environ.get("QSERVER_ZMQ_PUBLIC_KEY") or None,
                timeout_recv=0.8, timeout_send=0.3,
                console_monitor_poll_timeout=0.2, console_monitor_max_msgs=2000,
            )
            client.console_monitor.enable()
            next_status = 0.0
            while not self._stop_event.is_set():
                if time.monotonic() >= next_status:
                    try:
                        self.status_received.emit(client.status(reload=True), "")
                    except Exception as exc:
                        self.status_received.emit({}, str(exc))
                    next_status = time.monotonic() + 2.0
                messages = []
                # Bound each batch so sustained output cannot starve status/stop.
                for _ in range(100):
                    try:
                        messages.append(client.console_monitor.next_msg().get("msg", ""))
                    except client.RequestTimeoutError:
                        break
                if messages:
                    self.console_received.emit("".join(messages))
                self._stop_event.wait(0.1)
        except Exception as exc:
            self.status_received.emit({}, str(exc))
        finally:
            if client is not None:
                client.close()


class ConsoleView(QtWidgets.QWidget):
    """Bounded plain text with optional tail-following and no shell input."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.follow = QtWidgets.QCheckBox("Follow output")
        self.follow.setChecked(True)
        toolbar.addWidget(self.follow)
        for title, callback in (("Clear view", self.clear), ("Copy all", self.copy)):
            button = QtWidgets.QPushButton(title)
            button.setAutoDefault(False)
            button.clicked.connect(callback)
            toolbar.addWidget(button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.output = QtWidgets.QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        configure_console_font(self.output)
        self.output.document().setMaximumBlockCount(10000)
        layout.addWidget(self.output)

    @QtCore.Slot(str)
    def append(self, text):
        scrollbar = self.output.verticalScrollBar()
        previous = scrollbar.value()
        cursor = QtGui.QTextCursor(self.output.document())
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(clean_console_text(text))
        scrollbar.setValue(scrollbar.maximum() if self.follow.isChecked() else previous)

    def clear(self):
        self.output.clear()

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.output.toPlainText())


class ServiceRow(ServiceCard):
    def __init__(self, spec, panel):
        super().__init__(spec.title, f"{spec.description} • {spec.unit}", panel)
        self.spec = spec
        self.panel = panel
        self.properties = {}
        self._cpu_sample = None
        self._status_process = None
        self._control_process = None
        self._closed = False
        self.state = self.identity
        self.resources = ResourceGrid(parent=self)
        for key, caption in (("cpu", "CPU"), ("memory", "Memory"), ("tasks", "Tasks"), ("uptime", "Uptime")):
            self.resources.add_metric(key, caption)
        self.resources.setToolTip("Entire service including child workers. CPU: 100% is one fully used core.")
        self.content.addWidget(self.resources)
        self.buttons = {}
        for action in ("start", "restart", "stop"):
            self.buttons[action] = self.add_action(action.title(), lambda checked=False, a=action: self.control(a))
        self.console_button = self.add_action("Console / logs", lambda: panel.show_console(spec.unit))
        self.actions.addStretch()
        self.update_buttons()

    def update_buttons(self):
        for action, button in self.buttons.items():
            button.setEnabled(self.can_control(action))
            button.setToolTip("Local maintenance permission and a writable GUI are required. Stop/restart also require a fresh, idle Queue Server status.")

    def can_control(self, action):
        if (self._closed or action not in self.buttons or self._control_process is not None
                or not self.panel.local or not local_maintenance_allowed()
                or data_plugins.is_read_only() or self.properties.get("LoadState") != "loaded"):
            return False
        active = self.properties.get("ActiveState")
        if action == "start":
            return active in {"inactive", "failed"}
        return (active == "active" and time.monotonic() - self.panel.status_time < 6.0
                and idle_status(self.panel.queue_status))

    def refresh(self):
        if self._closed or not self.panel.local or self._status_process is not None:
            return
        process = QtCore.QProcess(self)
        self._status_process = process
        process.finished.connect(lambda code, state: self._status_finished(process, code))
        process.errorOccurred.connect(lambda error: self._status_error(process))
        timeout = QtCore.QTimer(process)
        timeout.setSingleShot(True)
        timeout.timeout.connect(process.kill)
        timeout.start(5000)
        process.start("systemctl", ["--user", "show", self.spec.unit,
                                    "--property=" + ",".join(PROPERTIES), "--no-pager"])

    def _status_error(self, process):
        if process is self._status_process and process.error() == QtCore.QProcess.FailedToStart:
            self._status_finished(process, -1)

    def _status_finished(self, process, code):
        if process is not self._status_process:
            return
        output = bytes(process.readAllStandardOutput()).decode(errors="replace")
        error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        self._status_process = None
        if code:
            self.properties = {}
            self._cpu_sample = None
            self.state.setText("Status unavailable: " + (error or process.errorString()))
            self.status_badge.set_status("Status unavailable", "error")
            self.resources.clear_values()
        else:
            self.apply_properties(parse_properties(output))
        self.update_buttons()
        process.deleteLater()

    def apply_properties(self, values, now=None):
        self.properties = values
        now = time.monotonic() if now is None else now
        active = values.get("ActiveState", "unknown")
        self.state.setText(f"{self.spec.description} • PID {values.get('MainPID', '—')}")
        self.state.setToolTip(f"{self.spec.unit}\n{values.get('LoadState', 'unknown')} • {active}/{values.get('SubState', 'unknown')}")
        if values.get("LoadState") == "not-found":
            badge, level = "Not installed", "attention"
        elif active == "active" and values.get("SubState") == "running":
            badge, level = "Running", "healthy"
        elif active in {"activating", "deactivating", "reloading"}:
            badge, level = active.title(), "active"
        elif active in {"failed", "inactive"}:
            badge, level = ("Failed" if active == "failed" else "Stopped"), "error"
        elif active == "active":
            badge, level = "Check service", "attention"
        else:
            badge, level = "Unknown", "neutral"
        self.status_badge.set_status(badge, level)
        cpu = "—"
        try:
            counter = int(values["CPUUsageNSec"])
            sample = (now, counter, values.get("MainPID"), values.get("ActiveEnterTimestampMonotonic"))
            previous = self._cpu_sample
            if previous and sample[2:] == previous[2:] and now > previous[0] and counter >= previous[1]:
                cpu = f"{(counter - previous[1]) / (now - previous[0]) / 1e7:.1f}%"
            self._cpu_sample = sample if active == "active" else None
        except (KeyError, ValueError):
            self._cpu_sample = None
        try:
            memory = format_bytes(int(values["MemoryCurrent"]))
        except (KeyError, ValueError):
            memory = "—"
        try:
            started = int(values["ActiveEnterTimestampMonotonic"]) / 1e6
            elapsed = max(0, int(now - started))
            uptime = f"{elapsed // 3600:d}h {elapsed % 3600 // 60:02d}m {elapsed % 60:02d}s" if started and active == "active" else "—"
        except (KeyError, ValueError):
            uptime = "—"
        for key, value in (("cpu", cpu), ("memory", memory), ("tasks", values.get("TasksCurrent", "—")), ("uptime", uptime)):
            self.resources.set_value(key, value)

    def control(self, action):
        if not self.can_control(action):
            return
        if action in {"stop", "restart"}:
            answer = QtWidgets.QMessageBox.question(self, f"{action.title()} {self.spec.title}?",
                f"{action.title()} {self.spec.unit}? This interrupts its clients.\nClose the worker environment before restarting Queue Server.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if answer != QtWidgets.QMessageBox.Yes or not self.can_control(action):
                return
        if self.spec.unit == "queue-server.service" and action != "start" and self.panel.queue_status.get("worker_environment_exists"):
            QtWidgets.QMessageBox.information(self, "Worker environment open", "Close the idle worker environment using the RunEngine controls first.")
            return
        process = QtCore.QProcess(self)
        self._control_process = process
        process.finished.connect(lambda code, state: self._control_finished(process, code))
        process.errorOccurred.connect(lambda error: self._control_failed_to_start(process))
        process.start("systemctl", ["--user", action, self.spec.unit])
        self.update_buttons()

    def _control_failed_to_start(self, process):
        if process.error() == QtCore.QProcess.FailedToStart:
            self._control_finished(process, -1)

    def _control_finished(self, process, code):
        if process is not self._control_process:
            return
        error = bytes(process.readAllStandardError()).decode(errors="replace").strip()
        self._control_process = None
        if code and not self._closed:
            QtWidgets.QMessageBox.warning(self, "Service control failed", error or process.errorString())
        process.deleteLater()
        self.refresh()
        self.update_buttons()

    def shutdown(self):
        self._closed = True
        for process in (self._status_process, self._control_process):
            if process is not None and process.state() != QtCore.QProcess.NotRunning:
                process.kill()
                process.waitForFinished(1000)


class BlueskyServersPanel(QtWidgets.QDialog):
    def __init__(self, host="localhost", parent=None):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Bluesky Servers")
        self.resize(1040, 880)
        self.local = platform.system() == "Linux" and is_local_host(host)
        self.queue_status = {}
        self.status_time = 0.0
        self._closed = False
        self._journals = []
        self._journal_views = {}
        layout = QtWidgets.QVBoxLayout(self)
        heading = QtWidgets.QLabel("Bluesky Servers")
        heading.setStyleSheet("font-size: 21px; font-weight: 600;")
        layout.addWidget(heading)
        self.note = QtWidgets.QLabel(
            f"Queue Server: {profile.endpoints(host).control}. " +
            ("Local service metrics include child workers; refreshed every 2 seconds." if self.local else
             "Local systemd metrics, journals and service controls are unavailable for this connection."))
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addWidget(status_legend(self))
        self.sections = QtWidgets.QTabWidget()
        layout.addWidget(self.sections, 1)
        overview, cards = overview_page(self)
        self.sections.addTab(overview, "Overview")
        health = QtWidgets.QWidget()
        health_layout = QtWidgets.QVBoxLayout(health)
        health_layout.setContentsMargins(0, 0, 0, 0)
        badges = QtWidgets.QHBoxLayout()
        self.health_badges = {}
        for key, title in (("manager_state", "Manager"), ("worker_environment_state", "Worker"), ("re_state", "RunEngine")):
            badge = StatusBadge(f"{title}: checking…", parent=health)
            self.health_badges[key] = badge
            badges.addWidget(badge)
        badges.addStretch()
        health_layout.addLayout(badges)
        self.queue_label = QtWidgets.QLabel("Connecting to Queue Server…")
        self.queue_label.setTextFormat(QtCore.Qt.PlainText)
        self.queue_label.setWordWrap(True)
        health_layout.addWidget(self.queue_label)
        self.rows = [ServiceRow(spec, self) for spec in SERVICES] if self.local else []
        if self.rows:
            self.rows[0].content.insertWidget(0, health)
        else:
            cards.addWidget(health)
        for row in self.rows:
            cards.addWidget(row)
        cards.addStretch()
        self.tabs = QtWidgets.QTabWidget()
        self.console = ConsoleView()
        self.console.append("Live RE manager / worker output (new messages only). Service startup logs are in the Queue Server journal tab.\n")
        self.tabs.addTab(self.console, "RE console")
        self.sections.addTab(self.tabs, "Console / logs")
        if self.local:
            for spec in SERVICES:
                self._start_journal(spec)
        self.monitor = QueueMonitor(host, self)
        self.monitor.status_received.connect(self._queue_status_received)
        self.monitor.console_received.connect(self.console.append)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.refresh()
        self.monitor.start()

    @QtCore.Slot(object, str)
    def _queue_status_received(self, status, error):
        self.queue_status = status
        self.status_time = time.monotonic() if status else 0.0
        if error or not status:
            self.queue_label.setText("Queue Server unreachable: " + (error or "No status received"))
        else:
            self.queue_label.setText(
                f"Queued: {status.get('items_in_queue', '—')}  •  History: {status.get('items_in_history', '—')}  •  "
                f"Background tasks: {status.get('worker_background_tasks', 0)}")
        for key, title in (("manager_state", "Manager"), ("worker_environment_state", "Worker"), ("re_state", "RunEngine")):
            value = status.get(key)
            if not status or error:
                text, level = "unreachable", "error"
            elif value == "idle":
                text, level = "idle", "healthy"
            elif value == "closed" or (key == "re_state" and not status.get("worker_environment_exists") and value is None):
                text, level = ("closed" if key != "re_state" else "not loaded"), "neutral"
            elif value in {"paused", "pausing", "suspended", "suspending"}:
                text, level = value, "attention"
            elif value in {"failed", "error"}:
                text, level = value, "error"
            elif value is None:
                text, level = "unknown", "neutral"
            else:
                text, level = str(value).replace("_", " "), "active"
            self.health_badges[key].set_status(f"{title}: {text}", level)
        for row in self.rows:
            row.update_buttons()

    def _start_journal(self, spec):
        view = ConsoleView()
        self._journal_views[spec.unit] = view
        self.tabs.addTab(view, spec.title + " journal")
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        process.readyReadStandardOutput.connect(lambda: view.append(decoder.decode(bytes(process.readAllStandardOutput()))))
        process.errorOccurred.connect(lambda error: view.append(f"\nJournal unavailable: {process.errorString()}\n") if not self._closed else None)
        process.finished.connect(lambda code, state: view.append(f"\nJournal stopped (exit {code}). Reopen the panel to reconnect.\n") if not self._closed else None)
        process.start("journalctl", ["--user", "--unit=" + spec.unit, "--lines=500", "--follow", "--output=short-iso", "--no-pager", "--no-hostname"])
        self._journals.append(process)

    def show_console(self, unit):
        self.sections.setCurrentWidget(self.tabs)
        view = self.console if unit == "queue-server.service" else self._journal_views.get(unit)
        if view is not None:
            self.tabs.setCurrentWidget(view)

    def refresh(self):
        for row in self.rows:
            row.refresh()
            row.update_buttons()

    @QtCore.Slot()
    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        self.timer.stop()
        self.monitor.stop()
        for row in self.rows:
            row.shutdown()
        for process in self._journals:
            if process.state() != QtCore.QProcess.NotRunning:
                process.kill()
                process.waitForFinished(1000)
        # Network requests are bounded; do not delete a running QThread.
        self.monitor.wait()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
