"""PyQt front end for the detached Bluesky environment updater."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets


class BlueskyEnvironmentUpdateDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Update Bluesky Python Environment")
        self.setWindowModality(QtCore.Qt.WindowModal)
        self.resize(860, 620)

        self.repo_root = Path(__file__).resolve().parents[1]
        self.helper = Path(__file__).resolve().with_name("bluesky_environment_update.py")
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        self.run_dir = (
            Path.home()
            / ".local"
            / "state"
            / "diffractometer-controls"
            / "updates"
            / timestamp
        )
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.status_path = self.run_dir / "status.json"
        self.log_path = self.run_dir / "update.log"
        self.summary_path = self.run_dir / "plan_summary.txt"
        self._last_log_size = 0
        self._last_status_signature = None
        self._apply_running = False
        self._check_process = None

        layout = QtWidgets.QVBoxLayout(self)
        warning = QtWidgets.QLabel(
            "This procedure updates the bluesky-server environment, backs up and "
            "conditionally migrates Tiled databases, and restarts Bluesky services "
            "and this GUI. No plan may be running."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("font-weight: 600;")
        layout.addWidget(warning)

        self.status_label = QtWidgets.QLabel("Preparing update preview…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        plan_group = QtWidgets.QGroupBox("Proposed changes")
        plan_layout = QtWidgets.QVBoxLayout(plan_group)
        self.plan_text = QtWidgets.QPlainTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setPlaceholderText("Mamba is resolving the environment…")
        plan_layout.addWidget(self.plan_text)
        splitter.addWidget(plan_group)

        log_group = QtWidgets.QGroupBox("Update log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        self.log_text = QtWidgets.QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self.log_text.setFont(fixed_font)
        log_layout.addWidget(self.log_text)
        splitter.addWidget(log_group)
        splitter.setSizes([190, 290])
        layout.addWidget(splitter, 1)

        path_label = QtWidgets.QLabel(f"Log and restore point: {self.run_dir}")
        path_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        path_label.setWordWrap(True)
        layout.addWidget(path_label)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        self.apply_button = QtWidgets.QPushButton("Apply Update")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._confirm_and_apply)
        button_row.addWidget(self.apply_button)
        self.restore_button = QtWidgets.QPushButton("Restore Previous")
        self.restore_button.setVisible(False)
        self.restore_button.clicked.connect(self._confirm_and_restore)
        button_row.addWidget(self.restore_button)
        self.close_button = QtWidgets.QPushButton("Cancel")
        self.close_button.clicked.connect(self.close)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.setInterval(300)
        self.poll_timer.timeout.connect(self._poll_files)
        self.poll_timer.start()
        QtCore.QTimer.singleShot(0, self._start_check)
        QtCore.QTimer.singleShot(0, self._center_on_parent)

    def _center_on_parent(self):
        parent = self.parentWidget()
        if parent is not None:
            center = parent.frameGeometry().center()
        else:
            screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
            if screen is None:
                screen = QtGui.QGuiApplication.primaryScreen()
            center = screen.availableGeometry().center() if screen is not None else QtCore.QPoint()
        frame = self.frameGeometry()
        frame.moveCenter(center)
        self.move(frame.topLeft())

    def _helper_arguments(self, mode):
        return [
            str(self.helper),
            str(mode),
            "--run-dir",
            str(self.run_dir),
            "--repo-root",
            str(self.repo_root),
        ]

    def _start_check(self):
        self.status_label.setText("Checking Queue Server and resolving package updates…")
        process = QtCore.QProcess(self)
        process.setProgram("/usr/bin/python3")
        process.setArguments(self._helper_arguments("check"))
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.finished.connect(self._on_check_finished)
        process.errorOccurred.connect(self._on_check_error)
        self._check_process = process
        process.start()

    def _on_check_finished(self, exit_code, exit_status):
        _ = exit_status
        if int(exit_code) != 0 and not self.status_path.exists():
            self.status_label.setText(f"Update preview failed with exit code {exit_code}.")
            self.close_button.setText("Close")

    def _on_check_error(self, error):
        self.status_label.setText(f"Could not launch update preview: {error}")
        self.close_button.setText("Close")

    def _confirm_and_apply(self):
        summary = self.plan_text.toPlainText().strip()
        preview = summary[:3500]
        if len(summary) > len(preview):
            preview += "\n…"
        response = QtWidgets.QMessageBox.warning(
            self,
            "Confirm Bluesky Environment Update",
            "The Queue Server worker and Bluesky services will be stopped. Verified "
            "database backups and an environment restore point will be created first.\n\n"
            f"{preview}\n\nProceed with the update?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if response != QtWidgets.QMessageBox.Yes:
            return

        self._launch_detached_helper("apply")

    def _confirm_and_restore(self):
        response = QtWidgets.QMessageBox.warning(
            self,
            "Restore Previous Environment",
            "This stops Bluesky services and restores the saved Conda revision, pip-only "
            "packages, and any database schemas touched by the failed update. Continue?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if response != QtWidgets.QMessageBox.Yes:
            return
        self._launch_detached_helper("restore")

    def _launch_detached_helper(self, mode):
        unit = f"bluesky-environment-{mode}-{self.run_dir.name.lower()}"
        command = [
            "systemd-run",
            "--user",
            "--collect",
            f"--unit={unit}",
        ]
        for key in (
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XAUTHORITY",
            "DBUS_SESSION_BUS_ADDRESS",
            "XDG_RUNTIME_DIR",
            "MITR_ALLOW_LOCAL_MAINTENANCE",
        ):
            value = os.environ.get(key)
            if value:
                command.append(f"--setenv={key}={value}")
        command.extend(["/usr/bin/python3", *self._helper_arguments(mode)])
        try:
            result = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Update Error", str(ex))
            return
        if result.returncode != 0:
            QtWidgets.QMessageBox.critical(
                self,
                "Update Error",
                "Could not launch the detached updater.\n\n" + (result.stdout or ""),
            )
            return
        self._apply_running = True
        self.apply_button.setEnabled(False)
        self.restore_button.setVisible(False)
        self.close_button.setText("Hide")
        self.status_label.setText(f"Detached {mode} process started. Waiting for status…")

    def _poll_files(self):
        self._poll_log()
        self._poll_summary()
        self._poll_status()

    def _poll_log(self):
        try:
            size = self.log_path.stat().st_size
        except OSError:
            return
        if size < self._last_log_size:
            self._last_log_size = 0
            self.log_text.clear()
        if size == self._last_log_size:
            return
        with self.log_path.open("r", encoding="utf-8", errors="replace") as stream:
            stream.seek(self._last_log_size)
            chunk = stream.read()
            self._last_log_size = stream.tell()
        if chunk:
            cursor = self.log_text.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            cursor.insertText(chunk)
            self.log_text.setTextCursor(cursor)
            self.log_text.ensureCursorVisible()

    def _poll_summary(self):
        if self.plan_text.toPlainText() or not self.summary_path.exists():
            return
        try:
            self.plan_text.setPlainText(self.summary_path.read_text(encoding="utf-8"))
        except OSError:
            pass

    def _poll_status(self):
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        signature = (payload.get("status"), payload.get("phase"), payload.get("message"))
        if signature == self._last_status_signature:
            return
        self._last_status_signature = signature
        self.status_label.setText(str(payload.get("message", "")))
        self.progress.setValue(int(payload.get("progress", 0) or 0))
        status = str(payload.get("status", ""))
        if status == "ready":
            self.apply_button.setEnabled(True)
            self.close_button.setText("Cancel")
        elif status == "failed":
            self._apply_running = False
            self.apply_button.setEnabled(False)
            self.close_button.setText("Close")
            if payload.get("restore_available"):
                self.restore_button.setVisible(True)
                self.status_label.setText(
                    f"{payload.get('message', 'Update failed')} Restore information was saved; "
                    "services may require recovery before use."
                )
        elif status == "success":
            self._apply_running = False
            self.apply_button.setEnabled(False)
            self.close_button.setText("Close")

    def closeEvent(self, event):
        if self._apply_running:
            response = QtWidgets.QMessageBox.question(
                self,
                "Hide Update Window?",
                "The detached updater will continue and write to its log. Hide this window?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if response != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
        super().closeEvent(event)
