#!/usr/bin/env python3
"""Online focus viewer that consumes Bluesky documents and reuses FocusOfflineWindow."""

from __future__ import annotations

import argparse
import datetime as _dt
import math
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Keep per-task CPU usage predictable in the online viewer process and any
# process-pool workers (across Linux/macOS/Windows).
for _env_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

from qtpy import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import numpy as np

try:
    from bluesky_queueserver_api import BFunc
    from bluesky_queueserver_api.zmq import REManagerAPI
except Exception:
    BFunc = None
    REManagerAPI = None

try:
    from focus_offline_viewer import FocusOfflineWindow, FrameInfo, _build_focus_program_icon
except Exception:
    from diffractometer_controls.focus_offline_viewer import (
        FocusOfflineWindow,
        FrameInfo,
        _build_focus_program_icon,
    )


def _is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


THEME_MODE_SETTINGS_KEY = "appearance/theme_mode"
SETTINGS_ORGANIZATION = "MITR"
SETTINGS_APPLICATION = "MITR"
DEFAULT_QT_STYLE = "Fusion"


def _desktop_prefers_dark() -> bool:
    override = os.environ.get("MITR_FORCE_DARK_MODE")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on", "dark"}

    gtk_theme = os.environ.get("GTK_THEME", "").strip().lower()
    if gtk_theme and "dark" in gtk_theme:
        return True

    for key in ("color-scheme", "gtk-theme"):
        try:
            proc = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", key],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except Exception:
            continue
        value = proc.stdout.strip().strip("'").lower()
        if key == "color-scheme" and value == "prefer-dark":
            return True
        if "dark" in value:
            return True
    return False


def _build_dark_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 80, 80))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(66, 153, 225))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(66, 153, 225))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(15, 15, 15))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Base, QtGui.QColor(38, 38, 38))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, QtGui.QColor(127, 127, 127))
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, QtGui.QColor(127, 127, 127))
    return palette


def _sync_pyqtgraph_palette(palette: QtGui.QPalette):
    bg = palette.color(QtGui.QPalette.Window)
    fg = palette.color(QtGui.QPalette.WindowText)
    pg.setConfigOption("background", (bg.red(), bg.green(), bg.blue()))
    pg.setConfigOption("foreground", (fg.red(), fg.green(), fg.blue()))


def _apply_saved_theme(app: QtWidgets.QApplication):
    QtCore.QCoreApplication.setOrganizationName(SETTINGS_ORGANIZATION)
    QtCore.QCoreApplication.setApplicationName(SETTINGS_APPLICATION)
    base_style = QtWidgets.QStyleFactory.create(DEFAULT_QT_STYLE)
    if base_style is not None:
        app.setStyle(base_style)
    base_palette = QtGui.QPalette(app.palette())
    settings = QtCore.QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
    mode = str(settings.value(THEME_MODE_SETTINGS_KEY, "system")).strip().lower()
    if mode not in {"system", "light", "dark"}:
        mode = "system"
    dark_requested = _desktop_prefers_dark() if mode == "system" else (mode == "dark")
    palette = _build_dark_palette() if dark_requested else base_palette
    app.setPalette(palette)
    _sync_pyqtgraph_palette(palette)


def _acquire_session_lock(session_id: Optional[str]):
    """Return a held QLockFile, or None if this viewer should not start."""
    sid = str(session_id or "").strip()
    if not sid:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sid).strip("._") or "session"
    lock_dir = Path(os.environ.get("TMPDIR", "/tmp")).expanduser()
    lock_path = lock_dir / f"diffractometer_controls_focus_{safe}.lock"
    lock = QtCore.QLockFile(str(lock_path))
    lock.setStaleLockTime(30000)
    try:
        if lock.tryLock(100):
            return lock
    except Exception:
        return None
    print(f"Focus viewer already running for session {sid}; exiting duplicate process.")
    return None


class QueueServerAdaptiveClient:
    """Submit adaptive focus commands to Queue Server via function_execute."""

    def __init__(
        self,
        *,
        session_id: str,
        zmq_control_addr: str,
        zmq_info_addr: str,
        user: str = "focus_online_viewer",
        user_group: str = "primary",
    ):
        if REManagerAPI is None or BFunc is None:
            raise RuntimeError(
                "bluesky_queueserver_api is required for adaptive command mode."
            )
        self.session_id = str(session_id).strip()
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        self._api = REManagerAPI(
            zmq_control_addr=str(zmq_control_addr),
            zmq_info_addr=str(zmq_info_addr),
        )
        self._user = str(user)
        self._user_group = str(user_group)

    def submit(self, command: str, payload: Optional[Dict] = None) -> Dict:
        item = BFunc(
            "adaptive_focus_submit_command",
            str(self.session_id),
            str(command),
            dict(payload or {}),
        )
        def _exec():
            return self._api.function_execute(
                item,
                run_in_background=True,
                user=self._user,
                user_group=self._user_group,
            )

        try:
            return _exec()
        except Exception as ex:
            msg = str(ex)
            # Queue Server may keep stale permissions in memory until reloaded.
            if ("not allowed" in msg.lower()) or ("permission" in msg.lower()):
                try:
                    self._api.permissions_reload()
                except Exception:
                    pass
                try:
                    return _exec()
                except Exception as ex2:
                    return {
                        "success": False,
                        "ok": False,
                        "error": str(ex2),
                        "command": str(command),
                    }
            return {
                "success": False,
                "ok": False,
                "error": msg,
                "command": str(command),
            }


class FocusOnlineBridge(QtCore.QObject):
    """Translate Bluesky documents into incremental frame updates for FocusOfflineWindow."""

    _frame_received = QtCore.Signal(str, float)
    _log_received = QtCore.Signal(str)
    _run_stopped = QtCore.Signal()
    _go_focus_requested = QtCore.Signal(str)
    _scan_focus_requested = QtCore.Signal(str, float)
    _extend_left_requested = QtCore.Signal()
    _extend_right_requested = QtCore.Signal()
    _mark_complete_requested = QtCore.Signal()
    _mark_aborted_requested = QtCore.Signal()

    def __init__(
        self,
        *,
        image_key: Optional[str] = None,
        motor_key: Optional[str] = None,
        stream_name: str = "primary",
        run_uid: Optional[str] = None,
        follow_latest: bool = True,
        reset_viewer_on_new_run: bool = True,
        on_go_to_focus=None,
        on_scan_around_focus=None,
        on_extend_left=None,
        on_extend_right=None,
        on_mark_complete=None,
        focus_metric_options=("mtf50", "lsf_sigma", "step_sigma"),
        default_focus_metric="mtf50",
        default_scan_step=0.1667,
        interval_ms: int = 200,
        max_workers_total: int = 8,
        bulk_workers: int = 1,
        full_workers: int = 6,
        full_cache_gb: float = 10.0,
        preprocess_mode: str = "gamma",
        preprocess_size: int = 5,
        file_wait_timeout_s: float = 30.0,
        file_wait_interval_ms: int = 250,
        run_file_name: Optional[str] = None,
        run_file_dir: Optional[str] = None,
        run_data_root: str = "/home/mitr_4dh4/Data",
        parent=None,
    ):
        super().__init__(parent=parent)
        self.image_key = image_key
        self.motor_key = motor_key
        self.stream_name = str(stream_name)
        self.run_uid_filter = str(run_uid).strip() if run_uid else None
        self.follow_latest = bool(follow_latest)
        self.reset_viewer_on_new_run = bool(reset_viewer_on_new_run)
        self.on_go_to_focus = on_go_to_focus
        self.on_scan_around_focus = on_scan_around_focus
        self.on_extend_left = on_extend_left
        self.on_extend_right = on_extend_right
        self.on_mark_complete = on_mark_complete
        self.focus_metric_options = tuple(str(v) for v in focus_metric_options)
        self.default_focus_metric = str(default_focus_metric)
        self.default_scan_step = float(max(1e-4, default_scan_step))
        self.interval_ms = int(max(50, interval_ms))
        self.max_workers_total = int(max(3, max_workers_total))
        self.bulk_workers = int(max(1, bulk_workers))
        self.full_workers = int(max(1, full_workers))
        self.full_cache_gb = float(max(0.25, full_cache_gb))
        self.preprocess_mode = str(preprocess_mode or "gamma")
        self.preprocess_size = int(max(1, preprocess_size))
        self.file_wait_timeout_s = float(max(1.0, file_wait_timeout_s))
        self.file_wait_interval_ms = int(max(50, file_wait_interval_ms))
        self.run_file_name = str(run_file_name or "").strip()
        self.run_file_dir = str(run_file_dir or "").strip()
        self.run_data_root = Path(str(run_data_root)).expanduser()

        self.window: Optional[FocusOfflineWindow] = None

        self._descriptor_stream: Dict[str, str] = {}
        self._descriptor_run_start: Dict[str, str] = {}
        self._descriptor_motor_key: Dict[str, str] = {}
        self._resource_docs: Dict[str, Dict] = {}
        self._datum_docs: Dict[str, Dict] = {}
        # The viewer is spawned from the plan's start document callback, so it may
        # miss that start doc and must still accept subsequent descriptors/events.
        self._active_run_uid: Optional[str] = self.run_uid_filter
        self._seen_paths = set()
        self._path_retry_count: Dict[str, int] = {}
        self._path_first_seen_ts: Dict[str, float] = {}
        self._warned_unresolved_datums = set()
        self._warned_no_image_key = False
        self._fallback_position = 0.0
        self._focus_metric_combo: Optional[QtWidgets.QComboBox] = None
        self._scan_step_spin: Optional[QtWidgets.QDoubleSpinBox] = None
        self._go_focus_button: Optional[QtWidgets.QPushButton] = None
        self._scan_focus_button: Optional[QtWidgets.QPushButton] = None
        self._extend_left_button: Optional[QtWidgets.QPushButton] = None
        self._extend_right_button: Optional[QtWidgets.QPushButton] = None
        self._complete_button: Optional[QtWidgets.QPushButton] = None
        self._complete_sent = False
        self._suppress_close_complete = False

        self._frame_received.connect(self._on_frame_received)
        self._log_received.connect(self._on_log_received)
        self._run_stopped.connect(self._on_run_stopped)
        self._go_focus_requested.connect(self._on_go_focus_requested)
        self._scan_focus_requested.connect(self._on_scan_focus_requested)
        self._extend_left_requested.connect(self._on_extend_left_requested)
        self._extend_right_requested.connect(self._on_extend_right_requested)
        self._mark_complete_requested.connect(self._on_mark_complete_requested)
        self._mark_aborted_requested.connect(self._on_mark_aborted_requested)

    def _ensure_window(self):
        if self.window is not None:
            return
        self.window = FocusOfflineWindow(
            frames=[],
            interval_ms=self.interval_ms,
            max_workers_total=self.max_workers_total,
            bulk_workers=self.bulk_workers,
            full_workers=self.full_workers,
            full_cache_gb=self.full_cache_gb,
            preprocess_mode=self.preprocess_mode,
            preprocess_size=self.preprocess_size,
            allow_file_open=False,
        )
        self.window.installEventFilter(self)
        self.window.setWindowTitle("Online Focus Scan Viewer")
        try:
            icon = _build_focus_program_icon()
            if not icon.isNull():
                self.window.setWindowIcon(icon)
        except Exception:
            pass
        self.window.show()
        self._bring_window_to_front(self.window)
        self._install_focus_controls()
        self.window._log("Online stream connected; waiting for frames...")

    @staticmethod
    def _bring_window_to_front(window: QtWidgets.QWidget):
        try:
            state = window.windowState()
            if state & QtCore.Qt.WindowMinimized:
                window.setWindowState(state & ~QtCore.Qt.WindowMinimized)
            window.show()
            window.raise_()
            window.activateWindow()
            # Some window managers apply focus changes asynchronously or ignore
            # a plain raise()/activate() for newly spawned processes. Use a
            # one-shot topmost pulse, then clear it.
            def _pulse_topmost():
                try:
                    window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
                    window.show()
                    window.raise_()
                    window.activateWindow()
                    QtCore.QTimer.singleShot(250, _clear_topmost)
                except Exception:
                    pass

            def _clear_topmost():
                try:
                    window.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, False)
                    window.show()
                    window.raise_()
                    window.activateWindow()
                except Exception:
                    pass

            QtCore.QTimer.singleShot(0, window.raise_)
            QtCore.QTimer.singleShot(0, window.activateWindow)
            QtCore.QTimer.singleShot(30, _pulse_topmost)
        except Exception:
            pass

    def _install_focus_controls(self):
        if self.window is None:
            return
        if self._focus_metric_combo is not None:
            return
        central = self.window.centralWidget()
        if central is None:
            return
        root_layout = central.layout()
        if root_layout is None or root_layout.count() <= 0:
            return
        control_item = root_layout.itemAt(0)
        control_row = control_item.layout() if control_item is not None else None
        if control_row is None:
            return
        metric_label = QtWidgets.QLabel("Focus metric:", self.window)
        combo = QtWidgets.QComboBox(self.window)
        opts = list(dict.fromkeys(self.focus_metric_options)) or ["mtf50"]
        combo.addItems(opts)
        if self.default_focus_metric in opts:
            combo.setCurrentText(self.default_focus_metric)
        elif "mtf50" in opts:
            combo.setCurrentText("mtf50")
        step_label = QtWidgets.QLabel("Step size:", self.window)
        step_spin = QtWidgets.QDoubleSpinBox(self.window)
        step_spin.setDecimals(4)
        step_spin.setRange(0.0001, 1000.0)
        step_spin.setSingleStep(0.01)
        step_spin.setValue(float(self.default_scan_step))
        step_spin.setKeyboardTracking(False)
        step_spin.setToolTip("Step size for Scan Around Focus (scroll mouse wheel to adjust).")
        go_btn = QtWidgets.QPushButton("Go to Focus", self.window)
        scan_btn = QtWidgets.QPushButton("Scan Around Focus", self.window)
        extend_left_btn = QtWidgets.QPushButton("Extend Left +3", self.window)
        extend_right_btn = QtWidgets.QPushButton("Extend Right +3", self.window)
        complete_btn = QtWidgets.QPushButton("Complete", self.window)
        go_btn.clicked.connect(lambda: self._go_focus_requested.emit(str(combo.currentText())))
        scan_btn.clicked.connect(
            lambda: self._scan_focus_requested.emit(
                str(combo.currentText()), float(step_spin.value())
            )
        )
        extend_left_btn.clicked.connect(lambda: self._extend_left_requested.emit())
        extend_right_btn.clicked.connect(lambda: self._extend_right_requested.emit())
        complete_btn.clicked.connect(lambda: self._mark_complete_requested.emit())
        # Append after the existing stretch spacer so this control group is right aligned.
        control_row.addWidget(metric_label)
        control_row.addWidget(combo)
        control_row.addWidget(step_label)
        control_row.addWidget(step_spin)
        control_row.addWidget(go_btn)
        control_row.addWidget(scan_btn)
        control_row.addWidget(extend_left_btn)
        control_row.addWidget(extend_right_btn)
        control_row.addWidget(complete_btn)
        self._focus_metric_combo = combo
        self._scan_step_spin = step_spin
        self._go_focus_button = go_btn
        self._scan_focus_button = scan_btn
        self._extend_left_button = extend_left_btn
        self._extend_right_button = extend_right_btn
        self._complete_button = complete_btn

    @QtCore.Slot(str)
    def _on_go_focus_requested(self, metric: str):
        if self.on_go_to_focus is None:
            self._log_received.emit("Go to Focus clicked, but no handler is attached.")
            return
        try:
            self.on_go_to_focus(str(metric))
        except Exception as ex:
            self._log_received.emit(f"Go to Focus handler failed: {ex}")

    @QtCore.Slot(str, float)
    def _on_scan_focus_requested(self, metric: str, step_size: float):
        if self.on_scan_around_focus is None:
            self._log_received.emit("Scan Around Focus clicked, but no handler is attached.")
            return
        try:
            try:
                self.on_scan_around_focus(str(metric), float(step_size))
            except TypeError:
                # Backward compatibility for older callbacks that only accept metric.
                self.on_scan_around_focus(str(metric))
        except Exception as ex:
            self._log_received.emit(f"Scan Around Focus handler failed: {ex}")

    @QtCore.Slot()
    def _on_extend_left_requested(self):
        if self.on_extend_left is None:
            self._log_received.emit("Extend Left +3 clicked, but no handler is attached.")
            return
        try:
            self.on_extend_left()
        except Exception as ex:
            self._log_received.emit(f"Extend Left +3 handler failed: {ex}")

    @QtCore.Slot()
    def _on_extend_right_requested(self):
        if self.on_extend_right is None:
            self._log_received.emit("Extend Right +3 clicked, but no handler is attached.")
            return
        try:
            self.on_extend_right()
        except Exception as ex:
            self._log_received.emit(f"Extend Right +3 handler failed: {ex}")

    @QtCore.Slot()
    def _on_mark_complete_requested(self):
        if self._complete_sent:
            return
        self._complete_sent = True
        if self._go_focus_button is not None:
            self._go_focus_button.setEnabled(False)
        if self._scan_focus_button is not None:
            self._scan_focus_button.setEnabled(False)
        if self._scan_step_spin is not None:
            self._scan_step_spin.setEnabled(False)
        if self._extend_left_button is not None:
            self._extend_left_button.setEnabled(False)
        if self._extend_right_button is not None:
            self._extend_right_button.setEnabled(False)
        if self._complete_button is not None:
            self._complete_button.setEnabled(False)
        self._log_received.emit("Focus workflow marked complete.")
        if self.on_mark_complete is None:
            return
        try:
            self.on_mark_complete()
        except Exception as ex:
            self._log_received.emit(f"Complete handler failed: {ex}")

    @QtCore.Slot()
    def _on_mark_aborted_requested(self):
        if self._complete_sent:
            return
        self._complete_sent = True
        self._log_received.emit("Focus workflow viewer closed; aborting adaptive session.")
        if self.on_mark_complete is None:
            return
        try:
            self.on_mark_complete("abort")
        except TypeError:
            self.on_mark_complete()
        except Exception as ex:
            self._log_received.emit(f"Abort handler failed: {ex}")

    def get_focus_target(self, metric: str = "mtf50") -> Optional[float]:
        if self.window is None:
            return None
        m = str(metric or "mtf50").strip().lower()
        if m == "mtf50":
            target = getattr(self.window, "_optimal_mtf50_position", np.nan)
        elif m == "lsf_sigma":
            target = getattr(self.window, "_optimal_psf_position", np.nan)
        else:
            target = getattr(self.window, "_optimal_focus_position", np.nan)
        try:
            target = float(target)
        except Exception:
            target = np.nan
        if np.isfinite(target):
            return target
        # Fallbacks if selected metric is unavailable.
        for attr in (
            "_optimal_mtf50_position",
            "_optimal_psf_position",
            "_optimal_focus_position",
        ):
            val = getattr(self.window, attr, np.nan)
            try:
                val = float(val)
            except Exception:
                val = np.nan
            if np.isfinite(val):
                return val
        return None

    def eventFilter(self, watched, event):
        if (
            self.window is not None
            and watched is self.window
            and event is not None
            and event.type() == QtCore.QEvent.Close
        ):
            if (not self._suppress_close_complete) and (not self._complete_sent):
                self._mark_aborted_requested.emit()
        return super().eventFilter(watched, event)

    def _detect_image_key(self, data: Dict) -> Optional[str]:
        if self.image_key and self.image_key in data:
            return self.image_key

        # Prefer keys that look like image/file metadata and resolve to path or datum.
        for k, v in data.items():
            lk = str(k).lower()
            if not any(t in lk for t in ("image", "file", "path", "tiff", "hdf")):
                continue
            if self._resolve_image_path(v) is not None:
                return str(k)

        # Fallback: any resolvable string-like payload.
        for k, v in data.items():
            if self._resolve_image_path(v) is not None:
                return str(k)
        return None

    @staticmethod
    def _coerce_text(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8", errors="ignore").strip()
            except Exception:
                return ""
        return str(value).strip()

    @staticmethod
    def _looks_like_path(text: str) -> bool:
        t = str(text or "").strip()
        if not t:
            return False
        if t.startswith(("file://", "/", "./", "../", "~")):
            return True
        if "\\" in t or "/" in t:
            # Avoid treating datum-like ids (e.g. "<uuid>/0") as file paths.
            if Path(t).suffix:
                return True
            if t.startswith((".", "~")):
                return True
            return False
        if len(t) >= 3 and t[1:3] in (":\\", ":/"):
            return True
        return Path(t).suffix.lower() in {".tif", ".tiff", ".h5", ".hdf5", ".png", ".jpg", ".jpeg"}

    @staticmethod
    def _looks_like_datum_id(text: str) -> bool:
        t = str(text or "").strip()
        if not t:
            return False
        if t in {"0", "1"}:
            return False
        if "/" in t and (Path(t).suffix == ""):
            return True
        if re.fullmatch(r"[0-9a-fA-F-]{8,}", t):
            return True
        return False

    @staticmethod
    def _trim_map_size(mapping: Dict, max_items: int = 20000):
        while len(mapping) > max_items:
            try:
                mapping.pop(next(iter(mapping)))
            except Exception:
                break

    def _resolve_path_from_datum_id(self, datum_id: str) -> Optional[str]:
        rec = self._datum_docs.get(str(datum_id), None)
        if rec is None:
            return None
        resource_uid = str(rec.get("resource", "")).strip()
        resource_doc = self._resource_docs.get(resource_uid, None)
        if resource_doc is None:
            return None

        resource_path = self._coerce_text(resource_doc.get("resource_path", ""))
        if not resource_path:
            return None
        root = self._coerce_text(resource_doc.get("root", ""))
        resource_kwargs = dict(resource_doc.get("resource_kwargs", {}) or {})
        datum_kwargs = dict(rec.get("datum_kwargs", {}) or {})
        template = self._coerce_text(resource_kwargs.get("template", ""))
        filename = self._coerce_text(resource_kwargs.get("filename", ""))

        def _join_root(path_text: str) -> Path:
            p = Path(path_text).expanduser()
            if p.is_absolute():
                return p
            if root:
                return (Path(root).expanduser() / p)
            return (Path.cwd() / p)

        point_number = datum_kwargs.get("point_number", datum_kwargs.get("index", None))
        try:
            pn = int(point_number)
        except Exception:
            pn = None
        point_indices = []
        if pn is not None:
            for n in (pn, pn + 1, pn - 1):
                if int(n) >= 0 and int(n) not in point_indices:
                    point_indices.append(int(n))

        base = _join_root(resource_path)
        dir_candidates = []
        if base.suffix:
            dir_candidates.append(base.parent)
        else:
            dir_candidates.append(base)
        if base.parent not in dir_candidates:
            dir_candidates.append(base.parent)

        candidates = []

        # Prefer resource template+filename (AD_TIFF style) when available.
        if template and filename:
            for d in dir_candidates:
                dir_text = str(d)
                if dir_text and (not dir_text.endswith(("/", "\\"))):
                    dir_text = dir_text + "/"
                for idx in (point_indices or [0]):
                    for fmt in (
                        (dir_text, filename, int(idx)),
                        {
                            "path": dir_text,
                            "directory": dir_text,
                            "filename": filename,
                            "point_number": int(idx),
                            "index": int(idx),
                        },
                    ):
                        try:
                            out = template % fmt
                        except Exception:
                            continue
                        if isinstance(out, str) and out.strip():
                            candidates.append(Path(out).expanduser())
                            break

        # Fallback: resource_path itself may be a template containing index tokens.
        if (not candidates) and ("%" in resource_path):
            for idx in (point_indices or [0]):
                for fmt in (
                    int(idx),
                    (int(idx),),
                    {"point_number": int(idx), "index": int(idx)},
                ):
                    try:
                        out = resource_path % fmt
                    except Exception:
                        continue
                    if isinstance(out, str) and out.strip():
                        candidates.append(_join_root(out))
                        break

        # Last fallback: synthesize common TIFF/HDF names from directory + filename.
        if (not candidates) and filename:
            for d in dir_candidates:
                for idx in (point_indices or [0]):
                    candidates.append(d / f"{filename}_{int(idx):04d}.tif")
                    candidates.append(d / f"{filename}_{int(idx):04d}.tiff")
                    candidates.append(d / f"{filename}_{int(idx):06d}.tif")
                    candidates.append(d / f"{filename}_{int(idx):06d}.tiff")
                    candidates.append(d / f"{filename}_{int(idx):06d}.h5")
                    candidates.append(d / f"{filename}_{int(idx):06d}.hdf5")

        if not candidates:
            # Do not return directory-only paths; caller expects image file path.
            if base.suffix:
                candidates.append(base)
            else:
                return None

        first_missing_file = None
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    return str(p.resolve())
            except Exception:
                pass
            if first_missing_file is None and p.suffix:
                try:
                    first_missing_file = str(p.resolve())
                except Exception:
                    first_missing_file = str(p)

        return first_missing_file

    def _resolve_image_path(self, value) -> Optional[str]:
        text = self._coerce_text(value)
        if not text:
            return None
        # Prefer datum resolution when value looks like a datum id token.
        if self._looks_like_datum_id(text) or (text in self._datum_docs):
            out = self._resolve_path_from_datum_id(text)
            if out:
                return out
            out = self._resolve_from_run_metadata(text)
            if out:
                return out
        if self._looks_like_path(text):
            return text
        out = self._resolve_path_from_datum_id(text)
        if out:
            return out
        out = self._resolve_from_run_metadata(text)
        if out:
            return out
        return self._resolve_path_from_datum_id(text)

    def _resolve_from_run_metadata(self, token: str) -> Optional[str]:
        if (not self.run_file_name) or (not self.run_file_dir):
            return None
        text = self._coerce_text(token)
        if not text:
            return None
        m = re.search(r"/(\d+)$", text)
        idx = int(m.group(1)) if m else None
        year_now = int(_dt.datetime.now().year)
        candidate_dirs = []
        for y in (year_now - 1, year_now, year_now + 1):
            p = self.run_data_root / f"{y}" / self.run_file_dir
            if p not in candidate_dirs:
                candidate_dirs.append(p)
        try:
            for p in sorted(self.run_data_root.glob(f"*/{self.run_file_dir}")):
                if p not in candidate_dirs:
                    candidate_dirs.append(p)
        except Exception:
            pass

        patterns = []
        if idx is not None:
            for n in (idx + 1, idx, idx + 2):
                if n < 0:
                    continue
                patterns.extend(
                    [
                        f"{self.run_file_name}_*_{int(n):04d}.tif",
                        f"{self.run_file_name}_*_{int(n):04d}.tiff",
                        f"{self.run_file_name}_*_{int(n):06d}.tif",
                        f"{self.run_file_name}_*_{int(n):06d}.tiff",
                    ]
                )
        else:
            patterns.extend(
                [
                    f"{self.run_file_name}_*.tif",
                    f"{self.run_file_name}_*.tiff",
                ]
            )

        for d in candidate_dirs:
            if not d.exists():
                continue
            for pat in patterns:
                try:
                    matches = sorted(
                        [p for p in d.glob(pat) if p.is_file()],
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                except Exception:
                    matches = []
                if matches:
                    try:
                        return str(matches[0].resolve())
                    except Exception:
                        return str(matches[0])
        return None

    def _descriptor_motor_key_from_doc(self, doc: Dict) -> Optional[str]:
        data_keys = dict((doc or {}).get("data_keys", {}) or {})
        object_keys = dict((doc or {}).get("object_keys", {}) or {})
        requested = str(self.motor_key or "").strip()

        candidates = []
        if requested:
            for key, entry in data_keys.items():
                object_name = str((entry or {}).get("object_name", "")).strip()
                key_text = str(key).strip()
                if "setpoint" in key_text.lower():
                    continue
                if key_text == requested or object_name == requested:
                    candidates.append(key_text)
            for object_name, keys in object_keys.items():
                if str(object_name).strip() != requested:
                    continue
                for key in list(keys or []):
                    key_text = str(key).strip()
                    if "setpoint" in key_text.lower():
                        continue
                    if key_text in data_keys:
                        candidates.append(key_text)

        if not candidates:
            for key, entry in data_keys.items():
                key_text = str(key).strip()
                if "setpoint" in key_text.lower():
                    continue
                object_name = str((entry or {}).get("object_name", "")).strip()
                text = f"{key_text} {object_name}".lower()
                if "focus" in text or "motor" in text or "position" in text:
                    candidates.append(key_text)

        for key in candidates:
            entry = data_keys.get(key, {}) or {}
            dtype = str(entry.get("dtype", "")).lower()
            shape = entry.get("shape", [])
            if dtype in {"number", "integer", "float"} or shape in ([], (), None):
                return str(key)
        return str(candidates[0]) if candidates else None

    def _detect_motor_key(self, data: Dict, *, descriptor_uid: str = "") -> Optional[str]:
        descriptor_key = self._descriptor_motor_key.get(str(descriptor_uid), "")
        if descriptor_key and descriptor_key in data:
            return descriptor_key
        if self.motor_key and self.motor_key in data:
            return self.motor_key
        candidates = []
        fallback_numeric = []
        for k, v in data.items():
            lk = str(k).lower()
            if "setpoint" in lk:
                continue
            if ("motor" in lk or "position" in lk) and _is_number(v):
                candidates.append(str(k))
            elif _is_number(v):
                if any(t in lk for t in ("count", "sum", "total", "mean", "sigma", "mtf", "psf", "lsf")):
                    continue
                fallback_numeric.append(str(k))
        if not candidates:
            # Common focus motor keys are often plain names like "cam_focus".
            for k in fallback_numeric:
                if "focus" in str(k).lower():
                    return str(k)
            return fallback_numeric[0] if fallback_numeric else None
        if "focus_sim_motor" in candidates:
            return "focus_sim_motor"
        for k in candidates:
            if "focus" in str(k).lower():
                return str(k)
        return candidates[0]

    def _accept_descriptor_for_run_stream(self, descriptor_uid: str) -> bool:
        stream = self._descriptor_stream.get(str(descriptor_uid), "")
        run_start_uid = self._descriptor_run_start.get(str(descriptor_uid), "")
        if self.run_uid_filter:
            if run_start_uid:
                if run_start_uid != self.run_uid_filter:
                    return False
            elif self._active_run_uid != self.run_uid_filter:
                return False
        if self.stream_name and stream and stream != self.stream_name:
            return False
        return True

    def _process_event_data(self, *, descriptor_uid: str, data: Dict):
        if not self._accept_descriptor_for_run_stream(descriptor_uid):
            return
        data = dict(data or {})
        image_key = self._detect_image_key(data)
        if image_key is None:
            if not self._warned_no_image_key:
                self._warned_no_image_key = True
                try:
                    keys = ", ".join(sorted(str(k) for k in data.keys()))
                except Exception:
                    keys = str(list(data.keys()))
                self._log_received.emit(
                    f"No image file key found in event data. keys=[{keys}]"
                )
            return

        motor_key = self._detect_motor_key(data, descriptor_uid=descriptor_uid)
        raw_image_value = data.get(image_key, "")
        image_path = self._resolve_image_path(raw_image_value)
        if not image_path:
            datum_id = self._coerce_text(raw_image_value)
            if datum_id and datum_id not in self._warned_unresolved_datums:
                self._warned_unresolved_datums.add(datum_id)
                self._log_received.emit(
                    "Could not resolve image datum/path "
                    f"(key='{image_key}', token='{datum_id}', "
                    f"file_name='{self.run_file_name}', file_dir='{self.run_file_dir}')"
                )
            return
        if motor_key is not None and _is_number(data.get(motor_key)):
            position = float(data.get(motor_key))
            self._fallback_position = position
        else:
            position = float(self._fallback_position)
            if motor_key is None:
                self._log_received.emit(
                    "No motor key found in Bluesky event data; using last known motor position."
                )
            elif not _is_number(data.get(motor_key)):
                self._log_received.emit(
                    f"Motor key '{motor_key}' was not numeric; using last known motor position."
                )

        self._frame_received.emit(image_path, float(position))

    def on_document(self, name: str, doc: Dict):
        """Bluesky callback entry point: subscribe this to RE/dispatcher."""
        if name == "start":
            uid = str(doc.get("uid", ""))
            self._active_run_uid = uid
            self._descriptor_stream.clear()
            self._descriptor_run_start.clear()
            self._descriptor_motor_key.clear()
            self._resource_docs.clear()
            self._datum_docs.clear()
            self._path_retry_count.clear()
            self._path_first_seen_ts.clear()
            self._warned_unresolved_datums.clear()
            self._warned_no_image_key = False
            if self.run_uid_filter and uid != self.run_uid_filter:
                return
            if self.reset_viewer_on_new_run:
                # Keep the pre-created placeholder window for the same run.
                # This avoids a close/reopen flicker when start arrives after launch.
                keep_placeholder = bool(
                    (self.window is not None)
                    and (len(getattr(self.window, "frames", [])) == 0)
                    and (not self._seen_paths)
                    and (self.run_uid_filter is not None)
                    and (uid == self.run_uid_filter)
                )
                if not keep_placeholder:
                    if self.window is not None:
                        self._suppress_close_complete = True
                        try:
                            self.window.close()
                        except Exception:
                            pass
                        finally:
                            self._suppress_close_complete = False
                    self.window = None
                    self._seen_paths.clear()
                    self._fallback_position = 0.0
            self._complete_sent = False
            self._log_received.emit(f"Run started: {uid}")
            return

        if name == "resource":
            uid = str(doc.get("uid", "")).strip()
            if uid:
                self._resource_docs[uid] = dict(doc or {})
                self._trim_map_size(self._resource_docs)
            return

        if name == "datum":
            datum_id = str(doc.get("datum_id", "")).strip()
            if datum_id:
                self._datum_docs[datum_id] = {
                    "resource": str(doc.get("resource", "")).strip(),
                    "datum_kwargs": dict(doc.get("datum_kwargs", {}) or {}),
                }
                self._trim_map_size(self._datum_docs)
            return

        if name == "datum_page":
            resource_uid = str(doc.get("resource", "")).strip()
            datum_ids = list(doc.get("datum_id", []) or [])
            datum_kwargs = doc.get("datum_kwargs", {}) or {}
            for i, datum_id in enumerate(datum_ids):
                did = str(datum_id).strip()
                if not did:
                    continue
                if isinstance(datum_kwargs, dict):
                    kw = {}
                    for k, vals in datum_kwargs.items():
                        try:
                            kw[str(k)] = vals[i]
                        except Exception:
                            pass
                else:
                    try:
                        kw = dict(datum_kwargs[i] or {})
                    except Exception:
                        kw = {}
                self._datum_docs[did] = {
                    "resource": resource_uid,
                    "datum_kwargs": kw,
                }
            self._trim_map_size(self._datum_docs)
            return

        if name == "descriptor":
            descriptor_uid = str(doc.get("uid", ""))
            stream_name = str(doc.get("name", ""))
            run_start_uid = str(doc.get("run_start", "")).strip()
            self._descriptor_stream[descriptor_uid] = stream_name
            self._descriptor_run_start[descriptor_uid] = run_start_uid
            motor_key = self._descriptor_motor_key_from_doc(doc)
            if motor_key:
                self._descriptor_motor_key[descriptor_uid] = str(motor_key)
            if (
                self.run_uid_filter
                and run_start_uid == self.run_uid_filter
                and self._active_run_uid != self.run_uid_filter
            ):
                self._active_run_uid = self.run_uid_filter
            return

        if name == "event":
            descriptor_uid = str(doc.get("descriptor", ""))
            data = dict(doc.get("data", {}) or {})
            self._process_event_data(descriptor_uid=descriptor_uid, data=data)
            return

        if name == "event_page":
            descriptor_uid = str(doc.get("descriptor", ""))
            page_data = dict(doc.get("data", {}) or {})
            keys = list(page_data.keys())
            n_items = 0
            for k in keys:
                v = page_data.get(k, [])
                try:
                    n_items = max(n_items, len(v))
                except Exception:
                    n_items = max(n_items, 1)
            for i in range(int(max(0, n_items))):
                row = {}
                for k in keys:
                    vals = page_data.get(k, [])
                    try:
                        row[k] = vals[i]
                    except Exception:
                        pass
                self._process_event_data(descriptor_uid=descriptor_uid, data=row)
            return

        if name == "stop":
            if self.run_uid_filter:
                run_start_uid = str(doc.get("run_start", "")).strip()
                if run_start_uid:
                    if run_start_uid != self.run_uid_filter:
                        return
                elif self._active_run_uid != self.run_uid_filter:
                    return
            exit_status = str(doc.get("exit_status", ""))
            self._log_received.emit(f"Run stopped: {exit_status or 'unknown'}")
            self._run_stopped.emit()

    @QtCore.Slot()
    def _on_run_stopped(self):
        """Ensure all streamed frames are queued for full filtering after run end."""
        if self.window is None:
            return
        total = int(len(self.window.frames))
        if total <= 0:
            return
        # Mark all streamed frames as seen so the window can treat the run as complete.
        try:
            self.window._seen_frame_indices.update(range(total))
        except Exception:
            pass

        # Queue in batches to keep the UI responsive on large runs.
        batch_size = 200

        def _queue_batch(i0: int):
            if self.window is None:
                return
            try:
                i1 = int(min(total, i0 + batch_size))
                for idx in range(i0, i1):
                    self.window._enqueue_full_prepare(idx)
                self.window._update_filter_queue_indicator()
                if i1 < total:
                    QtCore.QTimer.singleShot(0, lambda next_i=i1: _queue_batch(next_i))
                else:
                    self.window._log(
                        f"Run stop sync: queued full filtering for all streamed frames ({total})."
                    )
            except Exception as ex:
                self._log_received.emit(f"Run stop sync failed: {ex}")

        QtCore.QTimer.singleShot(0, lambda: _queue_batch(0))

    @QtCore.Slot(str)
    def _on_log_received(self, message: str):
        if self.window is not None:
            self.window._log(message)
        else:
            ts = QtCore.QDateTime.currentDateTime().toString("HH:mm:ss")
            print(f"[{ts}] {message}")

    @QtCore.Slot(str, float)
    def _on_frame_received(self, path_text: str, position: float):
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        norm = str(path.resolve())
        if norm in self._seen_paths:
            return
        self._seen_paths.add(norm)

        if not path.exists():
            now = float(time.monotonic())
            first_seen = float(self._path_first_seen_ts.get(norm, now))
            if norm not in self._path_first_seen_ts:
                self._path_first_seen_ts[norm] = now
                first_seen = now
            elapsed = max(0.0, now - first_seen)
            retries = int(self._path_retry_count.get(norm, 0)) + 1
            self._path_retry_count[norm] = retries
            if elapsed < self.file_wait_timeout_s:
                if retries in {1, 10, 25, 50, 100}:
                    self._log_received.emit(
                        f"Waiting for file to appear ({elapsed:.1f}s): {path.name}"
                    )
                self._seen_paths.discard(norm)
                QtCore.QTimer.singleShot(
                    int(self.file_wait_interval_ms),
                    lambda p=path_text, pos=float(position): self._frame_received.emit(p, pos),
                )
                return
            self._log_received.emit(
                f"Image file did not appear within {self.file_wait_timeout_s:.1f}s: {path}"
            )
            self._path_first_seen_ts.pop(norm, None)
        self._path_retry_count.pop(norm, None)
        self._path_first_seen_ts.pop(norm, None)

        self._ensure_window()
        if self.window is None:
            return

        idx = int(len(self.window.frames))
        self.window.frames.append(FrameInfo(index=idx, path=path, position=float(position)))
        if idx == 0:
            self.window._log(
                f"First streamed frame: {path.name} @ motor={float(position):.5f}"
            )
        self.window._update_filter_queue_indicator()
        self.window._log(
            f"Streamed frame {idx + 1}/{len(self.window.frames)}: {path.name} @ motor={float(position):.5f}"
        )
        if self.follow_latest:
            self.window._load_frame(idx)


def attach_to_run_engine(
    re,
    *,
    image_key: Optional[str] = None,
    motor_key: Optional[str] = None,
    stream_name: str = "primary",
    run_uid: Optional[str] = None,
    follow_latest: bool = True,
    reset_viewer_on_new_run: bool = True,
    on_go_to_focus=None,
    on_scan_around_focus=None,
    on_extend_left=None,
    on_extend_right=None,
    on_mark_complete=None,
    focus_metric_options=("mtf50", "lsf_sigma", "step_sigma"),
    default_focus_metric="mtf50",
    default_scan_step=0.1667,
    interval_ms: int = 200,
    max_workers_total: int = 8,
    bulk_workers: int = 1,
    full_workers: int = 6,
    full_cache_gb: float = 10.0,
    preprocess_mode: str = "gamma",
    preprocess_size: int = 5,
    file_wait_timeout_s: float = 30.0,
    file_wait_interval_ms: int = 250,
    run_file_name: Optional[str] = None,
    run_file_dir: Optional[str] = None,
    run_data_root: str = "/home/mitr_4dh4/Data",
) -> Tuple[FocusOnlineBridge, int]:
    """Attach online viewer to a local RunEngine stream."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    _ = app
    pg.setConfigOption("imageAxisOrder", "row-major")
    bridge = FocusOnlineBridge(
        image_key=image_key,
        motor_key=motor_key,
        stream_name=stream_name,
        run_uid=run_uid,
        follow_latest=follow_latest,
        reset_viewer_on_new_run=reset_viewer_on_new_run,
        on_go_to_focus=on_go_to_focus,
        on_scan_around_focus=on_scan_around_focus,
        on_extend_left=on_extend_left,
        on_extend_right=on_extend_right,
        on_mark_complete=on_mark_complete,
        focus_metric_options=focus_metric_options,
        default_focus_metric=default_focus_metric,
        default_scan_step=default_scan_step,
        interval_ms=interval_ms,
        max_workers_total=max_workers_total,
        bulk_workers=bulk_workers,
        full_workers=full_workers,
        full_cache_gb=full_cache_gb,
        preprocess_mode=preprocess_mode,
        preprocess_size=preprocess_size,
        file_wait_timeout_s=file_wait_timeout_s,
        file_wait_interval_ms=file_wait_interval_ms,
        run_file_name=run_file_name,
        run_file_dir=run_file_dir,
        run_data_root=run_data_root,
    )
    token = int(re.subscribe(bridge.on_document))
    return bridge, token


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Online focus viewer (ZMQ stream -> focus analysis UI)")
    p.add_argument("--zmq-address", type=str, default="localhost:5567", help="ZMQ publisher address as 'host:port'")
    p.add_argument("--image-key", type=str, default=None, help="Event data key containing image file path")
    p.add_argument("--motor-key", type=str, default=None, help="Event data key containing motor position")
    p.add_argument("--stream-name", type=str, default="primary", help="Bluesky stream name to consume")
    p.add_argument("--run-uid", type=str, default=None, help="Optional UID filter; ignore other runs")
    p.add_argument("--no-follow-latest", action="store_true", help="Do not auto-jump to newest frame")
    p.add_argument(
        "--keep-runs-combined",
        action="store_true",
        help="Append all runs into one viewer session instead of resetting on each new run.",
    )
    p.add_argument("--interval-ms", type=int, default=200, help="Playback interval for the base viewer")
    p.add_argument("--max-workers-total", type=int, default=8, help="Total worker cap")
    p.add_argument("--bulk-workers", type=int, default=1, help="Bulk/ROI workers")
    p.add_argument("--full-workers", type=int, default=6, help="Full filter process workers")
    p.add_argument("--full-cache-gb", type=float, default=10.0, help="Full filtered cache budget (GB)")
    p.add_argument(
        "--preprocess-mode",
        type=str,
        choices=["gamma", "tomopy_outlier", "median"],
        default="gamma",
        help="Prefilter mode for image processing.",
    )
    p.add_argument(
        "--preprocess-size",
        type=int,
        default=5,
        help="Kernel size for selected prefilter mode.",
    )
    p.add_argument(
        "--file-wait-timeout-s",
        type=float,
        default=30.0,
        help="Max time to wait for a just-triggered file to appear on disk.",
    )
    p.add_argument(
        "--file-wait-interval-ms",
        type=int,
        default=250,
        help="Retry interval while waiting for delayed file writes.",
    )
    p.add_argument(
        "--run-file-name",
        type=str,
        default="",
        help="Run metadata file_name (used as fallback for datum-id to file mapping).",
    )
    p.add_argument(
        "--run-file-dir",
        type=str,
        default="",
        help="Run metadata file_dir (used as fallback for datum-id to file mapping).",
    )
    p.add_argument(
        "--run-data-root",
        type=str,
        default="/home/mitr_4dh4/Data",
        help="Root directory for run file fallback mapping.",
    )
    p.add_argument("--session-id", type=str, default=None, help="Adaptive focus session id from plan metadata")
    p.add_argument(
        "--qserver-control-addr",
        type=str,
        default="tcp://localhost:60615",
        help="Queue Server control address (used when --session-id is set)",
    )
    p.add_argument(
        "--qserver-info-addr",
        type=str,
        default="tcp://localhost:60625",
        help="Queue Server info address (used when --session-id is set)",
    )
    p.add_argument("--qserver-user", type=str, default="focus_online_viewer", help="Queue Server API user name")
    p.add_argument("--qserver-user-group", type=str, default="primary", help="Queue Server API user group")
    p.add_argument(
        "--parent-pid",
        type=int,
        default=0,
        help="Optional launcher PID; viewer exits if this process is gone.",
    )
    p.add_argument(
        "--startup-timeout-s",
        type=float,
        default=90.0,
        help="Exit if no viewer window is created within this timeout.",
    )
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    session_lock = _acquire_session_lock(args.session_id) if args.session_id else None
    if args.session_id and session_lock is None:
        return 0
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    _apply_saved_theme(app)
    try:
        app_icon = _build_focus_program_icon()
        if not app_icon.isNull():
            app.setWindowIcon(app_icon)
    except Exception:
        pass
    pg.setConfigOption("imageAxisOrder", "row-major")

    bridge = FocusOnlineBridge(
        image_key=args.image_key,
        motor_key=args.motor_key,
        stream_name=args.stream_name,
        run_uid=args.run_uid,
        follow_latest=not bool(args.no_follow_latest),
        reset_viewer_on_new_run=not bool(args.keep_runs_combined),
        interval_ms=args.interval_ms,
        max_workers_total=args.max_workers_total,
        bulk_workers=args.bulk_workers,
        full_workers=args.full_workers,
        full_cache_gb=args.full_cache_gb,
        preprocess_mode=args.preprocess_mode,
        preprocess_size=args.preprocess_size,
        file_wait_timeout_s=args.file_wait_timeout_s,
        file_wait_interval_ms=args.file_wait_interval_ms,
        run_file_name=args.run_file_name,
        run_file_dir=args.run_file_dir,
        run_data_root=args.run_data_root,
    )
    bridge._ensure_window()

    if args.session_id:
        try:
            cmd_client = QueueServerAdaptiveClient(
                session_id=str(args.session_id),
                zmq_control_addr=str(args.qserver_control_addr),
                zmq_info_addr=str(args.qserver_info_addr),
                user=str(args.qserver_user),
                user_group=str(args.qserver_user_group),
            )

            def _submit(command: str, payload: Optional[Dict] = None):
                def _worker():
                    try:
                        resp = cmd_client.submit(command, payload=payload)
                        ok = bool(resp.get("success", resp.get("ok", False)))
                        if ok:
                            bridge._log_received.emit(
                                f"Adaptive command submitted: {command}"
                            )
                        else:
                            bridge._log_received.emit(
                                f"Adaptive command failed: {command} :: {resp}"
                            )
                    except Exception as ex:
                        bridge._log_received.emit(
                            f"Adaptive command failed: {command} :: {ex}"
                        )

                threading.Thread(
                    target=_worker,
                    daemon=True,
                    name=f"focus-cmd-{str(command)}",
                ).start()

            def _on_go_to_focus(metric: str):
                target = bridge.get_focus_target(metric)
                payload = {}
                if target is not None and np.isfinite(float(target)):
                    payload["target_position"] = float(target)
                payload["metric"] = str(metric)
                _submit("go_to_focus", payload=payload)

            def _on_scan_around_focus(metric: str, step_size: float):
                target = bridge.get_focus_target(metric)
                payload = {
                    "metric": str(metric),
                    "step_size": float(max(1e-4, float(step_size))),
                    "num_points": 7,
                }
                if target is not None and np.isfinite(float(target)):
                    payload["center"] = float(target)
                _submit("scan_around_focus", payload=payload)

            def _on_extend_left():
                _submit("extend_left", payload={"num_points": 3})

            def _on_extend_right():
                _submit("extend_right", payload={"num_points": 3})

            def _on_complete(command: str = "complete"):
                command_name = str(command or "complete").strip().lower()
                if command_name not in {"complete", "abort"}:
                    command_name = "complete"
                _submit(command_name, payload={})

            bridge.on_go_to_focus = _on_go_to_focus
            bridge.on_scan_around_focus = _on_scan_around_focus
            bridge.on_extend_left = _on_extend_left
            bridge.on_extend_right = _on_extend_right
            bridge.on_mark_complete = _on_complete
        except Exception as ex:
            print(f"Adaptive command client init failed: {ex}")

    try:
        from bluesky.callbacks.zmq import RemoteDispatcher
    except Exception:
        from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher

    dispatcher = RemoteDispatcher(args.zmq_address)
    dispatcher.subscribe(bridge.on_document)

    dispatch_thread = threading.Thread(target=dispatcher.start, daemon=True)
    dispatch_thread.start()

    # Guard 1: don't leave detached viewers if launcher is gone.
    parent_pid = int(max(0, int(args.parent_pid or 0)))
    if parent_pid > 0:
        parent_timer = QtCore.QTimer()
        parent_timer.setInterval(2000)

        def _parent_alive(pid: int) -> bool:
            try:
                os.kill(pid, 0)
                return True
            except Exception:
                return False

        def _check_parent():
            if not _parent_alive(parent_pid):
                print(f"Parent process {parent_pid} is gone; exiting focus viewer.")
                app.quit()

        parent_timer.timeout.connect(_check_parent)
        parent_timer.start()

    # Guard 2: if nothing was received, auto-exit so no unseen process lingers.
    startup_timeout_s = float(max(0.0, float(args.startup_timeout_s or 0.0)))
    if startup_timeout_s > 0:
        def _check_startup_timeout():
            if bridge.window is None:
                print(
                    f"No frames received within {startup_timeout_s:.1f}s; exiting focus viewer."
                )
                app.quit()
        QtCore.QTimer.singleShot(int(startup_timeout_s * 1000), _check_startup_timeout)

    try:
        return int(app.exec_())
    finally:
        try:
            dispatcher.stop()
        except Exception:
            pass
        try:
            if session_lock is not None:
                session_lock.unlock()
        except Exception:
            pass
        dispatch_thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
