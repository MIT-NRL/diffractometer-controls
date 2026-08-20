#!/usr/bin/env python3
"""Online focus viewer that consumes Bluesky documents and reuses FocusOfflineWindow."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import math
import os
import re
import signal
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
    from focus_offline_viewer import (
        DEFAULT_FOCUS_BULK_WORKERS,
        DEFAULT_FOCUS_FULL_CACHE_GB,
        DEFAULT_FOCUS_FULL_WORKERS,
        DEFAULT_FOCUS_MAX_WORKERS_TOTAL,
        DEFAULT_FOCUS_PREPROCESS_MODE,
        DEFAULT_FOCUS_PREPROCESS_SIZE,
        FocusOfflineWindow,
        FrameInfo,
        _apply_saved_theme,
        _build_focus_program_icon,
    )
except Exception:
    from diffractometer_controls.focus_offline_viewer import (
        DEFAULT_FOCUS_BULK_WORKERS,
        DEFAULT_FOCUS_FULL_CACHE_GB,
        DEFAULT_FOCUS_FULL_WORKERS,
        DEFAULT_FOCUS_MAX_WORKERS_TOTAL,
        DEFAULT_FOCUS_PREPROCESS_MODE,
        DEFAULT_FOCUS_PREPROCESS_SIZE,
        FocusOfflineWindow,
        FrameInfo,
        _apply_saved_theme,
        _build_focus_program_icon,
    )


def _is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


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
    owner_text = ""
    try:
        has_owner, owner_pid, owner_host, owner_app = lock.getLockInfo()
        if has_owner:
            owner_text = (
                f" owner_pid={owner_pid} owner_host={owner_host} "
                f"owner_app={owner_app}"
            )
    except Exception:
        pass
    print(
        f"Focus viewer already running for session {sid}; exiting duplicate process. "
        f"duplicate_pid={os.getpid()} parent_pid={os.getppid()} "
        f"DISPLAY={os.environ.get('DISPLAY', 'unset')}{owner_text}"
    )
    return None


def _has_received_focus_frame(bridge) -> bool:
    """Return whether the viewer has accepted its first valid image file."""
    window = getattr(bridge, "window", None)
    return bool(window is not None and getattr(window, "frames", ()))


def _has_observed_run_activity(bridge) -> bool:
    """Return whether any document for the tracked run has been received.

    Used by the startup guard: a live run whose first exposure is simply long
    must not be torn down, because exiting also aborts the adaptive session.
    """
    return bool(getattr(bridge, "_observed_run_activity", False))


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

    def submit(
        self,
        command: str,
        payload: Optional[Dict] = None,
        *,
        confirmation_timeout_s: float = 10.0,
    ) -> Dict:
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
            response = _exec()
        except Exception as ex:
            msg = str(ex)
            # Queue Server may keep stale permissions in memory until reloaded.
            if ("not allowed" in msg.lower()) or ("permission" in msg.lower()):
                try:
                    self._api.permissions_reload()
                except Exception:
                    pass
                try:
                    response = _exec()
                except Exception as ex2:
                    return {
                        "success": False,
                        "ok": False,
                        "error": str(ex2),
                        "command": str(command),
                    }
            else:
                return {
                    "success": False,
                    "ok": False,
                    "error": msg,
                    "command": str(command),
                }

        response = dict(response or {})
        if not bool(response.get("success", False)):
            response.setdefault("ok", False)
            return response
        task_uid = str(response.get("task_uid", "")).strip()
        if not task_uid:
            return {
                **response,
                "success": False,
                "ok": False,
                "error": "missing_task_uid",
            }

        try:
            self._api.wait_for_completed_task(
                task_uid,
                timeout=float(max(1.0, confirmation_timeout_s)),
                treat_not_found_as_completed=False,
            )
            task_reply = dict(self._api.task_result(task_uid) or {})
        except Exception as ex:
            return {
                **response,
                "success": False,
                "ok": False,
                "error": f"command_confirmation_failed: {ex}",
            }

        task_result = dict(task_reply.get("result", {}) or {})
        if task_reply.get("status") != "completed" or not bool(task_result.get("success", False)):
            return {
                **response,
                "success": False,
                "ok": False,
                "error": str(task_result.get("msg") or task_reply.get("msg") or "command_task_failed"),
            }
        return_value = task_result.get("return_value", {})
        if not isinstance(return_value, dict):
            return_value = {"ok": True, "return_value": return_value}
        return {
            **response,
            **return_value,
            "success": bool(return_value.get("ok", True)),
            "task_uid": task_uid,
        }


class FocusOnlineBridge(QtCore.QObject):
    """Translate Bluesky documents into incremental frame updates for FocusOfflineWindow."""

    _document_received = QtCore.Signal(str, object)
    _frame_received = QtCore.Signal(str, float)
    _log_received = QtCore.Signal(str)
    _run_stopped = QtCore.Signal()
    _go_focus_requested = QtCore.Signal(str)
    _scan_focus_requested = QtCore.Signal(str, float)
    _extend_left_requested = QtCore.Signal()
    _extend_right_requested = QtCore.Signal()
    _mark_complete_requested = QtCore.Signal()
    _mark_aborted_requested = QtCore.Signal()
    _terminal_command_failed = QtCore.Signal(str)
    _expected_frame_adjustment = QtCore.Signal(int)

    def __init__(
        self,
        *,
        image_key: Optional[str] = None,
        motor_key: Optional[str] = None,
        stream_name: str = "primary",
        run_uid: Optional[str] = None,
        expected_frame_count: Optional[int] = None,
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
        max_workers_total: int = DEFAULT_FOCUS_MAX_WORKERS_TOTAL,
        bulk_workers: Optional[int] = None,
        full_workers: Optional[int] = None,
        full_cache_gb: float = DEFAULT_FOCUS_FULL_CACHE_GB,
        preprocess_mode: str = DEFAULT_FOCUS_PREPROCESS_MODE,
        preprocess_size: int = DEFAULT_FOCUS_PREPROCESS_SIZE,
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
        try:
            parsed_expected_count = int(expected_frame_count)
        except (TypeError, ValueError):
            parsed_expected_count = 0
        self.expected_frame_count: Optional[int] = (
            parsed_expected_count if parsed_expected_count > 0 else None
        )
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
        self.bulk_workers = (
            int(max(1, bulk_workers)) if bulk_workers is not None else None
        )
        self.full_workers = (
            int(max(1, full_workers)) if full_workers is not None else None
        )
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
        # Set once any document for the tracked run arrives, so the startup guard
        # can distinguish "no run at all" from "run is live but still exposing".
        self._observed_run_activity = False
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
        self._focus_action_timer: Optional[QtCore.QTimer] = None

        # Documents and Queue Server replies arrive on background Python threads.
        # Explicit queued delivery is required before any slot touches widgets or
        # starts Qt timers; AutoConnection can otherwise execute a Python slot on
        # the emitting thread and produce QObject::startTimer warnings.
        queued = QtCore.Qt.QueuedConnection
        self._document_received.connect(self._on_document_received, queued)
        self._frame_received.connect(self._on_frame_received, queued)
        self._log_received.connect(self._on_log_received, queued)
        self._run_stopped.connect(self._on_run_stopped, queued)
        self._go_focus_requested.connect(self._on_go_focus_requested, queued)
        self._scan_focus_requested.connect(self._on_scan_focus_requested, queued)
        self._extend_left_requested.connect(self._on_extend_left_requested, queued)
        self._extend_right_requested.connect(self._on_extend_right_requested, queued)
        self._mark_complete_requested.connect(self._on_mark_complete_requested, queued)
        self._mark_aborted_requested.connect(self._on_mark_aborted_requested, queued)
        self._terminal_command_failed.connect(self._on_terminal_command_failed, queued)
        self._expected_frame_adjustment.connect(
            self._on_expected_frame_adjustment, queued
        )

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
            expected_frame_count=self.expected_frame_count,
        )
        self.window.warm_full_process_pool()
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

    @QtCore.Slot(int)
    def _on_expected_frame_adjustment(self, delta: int):
        """Update the acquisition target on the window's Qt thread."""
        if self.window is None:
            if self.expected_frame_count is not None:
                self.expected_frame_count = max(
                    0, int(self.expected_frame_count) + int(delta)
                )
            return
        updated = self.window.adjust_expected_stream_frames(int(delta))
        if updated is not None:
            self.expected_frame_count = int(updated)

    def show_window_now(self):
        """Create and raise the viewer before any frame has arrived.

        The window is opened when the plan starts so the operator can confirm the
        session is live and pre-position the ROI. Frames populate it afterwards.
        """
        self._ensure_window()
        if self.window is None:
            return
        self.window.statusBar().showMessage(
            "Waiting for the first image; set the ROI around the foil edge."
        )

    @staticmethod
    def _bring_window_to_front(window: QtWidgets.QWidget):
        try:
            state = window.windowState()
            if state & QtCore.Qt.WindowMinimized:
                window.setWindowState(state & ~QtCore.Qt.WindowMinimized)
            window.show()
            window.raise_()
            window.activateWindow()
            # Do not schedule topmost/raise pulses here. Besides causing the
            # original unmap/remap flicker on X11, a launch accidentally routed
            # through a dispatcher thread would also attempt to start Qt timers.
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
        extend_left_btn = QtWidgets.QPushButton("Extend Left (3 pts)", self.window)
        extend_right_btn = QtWidgets.QPushButton("Extend Right (3 pts)", self.window)
        extend_left_btn.setToolTip(
            "Acquire three points beyond the current left bound using the coarse-scan spacing."
        )
        extend_right_btn.setToolTip(
            "Acquire three points beyond the current right bound using the coarse-scan spacing."
        )
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
        combo.currentTextChanged.connect(lambda _text: self._refresh_focus_action_state())
        self._focus_action_timer = QtCore.QTimer(self)
        self._focus_action_timer.setInterval(250)
        self._focus_action_timer.timeout.connect(self._refresh_focus_action_state)
        self._focus_action_timer.start()
        self._refresh_focus_action_state()

    @QtCore.Slot()
    def _refresh_focus_action_state(self, *_args):
        if self._focus_metric_combo is None:
            return
        metric = str(self._focus_metric_combo.currentText())
        target = self.get_focus_target(metric)
        ready = bool((not self._complete_sent) and target is not None)
        for button in (self._go_focus_button, self._scan_focus_button):
            if button is not None:
                button.setEnabled(ready)
                button.setToolTip(
                    f"Validated full-quality target: {float(target):.5f}"
                    if ready
                    else "Waiting for at least five full-quality, in-range focus points."
                )

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

    @QtCore.Slot(str)
    def _on_terminal_command_failed(self, message: str):
        self._complete_sent = False
        if self._scan_step_spin is not None:
            self._scan_step_spin.setEnabled(True)
        for button in (
            self._extend_left_button,
            self._extend_right_button,
            self._complete_button,
        ):
            if button is not None:
                button.setEnabled(True)
        self._refresh_focus_action_state()
        self._log_received.emit(str(message))

    def get_focus_target(self, metric: str = "mtf50") -> Optional[float]:
        if self.window is None:
            return None
        try:
            target = self.window.validated_focus_target(str(metric))
        except Exception:
            target = None
        return float(target) if target is not None and np.isfinite(float(target)) else None

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
            # Score motor-like fields instead of accepting the first field that
            # contains "focus".  A detector named e.g. ``sim_focus_cam`` puts
            # that word in every detector field and previously caused an image
            # total or blur sigma to be used as the plot's motor coordinate.
            excluded_terms = (
                "image", "file", "path", "count", "sum", "total", "mean",
                "sigma", "mtf", "psf", "lsf", "blur", "stats", "array",
            )
            scored = []
            for order, (key, entry) in enumerate(data_keys.items()):
                key_text = str(key).strip()
                key_lower = key_text.lower()
                if "setpoint" in key_lower:
                    continue
                entry = entry or {}
                dtype = str(entry.get("dtype", "")).lower()
                shape = entry.get("shape", [])
                if not (dtype in {"number", "integer", "float"} or shape in ([], (), None)):
                    continue
                object_name = str(entry.get("object_name", "")).strip()
                object_lower = object_name.lower()
                combined = f"{key_lower} {object_lower}"
                score = 0
                if key_lower.endswith("_position") or key_lower == "position":
                    score += 120
                if "motor" in key_lower:
                    score += 100
                if "motor" in object_lower:
                    score += 80
                if "position" in key_lower:
                    score += 60
                if "position" in object_lower:
                    score += 40
                if "focus" in key_lower:
                    score += 20
                if "focus" in object_lower:
                    score += 10
                if any(term in combined for term in excluded_terms):
                    score -= 200
                if score > 0:
                    scored.append((score, -order, key_text))
            if scored:
                candidates.append(max(scored)[2])

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
        # An accepted event proves the tracked run is producing data, even if the
        # image file cannot be resolved yet.
        self._observed_run_activity = True
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
        # RemoteDispatcher invokes callbacks on its receiving thread. Queue the
        # complete document before touching bridge/window state; start handling
        # may close a window and therefore stop its Qt timers.
        self._document_received.emit(str(name), dict(doc or {}))

    @QtCore.Slot(str, object)
    def _on_document_received(self, name: str, doc: Dict):
        """Handle one Bluesky document on the bridge's Qt thread."""
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
            # Local RunEngine attachments receive the start document directly.
            # Spawned viewers normally receive this value through the CLI because
            # their subscription may begin after the start document was emitted.
            focus_md = dict(doc.get("focus_adaptive", {}) or {})
            plan_pattern_args = dict(doc.get("plan_pattern_args", {}) or {})
            try:
                start_expected_count = max(
                    0,
                    int(
                        plan_pattern_args.get(
                            "num_steps", focus_md.get("num_steps", 0)
                        )
                    ),
                )
            except (TypeError, ValueError):
                start_expected_count = 0
            if start_expected_count > 0:
                self.expected_frame_count = int(start_expected_count)
                if self.window is not None:
                    self.window.set_expected_stream_frames(start_expected_count)
            # This is the run we are tracking; it is live even before any frame.
            self._observed_run_activity = True
            if self.reset_viewer_on_new_run:
                # Keep an existing window that has not shown any frame yet.
                # The viewer is spawned from this plan's start document, so the
                # placeholder window it created is the window this run needs.
                # Closing and reopening it here caused the visible
                # appear/disappear/reappear flicker at scan start.
                keep_placeholder = bool(
                    (self.window is not None)
                    and (len(getattr(self.window, "frames", [])) == 0)
                    and (not self._seen_paths)
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
        self.window.mark_stream_complete()
        self.expected_frame_count = total
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
                    # If every frame was already prepared while the online gate
                    # was waiting for more acquisition, closing the stream is the
                    # state change that must retry finalization.
                    self.window._maybe_start_full_reprocess_after_scan()
                    self.window._maybe_finalize_full_metric_pass()
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
            return
        self._path_retry_count.pop(norm, None)
        self._path_first_seen_ts.pop(norm, None)

        self._ensure_window()
        if self.window is None:
            return

        idx = int(len(self.window.frames))
        self.window.frames.append(FrameInfo(index=idx, path=path, position=float(position)))
        # Pass the index so full-resolution processing is queued even if the
        # quick-pass load is refused by task backpressure.
        try:
            self.window.note_stream_frame_added(idx)
        except TypeError:
            self.window.note_stream_frame_added()
        if idx == 0:
            self.window._log(
                f"First streamed frame: {path.name} @ motor={float(position):.5f}"
            )
        self.window._update_filter_queue_indicator()
        self.window._log(
            "Streamed frame "
            f"{idx + 1}/{self.expected_frame_count or len(self.window.frames)}: "
            f"{path.name} @ motor={float(position):.5f}"
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
    expected_frame_count: Optional[int] = None,
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
    max_workers_total: int = DEFAULT_FOCUS_MAX_WORKERS_TOTAL,
    bulk_workers: Optional[int] = None,
    full_workers: Optional[int] = None,
    full_cache_gb: float = DEFAULT_FOCUS_FULL_CACHE_GB,
    preprocess_mode: str = DEFAULT_FOCUS_PREPROCESS_MODE,
    preprocess_size: int = DEFAULT_FOCUS_PREPROCESS_SIZE,
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
        expected_frame_count=expected_frame_count,
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
    p.add_argument(
        "--expected-frame-count",
        type=int,
        default=0,
        help=(
            "Expected image frames in the initial online acquisition; final edge "
            "locking waits until this many unique frames arrive."
        ),
    )
    p.add_argument("--no-follow-latest", action="store_true", help="Do not auto-jump to newest frame")
    p.add_argument(
        "--keep-runs-combined",
        action="store_true",
        help="Append all runs into one viewer session instead of resetting on each new run.",
    )
    p.add_argument("--interval-ms", type=int, default=200, help="Playback interval for the base viewer")
    p.add_argument(
        "--max-workers-total",
        type=int,
        default=DEFAULT_FOCUS_MAX_WORKERS_TOTAL,
        help="Total worker cap",
    )
    p.add_argument(
        "--bulk-workers",
        type=int,
        default=DEFAULT_FOCUS_BULK_WORKERS,
        help="Bulk/ROI workers",
    )
    p.add_argument(
        "--full-workers",
        type=int,
        default=DEFAULT_FOCUS_FULL_WORKERS,
        help="Full filter workers",
    )
    p.add_argument(
        "--full-cache-gb",
        type=float,
        default=DEFAULT_FOCUS_FULL_CACHE_GB,
        help="Full filtered cache budget (GB)",
    )
    p.add_argument(
        "--preprocess-mode",
        type=str,
        choices=["gamma", "tomopy_outlier", "median"],
        default=DEFAULT_FOCUS_PREPROCESS_MODE,
        help="Prefilter mode for image processing.",
    )
    p.add_argument(
        "--preprocess-size",
        type=int,
        default=DEFAULT_FOCUS_PREPROCESS_SIZE,
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
        expected_frame_count=args.expected_frame_count,
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
    command_executor = None
    pending_command_future = None
    terminal_future = None
    viewer_ready_callback = None

    if args.session_id:
        try:
            cmd_client = QueueServerAdaptiveClient(
                session_id=str(args.session_id),
                zmq_control_addr=str(args.qserver_control_addr),
                zmq_info_addr=str(args.qserver_info_addr),
                user=str(args.qserver_user),
                user_group=str(args.qserver_user_group),
            )
            command_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="focus-command",
            )

            def _submit(
                command: str,
                payload: Optional[Dict] = None,
                *,
                expected_additional_frames: int = 0,
            ):
                nonlocal pending_command_future
                command_name = str(command or "").strip().lower()
                is_terminal = command_name in {"complete", "abort"}
                expected_delta = max(0, int(expected_additional_frames))
                if (
                    not is_terminal
                    and pending_command_future is not None
                    and not pending_command_future.done()
                ):
                    bridge._log_received.emit(
                        f"Adaptive command ignored while another command is being confirmed: {command_name}"
                    )
                    return None

                # Open the new acquisition batch before submitting it so even a
                # zero-exposure detector cannot deliver its first image while the
                # viewer still considers the previous dataset complete.
                if expected_delta:
                    bridge._on_expected_frame_adjustment(expected_delta)

                def _worker():
                    try:
                        resp = cmd_client.submit(command_name, payload=payload)
                        ok = bool(resp.get("success", resp.get("ok", False)))
                        if ok:
                            bridge._log_received.emit(
                                f"Adaptive command confirmed: {command_name}"
                            )
                        else:
                            bridge._log_received.emit(
                                f"Adaptive command failed: {command_name} :: {resp}"
                            )
                            if expected_delta:
                                bridge._expected_frame_adjustment.emit(-expected_delta)
                            if is_terminal:
                                bridge._terminal_command_failed.emit(
                                    f"Terminal command failed: {command_name} :: {resp}"
                                )
                    except Exception as ex:
                        bridge._log_received.emit(
                            f"Adaptive command failed: {command_name} :: {ex}"
                        )
                        if expected_delta:
                            bridge._expected_frame_adjustment.emit(-expected_delta)
                        if is_terminal:
                            bridge._terminal_command_failed.emit(
                                f"Terminal command failed: {command_name} :: {ex}"
                            )
                try:
                    pending_command_future = command_executor.submit(_worker)
                except Exception:
                    if expected_delta:
                        bridge._on_expected_frame_adjustment(-expected_delta)
                    raise
                return pending_command_future

            def _on_go_to_focus(metric: str):
                target = bridge.get_focus_target(metric)
                if target is None or not np.isfinite(float(target)):
                    bridge._log_received.emit(
                        f"Go to Focus blocked: no validated {metric} target is ready."
                    )
                    return
                payload = {
                    "target_position": float(target),
                    "metric": str(metric),
                }
                return _submit(
                    "go_to_focus",
                    payload=payload,
                    expected_additional_frames=1,
                )

            def _on_scan_around_focus(metric: str, step_size: float):
                target = bridge.get_focus_target(metric)
                if target is None or not np.isfinite(float(target)):
                    bridge._log_received.emit(
                        f"Scan Around Focus blocked: no validated {metric} target is ready."
                    )
                    return
                payload = {
                    "metric": str(metric),
                    "step_size": float(max(1e-4, float(step_size))),
                    "num_points": 7,
                    "center": float(target),
                }
                return _submit(
                    "scan_around_focus",
                    payload=payload,
                    expected_additional_frames=7,
                )

            def _on_extend_left():
                return _submit(
                    "extend_left",
                    payload={"num_points": 3},
                    expected_additional_frames=3,
                )

            def _on_extend_right():
                return _submit(
                    "extend_right",
                    payload={"num_points": 3},
                    expected_additional_frames=3,
                )

            def _on_complete(command: str = "complete"):
                nonlocal terminal_future
                command_name = str(command or "complete").strip().lower()
                if command_name not in {"complete", "abort"}:
                    command_name = "complete"
                terminal_future = _submit(command_name, payload={})

            def _on_viewer_ready():
                _submit("viewer_ready", payload={"subscriber": "online_focus_viewer"})

            bridge.on_go_to_focus = _on_go_to_focus
            bridge.on_scan_around_focus = _on_scan_around_focus
            bridge.on_extend_left = _on_extend_left
            bridge.on_extend_right = _on_extend_right
            bridge.on_mark_complete = _on_complete
            viewer_ready_callback = _on_viewer_ready
        except Exception as ex:
            print(f"Adaptive command client init failed: {ex}")

    def _abort_unfinished_session():
        if args.session_id and not bridge._complete_sent and bridge.on_mark_complete is not None:
            bridge._mark_aborted_requested.emit()

    app.aboutToQuit.connect(_abort_unfinished_session)

    def _request_qt_shutdown(_signum, _frame):
        app.quit()

    for _signal_name in ("SIGTERM", "SIGINT"):
        _signal_value = getattr(signal, _signal_name, None)
        if _signal_value is not None:
            try:
                signal.signal(_signal_value, _request_qt_shutdown)
            except Exception:
                pass

    try:
        from bluesky.callbacks.zmq import RemoteDispatcher
    except Exception:
        from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher

    dispatcher = RemoteDispatcher(args.zmq_address)
    dispatcher.subscribe(bridge.on_document)

    dispatch_thread = threading.Thread(target=dispatcher.start, daemon=True)
    dispatch_thread.start()
    # Show the window as soon as the viewer process is up, rather than waiting for
    # the first image. The plan spawns this process from its start document, so
    # this is effectively "appears when the plan starts".
    bridge.show_window_now()
    if viewer_ready_callback is not None:
        # Avoid the PUB/SUB slow-joiner window: report readiness only after the
        # receiving thread has had time to establish its subscription.
        QtCore.QTimer.singleShot(500, viewer_ready_callback)

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

    # Guard 2: don't leave an idle viewer behind if no run ever reaches us. The
    # window is now shown at startup, so this must key off run activity rather
    # than frames: exiting also aborts the adaptive session, and a long first
    # exposure must never be mistaken for a dead session.
    startup_timeout_s = float(max(0.0, float(args.startup_timeout_s or 0.0)))
    if startup_timeout_s > 0:
        def _check_startup_timeout():
            if _has_observed_run_activity(bridge) or _has_received_focus_frame(bridge):
                return
            print(
                f"No run documents received within {startup_timeout_s:.1f}s; "
                "exiting focus viewer."
            )
            app.quit()
        QtCore.QTimer.singleShot(int(startup_timeout_s * 1000), _check_startup_timeout)

    try:
        return int(app.exec_())
    finally:
        if terminal_future is not None:
            try:
                terminal_future.result(timeout=12.0)
            except Exception:
                pass
        if command_executor is not None:
            try:
                command_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                command_executor.shutdown(wait=False)
            except Exception:
                pass
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
