#!/usr/bin/env python3
"""Online focus viewer that consumes Bluesky documents and reuses FocusOfflineWindow."""

from __future__ import annotations

import argparse
import math
import sys
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from qtpy import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np

try:
    from bluesky_queueserver_api import BFunc
    from bluesky_queueserver_api.zmq import REManagerAPI
except Exception:
    BFunc = None
    REManagerAPI = None

try:
    from focus_offline_viewer import FocusOfflineWindow, FrameInfo
except Exception:
    from diffractometer_controls.focus_offline_viewer import FocusOfflineWindow, FrameInfo


def _is_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


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
        return self._api.function_execute(
            item,
            run_in_background=True,
            user=self._user,
            user_group=self._user_group,
        )


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
        preprocess_mode: str = "tomopy_outlier",
        preprocess_size: int = 7,
        parent=None,
    ):
        super().__init__(parent=parent)
        self.image_key = image_key
        self.motor_key = motor_key
        self.stream_name = str(stream_name)
        self.run_uid_filter = run_uid
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
        self.preprocess_mode = str(preprocess_mode or "median")
        self.preprocess_size = int(max(1, preprocess_size))

        self.window: Optional[FocusOfflineWindow] = None

        self._descriptor_stream: Dict[str, str] = {}
        self._active_run_uid: Optional[str] = None
        self._seen_paths = set()
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
        step_label = QtWidgets.QLabel("Scan step:", self.window)
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
                self._log_received.emit("Viewer closed: marking focus workflow complete.")
                self._mark_complete_requested.emit()
        return super().eventFilter(watched, event)

    def _detect_image_key(self, data: Dict) -> Optional[str]:
        if self.image_key and self.image_key in data:
            return self.image_key
        for k, v in data.items():
            lk = str(k).lower()
            if ("image" in lk and "path" in lk) and isinstance(v, str):
                return str(k)
        return None

    def _detect_motor_key(self, data: Dict) -> Optional[str]:
        if self.motor_key and self.motor_key in data:
            return self.motor_key
        candidates = []
        for k, v in data.items():
            lk = str(k).lower()
            if "setpoint" in lk:
                continue
            if ("motor" in lk or "position" in lk) and _is_number(v):
                candidates.append(str(k))
        if not candidates:
            return None
        if "focus_sim_motor" in candidates:
            return "focus_sim_motor"
        return candidates[0]

    def on_document(self, name: str, doc: Dict):
        """Bluesky callback entry point: subscribe this to RE/dispatcher."""
        if name == "start":
            uid = str(doc.get("uid", ""))
            self._active_run_uid = uid
            self._descriptor_stream.clear()
            if self.run_uid_filter and uid != self.run_uid_filter:
                return
            if self.reset_viewer_on_new_run:
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

        if self.run_uid_filter and self._active_run_uid != self.run_uid_filter:
            return

        if name == "descriptor":
            self._descriptor_stream[str(doc.get("uid", ""))] = str(doc.get("name", ""))
            return

        if name == "event":
            descriptor_uid = str(doc.get("descriptor", ""))
            stream = self._descriptor_stream.get(descriptor_uid, "")
            if self.stream_name and stream != self.stream_name:
                return
            data = doc.get("data", {}) or {}
            image_key = self._detect_image_key(data)
            if image_key is None:
                return

            motor_key = self._detect_motor_key(data)
            image_path = str(data.get(image_key, "")).strip()
            if not image_path:
                return
            if motor_key is not None and _is_number(data.get(motor_key)):
                position = float(data.get(motor_key))
                self._fallback_position = position
            else:
                position = float(self._fallback_position)

            self._frame_received.emit(image_path, float(position))
            return

        if name == "stop":
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
        try:
            # Mark all streamed frames as seen so the window can treat the run as complete.
            self.window._seen_frame_indices.update(range(total))
            # Queue any missing full-filter jobs without requiring manual frame cycling.
            for idx in range(total):
                self.window._enqueue_full_prepare(idx)
            self.window._update_filter_queue_indicator()
            self.window._log(
                f"Run stop sync: queued full filtering for all streamed frames ({total})."
            )
        except Exception as ex:
            self._log_received.emit(f"Run stop sync failed: {ex}")

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

        if self.window is None:
            frames = [FrameInfo(index=0, path=path, position=float(position))]
            self.window = FocusOfflineWindow(
                frames=frames,
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
            self.window.show()
            self._bring_window_to_front(self.window)
            self._install_focus_controls()
            self.window._log("Online stream connected; waiting for frames...")
            self.window._log(
                f"First streamed frame: {path.name} @ motor={float(position):.5f}"
            )
            return

        idx = int(len(self.window.frames))
        self.window.frames.append(FrameInfo(index=idx, path=path, position=float(position)))
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
    preprocess_mode: str = "tomopy_outlier",
    preprocess_size: int = 7,
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
        choices=["median", "tomopy_outlier"],
        default="tomopy_outlier",
        help="Prefilter mode for image processing.",
    )
    p.add_argument(
        "--preprocess-size",
        type=int,
        default=7,
        help="Kernel size for selected prefilter mode.",
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
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
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
    )

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
                resp = cmd_client.submit(command, payload=payload)
                ok = bool(resp.get("success", resp.get("ok", False)))
                if ok:
                    bridge._on_log_received(
                        f"Adaptive command submitted: {command}"
                    )
                else:
                    bridge._on_log_received(
                        f"Adaptive command failed: {command} :: {resp}"
                    )

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

            def _on_complete():
                _submit("complete", payload={})

            bridge.on_go_to_focus = _on_go_to_focus
            bridge.on_scan_around_focus = _on_scan_around_focus
            bridge.on_extend_left = _on_extend_left
            bridge.on_extend_right = _on_extend_right
            bridge.on_mark_complete = _on_complete
        except Exception as ex:
            print(f"Adaptive command client init failed: {ex}")

    from bluesky.callbacks.zmq import RemoteDispatcher

    dispatcher = RemoteDispatcher(args.zmq_address)
    dispatcher.subscribe(bridge.on_document)

    dispatch_thread = threading.Thread(target=dispatcher.start, daemon=True)
    dispatch_thread.start()

    try:
        return int(app.exec_())
    finally:
        try:
            dispatcher.stop()
        except Exception:
            pass
        dispatch_thread.join(timeout=2.0)


if __name__ == "__main__":
    raise SystemExit(main())
