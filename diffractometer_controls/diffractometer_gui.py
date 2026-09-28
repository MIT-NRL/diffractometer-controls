import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

# from pydm.display import Display
from qtpy import QtCore, QtGui, QtWidgets
from pydm.widgets.channel import PyDMChannel

# from bluesky_widgets.qt.figures import QtFigure, QtFigures
# from bluesky_widgets.models.auto_plot_builders import AutoLines, AutoPlotter, AutoImages
# from bluesky_widgets.models.plot_builders import Lines, Images
from bluesky_widgets.models.run_engine_client import RunEngineClient

try:
    from . import display
except ImportError:
    import display


_DIFFRACTION_PLOT_SUPPORT = None


def _acquisition_monitor_from_start_doc(start_doc):
    """Return the generic exposure monitor declared by a diffraction run."""
    start_doc = dict(start_doc or {})
    monitor = start_doc.get("acquisition_monitor", {})
    if isinstance(monitor, Mapping) and str(monitor.get("active_pv", "") or "").strip():
        return dict(monitor)

    # Compatibility for HE3 runs recorded before acquisition-monitor metadata
    # was added. New runs always provide the PVs through the Start document.
    detector_type = str(start_doc.get("detector_type", "") or "").strip().lower()
    if detector_type == "he3psd":
        duration = dict(start_doc.get("plan_args", {}) or {}).get("acquire_time")
        return {
            "device": "he3psd",
            "active_pv": "4dh4:he3PSD:Acquire_RBV",
            "duration_pv": "4dh4:he3PSD:AcquireTime_RBV",
            "remaining_pv": "4dh4:he3PSD:AcquireTimeRemaining_RBV",
            "duration": duration,
        }
    return {}


class _DiffractionUnavailableWidget(QtWidgets.QFrame):
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setObjectName("diffractionUnavailablePanel")

        title = QtWidgets.QLabel("Diffraction viewer unavailable", self)
        title_font = QtGui.QFont(self.font())
        title_font.setPointSize(max(12, title_font.pointSize() + 1))
        title_font.setBold(True)
        title.setFont(title_font)

        body = QtWidgets.QLabel(str(message or ""), self)
        body.setWordWrap(True)
        body.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(body)
        layout.addStretch(1)
        self.setLayout(layout)


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _run_module_import_probe(module_name):
    command = [sys.executable, "-X", "faulthandler", "-c", f"import {module_name}"]
    try:
        result = subprocess.run(
            command,
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return False, str(exc)

    if result.returncode == 0:
        return True, ""

    details = (result.stderr or result.stdout or "").strip()
    if not details:
        details = f"Import probe exited with code {result.returncode}."
    return False, details


def _summarize_probe_error(text):
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return "Unknown import failure."


def _get_diffraction_plot_support():
    global _DIFFRACTION_PLOT_SUPPORT
    if _DIFFRACTION_PLOT_SUPPORT is not None:
        return dict(_DIFFRACTION_PLOT_SUPPORT)

    controller_ok, controller_error = _run_module_import_probe(
        "diffractometer_controls.diffraction_live_plot"
    )
    pyqtgraph_ok = False
    pyqtgraph_error = ""
    if controller_ok:
        pyqtgraph_ok, pyqtgraph_error = _run_module_import_probe(
            "diffractometer_controls.diffraction_live_plot_pyqtgraph"
        )

    support = {
        "controller_ok": controller_ok,
        "controller_error": controller_error,
        "pyqtgraph_ok": pyqtgraph_ok,
        "pyqtgraph_error": pyqtgraph_error,
    }

    if controller_ok:
        support["message"] = ""
    else:
        summary = _summarize_probe_error(controller_error)
        support["message"] = (
            "The diffraction plotting stack could not be imported in this Python "
            f"environment:\n{sys.executable}\n\n"
            f"Import failure:\n{summary}\n\n"
            "The rest of the GUI can still run, but the diffraction viewer is disabled "
            "until the plotting environment is repaired."
        )

    _DIFFRACTION_PLOT_SUPPORT = dict(support)
    return support


def _load_diffraction_plot_classes():
    from diffractometer_controls.diffraction_live_plot import (
        DiffractionHistoryViewer,
        DiffractionLivePlot,
        DiffractionPlotWidget,
    )

    plot_widget_pyqtgraph = None
    support = _get_diffraction_plot_support()
    if support.get("pyqtgraph_ok"):
        from diffractometer_controls.diffraction_live_plot_pyqtgraph import (
            DiffractionPlotWidgetPyQtGraph,
        )

        plot_widget_pyqtgraph = DiffractionPlotWidgetPyQtGraph

    return {
        "viewer": DiffractionHistoryViewer,
        "controller": DiffractionLivePlot,
        "matplotlib_plot": DiffractionPlotWidget,
        "pyqtgraph_plot": plot_widget_pyqtgraph,
    }

class MainScreen(display.MITRDisplay):
    _acquisition_document_received = QtCore.Signal(str, object)

    re_client: RunEngineClient

    def __init__(self, parent=None, args=None, macros=None, ui_filename='diffractometer_gui.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("MainScreen here")

    def ui_filename(self):
        return 'diffractometer_gui.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        from application import MITRApplication

        self._time_remaining_channel = None
        self._acquire_time_channel = None
        self._acquire_channel = None
        self._elapsed_time_channel = None
        self._manual_channels_connected = False
        self._document_subscription = None
        self._acquisition_document_subscription = None
        self._acquisition_run_uid = ""
        self._acquisition_monitor = {}
        self._pending_acquisition_start_doc = {}
        self._acquisition_active = False
        self._acquisition_seen_active = False
        self._exposure_started_monotonic = None
        self._acquire_time_total = 0.0
        self._time_remaining_value = 0.0
        self._elapsed_time_value = 0.0
        self._exposure_timer = QtCore.QTimer(self)
        self._exposure_timer.setInterval(100)
        self._exposure_timer.timeout.connect(self._on_exposure_timer)
        self._acquisition_document_received.connect(
            self._handle_acquisition_document,
            QtCore.Qt.QueuedConnection,
        )

        app = MITRApplication.instance()
        re_client = app.re_client

        support = _get_diffraction_plot_support()
        if support.get("controller_ok"):
            classes = _load_diffraction_plot_classes()
            # PyQtGraph is the primary backend: Matplotlib could not keep up
            # with live PSD updates and remains only as a compatibility fallback.
            plot_class = classes["pyqtgraph_plot"] or classes["matplotlib_plot"]
            plot_widget = plot_class()
            viewer = classes["viewer"](plot_widget)
            self._diffraction_live_plot = classes["controller"](viewer, re_client=re_client)
            self._document_subscription = app.document_dispatcher.subscribe(
                self._diffraction_live_plot.on_document
            )
        else:
            viewer = _DiffractionUnavailableWidget(support.get("message", ""))
            self._diffraction_live_plot = None

        self._acquisition_document_subscription = app.document_dispatcher.subscribe(
            self._queue_acquisition_document
        )
        self._setup_acquire_indicator()
        self._setup_time_remaining_progress()

        self.ui.Data_Viewer.layout().addWidget(viewer)

    def _set_manual_channels_connected(self, connected):
        connected = bool(connected)
        if connected == self._manual_channels_connected:
            return
        for channel in (
            self._acquire_channel,
            self._time_remaining_channel,
            self._acquire_time_channel,
            self._elapsed_time_channel,
        ):
            if channel is None:
                continue
            try:
                channel.connect() if connected else channel.disconnect()
            except Exception:
                pass
        self._manual_channels_connected = connected

    def deactivate_display(self):
        # Keep the document subscriptions while this cached display is hidden.
        # Runs may start and finish while the operator is on Run Extra; the
        # controller consumes those documents without keeping direct CA
        # subscriptions active, so the plot is complete when Viewer returns.
        controller = getattr(self, "_diffraction_live_plot", None)
        if controller is not None:
            controller.deactivate()
        self._set_manual_channels_connected(False)
        self._exposure_timer.stop()

    def activate_display(self):
        from application import MITRApplication

        controller = getattr(self, "_diffraction_live_plot", None)
        if controller is not None:
            controller.activate()
            if self._document_subscription is None:
                app = MITRApplication.instance()
                self._document_subscription = app.document_dispatcher.subscribe(
                    controller.on_document
                )
        if self._acquisition_document_subscription is None:
            app = MITRApplication.instance()
            self._acquisition_document_subscription = app.document_dispatcher.subscribe(
                self._queue_acquisition_document
            )
        pending_start = dict(getattr(self, "_pending_acquisition_start_doc", {}) or {})
        if pending_start and self._acquisition_run_uid:
            self._configure_acquisition_monitor(pending_start)
        self._set_manual_channels_connected(True)

    def _setup_acquire_indicator(self):
        old_widget = getattr(self.ui, "PyDMByteIndicator", None)
        row_layout = getattr(self.ui, "horizontalLayout_4", None)
        if old_widget is None or row_layout is None:
            return
        index = row_layout.indexOf(old_widget)
        if index < 0:
            return
        try:
            old_widget.channel = ""
        except Exception:
            pass
        row_layout.removeWidget(old_widget)
        old_widget.hide()

        self.acquire_indicator = QtWidgets.QLabel(self.ui)
        self.acquire_indicator.setObjectName("acquireIndicator")
        self.acquire_indicator.setFixedSize(40, 40)
        self.acquire_indicator.setAlignment(QtCore.Qt.AlignCenter)
        row_layout.insertWidget(index, self.acquire_indicator)
        self._set_acquire_indicator(False, "No active run")

    def _setup_time_remaining_progress(self):
        old_widget = getattr(self.ui, "PyDMLabel", None)
        row_layout = getattr(self.ui, "horizontalLayout", None)
        if old_widget is None or row_layout is None:
            return

        idx = row_layout.indexOf(old_widget)
        if idx < 0:
            return

        self.time_remaining_progress = QtWidgets.QProgressBar(self.ui)
        self.time_remaining_progress.setMinimumHeight(34)
        self.time_remaining_progress.setMaximumHeight(34)
        self.time_remaining_progress.setMinimumWidth(220)
        self.time_remaining_progress.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        progress_font = self.time_remaining_progress.font()
        progress_font.setPointSize(12)
        self.time_remaining_progress.setFont(progress_font)
        self.time_remaining_progress.setStyleSheet(
            "QProgressBar {"
            " border: 1px solid rgb(120,120,120);"
            " border-radius: 4px;"
            " background: rgb(235,235,235);"
            " color: rgb(10,10,10);"
            " text-align: center;"
            "}"
            "QProgressBar::chunk {"
            " background-color: rgb(120, 170, 255);"
            "}"
        )
        self.time_remaining_progress.setTextVisible(True)
        self.time_remaining_progress.setAlignment(QtCore.Qt.AlignCenter)
        self.time_remaining_progress.setRange(0, 1000)
        self.time_remaining_progress.setValue(0)
        self.time_remaining_progress.setFormat("No active run")
        self.time_remaining_progress.setEnabled(False)

        row_layout.removeWidget(old_widget)
        try:
            old_widget.channel = ""
        except Exception:
            pass
        old_widget.hide()
        row_layout.insertWidget(idx, self.time_remaining_progress)
        row_layout.setStretch(idx, 2)

    @staticmethod
    def _channel_address(pvname):
        address = str(pvname or "").strip()
        if not address:
            return ""
        if "://" in address:
            return address
        return f"ca://{address}"

    def _disconnect_acquisition_channels(self):
        self._exposure_timer.stop()
        for attribute in (
            "_acquire_channel",
            "_time_remaining_channel",
            "_acquire_time_channel",
            "_elapsed_time_channel",
        ):
            channel = getattr(self, attribute, None)
            if channel is not None:
                try:
                    channel.disconnect()
                except Exception:
                    pass
            setattr(self, attribute, None)
        self._manual_channels_connected = False

    def _make_acquisition_channel(self, pvname, slot):
        address = self._channel_address(pvname)
        if not address:
            return None
        channel = PyDMChannel(address=address, value_slot=slot)
        channel.connect()
        return channel

    def _configure_acquisition_monitor(self, start_doc):
        self._disconnect_acquisition_channels()
        self._acquisition_monitor = _acquisition_monitor_from_start_doc(start_doc)
        self._acquisition_active = False
        self._acquisition_seen_active = False
        self._exposure_started_monotonic = None
        self._elapsed_time_value = 0.0
        self._time_remaining_value = 0.0
        self._acquire_time_total = self._to_float(
            self._acquisition_monitor.get("duration"),
            default=0.0,
        )

        if not self._acquisition_monitor:
            self._set_acquire_indicator(False, "No exposure monitor for this run")
            self._update_time_remaining_progress(state="unavailable")
            return

        self._acquire_channel = self._make_acquisition_channel(
            self._acquisition_monitor.get("active_pv"),
            self._on_acquire_value_changed,
        )
        self._acquire_time_channel = self._make_acquisition_channel(
            self._acquisition_monitor.get("duration_pv"),
            self._on_acquire_time_changed,
        )
        self._time_remaining_channel = self._make_acquisition_channel(
            self._acquisition_monitor.get("remaining_pv"),
            self._on_time_remaining_changed,
        )
        self._elapsed_time_channel = self._make_acquisition_channel(
            self._acquisition_monitor.get("elapsed_pv"),
            self._on_elapsed_time_changed,
        )
        self._manual_channels_connected = True
        device = str(self._acquisition_monitor.get("device", "detector") or "detector")
        self._set_acquire_indicator(False, f"Waiting for {device} exposure")
        self._update_time_remaining_progress(state="waiting")

    def _queue_acquisition_document(self, name, doc):
        self._acquisition_document_received.emit(str(name), dict(doc or {}))

    @QtCore.Slot(str, object)
    def _handle_acquisition_document(self, name, doc):
        name = str(name or "")
        doc = dict(doc or {})
        if name == "start":
            if str(doc.get("experiment_type", "") or "").strip().lower() != "diffraction":
                return
            self._acquisition_run_uid = str(doc.get("uid", "") or "")
            self._pending_acquisition_start_doc = dict(doc)
            if getattr(self, "_navigation_active", True):
                self._configure_acquisition_monitor(doc)
            return
        if name != "stop":
            return
        run_uid = str(doc.get("run_start", "") or "")
        if self._acquisition_run_uid and run_uid and run_uid != self._acquisition_run_uid:
            return
        self._acquisition_run_uid = ""
        self._pending_acquisition_start_doc = {}
        self._disconnect_acquisition_channels()
        self._acquisition_active = False
        self._set_acquire_indicator(False, "No active run")
        self._update_time_remaining_progress(state="stopped")

    @staticmethod
    def _to_bool(value):
        if isinstance(value, str):
            return value.strip().lower() in {
                "1", "true", "on", "yes", "acquire", "acquiring", "counting"
            }
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return bool(int(value))
        except Exception:
            return bool(value)

    def _set_acquire_indicator(self, acquiring, tooltip=""):
        indicator = getattr(self, "acquire_indicator", None)
        if indicator is None:
            return
        color = "rgb(225, 45, 45)" if acquiring else "rgb(145, 145, 145)"
        indicator.setStyleSheet(
            "QLabel#acquireIndicator {"
            f" background-color: {color};"
            " border: 2px solid rgb(80, 80, 80);"
            " border-radius: 4px;"
            "}"
        )
        indicator.setToolTip(str(tooltip or ("Acquiring" if acquiring else "Idle")))

    def _on_acquire_value_changed(self, value):
        acquiring = self._to_bool(value)
        was_active = bool(self._acquisition_active)
        self._acquisition_active = acquiring
        device = str(self._acquisition_monitor.get("device", "detector") or "detector")
        if acquiring:
            if not was_active:
                self._acquisition_seen_active = True
                self._exposure_started_monotonic = time.monotonic()
                self._elapsed_time_value = 0.0
                self._time_remaining_value = max(0.0, self._acquire_time_total)
            self._set_acquire_indicator(True, f"{device} is acquiring")
            if not (
                self._acquisition_monitor.get("remaining_pv")
                or self._acquisition_monitor.get("elapsed_pv")
            ):
                self._exposure_timer.start()
            self._update_time_remaining_progress(state="acquiring")
            return

        self._exposure_timer.stop()
        self._set_acquire_indicator(False, f"{device} is idle")
        if was_active:
            self._time_remaining_value = 0.0
            self._elapsed_time_value = max(
                self._elapsed_time_value,
                self._acquire_time_total,
            )
        state = "complete" if self._acquisition_seen_active else "waiting"
        self._update_time_remaining_progress(state=state)

    def _on_time_remaining_changed(self, value):
        self._time_remaining_value = self._to_float(value, default=0.0)
        self._update_time_remaining_progress(
            state="acquiring" if self._acquisition_active else None
        )

    def _on_acquire_time_changed(self, value):
        self._acquire_time_total = self._to_float(value, default=0.0)
        self._update_time_remaining_progress(
            state="acquiring" if self._acquisition_active else None
        )

    def _on_elapsed_time_changed(self, value):
        self._elapsed_time_value = max(0.0, self._to_float(value, default=0.0))
        self._time_remaining_value = max(
            0.0,
            self._acquire_time_total - self._elapsed_time_value,
        )
        self._update_time_remaining_progress(
            state="acquiring" if self._acquisition_active else None
        )

    def _on_exposure_timer(self):
        if not self._acquisition_active or self._exposure_started_monotonic is None:
            return
        self._elapsed_time_value = max(
            0.0,
            time.monotonic() - self._exposure_started_monotonic,
        )
        self._time_remaining_value = max(
            0.0,
            self._acquire_time_total - self._elapsed_time_value,
        )
        self._update_time_remaining_progress(state="acquiring")

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return float(value)
        except Exception:
            return float(default)

    def _update_time_remaining_progress(self, *, state=None):
        if not hasattr(self, "time_remaining_progress"):
            return

        remaining = max(0.0, self._time_remaining_value)
        total = max(0.0, self._acquire_time_total)

        self.time_remaining_progress.setRange(0, 1000)
        if state == "unavailable":
            self.time_remaining_progress.setEnabled(False)
            self.time_remaining_progress.setValue(0)
            self.time_remaining_progress.setFormat("No exposure monitor")
            return
        if state == "stopped":
            self.time_remaining_progress.setEnabled(False)
            self.time_remaining_progress.setValue(0)
            self.time_remaining_progress.setFormat("No active run")
            return
        self.time_remaining_progress.setEnabled(True)
        if state == "waiting" or (
            state is None and not self._acquisition_active and not self._acquisition_seen_active
        ):
            self.time_remaining_progress.setValue(0)
            self.time_remaining_progress.setFormat("Waiting for exposure")
            return
        if state == "complete" or (
            state is None and not self._acquisition_active and self._acquisition_seen_active
        ):
            self.time_remaining_progress.setValue(1000)
            self.time_remaining_progress.setFormat("Exposure complete")
            return
        if total > 0:
            frac_done = 1.0 - min(1.0, remaining / total)
            self.time_remaining_progress.setValue(int(round(frac_done * 1000.0)))
        else:
            self.time_remaining_progress.setValue(0)

        self.time_remaining_progress.setFormat(f"{remaining:.1f} s remaining")
