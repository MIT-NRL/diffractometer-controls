import subprocess
import sys
from pathlib import Path

# from pydm.display import Display
from qtpy import QtCore, QtGui, QtWidgets
from pydm.widgets.channel import PyDMChannel

# from bluesky_widgets.qt.figures import QtFigure, QtFigures
# from bluesky_widgets.models.auto_plot_builders import AutoLines, AutoPlotter, AutoImages
# from bluesky_widgets.models.plot_builders import Lines, Images
from bluesky_widgets.models.run_engine_client import RunEngineClient
import display


_DIFFRACTION_PLOT_SUPPORT = None


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
        self._manual_channels_connected = False
        self._document_subscription = None
        self._acquire_time_total = 0.0
        self._time_remaining_value = 0.0

        app = MITRApplication.instance()
        re_client = app.re_client

        support = _get_diffraction_plot_support()
        if support.get("controller_ok"):
            classes = _load_diffraction_plot_classes()
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

        self._setup_time_remaining_progress()

        self.ui.Data_Viewer.layout().addWidget(viewer)

    def _set_manual_channels_connected(self, connected):
        connected = bool(connected)
        if connected == self._manual_channels_connected:
            return
        for channel in (self._time_remaining_channel, self._acquire_time_channel):
            if channel is None:
                continue
            try:
                channel.connect() if connected else channel.disconnect()
            except Exception:
                pass
        self._manual_channels_connected = connected

    def deactivate_display(self):
        from application import MITRApplication

        app = MITRApplication.instance()
        if self._document_subscription is not None:
            app.document_dispatcher.unsubscribe(self._document_subscription)
            self._document_subscription = None
        controller = getattr(self, "_diffraction_live_plot", None)
        if controller is not None:
            controller.deactivate()
        self._set_manual_channels_connected(False)

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
        self._set_manual_channels_connected(True)

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
        self.time_remaining_progress.setFormat("0.0 s remaining")

        row_layout.removeWidget(old_widget)
        old_widget.hide()
        row_layout.insertWidget(idx, self.time_remaining_progress)
        row_layout.setStretch(idx, 2)

        remaining_address = getattr(old_widget, "channel", None) or old_widget.property("channel")
        acquire_time_address = self._derive_acquire_time_address(remaining_address)

        if remaining_address:
            self._time_remaining_channel = PyDMChannel(
                address=remaining_address,
                value_slot=self._on_time_remaining_changed,
            )
            self._time_remaining_channel.connect()

        if acquire_time_address:
            self._acquire_time_channel = PyDMChannel(
                address=acquire_time_address,
                value_slot=self._on_acquire_time_changed,
            )
            self._acquire_time_channel.connect()
        self._manual_channels_connected = True

    @staticmethod
    def _derive_acquire_time_address(remaining_address):
        address = str(remaining_address or "").strip()
        if not address:
            return ""
        return address.replace("AcquireTimeRemaining_RBV", "AcquireTime_RBV")

    def _on_time_remaining_changed(self, value):
        self._time_remaining_value = self._to_float(value, default=0.0)
        self._update_time_remaining_progress()

    def _on_acquire_time_changed(self, value):
        self._acquire_time_total = self._to_float(value, default=0.0)
        self._update_time_remaining_progress()

    @staticmethod
    def _to_float(value, default=0.0):
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return float(value)
        except Exception:
            return float(default)

    def _update_time_remaining_progress(self):
        if not hasattr(self, "time_remaining_progress"):
            return

        remaining = max(0.0, self._time_remaining_value)
        total = max(0.0, self._acquire_time_total)

        self.time_remaining_progress.setRange(0, 1000)
        if total > 0:
            frac_done = 1.0 - min(1.0, remaining / total)
            self.time_remaining_progress.setValue(int(round(frac_done * 1000.0)))
        else:
            self.time_remaining_progress.setValue(0)

        self.time_remaining_progress.setFormat(f"{remaining:.1f} s remaining")
