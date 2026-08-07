import subprocess
import sys
from pathlib import Path

# from pydm.display import Display
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel)
from pydm.widgets.channel import PyDMChannel
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

# from bluesky_widgets.qt.figures import QtFigure, QtFigures
# from bluesky_widgets.models.auto_plot_builders import AutoLines, AutoPlotter, AutoImages
# from bluesky_widgets.models.plot_builders import Lines, Images
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.utils import install_remote_qt_kicker

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
    re_dispatcher: RemoteDispatcher
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
        from bluesky_widgets.utils.streaming import stream_documents_into_runs

        self._time_remaining_channel = None
        self._acquire_time_channel = None
        self._acquire_time_total = 0.0
        self._time_remaining_value = 0.0

        app = MITRApplication.instance()
        re_client = app.re_client

        # re_queue = QtRePlanQueue(re_client)
        # re_plan_editor = RePlanEditorWidget(re_client)
        # self.ui.RE_Queue.layout().addWidget(re_queue)
        # self.ui.RE_Plan_Editor.layout().addWidget(re_plan_editor)

        # figModel = Lines('motor',['det1','det2'],max_runs=3)
        # figModel = AutoLines(max_runs=1)
        # figModel = Lines('he3psd0_position_x[0]',['he3psd0_counts[0]'],max_runs=1)

        # viewer = QtFigures(figModel.figures)
        # self.runs = []
        # app.re_dispatcher.subscribe(stream_documents_into_runs(figModel.add_run))
        # app.re_dispatcher.subscribe(print)

        #Viewer for imaging data
        # figModel = Images("tiff1")
        # viewer = QtFigure(figModel.figure)
        # app.re_dispatcher.subscribe(stream_documents_into_runs(figModel.add_run))

        support = _get_diffraction_plot_support()
        if support.get("controller_ok"):
            classes = _load_diffraction_plot_classes()
            plot_class = classes["pyqtgraph_plot"] or classes["matplotlib_plot"]
            plot_widget = plot_class()
            viewer = classes["viewer"](plot_widget)
            self._diffraction_live_plot = classes["controller"](viewer, re_client=re_client)
            app.re_dispatcher.subscribe(self._diffraction_live_plot.on_document)
            if hasattr(app.re_dispatcher, "_waiting_for_start"):
                # The Qt RemoteDispatcher drops all mid-run documents until it
                # sees a 'start' doc. For diffraction we explicitly want to
                # support attaching during a run, so allow later event/event_page
                # docs through immediately.
                app.re_dispatcher._waiting_for_start = False
        else:
            viewer = _DiffractionUnavailableWidget(support.get("message", ""))
            self._diffraction_live_plot = None

        self._setup_time_remaining_progress()

        app.re_dispatcher.start()
        # install_remote_qt_kicker()

        # re_console = QtReConsoleMonitor(re_client)
        # re_queue_history = QtRePlanHistory(re_client)
        # self.ui.RE_Console.layout().addWidget(re_console)
        # self.ui.RE_Queue_History.layout().addWidget(re_queue_history)

        self.ui.Data_Viewer.layout().addWidget(viewer)

        # self.ui.pushButton.clicked.connect(self.printstuff)

        # self.ui.psdDisplay.addChannel(y_channel=f"ca://{self.macros['P']}det1:LiveCountsD0",color='black',symbolSize=5,symbol='o')

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
