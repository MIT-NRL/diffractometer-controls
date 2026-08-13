"""PyDM display for one ADReolink camera.

The image is intentionally monitored with p4p directly.  ADReolink publishes a
JPEG-compressed NTNDArray, while PyDM's normal image widget was designed around
flat waveform arrays.  A depth-one worker decodes only the newest JPEG away
from the Qt GUI thread and presents it with pyqtgraph.
"""

from __future__ import annotations

import io
import threading

import numpy as np
import pyqtgraph as pg
from PIL import Image
from p4p.client.thread import Context
from pydm import Display
from pydm.utilities.macro import parse_macro_string
from pydm.widgets import (
    PyDMEnumComboBox,
    PyDMLabel,
    PyDMLineEdit,
    PyDMPushButton,
    PyDMSlider,
    PyDMSpinbox,
)
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore, QtWidgets


class NTNDArrayPayload:
    __slots__ = ("data", "codec", "dimensions", "unique_id")

    def __init__(
        self,
        data: bytes,
        codec: str,
        dimensions: tuple[int, ...],
        unique_id: int,
    ):
        self.data = data
        self.codec = codec
        self.dimensions = dimensions
        self.unique_id = unique_id


def macro_dict(macros) -> dict[str, str]:
    """Return a case-tolerant PyDM macro dictionary."""
    if isinstance(macros, dict):
        result = {str(key): str(value) for key, value in macros.items()}
    elif isinstance(macros, str) and macros.strip():
        try:
            parsed = parse_macro_string(macros)
            result = (
                {str(key): str(value) for key, value in parsed.items()}
                if isinstance(parsed, dict)
                else {}
            )
        except Exception:
            result = {}
        if not result:
            for piece in macros.split(","):
                if "=" in piece:
                    key, value = piece.split("=", 1)
                    result[key.strip()] = value.strip()
    else:
        result = {}

    for key, value in list(result.items()):
        result.setdefault(key.lower(), value)
        result.setdefault(key.upper(), value)
    return result


def expand_macros(text: str, macros) -> str:
    value = text or ""
    for key, replacement in macro_dict(macros).items():
        value = value.replace("${" + key + "}", replacement)
    return value


def pva_pv_name(address: str) -> str:
    """Convert a PyDM-style PVA address to the PV name expected by p4p."""
    value = (address or "").strip()
    if value.startswith("pva://"):
        return value[6:]
    if value.startswith("pva:"):
        return value[4:].lstrip("/")
    return value


def capability_is_supported(value) -> bool:
    """Normalize numeric and enum-string capability readbacks from Channel Access."""
    if value is None:
        return False
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"supported", "enabled", "yes", "true", "on"}:
            return True
        if normalized in {"unsupported", "disabled", "no", "false", "off", ""}:
            return False
        try:
            return bool(int(normalized, 0))
        except ValueError:
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError, OverflowError):
        return False


def channel_value_slot(method):
    """Register a QObject method for every scalar type emitted by PyDM."""
    for value_type in (int, float, str, bool, object):
        method = QtCore.Slot(value_type)(method)
    return method


def extract_ntndarray_payload(value) -> NTNDArrayPayload:
    """Copy the fields needed by the asynchronous decoder from a p4p Value."""
    raw = value.get("value")
    if raw is None:
        raise ValueError("NTNDArray has no value payload")
    codec = value.get("codec", {})
    codec_name = str(codec.get("name", "") or "").strip().lower()
    dimensions = tuple(
        int(dimension.get("size", 0))
        for dimension in value.get("dimension", [])
    )
    return NTNDArrayPayload(
        data=np.asarray(raw, dtype=np.uint8).tobytes(),
        codec=codec_name,
        dimensions=dimensions,
        unique_id=int(value.get("uniqueId", -1)),
    )


def decode_ntndarray_payload(payload: NTNDArrayPayload) -> np.ndarray:
    """Decode an ADCore JPEG NTNDArray into a contiguous RGB image."""
    if payload.codec not in {"jpeg", "jpg"}:
        raise ValueError(
            f"Expected a JPEG-compressed NTNDArray, received codec "
            f"{payload.codec or '<none>'}"
        )
    with Image.open(io.BytesIO(payload.data)) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB")))


def image_view_options(frame_ndim: int, initialized: bool) -> dict:
    """Return pyqtgraph options which correctly initialize 8-bit levels."""
    auto = not initialized
    options = {
        "autoLevels": auto,
        "autoRange": auto,
        "autoHistogramRange": auto,
    }
    if frame_ndim == 3:
        options["axes"] = {"x": 1, "y": 0, "c": 2}
    elif frame_ndim == 2:
        options["axes"] = {"x": 1, "y": 0}
    return options


class _LatestFrameDecoder(QtCore.QThread):
    frame_ready = QtCore.Signal(object, object)
    decode_error = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._condition = threading.Condition()
        self._pending: NTNDArrayPayload | None = None
        self._stopping = False

    def submit(self, payload: NTNDArrayPayload) -> None:
        with self._condition:
            # Depth one by design: live viewing values freshness over history.
            self._pending = payload
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify()

    def run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                payload = self._pending
                self._pending = None
            try:
                frame = decode_ntndarray_payload(payload)
            except Exception as exc:
                self.decode_error.emit(str(exc))
            else:
                self.frame_ready.emit(frame, payload)


class ReolinkDisplay(Display):
    pva_connection_changed = QtCore.Signal(bool, str)

    def __init__(self, parent=None, args=None, macros=None):
        self._channels = []
        self._monitor = None
        self._pva_context = None
        self._decoder = None
        self._image_view = None
        self._image_initialized = False
        self._closed = False
        self._capability_rows = {}
        self._capability_states = {
            "ptz": False,
            "speed": False,
            "zoom": False,
            "focus": False,
            "autofocus": False,
        }
        self._reported_limits = {
            "zoom": [None, None],
            "focus": [None, None],
        }
        self._integer_sliders_configured = set()
        super().__init__(parent=parent, args=args, macros=macros)

        display_macros = self.macros() if callable(self.macros) else self.macros
        self._camera_macros = macro_dict(display_macros)
        self._prefix = self._camera_macros.get("P", "4dh4:")
        self._record = self._camera_macros.get("R", "Reolink1:")
        self._base = f"{self._prefix}{self._record}"
        image_default = f"{self._base}Pva1:Image"
        self._image_pv = pva_pv_name(
            expand_macros(
                self._camera_macros.get("IMAGE_PV", image_default),
                self._camera_macros,
            )
        )
        self._title = self._camera_macros.get(
            "TITLE", self._record.rstrip(":")
        )

        self._build_display()
        self._connect_capability_channels()
        self.pva_connection_changed.connect(self._on_pva_connection)
        self._start_image_monitor()
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._shutdown)

    def ui_filename(self):
        return "reolink_viewer.ui"

    def _ca(self, suffix: str) -> str:
        return f"ca://{self._base}{suffix}"

    def _build_display(self) -> None:
        root = self.ui.viewerLayout

        heading = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(self._title)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        heading.addWidget(title)
        heading.addStretch(1)
        self._pva_status = QtWidgets.QLabel("PVA: connecting")
        heading.addWidget(self._pva_status)
        root.addLayout(heading)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter, 1)

        image_panel = QtWidgets.QWidget()
        image_layout = QtWidgets.QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self._image_view = pg.ImageView(view=pg.PlotItem())
        self._image_view.ui.histogram.hide()
        self._image_view.ui.roiBtn.hide()
        self._image_view.ui.menuBtn.hide()
        self._image_view.view.setAspectLocked(True)
        self._image_view.view.invertY(True)
        self._image_view.view.hideAxis("left")
        self._image_view.view.hideAxis("bottom")
        self._image_view.view.setMenuEnabled(False)
        self._image_view.getImageItem().setOpts(axisOrder="row-major")
        self._image_view.setStyleSheet("background-color: black;")
        self._image_view.view.scene().sigMouseClicked.connect(
            self._on_plot_mouse_clicked
        )
        image_layout.addWidget(self._image_view, 1)
        self._frame_status = QtWidgets.QLabel(
            f"Waiting for pva://{self._image_pv}"
        )
        image_layout.addWidget(self._frame_status)
        splitter.addWidget(image_panel)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        controls = QtWidgets.QWidget()
        self._controls_layout = QtWidgets.QVBoxLayout(controls)
        self._controls_layout.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(controls)
        splitter.addWidget(scroll)
        splitter.setSizes([820, 360])

        self._add_acquisition_controls()
        self._add_ptz_controls()
        self._add_lens_controls()
        self._add_status_controls()
        self._add_snapshot_controls()
        self._controls_layout.addStretch(1)

    @staticmethod
    def _form_label(text: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return label

    def _label(self, suffix: str) -> PyDMLabel:
        label = PyDMLabel(init_channel=self._ca(suffix))
        label.setMinimumWidth(120)
        return label

    def _button(
        self,
        text: str,
        suffix: str,
        press_value,
        release_value=None,
    ) -> PyDMPushButton:
        button = PyDMPushButton(
            label=text,
            pressValue=press_value,
            releaseValue=release_value,
            init_channel=self._ca(suffix),
        )
        button.writeWhenRelease = release_value is not None
        button.setMinimumHeight(32)
        return button

    def _spinbox(self, suffix: str, precision: int = 0) -> PyDMSpinbox:
        box = PyDMSpinbox(init_channel=self._ca(suffix))
        box.precisionFromPV = False
        box.precision = precision
        box.showUnits = True
        return box

    def _slider(
        self, suffix: str, minimum: int, maximum: int
    ) -> PyDMSlider:
        slider = PyDMSlider(init_channel=self._ca(suffix))
        slider.userMinimum = minimum
        slider.userMaximum = maximum
        slider.userDefinedLimits = True
        slider.showLimitLabels = True
        slider.showValueLabel = True
        slider.setMinimumWidth(220)
        return slider

    def _add_capability_row(
        self,
        form: QtWidgets.QFormLayout,
        capability: str,
        title: str,
        widget: QtWidgets.QWidget,
    ) -> None:
        label = QtWidgets.QLabel(title)
        form.addRow(label, widget)
        self._capability_rows.setdefault(capability, []).extend((label, widget))

    def _add_acquisition_controls(self) -> None:
        group = QtWidgets.QGroupBox("Acquisition")
        layout = QtWidgets.QGridLayout(group)
        layout.addWidget(self._button("Start", "Acquire", 1), 0, 0)
        layout.addWidget(self._button("Stop", "Acquire", 0), 0, 1)
        layout.addWidget(self._form_label("State"), 1, 0)
        layout.addWidget(self._label("DetectorState_RBV"), 1, 1)
        layout.addWidget(self._form_label("Publish period"), 2, 0)
        layout.addWidget(self._spinbox("AcquirePeriod", precision=2), 2, 1)
        self._controls_layout.addWidget(group)

    def _add_ptz_controls(self) -> None:
        self._ptz_group = QtWidgets.QGroupBox("Pan / Tilt")
        outer = QtWidgets.QVBoxLayout(self._ptz_group)
        grid = QtWidgets.QGridLayout()
        directions = (
            ("↖", 5, 0, 0),
            ("↑", 1, 0, 1),
            ("↗", 6, 0, 2),
            ("←", 3, 1, 0),
            ("Stop", 0, 1, 1),
            ("→", 4, 1, 2),
            ("↙", 7, 2, 0),
            ("↓", 2, 2, 1),
            ("↘", 8, 2, 2),
        )
        for text, value, row, column in directions:
            suffix = "PTZStop" if value == 0 else "PTZDirection"
            press = 1 if value == 0 else value
            release = None if value == 0 else 0
            button = self._button(text, suffix, press, release)
            if value != 0:
                button.setToolTip("Press and hold to move; release to stop")
            grid.addWidget(button, row, column)
        outer.addLayout(grid)
        form = QtWidgets.QFormLayout()
        self._speed_control = self._slider("PTZSpeed", 1, 64)
        self._add_capability_row(
            form, "speed", "Speed", self._speed_control
        )
        form.addRow("Deadman", self._spinbox("PTZMoveTimeout", precision=2))
        outer.addLayout(form)
        self._controls_layout.addWidget(self._ptz_group)

    def _add_lens_controls(self) -> None:
        self._lens_group = QtWidgets.QGroupBox("Lens")
        form = QtWidgets.QFormLayout(self._lens_group)
        # Zoom limits are replaced with the camera-reported range after
        # capability discovery.  A nonzero initial span keeps PyDMSlider valid.
        # Keep the initial range deliberately broad.  The writable Zoom PV can
        # deliver its current value before the separate range PVs; a narrow
        # placeholder makes PyDMSlider reject that valid value and disable
        # itself.  The row remains hidden until both real limits arrive.
        self._zoom_control = self._slider("Zoom", -1.0e12, 1.0e12)
        self._add_capability_row(
            form, "zoom", "Zoom", self._zoom_control
        )
        self._add_capability_row(
            form, "zoom", "Zoom readback", self._label("Zoom_RBV")
        )
        self._focus_control = self._spinbox("Focus")
        self._focus_control.userDefinedLimits = True
        self._add_capability_row(
            form, "focus", "Focus", self._focus_control
        )
        self._add_capability_row(
            form, "focus", "Focus readback", self._label("Focus_RBV")
        )
        self._autofocus_control = PyDMEnumComboBox(
            init_channel=self._ca("AutoFocus")
        )
        self._add_capability_row(
            form, "autofocus", "Autofocus", self._autofocus_control
        )
        self._refocus_control = self._button("Refocus", "Refocus", 1)
        form.addRow(self._refocus_control)
        self._capability_rows.setdefault("refocus", []).append(
            self._refocus_control
        )
        self._controls_layout.addWidget(self._lens_group)
        self._refresh_capability_visibility()

    def _add_status_controls(self) -> None:
        group = QtWidgets.QGroupBox("Status")
        form = QtWidgets.QFormLayout(group)
        rows = (
            ("RTSP", "RTSPConnected_RBV"),
            ("Control", "ControlConnected_RBV"),
            ("Source rate", "MeasuredFPS_RBV"),
            ("Published rate", "PublishedFPS_RBV"),
            ("Frame age", "FrameAge_RBV"),
            ("Dropped frames", "DroppedFrames_RBV"),
            ("Reconnects", "ReconnectCount_RBV"),
            ("Capabilities", "CapabilityStatus_RBV"),
            ("Last error", "LastError_RBV"),
        )
        for title, suffix in rows:
            label = self._label(suffix)
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            form.addRow(title, label)
        self._controls_layout.addWidget(group)

    def _add_snapshot_controls(self) -> None:
        group = QtWidgets.QGroupBox("JPEG Snapshot")
        form = QtWidgets.QFormLayout(group)
        file_path = PyDMLineEdit(init_channel=self._ca("JPEG1:FilePath"))
        file_name = PyDMLineEdit(init_channel=self._ca("JPEG1:FileName"))
        form.addRow("Path", file_path)
        form.addRow("Name", file_name)
        form.addRow("Number", self._spinbox("JPEG1:FileNumber"))
        write = self._button("Save next frame", "JPEG1:WriteFile", 1)
        form.addRow(write)
        form.addRow("Result", self._label("JPEG1:WriteStatus"))
        full_name = self._label("JPEG1:FullFileName_RBV")
        full_name.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        form.addRow("Last file", full_name)
        self._controls_layout.addWidget(group)

    def _connect_channel(self, suffix: str, slot) -> None:
        channel = PyDMChannel(address=self._ca(suffix), value_slot=slot)
        channel.connect()
        self._channels.append(channel)

    def _connect_capability_channels(self) -> None:
        channels = (
            ("PTZSupported_RBV", self._on_ptz_supported),
            ("PTZSpeedSupported_RBV", self._on_speed_supported),
            ("ZoomSupported_RBV", self._on_zoom_supported),
            ("FocusSupported_RBV", self._on_focus_supported),
            ("AutoFocusSupported_RBV", self._on_autofocus_supported),
            ("ZoomMin_RBV", self._on_zoom_minimum),
            ("ZoomMax_RBV", self._on_zoom_maximum),
            ("FocusMin_RBV", self._on_focus_minimum),
            ("FocusMax_RBV", self._on_focus_maximum),
        )
        for suffix, slot in channels:
            self._connect_channel(suffix, slot)

    @channel_value_slot
    def _on_ptz_supported(self, value) -> None:
        self._set_capability("ptz", value)

    @channel_value_slot
    def _on_speed_supported(self, value) -> None:
        self._set_capability("speed", value)

    @channel_value_slot
    def _on_zoom_supported(self, value) -> None:
        self._set_capability("zoom", value)

    @channel_value_slot
    def _on_focus_supported(self, value) -> None:
        self._set_capability("focus", value)

    @channel_value_slot
    def _on_autofocus_supported(self, value) -> None:
        self._set_capability("autofocus", value)

    @channel_value_slot
    def _on_zoom_minimum(self, value) -> None:
        self._set_reported_limit("zoom", value, minimum=True)

    @channel_value_slot
    def _on_zoom_maximum(self, value) -> None:
        self._set_reported_limit("zoom", value, minimum=False)

    @channel_value_slot
    def _on_focus_minimum(self, value) -> None:
        self._set_reported_limit("focus", value, minimum=True)

    @channel_value_slot
    def _on_focus_maximum(self, value) -> None:
        self._set_reported_limit("focus", value, minimum=False)

    def _set_capability(self, name: str, value) -> None:
        self._capability_states[name] = capability_is_supported(value)
        if name in self._reported_limits:
            self._apply_reported_limits(name)
        else:
            self._refresh_capability_visibility()
        if name == "speed" and self._capability_states[name]:
            self._configure_integer_slider(name, self._speed_control)

    def _refresh_capability_visibility(self) -> None:
        if hasattr(self, "_ptz_group"):
            self._ptz_group.setVisible(self._capability_states["ptz"])
        for name, widgets in self._capability_rows.items():
            if name == "refocus":
                visible = (
                    self._capability_states["zoom"]
                    and self._capability_states["autofocus"]
                )
            else:
                visible = self._capability_states[name]
            if name in self._reported_limits:
                lower, upper = self._reported_limits[name]
                visible = (
                    visible
                    and lower is not None
                    and upper is not None
                    and lower < upper
                )
            for widget in widgets:
                widget.setVisible(visible)
        if hasattr(self, "_lens_group"):
            self._lens_group.setVisible(
                any(
                    self._capability_states[name]
                    for name in ("zoom", "focus", "autofocus")
                )
            )

    def _set_reported_limit(
        self, name: str, value, minimum: bool
    ) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            return
        self._reported_limits[name][0 if minimum else 1] = numeric
        self._apply_reported_limits(name)

    def _apply_reported_limits(self, name: str) -> None:
        lower, upper = self._reported_limits[name]
        if (
            not self._capability_states[name]
            or lower is None
            or upper is None
            or lower >= upper
        ):
            self._refresh_capability_visibility()
            return
        widget = {
            "zoom": self._zoom_control,
            "focus": self._focus_control,
        }[name]
        widget.userMinimum = lower
        widget.userMaximum = upper
        if name == "zoom":
            self._configure_integer_slider(name, widget)
        self._refresh_capability_visibility()

    def _configure_integer_slider(
        self, name: str, slider: PyDMSlider
    ) -> None:
        """Apply a unit step after PyDM has a value and usable limits."""
        if self._closed or name in self._integer_sliders_configured:
            return
        value = slider.value
        minimum = slider.minimum
        maximum = slider.maximum
        if (
            value is None
            or minimum is None
            or maximum is None
            or value < minimum
            or value > maximum
        ):
            QtCore.QTimer.singleShot(
                100,
                lambda: self._configure_integer_slider(name, slider),
            )
            return
        slider.step_size = 1
        if slider.step_size == 1:
            self._integer_sliders_configured.add(name)
        else:
            QtCore.QTimer.singleShot(
                100,
                lambda: self._configure_integer_slider(name, slider),
            )

    def _start_image_monitor(self) -> None:
        self._decoder = _LatestFrameDecoder(self)
        self._decoder.frame_ready.connect(self._on_frame)
        self._decoder.decode_error.connect(self._on_decode_error)
        self._decoder.start()
        try:
            self._pva_context = Context("pva", nt=False)
            self._monitor = self._pva_context.monitor(
                self._image_pv,
                self._on_pva_value,
                notify_disconnect=True,
            )
        except Exception as exc:
            self.pva_connection_changed.emit(False, str(exc))

    def _on_pva_value(self, value) -> None:
        if self._closed:
            return
        if isinstance(value, Exception):
            self.pva_connection_changed.emit(False, str(value))
            return
        try:
            payload = extract_ntndarray_payload(value)
        except Exception as exc:
            self.pva_connection_changed.emit(False, str(exc))
            return
        self.pva_connection_changed.emit(True, "")
        if self._decoder is not None:
            self._decoder.submit(payload)

    @QtCore.Slot(bool, str)
    def _on_pva_connection(self, connected: bool, detail: str) -> None:
        if connected:
            self._pva_status.setText("PVA: connected")
            self._pva_status.setStyleSheet("color: rgb(0, 145, 0);")
        else:
            self._pva_status.setText("PVA: disconnected")
            self._pva_status.setToolTip(detail)
            self._pva_status.setStyleSheet("color: rgb(190, 40, 40);")

    @QtCore.Slot(object, object)
    def _on_frame(self, frame: np.ndarray, payload: NTNDArrayPayload) -> None:
        if self._image_view is None:
            return
        auto = not self._image_initialized
        self._image_view.setImage(
            frame,
            **image_view_options(frame.ndim, self._image_initialized),
        )
        if auto:
            self._image_view.view.autoRange(padding=0.0)
        self._image_initialized = True
        height, width = frame.shape[:2]
        self._frame_status.setText(
            f"{width} × {height} RGB | {payload.codec.upper()} | "
            f"frame {payload.unique_id}"
        )

    @QtCore.Slot(str)
    def _on_decode_error(self, message: str) -> None:
        self._frame_status.setText(f"Image decode error: {message}")

    def _on_plot_mouse_clicked(self, event) -> None:
        try:
            if event.double() and event.button() == QtCore.Qt.LeftButton:
                self._image_view.view.autoRange(padding=0.0)
                event.accept()
        except Exception:
            pass

    @QtCore.Slot()
    def _shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._monitor is not None:
            self._monitor.close()
            self._monitor = None
        if self._pva_context is not None:
            self._pva_context.close()
            self._pva_context = None
        if self._decoder is not None:
            self._decoder.stop()
            self._decoder.wait(2000)
            self._decoder = None
        for channel in self._channels:
            try:
                channel.disconnect()
            except Exception:
                pass
        self._channels = []

    def closeEvent(self, event) -> None:
        self._shutdown()
        super().closeEvent(event)
