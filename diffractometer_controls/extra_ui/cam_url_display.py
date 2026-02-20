import numpy as np
import pyqtgraph as pg
from pydm import Display
from pydm.utilities.macro import parse_macro_string
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore


class URLCamDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None):
        self._channels = []
        self._size0 = 0
        self._size1 = 0
        self._size2 = 0
        self._color_mode = "MONO"
        self._array_data = None
        self._image_view = None
        self._image_initialized = False
        self._image_channel_template = ""

        super().__init__(parent=parent, args=args, macros=macros)
        self._install_image_widget()
        self._connect_image_channels()

    def ui_filename(self):
        return "cam_url.ui"

    def _install_image_widget(self):
        old_widget = self.ui.cameraImage
        self._image_channel_template = old_widget.property("imageChannel") or ""

        layout = self.ui.verticalLayout_4
        index = layout.indexOf(old_widget)

        image_view = pg.ImageView(view=pg.PlotItem())
        image_view.ui.histogram.hide()
        image_view.ui.roiBtn.hide()
        image_view.ui.menuBtn.hide()
        image_view.view.setAspectLocked(True)
        image_view.view.invertY(True)
        image_view.view.hideAxis("left")
        image_view.view.hideAxis("bottom")
        image_view.view.setMenuEnabled(False)
        image_view.getImageItem().setOpts(axisOrder="row-major")
        image_view.view.scene().sigMouseClicked.connect(self._on_plot_mouse_clicked)

        self._image_view = image_view

        layout.insertWidget(index, image_view)
        layout.removeWidget(old_widget)
        old_widget.hide()
        old_widget.deleteLater()

    def _macro_dict(self):
        macros = self.macros
        if isinstance(macros, dict):
            result = {str(k): str(v) for k, v in macros.items()}
        elif isinstance(macros, str) and macros.strip():
            try:
                parsed = parse_macro_string(macros)
                if isinstance(parsed, dict):
                    result = {str(k): str(v) for k, v in parsed.items()}
                else:
                    result = {}
            except Exception:
                result = {}
            if not result:
                for piece in macros.split(","):
                    if "=" not in piece:
                        continue
                    key, value = piece.split("=", 1)
                    result[key.strip()] = value.strip()
        else:
            result = {}

        # make lookup tolerant to key case
        for key, value in list(result.items()):
            result.setdefault(key.lower(), value)
            result.setdefault(key.upper(), value)
        return result

    def _expand_macros(self, text):
        value = text or ""
        for key, repl in self._macro_dict().items():
            value = value.replace("${" + key + "}", repl)
        return value

    def _normalized_ca(self, pv):
        if not pv:
            return ""
        pv = pv.strip()
        if pv.startswith("ca://"):
            return pv
        if pv.startswith("ca:"):
            return "ca://" + pv[3:]
        return "ca://" + pv

    def _image_base_pv(self):
        channel = self._expand_macros(self._image_channel_template)
        if "${im}" in channel or "${IM}" in channel:
            channel = channel.replace("${im}", "urlimage1:").replace("${IM}", "urlimage1:")
        channel = self._normalized_ca(channel)
        if channel.endswith("ArrayData"):
            return channel[:-len("ArrayData")]

        macros = self._macro_dict()
        prefix = macros.get("P", "")
        image = macros.get("im", "urlimage1:")
        return self._normalized_ca(f"{prefix}{image}")

    def _connect_channel(self, address, slot):
        if not address:
            return
        channel = PyDMChannel(address=address, value_slot=slot)
        channel.connect()
        self._channels.append(channel)

    def _connect_image_channels(self):
        base = self._image_base_pv()
        self._connect_channel(base + "ArrayData", self._on_array_data)
        self._connect_channel(base + "ArraySize0_RBV", self._on_size0)
        self._connect_channel(base + "ArraySize1_RBV", self._on_size1)
        self._connect_channel(base + "ArraySize2_RBV", self._on_size2)
        self._connect_channel(base + "ColorMode_RBV", self._on_color_mode)

    def _to_int(self, value):
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return int(value)
        except Exception:
            return 0

    def _on_size0(self, value):
        self._size0 = self._to_int(value)

    def _on_size1(self, value):
        self._size1 = self._to_int(value)

    def _on_size2(self, value):
        self._size2 = self._to_int(value)

    def _on_color_mode(self, value):
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        self._color_mode = value

    def _on_array_data(self, value):
        self._array_data = value
        self._render_image()

    def _effective_color_mode(self):
        enum_map = {
            0: "MONO",
            1: "BAYER",
            2: "RGB1",
            3: "RGB2",
            4: "RGB3",
            5: "YUV444",
            6: "YUV422",
            7: "YUV421",
        }
        mode = self._color_mode
        if isinstance(mode, (int, np.integer)):
            return enum_map.get(int(mode), "MONO")
        text = str(mode).strip().upper()
        if text.isdigit():
            return enum_map.get(int(text), "MONO")
        return text or "MONO"

    def _reshape_color(self, flat, mode):
        s0, s1, s2 = self._size0, self._size1, self._size2
        if s0 <= 0 or s1 <= 0 or s2 <= 0:
            return None
        expected = s0 * s1 * s2
        if flat.size < expected:
            return None
        flat = flat[:expected]

        mode = mode.upper()
        if mode == "RGB1":
            # [C, X, Y] -> [Y, X, C]
            image = flat.reshape((s2, s1, s0))
        elif mode == "RGB2":
            # [X, C, Y] -> [Y, X, C]
            image = flat.reshape((s2, s1, s0))
        elif mode == "RGB3":
            # [X, Y, C] in NDArray dim-order semantics -> [Y, X, C]
            image = flat.reshape((s2, s1, s0))
        else:
            return None

        if image.ndim != 3 or image.shape[2] not in (3, 4):
            return None

        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        elif np.issubdtype(image.dtype, np.integer) and image.dtype != np.uint8:
            max_val = max(int(np.iinfo(image.dtype).max), 1)
            image = np.clip((image.astype(np.float32) * (255.0 / max_val)), 0.0, 255.0).astype(np.uint8)

        return np.ascontiguousarray(image)

    def _reshape_mono(self, flat):
        if self._size0 > 0 and self._size1 > 0:
            expected = self._size0 * self._size1
            if flat.size >= expected:
                return flat[:expected].reshape((self._size1, self._size0))
        if self._size1 > 0 and self._size2 > 0:
            expected = self._size1 * self._size2
            if flat.size >= expected:
                return flat[:expected].reshape((self._size2, self._size1))
        return None

    def _reshape_frame(self):
        if self._array_data is None:
            return None
        data = np.asarray(self._array_data)
        if data.size == 0:
            return None
        if data.ndim == 2:
            return data

        flat = np.ravel(data)
        mode = self._effective_color_mode()
        if mode.startswith("RGB"):
            image = self._reshape_color(flat, mode)
            if image is not None:
                return image
        if self._size0 in (3, 4) and self._size1 > 0 and self._size2 > 0:
            image = self._reshape_color(flat, "RGB1")
            if image is not None:
                return image
        return self._reshape_mono(flat)

    def _render_image(self):
        if self._image_view is None:
            return
        frame = self._reshape_frame()
        if frame is None:
            return

        auto = not self._image_initialized
        kwargs = dict(autoLevels=auto, autoRange=auto, autoHistogramRange=auto)
        if frame.ndim == 3:
            kwargs["axes"] = {"x": 1, "y": 0, "c": 2}
        elif frame.ndim == 2:
            kwargs["axes"] = {"x": 1, "y": 0}
        self._image_view.setImage(frame, **kwargs)
        if auto:
            self._image_view.view.autoRange(padding=0.0)
        self._image_initialized = True

    def _on_plot_mouse_clicked(self, event):
        if self._image_view is None:
            return
        try:
            if hasattr(event, "double") and event.double():
                if not hasattr(event, "button") or event.button() == QtCore.Qt.LeftButton:
                    self._image_view.view.autoRange(padding=0.0)
                    if hasattr(event, "accept"):
                        event.accept()
        except Exception:
            pass

    def closeEvent(self, event):
        for channel in self._channels:
            try:
                channel.disconnect()
            except Exception:
                pass
        self._channels = []
        super().closeEvent(event)
