import sys
import numpy as np

# from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy import QtWidgets
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanEditor,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)

# from bluesky_widgets.qt.figures import QtFigure, QtFigures
# from bluesky_widgets.models.auto_plot_builders import AutoLines, AutoPlotter, AutoImages
from bluesky_widgets.models.plot_builders import Lines, Images
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky.utils import install_remote_qt_kicker

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display


class MainScreen(display.MITRDisplay):
    re_dispatcher: RemoteDispatcher
    re_client: RunEngineClient

    def __init__(self, parent=None, args=None, macros=None, ui_filename='imaging_viewer.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("MainScreen here")

    def ui_filename(self):
        return 'imaging_viewer.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        # Robust normalization: clip dark and bright outliers so hot pixels/gamma spots
        # do not dominate the displayed intensity range.
        self._norm_low_percentile = 1.0
        self._norm_high_percentile = 99.7
        self._low_slider_min_pct = 0.0
        self._low_slider_max_pct = 5.0
        self._high_slider_min_pct = 95.0
        self._high_slider_max_pct = 100.0
        self._gamma_value = 1.0
        self._last_image = None
        self._manual_levels_initialized = False
        self._startup_autoscale_attempts = 0
        self._startup_autoscale_max_attempts = 30
        self._level_slider_low_bound = 0.0
        self._level_slider_high_bound = 1.0

        image_view = self.ui.cameraImage
        self._install_display_controls(image_view)

        # Disable built-in full-range normalization; we set levels manually.
        if hasattr(image_view, "setNormalizeData"):
            image_view.setNormalizeData(False)
        else:
            image_view.setProperty("normalizeData", False)

        if hasattr(image_view, "newImageSignal"):
            image_view.newImageSignal.connect(self._apply_robust_normalization)

        # Also listen to the underlying ImageItem change signal to catch
        # first-image timing when PV data/shape channels connect in sequence.
        try:
            image_item = image_view.getImageItem()
            if image_item is not None and hasattr(image_item, "sigImageChanged"):
                image_item.sigImageChanged.connect(self._apply_robust_normalization)
        except Exception:
            pass

        # Startup retry loop: keep attempting autoscale until first valid image appears.
        self._startup_autoscale_timer = QtCore.QTimer(self)
        self._startup_autoscale_timer.setInterval(200)
        self._startup_autoscale_timer.timeout.connect(self._startup_autoscale_tick)
        self._startup_autoscale_timer.start()

    def _install_display_controls(self, image_view):
        main_layout = self.ui.verticalLayout
        controls = QtWidgets.QGroupBox("Image Settings", self.ui)
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(10, 6, 10, 6)
        controls_layout.setSpacing(12)

        self.auto_levels_checkbox = QtWidgets.QCheckBox("Auto levels")
        self.auto_levels_checkbox.setChecked(True)
        self.auto_levels_checkbox.toggled.connect(self._on_auto_levels_toggled)
        controls_layout.addWidget(self.auto_levels_checkbox)

        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setRange(-1e12, 1e12)
        self.min_spinbox.setSingleStep(1.0)
        self.min_spinbox.setMaximumWidth(140)
        self.min_spinbox.valueChanged.connect(self._on_manual_levels_changed)
        self.min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_slider.setRange(0, 1000)
        self.min_slider.valueChanged.connect(self._on_min_slider_changed)
        self.min_spinbox.valueChanged.connect(self._on_min_spinbox_changed)
        self.min_label = QtWidgets.QLabel("Min")
        min_block = QtWidgets.QWidget()
        min_layout = QtWidgets.QVBoxLayout(min_block)
        min_layout.setContentsMargins(0, 0, 0, 0)
        min_layout.setSpacing(2)
        min_top = QtWidgets.QHBoxLayout()
        min_top.setContentsMargins(0, 0, 0, 0)
        min_top.setSpacing(4)
        min_top.addWidget(self.min_label)
        min_top.addWidget(self.min_spinbox)
        min_layout.addLayout(min_top)
        min_layout.addWidget(self.min_slider)
        controls_layout.addWidget(min_block)

        self.max_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_spinbox.setDecimals(3)
        self.max_spinbox.setRange(-1e12, 1e12)
        self.max_spinbox.setSingleStep(1.0)
        self.max_spinbox.setMaximumWidth(140)
        self.max_spinbox.valueChanged.connect(self._on_manual_levels_changed)
        self.max_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_slider.setRange(0, 1000)
        self.max_slider.valueChanged.connect(self._on_max_slider_changed)
        self.max_spinbox.valueChanged.connect(self._on_max_spinbox_changed)
        self.max_label = QtWidgets.QLabel("Max")
        max_block = QtWidgets.QWidget()
        max_layout = QtWidgets.QVBoxLayout(max_block)
        max_layout.setContentsMargins(0, 0, 0, 0)
        max_layout.setSpacing(2)
        max_top = QtWidgets.QHBoxLayout()
        max_top.setContentsMargins(0, 0, 0, 0)
        max_top.setSpacing(4)
        max_top.addWidget(self.max_label)
        max_top.addWidget(self.max_spinbox)
        max_layout.addLayout(max_top)
        max_layout.addWidget(self.max_slider)
        controls_layout.addWidget(max_block)

        self.low_pct_spinbox = QtWidgets.QDoubleSpinBox()
        self.low_pct_spinbox.setDecimals(2)
        self.low_pct_spinbox.setRange(0.0, 49.9)
        self.low_pct_spinbox.setSingleStep(0.1)
        self.low_pct_spinbox.setValue(self._norm_low_percentile)
        self.low_pct_spinbox.valueChanged.connect(self._on_percentiles_changed)
        self.low_pct_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.low_pct_slider.setRange(
            int(round(self._low_slider_min_pct * 10.0)),
            int(round(self._low_slider_max_pct * 10.0)),
        )
        self.low_pct_slider.setValue(int(round(self._norm_low_percentile * 10.0)))
        self.low_pct_slider.valueChanged.connect(self._on_low_pct_slider_changed)
        self.low_pct_spinbox.valueChanged.connect(self._on_low_pct_spinbox_changed)
        self.low_pct_label = QtWidgets.QLabel("Low %")
        low_pct_block = QtWidgets.QWidget()
        low_pct_layout = QtWidgets.QVBoxLayout(low_pct_block)
        low_pct_layout.setContentsMargins(0, 0, 0, 0)
        low_pct_layout.setSpacing(2)
        low_pct_top = QtWidgets.QHBoxLayout()
        low_pct_top.setContentsMargins(0, 0, 0, 0)
        low_pct_top.setSpacing(4)
        low_pct_top.addWidget(self.low_pct_label)
        low_pct_top.addWidget(self.low_pct_spinbox)
        low_pct_layout.addLayout(low_pct_top)
        low_pct_layout.addWidget(self.low_pct_slider)
        controls_layout.addWidget(low_pct_block)

        self.high_pct_spinbox = QtWidgets.QDoubleSpinBox()
        self.high_pct_spinbox.setDecimals(2)
        self.high_pct_spinbox.setRange(50.1, 100.0)
        self.high_pct_spinbox.setSingleStep(0.1)
        self.high_pct_spinbox.setValue(self._norm_high_percentile)
        self.high_pct_spinbox.valueChanged.connect(self._on_percentiles_changed)
        self.high_pct_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.high_pct_slider.setRange(
            int(round(self._high_slider_min_pct * 10.0)),
            int(round(self._high_slider_max_pct * 10.0)),
        )
        self.high_pct_slider.setValue(int(round(self._norm_high_percentile * 10.0)))
        self.high_pct_slider.valueChanged.connect(self._on_high_pct_slider_changed)
        self.high_pct_spinbox.valueChanged.connect(self._on_high_pct_spinbox_changed)
        self.high_pct_label = QtWidgets.QLabel("High %")
        high_pct_block = QtWidgets.QWidget()
        high_pct_layout = QtWidgets.QVBoxLayout(high_pct_block)
        high_pct_layout.setContentsMargins(0, 0, 0, 0)
        high_pct_layout.setSpacing(2)
        high_pct_top = QtWidgets.QHBoxLayout()
        high_pct_top.setContentsMargins(0, 0, 0, 0)
        high_pct_top.setSpacing(4)
        high_pct_top.addWidget(self.high_pct_label)
        high_pct_top.addWidget(self.high_pct_spinbox)
        high_pct_layout.addLayout(high_pct_top)
        high_pct_layout.addWidget(self.high_pct_slider)
        controls_layout.addWidget(high_pct_block)

        self.gamma_spinbox = QtWidgets.QDoubleSpinBox()
        self.gamma_spinbox.setDecimals(2)
        self.gamma_spinbox.setRange(0.10, 5.00)
        self.gamma_spinbox.setSingleStep(0.05)
        self.gamma_spinbox.setValue(self._gamma_value)
        self.gamma_spinbox.valueChanged.connect(self._on_gamma_changed)
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(10, 500)
        self.gamma_slider.setValue(int(round(self._gamma_value * 100.0)))
        self.gamma_slider.valueChanged.connect(self._on_gamma_slider_changed)
        self.gamma_spinbox.valueChanged.connect(self._on_gamma_spinbox_changed)
        self.gamma_label = QtWidgets.QLabel("Gamma")
        gamma_block = QtWidgets.QWidget()
        gamma_layout = QtWidgets.QVBoxLayout(gamma_block)
        gamma_layout.setContentsMargins(0, 0, 0, 0)
        gamma_layout.setSpacing(2)
        gamma_top = QtWidgets.QHBoxLayout()
        gamma_top.setContentsMargins(0, 0, 0, 0)
        gamma_top.setSpacing(4)
        gamma_top.addWidget(self.gamma_label)
        gamma_top.addWidget(self.gamma_spinbox)
        gamma_layout.addLayout(gamma_top)
        gamma_layout.addWidget(self.gamma_slider)
        controls_layout.addWidget(gamma_block)

        self.colormap_combo = QtWidgets.QComboBox()
        self._colormap_options = self._discover_colormap_options(image_view)
        for label, _ in self._colormap_options:
            self.colormap_combo.addItem(label)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        self.colormap_combo.setMinimumWidth(140)
        controls_layout.addWidget(QtWidgets.QLabel("Color map"))
        controls_layout.addWidget(self.colormap_combo)
        current_cmap = image_view.property("colorMap")
        for i, (_, value) in enumerate(self._colormap_options):
            if value == current_cmap:
                self.colormap_combo.setCurrentIndex(i)
                break

        controls_layout.addStretch(1)
        controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        main_layout.insertWidget(1, controls)

        self._update_level_slider_bounds(0.0, 1.0)
        self._sync_level_sliders_from_spinboxes()
        self._on_auto_levels_toggled(True)
        self._apply_gamma_setting()

    def _discover_colormap_options(self, image_view):
        options = []

        meta = image_view.metaObject()
        prop_index = meta.indexOfProperty("colorMap")
        if prop_index >= 0:
            prop = meta.property(prop_index)
            if prop.isEnumType():
                enum = prop.enumerator()
                for i in range(enum.keyCount()):
                    key = enum.key(i)
                    value = enum.value(i)
                    options.append((key, value))

        if not options:
            # Keep a safe fallback for environments where enum metadata is unavailable.
            options.append(("Monochrome", image_view.property("colorMap")))
        return options

    def _set_image_levels(self, low, high):
        image_view = self.ui.cameraImage
        if high <= low:
            high = low + 1.0

        self._update_level_slider_bounds(low, high)
        self._sync_level_sliders_from_spinboxes()

        try:
            image_item = image_view.getImageItem()
            if image_item is not None and hasattr(image_item, "setLevels"):
                image_item.setLevels([float(low), float(high)])
        except Exception:
            pass

        if hasattr(image_view, "setColorMapMin"):
            image_view.setColorMapMin(float(low))
        else:
            image_view.setProperty("colorMapMin", float(low))

        if hasattr(image_view, "setColorMapMax"):
            image_view.setColorMapMax(float(high))
        else:
            image_view.setProperty("colorMapMax", float(high))

    def _on_auto_levels_toggled(self, enabled):
        self.min_spinbox.setEnabled(not enabled)
        self.max_spinbox.setEnabled(not enabled)
        self.min_label.setEnabled(not enabled)
        self.max_label.setEnabled(not enabled)
        self.min_slider.setEnabled(not enabled)
        self.max_slider.setEnabled(not enabled)
        self.low_pct_label.setEnabled(enabled)
        self.high_pct_label.setEnabled(enabled)
        self.low_pct_spinbox.setEnabled(enabled)
        self.high_pct_spinbox.setEnabled(enabled)
        self.low_pct_slider.setEnabled(enabled)
        self.high_pct_slider.setEnabled(enabled)
        if enabled:
            self._auto_levels_from_current_image()
        else:
            self._on_manual_levels_changed()

    def _on_manual_levels_changed(self, *args):
        if self.auto_levels_checkbox.isChecked():
            return
        low = self.min_spinbox.value()
        high = self.max_spinbox.value()
        self._set_image_levels(low, high)

    def _value_to_level_slider(self, value):
        span = self._level_slider_high_bound - self._level_slider_low_bound
        if span <= 0:
            return 0
        raw = (value - self._level_slider_low_bound) / span
        return int(round(max(0.0, min(1.0, raw)) * 1000.0))

    def _slider_to_level_value(self, slider_value):
        span = self._level_slider_high_bound - self._level_slider_low_bound
        if span <= 0:
            return self._level_slider_low_bound
        frac = slider_value / 1000.0
        return self._level_slider_low_bound + frac * span

    def _update_level_slider_bounds(self, low, high):
        if high <= low:
            high = low + 1.0
        span = high - low
        pad = max(span, abs(low) * 0.1, abs(high) * 0.1, 1.0)
        self._level_slider_low_bound = low - pad
        self._level_slider_high_bound = high + pad
        if self._level_slider_high_bound <= self._level_slider_low_bound:
            self._level_slider_high_bound = self._level_slider_low_bound + 1.0

    def _sync_level_sliders_from_spinboxes(self):
        min_sv = self._value_to_level_slider(self.min_spinbox.value())
        max_sv = self._value_to_level_slider(self.max_spinbox.value())
        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)
        self.min_slider.setValue(min_sv)
        self.max_slider.setValue(max_sv)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)

    def _on_percentiles_changed(self, *args):
        low = self.low_pct_spinbox.value()
        high = self.high_pct_spinbox.value()
        if low >= high:
            return
        self._norm_low_percentile = low
        self._norm_high_percentile = high
        if self.auto_levels_checkbox.isChecked():
            self._auto_levels_from_current_image()

    def _on_low_pct_slider_changed(self, value):
        new_low = value / 10.0
        self.low_pct_spinbox.blockSignals(True)
        self.low_pct_spinbox.setValue(new_low)
        self.low_pct_spinbox.blockSignals(False)
        self._on_percentiles_changed()

    def _on_high_pct_slider_changed(self, value):
        new_high = value / 10.0
        self.high_pct_spinbox.blockSignals(True)
        self.high_pct_spinbox.setValue(new_high)
        self.high_pct_spinbox.blockSignals(False)
        self._on_percentiles_changed()

    def _on_low_pct_spinbox_changed(self, value):
        slider_value = int(round(value * 10.0))
        self.low_pct_slider.blockSignals(True)
        self.low_pct_slider.setValue(slider_value)
        self.low_pct_slider.blockSignals(False)

    def _on_high_pct_spinbox_changed(self, value):
        slider_value = int(round(value * 10.0))
        self.high_pct_slider.blockSignals(True)
        self.high_pct_slider.setValue(slider_value)
        self.high_pct_slider.blockSignals(False)

    def _on_min_slider_changed(self, slider_value):
        new_low = self._slider_to_level_value(slider_value)
        self.min_spinbox.blockSignals(True)
        self.min_spinbox.setValue(new_low)
        self.min_spinbox.blockSignals(False)
        self._on_manual_levels_changed()

    def _on_max_slider_changed(self, slider_value):
        new_high = self._slider_to_level_value(slider_value)
        self.max_spinbox.blockSignals(True)
        self.max_spinbox.setValue(new_high)
        self.max_spinbox.blockSignals(False)
        self._on_manual_levels_changed()

    def _on_min_spinbox_changed(self, value):
        slider_value = self._value_to_level_slider(value)
        self.min_slider.blockSignals(True)
        self.min_slider.setValue(slider_value)
        self.min_slider.blockSignals(False)

    def _on_max_spinbox_changed(self, value):
        slider_value = self._value_to_level_slider(value)
        self.max_slider.blockSignals(True)
        self.max_slider.setValue(slider_value)
        self.max_slider.blockSignals(False)

    def _on_gamma_changed(self, value):
        self._gamma_value = float(value)
        self._apply_gamma_setting()
        self.ui.cameraImage.update()

    def _on_gamma_slider_changed(self, slider_value):
        new_gamma = slider_value / 100.0
        self.gamma_spinbox.blockSignals(True)
        self.gamma_spinbox.setValue(new_gamma)
        self.gamma_spinbox.blockSignals(False)
        self._on_gamma_changed(new_gamma)

    def _on_gamma_spinbox_changed(self, value):
        slider_value = int(round(value * 100.0))
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(slider_value)
        self.gamma_slider.blockSignals(False)

    def _apply_gamma_setting(self):
        image_view = self.ui.cameraImage
        gamma = float(self._gamma_value)
        if hasattr(image_view, "setGamma"):
            try:
                image_view.setGamma(gamma)
                return
            except Exception:
                pass
        image_view.setProperty("gamma", gamma)

        try:
            image_item = image_view.getImageItem()
            if image_item is not None and hasattr(image_item, "setOpts"):
                image_item.setOpts(gamma=gamma)
        except Exception:
            pass

    def _auto_levels_from_current_image(self):
        if self._last_image is None:
            return
        data = np.asarray(self._last_image)
        if data.size == 0:
            return
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return

        low, high = np.percentile(
            finite,
            [self._norm_low_percentile, self._norm_high_percentile],
        )
        if high <= low:
            high = low + 1.0

        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(float(low))
        self.max_spinbox.setValue(float(high))
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

        self._set_image_levels(low, high)

    def _on_colormap_changed(self, index):
        if index < 0 or index >= len(self._colormap_options):
            return
        _, cmap_value = self._colormap_options[index]
        image_view = self.ui.cameraImage
        if hasattr(image_view, "setColorMap"):
            try:
                image_view.setColorMap(cmap_value)
            except Exception:
                image_view.setProperty("colorMap", cmap_value)
        else:
            image_view.setProperty("colorMap", cmap_value)
        image_view.update()

    def _startup_autoscale_tick(self):
        self._startup_autoscale_attempts += 1
        self._apply_robust_normalization()

        if self._last_image is not None:
            self._startup_autoscale_timer.stop()
            return

        if self._startup_autoscale_attempts >= self._startup_autoscale_max_attempts:
            self._startup_autoscale_timer.stop()

    def _apply_robust_normalization(self, *args):
        image = None
        if args:
            candidate = args[0]
            if candidate is not None:
                image = candidate

        if image is None:
            image_view = self.ui.cameraImage
            try:
                image = image_view.getImageItem().image
            except Exception:
                image = getattr(image_view, "image", None)

        if image is None:
            return

        self._last_image = image
        if not self.auto_levels_checkbox.isChecked():
            if not self._manual_levels_initialized:
                self._auto_levels_from_current_image()
                self._manual_levels_initialized = True
            return

        data = np.asarray(image)
        if data.size == 0:
            return

        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return

        low, high = np.percentile(
            finite,
            [self._norm_low_percentile, self._norm_high_percentile],
        )
        if high <= low:
            high = low + 1.0

        self.min_spinbox.blockSignals(True)
        self.max_spinbox.blockSignals(True)
        self.min_spinbox.setValue(float(low))
        self.max_spinbox.setValue(float(high))
        self.min_spinbox.blockSignals(False)
        self.max_spinbox.blockSignals(False)

        self._set_image_levels(low, high)
