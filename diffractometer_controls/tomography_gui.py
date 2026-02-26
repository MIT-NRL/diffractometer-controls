import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
from pydm.widgets.channel import PyDMChannel

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

try:
    import pyqtgraph as pg
except Exception:
    pg = None
try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None
try:
    from scipy import ndimage as scipy_ndimage
except Exception:
    scipy_ndimage = None


class MainScreen(display.MITRDisplay):
    re_dispatcher: RemoteDispatcher
    re_client: RunEngineClient
    profile_compute_ready = QtCore.Signal(object)
    profile_worker_idle = QtCore.Signal()

    def __init__(self, parent=None, args=None, macros=None, ui_filename='tomography_gui.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("MainScreen here")

    def ui_filename(self):
        return 'tomography_gui.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        # Robust normalization: clip dark and bright outliers so hot pixels/gamma spots
        # do not dominate the displayed intensity range.
        self._default_norm_low_percentile = 1.0
        self._default_norm_high_percentile = 99.7
        self._norm_low_percentile = self._default_norm_low_percentile
        self._norm_high_percentile = self._default_norm_high_percentile
        self._low_slider_min_pct = 0.0
        self._low_slider_max_pct = 5.0
        self._high_slider_min_pct = 95.0
        self._high_slider_max_pct = 100.0
        self._default_gamma_value = 1.0
        self._gamma_value = self._default_gamma_value
        self._last_image = None
        self._manual_levels_initialized = False
        self._startup_autoscale_attempts = 0
        self._startup_autoscale_max_attempts = 30
        self._level_slider_low_bound = 0.0
        self._level_slider_high_bound = 65535.0
        self._base_lut = None
        self._acquire_channel = None
        self._time_remaining_channel = None
        self._acquire_time_channel = None
        self._acquire_time_total = 0.0
        self._time_remaining_value = 0.0
        self._histogram_curve = None
        self._histogram_plot_item = None
        self._histogram_low_line = None
        self._histogram_high_line = None
        self._histogram_update_pending = False
        self._histogram_max_samples = 200000
        self._histogram_bins = 128
        self.measure_line_checkbox = None
        self.measure_readout_label = None
        self.profile_checkbox = None
        self._measure_view_box = None
        self._measure_scene = None
        self._measure_line_item = None
        self._measure_enabled = False
        self._measure_drag_active = False
        self._measure_start_point = None
        self._profile_enabled = False
        self._profile_roi = None
        self._profile_popup = None
        self._profile_popup_filter_installed = False
        self._profile_orientation_combo = None
        self._profile_filter_checkbox = None
        self._profile_filter_method_combo = None
        self._profile_plot_widget = None
        self._profile_curve = None
        self._profile_worker_lock = threading.Lock()
        self._profile_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="profile-worker")
        self._profile_worker_future = None
        self._profile_pending_request = None
        self._profile_request_counter = 0
        self._profile_latest_requested_id = 0
        self._profile_roi_dragging = False
        # Higher preview target -> less downsampling, better drag-time profile quality.
        self._profile_preview_target_pixels = 90000
        self.profile_compute_ready.connect(self._on_profile_compute_ready)
        self.profile_worker_idle.connect(self._on_profile_worker_idle)

        image_view = self.ui.cameraImage
        self._install_display_controls(image_view)
        self._setup_time_remaining_progress()
        self._configure_acquire_indicators()
        self._enforce_pan_interaction()
        self._ensure_measure_overlay_ready()

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
        self.destroyed.connect(self._shutdown_profile_worker)

    def _get_image_viewbox(self):
        image_view = self.ui.cameraImage
        try:
            if hasattr(image_view, "getView"):
                view = image_view.getView()
                if view is not None:
                    # PyDMImageView wraps pyqtgraph.ImageView(view=PlotItem),
                    # so getView() typically returns a PlotItem. Use its ViewBox.
                    if hasattr(view, "getViewBox"):
                        view_box = view.getViewBox()
                        if view_box is not None and hasattr(view_box, "setMouseMode"):
                            return view_box
                    if hasattr(view, "setMouseMode") and hasattr(view, "mapSceneToView"):
                        return view
            if hasattr(image_view, "getViewBox"):
                view_box = image_view.getViewBox()
                if view_box is not None and hasattr(view_box, "setMouseMode"):
                    return view_box
        except Exception:
            return None
        return None

    def _enforce_pan_interaction(self):
        view_box = self._get_image_viewbox()
        if view_box is None:
            return
        try:
            pan_mode = getattr(view_box, "PanMode", None)
            if pan_mode is not None:
                view_box.setMouseMode(pan_mode)
            if hasattr(view_box, "setMouseEnabled"):
                view_box.setMouseEnabled(x=True, y=True)
        except Exception:
            pass

    def _set_measure_readout(self, length_px):
        if self.measure_readout_label is None:
            return
        if length_px is None:
            self.measure_readout_label.setText("Length: -- px")
            return
        self.measure_readout_label.setText(f"Length: {float(length_px):.1f} px")

    def _clear_measure_line(self):
        self._measure_drag_active = False
        self._measure_start_point = None
        if self._measure_line_item is not None:
            try:
                self._measure_line_item.setData([], [])
                self._measure_line_item.hide()
            except Exception:
                pass
        self._set_measure_readout(None)

    def _ensure_measure_overlay_ready(self):
        if pg is None:
            return

        view_box = self._get_image_viewbox()
        if view_box is None:
            return
        self._measure_view_box = view_box
        scene = None
        try:
            scene = view_box.scene()
        except Exception:
            scene = None
        self._measure_scene = scene

        if self._measure_line_item is None:
            try:
                self._measure_line_item = pg.PlotCurveItem(pen=pg.mkPen(255, 165, 0, width=2))
                self._measure_line_item.hide()
                view_box.addItem(self._measure_line_item, ignoreBounds=True)
            except Exception:
                self._measure_line_item = None
        else:
            try:
                if self._measure_line_item.scene() is None:
                    view_box.addItem(self._measure_line_item, ignoreBounds=True)
            except Exception:
                pass

    def _set_measure_interaction_enabled(self, enabled):
        self._ensure_measure_overlay_ready()

        view_box = self._measure_view_box
        scene = self._measure_scene

        if view_box is not None:
            try:
                if enabled and hasattr(view_box, "setMouseEnabled"):
                    view_box.setMouseEnabled(x=False, y=False)
                elif hasattr(view_box, "setMouseEnabled"):
                    view_box.setMouseEnabled(x=True, y=True)
            except Exception:
                pass

        if scene is not None:
            try:
                scene.removeEventFilter(self)
            except Exception:
                pass
            if enabled:
                try:
                    scene.installEventFilter(self)
                except Exception:
                    pass

    def _map_scene_to_image_point(self, scene_pos):
        view_box = self._get_image_viewbox()
        if view_box is None:
            return None
        try:
            point = view_box.mapSceneToView(scene_pos)
        except Exception:
            return None
        return point

    def _update_measure_line(self, start_point, end_point):
        if self._measure_line_item is None:
            return
        if start_point is None or end_point is None:
            return

        x0 = float(start_point.x())
        y0 = float(start_point.y())
        x1 = float(end_point.x())
        y1 = float(end_point.y())
        try:
            self._measure_line_item.setData([x0, x1], [y0, y1])
            self._measure_line_item.show()
        except Exception:
            return

        length_px = np.hypot(x1 - x0, y1 - y0)
        self._set_measure_readout(length_px)

    def _on_measure_line_toggled(self, enabled):
        self._measure_enabled = bool(enabled)
        self._set_measure_interaction_enabled(self._measure_enabled)

        if not self._measure_enabled:
            self._clear_measure_line()
            self._enforce_pan_interaction()

    def _profile_source_2d(self):
        if self._last_image is None:
            return None
        data = np.asarray(self._last_image)
        if data.size == 0:
            return None
        if data.ndim == 2:
            return data
        if data.ndim == 3:
            if data.shape[2] in (3, 4):
                return np.nanmean(data, axis=2)
            if data.shape[0] in (3, 4):
                return np.nanmean(data, axis=0)
            return np.nanmean(data, axis=2)
        squeezed = np.squeeze(data)
        if squeezed.ndim == 2:
            return squeezed
        return None

    def _default_profile_roi_geometry(self):
        data = self._profile_source_2d()
        if data is None:
            return 10.0, 10.0, 96.0, 96.0
        height, width = data.shape[:2]
        roi_w = max(8.0, float(width) * 0.25)
        roi_h = max(8.0, float(height) * 0.25)
        x = max(0.0, (float(width) - roi_w) * 0.5)
        y = max(0.0, (float(height) - roi_h) * 0.5)
        return x, y, roi_w, roi_h

    def _ensure_profile_roi(self):
        if pg is None:
            return

        view_box = self._get_image_viewbox()
        if view_box is None:
            return

        if self._profile_roi is None:
            x, y, roi_w, roi_h = self._default_profile_roi_geometry()
            try:
                roi = pg.RectROI(
                    [x, y],
                    [roi_w, roi_h],
                    pen=pg.mkPen(0, 180, 255, width=2),
                    movable=True,
                    rotatable=False,
                    removable=False,
                    resizable=True,
                )
                roi.addScaleHandle([1, 1], [0, 0])
                roi.addScaleHandle([0, 0], [1, 1])
                roi.sigRegionChanged.connect(self._on_profile_roi_changed)
                if hasattr(roi, "sigRegionChangeFinished"):
                    roi.sigRegionChangeFinished.connect(self._on_profile_roi_change_finished)
                self._profile_roi = roi
            except Exception:
                self._profile_roi = None
                return

        if self._profile_roi is not None:
            try:
                if self._profile_roi.scene() is None:
                    view_box.addItem(self._profile_roi, ignoreBounds=True)
            except Exception:
                pass
            self._profile_roi.setVisible(self._profile_enabled)

    def _ensure_profile_popup(self):
        if pg is None:
            return
        if self._profile_popup is not None:
            return

        popup = QtWidgets.QWidget(None)
        popup.setWindowTitle("ROI Profile")
        popup.setWindowFlag(QtCore.Qt.Window, True)
        popup.setWindowFlag(QtCore.Qt.Tool, True)
        popup.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, True)
        popup.resize(500, 320)

        layout = QtWidgets.QVBoxLayout(popup)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(QtWidgets.QLabel("Integration"))
        orientation_combo = QtWidgets.QComboBox()
        orientation_combo.addItems(["Vertical", "Horizontal"])
        orientation_combo.setCurrentIndex(0)
        orientation_combo.currentIndexChanged.connect(self._update_profile_plot)
        controls.addWidget(orientation_combo)
        filter_checkbox = QtWidgets.QCheckBox("Filter outliers")
        filter_checkbox.setChecked(False)
        filter_checkbox.toggled.connect(self._on_profile_filter_toggled)
        controls.addWidget(filter_checkbox)
        controls.addWidget(QtWidgets.QLabel("Method"))
        filter_method_combo = QtWidgets.QComboBox()
        filter_method_combo.addItem("Robust (MAD)", "mad")
        if scipy_stats is not None:
            filter_method_combo.addItem("Fast (Sigma clip)", "sigmaclip")
        filter_method_combo.addItem("Fast (Percentile)", "percentile")
        if scipy_ndimage is not None:
            filter_method_combo.addItem("Median (6x6)", "median6")
        fast_method = "sigmaclip" if scipy_stats is not None else "percentile"
        fast_idx = filter_method_combo.findData(fast_method)
        if fast_idx >= 0:
            filter_method_combo.setCurrentIndex(fast_idx)
        filter_method_combo.currentIndexChanged.connect(self._update_profile_plot)
        filter_method_combo.setEnabled(False)
        controls.addWidget(filter_method_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        plot_widget = pg.PlotWidget()
        plot_widget.showGrid(x=True, y=True, alpha=0.2)
        plot_item = plot_widget.getPlotItem()
        plot_item.setLabel("left", "Integrated intensity")
        plot_item.setLabel("bottom", "X pixel")
        curve = plot_item.plot(pen=pg.mkPen(0, 120, 230, width=2))
        layout.addWidget(plot_widget, 1)

        self._profile_popup = popup
        if not self._profile_popup_filter_installed:
            try:
                self._profile_popup.installEventFilter(self)
                self._profile_popup_filter_installed = True
            except Exception:
                pass
        self._profile_orientation_combo = orientation_combo
        self._profile_filter_checkbox = filter_checkbox
        self._profile_filter_method_combo = filter_method_combo
        self._profile_plot_widget = plot_widget
        self._profile_curve = curve

    def _on_profile_toggled(self, enabled):
        self._profile_enabled = bool(enabled)
        self._ensure_profile_roi()
        self._ensure_profile_popup()

        if self._profile_roi is not None:
            self._profile_roi.setVisible(self._profile_enabled)

        if self._profile_enabled:
            if self._profile_popup is not None:
                self._profile_popup.show()
                self._profile_popup.raise_()
                self._profile_popup.activateWindow()
            self._update_profile_plot()
        else:
            if self._profile_popup is not None:
                self._profile_popup.hide()
            self._profile_roi_dragging = False
            self._cancel_profile_requests()
            self._set_profile_curve_data([], [], "X pixel")

    def _on_profile_filter_toggled(self, enabled):
        checked = bool(enabled)
        if self._profile_filter_method_combo is not None:
            self._profile_filter_method_combo.setEnabled(checked)
        self._update_profile_plot()

    def _profile_filter_method(self):
        if self._profile_filter_method_combo is None:
            return "mad"
        method = self._profile_filter_method_combo.currentData()
        if isinstance(method, str) and method:
            return method
        return "mad"

    def _on_profile_roi_changed(self, *args):
        if not self._profile_enabled:
            return
        self._profile_roi_dragging = True
        request = self._build_profile_request(preview=True)
        if request is None:
            self._cancel_profile_requests()
            self._set_profile_curve_data([], [], "X pixel")
            return
        self._queue_profile_request(request)

    def _on_profile_roi_change_finished(self, *args):
        self._profile_roi_dragging = False
        self._update_profile_plot()

    def _shutdown_profile_worker(self, *args):
        with self._profile_worker_lock:
            worker = self._profile_worker
            self._profile_worker = None
            self._profile_worker_future = None
            self._profile_pending_request = None
        if worker is not None:
            try:
                worker.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                worker.shutdown(wait=False)
            except Exception:
                pass

    def _cancel_profile_requests(self):
        with self._profile_worker_lock:
            self._profile_pending_request = None
            self._profile_latest_requested_id += 1

    def _set_profile_curve_data(self, coord, profile, axis_label):
        if self._profile_curve is None or self._profile_plot_widget is None:
            return
        plot_item = self._profile_plot_widget.getPlotItem()
        plot_item.setLabel("bottom", axis_label)
        self._profile_curve.setData(coord, profile)
        plot_item.enableAutoRange(axis="xy", enable=True)

    def _preview_filter_method(self, method):
        if method == "median6":
            if scipy_stats is not None:
                return "sigmaclip"
            return "percentile"
        return method

    def _preview_stride(self, roi_data):
        total = int(np.size(roi_data))
        if total <= 0:
            return 1
        target = max(4000, int(self._profile_preview_target_pixels))
        stride = int(np.sqrt(float(total) / float(target)))
        return max(1, stride)

    def _build_profile_request(self, preview=False):
        if not self._profile_enabled:
            return None
        data_2d = self._profile_source_2d()
        if data_2d is None:
            return None
        self._ensure_profile_roi()
        bounds = self._roi_bounds_for_image(data_2d)
        if bounds is None:
            return None

        x0, x1, y0, y1 = bounds
        roi_data = np.asarray(data_2d[y0:y1, x0:x1], dtype=float)
        if roi_data.size == 0:
            return None

        mode = "vertical"
        if self._profile_orientation_combo is not None:
            mode = self._profile_orientation_combo.currentText().strip().lower()

        filter_enabled = self._profile_filter_checkbox is not None and self._profile_filter_checkbox.isChecked()
        filter_method = self._profile_filter_method() if filter_enabled else "mad"
        if preview:
            stride = self._preview_stride(roi_data)
            if stride > 1:
                roi_data = roi_data[::stride, ::stride]
            filter_method = self._preview_filter_method(filter_method)
            if mode.startswith("h"):
                y1 = y0 + roi_data.shape[0]
            else:
                x1 = x0 + roi_data.shape[1]

        return {
            "roi_data": np.array(roi_data, copy=True),
            "mode": mode,
            "x0": int(x0),
            "x1": int(x1),
            "y0": int(y0),
            "y1": int(y1),
            "filter_enabled": bool(filter_enabled),
            "filter_method": filter_method,
        }

    def _queue_profile_request(self, request):
        with self._profile_worker_lock:
            self._profile_request_counter += 1
            request["id"] = self._profile_request_counter
            self._profile_latest_requested_id = request["id"]
            self._profile_pending_request = request
            should_submit = self._profile_worker_future is None
        if should_submit:
            self._submit_next_profile_request()

    def _submit_next_profile_request(self):
        with self._profile_worker_lock:
            if self._profile_worker is None or self._profile_worker_future is not None:
                return
            request = self._profile_pending_request
            self._profile_pending_request = None
            if request is None:
                return
            future = self._profile_worker.submit(self._compute_profile_request, request)
            self._profile_worker_future = future
        future.add_done_callback(self._on_profile_compute_done)

    def _on_profile_compute_done(self, future):
        result = None
        try:
            result = future.result()
        except Exception:
            result = None
        if result is not None:
            try:
                self.profile_compute_ready.emit(result)
            except Exception:
                pass
        try:
            self.profile_worker_idle.emit()
        except Exception:
            pass

    def _on_profile_worker_idle(self):
        with self._profile_worker_lock:
            self._profile_worker_future = None
            has_pending = self._profile_pending_request is not None
        if has_pending:
            self._submit_next_profile_request()

    def _on_profile_compute_ready(self, result):
        if not isinstance(result, dict):
            return
        if not self._profile_enabled:
            return
        if self._profile_curve is None or self._profile_plot_widget is None:
            return
        request_id = int(result.get("id", 0))
        if request_id < int(self._profile_latest_requested_id):
            return
        self._set_profile_curve_data(
            result.get("coord", []),
            result.get("profile", []),
            result.get("axis_label", "X pixel"),
        )

    def _roi_bounds_for_image(self, data_2d):
        if self._profile_roi is None or data_2d is None:
            return None
        height, width = data_2d.shape[:2]
        try:
            pos = self._profile_roi.pos()
            size = self._profile_roi.size()
            x0f = float(pos.x())
            y0f = float(pos.y())
            x1f = x0f + float(size.x())
            y1f = y0f + float(size.y())
        except Exception:
            return None

        x0 = int(np.floor(min(x0f, x1f)))
        x1 = int(np.ceil(max(x0f, x1f)))
        y0 = int(np.floor(min(y0f, y1f)))
        y1 = int(np.ceil(max(y0f, y1f)))

        x0 = max(0, min(width, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, x1, y0, y1

    def _apply_profile_outlier_filter(self, roi_data, method=None):
        filtered = np.asarray(roi_data, dtype=float)
        finite = filtered[np.isfinite(filtered)]
        if finite.size == 0:
            return filtered

        if method is None:
            method = self._profile_filter_method()

        if method == "median6" and scipy_ndimage is not None:
            work = np.array(filtered, copy=True)
            nan_mask = ~np.isfinite(work)
            if np.any(nan_mask):
                work[nan_mask] = float(np.median(finite))
            try:
                result = scipy_ndimage.median_filter(work, size=6, mode="nearest")
            except Exception:
                return filtered
            if np.any(nan_mask):
                result[nan_mask] = np.nan
            return result

        if method == "percentile":
            low, high = np.nanpercentile(filtered, [1.0, 99.0])
        elif method == "sigmaclip" and scipy_stats is not None:
            try:
                _, low, high = scipy_stats.sigmaclip(finite, low=5.0, high=5.0)
            except Exception:
                low, high = np.nan, np.nan
        else:
            if finite.size < 8:
                return filtered
            median = float(np.median(finite))
            mad = float(np.median(np.abs(finite - median)))
            if mad <= 0:
                low, high = np.percentile(finite, [1.0, 99.0])
            else:
                robust_sigma = 1.4826 * mad
                low = median - 5.0 * robust_sigma
                high = median + 5.0 * robust_sigma

        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            return filtered
        return np.clip(filtered, low, high)

    def _compute_profile_request(self, request):
        roi_data = np.asarray(request.get("roi_data"), dtype=float)
        if roi_data.size == 0:
            return {
                "id": int(request.get("id", 0)),
                "coord": np.array([], dtype=float),
                "profile": np.array([], dtype=float),
                "axis_label": "X pixel",
            }

        if bool(request.get("filter_enabled", False)):
            roi_data = self._apply_profile_outlier_filter(
                roi_data,
                method=request.get("filter_method"),
            )

        x0 = int(request.get("x0", 0))
        x1 = int(request.get("x1", x0))
        y0 = int(request.get("y0", 0))
        y1 = int(request.get("y1", y0))
        mode = str(request.get("mode", "vertical")).strip().lower()

        if mode.startswith("h"):
            profile = np.nansum(roi_data, axis=1)
            coord = np.arange(y0, y1, dtype=float)
            axis_label = "Y pixel"
        else:
            profile = np.nansum(roi_data, axis=0)
            coord = np.arange(x0, x1, dtype=float)
            axis_label = "X pixel"

        return {
            "id": int(request.get("id", 0)),
            "coord": coord,
            "profile": profile,
            "axis_label": axis_label,
        }

    def _update_profile_plot(self, *args):
        if not self._profile_enabled:
            return
        if self._profile_curve is None or self._profile_plot_widget is None:
            return
        request = self._build_profile_request(preview=False)
        if request is None:
            self._cancel_profile_requests()
            self._set_profile_curve_data([], [], "X pixel")
            return
        self._queue_profile_request(request)

    def eventFilter(self, watched, event):
        if watched is self._profile_popup and event.type() == QtCore.QEvent.Close:
            if self.profile_checkbox is not None and self.profile_checkbox.isChecked():
                self.profile_checkbox.setChecked(False)
            return super().eventFilter(watched, event)

        if watched is self._measure_scene and self._measure_enabled:
            event_type = event.type()
            if event_type == QtCore.QEvent.GraphicsSceneMousePress:
                try:
                    if event.button() != QtCore.Qt.LeftButton:
                        return False
                    point = self._map_scene_to_image_point(event.scenePos())
                    if point is None:
                        return False
                    self._measure_drag_active = True
                    self._measure_start_point = QtCore.QPointF(point)
                    self._update_measure_line(self._measure_start_point, point)
                    event.accept()
                    return True
                except Exception:
                    return False

            if event_type == QtCore.QEvent.GraphicsSceneMouseMove and self._measure_drag_active:
                try:
                    point = self._map_scene_to_image_point(event.scenePos())
                    if point is None or self._measure_start_point is None:
                        return True
                    self._update_measure_line(self._measure_start_point, point)
                    event.accept()
                    return True
                except Exception:
                    return True

            if event_type == QtCore.QEvent.GraphicsSceneMouseRelease and self._measure_drag_active:
                try:
                    if event.button() != QtCore.Qt.LeftButton:
                        return False
                    point = self._map_scene_to_image_point(event.scenePos())
                    if point is not None and self._measure_start_point is not None:
                        self._update_measure_line(self._measure_start_point, point)
                    self._measure_drag_active = False
                    event.accept()
                    return True
                except Exception:
                    self._measure_drag_active = False
                    return True

        return super().eventFilter(watched, event)

    def _setup_time_remaining_progress(self):
        old_widget = self.ui.PyDMLabel_6
        row_layout = self.ui.horizontalLayout
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
        acquire_time_address = (
            getattr(self.ui.PyDMLineEdit_2, "channel", None)
            or self.ui.PyDMLineEdit_2.property("channel")
        )

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

    def _on_time_remaining_changed(self, value):
        self._time_remaining_value = self._to_float(value, default=0.0)
        self._update_time_remaining_progress()

    def _on_acquire_time_changed(self, value):
        self._acquire_time_total = self._to_float(value, default=0.0)
        self._update_time_remaining_progress()

    def _to_float(self, value, default=0.0):
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

        if total > 0:
            frac_done = 1.0 - min(1.0, remaining / total)
            self.time_remaining_progress.setRange(0, 1000)
            self.time_remaining_progress.setValue(int(round(frac_done * 1000.0)))
        else:
            self.time_remaining_progress.setRange(0, 1000)
            self.time_remaining_progress.setValue(0)

        self.time_remaining_progress.setFormat(f"{remaining:.1f} s remaining")

    def _configure_acquire_indicators(self):
        indicator = self.ui.PyDMByteIndicator_2

        if hasattr(indicator, "setCircles"):
            indicator.setCircles(False)
        else:
            indicator.setProperty("circles", False)

        indicator.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        indicator.setMinimumSize(50, 50)
        indicator.setMaximumSize(50, 50)
        self._set_indicator_on_off_colors()
        self._set_acquire_indicator_style(False)
        self._set_detector_state_acquire_style(False)

        try:
            self.ui.horizontalLayout_7.setStretch(0, 0)
            self.ui.horizontalLayout_7.setStretch(1, 1)
        except Exception:
            pass

        channel_address = getattr(indicator, "channel", None) or indicator.property("channel")
        if channel_address:
            self._acquire_channel = PyDMChannel(
                address=channel_address,
                value_slot=self._on_acquire_value_changed,
            )
            self._acquire_channel.connect()

    def _set_indicator_on_off_colors(self):
        indicator = self.ui.PyDMByteIndicator_2
        on_color = QtGui.QColor(220, 0, 0)
        off_color = QtGui.QColor(90, 90, 90)

        candidates = [
            ("setOnColor", on_color),
            ("setOffColor", off_color),
            ("setTrueColor", on_color),
            ("setFalseColor", off_color),
        ]
        for method_name, color in candidates:
            method = getattr(indicator, method_name, None)
            if callable(method):
                try:
                    method(color)
                except Exception:
                    pass

        prop_candidates = [
            ("onColor", on_color),
            ("offColor", off_color),
            ("trueColor", on_color),
            ("falseColor", off_color),
        ]
        for prop_name, color in prop_candidates:
            try:
                if indicator.metaObject().indexOfProperty(prop_name) >= 0:
                    indicator.setProperty(prop_name, color)
            except Exception:
                pass

    def _set_acquire_indicator_style(self, acquiring):
        if acquiring:
            style = (
                "background-color: rgb(220, 0, 0);"
                "border: 1px solid black;"
            )
        else:
            style = (
                "background-color: rgb(90, 90, 90);"
                "border: 1px solid black;"
            )
        self.ui.PyDMByteIndicator_2.setStyleSheet(style)

    def _set_detector_state_acquire_style(self, acquiring):
        if acquiring:
            style = "color: rgb(220, 0, 0); background-color: rgb(213, 213, 213);"
        else:
            style = "color: rgb(0, 216, 0); background-color: rgb(213, 213, 213);"
        self.ui.PyDMLabel_7.setStyleSheet(style)

    def _on_acquire_value_changed(self, value):
        if isinstance(value, str):
            acquiring = value.strip().lower() in {"1", "true", "on", "acquire", "acquiring"}
        else:
            acquiring = bool(value)
        self._set_acquire_indicator_style(acquiring)
        self._set_indicator_on_off_colors()
        self.ui.PyDMByteIndicator_2.update()
        self._set_detector_state_acquire_style(acquiring)

    def _install_display_controls(self, image_view):
        controls = QtWidgets.QWidget(self.ui)
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(6, 2, 6, 2)
        controls_layout.setSpacing(6)

        self.auto_levels_checkbox = QtWidgets.QCheckBox("Auto levels")
        self.auto_levels_checkbox.setChecked(True)
        self.auto_levels_checkbox.toggled.connect(self._on_auto_levels_toggled)
        self.reset_levels_button = QtWidgets.QPushButton("Reset Levels")
        self.reset_levels_button.clicked.connect(self._reset_levels_to_default)
        self.reset_levels_button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.reset_levels_button.setFixedWidth(self.reset_levels_button.sizeHint().width())
        auto_block = QtWidgets.QWidget()
        auto_layout = QtWidgets.QVBoxLayout(auto_block)
        auto_layout.setContentsMargins(0, 0, 0, 0)
        auto_layout.setSpacing(2)
        auto_layout.addWidget(self.auto_levels_checkbox)
        auto_layout.addWidget(self.reset_levels_button)
        controls_layout.addWidget(auto_block)

        self.measure_line_checkbox = QtWidgets.QCheckBox("Measure line")
        self.measure_line_checkbox.setChecked(False)
        self.measure_line_checkbox.toggled.connect(self._on_measure_line_toggled)
        self.measure_readout_label = QtWidgets.QLabel("Length: -- px")
        measure_font = self.measure_readout_label.font()
        measure_font.setPointSize(max(8, measure_font.pointSize() - 1))
        self.measure_readout_label.setFont(measure_font)
        measure_block = QtWidgets.QWidget()
        measure_layout = QtWidgets.QVBoxLayout(measure_block)
        measure_layout.setContentsMargins(0, 0, 0, 0)
        measure_layout.setSpacing(2)
        measure_layout.addWidget(self.measure_line_checkbox)
        measure_layout.addWidget(self.measure_readout_label)
        self.profile_checkbox = QtWidgets.QCheckBox("Profile")
        self.profile_checkbox.setChecked(False)
        self.profile_checkbox.toggled.connect(self._on_profile_toggled)
        if pg is None:
            self.profile_checkbox.setEnabled(False)
            self.profile_checkbox.setToolTip("pyqtgraph is unavailable.")
        measure_layout.addWidget(self.profile_checkbox)
        measure_block.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        measure_block.setMaximumWidth(165)

        self.min_spinbox = QtWidgets.QDoubleSpinBox()
        self.min_spinbox.setDecimals(3)
        self.min_spinbox.setRange(-1e12, 1e12)
        self.min_spinbox.setSingleStep(1.0)
        self.min_spinbox.setMaximumWidth(140)
        self.min_spinbox.valueChanged.connect(self._on_manual_levels_changed)
        self.min_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.min_slider.setRange(0, int(self._level_slider_high_bound))
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
        self.max_slider.setRange(0, int(self._level_slider_high_bound))
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

        # Keep these slider-style control blocks compact so they don't dominate row width.
        compact_block_max_width = 180
        compact_blocks = (min_block, max_block, low_pct_block, high_pct_block, gamma_block)
        for block in compact_blocks:
            block.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            block.setMinimumWidth(0)
            block.setMaximumWidth(compact_block_max_width)

        self.colormap_combo = QtWidgets.QComboBox()
        self._colormap_options = self._discover_colormap_options(image_view)
        for label, _ in self._colormap_options:
            self.colormap_combo.addItem(label)
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        self.colormap_combo.setMinimumWidth(128)
        self.colormap_combo.setMaximumWidth(128)
        color_map_block = QtWidgets.QWidget()
        color_map_layout = QtWidgets.QVBoxLayout(color_map_block)
        color_map_layout.setContentsMargins(0, 0, 0, 0)
        color_map_layout.setSpacing(2)
        color_map_label = QtWidgets.QLabel("Color map")
        color_map_top = QtWidgets.QHBoxLayout()
        color_map_top.setContentsMargins(0, 0, 0, 0)
        color_map_top.setSpacing(4)
        color_map_top.addWidget(color_map_label)
        color_map_top.addStretch(1)
        color_map_layout.addLayout(color_map_top)
        color_map_layout.addWidget(self.colormap_combo)
        color_map_layout.setAlignment(self.colormap_combo, QtCore.Qt.AlignLeft)
        color_map_block.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        controls_layout.addWidget(color_map_block)
        current_cmap = image_view.property("colorMap")
        for i, (_, value) in enumerate(self._colormap_options):
            if value == current_cmap:
                self.colormap_combo.setCurrentIndex(i)
                break

        controls_layout.addWidget(measure_block)

        histogram_block = self._create_histogram_block()
        if histogram_block is not None:
            controls_layout.addWidget(histogram_block)

        # Let non-compact blocks absorb extra width, keep slider blocks narrow.
        for block in compact_blocks:
            idx = controls_layout.indexOf(block)
            if idx >= 0:
                controls_layout.setStretch(idx, 0)
        idx = controls_layout.indexOf(auto_block)
        if idx >= 0:
            controls_layout.setStretch(idx, 0)
        idx = controls_layout.indexOf(color_map_block)
        if idx >= 0:
            controls_layout.setStretch(idx, 0)
        idx = controls_layout.indexOf(measure_block)
        if idx >= 0:
            controls_layout.setStretch(idx, 0)
        if histogram_block is not None:
            idx = controls_layout.indexOf(histogram_block)
            if idx >= 0:
                controls_layout.setStretch(idx, 1)
        controls.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        controls.setMaximumHeight(120)

        image_parent = image_view.parentWidget()
        image_parent_layout = image_parent.layout() if image_parent is not None else None
        if image_parent_layout is not None:
            idx = image_parent_layout.indexOf(image_view)
            if idx >= 0:
                image_parent_layout.removeWidget(image_view)
                container = QtWidgets.QWidget(image_parent)
                container_layout = QtWidgets.QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.setSpacing(4)
                image_view.setParent(container)
                container_layout.addWidget(image_view, 1)
                container_layout.addWidget(controls, 0)
                image_parent_layout.insertWidget(idx, container)
                self._image_controls_container = container

        self._set_level_slider_scale(int(self._level_slider_high_bound))
        self._sync_level_sliders_from_spinboxes()
        self._on_auto_levels_toggled(True)
        self._capture_base_lut()
        self._apply_gamma_setting()

    def _create_histogram_block(self):
        if pg is None:
            return None

        block = QtWidgets.QWidget()
        block.setMinimumWidth(145)
        block.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        layout = QtWidgets.QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        histogram_plot = pg.PlotWidget()
        histogram_plot.setMinimumWidth(145)
        histogram_plot.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        histogram_plot.setMinimumHeight(56)
        histogram_plot.setMaximumHeight(56)
        histogram_plot.setMenuEnabled(False)
        histogram_plot.setMouseEnabled(x=False, y=False)
        histogram_plot.hideButtons()

        plot_item = histogram_plot.getPlotItem()
        plot_item.hideAxis("left")
        plot_item.showAxis("bottom")
        plot_item.getAxis("bottom").setStyle(showValues=False)

        self._histogram_curve = plot_item.plot(
            pen=pg.mkPen(40, 100, 215, width=1.5),
            fillLevel=0.0,
            brush=pg.mkBrush(40, 100, 215, 70),
        )
        self._histogram_low_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(200, 55, 55, width=1))
        self._histogram_high_line = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(20, 140, 60, width=1))
        plot_item.addItem(self._histogram_low_line)
        plot_item.addItem(self._histogram_high_line)
        self._histogram_plot_item = plot_item

        layout.addWidget(histogram_plot)

        self._update_histogram_markers()
        return block

    def _discover_colormap_options(self, image_view):
        options = []
        seen = set()

        meta = image_view.metaObject()
        prop_index = meta.indexOfProperty("colorMap")
        if prop_index >= 0:
            prop = meta.property(prop_index)
            if prop.isEnumType():
                enum = prop.enumerator()
                for i in range(enum.keyCount()):
                    key = enum.key(i)
                    value = enum.value(i)
                    if not key:
                        continue
                    key_str = str(key).strip()
                    # Filter internal/sentinel entries from Qt enum metadata.
                    if key_str.startswith("__"):
                        continue
                    if key_str.lower() == "__nofirstline__":
                        continue
                    if key_str in seen:
                        continue
                    seen.add(key_str)
                    options.append((key_str, value))

        if options:
            # Stable presentation order independent of Qt introspection order.
            options = sorted(options, key=lambda kv: kv[0].casefold())

        if not options:
            # Keep a safe fallback for environments where enum metadata is unavailable.
            options.append(("Monochrome", image_view.property("colorMap")))
        return options

    def _set_image_levels(self, low, high):
        image_view = self.ui.cameraImage
        if high <= low:
            high = low + 1.0

        low = max(self._level_slider_low_bound, min(self._level_slider_high_bound, low))
        high = max(self._level_slider_low_bound, min(self._level_slider_high_bound, high))
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

        # Re-apply gamma LUT in case the widget refreshed its image internals.
        self._apply_gamma_setting()
        self._update_histogram_markers()

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

    def _reset_levels_to_default(self):
        self._norm_low_percentile = self._default_norm_low_percentile
        self._norm_high_percentile = self._default_norm_high_percentile
        self._gamma_value = self._default_gamma_value

        self.low_pct_spinbox.blockSignals(True)
        self.high_pct_spinbox.blockSignals(True)
        self.gamma_spinbox.blockSignals(True)
        self.low_pct_spinbox.setValue(self._norm_low_percentile)
        self.high_pct_spinbox.setValue(self._norm_high_percentile)
        self.gamma_spinbox.setValue(self._gamma_value)
        self.low_pct_spinbox.blockSignals(False)
        self.high_pct_spinbox.blockSignals(False)
        self.gamma_spinbox.blockSignals(False)

        self._on_low_pct_spinbox_changed(self._norm_low_percentile)
        self._on_high_pct_spinbox_changed(self._norm_high_percentile)
        self._on_gamma_spinbox_changed(self._gamma_value)
        self._apply_gamma_setting()

        self.auto_levels_checkbox.setChecked(True)
        self._auto_levels_from_current_image()

    def _value_to_level_slider(self, value):
        return int(round(max(self._level_slider_low_bound, min(self._level_slider_high_bound, value))))

    def _slider_to_level_value(self, slider_value):
        return float(max(self._level_slider_low_bound, min(self._level_slider_high_bound, slider_value)))

    def _set_level_slider_scale(self, bit_max):
        bit_max = int(max(1, bit_max))
        self._level_slider_low_bound = 0.0
        self._level_slider_high_bound = float(bit_max)
        self.min_slider.blockSignals(True)
        self.max_slider.blockSignals(True)
        self.min_slider.setRange(0, bit_max)
        self.max_slider.setRange(0, bit_max)
        self.min_slider.blockSignals(False)
        self.max_slider.blockSignals(False)
        self.min_spinbox.setRange(0.0, float(bit_max))
        self.max_spinbox.setRange(0.0, float(bit_max))

    def _infer_bit_max_from_image_data(self, data):
        dtype = data.dtype

        if np.issubdtype(dtype, np.integer):
            if dtype.itemsize <= 1:
                return 255
            if dtype.itemsize <= 2:
                return 65535

        finite = data[np.isfinite(data)]
        if finite.size > 0 and np.nanmax(finite) <= 255:
            return 255
        return 65535

    def _update_level_slider_scale_from_image(self, data):
        bit_max = self._infer_bit_max_from_image_data(data)
        if int(self._level_slider_high_bound) != bit_max:
            self._set_level_slider_scale(bit_max)

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
        try:
            image_item = image_view.getImageItem()
            if image_item is None:
                return

            if self._base_lut is None:
                self._capture_base_lut()
            base_lut = self._base_lut
            if base_lut is None:
                return

            if abs(gamma - 1.0) < 1e-6:
                image_item.setLookupTable(base_lut)
                return

            n = int(base_lut.shape[0])
            x = np.linspace(0.0, 1.0, n)
            idx = np.clip((x ** gamma) * (n - 1), 0, n - 1).astype(int)
            gamma_lut = base_lut[idx]
            image_item.setLookupTable(gamma_lut)
        except Exception:
            pass

    def _capture_base_lut(self):
        image_view = self.ui.cameraImage
        try:
            image_item = image_view.getImageItem()
            if image_item is None:
                return

            lut = getattr(image_item, "lut", None)
            if lut is None and hasattr(image_item, "getLookupTable"):
                try:
                    lut = image_item.getLookupTable(256, alpha=True)
                except Exception:
                    try:
                        lut = image_item.getLookupTable()
                    except Exception:
                        lut = None
            if lut is None:
                gray = np.arange(256, dtype=np.uint8)
                alpha = np.full(256, 255, dtype=np.uint8)
                lut = np.column_stack((gray, gray, gray, alpha))
            else:
                lut = np.array(lut, copy=True)
                if lut.ndim == 1:
                    alpha = np.full(lut.shape[0], 255, dtype=lut.dtype)
                    lut = np.column_stack((lut, lut, lut, alpha))
                elif lut.shape[1] == 3:
                    alpha = np.full((lut.shape[0], 1), 255, dtype=lut.dtype)
                    lut = np.hstack((lut, alpha))

            self._base_lut = lut
        except Exception:
            self._base_lut = None

    def _auto_levels_from_current_image(self):
        if self._last_image is None:
            return
        data = np.asarray(self._last_image)
        if data.size == 0:
            return
        self._update_level_slider_scale_from_image(data)
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
        QtCore.QTimer.singleShot(0, self._refresh_lut_and_gamma)

    def _refresh_lut_and_gamma(self):
        self._capture_base_lut()
        self._apply_gamma_setting()

    def _schedule_histogram_update(self):
        if self._histogram_curve is None:
            return
        if self._histogram_update_pending:
            return
        self._histogram_update_pending = True
        QtCore.QTimer.singleShot(120, self._update_histogram_plot)

    def _update_histogram_markers(self):
        if self._histogram_low_line is None or self._histogram_high_line is None:
            return
        if not hasattr(self, "min_spinbox") or not hasattr(self, "max_spinbox"):
            return
        self._histogram_low_line.setPos(float(self.min_spinbox.value()))
        self._histogram_high_line.setPos(float(self.max_spinbox.value()))

    def _update_histogram_plot(self):
        self._histogram_update_pending = False
        if self._histogram_curve is None:
            return
        if self._last_image is None:
            return

        data = np.asarray(self._last_image)
        if data.size == 0:
            return
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return

        sample = np.ravel(finite)
        if sample.size > self._histogram_max_samples:
            step = int(np.ceil(float(sample.size) / float(self._histogram_max_samples)))
            sample = sample[::step]
        if sample.size == 0:
            return

        bit_max = float(self._infer_bit_max_from_image_data(data))

        hist, edges = np.histogram(sample, bins=self._histogram_bins, range=(0.0, bit_max))
        centers = 0.5 * (edges[:-1] + edges[1:])
        self._histogram_curve.setData(centers, hist.astype(float))

        if self._histogram_plot_item is not None:
            ymax = float(np.max(hist)) if hist.size else 1.0
            if ymax <= 0:
                ymax = 1.0
            self._histogram_plot_item.setXRange(0.0, bit_max, padding=0.0)
            self._histogram_plot_item.setYRange(0.0, ymax * 1.05, padding=0.0)

        self._update_histogram_markers()

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

        # Some backends reset interaction mode when first frame is rendered.
        if self._measure_enabled:
            self._set_measure_interaction_enabled(True)
        else:
            self._enforce_pan_interaction()

        self._last_image = image
        self._schedule_histogram_update()
        if self._profile_enabled and not self._profile_roi_dragging:
            self._update_profile_plot()
        if not self.auto_levels_checkbox.isChecked():
            if not self._manual_levels_initialized:
                self._auto_levels_from_current_image()
                self._manual_levels_initialized = True
            return

        data = np.asarray(image)
        if data.size == 0:
            return
        self._update_level_slider_scale_from_image(data)

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
