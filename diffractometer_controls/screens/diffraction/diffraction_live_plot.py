from diffractometer_controls.site.mitr.profile import demo_active, pv_prefix
import math
import ast
import concurrent.futures
import os
import re
import threading
from collections.abc import Iterable

from diffractometer_controls.analysis.diffraction_history import (
    DIFFRACTION_EXPERIMENT_TYPE,
    _is_diffraction_start_doc,
    _make_history_entry,
    _preferred_catalog_names_for_mode,
    _resolve_catalog_name,
    _run_data_type,
    _run_start_stop_docs,
    _start_live_plot_fields,
    build_history_snapshot,
    fetch_recent_history_entries,
    select_history_stream,
)
from diffraction_analysis.metadata import run_outcome
from diffraction_analysis.history import history_entry_tooltip
from diffraction_analysis.widgets import run_status_icon

import numpy as np
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from control_ui.widgets.scalar_scan_widgets import ScalarCountReadout, ScalarScanTable
from control_ui.widgets.elided_label import ElidedLabel
from control_ui.widgets.row_geometry import match_control_heights
from control_ui.widgets.profile_colors import (
    HISTORY_ALPHA, LATEST_ALPHA, SERIES_COLORS as _CATEGORICAL_SERIES_COLORS,
    profile_colormap, profile_history_position,
)


try:
    from lmfit.models import GaussianModel, LinearModel
except Exception:
    GaussianModel = None
    LinearModel = None

try:
    from epics import PV, caget
except Exception:
    PV = None
    caget = None

try:
    from tiled.client import from_uri
except Exception:
    from_uri = None

try:
    from diffractometer_controls.analysis.tiled_auth import build_tiled_client_from_cached_tokens, get_tiled_uri
except Exception:
    try:
        from diffractometer_controls.analysis.tiled_auth import build_tiled_client_from_cached_tokens, get_tiled_uri
    except Exception:
        build_tiled_client_from_cached_tokens = None

        def get_tiled_uri(host=None):
            uri = os.environ.get("TILED_URI") or os.environ.get("MITR_TILED_URI")
            if uri:
                return uri
            host = host or os.environ.get("MITR_CONTROL_HOST") or os.environ.get("MITR_CONTROL_IP") or "localhost"
            port = os.environ.get("MITR_TILED_PORT", "8000")
            return f"http://{host}:{port}"

try:
    from diffractometer_controls.analysis.tiled_auth import ensure_tiled_login
except Exception:
    try:
        from diffractometer_controls.analysis.tiled_auth import ensure_tiled_login
    except Exception:
        ensure_tiled_login = None

try:
    from diffractometer_controls.site.mitr.bluesky_mode import BLUESKY_MODE_TEST, get_bluesky_active_mode
except Exception:
    try:
        from diffractometer_controls.site.mitr.bluesky_mode import BLUESKY_MODE_TEST, get_bluesky_active_mode
    except Exception:
        BLUESKY_MODE_TEST = "test"

        def get_bluesky_active_mode():
            return "production"


TILED_URI = get_tiled_uri()
RUN_UID_STATUS_PV = f"{pv_prefix()}Bluesky:Run:RunUID"
RUN_STATE_STATUS_PV = f"{pv_prefix()}Bluesky:Run:State"
DEFAULT_HISTORY_LIMIT = 100
matplotlib = None
FigureCanvas = None
NavigationToolbar = None



def _initialize_matplotlib():
    global matplotlib, FigureCanvas, NavigationToolbar
    if matplotlib is not None and FigureCanvas is not None and NavigationToolbar is not None:
        return
    import matplotlib as _matplotlib
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NavigationToolbar

    _matplotlib.use("Qt5Agg")
    import matplotlib.pyplot  # noqa: F401

    matplotlib = _matplotlib
    FigureCanvas = _FigureCanvas
    NavigationToolbar = _NavigationToolbar


def _coerce_array(value):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def _coerce_number(value):
    try:
        num = float(value)
    except Exception:
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


def _build_linear_axis(*, nbins, axis_min, axis_max):
    n = max(1, int(round(float(nbins))))
    return np.linspace(float(axis_min), float(axis_max), n)





def _rgba_to_hex(rgba):
    try:
        r, g, b = list(rgba)[:3]
    except Exception:
        return "#4a90e2"
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0.0, min(1.0, float(r))) * 255.0),
        int(max(0.0, min(1.0, float(g))) * 255.0),
        int(max(0.0, min(1.0, float(b))) * 255.0),
    )


def _safe_caget(pv_name, *, as_string=False):
    if caget is None:
        return None
    try:
        return caget(str(pv_name), timeout=1.0, connection_timeout=1.0, as_string=as_string)
    except Exception:
        return None


def _build_tiled_client(uri):
    if callable(build_tiled_client_from_cached_tokens):
        return build_tiled_client_from_cached_tokens(uri=uri)
    if from_uri is None:
        raise RuntimeError("tiled client is unavailable in this environment.")
    return from_uri(uri)


def _friendly_source_name(source):
    text = str(source or "").strip().replace("_", " ")
    text = " ".join(word.capitalize() for word in text.split())
    return re.sub(r"\bUsbctr\b", "USBCTR", text)


def _infer_field_source(data_key, label):
    channel_name = re.sub(
        r"^CTR\d+\s*[-:]\s*",
        "",
        str(label or "").strip(),
        flags=re.IGNORECASE,
    )
    channel_slug = re.sub(r"[^a-z0-9]+", "_", channel_name.lower()).strip("_")
    data_key = str(data_key or "")
    if channel_slug and data_key.lower().endswith(f"_{channel_slug}"):
        return data_key[: -(len(channel_slug) + 1)]
    return ""


def _qualify_duplicate_field_labels(fields):
    """Give same-named scalar channels stable, source-qualified identities."""
    fields = {str(key): dict(spec or {}) for key, spec in dict(fields or {}).items()}
    label_keys = {}
    for data_key, spec in fields.items():
        if str(spec.get("role", "signal") or "signal") != "signal":
            continue
        label = str(spec.get("label", "") or data_key)
        label_keys.setdefault(label, []).append(data_key)

    for label, data_keys in label_keys.items():
        if len(data_keys) < 2:
            continue
        short_label = re.sub(
            r"^CTR\d+\s*[-:]\s*",
            "",
            label,
            flags=re.IGNORECASE,
        )
        candidates = []
        for data_key in data_keys:
            spec = fields[data_key]
            source = str(spec.get("source", "") or "")
            if not source:
                source = _infer_field_source(data_key, label)
            if source:
                spec["source"] = source
            friendly_source = _friendly_source_name(source)
            candidates.append(
                f"{friendly_source}: {short_label}" if friendly_source else data_key
            )
        if len(set(candidates)) != len(candidates):
            candidates = [f"{candidate} [{key}]" for candidate, key in zip(candidates, data_keys)]
        for data_key, qualified_label in zip(data_keys, candidates):
            fields[data_key]["label"] = qualified_label
    return fields


def _fit_peak_position(x_values, y_values):
    if GaussianModel is None or LinearModel is None:
        return None

    x_arr = _coerce_array(x_values)
    y_arr = _coerce_array(y_values)
    if x_arr is None or y_arr is None or x_arr.shape != y_arr.shape:
        return None
    if x_arr.size < 7:
        return None

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if np.count_nonzero(finite) < 7:
        return None
    x_arr = x_arr[finite]
    y_arr = y_arr[finite]

    try:
        peak_index = int(np.nanargmax(y_arr))
    except Exception:
        return None

    peak_y = float(y_arr[peak_index])
    if not math.isfinite(peak_y):
        return None

    n_edge = max(3, min(12, x_arr.size // 10))
    edge_y = np.concatenate((y_arr[:n_edge], y_arr[-n_edge:]))
    baseline = float(np.nanmedian(edge_y)) if edge_y.size else float(np.nanmedian(y_arr))
    height = peak_y - baseline
    if not math.isfinite(height) or height <= 0:
        return None

    threshold = baseline + (0.25 * height)
    left = peak_index
    right = peak_index
    while left > 0 and y_arr[left - 1] >= threshold:
        left -= 1
    while right < (x_arr.size - 1) and y_arr[right + 1] >= threshold:
        right += 1

    half_window_min = 4
    left = max(0, left - 2)
    right = min(x_arr.size - 1, right + 2)
    while (right - left + 1) < (2 * half_window_min + 1):
        if left > 0:
            left -= 1
        if right < (x_arr.size - 1):
            right += 1
        if left == 0 and right == (x_arr.size - 1):
            break

    x_fit = x_arr[left : right + 1]
    y_fit = y_arr[left : right + 1]
    if x_fit.size < 7:
        return None

    span = float(x_fit[-1] - x_fit[0])
    if not math.isfinite(span) or abs(span) <= 0:
        return None

    dx = np.diff(x_fit)
    dx_mean = float(np.nanmedian(np.abs(dx))) if dx.size else 1.0
    sigma_guess = max(abs(span) / 6.0, dx_mean, 1e-3)
    center_guess = float(x_arr[peak_index])
    slope_guess = float((y_fit[-1] - y_fit[0]) / span)
    intercept_guess = float(np.nanmedian(y_fit) - (slope_guess * np.nanmedian(x_fit)))
    amplitude_guess = max(height * sigma_guess * math.sqrt(2.0 * math.pi), 1e-6)

    model = GaussianModel(prefix="g_") + LinearModel(prefix="b_")
    params = model.make_params()
    params["g_center"].set(value=center_guess, min=float(np.min(x_fit)), max=float(np.max(x_fit)))
    params["g_sigma"].set(
        value=sigma_guess,
        min=max(dx_mean * 0.5, 1e-6),
        max=max(abs(span), sigma_guess * 8.0),
    )
    params["g_amplitude"].set(value=amplitude_guess, min=0.0)
    params["b_slope"].set(value=slope_guess)
    params["b_intercept"].set(value=intercept_guess)

    try:
        result = model.fit(y_fit, params, x=x_fit, nan_policy="omit")
        center_param = result.params["g_center"]
        center = float(center_param.value)
        center_err = center_param.stderr
    except Exception:
        return None

    if not math.isfinite(center):
        return None
    if center < float(np.min(x_fit)) or center > float(np.max(x_fit)):
        return None
    if center_err is not None:
        try:
            center_err = float(center_err)
        except Exception:
            center_err = None
        if center_err is not None and (not math.isfinite(center_err) or center_err < 0):
            center_err = None
    return center, center_err


class DiffractionPlotWidget(QWidget):
    def __init__(self, parent=None):
        _initialize_matplotlib()
        super().__init__(parent)
        self._applying_theme = False
        self._last_toolbar_style = None
        self._last_canvas_style = None

        self.figure = matplotlib.figure.Figure()
        self.profile_axes, self.summary_axes, self.peak_axes = self._create_axes()
        self._apply_static_layout()

        canvas = FigureCanvas(self.figure)
        canvas.setMinimumWidth(640)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        canvas.setParent(self)
        self._canvas = canvas
        self._toolbar = NavigationToolbar(canvas, parent=self)
        self._canvas.mpl_connect("button_press_event", self._on_mouse_press)

        self._scalar_readout = ScalarCountReadout(self)
        self._scalar_table = ScalarScanTable(self)
        self._scalar_details = QWidget(self)
        scalar_details_layout = QHBoxLayout(self._scalar_details)
        scalar_details_layout.setContentsMargins(0, 0, 0, 0)
        scalar_details_layout.setSpacing(10)
        scalar_details_layout.addWidget(self._scalar_readout, 1)
        scalar_details_layout.addWidget(self._scalar_table, 1)
        self._scalar_details.hide()

        layout = QVBoxLayout()
        layout.addWidget(canvas, 2)
        layout.addWidget(self._scalar_details, 1)
        layout.addWidget(self._toolbar)
        self.setLayout(layout)
        self._main_layout = layout

        self._plot_mode = "1d"
        self._show_count_rate_gauge = True

        self._profile_history = {}
        self._live_profile_lines = {}
        self._profile_colormaps = {}
        self._detector_order = []
        self._detector_series_colors = {}
        self._summary_lines = {}
        self._summary_live_lines = {}
        self._summary_x = {}
        self._summary_y = {}
        self._peak_lines = {}
        self._peak_errorbars = {}
        self._peak_x = {}
        self._peak_y = {}
        self._peak_yerr = {}
        self._profile_x_limits = None
        self._summary_x_limits = None
        self._peak_x_limits = None
        self.reset()

    def sizeHint(self):
        size_hint = super().sizeHint()
        size_hint.setWidth(920)
        size_hint.setHeight(560)
        return size_hint

    def _create_axes(self):
        grid_spec = self.figure.add_gridspec(
            2,
            2,
            height_ratios=[2.2, 1.2],
            width_ratios=[1.0, 1.0],
        )
        profile_axes = self.figure.add_subplot(grid_spec[0, :])
        summary_axes = self.figure.add_subplot(grid_spec[1, 0])
        peak_axes = self.figure.add_subplot(grid_spec[1, 1])
        return profile_axes, summary_axes, peak_axes

    def _apply_static_layout(self):
        self.figure.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.09,
            top=0.92,
            hspace=0.34,
            wspace=0.24,
        )

    @QtCore.Slot(object)
    def reset(self, config=None):
        config = dict(config or {})
        plot_mode = str(config.get("plot_mode", "1d") or "1d").strip().lower()
        self._plot_mode = "scalar" if plot_mode == "scalar" else "1d"
        self._show_count_rate_gauge = bool(config.get("show_count_rate_gauge", True))
        run_title = str(config.get("run_title", "") or "")
        profile_title = str(config.get("profile_title", "Current Spectrum") or "Current Spectrum")
        summary_title = str(config.get("summary_title", "Total Counts") or "Total Counts")
        peak_title = str(
            config.get("peak_title", "Fitted Peak Position") or "Fitted Peak Position"
        )
        summary_x_label = str(config.get("summary_x_label", "Point") or "Point")
        summary_y_label = str(config.get("summary_y_label", "Total Counts") or "Total Counts")
        peak_y_label = str(config.get("peak_y_label", "Peak Position") or "Peak Position")
        profile_x_limits = self._normalize_axis_limits(config.get("profile_x_limits"))
        summary_x_limits = self._normalize_axis_limits(config.get("summary_x_limits"))
        peak_x_limits = self._normalize_axis_limits(config.get("peak_x_limits"))

        self.figure.clf()
        self.profile_axes, self.summary_axes, self.peak_axes = self._create_axes()
        self._apply_static_layout()
        if self._plot_mode == "scalar":
            self._main_layout.setStretch(0, 2)
            self._main_layout.setStretch(1, 1)
            self.profile_axes.set_visible(False)
            self.peak_axes.set_visible(False)
            self.summary_axes.set_position([0.08, 0.10, 0.90, 0.80])
            self._scalar_details.show()
            self._scalar_readout.setVisible(self._show_count_rate_gauge)
        else:
            self._main_layout.setStretch(0, 2)
            self._main_layout.setStretch(1, 1)
            self._scalar_details.hide()
        self.profile_axes.grid(alpha=0.25)
        self.summary_axes.grid(alpha=0.25)
        self.peak_axes.grid(alpha=0.25)
        self.profile_axes.tick_params(direction="in", which="both")
        self.summary_axes.tick_params(direction="in", which="both")
        self.peak_axes.tick_params(direction="in", which="both")
        self.profile_axes.set_title(profile_title)
        self.profile_axes.set_ylabel("Counts")
        self.profile_axes.set_xlabel("Detector Position")
        self.summary_axes.set_title(summary_title)
        self.summary_axes.set_xlabel(summary_x_label)
        self.summary_axes.set_ylabel(summary_y_label)
        self.peak_axes.set_title(peak_title)
        self.peak_axes.set_xlabel(summary_x_label)
        self.peak_axes.set_ylabel(peak_y_label)
        self.figure.suptitle(run_title)
        self._apply_theme_from_palette()
        self._profile_x_limits = profile_x_limits
        self._summary_x_limits = summary_x_limits
        self._peak_x_limits = peak_x_limits
        self._set_axis_x_limits(self.profile_axes, self._profile_x_limits)
        self._set_axis_x_limits(self.summary_axes, self._summary_x_limits)
        self._set_axis_x_limits(self.peak_axes, self._peak_x_limits)

        self._profile_history.clear()
        self._live_profile_lines.clear()
        self._profile_colormaps.clear()
        self._detector_order.clear()
        self._detector_series_colors.clear()
        self._summary_lines.clear()
        self._summary_live_lines.clear()
        self._summary_x.clear()
        self._summary_y.clear()
        self._peak_lines.clear()
        self._peak_errorbars.clear()
        self._peak_x.clear()
        self._peak_y.clear()
        self._peak_yerr.clear()
        self._scalar_readout.reset()
        self._scalar_table.reset_scan(summary_x_label)
        self._redraw()

    @QtCore.Slot(str, object, object)
    def set_profile(self, detector_name, x_values, y_values):
        x_arr = _coerce_array(x_values)
        y_arr = _coerce_array(y_values)
        if x_arr is None or y_arr is None:
            return
        if x_arr.shape != y_arr.shape:
            return

        had_live_preview = self._remove_live_profile_line(detector_name) is not None
        history = self._profile_history.setdefault(detector_name, [])
        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map

        (line,) = self.profile_axes.plot(
            x_arr,
            y_arr,
            color=color_map(0.9),
            marker="o",
            lw=1.5,
            mfc="none",
            ms=3,
            alpha=0.95,
            label=str(detector_name),
        )
        history.append(line)
        if had_live_preview:
            self._style_profile_history_line(
                line,
                color_map=color_map,
                total=len(history),
                index=len(history) - 1,
                is_latest=True,
            )
        else:
            self._restyle_profile_history(detector_name)

        self._sync_axis_x_limits(self.profile_axes, x_arr, fixed_limits=self._profile_x_limits)
        self._autoscale_y(self.profile_axes)
        self._update_legends()
        self._redraw()

    @QtCore.Slot(str, float, float)
    def append_summary_point(self, detector_name, x_value, y_value):
        series_x = self._summary_x.setdefault(detector_name, [])
        series_y = self._summary_y.setdefault(detector_name, [])
        series_x.append(float(x_value))
        series_y.append(float(y_value))
        color = self._get_detector_series_color(detector_name)
        self._remove_live_summary_line(detector_name)

        line = self._summary_lines.get(detector_name)
        if line is None:
            (line,) = self.summary_axes.plot(
                series_x,
                series_y,
                color=color,
                marker="o",
                lw=1,
                mfc="none",
                ms=4,
                label=str(detector_name),
            )
            self._summary_lines[detector_name] = line
        else:
            line.set_data(series_x, series_y)

        self._sync_axis_x_limits(
            self.summary_axes,
            series_x,
            fixed_limits=self._summary_x_limits,
        )
        self._autoscale_y(self.summary_axes)
        self._update_legends()
        if self._plot_mode == "scalar":
            self._scalar_table.append_point(detector_name, x_value, y_value)
        self._redraw()

    @QtCore.Slot(str, float, float)
    def update_live_summary_point(self, detector_name, x_value, y_value):
        color = self._get_detector_series_color(detector_name)
        base_x = list(self._summary_x.get(detector_name, []))
        base_y = list(self._summary_y.get(detector_name, []))
        draw_x = base_x + [float(x_value)]
        draw_y = base_y + [float(y_value)]

        line = self._summary_live_lines.get(detector_name)
        if line is None:
            (line,) = self.summary_axes.plot(
                draw_x,
                draw_y,
                color=color,
                marker="o",
                lw=1,
                mfc="none",
                ms=4,
                alpha=1.0,
                zorder=5,
                label="_nolegend_",
            )
            self._summary_live_lines[detector_name] = line
        else:
            line.set_data(draw_x, draw_y)
            line.set_color(color)

        self._sync_axis_x_limits(
            self.summary_axes,
            draw_x,
            fixed_limits=self._summary_x_limits,
        )
        self._autoscale_y(self.summary_axes)
        self._update_legends()
        self._redraw()

    @QtCore.Slot(str, float, object)
    def update_scalar_readout(self, detector_name, total_counts, elapsed_time=None):
        self._scalar_readout.update_reading(
            detector_name,
            total_counts,
            elapsed_time,
        )

    @QtCore.Slot(str, object, object)
    def update_live_profile(self, detector_name, x_values, y_values):
        x_arr = _coerce_array(x_values)
        y_arr = _coerce_array(y_values)
        if x_arr is None or y_arr is None:
            return
        if x_arr.shape != y_arr.shape:
            return

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map

        line = self._live_profile_lines.get(detector_name)
        if line is None:
            self._restyle_profile_history(detector_name, preview_new=True)
            (line,) = self.profile_axes.plot(
                x_arr,
                y_arr,
                color=color_map(0.9),
                marker="o",
                lw=1.5,
                ls="-",
                mfc="none",
                ms=3,
                alpha=0.95,
                zorder=5,
                label="_nolegend_",
            )
            self._live_profile_lines[detector_name] = line
            self._style_profile_history_line(
                line,
                color_map=color_map,
                total=max(1, len(self._profile_history.get(detector_name, [])) + 1),
                index=max(0, len(self._profile_history.get(detector_name, []))),
                is_latest=True,
            )
        else:
            line.set_data(x_arr, y_arr)
            line.set_color(color_map(0.9))

        self._sync_axis_x_limits(self.profile_axes, x_arr, fixed_limits=self._profile_x_limits)
        self._autoscale_y(self.profile_axes)
        self._redraw()

    @QtCore.Slot(str, float, float, object)
    def append_peak_point(self, detector_name, x_value, y_value, y_err):
        series_x = self._peak_x.setdefault(detector_name, [])
        series_y = self._peak_y.setdefault(detector_name, [])
        series_yerr = self._peak_yerr.setdefault(detector_name, [])
        series_x.append(float(x_value))
        series_y.append(float(y_value))
        color = self._get_detector_series_color(detector_name)
        try:
            err_value = None if y_err is None else float(y_err)
        except Exception:
            err_value = None
        if err_value is not None and (not math.isfinite(err_value) or err_value < 0):
            err_value = None
        series_yerr.append(err_value)

        old_container = self._peak_errorbars.get(detector_name)
        self._remove_errorbar_container(old_container)

        yerr = None
        if any(err is not None for err in series_yerr):
            yerr = [np.nan if err is None else float(err) for err in series_yerr]

        container = self.peak_axes.errorbar(
            series_x,
            series_y,
            yerr=yerr,
            color=color,
            ecolor=color,
            marker="o",
            lw=1,
            mfc="none",
            ms=4,
            capsize=3,
            elinewidth=1,
            label=str(detector_name),
        )
        self._peak_errorbars[detector_name] = container
        if getattr(container, "lines", None):
            self._peak_lines[detector_name] = container.lines[0]
        else:
            self._peak_lines.pop(detector_name, None)

        self._sync_axis_x_limits(
            self.peak_axes,
            series_x,
            fixed_limits=self._peak_x_limits,
        )
        self._autoscale_y(self.peak_axes)
        self._update_legends()
        self._redraw()

    @QtCore.Slot(str)
    def set_status(self, text):
        title = str(text or "")
        self.figure.suptitle(title)
        self._apply_theme_from_palette()
        self._redraw()

    def _update_legends(self):
        self._set_axis_legend(
            self.profile_axes,
            {
                detector_name: history[-1]
                for detector_name, history in self._profile_history.items()
                if history
            },
        )
        summary_lines = dict(self._summary_live_lines)
        summary_lines.update(self._summary_lines)
        self._set_axis_legend(self.summary_axes, summary_lines)
        self._set_axis_legend(self.peak_axes, self._peak_lines)
        self._style_legends()

    @staticmethod
    def _set_axis_legend(axes, line_map):
        handles = []
        labels = []
        for detector_name, line in line_map.items():
            if line is None:
                continue
            handles.append(line)
            labels.append(str(detector_name))
        legend = axes.get_legend()
        if handles:
            axes.legend(handles, labels, loc="best")
        elif legend is not None:
            legend.remove()

    def _remove_live_profile_line(self, detector_name):
        line = self._live_profile_lines.pop(detector_name, None)
        if line is None:
            return None
        try:
            line.remove()
        except Exception:
            pass
        return line

    def _remove_live_summary_line(self, detector_name):
        line = self._summary_live_lines.pop(detector_name, None)
        if line is None:
            return
        try:
            line.remove()
        except Exception:
            pass

    @QtCore.Slot()
    def clear_live_previews(self):
        for detector_name in tuple(self._live_profile_lines.keys()):
            removed_line = self._remove_live_profile_line(detector_name)
            if removed_line is not None:
                self._restyle_profile_history(detector_name)
        for detector_name in tuple(self._summary_live_lines.keys()):
            self._remove_live_summary_line(detector_name)
        self._redraw()

    def _restyle_profile_history(self, detector_name, *, preview_new=False):
        history = self._profile_history.get(detector_name, [])
        if not history:
            return

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map
        total = len(history) + (1 if preview_new else 0)
        for index, line in enumerate(history):
            is_latest = (not preview_new) and index == (len(history) - 1)
            self._style_profile_history_line(
                line,
                color_map=color_map,
                total=total,
                index=index,
                is_latest=is_latest,
            )

    @staticmethod
    def _style_profile_history_line(line, *, color_map, total, index, is_latest):
        color_pos = DiffractionPlotWidget._profile_history_color_position(total=total, index=index)
        if total <= 1:
            age = 1.0
        else:
            age = index / (total - 1)
        line.set_color(color_map(color_pos))
        line.set_alpha(LATEST_ALPHA if is_latest else HISTORY_ALPHA)
        line.set_linewidth(1.5 if is_latest else 1.0 + (0.3 * age))
        line.set_marker("o" if is_latest else "None")

    @staticmethod
    def _profile_history_color_position(*, total, index):
        return profile_history_position(total, index)

    def _on_mouse_press(self, event):
        if event is None or not getattr(event, "dblclick", False):
            return
        if getattr(event, "button", None) != 1:
            return
        axes = getattr(event, "inaxes", None)
        if axes is None:
            return
        if axes is self.profile_axes:
            self._reset_axis_view(self.profile_axes, fixed_limits=self._profile_x_limits)
        elif axes is self.summary_axes:
            self._reset_axis_view(self.summary_axes, fixed_limits=self._summary_x_limits)
        elif axes is self.peak_axes:
            self._reset_axis_view(self.peak_axes, fixed_limits=self._peak_x_limits)

    def _reset_axis_view(self, axes, *, fixed_limits=None):
        axes.relim()
        if fixed_limits is None:
            axes.autoscale_view(scalex=True, scaley=True)
        else:
            axes.autoscale_view(scalex=False, scaley=True)
            axes.set_xlim(*fixed_limits)
        self._redraw()

    def _select_profile_colormap(self, detector_name):
        if detector_name not in self._detector_order:
            self._detector_order.append(detector_name)
        index = self._detector_order.index(detector_name)
        background = self._palette().color(QtGui.QPalette.Base).getRgb()[:3]
        return profile_colormap(index, background)

    def _get_detector_series_color(self, detector_name):
        color = self._detector_series_colors.get(detector_name)
        if color is not None:
            return color

        if detector_name not in self._detector_order:
            self._detector_order.append(detector_name)
        index = self._detector_order.index(detector_name)
        color = _CATEGORICAL_SERIES_COLORS[index % len(_CATEGORICAL_SERIES_COLORS)]
        self._detector_series_colors[detector_name] = color
        return color

    def _palette(self):
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            return QtGui.QPalette(app.palette())
        return QtGui.QPalette(self.palette())

    def _apply_theme_from_palette(self):
        if self._applying_theme:
            return
        self._applying_theme = True
        try:
            pal = self._palette()
            figure_bg = pal.color(QtGui.QPalette.Window)
            axes_bg = pal.color(QtGui.QPalette.Base)
            profile_background = axes_bg.getRgb()[:3]
            if profile_background != getattr(self, '_profile_background', None):
                self._profile_background = profile_background
                self._profile_colormaps.clear()
                for detector in self._profile_history:
                    self._restyle_profile_history(detector, preview_new=detector in self._live_profile_lines)
                for detector, item in self._live_profile_lines.items():
                    color_map = self._profile_colormaps.setdefault(detector, self._select_profile_colormap(detector))
                    item.set_color(color_map(0.9))
                    item.set_alpha(LATEST_ALPHA)
            text = pal.color(QtGui.QPalette.WindowText)
            axis_text = pal.color(QtGui.QPalette.Text)
            grid = pal.color(QtGui.QPalette.Mid)
            edge = pal.color(QtGui.QPalette.Mid)

            self.figure.patch.set_facecolor(figure_bg.name())
            self.figure.patch.set_edgecolor(figure_bg.name())

            for axes in (self.profile_axes, self.summary_axes, self.peak_axes):
                axes.set_facecolor(axes_bg.name())
                axes.grid(color=grid.name(), alpha=0.35)
                axes.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    colors=axis_text.name(),
                    labelcolor=axis_text.name(),
                )
                axes.xaxis.label.set_color(text.name())
                axes.yaxis.label.set_color(text.name())
                axes.title.set_color(text.name())
                for spine in axes.spines.values():
                    spine.set_color(edge.name())

            suptitle = getattr(self.figure, "_suptitle", None)
            if suptitle is not None:
                suptitle.set_color(text.name())

            self._style_legends()
            self._style_toolbar(pal)
            canvas_style = f"background-color: {axes_bg.name()}; color: {text.name()};"
            if canvas_style != self._last_canvas_style:
                self._canvas.setStyleSheet(canvas_style)
                self._last_canvas_style = canvas_style
            self.setAutoFillBackground(True)
        finally:
            self._applying_theme = False

    def _style_legends(self):
        pal = self._palette()
        legend_bg = pal.color(QtGui.QPalette.Base).name()
        legend_edge = pal.color(QtGui.QPalette.Mid).name()
        legend_text = pal.color(QtGui.QPalette.Text).name()
        for axes in (self.profile_axes, self.summary_axes, self.peak_axes):
            legend = axes.get_legend()
            if legend is None:
                continue
            frame = legend.get_frame()
            frame.set_facecolor(legend_bg)
            frame.set_edgecolor(legend_edge)
            frame.set_alpha(0.9)
            for text in legend.get_texts():
                text.set_color(legend_text)

    @staticmethod
    def _remove_errorbar_container(container):
        if container is None:
            return
        try:
            data_line, caplines, barlinecols = container.lines
        except Exception:
            data_line, caplines, barlinecols = None, (), ()
        if data_line is not None:
            try:
                data_line.remove()
            except Exception:
                pass
        for group in (caplines, barlinecols):
            for artist in group or ():
                try:
                    artist.remove()
                except Exception:
                    pass

    @staticmethod
    def _normalize_axis_limits(value):
        if value is None:
            return None
        try:
            low, high = value
        except Exception:
            return None
        low = _coerce_number(low)
        high = _coerce_number(high)
        if low is None or high is None:
            return None
        if high < low:
            low, high = high, low
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
            low -= pad
            high += pad
        return float(low), float(high)

    @staticmethod
    def _set_axis_x_limits(axes, limits):
        if limits is None:
            return
        axes.set_xlim(*DiffractionPlotWidget._expand_limits_for_view(limits))

    def _sync_axis_x_limits(self, axes, x_values, *, fixed_limits=None):
        if fixed_limits is not None:
            return
        x_arr = _coerce_array(x_values)
        if x_arr is None or x_arr.size == 0:
            return
        finite = x_arr[np.isfinite(x_arr)]
        if finite.size == 0:
            return
        low = float(np.min(finite))
        high = float(np.max(finite))
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = max((high - low) * 0.02, 1e-9)
        axes.set_xlim(low - pad, high + pad)

    @staticmethod
    def _expand_limits_for_view(limits):
        low = float(limits[0])
        high = float(limits[1])
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = max((high - low) * 0.02, 0.25)
        return low - pad, high + pad

    @staticmethod
    def _autoscale_y(axes):
        axes.relim()
        axes.autoscale_view(scalex=False, scaley=True)

    def _style_toolbar(self, palette):
        bg = palette.color(QtGui.QPalette.Window).name()
        base = palette.color(QtGui.QPalette.Base).name()
        text = palette.color(QtGui.QPalette.ButtonText).name()
        border = palette.color(QtGui.QPalette.Mid).name()
        highlight = palette.color(QtGui.QPalette.Highlight).name()
        highlighted_text = palette.color(QtGui.QPalette.HighlightedText).name()
        toolbar_style = (
            "QToolBar {"
            f" background-color: {bg};"
            f" color: {text};"
            f" border-top: 1px solid {border};"
            "}"
            "QToolButton {"
            f" background-color: {base};"
            f" color: {text};"
            f" border: 1px solid {border};"
            " border-radius: 3px;"
            " padding: 3px 5px;"
            " margin: 1px;"
            "}"
            "QToolButton:hover {"
            f" background-color: {highlight};"
            f" color: {highlighted_text};"
            "}"
            "QLabel {"
            f" color: {text};"
            "}"
        )
        if toolbar_style != self._last_toolbar_style:
            self._toolbar.setStyleSheet(toolbar_style)
            self._last_toolbar_style = toolbar_style

    def changeEvent(self, event):
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
            QtCore.QEvent.StyleChange,
        ):
            self._apply_theme_from_palette()
            self._redraw()

    def _redraw(self):
        self.figure.canvas.draw_idle()


class DiffractionHistoryViewer(QWidget):
    tiled_enabled_toggled = QtCore.Signal(bool)
    previous_requested = QtCore.Signal()
    next_requested = QtCore.Signal()
    run_selected = QtCore.Signal(str)
    live_follow_toggled = QtCore.Signal(bool)
    analysis_requested = QtCore.Signal()

    def __init__(self, plot_widget, parent=None):
        super().__init__(parent)
        self._plot_widget = plot_widget
        self._history_entries = []
        self._history_enabled = False
        self._applying_theme = False
        self._last_theme_signature = None

        self._previous_button = QToolButton(self)
        self._previous_button.setArrowType(QtCore.Qt.LeftArrow)
        self._previous_button.setToolTip("Load older diffraction run")
        self._previous_button.clicked.connect(self.previous_requested.emit)

        self._run_combo = QComboBox(self)
        self._run_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._run_combo.setMinimumContentsLength(20)
        self._run_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._run_combo.setMaxVisibleItems(12)
        run_list = QListView(self._run_combo)
        run_list.setUniformItemSizes(True)
        run_list.setTextElideMode(QtCore.Qt.ElideRight)
        run_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        run_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._run_combo.setView(run_list)
        # Bound the popup itself; bounding only its view leaves an oversized frame.
        run_list.window().setMaximumSize(640, 360)
        self._run_combo.currentIndexChanged.connect(self._on_combo_index_changed)

        self._next_button = QToolButton(self)
        self._next_button.setArrowType(QtCore.Qt.RightArrow)
        self._next_button.setToolTip("Load newer diffraction run")
        self._next_button.clicked.connect(self.next_requested.emit)

        self._tiled_checkbox = QCheckBox("Tiled", self)
        self._tiled_checkbox.setChecked(False)
        self._tiled_checkbox.toggled.connect(self.tiled_enabled_toggled.emit)

        self._live_checkbox = QCheckBox("Live scan", self)
        self._live_checkbox.setChecked(True)
        self._live_checkbox.setToolTip('Follow current and new diffraction runs. Uncheck to hold the displayed run. Live following works without Tiled.')
        self._live_checkbox.toggled.connect(self.live_follow_toggled.emit)

        self._analysis_button = QToolButton(self)
        self._analysis_button.setText("Analyze runs…")
        self._analysis_button.setToolTip("Open diffraction history and metadata in a separate window")
        self._analysis_button.clicked.connect(self.analysis_requested.emit)

        self._status_label = ElidedLabel(self)
        self._status_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self._status_label.setFixedWidth(self.fontMetrics().averageCharWidth() * 32)
        self._status_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self._row_controls = (
            self._tiled_checkbox, self._previous_button, self._run_combo,
            self._next_button, self._live_checkbox, self._analysis_button, self._status_label,
        )

        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self._tiled_checkbox)
        controls_layout.addWidget(self._previous_button)
        controls_layout.addWidget(self._run_combo, 1)
        controls_layout.addWidget(self._next_button)
        controls_layout.addWidget(self._live_checkbox)
        controls_layout.addWidget(self._analysis_button)
        controls_layout.addWidget(self._status_label)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(controls_layout)
        plot_area = QWidget(self)
        plot_layout = QGridLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.addWidget(self._plot_widget, 0, 0)
        self._history_message = QLabel(plot_area)
        self._history_message.setAlignment(QtCore.Qt.AlignCenter)
        self._history_message.setWordWrap(True)
        self._history_message.setTextFormat(QtCore.Qt.PlainText)
        self._history_message.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._history_message.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._history_message.setMargin(40)
        plot_layout.addWidget(self._history_message, 0, 0)
        self._history_message.hide()
        layout.addWidget(plot_area, 1)
        self.setLayout(layout)

        self.set_history_state(
            {
                "entries": [],
                "selected_uid": "",
                "history_enabled": False,
                "tiled_enabled": False,
                "live_follow": True,
                "status_text": "Live only (Tiled off)",
            }
        )

    @QtCore.Slot(object)
    def set_history_state(self, state):
        state = dict(state or {})
        entries = list(state.get("entries", []) or [])
        selected_uid = str(state.get("selected_uid", "") or "")
        history_enabled = bool(state.get("history_enabled", False))
        tiled_enabled = bool(state.get("tiled_enabled", False))
        live_follow = bool(state.get("live_follow", True))
        status_text = str(state.get("status_text", "") or "")

        self._history_entries = entries
        self._history_enabled = history_enabled

        with QtCore.QSignalBlocker(self._run_combo):
            self._run_combo.clear()
            for entry in entries:
                unavailable = bool(entry.get('load_error')) or entry.get('file_status') == 'failed'
                self._run_combo.addItem(run_status_icon(entry.get('status', ''), unavailable=unavailable),
                                        str(entry.get("label", "")), str(entry.get("uid", "")))
                tooltip = history_entry_tooltip(entry)
                self._run_combo.setItemData(self._run_combo.count() - 1, tooltip, QtCore.Qt.ToolTipRole)
            target_index = -1
            if selected_uid:
                for index, entry in enumerate(entries):
                    if str(entry.get("uid", "")) == selected_uid:
                        target_index = index
                        break
            if target_index < 0 and entries:
                target_index = 0
            if target_index >= 0:
                self._run_combo.setCurrentIndex(target_index)

        with QtCore.QSignalBlocker(self._live_checkbox):
            self._live_checkbox.setChecked(live_follow)
        with QtCore.QSignalBlocker(self._tiled_checkbox):
            self._tiled_checkbox.setChecked(tiled_enabled)

        current_index = self._run_combo.currentIndex()
        has_entries = bool(entries)
        controls_enabled = tiled_enabled and history_enabled and has_entries
        self._run_combo.setEnabled(controls_enabled)
        self._previous_button.setEnabled(
            controls_enabled and current_index >= 0 and current_index < (len(entries) - 1)
        )
        self._next_button.setEnabled(controls_enabled and current_index > 0)
        self._run_combo.setToolTip(str(self._run_combo.currentData(QtCore.Qt.ToolTipRole) or self._run_combo.currentText()))
        self._live_checkbox.setEnabled(True)
        self._status_label.setText(status_text)
        self._apply_theme_from_palette()

    @QtCore.Slot(object)
    def reset(self, config=None):
        self.set_history_message('')
        return self._plot_widget.reset(config)

    @QtCore.Slot(str)
    def set_history_message(self, text):
        self._history_message.setText(text)
        self._history_message.setVisible(bool(text))
        if text:
            self._history_message.raise_()

    @QtCore.Slot(str, object, object)
    def set_profile(self, detector_name, x_values, y_values):
        return self._plot_widget.set_profile(detector_name, x_values, y_values)

    @QtCore.Slot(str, object, object)
    def update_live_profile(self, detector_name, x_values, y_values):
        return self._plot_widget.update_live_profile(detector_name, x_values, y_values)

    @QtCore.Slot(str, float, float)
    def append_summary_point(self, detector_name, x_value, y_value):
        return self._plot_widget.append_summary_point(detector_name, x_value, y_value)

    @QtCore.Slot(str, float, float)
    def update_live_summary_point(self, detector_name, x_value, y_value):
        return self._plot_widget.update_live_summary_point(detector_name, x_value, y_value)

    @QtCore.Slot(str, float, object)
    def update_scalar_readout(self, detector_name, total_counts, elapsed_time=None):
        slot = getattr(self._plot_widget, "update_scalar_readout", None)
        if callable(slot):
            return slot(detector_name, total_counts, elapsed_time)

    @QtCore.Slot(str, float, float, object)
    def append_peak_point(self, detector_name, x_value, y_value, y_err):
        return self._plot_widget.append_peak_point(detector_name, x_value, y_value, y_err)

    @QtCore.Slot(str)
    def set_status(self, text):
        return self._plot_widget.set_status(text)

    @QtCore.Slot()
    def clear_live_previews(self):
        return self._plot_widget.clear_live_previews()

    def _on_combo_index_changed(self, index):
        if index < 0 or index >= len(self._history_entries):
            return
        uid = str(self._run_combo.itemData(index) or "")
        if uid:
            self.run_selected.emit(uid)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event is None:
            return
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
            QtCore.QEvent.StyleChange,
            QtCore.QEvent.FontChange,
            QtCore.QEvent.ApplicationFontChange,
        ):
            self._apply_theme_from_palette()

    def _match_row_heights(self):
        height = match_control_heights(*self._row_controls)
        self._previous_button.setFixedWidth(height)
        self._next_button.setFixedWidth(height)

    def _apply_theme_from_palette(self):
        if self._applying_theme:
            return
        self._applying_theme = True
        try:
            pal = QtGui.QPalette(self.palette())
            base = pal.color(QtGui.QPalette.Base).name()
            window = pal.color(QtGui.QPalette.Window).name()
            text = pal.color(QtGui.QPalette.Text).name()
            button_text = pal.color(QtGui.QPalette.ButtonText).name()
            border = pal.color(QtGui.QPalette.Mid).name()
            highlight = pal.color(QtGui.QPalette.Highlight).name()
            highlighted_text = pal.color(QtGui.QPalette.HighlightedText).name()

            button_style = (
                "QToolButton {"
                f" background-color: {base};"
                f" color: {button_text};"
                f" border: 1px solid {border};"
                " border-radius: 3px;"
                " padding: 3px 5px;"
                "}"
                "QToolButton:hover {"
                f" background-color: {highlight};"
                f" color: {highlighted_text};"
                "}"
            )
            combo_style = (
                "QComboBox {"
                # Menu-style popups can ignore maxVisibleItems and expand to fit labels.
                " combobox-popup: 0;"
                f" background-color: {base};"
                f" color: {text};"
                f" border: 1px solid {border};"
                " border-radius: 3px;"
                " padding: 2px 6px;"
                " min-height: 24px;"
                "}"
                "QComboBox QAbstractItemView {"
                f" background-color: {base};"
                f" color: {text};"
                f" selection-background-color: {highlight};"
                f" selection-color: {highlighted_text};"
                "}"
            )
            checkbox_style = (
                "QCheckBox {"
                f" color: {text};"
                "}"
            )
            status_style = f"color: {text}; background-color: {window};"
            theme_signature = (
                window,
                button_style,
                combo_style,
                checkbox_style,
                status_style,
            )
            if theme_signature == self._last_theme_signature:
                self._match_row_heights()
                return

            self.setStyleSheet(f"background-color: {window};")
            self._previous_button.setStyleSheet(button_style)
            self._next_button.setStyleSheet(button_style)
            self._analysis_button.setStyleSheet(button_style)
            self._run_combo.setStyleSheet(combo_style)
            # A style change can restore the menu popup's hidden scrollbar policy.
            self._run_combo.view().setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            self._tiled_checkbox.setStyleSheet(checkbox_style)
            self._live_checkbox.setStyleSheet(checkbox_style)
            self._status_label.setStyleSheet(status_style)
            self._last_theme_signature = theme_signature
            self._match_row_heights()
        finally:
            self._applying_theme = False


class DiffractionLivePlot(QtCore.QObject):
    _reset_requested = QtCore.Signal(object)
    _clear_live_previews_requested = QtCore.Signal()
    _profile_updated = QtCore.Signal(str, object, object)
    _live_profile_updated = QtCore.Signal(str, object, object)
    _summary_point_updated = QtCore.Signal(str, float, float)
    _live_summary_point_updated = QtCore.Signal(str, float, float)
    _scalar_readout_updated = QtCore.Signal(str, float, object)
    _peak_point_updated = QtCore.Signal(str, float, float, object)
    _status_updated = QtCore.Signal(str)
    _history_state_updated = QtCore.Signal(object)
    _history_refresh_completed = QtCore.Signal(int, object)
    _history_load_completed = QtCore.Signal(int, object)
    _history_plot_message = QtCore.Signal(str)
    _local_reader_ready = QtCore.Signal(object)

    def __init__(self, widget, *, stream_name="primary", re_client=None, parent=None, auto_history=False):
        super().__init__(parent)
        self.widget = widget
        self.stream_name = str(stream_name or "primary")
        self.re_client = re_client

        queued = QtCore.Qt.QueuedConnection
        self._reset_requested.connect(self.widget.reset, queued)
        self._clear_live_previews_requested.connect(self.widget.clear_live_previews, queued)
        self._profile_updated.connect(self.widget.set_profile, queued)
        self._live_profile_updated.connect(self.widget.update_live_profile, queued)
        self._summary_point_updated.connect(self.widget.append_summary_point, queued)
        self._live_summary_point_updated.connect(self.widget.update_live_summary_point, queued)
        update_scalar_readout = getattr(self.widget, "update_scalar_readout", None)
        if callable(update_scalar_readout):
            self._scalar_readout_updated.connect(update_scalar_readout, queued)
        self._peak_point_updated.connect(self.widget.append_peak_point, queued)
        self._status_updated.connect(self.widget.set_status, queued)
        set_history_state = getattr(self.widget, "set_history_state", None)
        if callable(set_history_state):
            self._history_state_updated.connect(set_history_state, queued)
        self._history_refresh_completed.connect(self._on_history_refresh_completed, queued)
        self._history_load_completed.connect(self._on_history_load_completed, queued)
        self._local_reader_ready.connect(self._on_local_reader_ready, queued)
        if callable(getattr(self.widget, 'set_history_message', None)):
            self._history_plot_message.connect(self.widget.set_history_message, queued)

        self._descriptor_stream = {}
        self._descriptor_run_start = {}
        self._descriptor_data_keys = {}
        self._run_uid = None
        self._run_title = ""
        self._plan_name = ""
        self._data_type = "1d"
        self._live_plot_fields = {}
        self._run_detector_names = []
        self._motor_names = []
        self._summary_mode = "point"
        self._summary_x_label = "Point"
        self._summary_title = "Total Counts"
        self._summary_counter = 0
        self._primary_points_seen = 0
        self._summary_path_last_point = None
        self._summary_path_distance = 0.0
        self._detector_axis_cache = {}
        self._detector_axis_bounds = {}
        self._live_subscriptions = {}
        self._live_total_counts = {}
        self._live_elapsed_time = {}
        self._live_scalar_counts = {}
        self._scalar_count_labels = set()
        self._scalar_elapsed_keys = {}
        self._scalar_count_origins = {}
        self._scalar_sample_elapsed = {}
        self._scalar_elapsed_fallback = None
        self._armed_live_detectors = ()
        self._armed_live_field_signature = ()
        self._planned_summary_x = []
        self._displayed_run_uid = ""
        self._selected_history_uid = ""
        self._current_live_run_uid = ""
        self._live_run_active = False
        self._last_stopped_run_uid = ""
        self._current_live_start_doc = {}
        self._provisional_live_context = False
        self._rendered_primary_seq_nums = set()
        self._live_primary_rows = {}
        self._live_preview_lock = threading.RLock()
        self._live_preview_profiles = {}
        self._live_preview_values = {}
        self._live_backfill_pending = False
        self._history_access_changed = False
        self._backfill_retry_timer = QtCore.QTimer(self)
        self._backfill_retry_timer.setSingleShot(True)
        self._backfill_retry_timer.setInterval(5000)
        self._backfill_retry_timer.timeout.connect(self._request_startup_history_load)
        self._tiled_enabled = False
        self._live_follow_enabled = True
        self._display_mode = "live"
        self._history_limit = int(DEFAULT_HISTORY_LIMIT)
        self._history_entries = []
        self._history_catalog = None
        self._history_client = None
        self._history_catalog_name = ""
        self._history_mode = ""
        self._history_error = ""
        self._history_status_text = "Live only (Tiled off)"
        self._history_refresh_token = 0
        self._history_load_token = 0
        self._startup_history_uid = ""
        self._startup_history_retry_count = 0
        self._history_executor = None
        self._analysis_viewer = None
        self._active = False
        self._running_item_events_connected = False

        previous_requested = getattr(self.widget, "previous_requested", None)
        if previous_requested is not None:
            previous_requested.connect(self._on_history_previous_requested)
        next_requested = getattr(self.widget, "next_requested", None)
        if next_requested is not None:
            next_requested.connect(self._on_history_next_requested)
        run_selected = getattr(self.widget, "run_selected", None)
        if run_selected is not None:
            run_selected.connect(self._on_history_run_selected)
        tiled_enabled_toggled = getattr(self.widget, "tiled_enabled_toggled", None)
        if tiled_enabled_toggled is not None:
            tiled_enabled_toggled.connect(self._on_tiled_enabled_toggled)
        live_follow_toggled = getattr(self.widget, "live_follow_toggled", None)
        if live_follow_toggled is not None:
            live_follow_toggled.connect(self._on_live_follow_toggled)
        analysis_requested = getattr(self.widget, "analysis_requested", None)
        if analysis_requested is not None:
            analysis_requested.connect(self.open_analysis_viewer)
        self.destroyed.connect(self.shutdown)
        self.activate()
        self._emit_history_state()
        if auto_history:
            QtCore.QTimer.singleShot(0, self._connect_local_history_async)

    def _connect_local_history_async(self):
        if demo_active() or not self._active or self._history_executor is None:
            return
        from diffractometer_controls.analysis.tiled_auth import build_tiled_client_from_local_reader

        def connect():
            try:
                return build_tiled_client_from_local_reader()
            except Exception:
                return None  # Live acquisition remains usable when Tiled is offline.

        def completed(future):
            if future.cancelled():
                return
            try:
                self._local_reader_ready.emit(future.result())
            except RuntimeError:
                pass

        self._history_executor.submit(connect).add_done_callback(completed)

    def _on_local_reader_ready(self, client):
        if client is None or not self._active or self._history_access_changed:
            if client is not None and client is not self._history_client:
                context = getattr(client, "context", None)
                if context is not None:
                    context.close()
            return
        self._history_client = client
        self._tiled_enabled = True
        self._refresh_history_async(autoload_live=True)

    def _set_running_item_events_connected(self, connected):
        connected = bool(connected)
        if connected == self._running_item_events_connected:
            return
        events = getattr(self.re_client, "events", None)
        emitter = getattr(events, "running_item_changed", None)
        if emitter is None:
            return
        try:
            if connected:
                emitter.connect(self._on_running_item_changed)
            else:
                emitter.disconnect(self._on_running_item_changed)
        except Exception:
            return
        self._running_item_events_connected = connected

    def activate(self):
        """Resume model callbacks and background work for a visible screen."""
        if self._active:
            return
        self._active = True
        if self._history_executor is None:
            self._history_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="diffraction-history",
            )
        self._set_running_item_events_connected(True)
        QtCore.QTimer.singleShot(0, self._arm_current_running_item)
        if self._tiled_enabled:
            QtCore.QTimer.singleShot(0, self._request_startup_history_load)

    def deactivate(self):
        """Suspend callbacks and workers while PyDM caches the hidden screen."""
        self._active = False
        self._live_backfill_pending = False
        self._backfill_retry_timer.stop()
        self._set_running_item_events_connected(False)
        self._disarm_live_subscriptions()
        # Invalidate results from work that may complete after deactivation.
        self._history_refresh_token += 1
        self._history_load_token += 1
        self._shutdown_history_executor()

    def shutdown(self, *_args):
        window = self._analysis_viewer
        self._analysis_viewer = None
        if window is not None:
            try:
                window.close()
            except RuntimeError:
                pass
        self.deactivate()

    def open_analysis_viewer(self):
        """Open an independent, read-only diffraction browser without pausing live plots."""
        if demo_active():
            self._emit_history_state(status_text="Storage disabled in demo mode")
            return
        if self._analysis_viewer is not None:
            self._analysis_viewer.show()
            self._analysis_viewer.raise_()
            self._analysis_viewer.activateWindow()
            return
        client = self._history_client
        if client is None:
            client = ensure_tiled_login(parent=self.widget)
        if client is None:
            return
        from .diffraction_history_window import DiffractionHistoryWindow

        window = DiffractionHistoryWindow(
            client, mode=get_bluesky_active_mode(),
            selected_uid=self._displayed_run_uid, parent=self.widget,
        )
        self._analysis_viewer = window
        window.destroyed.connect(lambda: setattr(self, "_analysis_viewer", None))
        window.show()

    def _shutdown_history_executor(self, *_args):
        executor = getattr(self, "_history_executor", None)
        if executor is None:
            return
        self._history_executor = None
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def _arm_current_running_item(self):
        if (not self._active) or (not self._live_follow_enabled) or self._display_mode != "live":
            return
        running_item = dict(getattr(self.re_client, "_running_item", {}) or {})
        if not running_item:
            if self._current_live_run_uid and self._live_plot_fields:
                self._arm_live_plot_field_subscriptions(self._live_plot_fields)
            return
        if not self._ensure_live_context_from_running_item(running_item):
            self._disarm_live_subscriptions()
            return
        detector_names = self._extract_live_detector_names_from_running_item(running_item)
        if self._live_plot_fields:
            self._arm_live_plot_field_subscriptions(self._live_plot_fields)
        else:
            self._arm_live_subscriptions(detector_names)

    def _emit_history_state(self, *, status_text=None, selected_uid=None):
        if status_text is not None:
            self._history_status_text = str(status_text or "")
        if selected_uid is None:
            selected_uid = self._displayed_run_uid or self._current_live_run_uid
        history_enabled = (
            self._tiled_enabled and self._history_catalog is not None and not self._history_error
        )
        self._history_state_updated.emit(
            {
                "entries": list(self._history_entries),
                "selected_uid": str(selected_uid or ""),
                "history_enabled": history_enabled,
                "tiled_enabled": bool(self._tiled_enabled),
                "live_follow": bool(self._live_follow_enabled),
                "status_text": str(self._history_status_text or ""),
            }
        )

    def _schedule_history_refresh(self, *, delay_ms=0, autoload_live=False):
        if not self._active:
            return
        QtCore.QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda autoload_live=bool(autoload_live): self._refresh_history_async(
                autoload_live=autoload_live
            ),
        )

    def _refresh_history_async(self, *, autoload_live=False):
        if (not self._active) or (not self._tiled_enabled):
            return
        executor = getattr(self, "_history_executor", None)
        if executor is None:
            return
        self._history_refresh_token += 1
        token = int(self._history_refresh_token)
        if self._history_catalog is None and not self._history_error:
            self._emit_history_state(status_text="Connecting to diffraction history...")
        future = executor.submit(self._history_refresh_task, bool(autoload_live))
        future.add_done_callback(
            lambda fut, token=token: self._emit_history_refresh_result(token, fut)
        )

    def _emit_history_refresh_result(self, token, future):
        payload = {
            "entries": [],
            "catalog_name": self._history_catalog_name,
            "mode": self._history_mode,
            "autoload_live": False,
            "error": "",
        }
        try:
            payload.update(dict(future.result() or {}))
        except Exception as exc:
            payload["error"] = str(exc)
        self._history_refresh_completed.emit(int(token), payload)

    def _on_history_refresh_completed(self, token, payload):
        if int(token) != int(self._history_refresh_token):
            return
        payload = dict(payload or {})
        error_text = str(payload.get("error", "") or "")
        if error_text:
            self._history_error = error_text
            self._history_catalog = None
            self._history_catalog_name = ""
            self._history_entries = []
            self._emit_history_state(status_text=f"History unavailable: {error_text}")
            return

        self._history_error = ""
        self._history_catalog_name = str(
            payload.get("catalog_name", "") or self._history_catalog_name
        )
        self._history_mode = str(payload.get("mode", "") or self._history_mode)
        self._history_entries = list(payload.get("entries", []) or [])

        if self._history_entries:
            catalog_text = self._history_catalog_name or "diffraction catalog"
            self._emit_history_state(status_text=f"History: {catalog_text}")
        else:
            catalog_text = self._history_catalog_name or "diffraction catalog"
            self._emit_history_state(status_text=f"No diffraction runs found in {catalog_text}")

        if bool(payload.get("autoload_live")) and self._live_follow_enabled:
            self._request_startup_history_load()

    def _request_startup_history_load(self):
        if not self._active or self._live_backfill_pending or not self._tiled_enabled:
            return
        if not self._live_follow_enabled:
            return
        live_uid = self._read_active_run_uid()
        if not live_uid:
            if self._provisional_live_context:
                self._emit_history_state(status_text="Live diffraction run · waiting for run identity")
                self._backfill_retry_timer.start(5000)
                return
            if self._run_uid and self._run_uid != "__live_running__":
                # Automatic startup reads can complete after Stop, or run
                # again when returning to a cached screen. Keep the displayed
                # run instead of replacing it with an older or incomplete one.
                return
            latest_uid = ""
            if self._history_entries:
                latest_uid = str(self._history_entries[0].get("uid", "") or "").strip()
            if latest_uid:
                self._request_history_load(latest_uid, request_kind="latest")
            else:
                self._emit_history_state(status_text="No active diffraction run detected")
            return
        if str(live_uid) != str(self._startup_history_uid or ""):
            self._startup_history_retry_count = 0
        self._current_live_run_uid = str(live_uid)
        self._live_run_active = True
        self._startup_history_uid = str(live_uid)
        self._request_history_load(live_uid, request_kind="startup")

    def _request_history_load(self, uid, *, request_kind):
        if not self._tiled_enabled:
            return
        uid = str(uid or "").strip()
        if not uid:
            return
        executor = getattr(self, "_history_executor", None)
        if executor is None:
            return

        self._history_load_token += 1
        token = int(self._history_load_token)
        request_kind = str(request_kind or "history")
        self._live_backfill_pending = request_kind == "startup"
        if request_kind == "history":
            self._display_mode = "history"
            self._live_follow_enabled = False
            self._selected_history_uid = uid
            self._disarm_live_subscriptions()
            self._clear_live_previews_requested.emit()

        existing_entry = self._history_entry_for_uid(uid)
        target_label = str(existing_entry.get("label", uid[:8])) if existing_entry else uid[:8]
        if request_kind == "startup":
            prefix = "Loading active run"
        elif request_kind == "latest":
            prefix = "Loading latest run"
        else:
            prefix = "Loading history"
        self._emit_history_state(status_text=f"{prefix}: {target_label}", selected_uid=uid)

        future = executor.submit(self._history_load_task, uid, request_kind)
        future.add_done_callback(lambda fut, token=token, uid=uid, kind=request_kind:
                                 self._emit_history_load_result(token, fut, uid=uid, request_kind=kind))

    def _emit_history_load_result(self, token, future, *, uid='', request_kind='history'):
        payload = {
            "uid": uid,
            "request_kind": request_kind,
            "snapshot": None,
            "entry": None,
            "error": "",
        }
        try:
            payload.update(dict(future.result() or {}))
        except Exception as exc:
            payload["error"] = str(exc)
        self._history_load_completed.emit(int(token), payload)

    def _on_history_load_completed(self, token, payload):
        if int(token) != int(self._history_load_token):
            return
        self._live_backfill_pending = False
        payload = dict(payload or {})
        request_kind = str(payload.get("request_kind", "history") or "history")

        if request_kind == "latest" and (self._live_run_active or self._provisional_live_context):
            # QueueServer can reveal an already-running plan without a Start
            # document. A pending latest-run read must not replace that live view.
            self._arm_current_running_item()
            self._emit_history_state(status_text="Live diffraction run")
            if self._tiled_enabled:
                self._backfill_retry_timer.start(5000)
            return

        error_text = str(payload.get("error", "") or "")
        if error_text:
            prefix = "Startup history unavailable" if request_kind == "startup" else "History load failed"
            uid = str(payload.get('uid') or '')
            if request_kind != 'startup' and uid:
                entry = payload.get('entry') or self._history_entry_for_uid(uid) or {'uid': uid, 'label': uid[:8]}
                self._merge_history_entry({**entry, 'load_error': error_text})
                # A selection is a navigation position even when reading it
                # fails. Otherwise arrows keep retrying it from the last plot.
                self._selected_history_uid = uid
                self._displayed_run_uid = uid
                self._reset_requested.emit({'run_title': 'Historical run unavailable',
                                            'show_count_rate_gauge': False})
                self._history_plot_message.emit(f"Failed run / data unavailable\n{entry.get('title') or entry.get('label') or uid}\n{error_text}")
            self._emit_history_state(status_text=f"{prefix}: {error_text}", selected_uid=uid or None)
            if (
                request_kind == "startup"
                and self._live_follow_enabled
                and str(payload.get("uid", "") or "") == str(self._current_live_run_uid or "")
            ):
                self._startup_history_retry_count += 1
                self._backfill_retry_timer.start(5000)
            return

        uid = str(payload.get("uid", "") or "")
        entry = payload.get("entry")
        if isinstance(entry, dict):
            self._merge_history_entry(entry)
        snapshot = payload.get("snapshot")
        if not isinstance(snapshot, dict):
            self._emit_history_state(status_text="History load failed: no snapshot returned")
            return

        if request_kind == "startup":
            if not self._live_follow_enabled:
                return
            if self._display_mode != "live":
                return
            if uid != str(self._current_live_run_uid or ""):
                return
            rows = list(snapshot.get("rows", []) or [])
            has_rows = bool(rows)
            pending_reason = str(snapshot.get("pending_reason", "") or "").strip()
            if has_rows:
                self._startup_history_retry_count = 0
            self._display_mode = "live"
            self._displayed_run_uid = uid
            try:
                self._apply_live_backfill_snapshot(snapshot)
            except Exception as exc:
                self._emit_history_state(status_text=f"Live diffraction run · history unavailable: {exc}")
                self._backfill_retry_timer.start(5000)
                return
            if has_rows:
                self._emit_history_state(status_text="Live diffraction run", selected_uid=uid)
            else:
                status_text = "Live diffraction run (waiting for stored data)"
                if pending_reason:
                    status_text = f"Live diffraction run ({pending_reason})"
                self._emit_history_state(
                    status_text=status_text,
                    selected_uid=uid,
                )
                self._startup_history_retry_count += 1
                self._backfill_retry_timer.start(5000)
            return

        if request_kind == "latest":
            if not self._live_follow_enabled:
                return
            self._render_history_snapshot(snapshot, historical=True)
            self._display_mode = "live"
            self._displayed_run_uid = uid
            self._emit_history_state(
                status_text="Latest diffraction run (waiting for next live run)",
                selected_uid=uid,
            )
            return

        self.show_history_snapshot(snapshot)

    def show_history_snapshot(self, snapshot):
        """Replay a loaded PSD/scalar snapshot on either existing plot widget.

        Call on the Qt thread. Data loading can happen in a background worker
        using analysis.diffraction_history.build_history_snapshot.
        """
        self._history_load_token += 1
        self._display_mode = "history"
        self._live_backfill_pending = False
        self._backfill_retry_timer.stop()
        self._live_follow_enabled = False
        self._render_history_snapshot(snapshot, historical=True, register_descriptors=True)
        self._displayed_run_uid = str(snapshot.get("uid", ""))
        stop = snapshot.get("stop") or {}
        exit_status = str(stop.get("exit_status", "") or "")
        status = "Viewing historical diffraction run"
        if exit_status and exit_status != "success":
            status += f" ({exit_status}: {stop.get('reason', '')})"
        self._emit_history_state(status_text=status, selected_uid=self._displayed_run_uid)

    def _history_refresh_task(self, autoload_live):
        catalog, catalog_name, mode = self._ensure_history_catalog()
        entries = self._fetch_recent_history_entries(catalog, limit=self._history_limit)
        return {
            "entries": entries,
            "catalog_name": catalog_name,
            "mode": mode,
            "autoload_live": bool(autoload_live),
        }

    def _history_load_task(self, uid, request_kind):
        catalog, catalog_name, mode = self._ensure_history_catalog()
        run = catalog[str(uid)]
        start_doc, stop_doc = _run_start_stop_docs(run)
        if not _is_diffraction_start_doc(start_doc):
            raise RuntimeError("Selected run is not tagged as diffraction data.")

        entry = _make_history_entry(uid, start_doc, stop_doc)
        try:
            if entry['status'] == 'Failed' and request_kind != 'startup':
                raise RuntimeError(f"Recorded acquisition failure: {entry['reason'] or 'No reason recorded'}")
            snapshot = self._build_history_snapshot(
                run, start_doc, stop_doc,
                allow_empty=(str(request_kind or "history") == "startup"),
            )
        except Exception as exc:
            return {'uid': str(uid), 'request_kind': request_kind, 'entry': entry, 'error': str(exc)}

        return {
            "uid": str(uid),
            "request_kind": str(request_kind or "history"),
            "catalog_name": catalog_name,
            "mode": mode,
            "entry": entry,
            "snapshot": snapshot,
        }

    def _ensure_history_catalog(self):
        desired_mode = str(get_bluesky_active_mode() or "").strip().lower()
        if self._history_catalog is not None and self._history_mode == desired_mode:
            return self._history_catalog, self._history_catalog_name, self._history_mode

        client = self._history_client
        if client is None:
            client = _build_tiled_client(get_tiled_uri())
            self._history_client = client
        preferred_names = _preferred_catalog_names_for_mode(desired_mode)
        catalog_name = _resolve_catalog_name(client, preferred_names)
        catalog = client[catalog_name]

        self._history_catalog = catalog
        self._history_catalog_name = str(catalog_name or "")
        self._history_mode = str(desired_mode or "")
        return self._history_catalog, self._history_catalog_name, self._history_mode

    def _fetch_recent_history_entries(self, catalog, *, limit):
        return fetch_recent_history_entries(catalog, limit=limit)

    def _merge_history_entry(self, entry):
        entry = dict(entry or {})
        uid = str(entry.get("uid", "") or "").strip()
        if not uid:
            return
        merged = [item for item in self._history_entries if str(item.get("uid", "")) != uid]
        merged.append(entry)
        merged.sort(
            key=lambda item: (
                item.get("time") is None,
                -(float(item.get("time")) if item.get("time") is not None else 0.0),
            )
        )
        self._history_entries = merged[: self._history_limit]

    def _history_entry_for_uid(self, uid):
        uid = str(uid or "")
        for entry in self._history_entries:
            if str(entry.get("uid", "")) == uid:
                return entry
        return None

    def _history_index_for_uid(self, uid):
        uid = str(uid or "")
        for index, entry in enumerate(self._history_entries):
            if str(entry.get("uid", "")) == uid:
                return index
        return -1

    def _current_selected_history_uid(self):
        if self._display_mode == 'history' and self._selected_history_uid:
            return self._selected_history_uid
        if self._displayed_run_uid:
            return str(self._displayed_run_uid)
        if self._current_live_run_uid:
            return str(self._current_live_run_uid)
        if self._history_entries:
            return str(self._history_entries[0].get("uid", ""))
        return ""

    def _on_history_previous_requested(self):
        if not self._history_entries:
            return
        current_index = self._history_index_for_uid(self._current_selected_history_uid())
        if current_index < 0:
            current_index = 0
        next_index = current_index + 1
        if next_index >= len(self._history_entries):
            return
        uid = str(self._history_entries[next_index].get("uid", "") or "")
        if uid:
            self._request_history_load(uid, request_kind="history")

    def _on_history_next_requested(self):
        if not self._history_entries:
            return
        current_index = self._history_index_for_uid(self._current_selected_history_uid())
        if current_index <= 0:
            return
        uid = str(self._history_entries[current_index - 1].get("uid", "") or "")
        if uid:
            self._request_history_load(uid, request_kind="history")

    def _on_history_run_selected(self, uid):
        uid = str(uid or "").strip()
        if not uid:
            return
        if self._display_mode == "history" and uid == self._displayed_run_uid:
            return
        self._request_history_load(uid, request_kind="history")

    def _on_tiled_enabled_toggled(self, enabled):
        self._history_access_changed = True
        if demo_active():
            self._tiled_enabled = False
            self._emit_history_state(status_text="Storage disabled in demo mode")
            return
        enabled = bool(enabled)
        if enabled == self._tiled_enabled:
            return
        if enabled:
            if not callable(ensure_tiled_login):
                self._emit_history_state(status_text="Tiled login is unavailable in this environment.")
                return
            client = ensure_tiled_login(parent=self.widget)
            if client is None:
                self._emit_history_state(status_text="Live only (Tiled off)")
                self._emit_history_state()
                return
            self._tiled_enabled = True
            self._history_client = client
            self._history_catalog = None
            self._history_catalog_name = ""
            self._history_mode = ""
            self._history_error = ""
            self._history_entries = []
            self._emit_history_state(status_text="Connecting to diffraction history...")
            self._refresh_history_async(autoload_live=True)
            return

        self._tiled_enabled = False
        self._live_backfill_pending = False
        self._backfill_retry_timer.stop()
        self._history_client = None
        self._history_catalog = None
        self._history_catalog_name = ""
        self._history_mode = ""
        self._history_error = ""
        self._history_entries = []
        self._history_refresh_token += 1
        self._history_load_token += 1
        self._on_live_follow_toggled(True)
        self._emit_history_state(status_text="Live only (Tiled off)")

    def _on_live_follow_toggled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._live_follow_enabled:
            return
        # A late history read must not switch live following off again.
        self._history_load_token += 1
        self._live_backfill_pending = False
        self._backfill_retry_timer.stop()
        if enabled:
            self._live_follow_enabled = True
            self._display_mode = "live"
            active_uid = self._read_active_run_uid(allow_cached=False)
            if (
                isinstance(self._current_live_start_doc, dict)
                and self._current_live_start_doc
                and active_uid
                and self._current_live_start_doc.get('uid') == active_uid
                and str(self._displayed_run_uid or "") != str(self._current_live_run_uid or "")
            ):
                self._configure_run_display(self._current_live_start_doc, historical=False)
            self._arm_current_running_item()
            self._emit_history_state(status_text="Live diffraction run" if active_uid else 'Live follow enabled · waiting for next diffraction run')
            self._request_startup_history_load()
            return
        self._live_follow_enabled = False
        self._display_mode = "history"
        self._disarm_live_subscriptions()
        self._clear_live_previews_requested.emit()
        self._emit_history_state(status_text="History mode")

    def _read_active_run_uid(self, *, allow_cached=False):
        # State is an EPICS mbbo: request its label, not its integer value.
        state = str(_safe_caget(RUN_STATE_STATUS_PV, as_string=True) or '').strip().upper()
        if state in {'IDLE', 'DONE', 'ABORTED', 'FAILED', 'CLOSED', 'STALE'}:
            return ''
        if state in {'RUNNING', 'PAUSED', 'SUSPENDED'}:
            value = _safe_caget(RUN_UID_STATUS_PV)
            if value and str(value).strip() != self._last_stopped_run_uid:
                return str(value).strip()
        # Start/Stop documents also establish activity if status PVs cannot
        # be reached. A retained RunUID PV by itself is not activity evidence.
        if self._live_run_active and (allow_cached or self._current_live_run_uid):
            return str(self._current_live_run_uid or "").strip()
        return ""

    def _select_history_stream(self, run, start_doc):
        return select_history_stream(run, start_doc, stream_name=self.stream_name)

    def _build_history_snapshot(self, run, start_doc, stop_doc, *, allow_empty=False):
        return build_history_snapshot(
            run, start_doc, stop_doc, stream_name=self.stream_name, allow_empty=allow_empty,
        )

    def _configure_run_display(self, start_doc, *, historical, preserve_live_previews=False):
        start_doc = dict(start_doc or {})
        self._detector_axis_cache.clear()
        self._run_uid = str(start_doc.get("uid", ""))
        self._selected_history_uid = self._run_uid
        self._displayed_run_uid = str(self._run_uid or "")
        self._run_title = str(start_doc.get("title", "") or "")
        self._plan_name = str(start_doc.get("plan_name", "") or "")
        self._data_type = _run_data_type(start_doc)
        self._live_plot_fields = _qualify_duplicate_field_labels(
            _start_live_plot_fields(start_doc)
        )
        acquisition_monitor = dict(start_doc.get("acquisition_monitor", {}) or {})
        elapsed_pv = str(acquisition_monitor.get("elapsed_pv", "") or "").strip()
        has_elapsed_field = any(
            str(spec.get("role", "signal") or "signal") == "elapsed_time"
            for spec in self._live_plot_fields.values()
        )
        if self._data_type == "scalar" and elapsed_pv and not has_elapsed_field:
            self._live_plot_fields["__scalar_elapsed_time"] = {
                "label": "Elapsed Time",
                "role": "elapsed_time",
                "units": "s",
                "transport": "ca",
                "pv": elapsed_pv,
            }
        self._run_detector_names = self._normalize_detector_names(start_doc.get("detectors"))
        self._motor_names = self._normalize_motor_names(start_doc.get("motors"))
        self._summary_mode = "point"
        self._summary_counter = 0
        self._primary_points_seen = 0
        self._rendered_primary_seq_nums.clear()
        self._live_primary_rows.clear()
        if not preserve_live_previews:
            with self._live_preview_lock:
                self._live_preview_profiles.clear()
                self._live_preview_values.clear()
        self._summary_path_last_point = None
        self._summary_path_distance = 0.0
        self._planned_summary_x = []
        self._live_total_counts.clear()
        self._live_elapsed_time.clear()
        self._live_scalar_counts.clear()
        self._scalar_count_origins.clear()
        self._scalar_sample_elapsed.clear()

        det_config = dict(start_doc.get("det_config", {}) or {})
        axis_min = det_config.get("position_x_min", -209.21799055746422)
        axis_max = det_config.get("position_x_max", 209.21799055746422)
        try:
            axis_min = float(axis_min)
            axis_max = float(axis_max)
        except Exception:
            axis_min = -209.21799055746422
            axis_max = 209.21799055746422
        self._detector_axis_bounds = {"min": axis_min, "max": axis_max}

        summary_mode, summary_x_label, summary_title = self._choose_summary_config()
        self._summary_mode = summary_mode
        self._summary_x_label = summary_x_label
        self._summary_title = summary_title
        self._planned_summary_x = self._build_planned_summary_x(start_doc)
        peak_title = self._choose_peak_title()

        scalar_signal_specs = [
            spec
            for spec in self._live_plot_fields.values()
            if str(spec.get("role", "signal") or "signal") == "signal"
        ]
        scalar_units = {
            str(spec.get("units", "") or "").strip()
            for spec in scalar_signal_specs
            if str(spec.get("units", "") or "").strip()
        }
        self._scalar_count_labels = {
            str(spec.get("label", "") or data_key)
            for data_key, spec in self._live_plot_fields.items()
            if str(spec.get("role", "signal") or "signal") == "signal"
            and str(spec.get("units", "") or "").strip().lower() == "counts"
        }
        self._scalar_elapsed_keys = {}
        elapsed_fields = {
            key: spec for key, spec in self._live_plot_fields.items()
            if str(spec.get("role", "") or "") == "elapsed_time"
        }
        for data_key, spec in self._live_plot_fields.items():
            label = str(spec.get("label", "") or data_key)
            if label not in self._scalar_count_labels:
                continue
            source = str(spec.get("source", "") or _infer_field_source(data_key, label))
            keys = [
                key for key, elapsed_spec in elapsed_fields.items()
                if source and (
                    str(elapsed_spec.get("source", "") or "") == source
                    or key == f"{source}_time"
                )
            ]
            if not keys and len(elapsed_fields) == 1:
                key = next(iter(elapsed_fields))
                # The injected acquisition-monitor PV belongs to its device.
                monitor_source = str(acquisition_monitor.get("device", "") or "")
                if key == "__scalar_elapsed_time" and (
                    not source or not monitor_source or source == monitor_source
                ):
                    keys = [key]
                elif not source:
                    keys = [key]
            self._scalar_elapsed_keys[label] = keys
        plan_args = dict(start_doc.get("plan_args", {}) or {})
        self._scalar_elapsed_fallback = _coerce_number(plan_args.get("acquire_time"))
        if self._scalar_elapsed_fallback is None:
            acquire_times = dict(det_config.get("acquire_times", {}) or {})
            valid_times = [
                value
                for value in (_coerce_number(item) for item in acquire_times.values())
                if value is not None and value > 0
            ]
            if valid_times:
                self._scalar_elapsed_fallback = max(valid_times)
        show_count_rate_gauge = bool(
            self._data_type == "scalar" and self._scalar_count_labels
        )
        if self._data_type == "scalar" and scalar_units == {"counts"}:
            summary_y_label = "Counts"
        elif self._data_type == "scalar" and len(scalar_signal_specs) == 1:
            scalar_spec = scalar_signal_specs[0]
            scalar_label = str(scalar_spec.get("label", "") or "Value")
            scalar_unit = str(scalar_spec.get("units", "") or "").strip()
            summary_y_label = (
                f"{scalar_label} ({scalar_unit})" if scalar_unit else scalar_label
            )
        elif self._data_type == "scalar" and len(scalar_units) == 1:
            summary_y_label = f"Value ({next(iter(scalar_units))})"
        elif self._data_type == "scalar":
            summary_y_label = "Value"
        else:
            summary_y_label = "Total Counts"

        run_title = self._run_title or self._plan_name or "Diffraction Run"
        if historical:
            run_title = f"{run_title} [Historical Run]"
        self._reset_requested.emit(
            {
                "run_title": run_title,
                "plot_mode": "scalar" if self._data_type == "scalar" else "1d",
                "show_count_rate_gauge": show_count_rate_gauge,
                "profile_title": "Current PSD Profile",
                "summary_title": self._summary_title,
                "peak_title": peak_title,
                "summary_x_label": self._summary_x_label,
                "summary_y_label": summary_y_label,
                "peak_y_label": "Peak Position",
                "profile_x_limits": (axis_min, axis_max),
                "summary_x_limits": self._build_summary_x_limits(start_doc),
                "peak_x_limits": self._build_summary_x_limits(start_doc),
            }
        )
        if (
            self._active
            and not historical
            and self._live_follow_enabled
            and self._display_mode == "live"
        ):
            if self._live_plot_fields:
                self._arm_live_plot_field_subscriptions(self._live_plot_fields)
            else:
                self._arm_live_subscriptions(self._run_detector_names)
            if self._live_subscriptions:
                self._armed_live_detectors = tuple(self._run_detector_names)
        self._merge_history_entry(_make_history_entry(self._run_uid, start_doc))
        self._emit_history_state(selected_uid=self._run_uid)

    def _register_descriptor_docs(self, descriptor_docs):
        for descriptor_doc in list(descriptor_docs or []):
            descriptor_doc = dict(descriptor_doc or {})
            descriptor_uid = str(descriptor_doc.get("uid", "") or "").strip()
            if not descriptor_uid:
                continue
            self._descriptor_stream[descriptor_uid] = str(descriptor_doc.get("name", "") or "")
            self._descriptor_run_start[descriptor_uid] = str(
                descriptor_doc.get("run_start", "") or ""
            )
            self._descriptor_data_keys[descriptor_uid] = set(
                str(key) for key in dict(descriptor_doc.get("data_keys", {}) or {}).keys()
            )

    def _render_history_row(self, *, data, seq_num):
        if seq_num is not None:
            self._live_primary_rows[int(seq_num)] = dict(data)
        if self._data_type == "scalar":
            summary_x = self._extract_summary_x(data=data, seq_num=seq_num)
            self._emit_scalar_values(data, x_value=summary_x, live=False)
            seq_num_value = _coerce_number(seq_num)
            if seq_num_value is not None:
                seq_num_int = int(seq_num_value)
                self._rendered_primary_seq_nums.add(seq_num_int)
                self._primary_points_seen = max(int(self._primary_points_seen), seq_num_int)
            else:
                self._primary_points_seen += 1
            return

        detector_fields = self._extract_detector_fields(data)
        if not detector_fields:
            return

        summary_x = self._extract_summary_x(data=data, seq_num=seq_num)
        for detector_name, fields in detector_fields.items():
            counts = _coerce_array(fields.get("counts"))
            if counts is None:
                continue

            position_x = _coerce_array(fields.get("position_x"))
            if position_x is None:
                position_x = self._detector_axis_cache.get(detector_name)
            if position_x is None or position_x.shape != counts.shape:
                position_x = self._build_axis_from_counts(counts)
            if position_x is None or position_x.shape != counts.shape:
                continue

            self._detector_axis_cache[detector_name] = position_x
            self._profile_updated.emit(str(detector_name), position_x, counts)

            total_counts = _coerce_number(fields.get("total_counts"))
            if total_counts is None:
                try:
                    total_counts = float(np.nansum(counts))
                except Exception:
                    total_counts = None
            if total_counts is None:
                continue

            self._summary_point_updated.emit(
                str(detector_name),
                float(summary_x),
                float(total_counts),
            )

            fitted_peak = _fit_peak_position(position_x, counts)
            if fitted_peak is not None:
                peak_center, peak_center_err = fitted_peak
                self._peak_point_updated.emit(
                    str(detector_name),
                    float(summary_x),
                    float(peak_center),
                    peak_center_err,
                )
        seq_num_value = _coerce_number(seq_num)
        if seq_num_value is not None:
            seq_num_int = int(seq_num_value)
            self._rendered_primary_seq_nums.add(seq_num_int)
            self._primary_points_seen = max(int(self._primary_points_seen), seq_num_int)
        else:
            self._primary_points_seen += 1

    def _render_history_snapshot(
        self,
        snapshot,
        *,
        historical,
        register_descriptors=False,
        preserve_live_subscriptions=False,
    ):
        snapshot = dict(snapshot or {})
        start_doc = dict(snapshot.get("start", {}) or {})
        descriptor_docs = list(snapshot.get("descriptor_docs", []) or [])
        rows = list(snapshot.get("rows", []) or [])
        seq_nums = list(snapshot.get("seq_nums", []) or [])

        if not preserve_live_subscriptions:
            self._disarm_live_subscriptions()
        self._clear_live_previews_requested.emit()
        self._configure_run_display(start_doc, historical=historical)
        if register_descriptors:
            self._register_descriptor_docs(descriptor_docs)
        for index, row in enumerate(rows):
            seq_num = seq_nums[index] if index < len(seq_nums) else (index + 1)
            self._render_history_row(data=dict(row or {}), seq_num=seq_num)
        entry = _make_history_entry(snapshot.get('uid'), start_doc, snapshot.get('stop'))
        export = snapshot.get('file_export') or {}
        entry.update(file_status=export.get('status', ''), storage_error=export.get('error', ''))
        self._merge_history_entry(entry)

    def _apply_live_backfill_snapshot(self, snapshot):
        snapshot = dict(snapshot or {})
        start_doc = dict(snapshot.get("start", {}) or {})
        descriptor_docs = list(snapshot.get("descriptor_docs", []) or [])
        rows = list(snapshot.get("rows", []) or [])
        seq_nums = list(snapshot.get("seq_nums", []) or [])
        uid = str(start_doc.get("uid", "") or "").strip()

        if not uid:
            return
        if self._current_live_run_uid and uid != self._current_live_run_uid:
            return
        same_run = str(self._run_uid or "") in {"", "__live_running__", uid}
        if not same_run and self._current_live_run_uid != uid:
            return

        # A document can arrive while Tiled is reading earlier points. Rebuild
        # the union in acquisition order so old points never become the newest
        # spectrum, and path coordinates/summary colors remain correct.
        merged = {int(seq_nums[index] if index < len(seq_nums) else index + 1): dict(row)
                  for index, row in enumerate(rows)}
        if (same_run and not self._provisional_live_context and uid == self._run_uid
                and start_doc == self._current_live_start_doc
                and set(merged).issubset(self._rendered_primary_seq_nums)):
            self._register_descriptor_docs(descriptor_docs)
            return
        if same_run:
            merged.update(self._live_primary_rows)
        self._configure_run_display(start_doc, historical=False, preserve_live_previews=same_run)
        rows = [merged[seq] for seq in sorted(merged)]
        seq_nums = sorted(merged)

        if (
            not self._run_uid
            or str(self._run_uid) == "__live_running__"
            or str(self._run_uid) == uid
        ):
            self._current_live_run_uid = uid
            self._displayed_run_uid = uid
            self._current_live_start_doc = dict(start_doc)
            self._run_uid = uid
            if str(start_doc.get("title", "") or "").strip():
                self._run_title = str(start_doc.get("title", "") or "")
            if str(start_doc.get("plan_name", "") or "").strip():
                self._plan_name = str(start_doc.get("plan_name", "") or "")
            self._provisional_live_context = False

        self._register_descriptor_docs(descriptor_docs)

        for index, row in enumerate(rows):
            seq_num = seq_nums[index] if index < len(seq_nums) else (index + 1)
            try:
                seq_num_value = int(seq_num)
            except Exception:
                seq_num_value = index + 1
            if seq_num_value in self._rendered_primary_seq_nums:
                continue
            self._render_history_row(data=dict(row or {}), seq_num=seq_num_value)

        # Replaying completed points resets the plot. Restore the most recent
        # in-progress values even when no new CA callback follows the reset.
        # CA callbacks can arrive during replay. Read their latest values now,
        # and order restoration before any subsequent callbacks, rather than
        # restoring a stale copy captured before the redraw.
        with self._live_preview_lock:
            for detector, (axis, counts) in tuple(self._live_preview_profiles.items()):
                self._emit_live_profile(detector, axis, counts)
            live_x = self._estimate_live_summary_x()
            for label, (previous_x, value) in tuple(self._live_preview_values.items()):
                if self._data_type == "scalar":
                    # Provisional previews use raw data keys until metadata
                    # supplies display labels. Keep one series after relabeling.
                    new_label = str(self._live_plot_fields.get(label, {}).get("label") or label)
                    if new_label != label:
                        self._live_preview_values.pop(label, None)
                        label = new_label
                self._emit_live_summary(label, live_x if live_x is not None else previous_x, value)

    def on_document(self, name, doc):
        # PyDM caches this screen during navigation. Continue consuming run
        # documents while it is hidden so switching back does not lose a run;
        # deactivate() separately suspends direct CA and history work.
        if str(name) == "start":
            start_doc = dict(doc or {})
            if not _is_diffraction_start_doc(start_doc):
                return
            if self._live_follow_enabled:
                self._history_load_token += 1
                self._live_backfill_pending = False
                self._backfill_retry_timer.stop()
            self._descriptor_stream.clear()
            self._descriptor_run_start.clear()
            self._descriptor_data_keys.clear()
            self._current_live_run_uid = str(start_doc.get("uid", "") or "")
            self._live_run_active = True
            self._current_live_start_doc = dict(start_doc)
            self._provisional_live_context = False
            self._merge_history_entry(_make_history_entry(self._current_live_run_uid, start_doc))
            self._schedule_history_refresh(delay_ms=1500)
            if not self._live_follow_enabled:
                self._emit_history_state(
                    status_text="History mode",
                    selected_uid=self._displayed_run_uid or self._current_live_run_uid,
                )
                return
            self._display_mode = "live"
            self._on_start(start_doc)
            self._emit_history_state(status_text="Live diffraction run", selected_uid=self._current_live_run_uid)
            return

        if name == "descriptor":
            self._register_descriptor_docs([doc])
            return

        if name == "event":
            if (not self._live_follow_enabled) or self._display_mode != "live":
                return
            self._process_event(
                descriptor_uid=str(doc.get("descriptor", "")),
                data=dict(doc.get("data", {}) or {}),
                seq_num=doc.get("seq_num"),
                timestamps=dict(doc.get("timestamps", {}) or {}),
            )
            return

        if name == "event_page":
            if (not self._live_follow_enabled) or self._display_mode != "live":
                return
            descriptor_uid = str(doc.get("descriptor", ""))
            page_data = dict(doc.get("data", {}) or {})
            page_timestamps = dict(doc.get("timestamps", {}) or {})
            seq_nums = list(doc.get("seq_num", []) or [])
            keys = list(page_data.keys())
            n_items = len(seq_nums)
            if not n_items:
                for key in keys:
                    try:
                        n_items = max(n_items, len(page_data.get(key, [])))
                    except Exception:
                        n_items = max(n_items, 1)
            for index in range(int(max(0, n_items))):
                row = {}
                for key in keys:
                    values = page_data.get(key, [])
                    try:
                        row[key] = values[index]
                    except Exception:
                        pass
                seq_num = seq_nums[index] if index < len(seq_nums) else None
                self._process_event(
                    descriptor_uid=descriptor_uid,
                    data=row,
                    seq_num=seq_num,
                    timestamps={
                        key: values[index] for key, values in page_timestamps.items()
                        if index < len(values)
                    },
                )
            return

        if name == "stop":
            if str(doc.get('run_start') or '') != str(self._current_live_run_uid or ''):
                return
            self._last_stopped_run_uid = str(doc.get('run_start') or '')
            self._live_backfill_pending = False
            self._backfill_retry_timer.stop()
            if self._live_follow_enabled:
                self._history_load_token += 1
            if str(doc.get('run_start') or '') == str(self._current_live_run_uid or ''):
                self._live_run_active = False
                self._current_live_run_uid = ''
            if (not self._live_follow_enabled) or self._display_mode != "live":
                return
            exit_status = str(doc.get("exit_status", "") or "unknown")
            status_title = self._run_title or self._plan_name or "Diffraction Run"
            self._current_live_run_uid = ""
            self._disarm_live_subscriptions()
            self._clear_live_previews_requested.emit()
            self._status_updated.emit(f"{status_title} [{exit_status}]")
            self._schedule_history_refresh(delay_ms=2000)

    def _on_start(self, doc):
        self._configure_run_display(doc, historical=False)

    def _on_running_item_changed(self, event):
        if (not self._live_follow_enabled) or self._display_mode != "live":
            self._disarm_live_subscriptions()
            return
        running_item = dict(getattr(event, "running_item", {}) or {})
        if not self._ensure_live_context_from_running_item(running_item):
            self._disarm_live_subscriptions()
            return
        detector_names = self._extract_live_detector_names_from_running_item(running_item)
        if (
            not detector_names
            and self._armed_live_detectors
            and (self._current_live_run_uid or self._provisional_live_context)
        ):
            return
        if self._live_plot_fields:
            self._arm_live_plot_field_subscriptions(self._live_plot_fields)
        else:
            self._arm_live_subscriptions(detector_names)

    def _process_event(self, *, descriptor_uid, data, seq_num, timestamps=None):
        detector_fields = {}
        if self._data_type != "scalar":
            detector_fields = self._extract_detector_fields(data)
            if not detector_fields:
                return

        descriptor_uid = str(descriptor_uid or "")
        descriptor_known = (
            descriptor_uid in self._descriptor_stream
            or descriptor_uid in self._descriptor_run_start
            or descriptor_uid in self._descriptor_data_keys
        )
        if descriptor_known:
            if not self._accept_descriptor_run(descriptor_uid):
                return
            is_primary_stream = self._is_primary_descriptor(descriptor_uid)
        else:
            if (
                (not self._live_follow_enabled)
                or self._display_mode != "live"
                or (
                    (not self._provisional_live_context)
                    and not str(self._current_live_run_uid or "").strip()
                )
            ):
                return
            # On a mid-run connection, monitor descriptors may have been sent
            # before this GUI existed. Their seq_nums belong to separate
            # streams and must never be deduplicated as completed scan points.
            # Until we know the descriptor, show a live preview instead.
            is_primary_stream = False
            if self._tiled_enabled and not self._live_backfill_pending and not self._backfill_retry_timer.isActive():
                self._backfill_retry_timer.start(5000)

        summary_x = None
        if is_primary_stream:
            try:
                seq_num_value = int(seq_num) if seq_num is not None else None
            except Exception:
                seq_num_value = None
            if seq_num_value is not None and seq_num_value in self._rendered_primary_seq_nums:
                return
            with self._live_preview_lock:
                self._live_preview_profiles.clear()
                self._live_preview_values.clear()
            if seq_num_value is not None:
                if seq_num_value > max(self._rendered_primary_seq_nums, default=0) + 1 and self._tiled_enabled:
                    self._backfill_retry_timer.start(0)
                self._live_primary_rows[seq_num_value] = dict(data)
            summary_x = self._extract_summary_x(data=data, seq_num=seq_num)

        if self._data_type == "scalar":
            if is_primary_stream:
                self._emit_scalar_values(data, x_value=summary_x, live=False, timestamps=timestamps)
                if seq_num_value is not None:
                    self._rendered_primary_seq_nums.add(int(seq_num_value))
                    self._primary_points_seen = max(
                        int(self._primary_points_seen),
                        int(seq_num_value),
                    )
                else:
                    self._primary_points_seen += 1
            else:
                live_x = self._estimate_live_summary_x()
                if live_x is not None:
                    self._emit_scalar_values(data, x_value=live_x, live=True, timestamps=timestamps)
            return

        for detector_name, fields in detector_fields.items():
            if not is_primary_stream:
                counts = _coerce_array(fields.get("counts"))
                position_x = _coerce_array(fields.get("position_x"))
                if position_x is not None:
                    self._detector_axis_cache[detector_name] = position_x
                if counts is not None:
                    self._handle_live_counts_update(detector_name, counts)
                if fields.get("total_counts") is not None:
                    self._handle_live_total_counts_update(detector_name, fields["total_counts"])
                continue
            counts = _coerce_array(fields.get("counts"))
            if counts is None:
                continue

            position_x = _coerce_array(fields.get("position_x"))
            if position_x is None:
                position_x = self._detector_axis_cache.get(detector_name)
            if position_x is None or position_x.shape != counts.shape:
                position_x = self._build_axis_from_counts(counts)
            if position_x is None or position_x.shape != counts.shape:
                continue

            self._detector_axis_cache[detector_name] = position_x
            self._profile_updated.emit(str(detector_name), position_x, counts)
            total_counts = _coerce_number(fields.get("total_counts"))
            if total_counts is None:
                try:
                    total_counts = float(np.nansum(counts))
                except Exception:
                    total_counts = None
            if total_counts is None:
                continue
            self._summary_point_updated.emit(str(detector_name), float(summary_x), float(total_counts))

            fitted_peak = _fit_peak_position(position_x, counts)
            if fitted_peak is not None:
                peak_center, peak_center_err = fitted_peak
                self._peak_point_updated.emit(
                    str(detector_name),
                    float(summary_x),
                    float(peak_center),
                    peak_center_err,
                )
        if is_primary_stream:
            if seq_num_value is not None:
                self._rendered_primary_seq_nums.add(int(seq_num_value))
                self._primary_points_seen = max(int(self._primary_points_seen), int(seq_num_value))
            else:
                self._primary_points_seen += 1

    def _accept_descriptor_run(self, descriptor_uid):
        run_uid = self._descriptor_run_start.get(str(descriptor_uid), "")
        if self._provisional_live_context and run_uid:
            self._current_live_run_uid = str(run_uid)
            self._displayed_run_uid = str(run_uid)
            self._run_uid = str(run_uid)
            if isinstance(self._current_live_start_doc, dict) and self._current_live_start_doc:
                self._current_live_start_doc["uid"] = str(run_uid)
            self._provisional_live_context = False
        if self._run_uid and run_uid and run_uid != self._run_uid:
            return False
        return self._descriptor_has_detector_fields(descriptor_uid)

    @staticmethod
    def _normalize_detector_names(detectors):
        if detectors is None:
            return []
        if isinstance(detectors, str):
            text = detectors.strip()
            if not text:
                return []
            if text[0] in ("[", "(") and text[-1] in ("]", ")"):
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes)):
                    return [str(item).strip() for item in parsed if str(item).strip()]
            return [part.strip() for part in text.split(",") if part.strip()]
        if isinstance(detectors, Iterable):
            return [str(item).strip() for item in detectors if str(item).strip()]
        text = str(detectors).strip()
        return [text] if text else []

    def _extract_live_detector_names_from_running_item(self, running_item):
        if not running_item:
            return []
        if str(running_item.get("item_type", "") or "plan") != "plan":
            return []

        kwargs = dict(running_item.get("kwargs", {}) or {})
        detector_names = self._normalize_detector_names(kwargs.get("detectors"))
        if not detector_names:
            detector_names = self._normalize_detector_names(kwargs.get("detector"))

        live_detector_names = []
        for detector_name in detector_names:
            if self._get_live_counts_pv(detector_name):
                live_detector_names.append(detector_name)
        return live_detector_names

    def _ensure_live_context_from_running_item(self, running_item):
        if not running_item:
            return False
        if str(running_item.get("item_type", "") or "plan") != "plan":
            return False

        active_uid = str(self._read_active_run_uid(allow_cached=False) or "").strip()
        if (not active_uid and self._run_uid and not self._current_live_run_uid
                and not self._provisional_live_context):
            # QueueServer polls can still contain the finished plan after its
            # Stop. Only an active run identity or a new Start should replace
            # an established plot with a new acquisition context.
            return False
        provisional_uid = active_uid or str(self._current_live_run_uid or "").strip() or "__live_running__"
        if active_uid:
            self._current_live_run_uid = active_uid
            self._live_run_active = True

        if str(self._run_uid or "") == provisional_uid and self._current_live_start_doc:
            return True

        start_doc = {}
        if (
            isinstance(self._current_live_start_doc, dict)
            and str(self._current_live_start_doc.get("uid", "") or "").strip() == provisional_uid
        ):
            start_doc = dict(self._current_live_start_doc)
        else:
            start_doc = self._build_start_doc_from_running_item(running_item, uid=provisional_uid)

        if not start_doc:
            return False
        self._current_live_start_doc = dict(start_doc)
        self._provisional_live_context = not bool(active_uid)
        if str(self._run_uid or "") != provisional_uid:
            self._configure_run_display(start_doc, historical=False)
        return True

    def _build_start_doc_from_running_item(self, running_item, *, uid):
        running_item = dict(running_item or {})
        if str(running_item.get("item_type", "") or "plan") != "plan":
            return {}

        kwargs = dict(running_item.get("kwargs", {}) or {})
        plan_name = str(running_item.get("name", "") or "").strip()
        motors = self._extract_motor_names_from_running_item(running_item)
        plan_pattern_args = {}
        for key in (
            "motor",
            "motor1",
            "motor2",
            "motor_outer",
            "motor_inner",
            "start_pos",
            "stop_pos",
            "step_size",
            "num_steps",
            "start_pos1",
            "stop_pos1",
            "step_size1",
            "start_pos2",
            "stop_pos2",
            "step_size2",
            "start_pos_outer",
            "stop_pos_outer",
            "step_size_outer",
            "num_steps_outer",
            "start_pos_inner",
            "stop_pos_inner",
            "step_size_inner",
            "num_steps_inner",
            "position_list",
        ):
            if kwargs.get(key) is not None:
                plan_pattern_args[key] = kwargs.get(key)

        start_doc = {
            "uid": str(uid or "").strip(),
            "plan_name": plan_name,
            "experiment_type": DIFFRACTION_EXPERIMENT_TYPE,
        }
        detectors = self._normalize_detector_names(kwargs.get("detectors") or kwargs.get("detector"))
        if detectors:
            start_doc["detectors"] = detectors
        title = str(kwargs.get("title", "") or "").strip()
        if title:
            start_doc["title"] = title
        if motors:
            start_doc["motors"] = list(motors)
        if plan_pattern_args:
            start_doc["plan_pattern_args"] = dict(plan_pattern_args)

        num_points = kwargs.get("num_points", None)
        if num_points is None:
            num_points = kwargs.get("num_steps", None)
        if num_points is None and str(plan_name) in {"count_he3", "count_he3psd"}:
            num_points = kwargs.get("num_exposures", None)
        try:
            if num_points is not None:
                start_doc["num_points"] = int(num_points)
        except Exception:
            pass
        return start_doc

    def _extract_motor_names_from_running_item(self, running_item):
        kwargs = dict(dict(running_item or {}).get("kwargs", {}) or {})
        motors = self._normalize_motor_names(kwargs.get("motors"))
        if not motors:
            motors = self._normalize_motor_names(kwargs.get("motor"))
        if motors:
            return motors

        ordered_keys = ("motor1", "motor2", "motor_outer", "motor_inner")
        extracted = []
        for key in ordered_keys:
            value = str(kwargs.get(key, "") or "").strip()
            if value:
                extracted.append(value)
        return extracted

    def _arm_live_subscriptions(self, detector_names):
        unique_names = tuple(dict.fromkeys(detector_names))
        if unique_names == self._armed_live_detectors and self._live_subscriptions:
            return
        fields = {}
        for detector_name in unique_names:
            counts_pv_name = self._get_live_counts_pv(detector_name)
            if counts_pv_name:
                fields[f"{detector_name}_counts"] = {
                    "pv": counts_pv_name,
                    "transport": "ca",
                    "role": "profile",
                    "series": detector_name,
                }
            total_counts_pv_name = self._get_live_total_counts_pv(detector_name)
            if total_counts_pv_name:
                fields[f"{detector_name}_total_counts"] = {
                    "pv": total_counts_pv_name,
                    "transport": "ca",
                    "role": "summary",
                    "series": detector_name,
                }
        self._arm_live_plot_field_subscriptions(fields)
        self._armed_live_detectors = unique_names

    def _arm_live_plot_field_subscriptions(self, fields):
        ca_fields = {
            str(data_key): dict(spec or {})
            for data_key, spec in dict(fields or {}).items()
            if str(spec.get("transport", "") or "").lower() == "ca"
            and str(spec.get("pv", "") or "").strip()
        }
        signature = tuple(
            sorted(
                (
                    data_key,
                    str(spec.get("pv", "") or ""),
                    str(spec.get("role", "signal") or "signal"),
                    str(spec.get("series", "") or ""),
                )
                for data_key, spec in ca_fields.items()
            )
        )
        if signature == self._armed_live_field_signature:
            return

        self._disarm_live_subscriptions()
        self._armed_live_field_signature = signature
        if PV is None:
            return

        for data_key, spec in ca_fields.items():
            pv_name = str(spec.get("pv", "") or "")
            role = str(spec.get("role", "signal") or "signal")
            series = str(spec.get("series", "") or spec.get("label", "") or data_key)
            try:
                pv = PV(pv_name, auto_monitor=True)
                if role == "profile":
                    callback = (
                        lambda pvname=None, value=None, char_value=None, _series=series, **kwargs:
                        self._handle_live_counts_update(_series, value)
                    )
                elif role == "summary":
                    callback = (
                        lambda pvname=None, value=None, char_value=None, _series=series, **kwargs:
                        self._handle_live_total_counts_update(_series, value)
                    )
                elif role == "elapsed_time":
                    callback = (
                        lambda pvname=None, value=None, char_value=None, _key=data_key, **kwargs:
                        self._handle_live_elapsed_time_update(_key, value)
                    )
                else:
                    callback = (
                        lambda pvname=None, value=None, char_value=None, _series=series, **kwargs:
                        self._handle_live_scalar_update(_series, value)
                    )
                callback_index = pv.add_callback(callback)
                self._live_subscriptions[data_key] = [
                    {"pv": pv, "callback_index": callback_index}
                ]
            except Exception:
                continue

    def _disarm_live_subscriptions(self):
        for subscription_group in self._live_subscriptions.values():
            for subscription in subscription_group:
                pv = subscription.get("pv")
                callback_index = subscription.get("callback_index")
                if pv is not None and callback_index is not None:
                    try:
                        pv.remove_callback(callback_index)
                    except Exception:
                        pass
        self._live_subscriptions.clear()
        self._live_total_counts.clear()
        self._live_elapsed_time.clear()
        self._live_scalar_counts.clear()
        self._armed_live_detectors = ()
        self._armed_live_field_signature = ()

    def _handle_live_counts_update(self, detector_name, value):
        if (not self._live_follow_enabled) or self._display_mode != "live":
            return
        counts = _coerce_array(value)
        if counts is None:
            return

        position_x = self._detector_axis_cache.get(detector_name)
        if position_x is None or position_x.shape != counts.shape:
            position_x = self._build_axis_from_counts(counts)
        if position_x is None or position_x.shape != counts.shape:
            return

        self._detector_axis_cache[detector_name] = position_x
        self._emit_live_profile(str(detector_name), position_x, counts)
        should_emit_summary = True
        total_counts = _coerce_number(self._live_total_counts.get(detector_name))
        if total_counts is None:
            total_counts = _coerce_number(np.nansum(counts))
        elif self._get_live_total_counts_pv(detector_name) or any(
            str(spec.get("role", "") or "") == "summary"
            and str(spec.get("series", "") or "") == str(detector_name)
            for spec in self._live_plot_fields.values()
        ):
            should_emit_summary = False
        summary_x = self._estimate_live_summary_x()
        if should_emit_summary and total_counts is not None and summary_x is not None:
            self._emit_live_summary(
                str(detector_name),
                float(summary_x),
                float(total_counts),
            )

    def _handle_live_scalar_update(self, series_name, value):
        if (not self._live_follow_enabled) or self._display_mode != "live":
            return
        scalar_value = _coerce_number(value)
        summary_x = self._estimate_live_summary_x()
        if scalar_value is None or summary_x is None:
            return
        self._emit_live_summary(
            str(series_name),
            float(summary_x),
            float(scalar_value),
        )
        if str(series_name) in self._scalar_count_labels:
            self._live_scalar_counts[str(series_name)] = float(scalar_value)
            self._scalar_readout_updated.emit(
                str(series_name),
                float(scalar_value),
                self._current_scalar_elapsed_time(str(series_name)),
            )

    def _handle_live_elapsed_time_update(self, data_key, value):
        self._set_live_scalar_elapsed(data_key, value)

    def _set_live_scalar_elapsed(self, data_key, value):
        elapsed = _coerce_number(value)
        if elapsed is None:
            return
        self._live_elapsed_time[str(data_key)] = float(elapsed)
        for series_name, total_counts in self._live_scalar_counts.items():
            self._scalar_readout_updated.emit(
                str(series_name),
                float(total_counts),
                self._current_scalar_elapsed_time(str(series_name)),
            )

    def _current_scalar_elapsed_time(self, series_name):
        for key in self._scalar_elapsed_keys.get(str(series_name), []):
            value = _coerce_number(self._live_elapsed_time.get(key))
            if value is not None:
                return max(0.0, float(value))
        value = self._scalar_sample_elapsed.get(str(series_name))
        if value is not None:
            return value
        return self._scalar_elapsed_fallback

    def _track_scalar_count_time(self, data_key, label, value, timestamps):
        """Time soft-counter exposures from their reset when older workers
        publish counts without an elapsed-time monitor.
        """
        stamp = _coerce_number(dict(timestamps or {}).get(data_key))
        if stamp is None:
            return
        previous = self._scalar_count_origins.get(label)
        if value == 0 or (previous is not None and value < previous[1]):
            origin = stamp
        elif previous is not None:
            origin = previous[0]
        else:
            return  # Wait for a reset rather than guess an exposure start.
        self._scalar_count_origins[label] = (origin, float(value))
        self._scalar_sample_elapsed[label] = max(0.0, stamp - origin)

    @staticmethod
    def _get_live_counts_pv(detector_name):
        mapping = {
            "he3psd0": f"{pv_prefix()}he3PSD:Det0:LiveCounts",
            "he3psd7": f"{pv_prefix()}he3PSD:Det7:LiveCounts",
        }
        return mapping.get(str(detector_name))

    @staticmethod
    def _get_live_total_counts_pv(detector_name):
        mapping = {
            "he3psd0": f"{pv_prefix()}he3PSD:Det0:LiveTotalCounts",
            "he3psd7": f"{pv_prefix()}he3PSD:Det7:LiveTotalCounts",
        }
        return mapping.get(str(detector_name))

    def _handle_live_total_counts_update(self, detector_name, value):
        if (not self._live_follow_enabled) or self._display_mode != "live":
            return
        total_counts = _coerce_number(value)
        if total_counts is None:
            return
        self._live_total_counts[detector_name] = float(total_counts)
        summary_x = self._estimate_live_summary_x()
        if summary_x is None:
            return
        self._emit_live_summary(
            str(detector_name),
            float(summary_x),
            float(total_counts),
        )

    def _emit_live_profile(self, detector_name, axis, counts):
        with self._live_preview_lock:
            self._live_preview_profiles[detector_name] = (axis, counts)
            self._live_profile_updated.emit(detector_name, axis, counts)

    def _emit_live_summary(self, label, x_value, value):
        with self._live_preview_lock:
            self._live_preview_values[label] = (x_value, value)
            self._live_summary_point_updated.emit(label, float(x_value), float(value))

    def _estimate_live_summary_x(self):
        if self._summary_mode == "exposure":
            return float(int(self._primary_points_seen) + 1)
        if self._summary_mode == "motor":
            index = int(self._primary_points_seen)
            if index < len(self._planned_summary_x):
                return _coerce_number(self._planned_summary_x[index])
        return None

    def _build_planned_summary_x(self, start_doc):
        if self._summary_mode != "motor":
            return []
        plan_pattern_args = dict(start_doc.get("plan_pattern_args", {}) or {})
        start_pos = _coerce_number(plan_pattern_args.get("start_pos"))
        stop_pos = _coerce_number(plan_pattern_args.get("stop_pos"))
        num_steps = plan_pattern_args.get("num_steps")
        try:
            num_steps = int(num_steps)
        except Exception:
            num_steps = None
        if (
            start_pos is None
            or stop_pos is None
            or num_steps is None
            or num_steps <= 0
        ):
            return []
        if num_steps == 1:
            return [float(start_pos)]
        return list(np.linspace(float(start_pos), float(stop_pos), int(num_steps)))

    def _build_summary_x_limits(self, start_doc):
        if self._planned_summary_x:
            low = min(self._planned_summary_x)
            high = max(self._planned_summary_x)
            return DiffractionPlotWidget._normalize_axis_limits((low, high))

        try:
            num_points = int(start_doc.get("num_points"))
        except Exception:
            num_points = None
        if num_points is not None and num_points > 0:
            return DiffractionPlotWidget._normalize_axis_limits((1.0, float(num_points)))
        return None

    def _is_primary_descriptor(self, descriptor_uid):
        stream = self._descriptor_stream.get(str(descriptor_uid), "")
        if not self.stream_name:
            return True
        return stream == self.stream_name

    def _descriptor_has_detector_fields(self, descriptor_uid):
        data_keys = self._descriptor_data_keys.get(str(descriptor_uid), set())
        if self._data_type == "scalar":
            declared = {
                data_key
                for data_key, spec in self._live_plot_fields.items()
                if str(spec.get("role", "signal") or "signal") in {"signal", "elapsed_time"}
            }
            if declared.intersection(data_keys):
                return True
            return any(
                key == detector_name or key.startswith(f"{detector_name}_")
                for key in data_keys
                for detector_name in self._run_detector_names
            )
        for key in data_keys:
            if key.endswith("_counts") or key.endswith("_position_x") or key.endswith("_total_counts"):
                return True
        return False

    @staticmethod
    def _normalize_motor_names(motors):
        if motors is None:
            return []
        if isinstance(motors, str):
            text = motors.strip()
            if not text:
                return []
            if text[0] in ("[", "(") and text[-1] in ("]", ")"):
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None
                if isinstance(parsed, Iterable) and not isinstance(parsed, (str, bytes)):
                    out = []
                    for item in parsed:
                        item_text = str(item or "").strip()
                        if item_text:
                            out.append(item_text)
                    if out:
                        return out
            return [text]
        if isinstance(motors, Iterable):
            out = []
            for item in motors:
                text = str(item or "").strip()
                if text:
                    out.append(text)
            return out
        text = str(motors).strip()
        return [text] if text else []

    def _choose_summary_config(self):
        is_scalar = self._data_type == "scalar"
        scalar_labels = list(
            dict.fromkeys(
                str(spec.get("label", "") or data_key)
                for data_key, spec in self._live_plot_fields.items()
                if str(spec.get("role", "signal") or "signal") == "signal"
            )
        )
        if is_scalar and len(scalar_labels) == 1:
            value_name = scalar_labels[0]
        elif is_scalar:
            value_name = "Scalar Values"
        else:
            value_name = "Total Counts"
        if self._plan_name in {"count_he3", "count_he3psd", "count_scalar"}:
            return "exposure", "Acquisition Number", f"{value_name} vs Acquisition Number"
        if len(self._motor_names) == 1:
            return "motor", self._motor_names[0], f"{value_name} vs Position"
        if len(self._motor_names) > 1:
            return "path", "Path Length", f"{value_name} vs Scan Path"
        return "point", "Point", f"{value_name} vs Point"

    def _choose_peak_title(self):
        if self._summary_mode == "exposure":
            return "Fitted Peak Position vs Acquisition Number"
        if self._summary_mode == "motor":
            return "Fitted Peak Position vs Motor Position"
        if self._summary_mode == "path":
            return "Fitted Peak Position vs Scan Path"
        return "Fitted Peak Position vs Point"

    def _extract_summary_x(self, *, data, seq_num):
        if self._summary_mode == "motor" and len(self._motor_names) == 1:
            value = _coerce_number(data.get(self._motor_names[0]))
            if value is not None:
                return value
        if self._summary_mode == "path" and len(self._motor_names) > 1:
            value = self._extract_scan_path_x(data)
            if value is not None:
                return value
        if seq_num is not None:
            value = _coerce_number(seq_num)
            if value is not None:
                return value
        self._summary_counter += 1
        return float(self._summary_counter)

    def _extract_scan_path_x(self, data):
        point = []
        for motor_name in self._motor_names:
            value = _coerce_number(data.get(motor_name))
            if value is None:
                return None
            point.append(value)
        current = np.asarray(point, dtype=float)
        if self._summary_path_last_point is None:
            self._summary_path_last_point = current
            self._summary_path_distance = 0.0
            return 0.0
        self._summary_path_distance += float(
            np.linalg.norm(current - self._summary_path_last_point)
        )
        self._summary_path_last_point = current
        return self._summary_path_distance

    def _extract_scalar_values(self, data):
        data = dict(data or {})
        values = []
        for data_key, spec in self._live_plot_fields.items():
            if str(spec.get("role", "signal") or "signal") != "signal":
                continue
            value = _coerce_number(data.get(data_key))
            if value is None:
                continue
            label = str(spec.get("label", "") or data_key)
            values.append((label, value))
        if values:
            return values

        # Compatibility fallback for scalar runs written before live field
        # metadata was added. Restrict candidates to the declared detectors.
        for data_key, raw_value in data.items():
            data_key = str(data_key)
            if data_key in self._motor_names:
                continue
            if self._run_detector_names and not any(
                data_key == detector_name or data_key.startswith(f"{detector_name}_")
                for detector_name in self._run_detector_names
            ):
                continue
            if data_key.endswith("_time") or data_key.endswith("_clock_counts"):
                continue
            value = _coerce_number(raw_value)
            if value is not None:
                values.append((data_key, value))
        return values

    def _emit_scalar_values(self, data, *, x_value, live, timestamps=None):
        x_value = _coerce_number(x_value)
        if x_value is None:
            return
        data = dict(data or {})
        emit_value = self._emit_live_summary if live else self._summary_point_updated.emit
        for data_key, spec in self._live_plot_fields.items():
            if str(spec.get("role", "signal") or "signal") != "elapsed_time":
                continue
            elapsed_time = _coerce_number(data.get(data_key))
            if elapsed_time is not None:
                self._set_live_scalar_elapsed(data_key, elapsed_time)
        for label, value in self._extract_scalar_values(data):
            emit_value(str(label), float(x_value), float(value))
            if str(label) in self._scalar_count_labels:
                if live:
                    self._live_scalar_counts[str(label)] = float(value)
                    for data_key, spec in self._live_plot_fields.items():
                        if (
                            str(spec.get("role", "signal") or "signal") == "signal"
                            and str(spec.get("label", "") or data_key) == str(label)
                        ):
                            self._track_scalar_count_time(data_key, str(label), value, timestamps)
                elapsed_time = self._current_scalar_elapsed_time(str(label))
                if not live and not self._scalar_elapsed_keys.get(str(label)):
                    # The committed value covers the complete configured dwell.
                    elapsed_time = self._scalar_elapsed_fallback or elapsed_time
                self._scalar_readout_updated.emit(
                    str(label),
                    float(value),
                    elapsed_time,
                )

    @staticmethod
    def _extract_detector_fields(data):
        fields = {}
        for key, value in dict(data or {}).items():
            name = str(key)
            if name.endswith("_total_counts"):
                detector_name = name[: -len("_total_counts")]
                fields.setdefault(detector_name, {})["total_counts"] = value
            elif name.endswith("_position_x"):
                detector_name = name[: -len("_position_x")]
                fields.setdefault(detector_name, {})["position_x"] = value
            elif name.endswith("_counts"):
                detector_name = name[: -len("_counts")]
                fields.setdefault(detector_name, {})["counts"] = value
        return fields

    def _build_axis_from_counts(self, counts):
        nbins = counts.shape[0]
        axis_min = self._detector_axis_bounds.get("min", -209.21799055746422)
        axis_max = self._detector_axis_bounds.get("max", 209.21799055746422)
        return _build_linear_axis(nbins=nbins, axis_min=axis_min, axis_max=axis_max)
