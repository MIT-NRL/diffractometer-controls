import math
import ast
from collections.abc import Iterable

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.figure
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

try:
    from lmfit.models import GaussianModel, LinearModel
except Exception:
    GaussianModel = None
    LinearModel = None

try:
    from epics import PV
except Exception:
    PV = None


def _initialize_matplotlib():
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot  # noqa: F401


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

        self.figure = matplotlib.figure.Figure(constrained_layout=True)
        self.profile_axes, self.summary_axes, self.peak_axes = self._create_axes()

        canvas = FigureCanvas(self.figure)
        canvas.setMinimumWidth(640)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        canvas.updateGeometry()
        canvas.setParent(self)
        self._canvas = canvas
        self._toolbar = NavigationToolbar(canvas, parent=self)

        layout = QVBoxLayout()
        layout.addWidget(canvas)
        layout.addWidget(self._toolbar)
        self.setLayout(layout)

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

    @QtCore.Slot(object)
    def reset(self, config=None):
        config = dict(config or {})
        run_title = str(config.get("run_title", "") or "")
        profile_title = str(config.get("profile_title", "Current Spectrum") or "Current Spectrum")
        summary_title = str(config.get("summary_title", "Total Counts") or "Total Counts")
        peak_title = str(
            config.get("peak_title", "Fitted Peak Position") or "Fitted Peak Position"
        )
        summary_x_label = str(config.get("summary_x_label", "Point") or "Point")
        summary_y_label = str(config.get("summary_y_label", "Total Counts") or "Total Counts")
        peak_y_label = str(config.get("peak_y_label", "Peak Position") or "Peak Position")

        self.figure.clf()
        self.profile_axes, self.summary_axes, self.peak_axes = self._create_axes()
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
        self._redraw()

    @QtCore.Slot(str, object, object)
    def set_profile(self, detector_name, x_values, y_values):
        x_arr = _coerce_array(x_values)
        y_arr = _coerce_array(y_values)
        if x_arr is None or y_arr is None:
            return
        if x_arr.shape != y_arr.shape:
            return

        self._remove_live_profile_line(detector_name)
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
        self._restyle_profile_history(detector_name)

        self.profile_axes.relim()
        self.profile_axes.autoscale_view()
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

        self.summary_axes.relim()
        self.summary_axes.autoscale_view()
        self._update_legends()
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

        self.summary_axes.relim()
        self.summary_axes.autoscale_view()
        self._redraw()

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
        else:
            line.set_data(x_arr, y_arr)
            line.set_color(color_map(0.9))

        self.profile_axes.relim()
        self.profile_axes.autoscale_view()
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

        self.peak_axes.relim()
        self.peak_axes.autoscale_view()
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
        self._set_axis_legend(self.summary_axes, self._summary_lines)
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
            return
        try:
            line.remove()
        except Exception:
            pass

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
            self._remove_live_profile_line(detector_name)
        for detector_name in tuple(self._summary_live_lines.keys()):
            self._remove_live_summary_line(detector_name)
        self._redraw()

    def _restyle_profile_history(self, detector_name):
        history = self._profile_history.get(detector_name, [])
        if not history:
            return

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map
        total = len(history)
        for index, line in enumerate(history):
            if total <= 1:
                age = 1.0
                color_pos = 0.9
            else:
                age = index / (total - 1)
                color_pos = 0.15 + (0.75 * age)
            is_latest = index == (total - 1)
            line.set_color(color_map(color_pos))
            line.set_alpha(0.95 if is_latest else 0.35 + (0.15 * color_pos))
            line.set_linewidth(1.5 if is_latest else 0.8 + (0.3 * age))
            line.set_marker("o" if is_latest else "None")

    def _select_profile_colormap(self, detector_name):
        names = ("viridis", "plasma", "cividis", "magma", "turbo")
        if detector_name not in self._detector_order:
            self._detector_order.append(detector_name)
        index = self._detector_order.index(detector_name)
        return cm.get_cmap(names[index % len(names)])

    def _get_detector_series_color(self, detector_name):
        color = self._detector_series_colors.get(detector_name)
        if color is not None:
            return color

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map
        rgba = color_map(0.82)
        color = mcolors.to_hex(rgba, keep_alpha=False)
        self._detector_series_colors[detector_name] = color
        return color

    def _palette(self):
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            return QtGui.QPalette(app.palette())
        return QtGui.QPalette(self.palette())

    def _apply_theme_from_palette(self):
        pal = self._palette()
        figure_bg = pal.color(QtGui.QPalette.Window)
        axes_bg = pal.color(QtGui.QPalette.Base)
        text = pal.color(QtGui.QPalette.WindowText)
        axis_text = pal.color(QtGui.QPalette.Text)
        grid = pal.color(QtGui.QPalette.Mid)
        edge = pal.color(QtGui.QPalette.Mid)
        legend_bg = pal.color(QtGui.QPalette.Base)

        self.figure.patch.set_facecolor(figure_bg.name())
        self.figure.patch.set_edgecolor(figure_bg.name())
        self.figure.set_constrained_layout(True)

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
        self._canvas.setStyleSheet(
            f"background-color: {axes_bg.name()}; color: {text.name()};"
        )
        self.setAutoFillBackground(True)

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

    def _style_toolbar(self, palette):
        bg = palette.color(QtGui.QPalette.Window).name()
        base = palette.color(QtGui.QPalette.Base).name()
        text = palette.color(QtGui.QPalette.ButtonText).name()
        border = palette.color(QtGui.QPalette.Mid).name()
        highlight = palette.color(QtGui.QPalette.Highlight).name()
        highlighted_text = palette.color(QtGui.QPalette.HighlightedText).name()
        self._toolbar.setStyleSheet(
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


class DiffractionLivePlot(QtCore.QObject):
    _reset_requested = QtCore.Signal(object)
    _clear_live_previews_requested = QtCore.Signal()
    _profile_updated = QtCore.Signal(str, object, object)
    _live_profile_updated = QtCore.Signal(str, object, object)
    _summary_point_updated = QtCore.Signal(str, float, float)
    _live_summary_point_updated = QtCore.Signal(str, float, float)
    _peak_point_updated = QtCore.Signal(str, float, float, object)
    _status_updated = QtCore.Signal(str)

    def __init__(self, widget, *, stream_name="primary", re_client=None, parent=None):
        super().__init__(parent)
        self.widget = widget
        self.stream_name = str(stream_name or "primary")
        self.re_client = re_client

        self._reset_requested.connect(self.widget.reset)
        self._clear_live_previews_requested.connect(self.widget.clear_live_previews)
        self._profile_updated.connect(self.widget.set_profile)
        self._live_profile_updated.connect(self.widget.update_live_profile)
        self._summary_point_updated.connect(self.widget.append_summary_point)
        self._live_summary_point_updated.connect(self.widget.update_live_summary_point)
        self._peak_point_updated.connect(self.widget.append_peak_point)
        self._status_updated.connect(self.widget.set_status)

        self._descriptor_stream = {}
        self._descriptor_run_start = {}
        self._descriptor_data_keys = {}
        self._run_uid = None
        self._run_title = ""
        self._plan_name = ""
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
        self._armed_live_detectors = ()
        self._planned_summary_x = []

        events = getattr(self.re_client, "events", None)
        running_item_changed = getattr(events, "running_item_changed", None)
        if running_item_changed is not None:
            running_item_changed.connect(self._on_running_item_changed)

    def on_document(self, name, doc):
        if name == "start":
            self._on_start(doc)
            return

        if name == "descriptor":
            descriptor_uid = str(doc.get("uid", ""))
            self._descriptor_stream[descriptor_uid] = str(doc.get("name", ""))
            self._descriptor_run_start[descriptor_uid] = str(doc.get("run_start", ""))
            self._descriptor_data_keys[descriptor_uid] = set(
                str(key) for key in dict(doc.get("data_keys", {}) or {}).keys()
            )
            return

        if name == "event":
            self._process_event(
                descriptor_uid=str(doc.get("descriptor", "")),
                data=dict(doc.get("data", {}) or {}),
                seq_num=doc.get("seq_num"),
            )
            return

        if name == "event_page":
            descriptor_uid = str(doc.get("descriptor", ""))
            page_data = dict(doc.get("data", {}) or {})
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
                self._process_event(descriptor_uid=descriptor_uid, data=row, seq_num=seq_num)
            return

        if name == "stop":
            exit_status = str(doc.get("exit_status", "") or "unknown")
            status_title = self._run_title or self._plan_name or "Diffraction Run"
            self._disarm_live_subscriptions()
            self._clear_live_previews_requested.emit()
            self._status_updated.emit(f"{status_title} [{exit_status}]")

    def _on_start(self, doc):
        self._descriptor_stream.clear()
        self._descriptor_run_start.clear()
        self._descriptor_data_keys.clear()
        self._detector_axis_cache.clear()
        self._run_uid = str(doc.get("uid", ""))
        self._run_title = str(doc.get("title", "") or "")
        self._plan_name = str(doc.get("plan_name", "") or "")
        self._motor_names = self._normalize_motor_names(doc.get("motors"))
        self._summary_mode = "point"
        self._summary_counter = 0
        self._primary_points_seen = 0
        self._summary_path_last_point = None
        self._summary_path_distance = 0.0
        self._planned_summary_x = []

        det_config = dict(doc.get("det_config", {}) or {})
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
        self._planned_summary_x = self._build_planned_summary_x(doc)
        peak_title = self._choose_peak_title()

        run_title = self._run_title or self._plan_name or "Diffraction Run"
        self._reset_requested.emit(
            {
                "run_title": run_title,
                "profile_title": "Current PSD Profile",
                "summary_title": self._summary_title,
                "peak_title": peak_title,
                "summary_x_label": self._summary_x_label,
                "summary_y_label": "Total Counts",
                "peak_y_label": "Peak Position",
            }
        )

    def _on_running_item_changed(self, event):
        running_item = dict(getattr(event, "running_item", {}) or {})
        detector_names = self._extract_live_detector_names_from_running_item(running_item)
        self._arm_live_subscriptions(detector_names)

    def _process_event(self, *, descriptor_uid, data, seq_num):
        if not self._accept_descriptor_run(descriptor_uid):
            return
        detector_fields = self._extract_detector_fields(data)
        if not detector_fields:
            return

        is_primary_stream = self._is_primary_descriptor(descriptor_uid)
        summary_x = None
        if is_primary_stream:
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
            if is_primary_stream:
                self._profile_updated.emit(str(detector_name), position_x, counts)
            else:
                self._live_profile_updated.emit(str(detector_name), position_x, counts)

            if not is_primary_stream:
                continue
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
            self._primary_points_seen += 1

    def _accept_descriptor_run(self, descriptor_uid):
        run_uid = self._descriptor_run_start.get(str(descriptor_uid), "")
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

    def _arm_live_subscriptions(self, detector_names):
        unique_names = tuple(dict.fromkeys(detector_names))
        if unique_names == self._armed_live_detectors:
            return

        self._disarm_live_subscriptions()
        self._armed_live_detectors = unique_names

        if PV is None:
            return

        for detector_name in unique_names:
            pv_name = self._get_live_counts_pv(detector_name)
            if not pv_name:
                continue
            try:
                pv = PV(pv_name, auto_monitor=True)
                callback_index = pv.add_callback(
                    lambda pvname=None, value=None, char_value=None, _det=detector_name, **kwargs: (
                        self._handle_live_counts_update(_det, value)
                    )
                )
            except Exception:
                continue
            self._live_subscriptions[detector_name] = {
                "pv": pv,
                "callback_index": callback_index,
            }

    def _disarm_live_subscriptions(self):
        for subscription in self._live_subscriptions.values():
            pv = subscription.get("pv")
            callback_index = subscription.get("callback_index")
            if pv is not None and callback_index is not None:
                try:
                    pv.remove_callback(callback_index)
                except Exception:
                    pass
        self._live_subscriptions.clear()
        self._armed_live_detectors = ()

    def _handle_live_counts_update(self, detector_name, value):
        counts = _coerce_array(value)
        if counts is None:
            return

        position_x = self._detector_axis_cache.get(detector_name)
        if position_x is None or position_x.shape != counts.shape:
            position_x = self._build_axis_from_counts(counts)
        if position_x is None or position_x.shape != counts.shape:
            return

        self._detector_axis_cache[detector_name] = position_x
        self._live_profile_updated.emit(str(detector_name), position_x, counts)
        total_counts = _coerce_number(np.nansum(counts))
        summary_x = self._estimate_live_summary_x()
        if total_counts is not None and summary_x is not None:
            self._live_summary_point_updated.emit(
                str(detector_name),
                float(summary_x),
                float(total_counts),
            )

    @staticmethod
    def _get_live_counts_pv(detector_name):
        mapping = {
            "he3psd0": "4dh4:he3PSD:Det0:LiveCounts",
            "he3psd7": "4dh4:he3PSD:Det7:LiveCounts",
        }
        return mapping.get(str(detector_name))

    def _estimate_live_summary_x(self):
        if self._summary_mode == "exposure":
            return float(self._primary_points_seen + 1)
        if self._summary_mode == "motor":
            if self._primary_points_seen < len(self._planned_summary_x):
                return _coerce_number(self._planned_summary_x[self._primary_points_seen])
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

    def _is_primary_descriptor(self, descriptor_uid):
        stream = self._descriptor_stream.get(str(descriptor_uid), "")
        if not self.stream_name:
            return True
        return stream == self.stream_name

    def _descriptor_has_detector_fields(self, descriptor_uid):
        data_keys = self._descriptor_data_keys.get(str(descriptor_uid), set())
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
        if self._plan_name == "count_he3":
            return "exposure", "Exposure", "Total Counts vs Exposure"
        if len(self._motor_names) == 1:
            return "motor", self._motor_names[0], "Total Counts vs Position"
        if len(self._motor_names) > 1:
            return "path", "Path Length", "Total Counts vs Scan Path"
        return "point", "Point", "Total Counts vs Point"

    def _choose_peak_title(self):
        if self._summary_mode == "exposure":
            return "Fitted Peak Position vs Exposure"
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
