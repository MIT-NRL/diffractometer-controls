import math

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

_PROFILE_COLOR_STOPS = {
    "viridis": [(68, 1, 84), (58, 82, 139), (32, 144, 140), (94, 201, 98), (253, 231, 37)],
    "plasma": [(13, 8, 135), (84, 3, 160), (182, 55, 121), (251, 136, 97), (240, 249, 33)],
    "cividis": [(0, 32, 76), (40, 83, 107), (101, 129, 120), (170, 181, 115), (253, 234, 69)],
    "magma": [(0, 0, 4), (60, 15, 112), (140, 41, 129), (221, 73, 104), (252, 253, 191)],
    "turbo": [(48, 18, 59), (40, 120, 142), (70, 190, 111), (247, 209, 61), (165, 0, 38)],
}


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


def _coerce_rgba(color):
    if isinstance(color, QtGui.QColor):
        return (
            color.redF(),
            color.greenF(),
            color.blueF(),
            color.alphaF(),
        )
    if isinstance(color, str):
        text = color.strip()
        if text.startswith("#"):
            if len(text) == 7:
                return (
                    int(text[1:3], 16) / 255.0,
                    int(text[3:5], 16) / 255.0,
                    int(text[5:7], 16) / 255.0,
                    1.0,
                )
            if len(text) == 9:
                return (
                    int(text[1:3], 16) / 255.0,
                    int(text[3:5], 16) / 255.0,
                    int(text[5:7], 16) / 255.0,
                    int(text[7:9], 16) / 255.0,
                )
    try:
        values = list(color)
    except Exception:
        return (0.29, 0.56, 0.89, 1.0)
    if len(values) == 3:
        values.append(1.0)
    if len(values) < 4:
        return (0.29, 0.56, 0.89, 1.0)
    return tuple(max(0.0, min(1.0, float(v))) for v in values[:4])


def _rgba_to_hex(rgba):
    r, g, b, _a = _coerce_rgba(rgba)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * 255.0),
        int(g * 255.0),
        int(b * 255.0),
    )


def _interp_rgb(color_a, color_b, fraction):
    fraction = max(0.0, min(1.0, float(fraction)))
    return tuple(
        (float(a) + ((float(b) - float(a)) * fraction)) / 255.0
        for a, b in zip(color_a, color_b)
    ) + (1.0,)


def _make_profile_colormap(name):
    stops = list(_PROFILE_COLOR_STOPS.get(str(name), _PROFILE_COLOR_STOPS["viridis"]))

    def _color_map(position):
        pos = max(0.0, min(1.0, float(position)))
        if len(stops) == 1:
            return tuple(float(v) / 255.0 for v in stops[0]) + (1.0,)
        scaled = pos * (len(stops) - 1)
        low_index = int(math.floor(scaled))
        high_index = min(len(stops) - 1, low_index + 1)
        frac = scaled - low_index
        return _interp_rgb(stops[low_index], stops[high_index], frac)

    return _color_map


def _color_to_qcolor(color, alpha=1.0):
    rgba = _coerce_rgba(color)
    qcolor = QtGui.QColor(
        int(round(rgba[0] * 255.0)),
        int(round(rgba[1] * 255.0)),
        int(round(rgba[2] * 255.0)),
        int(round(max(0.0, min(1.0, float(alpha))) * 255.0)),
    )
    return qcolor


def _plot_data_finite_y(item):
    x_data, y_data = item.getData()
    if y_data is None:
        return np.asarray([], dtype=float)
    arr = np.asarray(y_data, dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


class DiffractionPlotWidgetPyQtGraph(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._applying_theme = False
        self._last_title_style = None

        self._title_label = QLabel(self)
        self._title_label.setAlignment(QtCore.Qt.AlignCenter)
        self._title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        title_font = QtGui.QFont(self.font())
        title_font.setPointSize(max(11, title_font.pointSize() + 1))
        title_font.setBold(True)
        self._title_label.setFont(title_font)

        self._profile_plot = pg.PlotWidget()
        self._summary_plot = pg.PlotWidget()
        self._peak_plot = pg.PlotWidget()
        self._profile_plot.setMinimumHeight(300)
        self._summary_plot.setMinimumHeight(180)
        self._peak_plot.setMinimumHeight(180)
        self._profile_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._summary_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._peak_plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.addWidget(self._profile_plot, 0, 0, 1, 2)
        grid.addWidget(self._summary_plot, 1, 0)
        grid.addWidget(self._peak_plot, 1, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._title_label)
        layout.addLayout(grid)
        self.setLayout(layout)

        self._profile_plot_item = self._profile_plot.getPlotItem()
        self._summary_plot_item = self._summary_plot.getPlotItem()
        self._peak_plot_item = self._peak_plot.getPlotItem()
        self._profile_legend = None
        self._summary_legend = None
        self._peak_legend = None

        self._profile_title = ""
        self._summary_title = ""
        self._peak_title = ""
        self._summary_x_label = ""
        self._summary_y_label = ""
        self._peak_y_label = ""
        self._run_title = ""

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

        self._configure_plot_widget(self._profile_plot)
        self._configure_plot_widget(self._summary_plot)
        self._configure_plot_widget(self._peak_plot)
        self._install_double_click_reset()
        self.reset()

    def sizeHint(self):
        size_hint = super().sizeHint()
        size_hint.setWidth(920)
        size_hint.setHeight(560)
        return size_hint

    @QtCore.Slot(object)
    def reset(self, config=None):
        config = dict(config or {})
        self._run_title = str(config.get("run_title", "") or "")
        self._profile_title = str(config.get("profile_title", "Current Spectrum") or "Current Spectrum")
        self._summary_title = str(config.get("summary_title", "Total Counts") or "Total Counts")
        self._peak_title = str(
            config.get("peak_title", "Fitted Peak Position") or "Fitted Peak Position"
        )
        self._summary_x_label = str(config.get("summary_x_label", "Point") or "Point")
        self._summary_y_label = str(config.get("summary_y_label", "Total Counts") or "Total Counts")
        self._peak_y_label = str(config.get("peak_y_label", "Peak Position") or "Peak Position")
        self._profile_x_limits = _normalize_axis_limits(config.get("profile_x_limits"))
        self._summary_x_limits = _normalize_axis_limits(config.get("summary_x_limits"))
        self._peak_x_limits = _normalize_axis_limits(config.get("peak_x_limits"))

        self._reset_plot_item(self._profile_plot_item, legend_attr="_profile_legend")
        self._reset_plot_item(self._summary_plot_item, legend_attr="_summary_legend")
        self._reset_plot_item(self._peak_plot_item, legend_attr="_peak_legend")

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

        self._apply_theme_from_palette()
        self._apply_plot_metadata()
        self._set_plot_x_limits(self._profile_plot_item, self._profile_x_limits)
        self._set_plot_x_limits(self._summary_plot_item, self._summary_x_limits)
        self._set_plot_x_limits(self._peak_plot_item, self._peak_x_limits)

    @QtCore.Slot(str, object, object)
    def set_profile(self, detector_name, x_values, y_values):
        x_arr = _coerce_array(x_values)
        y_arr = _coerce_array(y_values)
        if x_arr is None or y_arr is None or x_arr.shape != y_arr.shape:
            return

        had_live_preview = self._remove_live_profile_line(detector_name) is not None
        history = self._profile_history.setdefault(detector_name, [])
        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map

        line = self._create_curve(self._profile_plot_item, x_arr, y_arr)
        history.append(line)
        if had_live_preview:
            self._style_profile_history_item(
                line,
                color_map=color_map,
                total=len(history),
                index=len(history) - 1,
                is_latest=True,
            )
        else:
            self._restyle_profile_history(detector_name)

        self._sync_plot_x_limits(self._profile_plot_item, x_arr, fixed_limits=self._profile_x_limits)
        self._autoscale_y(self._profile_plot_item)
        self._update_legends()

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
            line = self._create_curve(self._summary_plot_item, series_x, series_y)
            self._summary_lines[detector_name] = line
        else:
            line.setData(series_x, series_y)
        self._style_series_curve(line, color=color, width=1.3, marker="o", alpha=0.95, z=3)

        self._sync_plot_x_limits(
            self._summary_plot_item,
            series_x,
            fixed_limits=self._summary_x_limits,
        )
        self._autoscale_y(self._summary_plot_item)
        self._update_legends()

    @QtCore.Slot(str, float, float)
    def update_live_summary_point(self, detector_name, x_value, y_value):
        color = self._get_detector_series_color(detector_name)
        base_x = list(self._summary_x.get(detector_name, []))
        base_y = list(self._summary_y.get(detector_name, []))
        draw_x = base_x + [float(x_value)]
        draw_y = base_y + [float(y_value)]

        line = self._summary_live_lines.get(detector_name)
        if line is None:
            line = self._create_curve(self._summary_plot_item, draw_x, draw_y)
            self._summary_live_lines[detector_name] = line
        else:
            line.setData(draw_x, draw_y)
        self._style_series_curve(line, color=color, width=1.2, marker="o", alpha=1.0, z=5)

        self._sync_plot_x_limits(
            self._summary_plot_item,
            draw_x,
            fixed_limits=self._summary_x_limits,
        )
        self._autoscale_y(self._summary_plot_item)

    @QtCore.Slot(str, object, object)
    def update_live_profile(self, detector_name, x_values, y_values):
        x_arr = _coerce_array(x_values)
        y_arr = _coerce_array(y_values)
        if x_arr is None or y_arr is None or x_arr.shape != y_arr.shape:
            return

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map

        line = self._live_profile_lines.get(detector_name)
        if line is None:
            self._restyle_profile_history(detector_name, preview_new=True)
            line = self._create_curve(self._profile_plot_item, x_arr, y_arr)
            self._live_profile_lines[detector_name] = line
            self._style_profile_history_item(
                line,
                color_map=color_map,
                total=max(1, len(self._profile_history.get(detector_name, [])) + 1),
                index=max(0, len(self._profile_history.get(detector_name, []))),
                is_latest=True,
            )
            line.setZValue(5)
        else:
            line.setData(x_arr, y_arr)
            self._style_profile_history_item(
                line,
                color_map=color_map,
                total=max(1, len(self._profile_history.get(detector_name, [])) + 1),
                index=max(0, len(self._profile_history.get(detector_name, []))),
                is_latest=True,
            )

        self._sync_plot_x_limits(self._profile_plot_item, x_arr, fixed_limits=self._profile_x_limits)
        self._autoscale_y(self._profile_plot_item)

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

        line = self._peak_lines.get(detector_name)
        if line is None:
            line = self._create_curve(self._peak_plot_item, series_x, series_y)
            self._peak_lines[detector_name] = line
        else:
            line.setData(series_x, series_y)
        self._style_series_curve(line, color=color, width=1.3, marker="o", alpha=0.95, z=3)

        err_item = self._peak_errorbars.get(detector_name)
        if err_item is None:
            err_item = pg.ErrorBarItem()
            self._peak_plot_item.addItem(err_item)
            self._peak_errorbars[detector_name] = err_item
        yerr = np.asarray([0.0 if err is None else float(err) for err in series_yerr], dtype=float)
        err_item.setData(
            x=np.asarray(series_x, dtype=float),
            y=np.asarray(series_y, dtype=float),
            top=yerr,
            bottom=yerr,
            beam=self._estimate_errorbar_beam(series_x),
        )
        err_item.setZValue(2)

        self._sync_plot_x_limits(
            self._peak_plot_item,
            series_x,
            fixed_limits=self._peak_x_limits,
        )
        self._autoscale_peak_y()
        self._update_legends()

    @QtCore.Slot(str)
    def set_status(self, text):
        self._run_title = str(text or "")
        self._title_label.setText(self._run_title)

    @QtCore.Slot()
    def clear_live_previews(self):
        for detector_name in tuple(self._live_profile_lines.keys()):
            removed_line = self._remove_live_profile_line(detector_name)
            if removed_line is not None:
                self._restyle_profile_history(detector_name)
        for detector_name in tuple(self._summary_live_lines.keys()):
            self._remove_live_summary_line(detector_name)

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
            self._apply_plot_metadata()
            self._update_legends()

    def _configure_plot_widget(self, plot_widget):
        plot_widget.showGrid(x=True, y=True, alpha=0.25)
        plot_widget.setMenuEnabled(False)
        plot_widget.hideButtons()
        plot_widget.setAntialiasing(False)

    def _install_double_click_reset(self):
        self._profile_plot.scene().sigMouseClicked.connect(
            lambda event, plot_item=self._profile_plot_item: self._on_plot_mouse_clicked(
                plot_item, event
            )
        )
        self._summary_plot.scene().sigMouseClicked.connect(
            lambda event, plot_item=self._summary_plot_item: self._on_plot_mouse_clicked(
                plot_item, event
            )
        )
        self._peak_plot.scene().sigMouseClicked.connect(
            lambda event, plot_item=self._peak_plot_item: self._on_plot_mouse_clicked(
                plot_item, event
            )
        )

    def _on_plot_mouse_clicked(self, plot_item, event):
        try:
            if hasattr(event, "double") and event.double():
                if (not hasattr(event, "button")) or event.button() == QtCore.Qt.LeftButton:
                    self._reset_plot_view(plot_item)
                    if hasattr(event, "accept"):
                        event.accept()
        except Exception:
            pass

    def _reset_plot_item(self, plot_item, *, legend_attr):
        legend = getattr(self, legend_attr, None)
        if legend is not None:
            try:
                legend.scene().removeItem(legend)
            except Exception:
                pass
        plot_item.clear()
        plot_item.showGrid(x=True, y=True, alpha=0.25)
        plot_item.getViewBox().enableAutoRange(x=False, y=True)
        setattr(self, legend_attr, plot_item.addLegend(offset=(12, 12)))

    def _apply_plot_metadata(self):
        self._title_label.setText(self._run_title)
        self._profile_plot_item.setTitle(self._profile_title)
        self._summary_plot_item.setTitle(self._summary_title)
        self._peak_plot_item.setTitle(self._peak_title)
        self._profile_plot_item.setLabel("left", "Counts")
        self._profile_plot_item.setLabel("bottom", "Detector Position")
        self._summary_plot_item.setLabel("left", self._summary_y_label)
        self._summary_plot_item.setLabel("bottom", self._summary_x_label)
        self._peak_plot_item.setLabel("left", self._peak_y_label)
        self._peak_plot_item.setLabel("bottom", self._summary_x_label)

    def _apply_theme_from_palette(self):
        if self._applying_theme:
            return
        self._applying_theme = True
        try:
            pal = self._palette()
            figure_bg = pal.color(QtGui.QPalette.Window)
            axes_bg = pal.color(QtGui.QPalette.Base)
            text = pal.color(QtGui.QPalette.WindowText)
            axis_text = pal.color(QtGui.QPalette.Text)
            edge = pal.color(QtGui.QPalette.Mid)
            legend_bg = pal.color(QtGui.QPalette.Base)

            title_style = (
                f"color: {text.name()}; background-color: {figure_bg.name()}; padding: 2px 4px;"
            )
            if title_style != self._last_title_style:
                self._title_label.setStyleSheet(title_style)
                self._last_title_style = title_style
            self.setAutoFillBackground(True)

            for plot_widget, plot_item, title in (
                (self._profile_plot, self._profile_plot_item, self._profile_title),
                (self._summary_plot, self._summary_plot_item, self._summary_title),
                (self._peak_plot, self._peak_plot_item, self._peak_title),
            ):
                plot_widget.setBackground((axes_bg.red(), axes_bg.green(), axes_bg.blue()))
                plot_item.showGrid(x=True, y=True, alpha=0.25)
                plot_item.getViewBox().setBorder(
                    pg.mkPen(edge.red(), edge.green(), edge.blue(), width=1)
                )
                plot_item.getAxis("left").setTextPen(
                    pg.mkPen(axis_text.red(), axis_text.green(), axis_text.blue(), width=1)
                )
                plot_item.getAxis("bottom").setTextPen(
                    pg.mkPen(axis_text.red(), axis_text.green(), axis_text.blue(), width=1)
                )
                plot_item.getAxis("left").setPen(
                    pg.mkPen(edge.red(), edge.green(), edge.blue(), width=1)
                )
                plot_item.getAxis("bottom").setPen(
                    pg.mkPen(edge.red(), edge.green(), edge.blue(), width=1)
                )
                plot_item.getAxis("left").setGrid(35)
                plot_item.getAxis("bottom").setGrid(35)
                plot_item.setTitle(title, color=text.name(), size="11pt")

            label_style = {"color": text.name()}
            self._profile_plot_item.setLabel("left", "Counts", **label_style)
            self._profile_plot_item.setLabel("bottom", "Detector Position", **label_style)
            self._summary_plot_item.setLabel("left", self._summary_y_label, **label_style)
            self._summary_plot_item.setLabel("bottom", self._summary_x_label, **label_style)
            self._peak_plot_item.setLabel("left", self._peak_y_label, **label_style)
            self._peak_plot_item.setLabel("bottom", self._summary_x_label, **label_style)

            for legend in (self._profile_legend, self._summary_legend, self._peak_legend):
                self._style_legend(
                    legend,
                    background=legend_bg,
                    edge=edge,
                    text_color=axis_text,
                )
        finally:
            self._applying_theme = False

    def _style_legend(self, legend, *, background, edge, text_color):
        if legend is None:
            return
        try:
            legend.setBrush(pg.mkBrush(background.red(), background.green(), background.blue(), 235))
            legend.setPen(pg.mkPen(edge.red(), edge.green(), edge.blue(), width=1))
        except Exception:
            pass
        for sample, label in getattr(legend, "items", []):
            try:
                label.setText(label.text, color=text_color.name())
            except Exception:
                try:
                    label.setAttr("color", text_color.name())
                except Exception:
                    pass

    def _update_legends(self):
        profile_items = {
            detector_name: history[-1]
            for detector_name, history in self._profile_history.items()
            if history
        }
        self._refresh_legend(self._profile_legend, profile_items)
        self._refresh_legend(self._summary_legend, self._summary_lines)
        self._refresh_legend(self._peak_legend, self._peak_lines)
        self._apply_theme_from_palette()

    def _refresh_legend(self, legend, line_map):
        if legend is None:
            return
        try:
            legend.clear()
        except Exception:
            pass
        for detector_name, item in line_map.items():
            if item is None:
                continue
            try:
                legend.addItem(item, str(detector_name))
            except Exception:
                pass

    def _reset_plot_view(self, plot_item):
        if plot_item is self._profile_plot_item:
            self._set_plot_x_limits(self._profile_plot_item, self._profile_x_limits)
            if self._profile_x_limits is None:
                self._autoscale_x(self._profile_plot_item)
            self._autoscale_y(self._profile_plot_item)
            return
        if plot_item is self._summary_plot_item:
            self._set_plot_x_limits(self._summary_plot_item, self._summary_x_limits)
            if self._summary_x_limits is None:
                self._autoscale_x(self._summary_plot_item)
            self._autoscale_y(self._summary_plot_item)
            return
        if plot_item is self._peak_plot_item:
            self._set_plot_x_limits(self._peak_plot_item, self._peak_x_limits)
            if self._peak_x_limits is None:
                self._autoscale_x(self._peak_plot_item)
            self._autoscale_peak_y()

    def _remove_live_profile_line(self, detector_name):
        line = self._live_profile_lines.pop(detector_name, None)
        if line is None:
            return None
        try:
            self._profile_plot_item.removeItem(line)
        except Exception:
            pass
        return line

    def _remove_live_summary_line(self, detector_name):
        line = self._summary_live_lines.pop(detector_name, None)
        if line is None:
            return None
        try:
            self._summary_plot_item.removeItem(line)
        except Exception:
            pass
        return line

    def _restyle_profile_history(self, detector_name, *, preview_new=False):
        history = self._profile_history.get(detector_name, [])
        if not history:
            return

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map
        total = len(history) + (1 if preview_new else 0)
        for index, item in enumerate(history):
            is_latest = (not preview_new) and index == (len(history) - 1)
            self._style_profile_history_item(
                item,
                color_map=color_map,
                total=total,
                index=index,
                is_latest=is_latest,
            )

    def _style_profile_history_item(self, item, *, color_map, total, index, is_latest):
        color_pos = self._profile_history_color_position(total=total, index=index)
        if total <= 1:
            age = 1.0
        else:
            age = index / (total - 1)
        alpha = 0.95 if is_latest else 0.35 + (0.15 * color_pos)
        width = 1.5 if is_latest else 0.8 + (0.3 * age)
        color = _color_to_qcolor(color_map(color_pos), alpha=alpha)
        pen = pg.mkPen(color=color, width=width)
        item.setPen(pen)
        item.setZValue(4 if is_latest else 2)
        if is_latest:
            item.setSymbol("o")
            item.setSymbolSize(5)
            item.setSymbolPen(pen)
            item.setSymbolBrush(pg.mkBrush(0, 0, 0, 0))
        else:
            item.setSymbol(None)

    @staticmethod
    def _profile_history_color_position(*, total, index):
        recent_rank = max(0, int(total) - 1 - int(index))
        recent_slots = np.asarray([0.90, 0.76, 0.62, 0.48, 0.34, 0.22, 0.15], dtype=float)
        slot_index = min(recent_rank, recent_slots.size - 1)
        return float(recent_slots[slot_index])

    def _style_series_curve(self, item, *, color, width, marker=None, alpha=1.0, z=3):
        qcolor = _color_to_qcolor(color, alpha=alpha)
        pen = pg.mkPen(qcolor, width=width)
        item.setPen(pen)
        item.setZValue(z)
        item.setSymbol(marker if marker else None)
        if marker:
            item.setSymbolSize(6)
            item.setSymbolPen(pen)
            item.setSymbolBrush(pg.mkBrush(0, 0, 0, 0))

    @staticmethod
    def _create_curve(plot_item, x_values, y_values):
        item = pg.PlotDataItem(
            np.asarray(x_values, dtype=float),
            np.asarray(y_values, dtype=float),
            connect="finite",
            antialias=False,
        )
        plot_item.addItem(item)
        return item

    @staticmethod
    def _set_plot_x_limits(plot_item, limits):
        if limits is None:
            return
        low, high = DiffractionPlotWidgetPyQtGraph._expand_limits_for_view(limits)
        plot_item.setXRange(float(low), float(high), padding=0.0)

    def _sync_plot_x_limits(self, plot_item, x_values, *, fixed_limits=None):
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
        plot_item.setXRange(low - pad, high + pad, padding=0.0)

    def _autoscale_x(self, plot_item):
        items = [item for item in plot_item.listDataItems() if item is not None]
        if not items:
            return
        arrays = []
        for item in items:
            x_data, _ = item.getData()
            if x_data is None:
                continue
            arr = np.asarray(x_data, dtype=float)
            if arr.size == 0:
                continue
            arr = arr[np.isfinite(arr)]
            if arr.size:
                arrays.append(arr)
        if not arrays:
            return
        low = min(float(np.min(arr)) for arr in arrays)
        high = max(float(np.max(arr)) for arr in arrays)
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = max((high - low) * 0.02, 1e-9)
        plot_item.setXRange(low - pad, high + pad, padding=0.0)

    @staticmethod
    def _expand_limits_for_view(limits):
        low = float(limits[0])
        high = float(limits[1])
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = max((high - low) * 0.02, 0.25)
        return low - pad, high + pad

    def _autoscale_y(self, plot_item):
        items = [
            item
            for item in plot_item.listDataItems()
            if item is not None
        ]
        if not items:
            return
        arrays = [_plot_data_finite_y(item) for item in items]
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            return
        low = min(float(np.min(arr)) for arr in arrays)
        high = max(float(np.max(arr)) for arr in arrays)
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = max((high - low) * 0.05, 1e-9)
        plot_item.setYRange(low - pad, high + pad, padding=0.0)

    def _autoscale_peak_y(self):
        arrays = []
        for detector_name, y_values in self._peak_y.items():
            y_arr = np.asarray(y_values, dtype=float)
            if y_arr.size == 0:
                continue
            err_values = self._peak_yerr.get(detector_name, [])
            err_arr = np.asarray(
                [0.0 if err is None else float(err) for err in err_values],
                dtype=float,
            )
            if err_arr.shape != y_arr.shape:
                err_arr = np.zeros_like(y_arr)
            arrays.append(y_arr - err_arr)
            arrays.append(y_arr + err_arr)
        arrays = [arr[np.isfinite(arr)] for arr in arrays if arr.size]
        arrays = [arr for arr in arrays if arr.size]
        if not arrays:
            return
        low = min(float(np.min(arr)) for arr in arrays)
        high = max(float(np.max(arr)) for arr in arrays)
        if math.isclose(low, high):
            pad = max(abs(low) * 0.05, 1.0)
        else:
            pad = max((high - low) * 0.05, 1e-9)
        self._peak_plot_item.setYRange(low - pad, high + pad, padding=0.0)

    def _select_profile_colormap(self, detector_name):
        names = ("viridis", "plasma", "cividis", "magma", "turbo")
        if detector_name not in self._detector_order:
            self._detector_order.append(detector_name)
        index = self._detector_order.index(detector_name)
        return _make_profile_colormap(names[index % len(names)])

    def _get_detector_series_color(self, detector_name):
        color = self._detector_series_colors.get(detector_name)
        if color is not None:
            return color

        color_map = self._profile_colormaps.get(detector_name)
        if color_map is None:
            color_map = self._select_profile_colormap(detector_name)
            self._profile_colormaps[detector_name] = color_map
        rgba = color_map(0.82)
        color = _rgba_to_hex(rgba)
        self._detector_series_colors[detector_name] = color
        return color

    @staticmethod
    def _estimate_errorbar_beam(series_x):
        if len(series_x) < 2:
            return 0.0
        finite = np.asarray(series_x, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size < 2:
            return 0.0
        diffs = np.diff(np.unique(np.sort(finite)))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return 0.0
        return float(np.median(diffs)) * 0.18

    def _palette(self):
        app = QtCore.QCoreApplication.instance()
        if app is not None:
            return QtGui.QPalette(app.palette())
        return QtGui.QPalette(self.palette())
