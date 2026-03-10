import math
from collections.abc import Iterable

import matplotlib.figure
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from qtpy import QtCore
from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QWidget


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


class DiffractionPlotWidget(QWidget):
    def __init__(self, parent=None):
        _initialize_matplotlib()
        super().__init__(parent)

        self.figure = matplotlib.figure.Figure(constrained_layout=True)
        self.profile_axes, self.summary_axes = self.figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [2.2, 1.2]}
        )

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

        self._profile_lines = {}
        self._summary_lines = {}
        self._summary_x = {}
        self._summary_y = {}
        self.reset()

    def sizeHint(self):
        size_hint = super().sizeHint()
        size_hint.setWidth(760)
        size_hint.setHeight(560)
        return size_hint

    @QtCore.Slot(object)
    def reset(self, config=None):
        config = dict(config or {})
        run_title = str(config.get("run_title", "") or "")
        profile_title = str(config.get("profile_title", "Current Spectrum") or "Current Spectrum")
        summary_title = str(config.get("summary_title", "Total Counts") or "Total Counts")
        summary_x_label = str(config.get("summary_x_label", "Point") or "Point")
        summary_y_label = str(config.get("summary_y_label", "Total Counts") or "Total Counts")

        self.figure.clf()
        self.profile_axes, self.summary_axes = self.figure.subplots(
            2, 1, gridspec_kw={"height_ratios": [2.2, 1.2]}
        )
        self.profile_axes.grid(alpha=0.25)
        self.summary_axes.grid(alpha=0.25)
        self.profile_axes.tick_params(direction="in", which="both")
        self.summary_axes.tick_params(direction="in", which="both")
        self.profile_axes.set_title(profile_title)
        self.profile_axes.set_ylabel("Counts")
        self.profile_axes.set_xlabel("Detector Position")
        self.summary_axes.set_title(summary_title)
        self.summary_axes.set_xlabel(summary_x_label)
        self.summary_axes.set_ylabel(summary_y_label)
        self.figure.suptitle(run_title)

        self._profile_lines.clear()
        self._summary_lines.clear()
        self._summary_x.clear()
        self._summary_y.clear()
        self._redraw()

    @QtCore.Slot(str, object, object)
    def set_profile(self, detector_name, x_values, y_values):
        x_arr = _coerce_array(x_values)
        y_arr = _coerce_array(y_values)
        if x_arr is None or y_arr is None:
            return
        if x_arr.shape != y_arr.shape:
            return

        line = self._profile_lines.get(detector_name)
        if line is None:
            (line,) = self.profile_axes.plot(
                x_arr,
                y_arr,
                marker="o",
                lw=1,
                mfc="none",
                ms=3,
                label=str(detector_name),
            )
            self._profile_lines[detector_name] = line
        else:
            line.set_data(x_arr, y_arr)

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

        line = self._summary_lines.get(detector_name)
        if line is None:
            (line,) = self.summary_axes.plot(
                series_x,
                series_y,
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

    @QtCore.Slot(str)
    def set_status(self, text):
        title = str(text or "")
        self.figure.suptitle(title)
        self._redraw()

    def _update_legends(self):
        if self._profile_lines:
            self.profile_axes.legend(loc="best")
        if self._summary_lines:
            self.summary_axes.legend(loc="best")

    def _redraw(self):
        self.figure.canvas.draw_idle()


class DiffractionLivePlot(QtCore.QObject):
    _reset_requested = QtCore.Signal(object)
    _profile_updated = QtCore.Signal(str, object, object)
    _summary_point_updated = QtCore.Signal(str, float, float)
    _status_updated = QtCore.Signal(str)

    def __init__(self, widget, *, stream_name="primary", parent=None):
        super().__init__(parent)
        self.widget = widget
        self.stream_name = str(stream_name or "primary")

        self._reset_requested.connect(self.widget.reset)
        self._profile_updated.connect(self.widget.set_profile)
        self._summary_point_updated.connect(self.widget.append_summary_point)
        self._status_updated.connect(self.widget.set_status)

        self._descriptor_stream = {}
        self._descriptor_run_start = {}
        self._run_uid = None
        self._run_title = ""
        self._plan_name = ""
        self._motor_names = []
        self._summary_x_label = "Point"
        self._summary_title = "Total Counts"
        self._summary_counter = 0
        self._detector_axis_cache = {}
        self._detector_axis_bounds = {}

    def on_document(self, name, doc):
        if name == "start":
            self._on_start(doc)
            return

        if name == "descriptor":
            descriptor_uid = str(doc.get("uid", ""))
            self._descriptor_stream[descriptor_uid] = str(doc.get("name", ""))
            self._descriptor_run_start[descriptor_uid] = str(doc.get("run_start", ""))
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
            self._status_updated.emit(f"{status_title} [{exit_status}]")

    def _on_start(self, doc):
        self._descriptor_stream.clear()
        self._descriptor_run_start.clear()
        self._detector_axis_cache.clear()
        self._run_uid = str(doc.get("uid", ""))
        self._run_title = str(doc.get("title", "") or "")
        self._plan_name = str(doc.get("plan_name", "") or "")
        self._motor_names = self._normalize_motor_names(doc.get("motors"))
        self._summary_counter = 0

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

        summary_x_label, summary_title = self._choose_summary_config()
        self._summary_x_label = summary_x_label
        self._summary_title = summary_title

        run_title = self._run_title or self._plan_name or "Diffraction Run"
        self._reset_requested.emit(
            {
                "run_title": run_title,
                "profile_title": "Current PSD Profile",
                "summary_title": self._summary_title,
                "summary_x_label": self._summary_x_label,
                "summary_y_label": "Total Counts",
            }
        )

    def _process_event(self, *, descriptor_uid, data, seq_num):
        if not self._accept_descriptor(descriptor_uid):
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
            self._summary_point_updated.emit(str(detector_name), float(summary_x), float(total_counts))

    def _accept_descriptor(self, descriptor_uid):
        stream = self._descriptor_stream.get(str(descriptor_uid), "")
        run_uid = self._descriptor_run_start.get(str(descriptor_uid), "")
        if self._run_uid and run_uid and run_uid != self._run_uid:
            return False
        if self.stream_name and stream and stream != self.stream_name:
            return False
        return True

    @staticmethod
    def _normalize_motor_names(motors):
        if motors is None:
            return []
        if isinstance(motors, str):
            text = motors.strip()
            return [text] if text else []
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
            return "Exposure", "Total Counts vs Exposure"
        if len(self._motor_names) == 1:
            return self._motor_names[0], "Total Counts vs Position"
        return "Point", "Total Counts vs Point"

    def _extract_summary_x(self, *, data, seq_num):
        if len(self._motor_names) == 1:
            value = _coerce_number(data.get(self._motor_names[0]))
            if value is not None:
                return value
        if seq_num is not None:
            value = _coerce_number(seq_num)
            if value is not None:
                return value
        self._summary_counter += 1
        return float(self._summary_counter)

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
