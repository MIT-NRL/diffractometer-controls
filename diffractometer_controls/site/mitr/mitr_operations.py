"""MITR operations display with bounded history and archive reconciliation."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from pydm.widgets.archiver_time_plot import (
    ArchivePlotCurveItem,
    PyDMArchiverTimePlot,
)
from pydm.widgets.timeplot import PyDMTimePlot
from pyqtgraph import ViewBox
from qtpy import QtCore, QtGui, QtWidgets

ARCHIVE_OPTIMIZED_BINS = 500
ARCHIVE_REFRESH_INTERVAL_MS = 60_000
ARCHIVE_REQUEST_STALE_SECONDS = 20.0
ZOOM_REFRESH_DEBOUNCE_MS = 1000
FOLLOW_LIVE_DEFAULT_WINDOW_MINUTES = 12 * 60
FOLLOW_LIVE_UPDATE_INTERVAL_MS = 1000
FOLLOW_LIVE_RIGHT_PADDING_FRACTION = 0.02
FOLLOW_LIVE_MIN_RIGHT_PADDING_SECONDS = 5.0
FOLLOW_LIVE_MAX_RIGHT_PADDING_SECONDS = 300.0

# MIT's core brand palette: https://brand.mit.edu/color
MIT_RED = "#750014"
MIT_BRIGHT_RED = "#ff1423"
LIGHT_PLOT_BACKGROUND = QtGui.QColor("#ffffff")
MIN_CURVE_CONTRAST = 3.0
LIGHT_MINOR_ALARM_COLOR = "#c68400"
DARK_MINOR_ALARM_COLOR = "#facc15"


class MITROperationsArchiveCurve(ArchivePlotCurveItem):
    """Archive curve that safely handles empty and overlapping responses."""

    def __init__(self, *args, **kwargs):
        self._new_live_value_pending = False
        self._archive_reconciled_through = None
        super().__init__(*args, **kwargs)
        self.error_bar.hide()
        self.show_extension_line = False

    def _finish_empty_archive_request(self):
        # PyDM's plot decrements its pending response count from this signal.
        # Empty data is still a completed response and must not clear the
        # curve or leave the plot waiting indefinitely.
        self.archive_data_received_signal.emit()

    @QtCore.Slot(np.ndarray)
    def receiveArchiveData(self, data):
        """Merge a valid response while preserving data on an empty one."""
        data = np.asarray(data) if data is not None else None
        if (
            data is None
            or data.ndim != 2
            or data.shape[0] < 2
            or data.shape[1] == 0
            or not self.isVisible()
        ):
            self._finish_empty_archive_request()
            return

        data = data[:, np.isfinite(data[0])]
        if data.shape[1] == 0:
            self._finish_empty_archive_request()
            return
        data = data[:, np.argsort(data[0], kind="stable")]
        latest_archive_x = float(data[0, -1])
        if self._archive_reconciled_through is None:
            self._archive_reconciled_through = latest_archive_x
        else:
            self._archive_reconciled_through = max(
                float(self._archive_reconciled_through),
                latest_archive_x,
            )

        # An archiver response is authoritative for every timestamp it has
        # reached.  Keep those samples in the archive buffer and remove the
        # now-reconciled local observations from the live buffer.  Previously,
        # the overlapping portion of each response was inserted back into the
        # live buffer; that prevented the archive/live boundary from advancing
        # during the periodic refresh.
        self._merge_archive_data(data)
        self._trim_live_through(latest_archive_x)

        self.data_changed.emit()
        # Optimized archive responses include min/max and deviation rows.
        # Keep the plots focused on their mean-value lines only.
        self.error_bar.hide()
        self.archive_data_received_signal.emit()

    def _repair_live_buffer(self):
        """Restore PyDM's expected 2-by-buffer-size live array if needed."""
        expected_shape = (2, int(self._bufferSize))
        if self.data_buffer.shape == expected_shape:
            self.points_accumulated = min(
                max(int(self.points_accumulated), 0),
                int(self._bufferSize),
            )
            return

        old_data = np.asarray(self.data_buffer)
        if old_data.ndim == 2 and old_data.shape[1] == 2:
            old_data = old_data.T

        points = 0
        if old_data.ndim == 2 and old_data.shape[0] >= 2:
            points = min(
                int(self.points_accumulated),
                int(old_data.shape[1]),
                int(self._bufferSize),
            )

        self.initialize_buffer()
        if points:
            self.data_buffer[:, -points:] = old_data[:2, -points:]
            self.points_accumulated = points

    @staticmethod
    def _merge_samples(existing, incoming, capacity):
        """Replace one time interval while preserving samples outside it."""
        incoming = np.asarray(incoming[:2])
        if existing is not None:
            min_x = incoming[0, 0]
            max_x = incoming[0, -1]
            outside_backfill = (existing[0] < min_x) | (existing[0] > max_x)
            merged = np.concatenate(
                (existing[:, outside_backfill], incoming),
                axis=1,
            )
        else:
            merged = incoming

        order = np.argsort(merged[0], kind="stable")
        merged = merged[:, order]
        # Keep the last value for duplicate timestamps, preferring the newly
        # received sample because it was appended after the cached data.
        _, reverse_indices = np.unique(
            merged[0, ::-1],
            return_index=True,
        )
        keep = np.sort(merged.shape[1] - 1 - reverse_indices)
        merged = merged[:, keep]
        return merged[:, -capacity:]

    def _merge_archive_data(self, data):
        """Merge archive results without replacing the full cached history."""
        archive_points = min(
            int(self.archive_points_accumulated),
            int(self.archive_data_buffer.shape[1]),
        )
        existing = (
            self.archive_data_buffer[:, -archive_points:]
            if archive_points
            else None
        )
        merged = self._merge_samples(
            existing,
            data[:2],
            int(self._archiveBufferSize),
        )
        self.initializeArchiveBuffer()
        self.archive_points_accumulated = int(merged.shape[1])
        self.archive_data_buffer[:, -self.archive_points_accumulated :] = merged
        self.error_bar_data = None

    def _trim_live_through(self, latest_archive_x):
        """Discard local samples covered by the latest archive response."""
        self._repair_live_buffer()
        live_points = min(
            int(self.points_accumulated),
            int(self.data_buffer.shape[1]),
        )
        if not live_points:
            return

        remaining = self.data_buffer[:, -live_points:]
        remaining = remaining[:, remaining[0] > float(latest_archive_x)]
        self.initialize_buffer()
        self.points_accumulated = int(remaining.shape[1])
        if self.points_accumulated:
            self.data_buffer[:, -self.points_accumulated :] = remaining

    def insert_live_data(self, data):
        """Merge a backfill into the live buffer without changing its shape."""
        self._repair_live_buffer()
        incoming = np.asarray(data[:2])
        if incoming.ndim != 2 or incoming.shape[1] == 0:
            return

        live_points = min(
            int(self.points_accumulated),
            int(self.data_buffer.shape[1]),
        )
        existing = self.data_buffer[:, -live_points:] if live_points else None
        merged = self._merge_samples(
            existing,
            incoming,
            int(self._bufferSize),
        )

        self.initialize_buffer()
        self.points_accumulated = int(merged.shape[1])
        if self.points_accumulated:
            self.data_buffer[:, -self.points_accumulated :] = merged

    @QtCore.Slot(float)
    @QtCore.Slot(int)
    def receiveNewValue(self, new_value):
        # AtFixedRate normally repeats the last value on every timer tick,
        # drawing an artificial line up to the present even when the PV has
        # not produced a new sample. Keep the rate limit, but only add a point
        # after an actual channel update.
        self.update_min_max_y_values(new_value)
        if self._update_mode == PyDMTimePlot.OnValueChange:
            self._repair_live_buffer()
            self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
            self.data_buffer[0, -1] = time.time()
            self.data_buffer[1, -1] = new_value
            if self.points_accumulated < self._bufferSize:
                self.points_accumulated += 1
            self.data_changed.emit()
        elif self._update_mode == PyDMTimePlot.AtFixedRate:
            self.latest_value = new_value
            self._new_live_value_pending = True

    @QtCore.Slot()
    def asyncUpdate(self):
        if (
            self._update_mode != PyDMTimePlot.AtFixedRate
            or not self._new_live_value_pending
        ):
            return
        self._new_live_value_pending = False
        self._repair_live_buffer()
        self.data_buffer = np.roll(self.data_buffer, -1, axis=1)
        self.data_buffer[0, -1] = time.time()
        self.data_buffer[1, -1] = self.latest_value
        if self.points_accumulated < self._bufferSize:
            self.points_accumulated += 1
        self.data_changed.emit()


class MITROperationsArchivePlot(PyDMArchiverTimePlot):
    """Archive plot with lighter history and periodic recent-data backfill."""

    def __init__(self, *args, **kwargs):
        self._applying_theme = False
        kwargs.setdefault("optimized_data_bins", ARCHIVE_OPTIMIZED_BINS)
        # Archive responses should not replace the range selected by the user.
        kwargs.setdefault("show_all", False)
        super().__init__(*args, **kwargs)
        self._archive_pending_since = None
        self._last_visible_span = None
        self._zoom_refresh_requested = False
        self._manual_x_navigation = False
        self._follow_live = False
        self._follow_window_seconds = FOLLOW_LIVE_DEFAULT_WINDOW_MINUTES * 60
        self._follow_controls_connected = False
        self.archive_request_finished.connect(self._archive_request_completed)

        # PyDM normally autoranges against the curve's entire stored history,
        # and its axis implementation disables Y autorange during a manual X
        # drag.  Keep each Y axis fitted to the portion of the curve in the
        # visible time window instead.
        QtCore.QTimer.singleShot(0, self._enable_visible_y_autoscale)
        self.plotItem.sigXRangeChangedManually.connect(
            self._enable_visible_y_autoscale
        )
        self.plotItem.sigXRangeChangedManually.connect(
            self._mark_manual_x_navigation
        )
        self.plotItem.sigXRangeChangedManually.connect(
            self._stop_follow_on_manual_navigation
        )

        self._zoom_refresh_timer = QtCore.QTimer(self)
        self._zoom_refresh_timer.setSingleShot(True)
        self._zoom_refresh_timer.setInterval(ZOOM_REFRESH_DEBOUNCE_MS)
        self._zoom_refresh_timer.timeout.connect(self._refresh_zoomed_range)
        self.plotItem.sigXRangeChangedManually.connect(self._schedule_zoom_refresh)
        # Restore-range, zoom-history, and autorange actions update X through
        # the regular range-changed signal rather than the manual one.
        self.plotItem.sigXRangeChanged.connect(self._schedule_zoom_refresh)
        # PyDM's View All action only enables autorange; when the cached data
        # already produces the same bounds it emits no usable span change.
        # Treat the action itself as an explicit archive refresh request.
        self.plotItem.getViewBox().menu.sigSetAutorange.connect(
            self._force_archive_refresh
        )
        self.plotItem.getViewBox().menu.sigRestoreRanges.connect(
            self._restore_default_follow_window
        )

        self._archive_refresh_timer = QtCore.QTimer(self)
        self._archive_refresh_timer.setInterval(ARCHIVE_REFRESH_INTERVAL_MS)
        self._archive_refresh_timer.timeout.connect(self._refresh_archived_data)
        self._archive_refresh_timer.start()

        self._follow_live_timer = QtCore.QTimer(self)
        self._follow_live_timer.setInterval(FOLLOW_LIVE_UPDATE_INTERVAL_MS)
        self._follow_live_timer.timeout.connect(self._update_follow_live_range)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            # ApplicationPaletteChange is not consistently delivered to plots
            # nested inside a loaded .ui or an inactive tab.  Listen to the
            # application signal so every custom PyQtGraph axis is refreshed
            # immediately when the user switches themes.
            app.paletteChanged.connect(self._apply_theme_from_palette)
        QtCore.QTimer.singleShot(0, self._connect_follow_controls)
        QtCore.QTimer.singleShot(0, self._apply_theme_from_palette)

    def createCurveItem(self, *args, **kwargs):
        curve = MITROperationsArchiveCurve(*args, **kwargs)
        curve._mitr_configured_color = QtGui.QColor(curve.color)
        curve.archive_data_received_signal.connect(self.archive_data_received)
        curve.prompt_archive_request.connect(self.requestDataFromArchiver)
        QtCore.QTimer.singleShot(0, self._apply_theme_from_palette)
        return curve

    @staticmethod
    def _relative_luminance(color):
        channels = []
        for value in (color.redF(), color.greenF(), color.blueF()):
            channels.append(
                value / 12.92
                if value <= 0.04045
                else ((value + 0.055) / 1.055) ** 2.4
            )
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    @classmethod
    def _contrast_ratio(cls, first, second):
        lighter, darker = sorted(
            (cls._relative_luminance(first), cls._relative_luminance(second)),
            reverse=True,
        )
        return (lighter + 0.05) / (darker + 0.05)

    @classmethod
    def _curve_color_for_background(cls, configured_color, background):
        """Preserve a trace hue while ensuring it remains visible."""
        configured_color = QtGui.QColor(configured_color)
        if cls._contrast_ratio(configured_color, background) >= MIN_CURVE_CONTRAST:
            return configured_color

        target = 255 if background.lightness() < 128 else 0
        red, green, blue = (
            configured_color.red(),
            configured_color.green(),
            configured_color.blue(),
        )
        for step in range(1, 11):
            blend = step / 10.0
            candidate = QtGui.QColor(
                round(red + (target - red) * blend),
                round(green + (target - green) * blend),
                round(blue + (target - blue) * blend),
            )
            if cls._contrast_ratio(candidate, background) >= MIN_CURVE_CONTRAST:
                return candidate
        return QtGui.QColor(target, target, target)

    def _application_palette(self):
        app = QtWidgets.QApplication.instance()
        if app is not None:
            return QtGui.QPalette(app.palette())
        return QtGui.QPalette(self.palette())

    def _apply_tab_accent_theme(self, palette):
        """Restore MIT brand accents without fixing the rest of the UI theme."""
        tab_widget = self.window().findChild(
            QtWidgets.QTabWidget,
            "PyDMTabWidget",
        )
        if tab_widget is None:
            return

        window_background = palette.color(QtGui.QPalette.Window).name()
        minor_alarm_color = (
            DARK_MINOR_ALARM_COLOR
            if palette.color(QtGui.QPalette.Window).lightness() < 128
            else LIGHT_MINOR_ALARM_COLOR
        )
        tab_widget.setStyleSheet(
            f"""
            QTabWidget#PyDMTabWidget::pane {{
                border: 3px solid {MIT_RED};
                background-color: {window_background};
            }}
            QTabWidget#PyDMTabWidget QTabBar::tab {{
                border: 2px solid {MIT_RED};
                margin-left: 5px;
                margin-right: 5px;
                margin-bottom: 3px;
                padding: 3px 4px;
                background-color: {MIT_RED};
                color: #ffffff;
            }}
            QTabWidget#PyDMTabWidget QTabBar::tab:hover,
            QTabWidget#PyDMTabWidget QTabBar::tab:selected {{
                background-color: {MIT_BRIGHT_RED};
            }}
            PyDMLabel#reactorThermalPowerValueLabel[alarmSeverity="1"],
            PyDMLabel#reactorPowerSixValueLabel[alarmSeverity="1"],
            PyDMLabel#reactorPowerFourValueLabel[alarmSeverity="1"],
            PyDMLabel#dwkOneValueLabel[alarmSeverity="1"],
            PyDMLabel#dwkTwoValueLabel[alarmSeverity="1"],
            PyDMLabel#dwkThreeValueLabel[alarmSeverity="1"],
            PyDMLabel#dwkFourValueLabel[alarmSeverity="1"] {{
                color: {minor_alarm_color};
            }}
            """.strip()
        )

    def _apply_theme_from_palette(self, *_args):
        """Apply the active Qt palette to all non-native plot elements."""
        if self._applying_theme:
            return
        self._applying_theme = True
        try:
            palette = self._application_palette()
            palette_background = palette.color(QtGui.QPalette.Base)
            text = palette.color(QtGui.QPalette.Text)
            edge = palette.color(QtGui.QPalette.Mid)
            dark_mode = self._relative_luminance(palette_background) < self._relative_luminance(
                text
            )
            background = (
                palette_background if dark_mode else QtGui.QColor(LIGHT_PLOT_BACKGROUND)
            )
            axis_line = text if dark_mode else edge
            axis_width = 1.5 if dark_mode else 1

            self._apply_tab_accent_theme(palette)
            self.setBackgroundColor(background)
            self.plotItem.getViewBox().setBorder(
                pg.mkPen(axis_line, width=axis_width)
            )
            for axis_info in self.plotItem.axes.values():
                axis = axis_info["item"]
                axis.setPen(pg.mkPen(axis_line, width=axis_width))
                axis.setTextPen(pg.mkPen(text, width=1))
                label_text = getattr(axis, "labelText", "")
                if label_text:
                    axis.setLabel(text=label_text, color=text.name())

            legend = getattr(self, "_legend", None)
            if legend is not None:
                legend_background = QtGui.QColor(background)
                legend_background.setAlpha(235)
                legend.setBrush(pg.mkBrush(legend_background))
                legend.setPen(pg.mkPen(edge, width=1))
                for _sample, label in getattr(legend, "items", ()):
                    try:
                        label.setText(label.text, color=text.name())
                    except Exception:
                        try:
                            label.setAttr("color", text.name())
                        except Exception:
                            pass

            for curve in getattr(self, "_curves", ()):
                configured = getattr(curve, "_mitr_configured_color", None)
                if configured is None:
                    configured = QtGui.QColor(curve.color)
                    curve._mitr_configured_color = configured
                curve.color = self._curve_color_for_background(
                    configured,
                    background,
                )
            self.viewport().update()
        finally:
            self._applying_theme = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
        ):
            QtCore.QTimer.singleShot(0, self._apply_theme_from_palette)

    def _mark_manual_x_navigation(self, *_args):
        self._manual_x_navigation = True

    def _stop_follow_on_manual_navigation(self, *_args):
        """Let a deliberate pan or zoom take control of the time range."""
        window = self.window()
        window._mitr_follow_reference_plot = self
        if not self._follow_live:
            return
        checkbox = window.findChild(
            QtWidgets.QCheckBox,
            "followLiveCheckBox",
        )
        if checkbox is not None:
            checkbox.setChecked(False)
        else:
            self.setFollowLive(False)

    def updateXAxis(self, update_immediately=False):
        """Let manual views—including future time—remain where the user put them."""
        if not getattr(self, "_manual_x_navigation", False):
            return super().updateXAxis(update_immediately)
        if self._curves:
            self._prev_x = self.plotItem.getAxis("bottom").range[0]

    def _connect_follow_controls(self):
        if self._follow_controls_connected:
            return
        window = self.window()
        checkbox = window.findChild(QtWidgets.QCheckBox, "followLiveCheckBox")
        window_spin = window.findChild(QtWidgets.QSpinBox, "liveWindowSpinBox")
        if checkbox is None or window_spin is None:
            return
        self._follow_controls_connected = True
        # These controls are shared by every plot.  Connect them once and let
        # the first plot act as the coordinator for the complete display.
        if getattr(checkbox, "_mitr_follow_controls_connected", False):
            return
        checkbox._mitr_follow_controls_connected = True
        checkbox.toggled.connect(self._set_all_follow_live)
        window_spin.valueChanged.connect(self._set_all_follow_window_minutes)
        self._set_all_follow_live(checkbox.isChecked())

    def _all_operations_plots(self):
        return self.window().findChildren(MITROperationsArchivePlot)

    def _follow_reference_plot(self, plots):
        window = self.window()
        previous = getattr(window, "_mitr_follow_reference_plot", None)
        visible = [plot for plot in plots if plot.isVisible()]
        if previous in visible:
            return previous
        if visible:
            return visible[0]
        if previous in plots:
            return previous
        return self

    @QtCore.Slot(bool)
    def _set_all_follow_live(self, enabled):
        """Apply the shared follow state and capture the current X width."""
        plots = self._all_operations_plots()
        if not plots:
            return

        reference = self._follow_reference_plot(plots)
        window = self.window()
        window._mitr_follow_reference_plot = reference
        window_spin = window.findChild(QtWidgets.QSpinBox, "liveWindowSpinBox")

        if enabled:
            if hasattr(window, "_mitr_follow_has_been_initialized"):
                min_x, max_x = reference.plotItem.getAxis("bottom").range
                span_seconds = max(60.0, float(max_x) - float(min_x))
            else:
                # Before the first redraw, the axis can still have
                # PyQtGraph's placeholder range.  Preserve the display's
                # configured 12-hour time span on startup.
                span_seconds = float(reference.getTimeSpan())
                window._mitr_follow_has_been_initialized = True
            minutes = max(1, int(round(span_seconds / 60.0)))
            if window_spin is not None:
                blocked = window_spin.blockSignals(True)
                window_spin.setValue(minutes)
                window_spin.blockSignals(blocked)
            for plot in plots:
                plot.setFollowWindowMinutes(minutes)
                plot.setFollowLive(True)
            return

        for plot in plots:
            plot.setFollowLive(False)

    @QtCore.Slot(int)
    def _set_all_follow_window_minutes(self, minutes):
        for plot in self._all_operations_plots():
            plot.setFollowWindowMinutes(minutes)

    def _restore_default_follow_window(self):
        """Keep the shared live-window control aligned with X/Y restore."""
        minutes = FOLLOW_LIVE_DEFAULT_WINDOW_MINUTES
        window_spin = self.window().findChild(
            QtWidgets.QSpinBox,
            "liveWindowSpinBox",
        )
        if window_spin is not None:
            blocked = window_spin.blockSignals(True)
            window_spin.setValue(minutes)
            window_spin.blockSignals(blocked)
        self._set_all_follow_window_minutes(minutes)

    @QtCore.Slot(bool)
    def setFollowLive(self, enabled):
        """Toggle a sliding window ending at the current time."""
        self._follow_live = bool(enabled)
        # Once this control has been initialized, X movement is governed by
        # either this timer or the user.  Do not let PyDM's independent
        # updateXAxis path contradict the checkbox state.
        self._manual_x_navigation = True
        if not self._follow_live:
            self._follow_live_timer.stop()
            return

        self.plotItem.disableXAutoRange()
        self.plotItem.setPlotAutoRangeVisibleOnly(visible_only_y=True)
        for axis in getattr(self, "_axes", ()):
            original_range = self.plotItem.axesOriginalRanges.get(axis.name)
            configured_for_autoscale = (
                original_range is None or original_range[0] is None
            )
            if configured_for_autoscale:
                axis.enable_auto_range()
        self._update_follow_live_range()
        self._follow_live_timer.start()
        self._request_follow_archive_refresh()

    @QtCore.Slot(int)
    def setFollowWindowMinutes(self, minutes):
        """Set the width of the sliding live window in whole minutes."""
        self._follow_window_seconds = max(60, int(minutes) * 60)
        if self._follow_live:
            self._update_follow_live_range()
            self._request_follow_archive_refresh()

    def _request_follow_archive_refresh(self):
        """Request the newly selected live window once, not on every tick."""
        self._zoom_refresh_requested = True
        self._zoom_refresh_timer.start()

    def _update_follow_live_range(self):
        if not self._follow_live:
            return
        now = time.time()
        right_padding = min(
            FOLLOW_LIVE_MAX_RIGHT_PADDING_SECONDS,
            max(
                FOLLOW_LIVE_MIN_RIGHT_PADDING_SECONDS,
                self._follow_window_seconds
                * FOLLOW_LIVE_RIGHT_PADDING_FRACTION,
            ),
        )
        max_x = now + right_padding
        min_x = max_x - self._follow_window_seconds
        self.plotItem.setXRange(min_x, max_x, padding=0)

    def _enable_visible_y_autoscale(self, *_args):
        """Fit every Y axis to data in the currently visible X range."""
        self.plotItem.setPlotAutoRangeVisibleOnly(visible_only_y=True)
        # In rectangle mode the user is deliberately selecting both an X and
        # a Y range.  Re-enabling Y autorange here would immediately undo the
        # vertical part of that selection and make rectangle zoom appear
        # broken.  Pan mode continues to restore visible-range Y autoscaling.
        if self.plotItem.getViewBox().state["mouseMode"] == ViewBox.RectMode:
            return
        for axis in getattr(self, "_axes", ()):
            original_range = self.plotItem.axesOriginalRanges.get(axis.name)
            configured_for_autoscale = (
                original_range is None or original_range[0] is None
            )
            if (
                axis.orientation in ("left", "right")
                and configured_for_autoscale
            ):
                axis.enable_auto_range()

    def getArchivePointBudget(self):
        return int(self.optimized_data_bins)

    def setArchivePointBudget(self, value):
        self.optimized_data_bins = max(1, int(value))

    archivePointBudget = QtCore.Property(
        int,
        getArchivePointBudget,
        setArchivePointBudget,
    )

    def _archive_request_completed(self):
        self._archive_pending_since = None
        if self._zoom_refresh_requested:
            self._zoom_refresh_timer.start()

    def _force_archive_refresh(self, *_args):
        # View All enables every Y axis as well; restore any axes intentionally
        # configured with fixed limits.
        for axis in getattr(self, "_axes", ()):
            original_range = self.plotItem.axesOriginalRanges.get(axis.name)
            if original_range is not None and original_range[0] is not None:
                axis.linkedView().enableAutoRange(ViewBox.YAxis, False)
                axis.linkedView().setYRange(*original_range, padding=0)
        self._zoom_refresh_requested = True
        self._zoom_refresh_timer.start()

    def _schedule_zoom_refresh(self, *_args):
        # PyQtGraph emits (view_box, (min_x, max_x)) for programmatic range
        # changes.  At that point the bottom AxisItem can still contain its
        # previous range, so prefer the range carried by the signal.
        manual_change = not _args
        # The follow timer shifts the range once per second without changing
        # its width.  Re-querying (or restarting the debounce timer) for every
        # shift can prevent any archive request from ever running.  Follow
        # enable/window changes explicitly request one refresh instead.
        if self._follow_live and not manual_change:
            return

        try:
            min_x, max_x = _args[-1]
        except (IndexError, TypeError, ValueError):
            min_x, max_x = self.plotItem.getAxis("bottom").range
        current_span = max(0.0, float(max_x) - float(min_x))
        reference_span = self._last_visible_span
        if reference_span is None:
            reference_span = float(self.getTimeSpan())
        self._last_visible_span = current_span

        span_changed = abs(current_span - reference_span) > max(
            1.0,
            0.01 * max(reference_span, 1.0),
        )
        if (
            not span_changed
            and not manual_change
            and not self._zoom_refresh_requested
        ):
            return

        self._zoom_refresh_requested = True
        self._zoom_refresh_timer.start()

    def _archived_curves(self):
        return [
            curve
            for curve in getattr(self, "_curves", ())
            if bool(getattr(curve, "use_archive_data", False))
        ]

    def _refresh_zoomed_range(self):
        """Reload the visible range after pan/zoom using a fixed point budget."""
        if int(getattr(self, "_pending_archive_responses", 0) or 0):
            self._zoom_refresh_requested = True
            return

        self._request_visible_archive_range()

    def _request_visible_archive_range(self, now=None):
        """Reload the visible window at the plot's configured archive density."""
        if now is None:
            now = time.time()

        min_x, max_x = self.plotItem.getAxis("bottom").range
        min_x = float(min_x)
        max_x = float(max_x)
        request_max_x = min(max_x, now)
        if request_max_x - min_x <= 1.0:
            return False

        curves = [curve for curve in self._archived_curves() if curve.isVisible()]
        if not curves:
            return False

        self._zoom_refresh_requested = False
        self._archive_request_queued = True
        self._pending_archive_responses += len(curves)
        self._archive_pending_since = now
        processing_command = f"optimized_{self.optimized_data_bins}"
        for curve in curves:
            curve.archive_data_request_signal.emit(
                min_x,
                request_max_x - 0.001,
                processing_command,
            )
        self.archive_request_started.emit()
        return True

    def _refresh_archived_data(self):
        """Rebuild the visible window at the same density used after a pan."""
        now = time.time()
        pending = int(getattr(self, "_pending_archive_responses", 0) or 0)
        if pending:
            if self._archive_pending_since is None:
                self._archive_pending_since = now
                return
            if now - self._archive_pending_since <= ARCHIVE_REQUEST_STALE_SECONDS:
                return
            # PyDM 1.29 does not clear its pending counter after a failed HTTP
            # request. Allow this periodic refresh to retry a stalled request.
            self._pending_archive_responses = 0
            self._archive_request_queued = False

        self._request_visible_archive_range(now=now)
