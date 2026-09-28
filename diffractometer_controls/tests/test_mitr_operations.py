import json
import unittest
from unittest.mock import patch

import numpy as np
from pydm.widgets.label import PyDMLabel
from qtpy import QtGui, QtWidgets

from diffractometer_controls.mitr_operations import (
    ARCHIVE_REFRESH_INTERVAL_MS,
    DARK_MINOR_ALARM_COLOR,
    LIGHT_MINOR_ALARM_COLOR,
    MIN_CURVE_CONTRAST,
    MIT_BRIGHT_RED,
    MIT_RED,
    MITROperationsArchiveCurve,
    MITROperationsArchivePlot,
)


class MITROperationsArchiveCurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _curve_with_live_data(self):
        curve = MITROperationsArchiveCurve()
        curve.setBufferSize(10)
        curve.setArchiveBufferSize(10)
        curve.data_buffer[:, -4:] = np.array(
            [
                [100.25, 101.25, 102.25, 103.25],
                [1.0, 2.0, 3.0, 4.0],
            ]
        )
        curve.points_accumulated = 4
        return curve

    def test_archive_response_replaces_reconciled_live_samples(self):
        curve = self._curve_with_live_data()

        curve.receiveArchiveData(
            np.array(
                [
                    [100.0, 101.0, 102.0, 103.0],
                    [10.0, 20.0, 30.0, 40.0],
                ]
            )
        )

        archive = curve.archive_data_buffer[
            :, -curve.archive_points_accumulated :
        ]
        live = curve.data_buffer[:, -curve.points_accumulated :]
        np.testing.assert_array_equal(
            archive,
            np.array(
                [
                    [100.0, 101.0, 102.0, 103.0],
                    [10.0, 20.0, 30.0, 40.0],
                ]
            ),
        )
        np.testing.assert_array_equal(
            live,
            np.array([[103.25], [4.0]]),
        )

    def test_periodic_refresh_matches_pan_density_and_visible_range(self):
        plot = MITROperationsArchivePlot()
        curve = self._curve_with_live_data()
        plot._curves.append(curve)
        plot.plotItem.getAxis("bottom").setRange(90.0, 210.0)
        requests = []
        curve.archive_data_request_signal.connect(
            lambda min_x, max_x, command: requests.append(
                (min_x, max_x, command)
            )
        )

        try:
            with patch(
                "diffractometer_controls.mitr_operations.time.time",
                return_value=200.0,
            ):
                plot._refresh_archived_data()

            periodic_request = requests[-1]
            self.assertTrue(plot._archive_refresh_timer.isActive())
            self.assertEqual(
                plot._archive_refresh_timer.interval(),
                ARCHIVE_REFRESH_INTERVAL_MS,
            )
            self.assertEqual(
                periodic_request,
                (90.0, 199.999, "optimized_500"),
            )
            self.assertEqual(plot._pending_archive_responses, 1)

            plot.archive_data_received()
            requests.clear()
            with patch(
                "diffractometer_controls.mitr_operations.time.time",
                return_value=200.0,
            ):
                plot._refresh_zoomed_range()

            self.assertEqual(requests, [periodic_request])
        finally:
            plot._archive_refresh_timer.stop()
            plot._follow_live_timer.stop()
            plot._zoom_refresh_timer.stop()

    def test_follow_live_restores_configured_y_autoscale(self):
        plot = MITROperationsArchivePlot()
        plot.setYAxes(
            [
                json.dumps(
                    {
                        "name": "Axis 1",
                        "orientation": "left",
                        "label": "Reactor Power",
                        "minRange": -0.2,
                        "maxRange": 6.2,
                        "autoRange": True,
                        "logMode": False,
                    }
                )
            ]
        )
        axis = plot._axes[0]
        axis.disable_auto_range()

        try:
            plot.setFollowLive(True)

            self.assertTrue(axis.auto_range)
        finally:
            plot._archive_refresh_timer.stop()
            plot._follow_live_timer.stop()
            plot._zoom_refresh_timer.stop()

    def test_plot_tracks_light_and_dark_application_palettes(self):
        original_palette = QtGui.QPalette(self.app.palette())
        plot = MITROperationsArchivePlot()
        plot.setYAxes(
            [
                json.dumps(
                    {
                        "name": "Axis 1",
                        "orientation": "left",
                        "label": "Value",
                        "minRange": -1.0,
                        "maxRange": 1.0,
                        "autoRange": True,
                        "logMode": False,
                    }
                )
            ]
        )
        curve = plot.addYChannel(
            name="Dark trace",
            color=QtGui.QColor("black"),
            yAxisName="Axis 1",
            useArchiveData=True,
        )

        light = QtGui.QPalette(original_palette)
        light.setColor(QtGui.QPalette.Base, QtGui.QColor(248, 248, 248))
        light.setColor(QtGui.QPalette.Text, QtGui.QColor(20, 20, 20))
        light.setColor(QtGui.QPalette.Mid, QtGui.QColor(128, 128, 128))
        dark = QtGui.QPalette(original_palette)
        dark.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        dark.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 240))
        dark.setColor(QtGui.QPalette.Mid, QtGui.QColor(100, 100, 100))

        try:
            self.app.setPalette(light)
            self.app.processEvents()
            self.assertEqual(
                plot.getBackgroundColor(),
                QtGui.QColor("white"),
            )
            self.assertEqual(
                plot._axes[0].textPen().color(),
                light.color(QtGui.QPalette.Text),
            )
            self.assertEqual(curve.color, QtGui.QColor("black"))

            self.app.setPalette(dark)
            self.app.processEvents()
            self.assertEqual(
                plot.getBackgroundColor(),
                dark.color(QtGui.QPalette.Base),
            )
            self.assertEqual(
                plot._axes[0].textPen().color(),
                dark.color(QtGui.QPalette.Text),
            )
            self.assertEqual(
                plot._axes[0].pen().color(),
                dark.color(QtGui.QPalette.Text),
            )
            self.assertEqual(plot._axes[0].pen().widthF(), 1.5)
            self.assertGreaterEqual(
                plot._contrast_ratio(curve.color, dark.color(QtGui.QPalette.Base)),
                MIN_CURVE_CONTRAST,
            )

            self.app.setPalette(light)
            self.app.processEvents()
            self.assertEqual(curve.color, QtGui.QColor("black"))
        finally:
            self.app.setPalette(original_palette)
            plot._archive_refresh_timer.stop()
            plot._follow_live_timer.stop()
            plot._zoom_refresh_timer.stop()

    def test_tab_theme_uses_official_mit_red_accents(self):
        original_palette = QtGui.QPalette(self.app.palette())
        window = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(window)
        tabs = QtWidgets.QTabWidget(window)
        tabs.setObjectName("PyDMTabWidget")
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        plot = MITROperationsArchivePlot(page)
        page_layout.addWidget(plot)
        warning_values = []
        for object_name in (
            "reactorPowerSixValueLabel",
            "dwkOneValueLabel",
        ):
            warning_value = PyDMLabel(page)
            warning_value.setObjectName(object_name)
            warning_value.setAlarmSensitiveContent(True)
            warning_value.setAlarmSeverity(1)
            page_layout.addWidget(warning_value)
            warning_values.append(warning_value)
        tabs.addTab(page, "Power")
        layout.addWidget(tabs)

        light = QtGui.QPalette(original_palette)
        light.setColor(QtGui.QPalette.Window, QtGui.QColor("white"))
        light.setColor(QtGui.QPalette.Base, QtGui.QColor("white"))
        light.setColor(QtGui.QPalette.Text, QtGui.QColor("black"))

        try:
            self.app.setPalette(light)
            plot._apply_theme_from_palette()

            style = tabs.styleSheet().lower()
            self.assertIn(MIT_RED, style)
            self.assertIn(MIT_BRIGHT_RED, style)
            self.assertIn("background-color: #ffffff", style)
            self.assertIn(f"color: {LIGHT_MINOR_ALARM_COLOR}", style)
            for warning_value in warning_values:
                warning_value.style().unpolish(warning_value)
                warning_value.style().polish(warning_value)
                self.assertEqual(
                    warning_value.palette().color(QtGui.QPalette.WindowText),
                    QtGui.QColor(LIGHT_MINOR_ALARM_COLOR),
                )

            dark = QtGui.QPalette(light)
            dark.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
            dark.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
            dark.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 240))
            self.app.setPalette(dark)
            plot._apply_theme_from_palette()
            self.assertIn(
                f"color: {DARK_MINOR_ALARM_COLOR}",
                tabs.styleSheet().lower(),
            )
        finally:
            self.app.setPalette(original_palette)
            plot._archive_refresh_timer.stop()
            plot._follow_live_timer.stop()
            plot._zoom_refresh_timer.stop()
            window.deleteLater()

    def test_older_archive_response_does_not_remove_newer_live_samples(self):
        curve = self._curve_with_live_data()

        curve.receiveArchiveData(
            np.array(
                [
                    [90.0, 95.0, 100.0],
                    [8.0, 9.0, 10.0],
                ]
            )
        )

        live = curve.data_buffer[:, -curve.points_accumulated :]
        np.testing.assert_array_equal(
            live,
            np.array(
                [
                    [100.25, 101.25, 102.25, 103.25],
                    [1.0, 2.0, 3.0, 4.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
