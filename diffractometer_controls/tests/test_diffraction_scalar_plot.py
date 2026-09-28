import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from diffractometer_controls import diffraction_live_plot as live_plot_module
from diffractometer_controls.diffraction_live_plot import DiffractionLivePlot
from diffractometer_controls.diffraction_live_plot_pyqtgraph import (
    DiffractionPlotWidgetPyQtGraph,
)


class _RecordingPlot(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.resets = []
        self.summary_points = []
        self.live_summary_points = []

    def reset(self, config=None):
        self.resets.append(dict(config or {}))

    def clear_live_previews(self):
        return None

    def set_profile(self, *args):
        return None

    def update_live_profile(self, *args):
        return None

    def append_summary_point(self, *args):
        self.summary_points.append(args)

    def update_live_summary_point(self, *args):
        self.live_summary_points.append(args)

    def append_peak_point(self, *args):
        return None

    def set_status(self, *args):
        return None


class TestDiffractionScalarPlot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _flush(self):
        self.app.processEvents()
        self.app.processEvents()

    def test_scalar_scan_uses_one_plot_and_live_then_final_points(self):
        widget = _RecordingPlot()
        controller = DiffractionLivePlot(widget)
        try:
            start = {
                "uid": "scalar-run",
                "title": "temperature scan",
                "plan_name": "scan_scalar",
                "experiment_type": "diffraction",
                "data_type": "scalar",
                "detectors": ["temperature"],
                "motors": ["test_motor"],
                "plan_pattern_args": {
                    "start_pos": 0.0,
                    "stop_pos": 2.0,
                    "num_steps": 3,
                },
                "live_plot_fields": {
                    "temperature": {
                        "label": "Temperature",
                        "role": "signal",
                        "units": "C",
                        "transport": "document",
                    }
                },
            }
            controller.on_document("start", start)
            controller.on_document(
                "descriptor",
                {
                    "uid": "temperature-monitor",
                    "run_start": "scalar-run",
                    "name": "temperature_monitor",
                    "data_keys": {"temperature": {"dtype": "number", "shape": []}},
                },
            )
            controller.on_document(
                "event",
                {
                    "descriptor": "temperature-monitor",
                    "seq_num": 1,
                    "data": {"temperature": 299.5},
                },
            )
            controller.on_document(
                "event",
                {
                    "descriptor": "temperature-monitor",
                    "seq_num": 2,
                    "data": {"temperature": 299.75},
                },
            )
            controller.on_document(
                "descriptor",
                {
                    "uid": "primary",
                    "run_start": "scalar-run",
                    "name": "primary",
                    "data_keys": {
                        "temperature": {"dtype": "number", "shape": []},
                        "test_motor": {"dtype": "number", "shape": []},
                    },
                },
            )
            controller.on_document(
                "event",
                {
                    "descriptor": "primary",
                    "seq_num": 1,
                    "data": {"temperature": 300.0, "test_motor": 0.0},
                },
            )
            self._flush()

            self.assertEqual(widget.resets[-1]["plot_mode"], "scalar")
            self.assertEqual(widget.resets[-1]["summary_y_label"], "Temperature (C)")
            self.assertEqual(widget.resets[-1]["summary_title"], "Temperature vs Position")
            self.assertEqual(
                widget.live_summary_points[-2:],
                [
                    ("Temperature", 0.0, 299.5),
                    ("Temperature", 0.0, 299.75),
                ],
            )
            self.assertEqual(widget.summary_points[-1], ("Temperature", 0.0, 300.0))
        finally:
            controller.shutdown()

    def test_old_psd_run_defaults_to_1d_layout(self):
        widget = _RecordingPlot()
        controller = DiffractionLivePlot(widget)
        try:
            controller.on_document(
                "start",
                {
                    "uid": "old-psd-run",
                    "plan_name": "scan_he3",
                    "experiment_type": "diffraction",
                    "detectors": ["he3psd0"],
                },
            )
            self._flush()
            self.assertEqual(widget.resets[-1]["plot_mode"], "1d")
        finally:
            controller.shutdown()

    def test_epics_scalar_metadata_arms_live_channel_access_preview(self):
        class FakePV:
            instances = {}

            def __init__(self, pvname, auto_monitor=True):
                self.pvname = pvname
                self.callback = None
                self.instances[pvname] = self

            def add_callback(self, callback):
                self.callback = callback
                return 1

            def remove_callback(self, callback_index):
                self.callback = None

        widget = _RecordingPlot()
        with mock.patch.object(live_plot_module, "PV", FakePV):
            controller = DiffractionLivePlot(widget)
            try:
                controller.on_document(
                    "start",
                    {
                        "uid": "counter-run",
                        "plan_name": "count_scalar",
                        "experiment_type": "diffraction",
                        "data_type": "scalar",
                        "detectors": ["usbctr"],
                        "live_plot_fields": {
                            "usbctr_beam_monitor": {
                                "label": "Beam Monitor",
                                "role": "signal",
                                "units": "counts",
                                "transport": "ca",
                                "pv": "TEST:scaler1.S2",
                            }
                        },
                    },
                )
                FakePV.instances["TEST:scaler1.S2"].callback(value=1234)
                self._flush()
                self.assertEqual(
                    widget.live_summary_points[-1],
                    ("Beam Monitor", 1.0, 1234.0),
                )
            finally:
                controller.shutdown()

    def test_hidden_viewer_keeps_document_data_and_rearms_live_pvs(self):
        class FakePV:
            instances = {}

            def __init__(self, pvname, auto_monitor=True):
                self.pvname = pvname
                self.callback = None
                self.instances[pvname] = self

            def add_callback(self, callback):
                self.callback = callback
                return 1

            def remove_callback(self, callback_index):
                self.callback = None

        widget = _RecordingPlot()
        with mock.patch.object(live_plot_module, "PV", FakePV):
            controller = DiffractionLivePlot(widget)
            try:
                controller.deactivate()
                controller.on_document(
                    "start",
                    {
                        "uid": "hidden-scalar-run",
                        "title": "hidden temperature scan",
                        "plan_name": "scan_scalar",
                        "experiment_type": "diffraction",
                        "data_type": "scalar",
                        "detectors": ["temperature"],
                        "motors": ["test_motor"],
                        "plan_pattern_args": {
                            "start_pos": 0.0,
                            "stop_pos": 1.0,
                            "num_steps": 2,
                        },
                        "live_plot_fields": {
                            "temperature": {
                                "label": "Temperature",
                                "role": "signal",
                                "transport": "ca",
                                "pv": "TEST:TEMP",
                            }
                        },
                    },
                )
                controller.on_document(
                    "descriptor",
                    {
                        "uid": "hidden-primary",
                        "run_start": "hidden-scalar-run",
                        "name": "primary",
                        "data_keys": {
                            "temperature": {"dtype": "number", "shape": []},
                            "test_motor": {"dtype": "number", "shape": []},
                        },
                    },
                )
                controller.on_document(
                    "event",
                    {
                        "descriptor": "hidden-primary",
                        "seq_num": 1,
                        "data": {"temperature": 301.0, "test_motor": 0.0},
                    },
                )
                self._flush()

                self.assertEqual(
                    widget.summary_points[-1],
                    ("Temperature", 0.0, 301.0),
                )
                self.assertNotIn("TEST:TEMP", FakePV.instances)

                controller.activate()
                self._flush()
                self.assertIsNotNone(FakePV.instances["TEST:TEMP"].callback)
            finally:
                controller.shutdown()

    def test_pyqtgraph_widget_switches_between_scalar_and_1d_layouts(self):
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        widget = DiffractionPlotWidgetPyQtGraph()
        host_layout.addWidget(widget)
        host.resize(700, 650)
        host.show()
        widget.reset({"plot_mode": "scalar"})
        self._flush()
        self.assertTrue(widget._profile_plot.isHidden())
        self.assertTrue(widget._peak_plot.isHidden())
        self.assertFalse(widget._summary_plot.isHidden())

        widget.reset({"plot_mode": "1d"})
        self._flush()
        self.assertFalse(widget._profile_plot.isHidden())
        self.assertFalse(widget._summary_plot.isHidden())
        self.assertFalse(widget._peak_plot.isHidden())
        self.assertLessEqual(
            abs(widget._summary_plot.width() - widget._peak_plot.width()),
            2,
        )
        self.assertLessEqual(
            widget._peak_plot.geometry().right(),
            widget.contentsRect().right(),
        )
        host.close()
        widget.deleteLater()
        host.deleteLater()

    def test_pyqtgraph_old_psd_profiles_remain_readable(self):
        widget = DiffractionPlotWidgetPyQtGraph()
        widget.reset({"plot_mode": "1d"})
        x_values = np.linspace(-1.0, 1.0, 9)
        widget.set_profile("PSD 0", x_values, np.arange(9, dtype=float))
        widget.set_profile("PSD 0", x_values, np.arange(9, dtype=float) + 1.0)

        older_line = widget._profile_history["PSD 0"][0]
        self.assertGreaterEqual(older_line.opts["pen"].color().alphaF(), 0.75)
        widget.deleteLater()

    def test_pyqtgraph_scalar_live_traces_have_a_legend(self):
        widget = DiffractionPlotWidgetPyQtGraph()
        widget.reset({"plot_mode": "scalar"})
        widget.update_live_summary_point("Beam Monitor", 1.0, 1200.0)
        widget.update_live_summary_point("He-3 Tube", 1.0, 450.0)

        labels = {
            str(getattr(label, "text", ""))
            for _sample, label in widget._summary_legend.items
        }
        self.assertEqual(labels, {"Beam Monitor", "He-3 Tube"})
        self.assertIs(widget._summary_legend.scene(), widget._summary_plot.scene())
        self.assertEqual(
            widget._summary_live_lines["Beam Monitor"].opts["pen"].color().name(),
            "#1f77b4",
        )
        self.assertEqual(
            widget._summary_live_lines["He-3 Tube"].opts["pen"].color().name(),
            "#ff7f0e",
        )
        widget.deleteLater()


if __name__ == "__main__":
    unittest.main()
