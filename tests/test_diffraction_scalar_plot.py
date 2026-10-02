import os
import pathlib
import runpy
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from diffractometer_controls.screens.diffraction import diffraction_live_plot as live_plot_module
from diffractometer_controls.screens.diffraction.diffraction_live_plot import DiffractionLivePlot
from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph


class _RecordingPlot(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.resets = []
        self.summary_points = []
        self.live_summary_points = []
        self.scalar_readouts = []

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

    def update_scalar_readout(self, *args):
        self.scalar_readouts.append(args)

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
                        "plan_args": {"acquire_time": 5.0},
                        "acquisition_monitor": {"elapsed_pv": "TEST:scaler1.T"},
                    },
                )
                FakePV.instances["TEST:scaler1.T"].callback(value=2.0)
                FakePV.instances["TEST:scaler1.S2"].callback(value=1234)
                self._flush()
                self.assertTrue(widget.resets[-1]["show_count_rate_gauge"])
                self.assertEqual(
                    widget.live_summary_points[-1],
                    ("Beam Monitor", 1.0, 1234.0),
                )
                self.assertEqual(
                    widget.scalar_readouts[-1],
                    ("Beam Monitor", 1234.0, 2.0),
                )
            finally:
                controller.shutdown()

    def test_document_elapsed_time_updates_simulated_counter_rate(self):
        widget = _RecordingPlot()
        controller = DiffractionLivePlot(widget)
        try:
            controller.on_document(
                "start",
                {
                    "uid": "sim-counter-run",
                    "plan_name": "count_scalar",
                    "experiment_type": "diffraction",
                    "data_type": "scalar",
                    "detectors": ["sim_usbctr_he3_tube"],
                    "live_plot_fields": {
                        "sim_usbctr_he3_tube": {
                            "label": "CTR2 - He-3 Tube",
                            "role": "signal",
                            "units": "counts",
                            "transport": "document",
                        },
                        "sim_usbctr_time": {
                            "label": "Elapsed Time",
                            "role": "elapsed_time",
                            "units": "s",
                            "transport": "document",
                        },
                    },
                    "plan_args": {"acquire_time": 5.0},
                },
            )
            controller.on_document(
                "descriptor",
                {
                    "uid": "sim-count-monitor",
                    "run_start": "sim-counter-run",
                    "name": "sim_usbctr_he3_tube_monitor",
                    "data_keys": {
                        "sim_usbctr_he3_tube": {"dtype": "number", "shape": []}
                    },
                },
            )
            controller.on_document(
                "descriptor",
                {
                    "uid": "sim-time-monitor",
                    "run_start": "sim-counter-run",
                    "name": "sim_usbctr_time_monitor",
                    "data_keys": {
                        "sim_usbctr_time": {"dtype": "number", "shape": []}
                    },
                },
            )
            controller.on_document(
                "event",
                {
                    "descriptor": "sim-count-monitor",
                    "seq_num": 1,
                    "data": {"sim_usbctr_he3_tube": 800.0},
                },
            )
            controller.on_document(
                "event",
                {
                    "descriptor": "sim-time-monitor",
                    "seq_num": 1,
                    "data": {"sim_usbctr_time": 2.0},
                },
            )
            self._flush()

            self.assertEqual(
                widget.scalar_readouts[-1],
                ("CTR2 - He-3 Tube", 800.0, 2.0),
            )
        finally:
            controller.shutdown()

    def test_sim_beam_monitor_plan_drives_rate_with_and_without_time_stream(self):
        from bluesky import RunEngine
        from ophyd.sim import SynAxis

        startup = pathlib.Path(__file__).resolve().parents[1] / "server" / "startup"
        # This startup file also defines an EPICS motor unrelated to the
        # counter. Replace that motor so the acquisition is fully in memory.
        with mock.patch("ophyd.EpicsMotor", side_effect=lambda prefix, name: SynAxis(name=name)):
            namespace = runpy.run_path(
                str(startup / "01-sim_devices.py"),
                init_globals={"sd": SimpleNamespace(baseline=[])},
            )
        for filename in ("89-plan_helpers.py", "90-plans_scalar.py"):
            path = startup / filename
            namespace["__file__"] = str(path)
            exec(compile(path.read_text(), str(path), "exec"), namespace)
        documents = []
        engine = RunEngine({})
        engine.subscribe(lambda name, doc: documents.append((name, doc)))
        engine(namespace["count_scalar"](
            "sim beam alone", detectors=namespace["sim_usbctr"].beam_monitor,
            acquire_time=1.6, num=2,
        ))

        for legacy_worker in (False, True):
            with self.subTest(legacy_worker=legacy_worker):
                widget = DiffractionPlotWidgetPyQtGraph()
                controller = DiffractionLivePlot(widget)
                ignored_descriptors = set()
                sample_rates = []
                final_readings = []
                elapsed = 0.0
                reset_stamp = None
                try:
                    for name, original in documents:
                        doc = dict(original)
                        if legacy_worker and name == "start":
                            doc["live_plot_fields"] = {
                                key: {k: v for k, v in spec.items() if k != "source"}
                                for key, spec in doc["live_plot_fields"].items()
                                if spec.get("role") != "elapsed_time"
                            }
                        if name == "descriptor" and doc["name"] == "sim_usbctr_time_monitor":
                            ignored_descriptors.add(doc["uid"])
                        if legacy_worker and doc.get("descriptor") in ignored_descriptors:
                            continue
                        controller.on_document(name, doc)
                        self._flush()
                        if name != "event":
                            continue
                        data = doc["data"]
                        if "sim_usbctr_time" in data:
                            elapsed = float(data["sim_usbctr_time"])
                        if "sim_usbctr_beam_monitor" not in data:
                            continue
                        counts = float(data["sim_usbctr_beam_monitor"])
                        stamp = doc["timestamps"]["sim_usbctr_beam_monitor"]
                        if counts == 0:
                            reset_stamp = stamp
                        if legacy_worker and reset_stamp is not None:
                            elapsed = stamp - reset_stamp
                        card = widget._scalar_readout.cards[0]
                        if 0.4 < elapsed < 1.4:
                            sample_rates.append(card.gauge.rate)
                        if doc["descriptor"] not in ignored_descriptors and counts > 0:
                            final_readings.append((counts, card.gauge.rate))
                    self.assertGreater(len(sample_rates), 4)
                    for rate in sample_rates:
                        self.assertGreater(rate, 1800.0)
                        self.assertLess(rate, 3000.0)
                    counts, rate = final_readings[-1]
                    self.assertAlmostEqual(rate, counts / 1.6, delta=1.0)
                    self.assertNotAlmostEqual(rate, counts, delta=100.0)
                    self.assertTrue(widget._scalar_readout.cards[1].isHidden())
                finally:
                    controller.shutdown()
                    widget.deleteLater()

    def test_same_channel_from_simulated_and_real_scalers_stays_separate(self):
        widget = _RecordingPlot()
        controller = DiffractionLivePlot(widget)
        try:
            controller.on_document(
                "start",
                {
                    "uid": "mixed-counter-run",
                    "plan_name": "count_scalar",
                    "experiment_type": "diffraction",
                    "data_type": "scalar",
                    "detectors": [
                        "sim_usbctr_beam_monitor",
                        "usbctr_beam_monitor",
                    ],
                    "live_plot_fields": {
                        "sim_usbctr_beam_monitor": {
                            "label": "CTR1 - Beam Monitor",
                            "role": "signal",
                            "units": "counts",
                            "transport": "document",
                        },
                        "usbctr_beam_monitor": {
                            "label": "CTR1 - Beam Monitor",
                            "role": "signal",
                            "units": "counts",
                            "transport": "document",
                        },
                        "sim_usbctr_time": {
                            "role": "elapsed_time", "transport": "document",
                        },
                        "usbctr_time": {
                            "role": "elapsed_time", "transport": "document",
                        },
                    },
                    "plan_args": {"acquire_time": 2.0},
                },
            )
            for descriptor_uid, data_key in (
                ("sim-beam-monitor", "sim_usbctr_beam_monitor"),
                ("real-beam-monitor", "usbctr_beam_monitor"),
            ):
                controller.on_document(
                    "descriptor",
                    {
                        "uid": descriptor_uid,
                        "run_start": "mixed-counter-run",
                        "name": f"{data_key}_monitor",
                        "data_keys": {data_key: {"dtype": "number", "shape": []}},
                    },
                )
            controller.on_document(
                "event",
                {
                    "descriptor": "sim-beam-monitor",
                    "seq_num": 1,
                    "data": {"sim_usbctr_beam_monitor": 600.0},
                },
            )
            controller.on_document(
                "event",
                {
                    "descriptor": "real-beam-monitor",
                    "seq_num": 1,
                    "data": {"usbctr_beam_monitor": 1200.0},
                },
            )
            self._flush()

            self.assertEqual(
                widget.live_summary_points[-2:],
                [
                    ("Sim USBCTR: Beam Monitor", 1.0, 600.0),
                    ("USBCTR: Beam Monitor", 1.0, 1200.0),
                ],
            )
            self.assertEqual(
                widget.scalar_readouts[-2:],
                [
                    ("Sim USBCTR: Beam Monitor", 600.0, 2.0),
                    ("USBCTR: Beam Monitor", 1200.0, 2.0),
                ],
            )
            for key, elapsed in (("sim_usbctr_time", 0.5), ("usbctr_time", 4.0)):
                controller.on_document("descriptor", {
                    "uid": f"{key}-descriptor", "run_start": "mixed-counter-run",
                    "name": f"{key}_monitor",
                    "data_keys": {key: {"dtype": "number", "shape": []}},
                })
                controller.on_document("event", {
                    "descriptor": f"{key}-descriptor", "seq_num": 1,
                    "data": {key: elapsed},
                })
            self._flush()
            self.assertEqual(widget.scalar_readouts[-2:], [
                ("Sim USBCTR: Beam Monitor", 600.0, 0.5),
                ("USBCTR: Beam Monitor", 1200.0, 4.0),
            ])
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
        self.assertFalse(widget._scalar_readout.isHidden())
        self.assertFalse(widget._scalar_table.isHidden())

        widget.append_summary_point("Beam Monitor", 0.0, 1200.0)
        widget.append_summary_point("He-3 Tube", 0.0, 450.0)
        self.assertEqual(widget._scalar_table.rowCount(), 1)
        self.assertEqual(widget._scalar_table.columnCount(), 3)
        self.assertEqual(widget._scalar_table.item(0, 0).text(), "0")
        self.assertEqual(widget._scalar_table.item(0, 1).text(), "1,200")
        self.assertEqual(widget._scalar_table.item(0, 2).text(), "450")

        widget.update_scalar_readout("Beam Monitor", 1200.0, 2.0)
        first_card = widget._scalar_readout.cards[0]
        self.assertEqual(first_card.name_label.text(), "Beam Monitor")
        self.assertEqual(first_card.total_value.text(), "1,200 cts")
        self.assertEqual(first_card.gauge.rate, 600.0)

        widget.update_scalar_readout("He-3 Tube", 450.0, 2.0)
        second_card = widget._scalar_readout.cards[1]
        self.assertFalse(second_card.isHidden())
        self.assertEqual(first_card.selected_series, "Beam Monitor")
        self.assertEqual(second_card.selected_series, "He-3 Tube")
        self.assertEqual(second_card.total_value.text(), "450 cts")
        self.assertEqual(second_card.gauge.rate, 225.0)

        widget.update_scalar_readout("Beam Monitor", 1500.0, 2.0)
        self.assertEqual(first_card.selected_series, "Beam Monitor")
        self.assertEqual(second_card.selected_series, "He-3 Tube")
        self.assertEqual(first_card.total_value.text(), "1,500 cts")

        widget.update_scalar_readout("Beam Monitor", 15000.0, 30.0)
        self.assertEqual(first_card.total_value.text(), "15,000 cts")
        self.assertEqual(first_card.gauge.rate, 500.0)
        self.assertEqual(first_card.gauge.scale, 100.0)

        widget.update_scalar_readout("CTR3", 90.0, 2.0)
        self.assertFalse(first_card.selector.isHidden())
        self.assertFalse(second_card.selector.isHidden())
        first_card.selector.setCurrentText("CTR3")
        self.assertEqual(first_card.name_label.text(), "CTR3")
        self.assertEqual(first_card.total_value.text(), "90 cts")
        self.assertEqual(first_card.gauge.rate, 45.0)

        widget.reset({"plot_mode": "1d"})
        self._flush()
        self.assertFalse(widget._profile_plot.isHidden())
        self.assertFalse(widget._summary_plot.isHidden())
        self.assertFalse(widget._peak_plot.isHidden())
        self.assertTrue(widget._scalar_readout.isHidden())
        self.assertTrue(widget._scalar_table.isHidden())
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
