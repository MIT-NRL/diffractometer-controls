import pathlib
import runpy
import inspect
import unittest
from types import SimpleNamespace

from bluesky import RunEngine
from ophyd import Component as Cpt, Device, Signal
from ophyd.sim import SynAxis
from ophyd.status import Status

from diffractometer_controls.plan_time_estimation import estimate_plan_runtime


STARTUP_DIR = pathlib.Path(__file__).resolve().parents[1] / "bluesky_config" / "startup"


class _TimedScalar(Device):
    scalar_plan_compatible = True

    acquire_time = Cpt(Signal, value=0.25, kind="config")
    value = Cpt(Signal, value=5.0, kind="hinted")

    def trigger(self):
        self.value.put(self.value.get() + 1)
        status = Status()
        status.set_finished()
        return status


class _TemperatureInputs(Device):
    ti1 = Cpt(Signal, value=21.5, kind="hinted")
    ti2 = Cpt(Signal, value=22.5, kind="hinted")


class _PassiveReadouts(Device):
    temperature = Cpt(_TemperatureInputs, "")
    scalar_readout_channels = ("temperature.ti1", "temperature.ti2")


class _RampSignal(Signal):
    """Passive test signal that changes on every read."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read_count = 0

    def get(self, **kwargs):
        self.read_count += 1
        return float(self.read_count)


def _load_scalar_plan_namespace():
    reactor_power = Signal(name="reactor_power", value=6.0)
    reactor_power.scalar_plan_hidden = True
    camera_gain = Signal(name="cam1_gain", value=1.0)
    camera_gain.scalar_plan_hidden = True
    namespace = {
        "__name__": "__main__",
        "temperature": Signal(name="temperature", value=295.0),
        "reactor_power": reactor_power,
        "cam1_gain": camera_gain,
        "timed_scalar": _TimedScalar(name="timed_scalar"),
        "usb2408": _PassiveReadouts(name="usb2408"),
        "ramp_temperature": _RampSignal(name="ramp_temperature", value=0.0),
        "test_motor": SynAxis(name="test_motor", value=1.5),
    }
    for filename in ("89-plan_helpers.py", "90-plans_scalar.py"):
        path = STARTUP_DIR / filename
        namespace["__file__"] = str(path)
        exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


class TestScalarPlans(unittest.TestCase):
    def setUp(self):
        self.namespace = _load_scalar_plan_namespace()
        self.documents = []
        self.RE = RunEngine({})
        self.RE.subscribe(lambda name, doc: self.documents.append((name, doc)))

    def _documents(self, name):
        return [doc for document_name, doc in self.documents if document_name == name]

    def _events_for_stream(self, stream_name):
        descriptors = {
            doc["uid"]: doc["name"]
            for doc in self._documents("descriptor")
        }
        return [
            doc
            for doc in self._documents("event")
            if descriptors.get(doc["descriptor"]) == stream_name
        ]

    def test_scalar_choices_include_signals_and_marked_devices(self):
        annotation = self.namespace["count_scalar"]._custom_parameter_annotation_
        choices = annotation["parameters"]["detectors"]["devices"]["ScalarReadables"]
        self.assertIn("temperature", choices)
        self.assertIn("timed_scalar", choices)
        self.assertIn("usb2408.temperature.ti1", choices)
        self.assertIn("usb2408.temperature.ti2", choices)
        self.assertNotIn("reactor_power", choices)
        self.assertNotIn("cam1_gain", choices)
        self.assertNotIn("test_motor", choices)
        self.assertEqual(
            annotation["parameters"]["file_type"]["enums"]["ScalarFileType"],
            ["csv", "nexus"],
        )

    def test_passive_board_channel_is_read_directly(self):
        channel = self.namespace["usb2408"].temperature.ti1
        acquisition_devices = self.namespace["_scalar_acquisition_devices"]([channel])
        self.assertEqual(acquisition_devices, [channel])

        self.RE(
            self.namespace["count_scalar"](
                "USB-2408 temperature",
                detectors=channel,
                num=1,
            )
        )
        start = self._documents("start")[0]
        self.assertEqual(start["detectors"], ["usb2408_temperature_ti1"])
        self.assertIn("usb2408_temperature_ti1", start["live_plot_fields"])
        event = self._events_for_stream("primary")[0]
        self.assertEqual(event["data"]["usb2408_temperature_ti1"], 21.5)
        self.assertEqual(
            start["live_plot_fields"]["usb2408_temperature_ti1"]["transport"],
            "document",
        )
        live_events = self._events_for_stream("usb2408_temperature_ti1_monitor")
        self.assertGreaterEqual(len(live_events), 1)
        self.assertEqual(live_events[-1]["data"]["usb2408_temperature_ti1"], 21.5)

    def test_zero_dwell_reads_passive_signal_once(self):
        signal = self.namespace["ramp_temperature"]
        self.RE(
            self.namespace["count_scalar"](
                "Immediate temperature",
                detectors=signal,
                acquire_time=0,
                num=1,
            )
        )
        start = self._documents("start")[0]
        event = self._events_for_stream("primary")[0]
        self.assertEqual(signal.read_count, 1)
        self.assertEqual(event["data"]["ramp_temperature"], 1.0)
        self.assertEqual(start["plan_args"]["acquire_time"], 0.0)
        self.assertEqual(start["det_config"]["software_dwell_time"], 0.0)
        self.assertNotIn("acquisition_monitor", start)

    def test_positive_dwell_averages_repeated_passive_reads(self):
        signal = self.namespace["ramp_temperature"]
        self.RE(
            self.namespace["count_scalar"](
                "Averaged temperature",
                detectors=signal,
                acquire_time=0.12,
                num=1,
            )
        )
        event = self._events_for_stream("primary")[0]
        live_events = self._events_for_stream("ramp_temperature_monitor")
        self.assertGreaterEqual(signal.read_count, 2)
        self.assertGreaterEqual(len(live_events), 2)
        self.assertGreater(
            live_events[-1]["data"]["ramp_temperature"],
            live_events[0]["data"]["ramp_temperature"],
        )
        self.assertGreater(event["data"]["ramp_temperature"], 1.0)
        self.assertLessEqual(
            event["data"]["ramp_temperature"],
            float(signal.read_count),
        )

    def test_count_scalar_records_metadata_and_events(self):
        self.RE(
            self.namespace["count_scalar"](
                "temperature count",
                sample="sample-a",
                detectors=self.namespace["temperature"],
                num=3,
            )
        )
        start = self._documents("start")[0]
        self.assertEqual(start["plan_name"], "count_scalar")
        self.assertEqual(start["experiment_type"], "diffraction")
        self.assertEqual(start["data_type"], "scalar")
        self.assertEqual(start["detector_type"], "scalar")
        self.assertEqual(start["file_type"], "csv")
        self.assertEqual(start["plan_args"]["file_type"], "csv")
        self.assertEqual(
            start["live_plot_fields"]["temperature"]["transport"],
            "document",
        )
        self.assertEqual(start["sample"], "sample-a")
        self.assertEqual(start["num_points"], 3)
        self.assertEqual(len(self._events_for_stream("primary")), 3)
        self.assertEqual(len(self._events_for_stream("temperature_monitor")), 3)

    def test_scalar_file_type_is_validated_and_recorded(self):
        self.RE(
            self.namespace["count_scalar"](
                "NeXus scalar count",
                detectors=self.namespace["temperature"],
                file_type="nexus",
            )
        )
        self.assertEqual(self._documents("start")[0]["file_type"], "nexus")

        with self.assertRaisesRegex(ValueError, "file_type"):
            self.RE(
                self.namespace["count_scalar"](
                    "Invalid export",
                    detectors=self.namespace["temperature"],
                    file_type="text",
                )
            )

    def test_count_scalar_restores_acquire_time(self):
        detector = self.namespace["timed_scalar"]
        self.RE(
            self.namespace["count_scalar"](
                "timed count",
                detectors=detector,
                acquire_time=0.1,
                num=2,
            )
        )
        self.assertEqual(detector.acquire_time.get(), 0.25)
        self.assertEqual(len(self._events_for_stream("primary")), 2)

    def test_scan_scalar_returns_motor_and_records_positions(self):
        motor = self.namespace["test_motor"]
        self.RE(
            self.namespace["scan_scalar"](
                "temperature scan",
                detectors=self.namespace["temperature"],
                motor=motor,
                start_pos=0,
                stop_pos=1,
                num_steps=3,
            )
        )
        start = self._documents("start")[0]
        self.assertEqual(start["plan_name"], "scan_scalar")
        self.assertEqual(start["num_points"], 3)
        self.assertEqual(len(self._events_for_stream("primary")), 3)
        self.assertEqual(len(self._events_for_stream("temperature_monitor")), 3)
        self.assertEqual(motor.position, 1.5)

    def test_scalar_runtime_estimators(self):
        count_estimate = estimate_plan_runtime(
            "count_scalar",
            kwargs={"num": 3, "acquire_time": 2.0, "delay": 0.5},
        )
        scan_estimate = estimate_plan_runtime(
            "scan_scalar",
            kwargs={"num_steps": 4, "acquire_time": 2.0},
        )
        self.assertEqual(count_estimate["estimated_total_time_s"], 7.0)
        self.assertEqual(count_estimate["estimated_total_units"], 3)
        self.assertEqual(scan_estimate["estimated_total_time_s"], 8.0)
        self.assertEqual(scan_estimate["estimated_total_units"], 4)

    def test_simulated_usbctr_publishes_live_monitor_documents(self):
        sim_namespace = runpy.run_path(
            str(STARTUP_DIR / "01-sim_devices.py"),
            init_globals={"sd": SimpleNamespace(baseline=[])},
        )
        detector = sim_namespace["sim_usbctr"]
        namespace = {
            "__name__": "__main__",
            "sim_usbctr": detector,
            "test_motor": SynAxis(name="test_motor", value=0.0),
        }
        for filename in ("89-plan_helpers.py", "90-plans_scalar.py"):
            path = STARTUP_DIR / filename
            namespace["__file__"] = str(path)
            exec(compile(path.read_text(), str(path), "exec"), namespace)

        choices = namespace["count_scalar"]._custom_parameter_annotation_["parameters"][
            "detectors"
        ]["devices"]["ScalarReadables"]
        self.assertNotIn("sim_usbctr", choices)
        self.assertEqual(
            choices,
            [f"sim_usbctr.{channel}" for channel in detector.scalar_channels],
        )
        self.assertNotIn("gauge_volume", inspect.signature(namespace["count_scalar"]).parameters)
        self.assertNotIn("gauge_volume", inspect.signature(namespace["scan_scalar"]).parameters)

        documents = []
        re = RunEngine({})
        re.subscribe(lambda name, doc: documents.append((name, doc)))
        re(
            namespace["count_scalar"](
                "simulated counter",
                detectors=detector.he3_tube,
                acquire_time=0.25,
                num=1,
            )
        )

        start = next(doc for name, doc in documents if name == "start")
        self.assertEqual(start["data_type"], "scalar")
        self.assertEqual(start["detector_type"], "usbctr08")
        self.assertEqual(
            start["live_plot_fields"]["sim_usbctr_he3_tube"]["transport"],
            "document",
        )
        self.assertNotIn("sim_usbctr_beam_monitor", start["live_plot_fields"])
        descriptors = {
            doc["uid"]: doc["name"]
            for name, doc in documents
            if name == "descriptor"
        }
        tube_events = [
            doc
            for name, doc in documents
            if name == "event"
            and descriptors.get(doc["descriptor"]) == "sim_usbctr_he3_tube_monitor"
        ]
        self.assertGreaterEqual(len(tube_events), 2)
        self.assertGreater(
            tube_events[-1]["data"]["sim_usbctr_he3_tube"],
            tube_events[0]["data"]["sim_usbctr_he3_tube"],
        )
        primary_events = [
            doc
            for name, doc in documents
            if name == "event" and descriptors.get(doc["descriptor"]) == "primary"
        ]
        self.assertEqual(
            set(primary_events[0]["data"]),
            {"sim_usbctr_he3_tube"},
        )

    def test_hardware_scaler_records_only_selected_channels(self):
        sim_namespace = runpy.run_path(
            str(STARTUP_DIR / "01-sim_devices.py"),
            init_globals={"sd": SimpleNamespace(baseline=[])},
        )
        detector = sim_namespace["sim_usbctr"]
        namespace = {
            "__name__": "__main__",
            "sim_usbctr": detector,
        }
        for filename in ("89-plan_helpers.py", "90-plans_scalar.py"):
            path = STARTUP_DIR / filename
            namespace["__file__"] = str(path)
            exec(compile(path.read_text(), str(path), "exec"), namespace)

        documents = []
        re = RunEngine({})
        re.subscribe(lambda name, doc: documents.append((name, doc)))
        re(
            namespace["count_scalar"](
                "selected counters",
                detectors=[detector.beam_monitor, detector.he3_tube],
                acquire_time=0.05,
                num=1,
            )
        )

        descriptors = {
            doc["uid"]: doc["name"]
            for name, doc in documents
            if name == "descriptor"
        }
        primary = next(
            doc
            for name, doc in documents
            if name == "event" and descriptors.get(doc["descriptor"]) == "primary"
        )
        self.assertEqual(
            set(primary["data"]),
            {"sim_usbctr_beam_monitor", "sim_usbctr_he3_tube"},
        )


if __name__ == "__main__":
    unittest.main()
