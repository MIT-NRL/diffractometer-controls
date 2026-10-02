"""Offline startup and public plan contracts; every external service is mocked."""

from contextlib import ExitStack
import importlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from bluesky import RunEngine
from ophyd import Signal
from ophyd.sim import SynAxis, make_fake_device
from diffractometer_controls.analysis.sim_focus import SimulatedFocusDetector, SimulatedFocusMotor

ROOT = Path(__file__).resolve().parents[1]
STARTUP = ROOT / "server/startup"


def execute(namespace, filename):
    path = STARTUP / filename
    namespace["__file__"] = str(path)
    exec(compile(path.read_text(encoding="utf8"), str(path), "exec"), namespace)


def normalized(value):
    if hasattr(value, "name") and not inspect.isclass(value):
        return {"device": value.name}
    if inspect.isclass(value):
        return value.__module__ + "." + value.__qualname__
    if isinstance(value, dict):
        return {str(key): normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(item) for item in value]
    if value is inspect.Parameter.empty:
        return "<empty>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class ServerStartupTests(unittest.TestCase):
    def test_ordered_simulated_plan_startup_matches_baseline_contracts(self):
        namespace = {"sd": SimpleNamespace(baseline=[]), "__name__": "__main__"}
        with patch("ophyd.EpicsMotor", side_effect=lambda prefix, name: SynAxis(name=name)), \
             patch("ophyd.EpicsSignal", side_effect=lambda prefix, name: Signal(name=name, value=0)), \
             patch("epics.caget", lambda *args, **kwargs: None), patch("epics.caput", lambda *args, **kwargs: None):
            for filename in ("01-sim_devices.py", "89-plan_helpers.py", "90-plans_scalar.py",
                             "91-plans_he3psd.py", "92-plans_imaging.py", "93-adaptive_plans_imaging.py"):
                if filename == "92-plans_imaging.py":
                    motor = SimulatedFocusMotor(name="test_focus", delay=0)
                    camera = SimulatedFocusDetector(name="cam1", motor=motor, image_shape=(32, 32))
                    camera.focus = motor
                    namespace["cam1"] = camera
                    for attribute in ("acquire_time", "gain", "offset"):
                        namespace["cam1_" + attribute] = getattr(camera.cam, attribute)
                execute(namespace, filename)
        baseline = json.loads((ROOT / "tests/reference/server_contracts.json").read_text(encoding="utf8"))["contracts"]
        for name, expected in baseline.items():
            with self.subTest(plan=name):
                function = namespace[name]
                signature = inspect.signature(function)
                actual = {
                    "name": function.__name__, "generator": inspect.isgeneratorfunction(function),
                    "parameters": [{"name": parameter.name, "kind": parameter.kind.name,
                                    "default": normalized(parameter.default),
                                    "annotation": normalized(parameter.annotation)}
                                   for parameter in signature.parameters.values()],
                    "annotations": normalized(getattr(function, "_custom_parameter_annotation_", None)),
                    "doc": inspect.getdoc(function),
                }
                self.assertEqual(actual, expected)

    def test_importing_definitions_does_not_construct_hardware(self):
        # Real device base classes remain intact; constructor and PV calls are
        # tripwires. Importing definitions must never invoke them.
        from ophyd import EpicsSignal, EpicsSignalRO, EpicsMotor
        with patch.object(EpicsSignal, "__init__", side_effect=AssertionError("EPICS constructor")), \
             patch.object(EpicsSignalRO, "__init__", side_effect=AssertionError("EPICS constructor")), \
             patch.object(EpicsMotor, "__init__", side_effect=AssertionError("motor constructor")), \
             patch("epics.caget", side_effect=AssertionError("PV read")), \
             patch("epics.caput", side_effect=AssertionError("PV write")):
            for name in ("server.devices.simulated", "server.devices.motors", "server.devices.detectors",
                         "server.devices.area_detector", "server.plans.helpers", "server.plans.scalar",
                         "server.plans.he3psd", "server.plans.imaging", "server.plans.adaptive_imaging",
                         "server.services.run_status", "server.services.file_directories"):
                importlib.reload(importlib.import_module(name))

    def test_full_numeric_startup_with_fake_devices_and_external_services(self):
        class FakeWriter:
            def __init__(self, *args, **kwargs):
                pass

            def receiver(self, *args):
                pass

            def __call__(self, *args):
                pass

        apstools = ModuleType("apstools")
        callbacks = ModuleType("apstools.callbacks")
        nexus = ModuleType("apstools.callbacks.nexus_writer")
        nexus.NXWriter = FakeWriter
        tiled_plugins = ModuleType("bluesky_tiled_plugins")
        tiled_plugins.TiledWriter = FakeWriter
        databroker = ModuleType("databroker")
        databroker.Broker = FakeWriter
        namespace = {"__name__": "__main__"}
        from ophyd import EpicsSignal, EpicsSignalRO
        fake_signal = make_fake_device(EpicsSignal)
        fake_readonly = make_fake_device(EpicsSignalRO)
        from server.services import file_directories
        original_factory = file_directories.create_service

        def fake_directory_service(context):
            definitions = original_factory(context)
            definitions["_start_file_dir_choices_stream"] = Mock()
            return definitions

        with ExitStack() as guards:
            guards.enter_context(patch.dict(sys.modules, {
                "apstools": apstools, "apstools.callbacks": callbacks,
                "apstools.callbacks.nexus_writer": nexus, "bluesky_tiled_plugins": tiled_plugins,
                "databroker": databroker,
            }))
            guards.enter_context(patch.dict("os.environ", {"TILED_API_KEY": "offline-test-key"}))
            guards.enter_context(patch("epics.caget", return_value=0))
            guards.enter_context(patch("epics.caput", return_value=True))
            guards.enter_context(patch("bluesky.callbacks.zmq.Publisher", return_value=FakeWriter()))
            guards.enter_context(patch("bluesky.utils.PersistentDict", return_value={}))
            guards.enter_context(patch("tiled.client.from_uri", return_value={"4dh4": {}, "testdb": {}}))
            guards.enter_context(patch("atexit.register"))
            guards.enter_context(patch("signal.signal"))
            guards.enter_context(patch.object(file_directories, "create_service", fake_directory_service))
            # Load the real classes first, then replace startup construction
            # with ophyd fake classes. No IOC/PVA/socket loop is started.
            for module_name in ("motors", "detectors", "area_detector"):
                module = importlib.import_module("server.devices." + module_name)
                from ophyd import Device
                for name, value in vars(module).copy().items():
                    if inspect.isclass(value) and issubclass(value, Device) and value.__module__ == module.__name__:
                        guards.enter_context(patch.object(module, name, make_fake_device(value)))
            def fake_camera(*args, **kwargs):
                motor = SynAxis(name="camera_focus")
                motor.user_readback = motor.readback
                camera = SimulatedFocusDetector(name=kwargs.get("name", "cam1"), motor=motor, image_shape=(32, 32))
                camera.focus = motor
                camera.x = SynAxis(name="camera_x")
                camera.x.user_readback = camera.x.readback
                camera.cam.nd_attributes_file = Signal(name="attributes", value="")
                return camera
            # TIFF filestore paths are POSIX paths on the Linux IOC host;
            # use an in-memory camera on Windows instead of constructing them.
            from server.devices import area_detector
            guards.enter_context(patch.object(area_detector, "MyZWODetector", side_effect=fake_camera))
            guards.enter_context(patch("ophyd.EpicsSignal", fake_signal))
            guards.enter_context(patch("ophyd.EpicsSignalRO", fake_readonly))
            guards.enter_context(patch("ophyd.EpicsMotor", side_effect=lambda prefix, name: SynAxis(name=name)))
            for file in sorted(STARTUP.glob("[0-9]*.py")):
                with self.subTest(startup=file.name):
                    execute(namespace, file.name)
            for name in ("RE", "sd", "cam1", "he3psd0", "he3psd7", "usb2408", "usbctr",
                         "sim_he3psd0", "sim_he3psd1", "sim_usbctr", "sim_focus_cam", "sim_focus_motor",
                         "count_scalar", "scan_scalar", "tomo_scan", "adaptive_imaging_focus_scan",
                         "_run_status_publisher", "he3_nexus_writer", "scalar_data_writer"):
                self.assertIn(name, namespace.keys())
            self.assertIsInstance(namespace["RE"], RunEngine)
            self.assertGreater(len(namespace["sd"].baseline), 10)
            namespace["_start_file_dir_choices_stream"].assert_called_once()
            self.assertIsNotNone(namespace["he3_nexus_writer_subscription"])
            self.assertIsNotNone(namespace["scalar_data_writer_subscription"])
        # Remove modules created with a deliberately stubbed external writer;
        # real writer-output tests still require the actual apstools package.
        sys.modules.pop("server.writers.he3_nexus", None)
        sys.modules.pop("server.writers.scalar", None)


if __name__ == "__main__":
    unittest.main()
