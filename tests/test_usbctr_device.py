import ast
import pathlib
import unittest
from unittest.mock import patch

import numpy as np
from ophyd import Component as Cpt
from ophyd import Device, EpicsSignal, EpicsSignalRO, FormattedComponent as FCpt
from ophyd.device import DeviceStatus
from ophyd.scaler import EpicsScaler
from ophyd.sim import make_fake_device
from ophyd.status import Status


STARTUP_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "server"
    / "devices"
    / "detectors.py"
)


def _load_usbctr08_scaler_class():
    """Load only the device class, without constructing real EPICS devices."""
    tree = ast.parse(STARTUP_FILE.read_text(), filename=str(STARTUP_FILE))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "USBCTR08Scaler"
    )
    namespace = {
        "np": np,
        "Cpt": Cpt,
        "Device": Device,
        "DeviceStatus": DeviceStatus,
        "EpicsScaler": EpicsScaler,
        "EpicsSignal": EpicsSignal,
        "EpicsSignalRO": EpicsSignalRO,
        "FCpt": FCpt,
    }
    exec(
        compile(
            ast.Module(body=[class_node], type_ignores=[]),
            str(STARTUP_FILE),
            "exec",
        ),
        namespace,
    )
    return namespace["USBCTR08Scaler"]


class USBCTR08ScalerTests(unittest.TestCase):
    def setUp(self):
        scaler_class = make_fake_device(_load_usbctr08_scaler_class())
        self.scaler = scaler_class("TEST:scaler1", name="usbctr")
        self.scaler.count.sim_put(0)
        self.scaler.freq.sim_put(10_000_000.0)
        self.scaler.preset_time.sim_put(5.0)
        self.scaler.pulse_frequency_readback.sim_put(999.8)

        # FakeEpicsSignal cannot complete integer puts for a string-valued
        # enum.  Preserve the real stage behavior with an immediate fake set.
        def set_count_mode(value, **_kwargs):
            self.scaler.count_mode.sim_put(value)
            status = Status()
            status.set_finished()
            return status

        self.scaler.count_mode.set = set_count_mode

    def test_stage_uses_actual_clock_and_reprocesses_preset_time(self):
        with patch.object(
            self.scaler.preset_time,
            "set",
            wraps=self.scaler.preset_time.set,
        ) as set_preset_time:
            self.scaler.stage()

        self.assertEqual(self.scaler.freq.get(), 999.8)
        set_preset_time.assert_called_once_with(5.0)

        self.scaler.unstage()
        self.assertEqual(self.scaler.freq.get(), 10_000_000.0)

    def test_stage_rejects_invalid_clock_readback_and_cleans_up(self):
        self.scaler.pulse_frequency_readback.sim_put(0.0)

        with self.assertRaisesRegex(RuntimeError, "frequency readback"):
            self.scaler.stage()

        self.assertEqual(self.scaler.freq.get(), 10_000_000.0)
        self.assertEqual(self.scaler.pulse_run.get(), 0)


if __name__ == "__main__":
    unittest.main()
