from server.devices.simulated import _build_position_axis, SimHE3PSD, SimUSBCTR08Scaler
import numpy as np
import threading
import time
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV,
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from bluesky_queueserver import register_device

try:
    from diffractometer_controls.analysis.sim_focus import SimulatedFocusDetector, SimulatedFocusMotor
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.analysis.sim_focus import SimulatedFocusDetector, SimulatedFocusMotor

sim_motor = EpicsMotor("4dh4:m6", name="sim_motor")

sim_he3psd0 = SimHE3PSD(
    name="sim_he3psd0",
    motor=sim_motor,
    peak_motor=2.9,
    amplitude_scale=3600.0,
    baseline_counts=28.0,
    width=18.0,
    center_offset=-24.0,
    center_motor_scale=16.0,
    shoulder_fraction=0.18,
    shoulder_offset=12.0,
    shoulder_width_scale=1.4,
    background_phase=0.2,
    noise_scale=0.9,
    random_seed=4,
)

sim_he3psd1 = SimHE3PSD(
    name="sim_he3psd1",
    motor=sim_motor,
    peak_motor=3.2,
    amplitude_scale=3000.0,
    baseline_counts=24.0,
    width=24.0,
    center_offset=26.0,
    center_motor_scale=-13.0,
    shoulder_fraction=0.10,
    shoulder_offset=-18.0,
    shoulder_width_scale=2.1,
    background_phase=1.1,
    noise_scale=1.1,
    random_seed=17,
)

sim_usbctr = SimUSBCTR08Scaler(name="sim_usbctr")
register_device("sim_usbctr", depth=2)

# End-to-end adaptive-focus simulation. These names appear as selectable
# detector/motor devices in the Queue Server plan editor.
sim_focus_motor = SimulatedFocusMotor(name="sim_focus_motor", value=0.0)
sim_focus_cam = SimulatedFocusDetector(
    name="sim_focus_cam",
    motor=sim_focus_motor,
    best_focus=0.0,
)
register_device("sim_focus_motor", depth=1)
# Keep the detector shallow and expose only its supported public scan axis.
register_device("sim_focus_cam", depth=2)
for _camera_axis_attr in ("acquire_time",):
    _camera_axis_name = f"sim_focus_cam_{_camera_axis_attr}"
    _camera_axis = getattr(sim_focus_cam.cam, _camera_axis_attr)
    _camera_axis.kind = "hinted"
    _camera_axis.scalar_plan_hidden = True
    globals()[_camera_axis_name] = _camera_axis
    register_device(_camera_axis_name, depth=1)
sd.baseline.append(sim_focus_motor)
