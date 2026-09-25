"""Create only IOC-backed simulated devices for the demo worker."""

import os

from bluesky_queueserver import register_device
from diffractometer_controls.demo_devices import create_demo_devices


_demo_prefix = os.environ.get("MITR_EPICS_PREFIX", "demo4dh4:")
globals().update(create_demo_devices(_demo_prefix))

for _name in (
    "stage1", "stage2", "pinhole", "sample_th", "det_psd_x",
    "analyzer1", "analyzer2", "he3psd0", "he3psd7", "cam1",
    "sim_focus_cam",
):
    register_device(_name, depth=3)

for _detector_name in ("cam1", "sim_focus_cam"):
    _detector = globals()[_detector_name]
    for _attr in ("acquire_time", "gain", "offset"):
        _alias = f"{_detector_name}_{_attr}"
        globals()[_alias] = getattr(_detector.cam, _attr)
        globals()[_alias].kind = "hinted"
        register_device(_alias, depth=1)

for _motor in (
    stage1.theta, stage2.theta, stage2.x, pinhole.y, sample_th,
    det_psd_x, analyzer1.curve, analyzer2.curve,
):
    sd.baseline.append(_motor)
