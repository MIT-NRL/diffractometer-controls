from server.devices.demo import create_demo_devices
from diffractometer_controls.site.mitr.profile import pv_prefix
from bluesky_queueserver import register_device
globals().update(create_demo_devices(pv_prefix()))
for _name in ("stage1", "stage2", "pinhole", "sample_th", "det_psd_x", "analyzer1", "analyzer2", "he3psd0", "he3psd7", "usbctr", "usb2408", "cam1", "sim_focus_cam"):
    register_device(_name, depth=2 if _name in ("cam1", "sim_focus_cam") else 3)
for _name in ("cam1", "sim_focus_cam"):
    for _attr in ("acquire_time", "gain", "offset"):
        _alias = f"{_name}_{_attr}"
        globals()[_alias] = getattr(globals()[_name].cam, _attr)
        globals()[_alias].kind = "hinted"
        globals()[_alias].scalar_plan_hidden = True
        register_device(_alias, depth=1)
for _motor in (stage1.theta, stage2.theta, stage2.x, pinhole.y, sample_th, det_psd_x):
    sd.baseline.append(_motor)
