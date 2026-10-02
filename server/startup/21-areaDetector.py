from server.config.paths import detector_xml
from server.devices.area_detector import ZWODetectorCam, QHYDetectorCam, ZWODetector, QHYDetector, MyTIFFPlugin, MyHDF5Plugin, SingleTriggerPause, MyZWODetector, MyQHYDetector, SimAreaDetector
import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsMotor, Signal,
                   cam)
from ophyd.device import DeviceStatus, do_not_wait_for_lazy_connection
from ophyd.status import Status, SubscriptionStatus

from ophyd.areadetector import (AreaDetector, SingleTrigger, SimDetector,
                                ImagePlugin, StatsPlugin, TIFFPlugin, HDF5Plugin, TransformPlugin,
                                CamBase, ADComponent as ADCpt, EpicsSignalWithRBV as SignalWithRBV,
                                DetectorBase)
from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite, FileStoreHDF5IterativeWrite
from ophyd import cam
from bluesky_queueserver import register_device
from epics import caput, caget, cainfo
import uuid
from datetime import datetime, timedelta

_CAMERA_ADVANCED_AXIS_ATTRS = ("acquire_time", "gain", "offset")

def _register_camera_advanced_axes(detector_name, detector):
    """Register only the camera controls deliberately supported as scan axes.

    Registering the complete AreaDetector tree at depth 3 makes Queue Server
    instantiate every lazy camera signal while it builds its device inventory.
    Some of the ZWO driver's optional ``*_RBV`` records are not present, so
    that eager discovery aborts startup.  Top-level aliases keep the supported
    controls addressable by Queue Server without traversing the camera tree.
    """
    camera = detector.cam
    with do_not_wait_for_lazy_connection(camera):
        for attr in _CAMERA_ADVANCED_AXIS_ATTRS:
            axis_name = f"{detector_name}_{attr}"
            axis = getattr(camera, attr)
            axis.kind = "hinted"
            axis.scalar_plan_hidden = True
            globals()[axis_name] = axis
            register_device(axis_name, depth=1)

# Enable when using the ZWO camera
if 1:
    cam1 = MyZWODetector(prefix='4dh4:',name='cam1',read_attrs=['tiff1','stats1.total','focus','x'])
    cam1.stats1.total.kind = "hinted"
    cam1.focus.user_readback.kind = "normal"
    cam1.x.user_readback.kind = "normal"
    register_device("cam1", depth=2)
    _register_camera_advanced_axes("cam1", cam1)
    cam1.cam.nd_attributes_file.set(detector_xml("tomoDetectorAttributes.xml"))
    caput("4dh4:TIFF1:AutoSave", 0) #Ensure the TIFF plugin does not auto save to prevent overwriting

    def _abort_detector_acquire(det):
        """Best-effort abort used by RunEngine pause hook."""
        try:
            if hasattr(det, "stop"):
                det.stop(success=False)
        except Exception:
            pass
        try:
            if hasattr(det, "tiff1") and hasattr(det.tiff1, "capture"):
                det.tiff1.capture.put(0, wait=False)
        except Exception:
            pass
        try:
            if hasattr(det, "cam") and hasattr(det.cam, "abort"):
                det.cam.abort.put(1, wait=False)
        except Exception:
            pass
        try:
            if hasattr(det, "cam") and hasattr(det.cam, "acquire"):
                det.cam.acquire.put(0, wait=False)
        except Exception:
            pass

    try:
        _existing_state_hook = RE.state_hook
        if getattr(_existing_state_hook, "_detector_abort_wrapper", False):
            _previous_state_hook = getattr(_existing_state_hook, "_detector_abort_previous", None)
        else:
            _previous_state_hook = _existing_state_hook

        def _state_hook_with_detector_abort(*args, _previous_hook=_previous_state_hook, **kwargs):
            state = kwargs.get("new_state", kwargs.get("state", None))
            str_args = [a for a in args if isinstance(a, str)]
            if state is None and str_args:
                state = str_args[0]
            if isinstance(state, str):
                state = state.strip().lower()

            if state in ("pausing", "paused", "suspending", "suspended"):
                _abort_detector_acquire(cam1)

            if callable(_previous_hook):
                return _previous_hook(*args, **kwargs)
            return None

        _state_hook_with_detector_abort._detector_abort_wrapper = True
        _state_hook_with_detector_abort._detector_abort_previous = _previous_state_hook
        RE.state_hook = _state_hook_with_detector_abort
    except Exception:
        pass

# Enable when using the QHY camera
if 0:
    cam1 = MyQHYDetector(prefix='4dh4:',name='cam1',read_attrs=['tiff1','stats1.total'])
    cam1.cam.nd_attributes_file.set(detector_xml("tomoDetectorAttributes.xml"))

sd.baseline.append(cam1.focus)
sd.baseline.append(cam1.x)

# Need to add stage sigs for create directory depth
