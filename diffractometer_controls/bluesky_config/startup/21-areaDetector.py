import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsMotor, Signal,
                   cam)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus

from ophyd.areadetector import (AreaDetector, SingleTrigger, SimDetector, 
                                ImagePlugin, StatsPlugin, TIFFPlugin, HDF5Plugin, TransformPlugin,
                                CamBase, ADComponent as ADCpt, EpicsSignalWithRBV as SignalWithRBV,
                                DetectorBase)
from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite, FileStoreHDF5IterativeWrite
from ophyd import cam
from epics import caput, caget, cainfo
import uuid
from datetime import datetime, timedelta

class ZWODetectorCam(CamBase):
    offset = ADCpt(SignalWithRBV, "Offset")
    abort = ADCpt(EpicsSignal, "Abort")

class QHYDetectorCam(CamBase):
    offset = ADCpt(SignalWithRBV, "Offset")
    readmode = ADCpt(SignalWithRBV, "ReadMode")
    abort = ADCpt(EpicsSignal, "Abort")

class ZWODetector(DetectorBase):
    cam = ADCpt(ZWODetectorCam, "cam1:")

class QHYDetector(DetectorBase):
    cam = ADCpt(QHYDetectorCam, "cam1:")


class MyTIFFPlugin(FileStoreTIFFIterativeWrite,TIFFPlugin):
    folder_name = Cpt(Signal, value="", kind="config")  # Add a folder_name component
    create_directory = Cpt(EpicsSignal, "CreateDirectory", kind="config")  # Add the CreateDirectory PV

    def make_filename(self):
        folder_name = self.folder_name.get()
        filename = self.file_name.get() + "_" + str(uuid.uuid4())[:8]
        formatter = datetime.now().strftime
        write_path = formatter(self.write_path_template) + "/" + folder_name + "/"
        read_path = formatter(self.read_path_template) + "/" + folder_name + "/"
        return filename, read_path, write_path
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stage_sigs.update(
            [("file_template","%s%s_%4.4d.tif"),
             ("auto_save", 1),
                
            ]
        )
        
    def stage(self):
        self.create_directory.set(-3).wait()
        return super().stage()
    # pass

class MyHDF5Plugin(FileStoreHDF5IterativeWrite,HDF5Plugin):
    layout_filename = Cpt(EpicsSignal, "XMLFileName", kind="config", string=True)
    layout_filename_valid = Cpt(EpicsSignal, "XMLValid_RBV", kind="omitted", string=True)
    nd_attr_status = Cpt(EpicsSignal, "NDAttributesStatus", kind="omitted", string=True)

class SingleTriggerPause(SingleTrigger):
    """SingleTrigger variant that aborts camera acquisition on stop().

    This is important for immediate RunEngine pause, which calls stop() on
    devices. If acquisition is in-flight, forcing TIFF autosave off and
    cam.acquire=0 can prevent finishing/writing the interrupted frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acq_status = None
        self._acq_in_progress = False

    def trigger(self):
        # Re-apply staged AutoSave before each acquisition. This makes resume
        # reliable even if RE state transitions differ across environments.
        try:
            if hasattr(self, "tiff1") and hasattr(self.tiff1, "auto_save"):
                staged_autosave = None
                try:
                    staged_autosave = self.tiff1.stage_sigs.get("auto_save")
                except Exception:
                    staged_autosave = None
                if staged_autosave is not None:
                    self.tiff1.auto_save.put(staged_autosave, wait=False)
        except Exception:
            pass

        self._acq_status = super().trigger()
        self._acq_in_progress = True
        try:
            self._acq_status.add_callback(lambda **kwargs: setattr(self, "_acq_in_progress", False))
        except Exception:
            pass
        return self._acq_status

    @staticmethod
    def _mark_status_aborted(status_obj):
        if status_obj is None:
            return
        try:
            if getattr(status_obj, "done", False):
                return
        except Exception:
            pass

        err = RuntimeError("Detector acquisition aborted by immediate pause/stop().")
        for method_name in ("set_exception", "_finished"):
            try:
                method = getattr(status_obj, method_name)
            except Exception:
                continue
            try:
                if method_name == "_finished":
                    method(success=False)
                else:
                    method(err)
                return
            except Exception:
                continue

    def stop(self, *, success=False):
        was_acquiring = False
        was_capturing = False
        interrupted_trigger = False

        try:
            interrupted_trigger = bool(self._acq_in_progress)
        except Exception:
            interrupted_trigger = False
        if not interrupted_trigger and self._acq_status is not None:
            try:
                interrupted_trigger = not bool(self._acq_status.done)
            except Exception:
                interrupted_trigger = False

        # Disable autosave first so interrupt does not commit the in-flight file.
        if interrupted_trigger and hasattr(self, "tiff1") and hasattr(self.tiff1, "auto_save"):
            try:
                self.tiff1.auto_save.put(0, wait=True)
            except Exception:
                pass

        # Stop file plugin capture first to avoid committing partial frames.
        if hasattr(self, "tiff1") and hasattr(self.tiff1, "capture"):
            try:
                was_capturing = bool(self.tiff1.capture.get())
            except Exception:
                was_capturing = False
            try:
                if was_capturing:
                    self.tiff1.capture.put(0, wait=False)
            except Exception:
                pass

        # Abort in-flight camera exposure ASAP.
        if hasattr(self, "cam") and hasattr(self.cam, "acquire"):
            try:
                was_acquiring = bool(self.cam.acquire.get())
            except Exception:
                was_acquiring = False
            # Some AD camera drivers expose a dedicated abort command that is
            # more immediate than toggling Acquire to 0.
            if hasattr(self.cam, "abort"):
                try:
                    self.cam.abort.put(1, wait=False)
                except Exception:
                    pass
            try:
                if was_acquiring:
                    self.cam.acquire.put(0, wait=False)
            except Exception:
                pass

        if interrupted_trigger or was_acquiring or was_capturing:
            self._mark_status_aborted(self._acq_status)

        self._acq_status = None
        self._acq_in_progress = False
        stop_ret = super().stop(success=success)
        # Restore to staged AutoSave setting (not a hardcoded global value).
        # This keeps manual EPICS operation safe when plugin is not staged.
        try:
            if hasattr(self, "tiff1") and hasattr(self.tiff1, "auto_save"):
                staged_autosave = None
                try:
                    staged_autosave = self.tiff1.stage_sigs.get("auto_save")
                except Exception:
                    staged_autosave = None
                if staged_autosave is not None:
                    self.tiff1.auto_save.put(staged_autosave, wait=False)
        except Exception:
            pass
        return stop_ret


class MyZWODetector(SingleTriggerPause, ZWODetector):
    cam = Cpt(ZWODetectorCam, "cam1:")
    image = Cpt(ImagePlugin, suffix='image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')
    transform1 = Cpt(TransformPlugin, "Trans1:")

    # Add the motors to the detector
    # cam_focus = EpicsMotorCustom("4dh4:m12",name="cam_focus",labels=["positioner"])
    # cam_x = EpicsMotorCustom("4dh4:m1",name="cam_x",labels=["positioner"])
    focus = Cpt(EpicsMotorCustom, "m12", name="focus", labels=["positioner"])
    x = Cpt(EpicsMotorCustom, "m1", name="x", labels=["positioner"])

    tiff1 = Cpt(
        MyTIFFPlugin,
        "TIFF1:",
        write_path_template="/home/mitr_4dh4/Data/%Y/",
        read_path_template="/home/mitr_4dh4/Data/%Y/",
    )

    
    # hdf1 = Cpt(
    #     MyHDF5Plugin,
    #     "HDF1:",
    #     write_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",
    #     read_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",        
    # )

class MyQHYDetector(SingleTriggerPause, QHYDetector):
    cam = Cpt(ZWODetectorCam, "cam1:")
    image = Cpt(ImagePlugin, suffix='image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')

    tiff1 = Cpt(
        MyTIFFPlugin,
        "TIFF1:",
        # write_path_template="/home/mitr_4dh4/Data/%Y/PSI_Experiment/",
        # read_path_template="/home/mitr_4dh4/Data/%Y/PSI_Experiment/",
        write_path_template="/home/mitr_4dh4/Data/%Y/",
        read_path_template="/home/mitr_4dh4/Data/%Y/",
    )
    # hdf1 = Cpt(
    #     MyHDF5Plugin,
    #     "HDF1:",
    #     write_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",
    #     read_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",        
    # )

class SimAreaDetector(SingleTriggerPause, SimDetector):
    cam = Cpt(cam.SimDetectorCam, "cam1:")
    image = Cpt(ImagePlugin, suffix='image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')

    tiff1 = Cpt(
        MyTIFFPlugin,
        "TIFF1:",
        write_path_template="/home/mitr_4dh4/Data/TestData/%Y/%m/%d/",
        read_path_template="/home/mitr_4dh4/Data/TestData/%Y/%m/%d/",
    )
    # hdf1 = Cpt(
    #     MyHDF5Plugin,
    #     "HDF1:",
    #     write_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",
    #     read_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",        
    # )

# Enable when using the ZWO camera
if 1:
    cam1 = MyZWODetector(prefix='4dh4:',name='cam1',read_attrs=['tiff1','stats1.total'])
    cam1.cam.nd_attributes_file.set("/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoDetectorAttributes.xml") 
    # caput("4dh4:TIFF1:CreateDirectory", -3)
    caput("4dh4:TIFF1:AutoSave", 0) #Ensure the TIFF plugin does not auto save to prevent overwriting

    def _abort_detector_acquire(det):
        """Best-effort abort used by RunEngine pause hook."""
        # Call device stop() first so it can handle interrupt cleanup.
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

    # In Queue Server, immediate pause can occur while RE waits on trigger
    # status. Hook RE state changes so pausing always aborts detector exposure.
    try:
        _previous_state_hook = RE.state_hook

        def _state_hook_with_detector_abort(*args, **kwargs):
            state = None
            str_args = [a for a in args if isinstance(a, str)]
            if str_args:
                state = str_args[-1]
            elif "state" in kwargs and isinstance(kwargs["state"], str):
                state = kwargs["state"]

            if state in ("pausing", "paused"):
                _abort_detector_acquire(cam1)

            if callable(_previous_state_hook):
                return _previous_state_hook(*args, **kwargs)
            return None

        RE.state_hook = _state_hook_with_detector_abort
    except Exception:
        pass

# Enable when using the QHY camera
if 0:
    cam1 = MyQHYDetector(prefix='4dh4:',name='cam1',read_attrs=['tiff1','stats1.total'])
    cam1.cam.nd_attributes_file.set("/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoDetectorAttributes.xml") 
    # caput("4dh4:TIFF1:CreateDirectory", -3)

sd.baseline.append(cam1.focus)
sd.baseline.append(cam1.x)

# cam_zwo.stage_sigs["cam.num_images"] = 1

# Need to add stage sigs for create directory depth
# cam_zwo.tiff1.stage_sigs[""]

# cam_zwo.cam.temperature.set(-20).wait()
# cam_zwo.hdf1.stage_sigs["layout_filename"] = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoLayoutDX.xml"
# cam_zwo.cam.stage_sigs["nd_attributes_file"] = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoDetectorAttributes.xml"
# cam_zwo.hdf1.stage_sigs["store_attr"] = "Yes"

# cam_sim = SimAreaDetector(prefix='4dh4:',name='cam1',read_attrs=['hdf1','stats1.total'])
# cam_sim.hdf1.stage_sigs["layout_filename"] = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoLayoutDX.xml"
# cam_sim.cam.stage_sigs["nd_attributes_file"] = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoDetectorAttributes.xml"
