"""Device definitions only; importing this module does not construct EPICS devices."""
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

from server.devices.motors import EpicsMotorCustom

class ZWODetectorCam(CamBase):
    # This ZWO IOC exposes ``Acquire`` as a single control/status PV; it does
    # not provide the AreaDetector-style ``Acquire_RBV`` record assumed by
    # CamBase.  SingleTrigger accesses this signal during detector
    # construction, so the inherited SignalWithRBV prevents worker startup.
    acquire = ADCpt(EpicsSignal, "Acquire")
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
            self._acq_status.add_callback(
                lambda status: setattr(self, "_acq_in_progress", False)
            )
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
        write_path_template="/home/mitr_4dh4/Data/Imaging/%Y/",
        read_path_template="/home/mitr_4dh4/Data/Imaging/%Y/",
    )

class MyQHYDetector(SingleTriggerPause, QHYDetector):
    cam = Cpt(ZWODetectorCam, "cam1:")
    image = Cpt(ImagePlugin, suffix='image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')

    tiff1 = Cpt(
        MyTIFFPlugin,
        "TIFF1:",
        # write_path_template="/home/mitr_4dh4/Data/%Y/PSI_Experiment/",
        # read_path_template="/home/mitr_4dh4/Data/%Y/PSI_Experiment/",
        write_path_template="/home/mitr_4dh4/Data/Imaging/%Y/",
        read_path_template="/home/mitr_4dh4/Data/Imaging/%Y/",
    )

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

__all__ = ['ADCpt', 'AreaDetector', 'CamBase', 'Cpt', 'DetectorBase', 'Device', 'DeviceStatus', 'EpicsMotor', 'EpicsMotorCustom', 'EpicsSignal', 'EpicsSignalRO', 'FileStoreHDF5IterativeWrite', 'FileStoreTIFFIterativeWrite', 'HDF5Plugin', 'ImagePlugin', 'MyHDF5Plugin', 'MyQHYDetector', 'MyTIFFPlugin', 'MyZWODetector', 'QHYDetector', 'QHYDetectorCam', 'Signal', 'SignalWithRBV', 'SimAreaDetector', 'SimDetector', 'SingleTrigger', 'SingleTriggerPause', 'StatsPlugin', 'Status', 'SubscriptionStatus', 'TIFFPlugin', 'TransformPlugin', 'ZWODetector', 'ZWODetectorCam', 'caget', 'cainfo', 'cam', 'caput', 'datetime', 'do_not_wait_for_lazy_connection', 'np', 'register_device', 'timedelta', 'uuid']
