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

class QHYDetectorCam(CamBase):
    offset = ADCpt(SignalWithRBV, "Offset")
    readmode = ADCpt(SignalWithRBV, "ReadMode")

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
    devices. If acquisition is in-flight, forcing cam.acquire=0 can prevent
    finishing/writing the interrupted frame. We also restore TIFF file_number
    to the value captured before trigger so resumed acquisition can reuse the
    same slot.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tiff_file_number_before_trigger = None

    def trigger(self):
        if hasattr(self, "tiff1") and hasattr(self.tiff1, "file_number"):
            try:
                self._tiff_file_number_before_trigger = int(self.tiff1.file_number.get())
            except Exception:
                self._tiff_file_number_before_trigger = None
        return super().trigger()

    def stop(self, *, success=False):
        was_acquiring = False

        # Abort in-flight camera exposure ASAP.
        if hasattr(self, "cam") and hasattr(self.cam, "acquire"):
            try:
                was_acquiring = bool(self.cam.acquire.get())
            except Exception:
                was_acquiring = False
            try:
                if was_acquiring:
                    self.cam.acquire.put(0, wait=False)
            except Exception:
                pass

        # If stop interrupted an exposure, roll back file_number so resume
        # can overwrite/reuse the interrupted slot.
        if was_acquiring and self._tiff_file_number_before_trigger is not None:
            if hasattr(self, "tiff1") and hasattr(self.tiff1, "file_number"):
                try:
                    self.tiff1.file_number.put(self._tiff_file_number_before_trigger, wait=False)
                except Exception:
                    pass

        return super().stop(success=success)


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
