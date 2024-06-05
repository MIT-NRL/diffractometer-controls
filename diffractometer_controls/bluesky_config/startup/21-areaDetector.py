import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsMotor, Signal,
                   cam)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus

from ophyd.areadetector import AreaDetector, SingleTrigger, SimDetector, ImagePlugin, StatsPlugin, TIFFPlugin, HDF5Plugin
from ophyd.areadetector.filestore_mixins import FileStoreTIFFIterativeWrite, FileStoreHDF5IterativeWrite
from ophyd import cam


class MyTIFFPlugin(FileStoreTIFFIterativeWrite,TIFFPlugin):
    pass

class MyHDF5Plugin(FileStoreHDF5IterativeWrite,HDF5Plugin):
    layout_filename = Cpt(EpicsSignal, "XMLFileName", kind="config", string=True)
    layout_filename_valid = Cpt(EpicsSignal, "XMLValid_RBV", kind="omitted", string=True)
    nd_attr_status = Cpt(EpicsSignal, "NDAttributesStatus", kind="omitted", string=True)

class ZWODetector(SingleTrigger, AreaDetector):
    cam = Cpt(cam.AreaDetectorCam, "cam1:")
    image = Cpt(ImagePlugin, suffix='image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')

    # tiff1 = Cpt(
    #     MyTIFFPlugin,
    #     "TIFF1:",
    #     write_path_template="/home/mitr_4dh4/Data/TestData/%Y/%m/%d/",
    #     read_path_template="/home/mitr_4dh4/Data/TestData/%Y/%m/%d/",
    # )
    hdf1 = Cpt(
        MyHDF5Plugin,
        "HDF1:",
        write_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",
        read_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",        
    )

class SimAreaDetector(SingleTrigger, SimDetector):
    cam = Cpt(cam.SimDetectorCam, "cam2:")
    image = Cpt(ImagePlugin, suffix='image1:')
    stats1 = Cpt(StatsPlugin, 'Stats1:')

    # tiff1 = Cpt(
    #     MyTIFFPlugin,
    #     "TIFF1:",
    #     write_path_template="/home/mitr_4dh4/Data/TestData/%Y/%m/%d/",
    #     read_path_template="/home/mitr_4dh4/Data/TestData/%Y/%m/%d/",
    # )
    hdf1 = Cpt(
        MyHDF5Plugin,
        "HDF1:",
        write_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",
        read_path_template="/home/mitr_4dh4/Data/TestData/HDF/%Y/%m/%d/",        
    )


# cam1 = ZWODetector(prefix='4dh4:',name='cam1',read_attrs=['tiff1','stats1.total'])
cam_sim = SimAreaDetector(prefix='4dh4:',name='cam2',read_attrs=['hdf1','stats1.total'])
cam_sim.hdf1.stage_sigs["layout_filename"] = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoLayout.xml"
cam_sim.cam.stage_sigs["nd_attributes_file"] = "/home/mitr_4dh4/Documents/GitHub/diffractometer-controls/diffractometer_controls/areaDetectorConfigXML/tomoDetectorAttributes.xml"