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
    pass

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