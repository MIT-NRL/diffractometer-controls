import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus


# Tomography/Imaging motors
sample_tomo_th = EpicsMotor("4dh4:m3",name="sample_tomo_th",labels=["positioner"])
sample_tomo_x = EpicsMotor("4dh4:m14",name="sample_tomo_x",labels=["positioner"])

cam_focus = EpicsMotor("4dh4:m12",name="cam_focus",labels=["positioner"])
cam_x = EpicsMotor("4dh4:m1",name="cam_x",labels=["positioner"])

# Diffraction motors
sample_th = EpicsMotor("4dh4:m10",name="sample_th",labels=["positioner"])

det_psd_x = EpicsMotor("4dh4:m9",name="det_psd_x",labels=["positioner"])


analyzer1_curve = EpicsMotor("4dh4:m11",name="analyzer1_curve")
analyzer2_curve = EpicsMotor("4dh4:m15",name="analyzer2_curve")
analyzer1_angle = EpicsMotor("4dh4:m13",name="analyzer1_angle")