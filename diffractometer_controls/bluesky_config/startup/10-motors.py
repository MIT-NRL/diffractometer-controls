import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus

sample_th = EpicsMotor("4dh4:m10",name="sample_th",labels=["positioner"])

det_psd_x = EpicsMotor("4dh4:m9",name="det_psd_x",labels=["positioner"])

analyzer_curve = EpicsMotor("4dh4:m11",name="analyzer_curve")