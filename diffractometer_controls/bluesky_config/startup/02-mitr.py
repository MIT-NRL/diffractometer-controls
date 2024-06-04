import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from bluesky import SupplementalData

sd: SupplementalData

reactor_power_6 = EpicsSignalRO("mitr:power_6",name="reactor_power_6")
reactor_power_4 = EpicsSignalRO("mitr:power_4",name="reactor_power_4")
reactor_power_thm = EpicsSignalRO("mitr:power_thm",name="reactor_power_thm")


sd.baseline.append(reactor_power_6)
sd.baseline.append(reactor_power_4)
sd.baseline.append(reactor_power_thm)