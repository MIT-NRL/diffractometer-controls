import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from bluesky import SupplementalData, RunEngine
from bluesky.suspenders import SuspendFloor, SuspendBoolLow

sd: SupplementalData
RE: RunEngine

reactor_power_6 = EpicsSignalRO("mitr:Power6",name="reactor_power_6")
reactor_power_4 = EpicsSignalRO("mitr:Power4",name="reactor_power_4")
reactor_power_thm = EpicsSignalRO("mitr:PowerThm",name="reactor_power_thm")

sd.monitors.append(reactor_power_6)
sd.baseline.append(reactor_power_6)
sd.baseline.append(reactor_power_4)
sd.baseline.append(reactor_power_thm)

# Install suspenders depending on the reactor power
sus = SuspendFloor(reactor_power_6, 4, resume_thresh=5)
RE.install_suspender(sus)