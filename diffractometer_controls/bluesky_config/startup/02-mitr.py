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
reactor_power_dwk1 = EpicsSignalRO("mitr:DWK1",name="reactor_power_dwk1")
reactor_power_dwk2 = EpicsSignalRO("mitr:DWK2",name="reactor_power_dwk2")
reactor_power_dwk3 = EpicsSignalRO("mitr:DWK3",name="reactor_power_dwk3")
reactor_power_dwk4 = EpicsSignalRO("mitr:DWK4",name="reactor_power_dwk4")

sd.monitors.append(reactor_power_6)
sd.baseline.append(reactor_power_6)
sd.baseline.append(reactor_power_4)
sd.baseline.append(reactor_power_thm)

# Install suspenders depending on the reactor power. Disable to test while the reactor is shutdown
if 1:
    reactor_power_suspender = SuspendFloor(reactor_power_6, 5, resume_thresh=5.2)
    RE.install_suspender(reactor_power_suspender)