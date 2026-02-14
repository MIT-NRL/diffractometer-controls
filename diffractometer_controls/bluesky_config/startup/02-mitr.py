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
    reactor_power_suspender_enable = EpicsSignal(
        "4dh4:Bluesky:SuspenderEnable", name="reactor_power_suspender_enable"
    )
    reactor_power_suspender_installed = EpicsSignal(
        "4dh4:Bluesky:SuspenderInstalled", name="reactor_power_suspender_installed"
    )

    def _is_reactor_power_suspender_installed():
        try:
            return reactor_power_suspender in RE.suspenders
        except Exception:
            return False

    def _publish_reactor_power_suspender_state():
        try:
            reactor_power_suspender_installed.put(
                int(_is_reactor_power_suspender_installed()), wait=False
            )
        except Exception:
            pass

    def _set_reactor_power_suspender_enabled(enable):
        enable = bool(enable)
        installed = _is_reactor_power_suspender_installed()
        if enable and not installed:
            RE.install_suspender(reactor_power_suspender)
        elif (not enable) and installed:
            RE.remove_suspender(reactor_power_suspender)
        _publish_reactor_power_suspender_state()

    def _queue_set_reactor_power_suspender(enable):
        try:
            RE.loop.call_soon_threadsafe(_set_reactor_power_suspender_enabled, bool(enable))
        except Exception:
            _set_reactor_power_suspender_enabled(bool(enable))

    def _on_reactor_power_suspender_enable_changed(value=None, **kwargs):
        try:
            enable = bool(int(float(value)))
        except Exception:
            return
        _queue_set_reactor_power_suspender(enable)

    reactor_power_suspender_enable.subscribe(
        _on_reactor_power_suspender_enable_changed, run=False
    )

    try:
        _initial_enable = bool(int(float(reactor_power_suspender_enable.get())))
    except Exception:
        _initial_enable = True

    _set_reactor_power_suspender_enabled(_initial_enable)
    try:
        reactor_power_suspender_enable.put(int(_initial_enable), wait=False)
    except Exception:
        pass
