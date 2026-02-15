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

# Install suspenders depending on reactor power and persist behavior from PV state.
reactor_power_suspender = SuspendFloor(reactor_power_6, 5, resume_thresh=5.2)
reactor_power_suspender_enable = EpicsSignal(
    "4dh4:Bluesky:SuspenderEnable", name="reactor_power_suspender_enable"
)
reactor_power_suspender_installed = EpicsSignal(
    "4dh4:Bluesky:SuspenderInstalled", name="reactor_power_suspender_installed"
)
_suspender_enable_feedback_write = False


def _is_reactor_power_suspender_installed():
    try:
        return len(_get_reactor_power_suspenders()) > 0
    except Exception:
        return False


def _get_reactor_power_suspenders():
    matched = []
    try:
        for susp in RE.suspenders:
            if not isinstance(susp, SuspendFloor):
                continue
            sig = getattr(susp, "_sig", None)
            sig_name = getattr(sig, "name", "")
            sig_pv = getattr(sig, "pvname", "")
            if sig is reactor_power_6 or sig_name == "reactor_power_6" or sig_pv == "mitr:Power6":
                matched.append(susp)
    except Exception:
        return []
    return matched


def _publish_reactor_power_suspender_state():
    global _suspender_enable_feedback_write
    installed = int(_is_reactor_power_suspender_installed())
    try:
        reactor_power_suspender_installed.put(installed, wait=False)
    except Exception:
        pass
    # Keep command PV synced to actual installed state so GUIs reflect truth.
    try:
        _suspender_enable_feedback_write = True
        reactor_power_suspender_enable.put(installed, wait=False)
    except Exception:
        pass
    finally:
        _suspender_enable_feedback_write = False


def _set_reactor_power_suspender_enabled(enable):
    enable = bool(enable)
    installed_suspenders = _get_reactor_power_suspenders()
    installed = len(installed_suspenders) > 0
    if enable and not installed:
        RE.install_suspender(reactor_power_suspender)
    elif (not enable) and installed:
        for susp in installed_suspenders:
            RE.remove_suspender(susp)
    _publish_reactor_power_suspender_state()


def _queue_set_reactor_power_suspender(enable):
    enable = bool(enable)
    loop = getattr(RE, "loop", None)
    try:
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(_set_reactor_power_suspender_enabled, enable)
        else:
            _set_reactor_power_suspender_enabled(enable)
    except Exception:
        _set_reactor_power_suspender_enabled(enable)


def _coerce_enable_value(value):
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on", "enable", "enabled"):
            return True
        if text in ("0", "false", "no", "off", "disable", "disabled"):
            return False
        return None
    try:
        return bool(int(float(value)))
    except Exception:
        return None


def _on_reactor_power_suspender_enable_changed(value=None, **kwargs):
    if _suspender_enable_feedback_write:
        return
    enable = _coerce_enable_value(value)
    if enable is None:
        return
    _queue_set_reactor_power_suspender(enable)


reactor_power_suspender_enable.subscribe(
    _on_reactor_power_suspender_enable_changed, run=True
)

try:
    _initial_enable = _coerce_enable_value(reactor_power_suspender_enable.get())
except Exception:
    _initial_enable = None
if _initial_enable is None:
    _initial_enable = True

_set_reactor_power_suspender_enabled(_initial_enable)
try:
    reactor_power_suspender_enable.put(int(_initial_enable), wait=False)
except Exception:
    pass
