from server.devices.detectors import PositionSignal, HE3PSD, USB2408TemperatureInputs, USB2408AnalogInputs, USB2408BinaryInputs, USB2408Readouts, USBCTR08Scaler
from server.devices.detectors import HE3PSD_POSITION_MIN, HE3PSD_POSITION_MAX
import numpy as np
from ophyd import (Device, Component as Cpt,FormattedComponent as FCpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV,
                   EpicsMotor, DerivedSignal)
from ophyd.device import DeviceStatus
from ophyd.scaler import EpicsScaler
from ophyd.status import Status, SubscriptionStatus
from bluesky_queueserver import register_device

he3psd0 = HE3PSD("4dh4:he3PSD:",det_num="Det0", name="he3psd0")
he3psd7 = HE3PSD("4dh4:he3PSD:",det_num="Det7", name="he3psd7")

usb2408 = USB2408Readouts("4dh4:USB2408:", name="usb2408")
register_device("usb2408", depth=3)

# Leave the class importable for plan/display development, but do not create
# EPICS connections until the CTR-08 has arrived and its IOC is enabled.
ENABLE_USBCTR08 = True

if ENABLE_USBCTR08:
    usbctr = USBCTR08Scaler("4dh4:USBCTR:scaler1", name="usbctr")
    register_device("usbctr", depth=2)
