# Local EPICS support corrections

## measComp USB-2408 optional digital expansion ports

`meascomp-usb2408-optional-expansion.patch` is applied to the control host's
measComp R4-4 `measCompApp/src/drvMultiFunction.cpp`.

ULDAQ advertises three digital ports for the USB-2408 family. Without the
expansion, reads of the additional ports return `ERR_BAD_PORT_TYPE`. The stock
driver exits the polling cycle before reading counters, analog inputs, or
thermocouples. Its temperature records remain undefined.

This correction skips that specific error on additional ports of the USB-2408
family under Linux. Errors on the main port and other devices keep their
existing handling. PV names and saved channel settings stay the same.

When replacing or upgrading measComp, check whether upstream has addressed this
case. If it remains necessary, apply this patch from the measComp source root
with `patch -p1 < /path/to/meascomp-usb2408-optional-expansion.patch`, rebuild the
support library, and restart the DAQ IOC during a suitable interruption. Use
atomic library replacement when installing while other IOCs are running.

Verified on the control host: channel 1 returned fresh readings around 22 °C
with `SEVR=0`, `STAT=0`, and `UDF=0`; the USB-CTR08 reconnected after restarting
`4dh4-daq.service`.

Background: [ULDAQ port-count discussion](https://github.com/mccdaq/uldaq/issues/35).
