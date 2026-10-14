"""Scalar and MCS records for the real USB-CTR displays and Ophyd device."""
import asyncio
import time
import numpy as np
from caproto import ChannelType
from caproto.server import PVGroup, pvproperty


class USBCTR(PVGroup):
    count = pvproperty(value=0, name="scaler1.CNT")
    mode = pvproperty(value=0, name="scaler1.CONT", dtype=ChannelType.ENUM, enum_strings=("OneShot", "AutoCount"))
    preset = pvproperty(value=0.2, name="scaler1.TP")
    elapsed = pvproperty(value=0.0, name="scaler1.T", read_only=True)
    frequency = pvproperty(value=1000.0, name="scaler1.FREQ")
    display_rate = pvproperty(value=10.0, name="scaler1.RATE")
    pulse_frequency = pvproperty(value=1000.0, name="PulseGen1Frequency")
    pulse_frequency_rbv = pvproperty(value=1000.0, name="PulseGen1Frequency_RBV", read_only=True)
    duty = pvproperty(value=0.5, name="PulseGen1DutyCycle")
    pulses = pvproperty(value=0, name="PulseGen1Count")
    pulse_run = pvproperty(value=0, name="PulseGen1Run")
    mcs_start = pvproperty(value=0, name="MCS:EraseStart")
    mcs_stop = pvproperty(value=0, name="MCS:StopAll")
    mcs_busy = pvproperty(value=0, name="MCS:Acquiring", read_only=True)
    mcs_hardware_busy = pvproperty(value=0, name="MCS:HardwareAcquiring", read_only=True)
    mcs_connected = pvproperty(value=1, name="MCS:SNL_Connected", read_only=True)
    dwell = pvproperty(value=0.1, name="MCS:Dwell")
    dwell_rbv = pvproperty(value=0.1, name="MCS:Dwell_RBV", read_only=True)
    points = pvproperty(value=32, name="MCS:NuseAll")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._count_task = self._mcs_task = None

    @pulse_frequency.putter
    async def pulse_frequency(self, instance, value):
        await self.pulse_frequency_rbv.write(float(value))
        return float(value)

    @dwell.putter
    async def dwell(self, instance, value):
        value = max(0.01, float(value))
        await self.dwell_rbv.write(value)
        return value

    @count.putter
    async def count(self, instance, value):
        if int(value):
            if int(self.mcs_hardware_busy.value):
                return 0
            if self._count_task is None or self._count_task.done():
                await instance.write(1, verify_value=False)
                self._count_task = asyncio.create_task(self._count_once())
            # The real scaler completes CA put-notify when counting finishes.
            # USBCTR08Scaler.trigger relies on that callback, so an immediate
            # return here would let plans read partially accumulated counts.
            try:
                await self._count_task
            except asyncio.CancelledError:
                pass
            return 0
        elif self._count_task is not None:
            self._count_task.cancel()
            self._count_task = None
        return int(bool(value))

    async def _count_once(self):
        started = time.monotonic()
        duration = max(0.0, float(self.preset.value))
        try:
            while True:
                elapsed = min(duration, time.monotonic() - started)
                await self.elapsed.write(elapsed)
                for index in range(1, 9):
                    rate = float(self.frequency.value) if index == 1 else (2400.0 if index == 2 else 850.0 / max(1, index - 2))
                    await getattr(self, f"counts_{index}").write(int(rate * elapsed))
                if elapsed >= duration:
                    # Publish count monitors before the completion monitor;
                    # cached EPICS reads in the real scaler consume them next.
                    await asyncio.sleep(0.05)
                    break
                await asyncio.sleep(min(0.03, duration - elapsed))
        finally:
            self._count_task = None
            await self.count.write(0, verify_value=False)

    @mcs_start.putter
    async def mcs_start(self, instance, value):
        if int(value) and not int(self.count.value) and (self._mcs_task is None or self._mcs_task.done()):
            self._mcs_task = asyncio.create_task(self._mcs_once())
        return 0

    @mcs_stop.putter
    async def mcs_stop(self, instance, value):
        if self._mcs_task is not None:
            task = self._mcs_task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            if self._mcs_task is task:
                self._mcs_task = None
        return 0

    async def _mcs_once(self):
        await self.mcs_busy.write(1)
        await self.mcs_hardware_busy.write(1)
        samples = [[] for _ in range(8)]
        try:
            for _ in range(min(2048, max(1, int(self.points.value)))):
                dwell = float(self.dwell.value)
                await asyncio.sleep(dwell)
                for channel in range(8):
                    samples[channel].append(int((2400 if channel == 1 else 850) * dwell))
                    await getattr(self, f"waveform_{channel+1}").write(samples[channel])
        finally:
            self._mcs_task = None
            await self.mcs_busy.write(0)
            await self.mcs_hardware_busy.write(0)


_fields = {}
for field in ("DLY", "DLY1", "TP1", "RAT1", "EGU"):
    _fields[field.lower()] = pvproperty(value="counts" if field == "EGU" else 0.0, name=f"scaler1.{field}")
for index in range(1, 33):
    _fields[f"counts_{index}"] = pvproperty(value=0, name=f"scaler1.S{index}", read_only=True)
    for field in ("NM", "PR", "G"):
        _fields[f"{field.lower()}_{index}"] = pvproperty(value=f"CTR{index-1}" if field == "NM" else 0, name=f"scaler1.{field}{index}")
for index in range(1, 9):
    _fields[f"waveform_{index}"] = pvproperty(value=np.zeros(1, dtype=np.int32), dtype=ChannelType.LONG, max_length=2048, name=f"MCS:mca{index}", read_only=True)
    _fields[f"enable_{index}"] = pvproperty(value=1, name=f"MCS:MCSCounter{index}Enable")
for field in ("ChannelAdvance", "TrigMode", "Point0Action"):
    _fields[field.lower()] = pvproperty(value=0, name=f"MCS:{field}")
USBCTRRecords = type("USBCTRRecords", (USBCTR,), _fields)

USB2408 = type("USB2408", (PVGroup,), {
    f"{field.lower()}{index}": pvproperty(value=20.0+index if field == "Ti" else index, name=f"{field}{index}", read_only=True)
    for field in ("Ti", "Ai", "Bi") for index in range(1, 9)
})
