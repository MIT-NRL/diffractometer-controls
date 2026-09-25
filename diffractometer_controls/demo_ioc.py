"""Loopback-only caproto IOC for the disconnected demonstration mode."""

from __future__ import annotations

import asyncio
import os

import numpy as np
from caproto import ChannelType
from caproto.ioc_examples.fake_motor_record import motor_record_simulator
from caproto.server import PVGroup, SubGroup, ioc_arg_parser, pvproperty, run

try:
    from .demo_simulation import (
        IMAGE_SHAPE,
        diffraction_spectrum,
        gaussian_image,
        slanted_edge_image,
    )
except ImportError:  # Direct execution by the Queue Server environment.
    from demo_simulation import (
        IMAGE_SHAPE,
        diffraction_spectrum,
        gaussian_image,
        slanted_edge_image,
    )


MOTOR_DESCRIPTIONS = {
    "m1": "Camera horizontal",
    "m3": "Tomography rotation",
    "m9": "Sample vertical",
    "m10": "Sample theta",
    "m11": "Analyzer 1 curvature",
    "m12": "Camera focus",
    "m13": "Stage 1 horizontal",
    "m14": "Stage 2 horizontal",
    "m15": "Analyzer 2 curvature",
    "m16": "Detector horizontal",
}


class DemoMotor(PVGroup):
    """A modest-speed motor record suitable for existing PyDM panels."""

    motor = pvproperty(value=0.0, name="", record="motor", precision=3)

    def __init__(self, *args, description="Demo motor", units="mm", **kwargs):
        super().__init__(*args, **kwargs)
        self.tick_rate_hz = 25.0
        self.defaults = {
            "velocity": 8.0,
            "precision": 3,
            "acceleration": 0.15,
            "resolution": 1e-6,
            "user_limits": (-100.0, 100.0),
        }
        self._description = description
        self._units = units

    @motor.startup
    async def motor(self, instance, async_lib):
        await instance.fields["DESC"].write(self._description)
        await instance.fields["EGU"].write(self._units)
        await instance.fields["TWV"].write(0.1)
        await instance.fields["CNEN"].write(1)
        await motor_record_simulator(
            instance,
            async_lib,
            self.defaults,
            tick_rate_hz=self.tick_rate_hz,
        )

    @motor.fields.tweak_motor_forward.putter
    async def _tweak_forward(fields, instance, value):
        if int(value):
            await fields.parent.write(
                float(fields.parent.value) + float(fields.tweak_step_size.value)
            )
        return 0

    @motor.fields.tweak_motor_reverse.putter
    async def _tweak_reverse(fields, instance, value):
        if int(value):
            await fields.parent.write(
                float(fields.parent.value) - float(fields.tweak_step_size.value)
            )
        return 0


class HE3Detector(PVGroup):
    acquire = pvproperty(value=0, name="Acquire")
    acquire_rbv = pvproperty(value=0, name="Acquire_RBV", read_only=True)
    acquire_time = pvproperty(value=0.2, name="AcquireTime")
    acquire_time_rbv = pvproperty(value=0.2, name="AcquireTime_RBV", read_only=True)
    acquire_time_remaining = pvproperty(value=0.0, name="AcquireTimeRemaining_RBV", read_only=True)
    nbins = pvproperty(value=350, name="NBins")
    nbins_rbv = pvproperty(value=350, name="NBins_RBV", read_only=True)
    soft_lld = pvproperty(value=0.0, name="SoftLLD")
    soft_lld_rbv = pvproperty(value=0.0, name="SoftLLD_RBV", read_only=True)
    position_x = pvproperty(
        value=np.linspace(-209.21799, 209.21799, 350).tolist(),
        name="PositionX",
        max_length=2048,
        dtype=ChannelType.DOUBLE,
        read_only=True,
    )
    counts = pvproperty(
        value=np.zeros(350, dtype=np.int32),
        name="Counts",
        max_length=2048,
        dtype=ChannelType.LONG,
        read_only=True,
    )
    total_counts = pvproperty(value=0.0, name="TotalCounts", read_only=True)
    det0_counts = pvproperty(
        value=np.zeros(350, dtype=np.int32), name="Det0:LiveCounts",
        max_length=2048, dtype=ChannelType.LONG, read_only=True,
    )
    det0_total = pvproperty(value=0.0, name="Det0:LiveTotalCounts", read_only=True)
    det7_counts = pvproperty(
        value=np.zeros(350, dtype=np.int32), name="Det7:LiveCounts",
        max_length=2048, dtype=ChannelType.LONG, read_only=True,
    )
    det7_total = pvproperty(value=0.0, name="Det7:LiveTotalCounts", read_only=True)

    def __init__(self, *args, detector_offset=0.0, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.detector_offset = float(detector_offset)
        self.seed = int(seed)
        self.frame = 0
        self._task = None

    @acquire_time.putter
    async def acquire_time(self, instance, value):
        value = max(0.001, float(value))
        await self.acquire_time_rbv.write(value)
        return value

    @nbins.putter
    async def nbins(self, instance, value):
        value = min(2048, max(1, int(value)))
        await self.nbins_rbv.write(value)
        return value

    @soft_lld.putter
    async def soft_lld(self, instance, value):
        await self.soft_lld_rbv.write(float(value))
        return value

    @acquire.putter
    async def acquire(self, instance, value):
        if int(value) and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._acquire())
            return 1
        return int(bool(value))

    async def _acquire(self):
        await self.acquire_rbv.write(1)
        exposure = float(self.acquire_time.value)
        await self.acquire_time_remaining.write(exposure)
        await asyncio.sleep(exposure)
        root = self.parent
        motor_position = float(root.m10.motor.value)
        axis, counts = diffraction_spectrum(
            motor_position,
            nbins=int(self.nbins.value),
            exposure=exposure,
            detector_offset=self.detector_offset,
            seed=self.seed + self.frame,
        )
        self.frame += 1
        threshold = float(self.soft_lld.value)
        if threshold > 0:
            counts = np.where(counts >= threshold, counts, 0).astype(np.int32)
        await self.position_x.write(axis)
        await self.counts.write(counts)
        await self.total_counts.write(float(np.sum(counts)))
        _, counts0 = diffraction_spectrum(
            motor_position, nbins=int(self.nbins.value), exposure=exposure,
            detector_offset=-0.08, seed=41 + self.frame,
        )
        _, counts7 = diffraction_spectrum(
            motor_position, nbins=int(self.nbins.value), exposure=exposure,
            detector_offset=0.10, seed=73 + self.frame,
        )
        if threshold > 0:
            counts0 = np.where(counts0 >= threshold, counts0, 0).astype(np.int32)
            counts7 = np.where(counts7 >= threshold, counts7, 0).astype(np.int32)
        await self.det0_counts.write(counts0)
        await self.det0_total.write(float(np.sum(counts0)))
        await self.det7_counts.write(counts7)
        await self.det7_total.write(float(np.sum(counts7)))
        await self.acquire_time_remaining.write(0.0)
        # Reset the setpoint and release the task before publishing completion.
        # The worker advances as soon as Acquire_RBV falls; doing this in the
        # opposite order allowed the next trigger to arrive while _task still
        # referred to the finishing acquisition, so alternate scan points
        # reused stale detector data.
        await self.acquire.write(0, verify_value=False)
        self._task = None
        await self.acquire_rbv.write(0)


class DemoCamera(PVGroup):
    acquire = pvproperty(value=0, name="Acquire")
    acquire_rbv = pvproperty(value=0, name="Acquire_RBV", read_only=True)
    acquire_time = pvproperty(value=0.1, name="AcquireTime")
    acquire_time_rbv = pvproperty(value=0.1, name="AcquireTime_RBV", read_only=True)
    time_remaining = pvproperty(value=0.0, name="TimeRemaining_RBV", read_only=True)
    detector_state = pvproperty(value=0, name="DetectorState_RBV", read_only=True)
    array_rate = pvproperty(value=0.0, name="ArrayRate_RBV", read_only=True)
    array_counter = pvproperty(value=0, name="ArrayCounter")
    array_counter_rbv = pvproperty(value=0, name="ArrayCounter_RBV", read_only=True)
    status_message = pvproperty(value="Idle", name="StatusMessage_RBV", read_only=True)
    size_x = pvproperty(value=IMAGE_SHAPE[1], name="SizeX_RBV", read_only=True)
    size_y = pvproperty(value=IMAGE_SHAPE[0], name="SizeY_RBV", read_only=True)
    array_size_x = pvproperty(value=IMAGE_SHAPE[1], name="ArraySizeX_RBV", read_only=True)
    array_size_y = pvproperty(value=IMAGE_SHAPE[0], name="ArraySizeY_RBV", read_only=True)
    gain = pvproperty(value=1.0, name="Gain")
    gain_rbv = pvproperty(value=1.0, name="Gain_RBV", read_only=True)
    offset = pvproperty(value=0.0, name="Offset")
    offset_rbv = pvproperty(value=0.0, name="Offset_RBV", read_only=True)
    temperature = pvproperty(value=-10.0, name="Temperature")
    temperature_actual = pvproperty(value=-10.0, name="TemperatureActual", read_only=True)
    total = pvproperty(value=0.0, name="Total", read_only=True)

    def __init__(self, *args, image_kind="gaussian", seed=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_kind = image_kind
        self.seed = int(seed)
        self.frame = 0
        self._task = None

    @acquire_time.putter
    async def acquire_time(self, instance, value):
        value = max(0.001, float(value))
        await self.acquire_time_rbv.write(value)
        return value

    @gain.putter
    async def gain(self, instance, value):
        await self.gain_rbv.write(float(value))
        return value

    @offset.putter
    async def offset(self, instance, value):
        await self.offset_rbv.write(float(value))
        return value

    @temperature.putter
    async def temperature(self, instance, value):
        await self.temperature_actual.write(float(value))
        return value

    @array_counter.putter
    async def array_counter(self, instance, value):
        await self.array_counter_rbv.write(int(value))
        return int(value)

    @acquire.putter
    async def acquire(self, instance, value):
        if int(value) and (self._task is None or self._task.done()):
            self._task = asyncio.create_task(self._acquire())
            return 1
        return int(bool(value))

    async def _acquire(self):
        root = self.parent
        exposure = float(self.acquire_time.value)
        await self.acquire_rbv.write(1)
        await self.detector_state.write(1)
        await self.status_message.write("Acquiring")
        await self.time_remaining.write(exposure)
        await asyncio.sleep(exposure)
        if self.image_kind == "gaussian":
            image = gaussian_image(
                float(root.m1.motor.value),
                float(root.m14.motor.value),
                exposure=exposure,
                seed=self.seed + self.frame,
            )
        else:
            image = slanted_edge_image(
                float(root.m12.motor.value),
                exposure=exposure,
                seed=self.seed + self.frame,
            )
        self.frame += 1
        counter = int(self.array_counter.value) + 1
        await self.array_counter.write(counter, verify_value=False)
        await self.total.write(float(np.sum(image)))
        await root.image_array.write(image.ravel())
        await root.image_size0.write(image.shape[1])
        await root.image_size1.write(image.shape[0])
        await self.array_rate.write(1.0 / exposure)
        await self.time_remaining.write(0.0)
        await self.detector_state.write(0)
        await self.status_message.write("Idle")
        await self.acquire.write(0, verify_value=False)
        self._task = None
        await self.acquire_rbv.write(0)


class DemoIOC(PVGroup):
    m1 = SubGroup(DemoMotor, prefix="m1", description=MOTOR_DESCRIPTIONS["m1"])
    m3 = SubGroup(DemoMotor, prefix="m3", description=MOTOR_DESCRIPTIONS["m3"], units="deg")
    m9 = SubGroup(DemoMotor, prefix="m9", description=MOTOR_DESCRIPTIONS["m9"])
    m10 = SubGroup(DemoMotor, prefix="m10", description=MOTOR_DESCRIPTIONS["m10"], units="deg")
    m11 = SubGroup(DemoMotor, prefix="m11", description=MOTOR_DESCRIPTIONS["m11"], units="counts")
    m12 = SubGroup(DemoMotor, prefix="m12", description=MOTOR_DESCRIPTIONS["m12"])
    m13 = SubGroup(DemoMotor, prefix="m13", description=MOTOR_DESCRIPTIONS["m13"], units="deg")
    m14 = SubGroup(DemoMotor, prefix="m14", description=MOTOR_DESCRIPTIONS["m14"])
    m15 = SubGroup(DemoMotor, prefix="m15", description=MOTOR_DESCRIPTIONS["m15"], units="counts")
    m16 = SubGroup(DemoMotor, prefix="m16", description=MOTOR_DESCRIPTIONS["m16"])

    allstop = pvproperty(value=0, name="allstop", record="bo")
    heartbeat = pvproperty(value=0, name="HEARTBEAT", read_only=True)
    frame_type = pvproperty(value=0, name="TS:FrameType")

    he3psd0 = SubGroup(HE3Detector, prefix="he3PSD:", detector_offset=-0.08, seed=41)
    cam1 = SubGroup(DemoCamera, prefix="cam1:", image_kind="gaussian", seed=101)
    sim_focus_cam = SubGroup(DemoCamera, prefix="simFocus:", image_kind="edge", seed=211)

    image_array = pvproperty(
        value=np.zeros(IMAGE_SHAPE[0] * IMAGE_SHAPE[1], dtype=np.int32),
        name="image1:ArrayData",
        max_length=IMAGE_SHAPE[0] * IMAGE_SHAPE[1],
        dtype=ChannelType.LONG,
        read_only=True,
    )
    image_size0 = pvproperty(value=IMAGE_SHAPE[1], name="image1:ArraySize0_RBV", read_only=True)
    image_size1 = pvproperty(value=IMAGE_SHAPE[0], name="image1:ArraySize1_RBV", read_only=True)

    run_state = pvproperty(value="IDLE", name="Bluesky:Run:State")
    run_suspended = pvproperty(value=0, name="Bluesky:Run:Suspended")
    run_start = pvproperty(value=0.0, name="Bluesky:Run:StartEpoch")
    run_finish = pvproperty(value=0.0, name="Bluesky:Run:FinishEpoch")
    run_update = pvproperty(value=0.0, name="Bluesky:Run:LastUpdateEpoch")
    run_done = pvproperty(value=0, name="Bluesky:Run:DoneUnits")
    run_total = pvproperty(value=0, name="Bluesky:Run:TotalUnits")
    run_plan = pvproperty(value="", name="Bluesky:Run:PlanName")
    run_uid = pvproperty(value="", name="Bluesky:Run:RunUID")
    run_suspend_since = pvproperty(value=0.0, name="Bluesky:Run:SuspendSinceEpoch")
    run_suspend_reason = pvproperty(value="", name="Bluesky:Run:SuspendReason")
    run_last_success = pvproperty(value=1, name="Bluesky:Run:LastRunSuccess")
    run_last_exit = pvproperty(value="", name="Bluesky:Run:LastRunExitStatus")

    @heartbeat.scan(period=1.0)
    async def heartbeat(self, instance, async_lib):
        await instance.write(0 if int(instance.value) else 1)

    @allstop.putter
    async def allstop(self, instance, value):
        if int(value):
            for name in MOTOR_DESCRIPTIONS:
                await getattr(self, name).motor.fields["STOP"].write(1)
        return 0


def main():
    prefix = os.environ.get("MITR_DEMO_EPICS_PREFIX", "demo4dh4:")
    ioc_options, run_options = ioc_arg_parser(
        default_prefix=prefix,
        desc="Disconnected diffractometer demonstration IOC",
        supported_async_libs=("asyncio",),
    )
    run_options["interfaces"] = ["127.0.0.1"]
    ioc = DemoIOC(**ioc_options)
    run(ioc.pvdb, **run_options)


if __name__ == "__main__":
    main()
