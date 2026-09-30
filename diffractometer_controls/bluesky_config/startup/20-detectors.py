import numpy as np
from ophyd import (Device, Component as Cpt,FormattedComponent as FCpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, DerivedSignal)
from ophyd.device import DeviceStatus
from ophyd.scaler import EpicsScaler
from ophyd.status import Status, SubscriptionStatus
from bluesky_queueserver import register_device


HE3PSD_POSITION_MIN = -209.21799055746422
HE3PSD_POSITION_MAX = 209.21799055746422


class PositionSignal(DerivedSignal):
    def inverse(self, value):
        nbins = max(1, int(round(float(value))))
        return np.linspace(HE3PSD_POSITION_MIN, HE3PSD_POSITION_MAX, nbins)

    def forward(self, value):
        return len(value)


class HE3PSD(Device):

    detector_type = "he3psd"
    plan_editor_group = "HE3_PSD"
    acquisition_monitor_signals = {
        "active": "acquire",
        "duration": "acquire_time",
        "remaining": "acquire_time_remaining",
    }
    live_plot_signals = {
        "counts": {"role": "profile", "label": "PSD Counts", "units": "counts"},
        "total_counts": {
            "role": "summary",
            "label": "Total Counts",
            "units": "counts",
        },
    }

    acquire = Cpt(EpicsSignalWithRBV, "Acquire",kind='config')
    acquire_time = Cpt(EpicsSignalWithRBV, "AcquireTime",kind='config')
    acquire_time_remaining = Cpt(
        EpicsSignalRO,
        "AcquireTimeRemaining_RBV",
        kind="omitted",
    )
    nbins = Cpt(EpicsSignalWithRBV, "NBins",kind='config')
    soft_lld = Cpt(EpicsSignalWithRBV, "SoftLLD",kind='config')

    position_x = Cpt(PositionSignal, derived_from="nbins", kind="hinted")

    counts = FCpt(EpicsSignalRO, "{prefix}{_det_num}:LiveCounts",name="counts",kind="hinted")

    total_counts = FCpt(EpicsSignalRO, "{prefix}{_det_num}:LiveTotalCounts",name="total_counts",kind="hinted")
    
    def trigger(self):
        def check_value(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            return (old_value == 1 and value == 0)

        self.acquire.set(1).wait()
        status = SubscriptionStatus(self.acquire, check_value)
        return status
    
    def __init__(self, prefix, det_num: str, **kwargs):
        self._det_num = det_num
        super().__init__(prefix, **kwargs)
    

he3psd0 = HE3PSD("4dh4:he3PSD:",det_num="Det0", name="he3psd0")
he3psd7 = HE3PSD("4dh4:he3PSD:",det_num="Det7", name="he3psd7")


class USB2408TemperatureInputs(Device):
    """Read-only USB-2408 thermocouple inputs (``Ti1`` through ``Ti8``)."""

    ti1 = Cpt(EpicsSignalRO, "Ti1", kind="hinted")
    ti2 = Cpt(EpicsSignalRO, "Ti2", kind="hinted")
    ti3 = Cpt(EpicsSignalRO, "Ti3", kind="hinted")
    ti4 = Cpt(EpicsSignalRO, "Ti4", kind="hinted")
    ti5 = Cpt(EpicsSignalRO, "Ti5", kind="hinted")
    ti6 = Cpt(EpicsSignalRO, "Ti6", kind="hinted")
    ti7 = Cpt(EpicsSignalRO, "Ti7", kind="hinted")
    ti8 = Cpt(EpicsSignalRO, "Ti8", kind="hinted")


class USB2408AnalogInputs(Device):
    """Read-only USB-2408 analog inputs (``Ai1`` through ``Ai8``)."""

    ai1 = Cpt(EpicsSignalRO, "Ai1", kind="hinted")
    ai2 = Cpt(EpicsSignalRO, "Ai2", kind="hinted")
    ai3 = Cpt(EpicsSignalRO, "Ai3", kind="hinted")
    ai4 = Cpt(EpicsSignalRO, "Ai4", kind="hinted")
    ai5 = Cpt(EpicsSignalRO, "Ai5", kind="hinted")
    ai6 = Cpt(EpicsSignalRO, "Ai6", kind="hinted")
    ai7 = Cpt(EpicsSignalRO, "Ai7", kind="hinted")
    ai8 = Cpt(EpicsSignalRO, "Ai8", kind="hinted")


class USB2408BinaryInputs(Device):
    """Read-only USB-2408 binary inputs (``Bi1`` through ``Bi8``)."""

    bi1 = Cpt(EpicsSignalRO, "Bi1", kind="hinted")
    bi2 = Cpt(EpicsSignalRO, "Bi2", kind="hinted")
    bi3 = Cpt(EpicsSignalRO, "Bi3", kind="hinted")
    bi4 = Cpt(EpicsSignalRO, "Bi4", kind="hinted")
    bi5 = Cpt(EpicsSignalRO, "Bi5", kind="hinted")
    bi6 = Cpt(EpicsSignalRO, "Bi6", kind="hinted")
    bi7 = Cpt(EpicsSignalRO, "Bi7", kind="hinted")
    bi8 = Cpt(EpicsSignalRO, "Bi8", kind="hinted")


class USB2408Readouts(Device):
    """Passive scalar readouts from a Measurement Computing USB-2408.

    Unlike a triggered detector, selecting one of these child signals should
    read only that signal. ``scalar_readout_channels`` exposes the supported
    leaves to the general scalar plans without staging or triggering the
    complete multifunction board.
    """

    temperature = Cpt(USB2408TemperatureInputs, "")
    analog = Cpt(USB2408AnalogInputs, "")
    binary = Cpt(USB2408BinaryInputs, "")
    scalar_readout_channels = (
        tuple(f"temperature.ti{channel}" for channel in range(1, 9))
        + tuple(f"analog.ai{channel}" for channel in range(1, 9))
        + tuple(f"binary.bi{channel}" for channel in range(1, 9))
    )


usb2408 = USB2408Readouts("4dh4:USB2408:", name="usb2408")
register_device("usb2408", depth=3)


class USBCTR08Scaler(EpicsScaler):
    """Measurement Computing USB-CTR08 exposed as an EPICS scaler.

    In timed scaler mode physical CTR0 is the clock/preset counter.  Its
    output must be wired to the gate inputs of CTR1 through CTR7.  The
    remaining counters can then be used for detector pulses.
    """

    # Used by the general scalar-plan device collector.  The scaler produces
    # scalar fields even though it is a multi-signal Device.
    scalar_plan_compatible = True
    detector_type = "usbctr08"
    clock_frequency_hz = 1000.0
    trigger_timeout_margin_s = 5.0
    acquisition_monitor_signals = {
        "active": "count",
        "duration": "preset_time",
        "elapsed": "time",
    }
    scalar_channels = (
        "clock_counts",
        "beam_monitor",
        "he3_tube",
        "counter_3",
        "counter_4",
        "counter_5",
        "counter_6",
        "counter_7",
    )
    live_plot_signals = {
        "clock_counts": {
            "role": "signal",
            "label": "CTR0 - Clock",
            "units": "counts",
        },
        "beam_monitor": {
            "role": "signal",
            "label": "CTR1 - Beam Monitor",
            "units": "counts",
        },
        "he3_tube": {
            "role": "signal",
            "label": "CTR2 - He-3 Tube",
            "units": "counts",
        },
        **{
            f"counter_{channel}": {
                "role": "signal",
                "label": f"CTR{channel}",
                "units": "counts",
            }
            for channel in range(3, 8)
        },
        "time": {
            "role": "elapsed_time",
            "label": "Elapsed Time",
            "units": "s",
        },
    }

    # EpicsScaler treats .T as configuration.  It is acquisition data for
    # this device because rates must use the actual elapsed time.
    time = Cpt(EpicsSignalRO, ".T", kind="normal")

    # Stable Ophyd names for the eight physical inputs.  In scaler mode CTR0
    # is the timing reference, so the first and second detector inputs are
    # CTR1 and CTR2 (.S2 and .S3), respectively.
    clock_counts = Cpt(EpicsSignalRO, ".S1", kind="normal")
    beam_monitor = Cpt(EpicsSignalRO, ".S2", kind="hinted")
    he3_tube = Cpt(EpicsSignalRO, ".S3", kind="hinted")
    counter_3 = Cpt(EpicsSignalRO, ".S4", kind="normal")
    counter_4 = Cpt(EpicsSignalRO, ".S5", kind="normal")
    counter_5 = Cpt(EpicsSignalRO, ".S6", kind="normal")
    counter_6 = Cpt(EpicsSignalRO, ".S7", kind="normal")
    counter_7 = Cpt(EpicsSignalRO, ".S8", kind="normal")

    # The scaler record requires Counter 0 to receive a real pulse train.
    # PulseGen1 is the board's TMR0 output; a short external jumper connects
    # TMR0 to C0IN.  A second jumper from C0OUT to each detector counter's
    # gate input makes Counter 0 the common hardware exposure gate.
    pulse_frequency = FCpt(
        EpicsSignal,
        "{self._board_prefix}PulseGen1Frequency",
        kind="config",
    )
    pulse_frequency_readback = FCpt(
        EpicsSignalRO,
        "{self._board_prefix}PulseGen1Frequency_RBV",
        kind="config",
    )
    pulse_duty_cycle = FCpt(
        EpicsSignal,
        "{self._board_prefix}PulseGen1DutyCycle",
        kind="config",
    )
    pulse_count = FCpt(
        EpicsSignal,
        "{self._board_prefix}PulseGen1Count",
        kind="config",
    )
    pulse_run = FCpt(
        EpicsSignal,
        "{self._board_prefix}PulseGen1Run",
        kind="omitted",
    )

    def __init__(self, *args, **kwargs):
        prefix = args[0] if args else kwargs.get("prefix", "")
        self._board_prefix = kwargs.pop("board_prefix", None) or str(
            prefix
        ).removesuffix("scaler1")
        # Use stable, descriptive data keys rather than the generic scaler's
        # 32-channel ``channels.chanN`` hierarchy.
        kwargs.setdefault(
            "read_attrs",
            [
                "clock_counts",
                "beam_monitor",
                "he3_tube",
                "counter_3",
                "counter_4",
                "counter_5",
                "counter_6",
                "counter_7",
                "time",
            ],
        )
        kwargs.setdefault(
            "configuration_attrs",
            [
                "preset_time",
                "freq",
                "count_mode",
                "delay",
                "pulse_frequency",
                "pulse_frequency_readback",
                "pulse_duty_cycle",
                "pulse_count",
            ],
        )
        super().__init__(*args, **kwargs)

        # One-shot mode and counter 0 as the only preset counter are required
        # for hardware-timed scaler acquisition.  Ophyd restores the previous
        # values when the device is unstaged.
        self.stage_sigs.update(
            [("count_mode", 0), ("gates.gate1", 1)]
            + [(f"gates.gate{channel}", 0) for channel in range(2, 9)]
            + [
                ("freq", self.clock_frequency_hz),
                ("pulse_frequency", self.clock_frequency_hz),
                ("pulse_duty_cycle", 0.5),
                ("pulse_count", 0),
                ("pulse_run", 1),
            ]
        )

    @property
    def acquire_time(self):
        """Alias for the scaler preset time used by acquisition plans."""
        return self.preset_time

    def stage(self):
        if self.count.get() != 0:
            raise RuntimeError(f"Cannot stage {self.name} while it is counting")
        staged = super().stage()
        try:
            # The scaler record converts preset time to CTR0 ticks only when
            # .TP is processed.  Plans may set .TP before staging, so changing
            # .FREQ here without reprocessing .TP would leave .PR1 calculated
            # from the old clock.  Use the pulse generator's actual frequency,
            # as required by the measComp scaler documentation, then rewrite
            # the current preset time to regenerate .PR1.
            actual_frequency = float(self.pulse_frequency_readback.get())
            if not np.isfinite(actual_frequency) or actual_frequency <= 0:
                raise RuntimeError(
                    f"Cannot stage {self.name}: PulseGen1 frequency readback "
                    f"is not positive ({actual_frequency!r})"
                )
            self.freq.set(actual_frequency).wait()
            self.preset_time.set(float(self.preset_time.get())).wait()
        except Exception:
            self.unstage()
            raise
        return staged

    def trigger(self):
        """Start one scaler exposure with a finite hardware-failure timeout."""
        acquire_time = max(0.0, float(self.preset_time.get()))
        timeout = acquire_time + max(
            self.trigger_timeout_margin_s,
            acquire_time * 0.1,
        )
        status = DeviceStatus(self, timeout=timeout)

        def acquisition_complete(**_kwargs):
            if not status.done:
                status.set_finished()

        try:
            # The scaler record's put callback completes when asynchronous
            # counting completes, not merely when CNT is written.
            self.count.put(1, wait=False, callback=acquisition_complete)
        except Exception as exc:
            status.set_exception(exc)
        return status

    def stop(self, *, success=False):
        # Device.stop() only visits child devices; explicitly stop the scaler
        # record as well so an interrupted Bluesky run stops the hardware.
        self.count.put(0, wait=False)
        return super().stop(success=success)


# Leave the class importable for plan/display development, but do not create
# EPICS connections until the CTR-08 has arrived and its IOC is enabled.
ENABLE_USBCTR08 = True

if ENABLE_USBCTR08:
    usbctr = USBCTR08Scaler("4dh4:USBCTR:scaler1", name="usbctr")
    # QueueServer and the plan editor expose direct components using dotted
    # paths such as ``usbctr.beam_monitor``.
    register_device("usbctr", depth=2)
