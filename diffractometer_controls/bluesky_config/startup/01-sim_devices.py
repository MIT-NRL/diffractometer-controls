import numpy as np
import threading
import time
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus


sim_motor = EpicsMotor("4dh4:m6", name="sim_motor")


def _build_position_axis(nbins):
    n = max(1, int(round(float(nbins))))
    return np.linspace(-209.21799055746422, 209.21799055746422, n)


class SimHE3PSD(Device):
    """
    Synthetic HE3 PSD detector with the same read interface as the live device.

    The spectrum is a Gaussian peak riding on a noisy background. The peak
    position and intensity both change with the simulation motor, with maximum
    intensity near motor position 3.
    """

    acquire = Cpt(Signal, value=0, kind="config")
    acquire_time = Cpt(Signal, value=0.2, kind="config")
    nbins = Cpt(Signal, value=350, kind="config")
    soft_lld = Cpt(Signal, value=0.0, kind="config")
    position_x = Cpt(Signal, value=_build_position_axis(350), kind="hinted")
    counts = Cpt(Signal, value=np.zeros(350, dtype=float), kind="hinted")
    total_counts = Cpt(Signal, value=0.0, kind="hinted")

    def __init__(
        self,
        *args,
        motor,
        peak_motor=3.0,
        amplitude_scale=3200.0,
        baseline_counts=25.0,
        width=22.0,
        center_offset=0.0,
        center_motor_scale=18.0,
        shoulder_fraction=0.0,
        shoulder_offset=0.0,
        shoulder_width_scale=1.6,
        background_phase=0.0,
        noise_scale=1.0,
        random_seed=0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._motor = motor
        self._peak_motor = float(peak_motor)
        self._amplitude_scale = float(amplitude_scale)
        self._baseline_counts = float(baseline_counts)
        self._width = float(width)
        self._center_offset = float(center_offset)
        self._center_motor_scale = float(center_motor_scale)
        self._shoulder_fraction = float(shoulder_fraction)
        self._shoulder_offset = float(shoulder_offset)
        self._shoulder_width_scale = float(shoulder_width_scale)
        self._background_phase = float(background_phase)
        self._noise_scale = float(noise_scale)
        self._rng = np.random.default_rng(int(random_seed))
        self._trigger_thread = None
        self.position_x.put(_build_position_axis(self.nbins.get()))
        self.counts.put(np.zeros(int(self.nbins.get()), dtype=float))
        self.total_counts.put(0.0)

    def _read_motor_position(self):
        try:
            return float(self._motor.position)
        except Exception:
            return 0.0

    def _gaussian_envelope(self, motor_pos):
        return float(np.exp(-0.5 * ((motor_pos - self._peak_motor) / 1.2) ** 2))

    def _peak_center(self, motor_pos):
        return self._center_offset + self._center_motor_scale * (motor_pos - self._peak_motor)

    def _generate_profile(self):
        nbins = max(1, int(round(float(self.nbins.get()))))
        axis = _build_position_axis(nbins)
        motor_pos = self._read_motor_position()
        envelope = self._gaussian_envelope(motor_pos)

        amplitude = self._amplitude_scale * (0.2 + 0.8 * envelope)
        center = self._peak_center(motor_pos)
        width = self._width * (1.0 + 0.10 * abs(motor_pos - self._peak_motor))

        background = self._baseline_counts * (
            1.0
            + 0.15 * np.cos(axis / 32.0)
            + 0.10 * np.sin((axis / 55.0) + self._background_phase + (0.3 * motor_pos))
        )
        peak = amplitude * np.exp(-0.5 * ((axis - center) / max(width, 1.0)) ** 2)
        shoulder = (
            self._shoulder_fraction
            * amplitude
            * np.exp(
                -0.5
                * (
                    (axis - (center + self._shoulder_offset))
                    / max(width * self._shoulder_width_scale, 1.0)
                )
                ** 2
            )
        )

        expected = np.clip(background + peak + shoulder, 0.0, None)
        noisy = self._rng.poisson(np.clip(expected, 0.0, None))
        noisy = noisy + self._rng.normal(
            loc=0.0,
            scale=self._noise_scale * np.sqrt(np.clip(expected, 1.0, None)),
            size=expected.shape,
        )

        counts = np.clip(np.rint(noisy), 0.0, None)
        return axis, counts.astype(float, copy=False)

    def _acquire_once(self, status):
        try:
            delay = max(0.0, float(self.acquire_time.get()))
            if delay > 0:
                time.sleep(delay)
            axis, counts = self._generate_profile()
            self.position_x.put(axis)
            self.counts.put(counts)
            self.total_counts.put(float(np.sum(counts)))
        except Exception as ex:
            self.acquire.put(0)
            status.set_exception(ex)
            return

        self.acquire.put(0)
        status.set_finished()

    def trigger(self):
        status = Status()
        self.acquire.put(1)
        self._trigger_thread = threading.Thread(
            target=self._acquire_once,
            args=(status,),
            name=f"{self.name}-trigger",
            daemon=True,
        )
        self._trigger_thread.start()
        return status


sim_he3psd0 = SimHE3PSD(
    name="sim_he3psd0",
    motor=sim_motor,
    peak_motor=2.9,
    amplitude_scale=3600.0,
    baseline_counts=28.0,
    width=18.0,
    center_offset=-24.0,
    center_motor_scale=16.0,
    shoulder_fraction=0.18,
    shoulder_offset=12.0,
    shoulder_width_scale=1.4,
    background_phase=0.2,
    noise_scale=0.9,
    random_seed=4,
)

sim_he3psd1 = SimHE3PSD(
    name="sim_he3psd1",
    motor=sim_motor,
    peak_motor=3.2,
    amplitude_scale=3000.0,
    baseline_counts=24.0,
    width=24.0,
    center_offset=26.0,
    center_motor_scale=-13.0,
    shoulder_fraction=0.10,
    shoulder_offset=-18.0,
    shoulder_width_scale=2.1,
    background_phase=1.1,
    noise_scale=1.1,
    random_seed=17,
)
