import numpy as np
import threading
import time
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from bluesky_queueserver import register_device

try:
    from diffractometer_controls.sim_focus import (
        SimulatedFocusDetector,
        SimulatedFocusMotor,
    )
except ModuleNotFoundError:
    from pathlib import Path
    import sys

    package_root = Path(__file__).resolve().parents[3]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.sim_focus import (
        SimulatedFocusDetector,
        SimulatedFocusMotor,
    )


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

    detector_type = "he3psd"
    plan_editor_group = "Simulated_HE3_PSD"
    live_plot_signals = {
        "counts": {
            "role": "profile",
            "label": "PSD Counts",
            "units": "counts",
            "transport": "document",
        },
        "total_counts": {
            "role": "summary",
            "label": "Total Counts",
            "units": "counts",
            "transport": "document",
        },
    }

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


class SimUSBCTR08Scaler(Device):
    """In-memory CTR-08 scaler simulator with live accumulated counts.

    Its read keys and component names match :class:`USBCTR08Scaler`. Because
    these are soft Ophyd Signals rather than EPICS PVs, scalar plans publish
    their intermediate changes through Bluesky monitor streams.
    """

    scalar_plan_compatible = True
    detector_type = "usbctr08"
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
            "transport": "document",
        },
        "beam_monitor": {
            "role": "signal",
            "label": "CTR1 - Beam Monitor",
            "units": "counts",
            "transport": "document",
        },
        "he3_tube": {
            "role": "signal",
            "label": "CTR2 - He-3 Tube",
            "units": "counts",
            "transport": "document",
        },
        **{
            f"counter_{channel}": {
                "role": "signal",
                "label": f"CTR{channel}",
                "units": "counts",
                "transport": "document",
            }
            for channel in range(3, 8)
        },
        "time": {
            "role": "elapsed_time",
            "label": "Elapsed Time",
            "units": "s",
            "transport": "document",
        },
    }

    count = Cpt(Signal, value=0, kind="omitted")
    acquire_time = Cpt(Signal, value=2.0, kind="config")
    clock_counts = Cpt(Signal, value=0, kind="normal")
    beam_monitor = Cpt(Signal, value=0, kind="hinted")
    he3_tube = Cpt(Signal, value=0, kind="hinted")
    counter_3 = Cpt(Signal, value=0, kind="normal")
    counter_4 = Cpt(Signal, value=0, kind="normal")
    counter_5 = Cpt(Signal, value=0, kind="normal")
    counter_6 = Cpt(Signal, value=0, kind="normal")
    counter_7 = Cpt(Signal, value=0, kind="normal")
    time = Cpt(Signal, value=0.0, kind="normal")

    def __init__(
        self,
        *args,
        beam_monitor_rate=2400.0,
        he3_tube_rate=850.0,
        update_period=0.1,
        clock_frequency=10_000_000.0,
        random_seed=37,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._beam_monitor_rate = float(beam_monitor_rate)
        self._he3_tube_rate = float(he3_tube_rate)
        self._update_period = max(0.02, float(update_period))
        self._clock_frequency = float(clock_frequency)
        self._rng = np.random.default_rng(int(random_seed))
        self._trigger_thread = None
        self._trigger_status = None
        self._stop_requested = threading.Event()

    def _reset_counts(self):
        self.clock_counts.put(0)
        self.beam_monitor.put(0)
        self.he3_tube.put(0)
        for attr in ("counter_3", "counter_4", "counter_5", "counter_6", "counter_7"):
            getattr(self, attr).put(0)
        self.time.put(0.0)

    def _acquire_once(self, status):
        target = max(0.0, float(self.acquire_time.get()))
        started = time.monotonic()
        previous_elapsed = 0.0
        beam_counts = 0
        tube_counts = 0
        try:
            while not self._stop_requested.is_set():
                elapsed = min(target, max(0.0, time.monotonic() - started))
                interval = max(0.0, elapsed - previous_elapsed)
                if interval:
                    beam_counts += int(self._rng.poisson(self._beam_monitor_rate * interval))
                    tube_counts += int(self._rng.poisson(self._he3_tube_rate * interval))
                    # Publish the new elapsed time first so each following
                    # counter update is divided by the matching interval in
                    # the live count-rate display.
                    self.time.put(float(elapsed))
                    self.clock_counts.put(int(round(self._clock_frequency * elapsed)))
                    self.beam_monitor.put(beam_counts)
                    self.he3_tube.put(tube_counts)
                    previous_elapsed = elapsed
                if elapsed >= target:
                    break
                time.sleep(min(self._update_period, max(0.0, target - elapsed)))
        except Exception as ex:
            self.count.put(0)
            status.set_exception(ex)
            return

        self.count.put(0)
        status.set_finished()

    def trigger(self):
        if self._trigger_status is not None and not self._trigger_status.done:
            raise RuntimeError(f"{self.name} is already counting")
        self._stop_requested.clear()
        self._reset_counts()
        self.count.put(1)
        status = Status()
        self._trigger_status = status
        self._trigger_thread = threading.Thread(
            target=self._acquire_once,
            args=(status,),
            name=f"{self.name}-trigger",
            daemon=True,
        )
        self._trigger_thread.start()
        return status

    def stop(self, *, success=False):
        self._stop_requested.set()
        self.count.put(0)
        return super().stop(success=success)


sim_usbctr = SimUSBCTR08Scaler(name="sim_usbctr")
register_device("sim_usbctr", depth=2)


# End-to-end adaptive-focus simulation. These names appear as selectable
# detector/motor devices in the Queue Server plan editor.
sim_focus_motor = SimulatedFocusMotor(name="sim_focus_motor", value=0.0)
sim_focus_cam = SimulatedFocusDetector(
    name="sim_focus_cam",
    motor=sim_focus_motor,
    best_focus=0.0,
)
register_device("sim_focus_motor", depth=1)
# Keep the detector shallow and expose only its supported public scan axis.
register_device("sim_focus_cam", depth=2)
for _camera_axis_attr in ("acquire_time",):
    _camera_axis_name = f"sim_focus_cam_{_camera_axis_attr}"
    _camera_axis = getattr(sim_focus_cam.cam, _camera_axis_attr)
    _camera_axis.kind = "hinted"
    # Available as an imaging scan axis, but not as a scalar detector.
    _camera_axis.scalar_plan_hidden = True
    globals()[_camera_axis_name] = _camera_axis
    register_device(_camera_axis_name, depth=1)
sd.baseline.append(sim_focus_motor)
