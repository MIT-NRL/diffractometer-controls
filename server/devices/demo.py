"""Ophyd devices backed exclusively by the demo caproto IOC."""

from __future__ import annotations

from ophyd import (
    Component as Cpt,
    Device,
    DerivedSignal,
    EpicsMotor,
    EpicsSignal,
    EpicsSignalRO,
    EpicsSignalWithRBV,
    FormattedComponent as FCpt,
    Signal,
)
from ophyd.pseudopos import (
    PseudoPositioner,
    PseudoSingle,
    pseudo_position_argument,
    real_position_argument,
)
from ophyd.status import SubscriptionStatus
from server.devices.detectors import USBCTR08Scaler, USB2408Readouts


class DemoHE3PositionSignal(DerivedSignal):
    def inverse(self, value):
        import numpy as np

        return np.linspace(-209.21799055746422, 209.21799055746422, max(1, int(value)))

    def forward(self, value):
        return len(value)


class DemoHE3PSD(Device):
    detector_type = "he3psd"
    plan_editor_group = "HE3_PSD"
    acquisition_monitor_signals = {"active": "acquire", "duration": "acquire_time", "remaining": "acquire_time_remaining"}
    live_plot_signals = {"counts": {"role": "profile", "label": "PSD Counts", "units": "counts"}, "total_counts": {"role": "summary", "label": "Total Counts", "units": "counts"}}
    acquire = Cpt(EpicsSignalWithRBV, "Acquire", kind="config")
    acquire_time = Cpt(EpicsSignalWithRBV, "AcquireTime", kind="config")
    acquire_time_remaining = Cpt(EpicsSignalRO, "AcquireTimeRemaining_RBV", kind="omitted")
    nbins = Cpt(EpicsSignalWithRBV, "NBins", kind="config")
    soft_lld = Cpt(EpicsSignalWithRBV, "SoftLLD", kind="config")
    position_x = Cpt(DemoHE3PositionSignal, derived_from="nbins", kind="hinted")
    counts = FCpt(EpicsSignalRO, "{prefix}{self._det_num}:LiveCounts", kind="hinted")
    total_counts = FCpt(EpicsSignalRO, "{prefix}{self._det_num}:LiveTotalCounts", kind="hinted")

    def __init__(self, prefix, *, det_num, **kwargs):
        self._det_num = det_num
        super().__init__(prefix, **kwargs)

    def trigger(self):
        status = SubscriptionStatus(
            self.acquire,
            lambda *, old_value, value, **kwargs: old_value == 1 and value == 0,
            run=False,
        )
        self.acquire.set(1)
        return status

    def stop(self, *, success=False):
        self.acquire.put(0, wait=False)
        return super().stop(success=success)


class DemoCameraControls(Device):
    acquire = Cpt(EpicsSignalWithRBV, "Acquire", kind="config")
    acquire_time = Cpt(EpicsSignalWithRBV, "AcquireTime", kind="config")
    time_remaining = Cpt(EpicsSignalRO, "TimeRemaining_RBV", kind="config")
    detector_state = Cpt(EpicsSignalRO, "DetectorState_RBV", kind="config")
    array_rate = Cpt(EpicsSignalRO, "ArrayRate_RBV", kind="config")
    array_counter = Cpt(EpicsSignalWithRBV, "ArrayCounter", kind="config")
    status_message = Cpt(EpicsSignalRO, "StatusMessage_RBV", kind="config")
    size_x = Cpt(EpicsSignalRO, "SizeX_RBV", kind="config")
    size_y = Cpt(EpicsSignalRO, "SizeY_RBV", kind="config")
    image_mode = Cpt(EpicsSignalWithRBV, "ImageMode", kind="config")
    gain = Cpt(EpicsSignalWithRBV, "Gain", kind="config")
    offset = Cpt(EpicsSignalWithRBV, "Offset", kind="config")
    temperature = Cpt(EpicsSignal, "Temperature", kind="config")
    temperature_actual = Cpt(EpicsSignalRO, "TemperatureActual", kind="config")


class DemoTIFFCompatibility(Device):
    """In-memory filename controls; the demo never writes image files."""

    file_name = Cpt(Signal, value="demo", kind="config")
    folder_name = Cpt(Signal, value="", kind="config")


class DemoStats(Device):
    total = Cpt(EpicsSignalRO, "Total", kind="hinted")


class DemoImagingDetector(Device):
    cam = Cpt(DemoCameraControls, "")
    stats1 = Cpt(DemoStats, "")
    tiff1 = Cpt(DemoTIFFCompatibility, "")
    image = FCpt(EpicsSignalRO, "{self._motor_prefix}image1:ArrayData", kind="hinted")
    focus = FCpt(EpicsMotor, "{self._motor_prefix}m12", labels={"positioner"})
    x = FCpt(EpicsMotor, "{self._motor_prefix}m1", labels={"positioner"})

    def __init__(self, prefix, *, motor_prefix, **kwargs):
        self._motor_prefix = motor_prefix
        super().__init__(prefix, **kwargs)
        self.stage_sigs[self.cam.image_mode] = 0

    def read(self):
        import numpy as np
        result = super().read()
        if self.image.name in result:
            shape = (int(self.cam.size_y.get()), int(self.cam.size_x.get()))
            result[self.image.name]["value"] = np.asarray(result[self.image.name]["value"]).reshape(shape)
        return result

    def describe(self):
        result = super().describe()
        if self.image.name in result:
            result[self.image.name]["shape"] = [int(self.cam.size_y.get()), int(self.cam.size_x.get())]
        return result

    def stop(self, *, success=False):
        self.cam.acquire.put(0, wait=False)
        return super().stop(success=success)

    def trigger(self):
        status = SubscriptionStatus(
            self.cam.acquire,
            lambda *, old_value, value, **kwargs: old_value == 1 and value == 0,
            run=False,
        )
        self.cam.acquire.set(1)
        return status

    @property
    def hints(self):
        return {"fields": [self.stats1.total.name]}


class Pinhole(Device):
    y = Cpt(EpicsMotor, "m16", labels={"positioner"})


class Stage1(Device):
    theta = Cpt(EpicsMotor, "m3", labels={"positioner"})


class Stage2(Device):
    theta = Cpt(EpicsMotor, "m13", labels={"positioner"})
    x = Cpt(EpicsMotor, "m14", labels={"positioner"})


class AnalyzerCurvature(PseudoPositioner):
    curve = Cpt(PseudoSingle, limits=(0, 0.7), egu="1/m")
    counts = Cpt(EpicsMotor, "", name="counts")

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        return self.RealPosition(counts=pseudo_pos.curve / 0.0005516111545194904)

    @real_position_argument
    def inverse(self, real_pos):
        return self.PseudoPosition(curve=real_pos.counts * 0.0005516111545194904)


def create_demo_devices(prefix="demo4dh4:"):
    """Build the stable public device namespace used by demo startup."""
    devices = {
        "stage1": Stage1(prefix, name="stage1", read_attrs=["theta"]),
        "stage2": Stage2(prefix, name="stage2", read_attrs=["theta", "x"]),
        "pinhole": Pinhole(prefix, name="pinhole", read_attrs=["y"]),
        "sample_th": EpicsMotor(f"{prefix}m10", name="sample_th", labels={"positioner"}),
        "det_psd_x": EpicsMotor(f"{prefix}m9", name="det_psd_x", labels={"positioner"}),
        "analyzer1": AnalyzerCurvature(f"{prefix}m11", name="analyzer1"),
        "analyzer2": AnalyzerCurvature(f"{prefix}m15", name="analyzer2"),
        "he3psd0": DemoHE3PSD(f"{prefix}he3PSD:", det_num="Det0", name="he3psd0"),
        "he3psd7": DemoHE3PSD(f"{prefix}he3PSD:", det_num="Det7", name="he3psd7"),
        "usbctr": USBCTR08Scaler(f"{prefix}USBCTR:scaler1", name="usbctr"),
        "usb2408": USB2408Readouts(f"{prefix}USB2408:", name="usb2408"),
        "cam1": DemoImagingDetector(
            f"{prefix}cam1:", motor_prefix=prefix, name="cam1",
            read_attrs=["image", "stats1.total", "focus", "x"],
        ),
        "sim_focus_cam": DemoImagingDetector(
            f"{prefix}simFocus:", motor_prefix=prefix, name="sim_focus_cam",
            read_attrs=["image", "stats1.total", "focus", "x"],
        ),
    }
    # The Gaussian camera's vertical centre follows stage2.x.  The component
    # is already exposed through stage2; this alias makes its coupling clear.
    devices["cam1_y"] = devices["stage2"].x
    return devices
