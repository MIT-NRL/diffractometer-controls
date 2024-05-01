from ophyd import Device, EpicsSignal, EpicsSignalRO, Component
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsMotor)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus


class EpicsSignalWithRBV(EpicsSignal):
    def __init__(self, prefix, **kwargs):
        super().__init__(prefix + "_RBV", write_pv=prefix, **kwargs)


class EpicsSignalWithRBV(EpicsSignal):
    def __init__(self, prefix, **kwargs):
        super().__init__(prefix + "_RBV", write_pv=prefix, **kwargs)

class HE3PSD(Device):
    acquire = Cpt(EpicsSignalWithRBV, ":Acquire")
    acquire_time = Cpt(EpicsSignalWithRBV, ":AcquireTime")
    nbins = Cpt(EpicsSignalWithRBV, ":NBins")

    counts0 = Cpt(EpicsSignalRO, ":CountsD0")
    counts1 = Cpt(EpicsSignalRO, ":CountsD1")
    counts2 = Cpt(EpicsSignalRO, ":CountsD2")
    counts3 = Cpt(EpicsSignalRO, ":CountsD3")
    counts4 = Cpt(EpicsSignalRO, ":CountsD4")
    counts5 = Cpt(EpicsSignalRO, ":CountsD5")
    counts6 = Cpt(EpicsSignalRO, ":CountsD6")
    counts7 = Cpt(EpicsSignalRO, ":CountsD7")
    total_counts = Cpt(EpicsSignalRO, ":TotalCounts")

    live_counts0 = Cpt(EpicsSignalRO, ":LiveCountsD0")
    live_counts1 = Cpt(EpicsSignalRO, ":LiveCountsD1")
    live_counts2 = Cpt(EpicsSignalRO, ":LiveCountsD2")
    live_counts3 = Cpt(EpicsSignalRO, ":LiveCountsD3")
    live_counts4 = Cpt(EpicsSignalRO, ":LiveCountsD4")
    live_counts5 = Cpt(EpicsSignalRO, ":LiveCountsD5")
    live_counts6 = Cpt(EpicsSignalRO, ":LiveCountsD6")
    live_counts7 = Cpt(EpicsSignalRO, ":LiveCountsD7")
    live_total_counts = Cpt(EpicsSignalRO, ":LiveTotalCounts")

    _default_read_attrs = (
        "counts0",
        "counts7",
        "total_counts"
    )

    _default_configuration_attrs = (
        "acquire_time",
        "nbins"
    )

    def trigger(self):
        def check_value(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            return (old_value == 1 and value == 0)

        self.acquire.set(1).wait()
        status = SubscriptionStatus(self.acquire, check_value)
        return status
    

det1 = HE3PSD("4dh4:det1", name="det1")