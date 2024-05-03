import numpy as np
from ophyd import (Device, Component as Cpt,
                   EpicsSignal, EpicsSignalRO, EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus


class EpicsSignalWithRBV(EpicsSignal):
    def __init__(self, prefix, **kwargs):
        super().__init__(prefix + "_RBV", write_pv=prefix, **kwargs)


class EpicsSignalWithRBV(EpicsSignal):
    def __init__(self, prefix, **kwargs):
        super().__init__(prefix + "_RBV", write_pv=prefix, **kwargs)

class HE3PSD(Device):
    acquire = Cpt(EpicsSignalWithRBV, ":Acquire",kind='config')
    acquire_time = Cpt(EpicsSignalWithRBV, ":AcquireTime",kind='config')
    nbins = Cpt(EpicsSignalWithRBV, ":NBins",kind='config')

    position_x = Cpt(Signal,value=np.linspace(-150,150,300))

    # counts0 = Cpt(EpicsSignalRO, ":CountsD0")
    # counts1 = Cpt(EpicsSignalRO, ":CountsD1")
    # counts2 = Cpt(EpicsSignalRO, ":CountsD2")
    # counts3 = Cpt(EpicsSignalRO, ":CountsD3")
    # counts4 = Cpt(EpicsSignalRO, ":CountsD4")
    # counts5 = Cpt(EpicsSignalRO, ":CountsD5")
    # counts6 = Cpt(EpicsSignalRO, ":CountsD6")
    # counts7 = Cpt(EpicsSignalRO, ":CountsD7")
    # total_counts = Cpt(EpicsSignalRO, ":TotalCounts")

    counts0 = Cpt(EpicsSignalRO, ":LiveCountsD0")
    counts1 = Cpt(EpicsSignalRO, ":LiveCountsD1",kind='omitted')
    counts2 = Cpt(EpicsSignalRO, ":LiveCountsD2",kind='omitted')
    counts3 = Cpt(EpicsSignalRO, ":LiveCountsD3",kind='omitted')
    counts4 = Cpt(EpicsSignalRO, ":LiveCountsD4",kind='omitted')
    counts5 = Cpt(EpicsSignalRO, ":LiveCountsD5",kind='omitted')
    counts6 = Cpt(EpicsSignalRO, ":LiveCountsD6",kind='omitted')
    counts7 = Cpt(EpicsSignalRO, ":LiveCountsD7")
    total_counts = Cpt(EpicsSignalRO, ":LiveTotalCounts")

    # _default_read_attrs = (
    #     "counts0",
    #     "counts7",
    #     "total_counts"
    # )

    # _default_configuration_attrs = (
    #     "acquire_time",
    #     "nbins"
    # )

    def trigger(self):
        def check_value(*, old_value, value, **kwargs):
            "Return True when the acquisition is complete, False otherwise."
            return (old_value == 1 and value == 0)

        self.acquire.set(1).wait()
        status = SubscriptionStatus(self.acquire, check_value)
        return status
    

he3psd = HE3PSD("4dh4:det1", name="he3psd")