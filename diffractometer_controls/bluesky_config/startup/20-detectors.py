import numpy as np
from ophyd import (Device, Component as Cpt,FormattedComponent as FCpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, DerivedSignal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus


HE3PSD_POSITION_MIN = -209.21799055746422
HE3PSD_POSITION_MAX = 209.21799055746422


class PositionSignal(DerivedSignal):
    def inverse(self, value):
        nbins = max(1, int(round(float(value))))
        return np.linspace(HE3PSD_POSITION_MIN, HE3PSD_POSITION_MAX, nbins)

    def forward(self, value):
        return len(value)


class HE3PSD(Device):

    acquire = Cpt(EpicsSignalWithRBV, "Acquire",kind='config')
    acquire_time = Cpt(EpicsSignalWithRBV, "AcquireTime",kind='config')
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
