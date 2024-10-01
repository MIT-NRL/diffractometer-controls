import numpy as np
from bluesky import SupplementalData, RunEngine
from ophyd import (Device, Component as Cpt, FormattedComponent as FCpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from ophyd.pseudopos import (PseudoPositioner, PseudoSingle, real_position_argument, pseudo_position_argument)

sd: SupplementalData

# Tomography/Imaging motors
sample_tomo_th = EpicsMotor("4dh4:m3",name="sample_tomo_th",labels=["positioner"])
sample_tomo_x = EpicsMotor("4dh4:m14",name="sample_tomo_x",labels=["positioner"])

cam_focus = EpicsMotor("4dh4:m12",name="cam_focus",labels=["positioner"])
cam_x = EpicsMotor("4dh4:m1",name="cam_x",labels=["positioner"])


#===============================================================================#
# Diffraction motors
sample_th = EpicsMotor("4dh4:m10",name="sample_th",labels=["positioner"])
sample_x = EpicsMotor("4dh4:m14",name="sample_x",labels=["positioner"])

det_psd_x = EpicsMotor("4dh4:m9",name="det_psd_x",labels=["positioner"])


analyzer1_angle = EpicsMotor("4dh4:m13",name="analyzer1_angle")
analyzer1_x = EpicsMotor("4dh4:m16",name="analyzer1_x")

class AnalyzerCurvature(PseudoPositioner):
    def __init__(self,
                 analyzer_motor_pv: str,
                 *args,
                 **kwargs
                 ):
        self.analyzer_motor_pv = analyzer_motor_pv
        super().__init__(*args, **kwargs)
        
    curve = Cpt(PseudoSingle, limits=(0,0.7), egu='1/m')

    counts = FCpt(EpicsMotor, "{analyzer_motor_pv}", name='counts')

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        return self.RealPosition(counts=pseudo_pos.curve/0.0005516111545194904)
    
    @real_position_argument
    def inverse(self, real_pos):
        return self.PseudoPosition(curve=real_pos.counts*0.0005516111545194904)
    

analyzer1_curve = AnalyzerCurvature("4dh4:m11",name="analyzer1_curve")
analyzer2_curve = AnalyzerCurvature("4dh4:m15",name="analyzer2_curve")


sd.baseline.append(sample_th)
sd.baseline.append(sample_x)
sd.baseline.append(det_psd_x)
sd.baseline.append(analyzer1_angle)
sd.baseline.append(analyzer1_x)
sd.baseline.append(analyzer1_curve)
sd.baseline.append(analyzer2_curve)