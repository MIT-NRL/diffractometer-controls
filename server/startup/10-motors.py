from server.devices.motors import EpicsMotorCustom, Pinhole, Stage1, Stage2, AnalyzerCurvature
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

# cam_focus = EpicsMotorCustom("4dh4:m12",name="cam_focus",labels=["positioner"])
# cam_x = EpicsMotorCustom("4dh4:m1",name="cam_x",labels=["positioner"])

# pinhole_y = EpicsMotorCustom("4dh4:m16",name="pinhole_y",labels=["positioner"])

# Define the pinhole
pinhole = Pinhole("4dh4:", name="pinhole", read_attrs=["y"])

# Define stages

# Define the stages
stage1 = Stage1("4dh4:", name="stage1", read_attrs=["theta"])
stage2 = Stage2("4dh4:", name="stage2", read_attrs=["theta", "x"])

#===============================================================================#
# Diffraction motors
sample_th = EpicsMotorCustom("4dh4:m10",name="sample_th",labels=["positioner"])
# sample_x = EpicsMotorCustom("4dh4:m14",name="sample_x",labels=["positioner"])

det_psd_x = EpicsMotorCustom("4dh4:m9",name="det_psd_x",labels=["positioner"])

# analyzer1_x = EpicsMotorCustom("4dh4:m16",name="analyzer1_x")

# analyzer2_x = EpicsMotorCustom("4dh4:m9",name="analyzer2_x")

analyzer1 = AnalyzerCurvature("4dh4:m11", name="analyzer1", read_attrs=["curve", "counts"])
analyzer2 = AnalyzerCurvature("4dh4:m15", name="analyzer2", read_attrs=["curve", "counts"])

sd.baseline.append(stage1.theta)
sd.baseline.append(stage2.theta)
sd.baseline.append(stage2.x)
sd.baseline.append(pinhole.y)
# sd.baseline.append(pinhole_y)

sd.baseline.append(sample_th)
# sd.baseline.append(sample_x)
sd.baseline.append(det_psd_x)
# sd.baseline.append(analyzer1_x)
sd.baseline.append(analyzer1.curve)
# sd.baseline.append(analyzer2_x)
sd.baseline.append(analyzer2.curve)

# class AnalyzerCurvature(PseudoPositioner):
#     def __init__(self,
#                  prefix: str,
#                  curve_motor_pv: str,
#                  x_motor_pv: str,
#                  th_motor_pv: str,
#                  *args,
#                  **kwargs
#                  ):
#         self._curve_motor_pv = curve_motor_pv
#         self._x_motor_pv = x_motor_pv
#         self._th_motor_pv = th_motor_pv
#         super().__init__(prefix,*args, **kwargs)

#     @pseudo_position_argument
#     def forward(self, pseudo_pos):
#         return self.RealPosition(counts=pseudo_pos.curve/0.0005516111545194904)

#     @real_position_argument
#     def inverse(self, real_pos):
#         return self.PseudoPosition(curve=real_pos.counts*0.0005516111545194904)

# analyzer1 = AnalyzerCurvature("4dh4:",curve_motor_pv="m11",x_motor_pv="m16",th_motor_pv="m13",name="analyzer1")
