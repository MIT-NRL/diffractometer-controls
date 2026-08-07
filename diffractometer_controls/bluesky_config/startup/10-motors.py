import numpy as np
from bluesky import SupplementalData, RunEngine
from ophyd import (Device, Component as Cpt, FormattedComponent as FCpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV, 
                   EpicsMotor, Signal)
from ophyd.device import DeviceStatus
from ophyd.status import Status, SubscriptionStatus
from ophyd.pseudopos import (PseudoPositioner, PseudoSingle, real_position_argument, pseudo_position_argument)

sd: SupplementalData

class EpicsMotorCustom(EpicsMotor):
    torque = Cpt(EpicsSignal, ".CNEN", kind="config", auto_monitor=True)

    def move(self, position, wait=True, **kwargs):
        """
        Move the motor to the specified position, ensuring torque is enabled.

        Parameters
        ----------
        position : float
            The target position to move to.
        wait : bool, optional
            Whether to wait for the motion to complete. Default is True.
        **kwargs : dict
            Additional arguments to pass to the parent class's move method.

        Returns
        -------
        status : MoveStatus
            The status object for the move.
        """
        # Check if torque enabled
        if self.torque.get() != 1:
            # Enable torque if not already enabled
            self.torque.set(1, settle_time=0.1).wait()
        
        # Call the parent class's move method
        return super().move(position, wait=wait, **kwargs)


# Tomography/Imaging motors
# sample_tomo_th = EpicsMotorCustom("4dh4:m3",name="sample_tomo_th",labels=["positioner"])
# sample_tomo_z = EpicsMotorCustom("4dh4:m14",name="sample_tomo_z",labels=["positioner"])

cam_focus = EpicsMotorCustom("4dh4:m12",name="cam_focus",labels=["positioner"])
# cam_x = EpicsMotorCustom("4dh4:m1",name="cam_x",labels=["positioner"])

# pinhole_y = EpicsMotorCustom("4dh4:m16",name="pinhole_y",labels=["positioner"])

class Pinhole(Device):
    y = Cpt(EpicsMotorCustom, "m16", name="y", labels=["positioner"])

# Define the pinhole
pinhole = Pinhole("4dh4:", name="pinhole")

# Define stages
class Stage1(Device):
    theta = Cpt(EpicsMotorCustom, "m3", name="theta", labels=["positioner"])

class Stage2(Device):
    theta = Cpt(EpicsMotorCustom, "m13", name="theta", labels=["positioner"])
    x = Cpt(EpicsMotorCustom, "m14", name="phi", labels=["positioner"])

# Define the stages
stage1 = Stage1("4dh4:", name="stage1")
stage2 = Stage2("4dh4:", name="stage2")


#===============================================================================#
# Diffraction motors
sample_th = EpicsMotorCustom("4dh4:m10",name="sample_th",labels=["positioner"])
# sample_x = EpicsMotorCustom("4dh4:m14",name="sample_x",labels=["positioner"])

det_psd_x = EpicsMotorCustom("4dh4:m9",name="det_psd_x",labels=["positioner"])


# analyzer1_th = EpicsMotorCustom("4dh4:m13",name="analyzer1_th")
# analyzer1_x = EpicsMotorCustom("4dh4:m16",name="analyzer1_x")

# analyzer2_th= EpicsMotorCustom("4dh4:m12",name="analyzer2_th")
# analyzer2_x = EpicsMotorCustom("4dh4:m9",name="analyzer2_x")

class AnalyzerCurvature(PseudoPositioner):
    def __init__(self,
                 analyzer_motor_pv: str,
                 *args,
                 **kwargs
                 ):
        self.analyzer_motor_pv = analyzer_motor_pv
        super().__init__(*args, **kwargs)
        
    curve = Cpt(PseudoSingle, limits=(0,0.7), egu='1/m')

    counts = FCpt(EpicsMotorCustom, "{analyzer_motor_pv}", name='counts')

    @pseudo_position_argument
    def forward(self, pseudo_pos):
        return self.RealPosition(counts=pseudo_pos.curve/0.0005516111545194904)
    
    @real_position_argument
    def inverse(self, real_pos):
        return self.PseudoPosition(curve=real_pos.counts*0.0005516111545194904)
    

analyzer1 = AnalyzerCurvature("4dh4:m11",name="analyzer1")
analyzer2 = AnalyzerCurvature("4dh4:m15",name="analyzer2")

sd.baseline.append(stage1.theta)
sd.baseline.append(stage2.theta)
sd.baseline.append(stage2.x)
sd.baseline.append(pinhole.y)
# sd.baseline.append(pinhole_y)

sd.baseline.append(sample_th)
# sd.baseline.append(sample_x)
sd.baseline.append(det_psd_x)
# sd.baseline.append(analyzer1_th)
# sd.baseline.append(analyzer1_x)
sd.baseline.append(analyzer1.curve)
# sd.baseline.append(analyzer2_th)
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
        
#     curve = Cpt(PseudoSingle, limits=(0,0.7), egu='1/m')
#     counts = FCpt(EpicsMotor, "{prefix}{_curve_motor_pv}", name='counts')

#     x = FCpt(EpicsMotor, "{prefix}{_x_motor_pv}", name='x')
#     th = FCpt(EpicsMotor, "{prefix}{_th_motor_pv}", name='th')

#     @pseudo_position_argument
#     def forward(self, pseudo_pos):
#         return self.RealPosition(counts=pseudo_pos.curve/0.0005516111545194904)
    
#     @real_position_argument
#     def inverse(self, real_pos):
#         return self.PseudoPosition(curve=real_pos.counts*0.0005516111545194904)
    
# analyzer1 = AnalyzerCurvature("4dh4:",curve_motor_pv="m11",x_motor_pv="m16",th_motor_pv="m13",name="analyzer1")