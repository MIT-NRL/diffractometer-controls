"""Device definitions only; importing this module does not construct EPICS devices."""
import numpy as np

from bluesky import SupplementalData, RunEngine

from ophyd import (Device, Component as Cpt, FormattedComponent as FCpt,
                   EpicsSignal, EpicsSignalRO, EpicsSignalWithRBV,
                   EpicsMotor, Signal)

from ophyd.device import DeviceStatus

from ophyd.status import Status, SubscriptionStatus

from ophyd.pseudopos import (PseudoPositioner, PseudoSingle, real_position_argument, pseudo_position_argument)

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

class Pinhole(Device):
    y = Cpt(EpicsMotorCustom, "m16", name="y", labels=["positioner"])

class Stage1(Device):
    theta = Cpt(EpicsMotorCustom, "m3", name="theta", labels=["positioner"])

class Stage2(Device):
    theta = Cpt(EpicsMotorCustom, "m13", name="theta", labels=["positioner"])
    x = Cpt(EpicsMotorCustom, "m14", name="phi", labels=["positioner"])

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

__all__ = ['AnalyzerCurvature', 'Cpt', 'Device', 'DeviceStatus', 'EpicsMotor', 'EpicsMotorCustom', 'EpicsSignal', 'EpicsSignalRO', 'EpicsSignalWithRBV', 'FCpt', 'Pinhole', 'PseudoPositioner', 'PseudoSingle', 'RunEngine', 'Signal', 'Stage1', 'Stage2', 'Status', 'SubscriptionStatus', 'SupplementalData', 'np', 'pseudo_position_argument', 'real_position_argument']
