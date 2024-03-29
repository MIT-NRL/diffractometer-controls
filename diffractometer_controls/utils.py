from typhos.plugins import register_signal#, SignalConnection, SignalPlugin
from ophyd import (Device, Component as Cpt,
                    EpicsSignal, EpicsSignalRO, EpicsMotor)


def register_all_signals(signal: Device) -> None:
    for attrs in signal.component_names:
        register_signal(getattr(signal,attrs))