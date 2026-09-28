"""Shared helpers for the custom USB-CTR08 PyDM displays."""
from __future__ import annotations

from collections.abc import Iterable

from pydm.utilities.macro import parse_macro_string
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore


def value_slot(method):
    """Accept the scalar and array value types emitted by PyDM plugins."""
    for value_type in (int, float, str, bool, object):
        method = QtCore.Slot(value_type)(method)
    return method


def macro_dict(macros) -> dict[str, str]:
    """Normalize a PyDM macro mapping or macro string."""
    if isinstance(macros, dict):
        return {str(key): str(value) for key, value in macros.items()}
    if isinstance(macros, str) and macros.strip():
        try:
            parsed = parse_macro_string(macros)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return {str(key): str(value) for key, value in parsed.items()}
        result = {}
        for item in macros.split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                result[key.strip()] = value.strip()
        return result
    return {}


def display_macros(display) -> dict[str, str]:
    macros = display.macros() if callable(display.macros) else display.macros
    return macro_dict(macros)


def ca_address(pv: str) -> str:
    return pv if "://" in pv else f"ca://{pv}"


def channel_waveform_suffix(channel_index: int) -> str:
    """Map physical CTR0..CTR7 to the EPICS mca1..mca8 waveform."""
    index = int(channel_index)
    if not 0 <= index <= 7:
        raise ValueError("USB-CTR08 channel must be between 0 and 7")
    return f"mca{index + 1}"


def scaler_count_field(channel_index: int) -> str:
    """Map physical CTR0..CTR7 to scaler fields S1..S8."""
    index = int(channel_index)
    if not 0 <= index <= 7:
        raise ValueError("USB-CTR08 channel must be between 0 and 7")
    return f"S{index + 1}"


def count_rate(counts: float, seconds: float) -> float:
    try:
        elapsed = float(seconds)
        value = float(counts)
    except (TypeError, ValueError):
        return 0.0
    if elapsed <= 0.0:
        return 0.0
    return max(0.0, value / elapsed)


def mcs_average_rate(
    samples: Iterable[float], dwell: float, bins: int = 1
) -> float:
    """Calculate the average rate over the newest completed MCS bins."""
    try:
        values = [max(0.0, float(value)) for value in samples]
        width = max(1, int(bins))
        actual_dwell = float(dwell)
    except (TypeError, ValueError):
        return 0.0
    if not values or actual_dwell <= 0.0:
        return 0.0
    selected = values[-width:]
    return sum(selected) / (len(selected) * actual_dwell)


class IntPVWriter(QtCore.QObject):
    value = QtCore.Signal(int)

    def __init__(self, pv: str, parent=None):
        super().__init__(parent)
        self.channel = PyDMChannel(
            address=ca_address(pv), value_signal=self.value
        )
        self.channel.connect()

    def close(self) -> None:
        self.channel.disconnect()


class FloatPVWriter(QtCore.QObject):
    value = QtCore.Signal(float)

    def __init__(self, pv: str, parent=None):
        super().__init__(parent)
        self.channel = PyDMChannel(
            address=ca_address(pv), value_signal=self.value
        )
        self.channel.connect()

    def close(self) -> None:
        self.channel.disconnect()
