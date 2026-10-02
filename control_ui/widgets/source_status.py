"""A source-status readout that hosts may connect to any data source."""

import re
from qtpy import QtWidgets


def numeric_value(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value).strip())
    return float(match.group()) if match else None


def value_style(value, *, low=1, high=5):
    value = numeric_value(value)
    color = "#334155" if value is None else (
        "#b91c1c" if value < low else "#15803d" if value > high else "#a16207"
    )
    return f"font-weight: 700; color: {color}; background: transparent; border: none;"


class SourceStatusIndicator(QtWidgets.QLabel):
    def __init__(self, parent=None, *, units="", low=1, high=5):
        super().__init__(parent)
        self.units, self.low, self.high = units, low, high
        self.set_value(None)

    def set_value(self, value):
        number = numeric_value(value)
        self.setText("Disconnected" if number is None else f"{number:g} {self.units}".strip())
        self.setStyleSheet(value_style(number, low=self.low, high=self.high))
