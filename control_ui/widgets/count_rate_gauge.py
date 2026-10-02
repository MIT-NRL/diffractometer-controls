"""Reusable analog count-rate gauge for PyDM and other Qt displays."""

from __future__ import annotations

import math

from qtpy import QtCore, QtGui, QtWidgets


def engineering_rate_text(rate: float) -> str:
    """Format a non-negative count rate with engineering prefixes."""
    rate = max(0.0, float(rate))
    if rate >= 1.0e9:
        return f"{rate / 1.0e9:.2f} Gcps"
    if rate >= 1.0e6:
        return f"{rate / 1.0e6:.2f} Mcps"
    if rate >= 1.0e3:
        return f"{rate / 1.0e3:.2f} kcps"
    return f"{rate:.1f} cps"


def decade_scale(rate: float, current: float | None = None) -> float:
    """Return a 0--10 dial multiplier with hysteresis around decades."""
    rate = max(0.0, float(rate))
    if current is None or current <= 0:
        if rate <= 10.0:
            return 1.0
        return 10.0 ** math.floor(math.log10(rate))

    scale = max(1.0, float(current))
    while rate > 10.5 * scale:
        scale *= 10.0
    while scale > 1.0 and rate < 0.8 * scale:
        scale /= 10.0
    return scale


class CountRateGauge(QtWidgets.QWidget):
    """Animated, auto-ranging count-rate gauge.

    The widget has no EPICS dependency. Call :meth:`set_rate` from any data
    source, which makes it reusable in PyDM, standalone Qt, and test displays.
    """

    def __init__(self, parent=None, title="COUNT RATE"):
        super().__init__(parent)
        self.setMinimumSize(420, 330)
        self._needle_value = 0.0
        self._rate_cps = 0.0
        self._scale = 1.0
        self._title = str(title)
        self._connected = True
        self._compact = False

        self._animation = QtCore.QPropertyAnimation(self, b"needleValue", self)
        self._animation.setDuration(350)
        self._animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)

    def get_needle_value(self) -> float:
        return self._needle_value

    def set_needle_value(self, value: float) -> None:
        self._needle_value = float(value)
        self.update()

    needleValue = QtCore.Property(
        float, get_needle_value, set_needle_value
    )

    @property
    def rate(self) -> float:
        return self._rate_cps

    @property
    def scale(self) -> float:
        return self._scale

    def set_title(self, title: str) -> None:
        self._title = str(title)
        self.update()

    def set_connected(self, connected: bool) -> None:
        self._connected = bool(connected)
        self.update()

    def set_compact(self, compact: bool) -> None:
        """Use a needle-focused layout for secondary, smaller gauges."""
        self._compact = bool(compact)
        self.update()

    def set_rate(self, rate_cps: float, *, animate: bool = True) -> None:
        rate = max(0.0, float(rate_cps))
        new_scale = decade_scale(rate, self._scale)
        if new_scale != self._scale:
            old_visual_rate = self._needle_value * self._scale
            self._scale = new_scale
            self._needle_value = min(old_visual_rate / self._scale, 10.0)

        self._rate_cps = rate
        target = min(rate / self._scale, 10.0)
        self._animation.stop()
        if animate and self.isVisible():
            self._animation.setStartValue(self._needle_value)
            self._animation.setEndValue(target)
            self._animation.start()
        else:
            self.set_needle_value(target)
        self.update()

    def multiplier_text(self) -> str:
        if self._scale >= 1.0e9:
            return f"× {self._scale / 1.0e9:g}G"
        if self._scale >= 1.0e6:
            return f"× {self._scale / 1.0e6:g}M"
        if self._scale >= 1.0e3:
            return f"× {self._scale / 1.0e3:g}k"
        return f"× {self._scale:g}"

    def _draw_multiplier_badge(self, painter, top, font):
        text = self.multiplier_text()
        metrics = QtGui.QFontMetricsF(font)
        badge = QtCore.QRectF(
            0, top, metrics.horizontalAdvance(text) + 24.0, metrics.height() + 8.0
        )
        badge.moveLeft((self.width() - badge.width()) / 2.0)
        palette = self.palette()
        dark = palette.color(QtGui.QPalette.Window).lightness() < 128
        background = QtGui.QColor("#555555" if dark else "#dddddd")
        if self._connected:
            foreground = QtGui.QColor("#f0f0f0" if dark else "#202020")
        else:
            foreground = palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(badge, 5.0, 5.0)
        painter.setPen(foreground)
        painter.setFont(font)
        painter.drawText(badge, QtCore.Qt.AlignCenter, text)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        palette = self.palette()
        foreground = palette.color(QtGui.QPalette.WindowText)
        muted = palette.color(QtGui.QPalette.PlaceholderText)
        if not self._connected:
            foreground = muted

        width, height = self.width(), self.height()
        multiplier_font = QtGui.QFont(self.font())
        multiplier_font.setPointSize(11 if self._compact else 13)
        multiplier_font.setBold(True)
        badge_height = QtGui.QFontMetricsF(multiplier_font).height() + 8.0
        center_x = width / 2.0
        if self._compact:
            # The open-bottom arc occupies 1 + sqrt(1/2) radii in height.
            # Fit the dial to that footprint, reserving space for its scale.
            radius = max(1.0, min(
                (width - 12.0) / 2.0, (height - badge_height - 12.0) / 1.7072
            ))
            center_y = radius + 6.0
        else:
            center_y = height * 0.52
            radius = min(width * 0.40, height * 0.40)
        start_angle = 225.0
        sweep = 270.0

        painter.setPen(QtGui.QPen(foreground, 5))
        dial_rect = QtCore.QRectF(
            center_x - radius,
            center_y - radius,
            2.0 * radius,
            2.0 * radius,
        )
        painter.drawArc(dial_rect, int(-45 * 16), int(270 * 16))

        for index in range(21):
            value = index / 2.0
            angle = math.radians(start_angle - (value / 10.0) * sweep)
            major = index % 2 == 0
            outer = radius * 0.98
            inner = radius * (0.80 if major else 0.88)
            x1 = center_x + inner * math.cos(angle)
            y1 = center_y - inner * math.sin(angle)
            x2 = center_x + outer * math.cos(angle)
            y2 = center_y - outer * math.sin(angle)
            painter.setPen(QtGui.QPen(foreground, 2 if major else 1))
            painter.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

        painter.setPen(foreground)
        painter.setFont(QtGui.QFont("Sans Serif", 8 if self._compact else 10))
        for value in range(0, 11, 2):
            angle = math.radians(start_angle - (value / 10.0) * sweep)
            label_radius = radius * 0.68
            x = center_x + label_radius * math.cos(angle)
            y = center_y - label_radius * math.sin(angle)
            painter.drawText(
                QtCore.QRectF(x - 18, y - 12, 36, 24),
                QtCore.Qt.AlignCenter,
                str(value),
            )

        angle = math.radians(
            start_angle - (self._needle_value / 10.0) * sweep
        )
        needle_length = radius * 0.77
        needle_end = QtCore.QPointF(
            center_x + needle_length * math.cos(angle),
            center_y - needle_length * math.sin(angle),
        )
        painter.setPen(QtGui.QPen(foreground, 4))
        painter.drawLine(QtCore.QPointF(center_x, center_y), needle_end)
        painter.setBrush(foreground)
        painter.drawEllipse(QtCore.QPointF(center_x, center_y), 7, 7)

        painter.setFont(QtGui.QFont("Sans Serif", 10 if self._compact else 13))
        painter.drawText(
            QtCore.QRectF(0, center_y - radius * 0.45, width, 30),
            QtCore.Qt.AlignCenter,
            self._title,
        )
        if self._compact:
            self._draw_multiplier_badge(
                painter, center_y + radius * 0.7072 + 3.0, multiplier_font
            )
            return

        painter.setFont(QtGui.QFont("Sans Serif", 19, QtGui.QFont.Bold))
        rate_text = engineering_rate_text(self._rate_cps)
        if not self._connected:
            rate_text = "DISCONNECTED"
        painter.drawText(
            QtCore.QRectF(0, center_y + radius * 0.40, width, 35),
            QtCore.Qt.AlignCenter,
            rate_text,
        )
        self._draw_multiplier_badge(
            painter,
            max(center_y + radius * 0.60, center_y + radius * 0.40 + 38.0),
            multiplier_font,
        )
