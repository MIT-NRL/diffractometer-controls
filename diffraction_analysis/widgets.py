"""Shared plot gestures and status icons, independent of instrument controls."""

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets


def reset_data_range(plot, curves):
    """Use full input arrays, independent of any displayed/clipped subset."""
    ranges = []
    for curve in curves:
        x, y = np.asarray(curve['x']), np.asarray(curve['y'])
        valid = np.isfinite(x) & np.isfinite(y)
        if np.any(valid):
            ranges.append((x[valid].min(), x[valid].max(), y[valid].min(), y[valid].max()))
    if ranges:
        bounds = np.array(ranges)
        plot.setRange(xRange=(bounds[:, 0].min(), bounds[:, 1].max()),
                      yRange=(bounds[:, 2].min(), bounds[:, 3].max()),
                      padding=0.03, disableAutoRange=True)


class ResettablePlotWidget(pg.PlotWidget):
    """A left double-click restores axes without editing fit settings."""

    def __init__(self, *args, reset_axes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_axes = reset_axes or self.getPlotItem().autoRange
        self.setToolTip('Double-click to reset axis limits.')
        self._message = QtWidgets.QLabel(self.viewport())
        self._message.setTextFormat(QtCore.Qt.PlainText)
        self._message.setWordWrap(True)
        self._message.setAlignment(QtCore.Qt.AlignCenter)
        self._message.setMargin(40)
        self._message.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._message.hide()

    def set_message(self, text):
        self._message.setText(text)
        self._message.setGeometry(self.viewport().rect())
        self._message.setVisible(bool(text))
        if text:
            self._message.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_message'):
            self._message.setGeometry(self.viewport().rect())

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._reset_axes()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


def run_status_icon(status, *, unavailable=False):
    """Draw symbols locally so they also work with limited X11 fonts."""
    status = str(status).lower()
    if unavailable or status in {'failed', 'aborted'}:
        color, symbol = '#bf3030', 'cross'
    elif status == 'complete':
        color, symbol = '#248543', 'check'
    elif status in {'partial', 'stopped'}:
        color, symbol = '#a86b05', 'bar'
    else:
        color, symbol = '#6c7782', 'bar'
    pixmap = QtGui.QPixmap(18, 18)
    pixmap.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawEllipse(1, 1, 16, 16)
    painter.setPen(QtGui.QPen(QtGui.QColor('white'), 2))
    if symbol == 'check':
        painter.drawLine(5, 9, 8, 12)
        painter.drawLine(8, 12, 13, 6)
    elif symbol == 'cross':
        painter.drawLine(6, 6, 12, 12)
        painter.drawLine(6, 12, 12, 6)
    else:
        painter.drawLine(5, 9, 13, 9)
    painter.end()
    return QtGui.QIcon(pixmap)
