"""Fit-parameter trends and per-point batch outcomes."""

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .batch import METRICS, coordinate_fields, parameter_trend
from .curves import data_keys
from .widgets import ResettablePlotWidget
from .colors import series_color


class FitTrends(QtWidgets.QWidget):
    point_selected = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = None
        self.records = []
        self.field = ''
        self.metric_combo = QtWidgets.QComboBox(self)
        self.metric_combo.addItems(METRICS)
        self.coordinate_combo = QtWidgets.QComboBox(self)
        self.peak_spin = QtWidgets.QSpinBox(self)
        self.peak_spin.setRange(1, 6)
        self.peak_spin.setMaximumWidth(80)
        self.summary = QtWidgets.QLabel(self)
        self.summary.setWordWrap(True)
        self.plot = ResettablePlotWidget(self)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setMinimumHeight(120)
        self.line = self.plot.plot(pen=pg.mkPen('#3b82c4', width=1), symbol='o', symbolSize=6, symbolBrush='#3b82c4', connect='finite')
        self.errors = pg.ErrorBarItem(beam=0, pen=pg.mkPen('#3b82c4'))
        self.plot.addItem(self.errors)
        self.table = QtWidgets.QTableWidget(0, 8, self)
        self.table.setHorizontalHeaderLabels(['Point', 'Coordinate', 'Status', 'Position', 'Height', 'Area', 'FWHM', 'Sigma'])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._select_row)
        row = QtWidgets.QHBoxLayout()
        for text, widget in [('Parameter', self.metric_combo), ('Peak', self.peak_spin), ('Versus', self.coordinate_combo)]:
            row.addWidget(QtWidgets.QLabel(text, self))
            row.addWidget(widget, 0 if widget is self.peak_spin else 1)
        graph = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(graph)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(graph)
        self.metric_combo.currentIndexChanged.connect(self._draw)
        self.coordinate_combo.currentIndexChanged.connect(self._draw)
        self.peak_spin.valueChanged.connect(self._draw)
        self.reset()

    def reset(self):
        self.snapshot = None
        self.records = []
        self.field = ''
        self.line.clear()
        self.errors.setData(x=np.array([]), y=np.array([]), height=None, top=np.array([]), bottom=np.array([]))
        self.table.setRowCount(0)
        self.coordinate_combo.clear()
        self.summary.setText('Fit one or all acquisition spectra to show parameter trends. Double-click a result row to inspect that fit.')

    def set_records(self, snapshot, records, peaks=1, *, field=''):
        self.snapshot = snapshot
        self.field = field
        self.records = sorted(records, key=lambda record: record['point'] if record['point'] is not None else -1)
        self.peak_spin.setMaximum(peaks)
        previous = self.coordinate_combo.currentData()
        with QtCore.QSignalBlocker(self.coordinate_combo):
            self.coordinate_combo.clear()
            self.coordinate_combo.addItem('Point number', '')
            fields = coordinate_fields(snapshot)
            for field in fields:
                self.coordinate_combo.addItem(field, field)
            motors = snapshot.get('start', {}).get('motors', [])
            target = previous if previous in fields else next((motor for motor in motors if motor in fields), '')
            self.coordinate_combo.setCurrentIndex(max(0, self.coordinate_combo.findData(target)))
        self._draw()

    def _draw(self, *_args):
        if self.snapshot is None:
            return
        metric = self.metric_combo.currentText()
        coordinate = self.coordinate_combo.currentData() or ''
        peak = self.peak_spin.value()
        x, y, error = parameter_trend(self.records, metric, peak, coordinate)
        if self.records and all(record['point'] is None for record in self.records):
            x[:] = np.nan
        self.line.setData(x, y, connect='finite')
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(error)
        self.errors.setData(x=x[valid], y=y[valid], height=None, top=error[valid], bottom=error[valid])
        keys = data_keys(self.snapshot)
        x_unit = keys.get(coordinate, {}).get('units', '')
        successes = sum(record['fit'] is not None for record in self.records)
        self.plot.setLabel('bottom', coordinate or 'Point number', units=x_unit or None)
        units = next((record.get('units', {}) for record in self.records if record['fit'] is not None), {})
        unit = units.get('intensity', '') if metric == 'Peak height' else units.get('coordinate', '')
        if metric == 'Peak area':
            unit = ' · '.join(value for value in (units.get('intensity', ''), units.get('coordinate', '')) if value)
        self.plot.setLabel('left', metric, units=unit or None)
        self.plot.setTitle(f'Peak {peak}: {metric}')
        scalar = self.records and all(record['point'] is None for record in self.records)
        self.summary.setText('Scalar trace: use the single-fit parameters; per-frame trends apply to 1D scans.' if scalar else
                             f'{successes} successful / {len(self.records)} attempted · failed fits and missing coordinates remain gaps; missing errors are not shown as zero.')
        self.table.setRowCount(len(self.records))
        self.table.horizontalHeaderItem(1).setText(coordinate or 'Point number')
        for index, record in enumerate(self.records):
            fit = record['fit']
            values = [str(record['sequence']) if record['point'] is not None else 'Scalar trace',
                      'Unavailable' if not np.isfinite(x[index]) else f'{x[index]:.8g}',
                      'Success' if fit is not None else record['error']]
            for suffix in ('center', 'height', 'amplitude', 'fwhm', 'sigma'):
                parameter = fit['parameters'].get(f'p{peak}_{suffix}') if fit is not None else None
                values.append(f"{parameter['value']:.8g}" if parameter else '')
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if column == 2:
                    item.setToolTip(value)
                self.table.setItem(index, column, item)
        self._apply_palette()

    def _select_row(self, row, _column):
        if row < len(self.records) and self.records[row]['point'] is not None:
            self.point_selected.emit(self.records[row]['point'])

    def _apply_palette(self):
        palette = self.palette()
        text = palette.color(palette.Text)
        self.plot.setBackground(palette.color(palette.Base))
        for name in ('left', 'bottom'):
            axis = self.plot.getAxis(name)
            axis.setPen(text)
            axis.setTextPen(text)
            axis.setLabel(axis.labelText, units=axis.labelUnits, color=text.name())
        title = self.plot.getPlotItem().titleLabel
        title.setText(title.text, color=text.name())
        if self.snapshot is not None:
            color = series_color(self.snapshot, self.field)
            self.line.setPen(pg.mkPen(color, width=1))
            self.line.setSymbolBrush(pg.mkBrush(color))
            self.line.setSymbolPen(pg.mkPen(color))
            self.errors.setData(pen=pg.mkPen(color))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange) and hasattr(self, 'plot'):
            self._apply_palette()
