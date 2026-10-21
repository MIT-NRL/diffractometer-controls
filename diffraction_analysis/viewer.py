"""Run-data display: overlay 1D spectra or browse acquisition positions."""

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .curves import axis_fields, default_axis, numeric, select_curve, signal_fields, signal_label
from .batch import coordinate_fields
from .widgets import ResettablePlotWidget, reset_data_range
from .colors import curve_color


class RunDataPanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = None
        self.curves = []
        self._curve_items = []
        self._updating = False
        self.signal_combo = QtWidgets.QComboBox(self)
        self.axis_combo = QtWidgets.QComboBox(self)
        self.show_all = QtWidgets.QCheckBox('Show all spectra', self)
        self.show_all.setChecked(True)
        self.point_combo = QtWidgets.QComboBox(self)
        self.previous_button = QtWidgets.QToolButton(self)
        self.previous_button.setArrowType(QtCore.Qt.LeftArrow)
        self.previous_button.setToolTip('Previous acquisition point')
        self.next_button = QtWidgets.QToolButton(self)
        self.next_button.setArrowType(QtCore.Qt.RightArrow)
        self.next_button.setToolTip('Next acquisition point')
        self.motor_combo = QtWidgets.QComboBox(self)
        self.position_combo = QtWidgets.QComboBox(self)
        self.position_combo.setToolTip('Recorded motor position and acquisition point')
        self.summary = QtWidgets.QLabel(self)
        self.summary.setTextFormat(QtCore.Qt.PlainText)
        self.summary.setWordWrap(True)
        selection = QtWidgets.QHBoxLayout()
        selection.addWidget(QtWidgets.QLabel('Signal', self))
        selection.addWidget(self.signal_combo, 1)
        selection.addWidget(QtWidgets.QLabel('X coordinate', self))
        selection.addWidget(self.axis_combo, 1)
        selection.addWidget(self.show_all)
        navigation = QtWidgets.QHBoxLayout()
        navigation.addWidget(self.previous_button)
        navigation.addWidget(self.point_combo, 1)
        navigation.addWidget(self.next_button)
        navigation.addWidget(QtWidgets.QLabel('Motor', self))
        navigation.addWidget(self.motor_combo)
        navigation.addWidget(self.position_combo, 1)
        self.plot = ResettablePlotWidget(self, reset_axes=lambda: reset_data_range(self.plot, self.curves))
        self.plot.setMinimumSize(200, 200)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.legend = self.plot.addLegend(offset=(10,10))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(selection)
        layout.addLayout(navigation)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.summary)
        self.signal_combo.currentIndexChanged.connect(self._source_changed)
        self.axis_combo.currentIndexChanged.connect(self._draw)
        self.point_combo.currentIndexChanged.connect(self._point_changed)
        self.point_combo.activated.connect(lambda _index: self.show_all.setChecked(False))
        self.show_all.toggled.connect(self._draw)
        self.previous_button.clicked.connect(lambda: self._step(-1))
        self.next_button.clicked.connect(lambda: self._step(1))
        self.motor_combo.currentIndexChanged.connect(self._positions_changed)
        self.position_combo.currentIndexChanged.connect(self._position_selected)
        self.position_combo.activated.connect(lambda _index: self.show_all.setChecked(False))
        self.reset()

    def reset(self, title='Select a diffraction run'):
        self.plot.set_message('')
        self.snapshot = None
        self.curves = []
        self._curve_items = []
        self._updating = True
        for combo in (self.signal_combo, self.axis_combo, self.point_combo, self.motor_combo, self.position_combo):
            combo.clear()
        self._updating = False
        self.plot.clear()
        self.legend.clear()
        self.plot.setTitle(title)
        self.summary.clear()
        self._navigation_enabled(False)
        self._apply_palette()

    def set_snapshot(self, snapshot):
        self.reset()
        self.snapshot = snapshot
        rows = snapshot.get('rows', [])
        fields = signal_fields(snapshot)
        profiles = [key for key in fields if rows and numeric(rows[0].get(key)) is not None and numeric(rows[0][key]).ndim == 1]
        self._updating = True
        if len(profiles) > 1:
            self.signal_combo.addItem('All profile signals', None)
        for field in fields:
            label = signal_label(snapshot, field)
            self.signal_combo.addItem(label, field)
            self.signal_combo.setItemData(self.signal_combo.count() - 1, field, QtCore.Qt.ToolTipRole)
        for index, _row in enumerate(rows):
            sequences = snapshot.get('seq_nums', [])
            sequence = sequences[index] if index < len(sequences) else index + 1
            self.point_combo.addItem(f'Point {sequence}', index)
        for field in coordinate_fields(snapshot):
            self.motor_combo.addItem(field, field)
        self.show_all.setChecked(True)
        self._updating = False
        self._positions_changed()
        self._source_changed()

    def _fields(self):
        if self.snapshot is None:
            return []
        selected = self.signal_combo.currentData()
        if selected is not None:
            return [selected]
        rows = self.snapshot.get('rows', [])
        return [field for field in signal_fields(self.snapshot)
                if rows and numeric(rows[0].get(field)) is not None and numeric(rows[0][field]).ndim == 1]

    def _source_changed(self, *_args):
        if self._updating or self.snapshot is None:
            return
        fields = self._fields()
        self._updating = True
        self.axis_combo.clear()
        if len(fields) > 1:
            self.axis_combo.addItem('Recorded coordinates / bin index', '')
        elif fields:
            point = self.point_combo.currentData() or 0
            field = fields[0]
            self.axis_combo.addItem('Bin / point number', '')
            for axis in axis_fields(self.snapshot, field, point):
                self.axis_combo.addItem(axis, axis)
            self.axis_combo.setCurrentIndex(max(0, self.axis_combo.findData(default_axis(self.snapshot, field, point))))
        self._updating = False
        self._draw()

    def _positions_changed(self, *_args):
        if self._updating or self.snapshot is None:
            return
        field = self.motor_combo.currentData()
        self._updating = True
        self.position_combo.clear()
        for index, row in enumerate(self.snapshot['rows']):
            value = numeric(row.get(field))
            label = f'{float(value):.8g}' if value is not None and value.ndim == 0 and np.isfinite(value) else 'Unavailable'
            self.position_combo.addItem(f'{label} · {self.point_combo.itemText(index)}', index)
        self.position_combo.setCurrentIndex(max(0, self.point_combo.currentIndex()))
        self._updating = False
        self._draw()

    def _position_selected(self, index):
        if not self._updating and index >= 0:
            self.show_all.setChecked(False)
            self.point_combo.setCurrentIndex(index)

    def _step(self, delta):
        self.show_all.setChecked(False)
        index = self.point_combo.currentIndex() + delta
        if 0 <= index < self.point_combo.count():
            self.point_combo.setCurrentIndex(index)

    def _point_changed(self, *_args):
        if not self._updating:
            with QtCore.QSignalBlocker(self.position_combo):
                self.position_combo.setCurrentIndex(self.point_combo.currentIndex())
            self._draw()

    def _navigation_enabled(self, profile):
        index = self.point_combo.currentIndex()
        self.previous_button.setEnabled(profile and index > 0)
        self.next_button.setEnabled(profile and index < self.point_combo.count() - 1)
        self.point_combo.setEnabled(profile)
        self.position_combo.setEnabled(profile and self.motor_combo.count() > 0)
        self.motor_combo.setEnabled(profile and self.motor_combo.count() > 0)
        self.show_all.setEnabled(profile)

    def _draw(self, *_args):
        if self._updating or self.snapshot is None:
            return
        self.plot.clear()
        self.legend.clear()
        self.curves = []
        self._curve_items = []
        rows = self.snapshot['rows']
        fields = self._fields()
        if not rows or not fields:
            return
        value = numeric(rows[0].get(fields[0]))
        profile = value is not None and value.ndim == 1
        self._navigation_enabled(profile)
        points = range(len(rows)) if profile and self.show_all.isChecked() else [self.point_combo.currentData() or 0]
        errors = []
        for field in fields:
            for point in points:
                axis = self.axis_combo.currentData() or (default_axis(self.snapshot, field, point) if len(fields) > 1 else '')
                try:
                    curve = select_curve(self.snapshot, field, point=point, x_field=axis)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                self.curves.append(curve)
                item = self.plot.plot(curve['x'], curve['y'], connect='finite')
                self._curve_items.append((item, curve))
                monotonic = np.all(np.isfinite(curve['x'])) and np.all(np.diff(curve['x']) > 0)
                item.setClipToView(bool(monotonic))
                item.setDownsampling(auto=bool(monotonic), method='peak')
                if len(fields) > 1 and point == points[-1]:
                    self.legend.addItem(item, field)
        if self.curves:
            curve = self.curves[0]
            self.plot.setLabel('bottom', curve['x_label'], units=curve['x_units'] or None)
            self.plot.setLabel('left', 'Intensity' if len(fields) > 1 else curve['y_label'], units=curve['y_units'] or None)
            title = self.snapshot.get('start', {}).get('title') or 'Diffraction run'
            self.plot.setTitle(title)
            selection = ('All acquisition spectra' if self.show_all.isChecked() else self.point_combo.currentText()) if profile else f'Scalar trace · {len(rows)} acquisition points'
            field = self.motor_combo.currentData()
            point = self.point_combo.currentData() or 0
            coordinate = rows[point].get(field)
            motor = f' · {field} = {coordinate}' if field and profile and not self.show_all.isChecked() else ''
            self.summary.setText(f'{selection} · {len(self.curves)} curves{motor}' + (f' · {len(errors)} unavailable' if errors else ''))
        elif errors:
            self.summary.setText(errors[0])
        self._apply_palette()

    def fitting_selection(self):
        fields = self._fields()
        if not fields or self.snapshot is None:
            return None
        point = self.point_combo.currentData() or 0
        axis = self.axis_combo.currentData() or (default_axis(self.snapshot, fields[0], point) if len(fields) > 1 else '')
        return fields[0], point, axis

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
        self.legend.setLabelTextColor(text.name())
        for _sample, label in self.legend.items:
            label.setText(label.text, color=text.name())
        for item, curve in self._curve_items:
            color = curve_color(self.snapshot, curve['field'], curve['point'], palette)
            newest = curve['point'] is None or curve['point'] == len(self.snapshot['rows']) - 1
            item.setPen(pg.mkPen(color, width=1.5 if newest else 1))
            item.setZValue(4 if newest else 2)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange) and hasattr(self, 'plot'):
            self._apply_palette()
