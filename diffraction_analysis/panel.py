"""Gauge-free scalar/1D plotting, explicit fit controls, and fit results."""

import json
from dataclasses import asdict

import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from .curves import axis_fields, default_axis, numeric, select_curve, signal_fields, signal_label
from .fitting import BACKGROUNDS, PEAK_MODELS, FitConfig
from .batch import coordinate_fields, export_batch
from .trends import FitTrends
from .widgets import ResettablePlotWidget, reset_data_range
from .colors import curve_color


class AnalysisPanel(QtWidgets.QWidget):
    fit_requested = QtCore.Signal(object, object)
    selection_changed = QtCore.Signal()
    context_changed = QtCore.Signal()
    batch_requested = QtCore.Signal(object, object)
    cancel_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.snapshot = None
        self.curve = None
        self.fit_result = None
        self.batch = None
        self._fit_context = None
        self._batch_busy = False
        self._updating = False
        self.signal_combo = QtWidgets.QComboBox(self)
        self.point_combo = QtWidgets.QComboBox(self)
        self.axis_combo = QtWidgets.QComboBox(self)
        self.peak_combo = QtWidgets.QComboBox(self)
        self.peak_combo.addItems(PEAK_MODELS)
        self.background_combo = QtWidgets.QComboBox(self)
        self.background_combo.addItems(BACKGROUNDS)
        self.background_combo.setCurrentText('Constant')
        self.peaks_spin = QtWidgets.QSpinBox(self)
        self.peaks_spin.setRange(1, 6)
        self.centers_edit = QtWidgets.QLineEdit(self)
        self.centers_edit.setPlaceholderText('Optional centers, comma separated')
        self.centers_edit.setToolTip('Supply one starting center per peak, inside the fitting window. Leave blank for an estimate.')
        self.weight_combo = QtWidgets.QComboBox(self)
        self.weight_combo.addItems(['Equal', 'Poisson (counts)'])
        self.weight_combo.setToolTip('Poisson weighting is intended for nonnegative raw counts; use equal weights for other scalar measurements.')
        self.min_spin = QtWidgets.QDoubleSpinBox(self)
        self.max_spin = QtWidgets.QDoubleSpinBox(self)
        for spin in (self.min_spin, self.max_spin):
            spin.setDecimals(8)
            spin.setRange(-1e15, 1e15)
            spin.setKeyboardTracking(False)
        self.full_range_button = QtWidgets.QPushButton('Full range', self)
        self.full_range_button.setToolTip('Use the entire spectrum as the fitting window. Clears stored fits.')
        self.fit_button = QtWidgets.QPushButton('Fit selected curve', self)
        self.fit_button.setDefault(True)
        self.clear_button = QtWidgets.QPushButton('Clear all fits', self)
        self.batch_button = QtWidgets.QPushButton('Fit all points', self)
        self.stop_button = QtWidgets.QPushButton('Stop', self)
        self.stop_button.setEnabled(False)
        self.progress = QtWidgets.QProgressBar(self)
        self.progress.hide()
        self.copy_button = QtWidgets.QPushButton('Copy fit results', self)
        self.copy_button.setEnabled(False)
        self.message = QtWidgets.QLabel('Select a run and signal. Fits run only when requested.', self)
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.message.setTextFormat(QtCore.Qt.PlainText)
        controls = QtWidgets.QWidget(self)
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)

        def form(title, rows):
            box = QtWidgets.QGroupBox(title, controls)
            layout = QtWidgets.QFormLayout(box)
            layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
            layout.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
            for label, widget in rows:
                layout.addRow(label, widget)
            controls_layout.addWidget(box)

        form('Data selection', [('Signal', self.signal_combo), ('Acquisition point', self.point_combo), ('X coordinate', self.axis_combo)])
        form('Peak model', [('Profile', self.peak_combo), ('Number of peaks', self.peaks_spin), ('Background', self.background_combo),
                            ('Starting centers', self.centers_edit), ('Weighting', self.weight_combo)])
        form('Fit window', [('Minimum', self.min_spin), ('Maximum', self.max_spin), ('', self.full_range_button)])
        controls_layout.addStretch(1)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(controls)
        right = QtWidgets.QWidget(self)
        right.setMinimumWidth(280)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 0, 4, 4)
        right_layout.addWidget(scroll, 1)
        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self.fit_button, 1)
        actions.addWidget(self.clear_button)
        right_layout.addLayout(actions)
        batch_actions = QtWidgets.QHBoxLayout()
        batch_actions.addWidget(self.batch_button, 1)
        batch_actions.addWidget(self.stop_button)
        right_layout.addLayout(batch_actions)
        right_layout.addWidget(self.progress)
        right_layout.addWidget(self.copy_button)
        right_layout.addWidget(self.message)

        self.plot = ResettablePlotWidget(self, reset_axes=self.reset_view)
        self.residual_plot = ResettablePlotWidget(self, reset_axes=self.reset_view)
        self.plot.setMinimumSize(200, 200)
        self.residual_plot.setMinimumHeight(100)
        self.residual_plot.setXLink(self.plot)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.residual_plot.showGrid(x=True, y=True, alpha=0.15)
        self.residual_plot.setLabel('left', 'Residual')
        self.residual_plot.addLine(y=0, pen=pg.mkPen('#888888', style=QtCore.Qt.DashLine))
        self.data_line = self.plot.plot(pen=pg.mkPen('#3b82c4', width=1), symbol='o', symbolSize=4, symbolBrush='#3b82c4')
        self.fit_line = self.plot.plot(pen=pg.mkPen('#d65c22', width=2))
        self.background_line = self.plot.plot(pen=pg.mkPen('#6c9d40', width=1.5, style=QtCore.Qt.DashLine))
        self.residual_line = self.residual_plot.plot(pen=None, symbol='o', symbolSize=4, symbolBrush='#3b82c4')
        for item in (self.data_line, self.fit_line, self.background_line, self.residual_line):
            item.curve.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
            item.scatter.setCacheMode(QtWidgets.QGraphicsItem.DeviceCoordinateCache)
        self.region = pg.LinearRegionItem(values=(0, 1), brush=pg.mkBrush(80, 130, 210, 20))
        for line, bound in zip(self.region.lines, ('lower', 'upper')):
            line.setPen(color=line.pen.color(), width=2)
            line.setHoverPen(color=line.hoverPen.color(), width=3)
            # Pixel-sized handles also widen InfiniteLine's hit area along
            # the whole boundary, independently of the plot's zoom level.
            line.addMarker('<|>', position=0.9, size=16)
            line.setCursor(QtCore.Qt.SizeHorCursor)
            line.setToolTip(f'Drag to adjust the {bound} fitting limit.')
        self.region.setZValue(-10)
        self.plot.addItem(self.region, ignoreBounds=True)
        self._window_timer = QtCore.QTimer(self)
        self._window_timer.setSingleShot(True)
        self._window_timer.setInterval(33)
        self._window_timer.timeout.connect(self._flush_region)
        self.region.sigRegionChanged.connect(self._preview_region)
        self.region.sigRegionChangeFinished.connect(self._region_changed)
        self.plot.setTitle('Selected diffraction curve')
        self.plot.legend = self.plot.addLegend(offset=(10, 10))
        self.plot.legend.addItem(self.data_line, 'Data')
        self.plot.legend.addItem(self.fit_line, 'Fit')
        self.plot.legend.addItem(self.background_line, 'Background')
        plots = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        plots.addWidget(self.plot)
        plots.addWidget(self.residual_plot)
        plots.setSizes([420, 120])
        plots.setStretchFactor(0, 3)
        plots.setStretchFactor(1, 1)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.splitter.addWidget(plots)
        self.splitter.addWidget(right)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)
        self.splitter.setSizes([700, 290])
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        self.results = QtWidgets.QTableWidget(0, 3)
        self.results.setHorizontalHeaderLabels(['Parameter', 'Value', 'Std. error'])
        self.results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results.horizontalHeader().setStretchLastSection(True)
        self.report = QtWidgets.QPlainTextEdit()
        self.report.setReadOnly(True)
        self.results_widget = QtWidgets.QTabWidget()
        self.results_widget.addTab(self.results, 'Parameters')
        self.results_widget.addTab(self.report, 'Fit report')
        self.trends = FitTrends(self)
        self.results_widget.addTab(self.trends, 'Parameter trends')
        self.results_widget.addTab(self.trends.table, 'All point results')
        self.trends.point_selected.connect(self.point_combo.setCurrentIndex)
        self.signal_combo.currentIndexChanged.connect(self._source_changed)
        self.point_combo.currentIndexChanged.connect(self._point_changed)
        self.axis_combo.currentIndexChanged.connect(self._draw_curve)
        self.fit_button.clicked.connect(self.request_fit)
        self.batch_button.clicked.connect(self.request_batch)
        self.stop_button.clicked.connect(self.cancel_requested.emit)
        self.clear_button.clicked.connect(self.clear_fit)
        self.full_range_button.clicked.connect(self._full_range)
        self.copy_button.clicked.connect(self._copy_results)
        for combo in (self.peak_combo, self.background_combo, self.weight_combo):
            combo.currentIndexChanged.connect(self._settings_changed)
        self.peaks_spin.valueChanged.connect(self._settings_changed)
        self.centers_edit.textChanged.connect(self._settings_changed)
        self.min_spin.valueChanged.connect(self._window_changed)
        self.max_spin.valueChanged.connect(self._window_changed)
        self.reset()
        self._apply_palette()

    def _apply_palette(self):
        palette = self.palette()
        text = palette.color(palette.Text)
        for widget in (self.plot, self.residual_plot):
            widget.setBackground(palette.color(palette.Base))
            for name in ('left', 'bottom'):
                axis = widget.getAxis(name)
                axis.setPen(text)
                axis.setTextPen(text)
                axis.setLabel(axis.labelText, units=axis.labelUnits, color=text.name())
        title = self.plot.getPlotItem().titleLabel
        title.setText(title.text, color=text.name())
        self.plot.legend.setLabelTextColor(text.name())
        for _sample, label in self.plot.legend.items:
            label.setText(label.text, color=text.name())
        if self.curve is not None:
            color = curve_color(self.snapshot, self.curve['field'], self.curve['point'], palette)
            self.data_line.setPen(pg.mkPen(color, width=1))
            self.data_line.setSymbolPen(pg.mkPen(color))
            self.data_line.setSymbolBrush(pg.mkBrush(color))
            self.residual_line.setSymbolPen(pg.mkPen(color))
            self.residual_line.setSymbolBrush(pg.mkBrush(color))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange) and hasattr(self, 'plot'):
            self._apply_palette()

    def reset(self, config=None):
        self.plot.set_message('')
        self._window_timer.stop()
        self.snapshot = None
        self.curve = None
        self._fit_context = None
        self._updating = True
        for combo in (self.signal_combo, self.point_combo, self.axis_combo):
            combo.clear()
        self._updating = False
        self.data_line.clear()
        self.clear_fit()
        self.fit_button.setEnabled(False)
        self.batch_button.setEnabled(False)
        self.region.hide()
        self.plot.setTitle((config or {}).get('run_title', 'Select a diffraction run'))

    def set_snapshot(self, snapshot):
        self.reset()
        self.snapshot = snapshot
        self._updating = True
        for field in signal_fields(snapshot):
            label = signal_label(snapshot, field)
            self.signal_combo.addItem(label, field)
            self.signal_combo.setItemData(self.signal_combo.count() - 1, field, QtCore.Qt.ToolTipRole)
        self._updating = False
        self._source_changed()

    def select_selection(self, field, point, axis):
        index = self.signal_combo.findData(field)
        if index >= 0 and index != self.signal_combo.currentIndex():
            self.signal_combo.setCurrentIndex(index)
        if 0 <= point < self.point_combo.count() and point != self.point_combo.currentIndex():
            self.point_combo.setCurrentIndex(point)
        index = self.axis_combo.findData(axis)
        if index >= 0 and index != self.axis_combo.currentIndex():
            self.axis_combo.setCurrentIndex(index)

    def _source_changed(self, *_args):
        if self._updating or self.snapshot is None:
            return
        self._updating = True
        self.point_combo.clear()
        field = self.signal_combo.currentData()
        rows = self.snapshot.get('rows', [])
        value = numeric(rows[0].get(field)) if rows else None
        is_profile = value is not None and value.ndim == 1
        for index, row in enumerate(rows if is_profile else rows[:1]):
            sequence = self.snapshot.get('seq_nums', [])[index] if index < len(self.snapshot.get('seq_nums', [])) else index + 1
            motors = ', '.join(f'{name}={row[name]:g}' for name in self.snapshot.get('start', {}).get('motors', []) if isinstance(row.get(name), (int, float)))
            self.point_combo.addItem(f'Point {sequence}' + (f' · {motors}' if motors else ''), index)
        self.point_combo.setEnabled(is_profile)
        self._updating = False
        self._point_changed()

    def _point_changed(self, *_args):
        if self._updating or self.snapshot is None:
            return
        field = self.signal_combo.currentData()
        if not field:
            self.message.setText('No numeric scalar or 1D signals are available.')
            return
        point = self.point_combo.currentData() or 0
        self._updating = True
        previous = self.axis_combo.currentData()
        self.axis_combo.clear()
        value = numeric(self.snapshot['rows'][point].get(field))
        self.axis_combo.addItem('Bin index' if value is not None and value.ndim == 1 else 'Point number', '')
        for key in axis_fields(self.snapshot, field, point):
            self.axis_combo.addItem(key, key)
        target = previous if previous in axis_fields(self.snapshot, field, point) else default_axis(self.snapshot, field, point)
        self.axis_combo.setCurrentIndex(max(0, self.axis_combo.findData(target)))
        self._updating = False
        self._draw_curve()

    def _draw_curve(self, *_args):
        if self._updating or self.snapshot is None:
            return
        context = (self.signal_combo.currentData(), self.axis_combo.currentData() or '')
        changed = context != self._fit_context
        if changed:
            self.clear_fit()
            self._fit_context = context
        else:
            self._clear_selected_fit()
            self.selection_changed.emit()
        try:
            self.curve = select_curve(self.snapshot, context[0], point=self.point_combo.currentData() or 0, x_field=context[1])
        except ValueError as exc:
            self.curve = None
            self.data_line.clear()
            self.fit_button.setEnabled(False)
            self.batch_button.setEnabled(False)
            self.message.setText(str(exc))
            return
        curve = self.curve
        self.data_line.setData(curve['x'], curve['y'])
        monotonic = np.all(np.isfinite(curve['x'])) and np.all(np.diff(curve['x']) > 0)
        self.data_line.setClipToView(bool(monotonic))
        # Region geometry must never trigger auto-ranging while dragging.
        self.plot.getViewBox().enableAutoRange(x=False, y=False)
        self.reset_view()
        self.plot.setTitle(curve['label'])
        self.plot.setLabel('bottom', curve['x_label'], units=curve['x_units'] or None)
        self.plot.setLabel('left', curve['y_label'], units=curve['y_units'] or None)
        self.residual_plot.setLabel('bottom', curve['x_label'], units=curve['x_units'] or None)
        self.fit_button.setEnabled(not self._batch_busy)
        self.batch_button.setEnabled(not self._batch_busy and curve['point'] is not None)
        if changed:
            self._full_range()
        self._display_selected_record()
        self._apply_palette()

    def reset_view(self):
        """Range from full inputs, never from clip-to-view's displayed subset."""
        if self.curve is None:
            return
        reset_data_range(self.plot, [self.curve])
        residual = self.residual_line.yData
        if residual is not None:
            finite = residual[np.isfinite(residual)]
            if len(finite):
                self.residual_plot.setYRange(float(finite.min()), float(finite.max()), padding=0.03)

    def clear_fit(self):
        self._clear_selected_fit()
        if self.batch is not None:
            self.batch = None
            self.trends.reset()
        self.copy_button.setEnabled(False)
        self.message.setText('No fit applied. Select a model and window, then fit one or all points.')
        self.selection_changed.emit()
        self.context_changed.emit()

    def _clear_selected_fit(self):
        self.fit_result = None
        self.fit_line.clear()
        self.background_line.clear()
        self.residual_line.clear()
        if self.results.rowCount():
            self.results.setRowCount(0)
        if not self.report.document().isEmpty():
            self.report.clear()

    def _settings_changed(self, *_args):
        if not self._updating:
            self.clear_fit()

    def _window_changed(self, *_args):
        if self._updating:
            return
        self._updating = True
        self.region.setRegion((self.min_spin.value(), self.max_spin.value()))
        self._updating = False
        self.clear_fit()

    def _preview_region(self):
        if not self._updating and not self._window_timer.isActive():
            self._window_timer.start()

    def _flush_region(self):
        if self._updating:
            return
        lo, hi = self.region.getRegion()
        with QtCore.QSignalBlocker(self.min_spin), QtCore.QSignalBlocker(self.max_spin):
            self.min_spin.setValue(lo)
            self.max_spin.setValue(hi)

    def _region_changed(self):
        if self._updating:
            return
        self._window_timer.stop()
        self._flush_region()
        self.clear_fit()

    def _full_range(self):
        if self.curve is None:
            return
        finite = self.curve['x'][np.isfinite(self.curve['x'])]
        if not len(finite):
            return
        self._updating = True
        self.min_spin.setValue(float(np.min(finite)))
        self.max_spin.setValue(float(np.max(finite)))
        self.region.setRegion((float(np.min(finite)), float(np.max(finite))))
        self.region.show()
        self._updating = False
        self.clear_fit()

    def _configuration(self):
        centers = tuple(float(value.strip()) for value in self.centers_edit.text().split(',') if value.strip())
        return FitConfig(peak=self.peak_combo.currentText(), background=self.background_combo.currentText(), peaks=self.peaks_spin.value(),
                         xmin=self.min_spin.value(), xmax=self.max_spin.value(), weighting=self.weight_combo.currentText(), centers=centers)

    def request_fit(self):
        if self.curve is None:
            return
        try:
            config = self._configuration()
        except ValueError:
            self.message.setText('Starting centers must be comma-separated numbers.')
            return
        self.message.setText('Fitting selected curve…')
        self.fit_requested.emit(self.curve, config)

    def request_batch(self):
        if self.curve is None or self.curve['point'] is None:
            return
        try:
            config = self._configuration()
        except ValueError:
            self.message.setText('Starting centers must be comma-separated numbers.')
            return
        self.batch_requested.emit(self.curve, config)

    def set_batch_busy(self, busy, done=0, total=0):
        self._batch_busy = bool(busy)
        self.stop_button.setEnabled(busy)
        self.batch_button.setEnabled(not busy and self.curve is not None and self.curve['point'] is not None)
        self.fit_button.setEnabled(not busy and self.curve is not None)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, max(1, total))
            self.progress.setValue(done)
            self.message.setText(f'Fitting scan: {done}/{total} points. Stop takes effect after the current point.')

    def show_fit(self, result, *, store=True):
        if store:
            if self.batch is None:
                self.batch = {'uid': self.curve['uid'], 'field': self.curve['field'], 'x_field': self.curve['x_field'],
                              'config': result['config'], 'records': [], 'cancelled': False,
                              'total': len(self.snapshot['rows']) if self.curve['point'] is not None else 1,
                              'selection': {key: value for key, value in self.curve.items() if key not in {'x', 'y'}}}
            point = self.curve['point']
            row_index = point if point is not None else 0
            sequences = self.snapshot.get('seq_nums', [])
            coordinates = {}
            for field in coordinate_fields(self.snapshot):
                value = numeric(self.snapshot['rows'][row_index].get(field))
                if value is not None and value.ndim == 0 and np.isfinite(value):
                    coordinates[field] = float(value)
            record = {'point': point, 'sequence': sequences[row_index] if row_index < len(sequences) else row_index + 1,
                      'coordinates': coordinates, 'error': '', 'fit': result,
                      'units': {'coordinate': self.curve['x_units'], 'intensity': self.curve['y_units']}}
            self.batch['records'] = [item for item in self.batch['records'] if item['point'] != point] + [record]
            self.trends.set_records(self.snapshot, self.batch['records'], self.peaks_spin.value(), field=self.batch['field'])
        self.fit_result = result
        self.fit_line.setData(result['x'], result['fit'])
        background = result['components'].get('bg_')
        if background is not None:
            self.background_line.setData(result['x'], np.broadcast_to(background, result['x'].shape))
        self.residual_line.setData(result['x'], result['residual'])
        self.results.setRowCount(len(result['parameters']))
        for row, (name, parameter) in enumerate(result['parameters'].items()):
            values = [name, f"{parameter['value']:.8g}", 'Unavailable' if parameter['stderr'] is None else f"{parameter['stderr']:.5g}"]
            for column, value in enumerate(values):
                self.results.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self.results.resizeColumnsToContents()
        self.report.setPlainText(result['report'])
        note = '' if result['uncertainties_available'] else ' · covariance unavailable'
        self.message.setText(f"Fit converged · {result['ndata']} points\nReduced χ² = {result['redchi']:.5g}\nAIC = {result['aic']:.5g} · BIC = {result['bic']:.5g}{note}")
        self.copy_button.setEnabled(True)

    def show_batch(self, batch):
        self._clear_selected_fit()
        self.batch = batch
        self.trends.set_records(self.snapshot, batch['records'], self.peaks_spin.value(), field=batch['field'])
        self._display_selected_record()
        self.copy_button.setEnabled(bool(batch['records']))
        successes = sum(record['fit'] is not None for record in batch['records'])
        state = 'Stopped' if batch['cancelled'] else 'Complete'
        self.message.setText(f"{state}: {successes} successful / {len(batch['records'])} attempted. Inspect trends and point results below.")
        self.results_widget.setCurrentWidget(self.trends)

    def _display_selected_record(self):
        if self.curve is None:
            return
        record = next((record for record in (self.batch or {}).get('records', []) if record['point'] == self.curve['point']), None)
        if record is None:
            self.message.setText('This point has no stored fit. Fit it or fit all acquisition points.')
        elif record['fit'] is not None:
            self.show_fit(record['fit'], store=False)
        else:
            self.message.setText(f"Fit failed at this point: {record['error']}")
        self.copy_button.setEnabled(bool((self.batch or {}).get('records')))

    def _copy_results(self):
        if self.batch is None:
            return
        QtWidgets.QApplication.clipboard().setText(json.dumps(export_batch(self.batch), indent=2, allow_nan=False))
