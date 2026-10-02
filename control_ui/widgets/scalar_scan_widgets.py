"""Reusable detail widgets for live scalar scans."""

from __future__ import annotations

import math

from qtpy import QtCore, QtGui, QtWidgets

from control_ui.widgets.count_rate_gauge import CountRateGauge


def _number_text(value: float) -> str:
    value = float(value)
    if math.isfinite(value) and value.is_integer():
        return f"{int(value):,d}"
    return f"{value:,.8g}"


class _CountGaugeCard(QtWidgets.QWidget):
    """One operator-selected total/count-rate readout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._readings = {}

        self.selector = QtWidgets.QComboBox(self)
        self.selector.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Fixed,
        )
        self.selector.currentTextChanged.connect(self._show_selected_reading)
        self.selector.hide()

        self.name_label = QtWidgets.QLabel("Counts", self)
        self.name_label.setAlignment(QtCore.Qt.AlignCenter)
        title_font = QtGui.QFont(self.name_label.font())
        title_font.setPointSize(max(10, title_font.pointSize()))
        title_font.setBold(True)
        self.name_label.setFont(title_font)

        self.total_value = QtWidgets.QLabel("0 cts", self)
        self.total_value.setAlignment(QtCore.Qt.AlignCenter)
        self.total_value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        value_font = QtGui.QFont(self.total_value.font())
        value_font.setPointSize(max(17, value_font.pointSize() + 7))
        value_font.setBold(True)
        self.total_value.setFont(value_font)

        self.gauge = CountRateGauge(self, title="cts/s")
        self.gauge.set_compact(True)
        self.gauge.setMinimumSize(140, 145)
        self.gauge.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.selector)
        layout.addWidget(self.name_label)
        layout.addWidget(self.total_value)
        layout.addWidget(self.gauge, 1)

    @property
    def selected_series(self) -> str:
        return str(self.selector.currentText() or "")

    def reset(self) -> None:
        self._readings.clear()
        with QtCore.QSignalBlocker(self.selector):
            self.selector.clear()
        self.selector.hide()
        self.name_label.setText("Counts")
        self.name_label.show()
        self.total_value.setText("0 cts")
        self.gauge.set_title("cts/s")
        self.gauge.set_rate(0.0, animate=False)

    def set_series(self, series_names, *, selected_index, selectable) -> None:
        previous = self.selected_series
        names = [str(name) for name in series_names]
        with QtCore.QSignalBlocker(self.selector):
            self.selector.clear()
            self.selector.addItems(names)
            target = self.selector.findText(previous)
            if target < 0:
                target = min(max(0, int(selected_index)), max(0, len(names) - 1))
            self.selector.setCurrentIndex(target)
        self.selector.setVisible(bool(selectable and len(names) > 1))
        self.name_label.setVisible(not selectable)
        self._show_selected_reading(self.selected_series)

    def set_readings(self, readings) -> None:
        self._readings = dict(readings or {})
        self._show_selected_reading(self.selected_series)

    @QtCore.Slot(str)
    def _show_selected_reading(self, series_name) -> None:
        reading = self._readings.get(str(series_name))
        if reading is None:
            return
        total_counts, rate = reading
        self.name_label.setText(str(series_name))
        self.total_value.setText(f"{_number_text(total_counts)} cts")
        self.gauge.set_rate(rate)


class ScalarCountReadout(QtWidgets.QWidget):
    """Show one or two stable CTR08 count-rate gauges.

    One selected counter produces one gauge and two counters produce two gauges.
    With three or more counters, two selectors let the operator independently
    choose which pair remains visible.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._readings = {}
        self._series_names = []
        self.cards = [_CountGaugeCard(self), _CountGaugeCard(self)]

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        for card in self.cards:
            layout.addWidget(card, 1)
        self.cards[1].hide()

    def reset(self) -> None:
        self._readings.clear()
        self._series_names.clear()
        for card in self.cards:
            card.reset()
            card.hide()
        self.cards[0].show()

    @QtCore.Slot(str, float, object)
    def update_reading(self, series_name, total_counts, elapsed_time=None) -> None:
        series_name = str(series_name or "Counts")
        total_counts = max(0.0, float(total_counts))
        try:
            elapsed = float(elapsed_time)
        except (TypeError, ValueError):
            elapsed = 0.0
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            elapsed = 0.0
        rate = total_counts / elapsed if elapsed else 0.0
        self._readings[series_name] = (total_counts, rate)

        if series_name not in self._series_names:
            self._series_names.append(series_name)
            self._configure_cards()
        for card in self.cards:
            card.set_readings(self._readings)

    def _configure_cards(self) -> None:
        count = len(self._series_names)
        selectable = count > 2
        visible_cards = min(2, count)
        for index, card in enumerate(self.cards):
            card.setVisible(index < visible_cards)
            if index < visible_cards:
                card.set_series(
                    self._series_names,
                    selected_index=index,
                    selectable=selectable,
                )


class ScalarScanTable(QtWidgets.QTableWidget):
    """Tabulate committed scalar values against scan position."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x_label = "Position"
        self._series_columns = {}
        self._current_x = None
        self._current_row = -1

        self.setColumnCount(1)
        self.setHorizontalHeaderLabels([self._x_label])
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeToContents
        )
        self.setMinimumSize(260, 190)

    def reset_scan(self, x_label="Position") -> None:
        self.clearContents()
        self.setRowCount(0)
        self.setColumnCount(1)
        self._x_label = str(x_label or "Position")
        self.setHorizontalHeaderLabels([self._x_label])
        self._series_columns.clear()
        self._current_x = None
        self._current_row = -1

    @QtCore.Slot(str, float, float)
    def append_point(self, series_name, x_value, y_value) -> None:
        series_name = str(series_name or "Value")
        x_value = float(x_value)
        y_value = float(y_value)
        column = self._ensure_series_column(series_name)

        same_x = (
            self._current_x is not None
            and math.isclose(x_value, self._current_x, rel_tol=1e-12, abs_tol=1e-12)
        )
        cell_is_empty = (
            self._current_row >= 0 and self.item(self._current_row, column) is None
        )
        if not same_x or not cell_is_empty:
            self._current_row = self.rowCount()
            self.insertRow(self._current_row)
            self.setItem(
                self._current_row,
                0,
                QtWidgets.QTableWidgetItem(_number_text(x_value)),
            )
            self._current_x = x_value

        value_item = QtWidgets.QTableWidgetItem(_number_text(y_value))
        value_item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.setItem(self._current_row, column, value_item)
        self.scrollToBottom()

    def _ensure_series_column(self, series_name: str) -> int:
        column = self._series_columns.get(series_name)
        if column is not None:
            return column
        column = self.columnCount()
        self.insertColumn(column)
        self.setHorizontalHeaderItem(column, QtWidgets.QTableWidgetItem(series_name))
        self._series_columns[series_name] = column
        return column
