"""Simple timed-count display for the USB-CTR08."""

from __future__ import annotations

from pydm import Display
from pydm.widgets import PyDMLabel, PyDMSpinbox
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore, QtWidgets

try:
    from diffractometer_controls.extra_ui.count_rate_gauge import CountRateGauge
    from diffractometer_controls.extra_ui.usbctr_common import (
        FloatPVWriter,
        IntPVWriter,
        ca_address,
        configure_spinbox,
        count_rate,
        display_macros,
        scaler_count_field,
        value_slot,
    )
except ModuleNotFoundError as exc:
    if exc.name != "diffractometer_controls":
        raise
    # PyDM loads Python displays directly from their filenames. In that mode
    # this directory is on sys.path, but the repository need not be installed
    # as a package.
    from count_rate_gauge import CountRateGauge
    from usbctr_common import (
        FloatPVWriter,
        IntPVWriter,
        ca_address,
        configure_spinbox,
        count_rate,
        display_macros,
        scaler_count_field,
        value_slot,
    )


class USBCTRCountDisplay(Display):
    """Configure a timed scaler count and show one selected counter."""

    def __init__(self, parent=None, args=None, macros=None):
        self._channels = []
        self._count_channel = None
        self._writers = []
        self._counts = 0.0
        self._elapsed = 0.0
        self._running = False
        self._mcs_busy = False
        self._display_rate = None
        self._clock_frequency = None
        self._preset_time = None
        self._start_pending = False
        super().__init__(parent=parent, args=args, macros=macros)

        values = display_macros(self)
        board_prefix = values.get("P", "4dh4:USBCTR:")
        self._board_prefix = board_prefix
        scaler_name = values.get("S", "scaler1")
        self._scaler_prefix = f"{board_prefix}{scaler_name}"
        self._mcs_prefix = values.get("MP", f"{board_prefix}MCS:")

        self._count_writer = IntPVWriter(f"{self._scaler_prefix}.CNT", self)
        self._mode_writer = IntPVWriter(f"{self._scaler_prefix}.CONT", self)
        self._rate_writer = FloatPVWriter(f"{self._scaler_prefix}.RATE", self)
        self._clock_writer = FloatPVWriter(f"{self._scaler_prefix}.FREQ", self)
        self._preset_writer = FloatPVWriter(f"{self._scaler_prefix}.TP", self)
        self._pulse_run_writer = IntPVWriter(
            f"{board_prefix}PulseGen1Run", self
        )
        self._writers.extend(
            (
                self._count_writer,
                self._mode_writer,
                self._rate_writer,
                self._clock_writer,
                self._preset_writer,
                self._pulse_run_writer,
            )
        )

        self._build_ui()
        self._connect_fixed_channels()
        self.channel_selector.setCurrentIndex(1)
        self._select_channel(1)
        self._refresh_controls()

    def _scaler_pv(self, field: str) -> str:
        return ca_address(f"{self._scaler_prefix}.{field}")

    def _build_ui(self) -> None:
        self.setWindowTitle("USB-CTR Count")
        self.resize(860, 540)
        outer = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel("USB-CTR Count")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        outer.addWidget(heading)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Counter channel"))
        self.channel_selector = QtWidgets.QComboBox()
        channel_names = {
            0: "clock",
            1: "beam monitor",
            2: "He-3 tube",
        }
        for index in range(8):
            description = channel_names.get(index, "counter")
            self.channel_selector.addItem(
                f"CTR{index} — {description}", index
            )
        self.channel_selector.currentIndexChanged.connect(
            self._select_channel
        )
        selector_row.addWidget(self.channel_selector)
        selector_row.addStretch(1)
        outer.addLayout(selector_row)

        content = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        measurement = QtWidgets.QWidget()
        measurement_layout = QtWidgets.QVBoxLayout(measurement)
        measurement_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.total_title = QtWidgets.QLabel("CTR0 TOTAL COUNTS")
        self.total_title.setAlignment(QtCore.Qt.AlignCenter)
        self.total_title.setStyleSheet("font-size: 17px; font-weight: 600;")
        measurement_layout.addWidget(self.total_title)
        self.total_counts = QtWidgets.QLabel("0")
        self.total_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.total_counts.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.total_counts.setStyleSheet("font-size: 46px; font-weight: 700;")
        measurement_layout.addWidget(self.total_counts)
        measurement_layout.addSpacing(8)
        self.gauge = CountRateGauge(title="CTR0 AVG RATE")
        self.gauge.set_compact(True)
        self.gauge.setMinimumSize(300, 235)
        self.gauge.setMaximumSize(340, 270)
        measurement_layout.addWidget(
            self.gauge, alignment=QtCore.Qt.AlignHCenter
        )
        measurement_layout.addStretch(1)
        content.addWidget(measurement)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)

        settings = QtWidgets.QGroupBox("Count settings")
        settings_form = QtWidgets.QFormLayout(settings)
        count_time = PyDMSpinbox(init_channel=self._scaler_pv("TP"))
        count_time.precisionFromPV = False
        count_time.precision = 3
        configure_spinbox(count_time, 0.001, 604800.0, single_step=1.0)
        settings_form.addRow("Count time (s)", count_time)
        update_rate = PyDMSpinbox(init_channel=self._scaler_pv("RATE"))
        update_rate.precisionFromPV = False
        update_rate.precision = 1
        configure_spinbox(update_rate, 0.1, 100.0, single_step=0.1)
        settings_form.addRow("Live update rate (Hz)", update_rate)
        settings_form.addRow(
            "Clock frequency (Hz)",
            PyDMLabel(
                init_channel=ca_address(
                    f"{self._board_prefix}PulseGen1Frequency_RBV"
                )
            ),
        )
        controls_layout.addWidget(settings)

        actions = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._start_count)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_count)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        controls_layout.addLayout(actions)

        readings = QtWidgets.QGroupBox("Selected-channel readings")
        readings_form = QtWidgets.QFormLayout(readings)
        self.average_label = QtWidgets.QLabel("—")
        self.elapsed_label = QtWidgets.QLabel("—")
        readings_form.addRow("Elapsed time (s)", self.elapsed_label)
        readings_form.addRow("Average count rate", self.average_label)
        readings_form.addRow(
            "Counter state", PyDMLabel(init_channel=self._scaler_pv("CNT"))
        )
        controls_layout.addWidget(readings)

        wiring = QtWidgets.QLabel(
            "Required jumpers: TMR0 to C0IN, then C0OUT to each detector "
            "counter gate in use (for C1IN, connect C0OUT to C1GT). CTR0 is "
            "the timing reference; detector inputs begin at CTR1."
        )
        wiring.setWordWrap(True)
        wiring.setStyleSheet(
            "background: #fff4ce; color: #3b3100; padding: 8px; "
            "border: 1px solid #d6b656;"
        )
        controls_layout.addWidget(wiring)

        self.status = QtWidgets.QLabel("Connecting to scaler records…")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        controls_layout.addStretch(1)
        content.addWidget(controls)
        content.setSizes([540, 360])
        outer.addWidget(content, 1)

    def _connect(self, pv: str, slot, connection_slot=None) -> PyDMChannel:
        channel = PyDMChannel(
            address=ca_address(pv),
            value_slot=slot,
            connection_slot=connection_slot,
        )
        channel.connect()
        self._channels.append(channel)
        return channel

    def _connect_fixed_channels(self) -> None:
        self._connect(f"{self._scaler_prefix}.T", self._on_elapsed)
        self._connect(f"{self._scaler_prefix}.CNT", self._on_running)
        self._connect(f"{self._scaler_prefix}.RATE", self._on_display_rate)
        self._connect(
            f"{self._board_prefix}PulseGen1Frequency_RBV",
            self._on_clock_frequency,
        )
        self._connect(f"{self._scaler_prefix}.TP", self._on_preset_time)
        self._connect(
            f"{self._mcs_prefix}HardwareAcquiring", self._on_mcs_busy
        )

    @QtCore.Slot(int)
    def _select_channel(self, index: int) -> None:
        if not hasattr(self, "gauge"):
            return
        index = max(0, min(7, int(index)))
        self.total_title.setText(f"CTR{index} TOTAL COUNTS")
        self.gauge.set_title(f"CTR{index} AVG RATE")
        self._counts = 0.0
        self._update_rate()
        if self._count_channel is not None:
            self._count_channel.disconnect()
            if self._count_channel in self._channels:
                self._channels.remove(self._count_channel)
        field = scaler_count_field(index)
        self._count_channel = self._connect(
            f"{self._scaler_prefix}.{field}",
            self._on_counts,
            self._on_count_connection,
        )

    @value_slot
    def _on_counts(self, value) -> None:
        try:
            self._counts = max(0.0, float(value))
        except (TypeError, ValueError):
            self._counts = 0.0
        self._update_rate()

    @QtCore.Slot(bool)
    def _on_count_connection(self, connected: bool) -> None:
        self.gauge.set_connected(connected)
        if not connected:
            self.status.setText("Selected scaler channel is disconnected.")

    @value_slot
    def _on_elapsed(self, value) -> None:
        try:
            self._elapsed = max(0.0, float(value))
        except (TypeError, ValueError):
            self._elapsed = 0.0
        self._update_rate()

    @value_slot
    def _on_running(self, value) -> None:
        self._running = bool(value)
        self._refresh_controls()

    @value_slot
    def _on_mcs_busy(self, value) -> None:
        self._mcs_busy = bool(value)
        if self._mcs_busy:
            self._start_pending = False
        self._refresh_controls()

    @value_slot
    def _on_display_rate(self, value) -> None:
        try:
            self._display_rate = float(value)
        except (TypeError, ValueError):
            self._display_rate = None

    @value_slot
    def _on_clock_frequency(self, value) -> None:
        try:
            self._clock_frequency = float(value)
        except (TypeError, ValueError):
            self._clock_frequency = None
        self._refresh_controls()

    @value_slot
    def _on_preset_time(self, value) -> None:
        try:
            self._preset_time = float(value)
        except (TypeError, ValueError):
            self._preset_time = None
        self._refresh_controls()

    def _update_rate(self) -> None:
        rate = count_rate(self._counts, self._elapsed)
        self.gauge.set_rate(rate)
        self.total_counts.setText(f"{self._counts:,.0f}")
        self.elapsed_label.setText(f"{self._elapsed:,.3f}")
        self.average_label.setText(f"{rate:,.3f} counts/s")

    @QtCore.Slot()
    def _start_count(self) -> None:
        if self._mcs_busy:
            self.status.setText(
                "Cannot start the timed counter while MCS acquisition is active."
            )
            return
        if self._clock_frequency is None or self._clock_frequency <= 0.0:
            self.status.setText(
                "Cannot start: the TMR0 clock frequency is unavailable."
            )
            return
        if self._preset_time is None or self._preset_time <= 0.0:
            self.status.setText("Cannot start: set Count time above zero.")
            return
        self._start_pending = True
        self._mode_writer.value.emit(0)
        self._clock_writer.value.emit(self._clock_frequency)
        self._pulse_run_writer.value.emit(1)
        if self._display_rate is None or self._display_rate <= 0.0:
            self._rate_writer.value.emit(5.0)
            self.status.setText(
                "Live update rate was zero; set to 5 Hz. Starting timed count."
            )
        else:
            self.status.setText("Configuring timed count…")
        QtCore.QTimer.singleShot(100, self._apply_preset_time)

    def _apply_preset_time(self) -> None:
        if self._start_pending and not self._mcs_busy:
            # scalerRecord only recalculates PR1 when TP is processed.  FREQ
            # must therefore be written first and TP reprocessed afterward.
            self._preset_writer.value.emit(self._preset_time)
            QtCore.QTimer.singleShot(100, self._finish_start_count)

    def _finish_start_count(self) -> None:
        if self._start_pending and not self._mcs_busy:
            self._start_pending = False
            self.status.setText("Timed count start requested.")
            self._count_writer.value.emit(1)

    @QtCore.Slot()
    def _stop_count(self) -> None:
        self._start_pending = False
        self.status.setText("Timed count stop requested.")
        self._count_writer.value.emit(0)

    def _refresh_controls(self) -> None:
        self.start_button.setEnabled(
            self._clock_frequency is not None
            and self._clock_frequency > 0.0
            and self._preset_time is not None
            and self._preset_time > 0.0
            and not self._running
            and not self._mcs_busy
            and not self._start_pending
        )
        self.stop_button.setEnabled(self._running or self._start_pending)
        if self._mcs_busy:
            self.status.setText(
                "MCS mode is active; timed-count Start is interlocked."
            )
        elif self._running:
            self.status.setText("Timed count is running.")
        elif self._clock_frequency is None or self._clock_frequency <= 0.0:
            self.status.setText("Waiting for the timer clock.")
        elif self._preset_time is None or self._preset_time <= 0.0:
            self.status.setText("Set Count time above zero before starting.")
        elif self._elapsed > 0.0:
            self.status.setText("Timed count is complete or stopped.")
        else:
            self.status.setText("Ready to count.")

    def closeEvent(self, event) -> None:
        for channel in tuple(self._channels):
            channel.disconnect()
        for writer in self._writers:
            writer.close()
        super().closeEvent(event)
