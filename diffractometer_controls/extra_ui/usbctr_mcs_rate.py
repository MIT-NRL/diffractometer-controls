"""Single-channel MCS count-rate display for the USB-CTR08."""

from __future__ import annotations

import numpy as np
from pydm import Display
from pydm.widgets import PyDMEnumComboBox, PyDMLabel, PyDMSpinbox
from pydm.widgets.channel import PyDMChannel
from qtpy import QtCore, QtWidgets

try:
    from diffractometer_controls.extra_ui.count_rate_gauge import CountRateGauge
    from diffractometer_controls.extra_ui.usbctr_common import (
        IntPVWriter,
        ca_address,
        channel_waveform_suffix,
        display_macros,
        mcs_average_rate,
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
        IntPVWriter,
        ca_address,
        channel_waveform_suffix,
        display_macros,
        mcs_average_rate,
        value_slot,
    )


class USBCTRMCSRateDisplay(Display):
    """Display one USB-CTR08 MCS channel as a live count rate."""

    def __init__(self, parent=None, args=None, macros=None):
        self._channels = []
        self._waveform_channel = None
        self._enable_channel = None
        self._writers = []
        self._samples = np.asarray([], dtype=float)
        self._dwell = 0.0
        self._acquiring = False
        self._scaler_counting = False
        self._scaler_auto = False
        self._scaler_busy = False
        self._selected_enabled = None
        self._restart_armed = False
        self._start_pending = False
        super().__init__(parent=parent, args=args, macros=macros)

        values = display_macros(self)
        board_prefix = values.get("P", "4dh4:USBCTR:")
        self._mcs_prefix = values.get("MP", f"{board_prefix}MCS:")
        scaler_name = values.get("S", "scaler1")
        self._scaler_prefix = f"{board_prefix}{scaler_name}"

        self._erase_start = IntPVWriter(
            f"{self._mcs_prefix}EraseStart", self
        )
        self._stop = IntPVWriter(f"{self._mcs_prefix}StopAll", self)
        self._channel_advance_writer = IntPVWriter(
            f"{self._mcs_prefix}ChannelAdvance", self
        )
        self._trigger_mode_writer = IntPVWriter(
            f"{self._mcs_prefix}TrigMode", self
        )
        self._point_zero_writer = IntPVWriter(
            f"{self._mcs_prefix}Point0Action", self
        )
        self._points_writer = IntPVWriter(f"{self._mcs_prefix}NuseAll", self)
        self._writers.extend(
            (
                self._erase_start,
                self._stop,
                self._channel_advance_writer,
                self._trigger_mode_writer,
                self._point_zero_writer,
                self._points_writer,
            )
        )

        self._build_ui()
        self._connect_fixed_channels()
        self._select_channel(0)
        self._refresh_controls()

    def _pv(self, suffix: str) -> str:
        return ca_address(f"{self._mcs_prefix}{suffix}")

    def _build_ui(self) -> None:
        self.setWindowTitle("USB-CTR08 MCS Count Rate")
        self.resize(940, 620)
        outer = QtWidgets.QVBoxLayout(self)

        heading = QtWidgets.QLabel("USB-CTR08 MCS Count-Rate Gauge")
        heading.setStyleSheet("font-size: 20px; font-weight: 600;")
        outer.addWidget(heading)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Counter channel"))
        self.channel_selector = QtWidgets.QComboBox()
        for index in range(8):
            self.channel_selector.addItem(
                f"CTR{index}  (EPICS mca{index + 1})", index
            )
        self.channel_selector.currentIndexChanged.connect(
            self._select_channel
        )
        top.addWidget(self.channel_selector)
        top.addSpacing(18)
        top.addWidget(QtWidgets.QLabel("Average newest bins"))
        self.average_bins = QtWidgets.QSpinBox()
        self.average_bins.setRange(1, 100)
        self.average_bins.setValue(1)
        self.average_bins.valueChanged.connect(self._update_rate)
        top.addWidget(self.average_bins)
        top.addStretch(1)
        outer.addLayout(top)

        content = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.gauge = CountRateGauge(title="CTR0 MCS RATE")
        content.addWidget(self.gauge)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls)

        acquisition = QtWidgets.QGroupBox("Acquisition")
        acquisition_form = QtWidgets.QFormLayout(acquisition)
        self.dwell_control = PyDMSpinbox(init_channel=self._pv("Dwell"))
        self.dwell_control.precisionFromPV = False
        self.dwell_control.precision = 6
        acquisition_form.addRow("Dwell setpoint (s)", self.dwell_control)
        acquisition_form.addRow(
            "Actual dwell (s)", PyDMLabel(init_channel=self._pv("Dwell_RBV"))
        )
        acquisition_form.addRow(
            "Time points", PyDMSpinbox(init_channel=self._pv("NuseAll"))
        )
        preset = PyDMSpinbox(init_channel=self._pv("PresetReal"))
        preset.precisionFromPV = False
        preset.precision = 3
        acquisition_form.addRow("Stop after (s, 0 disables)", preset)
        self.continuous = QtWidgets.QCheckBox("Restart when buffer completes")
        acquisition_form.addRow("Continuous", self.continuous)
        controls_layout.addWidget(acquisition)

        mode = QtWidgets.QGroupBox("MCS operating settings")
        mode_form = QtWidgets.QFormLayout(mode)
        mode_form.addRow(
            "Channel advance",
            PyDMEnumComboBox(init_channel=self._pv("ChannelAdvance")),
        )
        mode_form.addRow(
            "Trigger mode", PyDMEnumComboBox(init_channel=self._pv("TrigMode"))
        )
        mode_form.addRow(
            "Point zero", PyDMEnumComboBox(init_channel=self._pv("Point0Action"))
        )
        self.enable_control = PyDMEnumComboBox()
        mode_form.addRow("Selected counter enabled", self.enable_control)
        self.apply_recommended = QtWidgets.QCheckBox(
            "Internal / Low / Skip / 2048 points"
        )
        self.apply_recommended.setChecked(True)
        mode_form.addRow("Apply on Start", self.apply_recommended)
        recommendation = QtWidgets.QLabel(
            "Uncheck to preserve advanced MCS settings."
        )
        recommendation.setWordWrap(True)
        mode_form.addRow(recommendation)
        controls_layout.addWidget(mode)

        actions = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("Erase / Start")
        self.start_button.clicked.connect(self._start_acquisition)
        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_acquisition)
        actions.addWidget(self.start_button)
        actions.addWidget(self.stop_button)
        controls_layout.addLayout(actions)

        readings = QtWidgets.QGroupBox("Selected-channel readings")
        readings_form = QtWidgets.QFormLayout(readings)
        self.raw_counts = QtWidgets.QLabel("—")
        self.numeric_rate = QtWidgets.QLabel("—")
        self.numeric_rate.setStyleSheet("font-size: 18px; font-weight: 600;")
        self.points_read = QtWidgets.QLabel("0")
        readings_form.addRow("Newest dwell counts", self.raw_counts)
        readings_form.addRow("Calculated rate", self.numeric_rate)
        readings_form.addRow("Completed points", self.points_read)
        readings_form.addRow(
            "Elapsed", PyDMLabel(init_channel=self._pv("ElapsedReal"))
        )
        readings_form.addRow(
            "MCS state", PyDMLabel(init_channel=self._pv("Acquiring"))
        )
        readings_form.addRow(
            "Hardware state",
            PyDMLabel(init_channel=self._pv("HardwareAcquiring")),
        )
        controls_layout.addWidget(readings)

        self.status = QtWidgets.QLabel("Connecting to MCS records…")
        self.status.setWordWrap(True)
        controls_layout.addWidget(self.status)
        controls_layout.addStretch(1)
        content.addWidget(controls)
        content.setSizes([560, 380])
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
        self._connect(f"{self._mcs_prefix}Dwell_RBV", self._on_dwell)
        self._connect(f"{self._mcs_prefix}Acquiring", self._on_acquiring)
        self._connect(f"{self._scaler_prefix}.CNT", self._on_scaler_counting)
        self._connect(f"{self._scaler_prefix}.CONT", self._on_scaler_auto)

    @QtCore.Slot(int)
    def _select_channel(self, index: int) -> None:
        if not hasattr(self, "gauge"):
            return
        index = max(0, min(7, int(index)))
        self.gauge.set_title(f"CTR{index} MCS RATE")
        self._samples = np.asarray([], dtype=float)
        self._update_rate()

        if self._waveform_channel is not None:
            self._waveform_channel.disconnect()
            if self._waveform_channel in self._channels:
                self._channels.remove(self._waveform_channel)
        if self._enable_channel is not None:
            self._enable_channel.disconnect()
            if self._enable_channel in self._channels:
                self._channels.remove(self._enable_channel)

        waveform = channel_waveform_suffix(index)
        self._waveform_channel = self._connect(
            f"{self._mcs_prefix}{waveform}",
            self._on_waveform,
            self._on_waveform_connection,
        )
        self._selected_enabled = None
        self.enable_control.channel = self._pv(
            f"MCSCounter{index + 1}Enable"
        )
        self._enable_channel = self._connect(
            f"{self._mcs_prefix}MCSCounter{index + 1}Enable",
            self._on_counter_enabled,
        )

    @value_slot
    def _on_waveform(self, value) -> None:
        try:
            self._samples = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            self._samples = np.asarray([], dtype=float)
        self._update_rate()

    @QtCore.Slot(bool)
    def _on_waveform_connection(self, connected: bool) -> None:
        self.gauge.set_connected(connected)
        if not connected:
            self.status.setText("Selected MCS waveform is disconnected.")

    @value_slot
    def _on_dwell(self, value) -> None:
        try:
            self._dwell = max(0.0, float(value))
        except (TypeError, ValueError):
            self._dwell = 0.0
        self._update_rate()

    @value_slot
    def _on_counter_enabled(self, value) -> None:
        self._selected_enabled = bool(value)

    @value_slot
    def _on_scaler_counting(self, value) -> None:
        self._scaler_counting = bool(value)
        self._update_scaler_busy()

    @value_slot
    def _on_scaler_auto(self, value) -> None:
        self._scaler_auto = bool(value)
        self._update_scaler_busy()

    def _update_scaler_busy(self) -> None:
        self._scaler_busy = self._scaler_counting or self._scaler_auto
        if self._scaler_busy:
            self._start_pending = False
        self._refresh_controls()

    @value_slot
    def _on_acquiring(self, value) -> None:
        was_acquiring = self._acquiring
        self._acquiring = bool(value)
        self._refresh_controls()
        if (
            was_acquiring
            and not self._acquiring
            and self._restart_armed
            and self.continuous.isChecked()
        ):
            QtCore.QTimer.singleShot(200, self._restart_if_allowed)

    def _update_rate(self) -> None:
        bins = self.average_bins.value() if hasattr(self, "average_bins") else 1
        rate = mcs_average_rate(self._samples, self._dwell, bins)
        self.gauge.set_rate(rate)
        if self._samples.size:
            self.raw_counts.setText(f"{self._samples[-1]:,.0f}")
            self.points_read.setText(str(self._samples.size))
            self.numeric_rate.setText(f"{rate:,.3f} counts/s")
        else:
            self.raw_counts.setText("—")
            self.points_read.setText("0")
            self.numeric_rate.setText("—")

    @QtCore.Slot()
    def _start_acquisition(self) -> None:
        if self._scaler_busy:
            self.status.setText(
                "Cannot start MCS mode while the scaler counter is running."
            )
            return
        if self._selected_enabled is not True:
            self.status.setText(
                "The selected counter is disabled or its enable record is not "
                "connected. Enable it in the vendor screen before starting."
            )
            return
        self._restart_armed = True
        self._start_pending = True
        if self.apply_recommended.isChecked():
            self._channel_advance_writer.value.emit(0)
            self._trigger_mode_writer.value.emit(7)
            self._point_zero_writer.value.emit(2)
            self._points_writer.value.emit(2048)
            self.status.setText(
                "Applying Internal / Low / Skip / 2048, then starting MCS."
            )
            QtCore.QTimer.singleShot(150, self._finish_start_acquisition)
        else:
            self._finish_start_acquisition()

    def _finish_start_acquisition(self) -> None:
        if self._start_pending and not self._scaler_busy and not self._acquiring:
            self._start_pending = False
            self.status.setText("MCS acquisition start requested.")
            self._erase_start.value.emit(1)

    @QtCore.Slot()
    def _stop_acquisition(self) -> None:
        self._restart_armed = False
        self._start_pending = False
        self.status.setText("MCS stop requested.")
        self._stop.value.emit(1)

    def _restart_if_allowed(self) -> None:
        if self._restart_armed and not self._scaler_busy and not self._acquiring:
            self.status.setText("Restarting the MCS count-rate acquisition.")
            self._erase_start.value.emit(1)

    def _refresh_controls(self) -> None:
        self.start_button.setEnabled(not self._acquiring and not self._scaler_busy)
        self.stop_button.setEnabled(self._acquiring)
        if self._scaler_busy:
            mode = "AutoCount" if self._scaler_auto else "a timed count"
            self.status.setText(
                f"Scaler {mode} is active; MCS Start is interlocked."
            )
        elif self._acquiring:
            self.status.setText("MCS acquisition is running.")

    def closeEvent(self, event) -> None:
        self._restart_armed = False
        self._start_pending = False
        for channel in tuple(self._channels):
            channel.disconnect()
        for writer in self._writers:
            writer.close()
        super().closeEvent(event)
