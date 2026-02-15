import os
import re

from epics import caput
from pydm.display import ScreenTarget, load_file
from pydm.utilities import find_file, is_pydm_app
from pydm.widgets.channel import PyDMChannel
from pydm.widgets.label import PyDMLabel
from pydm.widgets.related_display_button import PyDMRelatedDisplayButton
from qtpy import QtCore
from qtpy.QtGui import QFont, QFontMetrics
from qtpy.QtWidgets import QCheckBox, QGroupBox, QPushButton

import display


class ReactorPowerDisplay(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename="extra_ui/reactor_power.ui"):
        self._group_box = None
        self._power_value_label = None
        self._operations_button = None
        self._suspender_checkbox = None
        self._suspender_enable_channel = None
        self._suspender_enable_pv = ""
        self._suspender_checkbox_updating = False
        super().__init__(parent, args, macros, ui_filename)

    def ui_filename(self):
        return "extra_ui/reactor_power.ui"

    def ui_filepath(self):
        return super().ui_filepath()

    @staticmethod
    def _coerce_float(value):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        text = str(value).strip()
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _apply_power_style(self, value):
        numeric_value = self._coerce_float(value)
        if numeric_value is None:
            self._power_value_label.setStyleSheet(
                "font-weight: 700; color: #334155; background: transparent; border: none;"
            )
            return

        if numeric_value < 1:
            color = "#b91c1c"  # red
        elif numeric_value > 5:
            color = "#15803d"  # green
        else:
            color = "#a16207"  # yellow/amber

        self._power_value_label.setStyleSheet(
            f"font-weight: 700; color: {color}; background: transparent; border: none;"
        )

    def _open_mitr_operations(self):
        base_path = ""
        macros = {}
        try:
            parent_file_path = self.loaded_file()
        except Exception:
            parent_file_path = ""

        if parent_file_path:
            base_path = os.path.dirname(parent_file_path)
            try:
                macros.update(self.macros() or {})
            except Exception:
                pass

        fname = find_file("mitr_operations.ui", base_path=base_path)
        if not fname:
            fname = find_file("extra_ui/mitr_operations.ui", base_path=base_path)
        if not fname:
            return

        if is_pydm_app():
            load_file(fname, macros=macros, target=ScreenTarget.NEW_PROCESS)
        else:
            load_file(fname, macros=macros, target=ScreenTarget.DIALOG)

    def _replace_related_button_with_native_button(self):
        related_button = self.findChild(PyDMRelatedDisplayButton, "mitrOperationsButton")
        if related_button is None:
            return

        layout = related_button.parentWidget().layout()
        idx = layout.indexOf(related_button) if layout else -1

        related_button.hide()
        if layout:
            layout.removeWidget(related_button)
        related_button.deleteLater()

        native_button = QPushButton("Operations", self)
        native_button.setStyleSheet("")
        native_button.clicked.connect(self._open_mitr_operations)
        if layout and idx >= 0:
            layout.insertWidget(idx, native_button)
        elif layout:
            layout.addWidget(native_button)
        self._operations_button = native_button

        def _sync_height():
            h = max(native_button.sizeHint().height(), native_button.minimumSizeHint().height())
            native_button.setMinimumHeight(h)
            native_button.setMaximumHeight(h)

        QtCore.QTimer.singleShot(0, _sync_height)
        QtCore.QTimer.singleShot(100, _sync_height)

    def _pv_prefix(self):
        prefix = ""
        try:
            macros = self.macros() or {}
            if isinstance(macros, dict):
                prefix = str(macros.get("P", "") or "").strip()
        except Exception:
            pass
        if not prefix:
            try:
                from application import MITRApplication

                app = MITRApplication.instance()
                main_window = getattr(app, "main_window", None)
                main_macros = getattr(main_window, "macros", None)
                if isinstance(main_macros, dict):
                    prefix = str(main_macros.get("P", "") or "").strip()
            except Exception:
                pass
        if not prefix:
            return ""
        if not prefix.endswith(":"):
            prefix = f"{prefix}:"
        return prefix

    def _on_suspender_checkbox_toggled(self, checked):
        if self._suspender_checkbox_updating:
            return
        if not self._suspender_enable_pv:
            return
        wrote = False
        try:
            wrote = bool(caput(self._suspender_enable_pv, 1 if checked else 0, wait=False))
        except Exception:
            wrote = False
        if wrote:
            return
        # Fallback if CA write is unavailable from this host.
        try:
            from application import MITRApplication

            app = MITRApplication.instance()
            re_api = getattr(app, "re_manager_api", None)
            if re_api is not None:
                cmd = f"_queue_set_reactor_power_suspender({bool(checked)})"
                try:
                    # Must run in background to be accepted while a plan is active.
                    re_api.script_upload(cmd, run_in_background=True)
                except TypeError:
                    # Compatibility with older API signatures.
                    re_api.script_upload(cmd)
        except Exception:
            pass

    def _on_suspender_enable_changed(self, value):
        checked = False
        if isinstance(value, str):
            checked = value.strip().lower() in ("1", "true", "yes", "on")
        else:
            try:
                checked = bool(int(float(value)))
            except Exception:
                checked = bool(value)

        if self._suspender_checkbox is None:
            return
        self._suspender_checkbox_updating = True
        try:
            self._suspender_checkbox.setChecked(checked)
        finally:
            self._suspender_checkbox_updating = False

    def _setup_suspender_checkbox(self):
        if not getattr(self, "_group_box", None):
            return
        layout = self._group_box.layout()
        if layout is None:
            return

        checkbox = QCheckBox("Suspender Enabled", self)
        checkbox.setChecked(True)
        checkbox.toggled.connect(self._on_suspender_checkbox_toggled)

        # Place just above the Operations button.
        if self._operations_button is not None:
            idx = layout.indexOf(self._operations_button)
            if idx < 0:
                idx = layout.count()
            layout.insertWidget(idx, checkbox)
        else:
            layout.addWidget(checkbox)
        self._suspender_checkbox = checkbox

        prefix = self._pv_prefix()
        if not prefix:
            checkbox.setEnabled(False)
            checkbox.setToolTip("Missing macro P: cannot resolve suspender PV prefix.")
            return

        self._suspender_enable_pv = f"{prefix}Bluesky:SuspenderEnable"
        enable_address = f"ca://{prefix}Bluesky:SuspenderEnable"
        self._suspender_enable_channel = PyDMChannel(
            address=enable_address,
            value_slot=self._on_suspender_enable_changed,
        )
        self._suspender_enable_channel.connect()

    def _fit_groupbox_title_font(self):
        if not getattr(self, "_group_box", None):
            return
        title = self._group_box.title() or "Reactor Power"
        available_width = max(self._group_box.width() - 16, 40)
        chosen_size = 10
        for point_size in range(14, 9, -1):
            font = QFont(self.font())
            font.setPointSize(point_size)
            font.setBold(True)
            if QFontMetrics(font).horizontalAdvance(title) <= available_width:
                chosen_size = point_size
                break
        self._group_box.setStyleSheet(
            "QGroupBox { "
            f"font-size: {chosen_size}px; font-weight: 700; margin-top: 8px; "
            "} "
            "QGroupBox::title { "
            "subcontrol-origin: margin; subcontrol-position: top center; padding: 0 4px; "
            "}"
        )

    def _fit_power_value_font(self):
        if not getattr(self, "_power_value_label", None):
            return
        text = self._power_value_label.text().strip()
        if not text:
            return
        rect = self._power_value_label.contentsRect().adjusted(4, 2, -4, -2)
        if rect.width() <= 0 or rect.height() <= 0:
            return
        base_font = QFont(self._power_value_label.font())
        chosen_size = 10
        for point_size in range(22, 9, -1):
            font = QFont(base_font)
            font.setPointSize(point_size)
            font.setBold(True)
            metrics = QFontMetrics(font)
            if metrics.horizontalAdvance(text) <= rect.width() and metrics.height() <= rect.height():
                chosen_size = point_size
                break
        fitted_font = QFont(base_font)
        fitted_font.setPointSize(chosen_size)
        fitted_font.setBold(True)
        self._power_value_label.setFont(fitted_font)

    def _apply_autofit(self):
        self._fit_groupbox_title_font()
        self._fit_power_value_font()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._apply_autofit)

    def customize_ui(self):
        self._group_box = self.findChild(QGroupBox, "groupBox")
        self._replace_related_button_with_native_button()
        self._setup_suspender_checkbox()

        self._power_value_label = self.findChild(PyDMLabel, "powerValueLabel")
        if self._power_value_label is None:
            return
        self._power_value_label.precisionFromPV = False
        self._power_value_label.precision = 2

        original_value_changed = self._power_value_label.value_changed

        def _value_changed_with_style(new_value):
            original_value_changed(new_value)
            self._apply_power_style(new_value)
            self._fit_power_value_font()

        self._power_value_label.value_changed = _value_changed_with_style
        self._apply_power_style(self._power_value_label.value)
        QtCore.QTimer.singleShot(0, self._apply_autofit)
