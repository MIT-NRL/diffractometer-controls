import re
import textwrap
import types

from qtpy import QtCore
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSizePolicy, QTextEdit, QPushButton, QToolTip
from bluesky_widgets.qt.run_engine_client import (
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)

import display

class REControlPanel(display.MITRDisplay):
    _running_plan_font_px = 15
    _running_plan_font_min_px = 11
    _running_plan_font_max_px = 28

    def __init__(self, parent=None, args=None, macros=None, ui_filename='re_control_panel.ui'):
        self._running_plan_font_px = 15
        self._running_plan_font_min_px = 11
        self._running_plan_font_max_px = 28
        self._pending_restyle_methods = set()
        self._restyle_flush_scheduled = False
        super().__init__(parent, args, macros, ui_filename)
        # print("REControlPanel here")
        # self.customize_ui()

    def ui_filename(self):
        return 're_control_panel.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def _current_palette(self):
        app = QApplication.instance()
        return app.palette() if app is not None else self.palette()

    @staticmethod
    def _set_stylesheet_if_changed(widget, stylesheet):
        if widget.styleSheet() != stylesheet:
            widget.setStyleSheet(stylesheet)

    @staticmethod
    def _set_text_if_changed(widget, text):
        if widget.text() != text:
            widget.setText(text)

    @staticmethod
    def _indicator_palette():
        return {
            "muted": ("#6b7280", "#f3f4f6", "#d1d5db", 1),
            "neutral": ("#334155", "#e2e8f0", "#cbd5e1", 1),
            "success": ("#065f46", "#d1fae5", "#a7f3d0", 1),
            "warning": ("#92400e", "#fef3c7", "#fcd34d", 1),
            "danger": ("#7f1d1d", "#fee2e2", "#fecaca", 1),
            "error": ("#ffffff", "#e300ff", "#a100b3", 2),
        }

    def _indicator_badge_style(self, semantic, *, font_px, weight, padding="0px 4px"):
        fg, bg, border, border_width = self._indicator_palette().get(
            semantic,
            self._indicator_palette()["neutral"],
        )
        return (
            f"font-size: {font_px}px; font-weight: {weight}; color: {fg}; "
            f"background-color: {bg}; border: {border_width}px solid {border}; "
            f"border-radius: 6px; padding: {padding};"
        )

    @staticmethod
    def _ensure_neutral_tooltip_style():
        app = QApplication.instance()
        if app is None:
            return
        pal = app.palette()
        is_dark = pal.color(QPalette.Window).lightness() < 128
        if is_dark:
            fg = pal.color(QPalette.Active, QPalette.WindowText)
            bg = pal.color(QPalette.Active, QPalette.Base)
        else:
            fg = pal.color(QPalette.Active, QPalette.WindowText)
            bg = pal.color(QPalette.Active, QPalette.Base)

        tooltip_palette = QPalette(QToolTip.palette())
        for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
            tooltip_palette.setColor(group, QPalette.ToolTipText, fg)
            tooltip_palette.setColor(group, QPalette.ToolTipBase, bg)
        QToolTip.setPalette(tooltip_palette)
        tooltip_font = app.font()
        tooltip_font.setPointSize(max(tooltip_font.pointSize(), 11))
        QToolTip.setFont(tooltip_font)

    def _schedule_restyle(self, *method_names):
        self._pending_restyle_methods.update(method_names)
        if self._restyle_flush_scheduled:
            return
        self._restyle_flush_scheduled = True
        QtCore.QTimer.singleShot(0, self._flush_scheduled_restyles)

    def _flush_scheduled_restyles(self):
        self._restyle_flush_scheduled = False
        if not self._pending_restyle_methods:
            return
        method_names = list(self._pending_restyle_methods)
        self._pending_restyle_methods.clear()
        for method_name in method_names:
            method = getattr(self, method_name, None)
            if callable(method):
                method()

    def _patch_label_set_text(self, label, *method_names):
        if label.property("_dc_restyle_text_patch_applied"):
            return
        original_set_text = label.setText

        def _patched_set_text(widget, text):
            original_set_text(text)
            self._schedule_restyle(*method_names)

        label.setText = types.MethodType(_patched_set_text, label)
        label.setProperty("_dc_restyle_text_patch_applied", True)

    def _style_re_connection_status_label(self):
        """Emphasize the RE manager connection state label."""
        status_styles = {
            "ONLINE": "font-size: 20px; font-weight: 700; color: #1f9d55; padding: 0px 2px;",
            "-----": "font-size: 17px; font-weight: 600; color: #7a7a7a; padding: 0px 2px;",
            "OFFLINE": "font-size: 20px; font-weight: 600; color: #b23b3b; padding: 0px 2px;",
            "OFF": "font-size: 20px; font-weight: 600; color: #7a7a7a; padding: 0px 2px;",
        }
        for label in self._re_manager.findChildren(QLabel):
            text = label.text().strip().upper()
            if text in status_styles:
                self._set_stylesheet_if_changed(label, status_styles[text])

    def _style_re_queue_state_label(self):
        """Emphasize queue run state in Queue Controls."""
        queue_state_styles = {
            "RUNNING": "font-size: 24px; font-weight: 800; color: #b00020;",
            "STOPPED": "font-size: 24px; font-weight: 800; color: #6b7280;",
        }
        for label in self._re_queue_controls.findChildren(QLabel):
            text = label.text().strip().upper()
            if text in queue_state_styles:
                self._set_stylesheet_if_changed(label, queue_state_styles[text])

    def _style_re_running_plan_widget(self):
        """Improve readability of the Running Plan panel."""
        body_font_px = getattr(self, "_running_plan_font_px", 15)
        min_font_px = getattr(self, "_running_plan_font_min_px", 11)
        section_font_px = body_font_px
        sub_hdr_font_px = max(body_font_px - 1, min_font_px)

        for layout in self._re_running_plan.findChildren(QHBoxLayout):
            layout.setSpacing(4)
            layout.setContentsMargins(2, 2, 2, 2)
        for layout in self._re_running_plan.findChildren(QVBoxLayout):
            layout.setSpacing(3)
            layout.setContentsMargins(2, 2, 2, 2)

        has_running_item = False
        manager_paused = False
        for status_label in self._re_status.findChildren(QLabel):
            status_text = status_label.text().strip().upper()
            if status_text.startswith("MANAGER:") and "PAUSED" in status_text:
                manager_paused = True
                break

        for text_edit in self._re_running_plan.findChildren(QTextEdit):
            if not text_edit.property("_dc_ctrl_wheel_zoom_enabled"):
                text_edit.installEventFilter(self)
                text_edit.setProperty("_dc_ctrl_wheel_zoom_enabled", True)
                text_edit.setToolTip("Use Ctrl+mouse wheel to change text size.")
            palette = self._current_palette()
            base_color = palette.color(QPalette.Base).name()
            text_color = palette.color(QPalette.Text).name()
            border_color = palette.color(QPalette.Mid).name()
            muted_color = palette.color(QPalette.Mid).name()
            text_edit_style = (
                "QTextEdit {"
                f"font-size: {body_font_px}px; "
                "padding: 4px; "
                f"border: 1px solid {border_color}; "
                "border-radius: 6px; "
                f"background-color: {base_color}; "
                f"color: {text_color};"
                "}"
            )
            self._set_stylesheet_if_changed(text_edit, text_edit_style)
            document_style = (
                f"body {{ line-height: 1.45; color: {text_color}; }} "
                f"b {{ color: {text_color}; font-weight: 700; }} "
                "b.dc-section-hdr { "
                "display: inline-block; "
                "margin-top: 8px; "
                "margin-bottom: 3px; "
                f"font-size: {section_font_px}px; "
                f"color: {text_color}; "
                "} "
                "b.dc-sub-hdr { "
                f"font-size: {sub_hdr_font_px}px; "
                f"color: {muted_color}; "
                "font-weight: 600; "
                "}"
            )
            if text_edit.document().defaultStyleSheet() != document_style:
                text_edit.document().setDefaultStyleSheet(document_style)
            html = text_edit.toHtml()
            formatted_html = self._format_running_plan_html(html)
            if formatted_html != html:
                text_edit.setHtml(formatted_html)
            has_running_item = "Plan Name:" in text_edit.toPlainText()

        for button in self._re_running_plan.findChildren(QPushButton):
            # Match global/default button styling used in the other RE widgets.
            button.setStyleSheet("")

        for label in self._re_running_plan.findChildren(QLabel):
            if label.text().strip().upper() == "RUNNING PLAN":
                if manager_paused and has_running_item:
                    self._set_stylesheet_if_changed(label, self._running_plan_badge_style("paused"))
                elif has_running_item:
                    self._set_stylesheet_if_changed(label, self._running_plan_badge_style("running"))
                else:
                    self._set_stylesheet_if_changed(label, self._running_plan_badge_style("idle"))

    def _running_plan_badge_style(self, state):
        semantic = {
            "paused": "warning",
            "running": "danger",
            "idle": "neutral",
        }.get(state, "neutral")
        weight = 800 if state in {"paused", "running"} else 700
        return self._indicator_badge_style(semantic, font_px=14, weight=weight, padding="1px 8px")

    @staticmethod
    def _format_running_plan_html(html):
        """Mark top-level running-plan section headers for better spacing."""
        if "dc-section-hdr" in html:
            return html
        section_headers = ("Plan Name:", "Arguments:", "Parameters:", "Metadata:", "Runs:")
        updated = html
        for header in section_headers:
            updated = updated.replace(
                f"<b>{header}</b>",
                f"<b class=\"dc-section-hdr\">{header}</b>",
            )
        # Remaining bold labels (parameter keys, metadata labels, etc.) are treated as sub-headings.
        updated = re.sub(
            r"<b>([^<:]+:)</b>",
            r"<b class=\"dc-sub-hdr\">\1</b>",
            updated,
        )
        return updated

    def eventFilter(self, watched, event):
        if (
            event.type() == QtCore.QEvent.Wheel
            and isinstance(watched, QTextEdit)
            and getattr(self, "_re_running_plan", None) is not None
            and self._re_running_plan.isAncestorOf(watched)
            and bool(event.modifiers() & QtCore.Qt.ControlModifier)
        ):
            delta = event.angleDelta().y() or event.pixelDelta().y()
            if delta > 0:
                step = 1
            elif delta < 0:
                step = -1
            else:
                step = 0
            if step:
                min_font_px = getattr(self, "_running_plan_font_min_px", 11)
                max_font_px = getattr(self, "_running_plan_font_max_px", 28)
                current_font_px = getattr(self, "_running_plan_font_px", 15)
                new_size = max(
                    min_font_px,
                    min(max_font_px, current_font_px + step),
                )
                if new_size != current_font_px:
                    self._running_plan_font_px = new_size
                    self._style_re_running_plan_widget()
            return True
        return super().eventFilter(watched, event)

    def _style_re_status_labels(self):
        """Style RE status rows with state-dependent backgrounds."""
        is_connected = bool(getattr(self._re_manager.model, "re_manager_connected", False))
        raw_status_values = {}
        for label in self._re_status.findChildren(QLabel):
            text = label.text().strip()
            if ":" not in text:
                continue
            prefix, raw_value = text.split(":", 1)
            raw_status_values[prefix.strip()] = raw_value.strip().upper()

        short_names = {
            "RE Environment": "Environment",
            "Worker state": "Worker",
            "Manager state": "Manager",
            "RE state": "Engine",
            "Queue AUTOSTART": "Autostart",
            "Queue STOP pending": "Pending",
            "Pending": "Pending",
            "Items in queue": "Queue Items",
            "Queue LOOP mode": "Loop Mode",
        }
        # RE panel color semantics:
        # - Gray: disconnected, unavailable, or environment closed.
        # - Blue-gray: connected and no queued items remain.
        # - Green: queue contains items while idle.
        # - Red: queue contains items while an active run/queue is executing.
        # - Amber: transitional states such as environment creation/startup,
        #   close and destroy.
        # - Error/failure color: high-contrast magenta with white text
        #   (#e300ff background, #ffffff text, #a100b3 border).
        state_semantics = {
            "RUNNING": "danger",
            "EXECUTING_QUEUE": "danger",
            "EXECUTING_PLAN": "danger",
            "EXECUTING_TASK": "danger",
            "PAUSED": "warning",
            "OPENING_ENVIRONMENT": "warning",
            "CREATING_ENVIRONMENT": "warning",
            "INITIALIZING": "warning",
            "CLOSING_ENVIRONMENT": "warning",
            "CLOSING": "warning",
            "DESTROYING_ENVIRONMENT": "warning",
            "IDLE": "success",
            "OPEN": "success",
            "CLOSED": "muted",
            "FAILED": "error",
            "ON": "success",
            "OFF": "muted",
            "YES": "danger",
            "NO": "success",
            "-": "muted",
        }
        env_value = raw_status_values.get("RE Environment", raw_status_values.get("Environment", ""))
        manager_value = raw_status_values.get("Manager state", raw_status_values.get("Manager", ""))
        engine_value = raw_status_values.get("RE state", raw_status_values.get("Engine", ""))
        pending_value = raw_status_values.get("Pending", raw_status_values.get("Queue STOP pending", ""))
        env_closed = env_value == "CLOSED"
        manager_state = manager_value.replace(" ", "_")
        engine_state = engine_value.replace(" ", "_")
        pending_state = pending_value.replace(" ", "_")
        execution_active = (
            manager_state in {"EXECUTING_QUEUE", "PAUSED"}
            or engine_state in {"RUNNING", "PAUSED"}
        )
        for label in self._re_status.findChildren(QLabel):
            text = label.text().strip()
            if ":" not in text:
                continue
            prefix, raw_value = text.split(":", 1)
            prefix = prefix.strip()
            if is_connected:
                value = raw_value.strip().upper()
                value_key = value.replace(" ", "_")
                display_value = value_key.replace("_", " ")
            else:
                # Prevent stale "green" states from persisting after disconnect.
                value_key = "-"
                display_value = "-"
            label_name = short_names.get(prefix, prefix)
            font_px = 14
            weight = 600
            semantic = state_semantics.get(value_key, "neutral")
            if label_name == "Manager" and (
                value_key.startswith("CREATING_")
                or value_key.startswith("OPENING_")
                or value_key.startswith("CLOSING_")
                or value_key.startswith("DESTROYING_")
                or value_key in {"INITIALIZING", "STARTING"}
            ):
                semantic = "warning"
            if label_name == "Pending":
                if (not is_connected) or env_closed:
                    semantic = "muted"
                elif not execution_active:
                    semantic = "neutral"
                elif value_key not in {"NONE", "-", ""}:
                    semantic = "warning"
                else:
                    semantic = "success"
            elif label_name == "Queue Items":
                try:
                    queue_items = int(raw_value.strip())
                except ValueError:
                    queue_items = None

                if (not is_connected) or env_closed or queue_items is None:
                    semantic = "muted"
                elif queue_items <= 0:
                    semantic = "neutral"
                elif manager_state == "PAUSED" or engine_state == "PAUSED":
                    semantic = "warning"
                elif "STOP" in pending_state and pending_state not in {"NONE", "-", ""}:
                    semantic = "warning"
                elif execution_active:
                    semantic = "danger"
                else:
                    semantic = "success"
            elif label_name == "Loop Mode":
                if (not is_connected) or env_closed:
                    semantic = "muted"
                elif value_key == "OFF":
                    semantic = "neutral"
                elif execution_active:
                    semantic = "danger"
                else:
                    semantic = "success"
            elif label_name == "Autostart":
                if (not is_connected) or env_closed:
                    semantic = "muted"
                elif value_key == "OFF":
                    semantic = "neutral"
                else:
                    semantic = "success"
            if label_name == "Manager":
                manager_display_overrides = {
                    "OPENING_ENVIRONMENT": "OPENING ENV",
                    "CREATING_ENVIRONMENT": "CREATING ENV",
                    "CLOSING_ENVIRONMENT": "CLOSING ENV",
                    "DESTROYING_ENVIRONMENT": "DESTROYING ENV",
                }
                display_value = manager_display_overrides.get(value_key, display_value)
            self._set_text_if_changed(label, f"{label_name}: {display_value}")
            self._set_stylesheet_if_changed(
                label,
                self._indicator_badge_style(semantic, font_px=font_px, weight=weight),
            )

    def _compact_re_status_layout(self):
        """Tighten spacing inside status panel to free horizontal space."""
        for layout in self._re_status.findChildren(QHBoxLayout):
            layout.setSpacing(4)
            layout.setContentsMargins(2, 2, 2, 2)
            for i in reversed(range(layout.count())):
                item = layout.itemAt(i)
                spacer = item.spacerItem()
                if spacer is not None and spacer.sizeHint().width() <= 20:
                    layout.takeAt(i)
        for layout in self._re_status.findChildren(QVBoxLayout):
            layout.setSpacing(4)
            layout.setContentsMargins(2, 2, 2, 2)

        # Test inset for RE Manager Status content position verification.
        for group_box in self._re_status.findChildren(QGroupBox):
            if group_box.layout():
                left, _, right, bottom = group_box.layout().getContentsMargins()
                group_box.layout().setContentsMargins(left, 10, right, bottom)

    def _compact_panel_layouts(self):
        """Reduce spacing between top-level RE widgets and inside their containers."""
        pad = 2
        panel_frames = (
            self.ui.RE_Connection,
            self.ui.RE_Worker,
            self.ui.RE_Status,
            self.ui.RE_Running,
            self.ui.RE_Queue_Controls,
            self.ui.RE_Plan_Execution,
        )
        for frame in panel_frames:
            if frame.layout():
                frame.layout().setSpacing(pad)
                frame.layout().setContentsMargins(pad, pad, pad, pad)

        # Keep outer containers flush. Additional outer margins combined with
        # fixed panel heights (200) can clip the bottom of embedded widgets.
        if self.ui.layout():
            self.ui.layout().setSpacing(0)
            self.ui.layout().setContentsMargins(0, 0, 0, 0)

        top_hbox = getattr(self.ui, "horizontalLayout", None)
        if top_hbox is not None:
            top_hbox.setSpacing(0)
            top_hbox.setContentsMargins(0, 0, 0, 0)

        top_grid = getattr(self.ui, "gridLayout", None)
        if top_grid is not None:
            top_grid.setSpacing(0)
            top_grid.setContentsMargins(0, 0, 0, 0)

    def _style_groupbox_titles(self):
        """Make panel titles larger and centered."""
        title_style = (
            "QGroupBox { font-size: 14px; font-weight: 700; margin-top: 8px; } "
            "QGroupBox::title { subcontrol-origin: margin; "
            "subcontrol-position: top center; padding: 0 4px; }"
        )
        for group_box in self.findChildren(QGroupBox):
            group_box.setStyleSheet(title_style)

    def _normalize_panel_heights(self):
        """Keep all RE top-row panels exactly the same height."""
        panel_height = 200
        widget_height = 196
        panel_frames = (
            self.ui.RE_Connection,
            self.ui.RE_Worker,
            self.ui.RE_Status,
            self.ui.RE_Running,
            self.ui.RE_Queue_Controls,
            self.ui.RE_Plan_Execution,
        )
        for frame in panel_frames:
            frame.setFixedHeight(panel_height)
            if frame.layout():
                frame.layout().setAlignment(QtCore.Qt.AlignTop)

        panel_widgets = (
            self._re_manager,
            self._re_environment,
            self._re_status,
            self._re_running_plan,
            self._re_queue_controls,
            self._re_plan_execution,
        )
        for widget in panel_widgets:
            widget.setFixedHeight(widget_height)
            widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            if widget.layout():
                widget.layout().setContentsMargins(1, 1, 1, 1)
                widget.layout().setSpacing(2)

        # Keep all groupboxes on the same normalized spacing model.
        for group_box in self.findChildren(QGroupBox):
            group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            group_box.setMaximumHeight(widget_height)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
        ):
            self._ensure_neutral_tooltip_style()
            self._style_re_connection_status_label()
            self._style_re_queue_state_label()
            self._style_re_status_labels()
            self._style_re_running_plan_widget()

    def prepare_for_detach(self):
        re_manager = getattr(self, "_re_manager", None)
        if re_manager is not None:
            try:
                re_manager._deactivate_updates = True
            except Exception:
                pass

    def _sync_from_model(self):
        re_manager = getattr(self, "_re_manager", None)
        if re_manager is None:
            return

        model = re_manager.model
        is_connected = getattr(model, "re_manager_connected", None)
        status = getattr(model, "re_manager_status", {}) or {}
        running_item = getattr(model, "_running_item", {}) or {}
        run_list = getattr(model, "_run_list", []) or []

        re_manager.slot_update_widgets(is_connected)
        self._re_environment.slot_update_widgets(bool(is_connected), status)
        self._re_status.slot_update_widgets(status)
        self._re_running_plan.slot_running_item_changed(running_item, run_list)
        self._re_running_plan.slot_update_widgets(bool(is_connected), status)
        self._re_queue_controls.slot_update_widgets(bool(is_connected), status)
        self._re_plan_execution.slot_update_widgets(bool(is_connected), status)

        if is_connected and not getattr(re_manager, "updates_activated", False):
            def _resume_status_updates():
                re_manager.updates_activated = True
                re_manager._deactivate_updates = False
                re_manager._first_connection = True
                re_manager._update_widget_states()
                re_manager.slot_update_widgets(True)
                re_manager._start_thread()

            QtCore.QTimer.singleShot(0, _resume_status_updates)

    def customize_ui(self):
        # button = self.ui.pushButton
        # print('Here')
        # button.clicked.connect(self.printstuff)

        from application import MITRApplication

        self._ensure_neutral_tooltip_style()
        app = MITRApplication.instance()
        re_client = app.re_client
        self._apply_re_client_exception_compatibility(re_client)
        # re_client = RunEngineClient(zmq_control_addr='tcp://192.168.0.14:60615')
        re_manager = QtReManagerConnection(re_client)
        re_environment = _NonBlockingQtReEnvironmentControls(re_client)
        re_status = _WorkerAwareQtReStatusMonitor(re_client)
        re_running_plan = QtReRunningPlan(re_client)
        re_queue_controls = QtReQueueControls(re_client)
        re_plan_execution = QtReExecutionControls(re_client)

        self.ui.RE_Connection.layout().addWidget(re_manager)
        self.ui.RE_Worker.layout().addWidget(re_environment)
        self.ui.RE_Status.layout().addWidget(re_status)
        self.ui.RE_Running.layout().addWidget(re_running_plan)
        self.ui.RE_Queue_Controls.layout().addWidget(re_queue_controls)
        self.ui.RE_Plan_Execution.layout().addWidget(re_plan_execution)

        self._re_manager = re_manager
        self._re_environment = re_environment
        self._re_status = re_status
        self._re_running_plan = re_running_plan
        self._re_queue_controls = re_queue_controls
        self._re_plan_execution = re_plan_execution

        for label in self._re_manager.findChildren(QLabel):
            self._patch_label_set_text(label, "_style_re_connection_status_label")
        for label in self._re_queue_controls.findChildren(QLabel):
            self._patch_label_set_text(label, "_style_re_queue_state_label")
        for label in self._re_status.findChildren(QLabel):
            self._patch_label_set_text(label, "_style_re_status_labels", "_style_re_running_plan_widget")
        for text_edit in self._re_running_plan.findChildren(QTextEdit):
            text_edit.textChanged.connect(
                lambda: self._schedule_restyle("_style_re_running_plan_widget")
            )

        self._compact_panel_layouts()
        self._normalize_panel_heights()
        self._style_groupbox_titles()
        self._style_re_connection_status_label()
        self._style_re_queue_state_label()
        self._compact_re_status_layout()
        self._style_re_status_labels()
        self._style_re_running_plan_widget()
        self._sync_from_model()

    @staticmethod
    def _apply_re_client_exception_compatibility(re_client):
        """Bridge API exception-name differences across Queue Server versions."""
        client_cls = type(getattr(re_client, "_client", None))
        if client_cls is type(None):
            return
        if (not hasattr(client_cls, "RequestError")) and hasattr(client_cls, "RequestFailedError"):
            client_cls.RequestError = client_cls.RequestFailedError
        if (not hasattr(client_cls, "ClientError")) and hasattr(client_cls, "HTTPClientError"):
            client_cls.ClientError = client_cls.HTTPClientError


    # def printstuff():
    #     print("button pressed")


class _NonBlockingQtReEnvironmentControls(QtReEnvironmentControls):
    """Issue env open/close/destroy requests without blocking the GUI thread."""

    def __init__(self, model, parent=None):
        super().__init__(model, parent=parent)
        self._pb_env_open.clicked.disconnect()
        self._pb_env_open.clicked.connect(self._pb_env_open_clicked)
        self._pb_env_close.clicked.disconnect()
        self._pb_env_close.clicked.connect(self._pb_env_close_clicked)
        self._pb_env_destroy.clicked.disconnect()
        self._pb_env_destroy.clicked.connect(self._pb_env_destroy_clicked)

    def _refresh_status_soon(self):
        QtCore.QTimer.singleShot(0, lambda: self.model.load_re_manager_status(unbuffered=True))
        QtCore.QTimer.singleShot(250, lambda: self.model.load_re_manager_status(unbuffered=True))

    def _pb_env_open_clicked(self):
        try:
            self.model._client.environment_open()
            self.model.activate_env_destroy(False)
            self._refresh_status_soon()
        except Exception as ex:
            print(f"Exception: {ex}")

    def _pb_env_close_clicked(self):
        try:
            self.model._client.environment_close()
            self._refresh_status_soon()
        except Exception as ex:
            print(f"Exception: {ex}")

    def _pb_env_destroy_clicked(self):
        try:
            if not self.model.env_destroy_activated:
                raise RuntimeError("'Destroy Environment' operation is not activated and can not be executed")
            self.model._client.environment_destroy()
            self.model.activate_env_destroy(False)
            self._refresh_status_soon()
        except Exception as ex:
            print(f"Exception: {ex}")


class _WorkerAwareQtReStatusMonitor(QtReStatusMonitor):
    """Repurpose the history field to show worker environment state."""

    def __init__(self, model, parent=None):
        super().__init__(model, parent=parent)
        self._lb_items_in_history_text = "Worker state: "
        self._lb_items_in_history.setText(self._lb_items_in_history_text + "-")
        self._lb_queue_stop_pending_text = "Pending: "
        self._lb_queue_stop_pending.setText(self._lb_queue_stop_pending_text + "-")
        self._install_indicator_tooltips()

    @staticmethod
    def _tooltip_text(title, description, readouts):
        readouts_wrapped = textwrap.fill(
            " | ".join(readouts),
            width=40,
            subsequent_indent="",
            break_long_words=False,
            break_on_hyphens=False,
        )
        description_wrapped = textwrap.fill(
            description,
            width=48,
            break_long_words=False,
            break_on_hyphens=False,
        )
        return f"{title}\n{description_wrapped}\n\nPossible readouts:\n{readouts_wrapped}"

    def _install_indicator_tooltips(self):
        tooltips = {
            self._lb_environment_exists: self._tooltip_text(
                "RE Environment",
                "Whether the RE Worker environment currently exists.",
                ["OPEN", "CLOSED"],
            ),
            self._lb_items_in_history: self._tooltip_text(
                "Worker State",
                "State of the RE Worker environment process.",
                ["INITIALIZING", "IDLE", "EXECUTING PLAN", "EXECUTING TASK", "CLOSING", "FAILED", "CLOSED"],
            ),
            self._lb_manager_state: self._tooltip_text(
                "Manager State",
                "State of the Queue Server manager.",
                [
                    "INITIALIZING",
                    "IDLE",
                    "PAUSED",
                    "CREATING ENVIRONMENT",
                    "STARTING QUEUE",
                    "EXECUTING QUEUE",
                    "EXECUTING TASK",
                    "CLOSING ENVIRONMENT",
                    "DESTROYING ENVIRONMENT",
                ],
            ),
            self._lb_re_state: self._tooltip_text(
                "RE State",
                "State of the Bluesky Run Engine inside the worker.",
                ["IDLE", "RUNNING", "PAUSING", "PAUSED", "STOPPING", "ABORTING", "HALTING", "SUSPENDING", "PANICKED"],
            ),
            self._lb_queue_autostart_enabled: self._tooltip_text(
                "Queue AUTOSTART",
                "Whether the queue starts the next available item automatically.",
                ["ON", "OFF", "-"],
            ),
            self._lb_queue_stop_pending: self._tooltip_text(
                "Pending Action",
                "Local combined indicator for Queue STOP pending and Pause pending.",
                ["NONE", "PAUSE", "STOP", "PAUSE+STOP", "-"],
            ),
            self._lb_queue_loop_mode: self._tooltip_text(
                "Queue LOOP Mode",
                "Whether the whole queue is configured to repeat by moving completed items "
                "to the back.",
                ["ON", "OFF"],
            ),
            self._lb_items_in_queue: self._tooltip_text(
                "Items in Queue",
                "Number of queued items still waiting in the plan queue.",
                ["0", "1", "2", "..."],
            ),
        }
        for widget, tooltip in tooltips.items():
            widget.setProperty("_dc_tooltip_text", tooltip)
            widget.setToolTip("")
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.ToolTip:
            tooltip_text = watched.property("_dc_tooltip_text")
            if tooltip_text:
                QToolTip.showText(event.globalPos(), str(tooltip_text))
                return True
            QToolTip.hideText()
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def slot_update_widgets(self, status):
        worker_exists = status.get("worker_environment_exists", None)
        worker_state = status.get("worker_environment_state", None)
        manager_state = status.get("manager_state", None)
        re_state = status.get("re_state", None)
        items_in_queue = status.get("items_in_queue", None)
        queue_autostart_enabled = bool(status.get("queue_autostart_enabled", False))
        queue_stop_pending = status.get("queue_stop_pending", None)
        pause_pending = bool(status.get("pause_pending", False))

        queue_mode = status.get("plan_queue_mode", None)
        queue_loop_enabled = queue_mode.get("loop", None) if queue_mode else None

        worker_state = worker_state.upper() if isinstance(worker_state, str) else worker_state
        manager_state = manager_state.upper() if isinstance(manager_state, str) else manager_state
        re_state = re_state.upper() if isinstance(re_state, str) else re_state

        self._set_label_text(
            self._lb_environment_exists,
            self._lb_environment_exists_text,
            "OPEN" if worker_exists else "CLOSED",
        )
        self._set_label_text(self._lb_items_in_history, self._lb_items_in_history_text, worker_state)
        self._set_label_text(self._lb_manager_state, self._lb_manager_state_text, manager_state)
        self._set_label_text(self._lb_re_state, self._lb_re_state_text, re_state)
        self._set_label_text(self._lb_items_in_queue, self._lb_items_in_queue_text, str(items_in_queue))
        autostart_text = "ON" if queue_autostart_enabled else "OFF"
        pending_parts = []
        if pause_pending:
            pending_parts.append("PAUSE")
        if queue_stop_pending:
            pending_parts.append("STOP")
        pending_text = "+".join(pending_parts) if pending_parts else "NONE"
        if not worker_exists:
            autostart_text = "-"
            pending_text = "-"
        self._set_label_text(
            self._lb_queue_autostart_enabled,
            self._lb_queue_autostart_enabled_text,
            autostart_text,
        )
        self._set_label_text(
            self._lb_queue_stop_pending,
            self._lb_queue_stop_pending_text,
            pending_text,
        )
        self._set_label_text(
            self._lb_queue_loop_mode,
            self._lb_queue_loop_mode_text,
            "ON" if queue_loop_enabled else "OFF",
        )
