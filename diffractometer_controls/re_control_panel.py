import re
import types

from qtpy import QtCore
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSizePolicy, QTextEdit, QPushButton
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
        pal = self._current_palette()
        is_dark = pal.color(QPalette.Window).lightness() < 128
        if is_dark:
            palette = {
                "paused": ("#fbbf24", "#3f2d09", "#b45309"),
                "running": ("#f87171", "#3f1114", "#b91c1c"),
                "idle": ("#cbd5e1", "#1e293b", "#475569"),
            }
        else:
            palette = {
                "paused": ("#fef3c7", "#92400e", "#fcd34d"),
                "running": ("#fee2e2", "#7f1d1d", "#fecaca"),
                "idle": ("#e2e8f0", "#334155", "#cbd5e1"),
            }
        bg, fg, border = palette.get(state, palette["idle"])
        weight = 800 if state in {"paused", "running"} else 700
        return (
            f"font-size: 15px; font-weight: {weight}; color: {fg}; "
            f"background-color: {bg}; border: 1px solid {border}; "
            "border-radius: 6px; padding: 2px 8px;"
        )

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
        short_names = {
            "RE Environment": "Environment",
            "Manager state": "Manager",
            "RE state": "Engine",
            "Items in history": "History",
            "Queue AUTOSTART": "Autostart",
            "Queue STOP pending": "Stop Pending",
            "Items in queue": "Queue Items",
            "Queue LOOP mode": "Loop Mode",
        }
        base_style = (
            "font-size: 13px; font-weight: 600; padding: 0px 4px; "
            "border: 1px solid #d1d5db; border-radius: 6px;"
        )
        emphasis_style = (
            "font-size: 14px; font-weight: 700; padding: 0px 4px; "
            "border: 1px solid #d1d5db; border-radius: 6px;"
        )
        state_styles = {
            "RUNNING": "color: #7f1d1d; background-color: #fee2e2;",
            "EXECUTING_QUEUE": "color: #7f1d1d; background-color: #fee2e2;",
            "PAUSED": "color: #92400e; background-color: #fef3c7;",
            "IDLE": "color: #065f46; background-color: #d1fae5;",
            "OPEN": "color: #065f46; background-color: #d1fae5;",
            "CLOSED": "color: #6b7280; background-color: #f3f4f6;",
            "ON": "color: #065f46; background-color: #d1fae5;",
            "OFF": "color: #6b7280; background-color: #f3f4f6;",
            "YES": "color: #7f1d1d; background-color: #fee2e2;",
            "NO": "color: #065f46; background-color: #d1fae5;",
            "-": "color: #6b7280; background-color: #f9fafb;",
        }
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
            if label_name in ("Manager", "Engine"):
                text_style = emphasis_style
            else:
                text_style = base_style
            state_style = state_styles.get(value_key, "color: #111827; background-color: #eef2ff;")
            if label_name == "Stop Pending" and value_key == "YES":
                state_style = "color: #92400e; background-color: #fef3c7;"
            self._set_text_if_changed(label, f"{label_name}: {display_value}")
            self._set_stylesheet_if_changed(label, f"{text_style} {state_style}")

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
            self._style_re_connection_status_label()
            self._style_re_queue_state_label()
            self._style_re_status_labels()
            self._style_re_running_plan_widget()

    def customize_ui(self):
        # button = self.ui.pushButton
        # print('Here')
        # button.clicked.connect(self.printstuff)

        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client
        # re_client = RunEngineClient(zmq_control_addr='tcp://192.168.0.14:60615')
        re_manager = QtReManagerConnection(re_client)
        re_environment = QtReEnvironmentControls(re_client)
        re_status = QtReStatusMonitor(re_client)
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


    # def printstuff():
    #     print("button pressed")
