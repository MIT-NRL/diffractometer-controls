import re

from qtpy import QtCore
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QSizePolicy, QTextEdit, QPushButton
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
    def __init__(self, parent=None, args=None, macros=None, ui_filename='re_control_panel.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("REControlPanel here")
        # self.customize_ui()

    def ui_filename(self):
        return 're_control_panel.ui'

    def ui_filepath(self):
        return super().ui_filepath()

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
                label.setStyleSheet(status_styles[text])

    def _style_re_queue_state_label(self):
        """Emphasize queue run state in Queue Controls."""
        queue_state_styles = {
            "RUNNING": "font-size: 24px; font-weight: 800; color: #b00020;",
            "STOPPED": "font-size: 24px; font-weight: 800; color: #6b7280;",
        }
        for label in self._re_queue_controls.findChildren(QLabel):
            text = label.text().strip().upper()
            if text in queue_state_styles:
                label.setStyleSheet(queue_state_styles[text])

    def _style_re_running_plan_widget(self):
        """Improve readability of the Running Plan panel."""
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
            palette = text_edit.palette()
            base_color = palette.color(QPalette.Base).name()
            text_color = palette.color(QPalette.Text).name()
            border_color = palette.color(QPalette.Mid).name()
            muted_color = palette.color(QPalette.Mid).name()
            text_edit.setStyleSheet(
                "QTextEdit {"
                "font-size: 13px; "
                "padding: 4px; "
                f"border: 1px solid {border_color}; "
                "border-radius: 6px; "
                f"background-color: {base_color}; "
                f"color: {text_color};"
                "}"
            )
            text_edit.document().setDefaultStyleSheet(
                f"body {{ line-height: 1.45; color: {text_color}; }} "
                f"b {{ color: {text_color}; font-weight: 700; }} "
                "b.dc-section-hdr { "
                "display: inline-block; "
                "margin-top: 8px; "
                "margin-bottom: 3px; "
                "font-size: 13px; "
                f"color: {text_color}; "
                "} "
                "b.dc-sub-hdr { "
                "font-size: 12px; "
                f"color: {muted_color}; "
                "font-weight: 600; "
                "}"
            )
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
                    label.setStyleSheet(
                        "font-size: 15px; font-weight: 800; color: #92400e; "
                        "background-color: #fef3c7; border: 1px solid #fcd34d; "
                        "border-radius: 6px; padding: 2px 8px;"
                    )
                elif has_running_item:
                    label.setStyleSheet(
                        "font-size: 15px; font-weight: 800; color: #7f1d1d; "
                        "background-color: #fee2e2; border: 1px solid #fecaca; "
                        "border-radius: 6px; padding: 2px 8px;"
                    )
                else:
                    label.setStyleSheet(
                        "font-size: 15px; font-weight: 700; color: #334155; "
                        "background-color: #e2e8f0; border: 1px solid #cbd5e1; "
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
            label.setText(f"{label_name}: {display_value}")
            label.setStyleSheet(f"{text_style} {state_style}")

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

        # Manager widget can report a larger implicit height due to inner groupbox margins.
        for group_box in self._re_manager.findChildren(QGroupBox):
            group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            group_box.setMaximumHeight(widget_height)
            # Keep controls clear of the groupbox title area.
            if group_box.layout():
                group_box.layout().setContentsMargins(6, 10, 6, 4)
                group_box.layout().setSpacing(4)

        # Lower status indicator rows within "RE Manager Status" groupbox.
        for group_box in self._re_status.findChildren(QGroupBox):
            group_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            group_box.setMaximumHeight(widget_height)
            if group_box.layout():
                left, _, right, bottom = group_box.layout().getContentsMargins()
                group_box.layout().setContentsMargins(left, 20, right, bottom)

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
        self._re_connection_status_timer = QtCore.QTimer(self)
        self._re_connection_status_timer.timeout.connect(
            self._style_re_connection_status_label
        )
        self._re_connection_status_timer.timeout.connect(
            self._style_re_queue_state_label
        )
        self._re_connection_status_timer.timeout.connect(
            self._style_re_status_labels
        )
        self._re_connection_status_timer.timeout.connect(
            self._style_re_running_plan_widget
        )
        self._re_connection_status_timer.start(500)
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
