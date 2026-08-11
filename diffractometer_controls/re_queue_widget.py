import copy
import threading
from pathlib import Path
import sys
import time
from datetime import datetime

from qtpy import QtCore
from qtpy.QtCore import Qt, Slot
from qtpy.QtWidgets import QGridLayout, QLabel, QHeaderView, QTableWidgetItem, QWidget

from bluesky_widgets.qt.run_engine_client import QtRePlanQueue

try:
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
        format_estimated_time,
        format_plan_summary,
    )
except ModuleNotFoundError:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from diffractometer_controls.plan_time_estimation import (
        build_estimation_context,
        estimate_plan_runtime,
        format_estimated_time,
        format_plan_summary,
    )

try:
    from epics import caget
except Exception:
    caget = None


class QtRePlanQueueEstimated(QtRePlanQueue):
    signal_estimates_ready = QtCore.Signal(int, object)
    _QUEUE_PARAMS_PER_LINE = 2

    def __init__(self, model, parent=None):
        super().__init__(model, parent=parent)
        self._table_column_labels = ("", "Name", "Parameters", "Est. Time", "USER", "GROUP")
        self._table.setColumnCount(len(self._table_column_labels))
        self._table.setHorizontalHeaderLabels(self._table_column_labels)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setStretchLastSection(True)

        self._estimate_cache = {}
        self._running_estimate = None
        self._allowed_param_names_cache = {}
        self._completion_total_s = None
        self._completion_includes_running = False
        self._estimate_request_id = 0
        self._pending_estimate_items = []
        self._pending_running_item = {}
        self._estimate_timer = QtCore.QTimer(self)
        self._estimate_timer.setSingleShot(True)
        self._estimate_timer.timeout.connect(self._start_estimate_thread)
        self._completion_refresh_timer = QtCore.QTimer(self)
        self._completion_refresh_timer.setInterval(30000)
        self._completion_refresh_timer.timeout.connect(self._refresh_completion_label)
        self._completion_refresh_timer.start()
        self.signal_estimates_ready.connect(self._on_estimates_ready)

        self._footer_widget = QWidget(self)
        self._total_time_label = QLabel("Total Est. Time: --", self._footer_widget)
        self._completion_label = QLabel("Est. Completion: --", self._footer_widget)
        footer_layout = QGridLayout(self._footer_widget)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setHorizontalSpacing(16)
        footer_layout.setVerticalSpacing(0)
        footer_layout.addWidget(self._total_time_label, 0, 0, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        footer_layout.addWidget(self._completion_label, 0, 1, alignment=Qt.AlignLeft | Qt.AlignVCenter)
        footer_layout.setColumnStretch(0, 1)
        footer_layout.setColumnStretch(1, 1)
        layout = self.layout()
        if layout is not None:
            layout.addWidget(self._footer_widget)

        self._running_item = dict(getattr(self.model, "_running_item", {}) or {})
        events = getattr(self.model, "events", None)
        if events is not None:
            running_item_changed = getattr(events, "running_item_changed", None)
            status_changed = getattr(events, "status_changed", None)
            if running_item_changed is not None:
                try:
                    running_item_changed.connect(self._on_model_running_item_changed)
                except Exception:
                    pass
            if status_changed is not None:
                try:
                    status_changed.connect(self._on_model_status_changed)
                except Exception:
                    pass

    @staticmethod
    def _format_completion_time(epoch_s):
        if epoch_s is None:
            return "--"
        try:
            dt = datetime.fromtimestamp(float(epoch_s))
        except Exception:
            return "--"
        now = datetime.now()
        if dt.date() == now.date():
            return dt.strftime("%I:%M %p").lstrip("0")
        return dt.strftime("%b %d %I:%M %p").replace(" 0", " ")

    def _current_running_item(self):
        return dict(getattr(self, "_running_item", {}) or {})

    @staticmethod
    def _normalize_item(item):
        if hasattr(item, "to_dict"):
            try:
                item = item.to_dict()
            except Exception:
                item = {}
        return item if isinstance(item, dict) else {}

    def _allowed_plan_parameter_names(self, plan_name):
        plan_name = str(plan_name or "")
        if not plan_name:
            return ()
        if plan_name in self._allowed_param_names_cache:
            return self._allowed_param_names_cache[plan_name]

        param_names = []
        try:
            item_params = self.model.get_allowed_plan_parameters(name=plan_name) or {}
        except Exception:
            item_params = {}

        for param in list(item_params.get("parameters", []) or []):
            if not isinstance(param, dict):
                continue
            name = str(param.get("name", "") or "")
            if name:
                param_names.append(name)

        result = tuple(param_names)
        self._allowed_param_names_cache[plan_name] = result
        return result

    def _normalize_plan_kwargs(self, item):
        item_dict = self._normalize_item(item)
        params = dict(item_dict.get("kwargs", {}) or {})
        if str(item_dict.get("item_type", "") or "plan") != "plan":
            return params

        param_names = self._allowed_plan_parameter_names(item_dict.get("name", ""))
        if not param_names:
            return params

        for index, value in enumerate(list(item_dict.get("args", ()) or ())):
            if index >= len(param_names):
                break
            param_name = param_names[index]
            if param_name not in params:
                params[param_name] = value
        return params

    def _schedule_estimate_update(self, plan_queue_items):
        self._pending_estimate_items = copy.deepcopy(list(plan_queue_items or []))
        self._pending_running_item = copy.deepcopy(self._current_running_item())
        self._estimate_request_id += 1
        self._estimate_timer.start(75)

    @classmethod
    def _wrap_parameter_text(cls, text):
        text = str(text or "").strip()
        if not text:
            return ""

        text = " ".join(text.split())
        parts = []
        for chunk in text.split(", "):
            if ": " in chunk:
                key, value = chunk.split(": ", 1)
                parts.append(f"{key}: {value}")
            else:
                parts.append(chunk)

        if len(parts) <= 1:
            return parts[0] if parts else text

        lines = []
        step = max(int(cls._QUEUE_PARAMS_PER_LINE), 1)
        for index in range(0, len(parts), step):
            lines.append(" | ".join(parts[index:index + step]))
        return "\n".join(lines)

    def _refresh_parameter_column(self):
        table = getattr(self, "_table", None)
        labels = getattr(self, "_table_column_labels", ())
        model = getattr(self, "model", None)
        items = getattr(self, "_plan_queue_items", ())
        if table is None or model is None or not labels:
            return
        try:
            parameters_col = labels.index("Parameters")
        except ValueError:
            return
        try:
            name_col = labels.index("Name")
        except ValueError:
            name_col = None

        fm = table.fontMetrics()
        params_width = fm.horizontalAdvance("Parameters") + 28
        for row, item in enumerate(items):
            try:
                raw_text = model.get_item_value_for_label(item=item, label="Parameters")
            except Exception:
                raw_text = ""
            wrapped_text = self._wrap_parameter_text(raw_text)
            table_item = table.item(row, parameters_col)
            if table_item is None:
                continue
            table_item.setText(wrapped_text)
            tooltip = raw_text
            item_dict = self._normalize_item(item)
            plan_name = str(item_dict.get("name", ""))
            if str(item_dict.get("item_type", "") or "plan") == "plan" and plan_name:
                item_uid = str(item_dict.get("item_uid", ""))
                estimate = self._estimate_cache.get(item_uid, {}) or {}
                summary = format_plan_summary(
                    plan_name,
                    self._normalize_plan_kwargs(item_dict),
                    estimated_time_s=estimate.get("estimated_total_time_s"),
                )
                tooltip = summary
                if raw_text:
                    tooltip += "\n\nParameters:\n" + raw_text
                if name_col is not None:
                    name_item = table.item(row, name_col)
                    if name_item is not None:
                        name_item.setToolTip(summary)
            table_item.setToolTip(tooltip)
            table_item.setTextAlignment(int(Qt.AlignLeft | Qt.AlignTop))
            line_width = 0
            for line in wrapped_text.splitlines() or [""]:
                line_width = max(line_width, fm.horizontalAdvance(line))
            params_width = max(params_width, line_width + 20)

        table.setColumnWidth(parameters_col, params_width)
        table.resizeRowsToContents()

    def _start_estimate_thread(self):
        request_id = int(self._estimate_request_id)
        items = copy.deepcopy(list(self._pending_estimate_items or []))
        running_item = copy.deepcopy(dict(self._pending_running_item or {}))

        def _worker():
            context = build_estimation_context(caget_func=caget)
            queue_results = []
            for item in items:
                item_dict = self._normalize_item(item)
                estimate = estimate_plan_runtime(
                    str(item_dict.get("name", "")),
                    kwargs=self._normalize_plan_kwargs(item_dict),
                    context=context,
                )
                queue_results.append(
                    {
                        "item_uid": str(item_dict.get("item_uid", "")),
                        "estimated_total_time_s": estimate.get("estimated_total_time_s"),
                        "estimated_total_units": estimate.get("estimated_total_units"),
                    }
                )
            running_result = None
            if isinstance(running_item, dict) and str(running_item.get("item_type", "") or "plan") == "plan":
                running_result = estimate_plan_runtime(
                    str(running_item.get("name", "")),
                    kwargs=self._normalize_plan_kwargs(running_item),
                    context=context,
                )
            self.signal_estimates_ready.emit(
                request_id,
                {
                    "queue_results": queue_results,
                    "running_item_uid": str(running_item.get("item_uid", "")),
                    "running_estimate": running_result,
                },
            )

        thread = threading.Thread(
            target=_worker,
            name=f"queue-estimate-{request_id}",
            daemon=True,
        )
        thread.start()

    def _refresh_completion_label(self):
        if self._completion_total_s is None:
            self._completion_label.setText("Est. Completion: --")
            self._completion_label.setToolTip("No completion estimate available")
            return

        self._completion_label.setText(
            f"Est. Completion: {self._format_completion_time(time.time() + float(self._completion_total_s))}"
        )
        self._completion_label.setToolTip(
            "Estimated finish based on queued items"
            + (" and the current running item." if self._completion_includes_running else ".")
        )

    def _apply_estimates_to_table(self):
        try:
            est_col = self._table_column_labels.index("Est. Time")
        except ValueError:
            return

        total_estimate = 0.0
        valid_estimates = 0
        for row, item in enumerate(list(self._plan_queue_items or [])):
            if not isinstance(item, dict):
                continue
            item_uid = str(item.get("item_uid", ""))
            estimate = self._estimate_cache.get(item_uid, None)
            time_s = None
            if isinstance(estimate, dict):
                time_s = estimate.get("estimated_total_time_s", None)

            table_item = self._table.item(row, est_col)
            if table_item is None:
                table_item = QTableWidgetItem("")
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(row, est_col, table_item)

            table_item.setText(format_estimated_time(time_s))
            if time_s is None:
                table_item.setToolTip("No estimate available")
            else:
                table_item.setToolTip(f"{float(time_s):.1f} s")
                total_estimate += float(time_s)
                valid_estimates += 1

        if valid_estimates:
            self._total_time_label.setText(f"Total Est. Time: {format_estimated_time(total_estimate)}")
        else:
            self._total_time_label.setText("Total Est. Time: --")

        running_time_s = None
        running_estimate = self._running_estimate if isinstance(self._running_estimate, dict) else None
        if running_estimate is not None:
            running_time_s = running_estimate.get("estimated_total_time_s", None)

        completion_total_s = total_estimate
        has_completion_estimate = valid_estimates > 0
        if running_time_s is not None:
            completion_total_s += float(running_time_s)
            has_completion_estimate = True

        if has_completion_estimate:
            self._completion_total_s = float(completion_total_s)
            self._completion_includes_running = running_time_s is not None
        else:
            self._completion_total_s = None
            self._completion_includes_running = False
        self._refresh_completion_label()

    @Slot(object, object)
    def slot_plan_queue_changed(self, plan_queue_items, selected_item_uids):
        super().slot_plan_queue_changed(plan_queue_items, selected_item_uids)
        self._refresh_parameter_column()
        self._apply_estimates_to_table()
        self._schedule_estimate_update(plan_queue_items)

    def _on_model_running_item_changed(self, *_args, **_kwargs):
        self._running_item = dict(getattr(self.model, "_running_item", {}) or {})
        self._apply_estimates_to_table()
        self._schedule_estimate_update(getattr(self, "_plan_queue_items", []) or [])

    def _on_model_status_changed(self, *_args, **_kwargs):
        self._running_item = dict(getattr(self.model, "_running_item", {}) or {})
        model_queue_items = list(getattr(self.model, "_plan_queue_items", []) or [])
        if model_queue_items and not list(getattr(self, "_plan_queue_items", []) or []):
            selected_uids = list(getattr(self.model, "selected_queue_item_uids", []) or [])
            self.slot_plan_queue_changed(model_queue_items, selected_uids)
            return
        self._apply_estimates_to_table()
        self._schedule_estimate_update(getattr(self, "_plan_queue_items", []) or [])

    @Slot(bool)
    def slot_update_widgets(self, is_connected):
        super().slot_update_widgets(is_connected)
        if not is_connected:
            return
        self._running_item = dict(getattr(self.model, "_running_item", {}) or {})
        model_queue_items = list(getattr(self.model, "_plan_queue_items", []) or [])
        selected_uids = list(getattr(self.model, "selected_queue_item_uids", []) or [])
        if model_queue_items:
            self.slot_plan_queue_changed(model_queue_items, selected_uids)
        else:
            self._apply_estimates_to_table()
            self._schedule_estimate_update(model_queue_items)

    @Slot(int, object)
    def _on_estimates_ready(self, request_id, results):
        if int(request_id) != int(self._estimate_request_id):
            return
        results = dict(results or {})
        cache = {}
        for result in list(results.get("queue_results", []) or []):
            if not isinstance(result, dict):
                continue
            item_uid = str(result.get("item_uid", ""))
            if item_uid:
                cache[item_uid] = dict(result)
        self._estimate_cache = cache
        running_item_uid = str(self._current_running_item().get("item_uid", ""))
        result_running_uid = str(results.get("running_item_uid", ""))
        self._running_estimate = results.get("running_estimate", None) if running_item_uid == result_running_uid else None
        self._refresh_parameter_column()
        self._apply_estimates_to_table()
