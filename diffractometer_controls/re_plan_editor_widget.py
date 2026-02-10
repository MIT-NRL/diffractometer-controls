import ast
import inspect
from qtpy.QtWidgets import QAbstractItemView, QComboBox, QTableWidgetItem

import bluesky_widgets.qt.run_engine_client as rec


class RePlanEditorTable(rec._QtRePlanEditorTable):
    """Table subclass that renders dropdowns for parameters when the
    plan metadata exposes choices via 'values' or 'devices'.
    """

    def _get_param_meta(self, p_name):
        # Try to obtain parameter metadata from the model's allowed plans
        try:
            item_name = self._queue_item.get("name", None) if self._queue_item else None
            if not item_name:
                return {}
            item_type = self._queue_item.get("item_type", "plan")
            if item_type == "plan":
                item_params = self.model.get_allowed_plan_parameters(name=item_name) or {}
            else:
                item_params = self.model.get_allowed_instruction_parameters(name=item_name) or {}
            return self._find_param_meta(item_params, p_name)
        except Exception:
            return {}

    @staticmethod
    def _find_param_meta(item_params, p_name):
        """
        Return parameter metadata for name `p_name` from allowed plan/instruction
        payloads. The payload uses a list of dicts (qserver), but we also accept
        dict-mapped shapes for compatibility.
        """
        if not isinstance(item_params, dict):
            return {}
        params = item_params.get("parameters", None)
        if isinstance(params, list):
            for p in params:
                try:
                    if isinstance(p, dict) and p.get("name") == p_name:
                        return p
                except Exception:
                    continue
            return {}
        if isinstance(params, dict):
            return params.get(p_name, {}) or {}
        return {}

    def _show_row_value(self, *, row):
        # Based on original implementation but uses metadata from the model
        def print_value(v):
            if isinstance(v, str):
                return f"'{v}'"
            else:
                return str(v)

        p = self._params[row]
        p_name = p["name"]
        value = p["value"]
        default_value = p["parameters"].default
        is_var_positional = p["parameters"].kind == inspect.Parameter.VAR_POSITIONAL
        is_var_keyword = p["parameters"].kind == inspect.Parameter.VAR_KEYWORD
        is_value_set = p["is_value_set"]
        is_editable = self._editable and (is_value_set or not ((default_value != inspect.Parameter.empty) or is_var_positional or is_var_keyword))

        description = self._params_descriptions.get("parameters", {}).get(p_name, None)
        if not description:
            description = f"Description for parameter '{p_name}' was not found ..."

        v = value if is_value_set else default_value
        s_value = "" if v == inspect.Parameter.empty else print_value(v)
        if not is_value_set and s_value:
            s_value += " (default)"

        # Set checkable item in column 1
        check_item = QTableWidgetItem()
        check_item.setFlags(check_item.flags() | rec.Qt.ItemIsUserCheckable)
        if default_value == inspect.Parameter.empty and not is_var_positional and not is_var_keyword:
            check_item.setFlags(check_item.flags() & ~rec.Qt.ItemIsEnabled)
            check_item.setCheckState(rec.Qt.Checked)
        else:
            if self._editable:
                check_item.setFlags(check_item.flags() | rec.Qt.ItemIsEnabled)
            else:
                check_item.setFlags(check_item.flags() & ~rec.Qt.ItemIsEnabled)

            check_item.setCheckState(rec.Qt.Checked if is_value_set else rec.Qt.Unchecked)

        self.setItem(row, 1, check_item)

        # Determine choices from parameter metadata. Only create dropdowns when
        # the plan/decorator provides an explicit 'devices' or 'values' list for
        # this parameter. Prefer metadata from the decorator (`meta`) and then
        # fall back to the model-provided allowed-plan parameters.
        meta = self._get_param_meta(p_name) or {}
        choices = None

        # Attempt to get model-level parameter metadata as a fallback
        item_name = self._queue_item.get("name", None) if self._queue_item else None
        try:
            item_params = (
                self.model.get_allowed_plan_parameters(name=item_name)
                if item_name
                else {}
            ) or {}
            pmeta = self._find_param_meta(item_params, p_name)
        except Exception:
            pmeta = {}

        # Helper to extract string names from the 'devices' metadata which may
        # be provided as a dict (mapping), a list/tuple, or a single string.
        def _extract_devices_field(field):
            if field is None:
                return []
            if isinstance(field, dict):
                vals = []
                for v in field.values():
                    if isinstance(v, (list, tuple)):
                        vals.extend(map(str, v))
                    else:
                        try:
                            vals.append(str(v))
                        except Exception:
                            continue
                return vals
            if isinstance(field, (list, tuple)):
                return [str(x) for x in field]
            # fallback: single value (string or object)
            return [str(field)]

        # Accept 'values' or 'devices' defined either at the top-level of the
        # parameter metadata or nested under 'annotation' (as produced by
        # bluesky-queueserver). Normalize both places into local variables.
        values_field = meta.get("values") if isinstance(meta, dict) else None
        if values_field is None:
            # Check under 'annotation' in item parameter metadata
            values_field = (pmeta.get("annotation") or {}).get("values") if isinstance(pmeta, dict) else None

        devices_field = None
        if isinstance(meta, dict):
            devices_field = meta.get("devices")
        if devices_field is None and isinstance(pmeta, dict):
            # Some transports nest the 'devices' under 'annotation'
            devices_field = (pmeta.get("annotation") or {}).get("devices")

        pmeta_devices_field = None
        if isinstance(pmeta, dict):
            pmeta_devices_field = pmeta.get("devices") or (pmeta.get("annotation") or {}).get("devices")

        def _is_bool_param():
            # Use annotation/type metadata first, then fall back to value/default types
            try:
                ann = (pmeta.get("annotation") or {}) if isinstance(pmeta, dict) else {}
                ann_type = ann.get("type")
                if ann_type is bool or ann_type == "bool":
                    return True
            except Exception:
                pass
            if isinstance(value, bool):
                return True
            if default_value is not inspect.Parameter.empty and isinstance(default_value, bool):
                return True
            return False

        def _expected_type():
            try:
                ann = (pmeta.get("annotation") or {}) if isinstance(pmeta, dict) else {}
                ann_type = ann.get("type")
                if ann_type in (int, float, bool, str):
                    return ann_type
                if isinstance(ann_type, str):
                    low = ann_type.lower()
                    if low == "int":
                        return int
                    if low == "float":
                        return float
                    if low == "bool":
                        return bool
                    if low == "str":
                        return str
            except Exception:
                pass
            return None

        # Cache expected type on params for validation
        try:
            p["expected_type"] = _expected_type()
        except Exception:
            pass

        has_values_meta = isinstance(values_field, (list, tuple)) and bool(values_field)

        # Build choices preferentially from explicit 'values', then from any
        # declared 'devices' in either the decorator metadata or the
        # model-provided parameter metadata. This avoids touching real
        # device objects in the GUI process — we only use their names.
        if has_values_meta:
            choices = [str(x) for x in values_field]
        else:
            devs = []
            devs.extend(_extract_devices_field(devices_field))
            devs.extend(_extract_devices_field(pmeta_devices_field))
            choices = devs if devs else None

            # Do NOT parse description text for choices (too noisy). Only use
            # explicit metadata or model-provided device lists. If those are
            # present, filter them to likely device names.
            if choices:
                # Filter helper: accept dotted or underscore names with letters/numbers
                import re

                def _is_device_name(s):
                    if not isinstance(s, str):
                        return False
                    s = s.strip()
                    if not s:
                        return False
                    if len(s) < 2:
                        return False
                    # Reject obvious non-device tokens
                    bad_tokens = ("typing", "union", "name", "annotation", "__movable__")
                    low = s.lower()
                    for b in bad_tokens:
                        if b in low:
                            return False
                    # Accept names like 'cam1.focus', 'stage1_theta', 'motor'
                    return bool(re.match(r"^[A-Za-z][_A-Za-z0-9\.\_]*$", s))

                choices = [c for c in choices if _is_device_name(c)]
                if choices:
                    # Prefer dotted subdevice names when both dotted and
                    # underscore aliases are present (e.g., stage1.theta over stage1_theta).
                    choice_set = set(choices)
                    filtered = []
                    for c in choices:
                        if "_" in c:
                            dotted = c.replace("_", ".", 1)
                            if dotted in choice_set:
                                continue
                        filtered.append(c)
                    choices = list(dict.fromkeys(filtered)) if filtered else None
                else:
                    choices = None

        if _is_bool_param():
            combo = QComboBox()
            combo.addItems(["False", "True"])
            cur_bool = None
            if is_value_set and isinstance(value, bool):
                cur_bool = value
            elif default_value is not inspect.Parameter.empty and isinstance(default_value, bool):
                cur_bool = default_value
            if cur_bool is not None:
                combo.setCurrentIndex(1 if cur_bool else 0)
            combo.setEnabled(True)
            combo.setToolTip(description)

            def _on_bool_change(*_args, _row=row, _combo=combo):
                try:
                    txt = _combo.currentText()
                    val = True if txt == "True" else False
                    self._params[_row]["value"] = val
                    self._params[_row]["is_value_set"] = True
                    self._params[_row]["is_user_modified"] = True
                    self.signal_cell_modified.emit()
                except Exception:
                    pass

            combo.currentIndexChanged.connect(_on_bool_change)
            self.setCellWidget(row, 2, combo)
        elif choices:
            combo = QComboBox()
            combo.addItems(choices)
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.addItem("")
            custom_index = combo.count() - 1
            le = combo.lineEdit()
            if le is not None:
                le.setReadOnly(True)

            def _toggle_custom_edit(idx, _combo=combo, _le=le, _custom_index=custom_index):
                if _le is None:
                    return
                _le.setReadOnly(idx != _custom_index)
            cur_text = None
            if is_value_set and (value != inspect.Parameter.empty):
                cur_text = str(value)
            elif default_value != inspect.Parameter.empty:
                cur_text = str(default_value)
            if cur_text is not None and cur_text in choices:
                combo.setCurrentIndex(choices.index(cur_text))
            else:
                combo.setCurrentIndex(custom_index)
            # Allow selection even when the parameter value is not yet set
            # so users can choose from the dropdown instead of typing.
            # Allow selection from the dropdown whenever choices exist so
            # users can pick without typing the name manually.
            combo.setEnabled(True)
            combo.setToolTip(description)

            def _on_combo_change(*_args, _row=row, _combo=combo):
                try:
                    txt = _combo.currentText()
                    if not str(txt).strip():
                        # Treat empty custom entry as "unset"
                        self._params[_row]["value"] = inspect.Parameter.empty
                        self._params[_row]["is_value_set"] = False
                        self._params[_row]["is_user_modified"] = True
                        self.signal_cell_modified.emit()
                        return
                    try:
                        val = ast.literal_eval(txt)
                    except Exception:
                        val = txt
                    self._params[_row]["value"] = val
                    self._params[_row]["is_value_set"] = True
                    self._params[_row]["is_user_modified"] = True
                    self.signal_cell_modified.emit()
                except Exception:
                    pass

            combo.currentIndexChanged.connect(_toggle_custom_edit)
            combo.currentIndexChanged.connect(_on_combo_change)
            if combo.lineEdit() is not None:
                combo.lineEdit().editingFinished.connect(_on_combo_change)
            _toggle_custom_edit(combo.currentIndex())
            self.setCellWidget(row, 2, combo)
        else:
            # No explicit choices provided by the decorator/model — render
            # a plain editable (or non-editable) value cell. We do NOT fall
            # back to scanning the worker YAML; dropdowns are only created
            # when explicit 'devices' or 'values' metadata was provided.
            value_item = QTableWidgetItem(s_value)
            if is_editable:
                value_item.setFlags(value_item.flags() | rec.Qt.ItemIsEditable)
            else:
                value_item.setFlags(value_item.flags() & ~rec.Qt.ItemIsEditable)

            if is_value_set:
                value_item.setFlags(value_item.flags() | rec.Qt.ItemIsEnabled)
            else:
                value_item.setFlags(value_item.flags() & ~rec.Qt.ItemIsEnabled)

            value_item.setToolTip(description)
            self.setItem(row, 2, value_item)

    def _validate_cell_values(self):
        if self._validation_disabled:
            return

        data_valid = True
        for n, p_index in enumerate(self._params_indices):
            p = self._params[p_index]
            if p["is_value_set"]:
                widget = self.cellWidget(n, 2)
                if widget is not None:
                    try:
                        if isinstance(widget, QComboBox):
                            txt = widget.currentText()
                            if not str(txt).strip():
                                # Empty entry is invalid for required params
                                is_required = (
                                    p["parameters"].default == inspect.Parameter.empty
                                    and p["parameters"].kind
                                    not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                                )
                                if is_required:
                                    cell_valid = False
                                    data_valid = False
                                else:
                                    cell_valid = True
                                # Do not override value when empty
                                continue
                            try:
                                p["value"] = ast.literal_eval(txt)
                            except Exception:
                                p["value"] = txt
                            cell_valid = True
                        else:
                            cell_valid = True
                    except Exception:
                        cell_valid = False
                        data_valid = False
                else:
                    table_item = self.item(n, 2)
                    if table_item:
                        cell_valid = True
                        cell_text = table_item.text()
                        exp_t = p.get("expected_type")
                        try:
                            p["value"] = ast.literal_eval(cell_text)
                        except Exception:
                            if exp_t in (int, float, bool):
                                try:
                                    if exp_t is bool:
                                        # Allow case-insensitive true/false
                                        low = str(cell_text).strip().lower()
                                        if low in ("true", "false"):
                                            p["value"] = (low == "true")
                                            cell_valid = True
                                        else:
                                            raise ValueError("Invalid bool")
                                    else:
                                        p["value"] = exp_t(cell_text)
                                        cell_valid = True
                                except Exception:
                                    cell_valid = False
                                    data_valid = False
                            else:
                                # Treat as a plain string if no strict type is expected
                                p["value"] = cell_text
                                cell_valid = True
                        else:
                            # Validate parsed value against expected type if known.
                            if exp_t is int:
                                if isinstance(p["value"], bool):
                                    cell_valid = False
                                    data_valid = False
                                elif isinstance(p["value"], float):
                                    if p["value"].is_integer():
                                        p["value"] = int(p["value"])
                                        cell_valid = True
                                    else:
                                        cell_valid = False
                                        data_valid = False
                                elif isinstance(p["value"], int):
                                    cell_valid = True
                                else:
                                    cell_valid = False
                                    data_valid = False
                            elif exp_t is float:
                                if isinstance(p["value"], (int, float)) and not isinstance(p["value"], bool):
                                    p["value"] = float(p["value"])
                                    cell_valid = True
                                else:
                                    cell_valid = False
                                    data_valid = False
                            elif exp_t is bool:
                                if isinstance(p["value"], bool):
                                    cell_valid = True
                                else:
                                    cell_valid = False
                                    data_valid = False

                        table_item.setForeground(self._text_color_valid if cell_valid else self._text_color_invalid)

        self.signal_parameters_valid.emit(data_valid)

    def closeEditor(self, editor, hint):
        super().closeEditor(editor, hint)
        # Ensure validation runs after edits are committed to the model.
        self._validate_cell_values()
        if self._enable_signal_cell_modified:
            self.signal_cell_modified.emit()

    def table_item_changed(self, table_item):
        try:
            row = self.row(table_item)
            column = self.column(table_item)
            if column == 1:
                is_checked = table_item.checkState() == rec.Qt.Checked
                if self._params[row]["is_value_set"] != is_checked:
                    if is_checked and self._params[row]["value"] == inspect.Parameter.empty:
                        self._params[row]["value"] = self._params[row]["parameters"].default

                    self._params[row]["is_value_set"] = is_checked

                    self._enable_signal_cell_modified = False
                    self._show_row_value(row=row)
                    self._enable_signal_cell_modified = True

            if column == 2:
                self._params[row]["is_user_modified"] = True

            if column in (1, 2):
                self._validate_cell_values()
                if self._enable_signal_cell_modified:
                    self.signal_cell_modified.emit()
        except ValueError:
            pass

    def _params_to_item(self, params, item):
        item = super()._params_to_item(params, item)
        try:
            kwargs = item.get("kwargs", {})
            if isinstance(kwargs, dict):
                for p in params:
                    try:
                        name = p["parameters"].name
                        if name not in kwargs:
                            continue
                        if p.get("value") is None and p["parameters"].default is None:
                            # Treat default None as unset (avoid sending None to qserver).
                            kwargs.pop(name, None)
                            continue
                        if not p.get("is_user_modified"):
                            continue
                    except Exception:
                        continue
        except Exception:
            pass
        return item


class RePlanEditorWidget(rec.QtRePlanEditor):
    """QtRePlanEditor that uses the custom table implementation.

    This class replaces the internal plan editor table with `RePlanEditorTable`
    so dropdowns and device choices provided by the model are rendered as
    combo boxes.
    """

    def __init__(self, model, parent=None):
        super().__init__(model, parent)
        try:
            # Replace the table used by the internal editor widget and
            # swap it into the layout so it is visible.
            old = self._plan_editor._wd_editor
            new = RePlanEditorTable(self.model, editable=old.editable, detailed=old.detailed)

            # Preserve current item (if any).
            try:
                new.show_item(item=old.queue_item, editable=old.editable)
            except Exception:
                pass

            # Reconnect signals expected by the editor.
            new.signal_parameters_valid.connect(self._plan_editor._slot_parameters_valid)
            new.signal_item_description_changed.connect(self._plan_editor._slot_item_description_changed)
            new.signal_cell_modified.connect(self._plan_editor._switch_to_editing_mode)

            # Replace in layout to ensure the new table is shown.
            layout = self._plan_editor.layout()
            if layout is not None:
                index = layout.indexOf(old)
                if index >= 0:
                    layout.removeWidget(old)
                    old.setParent(None)
                    layout.insertWidget(index, new)

            self._plan_editor._wd_editor = new
        except Exception:
            # Fall back silently if internal layout changes in future versions
            # of bluesky-widgets and attribute names differ.
            pass
