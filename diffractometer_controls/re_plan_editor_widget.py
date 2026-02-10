import ast
import inspect
from qtpy.QtWidgets import QComboBox, QTableWidgetItem

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

        if choices:
            combo = QComboBox()
            combo.addItems(choices)
            cur_text = None
            if is_value_set and (value != inspect.Parameter.empty):
                cur_text = str(value)
            elif default_value != inspect.Parameter.empty:
                cur_text = str(default_value)
            if cur_text is not None and cur_text in choices:
                combo.setCurrentIndex(choices.index(cur_text))
            # Allow selection even when the parameter value is not yet set
            # so users can choose from the dropdown instead of typing.
            # Allow selection from the dropdown whenever choices exist so
            # users can pick without typing the name manually.
            combo.setEnabled(True)
            combo.setToolTip(description)

            def _on_combo_change(idx, _row=row, _combo=combo):
                try:
                    txt = _combo.currentText()
                    try:
                        val = ast.literal_eval(txt)
                    except Exception:
                        val = txt
                    self._params[_row]["value"] = val
                    self._params[_row]["is_value_set"] = True
                    self.signal_cell_modified.emit()
                except Exception:
                    pass

            combo.currentIndexChanged.connect(_on_combo_change)
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
                        try:
                            p["value"] = ast.literal_eval(cell_text)
                        except Exception:
                            cell_valid = False
                            data_valid = False

                        table_item.setForeground(self._text_color_valid if cell_valid else self._text_color_invalid)

        self.signal_parameters_valid.emit(data_valid)


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
