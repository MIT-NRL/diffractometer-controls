import ast
import inspect
import json
import logging
import os
import re
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from qtpy import QtCore
from qtpy import QtGui
from qtpy.QtWidgets import QComboBox, QShortcut, QTableWidgetItem

import bluesky_widgets.qt.run_engine_client as rec

try:
    from bluesky_queueserver_api import BFunc
    from bluesky_queueserver_api.zmq import REManagerAPI
except Exception:
    BFunc = None
    REManagerAPI = None
try:
    import zmq
except Exception:
    zmq = None

_LOG = logging.getLogger(__name__)


class _DynamicChoicesComboBox(QComboBox):
    """ComboBox that emits a signal before opening the popup list."""

    signal_popup_about_to_show = QtCore.Signal()

    def showPopup(self):
        self.signal_popup_about_to_show.emit()
        super().showPopup()


class _CheckableChoicesComboBox(_DynamicChoicesComboBox):
    """Combo box with checkable popup items for multi-select parameters."""

    signal_selection_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setReadOnly(True)
        self._block_popup_hide = False
        self.view().viewport().installEventFilter(self)
        self._refresh_display_text()

    def eventFilter(self, obj, event):
        if obj is self.view().viewport():
            if event.type() == QtCore.QEvent.MouseButtonRelease:
                index = self.view().indexAt(event.pos())
                if index.isValid():
                    self._toggle_index(index)
                    return True
        return super().eventFilter(obj, event)

    def hidePopup(self):
        if self._block_popup_hide:
            self._block_popup_hide = False
            return
        super().hidePopup()

    def currentText(self):
        line_edit = self.lineEdit()
        if line_edit is not None:
            return line_edit.text()
        return super().currentText()

    def checked_items(self):
        items = []
        model = self.model()
        for row in range(model.rowCount()):
            item = model.item(row)
            if item is None:
                continue
            if item.checkState() == QtCore.Qt.Checked:
                items.append(str(item.text()))
        return items

    def set_choices(self, choices):
        current = self.checked_items()
        blocker = QtCore.QSignalBlocker(self.model())
        try:
            self.clear()
            for choice in list(choices or []):
                self.addItem(str(choice))
                item = self.model().item(self.count() - 1)
                if item is None:
                    continue
                item.setFlags(
                    QtCore.Qt.ItemIsEnabled
                    | QtCore.Qt.ItemIsUserCheckable
                    | QtCore.Qt.ItemIsSelectable
                )
                item.setData(QtCore.Qt.Unchecked, QtCore.Qt.CheckStateRole)
        finally:
            del blocker
        self.set_checked_items(current, emit_signal=False)

    def set_checked_items(self, values, *, emit_signal=True):
        selected = {str(v) for v in list(values or [])}
        blocker = QtCore.QSignalBlocker(self.model())
        try:
            for row in range(self.model().rowCount()):
                item = self.model().item(row)
                if item is None:
                    continue
                state = QtCore.Qt.Checked if item.text() in selected else QtCore.Qt.Unchecked
                item.setCheckState(state)
        finally:
            del blocker
        self._refresh_display_text()
        if emit_signal:
            self.signal_selection_changed.emit()

    def _toggle_index(self, index):
        item = self.model().itemFromIndex(index)
        if item is None:
            return
        item.setCheckState(
            QtCore.Qt.Unchecked if item.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
        )
        self._block_popup_hide = True
        self._refresh_display_text()
        self.signal_selection_changed.emit()

    def _refresh_display_text(self):
        selected = self.checked_items()
        text = repr(selected) if selected else ""
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setText(text)
        else:
            self.setEditText(text)


class RePlanEditorTable(rec._QtRePlanEditorTable):
    """Table subclass that renders dropdowns for parameters when the
    plan metadata exposes choices via 'values' or 'devices'.
    """

    signal_file_dir_choices_ready = QtCore.Signal(object, object)

    def __init__(self, model, parent=None, *, editable=False, detailed=True):
        super().__init__(model, parent, editable=editable, detailed=detailed)
        # Keep internal 3-column model for compatibility, but hide checkbox column.
        self.setColumnHidden(1, True)
        # Guard against recursive validation/closeEditor re-entry on some Qt paths.
        self._validity_emit_in_progress = False
        self._close_editor_in_progress = False
        self._validate_in_progress = False
        # Cache parameter metadata for the currently displayed item.
        self._cached_item_params_key = None
        self._cached_item_params = {}
        # Dynamic worker-side choices for `file_dir` parameters.
        self._file_dir_combos = weakref.WeakSet()
        self._file_dir_cached_choices = []
        self._file_dir_cache_ts = 0.0
        ttl_env = str(os.environ.get("MITR_FILE_DIR_CACHE_TTL_S", "2")).strip()
        try:
            self._file_dir_cache_ttl_s = max(2.0, float(ttl_env))
        except Exception:
            self._file_dir_cache_ttl_s = 2.0
        self._file_dir_api = None
        mode = str(os.environ.get("MITR_FILE_DIR_QUERY_MODE", "stream")).strip().lower()
        self._file_dir_query_mode = mode if mode in ("local", "worker", "stream") else "stream"
        self._file_dir_stream_addr = str(os.environ.get("MITR_FILE_DIR_STREAM_ADDR", "")).strip()
        self._file_dir_stream_topic = str(
            os.environ.get("MITR_FILE_DIR_STREAM_TOPIC", "file_dir_choices")
        ).strip() or "file_dir_choices"
        self._file_dir_stream_stop = threading.Event()
        self._file_dir_stream_thread = None
        self.signal_file_dir_choices_ready.connect(self._on_file_dir_choices_ready)
        self._file_dir_request_event = threading.Event()
        self._file_dir_query_thread = None
        self.destroyed.connect(self._on_destroyed)
        if self._file_dir_query_mode == "stream":
            self._start_file_dir_stream_subscriber()
        else:
            self._ensure_file_dir_query_thread()

    def _on_destroyed(self, *args):
        self.shutdown(wait=False)

    def shutdown(self, *, wait=False, timeout=0.2):
        self._file_dir_stream_stop.set()
        self._file_dir_request_event.set()
        try:
            self.signal_file_dir_choices_ready.disconnect(self._on_file_dir_choices_ready)
        except Exception:
            pass

        if not wait:
            return

        current = threading.current_thread()
        for thread_attr in ("_file_dir_stream_thread", "_file_dir_query_thread"):
            thread = getattr(self, thread_attr, None)
            if thread is None or thread is current:
                continue
            try:
                if thread.is_alive():
                    thread.join(timeout=timeout)
            except Exception:
                pass

    def _emit_file_dir_choices_ready(self, choices, error):
        if self._file_dir_stream_stop.is_set():
            return False
        try:
            self.signal_file_dir_choices_ready.emit(choices, error)
            return True
        except RuntimeError as ex:
            if "has been deleted" in str(ex):
                self._file_dir_stream_stop.set()
                return False
            raise

    def _current_item_key(self):
        if not self._queue_item:
            return None
        item_name = self._queue_item.get("name")
        if not item_name:
            return None
        item_type = self._queue_item.get("item_type", "plan")
        return (str(item_type), str(item_name))

    def _get_current_item_params(self):
        key = self._current_item_key()
        if key is None:
            self._cached_item_params_key = None
            self._cached_item_params = {}
            return {}
        if self._cached_item_params_key == key:
            return self._cached_item_params
        item_type, item_name = key
        try:
            if item_type == "instruction":
                item_params = self.model.get_allowed_instruction_parameters(name=item_name) or {}
            else:
                item_params = self.model.get_allowed_plan_parameters(name=item_name) or {}
        except Exception:
            item_params = {}
        self._cached_item_params_key = key
        self._cached_item_params = item_params if isinstance(item_params, dict) else {}
        return self._cached_item_params

    def show_item(self, *, item, editable=None):
        self._cached_item_params_key = None
        self._cached_item_params = {}
        self._file_dir_combos = weakref.WeakSet()
        super().show_item(item=item, editable=editable)

    def _param_index_from_row(self, row):
        if 0 <= row < len(self._params_indices):
            return self._params_indices[row]
        if 0 <= row < len(self._params):
            return row
        return None

    def _param_from_row(self, row):
        p_index = self._param_index_from_row(row)
        if p_index is None:
            return None, None
        return p_index, self._params[p_index]

    def _get_param_meta(self, p_name):
        # Try to obtain parameter metadata from the model's allowed plans
        try:
            item_params = self._get_current_item_params()
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

    @staticmethod
    def _is_file_dir_param(p_name):
        return str(p_name).strip().lower() == "file_dir"

    @staticmethod
    def _extract_file_dir_choices(payload):
        """Best-effort parser for Queue Server function_execute responses."""

        def _as_choices(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                out = []
                seen = set()
                for item in value:
                    s = str(item).strip()
                    if not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
                return out
            if isinstance(value, dict):
                for key in ("directories", "file_dirs", "dirs", "choices", "items"):
                    out = _as_choices(value.get(key))
                    if out:
                        return out
                for key in ("return_value", "result", "data", "payload"):
                    out = _as_choices(value.get(key))
                    if out:
                        return out
            return []

        return _as_choices(payload)

    def _get_re_manager_api(self):
        app = QtCore.QCoreApplication.instance()
        return getattr(app, "re_manager_api", None)

    @staticmethod
    def _get_api_control_addr(api):
        control_addr = str(getattr(api, "_zmq_control_addr", "") or "").strip()
        if control_addr:
            return control_addr
        client = getattr(api, "_client", None)
        return str(getattr(client, "_zmq_server_address", "") or "").strip()

    def _get_file_dir_api(self):
        # Isolate file-dir requests from the shared API object used by other
        # bluesky_widgets components to avoid connection churn/reloads.
        if self._file_dir_api is not None:
            return self._file_dir_api

        shared_api = self._get_re_manager_api()
        if shared_api is None:
            return None
        if REManagerAPI is None:
            return shared_api

        control_addr = self._get_api_control_addr(shared_api)
        info_addr = str(getattr(shared_api, "_zmq_info_addr", "") or "").strip()
        if not control_addr or not info_addr:
            return shared_api

        try:
            self._file_dir_api = REManagerAPI(
                zmq_control_addr=str(control_addr),
                zmq_info_addr=str(info_addr),
            )
        except Exception:
            self._file_dir_api = shared_api
        return self._file_dir_api

    def _default_file_dir_stream_addr(self):
        if self._file_dir_stream_addr:
            return self._file_dir_stream_addr
        api = self._get_re_manager_api()
        if api is None:
            return "tcp://localhost:5569"
        control_addr = self._get_api_control_addr(api)
        m = re.match(r"^tcp://([^:]+):\d+$", control_addr)
        host = m.group(1) if m else "localhost"
        if host in ("*", "0.0.0.0", "::"):
            host = "localhost"
        return f"tcp://{host}:5569"

    def _start_file_dir_stream_subscriber(self):
        if self._file_dir_stream_thread is not None:
            return
        if zmq is None:
            _LOG.warning("file_dir stream mode requested, but pyzmq is unavailable")
            self._file_dir_query_mode = "worker"
            self._ensure_file_dir_query_thread()
            return

        addr = self._default_file_dir_stream_addr()
        topic_b = self._file_dir_stream_topic.encode("utf-8", errors="ignore")

        def _stream_loop():
            ctx = None
            sock = None
            try:
                ctx = zmq.Context.instance()
                sock = ctx.socket(zmq.SUB)
                sock.setsockopt(zmq.LINGER, 0)
                sock.setsockopt(zmq.RCVHWM, 10)
                sock.setsockopt(zmq.SUBSCRIBE, topic_b)
                sock.connect(addr)
                poller = zmq.Poller()
                poller.register(sock, zmq.POLLIN)
                while not self._file_dir_stream_stop.is_set():
                    events = dict(poller.poll(100))
                    if sock not in events:
                        continue
                    try:
                        parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                    except Exception:
                        continue
                    payload = None
                    try:
                        if len(parts) >= 2:
                            payload = json.loads(parts[-1].decode("utf-8", errors="ignore"))
                        elif len(parts) == 1:
                            payload = json.loads(parts[0].decode("utf-8", errors="ignore"))
                    except Exception:
                        payload = None
                    if payload is None:
                        continue
                    choices = self._extract_file_dir_choices(payload)
                    if choices:
                        if not self._emit_file_dir_choices_ready(choices, None):
                            break
            except Exception as ex:
                if not self._file_dir_stream_stop.is_set():
                    _LOG.warning("file_dir stream subscriber stopped: %s", ex)
            finally:
                try:
                    if sock is not None:
                        sock.close(0)
                except Exception:
                    pass

        self._file_dir_stream_thread = threading.Thread(
            target=_stream_loop, name="file-dir-stream-subscriber", daemon=True
        )
        self._file_dir_stream_thread.start()

    @staticmethod
    def _build_list_file_dirs_item():
        kwargs = {"max_depth": 3, "max_items": 512}
        if BFunc is not None:
            return BFunc("list_imaging_file_dirs", **kwargs)
        return {
            "item_type": "function",
            "name": "list_imaging_file_dirs",
            "args": [],
            "kwargs": kwargs,
        }

    @staticmethod
    def _call_function_execute(api, item):
        variants = (
            {"run_in_background": False, "user": "plan_editor", "user_group": "primary"},
            {"run_in_background": False},
            {"user": "plan_editor", "user_group": "primary"},
            {},
        )
        last_ex = None
        for kwargs in variants:
            try:
                return api.function_execute(item, **kwargs)
            except Exception as ex:
                last_ex = ex
                continue
        if last_ex is not None:
            raise last_ex
        return api.function_execute(item)

    @staticmethod
    def _extract_task_uid(payload):
        if not isinstance(payload, dict):
            return None

        candidates = (
            payload.get("task_uid", None),
            payload.get("uid", None),
        )
        for uid in candidates:
            s = str(uid or "").strip()
            if s:
                return s

        for key in ("result", "return_value", "payload", "data", "item"):
            nested = payload.get(key, None)
            if isinstance(nested, dict):
                uid = RePlanEditorTable._extract_task_uid(nested)
                if uid:
                    return uid
        return None

    @staticmethod
    def _extract_function_execute_error(payload):
        if not isinstance(payload, dict):
            return None

        success = payload.get("success", None)
        if success is False:
            for key in ("msg", "message", "error", "err_msg"):
                txt = payload.get(key, None)
                if txt:
                    return str(txt)
            return "Function execution failed."

        status = str(payload.get("status", "")).strip().lower()
        if status in ("failed", "error", "rejected"):
            for key in ("msg", "message", "error", "err_msg"):
                txt = payload.get(key, None)
                if txt:
                    return str(txt)
            return f"Function task {status}."

        for key in ("result", "return_value", "payload", "data"):
            nested = payload.get(key, None)
            if isinstance(nested, dict):
                err = RePlanEditorTable._extract_function_execute_error(nested)
                if err:
                    return err
        return None

    @staticmethod
    def _is_pending_task_state(status):
        s = str(status or "").strip().lower()
        return s in ("created", "queued", "submitted", "running", "in_progress", "pending")

    @staticmethod
    def _is_pending_task_error(message, *, status=None):
        if RePlanEditorTable._is_pending_task_state(status):
            return True
        m = str(message or "").strip().lower()
        if not m:
            return False
        pending_markers = (
            "not completed",
            "still running",
            "in progress",
            "is running",
            "pending",
            "queued",
            "not available yet",
        )
        return any(marker in m for marker in pending_markers)

    @staticmethod
    def _fallback_local_file_dirs(max_items=256, max_depth=3):
        now = datetime.now()
        roots = []
        env_roots = str(os.environ.get("MITR_IMAGING_DATA_ROOTS", "")).strip()
        if env_roots:
            for part in env_roots.split(os.pathsep):
                part = str(part).strip()
                if not part:
                    continue
                roots.append(Path(now.strftime(part)).expanduser())
        env_root = str(os.environ.get("MITR_IMAGING_DATA_ROOT", "")).strip()
        if env_root:
            roots.append(Path(now.strftime(env_root)).expanduser())
        roots.append(Path(now.strftime("/home/mitr_4dh4/Data/%Y")).expanduser())
        roots.append(Path(now.strftime("~/Data/%Y")).expanduser())

        dedup_roots = []
        seen_roots = set()
        for root in roots:
            s = str(root)
            if s in seen_roots:
                continue
            seen_roots.add(s)
            dedup_roots.append(root)

        names = set()
        for root in dedup_roots:
            try:
                if not root.exists() or (not root.is_dir()):
                    continue
            except Exception:
                continue
            stack = [(Path(root), 0, "")]
            while stack:
                cur_path, depth, rel_prefix = stack.pop()
                if depth >= int(max_depth):
                    continue
                try:
                    with os.scandir(cur_path) as entries:
                        children = []
                        for entry in entries:
                            try:
                                if not entry.is_dir(follow_symlinks=False):
                                    continue
                                name = str(entry.name).strip()
                                if (not name) or name.startswith("."):
                                    continue
                                rel_path = f"{rel_prefix}/{name}" if rel_prefix else name
                                names.add(rel_path)
                                children.append((Path(cur_path) / name, depth + 1, rel_path))
                            except Exception:
                                continue
                except Exception:
                    continue
                for child in sorted(children, key=lambda x: x[2], reverse=True):
                    stack.append(child)
        return sorted(names, key=str.lower)[: int(max_items)]

    def _query_worker_file_dirs(self):
        api = self._get_file_dir_api()
        item = self._build_list_file_dirs_item()
        if api is None:
            return self._fallback_local_file_dirs()
        response = self._call_function_execute(api, item)
        err = self._extract_function_execute_error(response)
        if err:
            raise RuntimeError(err)
        choices = self._extract_file_dir_choices(response)
        if choices:
            return choices

        # Compatibility: some servers still return task UID even when
        # run_in_background=False; poll briefly for completion.
        task_uid = self._extract_task_uid(response)
        if task_uid and hasattr(api, "task_result"):
            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                task = None
                for kwargs in (
                    {"task_uid": task_uid, "user": "plan_editor", "user_group": "primary"},
                    {"task_uid": task_uid},
                ):
                    try:
                        task = api.task_result(**kwargs)
                        break
                    except Exception:
                        continue
                if task is None:
                    try:
                        task = api.task_result(task_uid)
                    except Exception:
                        break
                status = str((task or {}).get("status", "")).strip().lower() if isinstance(task, dict) else ""
                if self._is_pending_task_state(status):
                    time.sleep(0.05)
                    continue
                err = self._extract_function_execute_error(task)
                if err and (not self._is_pending_task_error(err, status=status)):
                    raise RuntimeError(err)
                choices = self._extract_file_dir_choices(task)
                if choices:
                    return choices
                time.sleep(0.05)

        # Queue Server is reachable but returned no choices.
        return []

    def _query_file_dirs(self):
        # Local mode may be enabled explicitly with MITR_FILE_DIR_QUERY_MODE=local.
        if self._file_dir_query_mode == "local":
            return self._fallback_local_file_dirs(max_items=512, max_depth=3)
        if self._file_dir_query_mode == "stream":
            return []
        return self._query_worker_file_dirs()

    def _ensure_file_dir_query_thread(self):
        if self._file_dir_query_thread is not None:
            return
        self._file_dir_query_thread = threading.Thread(
            target=self._file_dir_query_loop, name="file-dir-query-thread", daemon=True
        )
        self._file_dir_query_thread.start()

    def _file_dir_query_loop(self):
        while not self._file_dir_stream_stop.is_set():
            self._file_dir_request_event.wait()
            self._file_dir_request_event.clear()
            if self._file_dir_stream_stop.is_set():
                break

            err = None
            choices = []
            try:
                choices = self._query_file_dirs()
            except Exception as ex:
                err = str(ex)
            if not self._emit_file_dir_choices_ready(choices, err):
                break

    def _request_file_dir_choices(self, *, force=False):
        if self._file_dir_stream_stop.is_set():
            return
        if self._file_dir_query_mode == "stream":
            self._start_file_dir_stream_subscriber()
            if self._file_dir_cached_choices:
                self._apply_file_dir_choices_to_combos(self._file_dir_cached_choices)
            return
        if not force and self._file_dir_cached_choices:
            age = time.monotonic() - float(self._file_dir_cache_ts)
            if age < self._file_dir_cache_ttl_s:
                self._apply_file_dir_choices_to_combos(self._file_dir_cached_choices)
                return
        self._ensure_file_dir_query_thread()
        self._file_dir_request_event.set()

    def _on_file_dir_choices_ready(self, choices, error):
        if self._file_dir_stream_stop.is_set():
            return
        if error and (not choices):
            _LOG.warning("file_dir choices update failed: %s", str(error))
            return
        if not isinstance(choices, list):
            return
        if not choices:
            return
        self._file_dir_cached_choices = list(choices)
        self._file_dir_cache_ts = time.monotonic()
        self._apply_file_dir_choices_to_combos(self._file_dir_cached_choices)

    @staticmethod
    def _set_dynamic_combo_edit_state(combo):
        line_edit = combo.lineEdit()
        if line_edit is None:
            return
        # Keep `file_dir` combos editable even after selecting a suggested value,
        # so users can append subfolder paths.
        line_edit.setReadOnly(False)
        try:
            txt = str(combo.currentText() or "")
            line_edit.deselect()
            line_edit.setCursorPosition(len(txt))
        except Exception:
            pass

    @staticmethod
    def _set_combo_implicit_style(combo, is_implicit_default):
        """
        Gray only the displayed value for implicit/default state while keeping
        popup items readable at normal color.
        """
        line_edit = combo.lineEdit()
        if is_implicit_default:
            if line_edit is not None:
                line_edit.setStyleSheet("color: #777;")
                combo.setStyleSheet("")
            else:
                combo.setStyleSheet(
                    "QComboBox { color: #777; } "
                    "QComboBox QAbstractItemView { color: palette(text); }"
                )
        else:
            if line_edit is not None:
                line_edit.setStyleSheet("")
            combo.setStyleSheet("")

    @staticmethod
    def _set_combo_invalid_style(combo):
        """Show invalid combo value in red without tinting popup entries."""
        line_edit = combo.lineEdit()
        if line_edit is not None:
            line_edit.setStyleSheet("color: #b00020;")
            combo.setStyleSheet("")
        else:
            combo.setStyleSheet(
                "QComboBox { color: #b00020; } "
                "QComboBox QAbstractItemView { color: palette(text); }"
            )

    @staticmethod
    def _get_combo_base_tooltip(combo, fallback_tooltip=""):
        base = combo.property("dc_base_tooltip")
        if base is None:
            base = fallback_tooltip if fallback_tooltip is not None else combo.toolTip()
        return str(base or "")

    @staticmethod
    def _get_file_dir_max_depth():
        raw = str(os.environ.get("MITR_FILE_DIR_MAX_DEPTH", "3")).strip()
        try:
            depth = int(raw)
        except Exception:
            depth = 3
        return max(1, depth)

    @staticmethod
    def _file_dir_depth(value):
        txt = str(value or "").strip().replace("\\", "/")
        parts = [part for part in txt.split("/") if part and part != "."]
        return len(parts)

    def _is_valid_file_dir_value(self, value):
        return self._file_dir_depth(value) <= self._get_file_dir_max_depth()

    def _set_file_dir_combo_validation(
        self, combo, *, is_valid, is_implicit_default=False, value=None, fallback_tooltip=""
    ):
        base_tooltip = self._get_combo_base_tooltip(combo, fallback_tooltip)
        if is_valid:
            self._set_combo_implicit_style(combo, is_implicit_default)
            combo.setToolTip(base_tooltip)
            return

        self._set_combo_invalid_style(combo)
        txt = combo.currentText() if value is None else value
        cur_depth = self._file_dir_depth(txt)
        max_depth = self._get_file_dir_max_depth()
        helper = (
            f"file_dir helper: max {max_depth} subdirectories. "
            f"Current depth: {cur_depth}."
        )
        combo.setToolTip(f"{base_tooltip}\n\n{helper}" if base_tooltip else helper)

    def _populate_dynamic_combo_choices(self, combo, choices, *, current_text=None):
        if current_text is None:
            current_text = combo.currentText()
        out = []
        seen = set()
        for ch in list(choices or []):
            txt = str(ch).strip()
            if (not txt) or (txt in seen):
                continue
            seen.add(txt)
            out.append(txt)
        blocker = QtCore.QSignalBlocker(combo)
        try:
            combo.clear()
            combo.addItem("")
            combo.addItems(out)
            custom_index = 0
            combo.setProperty("dc_custom_index", custom_index)
            if str(current_text) in out:
                combo.setCurrentIndex(out.index(str(current_text)) + 1)
            else:
                combo.setCurrentIndex(custom_index)
                line_edit = combo.lineEdit()
                if line_edit is not None:
                    line_edit.setText(str(current_text or ""))
        finally:
            del blocker
        self._set_dynamic_combo_edit_state(combo)

    def _apply_file_dir_choices_to_combos(self, choices):
        for combo in list(self._file_dir_combos):
            if combo is None:
                continue
            self._populate_dynamic_combo_choices(combo, choices, current_text=combo.currentText())

    def _show_row_value(self, *, row):
        # Based on original implementation but uses metadata from the model
        def print_value(v):
            if isinstance(v, str):
                return f"'{v}'"
            else:
                return str(v)

        _, p = self._param_from_row(row)
        if p is None:
            return
        p_name = p["name"]
        value = p["value"]
        default_value = p["parameters"].default
        is_var_positional = p["parameters"].kind == inspect.Parameter.VAR_POSITIONAL
        is_var_keyword = p["parameters"].kind == inspect.Parameter.VAR_KEYWORD
        is_value_set = p["is_value_set"]
        is_editable = self._editable

        description = self._params_descriptions.get("parameters", {}).get(p_name, None)
        if not description:
            description = f"Description for parameter '{p_name}' was not found ..."

        v = value if is_value_set else default_value
        if isinstance(v, str) and v == "":
            s_value = ""
        else:
            s_value = "" if v == inspect.Parameter.empty else print_value(v)
        is_required_param = (
            default_value == inspect.Parameter.empty
            and not is_var_positional
            and not is_var_keyword
        )

        def _emit_modified_deferred():
            def _emit():
                if self._enable_signal_cell_modified:
                    self.signal_cell_modified.emit()

            QtCore.QTimer.singleShot(0, _emit)

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
        try:
            item_params = self._get_current_item_params()
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
                ann_candidates = [
                    ann_type,
                    ann.get("type_name"),
                    ann.get("full_type"),
                    ann.get("annotation"),
                ]
                for candidate in ann_candidates:
                    low = str(candidate or "").strip().lower()
                    if not low:
                        continue
                    if ("list[" in low) or ("dict[" in low):
                        continue
                    if "str" in low:
                        return str
                    if "bool" in low:
                        return bool
                    if "float" in low:
                        return float
                    if "int" in low:
                        return int
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
            if isinstance(value, str):
                return str
            if default_value is not inspect.Parameter.empty and isinstance(default_value, str):
                return str
            if isinstance(value, bool):
                return bool
            if default_value is not inspect.Parameter.empty and isinstance(default_value, bool):
                return bool
            if isinstance(value, int) and not isinstance(value, bool):
                return int
            if default_value is not inspect.Parameter.empty and isinstance(default_value, int) and not isinstance(default_value, bool):
                return int
            if isinstance(value, float):
                return float
            if default_value is not inspect.Parameter.empty and isinstance(default_value, float):
                return float
            return None

        def _is_multi_select_param():
            candidates = []
            for meta_source in (meta, pmeta):
                if not isinstance(meta_source, dict):
                    continue
                ann = meta_source.get("annotation")
                candidates.append(ann)
                if isinstance(ann, dict):
                    candidates.extend(
                        [
                            ann.get("type"),
                            ann.get("type_name"),
                            ann.get("full_type"),
                            ann.get("annotation"),
                        ]
                    )
            for candidate in candidates:
                if candidate in (list, tuple, set):
                    return True
                text = str(candidate or "").strip().lower()
                if ("typing.list[" in text) or text.startswith("list[") or ("list[" in text):
                    return True
            if isinstance(value, (list, tuple, set)):
                return True
            if default_value is not inspect.Parameter.empty and isinstance(default_value, (list, tuple, set)):
                return True
            return False

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

        if self._is_file_dir_param(p_name):
            combo = _DynamicChoicesComboBox()
            combo.setEditable(True)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.setEnabled(True)
            combo.setToolTip(description)
            combo.setProperty("dc_base_tooltip", description)
            if not is_value_set:
                self._set_file_dir_combo_validation(
                    combo,
                    is_valid=True,
                    is_implicit_default=True,
                    value="",
                    fallback_tooltip=description,
                )

            cur_text = ""
            if is_value_set and (value != inspect.Parameter.empty):
                cur_text = str(value)
            elif default_value != inspect.Parameter.empty:
                cur_text = str(default_value)

            initial_choices = []
            if isinstance(choices, list):
                initial_choices.extend(choices)
            initial_choices.extend(self._file_dir_cached_choices)
            self._populate_dynamic_combo_choices(combo, initial_choices, current_text=cur_text)
            self._file_dir_combos.add(combo)

            def _on_popup_open():
                txt = str(combo.currentText() or "")
                is_valid = self._is_valid_file_dir_value(txt) if txt.strip() else True
                self._set_file_dir_combo_validation(
                    combo,
                    is_valid=is_valid,
                    is_implicit_default=False,
                    value=txt,
                    fallback_tooltip=description,
                )
                self._request_file_dir_choices(force=True)

            def _on_combo_change(*_args, _row=row, _combo=combo):
                try:
                    p_index = self._param_index_from_row(_row)
                    if p_index is None:
                        return
                    txt = _combo.currentText()
                    if not str(txt).strip():
                        self._params[p_index]["value"] = inspect.Parameter.empty
                        self._params[p_index]["is_value_set"] = False
                        self._params[p_index]["is_user_modified"] = True
                        self._set_file_dir_combo_validation(
                            _combo,
                            is_valid=True,
                            is_implicit_default=True,
                            value=txt,
                            fallback_tooltip=description,
                        )
                        _emit_modified_deferred()
                        return
                    # Keep directory tokens as plain strings (do not literal-eval).
                    val = str(txt)
                    self._params[p_index]["value"] = val
                    self._params[p_index]["is_value_set"] = True
                    self._params[p_index]["is_user_modified"] = True
                    self._set_file_dir_combo_validation(
                        _combo,
                        is_valid=self._is_valid_file_dir_value(val),
                        is_implicit_default=False,
                        value=val,
                        fallback_tooltip=description,
                    )
                    _emit_modified_deferred()
                except Exception:
                    pass

            def _finalize_combo_commit(*_args, _combo=combo):
                try:
                    _combo.hidePopup()
                except Exception:
                    pass
                try:
                    view = _combo.view()
                    if view is not None:
                        view.clearSelection()
                except Exception:
                    pass
                le = _combo.lineEdit()
                if le is not None:
                    try:
                        le.deselect()
                        le.setSelection(0, 0)
                        le.setCursorPosition(len(str(le.text() or "")))
                        le.clearFocus()
                    except Exception:
                        pass
                try:
                    _combo.clearFocus()
                    self.setFocus(QtCore.Qt.OtherFocusReason)
                except Exception:
                    pass

            def _on_combo_commit(*_args, _combo=combo):
                _on_combo_change()
                _finalize_combo_commit(_combo=_combo)

            combo.signal_popup_about_to_show.connect(_on_popup_open)
            combo.currentIndexChanged.connect(lambda *_: self._set_dynamic_combo_edit_state(combo))
            combo.currentIndexChanged.connect(_on_combo_change)
            # "activated" fires when user confirms an item from popup (including Enter).
            combo.activated.connect(_finalize_combo_commit)
            # Ensure Enter commits even when popup is closed and focus is on combo.
            _commit_shortcut_return = QShortcut(QtGui.QKeySequence("Return"), combo)
            _commit_shortcut_enter = QShortcut(QtGui.QKeySequence("Enter"), combo)
            _commit_shortcut_return.activated.connect(_on_combo_commit)
            _commit_shortcut_enter.activated.connect(_on_combo_commit)
            combo._dc_commit_shortcuts = (_commit_shortcut_return, _commit_shortcut_enter)
            if combo.lineEdit() is not None:
                combo.lineEdit().editingFinished.connect(_on_combo_change)
                combo.lineEdit().returnPressed.connect(_on_combo_commit)
            self.setCellWidget(row, 2, combo)
            self._request_file_dir_choices(force=(not bool(self._file_dir_cached_choices)))
        elif choices and _is_multi_select_param():
            combo = _CheckableChoicesComboBox()
            combo.set_choices(choices)
            combo.setToolTip(description)
            combo.setProperty("dc_base_tooltip", description)

            selected_values = []
            if is_value_set and (value != inspect.Parameter.empty):
                if isinstance(value, (list, tuple, set)):
                    selected_values = [str(v) for v in value]
                else:
                    try:
                        parsed = ast.literal_eval(str(value))
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (list, tuple, set)):
                        selected_values = [str(v) for v in parsed]
            elif default_value != inspect.Parameter.empty:
                if isinstance(default_value, (list, tuple, set)):
                    selected_values = [str(v) for v in default_value]
                else:
                    try:
                        parsed = ast.literal_eval(str(default_value))
                    except Exception:
                        parsed = None
                    if isinstance(parsed, (list, tuple, set)):
                        selected_values = [str(v) for v in parsed]

            if selected_values:
                combo.set_checked_items(selected_values, emit_signal=False)
            if not is_value_set:
                self._set_combo_implicit_style(combo, True)

            def _on_multi_combo_popup_open(_combo=combo):
                self._set_combo_implicit_style(_combo, False)

            def _on_multi_combo_change(*_args, _row=row, _combo=combo):
                try:
                    p_index = self._param_index_from_row(_row)
                    if p_index is None:
                        return
                    selected = list(_combo.checked_items())
                    if not selected:
                        self._params[p_index]["value"] = inspect.Parameter.empty
                        self._params[p_index]["is_value_set"] = False
                        self._params[p_index]["is_user_modified"] = True
                        self._set_combo_implicit_style(_combo, True)
                        _emit_modified_deferred()
                        return
                    self._params[p_index]["value"] = selected
                    self._params[p_index]["is_value_set"] = True
                    self._params[p_index]["is_user_modified"] = True
                    self._set_combo_implicit_style(_combo, False)
                    _emit_modified_deferred()
                except Exception:
                    pass

            combo.signal_popup_about_to_show.connect(_on_multi_combo_popup_open)
            combo.signal_selection_changed.connect(_on_multi_combo_change)
            self.setCellWidget(row, 2, combo)
        elif _is_bool_param():
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
            if not is_value_set:
                self._set_combo_implicit_style(combo, True)

            def _on_bool_change(*_args, _row=row, _combo=combo):
                try:
                    p_index = self._param_index_from_row(_row)
                    if p_index is None:
                        return
                    txt = _combo.currentText()
                    val = True if txt == "True" else False
                    self._params[p_index]["value"] = val
                    self._params[p_index]["is_value_set"] = True
                    self._params[p_index]["is_user_modified"] = True
                    self._set_combo_implicit_style(_combo, False)
                    _emit_modified_deferred()
                except Exception:
                    pass

            combo.currentIndexChanged.connect(_on_bool_change)
            self.setCellWidget(row, 2, combo)
        elif choices:
            combo = _DynamicChoicesComboBox()
            # Keep an explicit empty first option that supports custom typing.
            allow_custom_entry = True
            combo.setEditable(allow_custom_entry)
            combo.setInsertPolicy(QComboBox.NoInsert)
            combo.addItem("")
            combo.addItems(choices)
            custom_index = 0
            le = combo.lineEdit()
            if le is not None:
                le.setReadOnly(True)

            def _toggle_custom_edit(idx, _combo=combo, _le=le, _custom_index=custom_index):
                if _le is None:
                    return
                _le.setReadOnly(idx != _custom_index)

            def _on_combo_popup_open(_combo=combo):
                # Start implicit/default values in gray, but switch to normal style
                # as soon as the user opens the menu.
                self._set_combo_implicit_style(_combo, False)

            cur_text = None
            if is_value_set and (value != inspect.Parameter.empty):
                cur_text = str(value)
            elif default_value != inspect.Parameter.empty:
                cur_text = str(default_value)
            has_cur_text = cur_text is not None and bool(str(cur_text).strip())
            if has_cur_text and cur_text in choices:
                combo.setCurrentIndex(choices.index(cur_text) + 1)
            elif is_required_param and choices:
                # Required dropdowns should start with a valid value, not blank/red.
                combo.setCurrentIndex(1)
            else:
                combo.setCurrentIndex(custom_index)
                if le is not None:
                    le.setText("" if cur_text is None else str(cur_text))
            # Allow selection even when the parameter value is not yet set
            # so users can choose from the dropdown instead of typing.
            # Allow selection from the dropdown whenever choices exist so
            # users can pick without typing the name manually.
            combo.setEnabled(True)
            combo.setToolTip(description)
            if not is_value_set:
                self._set_combo_implicit_style(combo, True)

            def _on_combo_change(*_args, _row=row, _combo=combo):
                try:
                    p_index = self._param_index_from_row(_row)
                    if p_index is None:
                        return
                    txt = _combo.currentText()
                    if not str(txt).strip():
                        # Treat empty custom entry as "unset"
                        self._params[p_index]["value"] = inspect.Parameter.empty
                        self._params[p_index]["is_value_set"] = False
                        self._params[p_index]["is_user_modified"] = True
                        self._set_combo_implicit_style(_combo, True)
                        _emit_modified_deferred()
                        return
                    exp_t = self._params[p_index].get("expected_type")
                    if exp_t is str:
                        val = txt
                    else:
                        try:
                            val = ast.literal_eval(txt)
                        except Exception:
                            val = txt
                    self._params[p_index]["value"] = val
                    self._params[p_index]["is_value_set"] = True
                    self._params[p_index]["is_user_modified"] = True
                    self._set_combo_implicit_style(_combo, False)
                    _emit_modified_deferred()
                except Exception:
                    pass

            combo.signal_popup_about_to_show.connect(_on_combo_popup_open)
            combo.currentIndexChanged.connect(_on_combo_change)
            combo.currentIndexChanged.connect(_toggle_custom_edit)
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

            # Value column remains enabled even for implicit defaults.
            value_item.setFlags(value_item.flags() | rec.Qt.ItemIsEnabled)

            if not is_value_set:
                value_item.setForeground(QtGui.QBrush(QtGui.QColor(120, 120, 120)))
            else:
                value_item.setForeground(self._text_color_valid)

            value_item.setToolTip(description)
            self.setItem(row, 2, value_item)

    def _validate_cell_values(self):
        if self._validation_disabled:
            return
        if self._validate_in_progress:
            return
        self._validate_in_progress = True

        try:
            data_valid = True
            for n, p_index in enumerate(self._params_indices):
                p = self._params[p_index]
                if p["is_value_set"]:
                    widget = self.cellWidget(n, 2)
                    if widget is not None:
                        try:
                            if isinstance(widget, QComboBox):
                                txt = widget.currentText()
                                is_file_dir = self._is_file_dir_param(p.get("name", ""))
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
                                        self._set_combo_invalid_style(widget)
                                    else:
                                        cell_valid = True
                                        self._set_combo_implicit_style(widget, True)
                                    # Do not override value when empty
                                    continue
                                if is_file_dir:
                                    p["value"] = str(txt)
                                    if self._is_valid_file_dir_value(txt):
                                        cell_valid = True
                                        self._set_file_dir_combo_validation(
                                            widget,
                                            is_valid=True,
                                            is_implicit_default=False,
                                            value=txt,
                                        )
                                    else:
                                        cell_valid = False
                                        data_valid = False
                                        self._set_file_dir_combo_validation(
                                            widget,
                                            is_valid=False,
                                            is_implicit_default=False,
                                            value=txt,
                                        )
                                else:
                                    exp_t = p.get("expected_type")
                                    if exp_t is str:
                                        p["value"] = txt
                                    else:
                                        try:
                                            p["value"] = ast.literal_eval(txt)
                                        except Exception:
                                            p["value"] = txt
                                    cell_valid = True
                                    self._set_combo_implicit_style(widget, False)
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
                            if exp_t is str:
                                p["value"] = cell_text
                                blocker = QtCore.QSignalBlocker(self)
                                try:
                                    table_item.setForeground(self._text_color_valid)
                                finally:
                                    del blocker
                                continue
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

                            blocker = QtCore.QSignalBlocker(self)
                            try:
                                table_item.setForeground(
                                    self._text_color_valid if cell_valid else self._text_color_invalid
                                )
                            finally:
                                del blocker

            if self._validity_emit_in_progress:
                return
            self._validity_emit_in_progress = True
            try:
                self.signal_parameters_valid.emit(data_valid)
            finally:
                self._validity_emit_in_progress = False
        finally:
            self._validate_in_progress = False

    def closeEditor(self, editor, hint):
        if self._close_editor_in_progress:
            return
        self._close_editor_in_progress = True
        try:
            super().closeEditor(editor, hint)
            # Ensure validation runs after edits are committed to the model.
            self._validate_cell_values()
            if self._enable_signal_cell_modified:
                self.signal_cell_modified.emit()
        finally:
            self._close_editor_in_progress = False

    def table_item_changed(self, table_item):
        try:
            if self._validation_disabled:
                return
            row = self.row(table_item)
            column = self.column(table_item)
            _, p = self._param_from_row(row)
            if p is None:
                return
            if column == 1:
                is_checked = table_item.checkState() == rec.Qt.Checked
                if p["is_value_set"] != is_checked:
                    if is_checked and p["value"] == inspect.Parameter.empty:
                        p["value"] = p["parameters"].default

                    p["is_value_set"] = is_checked

                    self._enable_signal_cell_modified = False
                    self._show_row_value(row=row)
                    self._enable_signal_cell_modified = True

            if column == 2:
                table_item_col2 = self.item(row, 2)
                if table_item_col2 is not None:
                    txt = table_item_col2.text()
                    default_val = p["parameters"].default
                    if (default_val != inspect.Parameter.empty) and (not str(txt).strip()):
                        # Clearing optional field reverts to implicit default.
                        p["is_value_set"] = False
                        p["value"] = inspect.Parameter.empty
                        blocker = QtCore.QSignalBlocker(self)
                        try:
                            table_item_col2.setForeground(QtGui.QBrush(QtGui.QColor(120, 120, 120)))
                        finally:
                            del blocker
                    else:
                        p["is_value_set"] = True
                        p["is_user_modified"] = True
                        blocker = QtCore.QSignalBlocker(self)
                        try:
                            table_item_col2.setForeground(self._text_color_valid)
                        finally:
                            del blocker

            if column in (1, 2):
                self._validate_cell_values()
                if self._enable_signal_cell_modified:
                    self.signal_cell_modified.emit()
        except (ValueError, IndexError):
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
        self._pending_parameters_valid = None
        self._parameters_valid_update_scheduled = False
        self.destroyed.connect(self._on_destroyed)
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
            new.signal_parameters_valid.connect(self._queue_parameters_valid_update)
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

    def _on_destroyed(self, *args):
        self.shutdown(wait=False)

    def shutdown(self, *, wait=False, timeout=0.2):
        editor = getattr(self._plan_editor, "_wd_editor", None)
        shutdown = getattr(editor, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown(wait=wait, timeout=timeout)
            except TypeError:
                shutdown()
            except Exception:
                pass

    def _queue_parameters_valid_update(self, is_valid):
        self._pending_parameters_valid = bool(is_valid)
        if self._parameters_valid_update_scheduled:
            return
        self._parameters_valid_update_scheduled = True
        QtCore.QTimer.singleShot(0, self._flush_parameters_valid_update)

    def _flush_parameters_valid_update(self):
        self._parameters_valid_update_scheduled = False
        if self._pending_parameters_valid is None:
            return
        is_valid = self._pending_parameters_valid
        self._pending_parameters_valid = None
        self._plan_editor._slot_parameters_valid(is_valid)
