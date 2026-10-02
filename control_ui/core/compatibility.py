"""Compatibility adaptations for the installed Bluesky widgets; explicitly installed."""
import time
from qtpy import QtCore, QtGui, QtWidgets
from bluesky_widgets.qt import run_engine_client as bw_run_engine_client

class BlueskyCompatibility:
    @staticmethod
    def _patch_bluesky_model_event_disconnects():
        widget_event_specs = {
            "_QtReViewer": (
                ("queue_item_selection_changed", "on_queue_item_selection_changed"),
                ("status_changed", "on_update_widgets"),
            ),
            "_QtReEditor": (
                ("allowed_plans_changed", "_on_allowed_plans_changed"),
                ("status_changed", "on_update_widgets"),
            ),
            "QtReManagerConnection": (("status_changed", "on_update_widgets"),),
            "QtReEnvironmentControls": (("status_changed", "on_update_widgets"),),
            "QtReQueueControls": (("status_changed", "on_update_widgets"),),
            "QtReExecutionControls": (("status_changed", "on_update_widgets"),),
            "QtReStatusMonitor": (("status_changed", "on_update_widgets"),),
            "QtRePlanQueue": (
                ("status_changed", "on_update_widgets"),
                ("plan_queue_changed", "on_plan_queue_changed"),
                ("queue_item_selection_changed", "on_queue_item_selection_changed"),
            ),
            "QtRePlanHistory": (
                ("status_changed", "on_update_widgets"),
                ("plan_history_changed", "on_plan_history_changed"),
                ("history_item_selection_changed", "on_history_item_selection_changed"),
            ),
            "QtReRunningPlan": (
                ("running_item_changed", "on_running_item_changed"),
                ("status_changed", "on_update_widgets"),
            ),
        }

        for class_name, event_specs in widget_event_specs.items():
            widget_cls = getattr(bw_run_engine_client, class_name, None)
            if widget_cls is None or getattr(widget_cls, "_dc_event_disconnect_patch_applied", False):
                continue

            original_init = widget_cls.__init__

            def _make_disconnect(specs):
                def _disconnect(self, *_args):
                    model = getattr(self, "model", None)
                    events = getattr(model, "events", None)
                    if events is None:
                        return
                    for emitter_name, callback_name in specs:
                        emitter = getattr(events, emitter_name, None)
                        callback = getattr(self, callback_name, None)
                        if emitter is None or callback is None:
                            continue
                        try:
                            emitter.disconnect(callback)
                        except Exception:
                            pass
                return _disconnect

            disconnect_method = _make_disconnect(event_specs)

            def _make_patched_init(orig_init, disconnect_cb):
                def _patched_init(self, *args, **kwargs):
                    orig_init(self, *args, **kwargs)
                    self._dc_disconnect_model_events = lambda: disconnect_cb(self)
                    # Qt supplies a generic QObject to destroyed; retain the
                    # original Python widget that owns the model callbacks.
                    self.destroyed.connect(lambda *_args: disconnect_cb(self))
                return _patched_init

            widget_cls.__init__ = _make_patched_init(original_init, disconnect_method)
            widget_cls._dc_event_disconnect_patch_applied = True


    @staticmethod
    def _patch_bluesky_button_widths():
        """
        Patch bluesky_widgets PushButtonMinimumWidth to compute width using
        Qt style metrics (more reliable on macOS than deprecated fm.width()).
        """
        pb_cls = bw_run_engine_client.PushButtonMinimumWidth
        if getattr(pb_cls, "_dc_width_patch_applied", False):
            return

        def _button_width(button):
            option = QtWidgets.QStyleOptionButton()
            option.initFrom(button)
            option.text = button.text()
            option.icon = button.icon()
            option.iconSize = button.iconSize()
            fm = button.fontMetrics()
            contents = QtCore.QSize(max(fm.horizontalAdvance(button.text()), 0), fm.height())
            width = button.style().sizeFromContents(
                QtWidgets.QStyle.CT_PushButton, option, contents, button
            ).width()
            if button.menu() is not None:
                width += button.style().pixelMetric(
                    QtWidgets.QStyle.PM_MenuButtonIndicator, option, button
                )
            # Keep width text-driven to avoid oversized macOS minimum hints.
            return max(width + 2, fm.horizontalAdvance(button.text()) + 12)

        def _patched_init(self, *args, **kwargs):
            QtWidgets.QPushButton.__init__(self, *args, **kwargs)

            def _apply():
                self.setFixedWidth(_button_width(self))

            _apply()
            # Apply again after style polish; this fixes macOS sizing drift.
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(_apply)
            timer.start(0)

        pb_cls.__init__ = _patched_init
        pb_cls._dc_width_patch_applied = True


    @staticmethod
    def _patch_bluesky_console_theme_refresh():
        console_cls = bw_run_engine_client.QtReConsoleMonitor
        if getattr(console_cls, "_dc_theme_refresh_patch_applied", False):
            return

        original_init = console_cls.__init__
        original_change_event = getattr(console_cls, "changeEvent", None)
        original_finished = getattr(console_cls, "_finished_receiving_console_output", None)
        original_process = getattr(console_cls, "_process_new_console_output", None)
        original_update = console_cls._update_console_output

        def _apply_console_palette(self):
            text_edit = getattr(self, "_text_edit", None)
            if text_edit is None:
                return

            app = QtWidgets.QApplication.instance()
            base_palette = QtGui.QPalette(app.palette() if app is not None else self.palette())
            disabled_base = base_palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Base)
            base_palette.setColor(QtGui.QPalette.Base, disabled_base)
            text_edit.setPalette(base_palette)
            viewport = text_edit.viewport()
            if viewport is not None:
                viewport.setPalette(base_palette)
                viewport.update()
            text_edit.update()

        def _patched_init(self, *args, **kwargs):
            self._dc_console_stop_requested = False
            self._dc_console_closed = False
            self._dc_console_worker_pending = False
            original_init(self, *args, **kwargs)
            self.destroyed.connect(lambda *_args: _mark_console_closed(self))
            _apply_console_palette(self)

        def _patched_change_event(self, event):
            if callable(original_change_event):
                original_change_event(self, event)
            else:
                QtWidgets.QWidget.changeEvent(self, event)
            if event.type() in (
                QtCore.QEvent.PaletteChange,
                QtCore.QEvent.ApplicationPaletteChange,
            ):
                _apply_console_palette(self)

        def _poll_console_once(self):
            client = getattr(self.model, "_client", None)
            console_monitor = getattr(client, "console_monitor", None)
            if client is None or console_monitor is None:
                return None

            request_timeout_error = getattr(client, "RequestTimeoutError", Exception)
            while not getattr(self, "_dc_console_stop_requested", False):
                try:
                    payload = console_monitor.next_msg(timeout=0.2)
                    if payload is None:
                        continue
                    return payload.get("time", None), payload.get("msg", None)
                except request_timeout_error:
                    continue
                except Exception as ex:
                    print(f"Exception occurred: {ex}")
                    if getattr(self, "_dc_console_stop_requested", False):
                        break
            return None

        def _patched_start_thread(self):
            # FunctionWorker.is_running stays True after finished in the
            # installed version. Track queued/running work until its signal.
            if self._dc_console_stop_requested or self._dc_console_worker_pending:
                return
            self._dc_console_worker_pending = True
            self._thread = bw_run_engine_client.FunctionWorker(lambda: _poll_console_once(self))
            self._thread.returned.connect(self._process_new_console_output)
            self._thread.finished.connect(self._finished_receiving_console_output)
            self._thread.start()

        def _patched_finished_receiving_console_output(self):
            self._dc_console_worker_pending = False
            if getattr(self, "_dc_console_stop_requested", False):
                return
            if callable(original_finished):
                original_finished(self)

        def _patched_process_new_console_output(self, result):
            if result is None or getattr(self, "_dc_console_stop_requested", False):
                return
            if callable(original_process):
                original_process(self, result)

        def _mark_console_closed(self):
            self._dc_console_closed = True
            self._dc_console_stop_requested = True

        def _pause_console(self):
            self._dc_console_stop_requested = True
            timer = getattr(self, "_dc_console_update_timer", None)
            if timer is not None:
                timer.stop()
            self._text_edit.setUpdatesEnabled(True)

        def _resume_console(self):
            if self._dc_console_closed:
                return
            self._dc_console_stop_requested = False
            self._text_edit.setUpdatesEnabled(True)
            self._start_thread()
            self._start_timer()

        def _shutdown_console(self):
            _mark_console_closed(self)
            _pause_console(self)

        def _patched_start_timer(self):
            if self._dc_console_stop_requested:
                return
            timer = getattr(self, "_dc_console_update_timer", None)
            if timer is None:
                timer = QtCore.QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(self._update_console_output)
                self._dc_console_update_timer = timer
            if not timer.isActive():
                timer.start(195)

        def _patched_update_console_output(self):
            if not getattr(self, "_dc_console_stop_requested", False):
                original_update(self)

        def _patched_display_text(self):
            # Preserve upstream scroll behavior, but own the delayed callback
            # with the text edit so Qt cancels it when a screen is destroyed.
            if self._is_slider_pressed or getattr(self, "_dc_console_stop_requested", False):
                return
            value = self._text_edit.verticalScrollBar().value()
            lines = self._text_list[:-1] if self._text_list and self._text_list[-1] == "" else self._text_list
            self._text = "\n".join(lines)
            self._text_edit.setUpdatesEnabled(False)
            self._text_edit.setText(self._text)
            self._text_edit.verticalScrollBar().setValue(value)
            timer = QtCore.QTimer(self._text_edit)
            timer.setSingleShot(True)

            def restore_scroll():
                self._text_edit.setUpdatesEnabled(True)
                if not getattr(self, "_dc_console_stop_requested", False):
                    maximum = self._text_edit.verticalScrollBar().maximum()
                    self._text_edit.verticalScrollBar().setValue(maximum if self._te_scrolled_to_bottom else value)
                timer.deleteLater()

            timer.timeout.connect(restore_scroll)
            timer.start(50)

        console_cls.__init__ = _patched_init
        console_cls.changeEvent = _patched_change_event
        console_cls._start_thread = _patched_start_thread
        console_cls._start_timer = _patched_start_timer
        console_cls._finished_receiving_console_output = _patched_finished_receiving_console_output
        console_cls._process_new_console_output = _patched_process_new_console_output
        console_cls._update_console_output = _patched_update_console_output
        console_cls._display_text = _patched_display_text
        console_cls.__del__ = _mark_console_closed
        console_cls._dc_pause_console = _pause_console
        console_cls._dc_resume_console = _resume_console
        console_cls._dc_shutdown_console = _shutdown_console
        console_cls._dc_apply_console_palette = _apply_console_palette
        console_cls._dc_theme_refresh_patch_applied = True


def _patch_bluesky_status_reload_shutdown(cls):
    """Make shared Queue Server polling safe to stop during navigation."""
    if getattr(cls, "_dc_status_reload_shutdown_patch_applied", False):
        return

    def _patched_reload_status(self):
        self.model.load_re_manager_status()
        remaining = max(float(getattr(self, "update_period", 0) or 0), 0.0)
        while remaining > 0:
            if getattr(self, "_deactivate_updates", False):
                break
            delay = min(0.05, remaining)
            time.sleep(delay)
            remaining -= delay

    def _patched_reload_complete(self):
        if not self._deactivate_updates:
            self._start_thread()
            return
        # The RunEngineClient model is shared by all screens. A worker from a
        # detached screen must not clear state after another screen attaches.
        detaching = bool(getattr(self, "_dc_detaching", False))
        self._dc_detaching = False
        if not detaching:
            self.model.clear_connection_status()
        self.updates_activated = False
        self._deactivate_updates = False
        self._update_widget_states()

    cls._reload_status = _patched_reload_status
    cls._reload_complete = _patched_reload_complete
    cls._dc_status_reload_shutdown_patch_applied = True

def install_bluesky_compatibility():
    BlueskyCompatibility._patch_bluesky_model_event_disconnects()
    BlueskyCompatibility._patch_bluesky_button_widths()
    BlueskyCompatibility._patch_bluesky_console_theme_refresh()
    _patch_bluesky_status_reload_shutdown(bw_run_engine_client.QtReManagerConnection)
