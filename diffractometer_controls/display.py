from pathlib import Path

from pydm import Display
from pydm.widgets.embedded_display import PyDMEmbeddedDisplay
from qtpy import QtWidgets

class MITRDisplay(Display):
    _RE_EMBEDDED_FILENAMES = {
        "re_control_panel.py",
        "re_plans.py",
        "re_extras.py",
    }

    def __init__(self, parent=None, args=None, macros=None, ui_filename=None, **kwargs):
        super().__init__(parent=parent, args=args, macros=macros, ui_filename=ui_filename, **kwargs)
        self._pending_embedded_reload = set()
        self.customize_ui()
        self._navigation_active = True

    def customize_ui(self):
        pass

    def activate_display(self):
        """Resume non-PyDM resources after this cached display is restored."""

    def deactivate_display(self):
        """Suspend non-PyDM resources before this display is cached."""

    def cleanup_before_navigation(self):
        if not getattr(self, "_navigation_active", True):
            return
        self._navigation_active = False
        try:
            self.deactivate_display()
        except Exception:
            pass

        pending_reload = set()
        for embedded in self.findChildren(PyDMEmbeddedDisplay):
            filename = Path(str(getattr(embedded, "filename", "") or "")).name
            if filename not in self._RE_EMBEDDED_FILENAMES:
                continue

            widget = getattr(embedded, "embedded_widget", None)
            if widget is None:
                continue

            disconnect_events = getattr(widget, "_dc_disconnect_model_events", None)
            if callable(disconnect_events):
                try:
                    disconnect_events()
                except Exception:
                    pass
            for child in widget.findChildren(QtWidgets.QWidget):
                child_disconnect = getattr(child, "_dc_disconnect_model_events", None)
                if callable(child_disconnect):
                    try:
                        child_disconnect()
                    except Exception:
                        pass

            cleanup = getattr(widget, "prepare_for_detach", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    pass

            embedded.embedded_widget = None
            try:
                embedded._needs_load = True
            except Exception:
                pass
            pending_reload.add(embedded)

        self._pending_embedded_reload = pending_reload

    def restore_after_navigation(self):
        if getattr(self, "_navigation_active", False):
            return
        pending_reload = tuple(getattr(self, "_pending_embedded_reload", ()) or ())
        self._pending_embedded_reload = set()
        for embedded in pending_reload:
            try:
                embedded.load_if_needed()
            except Exception:
                pass
        try:
            self.activate_display()
        except Exception:
            pass
        self._navigation_active = True
