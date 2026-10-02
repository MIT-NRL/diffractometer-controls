"""MITR adapter obtains application services before PyDM loads a screen."""
from qtpy import QtWidgets
from control_ui.core.display import ServiceDisplay

class MITRDisplay(ServiceDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename=None, services=None, **kwargs):
        app = QtWidgets.QApplication.instance()
        services = services or getattr(app, "control_services", None)
        super().__init__(parent, args, macros, ui_filename, services=services, **kwargs)
        self._navigation_active = True

    def cleanup_before_navigation(self):
        if self._navigation_active:
            self.deactivate_display()
            self._navigation_active = False

    def restore_after_navigation(self):
        if not self._navigation_active:
            self.activate_display()
            self._navigation_active = True
