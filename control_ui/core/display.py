"""Service-injected PyDM components with source-relative Designer resources."""

import inspect
from pathlib import Path

from pydm import Display


class ServiceDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None, ui_filename=None,
                 *, services, **kwargs):
        if services is None:
            raise ValueError("ControlServices must be supplied by the host application")
        from control_ui.core.compatibility import install_bluesky_compatibility
        install_bluesky_compatibility()
        self.services = services
        super().__init__(parent=parent, args=args, macros=macros,
                         ui_filename=ui_filename, **kwargs)
        self.customize_ui()

    def ui_filepath(self):
        filename = self.ui_filename()
        if filename is None:
            return None
        return str(Path(inspect.getfile(type(self))).resolve().parent / filename)

    def customize_ui(self):
        pass

    def activate_display(self):
        pass

    def deactivate_display(self):
        pass
