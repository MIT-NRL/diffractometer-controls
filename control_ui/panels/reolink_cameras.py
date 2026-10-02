"""Two-camera wrapper which owns the embedded ADReolink displays."""

from pydm import Display
from pydm.widgets.embedded_display import PyDMEmbeddedDisplay


class ReolinkCamerasDisplay(Display):
    def ui_filename(self):
        return "reolink_cameras.ui"

    def closeEvent(self, event):
        # PyDMEmbeddedDisplay hides children when its parent closes but does not
        # close them.  Explicitly stop their p4p monitors and decode workers.
        for embedded in self.findChildren(PyDMEmbeddedDisplay):
            child = embedded.embedded_widget
            if child is not None:
                child.close()
        super().closeEvent(event)
