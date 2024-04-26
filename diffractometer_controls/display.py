from pathlib import Path
from typing import Optional, Sequence

from ophyd import Device
from pydm import Display
from qtpy import QtWidgets
from qtpy.QtCore import Signal, Slot

class MITRDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None, ui_filename=None, **kwargs):
        super().__init__(parent=parent, args=args, macros=macros, ui_filename=ui_filename, **kwargs)
        # self.ui_filename = ui_filename
        self.customize_ui()

    # def ui_filename(self):
    #     return self.ui_filename
    
    # def ui_filepath(self):
    #     return super().ui_filepath()

    def customize_ui(self):
        pass

