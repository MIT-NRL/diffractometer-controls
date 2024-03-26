from pathlib import Path
from os import path
from pydm import Display, PyDMApplication
from numpy.random import rand
from ophyd import (Device, Component as Cpt,
                    EpicsSignal, EpicsSignalRO, EpicsMotor)
from typhos.plugins import register_signal, SignalConnection, HappiConnection, SignalPlugin


print('here')

motor1 = EpicsMotor('4dh4:m6',name='motor1')
# print(motor1.user_readback.value)
# print(motor1.component_names)
plugin = SignalPlugin()
register_signal(motor1)

class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None, ui_filename=None):
        super(MyDisplay, self).__init__(parent, args, macros, ui_filename)

        self.ui.PyDMEmbeddedDisplay._embedded_widget.cameraImage.newImageSignal.connect(self.show_value)
        # print(self.ui.PyDMEmbeddedDisplay._embedded_widget.__dir__())

        self.testValue = 0.0
        self.show_value(str(self.testValue))


    def ui_filename(self):
        return "main.ui"
    
    def ui_filepath(self):
        # Return the full path to the UI file
        return path.join(path.dirname(path.realpath(__file__)), self.ui_filename())

    def show_value(self, *args, **kwargs):
        self.ui.label_test.setText(str(rand()))


