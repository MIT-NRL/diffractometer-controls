from pathlib import Path
from os import path
import os
import time
from pydm import Display, PyDMApplication
from numpy.random import rand
from ophyd import (Device, Component as Cpt,
                    EpicsSignal, EpicsSignalRO, EpicsMotor)
from typhos.plugins import register_signal, SignalConnection, SignalPlugin
from IPython import start_ipython
from utils import register_all_signals


os.environ["EPICS_CA_ADDR_LIST"] = "10.149.6.227"
os.environ["EPICS_PVA_ADDR_LIST"] = "10.149.6.227"


motor1 = EpicsMotor('4dh4:m6',name='motor1')
# time.sleep(1)
# ic(motor1.read())
# motor1.move(-1)
# ic(motor1.read())

register_all_signals(motor1)

# sig = SignalConnection('sig://','motor1.user_readback')
# sig.add_listener(motor1)

ic(motor1.user_readback)

# print(motor1.read())
# print(motor1.user_readback.value)
# print(motor1.component_names)
# plugin = SignalPlugin()

class MyDisplay(Display):
    def __init__(self, parent=None, args=None, macros=None, ui_filename=None):
        super(MyDisplay, self).__init__(parent, args, macros, ui_filename)

        # self.ui.PyDMEmbeddedDisplay._embedded_widget.cameraImage.newImageSignal.connect(self.show_value)
        # print(self.ui.PyDMEmbeddedDisplay._embedded_widget.__dir__())

        # motor1 = EpicsMotor('4dh4:m6',name='motor1')
        # motor1.read()
        

        self.testValue = 0.0
        self.show_value(str(self.testValue))


    def ui_filename(self):
        return "../ui/main.ui"
    
    def ui_filepath(self):
        # Return the full path to the UI file
        return path.join(path.dirname(path.realpath(__file__)), self.ui_filename())

    def show_value(self, *args, **kwargs):
        self.ui.label_test.setText(str(rand()))


