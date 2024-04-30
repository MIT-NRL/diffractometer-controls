import logging
import subprocess
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import pydm
import pyqtgraph as pg
import qtawesome as qta
from pydm.application import PyDMApplication
from pydm.utilities.stylesheet import apply_stylesheet
from PyQt5.QtWidgets import QStyleFactory
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QAction

from main_window import MITRMainWindow
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher


log = logging.getLogger(__name__)


ui_dir = Path(__file__).parent


pg.setConfigOption("background", (252, 252, 252))
pg.setConfigOption("foreground", (0, 0, 0))


class MITRApplication(PyDMApplication):

    def __init__(self, ipaddress: str = 'localhost', use_main_window=False, *args, **kwargs):
        # Instantiate the parent class
        # (*ui_file* and *use_main_window* let us render the window here instead)

        # Create the RunEngineClient as part of the application attributes
        # These attributes need to be defined before the super().__init__ call so that the main window can access them
        self.re_client = RunEngineClient(zmq_control_addr=f'tcp://{ipaddress}:60615', zmq_info_addr=f'tcp://{ipaddress}:60625')
        # self.re_dispatcher = RemoteDispatcher(f'{ipaddress}:5567')

        super().__init__(ui_file='main_screen.py', use_main_window=use_main_window, *args, **kwargs)
 
        # self.ui_file = ui_file
        # self.main_window = MI

        # self.main_window.ui.

    def __del__(self):
        pass
        # self.re_dispatcher.stop()


    # Redefine the make_main_window method to use the MITRMainWindow class
    def make_main_window(self, stylesheet_path=None, home_file=None, macros=None, command_line_args=None):
        """
        Instantiate a new PyDMMainWindow, add it to the application's
        list of windows. Typically, this function is only called as part
        of starting up a new process, because PyDMApplications only have
        one window per process.
        """
        main_window = MITRMainWindow(
            # re_client=self.re_client,
            hide_nav_bar=self.hide_nav_bar,
            hide_menu_bar=self.hide_menu_bar,
            hide_status_bar=self.hide_status_bar,
            home_file=home_file,
            macros=macros,
            command_line_args=command_line_args,
        )


        self.main_window = main_window
        apply_stylesheet(stylesheet_path, widget=self.main_window)
        self.main_window.update_tools_menu()

        if self.fullscreen:
            main_window.enter_fullscreen()
        else:
            main_window.show()