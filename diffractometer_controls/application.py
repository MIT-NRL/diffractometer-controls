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
from bluesky_widgets.qt import run_engine_client as bw_run_engine_client

from main_window import MITRMainWindow
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api.zmq import REManagerAPI

log = logging.getLogger(__name__)


ui_dir = Path(__file__).parent


pg.setConfigOption("background", (252, 252, 252))
pg.setConfigOption("foreground", (0, 0, 0))


class MITRApplication(PyDMApplication):
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
            QtCore.QTimer.singleShot(0, _apply)

        pb_cls.__init__ = _patched_init
        pb_cls._dc_width_patch_applied = True

    def __init__(self, ipaddress: str = 'localhost', use_main_window=False, *args, **kwargs):
        # Instantiate the parent class
        # (*ui_file* and *use_main_window* let us render the window here instead)

        # Create the RunEngineClient as part of the application attributes
        # These attributes need to be defined before the super().__init__ call so that the main window can access them
        self.re_client = RunEngineClient(zmq_control_addr=f'tcp://{ipaddress}:60615', zmq_info_addr=f'tcp://{ipaddress}:60625')
        self.re_dispatcher = RemoteDispatcher(f'{ipaddress}:5568')
        self.re_manager_api = REManagerAPI(zmq_control_addr=f'tcp://{ipaddress}:60615', zmq_info_addr=f'tcp://{ipaddress}:60625')
        self._patch_bluesky_button_widths()

        super().__init__(ui_file='main_screen.ui', use_main_window=use_main_window, *args, **kwargs)
 
        # self.ui_file = ui_file
        # self.main_window = MI

        # self.main_window.ui.

    def __del__(self):
        pass
        # self.re_dispatcher.stop()


    # Redefine the make_main_window method to use the MITRMainWindow class
    def make_main_window(self, stylesheet_path=None, home_file=None, macros=None, command_line_args=None, **kwargs):
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
