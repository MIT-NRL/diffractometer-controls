from control_ui.core.themes import _palette_is_dark, _desktop_prefers_dark, _build_dark_palette, _sync_pyqtgraph_palette
from control_ui.core.compatibility import BlueskyCompatibility
from diffractometer_controls.site.mitr import profile
import logging
import os
import json
import subprocess
import sys
from pathlib import Path

import pyqtgraph as pg
from pydm import data_plugins
from pydm.application import PyDMApplication
from pydm.utilities import path_info
from pydm.utilities.stylesheet import apply_stylesheet
from PyQt5.QtWidgets import QStyleFactory
from qtpy import QtCore, QtGui, QtWidgets
from bluesky_widgets.qt import run_engine_client as bw_run_engine_client

from diffractometer_controls.main_window import MITRMainWindow
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api.zmq import REManagerAPI
from control_ui.core.document_dispatcher import DocumentDispatcherService
from control_ui.core.services import ControlServices

log = logging.getLogger(__name__)
THEME_MODE_SETTINGS_KEY = "appearance/theme_mode"
SETTINGS_ORGANIZATION = "MITR"
SETTINGS_APPLICATION = "MITR"
DEFAULT_QT_STYLE = "Fusion"


ui_dir = Path(__file__).parent










def get_app_settings():
    return QtCore.QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


class MITRApplication(BlueskyCompatibility, PyDMApplication):



    def __init__(self, ipaddress: str = 'localhost', ui_file: str = "main_screen.ui", use_main_window=False, *args, **kwargs):
        # Instantiate the parent class
        # (*ui_file* and *use_main_window* let us render the window here instead)

        # Create the RunEngineClient as part of the application attributes
        # These attributes need to be defined before the super().__init__ call so that the main window can access them
        zmq_public_key = os.environ.get("QSERVER_ZMQ_PUBLIC_KEY") or None
        endpoints = profile.endpoints(ipaddress)
        self.re_client = RunEngineClient(zmq_control_addr=endpoints.control, zmq_info_addr=endpoints.info)
        self.re_dispatcher = RemoteDispatcher(endpoints.documents)
        self.document_dispatcher = DocumentDispatcherService(self.re_dispatcher)
        self.re_manager_api = REManagerAPI(
            zmq_control_addr=endpoints.control,
            zmq_info_addr=endpoints.info,
            zmq_public_key=zmq_public_key,
        )
        self.control_services = ControlServices(
            self.re_client, self.document_dispatcher, self.re_manager_api
        )
        self._ipaddress = str(ipaddress)
        self._startup_ui_file = ui_file
        self._patch_bluesky_model_event_disconnects()
        self._patch_bluesky_button_widths()
        self._patch_bluesky_console_theme_refresh()
        self._theme_mode = "system"

        super().__init__(ui_file=ui_file, use_main_window=use_main_window, *args, **kwargs)
        # Start exactly one Qt receive loop after the initial display has had a
        # chance to register its callbacks.  Later displays only subscribe.
        self.document_dispatcher.start()
        base_style = QStyleFactory.create(DEFAULT_QT_STYLE)
        if base_style is not None:
            self.setStyle(base_style)
        self._base_style_name = self.style().objectName() or DEFAULT_QT_STYLE
        self._base_palette = QtGui.QPalette(self.palette())
        self.apply_theme_preference()

    def theme_mode(self):
        return getattr(self, "_theme_mode", "system")

    def is_dark_theme_active(self):
        try:
            return _palette_is_dark(self.palette())
        except Exception:
            return False

    def _refresh_theme_for_widgets(self):
        for widget in self.allWidgets():
            try:
                style = widget.style()
                if style is not None:
                    style.unpolish(widget)
                    style.polish(widget)
                widget.update()
            except Exception:
                pass

    def _refresh_console_monitors(self):
        for widget in self.allWidgets():
            apply_palette = getattr(widget, "_dc_apply_console_palette", None)
            if not callable(apply_palette):
                continue
            try:
                apply_palette(widget)
            except Exception:
                pass

    def apply_theme_preference(self, mode=None):
        settings = get_app_settings()
        if mode is None:
            mode = str(settings.value(THEME_MODE_SETTINGS_KEY, "system")).strip().lower()
        if mode not in {"system", "light", "dark"}:
            mode = "system"

        dark_requested = _desktop_prefers_dark("MITR_FORCE_DARK_MODE") if mode == "system" else (mode == "dark")

        base_style = QStyleFactory.create(self._base_style_name)
        if base_style is not None:
            self.setStyle(base_style)

        if dark_requested:
            palette = _build_dark_palette()
        else:
            palette = QtGui.QPalette(self._base_palette)

        self.setPalette(palette)
        _sync_pyqtgraph_palette(palette)
        self._theme_mode = mode
        self._refresh_theme_for_widgets()
        self._refresh_console_monitors()
        return mode

    def new_pydm_process(self, ui_file, macros=None, command_line_args=None):
        ui_file = os.path.expanduser(os.path.expandvars(str(ui_file)))
        base_dir, fname, file_args = path_info(ui_file)
        filepath = os.path.join(base_dir, fname)


        args = [
            sys.executable,
            "-m", "diffractometer_controls",
            "--ip-addr",
            self._ipaddress,
            "--displayfile",
            filepath,
            "--hide-menu-bar",
            "--hide-nav-bar",
            "--hide-status-bar",
        ]
        if self.fullscreen:
            args.append("--fullscreen")
        if self.perfmon:
            args.append("--perfmon")
        if data_plugins.is_read_only():
            args.append("--read-only")
        if self.stylesheet_path:
            args.extend(["--stylesheet", self.stylesheet_path])
        if macros is not None:
            args.extend(["-m", json.dumps(macros)])
        args.extend(["--log_level", logging.getLevelName(logging.getLogger("").getEffectiveLevel())])

        extra_args = list(self.display_args)
        extra_args.extend(file_args)
        if command_line_args is not None:
            extra_args.extend(command_line_args)
        if extra_args:
            args.append("--")
            args.extend(extra_args)

        subprocess.Popen(args, shell=False, cwd=str(profile.CHECKOUT_ROOT))


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
