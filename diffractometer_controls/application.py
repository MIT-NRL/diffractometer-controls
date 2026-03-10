import logging
import os
import json
import subprocess
import sys
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

import pydm
import pyqtgraph as pg
import qtawesome as qta
from pydm import data_plugins
from pydm.application import PyDMApplication
from pydm.utilities import path_info
from pydm.utilities.stylesheet import apply_stylesheet
from PyQt5.QtWidgets import QStyleFactory
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QAction
from bluesky_widgets.qt import run_engine_client as bw_run_engine_client

from main_window import MITRMainWindow
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher
# from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_queueserver_api.zmq import REManagerAPI

log = logging.getLogger(__name__)
THEME_MODE_SETTINGS_KEY = "appearance/theme_mode"
SETTINGS_ORGANIZATION = "MITR"
SETTINGS_APPLICATION = "MITR"
DEFAULT_QT_STYLE = "Fusion"


ui_dir = Path(__file__).parent


def _palette_is_dark(palette):
    window_color = palette.color(QtGui.QPalette.Window)
    return window_color.lightness() < 128


def _desktop_prefers_dark():
    override = os.environ.get("MITR_FORCE_DARK_MODE")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on", "dark"}

    gtk_theme = os.environ.get("GTK_THEME", "").strip().lower()
    if gtk_theme and "dark" in gtk_theme:
        return True

    for key in ("color-scheme", "gtk-theme"):
        try:
            proc = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", key],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except Exception:
            continue
        value = proc.stdout.strip().strip("'").lower()
        if key == "color-scheme" and value == "prefer-dark":
            return True
        if "dark" in value:
            return True

    return False


def _build_dark_palette():
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(240, 240, 240))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 80, 80))
    palette.setColor(QtGui.QPalette.Link, QtGui.QColor(66, 153, 225))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(66, 153, 225))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(15, 15, 15))
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.Base,
        QtGui.QColor(38, 38, 38),
    )
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.Window,
        QtGui.QColor(45, 45, 45),
    )
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.Text,
        QtGui.QColor(127, 127, 127),
    )
    palette.setColor(
        QtGui.QPalette.Disabled,
        QtGui.QPalette.ButtonText,
        QtGui.QColor(127, 127, 127),
    )
    return palette


def _sync_pyqtgraph_palette(palette):
    bg = palette.color(QtGui.QPalette.Window)
    fg = palette.color(QtGui.QPalette.WindowText)
    pg.setConfigOption("background", (bg.red(), bg.green(), bg.blue()))
    pg.setConfigOption("foreground", (fg.red(), fg.green(), fg.blue()))


def get_app_settings():
    return QtCore.QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


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

    @staticmethod
    def _patch_bluesky_console_theme_refresh():
        console_cls = bw_run_engine_client.QtReConsoleMonitor
        if getattr(console_cls, "_dc_theme_refresh_patch_applied", False):
            return

        original_init = console_cls.__init__
        original_change_event = getattr(console_cls, "changeEvent", None)

        def _apply_console_palette(self):
            text_edit = getattr(self, "_text_edit", None)
            if text_edit is None:
                return

            app = QtWidgets.QApplication.instance()
            base_palette = QtGui.QPalette(app.palette() if app is not None else self.palette())
            disabled_base = base_palette.color(QtGui.QPalette.Disabled, QtGui.QPalette.Base)
            base_palette.setColor(QtGui.QPalette.Base, disabled_base)
            text_edit.setPalette(base_palette)
            viewport = text_edit.viewport()
            if viewport is not None:
                viewport.setPalette(base_palette)
                viewport.update()
            text_edit.update()

        def _patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _apply_console_palette(self)

        def _patched_change_event(self, event):
            if callable(original_change_event):
                original_change_event(self, event)
            else:
                QtWidgets.QWidget.changeEvent(self, event)
            if event.type() in (
                QtCore.QEvent.PaletteChange,
                QtCore.QEvent.ApplicationPaletteChange,
            ):
                _apply_console_palette(self)

        console_cls.__init__ = _patched_init
        console_cls.changeEvent = _patched_change_event
        console_cls._dc_apply_console_palette = _apply_console_palette
        console_cls._dc_theme_refresh_patch_applied = True

    def __init__(self, ipaddress: str = 'localhost', ui_file: str = "main_screen.ui", use_main_window=False, *args, **kwargs):
        # Instantiate the parent class
        # (*ui_file* and *use_main_window* let us render the window here instead)

        # Create the RunEngineClient as part of the application attributes
        # These attributes need to be defined before the super().__init__ call so that the main window can access them
        self.re_client = RunEngineClient(zmq_control_addr=f'tcp://{ipaddress}:60615', zmq_info_addr=f'tcp://{ipaddress}:60625')
        self.re_dispatcher = RemoteDispatcher(f'{ipaddress}:5568')
        self.re_manager_api = REManagerAPI(zmq_control_addr=f'tcp://{ipaddress}:60615', zmq_info_addr=f'tcp://{ipaddress}:60625')
        self._ipaddress = str(ipaddress)
        self._startup_ui_file = ui_file
        self._patch_bluesky_button_widths()
        self._patch_bluesky_console_theme_refresh()
        self._theme_mode = "system"

        super().__init__(ui_file=ui_file, use_main_window=use_main_window, *args, **kwargs)
        base_style = QStyleFactory.create(DEFAULT_QT_STYLE)
        if base_style is not None:
            self.setStyle(base_style)
        self._base_style_name = self.style().objectName() or DEFAULT_QT_STYLE
        self._base_palette = QtGui.QPalette(self.palette())
        self.apply_theme_preference()
 
        # self.ui_file = ui_file
        # self.main_window = MI

        # self.main_window.ui.

    def __del__(self):
        pass
        # self.re_dispatcher.stop()

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

        dark_requested = _desktop_prefers_dark() if mode == "system" else (mode == "dark")

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
        launcher_path = ui_dir / "launcher.py"

        args = [
            sys.executable,
            str(launcher_path),
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

        subprocess.Popen(args, shell=False, cwd=str(ui_dir))


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
