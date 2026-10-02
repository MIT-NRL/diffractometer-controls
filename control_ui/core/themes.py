"""Shared Qt/plot palettes and desktop theme detection."""
import os
import subprocess
import pyqtgraph as pg
from qtpy import QtGui

def _palette_is_dark(palette):
    window_color = palette.color(QtGui.QPalette.Window)
    return window_color.lightness() < 128

def _desktop_prefers_dark(override_env="CONTROL_UI_FORCE_DARK_MODE"):
    override = os.environ.get(override_env)
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
