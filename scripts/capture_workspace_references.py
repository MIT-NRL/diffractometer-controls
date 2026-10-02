"""Capture the real operator screens with all external connections disabled.

Run with the existing Bluesky Conda environment from the repository root.
Screenshots and the environment/baseline metadata go into ignored artifacts/.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "diffractometer_controls"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UI_DIR))


class FakeDocuments:
    def __init__(self):
        self.callbacks = {}
        self.next_token = 0

    def subscribe(self, callback, name="all"):
        self.next_token += 1
        self.callbacks[self.next_token] = (callback, name)
        return self.next_token

    def unsubscribe(self, token):
        self.callbacks.pop(token, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/reference_layouts/baseline")
    parser.add_argument("--width", type=int, default=2200)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--platform", default="windows" if sys.platform == "win32" else "offscreen")
    args = parser.parse_args()
    os.environ["QT_QPA_PLATFORM"] = args.platform
    os.environ.setdefault("MITR_FILE_DIR_QUERY_MODE", "local")
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PYDM_DISPLAYS_PATH"] = os.pathsep.join(
        [str(UI_DIR), str(UI_DIR / "extra_ui")]
    )

    from qtpy import QtCore, QtGui, QtWidgets
    from pydm.application import PyDMApplication
    from pydm.widgets.channel import PyDMChannel
    from bluesky_widgets.models.run_engine_client import RunEngineClient
    from bluesky_widgets.qt import run_engine_client as rec

    # Use actual model/widgets but block polling and all external I/O. No
    # MITRApplication constructor, remote document loop, or production endpoint.
    with patch("epics.caget", lambda *args, **kwargs: None), \
         patch("epics.caput", lambda *args, **kwargs: None), \
         patch.object(PyDMChannel, "connect", lambda self, **kwargs: None), \
         patch.object(PyDMChannel, "disconnect", lambda self, **kwargs: None), \
         patch.object(rec.QtReManagerConnection, "_start_thread", lambda self: None), \
         patch.object(rec.QtReConsoleMonitor, "_start_thread", lambda self: None):
        from application import MITRApplication, _build_dark_palette
        app = PyDMApplication(use_main_window=False, command_line_args=[], read_only=True)
        app.setStyle("Fusion")
        app.setFont(QtGui.QFont("Segoe UI" if sys.platform == "win32" else "DejaVu Sans", 10))
        app.re_client = RunEngineClient(
            zmq_control_addr="tcp://127.0.0.1:1",
            zmq_info_addr="tcp://127.0.0.1:2",
        )
        app.re_client.load_re_manager_status = lambda: None
        app.re_client.start_console_output_monitoring = lambda: None
        app.document_dispatcher = FakeDocuments()
        app.re_manager_api = SimpleNamespace()
        MITRApplication._patch_bluesky_model_event_disconnects()
        MITRApplication._patch_bluesky_button_widths()
        MITRApplication._patch_bluesky_console_theme_refresh()
        # The compatibility patch replaces _start_thread, so apply the I/O
        # guard again after it. Keep guards alive until all widgets are gone.
        guards = ExitStack()
        guards.enter_context(patch.object(rec.QtReConsoleMonitor, "_start_thread", lambda self: None))
        guards.enter_context(patch.object(rec.QtReConsoleMonitor, "_start_timer", lambda self: None))
        try:
            from control_ui.core.services import ControlServices
            app.control_services = ControlServices(
                app.re_client, app.document_dispatcher, app.re_manager_api
            )
        except ModuleNotFoundError:
            pass  # The same capture command also runs before extraction.

        from diffractometer_controls.diffractometer_gui import MainScreen as Diffraction
        from diffractometer_controls.tomography_gui import MainScreen as Tomography

        light_palette = QtGui.QPalette(app.palette())
        captures = []
        screens = []
        try:
            for theme in ("light", "dark"):
                app.setPalette(_build_dark_palette() if theme == "dark" else light_palette)
                for name, screen_class in (("diffraction", Diffraction), ("tomography", Tomography)):
                    screen = screen_class(macros={"P": "4dh4:", "R": ""})
                    screens.append(screen)
                    screen.resize(args.width, args.height)
                    screen.show()
                    for _ in range(12):
                        app.processEvents()
                        time.sleep(0.05)
                    labels = screen.findChildren(QtWidgets.QLabel)
                    substitutions = [
                        (label.font().family(), QtGui.QFontInfo(label.font()).family())
                        for label in labels
                        if QtGui.QFontInfo(label.font()).family().startswith("Font Awesome")
                        and not label.font().family().startswith("Font Awesome")
                    ]
                    if substitutions:
                        raise RuntimeError(
                            "Text fonts resolve to an icon font; use the native Windows "
                            f"Qt platform for capture. Substitutions: {set(substitutions)}"
                        )
                    filename = f"{name}_{theme}.png"
                    if not screen.grab().save(str(output / filename)):
                        raise RuntimeError(f"Unable to save {filename}")
                    captures.append({"file": filename, "theme": theme,
                                     "screen": name, "width": screen.width(),
                                     "height": screen.height()})
                    captures[-1]["text_fonts"] = sorted({
                        QtGui.QFontInfo(w.font()).family()
                        for w in screen.findChildren(QtWidgets.QLabel)
                    })
                    captures[-1]["requested_fonts"] = sorted({w.font().family() for w in screen.findChildren(QtWidgets.QLabel)})
                    captures[-1]["app_font"] = app.font().toString()
                    captures[-1]["screen_font"] = screen.font().toString()
                    screen.cleanup_before_navigation()
                    screen.close()
                    app.processEvents()
            metadata = {
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "python": sys.executable,
                "qt": QtCore.qVersion(),
                "platform": app.platformName(),
                "mode": "disconnected; channel I/O and QueueServer polling disabled",
                "captures": captures,
                "source_sha256": {
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [
                        UI_DIR / "diffractometer_gui.py", UI_DIR / "diffractometer_gui.ui",
                        UI_DIR / "tomography_gui.py", UI_DIR / "tomography_gui.ui",
                        ROOT / "control_ui" / "layouts" / "experiment_workspace.py",
                        ROOT / "control_ui" / "layouts" / "experiment_workspace.ui",
                    ] if path.exists()
                },
            }
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(metadata, indent=2))
        finally:
            for screen in screens:
                screen.cleanup_before_navigation()
                workspace = getattr(screen, "workspace", None)
                if workspace is not None:
                    workspace.shutdown()
                screen.deleteLater()
            app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
            app.processEvents()
            app.document_dispatcher.callbacks.clear()
            guards.close()


if __name__ == "__main__":
    main()
