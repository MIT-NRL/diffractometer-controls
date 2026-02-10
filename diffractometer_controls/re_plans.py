import sys

from pydm.display import Display
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import (QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QScrollArea, QFrame,
    QApplication, QWidget, QLabel)
from bluesky_widgets.qt.run_engine_client import (
    QtReConsoleMonitor,
    QtReEnvironmentControls,
    QtReExecutionControls,
    QtReManagerConnection,
    QtRePlanEditor,
    QtRePlanHistory,
    QtRePlanQueue,
    QtReQueueControls,
    QtReRunningPlan,
    QtReStatusMonitor,
)
try:
    # Try to import the overrides module as a package (normal case).
    from diffractometer_controls.bluesky_config.plan_editor_overrides import (
        install_plan_editor_overrides,
        MyQtRePlanEditor,
    )
except Exception:
    # Fall back to loading the file directly when running without package context.
    try:
        import importlib.util
        import os

        _pkg_dir = os.path.join(os.path.dirname(__file__), "bluesky_config")
        _mod_path = os.path.join(_pkg_dir, "plan_editor_overrides.py")
        _spec = importlib.util.spec_from_file_location(
            "plan_editor_overrides", _mod_path
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        install_plan_editor_overrides = getattr(_mod, "install_plan_editor_overrides", lambda: None)
        MyQtRePlanEditor = getattr(_mod, "MyQtRePlanEditor", None) or QtRePlanEditor
    except Exception:
        # Final fallback: no-op installer and original QtRePlanEditor
        def install_plan_editor_overrides():
            return None

        MyQtRePlanEditor = QtRePlanEditor

from bluesky_widgets.models.run_engine_client import RunEngineClient
import display

class REPlans(display.MITRDisplay):
    def __init__(self, parent=None, args=None, macros=None, ui_filename='re_plans.ui'):
        super().__init__(parent, args, macros, ui_filename)
        # print("REScreen here")
        # self.customize_ui()

    def ui_filename(self):
        return 're_plans.ui'

    def ui_filepath(self):
        return super().ui_filepath()

    def customize_ui(self):
        from application import MITRApplication

        app = MITRApplication.instance()
        re_client = app.re_client

        # Install local plan-editor overrides (do this before creating widgets)
        try:
            install_plan_editor_overrides()
        except Exception:
            pass

        re_queue = QtRePlanQueue(re_client)
        # Create the standard QtRePlanEditor; `install_plan_editor_overrides`
        # monkeypatches the internal table class so the editor will use
        # `MyRePlanEditorTable` and show combo boxes for device choices.
        re_plan_editor = QtRePlanEditor(re_client)
        self.ui.RE_Queue.layout().addWidget(re_queue)
        self.ui.RE_Plan_Editor.layout().addWidget(re_plan_editor)
