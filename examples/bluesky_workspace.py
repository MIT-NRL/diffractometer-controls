"""Real shared Bluesky controls with a custom viewer and disconnected fake services.

Run: python examples/bluesky_workspace.py [--smoke-test]
The host owns services; replace them with your instrument's clients for live use.
"""

import argparse
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtpy import QtCore, QtWidgets
from bluesky_widgets.models.run_engine_client import RunEngineClient
from control_ui.core.compatibility import install_bluesky_compatibility
from control_ui.core.options import PlanEditorOptions
from control_ui.core.services import ControlServices
from control_ui.layouts.experiment_workspace import ExperimentWorkspace
from control_ui.widgets.source_status import SourceStatusIndicator


class FakeDocuments:
    def __init__(self):
        self.callbacks = {}
        self._token = 0

    def subscribe(self, callback, name="all"):
        self._token += 1
        self.callbacks[self._token] = callback
        return self._token

    def unsubscribe(self, token):
        self.callbacks.pop(token, None)


def fake_services():
    model = RunEngineClient(zmq_control_addr="tcp://127.0.0.1:1",
                            zmq_info_addr="tcp://127.0.0.1:2")
    # Keep the installed model's signals and editor metadata, with no transport.
    pause = threading.Event()
    model._client = SimpleNamespace(
        console_monitor=SimpleNamespace(next_msg=lambda **kwargs: (pause.wait(0.2) or None)),
        RequestTimeoutError=RuntimeError,
    )
    model.load_re_manager_status = lambda: None
    model.start_console_output_monitoring = lambda: None
    model._allowed_plans = {
        "example_count": {
            "name": "example_count", "description": "A local proposed plan",
            "parameters": [{"name": "duration", "kind": {"name": "POSITIONAL_OR_KEYWORD", "value": 1},
                            "default": "1.0", "annotation": {"type": "float"}}],
        }
    }
    return ControlServices(model, FakeDocuments(), SimpleNamespace())


def make_workspace(services):
    workspace = ExperimentWorkspace(services)
    workspace.install_bluesky_controls(
        source_status_factory=lambda: SourceStatusIndicator(units="mA", low=1, high=5),
        editor_options=PlanEditorOptions(query_mode="local"),
    )
    viewer = QtWidgets.QTableWidget(3, 2)
    viewer.setHorizontalHeaderLabels(["Position", "Signal"])
    for row, (position, value) in enumerate(((0, 12), (1, 34), (2, 19))):
        for column, number in enumerate((position, value)):
            viewer.setItem(row, column, QtWidgets.QTableWidgetItem(str(number)))
    controls = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(controls)
    layout.addWidget(QtWidgets.QLabel("Custom instrument controls"))
    button = QtWidgets.QPushButton("Propose an example count")
    button.clicked.connect(lambda: workspace.load_proposed_plan(
        {"item_type": "plan", "name": "example_count", "kwargs": {"duration": 2.0}}
    ))
    layout.addWidget(button)
    layout.addStretch()
    workspace.install_experiment(viewer=viewer, controls=controls,
                                 acquisition=QtWidgets.QLabel("Disconnected example • no acquisitions"))
    workspace.setWindowTitle("Instrument template — shared Bluesky workspace")
    return workspace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    install_bluesky_compatibility()
    workspace = make_workspace(fake_services())
    workspace.resize(1800, 1100)
    workspace.show()
    app.aboutToQuit.connect(workspace.shutdown)
    if args.smoke_test:
        assert not any(name.startswith("diffractometer_controls") for name in sys.modules)
        workspace.load_proposed_plan({"item_type": "plan", "name": "example_count", "kwargs": {"duration": 2.0}})
        workspace.deactivate()
        workspace.activate()
        QtCore.QTimer.singleShot(700, workspace.shutdown)
        QtCore.QTimer.singleShot(1200, app.quit)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
