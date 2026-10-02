"""A standalone fake-service workspace with no MITR, EPICS, or Bluesky imports."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qtpy import QtWidgets
from control_ui.core.services import ControlServices
from control_ui.layouts.experiment_workspace import ExperimentWorkspace


class ExamplePlanEditor(QtWidgets.QPlainTextEdit):
    def load_new_plan_item(self, item, *, preserve_existing=True):
        self.setPlainText(json.dumps(item, indent=2))
        return True


def main():
    app = QtWidgets.QApplication(sys.argv)
    services = ControlServices(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
    workspace = ExperimentWorkspace(services)
    workspace.setWindowTitle("Shared workspace — example instrument")
    editor = ExamplePlanEditor()
    editor.setPlaceholderText("Proposed plan")
    run_controls = QtWidgets.QPushButton("Load example plan")
    run_controls.clicked.connect(lambda: workspace.load_proposed_plan({
        "name": "example_scan", "kwargs": {"points": 10},
    }))
    workspace.install_shared_controls(
        source_status=QtWidgets.QLabel("Source status\nSimulated / available"),
        run_engine=run_controls, plans=editor,
        console_history=QtWidgets.QPlainTextEdit("Example console and history"),
        plan_editor=editor,
    )
    data = QtWidgets.QTableWidget(3, 2)
    data.setHorizontalHeaderLabels(["Position", "Intensity"])
    for row, (position, intensity) in enumerate([(0, 12), (1, 35), (2, 18)]):
        for col, value in enumerate((position, intensity)):
            data.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))
    progress = QtWidgets.QProgressBar()
    progress.setValue(60)
    progress.setFormat("Example acquisition: %p%")
    workspace.install_experiment(
        viewer=data, controls=QtWidgets.QLabel("Instrument-specific controls"),
        acquisition=progress,
    )
    workspace.resize(1400, 900)
    workspace.show()
    app.aboutToQuit.connect(workspace.shutdown)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
