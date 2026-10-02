"""Real Qt integration tests: missing GUI dependencies must fail this suite."""

import os
from pathlib import Path
import sys
import unittest
import warnings
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MITR_FILE_DIR_QUERY_MODE", "local")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diffractometer_controls"))
from diffractometer_controls.site.mitr.profile import display_directories
os.environ["PYDM_DISPLAYS_PATH"] = os.pathsep.join(map(str, display_directories()))

from qtpy import QtCore, QtWidgets
from bluesky_widgets.models.run_engine_client import RunEngineClient
from bluesky_widgets.qt import run_engine_client as rec
from pydm.widgets.channel import PyDMChannel
from pydm.display import load_file, ScreenTarget

from control_ui.core.compatibility import install_bluesky_compatibility
from control_ui.core.services import ControlServices
from control_ui.layouts.experiment_workspace import ExperimentWorkspace
from diffractometer_controls.screens.diffraction import diffractometer_gui
from diffractometer_controls.screens.tomography.tomography_gui import MainScreen as TomographyScreen


class FakeDocuments:
    def __init__(self):
        self.callbacks = {}
        self.token = 0

    def subscribe(self, callback):
        self.token += 1
        self.callbacks[self.token] = callback
        return self.token

    def unsubscribe(self, token):
        self.callbacks.pop(token, None)

    def emit(self, name, document):
        for callback in tuple(self.callbacks.values()):
            callback(name, document)


class WorkspaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_generic_layout_routes_plans_without_site_or_application_services(self):
        services = ControlServices(Mock(), Mock(), Mock())
        workspace = ExperimentWorkspace(services)
        self.addCleanup(workspace.deleteLater)
        editor = Mock()
        workspace.install_shared_controls(
            source_status=QtWidgets.QLabel("Beam status"),
            run_engine=QtWidgets.QWidget(), plans=QtWidgets.QWidget(),
            console_history=QtWidgets.QWidget(), plan_editor=editor,
        )
        workspace.install_experiment(
            viewer=QtWidgets.QWidget(), controls=QtWidgets.QWidget(),
            acquisition=QtWidgets.QWidget(),
        )
        item = {"name": "scan", "kwargs": {"duration": 2}}
        workspace.load_proposed_plan(item)
        editor.load_new_plan_item.assert_called_once_with(item, preserve_existing=True)
        self.assertIs(workspace.services, services)
        self.assertEqual(workspace.contentSplitter.count(), 3)
        self.assertGreaterEqual(workspace.plansSlot.minimumWidth(), 400)

    def test_component_lifecycle_is_idempotent_and_shutdown_is_terminal(self):
        workspace = ExperimentWorkspace(ControlServices(None, None, None))
        self.addCleanup(workspace.deleteLater)
        owner = Mock()
        workspace.register_component(owner)
        workspace.deactivate()
        workspace.deactivate()
        owner.deactivate.assert_called_once_with()
        workspace.activate()
        workspace.activate()
        owner.activate.assert_called_once_with()
        workspace.shutdown()
        workspace.shutdown()
        workspace.activate()
        self.assertEqual(owner.deactivate.call_count, 2)
        owner.shutdown.assert_called_once_with()
        self.assertEqual(owner.activate.call_count, 1)


class ExperimentScreenIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        install_bluesky_compatibility()
        self.guards = [
            patch.object(PyDMChannel, "connect", lambda *args, **kwargs: None),
            patch.object(PyDMChannel, "disconnect", lambda *args, **kwargs: None),
            patch.object(rec.QtReManagerConnection, "_start_thread", lambda self: None),
            patch.object(rec.QtReConsoleMonitor, "_start_thread", lambda self: None),
            patch.object(rec.QtReConsoleMonitor, "_start_timer", lambda self: None),
            patch("epics.caget", lambda *args, **kwargs: None),
            patch.object(diffractometer_gui, "_DIFFRACTION_PLOT_SUPPORT", {
                "controller_ok": True, "pyqtgraph_ok": True,
            }),
        ]
        for guard in self.guards:
            guard.start()
        self.client = RunEngineClient(
            zmq_control_addr="tcp://127.0.0.1:1", zmq_info_addr="tcp://127.0.0.1:2"
        )
        self.client.start_console_output_monitoring = lambda: None
        self.client._allowed_plans = {"empty_plan": {"name": "empty_plan", "parameters": []}}
        self.documents = FakeDocuments()
        self.services = ControlServices(self.client, self.documents, SimpleNamespace())
        self.screens = []
        self.initial_counts = self.callback_counts()

    def tearDown(self):
        for screen in self.screens:
            screen.workspace.shutdown()
            screen.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.processEvents()
        for guard in reversed(self.guards):
            guard.stop()

    def screen(self, cls):
        screen = cls(services=self.services, macros={"P": "4dh4:", "R": ""})
        self.screens.append(screen)
        self.app.processEvents()
        return screen

    def callback_counts(self):
        return {name: len(getattr(self.client.events, name).callbacks)
                for name in self.client.events}

    def test_both_modes_use_one_shell_and_keep_specific_tools(self):
        diffraction = self.screen(diffractometer_gui.MainScreen)
        tomography = self.screen(TomographyScreen)
        for screen in (diffraction, tomography):
            self.assertIsInstance(screen.workspace, ExperimentWorkspace)
            self.assertIs(screen.workspace.services, self.services)
            self.assertIs(screen.ui.epicsControls.parentWidget(), screen.workspace.controlsSlot)
        self.assertEqual(diffraction.workspace.viewerTabs.tabText(2), "Analyzer calculations")
        self.assertEqual(tomography.workspace.viewerTabs.tabText(1), "Tomo Calculator")
        self.assertIsNotNone(tomography.ui.cameraImage)
        self.assertIsNotNone(diffraction._diffraction_live_plot)

    def _load_mode_through_pydm(self, mode, filename):
        path = ROOT / "diffractometer_controls" / "screens" / mode / filename
        with patch.object(self.app, "control_services", self.services, create=True), \
             warnings.catch_warnings():
            warnings.filterwarnings("error", message="More than one Display class", category=RuntimeWarning)
            screen = load_file(str(path), macros={"P": "4dh4:", "R": ""}, target=ScreenTarget.HOME)
        self.screens.append(screen)
        self.app.processEvents()
        self.assertEqual(type(screen).__name__, "MainScreen")
        self.assertEqual(screen.loaded_file(), str(path))
        self.assertEqual(Path(screen.ui_filepath()), path.with_suffix(".ui"))
        self.assertIs(screen.workspace.services, self.services)
        self.assertIsNotNone(screen.workspace.plan_editor)
        return screen

    def test_diffraction_file_loads_its_main_screen_through_pydm(self):
        screen = self._load_mode_through_pydm("diffraction", "diffractometer_gui.py")
        self.assertIsNotNone(screen._diffraction_live_plot)
        self.assertEqual(screen.workspace.viewerTabs.tabText(2), "Analyzer calculations")

    def test_tomography_file_loads_its_main_screen_through_pydm(self):
        screen = self._load_mode_through_pydm("tomography", "tomography_gui.py")
        self.assertIsNotNone(screen.ui.cameraImage)
        self.assertEqual(screen.workspace.viewerTabs.tabText(1), "Tomo Calculator")

    def test_calculator_transfers_directly_to_the_installed_editor(self):
        tomography = self.screen(TomographyScreen)
        self.client._re_manager_connected = True
        self.assertTrue(tomography._load_tomography_recommendation_in_plan_editor(
            {"item_type": "plan", "name": "empty_plan", "kwargs": {}}
        ))
        editor = tomography.workspace.plan_editor
        self.assertEqual(editor._plan_editor._wd_editor.queue_item["name"], "empty_plan")

    def test_repeated_navigation_restores_callbacks_and_workers_without_duplicates(self):
        screen = self.screen(TomographyScreen)
        counts = self.callback_counts()
        for _ in range(4):
            screen.cleanup_before_navigation()
            screen.cleanup_before_navigation()
            self.assertIsNone(screen._profile_worker)
            self.assertIsNone(screen._live_filter_worker)
            editor = screen.workspace.plan_editor
            for owner in (editor._plan_viewer, editor._plan_editor):
                thread = owner._wd_editor._file_dir_query_thread
                if thread is not None:
                    self.assertFalse(thread.is_alive())
            screen.restore_after_navigation()
            screen.restore_after_navigation()
            self.app.processEvents()
            self.assertEqual(self.callback_counts(), counts)
            self.assertIsNotNone(screen._profile_worker)
            self.assertIsNotNone(screen._live_filter_worker)
        screen.workspace.shutdown()
        self.assertEqual(self.callback_counts(), self.initial_counts)

    def test_fast_navigation_does_not_start_duplicate_status_pollers(self):
        screen = self.screen(TomographyScreen)
        self.client._re_manager_connected = True
        with patch.object(rec.QtReManagerConnection, "_start_thread", autospec=True) as start:
            for _ in range(4):
                screen.cleanup_before_navigation()
                screen.restore_after_navigation()
            self.app.processEvents()
            self.assertEqual(start.call_count, 1)

    def test_destroying_a_screen_releases_owned_document_subscriptions(self):
        screen = self.screen(diffractometer_gui.MainScreen)
        self.screens.remove(screen)
        screen.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.assertEqual(self.documents.callbacks, {})

    def test_active_status_poller_restarts_once_after_acknowledging_stop(self):
        screen = self.screen(TomographyScreen)
        self.client._re_manager_connected = True
        panel = screen.workspace.runEngineSlot.layout().itemAt(0).widget()
        panel._re_manager.updates_activated = True
        screen.cleanup_before_navigation()
        panel._re_manager.updates_activated = False
        with patch.object(rec.QtReManagerConnection, "_start_thread", autospec=True) as start:
            screen.restore_after_navigation()
            self.app.processEvents()
            self.assertEqual(start.call_count, 1)

    def test_diffraction_retains_documents_while_hidden_and_unsubscribes_on_shutdown(self):
        screen = self.screen(diffractometer_gui.MainScreen)
        screen.cleanup_before_navigation()
        self.assertEqual(len(self.documents.callbacks), 2)
        controller = screen._diffraction_live_plot
        self.documents.emit("start", {
            "uid": "hidden-run", "time": 1.0, "plan_name": "count_scalar",
            "experiment_type": "diffraction", "data_type": "scalar",
            "detectors": ["sim_detector"], "motors": [],
        })
        self.app.processEvents()
        self.assertEqual(controller._current_live_run_uid, "hidden-run")
        screen.restore_after_navigation()
        self.assertEqual(controller._current_live_start_doc["uid"], "hidden-run")
        self.assertEqual(len(self.documents.callbacks), 2)
        screen.workspace.shutdown()
        self.assertEqual(self.documents.callbacks, {})

    def test_optional_plotting_failure_keeps_the_shared_controls_available(self):
        with patch.object(diffractometer_gui, "_DIFFRACTION_PLOT_SUPPORT", {
            "controller_ok": False, "message": "Plot backend unavailable",
        }):
            screen = self.screen(diffractometer_gui.MainScreen)
        self.assertIsNone(screen._diffraction_live_plot)
        self.assertIsNotNone(screen.workspace.plan_editor)
        self.assertIsNotNone(screen.findChild(QtWidgets.QFrame, "diffractionUnavailablePanel"))


if __name__ == "__main__":
    unittest.main()
