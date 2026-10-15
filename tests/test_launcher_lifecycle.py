import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from diffractometer_controls.launcher import _patch_bluesky_status_reload_shutdown
except Exception:
    from launcher import _patch_bluesky_status_reload_shutdown


class SourceCheckoutLaunchTests(unittest.TestCase):
    def check_help(self, arguments, cwd):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment["QT_QPA_PLATFORM"] = "offscreen"
        result = subprocess.run(
            [sys.executable, *arguments, "--help"], cwd=cwd, env=environment,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--displayfile", result.stdout)
        self.assertIn("--ip-addr", result.stdout)

    def test_direct_script_from_application_directory_without_installed_package(self):
        root = Path(__file__).resolve().parents[1]
        self.check_help(["-I", "launcher.py"], root / "diffractometer_controls")

    def test_direct_script_from_an_unrelated_directory(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            self.check_help(["-I", str(root / "diffractometer_controls/launcher.py")], directory)

    def test_module_entry_point_from_checkout_root(self):
        root = Path(__file__).resolve().parents[1]
        self.check_help(["-m", "diffractometer_controls"], root)

    def test_directory_entry_point_from_checkout_root_without_installed_package(self):
        root = Path(__file__).resolve().parents[1]
        self.check_help(["-I", "diffractometer_controls"], root)

    def test_directory_entry_point_from_an_unrelated_directory(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            self.check_help(["-I", str(root / "diffractometer_controls")], directory)

    def test_main_file_entry_point_from_application_directory(self):
        root = Path(__file__).resolve().parents[1]
        self.check_help(["-I", "__main__.py"], root / "diffractometer_controls")


class _Model:
    def __init__(self):
        self.load_calls = 0
        self.clear_calls = 0

    def load_re_manager_status(self):
        self.load_calls += 1

    def clear_connection_status(self):
        self.clear_calls += 1


class _ManagerConnection:
    def __init__(self):
        self.model = _Model()
        self.update_period = 0
        self._deactivate_updates = False
        self.updates_activated = True
        self.start_calls = 0
        self.update_widget_calls = 0

    def _start_thread(self):
        self.start_calls += 1

    def _update_widget_states(self):
        self.update_widget_calls += 1


class QueueServerPollingLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _patch_bluesky_status_reload_shutdown(_ManagerConnection)

    def test_active_poller_schedules_next_update(self):
        manager = _ManagerConnection()

        manager._reload_complete()

        self.assertEqual(manager.start_calls, 1)

    def test_detached_poller_does_not_clear_shared_model_state(self):
        manager = _ManagerConnection()
        manager._deactivate_updates = True
        manager._dc_detaching = True

        manager._reload_complete()

        self.assertEqual(manager.model.clear_calls, 0)
        self.assertFalse(manager.updates_activated)
        self.assertFalse(manager._deactivate_updates)
        self.assertEqual(manager.update_widget_calls, 1)

    def test_user_disconnect_still_clears_connection_state(self):
        manager = _ManagerConnection()
        manager._deactivate_updates = True

        manager._reload_complete()

        self.assertEqual(manager.model.clear_calls, 1)


if __name__ == "__main__":
    unittest.main()
