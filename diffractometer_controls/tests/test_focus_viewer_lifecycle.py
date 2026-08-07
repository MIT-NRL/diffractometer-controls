import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from diffractometer_controls.focus_online_viewer import _has_received_focus_frame
from diffractometer_controls.main_window import MITRMainWindow


class FocusViewerInterpreterTests(unittest.TestCase):
    def test_uses_running_interpreter_instead_of_stale_conda_prefix(self):
        running_python = "/opt/controls-env/bin/python"
        with (
            patch.object(sys, "executable", running_python),
            patch.dict(os.environ, {"CONDA_PREFIX": "/opt/stale-base"}),
        ):
            selected = MITRMainWindow._resolve_focus_python_executable(None)

        self.assertEqual(selected, Path(running_python))


class FocusViewerStartupTests(unittest.TestCase):
    def test_window_without_frames_has_not_completed_startup(self):
        bridge = SimpleNamespace(window=SimpleNamespace(frames=[]))

        self.assertFalse(_has_received_focus_frame(bridge))

    def test_first_valid_frame_completes_startup(self):
        bridge = SimpleNamespace(window=SimpleNamespace(frames=[object()]))

        self.assertTrue(_has_received_focus_frame(bridge))

    def test_missing_window_has_not_completed_startup(self):
        self.assertFalse(_has_received_focus_frame(SimpleNamespace(window=None)))


if __name__ == "__main__":
    unittest.main()
