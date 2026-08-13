import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydm.utilities import import_module_by_filename
from qtpy import QtCore, QtTest, QtWidgets

from diffractometer_controls.launcher import _configure_epics_search_addresses
from diffractometer_controls.extra_ui.ioc_control_panel import (
    IOC_SPECS,
    IOCConsoleWindow,
    camera_log_command,
    clean_console_text,
    format_bytes,
    format_count,
    format_percent,
    parse_process_status,
    process_command,
    screen_input_commands,
)


class IOCSearchAddressTests(unittest.TestCase):
    def test_launcher_enables_lan_discovery_and_adds_private_endpoints(self):
        environ = {
            "EPICS_CA_ADDR_LIST": "upstream-ioc.example.invalid",
            "EPICS_PVA_ADDR_LIST": "upstream-ioc.example.invalid",
            "MITR_EPICS_CA_ADDR_LIST": (
                "control-host.example.invalid:6101 "
                "control-host.example.invalid:6102"
            ),
            "MITR_EPICS_PVA_ADDR_LIST": (
                "control-host.example.invalid:6201 "
                "control-host.example.invalid:6202"
            ),
        }
        with patch.dict(os.environ, environ, clear=True):
            _configure_epics_search_addresses("localhost")
            self.assertEqual(os.environ["EPICS_CA_AUTO_ADDR_LIST"], "YES")
            self.assertEqual(os.environ["EPICS_PVA_AUTO_ADDR_LIST"], "YES")
            self.assertEqual(
                os.environ["EPICS_CA_ADDR_LIST"],
                "upstream-ioc.example.invalid "
                "localhost "
                "control-host.example.invalid:6101 "
                "control-host.example.invalid:6102",
            )
            self.assertEqual(
                os.environ["EPICS_PVA_ADDR_LIST"],
                "upstream-ioc.example.invalid "
                "localhost "
                "control-host.example.invalid:6201 "
                "control-host.example.invalid:6202",
            )

    def test_explicit_demo_host_uses_standard_protocol_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            _configure_epics_search_addresses("demo-host.example.invalid")

            self.assertEqual(
                os.environ["EPICS_CA_ADDR_LIST"],
                "localhost demo-host.example.invalid",
            )
            self.assertEqual(
                os.environ["EPICS_PVA_ADDR_LIST"],
                "localhost demo-host.example.invalid",
            )


class IOCControlCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_enter_in_command_field_does_not_clear_console(self):
        window = IOCConsoleWindow(IOC_SPECS[0])
        try:
            window._poll_timer.stop()
            window.output.setPlainText("existing console output")
            window.command_edit.clear()
            window.show()
            window.command_edit.setFocus()

            with patch.object(window, "_start_next_main_command_step") as start:
                QtTest.QTest.keyClick(window.command_edit, QtCore.Qt.Key_Return)
                self.app.processEvents()

            start.assert_called_once()
            self.assertEqual(window.output.toPlainText(), "existing console output")
            for button in (
                window.clear_button,
                window.copy_button,
                window.bottom_button,
                window.find_button,
                window.send_button,
            ):
                self.assertFalse(button.autoDefault())
                self.assertFalse(button.isDefault())
        finally:
            window.close()

    def test_main_console_sends_input_to_screen_without_shell(self):
        self.assertEqual(
            screen_input_commands(" dbl "),
            (
                (
                    "screen",
                    ["-S", "4dh4ioc", "-p", "0", "-X", "stuff", "dbl"],
                ),
                (
                    "screen",
                    ["-S", "4dh4ioc", "-p", "0", "-X", "stuff", "\r"],
                ),
            ),
        )
        self.assertEqual(screen_input_commands("a\nb")[0][-1][-1], "a b")
        self.assertEqual(
            screen_input_commands(""),
            (
                (
                    "screen",
                    ["-S", "4dh4ioc", "-p", "0", "-X", "stuff", "\r"],
                ),
            ),
        )

    def test_camera_console_follows_journal_read_only(self):
        program, arguments = camera_log_command()
        self.assertEqual(program, "journalctl")
        self.assertIn("--unit=reolinkioc.service", arguments)
        self.assertIn("--follow", arguments)
        self.assertIn("--lines=2000", arguments)

    def test_console_log_control_sequences_are_removed(self):
        self.assertEqual(
            clean_console_text(
                "start\x1b[31m red\x1b[0m\r\n^[[A^[[Bnext\b!"
            ),
            "start red\nnex!",
        )

    def test_resource_formatters(self):
        self.assertEqual(format_percent(6.727), "6.7%")
        self.assertEqual(format_bytes(501190656), "478.0 MiB")
        self.assertEqual(format_bytes(1148481536), "1.1 GiB")
        self.assertEqual(format_count(1234.0), "1,234")

    def test_each_ioc_has_a_distinct_stats_prefix(self):
        self.assertEqual(IOC_SPECS[0].stats_prefix, "4dh4")
        self.assertEqual(IOC_SPECS[1].stats_prefix, "4dh4:ReolinkIOC")

    def test_panel_supports_pydm_dynamic_module_loader(self):
        panel_path = (
            Path(__file__).resolve().parents[1]
            / "extra_ui"
            / "ioc_control_panel.py"
        )
        module = import_module_by_filename(str(panel_path))
        self.assertTrue(hasattr(module, "IOCControlPanel"))

    def test_camera_control_uses_systemd_without_shell(self):
        self.assertEqual(
            process_command("camera", "restart"),
            ("systemctl", ["--user", "restart", "reolinkioc.service"]),
        )

    def test_main_status_parses_screen_launcher_output(self):
        running, detail = parse_process_status(
            "main", "4dh4 is running locally in a screen session (pid=123)\n", 0
        )
        self.assertTrue(running)
        self.assertIn("pid=123", detail)

    def test_camera_status_includes_pid(self):
        running, detail = parse_process_status(
            "camera", "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=42\n", 0
        )
        self.assertTrue(running)
        self.assertEqual(detail, "active/running  •  PID 42")


if __name__ == "__main__":
    unittest.main()
