import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydm.utilities import import_module_by_filename
from qtpy import QtCore, QtTest, QtWidgets

from diffractometer_controls.launcher import (
    _configure_epics_search_addresses,
    _github_root,
    _preload_p4p,
)
from diffractometer_controls.site.mitr.ioc_control_panel import IOC_SPECS, IOCConsoleWindow, journal_command, clean_console_text, format_bytes, format_count, format_percent, parse_process_status, process_command
from diffractometer_controls.site.mitr.ioc_control_panel import _IOCRow, RESOURCE_METRICS


class IOCSearchAddressTests(unittest.TestCase):
    def test_linux_pva_preserves_existing_endpoints_without_list_syntax(self):
        with patch.dict(os.environ, {
            "EPICS_PVA_ADDR_LIST": "upstream.example.invalid:5076",
            "MITR_EPICS_PVA_ADDR_LIST": "control.example.invalid:6201",
        }, clear=True), patch(
            "diffractometer_controls.launcher.platform.system", return_value="Linux"
        ):
            _configure_epics_search_addresses("localhost")
            self.assertEqual(os.environ["EPICS_PVA_ADDR_LIST"],
                             "upstream.example.invalid:5076 localhost control.example.invalid:6201")
            _configure_epics_search_addresses("localhost")
            self.assertEqual(os.environ["EPICS_PVA_ADDR_LIST"],
                             "upstream.example.invalid:5076 localhost control.example.invalid:6201")

    def test_github_root_uses_override_when_configured(self):
        with patch.dict(os.environ, {"MITR_GITHUB_ROOT": "C:/custom/github"}, clear=True):
            self.assertEqual(_github_root(), Path("C:/custom/github"))

    def test_p4p_preload_is_optional(self):
        with patch(
            "diffractometer_controls.launcher.importlib.import_module",
            return_value=object(),
        ) as importer:
            self.assertTrue(_preload_p4p())
            importer.assert_called_once_with("p4p._p4p")

        with patch(
            "diffractometer_controls.launcher.importlib.import_module",
            side_effect=ImportError,
        ):
            self.assertFalse(_preload_p4p())

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
        with patch.dict(os.environ, environ, clear=True), patch(
            "diffractometer_controls.launcher.platform.system", return_value="Windows"
        ):
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
                "control-host.example.invalid:6201 "
                "control-host.example.invalid:6202",
            )

    def test_explicit_demo_host_uses_standard_protocol_defaults(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "diffractometer_controls.launcher.platform.system", return_value="Windows"
        ):
            _configure_epics_search_addresses("demo-host.example.invalid")

            self.assertEqual(
                os.environ["EPICS_CA_ADDR_LIST"],
                "localhost demo-host.example.invalid",
            )
            self.assertEqual(os.environ["EPICS_PVA_ADDR_LIST"], "")


class IOCControlCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_epics_health_is_independent_of_process_health_and_details_retain_metrics(self):
        with patch("pydm.widgets.channel.PyDMChannel.connect"), patch("pydm.widgets.channel.PyDMChannel.disconnect"):
            row = _IOCRow(IOC_SPECS[0], False)
            try:
                row.status_badge.set_status("Running", "healthy")
                for ok, text, level in ((True, "EPICS online", "healthy"),
                                        (False, "EPICS heartbeat stale", "attention"),
                                        (False, "EPICS disconnected", "error")):
                    row._set_epics_status(ok, text)
                    self.assertEqual(row.epics_label.level, level)
                    self.assertEqual(row.status_badge.level, "healthy")
                displayed = row.resources.values.keys() | row.diagnostics.values.keys()
                self.assertTrue({metric[1] for metric in RESOURCE_METRICS} <= displayed)
                self.assertTrue(row.diagnostics.isHidden())
                row.details_button.setChecked(True)
                self.assertFalse(row.diagnostics.isHidden())
                self.assertFalse(any(button.isEnabled() for button in row.buttons.values()))
            finally:
                row.close()
                row.deleteLater()
                self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_read_only_mode_allows_monitoring_but_blocks_process_and_console_commands(self):
        with patch("pydm.widgets.channel.PyDMChannel.connect"), patch("pydm.widgets.channel.PyDMChannel.disconnect"), \
             patch("pydm.data_plugins.is_read_only", return_value=True):
            row = _IOCRow(IOC_SPECS[0], True)
            window = IOCConsoleWindow(IOC_SPECS[0])
            try:
                self.assertFalse(any(button.isEnabled() for button in row.buttons.values()))
                self.assertTrue(window.output.isReadOnly())
                with patch.object(QtCore, "QProcess") as process:
                    row.control("start")
                    process.assert_not_called()
            finally:
                window.close()
                row.close()
                row.deleteLater()
                self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_all_consoles_are_read_only_journals(self):
        for spec in IOC_SPECS:
            with self.subTest(ioc=spec.key):
                program, arguments = journal_command(spec.key)
                self.assertEqual(program, "journalctl")
                self.assertIn("--unit=" + spec.service, arguments)
                self.assertIn("--follow", arguments)
                window = IOCConsoleWindow(spec)
                try:
                    self.assertTrue(window.output.isReadOnly())
                    self.assertFalse(hasattr(window, "command_edit"))
                    window._append_output("test_under_score\n")
                    self.assertEqual(window.output.toPlainText(), "test_under_score\n")
                    for button in (window.clear_button, window.copy_button,
                                   window.bottom_button, window.find_button):
                        self.assertFalse(button.autoDefault())
                finally:
                    window.close()

    def test_every_ioc_control_uses_its_own_service(self):
        for spec in IOC_SPECS:
            for action in ("start", "stop", "restart"):
                self.assertEqual(process_command(spec.key, action),
                                 ("systemctl", ["--user", action, spec.service]))
        with self.assertRaises(ValueError):
            process_command("unknown", "start")
        with self.assertRaises(ValueError):
            process_command("main", "enable")

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
        self.assertEqual(len(IOC_SPECS), 5)
        self.assertEqual(len({s.stats_prefix for s in IOC_SPECS}), 5)
        self.assertTrue(all(s.heartbeat == s.stats_prefix + ":HEARTBEAT" for s in IOC_SPECS))

    def test_panel_supports_pydm_dynamic_module_loader(self):
        panel_path = (
            Path(__file__).resolve().parents[1]
            / "diffractometer_controls" / "site" / "mitr"
            / "ioc_control_panel.py"
        )
        module = import_module_by_filename(str(panel_path))
        self.assertTrue(hasattr(module, "IOCControlPanel"))

    def test_camera_control_uses_systemd_without_shell(self):
        self.assertEqual(
            process_command("camera", "restart"),
            ("systemctl", ["--user", "restart", "reolinkioc.service"]),
        )

    def test_missing_service_is_reported(self):
        running, detail = parse_process_status("main", "LoadState=not-found\n", 1)
        self.assertFalse(running)
        self.assertEqual(detail, "service not installed")

    def test_camera_status_includes_pid(self):
        running, detail = parse_process_status(
            "camera", "LoadState=loaded\nActiveState=active\nSubState=running\nMainPID=42\n", 0
        )
        self.assertTrue(running)
        self.assertEqual(detail, "active/running  •  PID 42")


if __name__ == "__main__":
    unittest.main()
