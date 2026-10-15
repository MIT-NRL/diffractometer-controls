"""Service monitoring, maintenance guards and console ownership."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtWidgets

from diffractometer_controls.site.mitr import bluesky_servers as servers


class ServerMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.guards = [
            patch.object(servers.QueueMonitor, "start"),
            patch.object(servers.BlueskyServersPanel, "_start_journal"),
            patch.object(servers.ServiceRow, "refresh"),
            patch.object(servers, "local_maintenance_allowed", return_value=True),
            patch.object(servers.data_plugins, "is_read_only", return_value=False),
        ]
        for guard in self.guards:
            guard.start()
            self.addCleanup(guard.stop)

    def panel(self, host="localhost"):
        with patch.object(servers.platform, "system", return_value="Linux"):
            panel = servers.BlueskyServersPanel(host)
        self.addCleanup(panel.close)
        return panel

    def test_remote_connection_does_not_inspect_or_control_local_services(self):
        panel = self.panel("remote.example.invalid")
        self.assertFalse(panel.local)
        self.assertEqual(panel.rows, [])
        self.assertEqual(panel.tabs.count(), 1)
        servers.BlueskyServersPanel._start_journal.assert_not_called()

    def test_loopback_and_current_host_are_local(self):
        for host in ("localhost", "127.0.0.1", "::1", "[::1]", servers.socket.gethostname()):
            self.assertTrue(servers.is_local_host(host), host)
        self.assertFalse(servers.is_local_host("remote.example.invalid"))
        with patch.object(servers.socket, "getaddrinfo", return_value=[(None, None, None, None, ("192.0.2.7", 0))]):
            self.assertTrue(servers.is_local_host("192.0.2.7"))
            self.assertFalse(servers.is_local_host("192.0.2.8"))

    def test_cpu_delta_includes_workers_and_resets_on_restart(self):
        row = self.panel().rows[0]
        props = dict(LoadState="loaded", ActiveState="active", SubState="running", MainPID="42",
                     MemoryCurrent="104857600", TasksCurrent="16", CPUUsageNSec="1000000000",
                     ActiveEnterTimestampMonotonic="1000000")
        row.apply_properties(props, now=10)
        self.assertIn("CPU —", row.resources.text())
        row.apply_properties(dict(props, CPUUsageNSec="6000000000"), now=12)
        self.assertIn("CPU 250.0%", row.resources.text())
        self.assertIn("100.0 MiB", row.resources.text())
        self.assertIn("0h 00m 11s", row.resources.text())
        row.apply_properties(dict(props, MainPID="43"), now=14)
        self.assertIn("CPU —", row.resources.text())
        row.apply_properties(dict(props, CPUUsageNSec="[not set]", MemoryCurrent="[not set]"), now=16)
        self.assertIn("Memory —", row.resources.text())
        self.assertIn("CPU —", row.resources.text())

    def test_stale_unreachable_busy_and_readonly_status_block_destructive_controls(self):
        panel = self.panel()
        row = panel.rows[0]
        row.properties = {"LoadState": "loaded", "ActiveState": "active"}
        idle = dict(manager_state="idle", re_state="idle", worker_environment_state="idle",
                    running_item_uid="", worker_background_tasks=0)
        panel._queue_status_received(idle, "")
        self.assertTrue(row.can_control("restart"))
        for changes in ({"manager_state": "executing_queue"}, {"re_state": "paused"},
                        {"worker_environment_state": "executing_plan"},
                        {"running_item_uid": "running"}, {"worker_background_tasks": 1}):
            panel._queue_status_received(dict(idle, **changes), "")
            self.assertFalse(row.can_control("stop"), changes)
        panel._queue_status_received(idle, "")
        panel.status_time -= 10
        self.assertFalse(row.can_control("restart"))
        panel._queue_status_received({}, "Disconnected")
        self.assertFalse(row.can_control("stop"))
        self.assertIn("Disconnected", panel.queue_label.text())
        panel._queue_status_received(idle, "")
        with patch.object(servers.data_plugins, "is_read_only", return_value=True):
            self.assertFalse(row.can_control("restart"))
        with patch.object(servers, "local_maintenance_allowed", return_value=False):
            with patch.object(servers.QtCore, "QProcess") as process:
                row.control("restart")
                process.assert_not_called()

    def test_start_can_recover_stopped_service_with_queue_unreachable(self):
        row = self.panel().rows[0]
        row.properties = {"LoadState": "loaded", "ActiveState": "failed"}
        self.assertTrue(row.can_control("start"))
        self.assertFalse(row.can_control("stop"))
        row.properties["LoadState"] = "not-found"
        self.assertFalse(row.can_control("start"))

    def test_service_health_distinguishes_starting_stopped_failed_and_running(self):
        row = self.panel().rows[0]
        for active, substate, expected in (
            ("active", "running", "healthy"), ("active", "exited", "attention"),
            ("activating", "start", "active"), ("deactivating", "stop", "active"),
            ("inactive", "dead", "error"), ("failed", "failed", "error"),
        ):
            with self.subTest(state=active):
                row.apply_properties({"LoadState": "loaded", "ActiveState": active, "SubState": substate})
                self.assertEqual(row.status_badge.level, expected)
                self.assertTrue(row.status_badge.text())
        row.apply_properties({"LoadState": "not-found"})
        self.assertEqual(row.status_badge.level, "attention")

    def test_worker_health_does_not_treat_closed_as_failure_or_paused_as_idle(self):
        panel = self.panel()
        panel._queue_status_received(dict(manager_state="idle", worker_environment_state="closed",
                                         worker_environment_exists=False, re_state=None), "")
        self.assertEqual(panel.health_badges["manager_state"].level, "healthy")
        self.assertEqual(panel.health_badges["worker_environment_state"].level, "neutral")
        self.assertEqual(panel.health_badges["re_state"].level, "neutral")
        for state, expected in (("running", "active"), ("paused", "attention"), ("failed", "error")):
            panel._queue_status_received(dict(manager_state="executing_queue", worker_environment_state="executing_plan",
                                             worker_environment_exists=True, re_state=state), "")
            self.assertEqual(panel.health_badges["re_state"].level, expected)
        panel._queue_status_received({}, "Connection lost")
        self.assertTrue(all(badge.level == "error" for badge in panel.health_badges.values()))

    def test_console_shortcuts_select_the_corresponding_service_log(self):
        panel = self.panel()
        tiled_log = servers.ConsoleView()
        panel.tabs.addTab(tiled_log, "Tiled journal")
        panel._journal_views["tiled-server.service"] = tiled_log
        panel.rows[2].console_button.click()
        self.assertIs(panel.sections.currentWidget(), panel.tabs)
        self.assertIs(panel.tabs.currentWidget(), tiled_log)
        panel.rows[0].console_button.click()
        self.assertIs(panel.tabs.currentWidget(), panel.console)

    def test_open_worker_blocks_queue_restart(self):
        panel = self.panel()
        row = panel.rows[0]
        row.properties = {"LoadState": "loaded", "ActiveState": "active"}
        panel._queue_status_received(dict(manager_state="idle", re_state="idle",
            worker_environment_state="idle", worker_environment_exists=True), "")
        with patch.object(servers.QtWidgets.QMessageBox, "question", return_value=QtWidgets.QMessageBox.Yes), \
             patch.object(servers.QtWidgets.QMessageBox, "information"), \
             patch.object(servers.QtCore, "QProcess") as process:
            row.control("restart")
            process.assert_not_called()

    def test_console_plain_text_bounded_and_does_not_modify_server_output(self):
        view = servers.ConsoleView()
        self.addCleanup(view.close)
        view.append("<tag>\x1b[31mRED\x1b[0m\n")
        self.assertEqual(view.output.toPlainText(), "<tag>RED\n")
        self.assertTrue(view.output.isReadOnly())
        view.append("line\n" * 10020)
        self.assertLessEqual(view.output.document().blockCount(), 10000)
        view.clear()
        self.assertEqual(view.output.toPlainText(), "")

    def test_monitor_owns_api_console_and_closes_client(self):
        monitor = servers.QueueMonitor("localhost")
        api = Mock()
        api.RequestTimeoutError = TimeoutError
        api.status.return_value = {"manager_state": "idle"}

        def no_message():
            monitor.stop()
            raise TimeoutError()

        api.console_monitor.next_msg.side_effect = no_message
        with patch.object(servers, "REManagerAPI", return_value=api) as factory:
            monitor.run()
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["zmq_info_addr"], "tcp://localhost:60625")
        api.console_monitor.enable.assert_called_once()
        api.status.assert_called_once_with(reload=True)
        api.close.assert_called_once()

    def test_close_stops_timers_and_is_idempotent(self):
        panel = self.panel()
        panel.shutdown()
        panel.shutdown()
        self.assertFalse(panel.timer.isActive())
        self.assertTrue(panel.monitor._stop_event.is_set())
        self.assertTrue(all(row._closed for row in panel.rows))

    def test_panel_reference_is_retained_and_reused(self):
        from diffractometer_controls.main_window import MITRMainWindow

        panel = Mock()
        harness = SimpleNamespace(_bluesky_servers_panel=None,
                                  _bluesky_servers_panel_destroyed=Mock())
        with patch.object(servers, "BlueskyServersPanel", return_value=panel) as factory:
            MITRMainWindow.launch_bluesky_servers_panel(harness)
            MITRMainWindow.launch_bluesky_servers_panel(harness)
        factory.assert_called_once()
        self.assertIs(harness._bluesky_servers_panel, panel)
        panel.destroyed.connect.assert_called_once_with(harness._bluesky_servers_panel_destroyed)
        self.assertEqual(panel.show.call_count, 2)


if __name__ == "__main__":
    unittest.main()
