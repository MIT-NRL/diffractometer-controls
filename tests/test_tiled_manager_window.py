"""Qt behavior with fake services; no sockets or production configuration."""

from datetime import datetime, timedelta, timezone
import json
import threading
import time
import unittest
from unittest.mock import patch

from qtpy import QtCore, QtWidgets, QtTest

from tiled_manager.window import TiledManagerWindow, SecretDialog, date, key_expiry
from tiled_manager import auth


class FakeService:
    def __init__(self, uri, catalogs):
        self.uri = uri
        self.client = True
        self.closed = False
        self.manager = None

    def connect(self):
        return self.refresh()

    def refresh(self):
        return {'uri': self.uri, 'identity': 'reader', 'server_version': 'test-version',
                'client_version': 'test-version', 'checked_at': '2026-10-23T12:00:00+00:00',
                'response_ms': 5, 'catalogs': {'Production': {'state': 'Readable', 'count': 3}},
                'reader_key': {'first_eight': '12345678'}}

    def close_manager(self):
        self.manager = None

    def close(self):
        self.closed = True


class ManagerWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.policy = patch.object(auth, 'management_allowed', return_value=False)
        self.policy.start()
        self.addCleanup(self.policy.stop)
        self.window = TiledManagerWindow('http://test', catalogs={'Production': 'production'}, service_factory=FakeService)
        self.window.show()
        self.addCleanup(self.window.close)
        self.wait()

    def wait(self):
        deadline = time.monotonic() + 3
        while (self.window.busy or not self.window.snapshot) and time.monotonic() < deadline:
            QtTest.QTest.qWait(10)
        self.assertFalse(self.window.busy)

    def test_layout_monitoring_and_diagnostics_allowlist(self):
        self.assertEqual(self.window.tabs.count(), 4)
        self.assertIn('Connected as reader', self.window.banner.text())
        self.assertFalse(self.window.login_button.isEnabled())
        self.assertFalse(self.window.issue_button.isEnabled())
        data = json.loads(self.window.diagnostics.toPlainText())
        self.assertNotIn('reader_key', data)
        self.assertNotIn('password', data)
        self.assertEqual(data['catalogs']['Production']['count'], 3)

    def test_nonoverlap_and_stale_completion(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        self.window.submit('refresh', lambda: (gate.wait(2), self.window.service.refresh())[1])
        self.assertFalse(self.window.submit('refresh', lambda: None))
        old_generation = self.window.generation
        self.window.server.setText('http://different')
        self.window.connect_server()
        self.window._complete(old_generation, 'refresh', {'password': 'test-secret'}, None)
        self.assertNotIn('test-secret', self.window.diagnostics.toPlainText())
        gate.set()
        self.wait()
        self.assertEqual(self.window.snapshot['uri'], 'http://different')

    def test_lock_and_idle_timeout_clear_manager_controls(self):
        self.window.manager_active = True
        self.window.last_activity = time.monotonic() - 601
        self.window._tick()
        self.assertFalse(self.window.manager_active)
        self.assertEqual(self.window.keys_table.rowCount(), 0)
        self.assertFalse(self.window.issue_button.isEnabled())

    def test_modal_input_does_not_lose_worker_to_automatic_refresh(self):
        dialog = QtWidgets.QDialog(self.window)
        dialog.setModal(True)
        dialog.show()
        try:
            QtTest.QTest.qWait(10)
            self.assertIs(QtWidgets.QApplication.activeModalWidget(), dialog)
            with patch.object(self.window, 'refresh') as refresh:
                self.window._tick()
                refresh.assert_not_called()
            self.assertFalse(self.window.busy)
        finally:
            dialog.close()
        with patch.object(self.window, 'refresh') as refresh:
            self.window._tick()
            refresh.assert_called_once()

    def test_failed_operation_marks_status_stale_without_exception_contents(self):
        self.window._complete(self.window.generation, 'refresh', None, 'Unreachable: check server.')
        self.assertIn('stale', self.window.banner.text())
        self.assertIn('Unreachable', self.window.cards[0].text())

    def test_msgpack_datetime_keys_survive_repeated_refresh(self):
        now = datetime.now(timezone.utc)
        keys = [{'note': 'test-laptop', 'first_eight': '12345678',
                 'time_created': now, 'latest_activity': now,
                 'expiration_time': now + timedelta(days=365),
                 'scopes': ['read:metadata', 'read:data']}]
        for _ in range(2):
            self.window._complete(self.window.generation, 'keys', keys, None)
            self.assertEqual(self.window.keys_table.rowCount(), 1)
            self.assertNotEqual(self.window.keys_table.item(0, 2).text(), '—')
            self.assertNotEqual(self.window.keys_table.item(0, 3).text(), '—')
            self.assertNotIn('Invalid', self.window.keys_table.item(0, 4).text())

    def test_dates_accept_json_msgpack_and_epoch_and_handle_invalid_values(self):
        now = datetime(2026, 10, 23, 12, tzinfo=timezone.utc)
        forms = (now, now.replace(tzinfo=None), now.isoformat(),
                 '2026-10-23T12:00:00Z', now.timestamp())
        self.assertEqual(len({date(value) for value in forms}), 1)
        for value in forms:
            self.assertIn('≤30 days', key_expiry(value, now=now - timedelta(days=1)))
            self.assertIn('Expired', key_expiry(value, now=now + timedelta(days=1)))
        self.assertNotIn('⚠', key_expiry(now + timedelta(days=31), now=now))
        self.assertEqual(key_expiry(None), 'No expiry')
        self.assertEqual(key_expiry(''), 'No expiry')
        for value in ('bad date', {}, [], 1e99):
            self.assertEqual(key_expiry(value), 'Invalid expiry')
            self.assertEqual(date(value), '—')

    def test_secret_dialog_clears_owned_clipboard_and_failed_export_keeps_secret(self):
        dialog = SecretDialog('http://test', 'test-laptop', 'test-only-secret', '12345678', self.window)
        dialog.copy()
        with patch.object(QtWidgets.QFileDialog, 'getSaveFileName', return_value=('/invalid/test', '')), \
             patch.object(auth, 'export_setup', side_effect=OSError), \
             patch.object(QtWidgets.QMessageBox, 'warning') as warning:
            dialog.export()
            self.assertEqual(dialog.secret, 'test-only-secret')
            warning.assert_called_once()
        dialog.reject()
        self.assertEqual(dialog.secret, '')
        self.assertEqual(self.app.clipboard().text(), '')


if __name__ == '__main__':
    unittest.main()
