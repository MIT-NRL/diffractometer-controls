import unittest

try:
    from diffractometer_controls.launcher import _patch_bluesky_status_reload_shutdown
except Exception:
    from launcher import _patch_bluesky_status_reload_shutdown


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
