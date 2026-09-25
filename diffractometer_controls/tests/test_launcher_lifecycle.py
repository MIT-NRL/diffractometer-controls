import unittest
from unittest import mock

try:
    from diffractometer_controls import launcher
except Exception:
    import launcher

_patch_bluesky_status_reload_shutdown = launcher._patch_bluesky_status_reload_shutdown


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


class _FakeEpicsCA:
    def __init__(self):
        self.AUTO_CLEANUP = True
        self.disable_calls = 0
        self.finalize_calls = []

    def disable_ca_messages(self):
        self.disable_calls += 1

    def finalize_libca(self, *, maxtime):
        self.finalize_calls.append(maxtime)


class EpicsShutdownTests(unittest.TestCase):
    def test_process_exit_does_not_destroy_shared_libca_context(self):
        epics_ca = _FakeEpicsCA()

        with mock.patch.object(launcher.atexit, "unregister") as unregister:
            launcher._shutdown_epics_client(epics_ca)

        self.assertEqual(epics_ca.disable_calls, 1)
        self.assertEqual(epics_ca.finalize_calls, [])
        self.assertFalse(epics_ca.AUTO_CLEANUP)
        unregister.assert_called_once_with(epics_ca.finalize_libca)

    def test_late_initialization_cannot_register_an_exit_finalizer(self):
        epics_ca = _FakeEpicsCA()

        def initialize_when_silencing():
            # Pyepics initializes lazily, including when disabling messages.
            if epics_ca.AUTO_CLEANUP:
                launcher.atexit.register(epics_ca.finalize_libca)

        with mock.patch.object(epics_ca, "disable_ca_messages", side_effect=initialize_when_silencing):
            with mock.patch.object(launcher.atexit, "register") as register:
                launcher._shutdown_epics_client(epics_ca)

        register.assert_not_called()
        self.assertEqual(epics_ca.finalize_calls, [])

    def test_silencing_failure_does_not_leave_finalizer_registered(self):
        epics_ca = _FakeEpicsCA()
        with mock.patch.object(epics_ca, "disable_ca_messages", side_effect=RuntimeError):
            with mock.patch.object(launcher.atexit, "unregister") as unregister:
                launcher._shutdown_epics_client(epics_ca)

        unregister.assert_called_once_with(epics_ca.finalize_libca)
        self.assertFalse(epics_ca.AUTO_CLEANUP)
        self.assertEqual(epics_ca.finalize_calls, [])

    def test_shutdown_is_safe_when_repeated_or_epics_is_unused(self):
        epics_ca = _FakeEpicsCA()
        launcher._shutdown_epics_client(None)
        launcher._shutdown_epics_client(epics_ca)
        launcher._shutdown_epics_client(epics_ca)

        self.assertFalse(epics_ca.AUTO_CLEANUP)
        self.assertEqual(epics_ca.finalize_calls, [])


if __name__ == "__main__":
    unittest.main()
