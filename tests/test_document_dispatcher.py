import unittest
import pickle
from unittest.mock import Mock, patch

try:
    from control_ui.core.document_dispatcher import DocumentDispatcherService
except Exception:
    from control_ui.core.document_dispatcher import DocumentDispatcherService


class _FakeCallbackDispatcher:
    def __init__(self):
        self.started = 0
        self.stopped = 0
        self.subscriptions = {}
        self.unsubscribed = []
        self._next_token = 0

    def subscribe(self, callback, name="all"):
        token = self._next_token
        self._next_token += 1
        self.subscriptions[token] = (callback, name)
        return token

    def unsubscribe(self, token):
        self.unsubscribed.append(token)
        self.subscriptions.pop(token, None)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


class _RemoteDispatcherShape(_FakeCallbackDispatcher):
    """Match bluesky-widgets, which hides unsubscribe on a child object."""

    unsubscribe = None

    def __init__(self):
        super().__init__()
        self._dispatcher = _FakeCallbackDispatcher()

    def subscribe(self, callback, name="all"):
        return self._dispatcher.subscribe(callback, name)


class DocumentDispatcherServiceTests(unittest.TestCase):
    def test_real_receiver_delivers_midrun_documents_only_to_opted_in_subscribers(self):
        from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher

        event = {"descriptor": "old-descriptor", "seq_num": 50, "data": {"det_counts": [1, 2]}}
        start = {"uid": "next-run"}
        messages = [b" event " + pickle.dumps(event), b" start " + pickle.dumps(start),
                    b" event " + pickle.dumps(event)]
        with patch("bluesky_widgets.qt.zmq_dispatcher.zmq.Context") as context:
            context.return_value.socket.return_value.recv.side_effect = messages
            dispatcher = RemoteDispatcher(("127.0.0.1", 5568))
        service = DocumentDispatcherService(dispatcher, receive_partial_runs=True)
        live_callback, full_run_callback = Mock(), Mock()
        service.subscribe(live_callback, accept_partial=True)
        service.subscribe(full_run_callback)
        receiver = dispatcher._receive_data()
        try:
            dispatcher._process_result(next(receiver))
            live_callback.assert_called_once_with("event", event)
            full_run_callback.assert_not_called()
            dispatcher._process_result(next(receiver))
            dispatcher._process_result(next(receiver))
            self.assertEqual(live_callback.call_count, 3)
            self.assertEqual(full_run_callback.call_count, 2)
            full_run_callback.assert_called_with("event", event)
        finally:
            receiver.close()
            service.stop()

    def test_partial_mode_removes_its_start_observer_and_callback_wrappers(self):
        dispatcher = _FakeCallbackDispatcher()
        service = DocumentDispatcherService(dispatcher, receive_partial_runs=True)
        service.subscribe(Mock(), accept_partial=True)
        service.subscribe(Mock())
        self.assertEqual(service.subscription_count, 2)
        service.stop()
        self.assertEqual(dispatcher.subscriptions, {})
        self.assertEqual(service._callbacks, {})

    def test_start_and_stop_are_idempotent(self):
        dispatcher = _FakeCallbackDispatcher()
        service = DocumentDispatcherService(dispatcher)

        self.assertTrue(service.start())
        self.assertFalse(service.start())
        self.assertEqual(dispatcher.started, 1)

        self.assertTrue(service.stop())
        self.assertFalse(service.stop())
        self.assertEqual(dispatcher.stopped, 1)

    def test_subscriptions_are_removed_explicitly(self):
        dispatcher = _FakeCallbackDispatcher()
        service = DocumentDispatcherService(dispatcher)
        callback = lambda *_: None

        token = service.subscribe(callback, name="start")
        self.assertEqual(service.subscription_count, 1)
        self.assertEqual(dispatcher.subscriptions[token], (callback, "start"))

        service.unsubscribe(token)
        service.unsubscribe(token)
        self.assertEqual(service.subscription_count, 0)
        self.assertNotIn(token, dispatcher.subscriptions)

    def test_supports_bluesky_widgets_unsubscribe_shape(self):
        dispatcher = _RemoteDispatcherShape()
        service = DocumentDispatcherService(dispatcher)

        token = service.subscribe(lambda *_: None)
        service.unsubscribe(token)

        self.assertNotIn(token, dispatcher._dispatcher.subscriptions)

    def test_stop_removes_all_subscriptions(self):
        dispatcher = _FakeCallbackDispatcher()
        service = DocumentDispatcherService(dispatcher)
        service.subscribe(lambda *_: None)
        service.subscribe(lambda *_: None, name="event")

        service.stop()

        self.assertEqual(service.subscription_count, 0)
        self.assertEqual(dispatcher.subscriptions, {})

    def test_subscribe_after_stop_is_rejected(self):
        service = DocumentDispatcherService(_FakeCallbackDispatcher())
        service.stop()

        with self.assertRaises(RuntimeError):
            service.subscribe(lambda *_: None)


if __name__ == "__main__":
    unittest.main()
