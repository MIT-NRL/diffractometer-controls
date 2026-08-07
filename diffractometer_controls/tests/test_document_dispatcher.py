import unittest

try:
    from diffractometer_controls.document_dispatcher import DocumentDispatcherService
except Exception:
    from document_dispatcher import DocumentDispatcherService


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
