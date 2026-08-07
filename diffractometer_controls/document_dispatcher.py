"""Single-owner lifecycle adapter for the Bluesky Qt document dispatcher."""


class DocumentDispatcherService:
    """Start one dispatcher once and provide symmetric subscriptions.

    ``bluesky_widgets.qt.zmq_dispatcher.RemoteDispatcher`` exposes the
    underlying Bluesky ``Dispatcher.subscribe`` method directly, but does not
    expose its matching ``unsubscribe`` method.  Keeping that compatibility
    detail here prevents individual displays from reaching into dispatcher
    internals or starting additional receive loops.
    """

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher
        self._started = False
        self._stopped = False
        self._tokens = set()

    @property
    def started(self):
        return self._started

    @property
    def stopped(self):
        return self._stopped

    @property
    def subscription_count(self):
        return len(self._tokens)

    def subscribe(self, callback, name="all"):
        if self._stopped:
            raise RuntimeError("The document dispatcher has already stopped.")
        token = self.dispatcher.subscribe(callback, name)
        self._tokens.add(token)
        return token

    def unsubscribe(self, token):
        if token is None:
            return
        self._tokens.discard(token)
        unsubscribe = getattr(self.dispatcher, "unsubscribe", None)
        if not callable(unsubscribe):
            unsubscribe = getattr(
                getattr(self.dispatcher, "_dispatcher", None),
                "unsubscribe",
                None,
            )
        if callable(unsubscribe):
            unsubscribe(token)

    def start(self):
        if self._started or self._stopped:
            return False
        self._started = True
        try:
            self.dispatcher.start()
        except Exception:
            self._started = False
            raise
        return True

    def stop(self):
        if self._stopped:
            return False
        self._stopped = True
        for token in tuple(self._tokens):
            self.unsubscribe(token)
        self.dispatcher.stop()
        return True
