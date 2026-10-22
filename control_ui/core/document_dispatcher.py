"""Single-owner lifecycle adapter for the Bluesky Qt document dispatcher."""


class DocumentDispatcherService:
    """Start one dispatcher once and provide symmetric subscriptions.

    ``bluesky_widgets.qt.zmq_dispatcher.RemoteDispatcher`` exposes the
    underlying Bluesky ``Dispatcher.subscribe`` method directly, but does not
    expose its matching ``unsubscribe`` method.  Keeping that compatibility
    detail here prevents individual displays from reaching into dispatcher
    internals or starting additional receive loops.
    """

    def __init__(self, dispatcher, *, receive_partial_runs=False):
        self.dispatcher = dispatcher
        self._started = False
        self._stopped = False
        self._tokens = set()
        self._callbacks = {}
        self._receive_partial_runs = bool(receive_partial_runs)
        self._waiting_for_start = bool(getattr(dispatcher, "_waiting_for_start", True))
        self._start_token = None
        if self._receive_partial_runs:
            self._start_token = dispatcher.subscribe(self._on_start, "start")
            # bluesky-widgets otherwise drops every document until the next
            # Start. Diffraction can recover context or display a live preview
            # without that Start; other subscribers retain the original gate.
            if hasattr(dispatcher, "_waiting_for_start"):
                dispatcher._waiting_for_start = False

    def _on_start(self, *_args):
        self._waiting_for_start = False

    @property
    def started(self):
        return self._started

    @property
    def stopped(self):
        return self._stopped

    @property
    def subscription_count(self):
        return len(self._tokens)

    def subscribe(self, callback, name="all", *, accept_partial=False):
        if self._stopped:
            raise RuntimeError("The document dispatcher has already stopped.")
        if self._receive_partial_runs and not accept_partial:
            original_callback = callback

            def callback(doc_name, doc):
                if not self._waiting_for_start:
                    original_callback(doc_name, doc)

        token = self.dispatcher.subscribe(callback, name)
        self._tokens.add(token)
        self._callbacks[token] = callback
        return token

    def unsubscribe(self, token):
        if token is None:
            return
        self._tokens.discard(token)
        self._callbacks.pop(token, None)
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
        if self._start_token is not None:
            self.unsubscribe(self._start_token)
            self._start_token = None
        self.dispatcher.stop()
        return True
