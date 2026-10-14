"""Registered ownership for model subscriptions, Qt timers and component hooks."""
from qtpy import QtCore


def schedule_owned(owner, delay_ms, callback):
    """Cancel delayed UI work when its owner is deleted or deactivated."""
    timer = QtCore.QTimer(owner)
    timer.setSingleShot(True)
    timer.setProperty("dc_discard_on_deactivate", True)

    def deliver():
        try:
            callback()
        finally:
            timer.deleteLater()

    timer.timeout.connect(deliver)
    timer.start(delay_ms)
    return timer

class DisplayOwner:
    """Own the callbacks/timers introduced by one shared display factory."""

    def __init__(self, factory, services):
        events = services.re_client.events
        before = {name: getattr(events, name).callbacks for name in events}
        self.widget = factory()
        self._connections = [
            (getattr(events, name), callback)
            for name in events
            for callback in getattr(events, name).callbacks
            if callback not in before[name]
        ]
        self._timers = []
        self._active = True
        self._shutdown = False

    def deactivate(self):
        if not self._active:
            return
        hook = getattr(self.widget, "deactivate_display", None)
        if callable(hook):
            hook()
        timers = self.widget.findChildren(QtCore.QTimer)
        self._timers = [(timer, timer.interval()) for timer in timers
                        if timer.isActive() and not timer.property("dc_discard_on_deactivate")]
        for timer in timers:
            if timer.property("dc_discard_on_deactivate"):
                timer.stop()
                timer.deleteLater()
        for timer, _ in self._timers:
            timer.stop()
        for emitter, callback in self._connections:
            emitter.disconnect(callback)
        self._active = False

    def activate(self):
        if self._active or self._shutdown:
            return
        for emitter, callback in reversed(self._connections):
            emitter.connect(callback)
        for timer, interval in self._timers:
            timer.start(interval)
        hook = getattr(self.widget, "activate_display", None)
        if callable(hook):
            hook()
        self._active = True

    def shutdown(self):
        if self._shutdown:
            return
        self.deactivate()
        cleanup = getattr(self.widget, "prepare_for_detach", None)
        if callable(cleanup):
            cleanup()
        self._shutdown = True
