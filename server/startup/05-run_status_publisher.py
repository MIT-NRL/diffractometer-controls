import atexit
import signal
import time
from server.services.run_status import _RunStatusPublisher, _safe_caput

def _mark_worker_closed():
    now = time.time()
    _safe_caput("State", "CLOSED")
    _safe_caput("Suspended", 0)
    _safe_caput("SuspendSinceEpoch", 0.0)
    _safe_caput("SuspendReason", "")
    _safe_caput("FinishEpoch", now)
    _safe_caput("LastUpdateEpoch", now)

_run_status_publisher = _RunStatusPublisher()
RE.subscribe(_run_status_publisher)

def _publish_run_progress(*, done_units=None, total_units=None, finish_epoch=None, now=None):
    _run_status_publisher.publish_progress(
        done_units=done_units,
        total_units=total_units,
        finish_epoch=finish_epoch,
        now=now,
    )

_existing_state_hook = RE.state_hook
if getattr(_existing_state_hook, "_run_status_wrapper", False):
    _previous_state_hook = getattr(_existing_state_hook, "_run_status_previous", None)
else:
    _previous_state_hook = _existing_state_hook

def _state_hook_with_status(*args, _previous_hook=_previous_state_hook, **kwargs):
    state = kwargs.get("new_state", kwargs.get("state", None))
    if state is None:
        str_args = [a for a in args if isinstance(a, str)]
        if str_args:
            state = str_args[0]
    if isinstance(state, str):
        state_lower = state.lower()
        if _run_status_publisher._run_active:
            now = time.time()
            if state_lower in ("suspending", "suspended"):
                _run_status_publisher._set_paused(True, now=now)
                _run_status_publisher._set_suspended(True, now=now)
            elif state_lower in ("pausing", "paused"):
                _run_status_publisher._set_suspended(False, now=now)
                _run_status_publisher._set_paused(True, now=now)
            elif state_lower in ("running", "executing"):
                _run_status_publisher._set_suspended(False, now=now)
                _run_status_publisher._set_paused(False, now=now)
            elif state_lower == "idle" and _run_status_publisher._run_paused:
                # RE may transiently report idle while paused at a checkpoint.
                _safe_caput(
                    "State",
                    "SUSPENDED" if _run_status_publisher._run_suspended else "PAUSED",
                )
                _safe_caput("LastUpdateEpoch", now)

    if callable(_previous_hook):
        return _previous_hook(*args, **kwargs)
    return None

_state_hook_with_status._run_status_wrapper = True
_state_hook_with_status._run_status_previous = _previous_state_hook
RE.state_hook = _state_hook_with_status

atexit.register(_mark_worker_closed)

def _install_shutdown_signal(sig):
    previous_handler = signal.getsignal(sig)

    def _handler(signum, frame):
        _mark_worker_closed()
        if callable(previous_handler):
            return previous_handler(signum, frame)
        raise SystemExit(0)

    try:
        signal.signal(sig, _handler)
    except Exception:
        pass

for _sig in (signal.SIGTERM, signal.SIGINT):
    _install_shutdown_signal(_sig)
