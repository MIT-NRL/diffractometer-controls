import atexit
import signal
import time

from epics import caput


STATUS_PV_PREFIX = "4dh4:Bluesky:Run:"


def _safe_caput(suffix, value):
    try:
        caput(f"{STATUS_PV_PREFIX}{suffix}", value, wait=False)
    except Exception:
        pass


class _RunStatusPublisher:
    def __init__(self):
        self._descriptor_stream = {}
        self._run_uid = ""
        self._run_active = False
        self._run_paused = False
        self._run_suspended = False
        self._suspend_since_epoch = 0.0
        self._finish_epoch = 0.0
        self._finish_epoch_before_suspend = 0.0

    def _set_finish_epoch(self, value):
        self._finish_epoch = float(value)
        _safe_caput("FinishEpoch", self._finish_epoch)

    def _set_suspended(self, suspended, now, reason=""):
        suspended = bool(suspended)
        if suspended == self._run_suspended:
            return

        if suspended:
            self._run_suspended = True
            self._suspend_since_epoch = float(now)
            self._finish_epoch_before_suspend = self._finish_epoch
            _safe_caput("Suspended", 1)
            _safe_caput("SuspendSinceEpoch", self._suspend_since_epoch)
            _safe_caput("SuspendReason", str(reason)[:200])
            if self._run_active:
                _safe_caput("State", "SUSPENDED")
            self._set_finish_epoch(0.0)
            _safe_caput("LastUpdateEpoch", now)
        else:
            # Shift ETA by suspension duration so countdown resumes smoothly.
            if self._suspend_since_epoch > 0 and self._finish_epoch_before_suspend > 0:
                self._finish_epoch = self._finish_epoch_before_suspend + max(
                    0.0, float(now) - self._suspend_since_epoch
                )
            self._run_suspended = False
            self._suspend_since_epoch = 0.0
            self._finish_epoch_before_suspend = 0.0
            _safe_caput("Suspended", 0)
            _safe_caput("SuspendSinceEpoch", 0.0)
            _safe_caput("SuspendReason", "")
            if self._run_active and self._run_paused:
                _safe_caput("State", "PAUSED")
            elif self._run_active:
                _safe_caput("State", "RUNNING")
            if self._finish_epoch > 0:
                _safe_caput("FinishEpoch", self._finish_epoch)
            _safe_caput("LastUpdateEpoch", now)

    def __call__(self, name, doc):
        now = time.time()

        if name == "start":
            self._descriptor_stream = {}
            self._run_uid = str(doc.get("uid", ""))
            self._run_active = True
            self._run_paused = False
            self._run_suspended = False
            self._suspend_since_epoch = 0.0
            self._finish_epoch_before_suspend = 0.0
            plan_name = str(doc.get("plan_name", ""))
            est_total_s = float(doc.get("estimated_total_time_s", 0.0) or 0.0)
            est_total_units = int(doc.get("estimated_total_units", 0) or 0)
            finish_epoch = now + est_total_s if est_total_s > 0 else 0.0

            _safe_caput("State", "RUNNING")
            _safe_caput("Suspended", 0)
            _safe_caput("SuspendSinceEpoch", 0.0)
            _safe_caput("SuspendReason", "")
            _safe_caput("RunUID", self._run_uid)
            _safe_caput("PlanName", plan_name)
            _safe_caput("DoneUnits", 0)
            _safe_caput("TotalUnits", est_total_units)
            _safe_caput("StartEpoch", now)
            self._set_finish_epoch(finish_epoch)
            _safe_caput("LastUpdateEpoch", now)

        elif name == "descriptor":
            self._descriptor_stream[doc.get("uid", "")] = doc.get("name", "")

        elif name == "event":
            stream_name = self._descriptor_stream.get(doc.get("descriptor", ""), "")
            if stream_name == "progress":
                data = doc.get("data", {})
                if "done_units" in data:
                    _safe_caput("DoneUnits", int(data["done_units"]))
                if "total_units" in data:
                    _safe_caput("TotalUnits", int(data["total_units"]))
                if "finish_epoch" in data:
                    self._set_finish_epoch(float(data["finish_epoch"]))
                _safe_caput("LastUpdateEpoch", now)
                return

            if stream_name == "interruptions":
                data = doc.get("data", {})
                tokens = " ".join(str(v) for v in data.values() if v is not None).lower()
                resume_markers = (" resume", "resum", "released", "release")
                suspend_markers = ("suspend", "tripp", "suspender")
                if any(marker in tokens for marker in resume_markers):
                    self._set_suspended(False, now=now)
                elif any(marker in tokens for marker in suspend_markers):
                    self._set_suspended(True, now=now, reason=tokens)
                return

        elif name == "stop":
            self._run_active = False
            self._run_paused = False
            self._set_suspended(False, now=now)
            exit_status = str(doc.get("exit_status", "")).lower()
            if exit_status == "success":
                state = "DONE"
            elif exit_status == "abort":
                state = "ABORTED"
            else:
                state = "FAILED"

            _safe_caput("State", state)
            self._set_finish_epoch(now)
            _safe_caput("LastUpdateEpoch", now)


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
try:
    RE.record_interruptions = True
except Exception:
    pass


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
            if state_lower in ("pausing", "paused", "suspending", "suspended"):
                _run_status_publisher._run_paused = True
                _safe_caput("State", "SUSPENDED" if _run_status_publisher._run_suspended else "PAUSED")
                _safe_caput("LastUpdateEpoch", time.time())
            elif state_lower in ("running", "executing"):
                _run_status_publisher._run_paused = False
                _safe_caput("State", "SUSPENDED" if _run_status_publisher._run_suspended else "RUNNING")
                _safe_caput("LastUpdateEpoch", time.time())
            elif state_lower == "idle" and _run_status_publisher._run_paused:
                # RE may transiently report idle while paused at a checkpoint.
                _safe_caput("State", "SUSPENDED" if _run_status_publisher._run_suspended else "PAUSED")
                _safe_caput("LastUpdateEpoch", time.time())

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
