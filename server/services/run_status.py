"""Run status publisher definitions; no subscriptions or EPICS writes at import time."""
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
        self._run_uid = ""
        self._run_active = False
        self._run_paused = False
        self._pause_since_epoch = 0.0
        self._run_suspended = False
        self._suspend_since_epoch = 0.0
        self._finish_epoch = 0.0
        self._finish_epoch_before_pause = 0.0

    def _set_finish_epoch(self, value):
        self._finish_epoch = float(value)
        _safe_caput("FinishEpoch", self._finish_epoch)

    def _set_paused(self, paused, now):
        paused = bool(paused)
        if paused == self._run_paused:
            return

        if paused:
            self._run_paused = True
            self._pause_since_epoch = float(now)
            self._finish_epoch_before_pause = self._finish_epoch
            # Pause is an estimate hold: finish becomes pending until resume.
            self._set_finish_epoch(0.0)
            if self._run_active:
                _safe_caput("State", "SUSPENDED" if self._run_suspended else "PAUSED")
            _safe_caput("LastUpdateEpoch", now)
        else:
            if self._pause_since_epoch > 0 and self._finish_epoch_before_pause > 0:
                self._finish_epoch = self._finish_epoch_before_pause + max(
                    0.0, float(now) - self._pause_since_epoch
                )
            self._run_paused = False
            self._pause_since_epoch = 0.0
            self._finish_epoch_before_pause = 0.0
            if self._run_active:
                _safe_caput("State", "SUSPENDED" if self._run_suspended else "RUNNING")
            if (not self._run_suspended) and self._finish_epoch > 0:
                _safe_caput("FinishEpoch", self._finish_epoch)
            _safe_caput("LastUpdateEpoch", now)

    def _set_suspended(self, suspended, now, reason=""):
        suspended = bool(suspended)
        if suspended == self._run_suspended:
            return

        if suspended:
            self._run_suspended = True
            self._suspend_since_epoch = float(now)
            _safe_caput("Suspended", 1)
            _safe_caput("SuspendSinceEpoch", self._suspend_since_epoch)
            _safe_caput("SuspendReason", str(reason)[:200])
            if self._run_active:
                _safe_caput("State", "SUSPENDED")
            _safe_caput("LastUpdateEpoch", now)
        else:
            self._run_suspended = False
            self._suspend_since_epoch = 0.0
            _safe_caput("Suspended", 0)
            _safe_caput("SuspendSinceEpoch", 0.0)
            _safe_caput("SuspendReason", "")
            if self._run_active and self._run_paused:
                _safe_caput("State", "PAUSED")
            elif self._run_active:
                _safe_caput("State", "RUNNING")
            if (not self._run_paused) and self._finish_epoch > 0:
                _safe_caput("FinishEpoch", self._finish_epoch)
            _safe_caput("LastUpdateEpoch", now)

    def publish_progress(self, *, done_units=None, total_units=None, finish_epoch=None, now=None):
        """Publish runtime plan progress directly to the run-status PVs."""
        if not self._run_active:
            return

        now = time.time() if now is None else float(now)
        if done_units is not None:
            _safe_caput("DoneUnits", int(done_units))
        if total_units is not None:
            _safe_caput("TotalUnits", int(total_units))
        if finish_epoch is not None:
            self._finish_epoch = float(finish_epoch)
            if not self._run_paused:
                _safe_caput("FinishEpoch", self._finish_epoch)
        _safe_caput("LastUpdateEpoch", now)

    def __call__(self, name, doc):
        now = time.time()

        if name == "start":
            self._run_uid = str(doc.get("uid", ""))
            self._run_active = True
            self._run_paused = False
            self._pause_since_epoch = 0.0
            self._run_suspended = False
            self._suspend_since_epoch = 0.0
            self._finish_epoch_before_pause = 0.0
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

        elif name == "stop":
            self._run_active = False
            self._run_paused = False
            self._pause_since_epoch = 0.0
            self._finish_epoch_before_pause = 0.0
            self._run_suspended = False
            self._suspend_since_epoch = 0.0
            _safe_caput("Suspended", 0)
            _safe_caput("SuspendSinceEpoch", 0.0)
            _safe_caput("SuspendReason", "")
            exit_status = str(doc.get("exit_status", "")).lower()
            run_success = 1 if exit_status == "success" else 0
            _safe_caput("LastRunSuccess", run_success)
            _safe_caput("LastRunExitStatus", exit_status or "unknown")
            _safe_caput("State", "IDLE")
            self._set_finish_epoch(now)
            _safe_caput("LastUpdateEpoch", now)
