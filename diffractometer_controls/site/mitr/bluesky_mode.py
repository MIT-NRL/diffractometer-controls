import json
from pathlib import Path

try:
    from epics import caget, caput
except Exception:
    caget = None
    caput = None


BLUESKY_MODE_PRODUCTION = "production"
BLUESKY_MODE_TEST = "test"
DEFAULT_BLUESKY_MODE = BLUESKY_MODE_PRODUCTION
BLUESKY_MODE_STATE_PATH = Path("~/.mitr_bluesky_mode.json").expanduser()
BLUESKY_MODE_PV = "4dh4:Bluesky:Mode"
BLUESKY_ACTIVE_MODE_PV = "4dh4:Bluesky:ActiveMode"


def normalize_bluesky_mode(mode):
    mode_normalized = str(mode or "").strip().lower()
    if mode_normalized == BLUESKY_MODE_TEST:
        return BLUESKY_MODE_TEST
    return BLUESKY_MODE_PRODUCTION


def load_bluesky_mode_state():
    state = {"mode": DEFAULT_BLUESKY_MODE}
    try:
        data = json.loads(BLUESKY_MODE_STATE_PATH.read_text())
    except FileNotFoundError:
        return state
    except Exception:
        return state
    if isinstance(data, dict):
        state["mode"] = normalize_bluesky_mode(data.get("mode"))
    return state


def _read_bluesky_mode_pv():
    if caget is None:
        return None
    try:
        value = caget(BLUESKY_MODE_PV, timeout=1.0, connection_timeout=1.0)
    except Exception:
        return None
    if value is None:
        return None
    try:
        value_int = int(value)
    except Exception:
        return normalize_bluesky_mode(value)
    return BLUESKY_MODE_TEST if value_int == 1 else BLUESKY_MODE_PRODUCTION


def _write_bluesky_mode_pv(mode):
    if caput is None:
        return False
    mode_normalized = normalize_bluesky_mode(mode)
    value = 1 if mode_normalized == BLUESKY_MODE_TEST else 0
    try:
        result = caput(BLUESKY_MODE_PV, value, wait=True, timeout=1.5)
    except Exception:
        return False
    return bool(result)


def _read_bluesky_active_mode_pv():
    if caget is None:
        return None
    try:
        value = caget(BLUESKY_ACTIVE_MODE_PV, timeout=1.0, connection_timeout=1.0)
    except Exception:
        return None
    if value is None:
        return None
    try:
        value_int = int(value)
    except Exception:
        return normalize_bluesky_mode(value)
    return BLUESKY_MODE_TEST if value_int == 1 else BLUESKY_MODE_PRODUCTION


def publish_bluesky_active_mode(mode):
    if caput is None:
        return False
    mode_normalized = normalize_bluesky_mode(mode)
    value = 1 if mode_normalized == BLUESKY_MODE_TEST else 0
    try:
        result = caput(BLUESKY_ACTIVE_MODE_PV, value, wait=True, timeout=1.5)
    except Exception:
        return False
    return bool(result)


def save_bluesky_mode_state(mode):
    state = {"mode": normalize_bluesky_mode(mode)}
    BLUESKY_MODE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLUESKY_MODE_STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    _write_bluesky_mode_pv(state["mode"])
    return state


def get_bluesky_mode():
    mode_from_pv = _read_bluesky_mode_pv()
    if mode_from_pv is not None:
        return mode_from_pv
    return load_bluesky_mode_state()["mode"]


def get_bluesky_active_mode():
    mode_from_pv = _read_bluesky_active_mode_pv()
    if mode_from_pv is not None:
        return mode_from_pv
    return get_bluesky_mode()


def get_bluesky_history_path(mode=None):
    mode_normalized = normalize_bluesky_mode(mode if mode is not None else get_bluesky_mode())
    if mode_normalized == BLUESKY_MODE_TEST:
        return Path("~/.bluesky_history_test").expanduser()
    return Path("~/.bluesky_history").expanduser()


def get_bluesky_mode_display(mode=None):
    mode_normalized = normalize_bluesky_mode(mode if mode is not None else get_bluesky_mode())
    if mode_normalized == BLUESKY_MODE_TEST:
        return "TEST"
    return "PRODUCTION"
