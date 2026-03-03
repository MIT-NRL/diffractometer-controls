import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path


_FILE_DIR_CACHE = {
    "ts": 0.0,
    "directories": [],
    "roots": [],
}

_FILE_DIR_STREAM_STATE = {
    "started": False,
    "thread": None,
}

_FILE_DIR_MONITOR = {
    "last_non_stream_print_ts": 0.0,
    "stream_skip_busy_ts": 0.0,
    "non_stream_busy_ts": 0.0,
    "slow_scan_ts": 0.0,
}


def _resolve_imaging_roots():
    """Resolve worker-side imaging roots from detector templates and env vars."""
    now = datetime.now()
    g = globals()
    roots = []

    # Prefer roots derived from configured detector TIFF write templates.
    for det_name in ("cam1", "cam_zwo", "cam_qhy"):
        det = g.get(det_name, None)
        if det is None:
            continue
        try:
            tiff = getattr(det, "tiff1", None)
            template = getattr(tiff, "write_path_template", "")
            if template:
                roots.append(Path(now.strftime(str(template))).expanduser())
        except Exception:
            continue

    # Optional explicit roots from environment (platform path separator).
    env_roots = str(os.environ.get("MITR_IMAGING_DATA_ROOTS", "")).strip()
    if env_roots:
        for part in env_roots.split(os.pathsep):
            part = str(part).strip()
            if not part:
                continue
            try:
                roots.append(Path(now.strftime(part)).expanduser())
            except Exception:
                continue

    # Optional single root.
    env_root = str(os.environ.get("MITR_IMAGING_DATA_ROOT", "")).strip()
    if env_root:
        try:
            roots.append(Path(now.strftime(env_root)).expanduser())
        except Exception:
            pass

    # Beamline default path.
    roots.append(Path(now.strftime("/home/mitr_4dh4/Data/%Y")).expanduser())

    # Safe fallback default for current year under the worker user's home.
    roots.append(Path(now.strftime("~/Data/%Y")).expanduser())

    unique = []
    seen = set()
    for root in roots:
        s = str(root)
        if s in seen:
            continue
        seen.add(s)
        unique.append(root)
    return unique


def _rate_limited_print(key: str, message: str, min_interval_s: float = 2.0):
    now = float(time.time())
    try:
        last = float(_FILE_DIR_MONITOR.get(key, 0.0))
    except Exception:
        last = 0.0
    if (now - last) < float(max(0.1, min_interval_s)):
        return
    _FILE_DIR_MONITOR[key] = now
    print(message)


def _get_re_state():
    re_obj = globals().get("RE", None)
    if re_obj is None:
        return ""
    try:
        return str(getattr(re_obj, "state", "")).strip().lower()
    except Exception:
        return ""


def _re_busy_for_file_scan(state: str) -> bool:
    s = str(state or "").strip().lower()
    # Treat any non-idle active states as potentially performance-sensitive.
    busy_states = {
        "running",
        "paused",
        "pausing",
        "suspending",
        "suspended",
        "executing",
    }
    return s in busy_states


def list_imaging_file_dirs(
    max_items: int = 256,
    max_depth: int = 3,
    include_hidden: bool = False,
    force_refresh: bool = False,
):
    """Return candidate folder names for the `file_dir` plan parameter.

    This function is designed for Queue Server `function_execute` calls from
    remote GUIs and intentionally does lightweight local filesystem reads only.
    """
    now_mono = time.monotonic()
    try:
        max_items = int(max_items)
    except Exception:
        max_items = 256
    max_items = max(1, max_items)
    try:
        max_depth = int(max_depth)
    except Exception:
        max_depth = 3
    # Keep directory discovery bounded and lightweight.
    max_depth = max(1, min(max_depth, 3))

    # Monitor unexpected calls outside the dedicated stream publisher thread.
    # This is useful to detect accidental RE-manager function_execute traffic.
    cur_thread_name = str(threading.current_thread().name)
    re_state = _get_re_state()
    re_busy = _re_busy_for_file_scan(re_state)
    if cur_thread_name != "file-dir-choices-stream":
        _rate_limited_print(
            "last_non_stream_print_ts",
            "[file_dir_choices] non-stream invocation "
            f"thread='{cur_thread_name}' force_refresh={bool(force_refresh)} "
            f"max_depth={int(max_depth)} max_items={int(max_items)} re_state='{re_state or 'unknown'}'",
            min_interval_s=2.0,
        )
        if re_busy:
            _rate_limited_print(
                "non_stream_busy_ts",
                "[file_dir_choices] WARNING: non-stream invocation while RE is active "
                f"(state='{re_state}') thread='{cur_thread_name}'",
                min_interval_s=2.0,
            )

    # By default, avoid expensive forced rescans from the stream thread while
    # RE is actively executing a plan. Publish last known cache instead.
    skip_stream_scan_when_re_busy = str(
        os.environ.get("MITR_FILE_DIR_SKIP_STREAM_SCAN_WHEN_RE_BUSY", "1")
    ).strip().lower() not in ("0", "false", "off", "no")
    if (
        skip_stream_scan_when_re_busy
        and re_busy
        and (cur_thread_name == "file-dir-choices-stream")
        and bool(force_refresh)
    ):
        _rate_limited_print(
            "stream_skip_busy_ts",
            "[file_dir_choices] stream scan skipped while RE is active "
            f"(state='{re_state}'); publishing cached directories only",
            min_interval_s=2.0,
        )
        return {
            "ok": True,
            "directories": list(_FILE_DIR_CACHE["directories"]),
            "roots": list(_FILE_DIR_CACHE["roots"]),
            "cached": True,
            "skipped_for_active_run": True,
            "re_state": re_state,
        }

    if (not force_refresh) and (now_mono - float(_FILE_DIR_CACHE["ts"]) < 2.0):
        return {
            "ok": True,
            "directories": list(_FILE_DIR_CACHE["directories"]),
            "roots": list(_FILE_DIR_CACHE["roots"]),
            "cached": True,
        }

    scan_t0 = time.monotonic()
    roots = _resolve_imaging_roots()
    names = set()
    resolved_roots = []
    for root in roots:
        try:
            if not root.exists() or (not root.is_dir()):
                continue
        except Exception:
            continue

        resolved_roots.append(str(root))
        stack = [(Path(root), 0, "")]
        while stack:
            cur_path, depth, rel_prefix = stack.pop()
            if depth >= max_depth:
                continue
            try:
                with os.scandir(cur_path) as entries:
                    children = []
                    for entry in entries:
                        try:
                            if not entry.is_dir(follow_symlinks=False):
                                continue
                            name = str(entry.name).strip()
                            if not name:
                                continue
                            if (not include_hidden) and name.startswith("."):
                                continue
                            rel_path = f"{rel_prefix}/{name}" if rel_prefix else name
                            names.add(rel_path)
                            children.append((Path(cur_path) / name, depth + 1, rel_path))
                        except Exception:
                            continue
            except Exception:
                continue

            # Deterministic traversal order keeps dropdown ordering stable.
            for child in sorted(children, key=lambda x: x[2], reverse=True):
                stack.append(child)

    directories = sorted(names, key=str.lower)[:max_items]
    _FILE_DIR_CACHE["ts"] = now_mono
    _FILE_DIR_CACHE["directories"] = list(directories)
    _FILE_DIR_CACHE["roots"] = list(resolved_roots)

    scan_dt = max(0.0, float(time.monotonic() - scan_t0))
    try:
        warn_scan_s = float(os.environ.get("MITR_FILE_DIR_WARN_SCAN_S", "0.25"))
    except Exception:
        warn_scan_s = 0.25
    if scan_dt >= max(0.01, warn_scan_s):
        _rate_limited_print(
            "slow_scan_ts",
            "[file_dir_choices] scan duration warning "
            f"dt={scan_dt:.3f}s thread='{cur_thread_name}' "
            f"dirs={len(directories)} roots={len(resolved_roots)} re_state='{re_state or 'unknown'}'",
            min_interval_s=2.0,
        )

    return {
        "ok": True,
        "directories": directories,
        "roots": resolved_roots,
        "cached": False,
    }


def _start_file_dir_choices_stream():
    """Publish file-dir choices on a dedicated ZMQ PUB stream.

    This decouples frequent directory updates from RE Manager RPC traffic.
    """
    if _FILE_DIR_STREAM_STATE["started"]:
        return

    enabled = str(os.environ.get("MITR_FILE_DIR_STREAM_ENABLE", "1")).strip().lower()
    if enabled in ("0", "false", "off", "no"):
        return

    try:
        import zmq
    except Exception:
        return

    addr = str(os.environ.get("MITR_FILE_DIR_STREAM_ADDR", "tcp://*:5569")).strip()
    if not addr:
        addr = "tcp://*:5569"
    topic = str(os.environ.get("MITR_FILE_DIR_STREAM_TOPIC", "file_dir_choices")).strip()
    if not topic:
        topic = "file_dir_choices"

    try:
        interval_s = float(os.environ.get("MITR_FILE_DIR_STREAM_INTERVAL_S", "2.0"))
    except Exception:
        interval_s = 2.0
    interval_s = max(0.5, interval_s)

    try:
        max_items = int(os.environ.get("MITR_FILE_DIR_STREAM_MAX_ITEMS", "512"))
    except Exception:
        max_items = 512
    max_items = max(1, max_items)

    try:
        max_depth = int(os.environ.get("MITR_FILE_DIR_STREAM_MAX_DEPTH", "1"))
    except Exception:
        max_depth = 1
    max_depth = max(1, min(max_depth, 3))

    topic_b = topic.encode("utf-8", errors="ignore")

    def _pub_loop():
        ctx = None
        sock = None
        try:
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 10)
            sock.bind(addr)
            print(
                "[file_dir_choices] stream publisher started "
                f"addr='{addr}' topic='{topic}' interval_s={interval_s:.3f} "
                f"max_depth={max_depth} max_items={max_items}"
            )

            while True:
                payload = list_imaging_file_dirs(
                    max_items=max_items,
                    max_depth=max_depth,
                    include_hidden=False,
                    force_refresh=True,
                )
                payload["published_ts"] = float(time.time())
                try:
                    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    sock.send_multipart([topic_b, data], flags=zmq.NOBLOCK)
                except Exception:
                    pass
                time.sleep(interval_s)
        except Exception as ex:
            print(f"[file_dir_choices] stream publisher stopped: {ex}")
            return
        finally:
            try:
                if sock is not None:
                    sock.close(0)
            except Exception:
                pass

    th = threading.Thread(target=_pub_loop, name="file-dir-choices-stream", daemon=True)
    _FILE_DIR_STREAM_STATE["thread"] = th
    _FILE_DIR_STREAM_STATE["started"] = True
    th.start()


_start_file_dir_choices_stream()
