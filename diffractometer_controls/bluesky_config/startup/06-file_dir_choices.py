import json
import os
import threading
import time
import atexit
from datetime import datetime
from pathlib import Path


_FILE_DIR_CACHE = {
    "ts": 0.0,
    "directories": [],
    "roots": [],
}
_FILE_DIR_CACHE_LOCK = threading.Lock()

_FILE_DIR_STREAM_STATE = {
    "started": False,
    "thread": None,
    "snapshot_thread": None,
    "stop": threading.Event(),
    "observer": None,
}

_FILE_DIR_MONITOR = {
    "last_non_stream_print_ts": 0.0,
    "stream_skip_busy_ts": 0.0,
    "stream_skip_busy_active": False,
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
    roots.append(Path(now.strftime("/home/mitr_4dh4/Data/Imaging/%Y")).expanduser())

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


def _get_file_dir_cache_payload(*, cached=True, **extra):
    with _FILE_DIR_CACHE_LOCK:
        payload = {
            "ok": True,
            "directories": list(_FILE_DIR_CACHE["directories"]),
            "roots": list(_FILE_DIR_CACHE["roots"]),
            "cached": bool(cached),
        }
    if extra:
        payload.update(extra)
    return payload


def _update_file_dir_cache(*, ts, directories, roots):
    with _FILE_DIR_CACHE_LOCK:
        _FILE_DIR_CACHE["ts"] = float(ts)
        _FILE_DIR_CACHE["directories"] = list(directories)
        _FILE_DIR_CACHE["roots"] = list(roots)


def _file_dir_cache_is_empty():
    with _FILE_DIR_CACHE_LOCK:
        return not bool(_FILE_DIR_CACHE["directories"])


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
    if cur_thread_name not in ("file-dir-choices-stream", "file-dir-choices-snapshot"):
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
        os.environ.get("MITR_FILE_DIR_SKIP_STREAM_SCAN_WHEN_RE_BUSY", "0")
    ).strip().lower() not in ("0", "false", "off", "no")
    blocked_for_active_run = (
        skip_stream_scan_when_re_busy
        and re_busy
        and (cur_thread_name == "file-dir-choices-stream")
        and bool(force_refresh)
    )
    was_blocked = bool(_FILE_DIR_MONITOR.get("stream_skip_busy_active", False))
    if blocked_for_active_run:
        if not was_blocked:
            _FILE_DIR_MONITOR["stream_skip_busy_active"] = True
            print(
                "[file_dir_choices] stream scan skipped while RE is active "
                f"(state='{re_state}'); publishing cached directories only"
            )
        return _get_file_dir_cache_payload(
            cached=True,
            skipped_for_active_run=True,
            re_state=re_state,
        )
    if was_blocked:
        _FILE_DIR_MONITOR["stream_skip_busy_active"] = False

    with _FILE_DIR_CACHE_LOCK:
        cache_age_s = now_mono - float(_FILE_DIR_CACHE["ts"])
    if (not force_refresh) and (cache_age_s < 2.0):
        return _get_file_dir_cache_payload(cached=True)

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
    _update_file_dir_cache(ts=now_mono, directories=directories, roots=resolved_roots)

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

    Supports event-driven publish with watchdog and polling fallback.
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
    snapshot_addr = str(os.environ.get("MITR_FILE_DIR_SNAPSHOT_ADDR", "tcp://*:5570")).strip()
    if not snapshot_addr:
        snapshot_addr = "tcp://*:5570"
    topic = str(os.environ.get("MITR_FILE_DIR_STREAM_TOPIC", "file_dir_choices")).strip()
    if not topic:
        topic = "file_dir_choices"

    stream_mode = str(os.environ.get("MITR_FILE_DIR_STREAM_MODE", "watchdog")).strip().lower()
    if stream_mode in ("watch", "watchdog", "event", "events"):
        stream_mode = "watchdog"
    elif stream_mode in ("poll", "polling"):
        stream_mode = "poll"
    else:
        stream_mode = "watchdog"

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
        max_depth = int(os.environ.get("MITR_FILE_DIR_STREAM_MAX_DEPTH", "3"))
    except Exception:
        max_depth = 3
    max_depth = max(1, min(max_depth, 3))

    try:
        debounce_s = float(os.environ.get("MITR_FILE_DIR_STREAM_EVENT_DEBOUNCE_S", "0.25"))
    except Exception:
        debounce_s = 0.25
    debounce_s = max(0.05, debounce_s)

    try:
        heartbeat_s = float(os.environ.get("MITR_FILE_DIR_STREAM_HEARTBEAT_S", "60.0"))
    except Exception:
        heartbeat_s = 60.0
    heartbeat_s = max(5.0, heartbeat_s)

    try:
        full_rescan_s = float(os.environ.get("MITR_FILE_DIR_STREAM_FULL_RESCAN_S", "600.0"))
    except Exception:
        full_rescan_s = 600.0
    full_rescan_s = max(30.0, full_rescan_s)
    debug_watchdog = str(os.environ.get("MITR_FILE_DIR_WATCHDOG_DEBUG", "0")).strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )

    stop_event = _FILE_DIR_STREAM_STATE["stop"]
    stop_event.clear()

    topic_b = topic.encode("utf-8", errors="ignore")
    def _snapshot_loop():
        ctx = None
        sock = None
        try:
            if _file_dir_cache_is_empty():
                list_imaging_file_dirs(
                    max_items=max_items,
                    max_depth=max_depth,
                    include_hidden=False,
                    force_refresh=True,
                )
            ctx = zmq.Context()
            sock = ctx.socket(zmq.REP)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(snapshot_addr)
            print(f"[file_dir_choices] snapshot RPC started addr='{snapshot_addr}'")
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not stop_event.is_set():
                try:
                    events = dict(poller.poll(100))
                except Exception:
                    continue
                if sock not in events:
                    continue
                try:
                    raw = sock.recv(flags=zmq.NOBLOCK)
                except Exception:
                    continue
                try:
                    request = json.loads(raw.decode("utf-8", errors="ignore"))
                except Exception:
                    request = {}
                op = str((request or {}).get("op", "")).strip().lower()
                if op not in ("get_file_dir_cache", "snapshot", "get"):
                    payload = {
                        "ok": False,
                        "error": f"Unsupported operation: {op or 'unknown'}",
                        "directories": [],
                        "roots": [],
                        "cached": True,
                    }
                else:
                    payload = _get_file_dir_cache_payload(cached=True)
                payload["published_ts"] = float(time.time())
                try:
                    sock.send_json(payload, flags=zmq.NOBLOCK)
                except Exception:
                    continue
        except Exception as ex:
            print(
                "[file_dir_choices] snapshot RPC stopped: "
                f"addr='{snapshot_addr}' error={ex}"
            )
        finally:
            try:
                if sock is not None:
                    sock.close(0)
            except Exception:
                pass
            try:
                if ctx is not None:
                    ctx.term()
            except Exception:
                pass
            _FILE_DIR_STREAM_STATE["snapshot_thread"] = None

    def _pub_loop():
        ctx = None
        sock = None
        observer = None
        initial_payload = _get_file_dir_cache_payload(cached=True)
        try:
            # Dedicated context for this publisher thread avoids global-context
            # teardown races on interpreter shutdown/restart.
            ctx = zmq.Context()
            sock = ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 10)
            sock.bind(addr)
            print(
                "[file_dir_choices] stream publisher started "
                f"addr='{addr}' topic='{topic}' mode='{stream_mode}' "
                f"interval_s={interval_s:.3f} max_depth={max_depth} max_items={max_items}"
            )

            last_dirs_signature = None
            last_publish_mono = 0.0

            def _watchdog_log(msg):
                if debug_watchdog:
                    print(f"[file_dir_choices][watchdog] {msg}")

            def _scan_payload():
                return list_imaging_file_dirs(
                    max_items=max_items,
                    max_depth=max_depth,
                    include_hidden=False,
                    force_refresh=True,
                )

            def _publish_payload(payload, *, force=False):
                nonlocal last_dirs_signature
                nonlocal last_publish_mono
                if not isinstance(payload, dict):
                    return False
                dirs = payload.get("directories", [])
                try:
                    dirs_sig = tuple(str(x) for x in list(dirs))
                except Exception:
                    dirs_sig = tuple()
                if (not force) and (dirs_sig == last_dirs_signature):
                    return False
                payload = dict(payload)
                payload["published_ts"] = float(time.time())
                try:
                    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    sock.send_multipart([topic_b, data], flags=zmq.NOBLOCK)
                    last_dirs_signature = dirs_sig
                    last_publish_mono = time.monotonic()
                    _watchdog_log(
                        "published update "
                        f"dirs={len(dirs_sig)} force={bool(force)} cached={bool(payload.get('cached', False))}"
                    )
                    return True
                except Exception:
                    return False

            def _publish_cached_heartbeat():
                payload = _get_file_dir_cache_payload(cached=True, heartbeat=True)
                _publish_payload(payload, force=True)

            if stream_mode == "watchdog":
                try:
                    from watchdog.events import FileSystemEventHandler
                    from watchdog.observers import Observer
                except Exception as ex:
                    print(
                        "[file_dir_choices] watchdog mode unavailable; "
                        f"falling back to polling mode ({ex})"
                    )
                    mode_watchdog = False
                else:
                    mode_watchdog = True

                if mode_watchdog:
                    watch_state = {"dirty": True, "deadline": 0.0, "events": []}
                    watched_paths = set()
                    state_lock = threading.Lock()
                    cache_dirs = set()
                    cache_roots = []
                    configured_roots = []

                    class _DirEventHandler(FileSystemEventHandler):
                        def on_any_event(self, event):
                            try:
                                is_dir = bool(getattr(event, "is_directory", False))
                            except Exception:
                                is_dir = False
                            if not is_dir:
                                return
                            evt = str(getattr(event, "event_type", "")).strip().lower()
                            if evt not in ("created", "moved", "deleted"):
                                return
                            src = str(getattr(event, "src_path", "") or "").strip()
                            dest = str(getattr(event, "dest_path", "") or "").strip()
                            if dest:
                                _watchdog_log(f"event={evt} src='{src}' dest='{dest}'")
                            else:
                                _watchdog_log(f"event={evt} path='{src}'")
                            with state_lock:
                                watch_state["events"].append((evt, src, dest))
                                watch_state["dirty"] = True
                                watch_state["deadline"] = time.monotonic() + debounce_s

                    def _refresh_configured_roots():
                        nonlocal configured_roots
                        roots = []
                        for root in _resolve_imaging_roots():
                            try:
                                roots.append(str(Path(root).expanduser()))
                            except Exception:
                                continue
                        configured_roots = roots

                    def _schedule_watch_paths(obs):
                        roots = _resolve_imaging_roots()
                        for root in roots:
                            try:
                                root = Path(root).expanduser()
                            except Exception:
                                continue
                            if root.exists() and root.is_dir():
                                key = (str(root), True)
                                if key not in watched_paths:
                                    obs.schedule(handler, str(root), recursive=True)
                                    watched_paths.add(key)
                                    _watchdog_log(f"watch root='{root}' recursive=True")
                            else:
                                parent = root.parent
                                if parent.exists() and parent.is_dir():
                                    key = (str(parent), False)
                                    if key not in watched_paths:
                                        obs.schedule(handler, str(parent), recursive=False)
                                        watched_paths.add(key)
                                        _watchdog_log(f"watch parent='{parent}' recursive=False")

                    def _rel_from_path(path):
                        p = str(path or "").strip()
                        if not p:
                            return None
                        p = str(Path(p))
                        for root in configured_roots:
                            root_s = str(root).rstrip("/")
                            if p == root_s:
                                return ""
                            pref = root_s + "/"
                            if p.startswith(pref):
                                rel = p[len(pref) :].strip("/")
                                if not rel:
                                    return ""
                                parts = [x for x in rel.split("/") if x]
                                if len(parts) > int(max_depth):
                                    return None
                                if any(str(part).startswith(".") for part in parts):
                                    return None
                                return "/".join(parts)
                        return None

                    def _add_rel(rel):
                        if not rel:
                            return False
                        if rel in cache_dirs:
                            return False
                        cache_dirs.add(rel)
                        return True

                    def _remove_rel_prefix(rel):
                        if not rel:
                            return False
                        changed = False
                        if rel in cache_dirs:
                            cache_dirs.discard(rel)
                            changed = True
                        pref = rel + "/"
                        to_drop = [x for x in cache_dirs if x.startswith(pref)]
                        if to_drop:
                            changed = True
                            for x in to_drop:
                                cache_dirs.discard(x)
                        return changed

                    def _add_tree_from_abs(path):
                        rel = _rel_from_path(path)
                        if rel is None:
                            return False
                        changed = False
                        if rel:
                            changed = _add_rel(rel) or changed
                        # If path does not exist (race/delete), we can still keep the leaf rel.
                        p = Path(path)
                        if (not p.exists()) or (not p.is_dir()):
                            return changed
                        base_depth = 0 if not rel else len(rel.split("/"))
                        if base_depth >= int(max_depth):
                            return changed
                        stack = [(p, base_depth, rel)]
                        while stack:
                            cur_path, depth, rel_prefix = stack.pop()
                            if depth >= int(max_depth):
                                continue
                            try:
                                with os.scandir(cur_path) as entries:
                                    children = []
                                    for entry in entries:
                                        try:
                                            if not entry.is_dir(follow_symlinks=False):
                                                continue
                                            name = str(entry.name).strip()
                                            if (not name) or name.startswith("."):
                                                continue
                                            child_rel = f"{rel_prefix}/{name}" if rel_prefix else name
                                            if _add_rel(child_rel):
                                                changed = True
                                            children.append((Path(entry.path), depth + 1, child_rel))
                                        except Exception:
                                            continue
                            except Exception:
                                continue
                            for child in sorted(children, key=lambda x: x[2], reverse=True):
                                stack.append(child)
                        return changed

                    def _cache_payload_from_set():
                        dirs = sorted(cache_dirs, key=str.lower)[: int(max_items)]
                        roots = list(cache_roots)
                        _update_file_dir_cache(ts=time.monotonic(), directories=dirs, roots=roots)
                        return {
                            "ok": True,
                            "directories": dirs,
                            "roots": roots,
                            "cached": True,
                            "incremental": True,
                        }

                    def _apply_event(evt, src, dest):
                        changed = False
                        need_full_rescan = False
                        if evt == "created":
                            rel = _rel_from_path(src)
                            if rel == "":
                                need_full_rescan = True
                            elif rel is not None:
                                changed = _add_tree_from_abs(src) or changed
                        elif evt == "deleted":
                            rel = _rel_from_path(src)
                            if rel == "":
                                need_full_rescan = True
                            elif rel is not None:
                                changed = _remove_rel_prefix(rel) or changed
                        elif evt == "moved":
                            rel_src = _rel_from_path(src)
                            rel_dst = _rel_from_path(dest)
                            if (rel_src == "") or (rel_dst == ""):
                                need_full_rescan = True
                            if rel_src not in (None, ""):
                                changed = _remove_rel_prefix(rel_src) or changed
                            if rel_dst not in (None, ""):
                                changed = _add_tree_from_abs(dest) or changed
                        return changed, need_full_rescan

                    handler = _DirEventHandler()
                    observer = Observer()
                    _refresh_configured_roots()
                    _schedule_watch_paths(observer)
                    _FILE_DIR_STREAM_STATE["observer"] = observer
                    observer.start()
                    # Publish initial snapshot immediately.
                    initial_payload = _scan_payload()
                    cache_dirs = set(str(x) for x in list(initial_payload.get("directories", [])))
                    cache_roots = list(initial_payload.get("roots", []))
                    _publish_payload(initial_payload, force=True)

                    next_full_rescan = time.monotonic() + full_rescan_s
                    next_heartbeat = time.monotonic() + heartbeat_s
                    while not stop_event.is_set():
                        now = time.monotonic()
                        do_apply = False
                        pending_events = []
                        with state_lock:
                            if watch_state["dirty"] and (now >= watch_state["deadline"]):
                                do_apply = True
                                watch_state["dirty"] = False
                                pending_events = list(watch_state["events"])
                                watch_state["events"] = []
                        if do_apply:
                            _refresh_configured_roots()
                            _schedule_watch_paths(observer)
                            changed = False
                            need_full_rescan = False
                            for evt, src, dest in pending_events:
                                ch, full = _apply_event(evt, src, dest)
                                changed = changed or ch
                                need_full_rescan = need_full_rescan or full
                            if need_full_rescan:
                                full_payload = _scan_payload()
                                cache_dirs = set(str(x) for x in list(full_payload.get("directories", [])))
                                cache_roots = list(full_payload.get("roots", []))
                                _publish_payload(full_payload, force=False)
                                _watchdog_log("event batch triggered full rescan")
                            elif changed:
                                if not _publish_payload(_cache_payload_from_set(), force=False):
                                    _watchdog_log("incremental batch applied; published list unchanged")
                            else:
                                _watchdog_log("incremental batch applied; no cache changes")
                            next_heartbeat = time.monotonic() + heartbeat_s
                        if now >= next_full_rescan:
                            _refresh_configured_roots()
                            _schedule_watch_paths(observer)
                            full_payload = _scan_payload()
                            cache_dirs = set(str(x) for x in list(full_payload.get("directories", [])))
                            cache_roots = list(full_payload.get("roots", []))
                            _publish_payload(full_payload, force=False)
                            next_full_rescan = now + full_rescan_s
                        if now >= next_heartbeat:
                            _publish_payload(_cache_payload_from_set(), force=True)
                            next_heartbeat = now + heartbeat_s
                        stop_event.wait(0.1)
                else:
                    initial_payload = _scan_payload()
                    _publish_payload(initial_payload, force=True)
                    while not stop_event.is_set():
                        _publish_payload(_scan_payload(), force=True)
                        stop_event.wait(interval_s)
            else:
                initial_payload = _scan_payload()
                _publish_payload(initial_payload, force=True)
                while not stop_event.is_set():
                    _publish_payload(_scan_payload(), force=True)
                    stop_event.wait(interval_s)
        except Exception as ex:
            print(f"[file_dir_choices] stream publisher stopped: {ex}")
            return
        finally:
            try:
                obs = _FILE_DIR_STREAM_STATE.get("observer", None)
                if obs is not None:
                    obs.stop()
                    obs.join(timeout=1.0)
            except Exception:
                pass
            _FILE_DIR_STREAM_STATE["observer"] = None
            try:
                if sock is not None:
                    sock.close(0)
            except Exception:
                pass
            try:
                if ctx is not None:
                    ctx.term()
            except Exception:
                pass
            _FILE_DIR_STREAM_STATE["thread"] = None
            _FILE_DIR_STREAM_STATE["started"] = False

    snapshot_th = threading.Thread(
        target=_snapshot_loop, name="file-dir-choices-snapshot", daemon=True
    )
    _FILE_DIR_STREAM_STATE["snapshot_thread"] = snapshot_th
    snapshot_th.start()

    th = threading.Thread(target=_pub_loop, name="file-dir-choices-stream", daemon=True)
    _FILE_DIR_STREAM_STATE["thread"] = th
    _FILE_DIR_STREAM_STATE["started"] = True
    th.start()


def _stop_file_dir_choices_stream():
    th = _FILE_DIR_STREAM_STATE.get("thread", None)
    snapshot_th = _FILE_DIR_STREAM_STATE.get("snapshot_thread", None)
    try:
        _FILE_DIR_STREAM_STATE["stop"].set()
    except Exception:
        pass
    try:
        obs = _FILE_DIR_STREAM_STATE.get("observer", None)
        if obs is not None:
            obs.stop()
            obs.join(timeout=1.0)
    except Exception:
        pass
    _FILE_DIR_STREAM_STATE["observer"] = None
    try:
        if snapshot_th is not None and snapshot_th.is_alive() and (threading.current_thread() is not snapshot_th):
            snapshot_th.join(timeout=1.0)
    except Exception:
        pass
    _FILE_DIR_STREAM_STATE["snapshot_thread"] = None
    if th is None:
        _FILE_DIR_STREAM_STATE["started"] = False
        return
    try:
        if th.is_alive() and (threading.current_thread() is not th):
            th.join(timeout=1.0)
    except Exception:
        pass
    _FILE_DIR_STREAM_STATE["thread"] = None
    _FILE_DIR_STREAM_STATE["started"] = False


atexit.register(_stop_file_dir_choices_stream)
_start_file_dir_choices_stream()
