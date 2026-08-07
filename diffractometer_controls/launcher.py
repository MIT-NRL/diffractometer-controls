import argparse
import cProfile
import inspect
import logging
import os
import platform
import pstats
import sys
import threading
import time
import faulthandler
from pathlib import Path
from qtpy import QtCore, QtGui


def _load_simple_env_file(path):
    try:
        lines = Path(path).expanduser().read_text().splitlines()
    except FileNotFoundError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _configure_qt_highdpi():
    if hasattr(QtCore.Qt, "AA_EnableHighDpiScaling"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, "AA_UseHighDpiPixmaps"):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    rounding_policy_enum = getattr(QtCore.Qt, "HighDpiScaleFactorRoundingPolicy", None)
    set_rounding_policy = getattr(QtGui.QGuiApplication, "setHighDpiScaleFactorRoundingPolicy", None)
    if rounding_policy_enum is not None and callable(set_rounding_policy):
        set_rounding_policy(rounding_policy_enum.PassThrough)


def _patch_bluesky_status_reload_shutdown(cls):
    """Make shared Queue Server polling safe to stop during navigation."""
    if getattr(cls, "_dc_status_reload_shutdown_patch_applied", False):
        return

    def _patched_reload_status(self):
        self.model.load_re_manager_status()
        remaining = max(float(getattr(self, "update_period", 0) or 0), 0.0)
        while remaining > 0:
            if getattr(self, "_deactivate_updates", False):
                break
            delay = min(0.05, remaining)
            time.sleep(delay)
            remaining -= delay

    def _patched_reload_complete(self):
        if not self._deactivate_updates:
            self._start_thread()
            return
        # The RunEngineClient model is shared by all screens. A worker from a
        # detached screen must not clear state after another screen attaches.
        detaching = bool(getattr(self, "_dc_detaching", False))
        self._dc_detaching = False
        if not detaching:
            self.model.clear_connection_status()
        self.updates_activated = False
        self._deactivate_updates = False
        self._update_widget_states()

    cls._reload_status = _patched_reload_status
    cls._reload_complete = _patched_reload_complete
    cls._dc_status_reload_shutdown_patch_applied = True


def main():
    logger = logging.getLogger("")
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)-8s] - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    handler.setLevel("INFO")
    _configure_qt_highdpi()
    os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
    _load_simple_env_file("~/.config/diffractometer-controls/control.env")
    _load_simple_env_file("~/.config/bluesky-queueserver/client-zmq.env")

    from pydm import config

    # Set EPICS as the default protocol
    config.DEFAULT_PROTOCOL = "ca"

    # Define EPICS Support dir where all display files are stored
    if platform.system() == "Windows":
        separator = ';'
    else:
        separator = ':'
    path_list = [dirs.as_posix() for dirs in [Path('./extra_ui').absolute(),Path('./extra_ui/autoconvert').absolute()]]
    EPICS_SUPPORT = Path('/home/mitr_4dh4/EPICS/synApps-6-3/support')
    DISPLAY_PATH = os.getenv("PYDM_DISPLAYS_PATH",None)
    GITHUB = Path('/home/mitr_4dh4/Documents/GitHub')
    if DISPLAY_PATH is None:
        path_list_adl = [dirs.as_posix() for dirs in EPICS_SUPPORT.glob('**/*op/adl*')] + [dirs.as_posix() for dirs in EPICS_SUPPORT.glob('**/*opi/medm*')]
        path_list_custom = [
            path.as_posix()
            for path in GITHUB.glob('**/PyDM*')
            if path.is_dir()
        ]
        logger.debug(
            "Configured %d EPICS and %d custom display search paths.",
            len(path_list_adl),
            len(path_list_custom),
        )
        if len(path_list_adl) != 0:
            path_list.extend(path_list_adl)
        if len(path_list_custom) != 0:
            path_list.extend(path_list_custom)
        DISPLAY_PATH = separator.join(path_list)
    os.environ['PYDM_DISPLAYS_PATH'] = DISPLAY_PATH

    from pydm.utilities import setup_renderer

    setup_renderer()

    try:
        """
        We must import QtWebEngineWidgets before creating a QApplication
        otherwise we get the following error if someone adds a WebView at Designer:
        ImportError: QtWebEngineWidgets must be imported before a QCoreApplication instance is created
        """
        from qtpy import QtWebEngineWidgets  # noqa: F401
    except ImportError:
        logger.debug("QtWebEngine is not supported.")

    import pydm
    from application import MITRApplication
    from pydm.utilities.macro import parse_macro_string

    from bluesky_widgets.qt import threading as bw_threading
    from bluesky_widgets.qt import run_engine_client as bw_run_engine_client

    def _patch_bluesky_worker_safe_shutdown():
        """
        Avoid RuntimeError on app shutdown if worker signals are deleted
        before a worker thread finishes and emits.
        """
        worker_cls = bw_threading.WorkerBase
        if getattr(worker_cls, "_dc_safe_shutdown_patch_applied", False):
            return

        def _safe_emit(worker, signal_name, *args):
            try:
                signal = getattr(worker._signals, signal_name, None)
                if signal is None:
                    return
                signal.emit(*args)
            except RuntimeError as exc:
                msg = str(exc)
                if "has been deleted" not in msg:
                    raise

        def _patched_run(self):
            _safe_emit(self, "started")
            self._running = True
            try:
                result = self.work()
                _safe_emit(self, "returned", result)
            except Exception as exc:
                _safe_emit(self, "errored", exc)
            _safe_emit(self, "finished")

        worker_cls.run = _patched_run
        worker_cls._dc_safe_shutdown_patch_applied = True

    _patch_bluesky_worker_safe_shutdown()

    _patch_bluesky_status_reload_shutdown(
        bw_run_engine_client.QtReManagerConnection
    )

    parser = argparse.ArgumentParser(description="Python Display Manager")
    parser.add_argument(
        "--ip-addr",
        help="Specify the IP Address of the RE Manager to connect to.",
        default="localhost"
    )
    # parser.add_argument(
    #     "displayfile",
    #     help="A PyDM file to display." + "    Can be either a Qt .ui file, or a Python file.",
    #     nargs="?",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--homefile", help="Path to a PyDM file to return to when the home button is clicked in the navigation bar"
    # )
    parser.add_argument(
        "--displayfile",
        help="A PyDM file to display in the launched window.",
        default="main_screen.ui",
    )
    parser.add_argument(
        "--perfmon",
        action="store_true",
        help="Enable performance monitoring," + " and print CPU usage to the terminal.",
    )
    parser.add_argument("--profile", action="store_true", help="Enable cProfile function profiling, printing on exit.")
    parser.add_argument("--faulthandler", action="store_true", help="Enable faulthandler to trace segmentation faults.")
    parser.add_argument("--hide-nav-bar", action="store_true", help="Start PyDM with the navigation bar hidden.")
    parser.add_argument("--hide-menu-bar", action="store_true", help="Start PyDM with the menu bar hidden.")
    parser.add_argument("--hide-status-bar", action="store_true", help="Start PyDM with the status bar hidden.")
    parser.add_argument("--fullscreen", action="store_true", help="Start PyDM in full screen mode.")
    parser.add_argument("--read-only", action="store_true", help="Start PyDM in a Read-Only mode.")
    parser.add_argument(
        "--log_level",
        help="Configure level of log display",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="PyDM {version}".format(version=pydm.__version__),
        help="Show PyDM's version number and exit.",
    )
    parser.add_argument(
        "-m",
        "--macro",
        help="Specify macro replacements to use, in JSON object format."
        + "    Reminder: JSON requires double quotes for strings, "
        + "so you should wrap this whole argument in single quotes."
        + '  Example: -m \'{"sector": "LI25", "facility": "LCLS"}\''
        + "--or-- specify macro replacements as KEY=value pairs "
        + " using a comma as delimiter  If you want to uses spaces "
        + " after the delimiters or around the = signs, "
        + " wrap the entire set with quotes "
        + '  Example: -m "sector = LI25, facility=LCLS"',
    )
    parser.add_argument(
        "--stylesheet",
        help="Specify the full path to a CSS stylesheet file, which"
        + " can be used to customize the appearance of PyDM and"
        + " Qt widgets.",
        default=None,
    )
    parser.add_argument(
        "display_args",
        help="Arguments to be passed to the PyDM client application" + " (which is a QApplication subclass).",
        nargs=argparse.REMAINDER,
        default=None,
    )

    pydm_args = parser.parse_args()
    if not (os.environ.get("TILED_URI") or os.environ.get("MITR_TILED_URI")):
        os.environ["MITR_CONTROL_HOST"] = str(pydm_args.ip_addr)

    if pydm_args.profile:
        profile = cProfile.Profile()
        profile.enable()

    if pydm_args.faulthandler:
        faulthandler.enable()

    macros = None
    if pydm_args.macro is not None:
        macros = parse_macro_string(pydm_args.macro)

    if pydm_args.log_level:
        logger.setLevel(pydm_args.log_level)
        handler.setLevel(pydm_args.log_level)

    # Set default macros to connect to the 4dh4 IOC
    if macros is None:
        macros = dict(P='4dh4:',ioc='4dh4')

    # Set the EPICS CA and PVA addresses
    os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"
    os.environ["EPICS_CA_ADDR_LIST"] = os.environ.get("EPICS_CA_ADDR_LIST", "") + " " + pydm_args.ip_addr 
    os.environ["EPICS_PVA_ADDR_LIST"] = os.environ.get("EPICS_PVA_ADDR_LIST", "") + " " + pydm_args.ip_addr 
 
    if pydm_args.ip_addr == 'localhost':
        os.environ["EPICS_CA_AUTO_ADDR_LIST"] = "NO"

    # ic(macros)

    app = MITRApplication(
        ipaddress=str(pydm_args.ip_addr),
        ui_file=pydm_args.displayfile,
        command_line_args=pydm_args.display_args,
        perfmon=pydm_args.perfmon,
        hide_nav_bar=pydm_args.hide_nav_bar,
        hide_menu_bar=pydm_args.hide_menu_bar,
        hide_status_bar=pydm_args.hide_status_bar,
        fullscreen=pydm_args.fullscreen,
        read_only=pydm_args.read_only,
        macros=macros,
        stylesheet_path=pydm_args.stylesheet,
        # home_file=pydm_args.homefile,
    )

    base_path = os.path.dirname(os.path.realpath(__file__))
    icon_path_mask = os.path.join(base_path, "icons", "pydm_{}.png")

    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path_mask.format(16), QtCore.QSize(16, 16))
    app_icon.addFile(icon_path_mask.format(24), QtCore.QSize(24, 24))
    app_icon.addFile(icon_path_mask.format(32), QtCore.QSize(32, 32))
    app_icon.addFile(icon_path_mask.format(64), QtCore.QSize(64, 64))
    app_icon.addFile(icon_path_mask.format(128), QtCore.QSize(128, 128))
    app_icon.addFile(icon_path_mask.format(256), QtCore.QSize(256, 256))

    app.setWindowIcon(app_icon)
    app.setApplicationName("MITR")

    pydm.utilities.shortcuts.install_connection_inspector(parent=app.main_window)

    from bluesky_widgets.qt.threading import wait_for_workers_to_quit

    _shutdown_started = False

    def _wait_for_workers_bounded(timeout_ms=750):
        """Call wait_for_workers_to_quit without risking long UI-blocking hangs."""
        try:
            sig = inspect.signature(wait_for_workers_to_quit)
            params = list(sig.parameters.values())
        except Exception:
            params = []

        try:
            if params:
                p0 = params[0]
                name = p0.name.lower()
                default = p0.default
                timeout_arg = timeout_ms / 1000.0

                # Best-effort unit inference. Favor short, safe waits.
                if "msec" in name or "millisecond" in name:
                    timeout_arg = int(timeout_ms)
                elif "sec" in name:
                    timeout_arg = timeout_ms / 1000.0
                elif isinstance(default, (int, float)):
                    if default >= 100:
                        timeout_arg = int(timeout_ms)
                    else:
                        timeout_arg = timeout_ms / 1000.0

                wait_for_workers_to_quit(timeout_arg)
                return True
        except RuntimeError:
            # Timed out waiting for workers to exit.
            return False
        except Exception:
            pass

        # Unknown signature: run in daemon thread and wait briefly.
        done = False

        def _waiter():
            nonlocal done
            try:
                wait_for_workers_to_quit()
            except Exception:
                pass
            done = True

        waiter = threading.Thread(target=_waiter, daemon=True)
        waiter.start()
        waiter.join(timeout=timeout_ms / 1000.0)
        return done

    def _on_about_to_quit():
        nonlocal _shutdown_started
        if _shutdown_started:
            return
        _shutdown_started = True

        try:
            import epics.ca as epics_ca
        except Exception:
            epics_ca = None
        if epics_ca is not None:
            try:
                epics_ca.disable_ca_messages()
            except Exception:
                pass

        main_window = getattr(app, "main_window", None)
        cleanup = getattr(main_window, "cleanup_before_close", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass

        re_client = getattr(app, "re_client", None)
        stop_console_monitor = getattr(re_client, "stop_console_output_monitoring", None)
        if callable(stop_console_monitor):
            try:
                stop_console_monitor()
            except Exception:
                pass

        dispatcher_service = getattr(app, "document_dispatcher", None)
        if dispatcher_service is not None:
            try:
                dispatcher_service.stop()
            except Exception:
                pass

        t0 = time.monotonic()
        workers_done = _wait_for_workers_bounded(timeout_ms=750)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if (not workers_done) or elapsed_ms > 900:
            print(f"Worker shutdown exceeded target timeout ({elapsed_ms:.0f} ms).")

    app.aboutToQuit.connect(_on_about_to_quit)

    exit_code = app.exec_()

    try:
        import epics.ca as epics_ca
    except Exception:
        epics_ca = None

    if epics_ca is not None:
        try:
            epics_ca.disable_ca_messages()
        except Exception:
            pass
        try:
            epics_ca.finalize_libca(maxtime=1.0)
        except Exception:
            pass

    if pydm_args.profile:
        profile.disable()
        stats = pstats.Stats(
            profile,
            stream=sys.stdout,
        ).sort_stats(pstats.SortKey.CUMULATIVE)
        stats.print_stats()

    if pydm_args.faulthandler:
        faulthandler.disable()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
