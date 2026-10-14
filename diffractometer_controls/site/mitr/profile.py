"""MITR bindings for the reusable workspace. Importing this module performs no I/O."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from control_ui.core.options import PlanEditorOptions

CHECKOUT_ROOT = Path(__file__).resolve().parents[3]
APPLICATION_ROOT = CHECKOUT_ROOT / "diffractometer_controls"
ASSETS = APPLICATION_ROOT / "assets"
PV_PREFIX = "4dh4:"
ESTIMATION_PVS = {
    "image_bytes": "4dh4:cam1:ArraySize_RBV",
    "imaging_exposure_time_s": "4dh4:cam1:AcquireTime_RBV",
    "diffraction_acquire_time_s": "4dh4:he3PSD:AcquireTime_RBV",
}
DEFAULT_IMAGING_ROOTS = ("/home/mitr_4dh4/Data/%Y", "~/Data/%Y")
PANEL_MACROS = {"P": PV_PREFIX, "R": "", "ioc": "4dh4"}


def demo_active():
    return os.environ.get("MITR_DEMO_ACTIVE") == "1"


def pv_prefix():
    return os.environ.get("MITR_DEMO_EPICS_PREFIX", "demo4dh4:") if demo_active() else PV_PREFIX


def estimation_pvs():
    return {key: value.replace(PV_PREFIX, pv_prefix(), 1) for key, value in ESTIMATION_PVS.items()}


def panel_macros():
    return {"P": pv_prefix(), "R": "", "ioc": "demo4dh4" if demo_active() else "4dh4"}
SCREEN_PATHS = {
    "diffraction": APPLICATION_ROOT / "screens/diffraction/diffractometer_gui.py",
    "tomography": APPLICATION_ROOT / "screens/tomography/tomography_gui.py",
}


def epics_root():
    return Path(os.environ.get("MITR_EPICS_ROOT", str(Path.home() / "EPICS"))).expanduser()


def ioc_root():
    return Path(os.environ.get("MITR_IOC_ROOT", str(epics_root() / "IOCs/4dh4"))).expanduser()


def ioc_launcher():
    return Path(os.environ.get("MITR_IOC_LAUNCHER", str(ioc_root() / "iocBoot/ioc4dh4/softioc/4dh4.pl"))).expanduser()


def epics_support():
    return Path(os.environ.get("MITR_EPICS_SUPPORT", str(epics_root() / "synApps-6-3/support"))).expanduser()


def display_directories():
    return tuple(CHECKOUT_ROOT / name for name in (
        "diffractometer_controls", "diffractometer_controls/site/mitr",
        "diffractometer_controls/screens/diffraction", "diffractometer_controls/screens/tomography",
        "diffractometer_controls/analysis", "control_ui/widgets", "control_ui/panels",
    ))


def external_display_directories(github_root=None):
    """Find original EPICS displays, preferring the installed IOC's modules.

    envPaths identifies the versions actually built into the IOC. Searching
    those first avoids selecting an older areaDetector or measComp installation.
    No network addresses or support paths are stored in this checkout.
    """
    modules = []
    for boot in ("ioc4dh4-motors", "ioc4dh4"):
        env_paths = ioc_root() / "iocBoot" / boot / "envPaths"
        if env_paths.is_file():
            for name, value in re.findall(r'epicsEnvSet\("([^"\n]+)",\s*"([^"\n]+)"\)', env_paths.read_text()):
                if name not in {"TOP", "SUPPORT", "EPICS_BASE"} and "$" not in value:
                    modules.append(Path(value).expanduser())
            break
    roots = modules + [epics_support()]
    if github_root is not None:
        roots.append(Path(github_root).expanduser())
    patterns = ("**/op/adl*", "**/opi/medm*", "**/op/PyDM*", "**/op/pydm*")
    paths, seen = [], set()
    visited = set()
    for root in roots:
        if root in visited or not root.is_dir():
            continue
        visited.add(root)
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                resolved = path.resolve()
                if path.is_dir() and resolved not in seen:
                    paths.append(path)
                    seen.add(resolved)
    return tuple(paths)


def configure_display_paths(github_root=None):
    """Keep user overrides, then add local panels and installed source displays."""
    existing = os.environ.get("PYDM_DISPLAYS_PATH", "").split(os.pathsep)
    discovered = list(display_directories()) + list(external_display_directories(github_root))
    automatic = {path.resolve() for path in discovered}
    # A child GUI inherits the parent's generated path list. Rebuild those
    # entries in module order rather than promoting old discovery to overrides.
    paths = [Path(path).expanduser() for path in existing
             if path and Path(path).expanduser().resolve() not in automatic]
    paths += discovered
    archived = (CHECKOUT_ROOT / "vendor" / "displays").resolve()
    result, seen = [], set()
    for path in paths:
        resolved = path.resolve()
        if resolved == archived or archived in resolved.parents or resolved in seen:
            continue
        seen.add(resolved)
        result.append(str(path))
    os.environ["PYDM_DISPLAYS_PATH"] = os.pathsep.join(result)
    return tuple(result)


def screen_factory(mode):
    if mode == "diffraction":
        from diffractometer_controls.screens.diffraction.diffractometer_gui import MainScreen
    elif mode == "tomography":
        from diffractometer_controls.screens.tomography.tomography_gui import MainScreen
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")
    return MainScreen


def editor_options():
    if demo_active():
        return PlanEditorOptions(query_mode="local", local_roots=())
    def number(name, default, convert):
        try:
            return convert(os.environ.get(name, default))
        except (TypeError, ValueError):
            return default
    roots = tuple(filter(None, os.environ.get("MITR_IMAGING_DATA_ROOTS", "").split(os.pathsep)))
    root = os.environ.get("MITR_IMAGING_DATA_ROOT", "").strip()
    roots += ((root,) if root else ()) + DEFAULT_IMAGING_ROOTS
    mode = os.environ.get("MITR_FILE_DIR_QUERY_MODE", "stream").strip().lower()
    return PlanEditorOptions(
        query_mode=mode if mode in ("local", "worker", "stream") else "stream",
        cache_ttl_s=max(2.0, number("MITR_FILE_DIR_CACHE_TTL_S", 2.0, float)),
        stream_address=os.environ.get("MITR_FILE_DIR_STREAM_ADDR", "").strip(),
        snapshot_address=os.environ.get("MITR_FILE_DIR_SNAPSHOT_ADDR", "").strip(),
        stream_topic=os.environ.get("MITR_FILE_DIR_STREAM_TOPIC", "file_dir_choices").strip() or "file_dir_choices",
        local_roots=roots, max_depth=number("MITR_FILE_DIR_MAX_DEPTH", 3, int),
    )


def estimation_context(**overrides):
    from epics import caget
    from control_ui.widgets.plan_time_estimation import build_estimation_context
    return build_estimation_context(caget_func=caget, pv_mapping=estimation_pvs(), **overrides)


@dataclass(frozen=True)
class Endpoints:
    control: str
    info: str
    documents: str


def endpoints(host="localhost"):
    if demo_active():
        return Endpoints(os.environ["MITR_QSERVER_CONTROL_ADDR"], os.environ["MITR_QSERVER_INFO_ADDR"],
                         os.environ["MITR_DOCUMENT_ADDR"].removeprefix("tcp://"))
    return Endpoints(f"tcp://{host}:60615", f"tcp://{host}:60625", f"{host}:5568")
