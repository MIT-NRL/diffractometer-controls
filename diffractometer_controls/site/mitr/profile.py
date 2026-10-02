"""MITR bindings for the reusable workspace. Importing this module performs no I/O."""

import os
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
        "diffractometer_controls/analysis", "control_ui/widgets", "control_ui/panels", "vendor/displays",
    ))


def screen_factory(mode):
    if mode == "diffraction":
        from diffractometer_controls.screens.diffraction.diffractometer_gui import MainScreen
    elif mode == "tomography":
        from diffractometer_controls.screens.tomography.tomography_gui import MainScreen
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")
    return MainScreen


def editor_options():
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
    return build_estimation_context(caget_func=caget, pv_mapping=ESTIMATION_PVS, **overrides)


@dataclass(frozen=True)
class Endpoints:
    control: str
    info: str
    documents: str


def endpoints(host="localhost"):
    return Endpoints(f"tcp://{host}:60615", f"tcp://{host}:60625", f"{host}:5568")
