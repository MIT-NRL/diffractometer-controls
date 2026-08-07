
import os
import importlib.util
from bluesky import RunEngine
from bluesky.utils import PersistentDict
from bluesky.callbacks.zmq import Publisher

from pathlib import Path

def _load_bluesky_mode_helpers():
    module_path = Path(__file__).resolve().parents[2] / "bluesky_mode.py"
    spec = importlib.util.spec_from_file_location("mitr_bluesky_mode", module_path)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Unable to load bluesky mode helper from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bluesky_mode = _load_bluesky_mode_helpers()
BLUESKY_MODE_TEST = _bluesky_mode.BLUESKY_MODE_TEST
get_bluesky_history_path = _bluesky_mode.get_bluesky_history_path
get_bluesky_mode = _bluesky_mode.get_bluesky_mode
publish_bluesky_active_mode = _bluesky_mode.publish_bluesky_active_mode

BLUESKY_MODE = get_bluesky_mode()
publish_bluesky_active_mode(BLUESKY_MODE)
BLUESKY_HISTORY_PATH = get_bluesky_history_path(BLUESKY_MODE)
RE = RunEngine({})
RE.md = PersistentDict(str(BLUESKY_HISTORY_PATH))
RE.md["bluesky_mode"] = BLUESKY_MODE

publisher = Publisher('localhost:5567')
RE.subscribe(publisher)

print(f"Bluesky startup mode: {BLUESKY_MODE} (scan history: {BLUESKY_HISTORY_PATH})")

# Add default metadata
RE.md['facility'] = 'MITR'
RE.md['beamline_id'] = '4DH4'


from bluesky import SupplementalData
sd = SupplementalData()
RE.preprocessors.append(sd)

# Set up a Broker.
from databroker import Broker
try:
    from bluesky_tiled_plugins import TiledWriter
except ImportError:
    from bluesky.callbacks.tiled_writer import TiledWriter
from tiled.client import from_uri

def get_tiled_uri(host=None):
    uri = os.environ.get("TILED_URI") or os.environ.get("MITR_TILED_URI")
    if uri:
        return uri

    host = host or os.environ.get("MITR_CONTROL_HOST") or os.environ.get("MITR_CONTROL_IP") or "localhost"
    port = os.environ.get("MITR_TILED_PORT", "8000")
    return f"http://{host}:{port}"

key = os.environ.get("TILED_WRITER_API_KEY") or os.environ["TILED_API_KEY"]
client = from_uri(get_tiled_uri(), api_key=key)

def _catalog_exists(container, name):
    try:
        if name in container:
            return True
    except Exception:
        pass
    try:
        container[name]
    except Exception:
        return False
    return True


def _resolve_catalog_name(container, preferred_names):
    for name in preferred_names:
        if _catalog_exists(container, name):
            return name
    return preferred_names[0]


if BLUESKY_MODE == BLUESKY_MODE_TEST:
    BLUESKY_CATALOG_PREFERRED_NAMES = ["testdb"]
else:
    BLUESKY_CATALOG_PREFERRED_NAMES = [
        "4dh4",
        "4dh4_imaging",
        "4dh4_diffraction",
        "4dh4Collection",
    ]

BLUESKY_CATALOG_NAME = _resolve_catalog_name(client, BLUESKY_CATALOG_PREFERRED_NAMES)
BLUESKY_CATALOG = client[BLUESKY_CATALOG_NAME]
RE.md["bluesky_catalog"] = BLUESKY_CATALOG_NAME
if BLUESKY_CATALOG_NAME in {"4dh4", "testdb"}:
    db = BLUESKY_CATALOG
    RE.subscribe(TiledWriter(BLUESKY_CATALOG, max_array_size=512))
else:
    db = Broker(BLUESKY_CATALOG)
    RE.subscribe(db.insert)
print(f"Bluesky data catalog: {BLUESKY_CATALOG_NAME}")
if BLUESKY_CATALOG_NAME != BLUESKY_CATALOG_PREFERRED_NAMES[0]:
    print(
        "Requested preferred catalog "
        f"{BLUESKY_CATALOG_PREFERRED_NAMES[0]!r} was not available; "
        f"using {BLUESKY_CATALOG_NAME!r} instead."
    )


# from bluesky.magics import BlueskyMagics
# get_ipython().register_magics(BlueskyMagics)


# Use BestEffortCallback 
# TODO: Retire our use of BestEffortCallback, using a table from
# bluesky_widgets once one is available.
from bluesky.callbacks.best_effort import BestEffortCallback
bec = BestEffortCallback()
bec.disable_plots()
RE.subscribe(bec)


import matplotlib.pyplot as plt
# Make plots update live while scans run.
# from bluesky.utils import install_nb_kicker
# install_nb_kicker()

# convenience imports
# some of the * imports are for 'back-compatibility' of a sort -- we have
# taught BL staff to expect LiveTable and LivePlot etc. to be in their
# namespace
import numpy as np

import bluesky.callbacks
from bluesky.callbacks import *

import bluesky.plans
import bluesky.plans as bp
# from bluesky.plans import *

import bluesky.plan_stubs
import bluesky.plan_stubs as bps
# from bluesky.plan_stubs import *

import bluesky.preprocessors
import bluesky.preprocessors as bpp
# import bluesky.simulators
# from bluesky.simulators import *

from bluesky.simulators import summarize_plan, check_limits, plot_raster_path
#from bluesky.plan_tools import plot_raster_path

from ophyd.sim import det, motor, noisy_det

### make temperature motor alias
from ophyd.sim import motor3 as temperature
temperature.name = "temperature"

# from bluesky.utils import ProgressBarManager
# RE.waiting_hook = ProgressBarManager()

def md_info(default_md = RE.md):
    '''Formatted print of RunEngine metadata.'''

    print('Current peristent metadata for each scan are:')
    for info in default_md:
        val = default_md[info]
        print(f'    {info:_<30} : {val}')
    print('\n\n Use \'md_info()\' or \'RE.md\' to inspect again.')
