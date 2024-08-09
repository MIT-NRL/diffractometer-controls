
import os
from bluesky import RunEngine
from bluesky.utils import PersistentDict
from bluesky.callbacks.zmq import Publisher

from pathlib import Path

RE = RunEngine({})
RE.md = PersistentDict(str(Path("~/.bluesky_history").expanduser()))

publisher = Publisher('localhost:5567')
RE.subscribe(publisher)

# Add default metadata
RE.md['facility'] = 'MITR'
RE.md['beamline_id'] = '4DH4'


from bluesky import SupplementalData
sd = SupplementalData()
RE.preprocessors.append(sd)

# Set up a Broker.
from databroker import Broker
from tiled.client import from_uri

client = from_uri("http://127.0.0.1:8000", api_key="e752bc7528d4dd7bcca206fa8adbdc727b726d42e0f3564663ec3a06bb66b4b1")

# Use persistent database backed by MongoDB
# db = Broker(client['4dh4_imaging'])
# db = Broker(client['4dh4_diffraction'])
db = Broker(client['testdb'])

# db = Broker.named("temp")
# and subscribe it to the RunEngine
RE.subscribe(db.insert)


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
