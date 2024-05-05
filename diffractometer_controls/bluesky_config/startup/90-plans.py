import bluesky.plans
import bluesky.plans as bp
from bluesky.plans import *

import bluesky.plan_stubs
import bluesky.plan_stubs as bps
# from bluesky.plan_stubs import *

import bluesky.preprocessors
import bluesky.preprocessors as bpp

# from "20-detectors.py" import he3psd

monitor_and_count = bpp.monitor_during_decorator([he3psd.counts])(bp.count)