"""Minimal, session-only RunEngine configuration for demo mode."""

import os

from bluesky import RunEngine, SupplementalData
from bluesky.callbacks.best_effort import BestEffortCallback
from bluesky.callbacks.zmq import Publisher


if os.environ.get("MITR_DEMO_ACTIVE") != "1":
    raise RuntimeError("The demo Queue Server profile may only run in demo mode")

RE = RunEngine({})
RE.md.update(facility="MITR", beamline_id="4DH4", demo=True)
sd = SupplementalData()
RE.preprocessors.append(sd)

document_input = os.environ.get("MITR_DOCUMENT_INPUT_ADDR", "tcp://127.0.0.1:61567")
RE.subscribe(Publisher(document_input.removeprefix("tcp://")))
bec = BestEffortCallback()
bec.disable_plots()
RE.subscribe(bec)

print("Bluesky demo startup: in-memory metadata, no Tiled and no file writers")
