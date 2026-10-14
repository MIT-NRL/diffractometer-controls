"""Session-only metadata and documents; no Tiled or detector file writers."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
if os.environ.get("MITR_DEMO_ACTIVE") != "1":
    raise RuntimeError("The demo startup profile requires an isolated demo runtime")
from bluesky import RunEngine, SupplementalData
from bluesky.callbacks.zmq import Publisher
from bluesky.callbacks.best_effort import BestEffortCallback
RE = RunEngine({"facility": "MITR", "beamline_id": "4DH4", "demo": True})
sd = SupplementalData()
RE.preprocessors.append(sd)
RE.subscribe(Publisher(os.environ["MITR_DOCUMENT_INPUT_ADDR"].removeprefix("tcp://")))
bec = BestEffortCallback()
bec.disable_plots()
RE.subscribe(bec)
print("DEMO: local simulated IOC; metadata/images are in memory; storage disabled")
