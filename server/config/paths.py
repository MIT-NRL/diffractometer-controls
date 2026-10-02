"""Resolve detector XML files without embedding a checkout or IOC host path."""

import os
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent


def detector_xml(name):
    # An IOC on another host may need its own path to the same XML resource.
    root = Path(os.environ.get("MITR_DETECTOR_XML_DIR", str(CONFIG_ROOT))).expanduser()
    return str(root / name)
