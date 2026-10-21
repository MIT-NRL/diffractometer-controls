"""Shared diffraction storage defaults and existing PSD geometry."""

import os
from pathlib import Path

DIFFRACTION_ROOT = Path(os.environ.get(
    "MITR_DIFFRACTION_DATA_ROOT", "/home/mitr_4dh4/Data/Diffraction"
)).expanduser()
DIFFRACTION_TEST_ROOT = Path(os.environ.get(
    "MITR_DIFFRACTION_TEST_DATA_ROOT", "/home/mitr_4dh4/Data/TestData/Diffraction"
)).expanduser()
HE3PSD_POSITION_MIN = -209.21799055746422
HE3PSD_POSITION_MAX = 209.21799055746422
