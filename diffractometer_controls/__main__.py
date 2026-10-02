"""Launch from the checkout root with python -m diffractometer_controls."""

import os
from pathlib import Path
import sys


def main():
    ui_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(ui_dir))
    # The existing launcher resolves external displays and branding relative
    # to this directory. Preserve its configuration and local path overrides.
    os.chdir(ui_dir)
    from diffractometer_controls.launcher import main as launch
    launch()


if __name__ == "__main__":
    main()
