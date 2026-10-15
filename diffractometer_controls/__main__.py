"""Launch the source checkout without installation or working-directory changes."""

import sys
from pathlib import Path

# Python can execute this directory (or this file) without importing the
# package first. Add the checkout root before loading the shared launcher.
if __package__ in (None, ""):
    _checkout_root = str(Path(__file__).resolve().parents[1])
    if _checkout_root not in sys.path:
        sys.path.insert(0, _checkout_root)


def main():
    from diffractometer_controls.launcher import main as launch
    launch()


if __name__ == "__main__":
    main()
