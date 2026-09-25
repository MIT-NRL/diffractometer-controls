"""Load the shared diffraction plans against demo devices."""

import os
from pathlib import Path

_shared = Path(os.environ["MITR_DEMO_SHARED_STARTUP_DIR"]) / "91-plans_diffraction.py"
exec(compile(_shared.read_bytes(), str(_shared), "exec"), globals(), globals())
