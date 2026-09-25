"""Load the shared imaging plans against no-file demo cameras."""

import os
from pathlib import Path

_shared = Path(os.environ["MITR_DEMO_SHARED_STARTUP_DIR"]) / "92-plans_imaging.py"
exec(compile(_shared.read_bytes(), str(_shared), "exec"), globals(), globals())
