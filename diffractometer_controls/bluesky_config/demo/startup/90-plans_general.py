"""Load the shared general plans into the isolated demo namespace."""

import os
from pathlib import Path

_shared = Path(os.environ["MITR_DEMO_SHARED_STARTUP_DIR"]) / "90-plans_general.py"
exec(compile(_shared.read_bytes(), str(_shared), "exec"), globals(), globals())
