"""Publish progress to the demo IOC using the production callback logic."""

import os
from pathlib import Path

_shared = Path(os.environ["MITR_DEMO_SHARED_STARTUP_DIR"]) / "05-run_status_publisher.py"
exec(compile(_shared.read_bytes(), str(_shared), "exec"), globals(), globals())
