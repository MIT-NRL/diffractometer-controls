from pathlib import Path
_source = Path(__file__).resolve().parents[2] / "startup/05-run_status_publisher.py"
exec(compile(_source.read_text(), str(_source), "exec"), globals())
