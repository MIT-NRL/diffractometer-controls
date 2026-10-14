"""Execute the ordered demo profile through QueueServer's cross-platform module loader."""
from pathlib import Path
_startup = Path(__file__).with_name("startup")
for _file in sorted(_startup.glob("[0-9]*.py")):
    globals()["__file__"] = str(_file)
    exec(compile(_file.read_text(encoding="utf8"), str(_file), "exec"), globals())
