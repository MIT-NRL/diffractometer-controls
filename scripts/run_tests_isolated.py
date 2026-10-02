"""Run each test module in its own process and retain complete failure logs.

Qt tests share process-wide state; isolation also preserves results if a
native library aborts. --checkout permits comparison with a baseline archive.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/tests/unified"))
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["MITR_FILE_DIR_QUERY_MODE"] = "local"
    environment["PYTHONPATH"] = os.pathsep.join([str(checkout), str(checkout / "diffractometer_controls")])
    results = []
    for file in sorted((checkout / "diffractometer_controls" / "tests").glob("test_*.py")):
        module = f"diffractometer_controls.tests.{file.stem}"
        with (output / f"{file.stem}.log").open("w", encoding="utf8") as log:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", module], cwd=checkout,
                    env=environment, stdout=log, stderr=subprocess.STDOUT,
                    timeout=120,
                )
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        text = (output / f"{file.stem}.log").read_text(encoding="utf8", errors="replace")
        match = re.search(r"Ran (\d+) tests?", text)
        skipped = re.search(r"skipped=(\d+)", text)
        item = {"module": module, "exit_code": code,
                "tests": int(match.group(1)) if match else None,
                "skipped": int(skipped.group(1)) if skipped else 0}
        results.append(item)
        print(json.dumps(item), flush=True)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(item["exit_code"] != 0 for item in results))


if __name__ == "__main__":
    sys.exit(main())
