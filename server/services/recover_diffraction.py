"""Rebuild an interrupted NeXus acquisition into a new file from its journal."""

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

from server.writers.he3_nexus import HE3DiffractionNXWriter
from server.writers.scalar import ScalarDiffractionWriter


def recover_journal(journal, output):
    """Preserve original inputs and label a missing Stop as recovered/aborted.

    Recovery is offline and does not write to Tiled or clear HDF5 crash flags.
    A partially written final JSON line is ignored; malformed earlier lines
    fail instead of silently skipping acquisition records.
    """
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing file: {output}")
    if output.suffix not in {".nxs", ".h5", ".hdf5"}:
        raise ValueError("Choose a new .nxs, .h5, or .hdf5 output file")
    writer = None
    descriptors = {}
    counts = {}
    start = None
    last_time = 0
    stop_seen = False
    try:
        with Path(journal).open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith("\n") and not stream.read(1):
                        break
                    raise
                name, doc = record["name"], record["doc"]
                if stop_seen:
                    raise ValueError("Journal contains documents after Stop")
                last_time = max(last_time, float(doc.get("time", 0)))
                if name == "start":
                    if writer is not None:
                        raise ValueError("Journal contains more than one run")
                    if doc.get("file_type") == "csv":
                        raise ValueError("This recovery command rebuilds NeXus acquisitions, not CSV exports")
                    for writer_class in (HE3DiffractionNXWriter, ScalarDiffractionWriter):
                        if writer_class()._is_supported_run(doc):
                            class RecoveryWriter(writer_class):
                                def make_file_name(self):
                                    return output
                            writer = RecoveryWriter()
                            break
                    if writer is None:
                        raise ValueError("No diffraction NeXus writer accepts this run")
                    start = doc
                    writer.receiver(name, doc)
                    writer.metadata["recovery"] = {
                        "journal": str(Path(journal).resolve()), "recovered_at": time.time(),
                        "original_run_uid": doc["uid"],
                    }
                    continue
                if writer is None:
                    raise ValueError("Journal must begin with a Start document")
                if name == "descriptor":
                    descriptors[doc["uid"]] = doc["name"]
                if name == "event":
                    key = descriptors[doc["descriptor"]]
                    counts[key] = counts.get(key, 0) + 1
                if name == "stop":
                    stop_seen = True
                writer.receiver(name, doc)
        if writer is None:
            raise ValueError("Journal contains no Start document")
        if not stop_seen:
            writer.receiver("stop", {
                "uid": str(uuid4()), "run_start": start["uid"], "time": max(last_time, start["time"]),
                "exit_status": "abort", "num_events": counts,
                "reason": "Recovered interrupted acquisition; no original Stop document was recorded.",
            })
        return writer.output_nexus_file
    finally:
        if writer is not None:
            writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New NeXus filename; never overwrites an existing file")
    args = parser.parse_args()
    print(recover_journal(args.journal, args.output))


if __name__ == "__main__":
    main()
