"""Recover missing diffraction plotting metadata, with an audit/rollback file.

The default CLI operation is a dry run. Apply requires a write-capable Tiled
API key and an explicit backup path; the GUI reader key remains read-only.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path

from diffractometer_controls.analysis.diffraction_history import (
    _run_metadata, suggest_diffraction_start_updates,
)


def repair_catalog_metadata(catalog, *, backup_path=None):
    plans = []
    for uid, run in catalog.items():
        updates = suggest_diffraction_start_updates(run)
        if updates:
            plans.append({"uid": str(uid), "start_updates": updates})
    if backup_path is None or not plans:
        return plans
    path = Path(backup_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        for plan in plans:
            run = catalog[plan["uid"]]
            before = _run_metadata(run)
            # Re-read before writing so a prior repair or other metadata edit
            # is respected. Never overwrite a field that has since appeared.
            updates = suggest_diffraction_start_updates(run)
            if not updates:
                continue
            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
            history = before.get("metadata_enrichment", [])
            if not isinstance(history, list):
                raise ValueError("Existing metadata_enrichment must be a list; refusing to replace it.")
            provenance = {
                "time": timestamp, "source": "recorded diffraction plan and primary descriptors",
                "start_fields": sorted(updates), "version": 1,
            }
            record = {"uid": plan["uid"], "before_metadata": before,
                      "start_updates": updates, "provenance": provenance}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            base = getattr(run, "base", run)
            base.update_metadata({"start": {**before["start"], **updates},
                                  "metadata_enrichment": [*history, provenance]})
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    if args.apply and args.backup is None:
        parser.error("--apply requires --backup")
    from tiled.client import from_uri

    client = from_uri(args.uri, api_key=os.environ.get("TILED_WRITER_API_KEY") or os.environ.get("TILED_API_KEY"))
    try:
        changes = repair_catalog_metadata(client[args.catalog], backup_path=args.backup if args.apply else None)
        print(json.dumps({"applied": args.apply, "changes": changes}, indent=2))
    finally:
        client.context.close()


if __name__ == "__main__":
    main()
