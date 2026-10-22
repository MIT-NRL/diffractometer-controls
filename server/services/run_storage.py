"""Route original documents to run-local exports and HDF5 references to Tiled."""

import copy
import hashlib
import json
import logging
from pathlib import Path
import os
from uuid import NAMESPACE_URL, uuid5
from urllib.parse import urlsplit, urlunsplit

import h5py
import numpy as np
from event_model import RunRouter, unpack_event_page

from server.writers.base import json_default

logger = logging.getLogger(__name__)


class RunStorage:
    """One subscription, ordered file finalization, and independent run state.

    Register writer classes during startup, before running plans. Runs without
    an applicable file writer retain the ordinary TiledWriter behavior.
    """

    def __init__(self, tiled_writer, catalog=None):
        self.tiled_writer = tiled_writer
        self.catalog = catalog
        self.writer_classes = []
        self.last_result = None
        self._completion_errors = {}
        self._active_runs = {}
        self._router = RunRouter([self._factory])

    def add_writer(self, writer_class):
        self.writer_classes.append(writer_class)

    def _factory(self, name, doc):
        for writer_class in self.writer_classes:
            writer = writer_class()
            if writer._is_supported_run(doc):
                run = _FileRun(self, writer)
                self._active_runs[doc["uid"]] = run
                return [run], []
        return [self.tiled_writer], []

    def __call__(self, name, doc):
        result = self._router(name, doc)
        if name == "stop":
            error = self._completion_errors.pop(doc["run_start"], None)
            if error is not None:
                raise error
        return result

    def close(self):
        """Close unfinished files on an orderly worker exit; keep journals."""
        for run in list(self._active_runs.values()):
            close = getattr(run.writer, "close", None)
            if close is not None:
                close()
        self._active_runs.clear()


class _FileRun:
    def __init__(self, owner, writer):
        self.owner = owner
        self.writer = writer
        self.descriptors = {}
        self.emitted = set()
        self.sequence_numbers = {}
        self.array_keys = {}
        self.tiled_error = None
        self.registration_error = None
        self.storage_error = None
        self.live_resources = set()
        self.live_published = {}
        self.published_points = 0
        self.journal = None
        self.stop_sent = False

    def _record(self, name, doc):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"name": name, "doc": doc}, default=json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _send(self, name, doc):
        if self.tiled_error is not None:
            return
        external = name in {"stream_resource", "stream_datum"}
        if external and self.registration_error is not None:
            return
        try:
            if name == "stop":
                self.stop_sent = True
            self.owner.tiled_writer(name, doc)
        except Exception as exc:
            if external:
                self.registration_error = exc
            else:
                self.tiled_error = exc
            logger.exception("Tiled persistence failed; original documents remain in %s", self.journal)

    def _status(self, status, stop=None, error=None):
        if self.owner.catalog is None:
            return
        try:
            run = self.owner.catalog[self.writer.uid]
            base = getattr(run, "base", run)
            metadata = {"file_export": {"status": status, "run_uid": self.writer.uid,
                                        "error": error}}
            if getattr(self.writer, "incremental_enabled", False):
                export = metadata["file_export"]
                export.update(incremental=True, swmr=True, committed_points=self.published_points)
                if self.writer.live_file is not None:
                    export["file_uri"] = self.writer.live_file.resolve().as_uri()
            if stop is not None:
                metadata["stop"] = stop
            base.update_metadata(metadata)
        except Exception:
            logger.warning("Could not publish export status for run %s", self.writer.uid, exc_info=True)

    def _emit_descriptor(self, uid, event=None):
        if uid in self.emitted:
            return
        doc = copy.deepcopy(self.descriptors[uid])
        if self.owner.catalog is not None:
            for key in self.array_keys[uid]:
                doc["data_keys"][key]["external"] = "STREAM:"
                if event is not None:
                    doc["data_keys"][key]["dtype_numpy"] = np.asarray(event["data"][key]).dtype.str
        self._send("descriptor", doc)
        self.emitted.add(uid)

    def __call__(self, name, doc):
        try:
            return self._process(name, doc)
        except Exception as exc:
            self.storage_error = exc
            close = getattr(self.writer, "close", None)
            if close is not None:
                close()
            self._status("failed", error=str(exc))
            self.owner.last_result = {"run_uid": self.writer.uid, "status": "failed",
                                      "journal": str(self.journal), "error": str(exc)}
            raise
        finally:
            if name == "stop":
                self.owner._active_runs.pop(doc["run_start"], None)
                close = getattr(self.writer, "close", None)
                if close is not None:
                    close()

    def _process(self, name, doc):
        if name == "event_page":
            for event in unpack_event_page(doc):
                self("event", event)
            return
        if name == "start":
            self.writer.receiver(name, doc)
            if self.owner.catalog is not None:
                parts = urlsplit(str(self.owner.catalog.uri))
                catalog_uri = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1],
                                          parts.path, "", ""))
                self.writer.metadata["tiled_run_uri"] = catalog_uri.rstrip("/") + "/" + doc["uid"]
            self.journal = Path(str(self.writer.file_name) + ".documents.jsonl")
            # An existing journal must be recovered, never silently appended to.
            self.journal.touch(exist_ok=False)
            self._record(name, doc)
            self.owner.last_result = {"run_uid": doc["uid"], "status": "collecting", "journal": str(self.journal)}
            self._send(name, doc)
            self._status("collecting")
            return
        self._record(name, doc)
        if name == "descriptor":
            uid = doc["uid"]
            self.descriptors[uid] = copy.deepcopy(doc)
            self.sequence_numbers[uid] = []
            self.array_keys[uid] = {
                key for key, description in doc["data_keys"].items()
                if description["dtype"] == "array" and not description.get("external")
            }
            self.writer.receiver(name, doc)
            # The first event supplies the actual numpy dtype, including for
            # descriptors whose broad 'array' dtype does not specify it.
            return
        if name == "event":
            uid = doc["descriptor"]
            self.writer.receiver(name, doc)
            self._emit_descriptor(uid, doc)
            self.sequence_numbers[uid].append(doc["seq_num"])
            if self.owner.catalog is not None:
                doc = copy.deepcopy(doc)
                for field in ("data", "timestamps", "filled"):
                    for key in self.array_keys[uid]:
                        doc.get(field, {}).pop(key, None)
            self._send(name, doc)
            if self.owner.catalog is not None and getattr(self.writer, "live_file", None) is not None:
                # Making the file visible also makes earlier monitor readings
                # publishable. Catch up each stream without replaying datums.
                for descriptor_uid in self.descriptors:
                    self._register_live_event(descriptor_uid)
                if self.tiled_error is None and self.registration_error is None:
                    if self.descriptors[uid]["name"] == "primary":
                        self.published_points = len(self.sequence_numbers[uid])
                    self._status("collecting")
            return
        if name != "stop":
            self.writer.receiver(name, doc)
            self._send(name, doc)
            return

        result = {"run_uid": self.writer.uid, "status": "writing", "journal": str(self.journal)}
        self.owner.last_result = result
        self._status("writing")
        try:
            if self.storage_error is not None:
                raise self.storage_error
            self.writer.receiver(name, doc)
            output = self.writer.output_nexus_file or getattr(self.writer, "output_csv_file", None)
            if output is None:
                raise RuntimeError("Writer completed without an output file")
            for uid in self.descriptors:
                self._emit_descriptor(uid)
            # Failure before this point may leave scalar rows missing. Never
            # label that run complete merely because file registration works.
            events_complete = self.tiled_error is None
            if self.owner.catalog is not None and self.writer.output_nexus_file:
                for resource, datums in self._external_documents(output):
                    if resource["uid"] in self.live_resources:
                        continue
                    self._send("stream_resource", resource)
                    for datum in datums:
                        self._send("stream_datum", datum)
            self._send(name, doc)
            manifest = make_manifest(self.writer, output, self.descriptors, self.sequence_numbers)
            manifest["events_complete"] = events_complete
            pending = Path(str(output) + ".registration.json")
            write_json(pending, manifest)
            if self.tiled_error is not None or self.registration_error is not None:
                raise RuntimeError("Tiled did not persist all run documents") from self.tiled_error
            if self.owner.catalog is not None:
                register_export(self.owner.catalog, manifest)
            pending.unlink()
            self.journal.unlink()
            result.update(status="ready", file=str(output))
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            logger.exception("Run export or registration failed; recovery documents: %s", self.journal)
            if not self.stop_sent:
                self._send("stop", doc)
            self._status("failed", stop=doc, error=str(exc))
            # Let RunRouter release this run's buffers before reporting the
            # failure back to QueueServer.
            self.owner._completion_errors[self.writer.uid] = exc

    def _external_documents(self, output):
        for uid, descriptor in self.descriptors.items():
            seqs = self.sequence_numbers[uid]
            if not seqs:
                continue
            for key in sorted(self.array_keys[uid]):
                resource_uid = str(uuid5(NAMESPACE_URL, f"{self.writer.uid}/{uid}/{key}"))
                resource = {
                    "uid": resource_uid, "run_start": self.writer.uid,
                    "data_key": key, "uri": Path(output).resolve().as_uri(),
                    "mimetype": "application/x-hdf5",
                    "parameters": {
                        "dataset": f"/entry/instrument/bluesky/streams/{descriptor['name']}/{key}/value",
                        "join_method": "stack", "swmr": bool(getattr(self.writer, "incremental_enabled", False)),
                        "chunk_shape": [1, *descriptor["data_keys"][key]["shape"]],
                    },
                }
                yield resource, self._stream_datums(resource_uid, uid, seqs)

    @staticmethod
    def _stream_datums(resource_uid, descriptor_uid, seqs, first=0):
        for index in range(first + 1, len(seqs) + 1):
            if index == len(seqs) or seqs[index] != seqs[index - 1] + 1:
                yield {
                    "uid": f"{resource_uid}/{first}", "stream_resource": resource_uid,
                    "descriptor": descriptor_uid, "indices": {"start": first, "stop": index},
                    "seq_nums": {"start": seqs[first], "stop": seqs[index - 1] + 1},
                }
                first = index

    def _register_live_event(self, uid):
        """Advertise the unregistered committed suffix, preserving sequence gaps."""
        from server.services.tiled_streams import install_committed_hdf5_consolidator

        seqs = self.sequence_numbers[uid]
        if not seqs or not self.array_keys[uid] or self.tiled_error is not None or self.registration_error is not None:
            return
        install_committed_hdf5_consolidator()
        output = self.writer.live_file
        descriptor = self.descriptors[uid]
        for key in sorted(self.array_keys[uid]):
            resource_uid = str(uuid5(NAMESPACE_URL, f"{self.writer.uid}/{uid}/{key}"))
            if resource_uid not in self.live_resources:
                resource = {
                    "uid": resource_uid, "run_start": self.writer.uid,
                    "data_key": key, "uri": output.resolve().as_uri(),
                    "mimetype": "application/x-hdf5",
                    "parameters": {"dataset": f"/entry/instrument/bluesky/streams/{descriptor['name']}/{key}/value",
                                   "join_method": "stack", "swmr": True,
                                   "committed_prefix": True,
                                   "chunk_shape": [1, *descriptor["data_keys"][key]["shape"]]},
                }
                self._send("stream_resource", resource)
                if self.tiled_error is not None or self.registration_error is not None:
                    return
                self.live_resources.add(resource_uid)
            first = self.live_published.get(resource_uid, 0)
            for datum in self._stream_datums(resource_uid, uid, seqs, first):
                self._send("stream_datum", datum)
                if self.tiled_error is not None or self.registration_error is not None:
                    return
                self.live_published[resource_uid] = datum["indices"]["stop"]


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, default=json_default)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def make_manifest(writer, output, descriptors, sequence_numbers):
    digest = hashlib.sha256()
    with Path(output).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": "1.0", "run_uid": writer.uid,
        "catalog": writer.metadata.get("bluesky_catalog"),
        "file": str(Path(output).resolve()), "sha256": digest.hexdigest(),
        "format": "nexus" if writer.output_nexus_file else "csv",
        "start": writer.start_document, "stop": writer.stop_document,
        "descriptors": list(descriptors.values()), "sequence_numbers": sequence_numbers,
        "baseline_readings": writer.metadata.get("baseline_readings", {}),
        "incremental": bool(getattr(writer, "incremental_enabled", False)),
    }


def register_export(catalog, manifest):
    """Attach a completed file and summaries without modifying original Start."""
    from tiled.structures.core import StructureFamily
    from tiled.structures.bytes import BytesStructure
    from tiled.structures.data_source import Asset, DataSource, Management

    run = catalog[manifest["run_uid"]]
    base = getattr(run, "base", run)
    try:
        artifacts = base["artifacts"]
    except KeyError:
        artifacts = base.create_container(key="artifacts")
    output = Path(manifest["file"])
    metadata = {key: manifest[key] for key in ("schema_version", "run_uid", "sha256", "format")}
    metadata.update(file_uri=output.as_uri(), status="ready", error=None)
    if manifest.get("incremental"):
        metadata.update(incremental=True, swmr=True)
    kind = manifest["format"]
    try:
        existing = artifacts[kind]
    except KeyError:
        source = DataSource(
            structure_family=StructureFamily.container if kind == "nexus" else StructureFamily.bytes,
            mimetype="application/x-hdf5" if kind == "nexus" else "application/octet-stream",
            management=Management.external,
            structure=None if kind == "nexus" else BytesStructure(),
            parameters={"swmr": bool(manifest.get("incremental"))} if kind == "nexus" else {},
            assets=[Asset(data_uri=output.as_uri(), is_directory=False, parameter="data_uris", num=0)],
        )
        artifacts.new(key=kind, structure_family=source.structure_family,
                      data_sources=[source], metadata=metadata)
    else:
        if any(existing.metadata.get(key) != value for key, value in metadata.items()):
            raise ValueError("Existing artifact differs from the completed export")
    base.update_metadata({
        "baseline_readings": manifest["baseline_readings"],
        "file_export": metadata,
    })


def retry_registration(catalog, manifest_path):
    """Repair references after a completed file's registration failed.

    Scalar rows are never replayed or deleted. A failure during their original
    delivery needs document recovery instead and is explicitly rejected here.
    """
    from tiled.structures.array import ArrayStructure, BuiltinDtype
    from tiled.structures.core import StructureFamily
    from tiled.structures.data_source import Asset, DataSource, Management
    from tiled.client.utils import handle_error
    from tiled.utils import safe_json_dump

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["events_complete"]:
        raise ValueError("Scalar delivery was incomplete; recover the original document journal first")
    output = Path(manifest["file"])
    digest = hashlib.sha256()
    with output.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest["sha256"]:
        raise ValueError("Export checksum changed; refusing to register it")
    run = catalog[manifest["run_uid"]]
    base = getattr(run, "base", run)
    if base.metadata["start"]["uid"] != manifest["run_uid"]:
        raise ValueError("Tiled run UID differs from the export")
    if manifest["format"] == "nexus":
        with h5py.File(output, "r") as handle:
            for descriptor in manifest["descriptors"]:
                stream = descriptor["name"]
                if not manifest["sequence_numbers"][descriptor["uid"]]:
                    continue
                parent = base[stream]
                parent = getattr(parent, "base", parent)
                for key, description in descriptor["data_keys"].items():
                    if description["dtype"] != "array" or description.get("external"):
                        continue
                    dataset_path = f"/entry/instrument/bluesky/streams/{stream}/{key}/value"
                    dataset = handle[dataset_path]
                    source = DataSource(
                        structure_family=StructureFamily.array, mimetype="application/x-hdf5",
                        management=Management.external,
                        structure=ArrayStructure(
                            data_type=BuiltinDtype.from_numpy_dtype(dataset.dtype),
                            shape=dataset.shape,
                            chunks=((1,) * dataset.shape[0], *[(size,) for size in dataset.shape[1:]]),
                        ),
                        parameters={"dataset": dataset_path, "swmr": bool(manifest.get("incremental"))},
                        assets=[Asset(data_uri=output.as_uri(), is_directory=False,
                                      parameter="data_uris", num=0)],
                    )
                    if manifest.get("incremental"):
                        source.parameters["slice"] = f"0:{dataset.shape[0]}"
                    try:
                        node = parent[key]
                    except KeyError:
                        parent.new(key=key, structure_family=source.structure_family,
                                   data_sources=[source])
                    else:
                        existing, = node.data_sources()
                        if (existing.parameters.get("dataset") != dataset_path
                                or [asset.data_uri for asset in existing.assets] != [output.as_uri()]):
                            raise ValueError(f"Existing array {stream}/{key} references another source")
                        # TiledWriter may have created a zero-length source
                        # before its subsequent shape update failed.
                        if (node.shape != dataset.shape or node.dtype != dataset.dtype
                                or existing.parameters.get("slice") != source.parameters.get("slice")):
                            source.id = existing.id
                            response = node.context.http_client.put(
                                node.uri.replace("/metadata/", "/data_source/", 1),
                                content=safe_json_dump({"data_source": source}),
                            )
                            handle_error(response)
    register_export(catalog, manifest)
    base.update_metadata({"stop": manifest["stop"]})
    manifest_path.unlink()
    Path(str(output) + ".documents.jsonl").unlink(missing_ok=True)
    return manifest


def main():
    """Retry only completed exports, using the existing writer credentials."""
    import argparse
    from tiled.client import from_uri

    parser = argparse.ArgumentParser(description="Retry a completed run's HDF5/CSV registration")
    parser.add_argument("manifest", type=Path, help="The retained .registration.json file")
    parser.add_argument("--uri", default=os.environ.get("TILED_URI") or os.environ.get("MITR_TILED_URI"))
    args = parser.parse_args()
    if not args.uri:
        parser.error("Provide --uri or set TILED_URI")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    catalog_name = manifest.get("catalog")
    if not catalog_name:
        parser.error("The registration manifest has no catalog name")
    client = from_uri(args.uri, api_key=os.environ.get("TILED_WRITER_API_KEY") or os.environ.get("TILED_API_KEY"))
    retry_registration(client[catalog_name], args.manifest)
    print(f"Registered completed {manifest['format']} export for run {manifest['run_uid']}")


if __name__ == "__main__":
    main()
