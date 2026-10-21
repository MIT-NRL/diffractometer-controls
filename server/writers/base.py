"""Complete and validate a NeXus export before publishing its final filename."""

import copy
import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
from apstools.callbacks.nexus_writer import NXWriter
from event_model import unpack_event_page


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} as acquisition metadata")


class AtomicNXWriter(NXWriter):
    """Synchronous finalization avoids NXWriter's shared background-thread state.

    Acquisition still collects events normally. Only run completion waits for
    the export to close; callers can safely register ``output_nexus_file`` when
    ``stop`` returns. Each production run receives its own writer instance.
    """

    def clear(self):
        super().clear()
        self.start_document = None
        self.stop_document = None
        self.descriptor_documents = {}
        self.output_nexus_file = None

    def start(self, doc):
        super().start(copy.deepcopy(doc))
        self.start_document = copy.deepcopy(doc)

    def descriptor(self, doc):
        super().descriptor(doc)
        if self.scanning:
            self.descriptor_documents[doc["uid"]] = copy.deepcopy(doc)

    def stop(self, doc):
        self.stop_document = copy.deepcopy(doc)
        return super().stop(doc)

    def receiver(self, name, doc):
        if name == "event_page":
            for event in unpack_event_page(doc):
                super().receiver("event", event)
            return
        return super().receiver(name, doc)

    def write_metadata(self, parent):
        self.metadata["storage_schema_version"] = "1.0"
        group = super().write_metadata(parent)
        documents = self.create_NX_group(parent, "documents:NXcollection")
        documents.attrs["schema_version"] = "1.0"
        for name, value in (
            ("start", self.start_document),
            ("stop", self.stop_document),
            ("descriptors", list(self.descriptor_documents.values())),
        ):
            dataset = documents.create_dataset(
                name, data=json.dumps(value, default=json_default)
            )
            dataset.attrs["mime_type"] = "application/json"
        return group

    def write_streams(self, parent):
        # A stopped/aborted run can have a declared stream with no readings.
        # apstools indexes the first timestamp, so omit empty streams from its
        # presentation while retaining their original Descriptor documents.
        streams = self.streams
        self.streams = {
            name: [uid for uid in uids if any(
                entry["data"] for entry in self.acquisitions[uid]["data"].values()
            )]
            for name, uids in streams.items()
        }
        self.streams = {name: uids for name, uids in self.streams.items() if uids}
        try:
            return super().write_streams(parent)
        finally:
            self.streams = streams

    def writer(self):
        destination = Path(self.file_name or self.make_file_name())
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace an existing export: {destination}")
        # Keep partial output for diagnosis if writing or validation fails.
        temporary_path = destination.with_name(f"{destination.name}.{uuid4().hex}.partial")
        # Match the original writer's permissions (0666 filtered by umask),
        # including group access when Tiled runs under another account.
        descriptor = os.open(temporary_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        os.close(descriptor)
        try:
            with h5py.File(temporary_path, "w") as self.root:
                self.write_root(destination)
                self.root.attrs["diffractometer_schema_version"] = "1.0"
            self.root = None
            self.validate_file(temporary_path)
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            # link() atomically publishes without overwriting another export.
            os.link(temporary_path, destination)
            temporary_path.unlink()
            self.output_nexus_file = destination
        finally:
            self.root = None

    def validate_file(self, path):
        with h5py.File(path, "r") as handle:
            uid = handle["/entry/instrument/bluesky/metadata/run_start_uid"].asstr()[()]
            if uid != self.uid:
                raise ValueError("NeXus run UID does not match the acquisition")
            for acquisition in self.acquisitions.values():
                stream = acquisition["stream"]
                for key, entry in acquisition["data"].items():
                    if not entry["data"]:
                        continue
                    dataset = handle[f"/entry/instrument/bluesky/streams/{stream}/{key}/value"]
                    expected = np.asarray(entry["data"])
                    if dataset.shape != expected.shape:
                        raise ValueError(f"NeXus shape mismatch for {stream}/{key}")
