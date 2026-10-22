"""One-file NeXus acquisition with durable point commits and SWMR readers."""

import datetime as dt
import json
import os
from pathlib import Path
from uuid import uuid4

import h5py
import numpy as np
import yaml

from server.writers.base import AtomicNXWriter, json_default
from server.config.diffraction import HE3PSD_POSITION_MIN, HE3PSD_POSITION_MAX


STREAMS = "/entry/instrument/bluesky/streams"
DOCUMENTS = "/entry/instrument/bluesky/documents"


class DatasetSequence:
    """Small list-compatible view for the existing baseline summary helpers."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        value = self.dataset[index]
        return value.decode("utf-8") if isinstance(value, bytes) else value


def write_json_dataset(dataset, value):
    """UTF-8 bytes avoid SWMR's restriction on variable-length string writes."""
    raw = np.frombuffer(json.dumps(value, default=json_default).encode("utf-8"), dtype="u1")
    dataset.resize((len(raw),))
    dataset[:] = raw


class IncrementalNXWriter(AtomicNXWriter):
    """Append readings without retaining spectra in Python lists.

    All stream schemas must be declared before the first primary event. Changing
    descriptors fails clearly and leaves the original document journal and
    committed file for recovery, matching the existing one-descriptor layout.
    CSV subclasses retain their existing export implementation.
    """

    @property
    def incremental_enabled(self):
        return getattr(self, "selected_file_type", None) != "csv"

    def clear(self):
        self.close()
        super().clear()
        self.live_file = None
        self._temporary_file = None
        self._h5 = None
        self._frozen = False
        self._stream_specs = {}
        self._event_indices = []

    def close(self):
        handle = getattr(self, "_h5", None)
        if handle is not None:
            handle.close()
        self._h5 = None
        self.root = None

    def descriptor(self, doc):
        if not self.incremental_enabled:
            return super().descriptor(doc)
        if self.streams.get(doc["name"]):
            raise ValueError(f"Changing descriptors for {doc['name']} is unsupported by this NeXus layout")
        if self._frozen:
            raise ValueError("SWMR cannot add a stream or field after acquisition starts")
        super().descriptor(doc)
        self._stream_specs[doc["name"]] = doc["data_keys"]
        if self._h5 is not None:
            write_json_dataset(self._h5[f"{DOCUMENTS}/descriptors"], list(self.descriptor_documents.values()))

    def _json_node(self, parent, name, value):
        dataset = parent.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(4096,), dtype="u1")
        dataset.attrs["mime_type"] = "application/json"
        dataset.attrs["encoding"] = "utf-8"
        write_json_dataset(dataset, value)
        return dataset

    def _open_file(self):
        destination = Path(self.file_name)
        if destination.exists():
            raise FileExistsError(f"Refusing to replace an existing export: {destination}")
        self._temporary_file = destination.with_name(f"{destination.name}.{uuid4().hex}.partial")
        self._h5 = h5py.File(self._temporary_file, "x", libver="latest")
        self.root = self._h5
        self._building_incremental = True
        try:
            self.write_root(destination)
        finally:
            self._building_incremental = False

    def write_root(self, filename):
        # Existing offline layout helpers can still build a completed export.
        if not getattr(self, "_building_incremental", False):
            return super().write_root(filename)
        root = self._h5
        self.metadata["storage_schema_version"] = "2.0"
        root.attrs.update(default="entry", file_name=str(filename), creator=type(self).__name__,
                          diffractometer_schema_version="2.0", HDF5_Version=h5py.version.hdf5_version,
                          h5py_version=h5py.version.version, NeXus_release=self.nexus_release,
                          file_time=dt.datetime.now(dt.timezone.utc).isoformat(), instrument=self.instrument_name or "")
        entry = self.create_NX_group(root, "entry:NXentry")
        entry.create_dataset("start_time", data=dt.datetime.fromtimestamp(self.start_time, dt.timezone.utc).isoformat())
        entry.create_dataset("end_time", data=b"", dtype="S64")
        entry.create_dataset("duration", data=0.0).attrs["units"] = "s"
        entry.create_dataset("program_name", data="bluesky")
        instrument = self.create_NX_group(entry, "instrument:NXinstrument")
        instrument.attrs["name"] = self.instrument_name or ""
        bluesky = self.create_NX_group(instrument, "bluesky:NXnote")
        metadata = self.create_NX_group(bluesky, "metadata:NXnote")
        metadata.create_dataset("run_start_uid", data=self.uid)
        for key, value in self.metadata.items():
            if key == "baseline_readings":
                continue
            if isinstance(value, (dict, list, tuple)):
                plain = json.loads(json.dumps(value, default=json_default))
                dataset = metadata.create_dataset(key, data=yaml.safe_dump(plain, default_flow_style=False))
                dataset.attrs["text_format"] = "yaml"
            else:
                metadata.create_dataset(key, data=value if value is not None else "")
        self._json_node(metadata, "baseline_readings", {})
        bluesky["uid"] = metadata["run_start_uid"]
        documents = self.create_NX_group(bluesky, "documents:NXcollection")
        documents.attrs["schema_version"] = "2.0"
        documents.create_dataset("start", data=json.dumps(self.start_document, default=json_default))
        self._json_node(documents, "descriptors", list(self.descriptor_documents.values()))
        self._json_node(documents, "stop", None)
        self.create_NX_group(bluesky, "streams:NXcollection")
        storage = self.create_NX_group(bluesky, "storage:NXcollection")
        storage.create_dataset("state", data=b"collecting", dtype="S32")
        entry["entry_identifier"] = metadata["run_start_uid"]
        entry.create_dataset("title", data=str(self.metadata.get("title", "")))
        entry.create_dataset("plan_name", data=self.plan_name or "")
        sample = self.create_NX_group(entry, "sample:NXsample")
        sample.create_dataset("name", data=str(self.metadata.get("sample", "")))

    def _ensure_stream(self, uid, event=None):
        descriptor = self.descriptor_documents[uid]
        stream = descriptor["name"]
        group = self._h5[STREAMS].require_group(stream)
        if "committed" not in group:
            group.attrs.update(NX_class="NXcollection", uid=uid)
            group.create_dataset("committed", data=np.int64(0))
            for name, dtype in (("seq_num", "i8"), ("event_time", "f8")):
                group.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1024,), dtype=dtype)
        for key, spec in descriptor["data_keys"].items():
            if spec.get("external"):
                raise ValueError(f"External input {stream}/{key} requires its own file handler")
            entry = self.acquisitions[uid]["data"][key]
            if key not in group:
                field = group.create_group(key)
                field.attrs.update(NX_class="NXdata", signal="value", axes=["time"])
                if event is not None:
                    array = np.asarray(event["data"][key])
                    dtype = array.dtype
                else:
                    dtype = np.dtype(spec.get("dtype_numpy") or {
                        "integer": "i8", "number": "f8", "boolean": "?", "string": "S65536",
                    }.get(spec["dtype"], "f8"))
                if dtype.kind in "OUS":
                    dtype = np.dtype("S65536")
                shape = tuple(spec.get("shape", []))
                if any(size <= 0 for size in shape):
                    raise ValueError(f"Fixed, positive field dimensions are required for {key}")
                dataset = field.create_dataset("value", shape=(0, *shape), maxshape=(None, *shape),
                                               chunks=(1, *shape), dtype=dtype)
                self.add_dataset_attributes(dataset, entry, long_name=key)
                dataset.attrs["target"] = dataset.name
                for name in ("EPOCH", "time"):
                    field.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(1024,), dtype="f8").attrs["units"] = "s"
                if stream == "baseline":
                    for name in ("value_start", "value_end"):
                        field.create_dataset(name, shape=shape, dtype=dtype)
            entry["data"] = DatasetSequence(group[key]["value"])
            entry["time"] = DatasetSequence(group[key]["EPOCH"])
        return group

    def _create_live_data(self):
        primary = self._h5.get(f"{STREAMS}/primary")
        if primary is None:
            return
        entry = self._h5["entry"]
        fields = list(self._stream_specs["primary"])
        profiles = [key for key in fields if key.endswith("_counts") and not key.endswith("_total_counts")
                    and primary[key]["value"].ndim == 2]
        if profiles:
            for index, key in enumerate(profiles):
                group = self.create_NX_group(entry, f"{'data' if index == 0 else 'data_' + key[:-7]}:NXdata")
                group["counts"] = primary[key]["value"]
                bins = primary[key]["value"].shape[-1]
                group.create_dataset("position_x", data=np.linspace(HE3PSD_POSITION_MIN, HE3PSD_POSITION_MAX, bins)).attrs["units"] = "mm"
                group.attrs.update(signal="counts", axes=["event_index", "position_x"])
                dataset = group.create_dataset("event_index", shape=(0,), maxshape=(None,), chunks=(1024,), dtype="i8")
                self._event_indices.append(dataset)
                total = key[:-7] + "_total_counts"
                if total in primary:
                    group["total_counts"] = primary[total]["value"]
                group["EPOCH"] = primary[key]["EPOCH"]
                for motor in self.positioners:
                    if motor in primary:
                        group[motor] = primary[motor]["value"]
        else:
            group = self.create_NX_group(entry, "data:NXdata")
            for key in fields:
                group[key] = primary[key]["value"]
            signal = next((key for key in fields if key not in self.positioners), fields[0] if fields else "")
            group.attrs["signal"] = signal
            if self.positioners and self.positioners[0] in fields:
                group.attrs["axes"] = [self.positioners[0]]
        entry.attrs["default"] = "data"

    def _sync(self):
        self._h5.flush()
        # The default sec2 VFD returns an OS descriptor. Refuse to claim
        # durability if a different driver cannot provide one.
        os.fsync(self._h5.id.get_vfd_handle())

    def event(self, doc):
        if not self.incremental_enabled:
            return super().event(doc)
        if not self.scanning:
            return
        if self._h5 is None:
            self._open_file()
        group = self._ensure_stream(doc["descriptor"], doc)
        index = int(group["committed"][()])
        # Validate the whole point before changing extents; no silent truncation
        # or integer overflow is allowed when a device changes its dtype.
        values = {}
        for key, value in doc["data"].items():
            dataset = group[key]["value"]
            array = np.asarray(value)
            if dataset.dtype.kind == "S":
                array = np.asarray(str(value).encode("utf-8"))
            if array.shape != dataset.shape[1:] or not np.can_cast(array.dtype, dataset.dtype, casting="safe"):
                raise ValueError(f"Shape or dtype changed for {key}: {array.shape}/{array.dtype}")
            values[key] = array
        if set(values) != set(self.descriptor_documents[doc["descriptor"]]["data_keys"]):
            raise ValueError("Event fields do not match the declared schema")
        for key, value in values.items():
            field = group[key]
            for name, item in (("value", value), ("EPOCH", doc["timestamps"][key]),
                               ("time", doc["timestamps"][key] - (field["EPOCH"][0] if index else doc["timestamps"][key]))):
                dataset = field[name]
                dataset.resize((index + 1, *dataset.shape[1:]))
                dataset[index] = item
            if "value_start" in field:
                if index == 0:
                    field["value_start"][()] = value
                field["value_end"][()] = value
        for name, item in (("seq_num", doc["seq_num"]), ("event_time", doc["time"])):
            group[name].resize((index + 1,))
            group[name][index] = item
        if group.name.endswith("/primary"):
            if not self._frozen:
                for uid in self.descriptor_documents:
                    self._ensure_stream(uid)
                self._create_live_data()
                self._h5.swmr_mode = True
                self._frozen = True
            for dataset in self._event_indices:
                dataset.resize((index + 1,))
                dataset[index] = index
        self._sync()
        group["committed"][()] = index + 1
        self._sync()
        if self._frozen and self.live_file is None:
            # Publish only once the schema is frozen and the first full point
            # is committed. The path remains stable for every Tiled request.
            destination = Path(self.file_name)
            os.link(self._temporary_file, destination)
            self._temporary_file.unlink()
            self._sync_directory(destination.parent)
            self.live_file = destination

    @staticmethod
    def _sync_directory(path):
        if os.name == "posix":
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def writer(self):
        if not self.incremental_enabled or self._h5 is None:
            return super().writer()
        try:
            root = self._h5
            root["entry/end_time"][()] = dt.datetime.fromtimestamp(self.stop_time, dt.timezone.utc).isoformat().encode()
            root["entry/duration"][()] = self.stop_time - self.start_time
            write_json_dataset(root[f"{DOCUMENTS}/stop"], self.stop_document)
            write_json_dataset(root["/entry/instrument/bluesky/metadata/baseline_readings"], self.metadata.get("baseline_readings", {}))
            self._sync()
            path = self.live_file or self._temporary_file
            self.validate_file(path)
            root["/entry/instrument/bluesky/storage/state"][()] = b"complete"
            self._sync()
            if self.live_file is None:
                os.link(path, self.file_name)
                path.unlink()
                self._sync_directory(Path(self.file_name).parent)
        finally:
            self.close()
        # Closing SWMR also clears writer flags in the superblock. Sync those
        # final bytes before computing the artifact checksum or declaring ready.
        with Path(self.file_name).open("rb") as stream:
            os.fsync(stream.fileno())
        self.output_nexus_file = Path(self.file_name)

    def validate_file(self, path):
        if self._h5 is None:
            return super().validate_file(path)
        if self._h5["/entry/instrument/bluesky/metadata/run_start_uid"].asstr()[()] != self.uid:
            raise ValueError("NeXus run UID does not match the acquisition")
        for stream, specs in self._stream_specs.items():
            group = self._h5.get(f"{STREAMS}/{stream}")
            if group is None:
                continue
            count = int(group["committed"][()])
            for key, spec in specs.items():
                if group[key]["value"].shape != (count, *spec.get("shape", [])):
                    raise ValueError(f"Uncommitted NeXus tail for {stream}/{key}")
