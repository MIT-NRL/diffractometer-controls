"""Writer definitions; startup owns construction and RunEngine subscriptions."""
import csv
import datetime as dt
import json
import os
import pathlib
import re

import numpy as np

from apstools.callbacks.nexus_writer import NXWriter


SCALAR_DIFFRACTION_ROOT = pathlib.Path("/home/mitr_4dh4/Data/Diffraction")
SCALAR_DIFFRACTION_TEST_ROOT = pathlib.Path(
    "/home/mitr_4dh4/Data/TestData/Diffraction"
)
SCALAR_FILE_TYPES = {"csv", "nexus"}
SCALAR_TEST_CATALOG_NAMES = {"testdb"}


def _scalar_sanitize_filename_part(value):
    text = str(value or "").strip()
    if not text:
        return "run"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-") or "run"


def _scalar_sanitize_optional_title(value, *, max_length=48):
    text = str(value or "").strip()
    if not text:
        return ""
    return _scalar_sanitize_filename_part(text)[:max_length].rstrip("._-")


def _scalar_json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _scalar_metadata_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class ScalarDiffractionWriter(NXWriter):
    """Write scalar diffraction primary data as selected CSV or NeXus."""

    instrument_name = "4DH4"
    output_root = SCALAR_DIFFRACTION_ROOT
    test_output_root = SCALAR_DIFFRACTION_TEST_ROOT

    def clear(self):
        super().clear()
        self.selected_file_type = None
        self._file_name = None
        self._file_path = None
        self.output_csv_file = None
        self.output_nexus_file = None

    def _catalog_name(self):
        return str(self.metadata.get("bluesky_catalog", "") or "").strip()

    def _is_test_catalog(self):
        return self._catalog_name() in SCALAR_TEST_CATALOG_NAMES

    def _output_root_for_run(self):
        if self._is_test_catalog():
            return pathlib.Path(self.test_output_root)
        return pathlib.Path(self.output_root)

    @staticmethod
    def _requested_file_type(doc):
        return str(doc.get("file_type", "") or "").strip().lower()

    def _is_supported_run(self, doc):
        return (
            str(doc.get("experiment_type", "") or "").strip().lower()
            == "diffraction"
            and str(doc.get("data_type", "") or "").strip().lower()
            == "scalar"
            and self._requested_file_type(doc) in SCALAR_FILE_TYPES
        )

    def start(self, doc):
        if not self._is_supported_run(doc):
            self.clear()
            return

        selected_file_type = self._requested_file_type(doc)
        super().start(doc)
        self.selected_file_type = selected_file_type
        self.file_extension = "csv" if selected_file_type == "csv" else "nxs"
        self.file_name = self.make_file_name()
        self.file_name.parent.mkdir(parents=True, exist_ok=True)
        metadata_key = "csv_file" if selected_file_type == "csv" else "nexus_file"
        self.metadata[metadata_key] = str(self.file_name)

    def make_file_name(self):
        start_time = dt.datetime.fromtimestamp(self.start_time)
        directory = self._output_root_for_run() / start_time.strftime("%Y")
        parts = [
            start_time.strftime("%Y%m%d-%H%M%S"),
            f"S{int(self.scan_id or 0):05d}",
            _scalar_sanitize_filename_part(self.plan_name or "scalar"),
        ]
        title = _scalar_sanitize_optional_title(self.metadata.get("title", ""))
        if title:
            parts.append(title)
        parts.append(str(self.uid)[:7])
        filename = "-".join(parts) + f".{self.file_extension}"
        if self._is_test_catalog():
            filename = f"test-{filename}"
        return directory / filename

    def writer(self):
        self._capture_baseline_metadata()
        if self.selected_file_type == "csv":
            self._write_csv()
        elif self.selected_file_type == "nexus":
            super().writer()

    def _capture_baseline_metadata(self):
        readings = {}
        for descriptor_uid in list(self.streams.get("baseline", []) or []):
            descriptor = dict(self.acquisitions.get(descriptor_uid, {}) or {})
            for data_key, entry in dict(descriptor.get("data", {}) or {}).items():
                data_key = str(data_key)
                entry = dict(entry or {})
                values = list(entry.get("data", []) or [])
                if not values:
                    continue
                readings[data_key] = {
                    "start": _scalar_metadata_value(values[0]),
                    "end": _scalar_metadata_value(values[-1]),
                    "units": str(entry.get("units", "") or ""),
                }
        if readings:
            self.metadata["baseline_readings"] = readings
        return readings

    def write_metadata(self, parent):
        self._capture_baseline_metadata()
        return super().write_metadata(parent)

    @staticmethod
    def _csv_value(value):
        if value is None:
            return ""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple, dict)):
            return json.dumps(value, default=_scalar_json_default, separators=(",", ":"))
        return value

    def _primary_entries(self):
        entries = {}
        for descriptor_uid in list(self.streams.get("primary", []) or []):
            descriptor = dict(self.acquisitions.get(descriptor_uid, {}) or {})
            for key, entry in dict(descriptor.get("data", {}) or {}).items():
                entries[str(key)] = entry
        return entries

    def _header_metadata(self):
        values = {
            "format": "MITR scalar CSV v1",
            "uid": self.uid,
            "scan_id": self.scan_id,
            "plan_name": self.plan_name,
            "title": self.metadata.get("title", ""),
            "sample": self.metadata.get("sample", ""),
            "detectors": self.detectors,
            "motors": self.positioners,
            "start_time": dt.datetime.fromtimestamp(
                self.start_time, tz=dt.timezone.utc
            ).isoformat(),
            "stop_time": dt.datetime.fromtimestamp(
                self.stop_time, tz=dt.timezone.utc
            ).isoformat(),
            "exit_status": self.exit_status,
            "plan_args": self.metadata.get("plan_args", {}),
            "det_config": self.metadata.get("det_config", {}),
            "baseline_readings": self.metadata.get("baseline_readings", {}),
        }
        return values

    @staticmethod
    def _metadata_cell(value):
        if value is None:
            return ""
        value = _scalar_metadata_value(value)
        if isinstance(value, (list, tuple)):
            return "; ".join(str(_scalar_metadata_value(item)) for item in value)
        return value

    @classmethod
    def _flatten_metadata_rows(cls, prefix, value):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                nested_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from cls._flatten_metadata_rows(nested_prefix, nested_value)
            return
        yield [f"# {prefix}", cls._metadata_cell(value), "", ""]

    def _write_csv(self):
        entries = self._primary_entries()
        columns = list(entries)
        row_count = max(
            (len(dict(entry or {}).get("data", []) or []) for entry in entries.values()),
            default=0,
        )
        output_path = pathlib.Path(self.file_name)
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                metadata = self._header_metadata()
                baseline_readings = dict(metadata.pop("baseline_readings", {}) or {})
                reactor_power_readings = {
                    key: value
                    for key, value in baseline_readings.items()
                    if str(key).startswith("reactor_power")
                }
                other_baseline_readings = {
                    key: value
                    for key, value in baseline_readings.items()
                    if key not in reactor_power_readings
                }
                for key, value in metadata.items():
                    writer.writerows(self._flatten_metadata_rows(str(key), value))
                writer.writerow(["# MITR scalar metadata", "Start", "End", "Units"])
                for data_key, reading in reactor_power_readings.items():
                    reading = dict(reading or {})
                    writer.writerow(
                        [
                            f"# baseline.{data_key}",
                            self._metadata_cell(reading.get("start")),
                            self._metadata_cell(reading.get("end")),
                            self._metadata_cell(reading.get("units")),
                        ]
                    )
                for data_key, reading in other_baseline_readings.items():
                    reading = dict(reading or {})
                    writer.writerow(
                        [
                            f"# baseline.{data_key}",
                            self._metadata_cell(reading.get("start")),
                            self._metadata_cell(reading.get("end")),
                            self._metadata_cell(reading.get("units")),
                        ]
                    )
                writer.writerow([])
                writer.writerow(["sequence", "time", *columns])
                for index in range(row_count):
                    event_time = None
                    for entry in entries.values():
                        timestamps = list(dict(entry or {}).get("time", []) or [])
                        if index < len(timestamps):
                            event_time = timestamps[index]
                            break
                    row = [index + 1, event_time]
                    for column in columns:
                        data = list(dict(entries[column] or {}).get("data", []) or [])
                        value = data[index] if index < len(data) else None
                        row.append(self._csv_value(value))
                    writer.writerow(row)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        self.output_csv_file = output_path
