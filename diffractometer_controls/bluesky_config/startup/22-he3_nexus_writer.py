import datetime as dt
import pathlib
import re

import numpy as np

from apstools.callbacks.nexus_writer import NXWriter


HE3_DIFFRACTION_ROOT = pathlib.Path("/home/mitr_4dh4/Data/Diffraction")
HE3PSD_POSITION_MIN = -209.21799055746422
HE3PSD_POSITION_MAX = 209.21799055746422
TEST_CATALOG_NAMES = {"testdb"}
HE3_PLAN_NAMES = {
    "count_he3",
    "scan_he3",
    "scan_parallel_he3",
    "scan_list_he3",
    "scan2D_he3",
}


def _sanitize_filename_part(value):
    text = str(value or "").strip()
    if not text:
        return "run"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._-") or "run"


def _sanitize_optional_title(value, *, max_length=48):
    text = str(value or "").strip()
    if not text:
        return ""
    text = _sanitize_filename_part(text)
    return text[:max_length].rstrip("._-")


def _normalize_detector_names(detectors):
    return [str(det).strip() for det in (detectors or []) if str(det).strip()]


class HE3DiffractionNXWriter(NXWriter):
    instrument_name = "4DH4"
    file_extension = "nxs"
    output_root = HE3_DIFFRACTION_ROOT

    def clear(self):
        super().clear()
        self._file_name = None
        self._file_path = None
        self.output_nexus_file = None

    def _catalog_name(self):
        return str(self.metadata.get("bluesky_catalog", "") or "").strip()

    def _is_test_catalog(self):
        return self._catalog_name() in TEST_CATALOG_NAMES

    def _output_root_for_run(self):
        output_root = pathlib.Path(self.output_root)
        if self._is_test_catalog():
            return output_root / "Test"
        return output_root

    def _is_supported_run(self, doc):
        experiment_type = str(doc.get("experiment_type", "") or "").strip().lower()
        if experiment_type != "diffraction":
            return False

        detector_names = _normalize_detector_names(doc.get("detectors", []))
        if any(name.startswith("he3psd") for name in detector_names):
            return True

        plan_args = doc.get("plan_args", {}) or {}
        plan_detectors = _normalize_detector_names(plan_args.get("detectors", []))
        if any("he3psd" in name.lower() for name in plan_detectors):
            return True

        if detector_names or plan_detectors:
            return False

        plan_name = str(doc.get("plan_name", "") or "").strip()
        return plan_name in HE3_PLAN_NAMES

    def _get_primary_descriptor(self):
        primary_uids = list(self.streams.get("primary", []) or [])
        if len(primary_uids) != 1:
            return None
        primary_uid = primary_uids[0]
        return self.acquisitions.get(primary_uid, {}).get("data")

    def _get_primary_detector_fields(self, primary):
        detector_names = _normalize_detector_names(self.detectors)
        for detector_name in detector_names:
            if not detector_name.startswith("he3psd"):
                continue
            counts_key = f"{detector_name}_counts"
            if counts_key not in primary:
                continue
            return {
                "detector_name": detector_name,
                "counts": counts_key,
                "position_x": f"{detector_name}_position_x",
                "total_counts": f"{detector_name}_total_counts",
            }

        for field_name in primary:
            if not str(field_name).endswith("_counts"):
                continue
            detector_name = str(field_name)[: -len("_counts")]
            if detector_name.startswith("he3psd"):
                return {
                    "detector_name": detector_name,
                    "counts": str(field_name),
                    "position_x": f"{detector_name}_position_x",
                    "total_counts": f"{detector_name}_total_counts",
                }

        return None

    @staticmethod
    def _stack_field(entry):
        return np.asarray(list(entry.get("data", [])))

    @staticmethod
    def _collapse_invariant_axis(position_data):
        arr = np.asarray(position_data)
        if arr.ndim <= 1:
            return arr
        if arr.shape[0] == 0:
            return arr
        reference = arr[0]
        try:
            if np.allclose(arr, reference, equal_nan=True):
                return np.asarray(reference)
        except Exception:
            pass
        return arr

    def _create_dataset(self, parent, dataset_name, data, entry=None, long_name=None):
        ds = parent.create_dataset(dataset_name, data=data)
        ds.attrs["target"] = ds.name
        if entry is not None:
            self.add_dataset_attributes(ds, entry, long_name=long_name or dataset_name)
        elif long_name is not None:
            ds.attrs["long_name"] = self.h5string(long_name)
        return ds

    def _get_expected_nbins(self, counts_data):
        det_config = self.metadata.get("det_config", {}) or {}
        try:
            nbins = int(round(float(det_config.get("nbins"))))
            if nbins > 0:
                return nbins
        except Exception:
            pass

        arr = np.asarray(counts_data)
        if arr.ndim == 0:
            return 0
        return int(arr.shape[-1])

    def _build_fixed_position_axis(self, counts_data):
        nbins = self._get_expected_nbins(counts_data)
        if nbins <= 0:
            return None
        return np.linspace(HE3PSD_POSITION_MIN, HE3PSD_POSITION_MAX, nbins)

    def start(self, doc):
        if not self._is_supported_run(doc):
            self.clear()
            return

        super().start(doc)
        self.file_name = self.make_file_name()
        self.file_name.parent.mkdir(parents=True, exist_ok=True)
        self.metadata["nexus_file"] = str(self.file_name)

    def make_file_name(self):
        start_time = dt.datetime.fromtimestamp(self.start_time)
        file_dir = self._output_root_for_run() / start_time.strftime("%Y")
        scan_id = int(self.scan_id or 0)
        plan_name = _sanitize_filename_part(self.plan_name or "run")
        title_slug = _sanitize_optional_title(self.metadata.get("title", ""))
        parts = [
            start_time.strftime("%Y%m%d-%H%M%S"),
            f"S{scan_id:05d}",
            plan_name,
        ]
        if title_slug:
            parts.append(title_slug)
        parts.append(self.uid[:7])
        file_name = "-".join(parts) + f".{self.file_extension}"
        if self._is_test_catalog():
            file_name = f"test-{file_name}"
        return file_dir / file_name

    def write_data(self, parent):
        primary = self._get_primary_descriptor()
        if not primary:
            return super().write_data(parent)

        detector_fields = self._get_primary_detector_fields(primary)
        if not detector_fields:
            return super().write_data(parent)

        counts_key = detector_fields["counts"]
        position_key = detector_fields["position_x"]
        total_counts_key = detector_fields["total_counts"]

        counts_entry = primary.get(counts_key)
        if counts_entry is None:
            return super().write_data(parent)

        counts_stack = self._stack_field(counts_entry)
        if counts_stack.size == 0:
            return super().write_data(parent)

        position_entry = primary.get(position_key)
        total_counts_entry = primary.get(total_counts_key)

        nxdata = self.create_NX_group(parent, "data:NXdata")
        axes = []

        num_events = len(counts_entry.get("data", []))
        if num_events <= 1:
            counts_out = np.asarray(counts_entry["data"][0])
            self._create_dataset(nxdata, "counts", counts_out, counts_entry, long_name=counts_key)

            position_out = self._build_fixed_position_axis(counts_out)
            if position_out is None and position_entry is not None and position_entry.get("data"):
                position_out = np.asarray(position_entry["data"][0])
            if position_out is not None:
                self._create_dataset(
                    nxdata,
                    "position_x",
                    position_out,
                    position_entry,
                    long_name=position_key,
                )
                axes = ["position_x"]

            if total_counts_entry is not None and total_counts_entry.get("data"):
                total_counts_out = np.asarray(total_counts_entry["data"][0])
                self._create_dataset(
                    nxdata,
                    "total_counts",
                    total_counts_out,
                    total_counts_entry,
                    long_name=total_counts_key,
                )
        else:
            counts_out = counts_stack
            self._create_dataset(nxdata, "counts", counts_out, counts_entry, long_name=counts_key)

            event_index = np.arange(num_events, dtype=np.int64)
            event_index_ds = self._create_dataset(
                nxdata,
                "event_index",
                event_index,
                entry=None,
                long_name="event index",
            )
            event_index_ds.attrs["units"] = self.h5string("")
            axes = ["event_index"]

            position_out = self._build_fixed_position_axis(counts_out)
            if position_out is None and position_entry is not None and position_entry.get("data"):
                position_out = self._collapse_invariant_axis(self._stack_field(position_entry))
            if position_out is not None:
                self._create_dataset(
                    nxdata,
                    "position_x",
                    position_out,
                    position_entry,
                    long_name=position_key,
                )
                axes.append("position_x")

            if total_counts_entry is not None and total_counts_entry.get("data"):
                total_counts_out = np.asarray(total_counts_entry["data"])
                self._create_dataset(
                    nxdata,
                    "total_counts",
                    total_counts_out,
                    total_counts_entry,
                    long_name=total_counts_key,
                )

            motor_names = []
            motor_names.extend(list(self.data_map.get("motors", [])))
            for name in _normalize_detector_names(self.positioners):
                if name not in motor_names:
                    motor_names.append(name)

            for motor_name in motor_names:
                if motor_name in {"event_index", "position_x"}:
                    continue
                entry = primary.get(motor_name)
                if entry is None or not entry.get("data"):
                    continue
                motor_values = np.asarray(entry["data"])
                self._create_dataset(nxdata, motor_name, motor_values, entry, long_name=motor_name)

        epoch_values = np.asarray(counts_entry.get("time", []))
        if epoch_values.size > 0:
            epoch_out = epoch_values[0] if num_events <= 1 else epoch_values
            epoch_ds = self._create_dataset(nxdata, "EPOCH", epoch_out, entry=None, long_name="epoch time")
            epoch_ds.attrs["units"] = self.h5string("s")

        nxdata.attrs["signal"] = "counts"
        if axes:
            nxdata.attrs["axes"] = axes
        return nxdata


he3_nexus_writer = None
he3_nexus_writer_subscription = None

if "RE" in globals():
    he3_nexus_writer = HE3DiffractionNXWriter()
    he3_nexus_writer_subscription = RE.subscribe(he3_nexus_writer.receiver)
