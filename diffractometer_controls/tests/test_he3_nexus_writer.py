import importlib.util
import datetime
import pathlib
import tempfile
import unittest

import h5py
import numpy as np


STARTUP_FILE = pathlib.Path(__file__).resolve().parents[1] / "bluesky_config" / "startup" / "22-he3_nexus_writer.py"


def _load_writer_module():
    spec = importlib.util.spec_from_file_location("he3_nexus_writer_startup", STARTUP_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _normalize_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray):
        return [_normalize_attr(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_normalize_attr(item) for item in value]
    return value


def _build_start_doc(
    *,
    plan_name="count_he3",
    experiment_type="diffraction",
    detectors=None,
    motors=None,
    when=1715798400.0,
    bluesky_catalog="4dh4",
):
    return {
        "uid": "1234567890abcdef",
        "time": float(when),
        "scan_id": 17,
        "plan_name": str(plan_name),
        "experiment_type": str(experiment_type),
        "detectors": list(detectors or ["he3psd0"]),
        "motors": list(motors or []),
        "bluesky_catalog": str(bluesky_catalog),
        "title": "HE3 test",
        "sample": "sample-a",
        "gauge_volume": "gv-1",
    }


def _build_descriptor(*, start_uid, detector_name="he3psd0", motors=None):
    data_keys = {
        f"{detector_name}_counts": {
            "source": f"SIM:{detector_name}:COUNTS",
            "dtype": "array",
            "shape": [5],
            "units": "counts",
            "precision": 3,
            "object_name": detector_name,
        },
        f"{detector_name}_position_x": {
            "source": f"SIM:{detector_name}:POSITION",
            "dtype": "array",
            "shape": [5],
            "units": "mm",
            "precision": 6,
            "object_name": detector_name,
        },
        f"{detector_name}_total_counts": {
            "source": f"SIM:{detector_name}:TOTAL",
            "dtype": "number",
            "shape": [],
            "units": "counts",
            "precision": 3,
            "object_name": detector_name,
        },
    }
    object_keys = {
        detector_name: [
            f"{detector_name}_counts",
            f"{detector_name}_position_x",
            f"{detector_name}_total_counts",
        ]
    }
    for motor_name in list(motors or []):
        data_keys[motor_name] = {
            "source": f"SIM:{motor_name}",
            "dtype": "number",
            "shape": [],
            "units": "deg",
            "precision": 3,
            "object_name": motor_name,
        }
        object_keys[motor_name] = [motor_name]
    return {
        "uid": "descriptor-primary",
        "run_start": start_uid,
        "time": 1715798400.5,
        "name": "primary",
        "data_keys": data_keys,
        "object_keys": object_keys,
    }


def _build_event(*, descriptor_uid, seq_num, counts, position_x, total_counts, time_value, motors=None):
    data = {
        "he3psd0_counts": np.asarray(counts),
        "he3psd0_position_x": np.asarray(position_x),
        "he3psd0_total_counts": float(total_counts),
    }
    timestamps = {
        "he3psd0_counts": float(time_value),
        "he3psd0_position_x": float(time_value),
        "he3psd0_total_counts": float(time_value),
    }
    for motor_name, motor_value in dict(motors or {}).items():
        data[motor_name] = float(motor_value)
        timestamps[motor_name] = float(time_value)
    return {
        "uid": f"event-{seq_num}",
        "time": float(time_value),
        "seq_num": int(seq_num),
        "descriptor": descriptor_uid,
        "data": data,
        "timestamps": timestamps,
    }


class HE3DiffractionNXWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_writer_module()

    def _new_writer(self, root_path):
        writer = self.module.HE3DiffractionNXWriter()
        writer.output_root = pathlib.Path(root_path)
        return writer

    def _prepare_writer(self, writer, start_doc, descriptor_doc, events):
        writer.start(start_doc)
        primary_data = {}
        for key, entry in descriptor_doc["data_keys"].items():
            primary_data[key] = {
                "source": entry.get("source", "local"),
                "dtype": entry.get("dtype", ""),
                "shape": entry.get("shape", []),
                "units": entry.get("units", ""),
                "lower_ctrl_limit": entry.get("lower_ctrl_limit", ""),
                "upper_ctrl_limit": entry.get("upper_ctrl_limit", ""),
                "precision": entry.get("precision", 0),
                "object_name": entry.get("object_name", key),
                "data": [],
                "time": [],
                "external": entry.get("external") is not None,
            }
        for event_doc in events:
            for key, value in event_doc["data"].items():
                primary_data[key]["data"].append(value)
                primary_data[key]["time"].append(event_doc["timestamps"][key])
        writer.streams["primary"] = [descriptor_doc["uid"]]
        writer.acquisitions[descriptor_doc["uid"]] = {"stream": "primary", "data": primary_data}
        writer.data_map = {
            "detectors": list(descriptor_doc["object_keys"].get("he3psd0", [])),
            "motors": list(start_doc.get("motors", [])),
            "unassigned": [],
        }
        writer.stop_time = start_doc["time"] + 1.0
        writer.stop_reason = ""
        writer.exit_status = "success"
        writer.scanning = False

    def _write_file(self, writer):
        file_name = pathlib.Path(writer.file_name)
        with h5py.File(file_name, "w") as writer.root:
            writer.write_root(file_name)
        writer.root = None
        return file_name

    def test_accepts_only_he3_diffraction_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            writer.start(_build_start_doc(experiment_type="imaging"))
            self.assertFalse(writer.scanning)
            self.assertIsNone(writer.file_name)

            writer.start(_build_start_doc(detectors=["cam1"]))
            self.assertFalse(writer.scanning)
            self.assertIsNone(writer.file_name)

            writer.start(_build_start_doc())
            self.assertTrue(writer.scanning)
            self.assertIsNotNone(writer.file_name)

    def test_accepts_scan_he3_when_detector_names_only_exist_in_plan_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(plan_name="scan_he3", detectors=[])
            start_doc["detectors"] = None
            start_doc["plan_args"] = {
                "detectors": [
                    "HE3PSD(prefix='4dh4:he3PSD:', name='he3psd7', read_attrs=['position_x', 'counts'])"
                ]
            }
            writer.start(start_doc)
            self.assertTrue(writer.scanning)
            self.assertIsNotNone(writer.file_name)

    def test_filename_generation_uses_diffraction_root_and_readable_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(when=1715798400.0)
            writer.start(start_doc)
            file_name = pathlib.Path(writer.file_name)
            expected_prefix = datetime.datetime.fromtimestamp(start_doc["time"]).strftime("%Y%m%d-%H%M%S")
            self.assertEqual(file_name.parent, pathlib.Path(tmpdir) / "2024")
            self.assertEqual(file_name.name, f"{expected_prefix}-S00017-count_he3-HE3_test-1234567.nxs")
            self.assertEqual(writer.metadata["nexus_file"], str(file_name))

    def test_filename_omits_title_slug_when_title_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc()
            start_doc["title"] = ""
            writer.start(start_doc)
            file_name = pathlib.Path(writer.file_name)
            expected_prefix = datetime.datetime.fromtimestamp(start_doc["time"]).strftime("%Y%m%d-%H%M%S")
            self.assertEqual(file_name.name, f"{expected_prefix}-S00017-count_he3-1234567.nxs")

    def test_test_catalog_uses_test_root_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(when=1715798400.0, bluesky_catalog="testdb")
            writer.start(start_doc)
            file_name = pathlib.Path(writer.file_name)
            expected_prefix = datetime.datetime.fromtimestamp(start_doc["time"]).strftime("%Y%m%d-%H%M%S")
            self.assertEqual(file_name.parent, pathlib.Path(tmpdir) / "Test" / "2024")
            self.assertEqual(
                file_name.name,
                f"test-{expected_prefix}-S00017-count_he3-HE3_test-1234567.nxs",
            )
            self.assertEqual(writer.metadata["nexus_file"], str(file_name))

    def test_count_he3_single_event_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(plan_name="count_he3")
            descriptor_doc = _build_descriptor(start_uid=start_doc["uid"])
            event_doc = _build_event(
                descriptor_uid=descriptor_doc["uid"],
                seq_num=1,
                counts=[1, 2, 3, 4, 5],
                position_x=[-2, -1, 0, 1, 2],
                total_counts=15,
                time_value=start_doc["time"] + 0.1,
            )
            self._prepare_writer(writer, start_doc, descriptor_doc, [event_doc])
            file_name = self._write_file(writer)

            with h5py.File(file_name, "r") as handle:
                self.assertEqual(handle["/entry/data/counts"].shape, (5,))
                self.assertEqual(handle["/entry/data/position_x"].shape, (5,))
                self.assertNotIn("event_index", handle["/entry/data"])
                self.assertEqual(_normalize_attr(handle["/entry/data"].attrs["signal"]), "counts")
                self.assertEqual(_normalize_attr(handle["/entry/data"].attrs["axes"]), ["position_x"])
                self.assertIn("/entry/instrument/bluesky/streams/primary/he3psd0_counts/value", handle)
                self.assertEqual(
                    _normalize_attr(handle["/entry/instrument/bluesky/metadata/nexus_file"][()]),
                    str(file_name),
                )

    def test_curated_position_axis_is_generated_from_fixed_geometry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(plan_name="count_he3")
            descriptor_doc = _build_descriptor(start_uid=start_doc["uid"])
            event_doc = _build_event(
                descriptor_uid=descriptor_doc["uid"],
                seq_num=1,
                counts=[1, 2, 3, 4, 5],
                position_x=[10, 11, 12, 13, 14],
                total_counts=15,
                time_value=start_doc["time"] + 0.1,
            )
            descriptor_doc["data_keys"]["he3psd0_counts"]["shape"] = [5]
            start_doc["det_config"] = {"nbins": 5}
            self._prepare_writer(writer, start_doc, descriptor_doc, [event_doc])
            file_name = self._write_file(writer)

            with h5py.File(file_name, "r") as handle:
                axis = handle["/entry/data/position_x"][()]
                expected = np.linspace(-209.21799055746422, 209.21799055746422, 5)
                np.testing.assert_allclose(axis, expected)
                self.assertFalse(np.allclose(axis, np.array([10, 11, 12, 13, 14], dtype=float)))

    def test_count_he3_multi_event_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(plan_name="count_he3")
            descriptor_doc = _build_descriptor(start_uid=start_doc["uid"])
            events = [
                _build_event(
                    descriptor_uid=descriptor_doc["uid"],
                    seq_num=1,
                    counts=[1, 2, 3, 4, 5],
                    position_x=[-2, -1, 0, 1, 2],
                    total_counts=15,
                    time_value=start_doc["time"] + 0.1,
                ),
                _build_event(
                    descriptor_uid=descriptor_doc["uid"],
                    seq_num=2,
                    counts=[2, 3, 4, 5, 6],
                    position_x=[-2, -1, 0, 1, 2],
                    total_counts=20,
                    time_value=start_doc["time"] + 0.2,
                ),
            ]
            self._prepare_writer(writer, start_doc, descriptor_doc, events)
            file_name = self._write_file(writer)

            with h5py.File(file_name, "r") as handle:
                self.assertEqual(handle["/entry/data/counts"].shape, (2, 5))
                self.assertEqual(handle["/entry/data/position_x"].shape, (5,))
                self.assertEqual(handle["/entry/data/event_index"].shape, (2,))
                self.assertEqual(_normalize_attr(handle["/entry/data"].attrs["axes"]), ["event_index", "position_x"])

    def test_scan_he3_layout_includes_motor_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self._new_writer(tmpdir)
            start_doc = _build_start_doc(plan_name="scan_he3", motors=["theta"])
            descriptor_doc = _build_descriptor(start_uid=start_doc["uid"], motors=["theta"])
            events = [
                _build_event(
                    descriptor_uid=descriptor_doc["uid"],
                    seq_num=1,
                    counts=[1, 2, 3, 4, 5],
                    position_x=[-2, -1, 0, 1, 2],
                    total_counts=15,
                    time_value=start_doc["time"] + 0.1,
                    motors={"theta": 1.0},
                ),
                _build_event(
                    descriptor_uid=descriptor_doc["uid"],
                    seq_num=2,
                    counts=[2, 3, 4, 5, 6],
                    position_x=[-2, -1, 0, 1, 2],
                    total_counts=20,
                    time_value=start_doc["time"] + 0.2,
                    motors={"theta": 2.0},
                ),
                _build_event(
                    descriptor_uid=descriptor_doc["uid"],
                    seq_num=3,
                    counts=[3, 4, 5, 6, 7],
                    position_x=[-2, -1, 0, 1, 2],
                    total_counts=25,
                    time_value=start_doc["time"] + 0.3,
                    motors={"theta": 3.0},
                ),
            ]
            self._prepare_writer(writer, start_doc, descriptor_doc, events)
            file_name = self._write_file(writer)

            with h5py.File(file_name, "r") as handle:
                self.assertEqual(handle["/entry/data/counts"].shape, (3, 5))
                self.assertEqual(handle["/entry/data/theta"].shape, (3,))
                self.assertEqual(_normalize_attr(handle["/entry/data"].attrs["axes"]), ["event_index", "position_x"])

    def test_multi_motor_scan_layout_keeps_event_order_for_parallel_and_2d_runs(self):
        for plan_name in ("scan_parallel_he3", "scan2D_he3"):
            with self.subTest(plan_name=plan_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    writer = self._new_writer(tmpdir)
                    start_doc = _build_start_doc(plan_name=plan_name, motors=["theta", "phi"])
                    descriptor_doc = _build_descriptor(start_uid=start_doc["uid"], motors=["theta", "phi"])
                    events = [
                        _build_event(
                            descriptor_uid=descriptor_doc["uid"],
                            seq_num=1,
                            counts=[1, 2, 3, 4, 5],
                            position_x=[-2, -1, 0, 1, 2],
                            total_counts=15,
                            time_value=start_doc["time"] + 0.1,
                            motors={"theta": 1.0, "phi": 10.0},
                        ),
                        _build_event(
                            descriptor_uid=descriptor_doc["uid"],
                            seq_num=2,
                            counts=[2, 3, 4, 5, 6],
                            position_x=[-2, -1, 0, 1, 2],
                            total_counts=20,
                            time_value=start_doc["time"] + 0.2,
                            motors={"theta": 2.0, "phi": 20.0},
                        ),
                        _build_event(
                            descriptor_uid=descriptor_doc["uid"],
                            seq_num=3,
                            counts=[3, 4, 5, 6, 7],
                            position_x=[-2, -1, 0, 1, 2],
                            total_counts=25,
                            time_value=start_doc["time"] + 0.3,
                            motors={"theta": 3.0, "phi": 30.0},
                        ),
                    ]
                    self._prepare_writer(writer, start_doc, descriptor_doc, events)
                    file_name = self._write_file(writer)

                    with h5py.File(file_name, "r") as handle:
                        self.assertEqual(handle["/entry/data/counts"].shape, (3, 5))
                        self.assertEqual(handle["/entry/data/theta"].shape, (3,))
                        self.assertEqual(handle["/entry/data/phi"].shape, (3,))
                        np.testing.assert_array_equal(handle["/entry/data/event_index"][()], np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
