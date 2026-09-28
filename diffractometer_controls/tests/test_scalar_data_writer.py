import csv
import importlib.util
import pathlib
import tempfile
import time
import unittest

import h5py


STARTUP_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "bluesky_config"
    / "startup"
    / "23-scalar_data_writer.py"
)


def _load_writer_module():
    spec = importlib.util.spec_from_file_location("scalar_data_writer_startup", STARTUP_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _start_doc(file_type="csv", *, data_type="scalar", catalog="4dh4"):
    return {
        "uid": "abcdef1234567890",
        "time": 1715798400.0,
        "scan_id": 12,
        "plan_name": "scan_scalar",
        "experiment_type": "diffraction",
        "data_type": data_type,
        "file_type": file_type,
        "detectors": ["temperature"],
        "motors": ["theta"],
        "bluesky_catalog": catalog,
        "title": "Temperature scan",
        "sample": "sample-a",
        "plan_args": {"file_type": file_type, "acquire_time": 1.0},
        "det_config": {"software_dwell_time": 1.0},
    }


def _descriptor(start_uid):
    return {
        "uid": "primary-descriptor",
        "run_start": start_uid,
        "time": 1715798400.1,
        "name": "primary",
        "data_keys": {
            "temperature": {
                "source": "SIM:TEMP",
                "dtype": "number",
                "shape": [],
                "units": "C",
                "object_name": "temperature",
            },
            "theta": {
                "source": "SIM:THETA",
                "dtype": "number",
                "shape": [],
                "units": "deg",
                "object_name": "theta",
            },
        },
        "object_keys": {
            "temperature": ["temperature"],
            "theta": ["theta"],
        },
    }


def _event(descriptor_uid, seq_num, temperature, theta):
    timestamp = 1715798400.0 + seq_num
    return {
        "uid": f"event-{seq_num}",
        "descriptor": descriptor_uid,
        "seq_num": seq_num,
        "time": timestamp,
        "data": {"temperature": temperature, "theta": theta},
        "timestamps": {"temperature": timestamp, "theta": timestamp},
    }


def _baseline_descriptor(start_uid):
    return {
        "uid": "baseline-descriptor",
        "run_start": start_uid,
        "time": 1715798400.05,
        "name": "baseline",
        "data_keys": {
            "reactor_power_6": {
                "source": "SIM:POWER6",
                "dtype": "number",
                "shape": [],
                "units": "MW",
                "object_name": "reactor_power_6",
            },
            "sample_temperature": {
                "source": "SIM:SAMPLE:TEMP",
                "dtype": "number",
                "shape": [],
                "units": "C",
                "object_name": "sample_temperature",
            },
        },
        "object_keys": {
            "reactor_power_6": ["reactor_power_6"],
            "sample_temperature": ["sample_temperature"],
        },
    }


def _baseline_event(descriptor_uid, seq_num, reactor_power, temperature):
    timestamp = 1715798400.0 + seq_num * 0.01
    return {
        "uid": f"baseline-event-{seq_num}",
        "descriptor": descriptor_uid,
        "seq_num": seq_num,
        "time": timestamp,
        "data": {
            "reactor_power_6": reactor_power,
            "sample_temperature": temperature,
        },
        "timestamps": {
            "reactor_power_6": timestamp,
            "sample_temperature": timestamp,
        },
    }


class ScalarDiffractionWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_writer_module()

    def _run_documents(self, writer, file_type):
        start = _start_doc(file_type)
        descriptor = _descriptor(start["uid"])
        baseline = _baseline_descriptor(start["uid"])
        writer.receiver("start", start)
        writer.receiver("descriptor", baseline)
        writer.receiver(
            "event", _baseline_event(baseline["uid"], 1, 5.95, 24.0)
        )
        writer.receiver("descriptor", descriptor)
        writer.receiver("event", _event(descriptor["uid"], 1, 295.1, 0.0))
        writer.receiver("event", _event(descriptor["uid"], 2, 295.4, 0.5))
        writer.receiver(
            "event", _baseline_event(baseline["uid"], 2, 5.90, 24.5)
        )
        writer.receiver(
            "stop",
            {
                "uid": "stop-uid",
                "run_start": start["uid"],
                "time": 1715798403.0,
                "exit_status": "success",
                "reason": "",
            },
        )
        return start

    def test_csv_contains_comment_metadata_and_tabular_primary_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self.module.ScalarDiffractionWriter()
            writer.output_root = pathlib.Path(tmpdir)
            writer.test_output_root = pathlib.Path(tmpdir) / "TestData" / "Diffraction"
            self._run_documents(writer, "csv")

            output = pathlib.Path(writer.output_csv_file)
            self.assertTrue(output.exists())
            self.assertEqual(output.suffix, ".csv")
            text = output.read_text(encoding="utf-8")
            csv_rows = list(csv.reader(text.splitlines()))
            self.assertIn(
                ["# MITR scalar metadata", "Start", "End", "Units"],
                csv_rows,
            )
            self.assertIn(["# title", "Temperature scan", "", ""], csv_rows)
            self.assertIn(["# sample", "sample-a", "", ""], csv_rows)
            self.assertIn(["# plan_args.file_type", "csv", "", ""], csv_rows)
            self.assertIn(["# plan_args.acquire_time", "1.0", "", ""], csv_rows)
            self.assertIn(
                ["# baseline.reactor_power_6", "5.95", "5.9", "MW"],
                csv_rows,
            )
            self.assertIn(
                ["# baseline.sample_temperature", "24.0", "24.5", "C"],
                csv_rows,
            )
            self.assertLess(
                csv_rows.index(["# uid", "abcdef1234567890", "", ""]),
                csv_rows.index(
                    ["# MITR scalar metadata", "Start", "End", "Units"]
                ),
            )
            self.assertLess(
                csv_rows.index(
                    ["# MITR scalar metadata", "Start", "End", "Units"]
                ),
                csv_rows.index(
                    ["# baseline.reactor_power_6", "5.95", "5.9", "MW"]
                ),
            )
            self.assertLess(
                csv_rows.index(
                    ["# baseline.reactor_power_6", "5.95", "5.9", "MW"]
                ),
                csv_rows.index(
                    ["# baseline.sample_temperature", "24.0", "24.5", "C"]
                ),
            )

            rows = list(
                csv.DictReader(
                    line
                    for line in text.splitlines()
                    if line.strip() and not line.startswith("#")
                )
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["temperature"], "295.1")
            self.assertEqual(rows[1]["theta"], "0.5")

    def test_nexus_selection_writes_a_valid_nexus_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self.module.ScalarDiffractionWriter()
            writer.output_root = pathlib.Path(tmpdir)
            writer.test_output_root = pathlib.Path(tmpdir) / "TestData" / "Diffraction"
            self._run_documents(writer, "nexus")

            deadline = time.monotonic() + 10.0
            while writer.output_nexus_file is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertIsNotNone(writer.output_nexus_file)
            output = pathlib.Path(writer.output_nexus_file)
            self.assertEqual(output.suffix, ".nxs")
            with h5py.File(output, "r") as handle:
                self.assertIn("entry", handle)
                self.assertIn("instrument", handle["entry"])
                baseline = handle[
                    "/entry/instrument/bluesky/metadata/baseline_readings"
                ][()]
                if isinstance(baseline, bytes):
                    baseline = baseline.decode()
                self.assertIn("reactor_power_6", baseline)
                self.assertIn("sample_temperature", baseline)

    def test_ignores_non_scalar_and_unselected_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self.module.ScalarDiffractionWriter()
            writer.output_root = pathlib.Path(tmpdir)
            writer.test_output_root = pathlib.Path(tmpdir) / "TestData" / "Diffraction"
            writer.receiver("start", _start_doc("csv", data_type="1d"))
            self.assertFalse(writer.scanning)
            self.assertIsNone(writer.file_name)

            start = _start_doc("")
            writer.receiver("start", start)
            self.assertFalse(writer.scanning)
            self.assertIsNone(writer.file_name)

    def test_test_catalog_uses_test_directory_and_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = self.module.ScalarDiffractionWriter()
            writer.output_root = pathlib.Path(tmpdir)
            writer.test_output_root = pathlib.Path(tmpdir) / "TestData" / "Diffraction"
            start = _start_doc("csv", catalog="testdb")
            writer.receiver("start", start)
            output = pathlib.Path(writer.file_name)
            self.assertEqual(
                output.parent,
                pathlib.Path(tmpdir) / "TestData" / "Diffraction" / "2024",
            )
            self.assertTrue(output.name.startswith("test-"))


if __name__ == "__main__":
    unittest.main()
