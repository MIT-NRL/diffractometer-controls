"""Real HDF5 and in-process Tiled checks; no IOC or production server access."""

import copy
import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import h5py
import numpy as np
from bluesky_tiled_plugins import TiledWriter
from tiled.catalog import from_uri
from tiled.client import from_context
from tiled.client.context import Context
from tiled.config import Authentication
from tiled.server.app import build_app

from server.services.run_storage import RunStorage, register_export, retry_registration
from server.writers.he3_nexus import HE3DiffractionNXWriter
from server.writers.scalar import ScalarDiffractionWriter
from tests.test_he3_nexus_writer import _build_start_doc, _build_descriptor, _build_event
from diffractometer_controls.analysis.diffraction_history import build_history_snapshot, fetch_recent_history_entries


class RunStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.temporary.name)
        tree = from_uri(
            cls.root / "catalog.db", init_if_not_exists=True,
            writable_storage=[f"duckdb:///{cls.root}/tables.db"],
            readable_storage=[cls.root],
        )
        app = build_app(tree, authentication=Authentication(allow_anonymous_access=True))
        cls.context = Context.from_app(app)
        cls.catalog = from_context(cls.context)

    @classmethod
    def tearDownClass(cls):
        cls.context.close()
        cls.temporary.cleanup()

    def setUp(self):
        root = self.root

        class LocalWriter(HE3DiffractionNXWriter):
            output_root = root / "production"
            test_output_root = root / "test"

        self.writer_class = LocalWriter
        self.tiled_writer = TiledWriter(self.catalog, batch_size=1)
        self.storage = RunStorage(self.tiled_writer, self.catalog)
        self.storage.add_writer(LocalWriter)

    def documents(self, *, num=2, catalog="4dh4"):
        start = _build_start_doc(bluesky_catalog=catalog)
        start["uid"] = str(uuid4())
        descriptor = _build_descriptor(start_uid=start["uid"])
        descriptor["uid"] = str(uuid4())
        events = [
            _build_event(
                descriptor_uid=descriptor["uid"], seq_num=i + 1,
                counts=np.arange(5, dtype="int32") + i,
                position_x=np.linspace(-2, 2, 5), total_counts=10 + 5 * i,
                time_value=start["time"] + i + 0.1,
            )
            for i in range(num)
        ]
        stop = {
            "uid": str(uuid4()), "run_start": start["uid"],
            "time": start["time"] + num + 1, "exit_status": "success",
            "reason": "", "num_events": {"primary": num},
        }
        return start, descriptor, events, stop

    def acquire(self, documents):
        start, descriptor, events, stop = documents
        for name, doc in [("start", start), ("descriptor", descriptor),
                          *[("event", event) for event in events], ("stop", stop)]:
            self.storage(name, doc)
        return self.catalog[start["uid"]]

    def test_tiled_and_loader_read_a_stable_prefix_before_stop(self):
        start, descriptor, events, stop = self.documents(num=3)
        self.addCleanup(self.storage.close)
        self.storage("start", start)
        self.storage("descriptor", descriptor)
        for index, event in enumerate(events):
            self.storage("event", event)
            run = self.catalog[start["uid"]]
            self.assertEqual(run.metadata["file_export"]["status"], "collecting")
            self.assertEqual(run.metadata["file_export"]["committed_points"], index + 1)
            source, = run["primary/he3psd0_counts"].data_sources()
            self.assertTrue(source.parameters["swmr"])
            snapshot = build_history_snapshot(run)
            self.assertEqual(snapshot["seq_nums"], list(range(1, index + 2)))
            self.assertEqual(len(snapshot["rows"]), index + 1)
            np.testing.assert_array_equal(snapshot["rows"][-1]["he3psd0_counts"], np.arange(5) + index)
        self.storage("stop", stop)
        self.assertEqual(self.catalog[start["uid"]].metadata["file_export"]["status"], "ready")

    def test_array_monitors_before_first_primary_point_are_registered_in_full(self):
        start, descriptor, events, stop = self.documents(num=1)
        self.addCleanup(self.storage.close)
        monitor = copy.deepcopy(descriptor)
        monitor.update(uid=str(uuid4()), name="he3psd0_counts_monitor")
        monitor["data_keys"] = {"he3psd0_counts": monitor["data_keys"]["he3psd0_counts"]}
        monitor["object_keys"] = {"he3psd0": ["he3psd0_counts"]}
        self.storage("start", start)
        self.storage("descriptor", monitor)
        self.storage("descriptor", descriptor)

        def record_monitor(sequence):
            values = np.arange(5, dtype="int32") + sequence * 10
            self.storage("event", {
                "uid": str(uuid4()), "descriptor": monitor["uid"], "seq_num": sequence,
                "time": start["time"] + sequence / 100,
                "data": {"he3psd0_counts": values},
                "timestamps": {"he3psd0_counts": start["time"] + sequence / 100},
            })
            return values

        expected = [record_monitor(2), record_monitor(4)]
        self.storage("event", events[0])
        run = self.catalog[start["uid"]]
        np.testing.assert_array_equal(run["he3psd0_counts_monitor/he3psd0_counts"].read(), expected)
        expected.append(record_monitor(7))
        np.testing.assert_array_equal(run["he3psd0_counts_monitor/he3psd0_counts"].read(), expected)
        self.storage("stop", stop)
        np.testing.assert_array_equal(run["he3psd0_counts_monitor/he3psd0_counts"].read(), expected)
        self.assertEqual(run["he3psd0_counts_monitor/seq_num"].read().tolist(), [2, 4, 7])

    def test_tiled_failure_does_not_stop_incremental_disk_saving(self):
        start, descriptor, events, stop = self.documents(num=3)
        self.addCleanup(self.storage.close)
        self.storage("start", start)
        self.storage("descriptor", descriptor)
        self.storage("event", events[0])
        real_writer = self.storage.tiled_writer

        def offline(name, doc):
            if name == "event":
                raise OSError("Tiled offline")
            return real_writer(name, doc)

        self.storage.tiled_writer = offline
        for event in events[1:]:
            self.storage("event", event)
        partial = build_history_snapshot(self.catalog[start["uid"]])
        self.assertEqual(partial["seq_nums"], [1])
        with self.assertRaises(RuntimeError):
            self.storage("stop", stop)
        output = self.storage.last_result["journal"].removesuffix(".documents.jsonl")
        with h5py.File(output, "r") as handle:
            self.assertEqual(handle["entry/data/counts"].shape, (3, 5))

    def test_failure_before_tiled_start_still_saves_the_complete_file(self):
        documents = self.documents(num=2)
        self.storage.tiled_writer = lambda *args: (_ for _ in ()).throw(OSError("Tiled unavailable"))
        with self.assertRaises(RuntimeError):
            self.acquire(documents)
        journal = pathlib.Path(self.storage.last_result["journal"])
        output = pathlib.Path(str(journal).removesuffix(".documents.jsonl"))
        with h5py.File(output, "r") as handle:
            self.assertEqual(handle["entry/data/counts"].shape, (2, 5))
        self.assertTrue(journal.exists())
        manifest = json.loads(pathlib.Path(str(output) + ".registration.json").read_text())
        self.assertFalse(manifest["events_complete"])

    def test_failed_point_does_not_break_access_to_the_previous_committed_prefix(self):
        start, descriptor, events, stop = self.documents(num=2)
        self.addCleanup(self.storage.close)
        self.storage("start", start)
        self.storage("descriptor", descriptor)
        self.storage("event", events[0])
        writer = self.storage._active_runs[start["uid"]].writer
        with patch.object(writer, "_sync", side_effect=OSError("storage sync failed")):
            with self.assertRaises(OSError):
                self.storage("event", events[1])
        snapshot = build_history_snapshot(self.catalog[start["uid"]])
        self.assertEqual(snapshot["file_export"]["status"], "failed")
        self.assertEqual(snapshot["seq_nums"], [1])
        np.testing.assert_array_equal(snapshot["rows"][0]["he3psd0_counts"], np.arange(5))
        with self.assertRaises(OSError):
            self.storage("stop", stop)

    def test_arrays_read_from_hdf5_without_sql_or_zarr_copies(self):
        documents = self.documents()
        original = copy.deepcopy(documents)
        run = self.acquire(documents)
        array = run["primary/he3psd0_counts"]
        np.testing.assert_array_equal(array.read(), [np.arange(5), np.arange(5) + 1])
        np.testing.assert_array_equal(array[1, 1:4], np.arange(5)[1:4] + 1)
        source, = array.data_sources()
        self.assertEqual(source.management, "external")
        self.assertEqual(source.mimetype, "application/x-hdf5")
        self.assertEqual(array.dtype, np.dtype("int32"))
        frame = run["primary"].base["internal"].read()
        self.assertNotIn("he3psd0_counts", frame.columns)
        self.assertNotIn("he3psd0_position_x", frame.columns)
        self.assertEqual(frame["seq_num"].tolist(), [1, 2])
        self.assertEqual(self.storage.last_result["status"], "ready")
        path = pathlib.Path(self.storage.last_result["file"])
        reference = self.root / "permissions-reference"
        reference.write_text("created using the process umask")
        self.assertEqual(path.stat().st_mode & 0o777, reference.stat().st_mode & 0o777)
        self.assertEqual(source.assets[0].data_uri, path.as_uri())
        self.assertFalse(pathlib.Path(str(path) + ".documents.jsonl").exists())
        self.assertFalse(pathlib.Path(str(path) + ".registration.json").exists())
        self.assertEqual(documents[1], original[1])
        np.testing.assert_array_equal(documents[2][0]["data"]["he3psd0_counts"],
                                      original[2][0]["data"]["he3psd0_counts"])
        self.assertEqual(run.metadata["start"]["uid"], documents[0]["uid"])
        np.testing.assert_array_equal(run["artifacts/nexus/entry/data/counts"].read(), array.read())
        with h5py.File(path, "r") as handle:
            self.assertEqual(handle["entry/data/counts"].id,
                             handle["entry/instrument/bluesky/streams/primary/he3psd0_counts/value"].id)
            docs = handle["entry/instrument/bluesky/documents"]
            self.assertEqual(json.loads(docs["start"].asstr()[()]), documents[0])
            self.assertEqual(json.loads(docs["stop"][()].tobytes().decode("utf-8")), documents[3])

    def test_interleaved_runs_and_test_paths_keep_independent_metadata(self):
        first = self.documents()
        second = self.documents(num=1, catalog="testdb")
        second[0]["title"] = "second run"
        for documents in (first, second):
            self.storage("start", documents[0])
            self.storage("descriptor", documents[1])
        for documents in (second, first):
            for event in documents[2]:
                self.storage("event", event)
            self.storage("stop", documents[3])
        for documents, folder in ((first, "production"), (second, "test")):
            run = self.catalog[documents[0]["uid"]]
            path = pathlib.Path(run.metadata["file_export"]["file_uri"].removeprefix("file://"))
            self.assertTrue(path.is_relative_to(self.root / folder))
            with h5py.File(path) as handle:
                self.assertEqual(handle["entry/instrument/bluesky/metadata/title"].asstr()[()],
                                 documents[0]["title"])

    def test_declared_stream_with_no_events_still_exports_metadata(self):
        documents = self.documents(num=0)
        self.acquire(documents)
        self.assertEqual(self.storage.last_result["status"], "ready")
        with h5py.File(self.storage.last_result["file"]) as handle:
            self.assertIn("entry/instrument/bluesky/documents/stop", handle)

    def test_each_psd_gets_a_curated_view_with_its_own_bin_count(self):
        documents = self.documents()
        start, descriptor, events, stop = documents
        start["detectors"].append("he3psd7")
        second = _build_descriptor(start_uid=start["uid"], detector_name="he3psd7")
        for description in second["data_keys"].values():
            if description["dtype"] == "array":
                description["shape"] = [3]
        descriptor["data_keys"].update(second["data_keys"])
        descriptor["object_keys"].update(second["object_keys"])
        for event in events:
            event["data"].update(he3psd7_counts=np.arange(3),
                                 he3psd7_position_x=np.linspace(-2, 2, 3),
                                 he3psd7_total_counts=3)
            event["timestamps"].update({key: event["time"] for key in second["data_keys"]})
        self.acquire(documents)
        with h5py.File(self.storage.last_result["file"]) as handle:
            self.assertEqual(handle["entry/data/counts"].shape, (2, 5))
            self.assertEqual(handle["entry/data_he3psd7/counts"].shape, (2, 3))
            self.assertEqual(handle["entry/data_he3psd7/position_x"].shape, (3,))
            self.assertNotIn("data_he3psd0_total", handle["entry"])

    def test_small_and_full_size_profiles_keep_the_event_dimension(self):
        for bins in (1, 512):
            with self.subTest(bins=bins):
                documents = self.documents(num=1)
                for description in documents[1]["data_keys"].values():
                    if description["dtype"] == "array":
                        description["shape"] = [bins]
                documents[2][0]["data"].update(
                    he3psd0_counts=np.arange(bins, dtype="int32"),
                    he3psd0_position_x=np.linspace(-2, 2, bins),
                )
                run = self.acquire(documents)
                array = run["primary/he3psd0_counts"]
                self.assertEqual(array.shape, (1, bins))
                np.testing.assert_array_equal(array.read(), [np.arange(bins)])

    def test_history_loader_reads_registered_psd_arrays_and_ignores_artifacts(self):
        documents = self.documents()
        snapshot = build_history_snapshot(self.acquire(documents))
        self.assertEqual(snapshot["stream_name"], "primary")
        self.assertEqual(snapshot["seq_nums"], [1, 2])
        self.assertEqual(snapshot["file_export"]["status"], "ready")
        self.assertEqual(snapshot["start"], {**documents[0], "data_type": "1d"})
        self.assertEqual(snapshot["inferred_metadata"], {"data_type": "1d"})
        self.assertNotIn("data_type", documents[0])
        for row, event in zip(snapshot["rows"], documents[2]):
            for key, expected in event["data"].items():
                np.testing.assert_array_equal(row[key], expected)

    def test_history_loader_reads_pre_hdf5_tiled_arrays(self):
        documents = self.documents()
        for name, doc in [("start", documents[0]), ("descriptor", documents[1]),
                          *[("event", event) for event in documents[2]], ("stop", documents[3])]:
            self.tiled_writer(name, doc)
        snapshot = build_history_snapshot(self.catalog[documents[0]["uid"]])
        self.assertEqual(len(snapshot["rows"]), 2)
        self.assertEqual(snapshot["file_export"], {})
        np.testing.assert_array_equal(snapshot["rows"][1]["he3psd0_counts"], np.arange(5) + 1)

    def test_history_loader_reads_scalar_runs_in_both_file_formats(self):
        from tests.test_scalar_data_writer import _start_doc, _descriptor, _event
        root = self.root

        class LocalScalar(ScalarDiffractionWriter):
            output_root = root / "scalar-history"

        self.storage.add_writer(LocalScalar)
        for format_name in ("csv", "nexus"):
            with self.subTest(format=format_name):
                start = _start_doc(format_name)
                start["uid"] = str(uuid4())
                descriptor = _descriptor(start["uid"])
                descriptor["uid"] = str(uuid4())
                events = [_event(descriptor["uid"], i + 1, 295 + i, i * 0.5) for i in range(3)]
                stop = {"uid": str(uuid4()), "run_start": start["uid"], "time": start["time"] + 4,
                        "exit_status": "success", "reason": "", "num_events": {"primary": 3}}
                snapshot = build_history_snapshot(self.acquire((start, descriptor, events, stop)))
                self.assertEqual([row["temperature"] for row in snapshot["rows"]], [295, 296, 297])
                self.assertEqual([row["theta"] for row in snapshot["rows"]], [0, 0.5, 1])
                self.assertEqual(snapshot["start"]["live_plot_fields"]["temperature"]["units"], "C")
                self.assertNotIn("theta", snapshot["start"]["live_plot_fields"])

    def test_active_scalar_loading_is_independent_of_csv_or_nexus_export(self):
        from tests.test_scalar_data_writer import _start_doc, _descriptor, _event
        root = self.root

        class LocalScalar(ScalarDiffractionWriter):
            output_root = root / 'active-scalar'

        self.storage.add_writer(LocalScalar)
        self.addCleanup(self.storage.close)
        for format_name in ('csv', 'nexus'):
            with self.subTest(format=format_name):
                start = _start_doc(format_name)
                start['uid'] = str(uuid4())
                descriptor = _descriptor(start['uid'])
                descriptor['uid'] = str(uuid4())
                self.storage('start', start)
                self.storage('descriptor', descriptor)
                for seq in (2, 4):
                    self.storage('event', _event(descriptor['uid'], seq, 295 + seq, seq * 0.5))
                    snapshot = build_history_snapshot(self.catalog[start['uid']])
                    self.assertEqual(snapshot['seq_nums'], [2] if seq == 2 else [2, 4])
                    self.assertEqual(snapshot['rows'][-1]['temperature'], 295 + seq)
                    self.assertIsNone(snapshot['stop'])
                # A stale export watermark must not hide scalar SQL rows.
                run = self.catalog[start['uid']]
                base = getattr(run, 'base', run)
                base.update_metadata({'file_export': {'status': 'collecting', 'incremental': True,
                                                      'committed_points': 0}})
                self.assertEqual(len(build_history_snapshot(self.catalog[start['uid']])['rows']), 2)
                self.storage('stop', {'uid': str(uuid4()), 'run_start': start['uid'],
                                     'time': start['time'] + 5, 'exit_status': 'success',
                                     'reason': '', 'num_events': {'primary': 2}})

    def test_failed_scalar_file_export_keeps_tiled_readings_loadable(self):
        from tests.test_scalar_data_writer import _start_doc, _descriptor, _event
        root = self.root

        class LocalScalar(ScalarDiffractionWriter):
            output_root = root / 'failed-scalar-export'

        self.storage.add_writer(LocalScalar)
        for format_name in ('csv', 'nexus'):
            with self.subTest(format=format_name):
                start = _start_doc(format_name)
                start['uid'] = str(uuid4())
                descriptor = _descriptor(start['uid'])
                descriptor['uid'] = str(uuid4())
                events = [_event(descriptor['uid'], index + 1, 295 + index, index * 0.5) for index in range(2)]
                stop = {'uid': str(uuid4()), 'run_start': start['uid'], 'time': start['time'] + 3,
                        'exit_status': 'success', 'reason': '', 'num_events': {'primary': 2}}
                method = '_write_csv' if format_name == 'csv' else 'validate_file'
                with patch.object(LocalScalar, method, side_effect=OSError('Export disk unavailable')):
                    with self.assertRaises(OSError):
                        self.acquire((start, descriptor, events, stop))
                snapshot = build_history_snapshot(self.catalog[start['uid']])
                self.assertEqual(snapshot['seq_nums'], [1, 2])
                self.assertEqual([row['temperature'] for row in snapshot['rows']], [295, 296])
                self.assertEqual(snapshot['file_export']['status'], 'failed')
                self.assertIn('Export disk unavailable', snapshot['file_export']['error'])
                self.assertEqual(snapshot['stop'], stop)

    def test_history_search_sorts_nested_start_times_newest_first(self):
        uids = []
        for timestamp in (2100000000, 2100000002, 2100000001):
            docs = self.documents(num=1)
            docs[0]["time"] = timestamp
            self.acquire(docs)
            uids.append(docs[0]["uid"])
        entries = fetch_recent_history_entries(self.catalog, limit=3)
        self.assertEqual([entry["uid"] for entry in entries], [uids[1], uids[2], uids[0]])

    def test_metadata_enrichment_preserves_arrays_and_original_hdf5_documents(self):
        from server.services.diffraction_metadata import repair_catalog_metadata

        docs = self.documents(num=1)
        docs[0].pop("data_type", None)
        run = self.acquire(docs)
        path = pathlib.Path(self.storage.last_result["file"])
        before_file = path.read_bytes()
        backup = self.root / f"metadata-backup-{uuid4()}.jsonl"
        repair_catalog_metadata(self.catalog, backup_path=backup)
        current = self.catalog[docs[0]["uid"]]
        self.assertEqual(current.metadata["start"]["data_type"], "1d")
        self.assertEqual(current.metadata["stop"], docs[3])
        np.testing.assert_array_equal(current["primary/he3psd0_counts"].read(), [np.arange(5)])
        self.assertEqual(path.read_bytes(), before_file)
        self.assertIn("metadata_enrichment", current.metadata)

    def test_real_simulated_plan_preserves_configuration_and_baseline(self):
        from bluesky import RunEngine, SupplementalData
        from ophyd import Signal
        from ophyd.sim import SynAxis
        from server.context import StartupContext
        from server.devices.simulated import SimHE3PSD
        from server.plans.helpers import create_plans as helpers
        from server.plans.he3psd import create_plans as he3_plans

        first = SimHE3PSD(name="sim_he3psd0", motor=SynAxis(name="sample"))
        second = SimHE3PSD(name="sim_he3psd1", motor=SynAxis(name="sample2"))
        second.nbins.put(9)
        namespace = {"__name__": "__main__", first.name: first, second.name: second}
        context = StartupContext(namespace, runtime={"caget": lambda *args, **kwargs: None})
        context.publish(helpers(context))
        context.publish(he3_plans(context))
        engine = RunEngine({"bluesky_catalog": "testdb"})
        supplemental = SupplementalData(baseline=[Signal(name="reactor_power_6", value=6.0)])
        engine.preprocessors.append(supplemental)
        engine.subscribe(self.storage)
        uid, = engine(namespace["count_he3psd"](
            title="storage integration", detectors=[first, second], acquire_time=0.001, num=2,
        ))
        run = self.catalog[uid]
        self.assertEqual(run["primary/sim_he3psd0_counts"].shape, (2, 350))
        self.assertEqual(run["primary/sim_he3psd1_counts"].shape, (2, 9))
        self.assertEqual(run.metadata["start"]["detector_configurations"][second.name]["nbins"], 9)
        self.assertEqual(run.metadata["baseline_readings"]["reactor_power_6"]["start"], 6.0)
        with h5py.File(self.storage.last_result["file"]) as handle:
            self.assertIn("entry/data_sim_he3psd1/counts", handle)

    def test_real_scalar_nexus_plan_keeps_live_monitor_streams(self):
        from bluesky import RunEngine
        from tests.test_scalar_plans import _load_scalar_plan_namespace
        root = self.root

        class LocalScalar(ScalarDiffractionWriter):
            output_root = root / "scalar-monitor"
            test_output_root = root / "scalar-monitor-test"

        self.storage.add_writer(LocalScalar)
        namespace = _load_scalar_plan_namespace()
        engine = RunEngine({"bluesky_catalog": "testdb"})
        engine.subscribe(self.storage)
        uid, = engine(namespace["count_scalar"](
            "incremental scalar", detectors=namespace["temperature"],
            acquire_time=0.01, num=2, file_type="nexus",
        ))
        self.assertEqual(len(build_history_snapshot(self.catalog[uid])["rows"]), 2)
        with h5py.File(self.storage.last_result["file"]) as handle:
            streams = handle["entry/instrument/bluesky/streams"]
            self.assertIn("temperature_monitor", streams)
            self.assertGreaterEqual(streams["temperature_monitor/committed"][()], 2)

    def test_artifact_retry_preserves_file_and_does_not_duplicate_scalar_rows(self):
        documents = self.documents()
        with patch("server.services.run_storage.register_export", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                self.acquire(documents)
        journal = pathlib.Path(self.storage.last_result["journal"])
        path = pathlib.Path(str(journal).removesuffix(".documents.jsonl"))
        pending = pathlib.Path(str(path) + ".registration.json")
        self.assertTrue(path.exists())
        self.assertTrue(journal.exists())
        before = path.read_bytes()
        manifest = retry_registration(self.catalog, pending)
        register_export(self.catalog, manifest)  # Repeating registration is safe.
        self.assertEqual(path.read_bytes(), before)
        frame = self.catalog[documents[0]["uid"]]["primary"].base["internal"].read()
        self.assertEqual(frame["seq_num"].tolist(), [1, 2])
        self.assertFalse(journal.exists())
        self.assertFalse(pending.exists())

    def test_retry_repairs_external_registration_failure(self):
        documents = self.documents()
        real_writer = self.storage.tiled_writer

        def disconnect(name, doc):
            if name == "stream_datum":
                raise OSError("connection lost while registering arrays")
            return real_writer(name, doc)

        self.storage.tiled_writer = disconnect
        with self.assertRaises(RuntimeError):
            self.acquire(documents)
        journal = self.storage.last_result["journal"]
        pending = pathlib.Path(journal.removesuffix(".documents.jsonl") + ".registration.json")
        retry_registration(self.catalog, pending)
        run = self.catalog[documents[0]["uid"]]
        np.testing.assert_array_equal(run["primary/he3psd0_counts"].read(),
                                      [np.arange(5), np.arange(5) + 1])
        self.assertEqual(run.metadata["stop"]["uid"], documents[3]["uid"])

    def test_retry_rejects_changed_file_and_retains_recovery_documents(self):
        documents = self.documents()
        with patch("server.services.run_storage.register_export", side_effect=OSError("offline")):
            with self.assertRaises(OSError):
                self.acquire(documents)
        journal = pathlib.Path(self.storage.last_result["journal"])
        output = pathlib.Path(str(journal).removesuffix(".documents.jsonl"))
        pending = pathlib.Path(str(output) + ".registration.json")
        with output.open("ab") as stream:
            stream.write(b"changed after export")
        with self.assertRaisesRegex(ValueError, "checksum"):
            retry_registration(self.catalog, pending)
        self.assertTrue(journal.exists())
        self.assertTrue(pending.exists())

    def test_missing_scalar_delivery_is_not_marked_ready_by_registration_retry(self):
        documents = self.documents()
        real_writer = self.storage.tiled_writer

        def disconnect(name, doc):
            if name == "event":
                raise OSError("scalar delivery failed")
            return real_writer(name, doc)

        self.storage.tiled_writer = disconnect
        with self.assertRaises(RuntimeError):
            self.acquire(documents)
        journal = pathlib.Path(self.storage.last_result["journal"])
        output = pathlib.Path(str(journal).removesuffix(".documents.jsonl"))
        pending = pathlib.Path(str(output) + ".registration.json")
        with self.assertRaisesRegex(ValueError, "Scalar delivery was incomplete"):
            retry_registration(self.catalog, pending)
        with h5py.File(output) as handle:
            self.assertEqual(handle["entry/data/counts"].shape, (2, 5))
        self.assertEqual(self.catalog[documents[0]["uid"]].metadata["file_export"]["status"], "failed")
        self.assertTrue(journal.exists())

    def test_write_failure_never_publishes_final_file(self):
        documents = self.documents()
        with patch.object(self.writer_class, "write_root", side_effect=OSError("disk full")):
            self.storage("start", documents[0])
            self.storage("descriptor", documents[1])
            with self.assertRaises(OSError):
                self.storage("event", documents[2][0])
            # RunEngine emits Stop after an event callback fails. Exercise that
            # cleanup explicitly when using this document-level fixture.
            with self.assertRaises(OSError):
                self.storage("stop", documents[3])
        journal = pathlib.Path(self.storage.last_result["journal"])
        output = pathlib.Path(str(journal).removesuffix(".documents.jsonl"))
        self.assertFalse(output.exists())
        self.assertTrue(journal.exists())
        run = self.catalog[documents[0]["uid"]]
        self.assertEqual(run.metadata["file_export"]["status"], "failed")
        self.assertEqual(run.metadata["stop"]["uid"], documents[3]["uid"])
        records = [json.loads(line) for line in journal.read_text().splitlines()]
        self.assertEqual(records[-1]["name"], "stop")
        self.assertEqual(records[2]["doc"]["data"]["he3psd0_counts"], list(range(5)))

    def test_retry_repairs_zero_length_array_after_shape_update_fails(self):
        documents = self.documents()
        with patch("bluesky_tiled_plugins.writing.tiled_writer._RunWriter._update_data_source_for_node",
                   side_effect=OSError("connection lost during shape update")):
            with self.assertRaises(RuntimeError):
                self.acquire(documents)
        journal = self.storage.last_result["journal"]
        pending = pathlib.Path(journal.removesuffix(".documents.jsonl") + ".registration.json")
        retry_registration(self.catalog, pending)
        run = self.catalog[documents[0]["uid"]]
        self.assertEqual(run["primary/he3psd0_counts"].shape, (2, 5))
        np.testing.assert_array_equal(run["primary/he3psd0_counts"].read(),
                                      [np.arange(5), np.arange(5) + 1])

    def test_csv_is_registered_without_changing_its_existing_format(self):
        root = self.root

        class LocalScalar(ScalarDiffractionWriter):
            output_root = root / "scalar"

        self.storage.add_writer(LocalScalar)
        from tests.test_scalar_data_writer import _start_doc, _descriptor, _event
        start = _start_doc()
        start["uid"] = str(uuid4())
        descriptor = _descriptor(start["uid"])
        descriptor["uid"] = str(uuid4())
        event = _event(descriptor["uid"], 1, 295.1, 0)
        stop = {"uid": str(uuid4()), "run_start": start["uid"], "time": start["time"] + 2,
                "exit_status": "success", "reason": "", "num_events": {"primary": 1}}
        run = self.acquire((start, descriptor, [event], stop))
        source, = run["artifacts/csv"].data_sources()
        self.assertEqual(source.management, "external")
        self.assertIn("# MITR scalar metadata", pathlib.Path(self.storage.last_result["file"]).read_text())

    def test_existing_csv_and_nexus_exports_are_never_overwritten(self):
        from tests.test_scalar_data_writer import ScalarDiffractionWriterTests

        case = ScalarDiffractionWriterTests()
        for format_name in ("csv", "nexus"):
            with self.subTest(format=format_name):
                writer = ScalarDiffractionWriter()
                writer.output_root = self.root / "collisions" / str(uuid4())
                case._run_documents(writer, format_name)
                output = writer.output_csv_file or writer.output_nexus_file
                original = pathlib.Path(output).read_bytes()
                with self.assertRaises(FileExistsError):
                    case._run_documents(writer, format_name)
                self.assertEqual(pathlib.Path(output).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
