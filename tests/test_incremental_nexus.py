"""Real SWMR readers, durable point boundaries, and abrupt-process recovery."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from server.services.recover_diffraction import recover_journal
from server.writers.he3_nexus import HE3DiffractionNXWriter
from server.writers.incremental import DatasetSequence, DOCUMENTS, STREAMS
from tests.test_he3_nexus_writer import _build_start_doc, _build_descriptor, _build_event


class IncrementalNexusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.writer = HE3DiffractionNXWriter()
        self.writer.output_root = self.root
        self.addCleanup(self.writer.close)
        self.start = _build_start_doc()
        self.descriptor = _build_descriptor(start_uid=self.start["uid"])
        self.writer.receiver("start", self.start)
        self.writer.receiver("descriptor", self.descriptor)

    def event(self, seq):
        return _build_event(descriptor_uid=self.descriptor["uid"], seq_num=seq,
                            counts=np.arange(5, dtype="int32") + seq,
                            position_x=np.linspace(-2, 2, 5), total_counts=10 + seq,
                            time_value=self.start["time"] + seq)

    def read_in_other_process(self):
        script = """
import h5py, json, sys
with h5py.File(sys.argv[1], 'r', libver='latest', swmr=True) as handle:
    stream = handle['/entry/instrument/bluesky/streams/primary']
    print(json.dumps({'committed': int(stream['committed'][()]),
                      'counts': stream['he3psd0_counts/value'][()].tolist()}))
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.writer.live_file)],
                                capture_output=True, text=True, timeout=15, check=True)
        return json.loads(result.stdout)

    def test_other_process_reads_committed_points_while_writer_remains_open(self):
        self.writer.receiver("event", self.event(1))
        self.assertEqual(self.read_in_other_process(), {"committed": 1, "counts": [(np.arange(5) + 1).tolist()]})
        self.writer.receiver("event", self.event(2))
        self.assertEqual(self.read_in_other_process()["committed"], 2)
        self.assertIsNone(self.writer.output_nexus_file)
        self.assertIsNotNone(self.writer._h5)
        counts = self.writer._h5[f"{STREAMS}/primary/he3psd0_counts/value"]
        self.assertEqual(counts.maxshape, (None, 5))
        self.assertEqual(counts.chunks, (1, 5))
        self.assertEqual(counts.id, self.writer._h5["entry/data/counts"].id)

    def test_acquisitions_keep_disk_views_instead_of_buffering_spectra(self):
        for seq in range(1, 51):
            self.writer.receiver("event", self.event(seq))
        for entry in self.writer.acquisitions[self.descriptor["uid"]]["data"].values():
            self.assertIsInstance(entry["data"], DatasetSequence)
            self.assertEqual(len(entry["data"]), 50)

    def test_failed_flush_leaves_previous_commit_and_no_completed_export(self):
        self.writer.receiver("event", self.event(1))
        with patch.object(self.writer, "_sync", side_effect=OSError("sync failed")):
            with self.assertRaisesRegex(OSError, "sync failed"):
                self.writer.receiver("event", self.event(2))
        self.assertEqual(self.writer._h5[f"{STREAMS}/primary/committed"][()], 1)
        self.assertIsNone(self.writer.output_nexus_file)

    def test_late_stream_and_changed_field_shape_fail_without_losing_committed_data(self):
        self.writer.receiver("event", self.event(1))
        late = {**self.descriptor, "uid": "late", "name": "new_stream"}
        with self.assertRaisesRegex(ValueError, "SWMR cannot add"):
            self.writer.receiver("descriptor", late)
        bad = self.event(2)
        bad["data"]["he3psd0_counts"] = np.arange(3)
        with self.assertRaisesRegex(ValueError, "Shape or dtype changed"):
            self.writer.receiver("event", bad)
        self.assertEqual(self.read_in_other_process()["committed"], 1)

    def test_process_crash_preserves_swmr_data_and_journal_recovers_to_new_file(self):
        script = """
import os, sys
from pathlib import Path
import numpy as np
from server.services.run_storage import RunStorage
from server.writers.he3_nexus import HE3DiffractionNXWriter
from tests.test_he3_nexus_writer import _build_start_doc, _build_descriptor, _build_event
class Writer(HE3DiffractionNXWriter):
    output_root = Path(sys.argv[1])
storage = RunStorage(lambda *args: None)
storage.add_writer(Writer)
start = _build_start_doc()
desc = _build_descriptor(start_uid=start['uid'])
storage('start', start)
storage('descriptor', desc)
for seq in (1, 2):
    storage('event', _build_event(descriptor_uid=desc['uid'], seq_num=seq,
        counts=np.arange(5), position_x=np.linspace(-2, 2, 5), total_counts=10,
        time_value=start['time'] + seq))
os._exit(23)
"""
        crash_root = self.root / "crash"
        result = subprocess.run([sys.executable, "-c", script, str(crash_root)],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 23, result.stderr)
        original, = crash_root.rglob("*.nxs")
        journal, = crash_root.rglob("*.documents.jsonl")
        before = original.read_bytes()
        with h5py.File(original, "r", swmr=True, libver="latest") as handle:
            self.assertEqual(handle[f"{STREAMS}/primary/committed"][()], 2)
            self.assertEqual(handle["entry/instrument/bluesky/storage/state"][()], b"collecting")
        # A torn last journal line must not lose the earlier complete records.
        with journal.open("a") as stream:
            stream.write('{"name":"event","doc":')
        output = recover_journal(journal, self.root / "recovered.nxs")
        self.assertEqual(original.read_bytes(), before)
        self.assertTrue(journal.exists())
        with h5py.File(output, "r") as handle:
            stop = json.loads(handle[f"{DOCUMENTS}/stop"][()].tobytes())
            self.assertEqual(stop["exit_status"], "abort")
            self.assertEqual(stop["num_events"], {"primary": 2})
            self.assertIn("Recovered", stop["reason"])
            self.assertEqual(handle["entry/data/counts"].shape, (2, 5))
        with self.assertRaises(FileExistsError):
            recover_journal(journal, output)


if __name__ == "__main__":
    unittest.main()
