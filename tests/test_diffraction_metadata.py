import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from diffractometer_controls.analysis.diffraction_history import suggest_diffraction_start_updates
from server.services.diffraction_metadata import repair_catalog_metadata
from tests.test_diffraction_history_window import make_psd


class DiffractionMetadataTests(unittest.TestCase):
    def legacy_run(self):
        run = make_psd('legacy')
        run.metadata['start'].pop('data_type', None)
        run.metadata['start'].pop('detectors', None)
        run['primary'].read = Mock(side_effect=AssertionError('Metadata repair read event data'))
        return run

    def test_recovery_requires_matching_recorded_psd_fields(self):
        run = self.legacy_run()
        self.assertEqual(suggest_diffraction_start_updates(run), {'data_type': '1d', 'detectors': ['psd']})
        run['primary'].metadata['data_keys'].pop('psd_position_x')
        self.assertEqual(suggest_diffraction_start_updates(run), {})
        run['primary'].read.assert_not_called()

    def test_never_changes_explicit_imaging_or_untagged_synthetic_runs(self):
        for experiment_type, plan in [('imaging', 'scan_he3'), ('', 'codex_he3_array_writer_check'), ('', 'adaptive_imaging_focus_scan')]:
            run = self.legacy_run()
            run.metadata['start'].update(experiment_type=experiment_type, plan_name=plan)
            self.assertEqual(suggest_diffraction_start_updates(run), {})

    def test_preserves_existing_values(self):
        run = self.legacy_run()
        run.metadata['start'].update(data_type='scalar', detectors=['recorded-detector'])
        self.assertEqual(suggest_diffraction_start_updates(run), {})

    def test_dry_run_does_not_modify_metadata(self):
        run = self.legacy_run()
        before = copy.deepcopy(run.metadata)
        changes = repair_catalog_metadata({'legacy': run})
        self.assertEqual(changes[0]['start_updates']['data_type'], '1d')
        self.assertEqual(run.metadata, before)

    def test_apply_backs_up_before_write_and_is_idempotent(self):
        run = self.legacy_run()
        before = copy.deepcopy(run.metadata)
        with tempfile.TemporaryDirectory() as directory:
            backup = Path(directory) / 'backup.jsonl'

            def update(metadata):
                record = json.loads(backup.read_text())
                self.assertEqual(record['before_metadata'], before)
                run.metadata.update(metadata)

            run.update_metadata = Mock(side_effect=update)
            repair_catalog_metadata({'legacy': run}, backup_path=backup)
            self.assertEqual(backup.stat().st_mode & 0o777, 0o600)
            self.assertEqual(run.metadata['start']['detectors'], ['psd'])
            self.assertEqual(run.metadata['stop'], before['stop'])
            self.assertEqual(run.metadata['metadata_enrichment'][0]['start_fields'], ['data_type', 'detectors'])
            self.assertEqual(repair_catalog_metadata({'legacy': run}, backup_path=backup), [])
            run.update_metadata.assert_called_once()
