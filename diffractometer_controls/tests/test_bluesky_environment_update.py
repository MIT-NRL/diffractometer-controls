import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import diffractometer_controls.bluesky_environment_update as environment_update

from diffractometer_controls.bluesky_environment_update import (
    extract_json_document,
    extract_json_value,
    filter_pip_freeze,
    parse_pip_check_issues,
    parse_qserver_response,
    read_simple_env,
    redact_command,
    summarize_mamba_plan,
)


class BlueskyEnvironmentUpdateTests(unittest.TestCase):
    def test_main_rejects_unapproved_host_before_update_check(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(
                    environment_update,
                    "local_maintenance_allowed",
                    return_value=False,
                ),
                mock.patch.object(environment_update.UpdateContext, "check") as check,
            ):
                result = environment_update.main(
                    [
                        "check",
                        "--run-dir",
                        str(run_dir),
                        "--repo-root",
                        str(Path(directory)),
                    ]
                )

            self.assertEqual(result, 1)
            check.assert_not_called()
            state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("Local maintenance is disabled", state["message"])

    def test_summarizes_mamba_transaction(self):
        payload = {
            "actions": {
                "UNLINK": [
                    {"name": "tiled", "version": "0.2.11", "build_string": "old"},
                    {"name": "removed", "version": "1.0"},
                ],
                "LINK": [
                    {"name": "tiled", "version": "0.2.14", "build_string": "new"},
                    {"name": "opencv", "version": "5.0.0"},
                ],
            }
        }
        lines, changed = summarize_mamba_plan(payload)
        self.assertTrue(changed)
        self.assertIn("Update tiled: 0.2.11 -> 0.2.14", lines)
        self.assertIn("Install opencv: 5.0.0", lines)
        self.assertIn("Remove removed: 1.0", lines)

    def test_empty_mamba_transaction_is_noop(self):
        lines, changed = summarize_mamba_plan({"actions": {}})
        self.assertFalse(changed)
        self.assertEqual(lines, ["No Conda package changes are currently proposed."])

    def test_extracts_json_from_command_noise(self):
        payload = extract_json_document("prefix\n{\"success\": true, \"actions\": {}}\nsuffix")
        self.assertTrue(payload["success"])

    def test_extracts_json_array_from_command_noise(self):
        payload = extract_json_value('prefix\n[{"name": "nit"}]\nsuffix', list)
        self.assertEqual(payload, [{"name": "nit"}])

    def test_reads_simple_environment_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "values.env"
            path.write_text(
                "# comment\nexport FIRST=one\nSECOND=\"two words\"\nTHIRD='three'\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_simple_env(path),
                {"FIRST": "one", "SECOND": "two words", "THIRD": "three"},
            )

    def test_parses_qserver_response(self):
        result = parse_qserver_response(
            "Arguments: ['status']\n12:00 - MESSAGE:\n"
            "{'manager_state': 'idle', 'worker_environment_exists': True}"
        )
        self.assertEqual(result["manager_state"], "idle")
        self.assertTrue(result["worker_environment_exists"])

    def test_database_password_is_redacted(self):
        shown = redact_command(
            [
                "tiled",
                "catalog",
                "upgrade-database",
                "postgresql+asyncpg://tiled:secret@localhost:5432/catalog",
            ]
        )
        self.assertNotIn("secret", shown)
        self.assertIn("***", shown)

    def test_restore_requirements_include_only_pip_owned_packages(self):
        frozen = (
            "numpy==2.4.6\n"
            "neutron-imaging-tools @ git+https://example.test/nit.git@abc123\n"
            "adl2pydm==1.2.3\n"
        )
        selected = filter_pip_freeze(
            frozen,
            {"neutron-imaging-tools", "adl2pydm"},
        )
        self.assertEqual(
            selected,
            [
                "neutron-imaging-tools @ git+https://example.test/nit.git@abc123",
                "adl2pydm==1.2.3",
            ],
        )

    def test_pip_check_parser_ignores_incidental_warnings(self):
        output = (
            "WARNING: cache is not writable\n"
            "tomopy 1.14.4 has requirement numpy~=1.12, but you have numpy 2.4.6.\n"
        )
        self.assertEqual(
            parse_pip_check_issues(output),
            ["tomopy 1.14.4 has requirement numpy~=1.12, but you have numpy 2.4.6."],
        )


if __name__ == "__main__":
    unittest.main()
