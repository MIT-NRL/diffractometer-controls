import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import diffractometer_controls.bluesky_environment_update as environment_update

from diffractometer_controls.bluesky_environment_update import (
    UpdateContext,
    UpdateFailure,
    combined_plan_digest,
    explicit_snapshot_artifacts,
    explicit_snapshot_urls,
    extract_json_document,
    extract_json_value,
    find_duplicate_distribution_metadata,
    filter_pip_freeze,
    installed_conda_artifacts,
    is_conda_managed_pip_issue,
    parse_pip_check_issues,
    pip_check_issue_key,
    parse_qserver_response,
    read_simple_env,
    redact_command,
    summarize_mamba_plan,
)


class BlueskyEnvironmentUpdateTests(unittest.TestCase):
    @staticmethod
    def _write_distribution(prefix, directory_name, name, version):
        metadata_dir = (
            Path(prefix) / "lib" / "python3.13" / "site-packages" / directory_name
        )
        metadata_dir.mkdir(parents=True)
        (metadata_dir / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n\n",
            encoding="utf-8",
        )
        return metadata_dir

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
        self.assertIn("Upgrades (1):", lines)
        self.assertIn("  tiled 0.2.11 -> 0.2.14", lines)
        self.assertIn("New packages (1):", lines)
        self.assertIn("  opencv 5.0.0", lines)
        self.assertIn("Removals (1):", lines)
        self.assertIn("  removed 1.0", lines)

    def test_groups_downgrades_and_rebuilds(self):
        payload = {
            "actions": {
                "UNLINK": [
                    {"name": "numpy", "version": "2.4.0", "build_string": "old"},
                    {"name": "qt", "version": "6.8.0", "build_string": "old"},
                ],
                "LINK": [
                    {"name": "numpy", "version": "2.3.2", "build_string": "new"},
                    {"name": "qt", "version": "6.8.0", "build_string": "new"},
                ],
            }
        }
        lines, changed = summarize_mamba_plan(payload)
        self.assertTrue(changed)
        self.assertIn("Downgrades (1):", lines)
        self.assertIn("  numpy 2.4.0 -> 2.3.2", lines)
        self.assertIn("Rebuilds / unchanged version (1):", lines)
        self.assertIn("  qt 6.8.0 (old -> new)", lines)

    def test_empty_mamba_transaction_is_noop(self):
        lines, changed = summarize_mamba_plan({"actions": {}})
        self.assertFalse(changed)
        self.assertEqual(lines, ["No Conda package changes are currently proposed."])

    def test_identical_mamba_unlink_link_pair_is_ignored(self):
        package = {
            "name": "requests",
            "version": "2.34.2",
            "build_string": "pyhcf101f3_0",
            "sha256": "same-package-hash",
        }
        lines, changed = summarize_mamba_plan(
            {"actions": {"UNLINK": [package], "LINK": [dict(package)]}}
        )
        self.assertFalse(changed)
        self.assertEqual(lines, ["No Conda package changes are currently proposed."])

    def test_plan_digest_ignores_order_and_identical_relinks(self):
        first = {
            "actions": {
                "UNLINK": [
                    {"name": "same", "version": "1", "fn": "same-1.conda"},
                    {"name": "changed", "version": "1", "fn": "changed-1.conda"},
                ],
                "LINK": [
                    {"name": "same", "version": "1", "fn": "same-1.conda"},
                    {"name": "changed", "version": "2", "fn": "changed-2.conda"},
                ],
            }
        }
        reordered = {
            "actions": {
                "UNLINK": list(reversed(first["actions"]["UNLINK"])),
                "LINK": list(reversed(first["actions"]["LINK"])),
            }
        }
        self.assertEqual(
            combined_plan_digest(first, {}), combined_plan_digest(reordered, {})
        )
        changed_again = json.loads(json.dumps(first))
        changed_again["actions"]["LINK"][1]["version"] = "3"
        self.assertNotEqual(
            combined_plan_digest(first, {}), combined_plan_digest(changed_again, {})
        )

    def test_revalidation_rejects_a_changed_plan_before_snapshot(self):
        approved_environment = {"actions": {}}
        approved_update = {
            "actions": {"LINK": [{"name": "demo", "version": "2", "fn": "demo-2"}]}
        }
        changed_update = {
            "actions": {"LINK": [{"name": "demo", "version": "3", "fn": "demo-3"}]}
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            context.write_json_atomic(
                context.plan_path,
                {
                    "plan_digest": combined_plan_digest(
                        approved_environment, approved_update
                    ),
                    "metadata_repairs": [],
                },
            )
            try:
                with (
                    mock.patch.object(
                        context,
                        "_resolve_mamba_plan",
                        return_value=(approved_environment, changed_update),
                    ),
                    mock.patch.object(context, "_metadata_repairs", return_value=[]),
                ):
                    with self.assertRaisesRegex(UpdateFailure, "packages changed"):
                        context._revalidate_approved_plan()
            finally:
                context.close()

    def test_finds_only_safely_repairable_duplicate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            conda_meta = prefix / "conda-meta"
            conda_meta.mkdir()
            (conda_meta / "bluesky-1.15.1-test.json").write_text(
                json.dumps({"name": "bluesky", "version": "1.15.1"}),
                encoding="utf-8",
            )
            stale = self._write_distribution(
                prefix, "bluesky-0.0.0.dist-info", "bluesky", "0.0.0"
            )
            self._write_distribution(
                prefix, "bluesky-1.15.1.dist-info", "bluesky", "1.15.1"
            )

            repairs, ambiguous = find_duplicate_distribution_metadata(prefix)

            self.assertEqual(ambiguous, [])
            self.assertEqual(len(repairs), 1)
            self.assertEqual(repairs[0]["remove_version"], "0.0.0")
            self.assertEqual(Path(repairs[0]["path"]), stale)

    def test_ambiguous_duplicate_metadata_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            (prefix / "conda-meta").mkdir()
            self._write_distribution(prefix, "demo-1.dist-info", "demo", "1")
            self._write_distribution(prefix, "demo-2.dist-info", "demo", "2")
            repairs, ambiguous = find_duplicate_distribution_metadata(prefix)
            self.assertEqual(repairs, [])
            self.assertEqual(ambiguous[0]["name"], "demo")

    def test_metadata_repair_is_backed_up_before_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = root / "env"
            stale = self._write_distribution(
                prefix, "demo-0.dist-info", "demo", "0"
            )
            context = UpdateContext(root / "run", root)
            context.env_prefix = prefix
            context._manifest = {"metadata_backups": []}
            try:
                context._repair_duplicate_metadata(
                    [
                        {
                            "name": "demo",
                            "keep_version": "1",
                            "remove_version": "0",
                            "path": str(stale),
                        }
                    ]
                )
            finally:
                context.close()
            self.assertFalse(stale.exists())
            backup = Path(context._manifest["metadata_backups"][0]["backup"])
            self.assertTrue((backup / "METADATA").exists())

    def test_command_timeout_terminates_a_silent_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            started = time.monotonic()
            try:
                with self.assertRaisesRegex(UpdateFailure, "timed out"):
                    context.run(
                        [sys.executable, "-c", "import time; time.sleep(10)"],
                        capture=True,
                        timeout=0.1,
                    )
            finally:
                context.close()
            self.assertLess(time.monotonic() - started, 2)

    def test_command_heartbeat_is_written_and_cleared(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            context.set_state("test", "running", "Testing command", 10)
            try:
                with mock.patch.object(environment_update, "HEARTBEAT_INTERVAL", 0.02):
                    context.run(
                        [sys.executable, "-c", "import time; time.sleep(0.08)"],
                        capture=True,
                        timeout=2,
                    )
            finally:
                context.close()
            state = json.loads(context.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["message"], "Testing command")
            self.assertNotIn("command_elapsed_seconds", state)
            self.assertIn("Still running", context.log_path.read_text(encoding="utf-8"))

    def test_failed_apply_runs_automatic_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            with (
                mock.patch.object(environment_update, "local_maintenance_allowed", return_value=True),
                mock.patch.object(environment_update, "acquire_update_lock", return_value=None),
                mock.patch.object(
                    environment_update.UpdateContext,
                    "apply",
                    side_effect=UpdateFailure("apply broke"),
                ),
                mock.patch.object(
                    environment_update.UpdateContext,
                    "recover_failed_apply",
                    return_value="success",
                ) as recover,
                mock.patch.object(Path, "exists", return_value=True),
            ):
                result = environment_update.main(
                    ["apply", "--run-dir", str(run_dir), "--repo-root", str(directory)]
                )
            self.assertEqual(result, 1)
            recover.assert_called_once()
            state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["automatic_recovery"], "success")
            self.assertFalse(state["restore_available"])

    def test_automatic_recovery_rolls_back_services_without_restarting_gui(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            context.write_json_atomic(
                context.manifest_path,
                {
                    "services_stop_started": True,
                    "environment_update_started": True,
                    "databases_requiring_restore": ["catalog"],
                },
            )
            try:
                with (
                    mock.patch.object(context, "_stop_services") as stop,
                    mock.patch.object(context, "_restore_environment") as restore_env,
                    mock.patch.object(context, "_restore_databases") as restore_db,
                    mock.patch.object(context, "_smoke_test_environment") as smoke,
                    mock.patch.object(context, "_start_services") as start,
                    mock.patch.object(context, "_open_worker") as open_worker,
                ):
                    result = context.recover_failed_apply()
            finally:
                context.close()
            self.assertEqual(result, "success")
            stop.assert_called_once()
            restore_env.assert_called_once()
            restore_db.assert_called_once()
            smoke.assert_called_once()
            start.assert_called_once()
            open_worker.assert_called_once()
            manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["automatic_recovery_success"])

    def test_apply_finishes_services_and_waits_for_operator_gui_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            context._manifest = {}
            try:
                with (
                    mock.patch.object(context, "ensure_no_active_run"),
                    mock.patch.object(
                        context, "_revalidate_approved_plan", return_value=[]
                    ),
                    mock.patch.object(context, "_record_restore_point"),
                    mock.patch.object(context, "_backup_databases"),
                    mock.patch.object(context, "_close_worker"),
                    mock.patch.object(context, "_stop_services"),
                    mock.patch.object(context, "_repair_duplicate_metadata"),
                    mock.patch.object(context, "_update_environment"),
                    mock.patch.object(context, "_smoke_test_environment"),
                    mock.patch.object(context, "_migrate_databases"),
                    mock.patch.object(context, "_start_services") as start,
                    mock.patch.object(context, "_open_worker") as open_worker,
                    mock.patch.object(context, "_restart_gui") as restart_gui,
                ):
                    context.apply()
            finally:
                context.close()

            start.assert_called_once()
            open_worker.assert_called_once()
            restart_gui.assert_not_called()
            state = json.loads(context.state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "success")
            self.assertTrue(state["gui_restart_required"])

    def test_operator_restart_mode_restarts_gui_and_clears_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            context.write_json_atomic(context.manifest_path, {"success": True})
            try:
                with mock.patch.object(context, "_restart_gui") as restart_gui:
                    context.restart_gui()
            finally:
                context.close()

            restart_gui.assert_called_once()
            state = json.loads(context.state_path.read_text(encoding="utf-8"))
            self.assertFalse(state["gui_restart_required"])
            manifest = json.loads(context.manifest_path.read_text(encoding="utf-8"))
            self.assertIn("gui_restarted_at", manifest)

    def test_gui_restart_uses_scope_independent_of_updater_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            context.gui_script.parent.mkdir(parents=True, exist_ok=True)
            context.gui_script.write_text("#!/usr/bin/env perl\n", encoding="utf-8")
            try:
                with mock.patch.object(context, "run") as run:
                    context._restart_gui()
            finally:
                context.close()

            run.assert_called_once_with(
                [
                    "/usr/bin/systemd-run",
                    "--user",
                    "--scope",
                    "--quiet",
                    "/usr/bin/perl",
                    str(context.gui_script),
                    "restart",
                ],
                timeout=60,
            )

    def test_explicit_snapshot_artifacts_decodes_conda_filenames(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "explicit.txt"
            snapshot.write_text(
                "# generated\n@EXPLICIT\n"
                "https://example.test/linux-64/x264-1%21164-build.tar.bz2#hash\n",
                encoding="utf-8",
            )
            self.assertEqual(
                explicit_snapshot_artifacts(snapshot),
                {"x264-1!164-build.tar.bz2"},
            )

    def test_mamba_explicit_list_header_is_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "explicit.txt"
            snapshot.write_text(
                'List of packages in environment: "/tmp/demo"\n\n'
                "https://example.test/linux-64/demo-1-build.conda\n",
                encoding="utf-8",
            )
            self.assertEqual(
                explicit_snapshot_urls(snapshot),
                ["https://example.test/linux-64/demo-1-build.conda"],
            )
            self.assertEqual(
                explicit_snapshot_artifacts(snapshot), {"demo-1-build.conda"}
            )

    def test_installed_conda_artifacts_reads_package_records(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            conda_meta = prefix / "conda-meta"
            conda_meta.mkdir()
            (conda_meta / "demo.json").write_text(
                json.dumps({"name": "demo", "fn": "demo-1-build.conda"}),
                encoding="utf-8",
            )
            self.assertEqual(
                installed_conda_artifacts(prefix), {"demo-1-build.conda"}
            )

    def test_restore_recreates_exact_snapshot_with_mamba_and_keeps_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            prefix = root / "env"
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "old-marker").write_text("old", encoding="utf-8")
            (prefix / "conda-meta" / "demo-old.json").write_text(
                json.dumps({"name": "demo", "fn": "demo-2-build.conda"}),
                encoding="utf-8",
            )
            snapshot = context.run_dir / "conda-explicit.txt"
            snapshot.write_text(
                "@EXPLICIT\nhttps://example.test/linux-64/demo-1-build.conda\n",
                encoding="utf-8",
            )
            context.env_prefix = prefix
            context.env_python = prefix / "bin" / "python"
            context._manifest = {"conda_explicit": str(snapshot)}

            def create_environment(command, **_kwargs):
                self.assertEqual(command[0], str(environment_update.DEFAULT_MAMBA))
                self.assertEqual(command[1], "create")
                (prefix / "conda-meta").mkdir(parents=True)
                (prefix / "conda-meta" / "demo.json").write_text(
                    json.dumps({"name": "demo", "fn": "demo-1-build.conda"}),
                    encoding="utf-8",
                )

            try:
                with (
                    mock.patch.object(context, "run", side_effect=create_environment) as run,
                    mock.patch.object(context, "_verify_pip_consistency"),
                ):
                    context._restore_environment()
            finally:
                context.close()

            run.assert_called_once()
            backup = root / "run" / "environment-before-restore"
            self.assertEqual((backup / "old-marker").read_text(encoding="utf-8"), "old")
            self.assertEqual(installed_conda_artifacts(prefix), {"demo-1-build.conda"})

    def test_failed_mamba_recreation_restores_original_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            context = UpdateContext(root / "run", root)
            prefix = root / "env"
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "old-marker").write_text("old", encoding="utf-8")
            (prefix / "conda-meta" / "demo-old.json").write_text(
                json.dumps({"name": "demo", "fn": "demo-2-build.conda"}),
                encoding="utf-8",
            )
            snapshot = context.run_dir / "conda-explicit.txt"
            snapshot.write_text(
                "@EXPLICIT\nhttps://example.test/linux-64/demo-1-build.conda\n",
                encoding="utf-8",
            )
            context.env_prefix = prefix
            context.env_python = prefix / "bin" / "python"
            context._manifest = {"conda_explicit": str(snapshot)}
            try:
                with mock.patch.object(
                    context, "run", side_effect=UpdateFailure("create failed")
                ):
                    with self.assertRaisesRegex(UpdateFailure, "create failed"):
                        context._restore_environment()
            finally:
                context.close()

            self.assertEqual((prefix / "old-marker").read_text(encoding="utf-8"), "old")
            self.assertFalse((root / "run" / "environment-before-restore").exists())

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

    def test_pip_check_issue_key_ignores_requiring_package_version(self):
        before = (
            "apstools 1.7.10 has requirement bluesky!=1.11.0,>=1.6.2, "
            "but you have bluesky 0.0.0."
        )
        after = (
            "apstools 1.7.11 has requirement bluesky!=1.11.0,>=1.6.2, "
            "but you have bluesky 0.0.0."
        )
        self.assertEqual(pip_check_issue_key(before), pip_check_issue_key(after))

    def test_pip_check_issue_between_conda_packages_is_not_a_pip_failure(self):
        issue = (
            "tomopy 1.14.4 has requirement numpy!=1.22.4,~=1.12, "
            "but you have numpy 2.5.2."
        )
        self.assertTrue(is_conda_managed_pip_issue(issue, {"tomopy", "numpy"}))
        self.assertFalse(is_conda_managed_pip_issue(issue, {"numpy"}))

    def test_pip_owned_dependency_conflict_remains_a_failure(self):
        issue = (
            "beamline-addon 1.0 has requirement numpy<2, but you have numpy 2.5.2."
        )
        self.assertFalse(is_conda_managed_pip_issue(issue, {"numpy"}))


if __name__ == "__main__":
    unittest.main()
