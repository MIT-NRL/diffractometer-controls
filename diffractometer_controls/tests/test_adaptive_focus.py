import unittest

from diffractometer_controls.adaptive_focus import (
    FocusAdaptiveSessionStore,
    FocusScanSpec,
    build_focus_adaptive_metadata,
    normalize_focus_command,
)


class FocusAdaptiveSessionStoreTests(unittest.TestCase):
    def test_session_lifecycle(self):
        store = FocusAdaptiveSessionStore(now_func=lambda: 123.0)
        session_id = store.create(initial_state={"done_units": 0})

        snap = store.get_session(session_id)
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["status"], "created")
        self.assertEqual(snap["queued_commands"], 0)
        self.assertEqual(snap["state"]["done_units"], 0)

        store.update(session_id, status="awaiting_command", state_update={"done_units": 2})
        store.add_history(session_id, "coarse_scan_complete", {"done_units": 2})
        resp = store.submit_command(session_id, "complete", {"reason": "test"})
        self.assertTrue(resp["ok"])
        self.assertEqual(resp["queued_commands"], 1)

        cmd = store.pop_command(session_id)
        self.assertEqual(cmd["command"], "complete")
        self.assertEqual(cmd["payload"]["reason"], "test")

        snap = store.get_session(session_id)
        self.assertTrue(snap["ok"])
        self.assertEqual(snap["status"], "awaiting_command")
        self.assertEqual(snap["queued_commands"], 0)
        self.assertEqual(snap["history_count"], 1)
        self.assertEqual(snap["state"]["done_units"], 2)

        store.delete(session_id)
        self.assertFalse(store.get_session(session_id)["ok"])

    def test_missing_session_responses(self):
        store = FocusAdaptiveSessionStore()
        self.assertFalse(store.get_session("missing")["ok"])
        self.assertFalse(store.submit_command("missing", "complete")["ok"])
        self.assertIsNone(store.pop_command("missing"))


class FocusScanSpecTests(unittest.TestCase):
    def test_explicit_bounds(self):
        spec = FocusScanSpec.from_inputs(start_pos=1.0, stop_pos=2.0, num_steps=3)
        self.assertEqual(spec.scan_mode, "explicit_bounds")
        self.assertEqual(spec.positions, (1.0, 1.5, 2.0))
        self.assertEqual(spec.step_size, 0.5)

    def test_guess_range(self):
        spec = FocusScanSpec.from_inputs(focus_guess=10.0, scan_half_range=2.0, num_steps=5)
        self.assertEqual(spec.scan_mode, "guess_range")
        self.assertEqual(spec.start_pos, 8.0)
        self.assertEqual(spec.stop_pos, 12.0)
        self.assertEqual(spec.positions, (8.0, 9.0, 10.0, 11.0, 12.0))

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            FocusScanSpec.from_inputs(start_pos=2.0, stop_pos=1.0, num_steps=3)
        with self.assertRaises(ValueError):
            FocusScanSpec.from_inputs(focus_guess=1.0, scan_half_range=-1.0, num_steps=3)
        with self.assertRaises(ValueError):
            FocusScanSpec.from_inputs(num_steps=3)


class FocusCommandTests(unittest.TestCase):
    def test_terminal_commands(self):
        complete = normalize_focus_command("complete")
        abort = normalize_focus_command("abort")
        self.assertEqual(complete.kind, "terminal")
        self.assertEqual(complete.status, "completed")
        self.assertEqual(abort.kind, "terminal")
        self.assertEqual(abort.status, "aborted")

    def test_go_to_focus(self):
        action = normalize_focus_command("go_to_focus", {"target_position": 3.5})
        self.assertEqual(action.kind, "acquire")
        self.assertEqual(action.positions, (3.5,))
        self.assertEqual(action.state_update["last_position"], 3.5)

        fallback = normalize_focus_command("go_to_focus", {}, current_position=4.0)
        self.assertEqual(fallback.positions, (4.0,))

    def test_scan_around_focus(self):
        action = normalize_focus_command(
            "scan_around_focus",
            {"center": 5.0, "step_size": 0.25, "num_points": 4},
            coarse_step_size=1.0,
        )
        self.assertEqual(action.kind, "acquire")
        self.assertEqual(action.positions, (4.5, 4.75, 5.0, 5.25, 5.5))

        ignored = normalize_focus_command("scan_around_focus", {"step_size": 0.0})
        self.assertEqual(ignored.kind, "ignore")
        self.assertEqual(ignored.reason, "bad_step")

    def test_extend_commands(self):
        left = normalize_focus_command("extend_left", {"num_points": 2}, coarse_step_size=0.5, left_bound=1.0)
        right = normalize_focus_command("extend_right", {"num_points": 2}, coarse_step_size=0.5, right_bound=2.0)
        self.assertEqual(left.positions, (0.5, 0.0))
        self.assertEqual(right.positions, (2.5, 3.0))

        ignored = normalize_focus_command("extend_left", {}, coarse_step_size=None, left_bound=1.0)
        self.assertEqual(ignored.kind, "ignore")
        self.assertEqual(ignored.reason, "no_coarse_step")

    def test_unknown_command(self):
        action = normalize_focus_command("dance")
        self.assertEqual(action.kind, "ignore")
        self.assertEqual(action.reason, "unknown_command")


class FocusMetadataTests(unittest.TestCase):
    def test_metadata_shape_preserved(self):
        spec = FocusScanSpec.from_inputs(start_pos=1.0, stop_pos=2.0, num_steps=3)
        md = build_focus_adaptive_metadata(
            file_name="scan",
            file_dir="dir",
            detector_names=["cam1"],
            detector_config={"exposure_time": 1.0},
            motor_name="cam1_focus",
            scan_spec=spec,
            session_id="session-1",
            total_time=3.0,
            total_units=3,
            plan_patterns_module="bluesky.plan_patterns",
            md={"sample": "sample-a"},
        )
        self.assertEqual(md["plan_name"], "adaptive_imaging_focus_scan")
        self.assertEqual(md["focus_adaptive"]["session_id"], "session-1")
        self.assertEqual(md["focus_adaptive"]["command_submit_fn"], "adaptive_focus_submit_command")
        self.assertEqual(md["focus_adaptive"]["command_state_fn"], "adaptive_focus_get_session")
        self.assertIn("complete", md["focus_adaptive"]["accepted_commands"])
        self.assertEqual(md["sample"], "sample-a")


if __name__ == "__main__":
    unittest.main()
