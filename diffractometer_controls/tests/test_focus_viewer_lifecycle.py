import os
import inspect
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from diffractometer_controls import (
    focus_offline_viewer,
    focus_online_viewer,
    main_window as main_window_module,
)
from diffractometer_controls.focus_offline_viewer import FocusOfflineWindow, FrameInfo
from diffractometer_controls.focus_online_viewer import (
    FocusOnlineBridge,
    QueueServerAdaptiveClient,
    _has_observed_run_activity,
    _has_received_focus_frame,
)
from diffractometer_controls.main_window import MITRMainWindow


class FocusViewerInterpreterTests(unittest.TestCase):
    def test_ioc_panel_reference_is_retained_and_reused(self):
        panel = mock.Mock()
        panel.destroyed = mock.Mock()
        harness = SimpleNamespace(
            _ioc_control_panel=None,
            macros={"P": "4dh4:"},
            _ioc_control_panel_destroyed=mock.Mock(),
        )

        with patch(
            "diffractometer_controls.main_window.load_file", return_value=panel
        ) as loader:
            MITRMainWindow.launch_ioc_control_panel(harness)
            self.assertIs(harness._ioc_control_panel, panel)
            panel.destroyed.connect.assert_called_once_with(
                harness._ioc_control_panel_destroyed
            )

            MITRMainWindow.launch_ioc_control_panel(harness)

        loader.assert_called_once()
        panel.show.assert_called_once()
        panel.raise_.assert_called_once()
        panel.activateWindow.assert_called_once()

    def test_uses_running_interpreter_instead_of_stale_conda_prefix(self):
        running_python = "/opt/controls-env/bin/python"
        with (
            patch.object(sys, "executable", running_python),
            patch.dict(os.environ, {"CONDA_PREFIX": "/opt/stale-base"}),
        ):
            selected = MITRMainWindow._resolve_focus_python_executable(None)

        self.assertEqual(selected, Path(running_python))

    def test_forwarded_x_display_forces_xcb_and_drops_wayland_target(self):
        source = {
            "DISPLAY": "localhost:10.0",
            "XAUTHORITY": "/tmp/ssh-cookie",
            "SSH_CONNECTION": "192.0.2.10 50000 192.0.2.20 22",
            "WAYLAND_DISPLAY": "wayland-0",
            "QT_QPA_PLATFORM": "wayland",
        }

        child = MITRMainWindow._analysis_launcher_environment(source)

        self.assertEqual(child["DISPLAY"], "localhost:10.0")
        self.assertEqual(child["XAUTHORITY"], "/tmp/ssh-cookie")
        self.assertEqual(child["QT_QPA_PLATFORM"], "xcb")
        self.assertEqual(child["QT_XCB_GL_INTEGRATION"], "none")
        self.assertEqual(child["QT_OPENGL"], "software")
        self.assertEqual(child["LIBGL_ALWAYS_SOFTWARE"], "1")
        self.assertNotIn("WAYLAND_DISPLAY", child)

    def test_analysis_display_environment_can_be_overridden(self):
        source = {
            "DISPLAY": ":0",
            "MITR_ANALYSIS_DISPLAY": "localhost:12.0",
            "MITR_ANALYSIS_XAUTHORITY": "/tmp/alternate-cookie",
            "MITR_ANALYSIS_QT_QPA_PLATFORM": "xcb",
            "WAYLAND_DISPLAY": "wayland-0",
        }

        child = MITRMainWindow._analysis_launcher_environment(source)

        self.assertEqual(child["DISPLAY"], "localhost:12.0")
        self.assertEqual(child["XAUTHORITY"], "/tmp/alternate-cookie")
        self.assertEqual(child["QT_QPA_PLATFORM"], "xcb")
        self.assertEqual(child["QT_XCB_GL_INTEGRATION"], "none")
        self.assertEqual(child["QT_OPENGL"], "software")
        self.assertEqual(child["LIBGL_ALWAYS_SOFTWARE"], "1")
        self.assertNotIn("WAYLAND_DISPLAY", child)

    def test_forwarded_analysis_rendering_can_be_overridden(self):
        source = {
            "DISPLAY": "localhost:10.0",
            "SSH_CONNECTION": "192.0.2.10 50000 192.0.2.20 22",
            "MITR_ANALYSIS_QT_XCB_GL_INTEGRATION": "xcb_glx",
            "MITR_ANALYSIS_QT_OPENGL": "desktop",
            "MITR_ANALYSIS_LIBGL_ALWAYS_SOFTWARE": "0",
        }

        child = MITRMainWindow._analysis_launcher_environment(source)

        self.assertEqual(child["QT_XCB_GL_INTEGRATION"], "xcb_glx")
        self.assertEqual(child["QT_OPENGL"], "desktop")
        self.assertEqual(child["LIBGL_ALWAYS_SOFTWARE"], "0")

    def test_local_analysis_launch_does_not_force_software_rendering(self):
        child = MITRMainWindow._analysis_launcher_environment(
            {"DISPLAY": ":0", "QT_QPA_PLATFORM": "xcb"}
        )

        self.assertNotIn("QT_XCB_GL_INTEGRATION", child)
        self.assertNotIn("QT_OPENGL", child)
        self.assertNotIn("LIBGL_ALWAYS_SOFTWARE", child)

    def test_neutron_imaging_gui_uses_running_interpreter_module(self):
        status_bar = mock.Mock()
        harness = SimpleNamespace(
            _resolve_focus_python_executable=lambda: Path("/opt/controls-env/bin/python"),
            statusBar=lambda: status_bar,
        )
        process = SimpleNamespace(pid=4321)

        with (
            patch("diffractometer_controls.main_window.importlib.util.find_spec", return_value=object()),
            patch("diffractometer_controls.main_window.subprocess.Popen", return_value=process) as popen,
        ):
            MITRMainWindow.launch_neutron_imaging_gui(harness)

        popen.assert_called_once()
        self.assertEqual(
            popen.call_args.args[0],
            ["/opt/controls-env/bin/python", "-m", "neutron_imaging_gui"],
        )
        status_bar.showMessage.assert_called_once_with(
            "Launched Neutron Imaging GUI (pid=4321)", 3000
        )

    def test_neutron_imaging_gui_missing_package_warns_without_launch(self):
        harness = SimpleNamespace()

        with (
            patch("diffractometer_controls.main_window.importlib.util.find_spec", return_value=None),
            patch("diffractometer_controls.main_window.subprocess.Popen") as popen,
            patch("diffractometer_controls.main_window.QMessageBox.warning") as warning,
        ):
            MITRMainWindow.launch_neutron_imaging_gui(harness)

        popen.assert_not_called()
        warning.assert_called_once()


class FocusViewerStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_bring_to_front_does_not_recreate_native_x11_window(self):
        window = mock.Mock()
        window.windowState.return_value = QtCore.Qt.WindowNoState
        scheduled = []

        with patch.object(
            focus_online_viewer.QtCore.QTimer,
            "singleShot",
            side_effect=lambda _delay, callback: scheduled.append(callback),
        ):
            FocusOnlineBridge._bring_window_to_front(window)
            for callback in scheduled:
                callback()

        window.setWindowFlag.assert_not_called()
        self.assertEqual(scheduled, [])
        self.assertGreaterEqual(window.show.call_count, 1)
        self.assertGreaterEqual(window.raise_.call_count, 1)
        self.assertGreaterEqual(window.activateWindow.call_count, 1)

    def test_background_document_callback_runs_handler_on_qt_thread(self):
        received = threading.Event()

        class ProbeBridge(FocusOnlineBridge):
            def _on_document_received(self, _name, _document):
                self.received_thread = QtCore.QThread.currentThread()
                received.set()

        bridge = ProbeBridge()
        worker = threading.Thread(
            target=lambda: bridge.on_document("start", {"uid": "run-1"})
        )
        worker.start()
        worker.join(timeout=2)

        deadline = time.monotonic() + 2.0
        while not received.is_set() and time.monotonic() < deadline:
            self.app.processEvents()

        self.assertTrue(received.is_set())
        self.assertIs(bridge.received_thread, bridge.thread())

    def test_adaptive_launch_deduplication_occurs_in_gui_slot(self):
        harness = SimpleNamespace(
            _should_ignore_adaptive_focus_launch=mock.Mock(
                side_effect=[False, True]
            ),
            _launch_focus_online_viewer=mock.Mock(),
        )

        for _ in range(2):
            MITRMainWindow._on_adaptive_focus_plan_started(
                harness,
                "session-1",
                "run-1",
                "focus_file",
                "focus_dir",
                "sim_focus_motor_position",
                15,
            )

        harness._launch_focus_online_viewer.assert_called_once_with(
            session_id="session-1", run_uid="run-1"
        )
        self.assertEqual(harness._focus_online_file_name, "focus_file")
        self.assertEqual(harness._focus_online_file_dir, "focus_dir")
        self.assertEqual(
            harness._focus_online_motor_key, "sim_focus_motor_position"
        )
        self.assertEqual(harness._focus_online_expected_frame_count, 15)

    def test_adaptive_start_forwards_expected_coarse_frame_count(self):
        emitted = mock.Mock()
        harness = SimpleNamespace(
            adaptive_focus_plan_started=SimpleNamespace(emit=emitted)
        )

        MITRMainWindow._on_focus_bluesky_doc(
            harness,
            "start",
            {
                "uid": "run-1",
                "plan_name": "adaptive_imaging_focus_scan",
                "file_name": "focus_file",
                "file_dir": "focus_dir",
                "plan_pattern_args": {"num_steps": 15},
                "focus_adaptive": {
                    "session_id": "session-1",
                    "motor_event_key": "sim_focus_motor_position",
                },
            },
        )

        emitted.assert_called_once_with(
            "session-1",
            "run-1",
            "focus_file",
            "focus_dir",
            "sim_focus_motor_position",
            15,
        )

    def test_adaptive_launch_claim_is_process_wide(self):
        session_id = f"process-wide-{time.monotonic_ns()}"
        first_window = SimpleNamespace(_focus_launched_sessions=set())
        second_window = SimpleNamespace(_focus_launched_sessions=set())
        try:
            self.assertFalse(
                MITRMainWindow._should_ignore_adaptive_focus_launch(
                    first_window, session_id, "run-1"
                )
            )
            self.assertTrue(
                MITRMainWindow._should_ignore_adaptive_focus_launch(
                    second_window, session_id, "run-1"
                )
            )
        finally:
            with main_window_module._FOCUS_LAUNCH_GUARD:
                main_window_module._FOCUS_LAUNCHED_SESSIONS.discard(session_id)

    def test_window_without_frames_has_not_completed_startup(self):
        bridge = SimpleNamespace(window=SimpleNamespace(frames=[]))

        self.assertFalse(_has_received_focus_frame(bridge))

    def test_first_valid_frame_completes_startup(self):
        bridge = SimpleNamespace(window=SimpleNamespace(frames=[object()]))

        self.assertTrue(_has_received_focus_frame(bridge))

    def test_missing_window_has_not_completed_startup(self):
        self.assertFalse(_has_received_focus_frame(SimpleNamespace(window=None)))

    def test_bridge_does_not_create_an_eager_placeholder_window(self):
        bridge = FocusOnlineBridge()

        self.assertIsNone(bridge.window)

    def test_start_document_keeps_a_frameless_window_without_reopening(self):
        """The window opened at plan start must not be closed by the start doc."""
        window = SimpleNamespace(frames=[], close=mock.Mock())
        bridge = FocusOnlineBridge(run_uid="run-1")
        bridge.window = window
        bridge._log_received = mock.Mock()

        bridge._on_document_received("start", {"uid": "run-1"})

        window.close.assert_not_called()
        self.assertIs(bridge.window, window)

    def test_start_document_keeps_placeholder_even_without_a_uid_filter(self):
        """Viewers launched without --run-uid must not flicker either."""
        window = SimpleNamespace(frames=[], close=mock.Mock())
        bridge = FocusOnlineBridge()
        bridge.window = window
        bridge._log_received = mock.Mock()

        bridge._on_document_received("start", {"uid": "run-xyz"})

        window.close.assert_not_called()
        self.assertIs(bridge.window, window)

    def test_new_run_with_frames_still_resets_the_viewer(self):
        window = SimpleNamespace(frames=[object()], close=mock.Mock())
        bridge = FocusOnlineBridge()
        bridge.window = window
        bridge._log_received = mock.Mock()

        bridge._on_document_received("start", {"uid": "run-2"})

        window.close.assert_called_once()
        self.assertIsNone(bridge.window)

    def test_show_window_now_creates_the_window_before_any_frame(self):
        bridge = FocusOnlineBridge()
        created = SimpleNamespace(
            statusBar=lambda: SimpleNamespace(showMessage=mock.Mock()),
        )

        def _fake_ensure():
            bridge.window = created

        bridge._ensure_window = _fake_ensure

        bridge.show_window_now()

        self.assertIs(bridge.window, created)

    def test_start_document_marks_the_run_as_live_before_any_frame(self):
        """A long first exposure must not look like a dead session."""
        bridge = FocusOnlineBridge(run_uid="run-1")
        bridge._log_received = mock.Mock()

        self.assertFalse(_has_observed_run_activity(bridge))

        bridge._on_document_received("start", {"uid": "run-1"})

        self.assertTrue(_has_observed_run_activity(bridge))
        # No frame has arrived, but the startup guard must not exit.
        self.assertFalse(_has_received_focus_frame(bridge))

    def test_start_document_sets_expected_frame_count_for_local_attachment(self):
        bridge = FocusOnlineBridge(
            run_uid="run-1", reset_viewer_on_new_run=False
        )
        bridge._log_received = mock.Mock()
        window = SimpleNamespace(set_expected_stream_frames=mock.Mock())
        bridge.window = window

        bridge._on_document_received(
            "start",
            {
                "uid": "run-1",
                "plan_pattern_args": {"num_steps": 15},
            },
        )

        self.assertEqual(bridge.expected_frame_count, 15)
        window.set_expected_stream_frames.assert_called_once_with(15)

    def test_unrelated_run_does_not_mark_activity(self):
        bridge = FocusOnlineBridge(run_uid="run-1")
        bridge._log_received = mock.Mock()

        bridge._on_document_received("start", {"uid": "other-run"})

        self.assertFalse(_has_observed_run_activity(bridge))


class FocusViewerAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_online_and_offline_share_theme_implementation(self):
        self.assertIs(
            focus_online_viewer._apply_saved_theme,
            focus_offline_viewer._apply_saved_theme,
        )

    def test_saved_dark_theme_is_applied_to_offline_application(self):
        original_palette = self.app.palette()
        settings = mock.Mock()
        settings.value.return_value = "dark"
        try:
            with patch.object(focus_offline_viewer.QtCore, "QSettings", return_value=settings):
                focus_offline_viewer._apply_saved_theme(self.app)
            window_color = self.app.palette().color(
                focus_offline_viewer.QtGui.QPalette.Window
            )
            self.assertLess(window_color.lightness(), 128)
        finally:
            self.app.setPalette(original_palette)

    def test_online_and_offline_processing_defaults_match(self):
        offline = focus_offline_viewer.build_arg_parser().parse_args([])
        online = focus_online_viewer.build_arg_parser().parse_args([])

        for name in (
            "max_workers_total",
            "bulk_workers",
            "full_workers",
            "full_cache_gb",
            "preprocess_mode",
            "preprocess_size",
        ):
            with self.subTest(option=name):
                self.assertEqual(getattr(offline, name), getattr(online, name))

    def test_online_bridge_and_offline_window_processing_defaults_match(self):
        offline = inspect.signature(FocusOfflineWindow.__init__).parameters
        online = inspect.signature(FocusOnlineBridge.__init__).parameters

        for name in (
            "max_workers_total",
            "bulk_workers",
            "full_workers",
            "full_cache_gb",
            "preprocess_mode",
            "preprocess_size",
        ):
            with self.subTest(option=name):
                self.assertEqual(offline[name].default, online[name].default)


class FocusMotorKeyTests(unittest.TestCase):
    @staticmethod
    def _descriptor():
        return {
            "data_keys": {
                "sim_focus_cam_stats1_total": {
                    "dtype": "number", "shape": [], "object_name": "sim_focus_cam"
                },
                "sim_focus_cam_blur_sigma": {
                    "dtype": "number", "shape": [], "object_name": "sim_focus_cam"
                },
                "sim_focus_motor": {
                    "dtype": "number", "shape": [], "object_name": "sim_focus_motor"
                },
                "sim_focus_motor_setpoint": {
                    "dtype": "number", "shape": [], "object_name": "sim_focus_motor"
                },
                "sim_focus_motor_position": {
                    "dtype": "number", "shape": [], "object_name": "sim_focus_motor_position"
                },
            },
            "object_keys": {
                "sim_focus_cam": ["sim_focus_cam_stats1_total", "sim_focus_cam_blur_sigma"],
                "sim_focus_motor": ["sim_focus_motor", "sim_focus_motor_setpoint"],
                "sim_focus_motor_position": ["sim_focus_motor_position"],
            },
        }

    def test_auto_detection_does_not_use_focus_detector_metric(self):
        harness = SimpleNamespace(motor_key=None)

        key = FocusOnlineBridge._descriptor_motor_key_from_doc(harness, self._descriptor())

        self.assertEqual(key, "sim_focus_motor_position")

    def test_metadata_motor_event_key_is_selected_exactly(self):
        harness = SimpleNamespace(motor_key="sim_focus_motor_position")

        key = FocusOnlineBridge._descriptor_motor_key_from_doc(harness, self._descriptor())

        self.assertEqual(key, "sim_focus_motor_position")


class FocusFitSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_full_filter_pool_uses_spawn_context(self):
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            self.assertEqual(window._full_mp_context.get_start_method(), "spawn")
            window.warm_full_process_pool()
            self.assertEqual(len(window._full_warmup_futures), 1)
            self.assertGreater(window._full_warmup_futures[0].result(timeout=10), 0)
            self.assertEqual(window._full_process_pool.submit(sum, [1, 2, 3]).result(timeout=10), 6)
        finally:
            window.close()

    def test_full_filter_pool_falls_back_to_threads_after_submit_io_error(self):
        class FailingExecutor:
            def submit(self, *_args, **_kwargs):
                raise OSError(5, "Input/output error")

            def shutdown(self, **_kwargs):
                return None

        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        original = window._full_process_pool
        try:
            original.shutdown(wait=True, cancel_futures=True)
            window._full_process_pool = FailingExecutor()
            window._full_executor_kind = "process"
            window._full_warmup_attempted = False

            window.warm_full_process_pool()

            self.assertEqual(window._full_executor_kind, "thread")
            self.assertEqual(len(window._full_warmup_futures), 1)
            self.assertGreater(window._full_warmup_futures[0].result(timeout=10), 0)
        finally:
            window.close()

    def test_filter_progress_bar_tracks_completed_work_not_backlog(self):
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(3)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            window._full_prepared_indices = {0}
            window._seen_frame_indices = {0, 1, 2}
            window._full_queue.extend([1, 2])
            window._full_queued_indices.update({1, 2})

            window._update_filter_queue_indicator()

            self.assertEqual(window.filter_queue_bar.value(), 1)
            self.assertEqual(window.filter_queue_bar.maximum(), 3)
            self.assertEqual(window.filter_queue_bar.format(), "Full processing: 1/3")
            self.assertEqual(window.filter_queue_label.text(), "pending 2")

            window._bulk_reprocess_active = True
            window._bulk_reprocess_total = 3
            window._bulk_reprocess_done = 2
            window._update_filter_queue_indicator()

            self.assertEqual(window.filter_queue_bar.value(), 2)
            self.assertEqual(window.filter_queue_bar.format(), "Reprocessing: 2/3")
        finally:
            window.close()

    def test_filter_status_widgets_have_fixed_room_for_normal_messages(self):
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            bar = window.filter_queue_bar
            label = window.filter_queue_label
            self.assertEqual(bar.minimumWidth(), bar.maximumWidth())
            self.assertEqual(label.minimumWidth(), label.maximumWidth())
            self.assertGreaterEqual(
                bar.width(),
                QtGui.QFontMetrics(bar.font()).horizontalAdvance(
                    "Full processing: 99999/99999"
                ),
            )
            self.assertGreaterEqual(
                label.width(),
                QtGui.QFontMetrics(label.font()).horizontalAdvance(
                    "full 99999/99999"
                ),
            )
        finally:
            window.close()

    def test_progress_bar_ignores_unacquired_frames_and_tracks_only_processing(self):
        """Streaming in new frames must not make a busy queue read as complete."""
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(20)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            # Only 3 frames have been streamed so far: 2 processed, 1 still queued.
            window._full_prepared_indices = {0, 1}
            window._seen_frame_indices = {0, 1, 2}
            window._full_queue.append(2)
            window._full_queued_indices.add(2)

            window._update_filter_queue_indicator()

            # Denominator is admitted work (2 done + 1 pending), not len(frames).
            self.assertEqual(window.filter_queue_bar.maximum(), 3)
            self.assertEqual(window.filter_queue_bar.value(), 2)
            self.assertEqual(window.filter_queue_label.text(), "pending 1")

            # Draining the queue reports idle, even though most frames are unacquired.
            window._full_prepared_indices = {0, 1, 2}
            window._full_queue.clear()
            window._full_queued_indices.clear()
            window._update_filter_queue_indicator()

            self.assertEqual(window.filter_queue_label.text(), "idle")
            self.assertEqual(window.filter_queue_bar.format(), "Full processing: 3/3")
        finally:
            window.close()

    def test_progress_bar_counts_pending_metric_promotions(self):
        """Points still being promoted from quick to full must read as pending."""
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(4)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            # All images filtered, but two ROI refits are still in flight.
            window._full_prepared_indices = {0, 1, 2, 3}
            window._seen_frame_indices = {0, 1, 2, 3}
            window._full_refresh_active_indices = {2, 3}

            window._update_filter_queue_indicator()

            self.assertEqual(window.filter_queue_label.text(), "pending 2")
            self.assertEqual(window._full_processing_pending_count(), 2)
            # Each image is counted once across both processing stages, so the
            # denominator stays at the image count and the two frames awaiting
            # promotion are not yet reported as finished.
            self.assertEqual(window.filter_queue_bar.maximum(), 4)
            self.assertEqual(window.filter_queue_bar.value(), 2)
            self.assertEqual(window.filter_queue_bar.format(), "Full processing: 2/4")

            # Finishing the promotions completes the bar.
            window._full_refresh_active_indices = set()
            window._update_filter_queue_indicator()

            self.assertEqual(window.filter_queue_bar.format(), "Full processing: 4/4")
            self.assertEqual(window.filter_queue_label.text(), "idle")
        finally:
            window.close()

    def test_progress_bar_never_exceeds_the_image_count(self):
        """The bar is capped at the number of images in every processing state."""
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(5)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            states = {
                "one frame in every stage at once": lambda: (
                    window._full_queue.append(1),
                    window._full_active_indices.update({0}),
                    window._full_prepared_indices.update({0}),
                    window._full_refresh_active_indices.update({0}),
                ),
                "stale indices from a larger dataset": lambda: (
                    window._full_prepared_indices.update({0, 1, 40, 99}),
                    window._full_queue.append(77),
                    window._full_refresh_active_indices.update({88}),
                ),
                "all cached and all promoting": lambda: (
                    window._full_prepared_indices.update(range(5)),
                    window._full_refresh_active_indices.update(range(5)),
                ),
            }
            for name, mutate in states.items():
                with self.subTest(state=name):
                    window._full_queue.clear()
                    window._full_queued_indices.clear()
                    window._full_active_indices.clear()
                    window._full_refresh_active_indices.clear()
                    window._full_prepared_indices.clear()
                    mutate()

                    window._update_filter_queue_indicator()

                    bar = window.filter_queue_bar
                    self.assertLessEqual(bar.maximum(), len(frames))
                    self.assertLessEqual(bar.value(), bar.maximum())
                    self.assertGreaterEqual(bar.value(), 0)
        finally:
            window.close()

    def test_promoted_frame_drops_low_resolution_data_and_is_counted_once(self):
        """Promotion releases the quick pixels and must not double-charge memory."""
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(4)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            quick = np.zeros((40, 40))
            full = np.ones((40, 40))
            image_bytes = int(full.nbytes)

            window._cache_filtered(0, quick)
            self.assertEqual(window._quick_cache_bytes, image_bytes)

            window._cache_full_filtered(0, full)

            # One image resident, charged exactly once (to the full budget).
            self.assertEqual(
                window._quick_cache_bytes + window._full_cache_bytes, image_bytes
            )
            self.assertEqual(window._full_cache_bytes, image_bytes)
            self.assertEqual(window._quick_cache_bytes, 0)
            # The low-resolution array is gone; the entry aliases the full data.
            np.testing.assert_array_equal(window._quick_filtered_cache[0], full)
            self.assertIn(0, window._quick_alias_indices)
            self.assertEqual(window._quick_cache_owned_count(), 0)

            # Read paths still resolve to full-quality data.
            np.testing.assert_array_equal(window._get_filtered_image(0), full)
            np.testing.assert_array_equal(
                window._get_cached_for_bulk(0, use_quick_cache=True), full
            )
        finally:
            window.close()

    def test_quick_budget_evicts_owned_entries_but_keeps_aliases(self):
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(8)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            image = np.ones((40, 40))
            window._max_quick_cache_bytes = 2 * int(image.nbytes)
            # An alias costs nothing and must survive quick-cache pressure,
            # because evicting it would free no memory.
            window._cache_full_filtered(0, image)
            for i in range(1, 5):
                window._cache_filtered(i, np.full((40, 40), float(i)))

            self.assertIn(0, window._quick_filtered_cache)
            self.assertIn(0, window._quick_alias_indices)
            self.assertLessEqual(
                window._quick_cache_bytes, window._max_quick_cache_bytes
            )
            self.assertLessEqual(window._quick_cache_owned_count(), 2)
        finally:
            window.close()

    def test_full_cache_eviction_does_not_orphan_a_quick_alias(self):
        """An evicted full frame must not leave untracked bytes behind."""
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(4)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            image_bytes = int(np.ones((40, 40)).nbytes)
            window._max_full_cache_bytes = 2 * image_bytes

            for i in range(4):
                window._cache_full_filtered(i, np.full((40, 40), float(i)))

            orphaned = set(window._quick_alias_indices) - set(
                window._full_filtered_cache
            )
            self.assertEqual(orphaned, set())
            resident = len(window._full_filtered_cache) * image_bytes
            self.assertEqual(
                window._quick_cache_bytes + window._full_cache_bytes, resident
            )
        finally:
            window.close()

    def test_stale_filter_generation_points_remain_visible_but_do_not_fit(self):
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            positions = np.linspace(-1.0, 1.0, 7)
            window.frames = [
                FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(pos))
                for i, pos in enumerate(positions)
            ]
            window._results = {
                i: SimpleNamespace(
                    step_sigma=float(1.0 + pos**2),
                    step_sigma_stderr=np.nan,
                    psf_sigma=float(1.5 + pos**2),
                    mtf50=float(0.2 - 0.02 * pos**2),
                )
                for i, pos in enumerate(positions)
            }
            window._result_is_full = {i: False for i in range(len(positions))}
            window._stale_result_indices = set(range(len(positions)))

            window._update_metric_plot()

            plotted_x, _ = window.curve_sigma_quick.getData()
            np.testing.assert_allclose(plotted_x, positions)
            fit_x, _ = window.curve_quad_fit.getData()
            self.assertTrue(fit_x is None or len(fit_x) == 0)
            self.assertTrue(np.isnan(window._optimal_focus_position))
            self.assertIsNone(window.validated_focus_target("step_sigma"))
        finally:
            window.close()

    def test_appended_frame_reopens_full_metric_finalization(self):
        harness = SimpleNamespace(_full_prepare_refresh_requested=True)

        FocusOfflineWindow.note_stream_frame_added(harness)

        self.assertFalse(harness._full_prepare_refresh_requested)

    def test_streamed_frame_queues_full_processing_by_index(self):
        """Full processing must not depend on the quick load being accepted."""
        queued = []
        harness = SimpleNamespace(
            _full_prepare_refresh_requested=True,
            _enqueue_full_prepare=queued.append,
        )

        FocusOfflineWindow.note_stream_frame_added(harness, 7)

        self.assertFalse(harness._full_prepare_refresh_requested)
        self.assertEqual(queued, [7])

    def test_online_finalization_waits_for_expected_frames(self):
        """A temporarily drained online queue is not acquisition completion."""
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
            expected_frame_count=5,
        )
        try:
            window.frames = [
                FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
                for i in range(3)
            ]
            window._full_prepared_indices = {0, 1, 2}
            window._results = {
                i: focus_offline_viewer.FitResult(ok=True) for i in range(3)
            }
            window._result_is_full = {i: True for i in range(3)}

            with patch.object(
                window, "_try_lock_edge_from_focus_minimum"
            ) as lock_edge:
                window._maybe_finalize_full_metric_pass()

            lock_edge.assert_not_called()
            self.assertFalse(window._full_dynamic_results_ready)
            self.assertFalse(window._full_prepare_refresh_requested)
            self.assertFalse(window._is_dataset_complete())

            window.mark_stream_complete()
            self.assertEqual(window._expected_frame_count, 3)
            self.assertTrue(window._is_dataset_complete())
        finally:
            window.close()

    def test_online_finalization_runs_after_expected_frames_arrive(self):
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
            expected_frame_count=5,
        )
        try:
            window.frames = [
                FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
                for i in range(5)
            ]
            window._full_prepared_indices = set(range(5))
            window._results = {
                i: focus_offline_viewer.FitResult(ok=True) for i in range(5)
            }
            window._result_is_full = {i: True for i in range(5)}

            with patch.object(
                window, "_try_lock_edge_from_focus_minimum", return_value=False
            ) as lock_edge:
                window._maybe_finalize_full_metric_pass()

            lock_edge.assert_called_once_with()
            self.assertTrue(window._full_dynamic_results_ready)
            self.assertTrue(window._full_prepare_refresh_requested)
            self.assertTrue(window._is_dataset_complete())
        finally:
            window.close()

    def test_offline_finalization_still_uses_complete_file_list(self):
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            window.frames = [
                FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
                for i in range(3)
            ]
            window._full_prepared_indices = {0, 1, 2}
            window._results = {
                i: focus_offline_viewer.FitResult(ok=True) for i in range(3)
            }
            window._result_is_full = {i: True for i in range(3)}

            with patch.object(
                window, "_try_lock_edge_from_focus_minimum", return_value=False
            ) as lock_edge:
                window._maybe_finalize_full_metric_pass()

            lock_edge.assert_called_once_with()
            self.assertTrue(window._is_dataset_complete())
        finally:
            window.close()

    def test_adaptive_batch_temporarily_reopens_online_dataset(self):
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(5)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
            expected_frame_count=5,
        )
        try:
            window._full_prepared_indices = set(range(5))
            self.assertTrue(window._is_dataset_complete())

            self.assertEqual(window.adjust_expected_stream_frames(3), 8)
            self.assertFalse(window._is_dataset_complete())

            # A rejected command rolls back to the already received dataset.
            self.assertEqual(window.adjust_expected_stream_frames(-3), 5)
            self.assertTrue(window._is_dataset_complete())
        finally:
            window.close()

    def test_full_result_replaces_quick_image_for_displayed_frame(self):
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(2)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            quick = np.full((8, 8), 5.0)
            full = np.full((8, 8), 9.0)
            window.current_index = 0
            window._current_filtered = quick
            window._current_filtered_is_full = False

            shown = []
            window._set_display_image = shown.append
            window._request_analysis_current = lambda: None

            window._handle_full_prepare_result(0, window._preprocess_token, full, None)

            # The displayed image is swapped to the full-resolution version.
            self.assertEqual(len(shown), 1)
            np.testing.assert_array_equal(shown[0], full)
            self.assertIs(window._current_filtered, full)
            self.assertTrue(window._current_filtered_is_full)
            self.assertIn(0, window._full_prepared_indices)
        finally:
            window.close()

    def test_quick_load_does_not_regress_an_already_full_display(self):
        """A late quick-pass result must not overwrite a full-resolution image."""
        frames = [FrameInfo(index=0, path=Path("frame_0.tif"), position=0.0)]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            quick = np.full((8, 8), 5.0)
            full = np.full((8, 8), 9.0)
            window._cache_full_filtered(0, full)
            window._roi_initialized_from_first_frame = True

            shown = []
            window._set_display_image = shown.append
            window._request_analysis_current = lambda: None
            window._update_metric_plot = lambda: None

            window._on_task_done(
                "load_filter_display", int(window._load_generation), 0, quick, None
            )

            np.testing.assert_array_equal(shown[-1], full)
            self.assertIs(window._current_filtered, full)
            self.assertTrue(window._current_filtered_is_full)
        finally:
            window.close()

    def test_quick_pass_fit_is_not_labelled_full_quality(self):
        """A fit from a quick image stays gray even if full filtering just landed."""
        frames = [FrameInfo(index=0, path=Path("frame_0.tif"), position=0.0)]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            window.current_index = 0
            window._update_metric_plot = lambda: None
            window._update_profile_plot = lambda _r: None
            # Analysis was launched from the quick image...
            window._analysis_token = 5
            window._analysis_is_full_by_token = {5: False}
            # ...and full filtering completed for the same frame while it ran.
            window._full_prepared_indices.add(0)

            window._on_task_done(
                "analyze_current", 5, 0, focus_offline_viewer.FitResult(ok=True), None
            )

            self.assertFalse(window._result_is_full[0])
        finally:
            window.close()

    def test_full_pass_fit_is_labelled_full_quality(self):
        frames = [FrameInfo(index=0, path=Path("frame_0.tif"), position=0.0)]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            window.current_index = 0
            window._update_metric_plot = lambda: None
            window._update_profile_plot = lambda _r: None
            window._analysis_token = 5
            window._analysis_is_full_by_token = {5: True}

            window._on_task_done(
                "analyze_current", 5, 0, focus_offline_viewer.FitResult(ok=True), None
            )

            self.assertTrue(window._result_is_full[0])
        finally:
            window.close()

    def test_failed_full_processing_is_retried_then_abandoned(self):
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(2)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            window._max_full_prepare_retries = 2
            error = RuntimeError("bad read")

            self.assertTrue(window._requeue_failed_full_prepare(1, error))
            self.assertIn(1, window._full_queued_indices)

            # Clear the queue to allow the next retry to be admitted.
            window._full_queue.clear()
            window._full_queued_indices.clear()
            self.assertTrue(window._requeue_failed_full_prepare(1, error))

            window._full_queue.clear()
            window._full_queued_indices.clear()
            # Retries exhausted: stop re-queueing and report it once.
            self.assertFalse(window._requeue_failed_full_prepare(1, error))
            self.assertIn(1, window._full_prepare_abandoned)
        finally:
            window.close()

    def test_retry_missing_full_prepare_requeues_and_unlatches(self):
        frames = [
            FrameInfo(index=i, path=Path(f"frame_{i}.tif"), position=float(i))
            for i in range(4)
        ]
        window = FocusOfflineWindow(
            frames,
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            window._full_prepared_indices = {0, 1}
            window._full_prepare_refresh_requested = True
            window._pump_full_queue = lambda: None

            requeued = window.retry_missing_full_prepare()

            self.assertEqual(requeued, 2)
            self.assertEqual(window._full_queued_indices, {2, 3})
            self.assertFalse(window._full_prepare_refresh_requested)
        finally:
            window.close()

    def test_local_fit_accepts_an_in_range_vertex(self):
        x = np.linspace(0.0, 2.0, 9)
        y = (x - 1.0) ** 2 + 2.0

        fit = FocusOfflineWindow._local_parabola_fit(
            x,
            y,
            local_radius=1.0,
            find="min",
        )

        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit[2], 1.0, places=8)

    def test_local_fit_rejects_an_extrapolated_vertex(self):
        x = np.asarray([0.0, 1.0, 2.0])
        y = (x + 5.0) ** 2

        fit = FocusOfflineWindow._local_parabola_fit(
            x,
            y,
            local_radius=0.5,
            find="min",
        )

        self.assertIsNone(fit)

    def test_local_fit_discards_a_stale_edge_hint(self):
        x = np.linspace(-2.0, 2.0, 15)
        y = 0.155 - 0.02 * x**2

        fit = FocusOfflineWindow._local_parabola_fit(
            x,
            y,
            local_radius=0.5,
            find="max",
            center_hint=-2.0,
        )

        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit[2], 0.0, places=10)

    def test_simulated_extension_positions_and_fit_curves_are_plotted(self):
        window = FocusOfflineWindow(
            [],
            max_workers_total=3,
            bulk_workers=1,
            full_workers=1,
        )
        try:
            coarse = np.linspace(-2.0, 2.0, 15)
            spacing = float(coarse[1] - coarse[0])
            positions = np.concatenate(
                [coarse, -2.0 - spacing * np.arange(1.0, 4.0)]
            )
            window.frames = [
                FrameInfo(index=i, path=Path(f"sim_{i:04d}.tif"), position=float(pos))
                for i, pos in enumerate(positions)
            ]
            window._results = {
                i: SimpleNamespace(
                    step_sigma=float(1.2 + pos**2),
                    step_sigma_stderr=np.nan,
                    psf_sigma=float(2.0 + 0.8 * pos**2),
                    mtf50=float(0.20 - 0.01 * pos**2),
                )
                for i, pos in enumerate(positions)
            }
            window._result_is_full = {i: True for i in range(len(positions))}

            window._update_metric_plot()

            plotted_x, _ = window.curve_psf_full.getData()
            np.testing.assert_allclose(plotted_x, positions)
            self.assertAlmostEqual(plotted_x[-3], -2.0 - spacing, places=12)
            self.assertAlmostEqual(plotted_x[-1], -2.0 - 3.0 * spacing, places=12)
            for curve in (
                window.curve_quad_fit,
                window.curve_psf_fit,
                window.curve_mtf50_fit,
            ):
                fit_x, fit_y = curve.getData()
                self.assertGreater(len(fit_x), 3)
                self.assertTrue(np.isfinite(fit_x).all())
                self.assertTrue(np.isfinite(fit_y).all())
                self.assertTrue(curve.isVisible())
            self.assertAlmostEqual(window._optimal_focus_position, 0.0, places=10)
            self.assertAlmostEqual(window._optimal_psf_position, 0.0, places=10)
            self.assertAlmostEqual(window._optimal_mtf50_position, 0.0, places=10)
        finally:
            window.close()

    def test_focus_target_requires_full_quality_results(self):
        harness = SimpleNamespace(
            _shutting_down=False,
            _bulk_reprocess_active=False,
            _analysis_inflight=False,
            _full_refresh_active_indices=set(),
            _is_dataset_complete=lambda: True,
            _all_frames_full_prepared=lambda: True,
            _optimal_mtf50_position=2.0,
            _results={
                i: SimpleNamespace(mtf50=float(10 - (i - 2) ** 2))
                for i in range(5)
            },
            _result_is_full={i: True for i in range(5)},
            frames=[SimpleNamespace(position=float(i)) for i in range(5)],
        )

        target = FocusOfflineWindow.validated_focus_target(harness, "mtf50")
        self.assertEqual(target, 2.0)

        harness._is_dataset_complete = lambda: False
        self.assertIsNone(FocusOfflineWindow.validated_focus_target(harness, "mtf50"))

        harness._is_dataset_complete = lambda: True
        harness._result_is_full[4] = False
        self.assertIsNone(FocusOfflineWindow.validated_focus_target(harness, "mtf50"))

    def test_focus_target_is_blocked_while_processing(self):
        harness = SimpleNamespace(
            _shutting_down=False,
            _bulk_reprocess_active=True,
            _analysis_inflight=False,
            _full_refresh_active_indices=set(),
            _is_dataset_complete=lambda: True,
            _all_frames_full_prepared=lambda: True,
            _optimal_mtf50_position=2.0,
            _results={},
            _result_is_full={},
            frames=[],
        )

        self.assertIsNone(FocusOfflineWindow.validated_focus_target(harness, "mtf50"))


class AdaptiveCommandConfirmationTests(unittest.TestCase):
    def test_submit_waits_for_the_session_store_result(self):
        client = QueueServerAdaptiveClient.__new__(QueueServerAdaptiveClient)
        client.session_id = "session-1"
        client._user = "user"
        client._user_group = "primary"
        client._api = mock.Mock()
        client._api.function_execute.return_value = {
            "success": True,
            "task_uid": "task-1",
        }
        client._api.task_result.return_value = {
            "success": True,
            "status": "completed",
            "result": {
                "success": True,
                "return_value": {
                    "ok": True,
                    "accepted_command": "complete",
                },
            },
        }

        response = client.submit("complete")

        self.assertTrue(response["success"])
        self.assertTrue(response["ok"])
        self.assertEqual(response["accepted_command"], "complete")
        client._api.wait_for_completed_task.assert_called_once()


if __name__ == "__main__":
    unittest.main()
