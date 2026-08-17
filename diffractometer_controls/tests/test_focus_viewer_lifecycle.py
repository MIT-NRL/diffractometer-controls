import os
import sys
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qtpy import QtWidgets

from diffractometer_controls import focus_offline_viewer, focus_online_viewer
from diffractometer_controls.focus_offline_viewer import FocusOfflineWindow, FrameInfo
from diffractometer_controls.focus_online_viewer import (
    FocusOnlineBridge,
    QueueServerAdaptiveClient,
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
        self.assertNotIn("WAYLAND_DISPLAY", child)

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

    def test_appended_frame_reopens_full_metric_finalization(self):
        harness = SimpleNamespace(_full_prepare_refresh_requested=True)

        FocusOfflineWindow.note_stream_frame_added(harness)

        self.assertFalse(harness._full_prepare_refresh_requested)

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

        harness._result_is_full[4] = False
        self.assertIsNone(FocusOfflineWindow.validated_focus_target(harness, "mtf50"))

    def test_focus_target_is_blocked_while_processing(self):
        harness = SimpleNamespace(
            _shutting_down=False,
            _bulk_reprocess_active=True,
            _analysis_inflight=False,
            _full_refresh_active_indices=set(),
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
