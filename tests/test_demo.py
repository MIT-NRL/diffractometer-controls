"""Demo routing, CA acquisition semantics and owned-process cleanup."""
import asyncio
import os
import pickle
import socket
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diffractometer_controls.demo.runtime import DEFAULT_PORTS, DemoRuntime, DemoRuntimeError, demo_environment


class DemoRuntimeTests(unittest.TestCase):
    def test_demo_routes_are_local_and_profile_follows_prefix(self):
        ports = dict(DEFAULT_PORTS, epics_ca=16064)
        environment = demo_environment(ports)
        from diffractometer_controls.site.mitr import profile
        with patch.dict(os.environ, environment):
            self.assertEqual(profile.panel_macros()["P"], "demo4dh4:")
            self.assertIn("127.0.0.1", profile.endpoints("production-host").control)
            self.assertTrue(all(value.startswith("demo4dh4:") for value in profile.estimation_pvs().values()))
            self.assertEqual(profile.editor_options().local_roots, ())
        self.assertEqual(environment["EPICS_CA_AUTO_ADDR_LIST"], "NO")
        self.assertEqual(environment["EPICS_PVA_AUTO_ADDR_LIST"], "NO")
        self.assertEqual(environment["EPICS_PVA_ADDR_LIST"], "127.0.0.1:16064")

    def test_occupied_port_rejects_start(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            runtime = DemoRuntime(ports={"redis": listener.getsockname()[1]})
            with self.assertRaisesRegex(DemoRuntimeError, "redis"):
                runtime._check_ports()

    def test_additional_demo_panels_do_not_launch_adaptive_viewers(self):
        from diffractometer_controls.main_window import MITRMainWindow
        panel = SimpleNamespace(demo_mode=True, _focus_doc_subscription=None)
        with patch.dict(os.environ, {"MITR_DEMO_STACK_OWNER":"0"}):
            MITRMainWindow._start_adaptive_focus_listener(panel)
        self.assertIsNone(panel._focus_doc_subscription)

    def test_startup_failure_restores_environment_and_cleans_directory(self):
        previous = dict(os.environ)
        runtime = DemoRuntime()
        with patch.object(runtime, "_check_ports"), patch.object(runtime, "_spawn", side_effect=RuntimeError("startup failed")), patch("importlib.util.find_spec", return_value=object()):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                runtime.start()
        self.assertEqual(dict(os.environ), previous)
        self.assertIsNone(runtime.runtime_dir)
        self.assertEqual(runtime.processes, [])
        runtime.stop()

    def test_posix_cleanup_kills_owned_descendants_after_supervisor_exits(self):
        runtime = DemoRuntime()
        process = SimpleNamespace(pid=3456, poll=lambda: 0, wait=lambda **kw: 0)
        stream = SimpleNamespace(close=lambda: None)
        runtime.processes = [("owned", process, stream, None)]
        with patch("diffractometer_controls.demo.runtime.os.name", "posix"), patch("diffractometer_controls.demo.runtime.os.killpg", create=True) as kill, patch("diffractometer_controls.demo.runtime.signal.SIGKILL", 9, create=True):
            runtime.stop()
        import signal
        self.assertEqual(kill.call_args_list[0].args, (3456, signal.SIGTERM))
        self.assertEqual(kill.call_args_list[-1].args, (3456, 9))


class DemoIOCTests(unittest.IsolatedAsyncioTestCase):
    async def test_scaler_put_completion_waits_for_complete_counts(self):
        from diffractometer_controls.demo.scalar_ioc import USBCTRRecords
        scaler = USBCTRRecords(prefix="test:")
        await scaler.preset.write(0.1)
        await scaler.count.write(1)
        self.assertEqual(scaler.count.value, 0)
        self.assertEqual(scaler.counts_2.value, 240)
        self.assertEqual(scaler.counts_3.value, 85)
        self.assertEqual(scaler.elapsed.value, 0.1)

    async def test_scaler_stop_releases_pending_put(self):
        from diffractometer_controls.demo.scalar_ioc import USBCTRRecords
        scaler = USBCTRRecords(prefix="test:")
        await scaler.preset.write(10)
        counting = asyncio.create_task(scaler.count.write(1))
        await asyncio.sleep(0.03)
        await scaler.count.write(0)
        await asyncio.wait_for(counting, 1)
        self.assertEqual(scaler.count.value, 0)
        self.assertLess(scaler.elapsed.value, 10)

    async def test_mcs_stop_then_restart_keeps_new_acquisition_state(self):
        from diffractometer_controls.demo.scalar_ioc import USBCTRRecords
        scaler = USBCTRRecords(prefix="test:")
        await scaler.points.write(2)
        await scaler.mcs_start.write(1)
        old_task = scaler._mcs_task
        await asyncio.sleep(0.01)
        await scaler.mcs_stop.write(1)
        self.assertTrue(old_task.done())
        await scaler.mcs_start.write(1)
        new_task = scaler._mcs_task
        await asyncio.sleep(0.01)
        self.assertIs(scaler._mcs_task, new_task)
        self.assertEqual(scaler.mcs_busy.value, 1)
        await asyncio.wait_for(new_task, 1)
        self.assertEqual(scaler.mcs_busy.value, 0)

    async def test_camera_binning_and_multiple_frames(self):
        from diffractometer_controls.demo.ioc import DemoIOC
        ioc = DemoIOC(prefix="test:")
        camera = ioc.cam1
        await camera.bin_x.write(2)
        await camera.bin_y.write(4)
        await camera.num_images.write(2)
        await camera.image_mode.write(1)
        await camera.acquire_time.write(0.01)
        await camera.acquire.write(1)
        await asyncio.wait_for(camera._task, 2)
        self.assertEqual(camera.array_counter.value, 2)
        self.assertEqual(camera.num_images_counter.value, 2)
        self.assertEqual(camera.acquire_rbv.value, 0)
        self.assertEqual(ioc.image_array.value.size, 64 * 32)
        self.assertEqual(camera.size_x.value, 64)
        self.assertEqual(camera.size_y.value, 32)


class DemoHistoryIsolationTests(unittest.TestCase):
    def test_demo_blocks_automatic_and_manual_storage_connections(self):
        from qtpy import QtCore, QtWidgets
        from diffractometer_controls.screens.diffraction import diffraction_live_plot as live
        from diffractometer_controls.screens.diffraction.diffraction_live_plot_pyqtgraph import DiffractionPlotWidgetPyQtGraph

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with patch.dict(os.environ, demo_environment()), patch.object(live, "PV", None):
            widget = DiffractionPlotWidgetPyQtGraph()
            controller = live.DiffractionLivePlot(widget, auto_history=True)
            try:
                with patch.object(controller._history_executor, "submit") as submit, patch.object(live, "ensure_tiled_login") as login:
                    app.processEvents()
                    controller._on_tiled_enabled_toggled(True)
                    controller.open_analysis_viewer()
                    self.assertFalse(controller._tiled_enabled)
                    submit.assert_not_called()
                    login.assert_not_called()
            finally:
                controller.shutdown()
                widget.deleteLater()
                app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)


class FocusDocumentTests(unittest.TestCase):
    def test_focus_subscriber_accepts_descriptor_midrun_and_stops(self):
        from diffractometer_controls.demo.focus_viewer import FocusDocumentDispatcher
        descriptor = {"uid":"descriptor", "run_start":"known-run"}
        receiver = SimpleNamespace(closed=False, _prefix=b"", _deserializer=pickle.loads)
        def receive(_flags):
            receiver.closed = True
            return b" descriptor " + pickle.dumps(descriptor)
        receiver._socket = SimpleNamespace(recv=receive)
        generator = FocusDocumentDispatcher._receive_data(receiver)
        self.assertEqual(next(generator), ("descriptor", descriptor))
        with self.assertRaises(StopIteration):
            next(generator)


if __name__ == "__main__":
    unittest.main()
