"""Exercise the supported demo launcher, connected workspaces and focus viewer.

Uses the real local services and panels. Screenshots are optional; no production
endpoints or acquisition storage are used. Run this in the GUI environment.
"""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/demo-gui")
    parser.add_argument("--platform", default="windows" if os.name == "nt" else "offscreen")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["QT_QPA_PLATFORM"] = args.platform
    from diffractometer_controls.demo.runtime import demo_environment
    os.environ.update(demo_environment())
    from diffractometer_controls import launcher
    launcher._preload_p4p()
    from diffractometer_controls.application import MITRApplication
    from diffractometer_controls.main_window import MITRMainWindow
    from diffractometer_controls.site.mitr import profile
    from qtpy import QtCore, QtWidgets
    from control_ui.core.compatibility import install_bluesky_compatibility
    from diffractometer_controls.demo.focus_viewer import DemoFocusWindow

    failure = []
    focus = []
    original_init = MITRApplication.__init__

    def launch_focus(window, *, session_id, run_uid):
        viewer = DemoFocusWindow(SimpleNamespace(session_id=session_id, run_uid=run_uid,
            motor_key=window._focus_online_motor_key, parent_pid=0,
            zmq_address=window._focus_data_addr, qserver_control_addr=window._focus_qs_control_addr,
            qserver_info_addr=window._focus_qs_info_addr))
        focus.append(viewer)
        viewer.show()

    def initialize(app, *a, **kw):
        original_init(app, *a, **kw)
        QtCore.QTimer.singleShot(300, lambda: checks(app))

    def checks(app):
        api = app.re_manager_api

        def wait(predicate, label, timeout=30):
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                app.processEvents()
                if predicate():
                    return
                time.sleep(0.02)
            raise RuntimeError(f"Timed out: {label}")

        def run(name, kwargs):
            api.item_add({"name": name, "item_type": "plan", "kwargs": kwargs}, user="demo_gui_test", user_group="primary")
            api.queue_start()
            wait(lambda: api.status()["manager_state"] == "idle" and not api.status()["items_in_queue"], name)
            result = api.history_get()["items"][-1]["result"]
            assert result["exit_status"] == "completed", result

        try:
            window = app.main_window
            window.resize(2200, 1400)
            assert window.findChild(QtWidgets.QLabel, "demoModeBanner") is not None
            screen = window.display_widget()
            controls = screen.workspace.run_engine_controls
            controls._re_manager._pb_re_manager_connect_clicked()
            wait(lambda: bool(app.re_client._allowed_plans), "allowed plans")
            assert "scan_scalar" in app.re_client._allowed_plans
            run("count_scalar", {"title":"Connected GUI", "detectors":["usbctr.he3_tube", "usbctr.beam_monitor"], "num":2, "acquire_time":0.1})
            wait(lambda: bool(app.re_client._plan_history_items), "history refresh")
            assert screen._diffraction_live_plot._current_live_start_doc["plan_name"] == "count_scalar"
            from control_ui.panels.usbctr_count import USBCTRCountDisplay
            from control_ui.panels.usbctr_rate import USBCTRRateDisplay
            board_macros = {"P": profile.pv_prefix() + "USBCTR:", "S":"scaler1", "MP":profile.pv_prefix() + "USBCTR:MCS:"}
            count_panel = USBCTRCountDisplay(macros=board_macros)
            count_panel.show()
            wait(lambda: count_panel.start_button.isEnabled(), "USBCTR timed panel connected")
            count_panel.start_button.click()
            wait(lambda: count_panel._running, "USBCTR counting")
            wait(lambda: not count_panel._running, "USBCTR count completion")
            assert count_panel._counts > 0
            count_panel.close()
            rate_panel = USBCTRRateDisplay(macros=board_macros)
            rate_panel.show()
            wait(lambda: rate_panel.start_button.isEnabled(), "USBCTR MCS panel connected")
            rate_panel.start_button.click()
            wait(lambda: rate_panel._samples.size >= 3, "MCS waveform")
            assert np.isfinite(rate_panel._samples).all()
            rate_panel.stop_button.click()
            wait(lambda: not rate_panel._acquiring, "MCS stopped")
            rate_panel.close()
            window.grab().save(str(args.output / "diffraction.png"))
            window.open(str(profile.SCREEN_PATHS["tomography"]), macros=profile.panel_macros())
            app.processEvents()
            screen = window.display_widget()
            run("imaging", {"file_name":"gui", "file_dir":"", "num_exposures":2, "exposure_time":0.05})
            window.grab().save(str(args.output / "tomography.png"))
            api.item_add({"name":"adaptive_imaging_focus_scan", "item_type":"plan", "kwargs": {
                "file_name":"gui_focus", "file_dir":"", "detector":"sim_focus_cam", "motor":"cam1.focus",
                "start_pos":-0.2, "stop_pos":0.2, "num_steps":3, "exposure_time":0.05,
                "viewer_start_timeout_s":30}}, user="demo_gui_test", user_group="primary")
            api.queue_start()
            wait(lambda: focus and len(focus[0].metrics) >= 3, "adaptive viewer streamed frames")
            assert np.isfinite(focus[0].metrics).any(), focus[0].metrics
            focus[0].grab().save(str(args.output / "adaptive-focus.png"))
            focus[0].submit("complete")
            wait(lambda: api.status()["manager_state"] == "idle", "adaptive completion")
            assert api.history_get()["items"][-1]["result"]["exit_status"] == "completed"
            focus[0].finished = True
            focus[0].close()
            # Return through cached navigation and check polling has resumed.
            window.back()
            app.processEvents()
            run("count_he3psd", {"title":"Cached GUI", "detectors":["he3psd0", "he3psd7"], "num":1, "acquire_time":0.05})
            wait(lambda: len(app.re_client._plan_history_items) >= 4, "cached screen polling")
            app.new_pydm_process(str(ROOT / "control_ui/panels/usbctr_count.py"), macros=board_macros)
            wait(lambda: app._demo_child_processes and app._demo_child_processes[-1].poll() is None, "additional panel process")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.02)
            assert app._demo_child_processes[-1].poll() is None, "Additional panel failed to attach"
            print("Connected diffraction/tomography, adaptive MTF analysis and cached-screen polling passed", flush=True)
        except BaseException:
            failure.append(traceback.format_exc())
            print(failure[-1], flush=True)
        finally:
            for viewer in focus:
                viewer.close()
            app.quit()

    sys.argv = [sys.argv[0], "--demo", "--displayfile", str(profile.SCREEN_PATHS["diffraction"])]
    install_bluesky_compatibility()
    with patch.object(MITRApplication, "__init__", initialize), patch.object(MITRMainWindow, "_launch_focus_online_viewer", launch_focus):
        try:
            launcher.main()
        except SystemExit as exc:
            return 1 if failure else int(exc.code or 0)
    return int(bool(failure))


if __name__ == "__main__":
    sys.exit(main())
