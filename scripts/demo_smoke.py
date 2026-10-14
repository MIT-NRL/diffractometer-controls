"""Execute real demo plans through the isolated QueueServer, without a GUI."""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffractometer_controls.demo.runtime import DemoRuntime
import zmq


def wait_for(predicate, timeout, label, pump=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pump()
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise RuntimeError(f"Timed out: {label}")


def execute_function(api, name, *args):
    from bluesky_queueserver_api import BFunc
    reply = api.function_execute(BFunc(name, *args), run_in_background=True, user="demo_test", user_group="primary")
    api.wait_for_completed_task(reply["task_uid"], timeout=10)
    result = api.task_result(reply["task_uid"])["result"]
    if not result["success"]:
        raise RuntimeError(result)
    return result["return_value"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--advanced", action="store_true", help="Also test scans, tomography, adaptive commands and pause/abort")
    args = parser.parse_args()
    runtime = DemoRuntime(startup_timeout=args.timeout)
    context = zmq.Context()
    documents = []
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.LINGER, 0)

    def receive():
        while socket.poll(0):
            _prefix, name, document = socket.recv().split(b" ", 2)
            documents.append((name.decode(), pickle.loads(document)))

    try:
        runtime.start()
        socket.connect(runtime.environment["MITR_DOCUMENT_ADDR"])
        from bluesky_queueserver_api.zmq import REManagerAPI
        api = REManagerAPI(zmq_control_addr=runtime.environment["MITR_QSERVER_CONTROL_ADDR"],
                           zmq_info_addr=runtime.environment["MITR_QSERVER_INFO_ADDR"])
        print("Demo worker ready", flush=True)
        plans = api.plans_allowed(user_group="primary")["plans_allowed"]
        print("Plans:", sorted(plans), flush=True)
        items = [
            ("count_scalar", {"title": "Demo scalar", "detectors": ["usbctr.he3_tube", "usbctr.beam_monitor"], "acquire_time": 0.1, "num": 2}),
            ("count_he3psd", {"title": "Demo PSD", "detectors": ["he3psd0", "he3psd7"], "acquire_time": 0.05, "num": 2}),
            ("imaging", {"file_name": "demo", "file_dir": "", "exposure_time": 0.05, "num_exposures": 2}),
        ]
        if args.advanced:
            items += [
                ("scan_scalar", {"title": "Demo scalar scan", "detectors": ["usbctr.he3_tube", "usb2408.temperature.ti1"], "motor": "sample_th", "start_pos": -0.1, "stop_pos": 0.1, "num_steps": 3, "acquire_time": 0.05}),
                ("tomo_scan", {"file_name": "demo_tomo", "file_dir": "", "motor": "stage1.theta", "start_angle": 0, "stop_angle": 0.2, "num_projections": 3, "exposure_time": 0.05}),
            ]
        time.sleep(0.3)  # settle the document subscription before the first run
        for name, kwargs in items:
            print("Executing", name, flush=True)
            api.item_add({"name": name, "item_type": "plan", "kwargs": kwargs}, user="demo_test", user_group="primary")
            api.queue_start()
            def idle():
                status = api.status()
                return status["manager_state"] == "idle" and not status["items_in_queue"]
            wait_for(idle, args.timeout, name, receive)
            history = api.history_get()["items"]
            result = history[-1]["result"]
            print(json.dumps(result), flush=True)
            if result["exit_status"] != "completed":
                raise RuntimeError(result)
        if args.advanced:
            print("Executing adaptive imaging and refinement", flush=True)
            api.item_add({"name": "adaptive_imaging_focus_scan", "item_type": "plan", "kwargs": {
                "file_name": "demo_focus", "file_dir": "", "detector": "sim_focus_cam", "motor": "cam1.focus",
                "start_pos": -0.2, "stop_pos": 0.2, "num_steps": 3, "exposure_time": 0.05, "viewer_start_timeout_s": 20}}, user="demo_test", user_group="primary")
            api.queue_start()
            start = wait_for(lambda: next((doc for name, doc in documents if name == "start" and doc.get("plan_name") == "adaptive_imaging_focus_scan"), None), args.timeout, "adaptive start", receive)
            session = start["focus_adaptive"]["session_id"]
            def command(name, payload=None):
                result = execute_function(api, "adaptive_focus_submit_command", session, name, payload or {})
                if not result.get("ok"):
                    raise RuntimeError(result)
            def snapshot():
                return execute_function(api, "adaptive_focus_get_session", session)
            command("viewer_ready")
            wait_for(lambda: snapshot()["status"] == "awaiting_command", args.timeout, "coarse scan", receive)
            initial = snapshot()["state"]["done_units"]
            command("scan_around_focus", {"center": 0.0, "step_size": 0.05, "num_points": 3})
            wait_for(lambda: snapshot()["state"].get("done_units", 0) >= initial + 3, args.timeout, "refinement", receive)
            command("go_to_focus", {"position": 0.05})
            command("complete")
            wait_for(idle, args.timeout, "adaptive completion", receive)
            result = api.history_get()["items"][-1]["result"]
            assert result["exit_status"] == "completed", result
            print("Adaptive coarse/refine/go-to-focus/complete passed", flush=True)
            print("Checking pause/resume/abort", flush=True)
            api.item_add({"name": "count_scalar", "item_type": "plan", "kwargs": {"title": "Demo abort", "detectors": ["usbctr.he3_tube"], "num": 20, "acquire_time": 0.2}}, user="demo_test", user_group="primary")
            api.queue_start()
            wait_for(lambda: api.status()["re_state"] == "running", args.timeout, "running", receive)
            api.re_pause(option="deferred")
            wait_for(lambda: api.status()["re_state"] == "paused", args.timeout, "paused", receive)
            api.re_resume()
            wait_for(lambda: api.status()["re_state"] == "running", args.timeout, "resumed", receive)
            api.re_pause(option="deferred")
            wait_for(lambda: api.status()["re_state"] == "paused", args.timeout, "paused for abort", receive)
            api.re_abort()
            wait_for(lambda: api.status()["manager_state"] == "idle", args.timeout, "aborted", receive)
            assert api.history_get()["items"][-1]["result"]["exit_status"] == "aborted"
        receive()
        assert any(name == "event" for name, doc in documents), "No streamed events"
        assert not any(name in ("resource", "datum", "stream_resource", "stream_datum") for name, doc in documents), "Unexpected stored-data documents"
        assert all(doc.get("demo") for name, doc in documents if name == "start"), "Missing demo metadata"
        count_uid = next(doc["uid"] for name, doc in documents if name == "start" and doc.get("plan_name") == "count_scalar")
        descriptors = {doc["uid"] for name, doc in documents if name == "descriptor" and doc["run_start"] == count_uid and doc["name"] == "primary"}
        values = [doc["data"] for name, doc in documents if name == "event" and doc["descriptor"] in descriptors]
        assert len(values) == 2, values
        assert all(value["usbctr_he3_tube"] == 85 and value["usbctr_beam_monitor"] == 240 for value in values), values
        print(f"Received {len(documents)} documents; storage disabled", flush=True)
    finally:
        for label, _process, _stream, log_path in runtime.processes:
            if label == "queue-server":
                print(log_path.read_text(errors="replace")[-5000:], flush=True)
        owned = [process for _label, process, _stream, _path in runtime.processes]
        runtime.stop()
        socket.close()
        context.term()
        assert all(process.poll() is not None for process in owned), "Demo service left running"
        print("All owned demo services stopped", flush=True)


if __name__ == "__main__":
    main()
