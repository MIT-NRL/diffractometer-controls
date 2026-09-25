"""Lifecycle supervisor for the fully disconnected demo stack."""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_PORTS = {
    "redis": 6380,
    "qserver_control": 61615,
    "qserver_info": 61625,
    "documents_in": 61567,
    "documents_out": 61568,
    "epics_ca": 6064,
    "epics_repeater": 6065,
}


class DemoRuntimeError(RuntimeError):
    pass


def demo_environment(ports=None):
    ports = dict(DEFAULT_PORTS if ports is None else ports)
    prefix = "demo4dh4:"
    return {
        "MITR_DEMO_ACTIVE": "1",
        "MITR_EPICS_PREFIX": prefix,
        "MITR_DEMO_EPICS_PREFIX": prefix,
        "MITR_RUN_STATUS_PV_PREFIX": f"{prefix}Bluesky:Run:",
        "MITR_REDIS_ADDR": f"127.0.0.1:{ports['redis']}",
        "MITR_QSERVER_CONTROL_ADDR": f"tcp://127.0.0.1:{ports['qserver_control']}",
        "MITR_QSERVER_INFO_ADDR": f"tcp://127.0.0.1:{ports['qserver_info']}",
        "MITR_DOCUMENT_INPUT_ADDR": f"tcp://127.0.0.1:{ports['documents_in']}",
        "MITR_DOCUMENT_ADDR": f"tcp://127.0.0.1:{ports['documents_out']}",
        "EPICS_CA_AUTO_ADDR_LIST": "NO",
        "EPICS_CA_ADDR_LIST": "127.0.0.1",
        "EPICS_CA_SERVER_PORT": str(ports["epics_ca"]),
        "EPICS_CA_REPEATER_PORT": str(ports["epics_repeater"]),
        "EPICS_PVA_AUTO_ADDR_LIST": "NO",
        "EPICS_PVA_ADDR_LIST": "127.0.0.1",
        "EPICS_CA_MAX_ARRAY_BYTES": str(8 * 1024 * 1024),
    }


class DemoRuntime:
    """Own and supervise Redis, caproto, document proxy and RE Manager."""

    def __init__(self, *, ports=None, startup_timeout=45.0):
        self.ports = dict(DEFAULT_PORTS)
        if ports:
            self.ports.update(ports)
        self.startup_timeout = float(startup_timeout)
        self.processes = []
        self.runtime_dir = None
        self._stopped = False
        self.environment = dict(os.environ)
        self.environment.update(demo_environment(self.ports))
        self.environment["MITR_DEMO_STACK_OWNER"] = "1"
        # This stack is loopback-only.  Do not inherit beamline CURVE keys,
        # which would make the manager and an internal readiness probe use
        # mismatched security settings.
        for key in ("QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER", "QSERVER_ZMQ_PUBLIC_KEY"):
            self.environment.pop(key, None)
        package_dir = Path(__file__).resolve().parent
        demo_config = package_dir / "bluesky_config" / "demo"
        self.environment.update(
            MITR_DEMO_STARTUP_DIR=str(demo_config / "startup"),
            MITR_DEMO_PERMISSIONS=str(demo_config / "permissions.yaml"),
            MITR_DEMO_SHARED_STARTUP_DIR=str(package_dir / "bluesky_config" / "startup"),
        )
        package_root = str(package_dir.parent)
        old_pythonpath = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (package_root, old_pythonpath) if item
        )
        self.config_path = demo_config / "qserver_config.yml"

    @staticmethod
    def _executable(name):
        adjacent = Path(sys.executable).with_name(name)
        if adjacent.is_file():
            return str(adjacent)
        found = shutil.which(name)
        if found:
            return found
        raise DemoRuntimeError(
            f"Demo mode requires the '{name}' executable in the active environment or PATH."
        )

    @staticmethod
    def _port_available(port, socktype):
        sock = socket.socket(socket.AF_INET, socktype)
        try:
            if socktype == socket.SOCK_STREAM:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _check_ports(self):
        checks = [(name, port, socket.SOCK_STREAM) for name, port in self.ports.items()]
        checks.append(("epics_ca_udp", self.ports["epics_ca"], socket.SOCK_DGRAM))
        checks.append(("epics_repeater_udp", self.ports["epics_repeater"], socket.SOCK_DGRAM))
        occupied = [f"{name} ({port})" for name, port, kind in checks if not self._port_available(port, kind)]
        if occupied:
            raise DemoRuntimeError(
                "Demo mode cannot start because these loopback ports are occupied: "
                + ", ".join(occupied)
            )

    def _spawn(self, label, args):
        log_path = Path(self.runtime_dir.name) / f"{label}.log"
        log_file = log_path.open("ab", buffering=0)
        try:
            process = subprocess.Popen(
                args,
                cwd=str(Path(__file__).resolve().parent.parent),
                env=self.environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise
        self.processes.append((label, process, log_file, log_path))
        return process

    def _assert_alive(self):
        for label, process, _stream, log_path in self.processes:
            code = process.poll()
            if code is not None:
                try:
                    detail = log_path.read_text(errors="replace")[-3000:]
                except Exception:
                    detail = ""
                raise DemoRuntimeError(
                    f"Demo {label} exited during startup (status {code}).\n{detail}"
                )

    def _wait_tcp(self, port, label):
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            self._assert_alive()
            try:
                with socket.create_connection(("127.0.0.1", int(port)), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
        raise DemoRuntimeError(f"Timed out waiting for demo {label} on port {port}.")

    def _wait_manager_and_open(self):
        from bluesky_queueserver_api.zmq import REManagerAPI

        client = REManagerAPI(
            zmq_control_addr=self.environment["MITR_QSERVER_CONTROL_ADDR"],
            zmq_info_addr=self.environment["MITR_QSERVER_INFO_ADDR"],
        )
        deadline = time.monotonic() + self.startup_timeout
        last_error = None
        while time.monotonic() < deadline:
            self._assert_alive()
            try:
                status = client.status()
                if status.get("manager_state") == "idle":
                    break
            except Exception as exc:
                last_error = exc
            time.sleep(0.2)
        else:
            raise DemoRuntimeError(f"Demo Queue Server did not become ready: {last_error}")

        response = client.environment_open()
        if not response.get("success", False):
            raise DemoRuntimeError(f"Demo Queue Server environment failed to open: {response}")
        while time.monotonic() < deadline:
            self._assert_alive()
            status = client.status()
            if (
                status.get("manager_state") == "idle"
                and status.get("worker_environment_state") == "idle"
            ):
                return
            time.sleep(0.25)
        raise DemoRuntimeError("Timed out opening the demo Queue Server worker environment.")

    def start(self):
        self._check_ports()
        self.runtime_dir = tempfile.TemporaryDirectory(prefix="diffractometer-demo-")
        os.environ.update(self.environment)
        for key in ("QSERVER_ZMQ_PRIVATE_KEY_FOR_SERVER", "QSERVER_ZMQ_PUBLIC_KEY"):
            os.environ.pop(key, None)
        atexit.register(self.stop)
        try:
            redis = self._executable("redis-server")
            proxy = self._executable("bluesky-0MQ-proxy")
            manager = self._executable("start-re-manager")
            self._spawn(
                "redis",
                [redis, "--bind", "127.0.0.1", "--port", str(self.ports["redis"]),
                 "--save", "", "--appendonly", "no", "--dir", self.runtime_dir.name],
            )
            self._wait_tcp(self.ports["redis"], "Redis")
            self._spawn("ioc", [sys.executable, "-m", "diffractometer_controls.demo_ioc"])
            self._wait_tcp(self.ports["epics_ca"], "caproto IOC")
            self._spawn(
                "documents",
                [proxy, "--in-address", self.environment["MITR_DOCUMENT_INPUT_ADDR"],
                 "--out-address", self.environment["MITR_DOCUMENT_ADDR"]],
            )
            # ZMQ ports are not TCP-handshake endpoints until their event loop
            # runs, but successful binds are caught by the process liveness check.
            time.sleep(0.2)
            self._assert_alive()
            self._spawn(
                "queue-server",
                [manager, "--config", str(self.config_path),
                 "--zmq-control-addr", self.environment["MITR_QSERVER_CONTROL_ADDR"],
                 "--zmq-info-addr", self.environment["MITR_QSERVER_INFO_ADDR"],
                 "--redis-addr", self.environment["MITR_REDIS_ADDR"],
                 "--redis-name-prefix", "diffractometer_demo",
                 "--startup-dir", self.environment["MITR_DEMO_STARTUP_DIR"],
                 "--update-existing-plans-devices", "NEVER",
                 "--user-group-permissions", self.environment["MITR_DEMO_PERMISSIONS"],
                 "--user-group-permissions-reload", "ON_STARTUP"],
            )
            self._wait_manager_and_open()
        except BaseException:
            self.stop()
            raise
        return self

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        for _label, process, _stream, _log_path in reversed(self.processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    process.terminate()
        deadline = time.monotonic() + 4.0
        for _label, process, stream, _log_path in reversed(self.processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                process.wait(timeout=1.0)
            finally:
                stream.close()
        self.processes.clear()
        if self.runtime_dir is not None:
            self.runtime_dir.cleanup()
            self.runtime_dir = None
