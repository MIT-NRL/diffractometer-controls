"""Adaptive-focus viewer using streamed image arrays and the shared MTF analysis."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import signal
import sys
import time
import numpy as np
import zmq
from qtpy import QtCore, QtWidgets
import pyqtgraph as pg
from control_ui.core.document_dispatcher import DocumentDispatcherService
from bluesky_widgets.qt.zmq_dispatcher import RemoteDispatcher


class FocusDocumentDispatcher(RemoteDispatcher):
    """Receive a known adaptive run after its start document was published.

    Upstream waits for a new start document and has an unbounded receive loop.
    The parent has already received that document before launching this viewer;
    the window filters descriptors by its supplied run UID instead.
    """

    def _receive_data(self):
        while not self.closed:
            try:
                message = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.02)
                yield
                continue
            prefix, name, document = message.split(b" ", 2)
            if not self._prefix or prefix == self._prefix:
                yield name.decode(), self._deserializer(document)


class DemoFocusWindow(QtWidgets.QWidget):
    document_received = QtCore.Signal(str, object)
    analysis_ready = QtCore.Signal(float, object, object)
    command_ready = QtCore.Signal(str, object)

    def __init__(self, args):
        super().__init__()
        from diffractometer_controls.analysis.focus_online_viewer import QueueServerAdaptiveClient
        from diffractometer_controls.analysis.focus_offline_viewer import analyze_roi
        self.analyze_roi = analyze_roi
        self.args = args
        self.client = QueueServerAdaptiveClient(session_id=args.session_id,
            zmq_control_addr=args.qserver_control_addr, zmq_info_addr=args.qserver_info_addr)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-focus")
        self.positions, self.metrics = [], []
        self.descriptors = set()
        self.finished = False
        self._closed = False
        self.setWindowTitle("DEMO — Adaptive focus · Images in memory")
        self.resize(1000, 700)
        layout = QtWidgets.QVBoxLayout(self)
        self.image = pg.ImageView()
        self.image.ui.roiBtn.hide()
        self.image.ui.menuBtn.hide()
        layout.addWidget(self.image, 2)
        self.plot = pg.PlotWidget(title="MTF50 vs motor position")
        self.plot.setLabel("bottom", "Motor position")
        self.plot.setLabel("left", "MTF50", units="px⁻¹")
        self.curve = self.plot.plot(symbol="o")
        layout.addWidget(self.plot, 1)
        row = QtWidgets.QHBoxLayout()
        self.center = QtWidgets.QDoubleSpinBox()
        self.center.setRange(-100, 100)
        self.center.setDecimals(4)
        row.addWidget(QtWidgets.QLabel("Focus position"))
        row.addWidget(self.center)
        for title, command in (("Go to focus", "go_to_focus"), ("Refine", "scan_around_focus"),
                               ("Extend left", "extend_left"), ("Extend right", "extend_right"),
                               ("Complete", "complete"), ("Abort", "abort")):
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(lambda _=False, c=command: self.submit(c))
            row.addWidget(button)
        layout.addLayout(row)
        self.status = QtWidgets.QLabel("Waiting for the initial scan; storage disabled")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.document_received.connect(self.on_document)
        self.analysis_ready.connect(self.show_analysis)
        self.command_ready.connect(lambda command, result: self.status.setText(f"{command}: {result}"))
        self.documents = DocumentDispatcherService(FocusDocumentDispatcher(args.zmq_address.removeprefix("tcp://")))
        self.documents.subscribe(lambda name, doc: self.document_received.emit(name, doc))
        self.documents.start()
        self.parent_timer = QtCore.QTimer(self)
        self.parent_timer.timeout.connect(self.check_parent)
        self.parent_timer.start(1000)
        # Allow the document subscription to settle before releasing the coarse scan.
        self.ready_timer = QtCore.QTimer(self)
        self.ready_timer.setSingleShot(True)
        self.ready_timer.timeout.connect(lambda: self.submit("viewer_ready"))
        self.ready_timer.start(300)

    def submit(self, command):
        if self._closed:
            return
        payload = {"position": self.center.value(), "center": self.center.value()}
        self.status.setText(f"Sending {command}…")

        def work():
            result = self.client.submit(command, payload)
            try:
                self.command_ready.emit(command, result)
            except RuntimeError:
                pass
        self.executor.submit(work)

    def on_document(self, name, doc):
        if self._closed:
            return
        if name == "descriptor" and doc.get("run_start") == self.args.run_uid:
            self.descriptors.add(doc["uid"])
        if name == "stop" and doc.get("run_start") == self.args.run_uid:
            self.finished = True
            self.status.setText(f"Run {doc.get('exit_status', 'stopped')} — storage disabled")
        if name == "event_page" and doc.get("descriptor") in self.descriptors:
            for index in range(len(doc.get("seq_num", []))):
                self.on_document("event", {"descriptor": doc["descriptor"], "data": {k:v[index] for k,v in doc["data"].items()}})
        if name != "event" or doc.get("descriptor") not in self.descriptors:
            return
        data = doc.get("data", {})
        arrays = [np.asarray(value) for value in data.values() if isinstance(value, (np.ndarray, list)) and np.asarray(value).ndim == 2]
        if not arrays:
            return
        image = arrays[0].copy()
        position = float(data.get(self.args.motor_key, len(self.positions)))

        def work():
            try:
                height, width = image.shape
                result = self.analyze_roi(image, 0, width, 0, height, edge_median_size=3)
                self.analysis_ready.emit(position, image, result)
            except Exception as exc:
                try:
                    self.command_ready.emit("Analysis failed", str(exc))
                except RuntimeError:
                    pass
        self.executor.submit(work)

    def show_analysis(self, position, image, result):
        if self._closed:
            return
        self.image.setImage(image.T, autoRange=False)
        self.positions.append(position)
        self.metrics.append(float(result.mtf50))
        order = np.argsort(self.positions)
        self.curve.setData(np.asarray(self.positions)[order], np.asarray(self.metrics)[order])
        if np.isfinite(self.metrics).any():
            best = int(np.nanargmax(self.metrics))
            self.center.setValue(self.positions[best])
            self.status.setText(f"{len(self.positions)} frames · Best measured focus {self.positions[best]:.4f} · MTF50 {self.metrics[best]:.4f} · Storage disabled")

    def check_parent(self):
        if self.args.parent_pid:
            import psutil
            if not psutil.pid_exists(self.args.parent_pid):
                self.close()

    def closeEvent(self, event):
        if self._closed:
            return super().closeEvent(event)
        self._closed = True
        self.documents.stop()
        self.parent_timer.stop()
        self.ready_timer.stop()
        if not self.finished:
            self.executor.submit(self.client.submit, "abort", {})
        self.executor.shutdown(wait=False, cancel_futures=False)
        super().closeEvent(event)


def main():
    if os.environ.get("MITR_DEMO_ACTIVE") != "1":
        raise SystemExit("This viewer only runs with the isolated demo stack")
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("zmq-address", "run-uid", "session-id", "qserver-control-addr", "qserver-info-addr", "motor-key"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--parent-pid", type=int, default=0)
    args, _unused = parser.parse_known_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    window = DemoFocusWindow(args)
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    signal.signal(signal.SIGINT, lambda *_: window.close())
    window.show()
    app.exec_()


if __name__ == "__main__":
    main()
