"""Reusable read-only diffraction browser with explicit curve fitting."""

import concurrent.futures
import json
import threading

from qtpy import QtCore, QtWidgets

from .history import (
    _is_diffraction_start_doc, _run_metadata, _run_start_stop_docs,
    build_history_snapshot, fetch_recent_history_entries, format_history_entry, history_entry_tooltip,
)
from .panel import AnalysisPanel
from .fitting import fit_curve
from .batch import fit_snapshot
from .viewer import RunDataPanel
from .metadata import environment_readings, read_metadata_details, run_outcome, run_overview
from .widgets import run_status_icon


class DiffractionAnalysisWindow(QtWidgets.QMainWindow):
    _completed = QtCore.Signal(str, int, object)
    _batch_progress = QtCore.Signal(int, int, int)

    def __init__(self, client, *, catalogs, catalog=None, selected_uid="", parent=None, snapshot_loader=build_history_snapshot):
        super().__init__(parent, QtCore.Qt.Window)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Diffraction history and analysis")
        self.resize(1450, 900)
        self._client = client
        self._catalog_paths = dict(catalogs)
        if not self._catalog_paths:
            raise ValueError("At least one diffraction catalog is required.")
        self._snapshot_loader = snapshot_loader
        self._fit_token = 0
        self._batch_token = 0
        self._batch_cancel = None
        self._last_data_selection = None
        self._catalog = None
        self._entries = []
        self._overview = {}
        self._selected_uid = str(selected_uid or "")
        self._generation = 0
        self._load_token = 0
        self._closed = False
        self._futures = set()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="diffraction-analysis")
        self._completed.connect(self._on_completed, QtCore.Qt.QueuedConnection)
        self._batch_progress.connect(self._on_batch_progress, QtCore.Qt.QueuedConnection)

        self.catalog_combo = QtWidgets.QComboBox(self)
        for label, path in self._catalog_paths.items():
            self.catalog_combo.addItem(label, path)
        if catalog in self._catalog_paths:
            self.catalog_combo.setCurrentText(catalog)
        self.catalog_combo.currentIndexChanged.connect(self._catalog_changed)
        self.refresh_button = QtWidgets.QPushButton("Refresh", self)
        self.refresh_button.clicked.connect(self.refresh)
        self.load_more_button = QtWidgets.QPushButton("Load 100 older runs", self)
        self.load_more_button.clicked.connect(self._load_more)
        self.load_more_button.setEnabled(False)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("Filter title, sample, detector, plan, status or ID")
        self.search_edit.textChanged.connect(self._filter_entries)
        self.count_label = QtWidgets.QLabel("Connecting…", self)
        self.run_list = QtWidgets.QListWidget(self)
        self.run_list.setMinimumWidth(240)
        self.run_list.setTextElideMode(QtCore.Qt.ElideRight)
        self.run_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.run_list.currentItemChanged.connect(self._run_selected)

        left = QtWidgets.QWidget(self)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        catalog_row = QtWidgets.QHBoxLayout()
        catalog_row.addWidget(self.catalog_combo, 1)
        catalog_row.addWidget(self.refresh_button)
        left_layout.addLayout(catalog_row)
        left_layout.addWidget(self.search_edit)
        left_layout.addWidget(self.count_label)
        left_layout.addWidget(self.run_list, 1)
        left_layout.addWidget(self.load_more_button)

        self.analysis = AnalysisPanel(self)
        self.plot = self.analysis
        self.analysis.fit_requested.connect(self._request_fit)
        self.analysis.selection_changed.connect(self._invalidate_fit)
        self.analysis.context_changed.connect(self._invalidate_batch)
        self.analysis.batch_requested.connect(self._request_batch)
        self.analysis.cancel_requested.connect(self._cancel_batch)
        self.data_view = RunDataPanel(self)
        self.workspace_tabs = QtWidgets.QTabWidget(self)
        self.workspace_tabs.addTab(self.data_view, "Run data")
        self.workspace_tabs.addTab(self.analysis, "Fitting")
        self.overview_trees = []
        overview_columns = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        for heading in ('Run / sample', 'Acquisition', 'Storage'):
            tree = QtWidgets.QTreeWidget(self)
            tree.setHeaderLabels([heading, 'Value'])
            tree.setColumnWidth(0, 145)
            tree.setIndentation(12)
            tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
            tree.setAlternatingRowColors(True)
            tree.setTextElideMode(QtCore.Qt.ElideMiddle)
            overview_columns.addWidget(tree)
            self.overview_trees.append(tree)
        self.overview_tree = self.overview_trees[0]
        overview_columns.setSizes([400, 400, 400])
        self.copy_overview_button = QtWidgets.QPushButton("Copy overview", self)
        self.copy_overview_button.setEnabled(False)
        self.copy_overview_button.clicked.connect(self._copy_overview)
        overview = QtWidgets.QWidget(self)
        overview_layout = QtWidgets.QVBoxLayout(overview)
        overview_layout.setContentsMargins(4, 4, 4, 4)
        overview_layout.addWidget(overview_columns, 1)
        overview_layout.addWidget(self.copy_overview_button, 0, QtCore.Qt.AlignRight)
        self.metadata_tree = QtWidgets.QTreeWidget(self)
        self.metadata_tree.setHeaderLabels(["Metadata", "Value"])
        self.metadata_tree.setColumnWidth(0, 230)
        self.metadata_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.metadata_filter = QtWidgets.QLineEdit(self)
        self.metadata_filter.setPlaceholderText("Find metadata keys or values…")
        self.metadata_filter.setClearButtonEnabled(True)
        self.metadata_filter.textChanged.connect(self._filter_metadata)
        metadata_panel = QtWidgets.QWidget(self)
        metadata_layout = QtWidgets.QVBoxLayout(metadata_panel)
        metadata_layout.setContentsMargins(4, 4, 4, 4)
        metadata_layout.addWidget(self.metadata_filter)
        metadata_layout.addWidget(self.metadata_tree, 1)
        self.metadata_json = QtWidgets.QPlainTextEdit(self)
        self.metadata_json.setReadOnly(True)
        self.metadata_json.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.acquisition_tree = QtWidgets.QTreeWidget(self)
        self.acquisition_tree.setHeaderLabels(['Scan geometry', 'Recorded value'])
        self.acquisition_tree.setColumnWidth(0, 240)
        self.acquisition_tree.setAlternatingRowColors(True)
        self.acquisition_timing_tree = QtWidgets.QTreeWidget(self)
        self.acquisition_timing_tree.setHeaderLabels(['Timing / detector setup', 'Recorded value'])
        self.acquisition_timing_tree.setColumnWidth(0, 240)
        self.acquisition_timing_tree.setAlternatingRowColors(True)
        acquisition_columns = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        acquisition_columns.addWidget(self.acquisition_tree)
        acquisition_columns.addWidget(self.acquisition_timing_tree)
        acquisition_columns.setSizes([560, 560])
        self.environment_table = QtWidgets.QTableWidget(0, 5, self)
        self.environment_table.setHorizontalHeaderLabels(['Signal', 'Start', 'End', 'Units', 'Monitor stream'])
        self.environment_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.environment_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.environment_table.horizontalHeader().setStretchLastSection(True)
        self.environment_note = QtWidgets.QLabel(self)
        self.environment_note.setWordWrap(True)
        self.environment_note.setTextFormat(QtCore.Qt.PlainText)
        environment = QtWidgets.QWidget(self)
        environment_layout = QtWidgets.QVBoxLayout(environment)
        environment_layout.setContentsMargins(4, 4, 4, 4)
        environment_layout.addWidget(self.environment_table, 1)
        environment_layout.addWidget(self.environment_note)
        self.details_tabs = QtWidgets.QTabWidget(self)
        self.details_tabs.addTab(self.analysis.results_widget, "Fit results")
        self.details_tabs.addTab(overview, "Overview")
        self.details_tabs.addTab(acquisition_columns, 'Acquisition')
        self.details_tabs.addTab(environment, 'Environment / power')
        self.details_tabs.addTab(metadata_panel, "All metadata")
        self.details_tabs.addTab(self.metadata_json, "JSON / copy")
        self.details_tabs.setCurrentIndex(1)
        self.details_tabs.setTabVisible(0, False)
        self.workspace_tabs.currentChanged.connect(self._workspace_changed)
        self.details_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical, self)
        self.details_splitter.addWidget(self.workspace_tabs)
        self.details_splitter.addWidget(self.details_tabs)
        self.details_splitter.setStretchFactor(0, 3)
        self.details_splitter.setStretchFactor(1, 1)
        self.details_splitter.setSizes([560, 220])
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        self.main_splitter.addWidget(left)
        self.main_splitter.addWidget(self.details_splitter)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([285, 1165])
        self.setCentralWidget(self.main_splitter)
        self.statusBar().showMessage("Diffraction data only · read-only viewer")
        self._limit = 100
        QtCore.QTimer.singleShot(0, self.refresh)

    def _submit(self, kind, token, function, *args):
        if self._closed:
            return
        future = self._executor.submit(function, *args)
        self._futures.add(future)

        def completed(result):
            if self._closed or result.cancelled():
                return
            try:
                payload = result.result()
            except Exception as exc:
                payload = {"error": str(exc)}
            try:
                self._completed.emit(kind, token, payload)
            except RuntimeError:
                pass  # The parent GUI may already have been destroyed.

        future.add_done_callback(completed)

    def _catalog_changed(self, *_args):
        self._selected_uid = ""
        self._limit = 100
        self.refresh()

    def _clear_display(self, title="Select a diffraction run"):
        self.plot.reset({"run_title": title})
        self.data_view.reset(title)
        self._last_data_selection = None
        self.workspace_tabs.setCurrentIndex(0)
        self.metadata_tree.clear()
        self.acquisition_tree.clear()
        self.acquisition_timing_tree.clear()
        self.environment_table.setRowCount(0)
        self.environment_note.clear()
        for tree in self.overview_trees:
            tree.clear()
        self._overview = {}
        self.copy_overview_button.setEnabled(False)
        self.metadata_json.clear()

    def refresh(self):
        if self._closed:
            return
        self._generation += 1
        self._load_token += 1
        self._catalog = None
        self._entries = []
        with QtCore.QSignalBlocker(self.run_list):
            self.run_list.clear()
        self._clear_display("Loading diffraction history…")
        self.refresh_button.setEnabled(False)
        self.load_more_button.setEnabled(False)
        self.count_label.setText("Loading diffraction runs…")
        self.statusBar().showMessage("Loading catalog metadata…")
        self._submit("list", self._generation, self._list_task, self.catalog_combo.currentData(), self._limit)

    def _load_more(self):
        self._limit += 100
        self.refresh()

    def _list_task(self, name, limit):
        catalog = self._client[name]
        # The extra record tells us whether another page is available.
        entries = fetch_recent_history_entries(catalog, limit=limit + 1)
        return {"catalog": catalog, "catalog_name": name, "entries": entries[:limit], "has_more": len(entries) > limit}

    def _filter_entries(self, *_args):
        text = self.search_edit.text().strip().casefold()
        shown = 0
        for index, entry in enumerate(self._entries):
            match = text in " ".join(str(entry.get(key, "") or "") for key in ("title", "sample", "detectors", "plan_name", "scan_id", "uid", "status", "exit_status", "reason")).casefold()
            self.run_list.item(index).setHidden(not match)
            shown += int(match)
        self.count_label.setText(f"{shown} shown / {len(self._entries)} loaded · diffraction only")

    def _run_selected(self, item, _previous=None):
        if self._closed or self._catalog is None:
            return
        self._load_token += 1
        if item is None:
            self._selected_uid = ""
            self._clear_display()
            return
        self._selected_uid = str(item.data(QtCore.Qt.UserRole))
        self._clear_display("Loading selected run…")
        self.statusBar().showMessage(f"Loading {self._selected_uid}…")
        # Cancel queued selection reads; already running reads are discarded
        # by their token if another selection or catalog supersedes them.
        for future in tuple(self._futures):
            if future.done():
                self._futures.discard(future)
            else:
                future.cancel()
        self._submit("run", self._load_token, self._run_task, self._catalog, self._selected_uid)

    def _run_task(self, catalog, uid):
        run = catalog[uid]
        start, _ = _run_start_stop_docs(run)
        if not _is_diffraction_start_doc(start):
            raise ValueError("Selected run is not diffraction data.")
        metadata = read_metadata_details(run, _run_metadata(run))
        try:
            snapshot = self._snapshot_loader(run)
        except Exception as exc:
            return {"metadata": metadata, "error": str(exc)}
        metadata = {**metadata, "stream_descriptors": snapshot["descriptor_docs"]}
        if snapshot.get("inferred_metadata"):
            metadata["recovered_plotting_metadata"] = snapshot["inferred_metadata"]
        return {"metadata": metadata, "snapshot": snapshot}

    @QtCore.Slot(str, int, object)
    def _on_completed(self, kind, token, payload):
        if self._closed:
            return
        # Release completed payloads even when their selection token is stale.
        self._futures = {future for future in self._futures if not future.done()}
        if token != {"list": self._generation, "run": self._load_token, "fit": self._fit_token, "batch": self._batch_token}[kind]:
            return
        if kind == "list":
            self.refresh_button.setEnabled(True)
        if kind == "batch":
            self.analysis.set_batch_busy(False)
        if payload.get("error"):
            if kind in {"fit", "batch"}:
                self.analysis.message.setText(f"Fit failed: {payload['error']}")
                return
            if kind == 'run':
                self.data_view.reset('Run data unavailable')
                self.analysis.reset({'run_title': 'Run data unavailable'})
                message = f"Run data unavailable\n{payload['error']}"
                self.data_view.plot.set_message(message)
                self.analysis.plot.set_message(message)
                item = self.run_list.currentItem()
                if item is not None:
                    item.setIcon(run_status_icon('', unavailable=True))
                    item.setToolTip(item.toolTip() + '\n' + str(payload['error']))
            if "metadata" in payload:
                self._set_metadata(payload["metadata"])
            elif kind == 'list':
                self._clear_display("Unable to load diffraction history")
            if kind == "list":
                self.count_label.setText("History unavailable")
            self.statusBar().showMessage(f"Unable to load: {payload['error']}")
            return
        if kind == "fit":
            self.analysis.show_fit(payload["fit_result"])
            if self.workspace_tabs.currentIndex() == 1:
                self.details_tabs.setCurrentIndex(0)
            return
        if kind == "batch":
            self.analysis.show_batch(payload["batch"])
            if self.workspace_tabs.currentIndex() == 1:
                self.details_tabs.setCurrentIndex(0)
            return
        if kind == "list":
            self._catalog = payload["catalog"]
            self._entries = payload["entries"]
            self.load_more_button.setEnabled(payload["has_more"])
            with QtCore.QSignalBlocker(self.run_list):
                for entry in self._entries:
                    text = format_history_entry(entry, multiline=True)
                    item = QtWidgets.QListWidgetItem(text, self.run_list)
                    item.setIcon(run_status_icon(entry['status'], unavailable=bool(entry.get('load_error')) or entry.get('file_status') == 'failed'))
                    item.setData(QtCore.Qt.UserRole, entry["uid"])
                    item.setToolTip(history_entry_tooltip(entry))
            self._filter_entries()
            self.statusBar().showMessage(f"Catalog: {payload['catalog_name']} · diffraction data only · read-only")
            target = next((index for index, entry in enumerate(self._entries) if entry["uid"] == self._selected_uid), 0)
            if self._entries:
                self.run_list.setCurrentRow(target)
            return
        self._set_metadata(payload["metadata"], point_count=len(payload['snapshot']['rows']))
        self.analysis.set_snapshot(payload["snapshot"])
        self.data_view.set_snapshot(payload["snapshot"])
        snapshot = payload['snapshot']
        outcome = run_outcome(snapshot.get('start'), snapshot.get('stop'))
        self.statusBar().showMessage(
            f"{self._selected_uid} · {len(snapshot['rows'])} points · {outcome['status']}"
        )

    def _workspace_changed(self, index):
        fitting = index == 1
        self.details_tabs.setTabVisible(0, fitting)
        self.details_tabs.setCurrentIndex(0 if fitting else 1)
        height = max(600, sum(self.details_splitter.sizes()))
        ratio = 0.60 if fitting else 0.74
        self.details_splitter.setSizes([int(height * ratio), int(height * (1 - ratio))])
        selection = self.data_view.fitting_selection()
        if fitting and selection is not None and selection != self._last_data_selection:
            field, point, axis = selection
            self.analysis.select_selection(field, point, axis)
            self._last_data_selection = selection

    def _invalidate_batch(self):
        self._batch_token += 1
        if self._batch_cancel is not None:
            self._batch_cancel.set()
            self._batch_cancel = None
        self.analysis.set_batch_busy(False)

    def _cancel_batch(self):
        if self._batch_cancel is not None:
            self._batch_cancel.set()
            self.analysis.message.setText('Stopping after the current acquisition point…')

    def _request_batch(self, curve, config):
        self._fit_token += 1
        self._invalidate_batch()
        self._batch_cancel = threading.Event()
        token = self._batch_token
        self.analysis.set_batch_busy(True, 0, len(self.analysis.snapshot['rows']))
        self._submit('batch', token, self._batch_task, self.analysis.snapshot, curve, config, token, self._batch_cancel)

    def _batch_task(self, snapshot, curve, config, token, cancel):
        def progress(done, total):
            if not self._closed:
                try:
                    self._batch_progress.emit(token, done, total)
                except RuntimeError:
                    pass
        return {'batch': fit_snapshot(snapshot, curve['field'], curve['x_field'], config,
                                      progress=progress, cancelled=cancel.is_set)}

    @QtCore.Slot(int, int, int)
    def _on_batch_progress(self, token, done, total):
        if not self._closed and token == self._batch_token:
            self.analysis.set_batch_busy(True, done, total)

    def _invalidate_fit(self):
        self._fit_token += 1

    def _request_fit(self, curve, config):
        self._fit_token += 1
        self._submit("fit", self._fit_token, self._fit_task, curve, config)

    @staticmethod
    def _fit_task(curve, config):
        return {"fit_result": fit_curve(curve["x"], curve["y"], config)}

    def _set_metadata(self, metadata, *, point_count=None):
        self.metadata_tree.clear()
        for tree in self.overview_trees:
            tree.clear()
        self._overview = run_overview(metadata, point_count=point_count)
        self.copy_overview_button.setEnabled(True)
        self.metadata_json.setPlainText(json.dumps(metadata, indent=2, default=str, ensure_ascii=False))

        def add(parent, key, value):
            item = QtWidgets.QTreeWidgetItem(parent, [str(key), ""])
            if isinstance(value, dict):
                for name, child in value.items():
                    add(item, name, child)
            elif isinstance(value, (list, tuple)) and any(isinstance(v, dict) for v in value):
                for index, child in enumerate(value):
                    add(item, index, child)
            else:
                item.setText(1, json.dumps(value, ensure_ascii=False, default=str))
            item.setToolTip(0, str(key))
            item.setToolTip(1, item.text(1))
            return item

        def add_overview(parent, key, value):
            item = QtWidgets.QTreeWidgetItem(parent, [str(key), ''])
            if isinstance(value, dict):
                for name, child in value.items():
                    add_overview(item, name, child)
            else:
                text = ', '.join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
                item.setText(1, text)
                item.setToolTip(1, text)
            return item

        # Independent columns keep outcome, point counts and export health
        # visible together even when the lower pane is only a few rows tall.
        run = self._overview['Run']
        timing = {key: value for key, value in run.items() if key in {'Started', 'Ended', 'Duration'}}
        run_fields = {key: value for key, value in run.items() if key not in timing}
        sample = self._overview.get('Sample / experiment', {})
        columns = [
            {**run_fields, **sample},
            {**self._overview['Acquisition'], **timing},
            self._overview['Storage'],
        ]
        for tree, values in zip(self.overview_trees, columns):
            for key, value in values.items():
                item = add_overview(tree.invisibleRootItem(), key, value)
                if key == 'Status':
                    item.setToolTip(1, run.get('Status detail', value))
        self.acquisition_tree.clear()
        self.acquisition_timing_tree.clear()
        acquisition = {
            **{key: value for key, value in self._overview['Acquisition details'].items() if key not in {'Timing', 'Detector settings'}},
            'Point counts': {key: value for key, value in self._overview['Acquisition'].items() if key.startswith('Points ')},
        }
        for key, value in acquisition.items():
            add_overview(self.acquisition_tree.invisibleRootItem(), key, value)
        self.acquisition_tree.expandToDepth(1)
        timing_details = {**timing, **self._overview['Acquisition details']['Timing']}
        if self._overview['Acquisition details'].get('Detector settings'):
            timing_details['Detector settings'] = self._overview['Acquisition details']['Detector settings']
        for key, value in timing_details.items():
            add_overview(self.acquisition_timing_tree.invisibleRootItem(), key, value)
        self.acquisition_timing_tree.expandToDepth(0)
        readings = environment_readings(metadata)
        self.environment_table.setRowCount(len(readings))
        for row, reading in enumerate(readings):
            values = [reading['signal'], reading['start'], reading['end'], reading['units'], reading['monitor'] or 'Baseline only']
            for column, value in enumerate(values):
                text = 'Not recorded' if value is None else f'{value:.7g}' if isinstance(value, (int, float)) else str(value)
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(f"{text}\n{reading['source']}")
                self.environment_table.setItem(row, column, item)
        self.environment_table.resizeColumnsToContents()
        details = metadata.get('viewer_details') or {}
        note = ('Start/end are recorded baseline readings, not time-series extrema or averages. '
                'Reactor power is listed first; duplicate setpoints remain in All metadata.' if readings else
                'No baseline readbacks were recorded or could be read for this run.')
        if details.get('monitor_streams'):
            note += '\nRecorded monitor streams: ' + ', '.join(details['monitor_streams'])
        if details.get('notes'):
            note += '\n' + '\n'.join(details['notes'])
        self.environment_note.setText(note)
        for key in ('Baseline / environment', 'Plotting metadata recovered for this view'):
            if key in self._overview:
                add_overview(self.overview_trees[1].invisibleRootItem(), key, self._overview[key])

        root = self.metadata_tree.invisibleRootItem()
        preferred = ["start", "stop", "file_export", "baseline_readings", "metadata_enrichment"]
        ordered = [key for key in preferred if key in metadata]
        ordered.extend(key for key in metadata if key not in ordered)
        for key in ordered:
            item = add(root, key, metadata[key])
            if key in {"start", "file_export"}:
                item.setExpanded(True)
        self._filter_metadata()

    def _filter_metadata(self, *_args):
        text = self.metadata_filter.text().strip().casefold()

        def filter_item(item, ancestor_match=False):
            matches = bool(text) and text in f'{item.text(0)} {item.text(1)}'.casefold()
            children_match = False
            for index in range(item.childCount()):
                children_match = filter_item(item.child(index), ancestor_match or matches) or children_match
            visible = not text or ancestor_match or matches or children_match
            item.setHidden(not visible)
            if text and children_match:
                item.setExpanded(True)
            return visible

        for index in range(self.metadata_tree.topLevelItemCount()):
            filter_item(self.metadata_tree.topLevelItem(index))

    def _copy_overview(self):
        lines = []

        def append(key, value, depth=0):
            if isinstance(value, dict):
                lines.append('  ' * depth + str(key))
                for name, child in value.items():
                    append(name, child, depth + 1)
            else:
                lines.append('  ' * depth + f'{key}: {value}')

        for key, value in self._overview.items():
            append(key, value)
        QtWidgets.QApplication.clipboard().setText('\n'.join(lines))

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        if self._batch_cancel is not None:
            self._batch_cancel.set()
        self._generation += 1
        self._load_token += 1
        self._executor.shutdown(wait=False, cancel_futures=True)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
