"""Desktop Tiled dashboard. All service calls run on one serialized worker."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import weakref

from qtpy import QtCore, QtWidgets
from control_ui.core.logging_config import configure_http_logging

from . import auth
from .service import ManagerService, failure


def parse_date(value):
    """JSON uses strings; Tiled's MessagePack client also returns datetimes."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(value, timezone.utc)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    else:
        raise ValueError('Unsupported timestamp')
    # Naive API timestamps are UTC, not the GUI computer's local time.
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def date(value):
    try:
        return parse_date(value).astimezone().strftime('%Y-%m-%d %H:%M')
    except (ValueError, TypeError, OverflowError, OSError):
        return '—'


def key_expiry(value, *, now=None):
    if value is None or value == '':
        return 'No expiry'
    try:
        remaining = parse_date(value) - (now or datetime.now(timezone.utc))
        note = ' ⚠ Expired' if remaining.total_seconds() <= 0 else ' ⚠ ≤30 days' if remaining.total_seconds() <= 30 * 86400 else ''
        return date(value) + note
    except (ValueError, TypeError, OverflowError, OSError):
        return 'Invalid expiry'


def table(headers):
    widget = QtWidgets.QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
    widget.setAlternatingRowColors(True)
    widget.verticalHeader().hide()
    widget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    return widget


def fill(widget, rows):
    widget.setRowCount(len(rows))
    for row, values in enumerate(rows):
        for col, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(str(value if value is not None else '—'))
            item.setToolTip(item.text())
            widget.setItem(row, col, item)


class InstallDialog(QtWidgets.QDialog):
    def __init__(self, uri, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Install reader key on this computer')
        self.resize(600, 250)
        form = QtWidgets.QFormLayout(self)
        self.uri = QtWidgets.QLineEdit(uri)
        self.key = QtWidgets.QLineEdit()
        self.key.setEchoMode(QtWidgets.QLineEdit.Password)
        self.replace = QtWidgets.QCheckBox('Replace this computer’s existing reader settings')
        form.addRow('Server', self.uri)
        form.addRow('Reader key', self.key)
        form.addRow(self.replace)
        load = QtWidgets.QPushButton('Import setup file…')
        load.clicked.connect(self.load)
        form.addRow(load)
        text = QtWidgets.QLabel('The key will be verified online and saved privately for this computer.\n'
                               'Setup files contain a secret; remove transferred copies after installation.')
        text.setWordWrap(True)
        form.addRow(text)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Import reader setup', '', 'JSON (*.json)')
        if path:
            try:
                setup = auth.load_setup(path)
                self.uri.setText(setup['uri'])
                self.key.setText(setup['api_key'])
            except Exception:
                QtWidgets.QMessageBox.warning(self, 'Invalid setup', 'Could not read a valid reader setup file.')


class SecretDialog(QtWidgets.QDialog):
    def __init__(self, uri, label, secret, prefix, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Reader key created — save before closing')
        self.resize(640, 220)
        self.secret = secret
        self.uri, self.label = uri, label
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f'Computer: {label}   •   Key ID: {prefix}'))
        layout.addWidget(QtWidgets.QLabel('This secret cannot be retrieved again. Transfer it securely to the intended PC.'))
        self.edit = QtWidgets.QLineEdit(secret)
        self.edit.setReadOnly(True)
        self.edit.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(self.edit)
        row = QtWidgets.QHBoxLayout()
        for title, action in [('Copy key', self.copy), ('Export setup file…', self.export), ('Close', self.accept)]:
            button = QtWidgets.QPushButton(title)
            button.clicked.connect(action)
            row.addWidget(button)
        layout.addLayout(row)

    def copy(self):
        QtWidgets.QApplication.clipboard().setText(self.secret)

    def export(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export setup — contains a secret',
            str(Path.home() / 'tiled-reader-setup.json'), 'JSON (*.json)')
        if path:
            try:
                auth.export_setup(path, self.uri, self.secret, self.label)
            except Exception:
                QtWidgets.QMessageBox.warning(self, 'Export failed',
                    'Key is still valid. Choose a new private file path outside Git checkouts, '
                    'or copy the key before closing.')

    def done(self, result):
        if QtWidgets.QApplication.clipboard().text() == self.secret:
            QtWidgets.QApplication.clipboard().clear()
        self.edit.clear()
        self.secret = ''
        super().done(result)


class TiledManagerWindow(QtWidgets.QMainWindow):
    completed = QtCore.Signal(int, str, object, object)

    def __init__(self, uri, *, catalogs, parent=None, service_factory=ManagerService, policy_path=None):
        super().__init__(parent, QtCore.Qt.Window)
        configure_http_logging()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle('Tiled Manager')
        self.resize(1180, 850)
        self.catalogs = dict(catalogs)
        self.policy_path = policy_path
        self.service_factory = service_factory
        self.service = service_factory(uri, self.catalogs)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tiled-manager')
        self.generation = 0
        self.busy = False
        self.closed = False
        self.manager_active = False
        self.last_activity = time.monotonic()
        self.snapshot = {}
        self.run_rows = []
        self.key_rows = []
        self.events = []
        self.loaded_catalog = None
        self.secret_dialog = None
        self.completed.connect(self._complete, QtCore.Qt.QueuedConnection)
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('Tiled Manager')
        font = title.font(); font.setPointSize(font.pointSize() + 7); font.setBold(True); title.setFont(font)
        header.addWidget(title)
        header.addStretch()
        self.server = QtWidgets.QLineEdit(uri)
        self.server.setMinimumWidth(340)
        header.addWidget(self.server)
        self.connect_button = QtWidgets.QPushButton('Connect')
        self.connect_button.clicked.connect(self.connect_server)
        header.addWidget(self.connect_button)
        self.refresh_button = QtWidgets.QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)
        self.banner = QtWidgets.QLabel('Connecting…')
        self.banner.setWordWrap(True)
        layout.addWidget(self.banner)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._overview()
        self._catalogs()
        self._keys()
        self._diagnostics()
        self.tabs.currentChanged.connect(self._tab_changed)
        self.statusBar().showMessage('HTTP transport • Reader credentials may cross the LAN unencrypted')
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(10000)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        QtWidgets.QApplication.instance().installEventFilter(self)
        QtCore.QTimer.singleShot(0, self.connect_server)

    def _page(self, name):
        page = QtWidgets.QWidget()
        self.tabs.addTab(page, name)
        return QtWidgets.QVBoxLayout(page)

    def _tab_changed(self, index):
        if index == 1 and not self.run_rows:
            self.load_runs()

    def _overview(self):
        layout = self._page('Overview')
        cards = QtWidgets.QHBoxLayout()
        self.cards = []
        for title in ('API / identity', 'Catalogs', 'Selected data check'):
            box = QtWidgets.QGroupBox(title)
            inner = QtWidgets.QVBoxLayout(box)
            value = QtWidgets.QLabel('Not checked')
            value.setWordWrap(True)
            inner.addWidget(value)
            cards.addWidget(box, 1)
            self.cards.append(value)
        layout.addLayout(cards)
        self.catalog_table = table(['Catalog', 'Runs', 'Latest run', 'Check'])
        self.catalog_table.setMaximumHeight(200)
        layout.addWidget(self.catalog_table)
        acquisition = QtWidgets.QGroupBox('Acquisition & storage')
        inner = QtWidgets.QVBoxLayout(acquisition)
        self.acquisition = QtWidgets.QLabel('No acquisition information available')
        self.acquisition.setWordWrap(True)
        inner.addWidget(self.acquisition)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        inner.addWidget(self.progress)
        inner.addWidget(QtWidgets.QLabel('Disk space and process health require host monitoring (not included).'))
        layout.addWidget(acquisition)
        self.event_view = QtWidgets.QPlainTextEdit()
        self.event_view.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel('Checks & recent events — this window’s session only'))
        layout.addWidget(self.event_view, 1)

    def _catalogs(self):
        layout = self._page('Catalogs & files')
        row = QtWidgets.QHBoxLayout()
        self.catalog = QtWidgets.QComboBox()
        self.catalog.addItems(self.catalogs)
        self.catalog.currentTextChanged.connect(self.load_runs)
        row.addWidget(self.catalog)
        button = QtWidgets.QPushButton('Load latest 100 runs')
        button.clicked.connect(self.load_runs)
        row.addWidget(button)
        self.check_button = QtWidgets.QPushButton('Check selected run data')
        self.check_button.clicked.connect(self.check_run)
        row.addWidget(self.check_button)
        layout.addLayout(row)
        self.runs_table = table(['Scan / title', 'Started', 'Outcome', 'Export / registration', 'Committed'])
        self.runs_table.itemSelectionChanged.connect(self._run_selected)
        layout.addWidget(self.runs_table, 1)
        self.run_details = QtWidgets.QPlainTextEdit()
        self.run_details.setReadOnly(True)
        self.run_details.setMaximumHeight(220)
        layout.addWidget(self.run_details)

    def _keys(self):
        layout = self._page('Access & keys')
        row = QtWidgets.QHBoxLayout()
        self.login_button = QtWidgets.QPushButton('Manager sign in…')
        self.login_button.clicked.connect(self.login_manager)
        row.addWidget(self.login_button)
        self.logout_button = QtWidgets.QPushButton('Lock management')
        self.logout_button.clicked.connect(self.lock_manager)
        row.addWidget(self.logout_button)
        row.addStretch()
        install = QtWidgets.QPushButton('Install reader key on this PC…')
        install.clicked.connect(self.install)
        row.addWidget(install)
        login = QtWidgets.QPushButton('Reader password login…')
        login.clicked.connect(self.reader_login)
        row.addWidget(login)
        layout.addLayout(row)
        self.policy_label = QtWidgets.QLabel()
        self.policy_label.setWordWrap(True)
        layout.addWidget(self.policy_label)
        self.legacy = QtWidgets.QLabel('Local reader key: not checked')
        self.legacy.setWordWrap(True)
        layout.addWidget(self.legacy)
        self.keys_table = table(['Computer / label', 'Key ID', 'Created', 'Last used', 'Expires', 'Permissions'])
        layout.addWidget(self.keys_table, 1)
        row = QtWidgets.QHBoxLayout()
        self.label = QtWidgets.QLineEdit()
        self.label.setPlaceholderText('Computer label, e.g. office-laptop')
        self.label.setMaxLength(120)
        row.addWidget(self.label, 1)
        self.days = QtWidgets.QSpinBox()
        self.days.setRange(1, 3650)
        self.days.setValue(365)
        self.days.setSuffix(' days')
        row.addWidget(self.days)
        self.issue_button = QtWidgets.QPushButton('Issue reader key…')
        self.issue_button.clicked.connect(self.issue)
        row.addWidget(self.issue_button)
        layout.addLayout(row)
        row = QtWidgets.QHBoxLayout()
        self.rotate_button = QtWidgets.QPushButton('Create replacement for selected…')
        self.rotate_button.clicked.connect(self.rotate)
        row.addWidget(self.rotate_button)
        self.revoke_button = QtWidgets.QPushButton('Revoke selected…')
        self.revoke_button.clicked.connect(self.revoke)
        row.addWidget(self.revoke_button)
        layout.addLayout(row)
        note = QtWidgets.QLabel('Issued keys have fixed read-only permissions. Labels are not hardware bindings.\n'
            'Rotation keeps the old key valid until you explicitly revoke it after installation.\n'
            'Management uses native own-account permissions; other accounts’ keys cannot be changed.')
        note.setWordWrap(True)
        layout.addWidget(note)
        self._management_buttons()

    def _diagnostics(self):
        layout = self._page('Diagnostics')
        self.diagnostics = QtWidgets.QPlainTextEdit()
        self.diagnostics.setReadOnly(True)
        layout.addWidget(self.diagnostics)
        export = QtWidgets.QPushButton('Export sanitized diagnostics…')
        export.clicked.connect(self.export_diagnostics)
        layout.addWidget(export)

    def _event(self, message):
        self.events.append(f'{datetime.now().strftime("%H:%M:%S")}  {message}')
        self.events = self.events[-100:]
        self.event_view.setPlainText('\n'.join(self.events))

    def submit(self, name, function):
        if self.closed or self.busy:
            return False
        self.busy = True
        generation = self.generation
        reference = weakref.ref(self)
        future = self.executor.submit(function)
        def done(future):
            window = reference()
            if window is None or window.closed:
                return
            try:
                result, error = future.result(), None
            except Exception as exc:
                result, error = None, failure(exc)
            try:
                window.completed.emit(generation, name, result, error)
            except RuntimeError:
                pass
        future.add_done_callback(done)
        self._management_buttons()
        return True

    def _complete(self, generation, name, result, error):
        if self.closed or generation != self.generation:
            return
        self.busy = False
        if error:
            self.banner.setText(error + ' Previous status is stale.')
            self._event(f'{name}: {error}')
            if name in ('connect', 'refresh'):
                self.cards[0].setText(error)
                self.cards[0].setStyleSheet('color: #d79624;')
            if name == 'check':
                self.cards[2].setText(error)
            if name == 'issue':
                QtWidgets.QMessageBox.warning(self, 'Issuance not confirmed',
                    'Do not retry immediately. Refresh the key list: a key may have been created, '
                    'but its secret cannot be retrieved. Revoke that entry before issuing another.')
        elif name in ('connect', 'refresh', 'reader-login'):
            self._show_snapshot(result)
        elif name == 'runs':
            label, rows = result
            if label != self.catalog.currentText():
                QtCore.QTimer.singleShot(0, self.load_runs)
                return
            self.loaded_catalog = label
            self.run_rows = rows
            fill(self.runs_table, [(f'#{r["scan_id"]} {r["title"]}', date(r['time']), r['status'],
                                   r['export_status'], r['committed_points']) for r in rows])
        elif name == 'check':
            label, uid, checked = result
            self.cards[2].setText(f'{checked["state"]}\n{label} / {uid[:8]} • {date(time.time())}')
            row = self.runs_table.currentRow()
            if (self.loaded_catalog == label and 0 <= row < len(self.run_rows)
                    and self.run_rows[row]['uid'] == uid):
                self.run_details.appendPlainText('\nData check: ' + json.dumps(checked, indent=2))
            self._event('Selected data check: ' + checked['state'])
        elif name in ('manager-login', 'keys', 'revoke'):
            if name == 'manager-login':
                self.manager_active = True
                self.last_activity = time.monotonic()
            self._show_keys(result)
            self._event('Manager key list refreshed' if name != 'revoke' else 'Selected key revoked')
        elif name == 'issue':
            self._event('Reader key issued; old keys remain valid')
            self.secret_dialog = SecretDialog(self.service.uri, result['note'], result.pop('secret'), result['first_eight'], self)
            self.secret_dialog.exec_()
            self.secret_dialog = None
            self.submit('keys', self.service.keys)
        elif name == 'install':
            self.server.setText(result[1])
            self._event('Reader key installed on this computer')
            self.banner.setText('Reader key installed. Connect to its server to use it.')
            QtCore.QTimer.singleShot(0, self.connect_server)
        self._management_buttons()

    def _show_snapshot(self, result):
        self.snapshot = result
        self.banner.setText(f'Connected as {result["identity"]}  •  {result["uri"]}  •  '
                            f'Tiled {result["server_version"]} / client {result["client_version"]}  •  '
                            f'Checked {date(result["checked_at"])}')
        self.cards[0].setText(f'● Reachable\n{result["response_ms"]} ms for catalog checks')
        self.cards[0].setStyleSheet('color: #26a269;')
        entries = result['catalogs']
        readable = sum(v['state'] == 'Readable' for v in entries.values())
        self.cards[1].setText(f'{readable} / {len(entries)} readable')
        self.cards[1].setStyleSheet('color: #26a269;' if readable == len(entries) else 'color: #d79624;')
        fill(self.catalog_table, [(label, row.get('count'),
              (row.get('latest') or {}).get('title', '—'), row['state']) for label, row in entries.items()])
        latest = max((row['latest'] for row in entries.values() if row.get('latest')),
                     key=lambda r: r.get('time') or 0, default=None)
        if latest:
            points, expected = latest['committed_points'], latest['expected_points']
            self.acquisition.setText(f'Latest run #{latest["scan_id"]}: {latest["title"]}\n'
                f'{latest["status"]} • Export: {latest["export_status"]}\n'
                f'Committed points: {points if points is not None else "Not recorded"} • '
                f'Started: {date(latest["time"])}\n'
                'A missing stop document does not prove acquisition is still running.')
            self.progress.setRange(0, max(int(expected or points or 1), 1))
            self.progress.setValue(int(points or 0))
        key = result.get('reader_key')
        self.legacy.setText('Installed reader key: ' + (key['first_eight'] if key else 'password session') +
            '. Manager controls apply only to keys owned by its identity; older reader-account keys are preserved.')
        # Allowlist only; never dump context, whoami sessions, environment or exceptions.
        safe = {k: result[k] for k in ('uri', 'client_version', 'server_version', 'checked_at', 'response_ms', 'identity')}
        safe['catalogs'] = {label: {'state': row['state'], 'count': row.get('count')}
                            for label, row in entries.items()}
        self.diagnostics.setPlainText(json.dumps(safe, indent=2))
        self.statusBar().showMessage(('HTTP — unencrypted LAN traffic' if self.service.uri.startswith('http:') else 'HTTPS') +
                                     ' • Host monitoring not configured')

    def _show_keys(self, keys):
        self.key_rows = keys
        fill(self.keys_table, [(k.get('note'), k['first_eight'], date(k.get('time_created')),
             date(k.get('latest_activity')), key_expiry(k.get('expiration_time')),
             ', '.join(k.get('scopes') or [])) for k in keys])

    def _management_buttons(self):
        allowed = auth.management_allowed(self.service.uri, self.policy_path)
        self.login_button.setEnabled(allowed and not self.busy and not self.manager_active)
        self.logout_button.setEnabled(self.manager_active)
        for widget in (self.issue_button, self.rotate_button, self.revoke_button):
            widget.setEnabled(self.manager_active and not self.busy)
        self.policy_label.setText('Management unlocked — locks after 10 minutes of inactivity.' if self.manager_active else
            'Management locked. Sign in with the dedicated manager account.' if allowed else
            'Create/revoke keys on the configured control PC (SSH/X11 is supported). Reader installation is available here.')

    def connect_server(self):
        uri = self.server.text().strip()
        try:
            auth.server_id(uri)
        except ValueError:
            self.banner.setText('Enter a valid HTTP/HTTPS server URL without credentials.')
            return
        if self.busy and uri == self.service.uri:
            return
        self.generation += 1
        self.manager_active = False
        self.busy = False
        previous = self.service
        self.service = self.service_factory(uri, self.catalogs)
        self.executor.submit(previous.close)
        self.snapshot = {}
        self.loaded_catalog = None
        self.key_rows = self.run_rows = []
        fill(self.keys_table, []); fill(self.runs_table, []); fill(self.catalog_table, [])
        self.diagnostics.clear(); self.run_details.clear()
        self.acquisition.setText('Not checked')
        for card in self.cards:
            card.setText('Not checked')
        self.submit('connect', self.service.connect)

    def refresh(self):
        if not self.busy:
            self.submit('refresh', self.service.refresh)

    def _tick(self):
        if self.manager_active and time.monotonic() - self.last_activity >= 600:
            self.lock_manager()
        # A background refresh must not take the worker while the user enters
        # credentials, installs a key, or confirms revocation in a modal dialog.
        # Otherwise its accepted action can be silently rejected as busy.
        if QtWidgets.QApplication.activeModalWidget() is not None:
            return
        if self.isVisible() and not self.busy:
            if self.tabs.currentIndex() == 2 and self.manager_active:
                self.submit('keys', self.service.keys)
            else:
                self.refresh()

    def eventFilter(self, watched, event):
        if self.manager_active and event.type() in (QtCore.QEvent.MouseButtonPress, QtCore.QEvent.KeyPress, QtCore.QEvent.Wheel):
            if isinstance(watched, QtWidgets.QWidget) and (watched is self or self.isAncestorOf(watched)):
                self.last_activity = time.monotonic()
        return super().eventFilter(watched, event)

    def load_runs(self, *_):
        if self.loaded_catalog != self.catalog.currentText():
            self.run_rows = []
            fill(self.runs_table, [])
            self.run_details.clear()
        if not self.busy and self.service.client is not None:
            label = self.catalog.currentText()
            self.submit('runs', lambda service=self.service: (label, service.runs(label)))

    def _run_selected(self):
        row = self.runs_table.currentRow()
        if 0 <= row < len(self.run_rows):
            self.run_details.setPlainText(json.dumps(self.run_rows[row], indent=2))

    def check_run(self):
        row = self.runs_table.currentRow()
        if self.loaded_catalog == self.catalog.currentText() and 0 <= row < len(self.run_rows):
            label, uid = self.catalog.currentText(), self.run_rows[row]['uid']
            self.submit('check', lambda service=self.service: (label, uid, service.check_run(label, uid)))

    def login_manager(self):
        if not auth.management_allowed(self.service.uri, self.policy_path) or self.busy:
            return
        password, ok = QtWidgets.QInputDialog.getText(self, 'Manager login', 'Dedicated manager password:', QtWidgets.QLineEdit.Password)
        if ok:
            self.submit('manager-login', lambda service=self.service: service.login_manager(password, policy_path=self.policy_path))

    def lock_manager(self):
        self.manager_active = False
        self.generation += 1
        self.busy = False
        if self.secret_dialog:
            self.secret_dialog.reject()
        self.key_rows = []
        fill(self.keys_table, [])
        self.executor.submit(self.service.close_manager)
        self._management_buttons()

    def reader_login(self):
        if self.busy:
            return
        username, ok = QtWidgets.QInputDialog.getText(self, 'Reader login', 'Username:', text='4dh4-reader')
        if not ok:
            return
        password, ok = QtWidgets.QInputDialog.getText(self, 'Reader login', 'Reader password:', QtWidgets.QLineEdit.Password)
        if ok:
            self.submit('reader-login', lambda service=self.service: service.connect(username, password))

    def install(self):
        if self.busy:
            return
        dialog = InstallDialog(self.service.uri, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            uri, key, replace = dialog.uri.text().strip(), dialog.key.text(), dialog.replace.isChecked()
            dialog.key.clear()
            self.submit('install', lambda: (auth.install_reader(uri, key, replace=replace), uri))

    def issue(self, *_):
        if self.manager_active and not self.busy:
            label, days = self.label.text(), self.days.value()
            self.submit('issue', lambda service=self.service: service.issue(label, days))

    def rotate(self):
        row = self.keys_table.currentRow()
        if 0 <= row < len(self.key_rows):
            self.label.setText(self.key_rows[row].get('note') or '')
            self.issue()

    def revoke(self):
        row = self.keys_table.currentRow()
        if not self.manager_active or self.busy or not 0 <= row < len(self.key_rows):
            return
        key = self.key_rows[row]
        answer = QtWidgets.QMessageBox.question(self, 'Revoke reader key',
            f'Revoke {key.get("note") or "unnamed"} ({key["first_eight"]})?\nEvery copy of this key will lose access.')
        if answer == QtWidgets.QMessageBox.Yes:
            self.submit('revoke', lambda service=self.service: service.revoke(key['first_eight']))

    def export_diagnostics(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export sanitized diagnostics', '', 'JSON (*.json)')
        if path:
            try:
                with open(path, 'x', encoding='utf-8') as stream:
                    stream.write(self.diagnostics.toPlainText())
            except OSError:
                QtWidgets.QMessageBox.warning(self, 'Export failed', 'Choose a new writable file path.')

    def closeEvent(self, event):
        self.closed = True
        self.timer.stop()
        QtWidgets.QApplication.instance().removeEventFilter(self)
        if self.secret_dialog:
            self.secret_dialog.reject()
        self.executor.submit(self.service.close)
        self.executor.shutdown(wait=False)
        super().closeEvent(event)
