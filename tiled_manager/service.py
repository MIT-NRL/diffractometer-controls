"""Synchronous Tiled operations, called exclusively by the window's worker."""

from datetime import datetime, timezone
from importlib.metadata import version
import time
import json

import httpx
from tiled.client import from_context

from . import auth


def failure(exc):
    """Allowlisted error messages: never echo HTTP bodies, URLs, or secrets."""
    response = getattr(exc, 'response', None)
    code = getattr(response, 'status_code', None)
    if code in (401, 403) or isinstance(exc, PermissionError):
        return 'Unauthorized: install a reader key or check this account’s permissions.'
    if isinstance(exc, httpx.TransportError):
        return 'Unreachable: check the server address, VPN, and Tiled service.'
    if code == 404:
        return 'Unavailable: the catalog, run, or registered file was not found.'
    if isinstance(exc, (ValueError, FileExistsError)):
        return 'Invalid settings or existing file: check the input and replacement option.'
    return 'Unreadable: the operation failed. Check the selected catalog or file registration.'


def summary(uid, metadata):
    start, stop = metadata.get('start', {}), metadata.get('stop')
    export = metadata.get('file_export', {})
    return {'uid': str(uid), 'scan_id': start.get('scan_id'),
            'title': start.get('title') or start.get('plan_name') or str(uid)[:8],
            'time': start.get('time'), 'plan': start.get('plan_name'),
            'status': stop.get('exit_status', 'stopped') if stop else 'No stop recorded',
            'committed_points': export.get('committed_points'),
            'export_status': export.get('status', 'Not recorded'),
            'file_uri': export.get('file_uri'),
            'incremental': export.get('incremental', False),
            'expected_points': start.get('num_points'),
            'data_type': start.get('data_type'),
            'last_point_time': metadata.get('last_point_time')}


class ManagerService:
    def __init__(self, uri, catalogs):
        auth.server_id(uri)
        self.uri = uri
        self.catalogs = dict(catalogs)
        self.reader = self.client = self.manager = None

    def close_manager(self):
        context, self.manager = self.manager, None
        if context:
            try:
                context.logout()
            except Exception:
                pass
            finally:
                context.close()

    def close(self):
        self.close_manager()
        if self.reader:
            self.reader.close()
            self.reader = self.client = None

    def connect(self, username=None, password=None):
        if self.reader:
            self.reader.close()
        self.reader = self.client = None
        if username is None:
            self.reader, self.client = auth.connect_reader(self.uri)
        else:
            context = auth.password_context(self.uri, username, password)
            try:
                # A monitoring login must not silently become an administrator.
                info = context.whoami()
                scopes = set().union(*(set(role['scopes']) for role in info['roles']))
                if scopes != auth.READER_SCOPES:
                    raise PermissionError('Use the read-only reader account for monitoring.')
                self.client = from_context(context, remember_me=False)
                self.reader = context
            except Exception:
                context.close()
                raise
        return self.refresh()

    def refresh(self):
        if self.client is None:
            raise PermissionError('Reader connection is required.')
        before = time.monotonic()
        info = self.reader.server_info
        result = {'uri': self.uri, 'client_version': version('tiled'),
                  'server_version': str(getattr(info, 'library_version', 'Unavailable')),
                  'checked_at': datetime.now(timezone.utc).isoformat(), 'catalogs': {}}
        identity = self.reader.whoami()
        result['identity'] = ', '.join(item['id'] for item in identity.get('identities', [])) or 'Service reader'
        result['reader_key'] = self.reader.which_api_key() if self.reader.api_key else None
        for label, path in self.catalogs.items():
            try:
                catalog = self.client[path]
                count = len(catalog)
                latest = list(catalog.sort(('start.time', -1)).items()[:1])
                result['catalogs'][label] = {'state': 'Readable', 'count': count,
                    'latest': summary(latest[0][0], dict(latest[0][1].metadata)) if latest else None}
            except Exception as exc:
                result['catalogs'][label] = {'state': failure(exc)}
        result['response_ms'] = round((time.monotonic() - before) * 1000)
        return result

    def runs(self, label):
        catalog = self.client[self.catalogs[label]]
        return [summary(uid, dict(run.metadata))
                for uid, run in catalog.sort(('start.time', -1)).items()[:100]]

    def check_run(self, label, uid):
        """Read one scalar row or at most 16 array values, never a full spectrum."""
        run = self.client[self.catalogs[label]][uid]
        metadata = dict(run.metadata)
        primary = run['primary']
        base = getattr(primary, 'base', primary)
        children = list(base.items())
        children.sort(key=lambda item: str(item[0]) == 'internal')
        for name, child in children:
            if str(name) == 'internal':
                structure = child.structure()
                if not structure.columns:
                    continue
                # Table columns are exposed as arrays by Tiled. Slice the
                # array endpoint; table/full does not support row slicing.
                column = structure.columns[0]
                values = child[column].read(slice(0, 1))
                if not values.size:
                    continue
                return {'state': 'Readable', 'kind': 'Scalar', 'field': column,
                        'values_read': int(values.size)}
            if getattr(child, 'structure_family', None) == 'array':
                shape = child.structure().shape
                if not shape or any(n == 0 for n in shape):
                    continue
                selection = (0,) * (len(shape) - 1) + (slice(0, min(16, shape[-1])),)
                values = child.read(selection)
                return {'state': 'Readable', 'kind': 'Array', 'field': str(name),
                        'values_read': int(values.size),
                        'committed_points': metadata.get('file_export', {}).get('committed_points')}
        return {'state': 'Unavailable: no saved primary values to sample.'}

    def login_manager(self, password, *, policy_path=None):
        if not auth.management_allowed(self.uri, policy_path):
            raise PermissionError('Management is available only on the configured control PC.')
        policy = auth.read_private_config(policy_path or auth.POLICY_PATH)
        context = auth.password_context(self.uri, policy['username'], password)
        try:
            info = context.whoami()
            scopes = set().union(*(set(role['scopes']) for role in info['roles']))
            if scopes != auth.MANAGER_SCOPES:
                raise PermissionError('The dedicated reader-key-manager role is required.')
            self.close_manager()
            self.manager = context
            return self.keys()
        except Exception:
            context.close()
            raise

    def keys(self):
        if self.manager is None:
            raise PermissionError('Manager login required.')
        # Project only display-safe metadata; no tokens/session identifiers.
        keys = []
        for key in self.manager.whoami().get('api_keys', []):
            row = {name: key.get(name) for name in ('first_eight', 'note', 'scopes', 'expiration_time',
                                                   'latest_activity')}
            # Tiled 0.2.17 does not expose creation time in APIKey. Preserve it
            # with the label in the native note for keys issued by this tool.
            if (row['note'] or '').startswith('reader-pc:'):
                try:
                    note = json.loads(row['note'][10:])
                    row['note'], row['time_created'] = note['computer'], note['created']
                except (ValueError, KeyError, TypeError):
                    pass
            keys.append(row)
        return keys

    def issue(self, label, days=365):
        if self.manager is None:
            raise PermissionError('Manager login required.')
        label = label.strip()
        if not label or len(label) > 120 or any(ord(c) < 32 for c in label):
            raise ValueError('Enter a computer label of 1–120 printable characters.')
        if not 1 <= days <= 3650:
            raise ValueError('Expiration must be between 1 and 3650 days.')
        # Deliberately bypass automatic retry of a non-idempotent POST.
        # A lost reply leaves a listable key; its secret cannot be recovered.
        created = datetime.now(timezone.utc).isoformat()
        note = 'reader-pc:' + json.dumps({'computer': label, 'created': created},
                                       separators=(',', ':'), ensure_ascii=False)
        if len(note) > 255:
            raise ValueError('Computer label is too long after encoding; shorten it before issuing a key.')
        response = self.manager.http_client.post(
            self.manager.server_info.authentication.links.apikey,
            json={'scopes': sorted(auth.READER_SCOPES), 'note': note, 'expires_in': days * 86400})
        from tiled.client.utils import handle_error
        result = handle_error(response).json()
        result['note'], result['time_created'] = label, created
        return result

    def revoke(self, prefix):
        keys = self.keys()
        if len(prefix) != 8 or not any(key['first_eight'] == prefix for key in keys):
            raise ValueError('Select an existing key belonging to this manager.')
        self.manager.revoke_api_key(prefix)
        return self.keys()
