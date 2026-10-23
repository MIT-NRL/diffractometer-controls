"""Explicit credentials, portable setup files, and local management policy."""

import json
import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

import httpx
from tiled.client.context import Context, password_grant
from tiled.client import from_context
from tiled.client.auth import TiledAuth

from .reader_config import read_private_config, write_private_config

READER_SCOPES = frozenset({'read:metadata', 'read:data'})
MANAGER_SCOPES = READER_SCOPES | {'create:apikeys', 'revoke:apikeys'}
READER_PATH = Path.home() / '.config/diffractometer-controls/tiled-reader.json'
POLICY_PATH = Path.home() / '.config/tiled/manager-policy.json'


def server_id(uri):
    parsed = urlsplit(str(uri))
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or
            parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError('Use a server URL without credentials, query, or fragment.')
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80), parsed.path.rstrip('/')


def management_allowed(uri, policy_path=None):
    policy = read_private_config(policy_path or POLICY_PATH)
    try:
        return bool(policy and policy['hostname'] == socket.gethostname() and
                    server_id(policy['uri']) == server_id(uri))
    except (KeyError, TypeError, ValueError):
        return False


def new_context(uri, api_key=''):
    server_id(uri)
    context, parts = Context.from_any_uri(uri, api_key=api_key, timeout=httpx.Timeout(5.0))
    if not api_key:
        context.api_key = None  # Explicitly bypass any ambient acquisition key.
    return context, parts


def password_context(uri, username, password, provider='local'):
    context, _ = new_context(uri)
    try:
        tokens = password_grant(context.http_client,
                                f'{context.api_uri}auth/provider/{provider}/token',
                                provider, username, password)
        # configure_auth(remember_me=False) clears the shared on-disk token
        # cache in Tiled 0.2.17. Construct memory-only auth without touching it.
        context.http_client.auth = TiledAuth(
            context.server_info.authentication.links.refresh_session,
            context.http_client.cookies['tiled_csrf'], None, context.client_id,
            context.scopes)
        context.http_client.auth.sync_tokens(tokens)
        return context
    except Exception:
        context.close()
        raise


def connect_reader(uri, path=None):
    config = read_private_config(path or READER_PATH)
    key = ''
    if config and config.get('hostname') == socket.gethostname():
        if server_id(config.get('uri')) == server_id(uri):
            key = config.get('api_key', '')
    context, parts = new_context(uri, key)
    try:
        if not key and context.server_info.authentication.required:
            raise PermissionError('Install a reader key or sign in as a reader.')
        # No cached session or interactive prompt, especially no cached admin.
        client = from_context(context, node_path_parts=parts, remember_me=False)
        return context, client
    except Exception:
        context.close()
        raise


def install_reader(uri, key, *, path=None, replace=False):
    server_id(uri)
    if not isinstance(key, str) or not key.strip():
        raise ValueError('Enter a read-only API key.')
    destination = Path(path or READER_PATH)
    if destination.exists() and not replace:
        raise FileExistsError('Local reader settings exist; explicitly enable replacement.')
    context, _ = new_context(uri, key.strip())
    try:
        if set(context.which_api_key().get('scopes', [])) != READER_SCOPES:
            raise ValueError('The key must have exactly read:metadata and read:data permissions.')
    finally:
        context.close()
    write_private_config(destination, {'hostname': socket.gethostname(), 'uri': uri,
                                      'api_key': key.strip()}, replace=replace)
    return destination


def load_setup(path):
    if Path(path).stat().st_size > 16384:
        raise ValueError('Setup file is too large.')
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(data, dict) or data.get('format') != 'tiled-reader-setup-v1' or
            set(data) != {'format', 'uri', 'api_key', 'label'} or
            not isinstance(data['api_key'], str) or not data['api_key'].strip() or
            not isinstance(data['label'], str)):
        raise ValueError('Not a Tiled reader setup file.')
    server_id(data['uri'])
    return data


def export_setup(path, uri, key, label):
    """Portable plaintext credential, never an installed Windows DPAPI file."""
    server_id(uri)
    destination = Path(path).resolve()
    if any((parent / '.git').exists() for parent in destination.parents):
        raise ValueError('Secret setup files must be exported outside Git checkouts.')
    payload = json.dumps({'format': 'tiled-reader-setup-v1', 'uri': uri,
                          'api_key': key, 'label': label}, indent=2)
    # Exclusive creation prevents overwriting/symlink-following an existing file.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if os.name == 'nt':
            # Set an owner-only ACL before writing a portable plaintext secret.
            import subprocess
            import getpass
            result = subprocess.run(['icacls', str(path), '/inheritance:r', '/grant:r',
                                     f'{getpass.getuser()}:(F)'], capture_output=True)
            if result.returncode:
                raise OSError('Could not protect exported setup file permissions.')
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        Path(path).unlink(missing_ok=True)
        raise
    finally:
        if fd is not None:
            os.close(fd)
