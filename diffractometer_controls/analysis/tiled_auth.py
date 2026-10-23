import os
from pathlib import Path
import socket
from urllib.parse import urlsplit

import httpx
from qtpy import QtWidgets

from .reader_config import read_private_config

try:
    from tiled.client.context import Context, password_grant
    from tiled.client.constructors import from_context
except Exception:
    Context = None
    password_grant = None
    from_context = None


LOCAL_READER_CONFIG = Path.home() / ".config" / "diffractometer-controls" / "tiled-reader.json"


def _local_reader_config():
    """Read only the private configuration belonging to this user and host."""
    path = LOCAL_READER_CONFIG
    try:
        config = read_private_config(path)
        if not isinstance(config, dict) or config.get("hostname") != socket.gethostname():
            return None
        _reader_server_id(config.get("uri", ""))
        return config
    except (OSError, ValueError):
        return None


def get_tiled_uri(host=None):
    uri = os.environ.get("TILED_URI") or os.environ.get("MITR_TILED_URI")
    if uri:
        return uri

    # IDE/direct launches may not load control.env, and the QueueServer
    # launcher can use localhost even though Tiled binds to the LAN address.
    # Use the installed reader endpoint for local defaults, while preserving
    # explicit remote hosts and Tiled URI overrides.
    default_host = os.environ.get("MITR_CONTROL_HOST") or os.environ.get("MITR_CONTROL_IP")
    if host is None and str(default_host or "localhost").lower() in {"localhost", "127.0.0.1", "::1"}:
        config = _local_reader_config()
        if config and config.get("uri"):
            return config["uri"]

    host = host or os.environ.get("MITR_CONTROL_HOST") or os.environ.get("MITR_CONTROL_IP") or "localhost"
    port = os.environ.get("MITR_TILED_PORT", "8000")
    return f"http://{host}:{port}"


TILED_AUTH_PROVIDER = "local"


def _reader_server_id(uri):
    parsed = urlsplit(str(uri))
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Expected a Tiled server URL without embedded credentials.")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname, port, parsed.path.rstrip("/")


TILED_URI = get_tiled_uri()


def build_tiled_client_from_local_reader(uri=None):
    """Use the private reader key installed on this computer, if any.

    Bind automatic access to the recorded computer and server. A copied client
    configuration or a different requested server still uses normal login.
    Multiple computers may intentionally be configured with the same reader key.
    """
    config = _local_reader_config()
    if config is None:
        return None
    uri = uri or get_tiled_uri()
    if _reader_server_id(config.get("uri", "")) != _reader_server_id(uri):
        return None
    api_key = config.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        return None
    if Context is None or from_context is None:
        raise RuntimeError("Tiled authentication helpers are unavailable in this environment.")
    context, parts = Context.from_any_uri(uri, api_key=api_key)
    try:
        return from_context(context, node_path_parts=parts, remember_me=False)
    except Exception:
        context.close()
        raise


def _context_authenticated(context):
    try:
        value = getattr(context, "authenticated", False)
    except Exception:
        return False
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def _password_context(uri):
    # An acquisition API key in the process environment must not override
    # an explicit viewer login. An empty key bypasses Context's env fallback;
    # clear it before configuring OAuth tokens.
    context, parts = Context.from_any_uri(uri, api_key="")
    context.api_key = None
    return context, parts


def build_tiled_client_from_cached_tokens(uri=None):
    if Context is None or from_context is None:
        raise RuntimeError("Tiled authentication helpers are unavailable in this environment.")

    uri = uri or get_tiled_uri()
    context, node_path_parts = _password_context(uri)
    try:
        context.use_cached_tokens()
        if not _context_authenticated(context):
            raise RuntimeError("Cached Tiled login is not authenticated.")
        return from_context(context, node_path_parts=node_path_parts, remember_me=True)
    except Exception as exc:
        context.close()
        raise RuntimeError("No cached Tiled login is available.") from exc


def login_tiled(username, password, uri=None, provider=TILED_AUTH_PROVIDER, *, remember_me=True):
    if Context is None or password_grant is None or from_context is None:
        raise RuntimeError("Tiled authentication helpers are unavailable in this environment.")

    uri = uri or get_tiled_uri()
    context, node_path_parts = _password_context(uri)
    try:
        tokens = password_grant(
            context.http_client,
            f"{context.api_uri}auth/provider/{provider}/token",
            provider,
            str(username or ""),
            str(password or ""),
        )
        context.configure_auth(tokens, remember_me=bool(remember_me))
        return from_context(context, node_path_parts=node_path_parts, remember_me=bool(remember_me))
    except Exception:
        context.close()
        raise


class TiledLoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, uri=None):
        super().__init__(parent)
        self.setWindowTitle("Tiled Login")
        self.setModal(True)

        self.server_edit = QtWidgets.QLineEdit(uri or get_tiled_uri(), self)
        self.user_edit = QtWidgets.QLineEdit(self)
        self.user_edit.setText(os.environ.get("MITR_TILED_USERNAME", "4dh4-reader"))
        self.pass_edit = QtWidgets.QLineEdit(self)
        self.pass_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.remember_box = QtWidgets.QCheckBox("Remember login", self)
        self.remember_box.setChecked(True)

        form = QtWidgets.QFormLayout()
        form.addRow("Server URL", self.server_edit)
        form.addRow("Username", self.user_edit)
        form.addRow("Password", self.pass_edit)
        form.addRow("", self.remember_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

        self.pass_edit.setFocus()

    def credentials(self):
        return (
            self.user_edit.text().strip(),
            self.pass_edit.text(),
            bool(self.remember_box.isChecked()),
        )


def ensure_tiled_login(parent=None, *, uri=None):
    uri = uri or get_tiled_uri()
    # Prefer the explicitly installed read-only identity to any cached admin
    # session. Computers without a local reader configuration retain normal login.
    try:
        client = build_tiled_client_from_local_reader(uri=uri)
        if client is not None:
            return client
    except httpx.TransportError:
        QtWidgets.QMessageBox.critical(
            parent, "Tiled connection failed",
            f"Cannot reach Tiled at {uri}. Check the server address and Tiled service.",
        )
        return None
    except Exception:
        pass
    try:
        return build_tiled_client_from_cached_tokens(uri=uri)
    except Exception as exc:
        cause = exc.__cause__ or exc
        if isinstance(cause, httpx.TransportError):
            QtWidgets.QMessageBox.critical(
                parent, "Tiled connection failed",
                f"Cannot reach Tiled at {uri}. Check the server address and Tiled service.",
            )
            return None

    dialog = TiledLoginDialog(parent=parent, uri=uri)
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None

    username, password, remember_me = dialog.credentials()
    uri = dialog.server_edit.text().strip()
    if not username or not password:
        QtWidgets.QMessageBox.warning(parent, "Tiled Login", "Username and password are required.")
        return None

    try:
        return login_tiled(
            username=username,
            password=password,
            uri=uri,
            remember_me=remember_me,
        )
    except Exception as exc:
        QtWidgets.QMessageBox.critical(parent, "Tiled Login Failed", str(exc))
        return None
