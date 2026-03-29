from qtpy import QtWidgets

try:
    from tiled.client.context import Context, password_grant
    from tiled.client.constructors import from_context
except Exception:
    Context = None
    password_grant = None
    from_context = None


TILED_URI = "http://localhost:8000"
TILED_AUTH_PROVIDER = "local"


def _context_authenticated(context):
    try:
        value = getattr(context, "authenticated", False)
    except Exception:
        return False
    try:
        return bool(value() if callable(value) else value)
    except Exception:
        return False


def build_tiled_client_from_cached_tokens(uri=TILED_URI):
    if Context is None or from_context is None:
        raise RuntimeError("Tiled authentication helpers are unavailable in this environment.")

    context, node_path_parts = Context.from_any_uri(uri)
    try:
        context.use_cached_tokens()
    except Exception as exc:
        raise RuntimeError("No cached Tiled login is available.") from exc

    if not _context_authenticated(context):
        raise RuntimeError("Cached Tiled login is not authenticated.")

    return from_context(context, node_path_parts=node_path_parts, remember_me=True)


def login_tiled(username, password, uri=TILED_URI, provider=TILED_AUTH_PROVIDER, *, remember_me=True):
    if Context is None or password_grant is None or from_context is None:
        raise RuntimeError("Tiled authentication helpers are unavailable in this environment.")

    context, node_path_parts = Context.from_any_uri(uri)
    tokens = password_grant(
        context.http_client,
        f"{context.api_uri}auth/provider/{provider}/token",
        provider,
        str(username or ""),
        str(password or ""),
    )
    context.configure_auth(tokens, remember_me=bool(remember_me))
    return from_context(context, node_path_parts=node_path_parts, remember_me=bool(remember_me))


class TiledLoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tiled Login")
        self.setModal(True)

        self.user_edit = QtWidgets.QLineEdit(self)
        self.pass_edit = QtWidgets.QLineEdit(self)
        self.pass_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.remember_box = QtWidgets.QCheckBox("Remember login", self)
        self.remember_box.setChecked(True)

        form = QtWidgets.QFormLayout()
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

        self.user_edit.setFocus()

    def credentials(self):
        return (
            self.user_edit.text().strip(),
            self.pass_edit.text(),
            bool(self.remember_box.isChecked()),
        )


def ensure_tiled_login(parent=None, *, uri=TILED_URI):
    try:
        return build_tiled_client_from_cached_tokens(uri=uri)
    except Exception:
        pass

    dialog = TiledLoginDialog(parent=parent)
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return None

    username, password, remember_me = dialog.credentials()
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
