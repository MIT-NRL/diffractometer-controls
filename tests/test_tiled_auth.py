import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from diffractometer_controls.analysis import tiled_auth


class TiledAuthTests(unittest.TestCase):
    def test_local_endpoint_default_handles_ssh_and_launcher_localhost(self):
        config = {"uri": "http://control-server:8000"}
        for environment in ({}, {"MITR_CONTROL_HOST": "localhost"}, {"MITR_CONTROL_HOST": "127.0.0.1"}):
            with self.subTest(environment=environment), \
                 patch.dict(tiled_auth.os.environ, environment, clear=True), \
                 patch.object(tiled_auth, "_local_reader_config", return_value=config):
                self.assertEqual(tiled_auth.get_tiled_uri(), config["uri"])

    def test_endpoint_overrides_remain_explicit(self):
        for environment, expected in [
            ({"TILED_URI": "http://custom:1234"}, "http://custom:1234"),
            ({"MITR_TILED_URI": "http://custom:5678"}, "http://custom:5678"),
            ({"MITR_CONTROL_HOST": "remote-host", "MITR_TILED_PORT": "8001"}, "http://remote-host:8001"),
        ]:
            with self.subTest(environment=environment), \
                 patch.dict(tiled_auth.os.environ, environment, clear=True), \
                 patch.object(tiled_auth, "_local_reader_config", return_value={"uri": "http://control:8000"}):
                self.assertEqual(tiled_auth.get_tiled_uri(), expected)
        with patch.dict(tiled_auth.os.environ, {}, clear=True), \
             patch.object(tiled_auth, "_local_reader_config", return_value={"uri": "http://control:8000"}):
            self.assertEqual(tiled_auth.get_tiled_uri(host="explicit-host"), "http://explicit-host:8000")

    def test_remote_without_local_configuration_keeps_normal_endpoint_default(self):
        with patch.dict(tiled_auth.os.environ, {}, clear=True), \
             patch.object(tiled_auth, "_local_reader_config", return_value=None):
            self.assertEqual(tiled_auth.get_tiled_uri(), "http://localhost:8000")

    def test_unreachable_local_server_reports_connection_error_without_login(self):
        with patch.object(tiled_auth, "build_tiled_client_from_local_reader", side_effect=tiled_auth.httpx.ConnectError("refused")), \
             patch.object(tiled_auth, "build_tiled_client_from_cached_tokens") as cached, \
             patch.object(tiled_auth, "TiledLoginDialog") as dialog, \
             patch.object(tiled_auth.QtWidgets.QMessageBox, "critical") as error:
            self.assertIsNone(tiled_auth.ensure_tiled_login(uri="http://server:8000"))
        cached.assert_not_called()
        dialog.assert_not_called()
        self.assertIn("Cannot reach Tiled at http://server:8000", error.call_args.args[2])

    def test_unreachable_remote_server_does_not_prompt_for_password(self):
        error = RuntimeError("No cached Tiled login")
        error.__cause__ = tiled_auth.httpx.ConnectError("refused")
        with patch.object(tiled_auth, "build_tiled_client_from_local_reader", return_value=None), \
             patch.object(tiled_auth, "build_tiled_client_from_cached_tokens", side_effect=error), \
             patch.object(tiled_auth, "TiledLoginDialog") as dialog, \
             patch.object(tiled_auth.QtWidgets.QMessageBox, "critical") as critical:
            self.assertIsNone(tiled_auth.ensure_tiled_login(uri="http://server:8000"))
        dialog.assert_not_called()
        critical.assert_called_once()

    def local_config(self, directory):
        path = Path(directory) / "tiled-reader.json"
        path.write_text(json.dumps({"hostname": "control-pc", "uri": "http://server:8000",
                                    "api_key": "reader-key"}))
        path.chmod(0o600)
        return path

    def test_local_reader_uses_its_explicit_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.local_config(directory)
            context = Mock()
            with patch.object(tiled_auth, "LOCAL_READER_CONFIG", path), \
                 patch.object(tiled_auth.socket, "gethostname", return_value="control-pc"), \
                 patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])) as create, \
                 patch.object(tiled_auth, "from_context", return_value="reader"):
                self.assertEqual(tiled_auth.build_tiled_client_from_local_reader("http://server:8000/"), "reader")
            create.assert_called_once_with("http://server:8000/", api_key="reader-key")
            context.use_cached_tokens.assert_not_called()

    def test_remote_computer_or_changed_server_does_not_use_local_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.local_config(directory)
            for hostname, uri in [("remote-pc", "http://server:8000"),
                                  ("control-pc", "http://other-server:8000"),
                                  ("control-pc", "http://server:9000"),
                                  ("control-pc", "https://server:8000")]:
                with self.subTest(hostname=hostname, uri=uri), \
                     patch.object(tiled_auth, "LOCAL_READER_CONFIG", path), \
                     patch.object(tiled_auth.socket, "gethostname", return_value=hostname), \
                     patch.object(tiled_auth.Context, "from_any_uri") as create:
                    self.assertIsNone(tiled_auth.build_tiled_client_from_local_reader(uri))
                    create.assert_not_called()

    def test_missing_or_publicly_readable_key_does_not_enable_autologin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.local_config(directory)
            path.chmod(0o644)
            with patch.object(tiled_auth, "LOCAL_READER_CONFIG", path), \
                 patch.object(tiled_auth.Context, "from_any_uri") as create:
                self.assertIsNone(tiled_auth.build_tiled_client_from_local_reader("http://server:8000"))
                path.unlink()
                self.assertIsNone(tiled_auth.build_tiled_client_from_local_reader("http://server:8000"))
                create.assert_not_called()

    def test_local_reader_takes_precedence_over_cached_credentials(self):
        reader = object()
        with patch.object(tiled_auth, "build_tiled_client_from_local_reader", return_value=reader), \
             patch.object(tiled_auth, "build_tiled_client_from_cached_tokens") as cached, \
             patch.object(tiled_auth, "TiledLoginDialog") as dialog:
            self.assertIs(tiled_auth.ensure_tiled_login(uri="http://server"), reader)
        cached.assert_not_called()
        dialog.assert_not_called()

    def test_remote_login_keeps_remembered_session(self):
        reader = object()
        with patch.object(tiled_auth, "build_tiled_client_from_local_reader", return_value=None), \
             patch.object(tiled_auth, "build_tiled_client_from_cached_tokens", return_value=reader), \
             patch.object(tiled_auth, "TiledLoginDialog") as dialog:
            self.assertIs(tiled_auth.ensure_tiled_login(uri="http://server"), reader)
        dialog.assert_not_called()

    def test_remote_or_failed_local_login_falls_back_to_password_dialog(self):
        for result in (None, RuntimeError("key revoked")):
            with self.subTest(result=type(result).__name__), \
                 patch.object(tiled_auth, "build_tiled_client_from_local_reader") as local, \
                 patch.object(tiled_auth, "build_tiled_client_from_cached_tokens", side_effect=RuntimeError("no tokens")), \
                 patch.object(tiled_auth, "TiledLoginDialog") as dialog:
                if isinstance(result, Exception):
                    local.side_effect = result
                else:
                    local.return_value = result
                dialog.return_value.exec.return_value = tiled_auth.QtWidgets.QDialog.Rejected
                self.assertIsNone(tiled_auth.ensure_tiled_login(uri="http://server"))
                dialog.assert_called_once_with(parent=None, uri="http://server")

    def test_password_login_bypasses_environment_api_key_and_keeps_memory_session(self):
        context = Mock(api_uri="http://server/api/v1/")
        client = object()
        with patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])) as create, \
             patch.object(tiled_auth, "password_grant", return_value={"access_token": "token"}), \
             patch.object(tiled_auth, "from_context", return_value=client):
            result = tiled_auth.login_tiled("reader", "password", uri="http://server", remember_me=False)
        create.assert_called_once_with("http://server", api_key="")
        self.assertIsNone(context.api_key)
        context.configure_auth.assert_called_once_with({"access_token": "token"}, remember_me=False)
        context.close.assert_not_called()
        self.assertIs(result, client)

    def test_failed_login_closes_context(self):
        context = Mock(api_uri="http://server/api/v1/")
        with patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])), \
             patch.object(tiled_auth, "password_grant", side_effect=RuntimeError("bad login")):
            with self.assertRaisesRegex(RuntimeError, "bad login"):
                tiled_auth.login_tiled("reader", "wrong", uri="http://server")
        context.close.assert_called_once()

    def test_missing_cached_tokens_closes_context(self):
        context = Mock()
        context.use_cached_tokens.side_effect = RuntimeError("missing")
        with patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])):
            with self.assertRaisesRegex(RuntimeError, "No cached"):
                tiled_auth.build_tiled_client_from_cached_tokens(uri="http://server")
        context.close.assert_called_once()
