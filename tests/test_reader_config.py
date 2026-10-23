"""Private reader configuration without real keys or production requests."""

import contextlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from diffractometer_controls.analysis import configure_tiled_reader as setup
from diffractometer_controls.analysis import tiled_auth
from tiled_manager import reader_config


class ReaderConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "settings" / "tiled-reader.json"
        self.config = {"hostname": socket.gethostname(), "uri": "http://server:8000", "api_key": "test-reader-key"}

    @unittest.skipUnless(os.name == "posix", "POSIX permissions")
    def test_private_roundtrip_and_atomic_replacement(self):
        reader_config.write_private_config(self.path, self.config)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(reader_config.read_private_config(self.path), self.config)
        with self.assertRaises(FileExistsError):
            reader_config.write_private_config(self.path, {**self.config, "uri": "http://other"})
        self.assertEqual(reader_config.read_private_config(self.path), self.config)
        updated = {**self.config, "api_key": "replacement-test-key"}
        reader_config.write_private_config(self.path, updated, replace=True)
        self.assertEqual(reader_config.read_private_config(self.path), updated)
        self.assertEqual(list(self.path.parent.glob(".tiled-reader-*")), [])

    def test_windows_protects_server_identity_and_key_before_writing(self):
        sealed = b"opaque protected bytes"
        def protect(data, *, decrypt=False):
            if decrypt:
                self.assertEqual(data, sealed)
                return json.dumps(self.config).encode()
            self.assertEqual(json.loads(data), self.config)
            return sealed
        with patch.object(reader_config, "_platform", return_value="nt"), \
             patch.object(reader_config, "_windows_protect", side_effect=protect):
            reader_config.write_private_config(self.path, self.config)
            raw = self.path.read_text()
            self.assertNotIn(self.config["api_key"], raw)
            self.assertNotIn(self.config["uri"], raw)
            self.assertEqual(reader_config.read_private_config(self.path), self.config)
            self.path.write_text(json.dumps(self.config))
            self.assertIsNone(reader_config.read_private_config(self.path))

    def test_windows_unlock_failure_disables_autologin_without_revealing_key(self):
        self.path.parent.mkdir()
        self.path.write_text('{"dpapi":"Ynl0ZXM="}')
        with patch.object(reader_config, "_platform", return_value="nt"), \
             patch.object(reader_config, "_windows_protect", side_effect=OSError("wrong user")):
            self.assertIsNone(reader_config.read_private_config(self.path))

    @unittest.skipUnless(os.name == "nt", "Native Windows DPAPI requires Windows")
    def test_native_windows_dpapi_roundtrip(self):
        reader_config.write_private_config(self.path, self.config)
        self.assertNotIn(self.config["api_key"], self.path.read_text())
        self.assertEqual(reader_config.read_private_config(self.path), self.config)

    def test_shared_key_installs_as_a_reader_and_closes_validation_session(self):
        context = Mock()
        context.which_api_key.return_value = {"scopes": ["read:metadata", "read:data"]}
        with patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])):
            setup.install_reader_key(self.config["uri"], self.config["api_key"], path=self.path)
        context.close.assert_called_once()
        self.assertEqual(reader_config.read_private_config(self.path), self.config)
        with patch.object(tiled_auth, "LOCAL_READER_CONFIG", self.path), \
             patch.object(tiled_auth.Context, "from_any_uri", return_value=(Mock(), [])) as connect, \
             patch.object(tiled_auth, "from_context", return_value="reader"):
            self.assertEqual(tiled_auth.build_tiled_client_from_local_reader(self.config["uri"]), "reader")
            connect.assert_called_once_with(self.config["uri"], api_key=self.config["api_key"])

    def test_writer_or_inherited_key_is_not_installed(self):
        for scopes in (["inherit"], ["read:metadata"], ["read:metadata", "read:data", "write:data"]):
            context = Mock()
            context.which_api_key.return_value = {"scopes": scopes}
            with self.subTest(scopes=scopes), patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])):
                with self.assertRaisesRegex(ValueError, "restricted"):
                    setup.install_reader_key(self.config["uri"], self.config["api_key"], path=self.path)
                context.close.assert_called_once()
                self.assertFalse(self.path.exists())

    def test_invalid_uri_or_unverified_key_does_not_write_configuration(self):
        with self.assertRaises(ValueError):
            setup.install_reader_key("http://server/?api_key=not-a-real-key", "test", path=self.path)
        context = Mock()
        context.which_api_key.side_effect = RuntimeError("revoked")
        with patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])):
            with self.assertRaises(RuntimeError):
                setup.install_reader_key(self.config["uri"], self.config["api_key"], path=self.path)
        context.close.assert_called_once()
        self.assertFalse(self.path.exists())

    def test_setup_uses_hidden_input_and_does_not_print_the_key(self):
        output = io.StringIO()
        with patch.object(setup.getpass, "getpass", return_value=self.config["api_key"]), \
             patch.object(setup, "install_reader_key", return_value=self.path) as install, \
             contextlib.redirect_stdout(output):
            setup.main(["--uri", self.config["uri"]])
        install.assert_called_once_with(self.config["uri"], self.config["api_key"], replace=False)
        self.assertNotIn(self.config["api_key"], output.getvalue())

    def test_reader_scope_validation_failure_keeps_existing_configuration(self):
        self.path.parent.mkdir()
        self.path.write_text("existing settings")
        context = Mock()
        context.which_api_key.return_value = {"scopes": ["inherit"]}
        with patch.object(tiled_auth.Context, "from_any_uri", return_value=(context, [])):
            with self.assertRaises(ValueError):
                setup.install_reader_key(self.config["uri"], self.config["api_key"], path=self.path, replace=True)
        self.assertEqual(self.path.read_text(), "existing settings")
