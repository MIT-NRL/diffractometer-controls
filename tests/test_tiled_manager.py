"""Isolated auth DB and real Tiled HTTP authorization; no production access."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from tiled.authn_database.core import initialize_database, create_user, create_service
from tiled.authn_database.orm import APIKey, Role
from tiled.catalog import from_uri
from tiled.client import from_context
from tiled.client.context import Context
from tiled.config import Authentication, Database
from tiled.server.app import build_app

from server.services import provision_tiled_manager as provisioning
from tiled_manager import auth
from tiled_manager.service import ManagerService, failure


class ManagerAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.db_uri = f'sqlite+aiosqlite:///{cls.root}/auth.db'
        cls.secrets = {}
        async def seed():
            engine = create_async_engine(cls.db_uri)
            await initialize_database(engine)
            async with AsyncSession(engine, expire_on_commit=False) as db:
                await create_user(db, 'local', provisioning.READER_USER)
                writer_role = Role(name='writer', scopes=['read:metadata', 'read:data', 'write:data', 'write:metadata', 'create:node'])
                db.add(writer_role); await db.commit()
                writer = await create_service(db, 'writer')
                await provisioning.apply_roles(db)
                await db.commit()
                for label, principal, scopes in (
                    ('writer', writer, ['inherit']),
                    ('reader', await provisioning.principal_for(db, provisioning.READER_USER), sorted(auth.READER_SCOPES)),
                    ('manager', await provisioning.principal_for(db, provisioning.MANAGER_USER), ['inherit'])):
                    secret = secrets.token_bytes(36)
                    cls.secrets[label] = secret.hex()
                    db.add(APIKey(principal_id=principal.id, scopes=scopes, first_eight=secret.hex()[:8],
                                  hashed_secret=hashlib.sha256(secret).digest(), note=f'test-{label}'))
                await db.commit()
            await engine.dispose()
        asyncio.run(seed())
        from tiled.alembic_utils import stamp_head
        from tiled.authn_database.alembic_constants import ALEMBIC_INI_TEMPLATE_PATH, ALEMBIC_DIR
        stamp_head(ALEMBIC_INI_TEMPLATE_PATH, ALEMBIC_DIR, cls.db_uri)
        tree = from_uri(cls.root / 'catalog.db', init_if_not_exists=True,
                        writable_storage=[f'duckdb:///{cls.root}/tables.db', cls.root / 'arrays'])
        cls.app = build_app(tree, authentication=Authentication(providers=[{
            'provider': 'local', 'authenticator': 'tiled.authenticators:DictionaryAuthenticator',
            'args': {'users_to_passwords': {provisioning.MANAGER_USER: 'test-only-password'}}}]),
            server_settings={'database': Database(uri=cls.db_uri, init_if_not_exists=False)})
        cls.reader = Context.from_app(cls.app, api_key=cls.secrets['reader'])
        cls.manager = Context.from_app(cls.app, api_key=cls.secrets['manager'])
        cls.writer = Context.from_app(cls.app, api_key=cls.secrets['writer'])
        cls.catalog = from_context(cls.writer)

    @classmethod
    def tearDownClass(cls):
        for context in (cls.reader, cls.manager, cls.writer):
            context.close()
        cls.directory.cleanup()

    def test_server_enforces_role_boundaries(self):
        for context in (self.reader, self.manager):
            with self.subTest(context='reader' if context is self.reader else 'manager'):
                client = from_context(context)
                with self.assertRaises(Exception) as error:
                    client.create_container(key='forbidden')
                self.assertIn(error.exception.response.status_code, (401, 403))
                with self.assertRaises(Exception) as error:
                    client.write_array(np.array([1]), key='forbidden-array')
                self.assertIn(error.exception.response.status_code, (401, 403))
        endpoint = self.reader.server_info.authentication.links.apikey
        self.assertIn(self.reader.http_client.post(endpoint, json={'scopes': ['read:data']}).status_code, (401, 403))
        # Include CSRF cookie, so the permission check itself is exercised.
        response = self.reader.http_client.delete(endpoint, params={'first_eight': self.secrets['reader'][:8]},
            headers={'x-csrf': self.reader.http_client.cookies['tiled_csrf']})
        self.assertIn(response.status_code, (401, 403))
        with self.assertRaises(Exception) as error:
            self.manager.admin.create_api_key(self.reader.whoami()['uuid'], scopes=['read:data'])
        self.assertIn(error.exception.response.status_code, (401, 403))
        with self.assertRaises(Exception) as error:
            self.manager.create_api_key(scopes=['write:data'])
        self.assertIn(error.exception.response.status_code, (401, 403))
        created = self.catalog.write_array(np.array([1, 2]), key='writer-check')
        np.testing.assert_array_equal(from_context(self.reader)['writer-check'].read(), [1, 2])
        for context in (self.reader, self.manager):
            with self.assertRaises(Exception) as error:
                from_context(context)['writer-check'].update_metadata({'forbidden': True})
            self.assertIn(error.exception.response.status_code, (401, 403))
        with self.assertRaises(Exception) as error:
            self.manager.revoke_api_key(self.secrets['reader'][:8])
        self.assertEqual(error.exception.response.status_code, 404)
        self.assertTrue(self.reader.which_api_key())

    def test_expired_key_is_rejected_by_server(self):
        issued = self.manager.create_api_key(scopes=sorted(auth.READER_SCOPES), expires_in=-1,
                                             note='expired-disposable-test')
        context = Context.from_app(self.app, api_key=issued['secret'])
        try:
            with self.assertRaises(Exception) as error:
                context.which_api_key()
            self.assertEqual(error.exception.response.status_code, 401)
        finally:
            context.close()

    def test_issue_list_rotate_revoke_and_actual_independent_access(self):
        service = ManagerService('http://test', {})
        service.manager = self.manager
        first = service.issue('test-laptop')
        second = service.issue('test-laptop')
        self.assertEqual(set(first['scopes']), auth.READER_SCOPES)
        self.assertIsNotNone(first['expiration_time'])
        self.assertEqual(sum(k['note'] == 'test-laptop' for k in service.keys()), 2)
        key_context = Context.from_app(self.app, api_key=first['secret'])
        other_context = Context.from_app(self.app, api_key=second['secret'])
        try:
            self.assertTrue(key_context.whoami())
            service.revoke(first['first_eight'])
            with self.assertRaises(Exception) as error:
                key_context.which_api_key()
            self.assertEqual(error.exception.response.status_code, 401)
            self.assertTrue(other_context.whoami())
        finally:
            service.revoke(second['first_eight'])
            key_context.close(); other_context.close()

    def test_password_session_is_memory_only_and_manager_policy_is_checked(self):
        policy = self.root / 'policy.json'
        auth.write_private_config(policy, {'hostname': socket.gethostname(), 'uri': 'http://test',
                                          'username': provisioning.MANAGER_USER})
        service = ManagerService('http://test', {})
        def context(*args, **kwargs):
            return Context.from_app(self.app), []
        with patch.object(auth, 'new_context', side_effect=context):
            service.login_manager('test-only-password', policy_path=policy)
            self.assertIsNone(service.manager.http_client.auth.token_directory)
            service.close_manager()
        with self.assertRaises(PermissionError):
            service.login_manager('test-only-password', policy_path=self.root / 'missing')

    def test_scalar_and_array_checks_are_bounded_on_real_tiled(self):
        import pandas as pd
        production = self.catalog.create_container(key='production')
        scalar = production.create_container(key='scalar', metadata={'start': {'time': 1}})
        primary = scalar.create_container(key='primary')
        primary.write_dataframe(pd.DataFrame({'signal': np.arange(100)}), key='internal')
        array = production.create_container(key='array', metadata={'start': {'time': 2}})
        array.create_container(key='primary').write_array(np.ones((5, 200)), key='counts')
        service = ManagerService('http://test', {'Production': 'production'})
        service.reader, service.client = self.reader, from_context(self.reader)
        self.assertEqual(service.check_run('Production', 'scalar')['values_read'], 1)
        self.assertEqual(service.check_run('Production', 'array')['values_read'], 16)
        self.assertEqual(len(service.runs('Production')), 2)
        self.assertEqual(service.refresh()['catalogs']['Production']['count'], 2)


class ManagerConfigTests(unittest.TestCase):
    def test_host_policy_setup_validation_and_export_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / 'policy.json'
            auth.write_private_config(policy, {'hostname': socket.gethostname(), 'uri': 'http://server'})
            self.assertTrue(auth.management_allowed('http://server/', policy))
            self.assertFalse(auth.management_allowed('http://other', policy))
            with patch.object(auth.socket, 'gethostname', return_value='remote'):
                self.assertFalse(auth.management_allowed('http://server', policy))
            setup = root / 'setup.json'
            auth.export_setup(setup, 'http://server', 'test-only-secret', 'laptop')
            self.assertEqual(auth.load_setup(setup)['label'], 'laptop')
            if os.name == 'posix':
                self.assertEqual(setup.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                auth.export_setup(setup, 'http://server', 'different-test-secret', 'laptop')
            setup.write_text('{"format":"wrong"}')
            with self.assertRaises(ValueError):
                auth.load_setup(setup)
            checkout = root / 'checkout'
            (checkout / '.git').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'outside Git'):
                auth.export_setup(checkout / 'secret.json', 'http://server', 'test-only-secret', 'laptop')
            self.assertFalse((checkout / 'secret.json').exists())

    def test_install_validates_scope_and_records_destination_host(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Mock()
            context.which_api_key.return_value = {'scopes': sorted(auth.READER_SCOPES)}
            path = Path(directory) / 'reader.json'
            with patch.object(auth, 'new_context', return_value=(context, [])):
                auth.install_reader('http://server', 'test-only-secret', path=path)
            self.assertEqual(auth.read_private_config(path)['hostname'], socket.gethostname())
            context.close.assert_called_once()
            context.which_api_key.return_value = {'scopes': ['inherit']}
            with patch.object(auth, 'new_context', return_value=(context, [])), self.assertRaises(ValueError):
                auth.install_reader('http://server', 'test-only-secret', path=path, replace=True)
            self.assertEqual(auth.read_private_config(path)['api_key'], 'test-only-secret')

    def test_issuance_does_not_retry_after_lost_response(self):
        service = ManagerService('http://test', {})
        service.manager = Mock()
        service.manager.http_client.post.side_effect = TimeoutError('test-only-secret')
        with self.assertRaises(TimeoutError):
            service.issue('laptop')
        self.assertEqual(service.manager.http_client.post.call_count, 1)
        self.assertNotIn('test-only-secret', failure(TimeoutError('test-only-secret')))

    def test_issue_unicode_labels_and_reject_oversized_notes_before_request(self):
        service = ManagerService('http://test', {})
        service.manager = Mock()
        service.manager.http_client.post.return_value = httpx.Response(
            201, json={'secret': 'test-only-secret', 'first_eight': '12345678'},
            request=httpx.Request('POST', 'http://test/api/v1/auth/apikey'))
        label = 'é' * 120
        self.assertEqual(service.issue(label)['note'], label)
        sent = service.manager.http_client.post.call_args.kwargs['json']
        self.assertLessEqual(len(sent['note']), 255)
        service.manager.http_client.post.reset_mock()
        with self.assertRaises(ValueError):
            service.issue('"' * 120)
        service.manager.http_client.post.assert_not_called()

    def test_provision_check_apply_repeat_and_rollback_preserve_other_roles_and_keys(self):
        async def scenario(root):
            engine = create_async_engine(f'sqlite+aiosqlite:///{root}/auth.db')
            await initialize_database(engine)
            async with AsyncSession(engine, expire_on_commit=False) as db:
                reader = await create_user(db, 'local', provisioning.READER_USER)
                writer = await create_user(db, 'local', 'unrelated-writer')
                # The targeted reader starts with the site's existing shared role.
                role = (await db.execute(select(Role).where(Role.name == 'user'))).scalar_one()
                role.scopes = sorted(auth.MANAGER_SCOPES)
                await db.commit()
                original_id = writer.id
                db.add(APIKey(principal_id=reader.id, first_eight='12345678', hashed_secret=b'x' * 32,
                              scopes=sorted(auth.READER_SCOPES)))
                await db.commit()
            profile, env = root / 'config.yml', root / 'server.env'
            profile.write_text('authentication:\n  providers:\n  - provider: local\n'
                '    authenticator: tiled.authenticators:DictionaryAuthenticator\n'
                '    args:\n      users_to_passwords:\n        4dh4-reader: "${READER_PASSWORD}"\n')
            env.write_text('READER_PASSWORD=test-only-password\nTILED_WRITER_API_KEY=test-only-writer\n')
            original = profile.read_bytes(), env.read_bytes()
            kwargs = dict(profile_path=profile, env_path=env, private_dir=root, uri='http://test')
            checked = await provisioning.provision(engine, **kwargs)
            self.assertEqual(checked['mode'], 'check')
            self.assertEqual((profile.read_bytes(), env.read_bytes()), original)
            result = await provisioning.provision(engine, apply=True, **kwargs)
            password = provisioning.environment(env)['TILED_KEY_MANAGER_PASSWORD']
            await provisioning.provision(engine, apply=True, **kwargs)
            self.assertEqual(provisioning.environment(env)['TILED_KEY_MANAGER_PASSWORD'], password)
            async with AsyncSession(engine) as db:
                reader = await provisioning.principal_for(db, provisioning.READER_USER)
                self.assertEqual({r.name for r in reader.roles}, {provisioning.READER_ROLE})
                writer = await provisioning.principal_for(db, 'unrelated-writer')
                self.assertEqual(writer.id, original_id)
                self.assertEqual({r.name for r in writer.roles}, {'user'})
                self.assertEqual(len(list((await db.execute(select(APIKey))).unique().scalars())), 1)
            await provisioning.rollback(engine, result['backup'])
            self.assertEqual((profile.read_bytes(), env.read_bytes()), original)
            await engine.dispose()
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(scenario(Path(directory)))


if __name__ == '__main__':
    unittest.main()
