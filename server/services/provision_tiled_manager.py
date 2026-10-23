"""Provision own-account reader-key management without touching writer keys.

Check is the default. Apply saves private backups before changing configuration
or the authentication DB. Tiled must be restarted separately when idle.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import shlex
import socket

from sqlalchemy import select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import selectinload
import yaml

from tiled.authn_database.orm import Principal, Identity, Role
from tiled_manager.auth import READER_SCOPES, MANAGER_SCOPES, server_id
from tiled_manager.reader_config import write_private_config

READER_ROLE = 'diffraction-reader'
MANAGER_ROLE = 'reader-key-manager'
READER_USER = '4dh4-reader'
MANAGER_USER = '4dh4-key-manager'


def environment(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, sep, value = line.partition('=')
        if sep:
            parts = shlex.split(value, comments=True)
            values[key.strip()] = parts[0] if parts else ''
    return values


def database_url(values):
    return URL.create('postgresql+asyncpg', username=values['TILED_PGUSER'],
        password=values['TILED_PGPASSWORD'], host=values['TILED_PGHOST'],
        port=int(values['TILED_PGPORT']), database=values['TILED_AUTH_DB'])


async def principal_for(db, username):
    return (await db.execute(select(Principal).join(Principal.identities)
        .where(Identity.provider == 'local', Identity.id == username)
        .options(selectinload(Principal.identities)))).unique().scalar_one_or_none()


async def describe(db):
    result = {'roles': {}, 'assignments': {}}
    for name in (READER_ROLE, MANAGER_ROLE):
        role = (await db.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
        result['roles'][name] = None if role is None else {'scopes': list(role.scopes), 'description': role.description}
    for user in (READER_USER, MANAGER_USER):
        principal = await principal_for(db, user)
        result['assignments'][user] = None if principal is None else {
            'uuid': str(principal.uuid), 'roles': [role.name for role in principal.roles]}
    return result


async def apply_roles(db):
    roles = {}
    for name, scopes in ((READER_ROLE, READER_SCOPES), (MANAGER_ROLE, MANAGER_SCOPES)):
        role = (await db.execute(select(Role).where(Role.name == name))).scalar_one_or_none()
        if role is None:
            role = Role(name=name, scopes=sorted(scopes), description=f'Tiled Manager: {name}')
            db.add(role)
        else:
            role.scopes = sorted(scopes)
        roles[name] = role
    await db.flush()
    for user, role_name in ((READER_USER, READER_ROLE), (MANAGER_USER, MANAGER_ROLE)):
        principal = await principal_for(db, user)
        if principal is None:
            if user == READER_USER:
                raise ValueError('Existing reader identity is missing; refusing to provision an unknown installation.')
            principal = Principal(type='user', roles=[roles[role_name]], identities=[])
            db.add(principal)
            await db.flush()
            db.add(Identity(id=user, provider='local', principal_id=principal.id))
        else:
            # Refuse unexpected assignments instead of silently stripping them.
            if any(role.name not in {'user', role_name} for role in principal.roles):
                raise ValueError('Unexpected role assignment; inspect before applying.')
            principal.roles = [roles[role_name]]
    await db.flush()


async def provision(engine, *, profile_path, env_path, private_dir, uri, apply=False):
    server_id(uri)
    profile_path, env_path, private_dir = map(Path, (profile_path, env_path, private_dir))
    checkout = Path(__file__).resolve().parents[2]
    if any(path.resolve().is_relative_to(checkout) for path in (profile_path, env_path, private_dir)):
        raise ValueError('Server credentials, policy, and backups must be outside the checkout.')
    profile = yaml.safe_load(profile_path.read_text())
    providers = profile['authentication']['providers']
    provider = next(p for p in providers if p.get('provider') == 'local')
    if provider.get('authenticator') != 'tiled.authenticators:DictionaryAuthenticator':
        raise ValueError('This provisioning command supports the installed local dictionary provider only.')
    users = provider['args']['users_to_passwords']
    if READER_USER not in users:
        raise ValueError('Shared reader login is missing from provider configuration.')
    values = environment(env_path)
    if 'TILED_KEY_MANAGER_PASSWORD' in values and not values['TILED_KEY_MANAGER_PASSWORD']:
        raise ValueError('Existing manager password is empty; refusing to change it implicitly.')
    async with AsyncSession(engine, expire_on_commit=False) as db:
        before = await describe(db)
        if not apply:
            return {'mode': 'check', 'before': before,
                    'desired': {READER_USER: sorted(READER_SCOPES), MANAGER_USER: sorted(MANAGER_SCOPES)}}
        backup = private_dir / ('manager-backup-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        backup.mkdir(parents=True, mode=0o700)
        original = {profile_path: profile_path.read_bytes(), env_path: env_path.read_bytes()}
        generated = [private_dir / 'manager-policy.json', private_dir / 'manager-login.json']
        original.update({p: p.read_bytes() if p.exists() else None for p in generated})
        # Backups contain server credentials and must never be in the checkout.
        for index, (path, content) in enumerate(original.items()):
            if content is not None:
                target = backup / f'{index}-{path.name}'
                fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, 'wb') as stream:
                    stream.write(content)
        write_private_config(backup / 'roles.json', before)
        write_private_config(backup / 'files.json', {str(p): f'{i}-{p.name}' if v is not None else None
                                                   for i, (p, v) in enumerate(original.items())})
        password = values.get('TILED_KEY_MANAGER_PASSWORD') or secrets.token_urlsafe(30)
        existing = users.get(MANAGER_USER)
        if existing and existing != '${TILED_KEY_MANAGER_PASSWORD}':
            raise ValueError('Manager login already has a different password source; refusing to replace it.')
        users[MANAGER_USER] = '${TILED_KEY_MANAGER_PASSWORD}'
        env_text = env_path.read_text()
        if 'TILED_KEY_MANAGER_PASSWORD' not in values:
            env_text += f'\nTILED_KEY_MANAGER_PASSWORD={password}\n'
        try:
            await apply_roles(db)
            for path, content in ((profile_path, yaml.safe_dump(profile, sort_keys=False).encode()),
                                  (env_path, env_text.encode())):
                # Same-directory atomic replace, retaining private file permissions.
                import tempfile
                fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.manager-')
                try:
                    with os.fdopen(fd, 'wb') as stream:
                        stream.write(content); stream.flush(); os.fsync(stream.fileno())
                    os.replace(tmp, path)
                finally:
                    Path(tmp).unlink(missing_ok=True)
            write_private_config(generated[0], {'hostname': socket.gethostname(), 'uri': uri,
                                                'username': MANAGER_USER}, replace=True)
            write_private_config(generated[1], {'uri': uri, 'username': MANAGER_USER,
                                                'password': password}, replace=True)
            await db.commit()
        except Exception:
            await db.rollback()
            for path, content in original.items():
                if content is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(content)
                    path.chmod(0o600)
            raise
    return {'mode': 'applied', 'backup': str(backup), 'manager_login_file': str(generated[1]),
            'restart_required': True}


async def rollback(engine, backup):
    """Restore role assignments/configuration; never delete issued API keys."""
    backup = Path(backup)
    previous = json.loads((backup / 'roles.json').read_text())
    files = json.loads((backup / 'files.json').read_text())
    async with AsyncSession(engine) as db:
        for user, assignment in previous['assignments'].items():
            principal = await principal_for(db, user)
            if principal is not None:
                names = assignment['roles'] if assignment else []
                principal.roles = list((await db.execute(select(Role).where(Role.name.in_(names)))).scalars())
        for name, state in previous['roles'].items():
            if state:
                role = (await db.execute(select(Role).where(Role.name == name))).scalar_one()
                role.scopes, role.description = state['scopes'], state['description']
        await db.commit()
    for path, saved in files.items():
        path = Path(path)
        if saved is None:
            path.unlink(missing_ok=True)
        else:
            path.write_bytes((backup / saved).read_bytes())
            path.chmod(0o600)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uri', required=True)
    parser.add_argument('--profile', type=Path, default=Path.home() / '.config/tiled/profiles/config.yml')
    parser.add_argument('--env-file', type=Path, default=Path.home() / '.config/tiled/tiled-server.env')
    parser.add_argument('--private-dir', type=Path, default=Path.home() / '.config/tiled')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--apply', action='store_true')
    action.add_argument('--rollback', type=Path)
    args = parser.parse_args(argv)
    async def run():
        engine = create_async_engine(database_url(environment(args.env_file)))
        try:
            if args.rollback:
                await rollback(engine, args.rollback)
                return {'mode': 'rolled back', 'restart_required': True}
            return await provision(engine, profile_path=args.profile, env_path=args.env_file,
                private_dir=args.private_dir, uri=args.uri, apply=args.apply)
        finally:
            await engine.dispose()
    try:
        print(json.dumps(asyncio.run(run()), indent=2))
    except Exception:
        parser.exit(1, 'Provisioning failed; no secrets displayed. Inspect configuration and authentication DB access.\n')


if __name__ == '__main__':
    main()
