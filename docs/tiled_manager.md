# Tiled Manager

Open **Control System → Tiled Manager**, or run from the checkout and its Python
environment:

```console
python -m tiled_manager
python -m tiled_manager --uri http://CONTROL_HOST:8000 --catalog Production=4dh4 --catalog Test=testdb
```

The independent window imports no EPICS, QueueServer, or instrument screens.
Other installations pass their own catalog labels/paths. It inherits the host
application's theme; standalone mode uses the desktop Qt theme. The interface
follows the proposed Overview / Catalogs & Files / Access & Keys / Diagnostics
mockup, using status text as well as colors.

## Monitoring

The installed reader key is preferred; no cached administrator session or
ambient writer key is used. **Access & keys → Reader password login** provides
an explicit alternative. Invalid/revoked keys leave the window available for
installation or password login. The existing diffraction loader retains its
normal key/cached-reader-session/password fallback.

Visible status refreshes every 10 seconds on a serialized background worker.
HTTP requests have five-second timeouts; Tiled's normal client retry policy can
extend the duration of a read operation. Operations do not overlap, and old
server responses cannot replace a new connection's status. Failed refreshes
mark previous status stale. The API, catalog queries, and actual data reads are
separate checks.

The control GUI and standalone manager suppress routine HTTPX/HTTPCore request
messages by default while retaining warnings, errors, and application failure
handling. Set `TILED_HTTP_DEBUG=1` before launching to restore HTTP request
logging for troubleshooting. Standalone mode also enables debug output; the
control GUI's logging level still controls its handler verbosity. This setting
does not suppress Qt/X11 graphics warnings.

Catalogs & Files lists the latest 100 runs, recorded run outcome, export status,
committed point counts, and file registration information. An explicit selected
run check reads at most 16 values from an array, or one value from a scalar
column, through Tiled. Table-column metadata may require Tiled to materialize
that column server-side; the client does not download the full column.

A missing stop document means **No stop recorded**, not proof that a plan is
still running. Missing timestamps and commit markers are shown as unavailable;
long exposures are not treated as writer failures. This version does not query
disk space, systemd, PostgreSQL health internals, or server logs. Session events
are local UI events, not a complete persistent server audit trail. Diagnostics
export includes only endpoint, identity, versions, timestamps, timing and
catalog check results, not credentials or arbitrary exception bodies.

## Permissions

| Identity / credential | Permissions |
|---|---|
| Shared `4dh4-reader` password account | Read metadata and data |
| Each issued per-PC API key | Exactly `read:metadata`, `read:data` |
| `4dh4-key-manager`, role `reader-key-manager` | Read metadata/data; create/revoke its own keys |
| Existing acquisition writer | Unchanged |

The manager has no data-write scopes, global key-admin scopes, or ability to
manage another identity's keys. Native Tiled allows it to issue keys carrying
its own management scopes through the API; this window deliberately issues only
read scopes. A strict server-side reader-only issuance service is deferred.

Management is enabled only on the hostname and endpoint in the private
`~/.config/tiled/manager-policy.json`. X11 display location does not matter:
SSH/X11 runs on the control PC. This is an application guard, not a server
firewall restriction. Knowing the password still permits use of the native API
from another reachable machine. HTTP traffic on the LAN is not encrypted; use
management on the control PC, with SSH/X11 for remote operation. HTTPS, tunnel
automation, firewall changes, and stronger identity-provider integration are
separate deployment work.

Manager login uses an independent memory-only session and does not modify the
reader token cache. It locks after ten minutes of user inactivity, on server
change, logout, or closure. Cached administrator credentials are never reused.

## One key per PC

1. On the control PC, open Access & Keys and sign in with the dedicated manager
   password from `~/.config/tiled/manager-login.json` (owner-only; outside git).
2. Enter the destination computer's label and issue a key. Default expiration is
   365 days; entries approaching expiration within 30 days are marked.
3. Copy the secret, or export a portable setup JSON file. Secrets are available
   only at creation; the server stores a hash, not a recoverable secret.
   Creation timestamps for this tool's keys are retained in their native Tiled
   notes because Tiled 0.2.17's API does not expose a creation timestamp.
4. Transfer the secret/setup file over SSH/SCP. On the destination PC, use
   **Install reader key on this PC** to paste or import it. Confirm the server
   and explicitly enable replacement if settings already exist.
5. Installation verifies the key's exact scopes online, records the destination
   hostname, and saves private local settings. POSIX uses owner-only files;
   Windows uses current-user DPAPI. Portable exports contain a plaintext secret
   and use restrictive permissions (owner ACL on Windows). Remove transferred
   copies after installation. Native Windows checks require Windows validation.

Export never overwrites an existing file and refuses destinations inside Git
checkouts. Its file dialog starts in the user's home directory. The secret dialog clears its owned
clipboard contents when closed; paste into the destination before closing, or
use the exported setup file. Clipboard history or third-party clipboard tools
may retain their own copies.

**Create replacement** leaves the old key valid. Install/test the new key, then
explicitly revoke the old entry by label and eight-character identifier. A
revocation affects every copy of that key. Deleting a local file does not
revoke the key. Computer labels are not hardware bindings.

If creation loses its response, it is not automatically retried. Refresh the
key list and revoke the unconfirmed entry before creating another. If export
fails, keep the creation dialog open and choose another path or copy the key.
The control PC's existing reader-account key is preserved and displayed
separately; the new manager cannot revoke keys belonging to the old identity.

## Provisioning and rollback

The new command defaults to a read-only check:

```console
python -m server.services.provision_tiled_manager --uri http://CONTROL_HOST:8000
python -m server.services.provision_tiled_manager --uri http://CONTROL_HOST:8000 --apply
```

It reads the existing private profile and environment, creates dedicated reader
and manager roles, changes only those identities' assignments, and adds the
manager password reference to the existing local provider. Unknown role
assignments are rejected for review. Other roles, users, keys, and acquisition
credentials are preserved. Repeated application preserves the manager password.
It never calls the older `provision_local_auth.py`, which rotates writer keys.

Each application creates a private `manager-backup-*` directory containing
original configuration, original targeted role definitions/assignments, and a
file manifest. These contain secrets and remain outside the checkout. A failed
application rolls back its DB transaction and restores affected files. Apply
when acquisition is idle, then restart only Tiled, verify reader access to both
catalogs, manager authorization, and a disposable issue/read/revoke cycle.

To restore the saved configuration and role assignments:

```console
python -m server.services.provision_tiled_manager --uri http://CONTROL_HOST:8000 --rollback /private/path/manager-backup-TIMESTAMP
```

Restart Tiled when idle after rollback. Rollback does not delete API keys or
principals; a newly created manager principal is left without roles, and its
password is removed from the restored provider configuration. Existing
sessions may retain their token scopes until expiration. Treat emergency
session/key revocation as a separate administrator operation.

Native APIs: [Tiled keys](https://blueskyproject.io/tiled/user-guide/api-keys.html)
and [Tiled scopes](https://blueskyproject.io/tiled/reference/scopes.html).

## Validation limits

Native Windows DPAPI and exported-file ACL behavior still need a Windows test
run. Linux tests exercise their platform-independent handling with mocks.

The installed Tiled 0.2.17 authentication cleanup task raises an error in
`purge_expired`: it calls datetime `replace` on a SQLAlchemy column expression.
This is outside the repository's manager code. Server authorization tests
confirm that expired API keys are still rejected on use and explicit revocation
works; periodic removal of unused expired records needs a separate dependency
fix. The regression logs retain this background-task error rather than hiding
it. The manager does not patch the installed Tiled package; cleanup needs a
separate dependency fix.
