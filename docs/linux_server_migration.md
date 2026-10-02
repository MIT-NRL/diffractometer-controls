# Linux control-host migration checklist

The refactor changes checkout paths. It does not deploy to the separate Linux
PC, upgrade Conda/Mamba, change endpoints, or restart hardware. Apply this
checklist on that PC during an approved maintenance window with acquisition
idle. Old module/display paths have no compatibility shims.

## Record the running configuration

Record deployed Git revision, checkout, Conda interpreter, IOC XML paths, data
directories and service arguments. Back up private environment files and unit
overrides outside Git. Inspect the existing MITR user services without changing
them:

```console
systemctl --user cat queue-server.service
systemctl --user cat bluesky-proxy.service
systemctl --user cat tiled-server.service
systemctl --user status queue-server.service bluesky-proxy.service tiled-server.service
```

Preserve actual unit names and existing options. For system services, inspect
the corresponding units without `--user`. Record QueueServer inventory,
permissions, metadata/history mode, writer locations, Redis configuration,
Tiled catalog and authentication. Keep credential values out of logs and Git.

## Update checkout references together

Set `CONTROL_CHECKOUT` to the actual absolute clone path in the QueueServer
unit's private environment file or override. The tracked QueueServer config
expands environment variables:

```text
CONTROL_CHECKOUT=<absolute checkout root>
```

| Previous resource | New resource |
| --- | --- |
| `diffractometer_controls/bluesky_config/startup/` | `server/startup/` |
| `diffractometer_controls/bluesky_config/qserver_config.yml` | `server/config/qserver_config.yml` |
| startup permissions YAML | `server/config/user_group_permissions.yaml` |
| `diffractometer_controls/areaDetectorConfigXML/` | `server/config/` |
| `diffractometer_controls/4dh4gui.pl` | `scripts/4dh4gui.pl` |
| root experiment displays | `diffractometer_controls/screens/{diffraction,tomography}/` |
| `extra_ui/autoconvert/` | `vendor/displays/` |

Set QueueServer's `WorkingDirectory` to the checkout root and change its existing
`start-re-manager --config` argument to the new absolute config path. Keep its
existing absolute Conda executable and all other arguments. The command has
this form when the existing environment is active:

```console
start-re-manager --config "$CONTROL_CHECKOUT/server/config/qserver_config.yml"
```

The config's startup and permissions paths use `${CONTROL_CHECKOUT}`. Update
private wrapper scripts/config copies that override those paths. Move generated
inventory paths if used; generated inventory remains untracked. Numbered startup
filenames/order are unchanged. `00-base.py` adds the checkout root for imports;
no installation is required.

Keep network addresses/ports, private ZMQ key, Redis settings, permission content
and group names. Proxy/Tiled services need changes only if they refer to moved
files. Their other deployment settings stay as they are.

Detector XML defaults to source-relative `server/config/`. If the IOC is remote
or sees another filesystem, set `MITR_DETECTOR_XML_DIR` in the worker's private
configuration to the directory visible to that IOC and place matching XML there.
Confirm attributes/layout readbacks and successful XML loading.

## Preserve host settings and GUI launchers

Retain existing overrides. These paths can be configured outside Git:

| Variable | Purpose |
| --- | --- |
| `MITR_CONTROL_CHECKOUT` | GUI session checkout; defaults to script's parent |
| `MITR_CONDA_ACTIVATE`, `MITR_CONDA_ENV` | Existing GUI Conda activation/environment |
| `MITR_GITHUB_ROOT` | Existing local repository-tools root |
| `MITR_EPICS_ROOT`, `MITR_IOC_ROOT`, `MITR_IOC_LAUNCHER` | Local EPICS tools and IOC launcher |
| `MITR_EPICS_SUPPORT` | Installed synApps support/display tree |
| `PYDM_DISPLAYS_PATH` | External display paths, preserved before project entries |
| `MITR_DETECTOR_XML_DIR` | XML location visible to detector IOC |
| `MITR_IMAGING_DATA_ROOT`, `MITR_IMAGING_DATA_ROOTS` | Existing image directory choices |
| `MITR_FILE_DIR_*` | Existing local/worker/stream discovery options |
| `TILED_URI`, `MITR_TILED_URI`, `TILED_WRITER_API_KEY`, `TILED_API_KEY` | Existing catalog endpoint/authentication |
| `MITR_EPICS_CA_ADDR_LIST`, `MITR_EPICS_PVA_ADDR_LIST` | Existing private IOC endpoints |

Local EPICS defaults use the current user's home, retaining the same locations
for the existing `mitr_4dh4` account. Detector/writer data templates are unchanged;
review those site defaults before adapting this server elsewhere. Private path
values belong in host configuration, rather than personal absolute paths in Git.

From the checkout root:

```console
python -m diffractometer_controls
perl scripts/4dh4gui.pl run
```

The session script also supports `start`, `stop`, `restart` and `console`. Logs
remain under `~/.local/state/diffractometer-controls/`. Update desktop/session
wrappers to the relocated script; retain display/Xauthority and maintenance
permission settings.

## Verify before enabling acquisition

Run offline checks in the existing dependency-complete Linux environment:

```console
python -m unittest tests.test_server_startup tests.test_repository_structure
python scripts/run_tests_isolated.py --output artifacts/tests/control-host
```

Startup tests use fake devices and mock Tiled, writers and external services;
they do not certify installed services or writer output. Run the actual scalar
and NeXus writer suites with `apstools` available on the control host. It is
missing in the development Windows environment.

After reviewing updated units/configuration, reload/restart only services whose
paths changed using the site's existing procedure. Confirm status/journals and
single-process ownership. Open the QueueServer environment and compare inventory
with the baseline: device/plan names, defaults, signatures, annotations and
permissions. Check production/test history and Tiled selection.

Perform hardware acceptance: reactor suspenders/status, PV connections,
motors/scalers/camera panels, startup subscriptions, scalar/PSD/imaging documents,
representative writer files and XML readbacks. Verify repeated navigation and
shutdown without duplicate loops/timers or orphaned focus/profile workers.
Compare both themes and gauges with reference images. Test external displays
against installed support paths (`vendor/displays/external_references.json`).

## Roll back

Restore the recorded revision and its matching private configuration/service
arguments together, following the same idle/service procedure. Do not mix old
startup paths with the new checkout. Keep data/history/config backups intact.
No remote changes were performed during this refactor.
