# Linux control-host migration checklist

This guide covers checkout migration and the manually started IOC split.
Repository updates alone do not deploy host configuration or restart hardware.
Apply deployment steps during a maintenance window with acquisition idle.
Old module/display paths have no compatibility shims.

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

Keep any QueueServer emergency-unlock value in its private service environment
using `QSERVER_EMERGENCY_LOCK_KEY_FOR_SERVER`; no unlock key belongs in the
tracked YAML. The control host sources this alongside its private ZMQ key.

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

If `4dh4gui` was a shell alias, update it to the relocated launcher; aliases in
an existing terminal retain their old target until the shell configuration is
reloaded. The command on the control host is installed as
`~/.local/bin/4dh4gui`, which invokes `scripts/4dh4gui.pl` through Perl. Its Bash
alias points to that command. Reload the alias before launching from an existing
terminal:

```console
source ~/.bashrc
4dh4gui start
```

## Verify before enabling acquisition

Open **Bluesky Servers** beside **IOCs** in the GUI toolbar to check the actual
user services (`queue-server.service`, `bluesky-proxy.service`, and
`tiled-server.service`). CPU and memory account for the entire service, including
its worker children. CPU is sampled every two seconds; 100% means one fully used
core. The RE console receives new QueueServer messages independently of the
experiment console. The journal tabs retain recent startup output and follow new
entries. Closing the panel stops its subscribers, timers and journal processes.

Start/stop/restart require the existing `MITR_ALLOW_LOCAL_MAINTENANCE` setting
and a writable GUI on the local Linux host. Stop/restart require a recent idle
QueueServer status with no running/paused plan or background tasks. Close the
worker environment through the RunEngine controls before stopping or restarting
QueueServer. In read-only mode, status, resource usage and logs remain available.
The panel does not start the EPICS IOCs. A healthy manager with a closed worker
does not establish that the main IOC or detector PVs are available; verify those
through **IOCs** before opening the worker environment.

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

## Independent EPICS IOCs

The control host uses separate foreground user services. Every service is manual:
`Restart=no`, no `[Install]` target, no dependencies that start another IOC.
Reolink was disabled in `default.target.wants` as part of this migration. Starting
the GUI only monitors these services; it does not start them.

| Device group | User service | Statistics prefix |
| --- | --- | --- |
| Motors, instrument metadata and Bluesky status | `4dh4-motors.service` | `4dh4:IOC:Motors` |
| He3 PSD | `4dh4-psd.service` | `4dh4:IOC:PSD` |
| ZWO or QHY and their AreaDetector plugins | `4dh4-imaging.service` | `4dh4:IOC:Imaging` |
| USBCTR and USB2408 | `4dh4-daq.service` | `4dh4:IOC:DAQ` |
| Reolink cameras | `reolinkioc.service` | `4dh4:IOC:Reolink` |

Each prefix exposes `:HEARTBEAT`, `:UPTIME`, CPU, memory and connection statistics.
Aliases preserve the original `4dh4:*` and `4dh4:ReolinkIOC:*` statistics names.
Device PV names are unchanged, so existing screens and Ophyd devices use the
same channels. Only one IOC owns each device record. Shared instrument records
and the legacy global autosave-status PVs belong to the motors IOC.

The **IOCs** panel provides individual start, restart, stop and read-only journal
controls. These use `systemctl --user` and `journalctl --user`; Screen and IOC shell
input are not used. Service operation still obeys existing local maintenance and
GUI read-only rules. The old host `4dh4ioc` convenience launcher now delegates to
the motors service only. Start the other groups separately through the panel.

The four groups use the installed `~/EPICS/IOCs/4dh4` binary with separate
`iocBoot/ioc4dh4-{motors,psd,imaging,daq}` startup and autosave directories. Their
systemd runner source is [ioc_main.c](../server/iocs/ioc_main.c). It stays in the
foreground without stdin and handles SIGTERM by running EPICS exit hooks and
flushing output. After those hooks, it avoids the installed Newport driver's
hanging C++ static destructor. `KillMode=mixed` lets the IOC finish cleanup before
systemd removes any remaining CA repeater child.

Private `~/.config/epics/network.env` contains distinct CA server, PVA server and
PVA search ports for every IOC. The motors and Reolink ports were retained; new
ports were checked before allocation. All new groups bind the same configured
server interface and disable automatic beacons consistently. Client endpoint
lists include every group. A custom port changes discovery; it is not an access
control. No access-security policy or firewall changes were introduced. Reolink
retains CA statistics/control and PVA image transport; all four new groups also
serve their record databases over PVA.

`MITR_CAMERA_BACKEND=zwo` (default) or `qhy` in
`~/.config/diffractometer-controls/control.env` selects both the imaging IOC startup
and the worker's camera device. To change it, first close the idle worker and stop
the imaging IOC, update that setting, then start the imaging IOC, restart Queue
Server to reload its environment and reopen the worker. ZWO and QHY use the same
canonical camera/plugin PVs, so they must never run simultaneously. The live
cutover was verified with ZWO; QHY requires acceptance with that camera connected.

The renderer reads installed support paths and device configuration, partitions
legacy autosaves while preserving array entries, and generates staged startup
files, compatibility aliases and units without starting anything:

```console
python -m server.iocs.deployment --ioc-root ~/EPICS/IOCs/4dh4 --stage artifacts/deployment/new-ioc-stage
```

Use a new staging directory each time. This renderer is for the original
monolithic configuration on this host, not an unattended installer. Inspect the
result, check port availability, back up the host IOC tree, binary, unit files and
private network settings, then close an idle worker and stop the old IOC before
installing or starting the split. Never copy legacy autosaves over running split
IOCs: their separate autosaves become authoritative after migration. ADCore's
plugin startup copy omits its extra global autosave-status database to prevent
duplicate PVs. Pass an explicit macro string when loading compatibility aliases;
the installed Lua database hook does not accept a null macro argument. Reolink
loads its alias database using the absolute `$(TOP)/iocBoot/iocReolink/` path,
because its database include path does not contain its boot directory.

For the deployed cutover, backups are under
`~/.local/state/diffractometer-controls/ioc-split-*`; the exact path is recorded in
`artifacts/deployment/ioc-split-backup.txt`. Live verification and inventory reports
are retained under `artifacts/deployment/ioc-split-final/`. A pre-existing 41-character
tomography record description was shortened in the external IOC source and installed
database so the complete tomography database can load. Hardware addresses, access
controls and acquisition timing were preserved.

## Roll back

Restore the recorded revision and its matching private configuration/service
arguments together, following the same idle/service procedure. Do not mix old
startup paths with the new checkout. Keep data/history/config backups intact.
For IOC rollback, stop all split services before restoring the original monolithic
startup, executable, private network configuration and launcher from the host
backup. Restore matching Reolink startup and service settings as well; avoid
starting two owners of the same hardware or PVs.
