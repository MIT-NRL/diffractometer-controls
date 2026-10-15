# Diffractometer controls

MITR diffraction and tomography share a Designer-editable operator workspace.
The same Run Engine controls, plan editor, queue, console and layout can host
another instrument's viewer and EPICS panels.

Activate the existing Conda/Mamba Bluesky environment and run from the checkout
root; no package installation or dependency upgrade is needed:

```console
python -m diffractometer_controls
```

The directory form `python diffractometer_controls` also works from the checkout
root and starts the same launcher.

Direct script and IDE launches also work without installing the project:

```console
cd diffractometer_controls
python launcher.py
```

You can also run `python /path/to/checkout/diffractometer_controls/launcher.py`
from another directory. The launcher resolves imports and bundled displays from
its own location.

Use **Bluesky Servers**, beside **IOCs** in the home toolbar (also under
**Control System**), to monitor Queue Server, the document proxy and Tiled.
The panel shows service state, PID, CPU, memory, task count, uptime, live journals
and an independent RunEngine console. Local service controls respect the existing
maintenance permission and GUI read-only mode. Remote connections show Queue
Server health and its console; local systemd controls are available on Linux.

Both service panels share an **Overview** with status cards, aligned resource
values and the same action order. Green indicates healthy, amber needs attention,
red indicates offline or failed, and blue indicates active work. Bluesky console
output and journals live under **Console / logs**; each service's console button
opens the relevant view. IOC host/connection details expand on each card, with
camera acquisition diagnostics under **Camera health**. Process state and
EPICS/worker connectivity are shown independently.

The IOC panel controls five independent, manually started user services: motors
and shared instrument records, He3 PSD, ZWO/QHY imaging, counter/analog/digital
cards, and Reolink cameras. Consoles are read-only service journals; no IOC shell
input is needed. Each IOC has its own heartbeat, resource statistics and autosave
files. Existing device PV names remain the same. See
[IOC deployment](docs/linux_server_migration.md#independent-epics-iocs) for service
names, camera selection and host configuration.

```text
control_ui/                 reusable services, widgets, layout and EPICS panels
diffractometer_controls/    application, experiment screens, analysis and MITR profile
server/                    ordered startup, device/plan definitions, writers and config
vendor/displays/           generated EPICS display resources
scripts/                   Linux session launcher and local verification tools
examples/                  disconnected workspaces and existing simulation examples
tests/                     GUI, scientific and offline server checks
docs/                      architecture, reference layouts and Linux migration guide
```

Start with the [instrument template guide](docs/instrument_template.md) and
[shared-workspace details](docs/shared_experiment_workspace.md). Try the actual
Bluesky controls with fake services:

```console
python examples/bluesky_workspace.py
```

The separate Linux control PC requires the path changes in the
[server migration checklist](docs/linux_server_migration.md). Existing endpoints,
permissions, settings keys, metadata and output formats are preserved. Credentials
and host-specific paths belong in private configuration outside Git.

For offline verification in a dependency-complete environment:

```console
python scripts/run_tests_isolated.py --output artifacts/tests/local
```

The runner records every module and treats skipped tests as incomplete coverage.
See [verification and reference captures](docs/reference_layouts/README.md) for
the known baseline limitations. Local captures and logs stay under ignored
`artifacts/`; selected references are in `docs/reference_layouts/`.
