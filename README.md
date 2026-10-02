# Diffractometer controls

MITR diffraction and tomography share a Designer-editable operator workspace.
The same Run Engine controls, plan editor, queue, console and layout can host
another instrument's viewer and EPICS panels.

Activate the existing Conda/Mamba Bluesky environment and run from the checkout
root; no package installation or dependency upgrade is needed:

```console
python -m diffractometer_controls
```

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
