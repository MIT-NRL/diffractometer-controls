# Shared experiment workspace

Diffraction and tomography now compose the same
`control_ui/layouts/experiment_workspace.ui` and `ExperimentWorkspace` class.
Edit this file in Qt Designer to change the common header, plan/queue column,
viewer tabs, acquisition strip, or control column. The mode `.ui` files contain
only their existing viewer, acquisition controls, and EPICS panels; diffraction
also supplies its analyzer tab and tomography supplies its calculator tab.

Run the application from the checkout root in the existing Bluesky environment:

```console
python -m diffractometer_controls
```

No installation step or environment upgrade is required. Existing launcher
configuration, settings, endpoints, and local machine overrides remain in use.

## Reuse

The host creates its transports once and passes
`ControlServices(re_client, documents, re_manager_api)` to its screens. The
generic workspace imports Qt and the service contract, and has no MITR,
application, server, or instrument imports. The MITR composition adapter stays
in `diffractometer_controls/experiment_screen.py`.

Call `install_shared_controls` with source status, Run Engine controls,
plans/queue, console/history, and an explicit editor handle. Call
`install_experiment` with any Qt viewer, acquisition widget, and instrument
controls. Use `add_tab` for additional tools, and `load_proposed_plan` to send
a recommendation to the editor while preserving existing user parameters.

Register component owners with `activate`, `deactivate`, and `shutdown` hooks.
The workspace invokes them once per state transition. MITR shared components
retain their widgets and editor state, pause owned model callbacks and timers
while hidden, and restore their workers on return. Hidden diffraction continues
to receive documents while its direct channels are suspended; final shutdown
removes its document subscriptions. The application continues to own the one
document dispatcher and shared transports.

This example uses the shell without importing MITR, EPICS, or Bluesky:

```console
python examples/shared_experiment_workspace.py
```

## Reference captures

`artifacts/` is an existing Git ignore rule. The local reference captures are
in `artifacts/reference_layouts/baseline/` and the unified layouts are in
`artifacts/reference_layouts/unified/`. Each directory contains diffraction and
tomography light/dark PNGs plus dimensions, Qt version, font diagnostics, and
Git revision in `metadata.json`. Original mode Designer files are preserved
in the baseline `source/` directory.

To reproduce the unified references:

```console
python scripts/capture_workspace_references.py --output artifacts/reference_layouts/unified
```

The capture script uses actual widgets with fake document services, dummy
local QueueServer addresses, disabled polling/console transport, and blocked
EPICS reads/writes. Blank viewers and `#####` PV readouts are expected when
disconnected. It runs at 2200 × 1400 by default.

Use the native Windows Qt platform on Windows. In this environment, Qt's
offscreen platform resolves ordinary text fonts to `Font Awesome 5 Free`,
substituting icons for letters. Native captures resolve the requested Segoe UI
font correctly. This is a capture-platform issue; the production theme and
scientific display algorithms were preserved. Both themes retain existing
semantic status colors and EPICS panel colors.

## Verification

```console
python -m unittest diffractometer_controls.tests.test_experiment_workspace
```

These tests require the GUI dependencies and do not skip failed imports. They
exercise both screen compositions, direct calculator/editor transfer,
repeated navigation, callback and worker ownership, hidden diffraction
documents, optional plotting failure, and final shutdown. Existing screen,
scalar plotting, calculator, queue/editor, and document dispatcher tests also
run in the same Bluesky environment.

The complete suite was also run module-by-module against an archived baseline
at `8064966` (including the two existing local edits) and the unified checkout.
Both runs have the same five failing modules: updater subprocess tests, focus
viewer lifecycle, Reolink viewer, and the two writer suites. The writer suites
lack `apstools` in this environment. No imports were silently skipped and no
new failing modules were introduced. Logs and per-module summaries are saved
under `artifacts/tests/{baseline,unified}/`. Environment repairs and server
deployment remain separate work.
