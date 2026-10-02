# Shared experiment workspace

Both experiment screens compose `control_ui/layouts/experiment_workspace.ui`
through `ExperimentWorkspace`. Edit the common shell in Qt Designer: source
status at top left, Run Engine controls across the top, plan/queue on the left,
central viewer tabs, acquisition strip, and experiment controls on the right.
The splitter minimums and wider default viewer allocation are preserved.

Mode files under `diffractometer_controls/screens/{diffraction,tomography}/`
contain their original viewer, acquisition strip and EPICS controls. Diffraction
keeps its PSD/scalar plotting, gauges, count table and analyzer; tomography keeps
its image tools, levels, filtering, normalization, profiles and calculator.
Scientific implementations live in the mode or `analysis/` modules.

The application creates `ControlServices(re_client, documents, re_manager_api)`
once. Shared widgets receive services explicitly and never import the
application, site or server. `site/mitr/experiment_screen.py` assembles the mode
and supplies reactor display, macros, directory options and estimation context
from the Python profile.

`install_bluesky_controls` builds the shared widgets and registers their owners.
`install_experiment(viewer=..., controls=..., acquisition=...)` fills the
instrument slots; `add_tab` inserts tools. Calculators call
`load_proposed_plan(item)` directly, preserving compatible editor parameters
without widget-tree searches or delayed construction retries.

Registered components supply `activate`, `deactivate` and `shutdown`.
Transitions are idempotent. Shared displays pause owned callbacks, timers and
workers while hidden; the host owns the dispatcher and transports. Diffraction
retains hidden document processing and pauses its direct channels. Tomography
cancels image/profile workers and resumes on activation. Final shutdown removes
subscriptions and stops workers.

The custom editor uses the injected QueueServer API and `PlanEditorOptions`,
with no MITR environment reads or data roots in shared code. Queue estimates
coalesce rapid changes and discard cancelled work. Upstream adaptations are
installed once from `control_ui/core/compatibility.py`. Unsupported editor
internals raise an explicit error; delayed callbacks belong to their Qt widgets,
and model callbacks are released on destruction.

Theme and plotting palettes live in `control_ui/core/themes.py`. Reactor actions
and suspenders remain site-specific. `SourceStatusIndicator` is a reusable
numeric source readout. See the [template guide](instrument_template.md) and
[reference captures](reference_layouts/README.md) for extension examples,
light/dark comparisons, font diagnosis and verification limitations.
