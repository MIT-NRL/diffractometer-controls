# Use this project for another control system

Reuse `control_ui` as the common operator layer. An instrument supplies its
viewer, acquisition/status widget, EPICS controls and service-owning host.
Start from `examples/bluesky_workspace.py`: it uses the real shared editor,
queue, Run Engine controls and console with disconnected fake services, and
imports no MITR application modules.

```console
python examples/bluesky_workspace.py
python examples/bluesky_workspace.py --smoke-test
```

`examples/shared_experiment_workspace.py` demonstrates the lighter shell with
simple fake controls. Both examples run directly from a clone.

## Compose the workspace

Create clients and the document dispatcher once in your host. `documents`
supports `subscribe(callback, name="all")` and `unsubscribe(token)`;
`re_client` is the Bluesky widgets model, and `re_manager_api` is the host's
configured QueueServer API including authentication.

```python
from control_ui.core.services import ControlServices
from control_ui.core.options import PlanEditorOptions
from control_ui.layouts.experiment_workspace import ExperimentWorkspace
from control_ui.widgets.source_status import SourceStatusIndicator

services = ControlServices(re_client, documents, re_manager_api)
workspace = ExperimentWorkspace(services)
workspace.install_bluesky_controls(
    source_status_factory=lambda: SourceStatusIndicator(units="mA"),
    editor_options=PlanEditorOptions(query_mode="local", local_roots=()),
    estimation_context=your_runtime_context,
)
workspace.install_experiment(
    viewer=your_viewer,
    acquisition=your_acquisition_status,
    controls=your_epics_controls,
)
workspace.register_component(your_component_owner)
workspace.add_tab(your_calculator, "Calculator")
```

Viewer/panels can be any Qt widgets, including PyDM displays. Keep maintained
`.ui` files beside their Python components. `ServiceDisplay` resolves its UI
relative to the source module and requires explicit services. A new site's
adapter can supply PV mappings, macros, branding, endpoints and screen factories
following `diffractometer_controls/site/mitr/profile.py`. Application actions
and site metadata belong in that adapter.

Send proposed plans through the explicit editor interface:

```python
workspace.load_proposed_plan({
    "item_type": "plan", "name": "your_count",
    "kwargs": {"duration": 2.0},
})
```

Your allowed-plan inventory must advertise the name and parameter annotations.
The editor retains device selectors, dynamic choices, validation and queue
editing. `PlanEditorOptions` configures local/worker/stream directory queries,
roots, cache TTL, stream/snapshot addresses, topic and worker function. Supply a
matching worker service when enabling remote choices. The queue accepts an
estimation-context provider for instrument-specific acquisition times/PV mappings.
Optional viewer plotting backends belong to the instrument component.

## Own lifecycle and connections

Implement an owner with `activate`, `deactivate` and `shutdown`. Activation
resumes channels/workers once; deactivation pauses them while retaining required
document processing; shutdown unsubscribes documents and stops all remaining
work. Register the owner with the workspace and connect the host's final quit to
`workspace.shutdown`. The host then stops its own dispatcher and transports.

`DisplayOwner` handles callbacks and Qt timers introduced by a shared-display
factory. Cache the workspace and call `deactivate`/`activate` for navigation to
retain editor state. Do not construct another dispatcher, API, polling loop or
console monitor on each navigation. Delayed callbacks must be owned by their
widgets so deletion cancels them.

`control_ui/core/compatibility.py` centralizes the upstream adaptations and is
idempotent. Keep patches here and test against the installed Bluesky environment
before changing dependency versions. Required GUI tests need Qt, PyDM and
Bluesky dependencies; import failures are failures.

## Extend the server

Put device classes in `server/devices`, plans in `server/plans`, writers in
`server/writers`, and status/services in `server/services`. Definition imports
must not instantiate hardware or start subscriptions. Numbered
`server/startup/*.py` entry points perform those actions explicitly.

Plan factories receive `StartupContext(namespace, runtime=...)`. Resolve device
defaults and dynamic annotations inside the factory after earlier startup files
instantiate devices, then publish the result:

```python
from server.context import StartupContext
from server.plans.scalar import create_plans

context = StartupContext(globals())
context.publish(create_plans(context))
```

MITR startup preserves original ordering and public contracts. Another site
should replace its configuration/startup and establish its own inventory
contract. Credentials, machine paths, service units and private environment
files stay outside Git. See the [Linux checklist](linux_server_migration.md).

Validate both themes, editor/queue workflows, proposed-plan transfer, repeated
navigation and shutdown with fake services. Check panels/macros and external
vendor displays. Use fake devices and mocked services for ordered startup before
hardware acceptance. Structural tests guard the shared-package boundary,
initializers and Designer resource imports.
