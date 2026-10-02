# Reference layouts and verification

`baseline/` contains the original diffraction/tomography light and dark layouts
and two scalar-gauge captures from archived revision `8064966`. `current/`
contains the same six views after composition and relocation, at revision
`6de16bc`. All captures use 2200 × 1400 windows on native Windows Qt with Fusion
style. JSON records revision, dimensions, theme, Qt platform, resolved fonts,
source hashes and image hashes. Personal interpreter paths are omitted.

Inputs are disconnected fake services and, for scalar views, two document-only
simulated counters. EPICS reads/writes/channel connections and QueueServer
polling/console transports are blocked. No production plan runs. The archived
scalar recapture saved its images/metadata, then exited 5 during old GUI teardown;
this is recorded in baseline metadata. The final capture exits cleanly.

Reproduce the current views using the existing Bluesky environment:

```console
python scripts/capture_workspace_references.py --output artifacts/reference_layouts/local
```

On Windows, use the default native `windows` Qt platform. This environment's
`offscreen` platform resolves ordinary text to `Font Awesome 5 Free`, producing
icon-like symbols instead of letters. Native captures resolve Segoe UI/Courier
New correctly. The script checks font substitutions and fails on unexpected
icon-font fallback. Disconnected PV placeholders such as `#####` are expected.

The original mode viewers, tools and panels remain; the outer shell is shared.
Both themes preserve semantic status and EPICS-panel colors. Scalar references
show matching simulated count values, gauges and table. New source-relative
paths fix plotting-backend discovery after the folder move.

## Test results

`verification.json` records the complete isolated baseline and final runs.
The final suite has 181 passing tests across 21 modules and no skipped tests.
It includes real Qt tests for both screen compositions, queue/editor workflows,
calculator transfer, hidden diffraction documents, repeated navigation,
destruction callbacks, timers/workers and optional plotting fallback.

All 22 original public plan contracts match. Offline checks execute all 15
numeric startup files with fake devices and mocked services, and guard imports
of server definitions against hardware construction and EPICS I/O. Structural
checks validate Designer imports/bundled resource paths and prohibit
application/site/server imports from shared controls.

The baseline had five failing modules. The relocated Reolink source-loader test
now reads UTF-8 explicitly and passes on Windows. Four baseline limitations
remain:

| Suite | Development-environment limitation |
| --- | --- |
| Updater subprocess tests | Windows selectors cannot select subprocess pipes (`WinError 10038`) |
| Focus viewer lifecycle | Native process abort in the existing environment |
| HE3 NeXus writer | `apstools` unavailable |
| Scalar writer | `apstools` unavailable |

These failures are reported, with no silent skipped imports or dependency
upgrades. The standalone real-Bluesky fake-service example and supported
application `--help` entry point also pass. Full logs remain in ignored
`artifacts/`. Actual writer outputs, Linux service operation and hardware
acceptance are outstanding deployment checks in the
[control-host checklist](../linux_server_migration.md).

## View captures

| View | Baseline | Current |
| --- | --- | --- |
| Diffraction, light | [PNG](baseline/diffraction_light.png) | [PNG](current/diffraction_light.png) |
| Diffraction, dark | [PNG](baseline/diffraction_dark.png) | [PNG](current/diffraction_dark.png) |
| Scalar gauges, light | [PNG](baseline/diffraction_scalar_light.png) | [PNG](current/diffraction_scalar_light.png) |
| Scalar gauges, dark | [PNG](baseline/diffraction_scalar_dark.png) | [PNG](current/diffraction_scalar_dark.png) |
| Tomography, light | [PNG](baseline/tomography_light.png) | [PNG](current/tomography_light.png) |
| Tomography, dark | [PNG](baseline/tomography_dark.png) | [PNG](current/tomography_dark.png) |
