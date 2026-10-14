# Local demo

Activate `bluesky-server` (or your existing GUI/server environment), then run
from the checkout root:

```console
python -m pip install --no-deps -r requirementsDemo.txt
python -m diffractometer_controls --demo
```

The optional requirements add caproto, fakeredis and sortedcontainers without
upgrading the existing Bluesky/Qt stack. fakeredis supplies an in-memory Redis
TCP server; no `redis-server` executable or system service is required. The
normal GUI/server dependencies in `environmentBluesky.yml` are still needed.
On Windows, activate Conda before launching: NumPy's native libraries may fail
if an absolute interpreter path is used without the environment's library path.

The red banner identifies simulated hardware and disabled storage. Open the
diffraction or tomography workspace, click **Connect**, choose a plan, and run
the queue normally. **Scalar count** and **Count rate** open the real USB-CTR
panels attached to the same demo stack. Motor panels and camera controls use
Channel Access to the local IOC.

Supported devices include motors, two HE3 PSD channels, the production
USBCTR08Scaler device, USB2408 temperature/analog/binary readouts, and two
128×128 cameras. `cam1` produces a Gaussian image; `sim_focus_cam` produces a
slanted edge whose sharpness varies with focus. Camera exposure, gain, offset,
binning, single/multiple/continuous acquisition, and counters are simulated.
The signals model control behavior; their values are not a detector calibration.

The demo loads the current helper, scalar, HE3, imaging, tomography and adaptive
plan factories through a separate startup module. Production device startup,
reactor suspenders and production service-management actions are excluded.
The main GUI automatically opens a memory-only viewer for
`adaptive_imaging_focus_scan`. Choose `sim_focus_cam` as the detector for a
slanted-edge focus test, for example:

```python
{
    "name": "adaptive_imaging_focus_scan",
    "item_type": "plan",
    "kwargs": {
        "file_name": "demo_focus", "file_dir": "",
        "detector": "sim_focus_cam", "motor": "cam1.focus",
        "start_pos": -0.5, "stop_pos": 0.5, "num_steps": 7,
        "exposure_time": 0.1,
    },
}
```

The viewer reuses the existing MTF analysis and adaptive command protocol.
Refine, extend, go to focus, complete and abort operate on the running plan.
Closing an unfinished viewer requests abort.

## Isolation and lifetime

All services bind to loopback with the `demo4dh4:` PV prefix. CA/PVA automatic
network searches are disabled and private production configuration files are
not loaded. Default ports are:

| Service | Port |
| --- | --- |
| In-memory Redis | 6380 |
| QueueServer control / information | 61615 / 61625 |
| Document proxy input / output | 61567 / 61568 |
| Channel Access IOC / repeater | 6064 / 6065 |

An occupied port stops startup with an error; the demo does not reuse or stop
another server. Additional GUI windows inherit the demo endpoints and do not
start another stack. Closing the owning GUI stops its additional windows and
services. Windows uses a Job Object for the service process tree; Linux and
macOS use dedicated process groups, including descendant cleanup after a
supervisor has exited.

Run metadata, queue/history and images remain in memory for the session. Tiled,
TIFF, scalar CSV and NeXus writers are disabled. Filename/directory parameters
remain for plan compatibility but do not produce acquisition files. Temporary
service logs are removed on shutdown. Qt may still save its normal appearance
and layout preferences.

## Verification

```console
python -m unittest tests.test_demo tests.test_widget_regressions
python scripts/demo_smoke.py --advanced
python scripts/demo_gui_smoke.py
```

The service smoke test executes scalar/PSD/imaging counts, a scalar scan,
tomography, adaptive refinement/focus commands, and pause/resume/abort. It
checks deterministic scaler readings, streamed documents, absence of stored
file-resource documents, and stopped processes. The GUI smoke test exercises
connected diffraction/tomography, timed scalar and MCS panels, streamed MTF
analysis, cached navigation and an additional panel process. Screenshots go to
ignored `artifacts/demo-gui/`. Its focus viewer runs in the GUI test process;
ordinary application operation launches a separate viewer process.

These integration checks were run on Windows. Linux/macOS cleanup is implemented
and its process-group behavior has a regression test; the full integration
commands still need to run on those operating systems. Existing writer tests
require `apstools`, which is unrelated to storage-disabled demo operation.
