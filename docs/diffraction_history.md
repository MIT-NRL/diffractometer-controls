# Diffraction history and viewer options

The existing history loader was in
`diffractometer_controls/screens/diffraction/diffraction_live_plot.py`, alongside
`DiffractionHistoryViewer` and the live plotting controller. Its data-reading
code now lives in the reusable `diffraction_analysis/history.py` package. The
old import path forwards to it for the control GUI.
The control GUI uses that shared loader; a separate application can use it too.

## GUI popout

Click **Analyze runs…** beside the embedded history controls. This opens a
separate window within the control GUI, leaving the live plot and acquisition
controls independent. It uses the same automatic local reader access or remote
login as the embedded history viewer. Enabling the embedded Tiled checkbox first
is optional.

The left pane selects **Production** or **Test**, lists diffraction runs newest
first, and filters the loaded list by title, sample, plan, status, scan ID or UID. It
shows the number of loaded and matching runs.

The main workspace has two tabs. **Run data** opens first and overlays all PSD
spectra by default. Choose a signal or all profile signals; switch to individual
points with the checkbox, arrows, point selector or recorded motor-position
selector. Scalar runs show their selected trace. This tab contains no fitting
controls.

**Fitting** has the selected spectrum and residuals, plus explicit model/window
controls. Fit one curve or **Fit all points** of the selected detector. Batch
progress and Stop are available. The lower Fit results tabs include **Parameter
trends**, which plots position, height, area, FWHM or sigma against a chosen
motor/readback coordinate with available error bars. Its all-point table lists
successes and failures; double-click a row to inspect that point. Navigating
between points retains their fits. Model/window changes invalidate previous
results, and failed points remain gaps. Live acquisition peak tracking is
unchanged.

Metadata remains available below either task. Fit results are shown only with
the Fitting tab. Bounds dragging now previews values at a limited rate, blocks
control feedback, caches plot rendering and keeps the view range fixed. It
invalidates results on release rather than rebuilding them during the drag.
Double-click any plot to reset axis limits without changing the fitting window
or results. The lower **Overview** presents run/sample, acquisition and storage
side by side. **All metadata** supports key/value search; full JSON and a copied
overview are also available. Acquisition and export status are separate.

Both the embedded selector and analysis browser show status icons: a green
check for completion, a red cross for failure/abort or an unsuccessful data
read, and an amber marker for stopped/partial runs. Missing Stop documents are
labelled Unfinished. Bluesky can record an operator stop as success, so only
known point-count shortfalls justify a Partial label; older records cannot
always distinguish completion from a stop.

In the control GUI, a failed acquisition or unavailable stored data clears the
old plots and displays an explanation over the empty plot area. That selection
still advances the navigation position, so the arrows continue to older/newer
runs instead of repeatedly retrying the same failure. The analysis popout can
still inspect available partial data from a failed acquisition. None of these
viewer actions changes the recorded acquisition or export status in Tiled.

The embedded history bar gives its buttons stable widths, lets the selector
fill the remaining space, and elides long status/title text with full tooltips.
Names no longer alter the layout. **Live scan** remains enabled independently
of Tiled: checked follows current/new documents and live detector updates;
unchecked holds the displayed run. Selecting history unchecks it, and turning
Tiled off resumes live following. Late history requests cannot override a new
live selection. An idle run-status PV means the retained last RunUID is not an
active acquisition; Start/Stop documents provide activity evidence when the
status PVs cannot be read. Resuming partway through a run shows current/future
data; completed HDF5-backed history becomes available after finalization.

PSD profile history uses blue-to-green and orange-to-red ramps from newest to
oldest for the first two detectors: newest spectra/previews are blue and orange;
older spectra become green and red. Additional detectors use violet/green/rose ramps. Summary
intensity and peak-position series retain their fixed blue/orange identities.
The ramps use theme-appropriate brightness and bounded contrast after alpha
blending. Existing spectra and previews are restyled when the palette changes.
The recent-history progression, newest-point markers, and live peak fitting
remain in place in both plotting backends.

The lower **Acquisition** tab presents geometry and timing side by side, including
recorded start/stop/step/point parameters, per-point exposure/software dwell,
delay, detector settings and elapsed versus estimated run duration.
**Environment / power** shows baseline start/end readbacks and recorded units,
with reactor power first and the names of available monitor streams.
Older baseline summaries are recovered for display from stored scalar data;
missing units/readings remain missing, and no historical records are changed.
Gauge volume, facility, beamline and operating mode are included in the overview
when recorded.
See the [package guide](../diffraction_analysis/README.md) for batch semantics,
numerical assumptions and rendering details.

The embedded history limit remains **100**. Its dropdown can show up to 20 rows
at once, subject to screen space; scroll to see the rest. The popout initially
loads 100 runs and offers **Load 100 older runs** if there are more. Filtering
applies to the loaded records. Production and test are separate catalogs; at
implementation, they contained 7 and 84 tagged diffraction runs respectively.
Imaging, focus scans, and untagged synthetic writer checks are excluded.

Closing the popout stops its background work. A failed or unfinished file export
still displays the available metadata and an error; it clears the preceding
plot so data from a different run is not mistaken for the selection. The viewer
does not write metadata or start hardware connections.

## Shared loader API

```python
from diffractometer_controls.analysis.diffraction_history import (
    build_history_snapshot,
    fetch_recent_history_entries,
)

catalog = client["4dh4"]  # Use "testdb" explicitly for test scans.
entries = fetch_recent_history_entries(catalog, limit=100)
snapshot = build_history_snapshot(catalog[entries[0]["uid"]])
```

The loader itself needs no Qt, EPICS, or QueueServer connection. It returns a
dictionary with `uid`, `start`, `stop`, `stream_name`, `descriptor_docs`, `rows`,
`seq_nums`, `file_export`, and `pending_reason`. Each row has the same detector
and motor field names as an acquisition event. Sequence numbers are retained
when the stored stream exposes them; older streams without them use row order.

The loader handles direct run/stream layouts and a `streams` namespace. New
HDF5-backed arrays are read through Tiled, so client computers do not need the
server's data directories mounted. Previous Tiled-stored arrays are supported,
including numeric array strings returned by older SQL storage. Missing or
truncated arrays raise an error rather than producing misleading profiles.

Scalar runs use their recorded `live_plot_fields`. For older scalar runs that
lack these, the loader derives fields from recorded detector identities and
numeric descriptor fields, preserving units and excluding motor fields. It
adds the derived plotting fields to the returned snapshot without changing
catalog metadata. Baseline and monitor streams do not replace an available
primary acquisition stream.

Recognized PSD plans can also recover absent `data_type` and detector identities
when matching one-dimensional counts and position descriptors exist. These
additions are exposed separately as `inferred_metadata` and used in the plotting
snapshot. The popout's metadata tree shows the catalog's recorded metadata and
any recovered plotting fields separately.

New file exports must have `file_export.status == "ready"`. An unfinished or
failed export produces a descriptive error. `allow_empty=True` is reserved for
live-startup backfill: it returns an empty snapshot with `pending_reason` until
the arrays are registered. Live documents continue to supply acquisition plots.
Legacy runs without export-status metadata are read using their stored streams.

## Reuse and fitting

The reusable browser, fitting engine and analysis panel are in
[`diffraction_analysis`](../diffraction_analysis/README.md). The package accepts
an authenticated Tiled client and a mapping of display labels to catalog paths;
it has no MITR, EPICS, QueueServer, writer or local-login dependency. The
instrument adapter supplies `4dh4` and `testdb`, and the surrounding control GUI
handles authentication.

The analysis panel handles scalar traces and individual 1D frames with arbitrary
signal names. It has no gauges and no automatic fits. Fit results include peak
parameters, available standard errors, FWHM/height, residuals, reduced chi-square,
AIC/BIC, and a copyable report containing run/selection provenance. These results
are not written back to the acquisition file or catalog. See the package guide
for API examples, numerical assumptions and layout references.

## Older production PSD archive

At the archive inspection, the production Tiled catalog had 7 diffraction entries
among 41 total runs. Separately, `~/Data/Diffraction/2024/` contained **165 NeXus/HDF5 files** from
August–December 2024, including `count_he3` and `scan_he3` runs. A sample of file
headers confirmed stored PSD counts and acquisition frames. Those files are
not represented by the seven production diffraction entries at that time. The
archive was inspected read-only; no bulk import, file rewrite or broad legacy
catalog compatibility was added.

## Legacy metadata repair

`server.services.diffraction_metadata` audits missing PSD plotting fields from
recorded plans and primary descriptors without reading or rewriting arrays.
The default is a dry run:

```sh
python -m server.services.diffraction_metadata --uri "$MITR_TILED_URI" --catalog testdb
```

To apply, use an administrative shell with a write-capable `TILED_WRITER_API_KEY`
or `TILED_API_KEY`, add `--apply`, and specify a new `--backup /private/path.jsonl`.
The GUI's reader key cannot perform repairs. Each original metadata record is
written and fsynced to a mode-600 backup before the catalog update. Existing
fields are preserved; `metadata_enrichment` records the evidence and added field
names. Repeating the repair is a no-op once the fields exist.

The initial repair adds `data_type: 1d` on 33 old test PSD runs and detector names
on 26 of those runs. Four failed/aborted runs without primary descriptors remain
unchanged. Production requires no repairs. Backups are stored privately under
`~/.config/tiled/metadata-backups/`. Original HDF5 files and their acquisition
documents stay intact; enrichment is identified as a later catalog annotation.

## Shared reader login

The server already had a `local` password provider and separate admin, reader,
and acquisition-writer credentials. The shared `4dh4-reader` account was verified
and its password replaced with a generated memorable passphrase. Its role can
read metadata and data in both catalogs and manage its own API keys; it cannot
create or modify scan data. The acquisition writer key and admin password were
left unchanged.

The control computer's private `~/.config/tiled/reader-login.txt` contains the
server URL, username, and new password. It and `tiled-server.env` have mode 600.
Credentials are not included in this repository. Tiled has been restarted and
the new password verified.

On the control computer, enabling **Tiled** in the existing history viewer now
connects automatically as the reader. Its dedicated API key grants only
`read:metadata` and `read:data`, and is stored in the private mode-600 file
`~/.config/diffractometer-controls/tiled-reader.json`. The file records the
computer's hostname and the server URL; automatic login applies only when both
match. This uses [Tiled's API-key support](https://blueskyproject.io/tiled/how-to/api-keys.html)
without loading the server's admin password or acquisition writer key into the
GUI. It takes precedence over cached logins on the control computer. Removing
the local configuration disables automatic login; the key can also be revoked
in Tiled. A missing, invalid, or revoked key falls back to normal login.

SSH/X11 launches still run the GUI process on the control computer, so they use
the same automatic reader access. The host's `control.env` sets `MITR_TILED_URI`
explicitly, independently of the QueueServer host. For IDE/direct launches that
do not load that environment, a missing or loopback control-host default uses
the server URL recorded in the local reader configuration. Explicit Tiled URI
overrides and explicit remote control hosts continue to take precedence.
This matters because Tiled listens on the control computer's LAN address;
`localhost:8000` is not its listening endpoint. Connection failures show a
connection error rather than requesting a password for an unreachable server.

On another computer, open the GUI's **Tiled** login, enter the server URL from
`reader-login.txt`, and use `4dh4-reader` with the new passphrase. The dialog defaults to
that username. For a persistent server default, set `MITR_TILED_URI`; optionally
set `MITR_TILED_USERNAME`. Existing `TILED_URI` takes precedence if set.

“Remember login” uses Tiled's token cache; the GUI does not save the password.
An unchecked box now works because the GUI keeps the authenticated in-memory
client instead of discarding it and reconnecting through cached tokens. Viewer
password login also bypasses any ambient acquisition API key. See Tiled's
[authentication documentation](https://blueskyproject.io/tiled/reference/authentication.html)
for the session/token mechanism.

Restart the control GUI to load the updated client code. This reader work does
not require reopening the QueueServer worker or starting an imaging IOC.
