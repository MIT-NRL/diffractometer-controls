# Reusable diffraction analysis

This package browses Tiled diffraction runs and fits scalar traces or individual
1D acquisition frames. It imports no instrument application, QueueServer,
EPICS, server writer, or local authentication configuration.

- `history.py`: Qt-free run queries, descriptors, normalized event rows, and
  external-array reads through Tiled.
- `curves.py`: numeric signal, frame, and coordinate selection.
- `colors.py`: stable signal identities using the shared live plot colors.
- `metadata.py`: evidence-based run outcomes and a read-only overview.
- `widgets.py`: shared status icons and double-click plot-axis reset.
- `fitting.py`: Qt-free lmfit models, validation, fit results and residuals.
- `viewer.py`: run-data overlays, point arrows and recorded motor-position selection.
- `panel.py`: explicit model/window controls and selected-curve/residual plots.
- `batch.py`: Qt-free per-frame fitting, cancellation, failed-point records and trend arrays.
- `trends.py`: parameter plots with available errors, motor-coordinate selection, and per-point results.
- `window.py`: run browser, asynchronous loading/fitting, metadata inspector.

The MITR adapter is
`diffractometer_controls/screens/diffraction/diffraction_history_window.py`.
It supplies the production/test catalog names. Its caller supplies an already
logged-in Tiled client; credentials and host policy stay outside this package.
The old `diffractometer_controls.analysis.diffraction_history` import path is a
thin forwarding module for the acquisition GUI and existing callers.

## Embed on another system

```python
from diffraction_analysis.window import DiffractionAnalysisWindow

window = DiffractionAnalysisWindow(
    authenticated_tiled_client,
    catalogs={"Experiment runs": "runs", "Commissioning": "test-runs"},
    catalog="Experiment runs",
)
window.show()
```

Create a QApplication first, retain the window, and close it when the host
application exits. Catalog values are paths relative to the supplied client.
The default query includes only `start.experiment_type == "diffraction"` and
loads 100 records initially, with explicit requests for older records. Changing
run, signal, spectrum coordinate, model or fitting window invalidates older fit
results, including work still running in a background thread. Navigating between
acquisition points preserves completed fits and redisplays the selected result.

Both history selectors display a green check for recorded success, a red cross
for failure/abort or unavailable data, and an amber marker for partial/stopped
runs. The analysis list can filter by status or recorded reason. A missing Stop
document is **Unfinished**, not a claim that the run is active. Bluesky's
`RunEngine.stop()` may record `success`; a successful exit with fewer primary
events than a recorded planned count is labelled **Partial**, without guessing
the cause. Historical records without enough evidence cannot distinguish that
operator stop from completion. The original exit status and reason remain visible.

The lower **Overview** shows run/sample, acquisition and storage in independent
columns so a file export failure is distinct from an acquisition failure.
It includes recorded/loaded/planned counts, local timestamps with timezone,
duration, and available file identity/checksum. **All metadata** searches keys
and nested values; **JSON / copy** preserves the complete metadata view.
**Copy overview** copies the readable summary. These views do not rewrite data.

**Acquisition** places recorded scan geometry beside timing and detector setup:
start/stop/step, point count, list positions and multi-motor/grid parameters,
exposure or software dwell per point, delay, wall-clock duration and runtime
estimate. Available PSD bin counts and lower thresholds are also shown.
Geometry uses recorded motor units; an exposure setting is distinct from the
elapsed duration of the whole run. Missing parameters are not invented.

**Environment / power** lists reactor-power baseline readings first, followed
by instrument readbacks, with start/end values and recorded units. Recorded
monitor stream names indicate where additional time-series readings exist;
baseline endpoints are not reported as their minima, maxima or averages.
Older runs can recover a display-only baseline summary from their stored
baseline stream; no PV connections, array reads, or catalog writes are needed.
Recovered information is explicitly labelled `viewer_details` in the JSON.
Duplicate setpoints remain available under All metadata.

Runs should record `data_type` (`scalar` or `1d`), detector and motor identities,
and descriptor data-key dtype/shape/units. `live_plot_fields` labels and roles
are used when available. Arbitrarily named numeric 1D signals are supported;
this is not restricted to He3 or `_counts` fields. The user can select a matching
array coordinate, or a scalar motor coordinate for a scalar trace. Missing
coordinate metadata leaves an explicit bin/point-number axis.

An alternate normalized loader can be passed as `snapshot_loader`. Its result
uses `start`, `stop`, `rows`, `seq_nums`, and `descriptor_docs`; see `history.py`.
HDF5 references are resolved by the Tiled server, so client file mounts are not
required. This package does not import old archives into a catalog.
Incremental runs can be inspected before Stop: the loader freezes the advertised
committed point count and reads that prefix consistently, even when another
point is being saved. It does not include an unfinished exposure or change an
unfinished/failed acquisition into a successful run.
Scalar scans read their primary Tiled table independently of CSV/NeXus export
status or HDF5 point counts. Values and sequence numbers come from one table
snapshot; pending/failed exports remain visible as separate metadata. Live
monitor samples are not substituted for completed scalar acquisition points.

## Run data and fitting tabs

The window opens on **Run data**. PSD/1D scans default to **Show all spectra**.
With multiple profile signals, **All profile signals** overlays every signal and
acquisition frame; choosing a signal restricts the overlay to that detector.
The same shared age ramp is used by live plots and this viewer: newest spectra
are blue/orange and older ones progress toward green/red. Isolating a detector
or point retains that color identity. Selected fitting data/residuals match the
corresponding spectrum; scalar traces and fit-parameter trends use fixed series
colors. Palette changes restyle existing curves without changing the selected
point, plot range or stored fits. In an overlay, each detector's legend sample
uses its newest spectrum.
Uncheck Show all for a single acquisition point. Arrows, the point selector and
the recorded motor-position selector navigate individual frames. Scalar data
shows the complete selected trace against its recorded coordinate. There are no
fit controls or automatic fits on this tab.

**Fitting** takes the data tab's selected signal/frame/coordinate when that
selection changes. Choose the common model and window, then **Fit selected
curve** or **Fit all points**. Fit all means every acquisition frame of the
selected 1D signal, not every detector. A scalar scan is one trace, so its batch
button is disabled. The same window, weighting and starting-center policy apply
to every frame. Batch fitting runs in a worker with progress and **Stop**;
stopping preserves completed results and takes effect after the current frame.

**Fit results → Parameter trends** offers parameter trends and an all-point results
table. Choose peak position, peak height, integrated peak area, FWHM or sigma;
choose the peak component and recorded motor/readback coordinate. Point number
is the explicit fallback when no motor coordinate is available. Failed fits and
missing coordinates are NaN gaps, never zeros. Available standard errors are
shown as error bars; unavailable errors are omitted. Scalar-trace parameters
remain in the single-fit results rather than pretending to be per-frame trends.
Double-click a point-result row to inspect that spectrum and its stored fit.
Separate single-frame fits accumulate too. Changing model or window clears the
scan results, while point navigation retains them. Copy results exports every
attempted point, its coordinates, configuration, parameters and failure status.
Peak component numbers follow center order; crossing or ambiguous peaks need
review rather than being treated as physical identity tracking.

The interactive fit region is excluded from auto-ranging, and curve rendering
is cached in device coordinates. While dragging, only the bounds preview is
updated, throttled to about 30 Hz with blocked control-feedback signals. Result
invalidation occurs once on release. The drag does not load arrays, run a fit,
or rebuild result tables on each mouse event. View clipping/downsampling is
used only with increasing coordinates and affects rendering, never fit inputs.
Double-click any plot to restore its axis limits. The selected-curve and
residual plots share their X range; resetting either restores the full curve
and residual Y range without changing the fitting window or stored fits.
Full data arrays determine bounds rather than a clipped rendering cache, which
prevents a narrow old zoom carrying into a newly loaded spectrum.
The relevant APIs are described by PyQtGraph's
[region signals](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/linearregionitem.html)
and [plot performance guidance](https://pyqtgraph.readthedocs.io/en/latest/api_reference/graphicsItems/plotdataitem.html).

## Fit without Qt

```python
from diffraction_analysis.fitting import FitConfig, fit_curve

result = fit_curve(x, y, FitConfig(
    peak="Pseudo-Voigt", background="Linear", xmin=-20, xmax=25,
    peaks=1, weighting="Equal",
))
center = result["parameters"]["p1_center"]
```

Supported profiles are Gaussian, Lorentzian, Voigt and pseudo-Voigt. One to six
positive peaks of the selected type can be combined with no background, a
constant, or a linear background. Optional comma-separated starting centers in
the GUI help resolve overlapping peaks. Centers stay ordered within the selected
window. Voigt Gaussian and Lorentzian widths are fitted independently.

Fit windows include both endpoints; nonfinite points are omitted and remaining
coordinates are sorted. At least seven distinct finite coordinates, and more
points than free parameters, are required. Constant data and invalid windows
produce an explanatory error without replacing the data plot. Equal weighting
is the default. `Poisson (counts)` uses `1/sqrt(max(y,1))` weights for nonnegative
raw counts, not calibrated/averaged scalar signals.

The orange curve is the fitted total, the dashed green curve is the background,
and the lower plot contains unweighted data-minus-fit residuals. The results
include parameter estimates, available covariance standard errors, FWHM, peak
height, reduced chi-square, AIC/BIC and the lmfit report. `amplitude` means peak
**area**, not height. Sigma has different meanings across the profiles; use FWHM
for width comparisons. Missing covariance is reported as unavailable, not zero.
Fit scores should be compared only for the same data and weighting. Copying
results includes the run UID, field, frame, coordinate, configuration and units.
Fits remain local analysis results and do not modify Tiled or acquisition files.

## Layout rationale

Run navigation is shared on the left. Run data and Fitting are separate task
tabs; the data overlay or selected fit and residuals occupy the flexible center;
related selection/model/window controls are grouped in a scrollable form on the
right. Results and metadata share the lower tabs. Splitters let smaller and X11
screens allocate space without fixed widget geometry. This follows Qt's
[layout guidance](https://doc.qt.io/qt-6/designer-layouts.html), particularly form
layouts and user-adjustable splitters. No installed skill specifically targeted
desktop Qt GUI design. Peak definitions and composite backgrounds follow the
[lmfit built-in models](https://lmfit.github.io/lmfit-py/builtin_models.html).

Dependencies: NumPy, SciPy and lmfit for curve fitting; QtPy, a Qt binding and
PyQtGraph for the GUI; Tiled and the Bluesky Tiled client plugins for catalog
access. These are available in the control-system environment.
