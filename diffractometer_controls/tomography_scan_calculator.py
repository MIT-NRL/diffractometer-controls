"""Tomography sampling and neutron-count planning helper.

The angular recommendation uses the parallel-beam Crowther sampling
criterion, ``N_theta = pi * D / (2 * delta)``, where ``D`` is the object
diameter and ``delta`` is the requested transverse resolution.  Count-based
SNR values are Poisson estimates, not reconstructed-voxel SNR predictions.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from qtpy import QtCore, QtGui, QtWidgets

try:
    from diffractometer_controls.tomography_scan_parameters import (
        paired_second_half_indices,
    )
except ModuleNotFoundError:
    # PyDM loads displays directly by filename with this directory on sys.path,
    # so the package root is not necessarily importable in that execution mode.
    from tomography_scan_parameters import paired_second_half_indices


REFERENCE_FLUX_N_CM2_S = 3.65e6
REFERENCE_L_OVER_D = 554.0
REFERENCE_PINHOLE_MM = 8.0
DEFAULT_PIXEL_SIZE_UM = 19.6775
DEFAULT_OBJECT_DIAMETER_MM = 30.0
MIN_TARGET_RESOLUTION_UM = 20.0
DEFAULT_SPATIAL_RESOLUTION_UM = max(MIN_TARGET_RESOLUTION_UM, 2.0 * DEFAULT_PIXEL_SIZE_UM)
DEFAULT_EXPOSURE_TIME_S = 60.0
DEFAULT_DETECTION_EFFICIENCY_PERCENT = 80.0
DEFAULT_TILT_CORRECTION_PROJECTIONS = 20
DEFAULT_FRAME_OVERHEAD_S = 1.707510645

ACQUISITION_MODE_HALF = "half"
ACQUISITION_MODE_SPARSE_TILT = "sparse_tilt"
ACQUISITION_MODE_FULL = "full_360"


class CompactDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox that omits insignificant trailing decimal zeroes."""

    def textFromValue(self, value):
        text = super().textFromValue(value)
        decimal_point = self.locale().decimalPoint()
        if decimal_point in text:
            text = text.rstrip("0").rstrip(decimal_point)
        return "0" if text in ("-0", "+0") else text


def nominal_spatial_resolution_um(pixel_size_um: float) -> float:
    """Return the nominal two-pixel resolution with the fixed 20 µm floor.

    The GUI starts at this value but leaves the target editable for evaluating
    alternate detector configurations.
    """

    if not math.isfinite(float(pixel_size_um)) or float(pixel_size_um) <= 0:
        raise ValueError("pixel_size_um must be finite and greater than zero")
    return max(MIN_TARGET_RESOLUTION_UM, 2.0 * float(pixel_size_um))


def paired_second_half_angles_deg(
    base_projection_count: int,
    extra_projection_count: int,
) -> tuple[float, ...]:
    """Select evenly distributed second-half angles paired to the base grid.

    Each returned angle is exactly 180° above an acquired 0–180° base angle.
    The already acquired 180° position is excluded and 360° is always the
    final angle. The estimator separately requires at least two projections
    when this is used as a sparse correction set.
    """

    base_projection_count = int(base_projection_count)
    extra_projection_count = int(extra_projection_count)
    if base_projection_count < 2:
        raise ValueError("base_projection_count must be at least two")
    paired_indices = paired_second_half_indices(
        base_projection_count,
        extra_projection_count,
    )
    base_intervals = base_projection_count - 1
    base_step_deg = 180.0 / float(base_intervals)
    return tuple(180.0 + index * base_step_deg for index in paired_indices)


def _crowther_projection_counts(
    object_diameter_mm: float,
    resolution_um: float,
    sampling_fraction: float,
) -> tuple[float, int, int, float]:
    """Return resolvable elements, inclusive counts, and base angular step."""

    resolvable_elements = float(object_diameter_mm) * 1000.0 / float(resolution_um)
    full_sampling_exact = (math.pi / 2.0) * resolvable_elements
    full_sampling_intervals = max(1, int(math.ceil(full_sampling_exact)))
    recommended_intervals = max(
        1,
        int(math.ceil(full_sampling_exact * float(sampling_fraction))),
    )
    full_sampling_projections = full_sampling_intervals + 1
    recommended_projections = recommended_intervals + 1
    angular_step_deg = 180.0 / float(recommended_intervals)
    return (
        resolvable_elements,
        full_sampling_projections,
        recommended_projections,
        angular_step_deg,
    )


@dataclass(frozen=True)
class TomographyScanEstimate:
    object_diameter_mm: float
    sample_detector_distance_mm: float
    geometric_blur_um: float
    geometry_limited_resolution_um: float
    geometry_limited_resolvable_elements: float
    geometry_limited_full_sampling_projections: int
    geometry_limited_recommended_projections: int
    geometry_limited_total_angular_positions: int
    geometry_limited_total_frames: int
    geometry_limited_estimated_scan_time_s: float
    resolvable_elements: float
    full_sampling_projections: int
    recommended_projections: int
    angular_step_deg: float
    acquisition_mode: str
    tilt_correction_projections: int
    tilt_correction_step_deg: float | None
    tilt_correction_angles_deg: tuple[float, ...]
    total_angular_positions: int
    total_frames: int
    estimated_flux_n_cm2_s: float
    implied_pinhole_mm: float
    incident_neutrons_per_pixel_frame: float
    detected_neutrons_per_pixel_projection: float
    projection_count_snr: float
    projection_relative_noise_percent: float
    base_set_count_snr: float
    acquired_set_count_snr: float
    estimated_scan_time_s: float


def estimate_tomography_scan(
    *,
    object_diameter_mm: float,
    spatial_resolution_um: float,
    sampling_fraction: float,
    exposure_time_s: float,
    pixel_size_um: float,
    l_over_d: float,
    sample_detector_distance_mm: float = 0.0,
    transmission_fraction: float = 1.0,
    detector_efficiency_fraction: float = 1.0,
    frames_per_angle: int = 1,
    tilt_correction_projections: int = 0,
    full_360_scan: bool = False,
    reference_flux_n_cm2_s: float = REFERENCE_FLUX_N_CM2_S,
    reference_l_over_d: float = REFERENCE_L_OVER_D,
    reference_pinhole_mm: float = REFERENCE_PINHOLE_MM,
    frame_overhead_s: float = DEFAULT_FRAME_OVERHEAD_S,
) -> TomographyScanEstimate:
    """Calculate angular sampling, count statistics, and approximate time.

    Flux is scaled from the reference measurement as ``(L/D)^-2``.  This
    assumes the collimator length, source brightness, beam spectrum, and
    operating power are unchanged.  SNR is the ideal Poisson count SNR after
    applying transmission and detector efficiency.
    """

    positive_values = {
        "object_diameter_mm": object_diameter_mm,
        "spatial_resolution_um": spatial_resolution_um,
        "exposure_time_s": exposure_time_s,
        "pixel_size_um": pixel_size_um,
        "l_over_d": l_over_d,
        "reference_flux_n_cm2_s": reference_flux_n_cm2_s,
        "reference_l_over_d": reference_l_over_d,
        "reference_pinhole_mm": reference_pinhole_mm,
    }
    for name, value in positive_values.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and greater than zero")

    if not math.isfinite(float(sampling_fraction)) or not 0 < float(sampling_fraction) <= 1:
        raise ValueError("sampling_fraction must be greater than zero and at most one")
    for name, value in (
        ("transmission_fraction", transmission_fraction),
        ("detector_efficiency_fraction", detector_efficiency_fraction),
    ):
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be between zero and one")

    frames_per_angle = int(frames_per_angle)
    if frames_per_angle < 1:
        raise ValueError("frames_per_angle must be at least one")
    tilt_correction_projections = int(tilt_correction_projections)
    if tilt_correction_projections < 0:
        raise ValueError("tilt_correction_projections must be non-negative")
    if not bool(full_360_scan) and tilt_correction_projections == 1:
        raise ValueError("a sparse paired correction set requires at least two projections")
    if not math.isfinite(float(frame_overhead_s)) or float(frame_overhead_s) < 0:
        raise ValueError("frame_overhead_s must be finite and non-negative")
    if (
        not math.isfinite(float(sample_detector_distance_mm))
        or float(sample_detector_distance_mm) < 0
    ):
        raise ValueError("sample_detector_distance_mm must be finite and non-negative")
    object_diameter_mm = float(object_diameter_mm)
    sample_detector_distance_mm = float(sample_detector_distance_mm)
    geometric_blur_um = sample_detector_distance_mm * 1000.0 / float(l_over_d)
    geometry_limited_resolution_um = max(
        float(spatial_resolution_um),
        geometric_blur_um,
    )
    (
        resolvable_elements,
        full_sampling_projections,
        recommended_projections,
        angular_step_deg,
    ) = _crowther_projection_counts(
        object_diameter_mm,
        float(spatial_resolution_um),
        float(sampling_fraction),
    )
    (
        geometry_limited_resolvable_elements,
        geometry_limited_full_sampling_projections,
        geometry_limited_recommended_projections,
        _geometry_limited_angular_step_deg,
    ) = _crowther_projection_counts(
        object_diameter_mm,
        geometry_limited_resolution_um,
        float(sampling_fraction),
    )

    if bool(full_360_scan):
        acquisition_mode = ACQUISITION_MODE_FULL
        # 180° is already in the base set. Add only the remaining positions
        # through 360° so both halves use the same angular spacing.
        tilt_correction_projections = recommended_projections - 1
    elif tilt_correction_projections > 0:
        acquisition_mode = ACQUISITION_MODE_SPARSE_TILT
    else:
        acquisition_mode = ACQUISITION_MODE_HALF

    total_angular_positions = recommended_projections + tilt_correction_projections
    tilt_correction_angles_deg = paired_second_half_angles_deg(
        recommended_projections,
        tilt_correction_projections,
    )
    if acquisition_mode == ACQUISITION_MODE_FULL:
        tilt_correction_step_deg = angular_step_deg
    else:
        # Sparse extras exclude the already acquired 180° endpoint and include
        # 360°, giving the requested count evenly across the second half.
        tilt_correction_step_deg = (
            180.0 / float(tilt_correction_projections)
            if tilt_correction_projections > 0
            else None
        )
    total_frames = total_angular_positions * frames_per_angle

    if acquisition_mode == ACQUISITION_MODE_FULL:
        geometry_limited_total_angular_positions = (
            2 * geometry_limited_recommended_projections - 1
        )
    elif acquisition_mode == ACQUISITION_MODE_SPARSE_TILT:
        geometry_limited_total_angular_positions = (
            geometry_limited_recommended_projections + tilt_correction_projections
        )
    else:
        geometry_limited_total_angular_positions = geometry_limited_recommended_projections
    geometry_limited_total_frames = (
        geometry_limited_total_angular_positions * frames_per_angle
    )

    ld_ratio = float(reference_l_over_d) / float(l_over_d)
    estimated_flux = float(reference_flux_n_cm2_s) * ld_ratio**2
    implied_pinhole_mm = float(reference_pinhole_mm) * ld_ratio

    pixel_size_cm = float(pixel_size_um) * 1.0e-4
    incident_per_frame = (
        estimated_flux
        * pixel_size_cm**2
        * float(exposure_time_s)
    )
    detected_per_projection = (
        incident_per_frame
        * float(transmission_fraction)
        * float(detector_efficiency_fraction)
        * frames_per_angle
    )
    projection_count_snr = math.sqrt(detected_per_projection)
    projection_relative_noise_percent = (
        100.0 / projection_count_snr if projection_count_snr > 0 else math.inf
    )
    base_set_count_snr = math.sqrt(detected_per_projection * recommended_projections)
    acquired_set_count_snr = math.sqrt(detected_per_projection * total_angular_positions)

    estimated_scan_time_s = total_frames * (float(exposure_time_s) + float(frame_overhead_s))
    geometry_limited_estimated_scan_time_s = geometry_limited_total_frames * (
        float(exposure_time_s) + float(frame_overhead_s)
    )

    return TomographyScanEstimate(
        object_diameter_mm=object_diameter_mm,
        sample_detector_distance_mm=sample_detector_distance_mm,
        geometric_blur_um=geometric_blur_um,
        geometry_limited_resolution_um=geometry_limited_resolution_um,
        geometry_limited_resolvable_elements=geometry_limited_resolvable_elements,
        geometry_limited_full_sampling_projections=(
            geometry_limited_full_sampling_projections
        ),
        geometry_limited_recommended_projections=(
            geometry_limited_recommended_projections
        ),
        geometry_limited_total_angular_positions=(
            geometry_limited_total_angular_positions
        ),
        geometry_limited_total_frames=geometry_limited_total_frames,
        geometry_limited_estimated_scan_time_s=(
            geometry_limited_estimated_scan_time_s
        ),
        resolvable_elements=resolvable_elements,
        full_sampling_projections=full_sampling_projections,
        recommended_projections=recommended_projections,
        angular_step_deg=angular_step_deg,
        acquisition_mode=acquisition_mode,
        tilt_correction_projections=tilt_correction_projections,
        tilt_correction_step_deg=tilt_correction_step_deg,
        tilt_correction_angles_deg=tilt_correction_angles_deg,
        total_angular_positions=total_angular_positions,
        total_frames=total_frames,
        estimated_flux_n_cm2_s=estimated_flux,
        implied_pinhole_mm=implied_pinhole_mm,
        incident_neutrons_per_pixel_frame=incident_per_frame,
        detected_neutrons_per_pixel_projection=detected_per_projection,
        projection_count_snr=projection_count_snr,
        projection_relative_noise_percent=projection_relative_noise_percent,
        base_set_count_snr=base_set_count_snr,
        acquired_set_count_snr=acquired_set_count_snr,
        estimated_scan_time_s=estimated_scan_time_s,
    )


def tomography_plan_item_from_estimate(
    estimate: TomographyScanEstimate,
    *,
    exposure_time_s: float,
    frames_per_angle: int,
) -> dict:
    """Build the Queue Server ``tomo_scan`` item for an estimate."""

    return {
        "item_type": "plan",
        "name": "tomo_scan",
        "kwargs": {
            "exposure_time": float(exposure_time_s),
            "num_projections": int(estimate.recommended_projections),
            "start_angle": 0.0,
            "stop_angle": 180.0,
            "num_exposures": int(frames_per_angle),
            "include_stop_angle": True,
            "tilt_correction_projections": (
                int(estimate.tilt_correction_projections)
                if estimate.acquisition_mode == ACQUISITION_MODE_SPARSE_TILT
                else 0
            ),
            "full_360_scan": estimate.acquisition_mode == ACQUISITION_MODE_FULL,
        },
    }


def format_duration(seconds: float) -> str:
    """Format a non-negative duration without hiding useful precision."""

    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d} h {minutes:02d} min {secs:02d} s"
    return f"{minutes:d} min {secs:02d} s"


class TomographyScanCalculator(QtWidgets.QWidget):
    """Interactive calculator embedded in the tomography display."""

    object_diameter_mm_changed = QtCore.Signal(float)
    plan_editor_requested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TomographyScanCalculator")
        self._result_labels = {}
        self._estimated_scan_time_s = None
        self._last_estimate = None
        self._queue_server_connected = False
        self._build_ui()
        self._connect_inputs()
        self.update_estimate()
        self._completion_timer = QtCore.QTimer(self)
        self._completion_timer.setInterval(30000)
        self._completion_timer.timeout.connect(self._update_completion_time)
        self._completion_timer.start()

    @staticmethod
    def _double_spin(
        value,
        minimum,
        maximum,
        *,
        decimals=3,
        suffix="",
        step=None,
        tooltip="",
    ):
        spin = CompactDoubleSpinBox()
        spin.setDecimals(int(decimals))
        spin.setRange(float(minimum), float(maximum))
        spin.setValue(float(value))
        spin.setSuffix(str(suffix))
        if step is not None:
            spin.setSingleStep(float(step))
        spin.setKeyboardTracking(False)
        spin.setToolTip(str(tooltip))
        return spin

    @staticmethod
    def _configure_form(form):
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(8)

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        root.addWidget(scroll)

        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(18, 14, 18, 14)
        content_layout.setSpacing(12)

        title = QtWidgets.QLabel("Tomography Scan Calculator", content)
        title_font = title.font()
        title_font.setPointSize(max(14, title_font.pointSize() + 3))
        title_font.setBold(True)
        title.setFont(title_font)
        content_layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Estimate angular sampling with the parallel-beam Crowther criterion, then "
            "evaluate exposure, neutron counts, and approximate acquisition time.",
            content,
        )
        subtitle.setWordWrap(True)
        content_layout.addWidget(subtitle)

        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(14)
        content_layout.addLayout(columns)

        inputs_column = QtWidgets.QVBoxLayout()
        inputs_column.setSpacing(12)
        columns.addLayout(inputs_column, 1)

        sampling_group = QtWidgets.QGroupBox("Angular sampling", content)
        sampling_form = QtWidgets.QFormLayout(sampling_group)
        self._configure_form(sampling_form)

        self.object_diameter_mm_spin = self._double_spin(
            DEFAULT_OBJECT_DIAMETER_MM,
            0.001,
            10000.0,
            decimals=2,
            suffix=" mm",
            step=1.0,
            tooltip=(
                "Physical object width. Enter it directly, or use the Imaging Viewer "
                "measurement line to convert a pixel length using the current pixel size."
            ),
        )
        sampling_form.addRow("Object diameter:", self.object_diameter_mm_spin)

        self.spatial_resolution_spin = self._double_spin(
            DEFAULT_SPATIAL_RESOLUTION_UM,
            MIN_TARGET_RESOLUTION_UM,
            10000.0,
            decimals=4,
            suffix=" µm",
            step=1.0,
            tooltip=(
                "Resolution used by the Crowther calculation. The default is twice "
                "the pixel size, with a fixed 20 µm lower limit. Edit this to evaluate "
                "other configurations."
            ),
        )
        sampling_form.addRow("Target spatial resolution:", self.spatial_resolution_spin)

        self.pixel_size_spin = self._double_spin(
            DEFAULT_PIXEL_SIZE_UM,
            0.1,
            10000.0,
            decimals=4,
            suffix=" µm",
            step=1.0,
            tooltip=(
                "Effective pixel pitch at the sample. This controls physical measurement "
                "conversion and estimated neutron counts per pixel."
            ),
        )
        sampling_form.addRow("Pixel size at sample:", self.pixel_size_spin)

        self.sampling_percent_spin = self._double_spin(
            80.0,
            1.0,
            100.0,
            decimals=0,
            suffix=" %",
            step=5.0,
            tooltip="Fraction of the full Crowther angular-sampling requirement.",
        )
        sampling_form.addRow("Sampling target:", self.sampling_percent_spin)

        self.angular_coverage_combo = QtWidgets.QComboBox(sampling_group)
        self.angular_coverage_combo.addItem("0–180° only", ACQUISITION_MODE_HALF)
        self.angular_coverage_combo.addItem(
            "Sparse 180–360° tilt views",
            ACQUISITION_MODE_SPARSE_TILT,
        )
        self.angular_coverage_combo.addItem("Full 0–360° scan", ACQUISITION_MODE_FULL)
        self.angular_coverage_combo.setToolTip(
            "Use a normal 0–180° scan, add a sparse second-half tilt-correction set, "
            "or repeat the full angular density from 180–360°."
        )
        sampling_form.addRow("Angular coverage:", self.angular_coverage_combo)

        self.tilt_projection_count_spin = QtWidgets.QSpinBox(sampling_group)
        self.tilt_projection_count_spin.setRange(2, 1000)
        self.tilt_projection_count_spin.setValue(DEFAULT_TILT_CORRECTION_PROJECTIONS)
        self.tilt_projection_count_spin.setSuffix(" views")
        self.tilt_projection_count_spin.setEnabled(False)
        self.tilt_projection_count_spin.setToolTip(
            "Number of approximately evenly spaced second-half views. Each is selected "
            "from a base projection angle + 180°, and the final view is always 360°."
        )
        sampling_form.addRow("Sparse extra projections:", self.tilt_projection_count_spin)
        inputs_column.addWidget(sampling_group)

        acquisition_group = QtWidgets.QGroupBox("Acquisition and beam assumptions", content)
        acquisition_form = QtWidgets.QFormLayout(acquisition_group)
        self._configure_form(acquisition_form)

        self.exposure_spin = self._double_spin(
            DEFAULT_EXPOSURE_TIME_S,
            0.001,
            86400.0,
            decimals=3,
            suffix=" s",
            step=0.1,
        )
        acquisition_form.addRow("Exposure / frame:", self.exposure_spin)

        self.frames_per_angle_spin = QtWidgets.QSpinBox(acquisition_group)
        self.frames_per_angle_spin.setRange(1, 1000)
        self.frames_per_angle_spin.setValue(1)
        self.frames_per_angle_spin.setSuffix(" frame")
        acquisition_form.addRow("Frames / angle:", self.frames_per_angle_spin)

        self.l_over_d_spin = self._double_spin(
            REFERENCE_L_OVER_D,
            1.0,
            100000.0,
            decimals=1,
            step=10.0,
            tooltip="Beam collimation ratio. Reference measurement: L/D=554 with an 8 mm pinhole.",
        )
        acquisition_form.addRow("L/D:", self.l_over_d_spin)

        self.sample_detector_distance_spin = self._double_spin(
            DEFAULT_OBJECT_DIAMETER_MM / 2.0,
            0.0,
            10000.0,
            decimals=2,
            suffix=" mm",
            step=1.0,
            tooltip=(
                "Distance from the rotation axis (sample center) to the scintillator. "
                "The radius estimate assumes the nearest sample surface is at the detector."
            ),
        )
        self.estimate_sdd_from_radius_check = QtWidgets.QCheckBox(
            "Use object radius",
            acquisition_group,
        )
        self.estimate_sdd_from_radius_check.setChecked(True)
        self.estimate_sdd_from_radius_check.setToolTip(
            "Keep the sample-center distance equal to half the object diameter."
        )
        self.sample_detector_distance_spin.setEnabled(False)
        distance_row = QtWidgets.QHBoxLayout()
        distance_row.setContentsMargins(0, 0, 0, 0)
        distance_row.setSpacing(8)
        distance_row.addWidget(self.sample_detector_distance_spin, 1)
        distance_row.addWidget(self.estimate_sdd_from_radius_check)
        acquisition_form.addRow("Sample center–detector:", distance_row)

        self.reference_flux_spin = self._double_spin(
            REFERENCE_FLUX_N_CM2_S,
            1.0,
            1.0e12,
            decimals=0,
            suffix=" n/cm²/s",
            step=1.0e5,
            tooltip="Measured open-beam flux at the L/D=554, 8 mm pinhole reference condition.",
        )
        acquisition_form.addRow("Reference flux @ L/D 554:", self.reference_flux_spin)

        self.transmission_spin = self._double_spin(
            100.0,
            0.0,
            100.0,
            decimals=1,
            suffix=" %",
            step=5.0,
            tooltip="Estimated neutron transmission through the sample at the pixel of interest.",
        )
        acquisition_form.addRow("Sample transmission:", self.transmission_spin)

        self.efficiency_spin = self._double_spin(
            DEFAULT_DETECTION_EFFICIENCY_PERCENT,
            0.0,
            100.0,
            decimals=1,
            suffix=" %",
            step=5.0,
            tooltip=(
                "Approximate thermal-neutron capture efficiency. The 80% default is based "
                "on reported stopping power for a roughly 20 µm natural-Gadox screen; it "
                "is not a measured whole-camera DQE."
            ),
        )
        acquisition_form.addRow("Neutron capture efficiency:", self.efficiency_spin)

        self.frame_overhead_spin = self._double_spin(
            DEFAULT_FRAME_OVERHEAD_S,
            0.0,
            3600.0,
            decimals=3,
            suffix=" s",
            step=0.1,
            tooltip="Current full-frame transfer/write estimate for the 4552×4552 UInt16 image.",
        )
        acquisition_form.addRow("Frame write overhead:", self.frame_overhead_spin)

        inputs_column.addWidget(acquisition_group)
        inputs_column.addStretch(1)

        results_group = QtWidgets.QGroupBox("Recommendation", content)
        results_layout = QtWidgets.QFormLayout(results_group)
        self._configure_form(results_layout)
        columns.addWidget(results_group, 1)

        result_rows = (
            ("object_diameter", "Measured object diameter:"),
            ("geometric_blur", "Geometric blur at sample center:"),
            ("geometry_limited_resolution", "Estimated geometry-limited resolution:"),
            ("resolvable_elements", "Target resolvable elements across object:"),
            ("full_projections", "Target full Crowther requirement:"),
            ("recommended_projections", "Target 0–180° projections:"),
            ("geometry_limited_projections", "Geometry-limited 0–180° projections:"),
            ("angular_step", "0–180° angular step:"),
            ("acquisition_mode", "Angular coverage:"),
            ("tilt_projections", "Additional 180–360° projections:"),
            ("tilt_step", "Nominal additional spacing:"),
            ("tilt_pairing", "180° pair alignment:"),
            ("total_positions", "Total angular positions:"),
            ("total_frames", "Total acquired frames:"),
            ("implied_pinhole", "Implied pinhole at same L:"),
            ("flux", "Estimated incident flux:"),
            ("incident_counts", "Incident n / pixel / frame:"),
            ("detected_counts", "Detected n / pixel / projection:"),
            ("projection_snr", "Per-projection count SNR:"),
            ("projection_noise", "Per-projection relative noise:"),
            ("base_set_snr", "Base-set aggregate count SNR:"),
            ("acquired_set_snr", "All-acquired aggregate count SNR:"),
            ("scan_time", "Target acquisition time:"),
            ("geometry_limited_scan_time", "Geometry-limited acquisition time:"),
            ("completion_time", "Target completion if started now:"),
        )
        for key, label_text in result_rows:
            value_label = QtWidgets.QLabel("--", results_group)
            value_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            value_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            value_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
            value_label.setFont(value_font)
            results_layout.addRow(label_text, value_label)
            self._result_labels[key] = value_label

        suggested_group = QtWidgets.QGroupBox("Suggested scan parameters", content)
        suggested_layout = QtWidgets.QHBoxLayout(suggested_group)
        self.suggested_values_edit = QtWidgets.QLineEdit(suggested_group)
        self.suggested_values_edit.setReadOnly(True)
        self.copy_button = QtWidgets.QPushButton("Copy", suggested_group)
        self.copy_button.clicked.connect(self._copy_suggested_values)
        suggested_layout.addWidget(self.suggested_values_edit, 1)
        suggested_layout.addWidget(self.copy_button)
        content_layout.addWidget(suggested_group)

        self.use_in_plan_editor_button = QtWidgets.QPushButton(
            "Use Recommendation in Plan Editor",
            content,
        )
        self.use_in_plan_editor_button.setEnabled(False)
        self.use_in_plan_editor_button.setToolTip(
            "Connect to the Queue Server to load these values into the tomo_scan editor."
        )
        self.use_in_plan_editor_button.clicked.connect(
            self._request_plan_editor_update
        )
        content_layout.addWidget(self.use_in_plan_editor_button)

        self.warning_label = QtWidgets.QLabel(content)
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("QLabel { color: #b35a00; font-weight: 600; }")
        content_layout.addWidget(self.warning_label)

        assumptions = QtWidgets.QLabel(
            "Model assumptions: Nθ = πD/(2Δ), where D/Δ is the number of resolvable "
            "elements across the measured object, scaled by the selected sampling target. "
            "Geometric blur is sample-center distance divided by L/D. The primary "
            "recommendation uses the selected target resolution; the separate geometry-limited "
            "screening estimate uses the larger of target resolution and geometric blur. A "
            "measured slanted-edge MTF should replace that screening estimate when available. "
            "The base set includes both 0° and 180°; sparse or full second-half sets "
            "exclude the duplicate 180° position, include 360°, and select every extra "
            "view from an acquired base angle shifted by exactly 180°. "
            "Flux is scaled from the entered reference at L/D=554 and an 8 mm pinhole as "
            "(L/D)⁻². Count SNR assumes independent Poisson statistics. Aggregate count "
            "SNR is not reconstructed-voxel SNR and excludes flat/dark uncertainty, "
            "camera/scintillator noise, artifacts, and reconstruction filtering. Time "
            "includes exposure and frame-write overhead, but excludes motor motion, setup, "
            "cooling, and final return motion.",
            content,
        )
        assumptions.setWordWrap(True)
        assumptions.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        assumptions.setStyleSheet("QLabel { color: palette(mid); }")
        content_layout.addWidget(assumptions)
        content_layout.addStretch(1)

    def _connect_inputs(self):
        for spin in (
            self.object_diameter_mm_spin,
            self.spatial_resolution_spin,
            self.pixel_size_spin,
            self.sampling_percent_spin,
            self.exposure_spin,
            self.frames_per_angle_spin,
            self.l_over_d_spin,
            self.sample_detector_distance_spin,
            self.reference_flux_spin,
            self.transmission_spin,
            self.efficiency_spin,
            self.frame_overhead_spin,
        ):
            spin.valueChanged.connect(self.update_estimate)
        self.angular_coverage_combo.currentIndexChanged.connect(
            self._on_angular_coverage_changed
        )
        self.tilt_projection_count_spin.valueChanged.connect(self.update_estimate)
        self.object_diameter_mm_spin.valueChanged.connect(
            self._on_object_diameter_changed
        )
        self.estimate_sdd_from_radius_check.toggled.connect(
            self._on_radius_distance_toggled
        )

    def _on_object_diameter_changed(self, diameter_mm):
        diameter_mm = float(diameter_mm)
        self.object_diameter_mm_changed.emit(diameter_mm)
        if self.estimate_sdd_from_radius_check.isChecked():
            self.sample_detector_distance_spin.setValue(diameter_mm / 2.0)

    def _on_radius_distance_toggled(self, checked):
        checked = bool(checked)
        self.sample_detector_distance_spin.setEnabled(not checked)
        if checked:
            self.sample_detector_distance_spin.setValue(
                self.object_diameter_mm_spin.value() / 2.0
            )
        self.update_estimate()

    def _on_angular_coverage_changed(self, *_args):
        is_sparse = self.angular_coverage_combo.currentData() == ACQUISITION_MODE_SPARSE_TILT
        self.tilt_projection_count_spin.setEnabled(is_sparse)
        self.update_estimate()

    def set_object_diameter_mm(self, diameter_mm):
        """Populate the physical object diameter in millimeters."""

        try:
            diameter_mm = float(diameter_mm)
        except (TypeError, ValueError):
            return
        if not math.isfinite(diameter_mm) or diameter_mm <= 0:
            return
        bounded = min(
            self.object_diameter_mm_spin.maximum(),
            max(self.object_diameter_mm_spin.minimum(), diameter_mm),
        )
        self.object_diameter_mm_spin.setValue(bounded)

    def _copy_suggested_values(self):
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self.suggested_values_edit.text())

    def set_queue_server_connected(self, connected):
        """Enable plan transfer only while the Queue Server is connected."""

        self._queue_server_connected = bool(connected)
        self._update_plan_editor_button_state()

    def _update_plan_editor_button_state(self):
        enabled = self._queue_server_connected and self._last_estimate is not None
        self.use_in_plan_editor_button.setEnabled(enabled)
        if self._queue_server_connected:
            tooltip = (
                "Replace the tomography values in the Plan Editor with this recommendation."
                if self._last_estimate is not None
                else "Enter valid tomography inputs before loading the Plan Editor."
            )
        else:
            tooltip = (
                "Connect to the Queue Server to load these values into the tomo_scan editor."
            )
        self.use_in_plan_editor_button.setToolTip(tooltip)

    def recommended_plan_item(self):
        """Return a Queue Server tomo_scan item for the current recommendation."""

        estimate = self._last_estimate
        if estimate is None:
            return None
        return tomography_plan_item_from_estimate(
            estimate,
            exposure_time_s=self.exposure_spin.value(),
            frames_per_angle=self.frames_per_angle_spin.value(),
        )

    def _request_plan_editor_update(self):
        if not self._queue_server_connected:
            return
        item = self.recommended_plan_item()
        if item is not None:
            self.plan_editor_requested.emit(item)

    def _set_result(self, key, text):
        label = self._result_labels.get(key)
        if label is not None:
            label.setText(str(text))

    def _update_completion_time(self):
        if self._estimated_scan_time_s is None:
            self._set_result("completion_time", "--")
            return
        completion = datetime.now().astimezone() + timedelta(
            seconds=float(self._estimated_scan_time_s)
        )
        self._set_result(
            "completion_time",
            completion.strftime("%Y-%m-%d %I:%M:%S %p %Z"),
        )

    @QtCore.Slot()
    def update_estimate(self, *_args):
        acquisition_mode = self.angular_coverage_combo.currentData()
        try:
            estimate = estimate_tomography_scan(
                object_diameter_mm=self.object_diameter_mm_spin.value(),
                spatial_resolution_um=self.spatial_resolution_spin.value(),
                sampling_fraction=self.sampling_percent_spin.value() / 100.0,
                exposure_time_s=self.exposure_spin.value(),
                pixel_size_um=self.pixel_size_spin.value(),
                l_over_d=self.l_over_d_spin.value(),
                sample_detector_distance_mm=self.sample_detector_distance_spin.value(),
                reference_flux_n_cm2_s=self.reference_flux_spin.value(),
                transmission_fraction=self.transmission_spin.value() / 100.0,
                detector_efficiency_fraction=self.efficiency_spin.value() / 100.0,
                frames_per_angle=self.frames_per_angle_spin.value(),
                tilt_correction_projections=(
                    self.tilt_projection_count_spin.value()
                    if acquisition_mode == ACQUISITION_MODE_SPARSE_TILT
                    else 0
                ),
                full_360_scan=acquisition_mode == ACQUISITION_MODE_FULL,
                frame_overhead_s=self.frame_overhead_spin.value(),
            )
        except ValueError as exc:
            self._estimated_scan_time_s = None
            self._last_estimate = None
            self._update_plan_editor_button_state()
            for key in self._result_labels:
                self._set_result(key, "--")
            self.warning_label.setText(str(exc))
            self.suggested_values_edit.clear()
            return

        self._last_estimate = estimate
        self._update_plan_editor_button_state()

        self._set_result("object_diameter", f"{estimate.object_diameter_mm:.2f} mm")
        self._set_result("geometric_blur", f"{estimate.geometric_blur_um:.2f} µm")
        self._set_result(
            "geometry_limited_resolution",
            f"{estimate.geometry_limited_resolution_um:.2f} µm",
        )
        self._set_result("resolvable_elements", f"{estimate.resolvable_elements:,.1f}")
        self._set_result("full_projections", f"{estimate.full_sampling_projections:,d}")
        self._set_result("recommended_projections", f"{estimate.recommended_projections:,d}")
        self._set_result(
            "geometry_limited_projections",
            f"{estimate.geometry_limited_recommended_projections:,d}",
        )
        self._set_result("angular_step", f"{estimate.angular_step_deg:.6g}°")
        mode_labels = {
            ACQUISITION_MODE_HALF: "0–180° only",
            ACQUISITION_MODE_SPARSE_TILT: "0–180° + sparse tilt views",
            ACQUISITION_MODE_FULL: "Full 0–360°",
        }
        self._set_result(
            "acquisition_mode",
            mode_labels.get(estimate.acquisition_mode, estimate.acquisition_mode),
        )
        self._set_result("tilt_projections", f"{estimate.tilt_correction_projections:,d}")
        self._set_result(
            "tilt_step",
            (
                f"{estimate.tilt_correction_step_deg:.6g}°"
                if estimate.tilt_correction_step_deg is not None
                else "--"
            ),
        )
        self._set_result(
            "tilt_pairing",
            (
                "Exact base-angle pairs; includes 360°"
                if estimate.tilt_correction_angles_deg
                else "--"
            ),
        )
        self._set_result("total_positions", f"{estimate.total_angular_positions:,d}")
        self._set_result("total_frames", f"{estimate.total_frames:,d}")
        self._set_result("implied_pinhole", f"{estimate.implied_pinhole_mm:.3g} mm")
        self._set_result("flux", f"{estimate.estimated_flux_n_cm2_s:.3g} n/cm²/s")
        self._set_result(
            "incident_counts",
            f"{estimate.incident_neutrons_per_pixel_frame:.3g}",
        )
        self._set_result(
            "detected_counts",
            f"{estimate.detected_neutrons_per_pixel_projection:.3g}",
        )
        self._set_result("projection_snr", f"{estimate.projection_count_snr:.3g}")
        relative_noise = estimate.projection_relative_noise_percent
        self._set_result(
            "projection_noise",
            f"{relative_noise:.3g} %" if math.isfinite(relative_noise) else "∞",
        )
        self._set_result("base_set_snr", f"{estimate.base_set_count_snr:.3g}")
        self._set_result("acquired_set_snr", f"{estimate.acquired_set_count_snr:.3g}")
        self._set_result("scan_time", format_duration(estimate.estimated_scan_time_s))
        self._set_result(
            "geometry_limited_scan_time",
            format_duration(estimate.geometry_limited_estimated_scan_time_s),
        )
        self._estimated_scan_time_s = estimate.estimated_scan_time_s
        self._update_completion_time()

        if estimate.acquisition_mode == ACQUISITION_MODE_FULL:
            suggested = (
                f"num_projections={estimate.recommended_projections}, "
                f"exposure_time={self.exposure_spin.value():g}, "
                "start_angle=0, stop_angle=180, include_stop_angle=True, "
                "full_360_scan=True"
            )
        elif estimate.acquisition_mode == ACQUISITION_MODE_SPARSE_TILT:
            suggested = (
                f"num_projections={estimate.recommended_projections}, "
                f"exposure_time={self.exposure_spin.value():g}, "
                f"tilt_correction_projections={estimate.tilt_correction_projections}, "
                "start_angle=0, stop_angle=180, include_stop_angle=True"
            )
        else:
            suggested = (
                f"num_projections={estimate.recommended_projections}, "
                f"exposure_time={self.exposure_spin.value():g}, "
                "start_angle=0, stop_angle=180, include_stop_angle=True"
            )
        if self.frames_per_angle_spin.value() != 1:
            suggested += f", num_exposures={self.frames_per_angle_spin.value()}"
        self.suggested_values_edit.setText(suggested)

        warnings = []
        if estimate.geometric_blur_um > self.spatial_resolution_spin.value():
            warnings.append(
                "The selected target is finer than the geometric-blur estimate at the "
                "sample center. Keep the target-based plan until a measured MTF supports "
                "using the lower geometry-limited projection count."
            )
        if self.spatial_resolution_spin.value() < 2.0 * self.pixel_size_spin.value():
            warnings.append(
                "Target resolution is finer than two pixels; this exploratory setting "
                "may overstate the number of resolvable elements."
            )
        if self.efficiency_spin.value() >= 99.999:
            warnings.append("Detection efficiency is 100%, so the displayed SNR is an ideal upper bound.")
        if self.transmission_spin.value() >= 99.999:
            warnings.append("Sample transmission is 100%; attenuating regions will have lower SNR.")
        self.warning_label.setText("  ".join(warnings))
