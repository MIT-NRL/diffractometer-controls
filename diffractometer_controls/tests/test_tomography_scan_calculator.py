import math
import unittest

from diffractometer_controls.tomography_scan_calculator import (
    ACQUISITION_MODE_FULL,
    ACQUISITION_MODE_HALF,
    ACQUISITION_MODE_SPARSE_TILT,
    DEFAULT_FRAME_OVERHEAD_S,
    REFERENCE_FLUX_N_CM2_S,
    nominal_spatial_resolution_um,
    paired_second_half_angles_deg,
    estimate_tomography_scan,
    format_duration,
    tomography_plan_item_from_estimate,
)
from diffractometer_controls.plan_time_estimation import (
    estimate_plan_runtime,
    format_plan_summary,
    format_tomography_plan_summary,
)
from diffractometer_controls.tomography_scan_parameters import (
    extend_tomography_angles_deg,
)


class TomographyScanEstimateTests(unittest.TestCase):
    def _estimate(self, **overrides):
        inputs = {
            "object_diameter_mm": 89.57198,
            "spatial_resolution_um": 40.0,
            "sampling_fraction": 0.8,
            "exposure_time_s": 1.0,
            "pixel_size_um": 19.6775,
            "l_over_d": 554.0,
            "transmission_fraction": 1.0,
            "detector_efficiency_fraction": 1.0,
            "frames_per_angle": 1,
            "tilt_correction_projections": 0,
            "frame_overhead_s": DEFAULT_FRAME_OVERHEAD_S,
        }
        inputs.update(overrides)
        return estimate_tomography_scan(**inputs)

    def test_full_field_80_percent_sampling(self):
        estimate = self._estimate()

        self.assertAlmostEqual(estimate.object_diameter_mm, 89.57198)
        self.assertAlmostEqual(estimate.resolvable_elements, 2239.2995)
        self.assertEqual(estimate.full_sampling_projections, 3519)
        self.assertEqual(estimate.recommended_projections, 2815)
        self.assertAlmostEqual(estimate.angular_step_deg, 180.0 / 2814.0)
        self.assertAlmostEqual(
            (estimate.recommended_projections - 1) * estimate.angular_step_deg,
            180.0,
        )
        self.assertEqual(estimate.total_angular_positions, 2815)
        self.assertEqual(estimate.total_frames, 2815)
        self.assertEqual(estimate.acquisition_mode, ACQUISITION_MODE_HALF)

    def test_nominal_resolution_uses_two_pixels_with_fixed_floor(self):
        self.assertAlmostEqual(nominal_spatial_resolution_um(19.6775), 39.355)
        self.assertAlmostEqual(nominal_spatial_resolution_um(5.0), 20.0)
        self.assertAlmostEqual(nominal_spatial_resolution_um(30.0), 60.0)

    def test_minimum_sparse_pair_set_is_270_and_360_on_compatible_grid(self):
        self.assertEqual(paired_second_half_angles_deg(5, 2), (270.0, 360.0))

    def test_physical_object_diameter_is_independent_of_pixel_size(self):
        fine_pixels = self._estimate(pixel_size_um=10.0)
        coarse_pixels = self._estimate(pixel_size_um=30.0)

        self.assertEqual(fine_pixels.object_diameter_mm, coarse_pixels.object_diameter_mm)
        self.assertEqual(fine_pixels.resolvable_elements, coarse_pixels.resolvable_elements)
        self.assertEqual(
            fine_pixels.recommended_projections,
            coarse_pixels.recommended_projections,
        )

    def test_geometry_limit_is_reported_without_changing_target_recommendation(self):
        target_only = self._estimate(spatial_resolution_um=40.0)
        estimate = self._estimate(
            spatial_resolution_um=40.0,
            sample_detector_distance_mm=50.0,
            l_over_d=500.0,
        )

        self.assertAlmostEqual(estimate.geometric_blur_um, 100.0)
        self.assertAlmostEqual(estimate.geometry_limited_resolution_um, 100.0)
        self.assertEqual(estimate.recommended_projections, target_only.recommended_projections)
        self.assertAlmostEqual(
            estimate.geometry_limited_resolvable_elements,
            estimate.object_diameter_mm * 1000.0 / 100.0,
        )
        geometry_intervals = math.ceil(
            (math.pi / 2.0) * estimate.geometry_limited_resolvable_elements * 0.8
        )
        self.assertEqual(
            estimate.geometry_limited_recommended_projections,
            geometry_intervals + 1,
        )
        self.assertLess(
            estimate.geometry_limited_recommended_projections,
            estimate.recommended_projections,
        )

    def test_geometry_limit_never_claims_finer_resolution_than_target(self):
        estimate = self._estimate(
            spatial_resolution_um=40.0,
            sample_detector_distance_mm=10.0,
            l_over_d=1000.0,
        )

        self.assertAlmostEqual(estimate.geometric_blur_um, 10.0)
        self.assertAlmostEqual(estimate.geometry_limited_resolution_um, 40.0)
        self.assertEqual(
            estimate.geometry_limited_recommended_projections,
            estimate.recommended_projections,
        )

    def test_flux_scales_with_inverse_square_of_l_over_d(self):
        at_reference = self._estimate(l_over_d=554.0)
        at_double_ld = self._estimate(l_over_d=1108.0)

        self.assertAlmostEqual(at_reference.estimated_flux_n_cm2_s, REFERENCE_FLUX_N_CM2_S)
        self.assertAlmostEqual(
            at_double_ld.estimated_flux_n_cm2_s,
            REFERENCE_FLUX_N_CM2_S / 4.0,
        )
        self.assertAlmostEqual(at_double_ld.implied_pinhole_mm, 4.0)

    def test_count_snr_includes_transmission_efficiency_and_frame_average(self):
        estimate = self._estimate(
            transmission_fraction=0.5,
            detector_efficiency_fraction=0.4,
            frames_per_angle=4,
        )

        incident = estimate.incident_neutrons_per_pixel_frame
        expected_detected = incident * 0.5 * 0.4 * 4
        self.assertAlmostEqual(estimate.detected_neutrons_per_pixel_projection, expected_detected)
        self.assertAlmostEqual(estimate.projection_count_snr, math.sqrt(expected_detected))
        self.assertAlmostEqual(
            estimate.base_set_count_snr,
            math.sqrt(expected_detected * estimate.recommended_projections),
        )

    def test_tilt_correction_adds_sparse_second_half(self):
        base = self._estimate()
        tilt = self._estimate(tilt_correction_projections=30)

        self.assertEqual(tilt.recommended_projections, base.recommended_projections)
        self.assertEqual(tilt.acquisition_mode, ACQUISITION_MODE_SPARSE_TILT)
        self.assertEqual(tilt.tilt_correction_projections, 30)
        self.assertAlmostEqual(tilt.tilt_correction_step_deg, 6.0)
        self.assertEqual(len(tilt.tilt_correction_angles_deg), 30)
        self.assertAlmostEqual(tilt.tilt_correction_angles_deg[-1], 360.0)
        for angle in tilt.tilt_correction_angles_deg:
            paired_base_index = (angle - 180.0) / tilt.angular_step_deg
            self.assertAlmostEqual(paired_base_index, round(paired_base_index))
        self.assertEqual(tilt.total_angular_positions, base.total_angular_positions + 30)
        self.assertEqual(tilt.total_frames, base.total_frames + 30)
        self.assertEqual(
            tilt.geometry_limited_total_angular_positions,
            tilt.geometry_limited_recommended_projections + 30,
        )
        expected_snr_ratio = math.sqrt(tilt.total_angular_positions / base.total_angular_positions)
        self.assertAlmostEqual(
            tilt.acquired_set_count_snr,
            expected_snr_ratio * base.acquired_set_count_snr,
        )
        expected_time = 30 * (1.0 + DEFAULT_FRAME_OVERHEAD_S)
        self.assertAlmostEqual(tilt.estimated_scan_time_s, base.estimated_scan_time_s + expected_time)

        item = tomography_plan_item_from_estimate(
            tilt,
            exposure_time_s=60.0,
            frames_per_angle=2,
        )
        self.assertEqual(item["name"], "tomo_scan")
        self.assertEqual(item["kwargs"]["num_projections"], tilt.recommended_projections)
        self.assertEqual(item["kwargs"]["tilt_correction_projections"], 30)
        self.assertFalse(item["kwargs"]["full_360_scan"])
        self.assertEqual(item["kwargs"]["stop_angle"], 180.0)
        self.assertTrue(item["kwargs"]["include_stop_angle"])

    def test_full_360_scan_repeats_base_density_in_second_half(self):
        base = self._estimate()
        full = self._estimate(full_360_scan=True, tilt_correction_projections=20)

        self.assertEqual(full.acquisition_mode, ACQUISITION_MODE_FULL)
        self.assertEqual(full.tilt_correction_projections, base.recommended_projections - 1)
        self.assertEqual(full.total_angular_positions, 2 * base.recommended_projections - 1)
        self.assertEqual(
            full.geometry_limited_total_angular_positions,
            2 * full.geometry_limited_recommended_projections - 1,
        )
        self.assertAlmostEqual(full.tilt_correction_step_deg, base.angular_step_deg)
        self.assertAlmostEqual(
            180.0 + full.tilt_correction_projections * full.tilt_correction_step_deg,
            360.0,
        )
        self.assertAlmostEqual(full.tilt_correction_angles_deg[-1], 360.0)
        expected_ratio = full.total_frames / base.total_frames
        self.assertAlmostEqual(
            full.estimated_scan_time_s,
            expected_ratio * base.estimated_scan_time_s,
        )
        item = tomography_plan_item_from_estimate(
            full,
            exposure_time_s=60.0,
            frames_per_angle=1,
        )
        self.assertEqual(item["kwargs"]["num_projections"], base.recommended_projections)
        self.assertEqual(item["kwargs"]["tilt_correction_projections"], 0)
        self.assertTrue(item["kwargs"]["full_360_scan"])

    def test_executed_sparse_angles_match_calculator_pairing(self):
        base_angles = tuple(index * 45.0 for index in range(5))
        extended = extend_tomography_angles_deg(
            base_angles,
            tilt_correction_projections=2,
        )

        self.assertEqual(extended, (0.0, 45.0, 90.0, 135.0, 180.0, 270.0, 360.0))

    def test_executed_full_scan_preserves_base_spacing_and_one_180(self):
        extended = extend_tomography_angles_deg(
            (0.0, 60.0, 120.0, 180.0),
            full_360_scan=True,
        )

        self.assertEqual(
            extended,
            (0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
        )
        with self.assertRaises(ValueError):
            extend_tomography_angles_deg(
                (0.0, 60.0, 120.0, 180.0),
                tilt_correction_projections=2,
                full_360_scan=True,
            )

    def test_queue_runtime_estimate_counts_sparse_and_full_positions(self):
        context = {"image_bytes": 0.0, "transfer_time_per_bytes": 0.0}
        sparse = estimate_plan_runtime(
            "tomo_scan",
            kwargs={
                "num_projections": 100,
                "exposure_time": 2.0,
                "num_exposures": 3,
                "tilt_correction_projections": 20,
            },
            context=context,
        )
        full = estimate_plan_runtime(
            "tomo_scan",
            kwargs={
                "num_projections": 100,
                "exposure_time": 2.0,
                "num_exposures": 3,
                "full_360_scan": True,
            },
            context=context,
        )

        self.assertEqual(sparse["estimated_total_units"], 120 * 3)
        self.assertEqual(sparse["estimated_total_time_s"], 120 * 3 * 2.0)
        self.assertEqual(full["estimated_total_units"], 199 * 3)
        self.assertEqual(full["estimated_total_time_s"], 199 * 3 * 2.0)

    def test_tomography_summary_reports_resolved_sparse_scan(self):
        summary = format_tomography_plan_summary(
            {
                "num_projections": 2862,
                "start_angle": 0.0,
                "stop_angle": 180.0,
                "include_stop_angle": True,
                "tilt_correction_projections": 20,
                "full_360_scan": False,
                "exposure_time": 60.0,
                "num_exposures": 1,
            },
            estimated_time_s=177840.0,
        )

        self.assertIn("2,862 projections", summary)
        self.assertIn("step 0.0629151°", summary)
        self.assertIn("20 exact 180° pairs", summary)
        self.assertIn("0° ✓ included", summary)
        self.assertIn("180° ✓ included", summary)
        self.assertIn("360° ✓ included", summary)
        self.assertIn("2,882 angular positions", summary)
        self.assertIn("Estimated acquisition: 49 h 24 min 00 s", summary)

    def test_tomography_summary_flags_off_by_one_landmark_grid(self):
        common = {
            "start_angle": 0.0,
            "stop_angle": 360.0,
            "include_stop_angle": True,
            "exposure_time": 1.0,
        }
        missing = format_tomography_plan_summary(
            {**common, "num_projections": 960}
        )
        included = format_tomography_plan_summary(
            {**common, "num_projections": 961}
        )

        self.assertIn("0° ✓ included", missing)
        self.assertIn("180° ✗ missing", missing)
        self.assertIn("360° ✓ included", missing)
        self.assertIn("180° ✓ included", included)

    def test_tomography_summary_audits_custom_range_without_blocking_it(self):
        summary = format_tomography_plan_summary(
            {
                "start_angle": 0.0,
                "stop_angle": 230.0,
                "num_projections": 100,
                "include_stop_angle": True,
            }
        )

        self.assertIn("Coverage: base 0–230° only", summary)
        self.assertIn("0° ✓ included", summary)
        self.assertIn("180° ✗ missing", summary)
        self.assertIn("360° — not requested", summary)

    def test_general_plan_summary_resolves_imaging_acquisition(self):
        summary = format_plan_summary(
            "imaging",
            {
                "file_name": "flat",
                "file_dir": "proposal/sample",
                "exposure_time": 12.5,
                "num_exposures": 4,
                "gain": 120,
            },
            estimated_time_s=55.0,
        )

        self.assertIn("Imaging acquisition", summary)
        self.assertIn("4 frame(s)", summary)
        self.assertIn("Exposure: 12.5 s", summary)
        self.assertIn("Estimated acquisition: 55 s", summary)
        self.assertIn("proposal/sample/flat", summary)

    def test_general_plan_summary_resolves_imaging_motor_scan(self):
        summary = format_plan_summary(
            "imaging_scan",
            {
                "motor": "stage1.x",
                "start_pos": -2.0,
                "stop_pos": 2.0,
                "num_steps": 9,
                "exposure_time": 3.0,
                "num_exposures": 2,
            },
        )

        self.assertIn("Imaging motor scan", summary)
        self.assertIn("Range: -2 → 2 | step 0.5", summary)
        self.assertIn("9 inclusive position(s)", summary)
        self.assertIn("18 frame(s)", summary)
        self.assertIn("Exposure-only time: 54 s", summary)

    def test_imaging_acquire_time_scan_estimate_sums_scan_values(self):
        estimate = estimate_plan_runtime(
            "imaging_scan",
            kwargs={
                "detector": "cam1",
                "motor": "cam1_acquire_time",
                "start_pos": 1.0,
                "stop_pos": 5.0,
                "num_steps": 3,
                "num_exposures": 2,
            },
            context={
                # A different current camera value must not affect this scan.
                "imaging_exposure_time_s": 99.0,
                "image_bytes": 0.0,
                "transfer_time_per_bytes": 0.0,
            },
        )

        self.assertEqual(estimate["estimated_total_units"], 6)
        self.assertAlmostEqual(estimate["estimated_total_time_s"], 18.0)

    def test_imaging_acquire_time_scan_step_size_matches_scan_positions(self):
        estimate = estimate_plan_runtime(
            "imaging_scan",
            kwargs={
                "detector": "cam1",
                "motor": "cam1_acquire_time",
                "start_pos": 0.2,
                "stop_pos": 0.5,
                "step_size": 0.1,
                "num_exposures": 3,
            },
            context={"image_bytes": 0.0, "transfer_time_per_bytes": 0.0},
        )

        # 0.2 + 0.3 + 0.4 + 0.5 seconds at each of three exposures.
        self.assertEqual(estimate["estimated_total_units"], 12)
        self.assertAlmostEqual(estimate["estimated_total_time_s"], 4.2)

    def test_imaging_discrete_acquire_time_estimate_sums_exact_positions(self):
        estimate = estimate_plan_runtime(
            "imaging_scan_discrete",
            kwargs={
                "detector": "cam1",
                "motor": "cam1_acquire_time",
                "positions": [0.1, 0.5, 0.2],
                "num_exposures": 2,
            },
            context={"image_bytes": 0.0, "transfer_time_per_bytes": 0.0},
        )

        self.assertEqual(estimate["estimated_total_units"], 6)
        self.assertAlmostEqual(estimate["estimated_total_time_s"], 1.6)

    def test_imaging_relative_acquire_time_estimate_uses_run_start_values(self):
        queued_estimate = estimate_plan_runtime(
            "imaging_scan_rel",
            kwargs={
                "detector": "cam1",
                "motor": "cam1_acquire_time",
                "start_relative": -0.1,
                "stop_relative": 0.1,
                "num_steps": 3,
            },
            context={"image_bytes": 0.0, "transfer_time_per_bytes": 0.0},
        )
        worker_estimate = estimate_plan_runtime(
            "imaging_scan_rel",
            kwargs={
                "detector": "cam1",
                "motor": "cam1_acquire_time",
                "start_relative": -0.1,
                "stop_relative": 0.1,
                "num_steps": 3,
                "resolved_positions": [0.4, 0.5, 0.6],
            },
            context={"image_bytes": 0.0, "transfer_time_per_bytes": 0.0},
        )

        self.assertEqual(queued_estimate["estimated_total_units"], 3)
        self.assertIsNone(queued_estimate["estimated_total_time_s"])
        self.assertEqual(worker_estimate["estimated_total_units"], 3)
        self.assertAlmostEqual(worker_estimate["estimated_total_time_s"], 1.5)

    def test_imaging_acquire_time_summary_reports_the_exposure_sweep(self):
        summary = format_plan_summary(
            "imaging_scan",
            {
                "detector": "cam1",
                "motor": "cam1_acquire_time",
                "start_pos": 1.0,
                "stop_pos": 5.0,
                "num_steps": 3,
                "num_exposures": 2,
            },
        )

        self.assertIn("Imaging acquire-time scan", summary)
        self.assertIn("Exposure sweep: 1 → 5 s | average 3 s", summary)
        self.assertIn("Exposure-only time: 18 s", summary)

    def test_imaging_scan_variant_summaries_identify_their_position_forms(self):
        discrete_summary = format_plan_summary(
            "imaging_scan_discrete",
            {
                "motor": "stage1.x",
                "positions": [1.0, 2.5, 4.0],
                "exposure_time": 2.0,
            },
        )
        relative_summary = format_plan_summary(
            "imaging_scan_rel",
            {
                "motor": "stage1.x",
                "start_relative": -1.0,
                "stop_relative": 1.0,
                "num_steps": 3,
                "exposure_time": 2.0,
            },
        )

        self.assertIn("Imaging discrete motor scan", discrete_summary)
        self.assertIn("Positions (3): 1, 2.5, 4", discrete_summary)
        self.assertIn("Imaging relative motor scan", relative_summary)
        self.assertIn("Relative range: -1 → 1 | step 1", relative_summary)

    def test_general_plan_summary_resolves_adaptive_focus_initial_scan(self):
        summary = format_plan_summary(
            "adaptive_imaging_focus_scan",
            {
                "focus_guess": 5.0,
                "scan_half_range": 2.0,
                "num_steps": 9,
                "exposure_time": 1.5,
            },
        )

        self.assertIn("Adaptive imaging focus scan", summary)
        self.assertIn("Initial range: 3 → 7 | step 0.5", summary)
        self.assertIn("9 position(s)", summary)
        self.assertIn("adaptive requests may add frames", summary)

    def test_general_plan_summary_has_fallback_for_other_plans(self):
        summary = format_plan_summary(
            "scan_he3",
            {
                "motor": "stage1.x",
                "start_pos": 0.0,
                "stop_pos": 10.0,
                "step_size": 2.0,
                "acquire_time": 4.0,
            },
            estimated_time_s=24.0,
        )

        self.assertIn("Scan He3", summary)
        self.assertIn("Range: 0 → 10 | 6 position(s) | step 2", summary)
        self.assertIn("Estimated acquisition: 24 s", summary)

    def test_time_includes_exposure_and_frame_overhead_only(self):
        estimate = self._estimate(
            exposure_time_s=2.0,
            frame_overhead_s=1.5,
            frames_per_angle=2,
        )
        expected = estimate.total_frames * 3.5
        self.assertAlmostEqual(estimate.estimated_scan_time_s, expected)
        geometry_expected = estimate.geometry_limited_total_frames * 3.5
        self.assertAlmostEqual(
            estimate.geometry_limited_estimated_scan_time_s,
            geometry_expected,
        )

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            self._estimate(sampling_fraction=1.01)

    def test_single_sparse_projection_is_rejected(self):
        with self.assertRaises(ValueError):
            self._estimate(tilt_correction_projections=1)

    def test_duration_format(self):
        self.assertEqual(format_duration(59.25), "59.2 s")
        self.assertEqual(format_duration(61), "1 min 01 s")
        self.assertEqual(format_duration(3661), "1 h 01 min 01 s")


if __name__ == "__main__":
    unittest.main()
