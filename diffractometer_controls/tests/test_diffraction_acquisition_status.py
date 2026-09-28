import unittest

from diffractometer_controls.diffractometer_gui import (
    _acquisition_monitor_from_start_doc,
)


class DiffractionAcquisitionStatusTests(unittest.TestCase):
    def test_monitor_pvs_are_taken_from_start_document(self):
        monitor = {
            "device": "usbctr",
            "active_pv": "4dh4:USBCTR:scaler1.CNT",
            "duration_pv": "4dh4:USBCTR:scaler1.TP",
            "elapsed_pv": "4dh4:USBCTR:scaler1.T",
            "duration": 5.0,
        }
        self.assertEqual(
            _acquisition_monitor_from_start_doc(
                {
                    "experiment_type": "diffraction",
                    "detector_type": "usbctr08",
                    "acquisition_monitor": monitor,
                }
            ),
            monitor,
        )

    def test_passive_scalar_has_no_exposure_monitor(self):
        self.assertEqual(
            _acquisition_monitor_from_start_doc(
                {
                    "experiment_type": "diffraction",
                    "data_type": "scalar",
                    "detector_type": "scalar",
                }
            ),
            {},
        )

    def test_old_he3_run_keeps_compatibility_monitor(self):
        monitor = _acquisition_monitor_from_start_doc(
            {
                "experiment_type": "diffraction",
                "detector_type": "he3psd",
                "plan_args": {"acquire_time": 12.0},
            }
        )
        self.assertEqual(monitor["active_pv"], "4dh4:he3PSD:Acquire_RBV")
        self.assertEqual(
            monitor["remaining_pv"],
            "4dh4:he3PSD:AcquireTimeRemaining_RBV",
        )
        self.assertEqual(monitor["duration"], 12.0)


if __name__ == "__main__":
    unittest.main()
