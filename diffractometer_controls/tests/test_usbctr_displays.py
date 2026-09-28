import unittest
from pathlib import Path
from xml.etree import ElementTree
from diffractometer_controls.extra_ui.count_rate_gauge import (
    decade_scale,
    engineering_rate_text,
)
from diffractometer_controls.extra_ui.usbctr_common import (
    channel_waveform_suffix,
    count_rate,
    mcs_average_rate,
    scaler_count_field,
)


class CountRateCalculationTests(unittest.TestCase):
    def test_channel_mapping_uses_physical_zero_based_labels(self):
        self.assertEqual(channel_waveform_suffix(0), "mca1")
        self.assertEqual(channel_waveform_suffix(7), "mca8")
        self.assertEqual(scaler_count_field(0), "S1")
        self.assertEqual(scaler_count_field(7), "S8")

    def test_channel_mapping_rejects_invalid_channels(self):
        for channel in (-1, 8):
            with self.subTest(channel=channel):
                with self.assertRaises(ValueError):
                    channel_waveform_suffix(channel)

    def test_mcs_rate_uses_actual_dwell_and_newest_bins(self):
        self.assertEqual(mcs_average_rate([10, 20, 30], 0.5, 1), 60.0)
        self.assertEqual(mcs_average_rate([10, 20, 30], 0.5, 2), 50.0)

    def test_rate_calculations_handle_zero_time(self):
        self.assertEqual(count_rate(100, 0), 0.0)
        self.assertEqual(mcs_average_rate([100], 0), 0.0)

    def test_running_average_rate(self):
        self.assertEqual(count_rate(1250, 2.5), 500.0)


class CountRateGaugeFormattingTests(unittest.TestCase):
    def test_engineering_format(self):
        self.assertEqual(engineering_rate_text(7420), "7.42 kcps")
        self.assertEqual(engineering_rate_text(2.5e6), "2.50 Mcps")

    def test_decade_scale_hysteresis(self):
        self.assertEqual(decade_scale(7420), 1000.0)
        self.assertEqual(decade_scale(790, 1000), 100.0)
        self.assertEqual(decade_scale(810, 1000), 1000.0)
        self.assertEqual(decade_scale(10501, 1000), 10000.0)


class USBCTRMenuTests(unittest.TestCase):
    def test_all_controls_menu_lists_vendor_and_custom_displays(self):
        ui_path = (
            Path(__file__).resolve().parents[1] / "extra_ui" / "4dh4All.ui"
        )
        root = ElementTree.parse(ui_path).getroot()
        button = root.find(
            ".//widget[@name='PyDMRelatedDisplayButton_84']"
        )
        self.assertIsNotNone(button)

        filenames = [
            item.text
            for item in button.findall(
                "property[@name='filenames']/stringlist/string"
            )
        ]
        titles = [
            item.text
            for item in button.findall(
                "property[@name='titles']/stringlist/string"
            )
        ]
        macros = [
            item.text
            for item in button.findall(
                "property[@name='macros']/stringlist/string"
            )
        ]

        self.assertEqual(
            filenames,
            [
                "USBCTR.adl",
                "usbctr_mcs_rate.py",
                "usbctr_timed_counter.py",
            ],
        )
        self.assertEqual(
            titles,
            [
                "USB-CTR08 module",
                "MCS count-rate gauge",
                "Basic timed counter",
            ],
        )
        self.assertEqual(len(macros), len(filenames))
        self.assertIn("MP=${P}USBCTR:MCS:", macros[1])
        self.assertIn("S=scaler1", macros[2])


if __name__ == "__main__":
    unittest.main()
