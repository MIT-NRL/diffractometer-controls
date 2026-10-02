import unittest
from pathlib import Path
from xml.etree import ElementTree
from control_ui.widgets.count_rate_gauge import decade_scale, engineering_rate_text
from control_ui.panels.usbctr_common import MCS_CHANNEL_ADVANCE_INTERNAL, MCS_POINT_ZERO_SKIP, MCS_TRIGGER_LOW_LEVEL, channel_waveform_suffix, configure_spinbox, count_rate, mcs_average_rate, scaler_count_field


class _SpinboxStub:
    def setSingleStep(self, value):
        self.single_step = value


class CountRateCalculationTests(unittest.TestCase):
    def test_spinbox_configuration_supplies_missing_epics_limits(self):
        widget = _SpinboxStub()
        configure_spinbox(widget, 0.001, 100.0, single_step=0.1)
        self.assertEqual(widget.userMinimum, 0.001)
        self.assertEqual(widget.userMaximum, 100.0)
        self.assertTrue(widget.userDefinedLimits)
        self.assertTrue(widget.writeOnPress)
        self.assertFalse(widget.showStepExponent)
        self.assertEqual(widget.single_step, 0.1)

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

    def test_recommended_mcs_settings_use_epics_enum_selections(self):
        self.assertEqual(MCS_CHANNEL_ADVANCE_INTERNAL, 0)
        self.assertEqual(MCS_TRIGGER_LOW_LEVEL, 3)
        self.assertEqual(MCS_POINT_ZERO_SKIP, 2)


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
        extra_ui = Path(__file__).resolve().parents[1] / "control_ui" / "panels"
        ui_path = Path(__file__).resolve().parents[1] / "diffractometer_controls/site/mitr/4dh4All.ui"
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
                "../../../control_ui/panels/usbctr_count.py",
                "../../../control_ui/panels/usbctr_rate.py",
            ],
        )
        self.assertEqual(
            titles,
            [
                "USB-CTR08 module",
                "Count",
                "Count rate",
            ],
        )
        self.assertEqual(len(macros), len(filenames))
        self.assertIn("S=scaler1", macros[1])
        self.assertIn("MP=${P}USBCTR:MCS:", macros[2])
        self.assertTrue((extra_ui / "usbctr_count.py").is_file())
        self.assertTrue((extra_ui / "usbctr_rate.py").is_file())
        self.assertFalse((extra_ui / "usbctr_timed_counter.py").exists())
        self.assertFalse((extra_ui / "usbctr_mcs_rate.py").exists())


if __name__ == "__main__":
    unittest.main()
