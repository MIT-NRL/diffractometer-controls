import io
import unittest
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image
from p4p import Value
from p4p.nt import NTNDArray

from diffractometer_controls.extra_ui.reolink_display import (
    capability_is_supported,
    decode_ntndarray_payload,
    expand_macros,
    extract_ntndarray_payload,
    image_view_options,
    macro_dict,
    pva_pv_name,
)


class ReolinkViewerAddressTests(unittest.TestCase):
    def test_capability_enum_text_and_numbers_are_normalized(self):
        for value in (1, True, "1", "Supported", "enabled", "yes"):
            with self.subTest(value=value):
                self.assertTrue(capability_is_supported(value))
        for value in (0, False, "0", "Unsupported", "disabled", None):
            with self.subTest(value=value):
                self.assertFalse(capability_is_supported(value))

    def test_macros_expand_for_second_camera(self):
        macros = macro_dict("P=4dh4:,R=Reolink2:")
        self.assertEqual(
            expand_macros("${P}${R}Pva1:Image", macros),
            "4dh4:Reolink2:Pva1:Image",
        )

    def test_pydm_address_is_normalized_for_p4p(self):
        self.assertEqual(
            pva_pv_name("pva://4dh4:Reolink1:Pva1:Image"),
            "4dh4:Reolink1:Pva1:Image",
        )

    def test_4dh4all_launches_two_camera_viewer(self):
        ui_path = (
            Path(__file__).resolve().parents[1] / "extra_ui" / "4dh4All.ui"
        )
        root = ElementTree.parse(ui_path).getroot()
        button = root.find(
            ".//widget[@name='PyDMRelatedDisplayButton_34']"
        )

        self.assertIsNotNone(button)
        self.assertEqual(
            button.find("property[@name='text']/string").text,
            "Reolink Cameras",
        )
        self.assertEqual(
            button.find("property[@name='filenames']/stringlist/string").text,
            "reolink_cameras.py",
        )
        self.assertEqual(
            button.find("property[@name='macros']/stringlist/string").text,
            "P=${P}",
        )

    def test_viewer_exposes_epics_refocus_command(self):
        viewer_path = (
            Path(__file__).resolve().parents[1]
            / "extra_ui"
            / "reolink_display.py"
        )
        source = viewer_path.read_text()
        self.assertIn('self._button("Refocus", "Refocus", 1)', source)

    def test_toolbar_camera_shortcut_uses_reolink_viewer(self):
        main_window_path = (
            Path(__file__).resolve().parents[1] / "main_window.py"
        )
        source = main_window_path.read_text()
        launch_method = source.split(
            "    def launch_camera_viewer(self):", 1
        )[1].split("    def launch_focus_program(self):", 1)[0]
        self.assertIn('"extra_ui/reolink_cameras.py"', launch_method)
        self.assertNotIn('"extra_ui/cam_url.ui"', launch_method)


class ReolinkViewerDecodeTests(unittest.TestCase):
    def test_first_rgb_frame_initializes_pyqtgraph_levels(self):
        first = image_view_options(frame_ndim=3, initialized=False)
        later = image_view_options(frame_ndim=3, initialized=True)

        self.assertTrue(first["autoLevels"])
        self.assertTrue(first["autoRange"])
        self.assertTrue(first["autoHistogramRange"])
        self.assertEqual(first["axes"], {"x": 1, "y": 0, "c": 2})
        self.assertFalse(later["autoLevels"])

    def test_jpeg_ntndarray_decodes_to_rgb(self):
        source = np.zeros((12, 20, 3), dtype=np.uint8)
        source[:, :, 0] = 220
        encoded = io.BytesIO()
        Image.fromarray(source, "RGB").save(encoded, format="JPEG", quality=95)
        jpeg = np.frombuffer(encoded.getvalue(), dtype=np.uint8)

        value = Value(
            NTNDArray.buildType(),
            {
                "value": ("ubyteValue", jpeg),
                "codec": {"name": "jpeg", "parameters": 0},
                "compressedSize": jpeg.size,
                "uncompressedSize": source.size,
                "uniqueId": 42,
                "dimension": [
                    {"size": 3},
                    {"size": source.shape[1]},
                    {"size": source.shape[0]},
                ],
            },
        )

        payload = extract_ntndarray_payload(value)
        frame = decode_ntndarray_payload(payload)

        self.assertEqual(payload.unique_id, 42)
        self.assertEqual(payload.codec, "jpeg")
        self.assertEqual(frame.shape, source.shape)
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreater(float(frame[:, :, 0].mean()), 190.0)

    def test_non_jpeg_payload_is_rejected(self):
        value = Value(
            NTNDArray.buildType(),
            {
                "value": ("ubyteValue", np.arange(6, dtype=np.uint8)),
                "codec": {"name": "", "parameters": 0},
                "uniqueId": 1,
                "dimension": [{"size": 3}, {"size": 2}],
            },
        )
        with self.assertRaisesRegex(ValueError, "JPEG-compressed"):
            decode_ntndarray_payload(extract_ntndarray_payload(value))


if __name__ == "__main__":
    unittest.main()
