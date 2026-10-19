"""Real converter/Qt checks for MEDM text sizing, contrast and live themes."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtGui, QtWidgets
from adl2pydm import adl_parser, output_handler
from pydm.display import load_adl_file, load_ui_file
from pydm.widgets.channel import PyDMChannel
from pydm.widgets.drawing import PyDMDrawingRectangle
from pydm.widgets.embedded_display import PyDMEmbeddedDisplay
from pydm.utilities.stylesheet import global_style

from control_ui.core.adl_display import AdlDisplayAdapter
from control_ui.core.themes import _build_dark_palette

FIXTURE = Path(__file__).parent / "fixtures" / "adl_text_layout.adl"


def contrast(first, second):
    def luminance(color):
        channels = (color.redF(), color.greenF(), color.blueF())
        linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in channels]
        return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))

    a, b = sorted((luminance(first), luminance(second)))
    return (b + .05) / (a + .05)


class AdlDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        cls.app.setStyle("Fusion")
        cls.light = QtGui.QPalette(cls.app.palette())
        cls.adapter = AdlDisplayAdapter(cls.app)
        cls.app._adl_display_adapter = cls.adapter
        # Keep these guards alive through widget destruction: no IOC needed.
        cls.guards = [patch.object(PyDMChannel, "connect"), patch.object(PyDMChannel, "disconnect")]
        for guard in cls.guards:
            guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.app.setPalette(cls.light)
        cls.app.removeEventFilter(cls.adapter)
        cls.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        for guard in reversed(cls.guards):
            guard.stop()

    def setUp(self):
        self.app.setPalette(self.light)
        self.original_stylesheet = self.app.styleSheet()
        self.widgets = []

    def tearDown(self):
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        self.app.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
        self.app.setPalette(self.light)
        self.app.setStyleSheet(self.original_stylesheet)

    def load(self, loader=load_adl_file, path=FIXTURE):
        screen = loader(str(path), macros={"P": "TEST:"})
        self.widgets.append(screen)
        screen.show()
        self.app.processEvents()
        return screen

    def theme(self, palette):
        self.app.setPalette(palette)
        self.adapter.refresh()
        self.app.processEvents()

    def assert_text_fits(self, label):
        label.grab()  # Also exercise fitting after a PyDM value/text change.
        metrics = label.fontMetrics()
        self.assertLessEqual(metrics.horizontalAdvance(label.text()), label.contentsRect().width() - 2)
        self.assertTrue(label.alignment() & QtCore.Qt.AlignVCenter)
        # Compare the actual rendered ink with a taller label using the same
        # font. Equal ink counts prove that the smaller box clips no glyphs.
        copies = []
        for height in (label.height(), label.height() + 60):
            copy = QtWidgets.QLabel(label.text())
            copy.setFont(label.font())
            copy.setAlignment(label.alignment())
            copy.setStyleSheet("color: black; background-color: white;")
            copy.resize(label.width(), height)
            rendered = copy.grab().toImage()
            copies.append(sum(rendered.pixelColor(x, y) != QtGui.QColor("white")
                              for y in range(rendered.height()) for x in range(rendered.width())))
            copy.deleteLater()
        self.assertEqual(*copies)

    def test_usb2408_header_text_fits_and_keeps_authored_alignment(self):
        screen = self.load()
        model = screen.findChild(QtWidgets.QLabel, "text_1")
        title = screen.findChild(QtWidgets.QLabel, "text")
        readback = screen.findChild(QtWidgets.QLabel, "text_update")
        readback.setText("USB-2408-2AO")
        for label in (model, title, readback):
            self.assert_text_fits(label)
        # A 20-pixel motor/device label must retain useful visible text height,
        # even when the font's unused line spacing exceeds the widget height.
        self.assertGreaterEqual(model.fontMetrics().tightBoundingRect(model.text()).height(), 12)
        self.assertTrue(model.alignment() & QtCore.Qt.AlignRight)
        self.assertTrue(title.alignment() & QtCore.Qt.AlignHCenter)
        self.assertTrue(readback.alignment() & QtCore.Qt.AlignLeft)

    def test_selectable_readbacks_are_centered_and_complete_with_pydm_padding(self):
        # Production uses PyDM's global stylesheet, including alarm padding.
        # A plain QLabel copy without selection/padding misses the clipping.
        self.app.setStyleSheet(global_style())
        label = self.load().findChild(QtWidgets.QLabel, "text_update")

        def ink(image):
            return [(x, y) for y in range(image.height()) for x in range(image.width())
                    if (color := image.pixelColor(x, y)).blue() > color.red() + 30
                    and color.blue() > color.green() + 30]

        for value in ("01FDC620", "1.02", "100", "gypqy"):
            with self.subTest(value=value):
                label.setText(value)
                self.app.processEvents()
                image = label.grab().toImage()
                pixels = ink(image)
                self.assertTrue(pixels)
                top, bottom = min(y for x, y in pixels), max(y for x, y in pixels)
                self.assertLessEqual(abs(top - (image.height() - 1 - bottom)), 2)

                reference = QtWidgets.QLabel(value)
                reference.setTextInteractionFlags(label.textInteractionFlags())
                reference.setAlignment(label.alignment())
                reference.setStyleSheet("color: rgb(10,0,184); background-color: rgb(187,187,187); padding: 2px;")
                reference.resize(label.width(), 100)
                reference.ensurePolished()
                reference.setFont(label.font())
                complete_ink = ink(reference.grab().toImage())
                self.assertGreaterEqual(len(pixels), .95 * len(complete_ink))
                self.assertLessEqual(len(pixels), 1.05 * len(complete_ink))
                reference.deleteLater()
        self.assertTrue(label.textInteractionFlags() & QtCore.Qt.TextSelectableByMouse)
        label.setSelection(1, 3)
        self.adapter.refresh()
        self.assertEqual(label.selectedText(), "ypq")

    def test_embedded_panel_bottom_buttons_are_visible_without_overlapping_neighbors(self):
        self.app.setStyleSheet(global_style())
        screen = self.load(path=FIXTURE.with_name("adl_embedded_layout.adl"))
        panel = screen.findChild(PyDMEmbeddedDisplay, "composite")
        self.app.processEvents()
        self.assertEqual(panel.height(), 120)
        self.assertLessEqual(panel.geometry().bottom(), screen.height() - 1)
        buttons = {button.text(): button for button in panel.findChildren(QtWidgets.QPushButton)}
        for text in ("Start", "Stop"):
            button = buttons[text]
            self.assertEqual(button.visibleRegion().boundingRect(), button.rect())
            position = button.mapTo(screen, button.rect().center())
            self.assertIs(screen.childAt(position), button)

        # When an adjacent panel occupies the gap, keep its controls clear.
        neighbor = QtWidgets.QWidget(screen)
        neighbor.setGeometry(10, 125, 360, 30)
        neighbor.show()
        panel.resize(360, 100)
        self.adapter.refresh()
        self.app.processEvents()
        self.assertLess(panel.geometry().bottom(), neighbor.geometry().top())

    def test_readback_refits_on_value_changes_and_regains_size(self):
        readback = self.load().findChild(QtWidgets.QLabel, "text_update")
        readback.setText("2408")
        self.assert_text_fits(readback)
        initial_size = readback.font().pointSizeF()
        readback.setText("USB-2408-2AO long model")
        self.assert_text_fits(readback)
        self.assertLess(readback.font().pointSizeF(), initial_size)
        readback.setText("2408")
        self.assert_text_fits(readback)
        self.assertEqual(readback.font().pointSizeF(), initial_size)

    def test_motor_screen_keeps_larger_text_and_uses_spare_label_space(self):
        from diffractometer_controls.site.mitr.profile import external_display_directories
        path = next((root / "motorx_all.adl" for root in external_display_directories()
                     if (root / "motorx_all.adl").is_file()), None)
        if path is None:
            self.skipTest("EPICS motor ADL sources are not installed")
        screen = self.load(load_adl_file, path)
        labels = {label.text(): label for label in screen.findChildren(QtWidgets.QLabel)}
        for text in ("Drive", "Readback", "Normal", "Backlash", "Jog", "Maximum Spd"):
            self.assert_text_fits(labels[text])
        for text in ("Drive", "Readback", "Normal", "Backlash"):
            # Keep the motor screen's visible 11-pixel lettering at normal
            # DPI, and at least that much visible height at higher display DPI.
            self.assertGreaterEqual(labels[text].fontMetrics().tightBoundingRect(text).height(), 11)
        maximum = labels["Maximum Spd"]
        self.assertGreater(maximum.width(), 73)
        self.assertLessEqual(maximum.geometry().right(), 93)
        self.assertGreaterEqual(maximum.fontMetrics().tightBoundingRect(maximum.text()).height(), 9)
        self.theme(_build_dark_palette())
        for text in ("Readback", "Maximum Spd"):
            label = labels[text]
            self.assertGreaterEqual(contrast(label.palette().color(QtGui.QPalette.WindowText),
                                             screen.palette().color(QtGui.QPalette.Window)), 4.5)

    def test_late_adl_load_in_dark_mode_has_readable_text_and_colored_inputs(self):
        self.theme(_build_dark_palette())
        screen = self.load()
        for label in screen.findChildren(QtWidgets.QLabel):
            if label.objectName().startswith("text"):
                self.assertGreaterEqual(contrast(label.palette().color(QtGui.QPalette.WindowText),
                                                 label.palette().color(QtGui.QPalette.Window)), 4.5)
        entry = screen.findChild(QtWidgets.QLineEdit, "text_entry")
        self.assertEqual(entry.palette().color(QtGui.QPalette.Base).name(), "#73dfff")
        self.assertEqual(entry.palette().color(QtGui.QPalette.Text).name(), "#000000")
        red = screen.findChild(PyDMDrawingRectangle, "rectangle_1")
        self.assertEqual(red.brush.color().name(), "#fd0000")

    def test_theme_switches_restore_original_styles_and_drawing_colors(self):
        screen = self.load()
        original = {widget: widget.styleSheet() for widget in [screen, *screen.findChildren(QtWidgets.QWidget)]}
        outline = screen.findChild(PyDMDrawingRectangle, "rectangle")
        original_pen = QtGui.QColor(outline.penColor)
        for _ in range(2):
            self.theme(_build_dark_palette())
            self.assertGreaterEqual(contrast(outline.penColor, self.app.palette().color(QtGui.QPalette.Window)), 4.5)
            self.theme(self.light)
            for widget, stylesheet in original.items():
                self.assertEqual(widget.styleSheet(), stylesheet)
            self.assertEqual(outline.penColor, original_pen)

    def test_saved_ui_with_older_canvas_selector_also_follows_theme(self):
        parsed = adl_parser.MedmMainWidget(str(FIXTURE))
        parsed.parseAdlBuffer(parsed.getAdlLines(str(FIXTURE)))
        converter = output_handler.Widget2Pydm()
        converter.write_ui(parsed, None)
        xml = converter.writer.generate_ui_contents().replace("#screen {", "PyDMFrame#screen {")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "converted.ui"
            path.write_text(xml)
            screen = self.load(load_ui_file, path)
        self.theme(_build_dark_palette())
        self.assertEqual(screen.palette().color(QtGui.QPalette.Window).name(), "#2d2d2d")
        self.assertGreaterEqual(contrast(screen.text_1.palette().color(QtGui.QPalette.WindowText),
                                         screen.palette().color(QtGui.QPalette.Window)), 4.5)

    def test_camera_readbacks_keep_gray_fill_and_readable_fitted_text(self):
        path = FIXTURE.parents[2] / "diffractometer_controls/site/mitr/tomo_cam_controls.ui"
        screen = self.load(load_ui_file, path)
        readbacks = [w for w in screen.findChildren(QtWidgets.QLabel)
                     if w.objectName().startswith("text_update")]
        self.assertEqual(len(readbacks), 16)
        mode = screen.findChild(QtWidgets.QLabel, "text_update_5")
        mode.setText("Continuous")
        for palette in (_build_dark_palette(), self.light, _build_dark_palette()):
            self.theme(palette)
            for label in readbacks:
                self.assertEqual(label.palette().color(QtGui.QPalette.Window).name(), "#bbbbbb")
                self.assertGreaterEqual(contrast(label.palette().color(QtGui.QPalette.WindowText),
                                                 label.palette().color(QtGui.QPalette.Window)), 4.5)
            self.assert_text_fits(mode)
            caption = screen.findChild(QtWidgets.QLabel, "text_17")
            # MITRApplication repolishes widgets after changing its palette.
            caption.style().unpolish(caption)
            caption.style().polish(caption)
            self.assertEqual(caption.palette().color(QtGui.QPalette.WindowText),
                             palette.color(QtGui.QPalette.WindowText))

    def test_ordinary_widgets_and_alarm_rules_are_preserved(self):
        normal = QtWidgets.QLabel("Ordinary label")
        normal.setObjectName("status")
        normal.setStyleSheet("QLabel#status { color: rgb(255, 0, 0); }")
        normal.setFont(QtGui.QFont("DejaVu Sans", 20))
        self.widgets.append(normal)
        original_font = normal.font()
        original_style = normal.styleSheet()
        screen = self.load()
        label = screen.findChild(QtWidgets.QLabel, "text_update")
        alarm_rule = '\nPyDMLabel[alarmSeverity="2"] { color: rgb(255, 0, 0); }'
        label.setStyleSheet(label.styleSheet() + alarm_rule)
        self.theme(_build_dark_palette())
        self.assertTrue(label.styleSheet().endswith(alarm_rule))
        self.assertEqual(normal.styleSheet(), original_style)
        self.assertEqual(normal.font(), original_font)

    def test_palette_updates_and_external_stylesheet_edits_do_not_use_stale_colors(self):
        screen = self.load()
        self.theme(_build_dark_palette())
        palette = _build_dark_palette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#151515"))
        self.theme(palette)
        self.assertEqual(screen.palette().color(QtGui.QPalette.Window).name(), "#151515")
        label = screen.findChild(QtWidgets.QLabel, "text_1")
        new_style = "QLabel#text_1 { color: rgb(20, 20, 20); }"
        label.setStyleSheet(new_style)
        self.app.processEvents()
        self.theme(self.light)
        self.assertEqual(label.styleSheet(), new_style)


if __name__ == "__main__":
    unittest.main()
