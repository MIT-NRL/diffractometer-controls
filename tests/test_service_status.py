"""Status text and color must remain readable when the application theme changes."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtGui, QtWidgets
from control_ui.widgets.service_status import StatusBadge, configure_console_font


def luminance(color):
    channels = [color.redF(), color.greenF(), color.blueF()]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
    return sum(value * weight for value, weight in zip(linear, (.2126, .7152, .0722)))


class StatusReadabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_console_font_survives_global_theme_font_rules(self):
        previous = self.app.styleSheet()
        output = QtWidgets.QPlainTextEdit()
        try:
            self.app.setStyleSheet('* { font-family: sans-serif; font-size: 12pt; }')
            configure_console_font(output)
            output.setPlainText('test_under_score iii WWW')
            output.show()
            self.app.processEvents()
            metrics = output.fontMetrics()
            self.assertEqual(metrics.horizontalAdvance('iii'), metrics.horizontalAdvance('WWW'))
            self.assertIn('test_under_score', output.toPlainText())
        finally:
            output.close()
            self.app.setStyleSheet(previous)

    def test_state_text_has_contrast_in_both_themes_and_updates_on_theme_change(self):
        original = QtGui.QPalette(self.app.palette())
        badges = [StatusBadge(level.title(), level) for level in StatusBadge._COLORS]
        try:
            styles = []
            for background in ("#f0f0f0", "#2d2d2d"):
                palette = QtGui.QPalette(original)
                palette.setColor(QtGui.QPalette.Window, QtGui.QColor(background))
                self.app.setPalette(palette)
                self.app.processEvents()
                styles.append([badge.styleSheet() for badge in badges])
                for badge in badges:
                    with self.subTest(background=background, state=badge.level):
                        foreground = luminance(badge.palette().color(QtGui.QPalette.WindowText))
                        fill = luminance(badge.palette().color(QtGui.QPalette.Window))
                        self.assertGreaterEqual((max(foreground, fill) + .05) / (min(foreground, fill) + .05), 4.5)
                        self.assertEqual(badge.accessibleName(), badge.level.title())
            self.assertNotEqual(styles[0], styles[1])
        finally:
            self.app.setPalette(original)
            for badge in badges:
                badge.close()


if __name__ == "__main__":
    unittest.main()
