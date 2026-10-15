"""Make adl2pydm displays readable with native Qt fonts and dark palettes.

Keep the source stylesheet on each widget so returning to light mode restores
the MEDM colors. Work on Qt widgets rather than the converter: this also covers
saved .ui files, embedded displays, and displays opened after startup.
"""

import re

from qtpy import QtCore, QtGui, QtWidgets
from pydm.widgets.drawing import PyDMDrawingLine, PyDMDrawingRectangle
from pydm.widgets.embedded_display import PyDMEmbeddedDisplay

from control_ui.core.themes import _palette_is_dark


_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_CONVERTER_SELECTOR = re.compile(r"(?:PyDM\w+#[\w-]+|QLabel#text(?:_\d+)?|#screen)")
_TEXT_NAME = re.compile(r"text(?:_update)?(?:_\d+)?")
_INTERIOR_SELECTOR = re.compile(r"Q(?:Label|LineEdit|PushButton|RadioButton|ComboBox|Slider)(?:::[\w:.-]+)?")
_COLOR = re.compile(
    r"(?P<prefix>(?:^|;)\s*(?P<property>color|background-color)\s*:\s*)"
    r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def _neutral(color):
    return max(color.red(), color.green(), color.blue()) - min(color.red(), color.green(), color.blue()) <= 12


def _luminance(color):
    linear = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4
              for v in (color.redF(), color.greenF(), color.blueF())]
    return sum(v * w for v, w in zip(linear, (.2126, .7152, .0722)))


def _contrast(first, second):
    a, b = sorted((_luminance(first), _luminance(second)))
    return (b + .05) / (a + .05)


def _readable_color(foreground, background):
    """Retain a semantic color's hue, mixing toward white/black as needed."""
    if _contrast(foreground, background) >= 4.5:
        return foreground
    target = max((QtGui.QColor("white"), QtGui.QColor("black")),
                 key=lambda color: _contrast(color, background))
    for step in range(1, 101):
        channels = [round(a + (b - a) * step / 100)
                    for a, b in zip(foreground.getRgb()[:3], target.getRgb()[:3])]
        color = QtGui.QColor(*channels)
        if _contrast(color, background) >= 4.5:
            return color
    return target


def _roles(selector):
    if "Button" in selector:
        return QtGui.QPalette.Button, QtGui.QPalette.ButtonText
    if "LineEdit" in selector or "ComboBox" in selector:
        return QtGui.QPalette.Base, QtGui.QPalette.Text
    return QtGui.QPalette.Window, QtGui.QPalette.WindowText


def _dark_stylesheet(source, palette, inherited_background, preserve_background=False):
    # Require the converter's class/object-name selector before considering
    # interior Qt controls. Leave application styles and alarm selectors alone.
    if not any(_CONVERTER_SELECTOR.fullmatch(m[1].strip()) for m in _RULE.finditer(source)):
        return source

    def adapt_rule(rule):
        selector, body = rule[1].strip(), rule[2]
        if not (_CONVERTER_SELECTOR.fullmatch(selector) or _INTERIOR_SELECTOR.fullmatch(selector)):
            return rule[0]
        colors = {}
        for match in _COLOR.finditer(body):
            colors[match["property"].lower()] = QtGui.QColor(*map(int, match.group(3, 4, 5)))
        background_role, foreground_role = _roles(selector)
        background = colors.get("background-color", inherited_background)
        if "background-color" in colors and _neutral(background) and not preserve_background:
            background = palette.color(background_role)

        def adapt_color(match):
            name = match["property"].lower()
            color = QtGui.QColor(*map(int, match.group(3, 4, 5)))
            if name == "background-color":
                replacement = background
            else:
                if _neutral(color) and ("background-color" not in colors or _neutral(colors["background-color"])):
                    color = palette.color(foreground_role)
                replacement = _readable_color(color, background)
            return match["prefix"] + replacement.name()

        # Older saved conversions use PyDMFrame#screen, but PyDM loads the
        # canvas onto a Display wrapper. Match its object name in dark mode.
        if selector == "PyDMFrame#screen":
            selector = "#screen"
        return selector + " {" + _COLOR.sub(adapt_color, body) + "}"

    return _RULE.sub(adapt_rule, source)


class AdlDisplayAdapter(QtCore.QObject):
    """Application-owned adapter for converted widgets, including late loads."""

    def __init__(self, application):
        super().__init__(application)
        self._application = application
        self._updating = False
        application.installEventFilter(self)

    def refresh(self):
        # Parents first: text without an explicit fill must contrast with its
        # inherited canvas, including deliberately colored MEDM panels.
        def depth(widget):
            result = 0
            while widget.parentWidget() is not None:
                result += 1
                widget = widget.parentWidget()
            return result

        for widget in sorted(self._application.allWidgets(), key=depth):
            self._fit_embedded_display(widget)
            self._apply(widget)
            self._apply_graphic(widget)
            self._fit_text(widget, allow_widen=True)

    def _fit_embedded_display(self, widget):
        if self._updating or not isinstance(widget, PyDMEmbeddedDisplay):
            return
        if not re.fullmatch(r"composite(?:_\d+)?", widget.objectName()):
            return
        content = widget.embedded_widget
        parent = widget.parentWidget()
        if (content is None or content.objectName() != "screen" or parent is None
                or QtWidgets.QWidget.layout(parent) is not None):
            return
        bounds = content.childrenRect()
        geometry = widget.geometry()
        right, bottom = parent.width(), parent.height()
        for sibling in parent.children():
            if (not isinstance(sibling, QtWidgets.QWidget) or sibling is widget
                    or isinstance(sibling, (PyDMDrawingRectangle, PyDMDrawingLine))):
                continue
            other = sibling.geometry()
            if other.left() >= geometry.right() and other.top() <= geometry.bottom() and other.bottom() >= geometry.top():
                right = min(right, other.left())
            if other.top() >= geometry.bottom() and other.left() <= geometry.right() and other.right() >= geometry.left():
                bottom = min(bottom, other.top())
        width = max(widget.width(), min(bounds.right() + 1, right - geometry.left()))
        height = max(widget.height(), min(bounds.bottom() + 1, bottom - geometry.top()))
        if (width, height) != (widget.width(), widget.height()):
            self._updating = True
            try:
                widget.resize(width, height)
            finally:
                self._updating = False

    def _apply_graphic(self, widget):
        if self._updating or not isinstance(widget, (PyDMDrawingRectangle, PyDMDrawingLine)):
            return
        if not re.fullmatch(r"(?:rectangle|line)(?:_\d+)?", widget.objectName()):
            return
        if widget.channel or widget.rules or widget.alarmSensitiveContent or widget.alarmSensitiveBorder:
            return
        current = (QtGui.QColor(widget.penColor), QtGui.QBrush(widget.brush))
        previous = getattr(widget, "_adl_graphic_colors", None)
        source = previous[0] if previous is not None and current == previous[1] else current
        pen, brush = QtGui.QColor(source[0]), QtGui.QBrush(source[1])
        palette = self._application.palette()
        if _palette_is_dark(palette):
            if _neutral(pen):
                pen = _readable_color(pen, palette.color(QtGui.QPalette.Window))
            if _neutral(brush.color()):
                color = palette.color(QtGui.QPalette.Button)
                color.setAlpha(brush.color().alpha())
                brush.setColor(color)
        widget._adl_graphic_colors = (source, (pen, brush))
        self._updating = True
        try:
            if current[0] != pen:
                widget.penColor = pen
            if current[1] != brush:
                widget.brush = brush
        finally:
            self._updating = False

    def _widen_static_label(self, widget, font):
        # Some motor labels have spare space before the next field. Use that
        # gap before reducing type size, without moving their left edge or
        # changing centered/right-aligned labels' authored anchors.
        parent = widget.parentWidget()
        if (type(widget) is not QtWidgets.QLabel or parent is None
                or QtWidgets.QWidget.layout(parent) is not None
                or not widget.alignment() & QtCore.Qt.AlignLeft):
            return
        geometry = widget.geometry()
        previous = getattr(widget, "_adl_text_width", None)
        original = previous[0] if previous and geometry.width() == previous[1] else geometry.width()
        right = parent.width()
        for sibling in parent.children():
            if (not isinstance(sibling, QtWidgets.QWidget) or sibling is widget
                    or isinstance(sibling, (PyDMDrawingRectangle, PyDMDrawingLine))):
                continue
            bounds = sibling.geometry()
            if (bounds.left() > geometry.left()
                    and bounds.top() <= geometry.center().y() <= bounds.bottom()):
                right = min(right, bounds.left() - 2)
        desired = int(QtGui.QFontMetricsF(font, widget).horizontalAdvance(widget.text()) + 3)
        width = max(original, min(desired, right - geometry.left()))
        widget._adl_text_width = (original, width)
        if width != widget.width():
            self._updating = True
            try:
                widget.resize(width, widget.height())
            finally:
                self._updating = False

    def _fit_text(self, widget, allow_widen=False):
        if self._updating or not isinstance(widget, QtWidgets.QLabel):
            return
        if not _TEXT_NAME.fullmatch(widget.objectName()) or widget.wordWrap():
            return
        if not any(_CONVERTER_SELECTOR.fullmatch(m[1].strip()) for m in _RULE.finditer(widget.styleSheet())):
            return
        document = (widget.findChild(QtGui.QTextDocument)
                    if widget.textFormat() != QtCore.Qt.RichText else None)
        current = widget.font()
        previous = getattr(widget, "_adl_font_fit", None)
        source = previous[0] if previous is not None and current == previous[1] else current
        if allow_widen:
            self._widen_static_label(widget, source)
        area = widget.contentsRect()
        width = area.width() - 2 * widget.margin() - max(0, widget.indent()) - 2
        height = area.height() - 2 * widget.margin() - 2
        if width <= 0 or height <= 0:
            return
        key = (source.toString(), widget.text(), width, height, widget.logicalDpiY(),
               widget.indent(), widget.frameWidth(), document is not None)
        if previous is not None and key == previous[2]:
            self._center_selectable_text(widget, document)
            return
        fitted = QtGui.QFont(source)
        # QLabel centers a full line box, including unused ascent/descent.
        # Allow that box to exceed the widget when the visible glyphs still
        # fit; fitting the full line height makes motor labels needlessly tiny.
        size = source.pointSizeF()
        if size <= 0:
            size = source.pixelSize() * 72 / widget.logicalDpiY()
        minimum_size = min(size, 4 * 72 / widget.logicalDpiY())
        while size > minimum_size:
            metrics = QtGui.QFontMetricsF(fitted, widget)
            ink = metrics.tightBoundingRect(widget.text() or "0")
            baseline = (height - metrics.height()) / 2 + metrics.ascent()
            fits_height = (ink.height() <= height if document is not None
                           else baseline + ink.top() >= 0 and baseline + ink.bottom() <= height)
            # QLabel adds half an 'x' of automatic indentation to framed,
            # left/right-aligned text (including PyDM alarm padding).
            auto_indent = (int(metrics.horizontalAdvance("x")) // 2
                           if widget.indent() < 0 and widget.frameWidth() > 0
                           and widget.alignment() & (QtCore.Qt.AlignLeft | QtCore.Qt.AlignRight) else 0)
            if (fits_height
                    and metrics.horizontalAdvance(widget.text()) <= width - auto_indent):
                break
            size = max(minimum_size, size - .25)
            fitted.setPointSizeF(size)
        widget._adl_font_fit = (QtGui.QFont(source), fitted, key)
        self._updating = True
        try:
            if fitted != current:
                widget.setFont(fitted)
            alignment = widget.alignment()
            widget.setAlignment((alignment & ~QtCore.Qt.AlignVertical_Mask) | QtCore.Qt.AlignVCenter)
            self._center_selectable_text(widget, document)
        finally:
            self._updating = False

    def _center_selectable_text(self, widget, document):
        # Selectable QLabel text uses a QTextDocument. When its line box is
        # taller than the padded label, Qt top-aligns that document and clips
        # the glyphs, even with AlignVCenter. Fit its paragraph to the content
        # height and center the glyph baseline while retaining native selection.
        if document is None or document.blockCount() != 1 or "\n" in widget.text():
            return
        height = widget.contentsRect().height() - 2 * widget.margin()
        if height <= 0:
            return
        cursor = QtGui.QTextCursor(document)
        paragraph = cursor.blockFormat()
        if paragraph.lineHeightType() != QtGui.QTextBlockFormat.FixedHeight or paragraph.lineHeight() != height:
            paragraph.setLineHeight(height, QtGui.QTextBlockFormat.FixedHeight)
            cursor.setBlockFormat(paragraph)
        document.documentLayout().documentSize()
        layout = document.begin().layout()
        if layout.lineCount() != 1:
            return
        line = layout.lineAt(0)
        ink = QtGui.QFontMetricsF(widget.font(), widget).tightBoundingRect(widget.text() or "0")
        top = (height - ink.height()) / 2 - ink.top() - line.ascent()
        if abs(line.y() - top) > .01:
            line.setPosition(QtCore.QPointF(line.x(), top))

    def _apply(self, widget):
        if self._updating:
            return
        current = widget.styleSheet()
        previous = getattr(widget, "_adl_theme_styles", None)
        source = previous[0] if previous is not None and current == previous[1] else current
        if not source:
            return
        palette = self._application.palette()
        themed = source
        if _palette_is_dark(palette):
            background = widget.palette().color(widget.backgroundRole())
            # Readbacks may deliberately keep their authored fill to remain
            # visually distinct from the canvas. Still check text contrast.
            themed = _dark_stylesheet(source, palette, background,
                                      preserve_background=bool(widget.property("preserveAdlBackground")))
        if themed == current:
            return
        widget._adl_theme_styles = (source, themed)
        self._updating = True
        try:
            widget.setStyleSheet(themed)
        finally:
            self._updating = False

    def eventFilter(self, watched, event):
        if event.type() in (QtCore.QEvent.Polish, QtCore.QEvent.StyleChange):
            if isinstance(watched, QtWidgets.QWidget):
                self._apply(watched)
                self._apply_graphic(watched)
                self._fit_text(watched)
        elif event.type() in (QtCore.QEvent.Paint, QtCore.QEvent.Resize, QtCore.QEvent.FontChange):
            # PyDM updates QLabel text directly when values arrive. Paint also
            # catches those changes, and the cache avoids work for unchanged text.
            if isinstance(watched, QtWidgets.QLabel):
                self._fit_text(watched, allow_widen=event.type() == QtCore.QEvent.Paint)
            elif isinstance(watched, PyDMEmbeddedDisplay):
                self._fit_embedded_display(watched)
        return False
