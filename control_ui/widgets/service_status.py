"""Shared presentation for process status, resource values and service controls."""

from qtpy import QtCore, QtGui, QtWidgets


def configure_console_font(output):
    """Keep log glyphs legible when application styles override widget fonts."""
    database = QtGui.QFontDatabase()
    font = database.systemFont(database.FixedFont)
    candidates = (font.family(), "DejaVu Sans Mono", "Liberation Mono", "Consolas", "Menlo")
    family = next((name for name in candidates if database.isFixedPitch(name)), "monospace")
    font.setFamily(family)
    font.setPointSize(10)
    font.setFixedPitch(True)
    output.setFont(font)
    # setFont alone loses to the application's global stylesheet font rule.
    output.setStyleSheet('QPlainTextEdit { font-family: "' + family + '"; font-size: 10pt; }')


class StatusBadge(QtWidgets.QLabel):
    """Pair an explicit state name with colors readable in either theme."""

    _COLORS = {
        "healthy": (("#e6f4ea", "#1b5e20"), ("#173d2a", "#9be0af")),
        "attention": (("#fff3d6", "#795000"), ("#493817", "#ffd584")),
        "error": (("#fce8e6", "#9c231b"), ("#4a2424", "#ffaaa5")),
        "active": (("#e8f0fe", "#174ea6"), ("#203955", "#a9ceff")),
        "neutral": (("#e9ecef", "#454c54"), ("#383d43", "#d4d9df")),
    }

    def __init__(self, text="Checking…", level="neutral", parent=None):
        super().__init__(parent)
        self.level = level
        self.setTextFormat(QtCore.Qt.PlainText)
        self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)
        self.set_status(text, level)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.paletteChanged.connect(self._application_palette_changed)

    @QtCore.Slot(QtGui.QPalette)
    def _application_palette_changed(self, _palette):
        self._apply_colors()

    def set_status(self, text, level):
        if level not in self._COLORS:
            raise ValueError(f"Unknown status level: {level}")
        self.level = level
        self.setProperty("statusLevel", level)
        self.setText(str(text))
        self.setAccessibleName(str(text))
        self._apply_colors()

    def _apply_colors(self):
        if getattr(self, "_applying_colors", False):
            return
        app = QtWidgets.QApplication.instance()
        palette = app.palette() if app is not None else self.palette()
        dark = palette.color(QtGui.QPalette.Window).lightness() < 128
        background, foreground = self._COLORS[self.level][int(dark)]
        self._applying_colors = True
        try:
            self.setStyleSheet(
                f"color: {foreground}; background-color: {background};"
                "border: none; border-radius: 5px; padding: 4px 9px; font-weight: 600;"
            )
        finally:
            self._applying_colors = False

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "level") and event.type() in (QtCore.QEvent.PaletteChange, QtCore.QEvent.ApplicationPaletteChange):
            self._apply_colors()


class ResourceGrid(QtWidgets.QGroupBox):
    """Aligned, selectable values instead of one long resource-status sentence."""

    def __init__(self, title="Resource usage", parent=None, columns=4):
        super().__init__(title, parent)
        self.values = {}
        self._captions = {}
        self._columns = columns
        self._grid = QtWidgets.QGridLayout(self)
        self._grid.setContentsMargins(10, 5, 10, 7)
        self._grid.setHorizontalSpacing(18)
        self._grid.setVerticalSpacing(8)
        for column in range(columns):
            self._grid.setColumnStretch(column, 1)

    def add_metric(self, key, caption, value=None, tooltip=""):
        index = len(self.values)
        cell = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QtWidgets.QLabel(caption)
        layout.addWidget(label)
        if value is None:
            value = QtWidgets.QLabel("—")
        font = value.font()
        font.setBold(True)
        value.setFont(font)
        value.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        value.setMinimumWidth(65)
        layout.addWidget(value)
        cell.setToolTip(tooltip)
        self.values[key] = value
        self._captions[key] = caption
        self._grid.addWidget(cell, index // self._columns, index % self._columns)
        return value

    def set_value(self, key, value):
        self.values[key].setText(str(value))

    def text(self):
        """Compact equivalent useful for status reports and accessibility."""
        return "  •  ".join(f"{self._captions[key]} {value.text()}" for key, value in self.values.items())

    def clear_values(self):
        for value in self.values.values():
            value.setText("—")


class ServiceCard(QtWidgets.QFrame):
    """Consistent identity, process state, metrics and action placement."""

    def __init__(self, title, identity, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setObjectName("serviceStatusCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        header = QtWidgets.QHBoxLayout()
        title_label = QtWidgets.QLabel(title)
        title_font = title_label.font()
        title_font.setPointSize(title_font.pointSize() + 2)
        title_font.setBold(True)
        title_label.setFont(title_font)
        header.addWidget(title_label)
        header.addStretch()
        self.status_badge = StatusBadge(parent=self)
        header.addWidget(self.status_badge)
        layout.addLayout(header)
        self.identity = QtWidgets.QLabel(identity)
        self.identity.setTextFormat(QtCore.Qt.PlainText)
        self.identity.setWordWrap(True)
        self.identity.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.identity)
        self.content = QtWidgets.QVBoxLayout()
        self.content.setSpacing(6)
        layout.addLayout(self.content)
        self.actions = QtWidgets.QHBoxLayout()
        self.actions.setSpacing(8)
        layout.addLayout(self.actions)

    def add_action(self, text, callback):
        button = QtWidgets.QPushButton(text)
        button.setAutoDefault(False)
        button.setDefault(False)
        button.clicked.connect(callback)
        self.actions.addWidget(button)
        return button


def overview_page(parent=None):
    """A scrollable overview keeps all cards reachable in smaller windows."""
    scroll = QtWidgets.QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
    body = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(body)
    layout.setContentsMargins(4, 8, 4, 8)
    layout.setSpacing(12)
    scroll.setWidget(body)
    return scroll, layout


def status_legend(parent=None):
    widget = QtWidgets.QWidget(parent)
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    for text, level in (("Healthy", "healthy"), ("Attention", "attention"),
                        ("Offline / fault", "error"), ("Active", "active")):
        layout.addWidget(StatusBadge(text, level, widget))
    layout.addStretch()
    return widget
