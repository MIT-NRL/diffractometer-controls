"""Consistent control heights that respect the current Qt font and style."""


def match_control_heights(*widgets, minimum=34):
    controls = [widget for widget in widgets if widget is not None]
    for widget in controls:
        widget.ensurePolished()
    height = max([minimum, *(
        max(widget.sizeHint().height(), widget.fontMetrics().height() + 12)
        for widget in controls
    )])
    for widget in controls:
        widget.setFixedHeight(height)
    return height
