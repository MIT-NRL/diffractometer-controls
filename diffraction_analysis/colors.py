"""Use the shared live palette with stable historical signal identities."""

from qtpy import QtGui

from control_ui.widgets.profile_colors import (
    HISTORY_ALPHA, LATEST_ALPHA, SERIES_COLORS, profile_colormap, profile_history_position,
)
from .curves import numeric, signal_fields


def signal_color_index(snapshot, field):
    fields = signal_fields(snapshot or {})
    rows = (snapshot or {}).get('rows', [])
    profiles = [key for key in fields if rows and numeric(rows[0].get(key)) is not None
                and numeric(rows[0][key]).ndim == 1]
    order = profiles if field in profiles else fields
    return order.index(field) if field in order else 0


def series_color(snapshot, field):
    index = signal_color_index(snapshot, field)
    return QtGui.QColor(SERIES_COLORS[index % len(SERIES_COLORS)])


def curve_color(snapshot, field, point, palette):
    if point is None:
        return series_color(snapshot, field)
    index = signal_color_index(snapshot, field)
    total = len(snapshot.get('rows', []))
    ramp = profile_colormap(index, palette.color(palette.Base).getRgb()[:3])
    color = QtGui.QColor.fromRgbF(*ramp(profile_history_position(total, point)))
    color.setAlphaF(LATEST_ALPHA if point == total - 1 else HISTORY_ALPHA)
    return color
