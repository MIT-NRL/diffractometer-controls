"""Detector-specific history ramps with readable old spectra on either theme."""

HISTORY_ALPHA = 0.88
LATEST_ALPHA = 0.98
SERIES_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')


def profile_history_position(total, index):
    """Shared acquisition-age positions for live and historical spectra."""
    recent_rank = max(0, int(total) - 1 - int(index))
    slots = (0.90, 0.76, 0.62, 0.48, 0.34, 0.22, 0.15)
    return slots[min(recent_rank, len(slots) - 1)]

# Stops run from oldest to newest. Latest blue/orange spectra match the
# detector identities used by the intensity and peak-position plots.
_DARK = (
    ((181, 242, 120), (119, 223, 122), (53, 212, 180), (76, 200, 228), (85, 167, 242)),
    ((245, 76, 89), (253, 94, 73), (255, 122, 67), (255, 158, 72), (255, 150, 55)),
    ((157, 134, 215), (196, 172, 241), (236, 220, 255)),
    ((94, 168, 106), (139, 210, 132), (220, 246, 173)),
    ((204, 124, 158), (236, 170, 188), (255, 221, 222)),
)
_LIGHT = (
    ((42, 130, 62), (30, 151, 103), (13, 148, 151), (28, 134, 180), (48, 113, 190)),
    ((155, 25, 50), (182, 29, 59), (211, 55, 47), (218, 83, 35), (206, 117, 22)),
    ((152, 123, 183), (121, 82, 156), (82, 40, 120)),
    ((104, 150, 96), (57, 119, 66), (24, 82, 48)),
    ((189, 116, 140), (165, 65, 105), (118, 27, 77)),
)


def _luminance(rgb):
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return sum(a * b for a, b in zip(linear, (0.2126, 0.7152, 0.0722)))


def contrast_ratio(foreground, background, alpha=1.0):
    composite = [alpha * f + (1 - alpha) * b for f, b in zip(foreground, background)]
    a, b = sorted((_luminance(composite), _luminance(background)))
    return (b + 0.05) / (a + 0.05)


def profile_colormap(index, background):
    """Return an RGBA ramp, bounding contrast after old-line alpha blending."""
    background = tuple(float(c) / 255 for c in background[:3])
    dark = _luminance(background) < 0.25
    families = _DARK if dark else _LIGHT
    stops = families[index % len(families)]
    target = 1.0 if dark else 0.0

    def color(position):
        position = float(position)
        if index % len(families) < 2:
            # Both plotters use .15 for oldest and .90 for newest/preview.
            # Use the full ramp so newest spectra reach blue/orange exactly.
            position = (position - 0.15) / 0.75
        scaled = max(0.0, min(1.0, position)) * (len(stops) - 1)
        low = min(int(scaled), len(stops) - 2)
        fraction = scaled - low
        rgb = tuple((a + (b - a) * fraction) / 255 for a, b in zip(stops[low], stops[low + 1]))
        # Do not let theme background or age blending erase old spectra.
        for _ in range(24):
            if contrast_ratio(rgb, background, HISTORY_ALPHA) >= 3:
                break
            rgb = tuple(c + (target - c) * 0.12 for c in rgb)
        return (*rgb, 1.0)

    return color
