"""Deterministic analytic signals used by the disconnected demo IOC."""

from __future__ import annotations

import numpy as np


DIFFRACTION_AXIS_LIMIT = 209.21799055746422
IMAGE_SHAPE = (128, 128)


def diffraction_spectrum(
    motor_position: float,
    *,
    nbins: int = 350,
    exposure: float = 0.2,
    optimum: float = 3.0,
    detector_offset: float = 0.0,
    seed: int = 0,
):
    """Return an HE3-like axis and noisy Gaussian spectrum.

    The optimum is intentionally easy to identify: it is simultaneously the
    narrowest and strongest response.  Supplying the same arguments and seed
    produces the same result, which keeps demo tests repeatable.
    """
    nbins = max(1, int(nbins))
    exposure = max(0.0, float(exposure))
    distance = float(motor_position) - (float(optimum) + float(detector_offset))
    axis = np.linspace(-DIFFRACTION_AXIS_LIMIT, DIFFRACTION_AXIS_LIMIT, nbins)
    # Keep the peak conspicuous at the motor's default position (zero), while
    # retaining a clear optimization target near 3 degrees.  The previous
    # envelope was narrow enough that the starting spectrum mostly looked
    # like a broad background feature.
    sigma = 10.0 + 1.2 * distance * distance
    amplitude = 30000.0 * np.exp(-0.5 * (distance / 2.0) ** 2)
    center = 14.0 * distance + 8.0 * float(detector_offset)
    expected_rate = 22.0 + amplitude * np.exp(-0.5 * ((axis - center) / sigma) ** 2)
    counts = np.random.default_rng(int(seed)).poisson(expected_rate * exposure)
    return axis, counts.astype(np.int32, copy=False)


def gaussian_image(
    motor_x: float,
    motor_y: float,
    *,
    exposure: float = 0.1,
    shape=IMAGE_SHAPE,
    seed: int = 0,
):
    """Return a 2-D Gaussian whose centre follows two demo motors."""
    rows, cols = (int(shape[0]), int(shape[1]))
    yy, xx = np.indices((rows, cols), dtype=float)
    cx = (cols - 1) / 2.0 + 5.0 * float(motor_x)
    cy = (rows - 1) / 2.0 + 5.0 * float(motor_y)
    rate = 6.0 + 2200.0 * np.exp(
        -0.5 * (((xx - cx) / 12.0) ** 2 + ((yy - cy) / 10.0) ** 2)
    )
    return np.random.default_rng(int(seed)).poisson(
        rate * max(0.0, float(exposure))
    ).astype(np.int32)


def slanted_edge_image(
    focus_position: float,
    *,
    exposure: float = 0.1,
    optimum: float = 0.0,
    shape=IMAGE_SHAPE,
    seed: int = 0,
):
    """Return a slanted edge with blur increasing quadratically off focus."""
    rows, cols = (int(shape[0]), int(shape[1]))
    yy, xx = np.indices((rows, cols), dtype=float)
    distance = float(focus_position) - float(optimum)
    blur = 0.8 + 1.6 * distance * distance
    signed_distance = (xx - (cols - 1) / 2.0) + 0.32 * (yy - (rows - 1) / 2.0)
    # erf would add a scipy dependency to the IOC.  tanh gives a compact,
    # smooth edge with the same useful focus behaviour.
    edge = 0.5 * (1.0 + np.tanh(signed_distance / max(blur, 0.1)))
    rate = 10.0 + 1400.0 * edge
    return np.random.default_rng(int(seed)).poisson(
        rate * max(0.0, float(exposure))
    ).astype(np.int32)


def edge_blur_width(focus_position: float, optimum: float = 0.0) -> float:
    """Expose the analytic focus response for future adaptive-plan tests."""
    distance = float(focus_position) - float(optimum)
    return 0.8 + 1.6 * distance * distance
