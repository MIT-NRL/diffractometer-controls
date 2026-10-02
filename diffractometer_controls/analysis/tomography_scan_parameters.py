"""Shared angular-position helpers for tomography planning and execution."""

import math
from collections.abc import Sequence


def paired_second_half_indices(
    base_projection_count: int,
    extra_projection_count: int,
) -> tuple[int, ...]:
    """Return evenly distributed base indices for exact 180-degree pairs.

    Index zero is omitted because its pair is the already acquired 180-degree
    endpoint. The final index is always selected, which guarantees a 360-degree
    projection for a base scan spanning 0 through 180 degrees.
    """

    base_projection_count = int(base_projection_count)
    extra_projection_count = int(extra_projection_count)
    if base_projection_count < 2:
        raise ValueError("base_projection_count must be at least two")
    if extra_projection_count == 0:
        return ()
    if extra_projection_count < 2:
        raise ValueError("a sparse paired correction set requires at least two projections")

    base_intervals = base_projection_count - 1
    if extra_projection_count > base_intervals:
        raise ValueError(
            "extra_projection_count cannot exceed the number of available base-angle pairs"
        )

    return tuple(
        int(math.floor((index * base_intervals / extra_projection_count) + 0.5))
        for index in range(1, extra_projection_count + 1)
    )


def extend_tomography_angles_deg(
    base_angles_deg: Sequence[float],
    *,
    tilt_correction_projections: int = 0,
    full_360_scan: bool = False,
) -> tuple[float, ...]:
    """Extend an inclusive 0–180° base grid with paired second-half views."""

    base_angles = tuple(float(angle) for angle in base_angles_deg)
    if len(base_angles) < 2:
        raise ValueError("the base tomography scan must contain at least two projections")
    if not math.isclose(base_angles[0], 0.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("paired or full second-half scans require a 0° base start")
    if not math.isclose(base_angles[-1], 180.0, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("paired or full second-half scans require an included 180° endpoint")

    if bool(full_360_scan):
        if int(tilt_correction_projections) != 0:
            raise ValueError(
                "full_360_scan and tilt_correction_projections are mutually exclusive"
            )
        extra_angles = tuple(angle + 180.0 for angle in base_angles[1:])
    else:
        indices = paired_second_half_indices(
            len(base_angles),
            int(tilt_correction_projections),
        )
        extra_angles = tuple(base_angles[index] + 180.0 for index in indices)

    return base_angles + extra_angles
