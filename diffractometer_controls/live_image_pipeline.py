"""State management for reversible live-image display processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LiveFilterRequest:
    frame_id: int
    generation: int
    image: np.ndarray


class LiveImageState:
    """Keep immutable raw frames separate from optional filtered derivatives."""

    def __init__(self):
        self.raw_frame: Optional[np.ndarray] = None
        self.raw_exposure_s: Optional[float] = None
        self.frame_id = 0
        self.filter_enabled = False
        self.filter_generation = 0
        self.filtered_frame: Optional[np.ndarray] = None
        self.filtered_frame_id: Optional[int] = None

    def ingest_raw(self, image, *, exposure_s=None) -> int:
        raw = np.array(image, copy=True)
        if raw.size == 0:
            raise ValueError("A live raw frame may not be empty.")
        self.frame_id += 1
        self.raw_frame = raw
        self.raw_exposure_s = exposure_s
        self.filtered_frame = None
        self.filtered_frame_id = None
        return self.frame_id

    def set_filter_enabled(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if enabled == self.filter_enabled:
            return False
        self.filter_enabled = enabled
        self.filter_generation += 1
        self.filtered_frame = None
        self.filtered_frame_id = None
        return True

    def make_filter_request(self) -> Optional[LiveFilterRequest]:
        if not self.filter_enabled or self.raw_frame is None:
            return None
        return LiveFilterRequest(
            frame_id=int(self.frame_id),
            generation=int(self.filter_generation),
            image=self.raw_frame,
        )

    def accept_filtered(self, image, *, frame_id: int, generation: int) -> bool:
        if not self.filter_enabled:
            return False
        if int(frame_id) != int(self.frame_id):
            return False
        if int(generation) != int(self.filter_generation):
            return False
        if self.raw_frame is None:
            return False
        filtered = np.asarray(image)
        if filtered.shape != self.raw_frame.shape:
            return False
        self.filtered_frame = np.array(filtered, copy=True)
        self.filtered_frame_id = int(frame_id)
        return True

    def display_source(self) -> Optional[np.ndarray]:
        if (
            self.filter_enabled
            and self.filtered_frame is not None
            and self.filtered_frame_id == self.frame_id
        ):
            return self.filtered_frame
        return self.raw_frame

    def compose_display(self, normalize=None) -> Optional[np.ndarray]:
        source = self.display_source()
        if source is None or normalize is None:
            return source
        normalized = normalize(source)
        return source if normalized is None else np.asarray(normalized)
