"""Cheap local tracker for the final approach (between Gemini calls).

TODO(merge milestone): seed with the grounded box at the scan pose, track with
OpenCV (e.g. CSRT/KCF or template match) while descending, hand off to the
ToF for the last few cm.
"""
from __future__ import annotations

import numpy as np


class Tracker:
    def init(self, image: np.ndarray, box_px: tuple[float, float, float, float]) -> None:
        raise NotImplementedError

    def update(self, image: np.ndarray) -> tuple[float, float, float, float] | None:
        raise NotImplementedError
