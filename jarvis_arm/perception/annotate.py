"""Draw detections on a COPY of a frame, for logs and the demo tool.

Annotated images are for humans only; the model always gets the raw frame.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .grounding import Detection

_COLOURS = [(0, 200, 255), (255, 120, 0), (80, 220, 80), (220, 60, 220), (0, 80, 255), (255, 220, 0)]


def draw_detections(image: np.ndarray, detections: list["Detection"]) -> np.ndarray:
    import cv2  # vision extra

    out = image.copy()
    scale = max(out.shape[:2]) / 1000
    thick = max(1, round(2 * scale))
    for i, d in enumerate(detections):
        c = _COLOURS[i % len(_COLOURS)]
        u, v = (int(round(p)) for p in d.point_px)
        if d.box_px:
            x1, y1, x2, y2 = (int(round(p)) for p in d.box_px)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, thick)
        cv2.circle(out, (u, v), max(4, round(7 * scale)), c, -1)
        cv2.circle(out, (u, v), max(4, round(7 * scale)), (0, 0, 0), 1)
        text = f"{i}: {d.label}"
        fs = 0.6 * scale + 0.2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick + 2)
        tx = u + 10 if u + 10 + tw < out.shape[1] else max(0, u - 10 - tw)   # keep on-image
        org = (tx, max(th + 4, v - 10))
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), thick + 2)
        cv2.putText(out, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, c, thick)
    return out
