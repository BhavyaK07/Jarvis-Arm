"""Pixel -> table-plane projection (replaces legacy px_to_table_landscape).

Cast the pixel's ray from the camera and intersect it with the plane z = table_z
in the arm base frame. The camera pose comes from FK of the current wrist pose
composed with the fixed wrist->camera transform (both TODO until kinematics +
calibration milestones).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: tuple[float, ...] = ()   # OpenCV distortion coeffs; undistort before projecting

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def pixel_to_table(u: float, v: float, intr: Intrinsics, T_base_cam: np.ndarray,
                   table_z: float = 0.0) -> tuple[float, float] | None:
    """Return (x, y) on the table in base frame, or None if the ray misses
    (parallel to table or pointing away). Camera frame is OpenCV: +z forward,
    +x right, +y down. T_base_cam is the 4x4 pose of the camera in base frame."""
    ray_cam = np.array([(u - intr.cx) / intr.fx, (v - intr.cy) / intr.fy, 1.0])
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    d = R @ ray_cam
    if abs(d[2]) < 1e-9:
        return None
    s = (table_z - t[2]) / d[2]
    if s <= 0:
        return None
    p = t + s * d
    return float(p[0]), float(p[1])


def load_intrinsics(path: str | Path) -> Intrinsics:
    with open(path) as f:
        return Intrinsics(**yaml.safe_load(f))


def save_intrinsics(intr: Intrinsics, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy,
                        "dist": list(intr.dist)}, f)
