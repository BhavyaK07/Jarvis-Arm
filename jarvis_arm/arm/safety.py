"""Safety rules that never go through an LLM."""
from __future__ import annotations

import math
import re

from ..config import JointCfg, WorkspaceCfg

# "stop" anywhere as a word, plus a few panic synonyms. Deliberately greedy:
# a false halt is cheap, a missed halt is not.
_STOP_RE = re.compile(r"\b(stop|stopp?ing|halt|freeze|abort|emergency)\b", re.IGNORECASE)


def is_stop_command(transcript: str | None) -> bool:
    return bool(transcript) and bool(_STOP_RE.search(transcript))


class SafetyViolation(ValueError):
    pass


def check_joint_targets(targets_deg: dict[str, float], joints: dict[str, JointCfg]) -> None:
    """Refuse (raise) - never clamp - targets outside configured limits."""
    for name, deg in targets_deg.items():
        if name not in joints:
            raise SafetyViolation(f"unknown joint '{name}'")
        j = joints[name]
        if not math.isfinite(deg) or not (j.min_deg <= deg <= j.max_deg):
            raise SafetyViolation(
                f"{name} target {deg:.1f} deg outside [{j.min_deg}, {j.max_deg}]")


def check_workspace(x: float, y: float, z: float, ws: WorkspaceCfg) -> None:
    r = math.hypot(x, y)
    if not (ws.min_radius_m <= r <= ws.max_radius_m):
        raise SafetyViolation(f"target radius {r:.3f} m outside [{ws.min_radius_m}, {ws.max_radius_m}]")
    if z < ws.table_z_m + ws.min_z_m:
        raise SafetyViolation(f"target z {z:.3f} m below table clearance")
