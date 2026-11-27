"""Perception skills: scan the table, find a described object."""
from __future__ import annotations

import time

from pydantic import BaseModel, Field

from ..perception.grounding import ALL_OBJECTS, GroundingError
from .context import SkillContext
from .registry import SkillRegistry, SkillResult


class FindArgs(BaseModel):
    description: str = Field(min_length=1, max_length=120,
                             description="Natural-language description, e.g. 'red cup'")


SETTLE_S = 0.3          # let vibration die down before trusting a frame
FRAME_TIMEOUT_S = 3.0


def _fresh_frame(ctx: SkillContext):
    """A frame captured after the arm stopped moving - not one from mid-move."""
    t_ready = time.monotonic() + SETTLE_S
    deadline = t_ready + FRAME_TIMEOUT_S
    while time.monotonic() < deadline:
        f = ctx.camera.latest()
        if f is not None and f.t >= t_ready:
            return f
        time.sleep(0.02)
    return None


def _look(ctx: SkillContext, query: str):
    """Go to scan pose, ground `query`, update the scene. Returns (objects, error)."""
    r = ctx.controller.move_to_named("scan")
    if not r.ok:
        return None, f"could not reach scan pose: {r.reason}"
    frame = _fresh_frame(ctx)
    if frame is None:
        return None, "no camera frame"
    if ctx.pixel_to_table is None:
        return None, "camera not calibrated: cannot convert pixels to table positions"
    try:
        detections = ctx.grounder.ground(frame.image, query)
    except GroundingError as e:
        return None, f"vision failed: {e}"
    found, skipped = [], 0
    for d in detections:
        xy = ctx.pixel_to_table(*d.point_px)
        if xy is None:
            skipped += 1
            continue
        found.append(ctx.scene.observe(d.label, xy[0], xy[1], d.confidence))
    return (found, skipped), None


def register(reg: SkillRegistry) -> None:
    @reg.register("scan", "Look at the whole table from the scan pose and refresh the list of visible objects.")
    def scan(ctx: SkillContext, _args) -> SkillResult:
        res, err = _look(ctx, ALL_OBJECTS)
        if err:
            return SkillResult(False, f"scan failed: {err}")
        found, skipped = res
        ctx.scene.mark_unseen_missing({o.id for o in found})
        return SkillResult(True, f"see {len(found)} object(s)",
                           {"objects": [o.summary() for o in found], "unprojectable": skipped})

    @reg.register("find", "Locate objects matching a description. Returns matching object ids.", FindArgs)
    def find(ctx: SkillContext, args: FindArgs) -> SkillResult:
        res, err = _look(ctx, args.description)
        if err:
            return SkillResult(False, f"find failed: {err}")
        found, _ = res
        if not found:
            return SkillResult(False, f"no '{args.description}' visible from the scan pose")
        return SkillResult(True, f"found {len(found)} match(es)", {"objects": [o.summary() for o in found]})
