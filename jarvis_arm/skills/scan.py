"""Perception skills: scan the table, find a described object."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .context import SkillContext
from .registry import SkillRegistry, SkillResult


class FindArgs(BaseModel):
    description: str = Field(min_length=1, max_length=120,
                             description="Natural-language description, e.g. 'red cup'")


def _look(ctx: SkillContext, query: str):
    """Go to scan pose, ground `query`, update the scene. Returns (objects, error)."""
    r = ctx.controller.move_to_named("scan")
    if not r.ok:
        return None, f"could not reach scan pose: {r.reason}"
    frame = ctx.camera.latest()
    if frame is None:
        return None, "no camera frame"
    if ctx.pixel_to_table is None:
        return None, "camera not calibrated: cannot convert pixels to table positions"
    found, skipped = [], 0
    for d in ctx.grounder.ground(frame.image, query):
        xy = ctx.pixel_to_table(*d.point_px)
        if xy is None:
            skipped += 1
            continue
        found.append(ctx.scene.observe(d.label, xy[0], xy[1], d.confidence))
    return (found, skipped), None


def register(reg: SkillRegistry) -> None:
    @reg.register("scan", "Look at the whole table from the scan pose and refresh the list of visible objects.")
    def scan(ctx: SkillContext, _args) -> SkillResult:
        res, err = _look(ctx, "all objects on the table")
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
