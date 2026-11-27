"""Place the held object on/in another object, or at a named table location."""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from ..arm.safety import SafetyViolation, check_workspace
from .context import SkillContext
from .registry import SkillRegistry, SkillResult

PLACE_Z_M = 0.05   # release height above table (refined per target later)


class PlaceArgs(BaseModel):
    object_id: str | None = Field(None, pattern=r"^obj\d+$",
                                  description="Put the held object on/in this scene object")
    location: str | None = Field(None, description="Or: a named table location, e.g. 'left'")

    @model_validator(mode="after")
    def _one(self) -> "PlaceArgs":
        if (self.object_id is None) == (self.location is None):
            raise ValueError("give exactly one of object_id or location")
        return self


def register(reg: SkillRegistry) -> None:
    @reg.register("place", "Place the held object onto/into a scene object, or at a named location.", PlaceArgs)
    def place(ctx: SkillContext, args: PlaceArgs) -> SkillResult:
        held = ctx.scene.held()
        if held is None:
            return SkillResult(False, "not holding anything")
        if args.object_id:
            tgt = ctx.scene.get(args.object_id)
            if tgt is None:
                return SkillResult(False, f"no object {args.object_id}")
            if tgt.id == held.id:
                return SkillResult(False, "cannot place an object on itself")
            x, y, where = tgt.x, tgt.y, f"{tgt.id} ({tgt.label})"
        else:
            loc = ctx.cfg.locations.get(args.location)
            if loc is None:
                return SkillResult(False, f"unknown location '{args.location}'; known: {sorted(ctx.cfg.locations)}")
            (x, y), where = loc, args.location
        try:
            check_workspace(x, y, ctx.cfg.workspace.table_z_m + PLACE_Z_M, ctx.cfg.workspace)
        except SafetyViolation as e:
            return SkillResult(False, f"place refused: {e}")
        if not ctx.simulated:
            return SkillResult(False, "place not implemented on this backend yet (needs IK); works in --offline mock mode")

        # (simulated move above target + descend)
        ctx.sim_world.release(x, y)
        r = ctx.controller.set_gripper(ctx.cfg.gripper.open_deg)
        if not r.ok:
            return SkillResult(False, f"place failed opening gripper: {r.reason}")
        ctx.scene.set_status(held.id, "placed", x, y)
        return SkillResult(True, f"placed {held.id} ({held.label}) at {where}")
