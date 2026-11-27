"""Pick an object by scene id.

Real sequence (TODO, needs IK + calibration + tracker):
  open -> hover above (x, y) -> track/align -> descend until ToF < threshold
  -> close -> lift -> ToF confirms object -> mark held.
In mock mode the grasp is simulated against MockWorld, including the ToF check,
so the agent sees realistic failures (stale positions -> "ToF shows no object").
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .context import SkillContext
from .registry import SkillRegistry, SkillResult

TOF_HELD_MAX_MM = 60


class PickArgs(BaseModel):
    object_id: str = Field(pattern=r"^obj\d+$", description="Scene object id from scan/find, e.g. 'obj3'")


def register(reg: SkillRegistry) -> None:
    @reg.register("pick", "Pick up a scene object by its id (from scan/find).", PickArgs)
    def pick(ctx: SkillContext, args: PickArgs) -> SkillResult:
        obj = ctx.scene.get(args.object_id)
        if obj is None:
            return SkillResult(False, f"no object {args.object_id}; run scan or find first")
        if (held := ctx.scene.held()) is not None:
            return SkillResult(False, f"already holding {held.id} ({held.label}); place it first")
        if obj.status not in ("on_table", "placed"):
            return SkillResult(False, f"{obj.id} is {obj.status}; rescan")
        if not ctx.simulated:
            return SkillResult(False, "pick not implemented on this backend yet (needs IK); works in --offline mock mode")

        r = ctx.controller.set_gripper(ctx.cfg.gripper.open_deg)
        if not r.ok:
            return SkillResult(False, f"pick failed opening gripper: {r.reason}")
        # (simulated approach/descend)
        ctx.sim_world.try_grasp(obj.x, obj.y)
        ctx.controller.set_gripper(ctx.cfg.gripper.closed_deg, timeout_s=1.0)
        r = ctx.controller.move_to_named("scan")   # lift
        if not r.ok:
            return SkillResult(False, f"pick failed lifting: {r.reason}")
        mm = ctx.tof.read_mm()
        if mm is None or mm > TOF_HELD_MAX_MM:
            ctx.controller.set_gripper(ctx.cfg.gripper.open_deg)
            ctx.scene.set_status(obj.id, "missing")
            return SkillResult(False, "pick failed: ToF shows no object in gripper (object moved? rescan)")
        ctx.scene.set_status(obj.id, "held")
        return SkillResult(True, f"holding {obj.id} ({obj.label})", {"tof_mm": mm})
