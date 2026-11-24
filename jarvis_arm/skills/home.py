"""Basic motion skills: home, named poses, gripper."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .context import SkillContext
from .registry import SkillRegistry, SkillResult


class MoveToArgs(BaseModel):
    pose: str = Field(description="Name of a predefined pose, e.g. 'home' or 'scan'")


def register(reg: SkillRegistry) -> None:
    @reg.register("home", "Move the arm to its safe home pose.")
    def home(ctx: SkillContext, _args) -> SkillResult:
        r = ctx.controller.move_to_named("home")
        return SkillResult(r.ok, "at home" if r.ok else f"home failed: {r.reason}")

    @reg.register("move_to", "Move the arm to a predefined named pose.", MoveToArgs)
    def move_to(ctx: SkillContext, args: MoveToArgs) -> SkillResult:
        r = ctx.controller.move_to_named(args.pose)
        return SkillResult(r.ok, f"at {args.pose}" if r.ok else f"move_to failed: {r.reason}")

    @reg.register("open_gripper", "Open the gripper fully.")
    def open_gripper(ctx: SkillContext, _args) -> SkillResult:
        r = ctx.controller.set_gripper(ctx.cfg.gripper.open_deg)
        return SkillResult(r.ok, "gripper open" if r.ok else f"open failed: {r.reason}")

    @reg.register("close_gripper", "Close the gripper.")
    def close_gripper(ctx: SkillContext, _args) -> SkillResult:
        # An object in the jaws stalls the gripper short of closed_deg; that
        # times out the wait, which is expected - so don't wait for completion.
        r = ctx.controller.set_gripper(ctx.cfg.gripper.closed_deg, timeout_s=2.0)
        return SkillResult(True, "gripper closed", {"reached_closed": r.ok})
