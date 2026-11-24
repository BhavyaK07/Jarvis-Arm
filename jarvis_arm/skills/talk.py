"""Talking skills: say something, ask the user a question."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .context import SkillContext
from .registry import SkillRegistry, SkillResult


class SayArgs(BaseModel):
    text: str = Field(min_length=1, max_length=200)


class AskArgs(BaseModel):
    question: str = Field(min_length=1, max_length=200)


def register(reg: SkillRegistry) -> None:
    @reg.register("say", "Say a short sentence out loud (progress updates).", SayArgs)
    def say(ctx: SkillContext, args: SayArgs) -> SkillResult:
        ctx.say(args.text)
        return SkillResult(True, "said")

    @reg.register("ask_user", "Ask the user a clarifying question and wait for the spoken answer.", AskArgs)
    def ask_user(ctx: SkillContext, args: AskArgs) -> SkillResult:
        answer = ctx.ask_user(args.question)
        if not answer:
            return SkillResult(False, "no answer from user")
        return SkillResult(True, "user answered", {"answer": answer})
