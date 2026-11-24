"""Skill registry: the ONLY things an LLM can make the arm do.

Every call is validated against a pydantic model; bad args come back as a
structured failure the agent can read, never an exception into the arm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


@dataclass
class SkillResult:
    ok: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "message": self.message, **({"data": self.data} if self.data else {})}


class NoArgs(BaseModel):
    pass


@dataclass
class Skill:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[Any, Any], SkillResult]   # (ctx, args) -> SkillResult


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, name: str, description: str, args_model: type[BaseModel] = NoArgs):
        def deco(fn):
            if name in self._skills:
                raise ValueError(f"skill '{name}' already registered")
            self._skills[name] = Skill(name, description, args_model, fn)
            return fn
        return deco

    @property
    def names(self) -> list[str]:
        return list(self._skills)

    def call(self, ctx: Any, name: str, raw_args: dict[str, Any] | None) -> SkillResult:
        skill = self._skills.get(name)
        if skill is None:
            return SkillResult(False, f"unknown skill '{name}'. Available: {self.names}")
        try:
            args = skill.args_model.model_validate(raw_args or {})
        except ValidationError as e:
            errs = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            return SkillResult(False, f"invalid args for {name}: {errs}")
        try:
            return skill.fn(ctx, args)
        except Exception as e:  # noqa: BLE001 - a skill bug must not kill the task loop
            log.exception("skill %s crashed", name)
            return SkillResult(False, f"{name} crashed: {type(e).__name__}: {e}")

    def schemas(self) -> list[dict[str, Any]]:
        """Function declarations: {name, description, parameters(JSON schema)}."""
        out = []
        for s in self._skills.values():
            params = s.args_model.model_json_schema()
            params.pop("title", None)
            for p in params.get("properties", {}).values():
                p.pop("title", None)
            out.append({"name": s.name, "description": s.description, "parameters": params})
        return out
