"""Task agents: turn one instruction into a sequence of validated skill calls.

The agent never touches hardware: it only gets `call(name, args)`, which the
TaskRunner wraps with budget, retry limits, cancellation and logging.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ..skills.registry import SkillResult

SkillCaller = Callable[[str, dict[str, Any]], SkillResult]


@dataclass
class StepRecord:
    skill: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class TaskResult:
    ok: bool
    summary: str
    steps: list[StepRecord] = field(default_factory=list)


class TaskAgent(Protocol):
    def run(self, instruction: str, call: SkillCaller, tools: list[dict[str, Any]]) -> TaskResult: ...


# ---------------------------------------------------------------------------
class ScriptedAgent:
    """Offline rule-based planner. Good enough to drive the mock stack and the
    tests; the real planner is GeminiAgent."""

    _PLACE = re.compile(r"\b(?:put|place|move|drop)\s+(?:the\s+)?(.+?)\s+(?:in|into|on|onto|to|at|inside)\s+(?:the\s+)?(.+)", re.I)
    _PICK = re.compile(r"\b(?:pick\s+up|pick|grab|get|take)\s+(?:the\s+)?(.+)", re.I)

    def __init__(self, locations: list[str] | None = None):
        self.locations = set(locations or [])

    def run(self, instruction: str, call: SkillCaller, tools: list[dict[str, Any]]) -> TaskResult:
        text = instruction.strip().rstrip(".!?")
        low = text.lower()
        if m := self._PLACE.search(text):
            return self._pick_and_place(call, m.group(1), m.group(2))
        if m := self._PICK.search(text):
            ok, msg, _ = self._pick(call, m.group(1))
            return TaskResult(ok, msg)
        if re.search(r"\b(scan|look|see|what'?s on)\b", low):
            r = call("scan", {})
            labels = [o["label"] for o in r.data.get("objects", [])]
            return TaskResult(r.ok, f"I see: {', '.join(labels) or 'nothing'}" if r.ok else r.message)
        if "home" in low:
            r = call("home", {})
            return TaskResult(r.ok, r.message)
        if re.search(r"\bopen\b", low):
            r = call("open_gripper", {})
            return TaskResult(r.ok, r.message)
        if re.search(r"\bclose\b", low):
            r = call("close_gripper", {})
            return TaskResult(r.ok, r.message)
        return TaskResult(False, f"offline planner can't handle '{instruction}'")

    @staticmethod
    def _best(r: SkillResult) -> dict | None:
        objs = r.data.get("objects", []) if r.ok else []
        return max(objs, key=lambda o: o["confidence"]) if objs else None

    def _pick(self, call: SkillCaller, what: str) -> tuple[bool, str, dict | None]:
        for _attempt in range(2):           # verify -> rescan -> retry once
            obj = self._best(call("find", {"description": what}))
            if obj is None:
                return False, f"couldn't find the {what}", None
            r = call("pick", {"object_id": obj["id"]})
            if r.ok:
                return True, f"picked up the {obj['label']}", obj
        return False, f"pick of the {what} failed: {r.message}", None

    def _pick_and_place(self, call: SkillCaller, what: str, where: str) -> TaskResult:
        ok, msg, obj = self._pick(call, what)
        if not ok:
            return TaskResult(False, msg)
        if where.lower() in self.locations:
            r = call("place", {"location": where.lower()})
        else:
            tgt = self._best(call("find", {"description": where}))
            if tgt is None:
                call("home", {})
                return TaskResult(False, f"I'm holding the {obj['label']} but can't find the {where}")
            r = call("place", {"object_id": tgt["id"]})
        call("home", {})
        return TaskResult(r.ok, f"put the {obj['label']} in the {where}" if r.ok else r.message)


# ---------------------------------------------------------------------------
class GeminiAgent:
    """TODO(step 5): tool-calling loop on Gemini Robotics-ER.

    Each turn: send instruction + scene summary + latest scan-pose frame +
    `tools` as function declarations (automatic function calling OFF). Execute
    exactly the returned function call via `call`, append the structured
    SkillResult as the function response, repeat until the model answers in
    text (= done) or `call` raises TaskAborted (budget/cancel/halt).
    """

    def __init__(self, model: str, api_key: str | None):
        self.model, self.api_key = model, api_key

    def run(self, instruction: str, call: SkillCaller, tools: list[dict[str, Any]]) -> TaskResult:
        raise NotImplementedError("GeminiAgent is step 5 of the build order")
