"""Tool schemas exposed to LLMs (replaces legacy COMMAND_PROMPT)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..skills.registry import SkillRegistry


class DoTaskArgs(BaseModel):
    instruction: str = Field(min_length=2, max_length=300)


# OpenAI-compatible tool for the Groq chat model.
DO_TASK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "do_task",
        "description": (
            "Hand a physical task to the robot arm, e.g. 'put the red cup in the bowl', "
            "'pick up the green block', 'go home', 'what's on the table'. The instruction is "
            "the user's request restated clearly and completely in one sentence. The arm "
            "plans and executes it itself; you never give angles or coordinates."
        ),
        "parameters": DoTaskArgs.model_json_schema(),
    },
}


def skill_declarations(registry: SkillRegistry) -> list[dict[str, Any]]:
    """Skill schemas for the task agent (Gemini function declarations shape)."""
    return registry.schemas()
