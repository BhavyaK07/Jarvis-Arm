"""Conversational front-end (TARS persona).

The chat model never controls the arm directly: for anything physical it calls
the `do_task(instruction)` tool, which main.py hands to the TaskRunner.
"""
from __future__ import annotations

import json
import logging
import random
import re
import threading
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from ..config import ChatCfg
from .agent import TaskResult
from .tools import DO_TASK_TOOL, DoTaskArgs

log = logging.getLogger(__name__)

ACKS = ["On it.", "Working on it.", "Consider it done. Eventually.", "Right away.",
        "Engaging servos. Stand back."]


def persona_prompt(humour: int) -> str:
    return f"""
You are Jarvis, an AI assistant embodied in a desktop robotic arm with a camera and
a gripper. Your personality is modelled after TARS from Interstellar: dry wit,
playful sarcasm, occasional dark humour, but always competent and helpful when it
matters. Your humour setting is {humour}%. Act accordingly.

Your replies are spoken aloud, so write plain sentences: no markdown, lists or emoji.
Match reply length to the question; short questions get short answers. No filler.
Remember context and build on it naturally; ask a follow-up question occasionally.

When the user wants anything physical done with the arm (move, pick, place, look,
scan, open/close the gripper, go home), call the do_task tool with a clear,
complete one-sentence instruction. Do not describe angles or coordinates. Do not
pretend you did a physical task without calling the tool. If the request is
ambiguous ("put it there"), ask what they mean instead of calling the tool.

When you receive a [task finished] note, tell the user the outcome in one short
sentence, in character. If it failed, say why plainly.
""".strip()


@dataclass
class ChatTurn:
    reply: str
    task: str | None = None


class GroqChat:
    def __init__(self, cfg: ChatCfg, api_key: str, client: Any | None = None):
        if client is None:
            from groq import Groq  # lazy: voice extra
            client = Groq(api_key=api_key)
        self.client = client
        self.cfg = cfg
        self.system = {"role": "system", "content": persona_prompt(cfg.humour)}
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()   # respond() (main) and report() (task thread)

    # -------------------------------------------------------------- history
    def _append(self, msg: dict[str, Any]) -> None:
        self.history.append(msg)
        limit = self.cfg.history_turns * 2
        if len(self.history) > limit:
            cut = len(self.history) - limit
            # never start on a tool/assistant message (orphaned tool results 400)
            while cut < len(self.history) and self.history[cut]["role"] != "user":
                cut += 1
            self.history = self.history[cut:]

    def _create(self, tools: bool, extra: list[dict] | None = None):
        kw: dict[str, Any] = dict(model=self.cfg.model, temperature=self.cfg.temperature,
                                  max_tokens=self.cfg.max_tokens,
                                  messages=[self.system, *self.history, *(extra or [])])
        if self.cfg.reasoning_effort:
            kw.update(reasoning_effort=self.cfg.reasoning_effort)
        if tools:
            kw.update(tools=[DO_TASK_TOOL], tool_choice="auto")
        return self.client.chat.completions.create(**kw).choices[0].message

    # ----------------------------------------------------------------- API
    def respond(self, text: str) -> ChatTurn:
        with self._lock:
            return self._respond(text)

    def _respond(self, text: str) -> ChatTurn:
        self._append({"role": "user", "content": text})
        msg = self._create(tools=True)
        content = (msg.content or "").strip()
        calls = list(msg.tool_calls or [])
        if not calls:
            self._append({"role": "assistant", "content": content})
            return ChatTurn(content)

        self._append({"role": "assistant", "content": content, "tool_calls": [
            {"id": c.id, "type": "function",
             "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]})
        task = None
        for c in calls:
            result = "ignored: only one task at a time"
            if task is None and c.function.name == "do_task":
                try:
                    task = DoTaskArgs.model_validate(json.loads(c.function.arguments or "{}")).instruction
                    result = "task started; you'll get a [task finished] note when it's done"
                except (json.JSONDecodeError, ValidationError) as e:
                    result = f"invalid do_task arguments: {e}"
            elif c.function.name != "do_task":
                result = f"unknown tool {c.function.name}"
            self._append({"role": "tool", "tool_call_id": c.id, "content": result})
        reply = content or random.choice(ACKS)
        if not content:
            self._append({"role": "assistant", "content": reply})
        return ChatTurn(reply, task)

    def report(self, instruction: str, result: TaskResult) -> str:
        with self._lock:
            return self._report(instruction, result)

    def _report(self, instruction: str, result: TaskResult) -> str:
        note = {"role": "user", "content":
                f"[task finished] instruction: {instruction!r}; success: {result.ok}; details: {result.summary}"}
        try:
            reply = (self._create(tools=False, extra=[note]).content or "").strip()
        except Exception:  # noqa: BLE001
            log.exception("task report failed")
            reply = ""
        reply = reply or (f"Done: {result.summary}." if result.ok else f"That didn't work: {result.summary}.")
        self._append(note)
        self._append({"role": "assistant", "content": reply})
        return reply


class OfflineChat:
    """No-network stand-in: physical-sounding requests become tasks."""

    _ARM = re.compile(r"\b(pick|grab|take|get|put|place|move|drop|scan|look|see|home|open|close|gripper|hand)\b", re.I)

    def respond(self, text: str) -> ChatTurn:
        if self._ARM.search(text):
            return ChatTurn(random.choice(ACKS), task=text)
        return ChatTurn("I'm in offline mode, so my wit is running on reserve power. Ask me to move something.")

    def report(self, instruction: str, result: TaskResult) -> str:
        return f"Done. {result.summary}." if result.ok else f"That didn't work: {result.summary}."
