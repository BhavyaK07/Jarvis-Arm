"""Runs one task at a time in a background thread so the voice loop can keep
listening (for "stop" and for ask_user answers) while the arm works."""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..brain.agent import StepRecord, TaskAgent, TaskResult
from ..brain.tools import skill_declarations
from ..config import AgentCfg
from ..skills.context import SkillContext
from ..skills.registry import SkillRegistry, SkillResult

log = logging.getLogger(__name__)


class TaskAborted(Exception):
    pass


class TaskRunner:
    def __init__(self, agent: TaskAgent, registry: SkillRegistry, ctx: SkillContext,
                 cfg: AgentCfg, log_dir: str | Path = "logs", answer_timeout_s: float = 30.0):
        self.agent, self.registry, self.ctx, self.cfg = agent, registry, ctx, cfg
        self.log_path = Path(log_dir) / "tasks.jsonl"
        self.answer_timeout_s = answer_timeout_s
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._cancel_reason = ""
        self._answers: queue.Queue[str] = queue.Queue()
        self._awaiting = threading.Event()
        ctx.ask_user = self.ask       # route the ask_user skill through us

    # ------------------------------------------------------------ public
    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def awaiting_answer(self) -> bool:
        return self._awaiting.is_set()

    def start(self, instruction: str, on_done: Callable[[TaskResult], None]) -> bool:
        if self.busy:
            return False

        def _target():
            res = self.run_sync(instruction)
            try:
                on_done(res)
            except Exception:  # noqa: BLE001
                log.exception("on_done callback failed")

        self._thread = threading.Thread(target=_target, name="task", daemon=True)
        self._thread.start()
        return True

    def cancel(self, reason: str = "cancelled") -> None:
        self._cancel_reason = reason
        self._cancel.set()

    def provide_answer(self, text: str) -> None:
        self._answers.put(text)

    def ask(self, question: str) -> str | None:
        while not self._answers.empty():
            self._answers.get_nowait()
        self.ctx.say(question)
        self._awaiting.set()
        try:
            deadline = time.monotonic() + self.answer_timeout_s
            while time.monotonic() < deadline and not self._cancel.is_set():
                try:
                    return self._answers.get(timeout=0.2)
                except queue.Empty:
                    continue
            return None
        finally:
            self._awaiting.clear()

    def run_sync(self, instruction: str) -> TaskResult:
        self._cancel.clear()
        self._cancel_reason = ""
        steps: list[StepRecord] = []
        fails: Counter[str] = Counter()
        n_calls = 0

        def call(name: str, args: dict[str, Any]) -> SkillResult:
            nonlocal n_calls
            if self._cancel.is_set():
                raise TaskAborted(self._cancel_reason or "cancelled")
            if self.ctx.controller.halted:
                raise TaskAborted(f"arm halted: {self.ctx.controller.halt_reason}")
            n_calls += 1
            if n_calls > self.cfg.max_calls:
                raise TaskAborted(f"step budget of {self.cfg.max_calls} calls exceeded")
            key = f"{name}:{json.dumps(args, sort_keys=True)}"
            if fails[key] > self.cfg.max_retries_per_step:
                res = SkillResult(False, f"retry limit reached for {name}{args}; replan or give up")
            else:
                res = self.registry.call(self.ctx, name, args)
                if not res.ok:
                    fails[key] += 1
            steps.append(StepRecord(name, args, res.to_dict()))
            log.info("skill %s(%s) -> %s", name, args, res.message)
            if self.ctx.controller.halted or self._cancel.is_set():
                raise TaskAborted(self._cancel_reason or f"arm halted: {self.ctx.controller.halt_reason}")
            return res

        t0 = time.time()
        try:
            result = self.agent.run(instruction, call, skill_declarations(self.registry))
            result.steps = steps
        except TaskAborted as e:
            result = TaskResult(False, f"task aborted: {e}", steps)
        except Exception as e:  # noqa: BLE001
            log.exception("agent crashed")
            result = TaskResult(False, f"agent error: {type(e).__name__}: {e}", steps)
        self._log(instruction, result, t0)
        return result

    # ------------------------------------------------------------ logging
    def _log(self, instruction: str, res: TaskResult, t0: float) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "t": t0, "duration_s": round(time.time() - t0, 2), "instruction": instruction,
                    "ok": res.ok, "summary": res.summary,
                    "steps": [s.__dict__ for s in res.steps],
                }, default=str) + "\n")
        except OSError:
            log.exception("could not write task log")
