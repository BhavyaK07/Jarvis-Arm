import json
from types import SimpleNamespace

from jarvis_arm.brain.agent import ScriptedAgent, TaskResult
from jarvis_arm.brain.chat import GroqChat
from jarvis_arm.config import ChatCfg
from jarvis_arm.executive.task_runner import TaskRunner


def test_registry_validates(mock_ctx, registry):
    assert not registry.call(mock_ctx, "teleport", {}).ok
    r = registry.call(mock_ctx, "pick", {"object_id": "DROP TABLE"})
    assert not r.ok and "invalid args" in r.message
    r = registry.call(mock_ctx, "place", {"object_id": "obj1", "location": "left"})
    assert not r.ok and "exactly one" in r.message
    names = {s["name"] for s in registry.schemas()}
    assert {"scan", "find", "pick", "place", "move_to", "open_gripper", "close_gripper",
            "home", "say", "ask_user"} <= names


def test_scan_find_pick_place(mock_ctx, registry):
    r = registry.call(mock_ctx, "scan", {})
    assert r.ok and len(r.data["objects"]) == 3
    cup = registry.call(mock_ctx, "find", {"description": "red cup"}).data["objects"][0]
    bowl = registry.call(mock_ctx, "find", {"description": "bowl"}).data["objects"][0]
    assert registry.call(mock_ctx, "pick", {"object_id": cup["id"]}).ok
    assert not registry.call(mock_ctx, "pick", {"object_id": bowl["id"]}).ok  # hands full
    assert registry.call(mock_ctx, "place", {"object_id": bowl["id"]}).ok
    assert mock_ctx.scene.get(cup["id"]).status == "placed"
    assert mock_ctx.sim_world.held is None


def test_pick_detects_moved_object_with_tof(mock_ctx, registry):
    cup = registry.call(mock_ctx, "find", {"description": "red cup"}).data["objects"][0]
    obj = next(o for o in mock_ctx.sim_world.objects if o.label == "red cup")
    obj.x += 0.1                                 # someone moved it after the scan
    r = registry.call(mock_ctx, "pick", {"object_id": cup["id"]})
    assert not r.ok and "ToF" in r.message


def test_task_runner_end_to_end(mock_ctx, registry, cfg):
    runner = TaskRunner(ScriptedAgent(list(cfg.locations)), registry, mock_ctx, cfg.agent, cfg.logging.dir)
    res = runner.run_sync("put the red cup in the bowl")
    assert res.ok, res.summary
    assert [s.skill for s in res.steps] == ["find", "pick", "find", "place", "home"]
    line = json.loads(open(f"{cfg.logging.dir}/tasks.jsonl").readline())
    assert line["ok"] and line["instruction"] == "put the red cup in the bowl"


def test_task_runner_budget_and_halt(mock_ctx, registry, cfg):
    class Looper:
        def run(self, instruction, call, tools):
            while True:
                call("scan", {})

    cfg.agent.max_calls = 3
    runner = TaskRunner(Looper(), registry, mock_ctx, cfg.agent, cfg.logging.dir)
    res = runner.run_sync("loop forever")
    assert not res.ok and "budget" in res.summary and len(res.steps) == 3

    mock_ctx.controller.halt("test")
    res = runner.run_sync("loop forever")
    assert not res.ok and "halted" in res.summary


def test_retry_limit_per_step(mock_ctx, registry, cfg):
    class Stubborn:
        def run(self, instruction, call, tools):
            for _ in range(5):
                call("pick", {"object_id": "obj99"})
            return TaskResult(False, "gave up")

    runner = TaskRunner(Stubborn(), registry, mock_ctx, cfg.agent, cfg.logging.dir)
    msgs = [s.result["message"] for s in runner.run_sync("x").steps]
    attempts = cfg.agent.max_retries_per_step + 1
    assert sum("retry limit" in m for m in msgs) == 5 - attempts


# ---------------------------------------------------------------- chat
def _msg(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


class FakeGroq:
    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        return self.replies.pop(0)


def test_chat_do_task_tool_call():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(
        name="do_task", arguments=json.dumps({"instruction": "put the red cup in the bowl"})))
    fake = FakeGroq([_msg(None, [call]), _msg("Cup is in the bowl. Applause optional.")])
    chat = GroqChat(ChatCfg(), "k", client=fake)
    turn = chat.respond("put the red cup in the bowl please")
    assert turn.task == "put the red cup in the bowl" and turn.reply
    roles = [m["role"] for m in chat.history]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert "bowl" in chat.report(turn.task, TaskResult(True, "done"))


def test_chat_history_capped_and_starts_with_user():
    fake = FakeGroq([_msg(f"r{i}") for i in range(50)])
    chat = GroqChat(ChatCfg(history_turns=3), "k", client=fake)
    for i in range(20):
        chat.respond(f"q{i}")
    assert len(chat.history) <= 6 and chat.history[0]["role"] == "user"
    assert fake.requests[-1]["messages"][0]["role"] == "system"


def test_halt_during_step_aborts_task(mock_ctx, registry, cfg):
    class HaltMidway:
        def run(self, instruction, call, tools):
            mock_ctx.controller.halt("voice stop")
            call("find", {"description": "red cup"})
            return TaskResult(False, "should not get here")

    runner = TaskRunner(HaltMidway(), registry, mock_ctx, cfg.agent, cfg.logging.dir)
    res = runner.run_sync("x")
    assert "aborted" in res.summary and "voice stop" in res.summary
