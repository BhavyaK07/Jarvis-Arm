"""Entry point.

    python -m jarvis_arm --text --offline     # no mic, no keys, mock arm
    python -m jarvis_arm --text               # type, but real Groq chat/TTS
    python -m jarvis_arm                      # full voice (mock arm by default)
    python -m jarvis_arm --hardware           # real servo bus (after bring-up!)
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

from .arm.controller import ArmController
from .arm.safety import is_stop_command
from .brain.agent import GeminiAgent, ScriptedAgent, TaskAgent, TaskResult
from .brain.chat import GroqChat, OfflineChat
from .brain.tts import Speaker
from .brain.voice import MicInput, TextInput, VoiceInput
from .config import Config, load_config
from .drivers.mocks import LinearProjection, MockCamera, MockServoBus, MockToF, MockWorld
from .executive.task_runner import TaskRunner
from .perception.grounding import MockGrounder
from .skills import SkillContext, build_registry
from .world.scene import Scene

log = logging.getLogger("jarvis_arm")

_RESUME_RE = re.compile(r"^\s*(resume|unfreeze|carry on|you can move)\b", re.I)
_QUIT_RE = re.compile(r"^\s*(quit|exit|shut ?down)\s*$", re.I)


@dataclass
class App:
    cfg: Config
    controller: ArmController
    camera: object
    speaker: Speaker
    chat: GroqChat | OfflineChat
    runner: TaskRunner
    scene: Scene
    voice: VoiceInput | None = None


def setup_logging(cfg: Config, verbose: bool) -> None:
    Path(cfg.logging.dir).mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(Path(cfg.logging.dir) / "jarvis.log", maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    root = logging.getLogger()
    root.setLevel(getattr(logging, cfg.logging.level.upper(), logging.INFO))
    root.handlers[:] = [fh, ch]


def build_app(cfg: Config, *, text: bool, offline: bool, tts: bool = True) -> App:
    """Construct and start everything. Mock vs hardware is cfg.mode."""
    keys = {k: os.getenv(k) for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "ELEVENLABS_API_KEY")}
    keys["ELEVENLABS_API_KEY"] = keys["ELEVENLABS_API_KEY"] or os.getenv("ELEVENLAB_API_KEY")  # legacy name

    # ---- drivers
    sim_world = proj = None
    if cfg.mode == "mock":
        sim_world = MockWorld()
        proj = LinearProjection(cfg.camera.width, cfg.camera.height)
        bus = MockServoBus([j.id for j in cfg.joints])
        camera = MockCamera(cfg.camera.width, cfg.camera.height)
        tof = MockToF(sim_world)
        grounder = MockGrounder(sim_world, proj)
    else:
        from .drivers.camera import OpenCVCamera
        from .drivers.servo_bus import FeetechServoBus
        from .drivers.tof import VL53L1XToF
        from .perception.grounding import GeminiGrounder
        bus = FeetechServoBus(cfg.servo_bus.port, cfg.servo_bus.baudrate)
        camera = OpenCVCamera(cfg.camera.index, cfg.camera.width, cfg.camera.height)
        tof = VL53L1XToF()
        grounder = GeminiGrounder(cfg.agent.model, keys["GEMINI_API_KEY"])

    controller = ArmController(bus, cfg)
    controller.start()
    camera.start()

    # ---- brain
    speaker = Speaker(cfg.tts, keys["ELEVENLABS_API_KEY"], enabled=tts and not offline)
    if offline or not keys["GROQ_API_KEY"]:
        if not offline:
            log.warning("GROQ_API_KEY missing: using offline chat")
        chat: GroqChat | OfflineChat = OfflineChat()
    else:
        chat = GroqChat(cfg.chat, keys["GROQ_API_KEY"])

    agent: TaskAgent
    if offline or cfg.mode == "mock":
        # TODO(step 5): use GeminiAgent here too once it exists.
        agent = ScriptedAgent(list(cfg.locations))
    else:
        agent = GeminiAgent(cfg.agent.model, keys["GEMINI_API_KEY"])

    scene = Scene()
    ctx = SkillContext(
        cfg=cfg, controller=controller, camera=camera, tof=tof, grounder=grounder,
        scene=scene, say=speaker.say, ask_user=lambda q: None,
        pixel_to_table=proj.to_table if proj else None,   # hardware: TODO calibration
        sim_world=sim_world,
    )
    runner = TaskRunner(agent, build_registry(), ctx, cfg.agent, cfg.logging.dir)

    if text or cfg.voice.input == "text":
        voice: VoiceInput = TextInput()
    else:
        voice = MicInput(cfg.voice, keys["GROQ_API_KEY"] or "")
    return App(cfg, controller, camera, speaker, chat, runner, scene, voice)


def handle_utterance(app: App, text: str) -> None:
    """One user utterance -> safety check -> chat -> maybe start a task."""
    # 1. Safety first, no LLM involved.
    if is_stop_command(text):
        app.runner.cancel("user said stop")
        app.controller.halt("voice stop")
        app.speaker.say("Stopped. Say resume when it's safe.")
        return
    if _RESUME_RE.match(text):
        app.controller.resume()
        app.speaker.say("Resuming. Motors are live.")
        return
    # 2. A running task asked a question: this utterance is the answer.
    if app.runner.awaiting_answer:
        app.runner.provide_answer(text)
        return
    # 3. Conversation.
    try:
        turn = app.chat.respond(text)
    except Exception as e:  # noqa: BLE001
        log.exception("chat failed")
        app.speaker.say("Give me a moment." if "429" in str(e) else "Something went wrong with my brain.")
        return
    if turn.reply:
        app.speaker.say(turn.reply)
    if turn.task:
        instruction = turn.task

        def done(res: TaskResult) -> None:
            app.speaker.say(app.chat.report(instruction, res))

        if not app.runner.start(instruction, done):
            app.speaker.say("I'm still busy with the last task. Say stop to cancel it.")


def run(app: App) -> None:
    timeout = app.cfg.voice.convo_timeout_s
    app.speaker.say("Jarvis online." + (" Say 'Hey Jarvis'." if app.voice.needs_wake_ack else ""))
    while True:
        app.voice.wait_for_wake()
        if app.voice.needs_wake_ack:
            app.speaker.say("Yeah?")
        last = time.monotonic()
        while True:
            # Stay awake while a task runs so "stop" doesn't need the wake word.
            if app.voice.needs_wake_ack and not app.runner.busy and time.monotonic() - last > timeout:
                app.speaker.say("Going to sleep.")
                break
            text = app.voice.listen()
            if not text:
                continue
            if _QUIT_RE.match(text):
                raise KeyboardInterrupt
            if app.voice.needs_wake_ack:
                print(f"you said: {text}", flush=True)
            handle_utterance(app, text)
            last = time.monotonic()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jarvis", description="Voice-controlled SO-101 arm")
    p.add_argument("--config", help="path to config.yaml (default: packaged one)")
    p.add_argument("--text", action="store_true", help="type instead of using the microphone")
    p.add_argument("--offline", action="store_true", help="no cloud APIs: offline chat + scripted agent")
    p.add_argument("--hardware", action="store_true", help="use the real servo bus (overrides mode)")
    p.add_argument("--no-tts", action="store_true", help="print replies only")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    load_dotenv()
    cfg = load_config(a.config)
    if a.hardware:
        cfg.mode = "hardware"
    setup_logging(cfg, a.verbose)

    app = build_app(cfg, text=a.text, offline=a.offline, tts=not a.no_tts)
    try:
        run(app)
    except KeyboardInterrupt:
        print()
    finally:
        app.runner.cancel("shutdown")
        app.controller.shutdown()
        app.camera.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
