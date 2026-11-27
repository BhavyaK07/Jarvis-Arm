# Jarvis-Arm

A voice-controlled SO-101 robotic arm. You give a task out loud ("put the red cup in the
bowl"); the arm finds the objects with open-vocabulary grounding, plans with validated
skills, acts one step at a time, checks each step, and retries or replans when one fails.

The previous Arduino/YOLO version is archived in `legacy/` and under the git tag `v0-arduino`.

## Quick start (no hardware or API keys needed)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
python -m jarvis_arm --text --offline        # mock arm, typed input, rule-based planner
pytest
```

Grounding on your own photos (needs `.[vision]` and `GEMINI_API_KEY`):

```bash
python -m jarvis_arm.tools.ground_demo photos/ -q "red cup" -q "bowl"   # annotated copies -> logs/grounding_demo/
```

Try: `what do you see`, `put the red cup in the bowl`, `pick up the green block`,
`stop`, `resume`, `go home`, `quit`.

## Simulator

```bash
pip install -e '.[sim]' && python -m jarvis_arm.tools.fetch_so101_model
python -m jarvis_arm --sim --text --offline         # whole robot on MuJoCo physics
```

See [docs/sim_guide.md](docs/sim_guide.md) for how it works and exercises.

## With cloud APIs / voice

```bash
pip install -e '.[voice,vision]'
cp .env.example .env                          # fill in GROQ / GEMINI / ELEVENLABS keys
brew install mpv                              # ElevenLabs playback (Pi: sudo apt install mpv)
python -m jarvis_arm --text                   # Groq chat persona, typed input
python -m jarvis_arm                          # "Hey Jarvis" wake word + mic
```

TTS falls back through ElevenLabs → Piper → espeak → macOS `say` → print, using the first
engine that works. On the Pi, install `espeak-ng` or Piper as the offline fallback.

## Architecture

```
mic ─► brain/voice (wake word, VAD, Whisper) ─► "stop"? ──yes──► controller.halt()  (no LLM)
                                                   │no
                                                   ▼
                     brain/chat (Groq, TARS persona) ── do_task(instruction) ──►
executive/task_runner (background thread; call budget, per-step retry cap, cancel, logs/tasks.jsonl)
          │  call(skill, args)  (schema-validated)
          ▼
skills/  scan · find · pick · place · move_to · home · open/close_gripper · say · ask_user
          │
          ├─ perception/  grounding (Gemini Robotics-ER), calibration (pixel → table plane), tracker
          ├─ world/scene  objects: id, label, x, y, confidence, last_seen, status
          └─ arm/controller  ← single bus-owner thread, torque-hold halt, load/temp watchdog
                 └─ drivers/  servo_bus (Feetech STS3215), camera, tof, mocks
```

The LLMs never output joint angles or coordinates. Every skill returns a structured
`SkillResult` (e.g. `pick failed: ToF shows no object in gripper`).

## Status

| Area | State |
|---|---|
| Mock arm / camera / ToF / world, controller, safety, scene, skill registry, task runner | working, tested |
| Voice front-end (wake word, VAD, Groq Whisper), Groq chat with `do_task`, TTS chain | ported, smoke-tested |
| `ScriptedAgent` (offline, rule-based) | working on mocks |
| `GeminiGrounder` (step 4): Robotics-ER 2 points/boxes, retries, per-call logs; `tools/ground_demo.py` | working, tested on synthetic images; real desk photos pending |
| `GeminiAgent` (step 5) | stub |
| `FeetechServoBus` | written from the STS3215 control table, **untested on hardware** |
| MuJoCo sim backend (`--sim`): physics servo bus, rendered wrist camera, ToF ray, ground-truth grounder; see [docs/sim_guide.md](docs/sim_guide.md) | working, tested |
| SO-101 FK/IK, camera calibration, tracker, VL53L1X driver | TODO |

## Config

`jarvis_arm/config.yaml` holds the joint map (IDs, `offset_ticks`, sign, limits), named
poses, table locations, workspace limits, models, and servo watchdog thresholds. Secrets
live only in `.env`.
