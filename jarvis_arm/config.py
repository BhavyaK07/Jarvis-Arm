"""Typed config loaded from config.yaml (validated with pydantic)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


class LoggingCfg(BaseModel):
    level: str = "INFO"
    dir: str = "logs"


class VoiceCfg(BaseModel):
    input: Literal["mic", "text"] = "mic"
    wake_word: str = "hey_jarvis"
    wake_threshold: float = 0.3
    sample_rate: int = 16000
    vad_aggressiveness: int = Field(2, ge=0, le=3)
    silence_s: float = 1.2
    max_utterance_s: float = 10
    convo_timeout_s: float = 12
    stt_model: str = "whisper-large-v3-turbo"
    language: str = "en"


class ElevenLabsCfg(BaseModel):
    voice_id: str = "pNInz6obpgDQGcFmaJgB"
    model_id: str = "eleven_flash_v2_5"
    output_format: str = "mp3_44100_128"


class PiperCfg(BaseModel):
    model_path: str | None = None


class TtsCfg(BaseModel):
    engines: list[str] = ["elevenlabs", "piper", "espeak", "say", "print"]
    elevenlabs: ElevenLabsCfg = ElevenLabsCfg()
    piper: PiperCfg = PiperCfg()


class ChatCfg(BaseModel):
    model: str = "qwen/qwen3.8-27b"
    reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    temperature: float = 0.75
    max_tokens: int = 600
    history_turns: int = 12
    humour: int = Field(75, ge=0, le=100)


class AgentCfg(BaseModel):
    model: str = "gemini-robotics-er-2-preview"
    max_calls: int = Field(15, ge=1)
    max_retries_per_step: int = Field(2, ge=0)


class CameraCfg(BaseModel):
    index: int = 0
    width: int = 1280
    height: int = 720


class ServoBusCfg(BaseModel):
    port: str = "/dev/ttyACM0"
    baudrate: int = 1_000_000
    control_hz: float = 50
    telemetry_hz: float = 5
    max_load_pct: float = 80
    max_temp_c: float = 60
    default_speed_dps: float = 60


class JointCfg(BaseModel):
    name: str
    id: int
    offset_ticks: int = 2048
    sign: Literal[1, -1] = 1
    min_deg: float
    max_deg: float

    @model_validator(mode="after")
    def _limits(self) -> "JointCfg":
        if self.min_deg >= self.max_deg:
            raise ValueError(f"joint {self.name}: min_deg must be < max_deg")
        return self


class GripperCfg(BaseModel):
    open_deg: float = 90
    closed_deg: float = 0


class WorkspaceCfg(BaseModel):
    table_z_m: float = 0.0
    min_radius_m: float = 0.08
    max_radius_m: float = 0.38
    min_z_m: float = 0.005


class Config(BaseModel):
    mode: Literal["mock", "hardware"] = "mock"
    logging: LoggingCfg = LoggingCfg()
    voice: VoiceCfg = VoiceCfg()
    tts: TtsCfg = TtsCfg()
    chat: ChatCfg = ChatCfg()
    agent: AgentCfg = AgentCfg()
    camera: CameraCfg = CameraCfg()
    servo_bus: ServoBusCfg = ServoBusCfg()
    joints: list[JointCfg]
    gripper: GripperCfg = GripperCfg()
    named_poses: dict[str, list[float]] = {}
    locations: dict[str, tuple[float, float]] = {}
    workspace: WorkspaceCfg = WorkspaceCfg()

    @model_validator(mode="after")
    def _check(self) -> "Config":
        names = [j.name for j in self.joints]
        ids = [j.id for j in self.joints]
        if len(set(names)) != len(names) or len(set(ids)) != len(ids):
            raise ValueError("joint names and ids must be unique")
        if not names or names[-1] != "gripper":
            raise ValueError("last joint must be 'gripper'")
        n_arm = len(self.joints) - 1
        for pose, vals in self.named_poses.items():
            if len(vals) != n_arm:
                raise ValueError(f"named pose '{pose}' needs {n_arm} values, got {len(vals)}")
        return self

    @property
    def arm_joints(self) -> list[JointCfg]:
        return self.joints[:-1]


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
