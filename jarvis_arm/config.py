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


class GroundingCfg(BaseModel):
    model: str = "gemini-robotics-er-2-preview"
    api_version: str = "v1beta"          # ER 2 is not served on v1
    thinking_level: Literal["minimal", "low", "medium", "high"] = "low"
    temperature: float = Field(0.5, ge=0, le=2)
    max_items: int = Field(10, ge=1, le=25)
    max_side_px: int = Field(1280, ge=256)   # downscale longer side before upload
    jpeg_quality: int = Field(85, ge=30, le=100)
    timeout_s: float = 20
    retries: int = Field(1, ge=0)
    log_images: bool = True


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
    overload_hold_s: float = 0.6     # load must stay above max_load_pct this long to halt
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


class SimObjectCfg(BaseModel):
    label: str
    shape: Literal["box", "cylinder", "sphere"]
    # box: half-extents [x, y, z]; cylinder: [radius, half_height]; sphere: [radius]
    size: list[float]
    pos: tuple[float, float]                    # table (x, y), base frame metres
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
    mass: float = Field(0.05, gt=0)

    @model_validator(mode="after")
    def _size(self) -> "SimObjectCfg":
        need = {"box": 3, "cylinder": 2, "sphere": 1}[self.shape]
        if len(self.size) != need or min(self.size) <= 0:
            raise ValueError(f"{self.label}: {self.shape} needs {need} positive size values")
        return self

    @property
    def half_height(self) -> float:
        return self.size[2] if self.shape == "box" else self.size[1] if self.shape == "cylinder" else self.size[0]


class SimCfg(BaseModel):
    model_path: str = "third_party/so101_mjcf/so101_new_calib_camera.xml"
    render_width: int = 640
    render_height: int = 480
    fps: float = 10
    camera_fovy_deg: float = 70      # vertical FOV; ~100 deg diagonal at 4:3
    realtime: bool = True
    objects: list[SimObjectCfg] = []


class Config(BaseModel):
    mode: Literal["mock", "sim", "hardware"] = "mock"
    logging: LoggingCfg = LoggingCfg()
    voice: VoiceCfg = VoiceCfg()
    tts: TtsCfg = TtsCfg()
    chat: ChatCfg = ChatCfg()
    agent: AgentCfg = AgentCfg()
    grounding: GroundingCfg = GroundingCfg()
    camera: CameraCfg = CameraCfg()
    servo_bus: ServoBusCfg = ServoBusCfg()
    joints: list[JointCfg]
    gripper: GripperCfg = GripperCfg()
    named_poses: dict[str, list[float]] = {}
    locations: dict[str, tuple[float, float]] = {}
    workspace: WorkspaceCfg = WorkspaceCfg()
    sim: SimCfg = SimCfg()

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
