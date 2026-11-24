"""Simulated hardware so the whole stack runs on a laptop with no servos.

MockWorld is ground truth: objects on the table in base-frame metres.
MockGrounder "sees" it through a fixed linear pixel<->table map (stand-in for
real intrinsics + FK), so skills exercise the same px -> table path as on
hardware.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .camera import Frame
from .servo_bus import TICK_MAX, ServoTelemetry


# --------------------------------------------------------------------- servos
class MockServoBus:
    """Servos move toward their goal at the commanded speed in real time."""

    def __init__(self, ids: Iterable[int], start_tick: int = 2048):
        self._pos = {i: float(start_tick) for i in ids}
        self._goal = dict(self._pos)
        self._speed = {i: 0.0 for i in self._pos}
        self._torque = {i: False for i in self._pos}
        self._t = time.monotonic()
        self.is_open = False
        self.load_override: dict[int, float] = {}   # tests: force a load reading
        self.temp_override: dict[int, float] = {}

    def _advance(self) -> None:
        now = time.monotonic()
        dt, self._t = now - self._t, now
        for i, p in self._pos.items():
            if not self._torque[i]:
                continue
            err = self._goal[i] - p
            step = self._speed[i] * dt if self._speed[i] > 0 else abs(err)
            self._pos[i] = p + max(-step, min(step, err))

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def ping(self, servo_id: int) -> bool:
        return servo_id in self._pos

    def set_torque(self, ids: Iterable[int], enable: bool) -> None:
        self._advance()
        for i in ids:
            self._torque[i] = enable
            if enable:
                self._goal[i] = self._pos[i]

    def sync_write_goals(self, goals: dict[int, int], speed: int) -> None:
        self._advance()
        for i, g in goals.items():
            self._goal[i] = float(max(0, min(int(g), TICK_MAX)))
            self._speed[i] = float(speed)

    def read_positions(self, ids: Iterable[int]) -> dict[int, int]:
        self._advance()
        return {i: int(round(self._pos[i])) for i in ids}

    def read_telemetry(self, ids: Iterable[int]) -> dict[int, ServoTelemetry]:
        self._advance()
        return {
            i: ServoTelemetry(
                position=int(round(self._pos[i])),
                load_pct=self.load_override.get(i, 5.0 if self._torque[i] else 0.0),
                voltage_v=12.0,
                temp_c=self.temp_override.get(i, 32.0),
            )
            for i in ids
        }


# ---------------------------------------------------------------------- world
@dataclass
class MockObject:
    label: str
    x: float
    y: float


@dataclass
class MockWorld:
    objects: list[MockObject] = field(default_factory=lambda: [
        MockObject("red cup", 0.22, 0.05),
        MockObject("blue bowl", 0.25, -0.10),
        MockObject("green block", 0.18, 0.12),
    ])
    held: MockObject | None = None
    grasp_radius_m: float = 0.03
    lock: threading.Lock = field(default_factory=threading.Lock)

    def try_grasp(self, x: float, y: float) -> MockObject | None:
        with self.lock:
            if self.held:
                return None
            for o in self.objects:
                if math.hypot(o.x - x, o.y - y) <= self.grasp_radius_m:
                    self.objects.remove(o)
                    self.held = o
                    return o
            return None

    def release(self, x: float, y: float) -> MockObject | None:
        with self.lock:
            o, self.held = self.held, None
            if o:
                o.x, o.y = x, y
                self.objects.append(o)
            return o


class LinearProjection:
    """Mock camera model: image centre looks at (x0, y0); fixed m/px scale.
    Image u (right) -> -y (robot left is +y), image v (down) -> -x."""

    def __init__(self, width: int = 1280, height: int = 720,
                 x0: float = 0.22, y0: float = 0.0, m_per_px: float = 0.0005):
        self.w, self.h, self.x0, self.y0, self.s = width, height, x0, y0, m_per_px

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return self.w / 2 - (y - self.y0) / self.s, self.h / 2 - (x - self.x0) / self.s

    def to_table(self, u: float, v: float) -> tuple[float, float] | None:
        if not (0 <= u < self.w and 0 <= v < self.h):
            return None
        return self.x0 - (v - self.h / 2) * self.s, self.y0 - (u - self.w / 2) * self.s


# --------------------------------------------------------------------- camera
class MockCamera:
    def __init__(self, width: int = 1280, height: int = 720):
        self.w, self.h = width, height

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self, max_age_s: float = 1.0) -> Frame | None:
        return Frame(np.full((self.h, self.w, 3), 128, np.uint8), time.monotonic())


# ------------------------------------------------------------------------ ToF
class MockToF:
    """Reports ~25 mm when the world says something is in the gripper."""

    def __init__(self, world: MockWorld):
        self.world = world

    def read_mm(self) -> int | None:
        return 25 if self.world.held else None
