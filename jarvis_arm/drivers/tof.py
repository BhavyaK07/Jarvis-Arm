"""VL53L1X time-of-flight sensor in the gripper (final approach + grasp check)."""
from __future__ import annotations

from typing import Protocol


class ToF(Protocol):
    def read_mm(self) -> int | None:
        """Distance in mm, or None if no target in range / read failed."""
        ...


class VL53L1XToF:
    """I2C VL53L1X on the Pi. TODO(hardware track): implement once the
    breakout is in hand (candidate lib: `vl53l1x` / adafruit-circuitpython-vl53l1x)."""

    def __init__(self, i2c_bus: int = 1, address: int = 0x29):
        self.i2c_bus, self.address = i2c_bus, address

    def read_mm(self) -> int | None:
        raise NotImplementedError("VL53L1X driver not implemented yet")
