"""Everything a skill is allowed to touch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..arm.controller import ArmController
from ..config import Config
from ..drivers.camera import Camera
from ..drivers.mocks import MockWorld
from ..drivers.tof import ToF
from ..perception.grounding import Grounder
from ..world.scene import Scene

PixelToTable = Callable[[float, float], "tuple[float, float] | None"]


@dataclass
class SkillContext:
    cfg: Config
    controller: ArmController
    camera: Camera
    tof: ToF
    grounder: Grounder
    scene: Scene
    say: Callable[[str], None]
    ask_user: Callable[[str], "str | None"]
    # (u, v) -> table (x, y) at the CURRENT arm pose. None until the camera is
    # calibrated and FK exists (hardware) - skills must report that clearly.
    pixel_to_table: PixelToTable | None = None
    # Mock mode only: ground-truth world used to simulate grasps.
    sim_world: MockWorld | None = None

    @property
    def simulated(self) -> bool:
        return self.sim_world is not None
