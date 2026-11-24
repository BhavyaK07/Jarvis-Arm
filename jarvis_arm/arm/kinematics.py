"""Joint <-> servo-tick mapping, plus FK/IK for the SO-101 (TODO).

Convention: joint angle 0 deg  <->  offset_ticks;  +deg = sign * +ticks.
"""
from __future__ import annotations

from ..config import JointCfg
from ..drivers.servo_bus import TICK_MAX, TICKS_PER_REV

DEG_PER_TICK = 360.0 / TICKS_PER_REV


class Unreachable(ValueError):
    """IK target outside the reachable workspace. Never scale - refuse."""


class JointMap:
    def __init__(self, joints: list[JointCfg]):
        self.joints = joints
        self.by_name = {j.name: j for j in joints}
        self.names = [j.name for j in joints]
        self.ids = [j.id for j in joints]

    def deg_to_ticks(self, name: str, deg: float) -> int:
        j = self.by_name[name]
        t = j.offset_ticks + j.sign * deg / DEG_PER_TICK
        return int(round(max(0, min(TICK_MAX, t))))

    def ticks_to_deg(self, name: str, ticks: int) -> float:
        j = self.by_name[name]
        return j.sign * (ticks - j.offset_ticks) * DEG_PER_TICK

    @staticmethod
    def dps_to_tick_speed(dps: float) -> int:
        """STS3215 goal-speed unit is ticks/s."""
        return max(1, int(round(dps / DEG_PER_TICK)))


# --------------------------------------------------------------------------
# TODO(kinematics milestone): SO-101 geometry from the URDF in
# TheRobotStudio/SO-ARM100 (Simulation/SO101/so101_new_calib.urdf).
#   fk(joint_deg)            -> 4x4 gripper pose in base frame
#   camera_pose(joint_deg)   -> 4x4 camera pose (fk @ T_wrist_cam)
#   ik(x, y, z, pitch_deg)   -> joint_deg, raises Unreachable (never scales)
# --------------------------------------------------------------------------
def fk(joint_deg: dict[str, float]):
    raise NotImplementedError("SO-101 FK not implemented yet")


def ik(x: float, y: float, z: float, pitch_deg: float = -90.0) -> dict[str, float]:
    raise NotImplementedError("SO-101 IK not implemented yet")
