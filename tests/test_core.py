import math
import time

import numpy as np
import pytest

from jarvis_arm.arm.kinematics import JointMap
from jarvis_arm.arm.safety import SafetyViolation, check_joint_targets, check_workspace, is_stop_command
from jarvis_arm.drivers.mocks import LinearProjection
from jarvis_arm.drivers.servo_bus import _decode_signed
from jarvis_arm.perception.calibration import Intrinsics, pixel_to_table
from jarvis_arm.world.scene import Scene


@pytest.mark.parametrize("text,expected", [
    ("stop", True), ("STOP!", True), ("please stop now", True), ("halt", True),
    ("stop the arm", True), ("emergency", True),
    ("put the cup down", False), ("unstoppable", False), ("", False), (None, False),
])
def test_stop_detection(text, expected):
    assert is_stop_command(text) is expected


def test_joint_map_roundtrip(cfg):
    jm = JointMap(cfg.joints)
    for deg in (-90.0, -12.3, 0.0, 45.0, 100.0):
        t = jm.deg_to_ticks("elbow_flex", deg)
        assert abs(jm.ticks_to_deg("elbow_flex", t) - deg) < 0.1
    cfg.joints[0].sign = -1
    jm = JointMap(cfg.joints)
    assert jm.deg_to_ticks("shoulder_pan", 90) == 2048 - 1024


def test_joint_limits_refuse_not_clamp(cfg):
    joints = {j.name: j for j in cfg.joints}
    check_joint_targets({"elbow_flex": 10}, joints)
    with pytest.raises(SafetyViolation):
        check_joint_targets({"elbow_flex": 500}, joints)
    with pytest.raises(SafetyViolation):
        check_joint_targets({"elbow_flex": math.nan}, joints)
    with pytest.raises(SafetyViolation):
        check_joint_targets({"nope": 0}, joints)


def test_workspace(cfg):
    check_workspace(0.2, 0.0, 0.05, cfg.workspace)
    with pytest.raises(SafetyViolation):
        check_workspace(0.6, 0.0, 0.05, cfg.workspace)
    with pytest.raises(SafetyViolation):
        check_workspace(0.2, 0.0, -0.01, cfg.workspace)


def test_sign_magnitude_decode():
    assert _decode_signed(0x0064, 10) == 100
    assert _decode_signed(0x0464, 10) == -100


def test_pixel_to_table_straight_down():
    intr = Intrinsics(fx=1000, fy=1000, cx=640, cy=360)
    # Camera 0.3 m above (0.2, 0) looking straight down: cam +z -> base -z,
    # cam +x -> base -y, cam +y -> base -x.
    T = np.eye(4)
    T[:3, :3] = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
    T[:3, 3] = [0.2, 0.0, 0.3]
    assert pixel_to_table(640, 360, intr, T) == pytest.approx((0.2, 0.0))
    x, y = pixel_to_table(740, 360, intr, T)   # 100 px right -> 3 cm robot-right
    assert (x, y) == pytest.approx((0.2, -0.03))
    T_up = T.copy()
    T_up[:3, :3] = np.diag([1, 1, 1])          # looking up: ray never hits table
    assert pixel_to_table(640, 360, intr, T_up) is None


def test_linear_projection_roundtrip():
    p = LinearProjection()
    assert p.to_table(*p.to_pixel(0.25, -0.1)) == pytest.approx((0.25, -0.1))


def test_scene_merges_and_tracks():
    s = Scene()
    a = s.observe("red cup", 0.2, 0.0, 0.9)
    b = s.observe("red cup", 0.21, 0.01, 0.8)
    c = s.observe("red cup", 0.3, 0.1, 0.8)
    assert a.id == b.id != c.id
    s.mark_unseen_missing({a.id})
    assert s.get(c.id).status == "missing"


def test_controller_moves_and_halts(controller):
    r = controller.move_to_named("scan")
    assert r.ok, r.reason
    pos = controller.joint_positions()
    assert pos["elbow_flex"] == pytest.approx(60, abs=2)
    assert not controller.move_joints({"elbow_flex": 999}).ok   # refused
    controller.halt("test")
    r = controller.move_to_named("home")
    assert not r.ok and "halted" in r.reason
    controller.resume()
    assert controller.move_to_named("home").ok
    assert controller.is_alive()


def test_halt_mid_move_holds_position(cfg, controller):
    cfg.servo_bus.default_speed_dps = 20
    controller.move_joints({"shoulder_pan": 90}, wait=False)
    time.sleep(0.3)
    controller.halt("test")
    time.sleep(0.2)
    p1 = controller.joint_positions()["shoulder_pan"]
    time.sleep(0.3)
    p2 = controller.joint_positions()["shoulder_pan"]
    assert 0 < p1 < 30 and abs(p2 - p1) < 0.5


def test_overload_triggers_halt(controller, bus, cfg):
    bus.load_override[cfg.joints[1].id] = 95.0
    time.sleep(0.5)
    assert controller.halted and "overload" in controller.halt_reason
