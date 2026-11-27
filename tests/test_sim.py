"""Simulator tests. Skipped unless MuJoCo is installed and the SO-101 model has
been fetched (python -m jarvis_arm.tools.fetch_so101_model)."""
import math
import time
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from jarvis_arm.arm.controller import ArmController  # noqa: E402
from jarvis_arm.config import load_config  # noqa: E402

if not Path(load_config().sim.model_path).exists():
    pytest.skip("SO-101 model not fetched", allow_module_level=True)

from jarvis_arm.drivers.sim import (  # noqa: E402
    SimCamera, SimGrounder, SimServoBus, SimToF, SimWorld, render_once,
)
from jarvis_arm.perception.grounding import ALL_OBJECTS  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    c.logging.dir = str(tmp_path)
    c.servo_bus.default_speed_dps = 180
    return c


@pytest.fixture
def world(cfg):
    w = SimWorld(cfg, realtime=False)
    yield w
    w.stop()


def _pose(world, cfg, name):
    for j, deg in zip(cfg.arm_joints, cfg.named_poses[name]):
        world.set_joint(j.name, math.radians(deg), also_ctrl=True)
    mujoco.mj_forward(world.model, world.data)


def test_config_joints_match_mjcf(world, cfg):
    m = world.model
    for j in cfg.joints:
        lo, hi = np.degrees(m.jnt_range[m.joint(j.name).id])
        assert lo - 0.5 <= j.min_deg and j.max_deg <= hi + 0.5, f"{j.name} limits outside MJCF range"


def test_named_poses_are_collision_free(world, cfg):
    for name in cfg.named_poses:
        _pose(world, cfg, name)
        assert world.data.ncon == 0, f"pose {name} is in collision"


def test_servo_bus_holds_and_moves_under_physics(world, cfg):
    bus = SimServoBus(world, cfg)
    jm = bus.jm
    world.step(0.5)                                    # gravity: home must hold
    home = dict(zip([j.name for j in cfg.arm_joints], cfg.named_poses["home"]))
    pos = bus.read_positions(jm.ids)
    for j in cfg.arm_joints:
        assert jm.ticks_to_deg(j.name, pos[j.id]) == pytest.approx(home[j.name], abs=3)
    pan = cfg.joints[0]
    bus.sync_write_goals({pan.id: jm.deg_to_ticks(pan.name, 30)}, speed=jm.dps_to_tick_speed(60))
    world.step(0.25)                                   # speed-limited: ~15 deg after 0.25 s
    mid = jm.ticks_to_deg(pan.name, bus.read_positions([pan.id])[pan.id])
    assert 5 < mid < 25
    world.step(1.0)
    assert jm.ticks_to_deg(pan.name, bus.read_positions([pan.id])[pan.id]) == pytest.approx(30, abs=1.5)


def test_torque_off_goes_limp(world, cfg):
    bus = SimServoBus(world, cfg)
    lift = cfg.joints[1]
    _pose(world, cfg, "scan")
    bus.set_torque([j.id for j in cfg.joints], True)
    world.step(0.3)
    before = world.joint_rad(lift.name)
    bus.set_torque([j.id for j in cfg.joints], False)
    world.step(1.0)
    assert abs(world.joint_rad(lift.name) - before) > math.radians(5)   # arm sags


def test_wrist_camera_sees_all_objects_from_scan(world, cfg):
    _pose(world, cfg, "scan")
    img = render_once(world)
    assert img.shape == (cfg.sim.render_height, cfg.sim.render_width, 3) and img.std() > 10
    dets = SimGrounder(world).ground(img, ALL_OBJECTS, boxes=True)
    assert {d.label for d in dets} == {o.label for o in cfg.sim.objects}
    for d in dets:
        x1, y1, x2, y2 = d.box_px
        assert x1 <= d.point_px[0] <= x2 and y1 <= d.point_px[1] <= y2


def test_pixel_to_table_error_is_height_bias(world, cfg):
    """Projecting an object's CENTRE pixel onto the table plane overshoots by
    about half_height * tan(view angle). Documented in docs/sim_guide.md."""
    _pose(world, cfg, "scan")
    img = render_once(world)
    truth = {o.label: o for o in cfg.sim.objects}
    for d in SimGrounder(world).ground(img, ALL_OBJECTS):
        x, y = world.pixel_to_table(*d.point_px)
        o = truth[d.label]
        err = math.hypot(x - o.pos[0], y - o.pos[1])
        assert err < 1.2 * o.half_height, f"{d.label}: {err * 100:.1f} cm"
    # A point ON the table projects back exactly (the math itself is right).
    T, intr = world.camera_pose_cv(), world.intrinsics()
    p = np.array([0.22, 0.01, cfg.workspace.table_z_m])
    pc = T[:3, :3].T @ (p - T[:3, 3])
    u, v = intr.fx * pc[0] / pc[2] + intr.cx, intr.fy * pc[1] / pc[2] + intr.cy
    assert world.pixel_to_table(u, v) == pytest.approx((0.22, 0.01), abs=1e-6)


def test_grounder_skips_occluded_and_offscreen(world, cfg):
    _pose(world, cfg, "home")                          # camera points at the floor near the base
    dets = SimGrounder(world).ground(render_once(world), ALL_OBJECTS)
    assert len(dets) < len(cfg.sim.objects)


def test_tof_sees_table_and_object(world, cfg):
    _pose(world, cfg, "scan")
    tof = SimToF(world)
    mm = tof.read_mm()
    assert mm is not None and 100 < mm < 400           # table, ~20-30 cm away along the tilted axis


def test_controller_and_camera_end_to_end(cfg):
    world = SimWorld(cfg)
    bus, cam = SimServoBus(world, cfg), SimCamera(world)
    world.start()
    ctl = ArmController(bus, cfg)
    ctl.start()
    cam.start()
    try:
        r = ctl.move_to_named("scan")
        assert r.ok, r.reason
        time.sleep(0.5)
        f = cam.latest()
        assert f is not None and f.image.shape[2] == 3 and cam.frames_rendered > 0
        assert ctl.is_alive()
        ctl.halt("test")
        time.sleep(0.2)
        a = world.joint_rad("shoulder_lift")
        time.sleep(0.5)
        assert abs(world.joint_rad("shoulder_lift") - a) < math.radians(1)   # torque-hold
    finally:
        ctl.shutdown()
        cam.stop()
        world.stop()
