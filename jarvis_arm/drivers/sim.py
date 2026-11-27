"""MuJoCo simulator backend.

Implements the SAME interfaces as the real drivers, so everything above
drivers/ (controller, skills, task runner, agent) runs unchanged on physics:

    SimServoBus  -> ServoBus   ticks in/out, goal speed, torque on/off, load telemetry
    SimCamera    -> Camera     rendered wrist-camera frames (BGR, latest-frame thread)
    SimToF       -> ToF        ray cast from the palm along the approach axis
    SimGrounder  -> Grounder   ground-truth "detections" (free, instant, no API)
    SimWorld.pixel_to_table    the REAL ray/table-plane math, fed with the sim's
                               exact camera pose + intrinsics

See docs/sim_guide.md for how it works.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")   # offscreen GL that works off the main thread

import mujoco
import numpy as np

from ..arm.kinematics import DEG_PER_TICK, JointMap
from ..config import Config, SimCfg, SimObjectCfg
from ..perception.calibration import Intrinsics, pixel_to_table
from ..perception.grounding import ALL_OBJECTS, Detection
from .camera import Frame
from .servo_bus import ServoTelemetry

log = logging.getLogger(__name__)

WRIST_CAM = "wrist_cam"
OVERVIEW_CAM = "overview"
OBJECT_GEOM_GROUP = 0          # table + objects; the robot's collision meshes are group 3
# Wrist camera in the `gripper` body frame: at the 32x32 module's lens on the hex-nut
# mount, looking along the approach axis (gripper -Z), image-up = +Y (jaws at
# the bottom of the image). Identity orientation because MuJoCo cameras look
# along their own -Z with +Y up.
WRIST_CAM_POS = (0.0025, 0.067, -0.021)   # just in front of the lens
PALM_OFFSET_M = 0.03           # ToF sits this far along the approach axis from the gripper origin


def _lookat_quat(pos, target, up=(0, 0, 1)) -> np.ndarray:
    z = np.asarray(pos, float) - np.asarray(target, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, y, z]).flatten())
    return q


# --------------------------------------------------------------------- model
def build_model(cfg: Config) -> mujoco.MjModel:
    """Official SO-101 MJCF + our table, lights, wrist camera and objects."""
    sc = cfg.sim
    path = Path(sc.model_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run: python -m jarvis_arm.tools.fetch_so101_model")
    spec = mujoco.MjSpec.from_file(str(path))
    spec.visual.global_.offwidth = max(1280, sc.render_width)
    spec.visual.global_.offheight = max(960, sc.render_height)
    wb = spec.worldbody
    tz = cfg.workspace.table_z_m

    wb.add_light(name="key", pos=[0.3, -0.3, 1.2], dir=[-0.2, 0.2, -1], castshadow=True,
                 diffuse=[0.7, 0.7, 0.7])
    wb.add_light(name="fill", pos=[-0.4, 0.4, 1.0], dir=[0.3, -0.3, -1], castshadow=False,
                 diffuse=[0.35, 0.35, 0.35])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.05],
                pos=[0, 0, tz], rgba=[0.66, 0.55, 0.42, 1], group=OBJECT_GEOM_GROUP,
                friction=[1.0, 0.005, 0.0001])

    spec.body("gripper").add_camera(name=WRIST_CAM, pos=list(WRIST_CAM_POS), quat=[1, 0, 0, 0],
                                    fovy=sc.camera_fovy_deg)
    ov_pos, ov_target = [0.62, -0.42, 0.42], [0.18, 0.0, 0.08]
    wb.add_camera(name=OVERVIEW_CAM, pos=ov_pos, quat=list(_lookat_quat(ov_pos, ov_target)), fovy=45)

    shapes = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
    for i, o in enumerate(sc.objects):
        b = wb.add_body(name=f"sim_obj{i}", pos=[o.pos[0], o.pos[1], tz + o.half_height + 1e-4])
        b.add_freejoint()
        size = list(o.size) + [0.0] * (3 - len(o.size))
        b.add_geom(name=f"sim_obj{i}_geom", type=shapes[o.shape], size=size, rgba=list(o.rgba),
                   mass=o.mass, group=OBJECT_GEOM_GROUP, friction=[1.2, 0.01, 0.001], condim=4)
    return spec.compile()


# --------------------------------------------------------------------- world
class SimWorld:
    """Owns MjModel/MjData and steps physics in a background thread.

    Everything that touches `data` holds `self.lock`. Drivers register
    pre-step callbacks (e.g. the servo bus moving its setpoints)."""

    def __init__(self, cfg: Config, realtime: bool | None = None):
        self.cfg = cfg
        self.sc: SimCfg = cfg.sim
        self.model = build_model(cfg)
        self.data = mujoco.MjData(self.model)
        self.lock = threading.RLock()
        self.realtime = self.sc.realtime if realtime is None else realtime
        self._prestep: list[Callable[[float], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._viewer = None
        self.sim_time_offset = 0.0
        self.objects: list[tuple[SimObjectCfg, int]] = [
            (o, self.model.body(f"sim_obj{i}").id) for i, o in enumerate(self.sc.objects)]
        # Start at home with the gripper open, so nothing starts in collision.
        home = cfg.named_poses.get("home")
        if home is not None:
            for j, deg in zip(cfg.arm_joints, home):
                self.set_joint(j.name, math.radians(deg), also_ctrl=True)
        self.set_joint("gripper", math.radians(cfg.gripper.open_deg), also_ctrl=True)
        mujoco.mj_forward(self.model, self.data)

    # -------------------------------------------------------------- helpers
    def set_joint(self, name: str, rad: float, also_ctrl: bool = False) -> None:
        with self.lock:
            self.data.qpos[self.model.joint(name).qposadr[0]] = rad
            if also_ctrl:
                self.data.ctrl[self.model.actuator(name).id] = rad

    def joint_rad(self, name: str) -> float:
        with self.lock:
            return float(self.data.qpos[self.model.joint(name).qposadr[0]])

    def add_prestep(self, fn: Callable[[float], None]) -> None:
        self._prestep.append(fn)

    def step(self, seconds: float) -> None:
        """Advance physics by `seconds` (used by tests / non-realtime runs)."""
        dt = self.model.opt.timestep
        with self.lock:
            for _ in range(max(1, round(seconds / dt))):
                for fn in self._prestep:
                    fn(dt)
                mujoco.mj_step(self.model, self.data)

    # ----------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="sim-physics", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._viewer is not None:
            self._viewer.close()

    def _run(self) -> None:
        chunk = 0.01   # 10 ms of sim per lock hold
        t_wall = time.monotonic()
        while not self._stop.is_set():
            self.step(chunk)
            if self._viewer is not None:
                with self.lock:
                    self._viewer.sync()
            if self.realtime:
                t_wall += chunk
                lag = t_wall - time.monotonic()
                if lag > 0:
                    time.sleep(lag)
                elif lag < -0.5:          # fell far behind (debugger, slow machine): resync
                    t_wall = time.monotonic()

    def launch_viewer(self) -> None:
        """Live 3D window. On macOS the process must be started with `mjpython`."""
        import mujoco.viewer
        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ------------------------------------------------------------ cameras
    def intrinsics(self, width: int | None = None, height: int | None = None) -> Intrinsics:
        w, h = width or self.sc.render_width, height or self.sc.render_height
        fovy = math.radians(self.model.cam(WRIST_CAM).fovy[0])
        f = (h / 2) / math.tan(fovy / 2)
        return Intrinsics(fx=f, fy=f, cx=w / 2, cy=h / 2)

    def camera_pose_cv(self, cam: str = WRIST_CAM) -> np.ndarray:
        """4x4 pose of the camera in the base(=world) frame, OpenCV axes
        (+x right, +y down, +z forward). MuJoCo cameras use +y up, -z forward."""
        with self.lock:
            c = self.data.cam(cam)
            R = c.xmat.reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, c.xpos
            return T

    def pixel_to_table(self, u: float, v: float) -> tuple[float, float] | None:
        return pixel_to_table(u, v, self.intrinsics(), self.camera_pose_cv(), self.cfg.workspace.table_z_m)

    # ------------------------------------------------------------ queries
    def object_states(self) -> list[dict]:
        """Ground truth: label, centre position, half-height, body id."""
        with self.lock:
            return [{"label": o.label, "pos": self.data.xpos[bid].copy(),
                     "R": self.data.xmat[bid].reshape(3, 3).copy(), "cfg": o, "body": bid}
                    for o, bid in self.objects]

    def ray_m(self, origin, direction, groups=(OBJECT_GEOM_GROUP,)) -> tuple[float, int]:
        """Distance (m) to the first geom in `groups` along a ray; (-1, -1) if none."""
        gg = np.zeros(6, np.uint8)
        for g in groups:
            gg[g] = 1
        geomid = np.full(1, -1, np.int32)
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        with self.lock:
            dist = mujoco.mj_ray(self.model, self.data, np.asarray(origin, float), d, gg, 1, -1, geomid)
        return float(dist), int(geomid[0])


# --------------------------------------------------------------------- servos
class SimServoBus:
    """ServoBus on top of MuJoCo position actuators.

    Like an STS3215: the bus moves an internal setpoint toward the goal at the
    commanded speed (ticks/s; 0 = as fast as possible) and the actuator's PD
    loop tracks it. Torque off zeroes the actuator gains, so the joint goes limp.
    """

    def __init__(self, world: SimWorld, cfg: Config):
        self.w = world
        self.jm = JointMap(cfg.joints)
        m = world.model
        self.name_by_id = {j.id: j.name for j in cfg.joints}
        self.act = {n: m.actuator(n).id for n in self.jm.names}
        self.ctrlrange = {n: tuple(m.actuator_ctrlrange[self.act[n]]) for n in self.jm.names}
        self.force_max = {n: float(abs(m.actuator_forcerange[self.act[n]]).max()) for n in self.jm.names}
        self._gain = {n: m.actuator_gainprm[self.act[n]].copy() for n in self.jm.names}
        self._bias = {n: m.actuator_biasprm[self.act[n]].copy() for n in self.jm.names}
        self._sp = {n: world.joint_rad(n) for n in self.jm.names}
        self._goal = dict(self._sp)
        self._speed = {n: 0.0 for n in self.jm.names}   # rad/s, 0 = max
        self._torque = {n: True for n in self.jm.names}
        self.is_open = False
        world.add_prestep(self._update)

    def _update(self, dt: float) -> None:      # called under world.lock
        for n, a in self.act.items():
            if not self._torque[n]:
                continue
            err = self._goal[n] - self._sp[n]
            step = self._speed[n] * dt if self._speed[n] > 0 else abs(err)
            self._sp[n] += max(-step, min(step, err))
            self.w.data.ctrl[a] = self._sp[n]

    def open(self) -> None:
        self.is_open = True

    def close(self) -> None:
        self.is_open = False

    def ping(self, servo_id: int) -> bool:
        return servo_id in self.name_by_id

    def set_torque(self, ids, enable: bool) -> None:
        m = self.w.model
        with self.w.lock:
            for i in ids:
                n = self.name_by_id[i]
                a = self.act[n]
                if enable:
                    m.actuator_gainprm[a], m.actuator_biasprm[a] = self._gain[n], self._bias[n]
                    self._sp[n] = self._goal[n] = self.w.joint_rad(n)
                else:
                    m.actuator_gainprm[a] = 0
                    m.actuator_biasprm[a] = 0
                self._torque[n] = enable

    def sync_write_goals(self, goals: dict[int, int], speed: int) -> None:
        with self.w.lock:
            for i, ticks in goals.items():
                n = self.name_by_id[i]
                lo, hi = self.ctrlrange[n]
                self._goal[n] = min(hi, max(lo, math.radians(self.jm.ticks_to_deg(n, ticks))))
                self._speed[n] = math.radians(speed * DEG_PER_TICK)

    def read_positions(self, ids) -> dict[int, int]:
        return {i: self.jm.deg_to_ticks(self.name_by_id[i], math.degrees(self.w.joint_rad(self.name_by_id[i])))
                for i in ids}

    def read_telemetry(self, ids) -> dict[int, ServoTelemetry]:
        out = {}
        with self.w.lock:
            for i in ids:
                n = self.name_by_id[i]
                force = float(self.w.data.actuator_force[self.act[n]])
                out[i] = ServoTelemetry(
                    position=self.jm.deg_to_ticks(n, math.degrees(self.w.joint_rad(n))),
                    load_pct=100.0 * force / self.force_max[n], voltage_v=12.0, temp_c=35.0)
        return out


# --------------------------------------------------------------------- camera
class SimCamera:
    """Renders the wrist camera in its own thread, like a real latest-frame camera."""

    def __init__(self, world: SimWorld, cam: str = WRIST_CAM):
        self.w, self.cam = world, cam
        self.width, self.height = world.sc.render_width, world.sc.render_height
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.frames_rendered = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sim-camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        renderer = mujoco.Renderer(self.w.model, self.height, self.width)   # GL context lives in this thread
        period = 1.0 / self.w.sc.fps
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                with self.w.lock:
                    renderer.update_scene(self.w.data, camera=self.cam)   # snapshot of the scene
                rgb = renderer.render()
                with self._lock:
                    self._frame = Frame(np.ascontiguousarray(rgb[:, :, ::-1]), time.monotonic())
                self.frames_rendered += 1
                time.sleep(max(0.0, period - (time.monotonic() - t0)))
        finally:
            renderer.close()

    def latest(self, max_age_s: float = 1.0) -> Frame | None:
        with self._lock:
            f = self._frame
        if f is None or time.monotonic() - f.t > max_age_s:
            return None
        return f


def render_once(world: SimWorld, cam: str = WRIST_CAM, width: int | None = None,
                height: int | None = None) -> np.ndarray:
    """One BGR image from `cam` (tools/tests; call from a single thread)."""
    w, h = width or world.sc.render_width, height or world.sc.render_height
    r = mujoco.Renderer(world.model, h, w)
    try:
        with world.lock:
            r.update_scene(world.data, camera=cam)
        return np.ascontiguousarray(r.render()[:, :, ::-1])
    finally:
        r.close()


# ------------------------------------------------------------------------ ToF
class SimToF:
    """VL53L1X stand-in: ray from the palm along the approach axis (gripper -Z),
    hitting only table/objects (not the robot's own jaws)."""

    MAX_RANGE_MM = 1300   # VL53L1X short-distance mode

    def __init__(self, world: SimWorld):
        self.w = world
        self.gripper = world.model.body("gripper").id

    def read_mm(self) -> int | None:
        with self.w.lock:
            R = self.w.data.xmat[self.gripper].reshape(3, 3)
            approach = -R[:, 2]
            origin = self.w.data.xpos[self.gripper] + PALM_OFFSET_M * approach
        dist, _ = self.w.ray_m(origin, approach)
        if dist < 0 or dist * 1000 > self.MAX_RANGE_MM:
            return None
        return int(round(dist * 1000))


# ------------------------------------------------------------------ grounding
class SimGrounder:
    """Ground-truth grounding: projects each matching object's CENTRE into the
    wrist camera (roughly where a model points: middle of the visible blob).
    Skips objects that are off-screen or hidden behind something."""

    def __init__(self, world: SimWorld, cam: str = WRIST_CAM):
        self.w, self.cam = world, cam

    def _project(self, p_world, T, intr) -> tuple[float, float] | None:
        pc = T[:3, :3].T @ (np.asarray(p_world) - T[:3, 3])
        if pc[2] <= 1e-6:
            return None
        return intr.fx * pc[0] / pc[2] + intr.cx, intr.fy * pc[1] / pc[2] + intr.cy

    def ground(self, image: np.ndarray, query: str, boxes: bool = False) -> list[Detection]:
        h, w = image.shape[:2]
        intr = self.w.intrinsics(w, h)
        T = self.w.camera_pose_cv(self.cam)
        q = set(query.lower().split())
        out = []
        for o in self.w.object_states():
            if query != ALL_OBJECTS and not q & set(o["label"].lower().split()):
                continue
            uv = self._project(o["pos"], T, intr)
            if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                continue
            # occlusion: first thing the ray from the camera hits must be this object
            ray = o["pos"] - T[:3, 3]
            dist, geom = self.w.ray_m(T[:3, 3], ray, groups=(OBJECT_GEOM_GROUP, 3))
            if geom >= 0 and self.w.model.geom_bodyid[geom] != o["body"] and dist < np.linalg.norm(ray) - 1e-3:
                continue
            box = None
            if boxes:
                c = o["cfg"]
                hx, hy, hz = (c.size if c.shape == "box" else
                              [c.size[0], c.size[0], c.half_height])
                corners = [o["pos"] + o["R"] @ np.array([sx * hx, sy * hy, sz * hz])
                           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                pts = [p for p in (self._project(cn, T, intr) for cn in corners) if p]
                if pts:
                    us, vs = zip(*pts)
                    box = (max(0, min(us)), max(0, min(vs)), min(w, max(us)), min(h, max(vs)))
            out.append(Detection(o["label"], uv, box, 1.0))
        return out
