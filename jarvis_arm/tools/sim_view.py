"""Explore the SO-101 simulation.

Interactive 3D viewer (drag the actuator sliders in the right-hand panel to move joints):
    python -m jarvis_arm.tools.sim_view
    python -m jarvis_arm.tools.sim_view --pose scan

Headless snapshots + numbers (no window; works over SSH):
    python -m jarvis_arm.tools.sim_view --pose scan --snapshot logs/sim
    python -m jarvis_arm.tools.sim_view --pose "0,-25,5,85,90" --snapshot logs/sim

Prints where the gripper tip and the wrist camera are (forward kinematics computed
by MuJoCo), whether anything is in collision, and the ToF reading; writes
overview.png (fixed external camera) and wrist.png (what the robot sees).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


def _parse_pose(text: str, cfg) -> list[float]:
    if text in cfg.named_poses:
        return list(cfg.named_poses[text])
    vals = [float(v) for v in text.split(",")]
    if len(vals) != len(cfg.arm_joints):
        raise SystemExit(f"pose needs {len(cfg.arm_joints)} comma-separated degrees "
                         f"({', '.join(j.name for j in cfg.arm_joints)}) or one of {sorted(cfg.named_poses)}")
    return vals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="home", help="named pose or comma-separated joint degrees")
    ap.add_argument("--gripper", type=float, help="gripper degrees (default: open)")
    ap.add_argument("--snapshot", metavar="DIR", help="save images + print numbers instead of opening a window")
    ap.add_argument("--config")
    a = ap.parse_args(argv)

    import mujoco

    from ..config import load_config
    from ..drivers.sim import OVERVIEW_CAM, SimToF, SimWorld, render_once

    cfg = load_config(a.config)
    world = SimWorld(cfg, realtime=False)
    for j, deg in zip(cfg.arm_joints, _parse_pose(a.pose, cfg)):
        world.set_joint(j.name, math.radians(deg), also_ctrl=True)
    if a.gripper is not None:
        world.set_joint("gripper", math.radians(a.gripper), also_ctrl=True)
    m, d = world.model, world.data
    mujoco.mj_forward(m, d)

    if not a.snapshot:
        import mujoco.viewer
        print("Viewer controls: double-click a body to select; Ctrl+right-drag to push it;\n"
              "right panel > Control: sliders move the joints (ctrl = target angle, radians).")
        mujoco.viewer.launch(m, d)
        return 0

    import cv2
    import numpy as np

    out = Path(a.snapshot)
    out.mkdir(parents=True, exist_ok=True)
    tip = d.site("gripperframe").xpos
    cam = world.camera_pose_cv()
    look = cam[:3, 2]
    print(f"pose {a.pose}: " + ", ".join(f"{j.name}={math.degrees(world.joint_rad(j.name)):.0f}"
                                         for j in cfg.joints))
    print(f"gripper tip      x={tip[0]:.3f} y={tip[1]:.3f} z={tip[2]:.3f} m")
    print(f"wrist camera at  x={cam[0, 3]:.3f} y={cam[1, 3]:.3f} z={cam[2, 3]:.3f} m, "
          f"looking {math.degrees(math.acos(max(-1, min(1, -look[2])))):.0f} deg from straight down")
    hit = world.pixel_to_table(cfg.sim.render_width / 2, cfg.sim.render_height / 2)
    if hit:
        print(f"image centre hits the table at x={hit[0]:.3f} y={hit[1]:.3f}")
    contacts = {tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)))
                for c in d.contact[:d.ncon]}
    print(f"contacts: {sorted(contacts) or 'none'}")
    print(f"ToF: {SimToF(world).read_mm()} mm")
    cv2.imwrite(str(out / "overview.png"), render_once(world, OVERVIEW_CAM, 800, 600))
    cv2.imwrite(str(out / "wrist.png"), render_once(world))
    print(f"wrote {out / 'overview.png'} and {out / 'wrist.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
