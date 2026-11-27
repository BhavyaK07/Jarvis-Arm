# MuJoCo simulator guide

The simulator is a fake set of hardware with real physics. It implements the same
interfaces as the real servo bus, camera and ToF sensor, so everything above
`drivers/` (controller, safety, skills, task runner, agent) runs **unchanged** on it.
You can develop kinematics, calibration math and the agent loop before the arm exists,
and find bugs that would otherwise show up on the real arm.

## 1. Setup

```bash
source .venv/bin/activate
pip install -e '.[sim]'                          # MuJoCo + OpenCV
python -m jarvis_arm.tools.fetch_so101_model     # official SO-101 model -> third_party/so101_mjcf/ (16 MB, gitignored)
pytest tests/test_sim.py                         # 9 checks; all should pass
```

Then try it:

```bash
python -m jarvis_arm.tools.sim_view                          # 3D window; move joints with the sliders
python -m jarvis_arm.tools.sim_view --pose scan --snapshot logs/sim   # images + numbers, no window
python -m jarvis_arm --sim --text --offline                  # the whole robot, on physics
mjpython -m jarvis_arm --sim --text --offline --viewer       # same, with a live 3D window (macOS needs mjpython)
python -m jarvis_arm --sim --text --offline --gemini-vision  # real Gemini on rendered frames
```

In the robot, try `what do you see`, `go home`, `stop`, `resume`. `pick` reports
"needs IK", because inverse kinematics is the next milestone.

## 2. MuJoCo in six ideas

**1. The model is an XML file (MJCF).** A tree of **bodies** (rigid parts), each with:
- **joints**: how it moves relative to its parent (`hinge` = revolute)
- **geoms**: shapes for collision and drawing (meshes, boxes, cylinders)
- **sites**: named reference points (`gripperframe` = the gripper tip)
- **cameras**

**Actuators** drive joints. Our file is `so101_new_calib_camera.xml`, exported from
the SO-101 CAD by TheRobotStudio. Open it and read the `<body>` nesting:
base → shoulder → upper_arm → lower_arm → wrist → gripper → moving_jaw.

**2. `MjModel` vs `MjData`.** `MjModel` is everything that doesn't change: geometry,
masses, joint ranges, gains. `MjData` is the current state:
- `qpos` joint angles, `qvel` velocities, `ctrl` actuator commands
- `xpos`/`xmat` where every body is, `contact` current contacts

You change the robot through `data`, and occasionally tune `model`.

**3. Two functions do almost everything.**
- `mj_forward(model, data)` computes everything from the current `qpos` (positions,
  contacts, sensor values) **without** advancing time. This is forward kinematics:
  set the angles, call it, read `data.site("gripperframe").xpos`.
- `mj_step(model, data)` advances physics by one `timestep` (2 ms here): forces,
  contacts, integration. Our physics thread calls it 500 times per second.

**4. Position actuators are servos.** `<position kp=998 kv=2.7 forcerange=±2.94>`
applies `torque = kp·(ctrl − qpos) − kv·qvel`, clipped to ±2.94 N·m. That limit
equals the 12V STS3215's 30 kg·cm stall torque, so the sim arm is exactly as strong
as yours. Setting `ctrl` is like writing Goal Position.

**5. Cameras look down their own −Z axis, with +Y up.** OpenCV (and our
calibration math) uses +Z forward and +Y down, so converting between them flips two
axes: `R_cv = R_mujoco · diag(1, −1, −1)`. That one line is in
`SimWorld.camera_pose_cv`. `mujoco.Renderer` renders any camera to a numpy image.

**6. Rays and contacts answer "what's there?"** `mj_ray` returns the distance to the
first geom along a ray (our ToF sensor, and the occlusion check).
`data.contact[:data.ncon]` lists everything currently touching (how we checked the
named poses are collision-free).

## 3. How the sim plugs into the robot

```
skills / task runner / agent      (unchanged)
          │
     ArmController ── single bus-owner thread, halt, watchdog   (unchanged)
          │ ticks
   ┌──────┴──────────────────────────── drivers/sim.py ────────────────────────────┐
   │ SimServoBus   ticks → joint deg (JointMap) → radians → setpoint, moved toward │
   │               the goal at the commanded speed each 2 ms → data.ctrl          │
   │               torque off = actuator gains set to 0 (the arm goes limp)        │
   │               load % = actuator_force / forcerange                            │
   │ SimCamera     own thread, mujoco.Renderer on "wrist_cam", 10 fps, BGR frames  │
   │ SimToF        mj_ray from the palm along the gripper's approach axis          │
   │ SimGrounder   ground truth: projects each object's centre into the image,     │
   │               skips off-screen / occluded ones (free, instant)                │
   │ SimWorld      MjModel+MjData, physics thread (real time), one lock,           │
   │               exact camera pose + intrinsics → the REAL pixel_to_table()      │
   └───────────────────────────────────────────────────────────────────────────────┘
```

`build_model()` loads the official SO-101 file unmodified and adds a table, lights,
the `wrist_cam` camera and the objects from `config.yaml` (`sim.objects`), using
MuJoCo's `MjSpec` API, which edits a model in Python instead of XML.

**Why a separate render thread?** The real camera delivers frames on its own thread,
and the sim behaves the same way, so timing bugs show up here. The scan skill now
waits for a frame captured *after* the arm stopped. A frame from mid-move would put
objects in the wrong place, on the real arm too.

## 4. Conventions (the sim defines them)

| Item | Convention |
|---|---|
| Base frame | origin at the base, **+X forward**, +Y = robot's left, +Z up; table at z = 0 |
| Joint zero | middle of each joint's range (SO-101 "new calibration") |
| Signs | +pan turns toward −Y (right); +lift / +elbow / +wrist_flex move the tip down; +gripper opens |
| Units | config and skills use degrees, MuJoCo uses radians, the bus uses ticks (4096/rev) |

When the real arm arrives, joint calibration finds each servo's `offset_ticks` and
`sign` so that **the real arm's angles match the sim's**. After that, any pose you
develop in sim means the same thing on hardware.

## 5. What the sim has already caught

1. **The wrist camera was inside its own housing.** The camera mesh's origin isn't the
   lens. I measured the mesh and moved the camera to its front face.
2. **Tall objects are estimated too far away.** A detection marks the middle of the
   visible object, but `pixel_to_table` intersects that ray with the *table* plane:

   ```
   camera
      \
       \  ray through the cup's centre pixel
        \
      ┌──x──┐ ← the point the detector picked (half-height up)
      │ cup │
      └─────┘\
   ──────────── x ← where we think the cup is (too far)
   ```

   Error ≈ half-height × tan(viewing angle): 2.3 cm for the 8 cm cup, 0.5 cm for the
   3 cm block. Possible fixes (for the calibration milestone): intersect at an
   estimated object height, use the bottom of the bounding box, or let the ToF
   correct it on approach. `test_pixel_to_table_error_is_height_bias` pins it down.
3. **The overload watchdog was trigger-happy.** On fast moves the elbow briefly hits
   100% torque while accelerating (real servos do too), and one reading halted the
   arm. It now requires `overload_hold_s` (0.6 s) of sustained overload.
4. **The arm is small.** The camera tops out about 20 cm above the table. The scan
   pose `[0, −25, 5, 85, 90]` puts it at ~18 cm, tilted 25°, centred on (0.23, 0).
   With roll at 0 the camera (offset 7 cm from the gripper axis) looks off to the
   side, which is why the scan pose rolls the wrist 90°.
5. **The old home pose put the gripper on the table.** The new `home`
   `[0, −80, 60, 90, 0]` is folded, collision-free and clear of the objects.

## 6. Exercises (in order)

1. **Drive it by hand.** `python -m jarvis_arm.tools.sim_view`, open the Control
   panel on the right, and drag the sliders. Find a pose where the gripper points
   straight down at (0.25, 0), and read the angles in the Joints panel.
2. **Forward kinematics in 5 lines.**
   ```python
   import math, mujoco
   from jarvis_arm.config import load_config
   from jarvis_arm.drivers.sim import SimWorld
   w = SimWorld(load_config(), realtime=False)
   for name, deg in zip(["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"], [0,-25,5,85,90]):
       w.set_joint(name, math.radians(deg))
   mujoco.mj_forward(w.model, w.data); print(w.data.site("gripperframe").xpos)
   ```
   Change one angle at a time and predict which way the tip moves before you run it.
3. **Change the scene.** Add an object under `sim.objects` in `config.yaml` (say a
   yellow ball, `shape: sphere`), then run `sim_view --pose scan --snapshot logs/sim`.
   Does the camera see it?
4. **Change the lens.** Set `camera_fovy_deg` to 50 and then 90, and compare `wrist.png`.
   This shows how much a wide lens matters for "one scan sees everything".
5. **Kinematics milestone (next):** write your own FK for the 5 arm joints from the
   link offsets in the MJCF, and test it against `mj_forward` for 1000 random poses
   (must agree to < 1 mm). Then IK: (x, y, z, pitch) → angles, refusing unreachable
   targets. MuJoCo is the answer key.

## 7. What the sim does NOT tell you

- **Servo dynamics are approximate.** The gains come from a community estimate. Real
  servos have backlash, deadband and slop (test yours on arrival).
- **Rendered images are cleaner than real ones:** no blur, noise, glare or clutter.
  Use real photos for perception (the grounding demo).
- **Soft TPU fingers aren't modelled:** the jaw is rigid. Grasp success in sim is
  optimistic.
- **Calibration is perfect in sim.** The camera pose is known exactly; on the real arm
  it comes from FK plus a measured wrist→camera transform, with error.
- **No cables, table flex or base wobble.**

Treat the sim as the place to get geometry, logic and timing right, then measure
everything again on the real arm.
