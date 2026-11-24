# tools/

Bring-up and calibration scripts (run as `python -m jarvis_arm.tools.<name>`). Planned:

- `servo_setup`: ping/scan the bus, set servo IDs one at a time, sweep the full range, check for slop
- `joint_calib`: record `offset_ticks`, `sign`, and limits per joint into config.yaml
- `camera_calib`: checkerboard intrinsics → `calib/intrinsics.yaml`
