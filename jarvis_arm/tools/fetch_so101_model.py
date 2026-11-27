"""Download the official SO-101 MuJoCo model (MJCF + meshes) for the simulator.

    python -m jarvis_arm.tools.fetch_so101_model            # -> third_party/so101_mjcf/

Files come from TheRobotStudio/SO-ARM100 (Apache-2.0) at a PINNED commit, so
the sim never changes under you when upstream does. Bump SO_ARM100_SHA on
purpose, then re-run the sim tests.
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

SO_ARM100_SHA = "141aea7bcac2744d92998e30268c413202a56f96"   # main @ 2026-11-20
BASE = f"https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/{SO_ARM100_SHA}"
MODEL_FILE = "so101_new_calib_camera.xml"   # new calibration (zero = mid-range) + 32x32 wrist cam mount
DEFAULT_DEST = Path("third_party/so101_mjcf")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "jarvis-arm-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch(dest: Path = DEFAULT_DEST, force: bool = False) -> Path:
    dest = Path(dest)
    (dest / "assets").mkdir(parents=True, exist_ok=True)
    xml_path = dest / MODEL_FILE
    xml = _get(f"{BASE}/Simulation/SO101/{MODEL_FILE}").decode()
    xml_path.write_text(xml)
    meshes = sorted(set(re.findall(r'<mesh\s+file="([^"]+)"', xml)))
    for i, m in enumerate(meshes, 1):
        if "/" in m or ".." in m:
            raise ValueError(f"unexpected mesh path {m!r}")
        out = dest / "assets" / m
        if out.exists() and not force:
            continue
        print(f"  [{i}/{len(meshes)}] {m}")
        out.write_bytes(_get(f"{BASE}/Simulation/SO101/assets/{m}"))
    (dest / "LICENSE").write_bytes(_get(f"{BASE}/LICENSE"))
    (dest / "SOURCE.txt").write_text(
        f"TheRobotStudio/SO-ARM100 @ {SO_ARM100_SHA}\nSimulation/SO101/{MODEL_FILE} + assets/\n"
        "Licensed Apache-2.0 (see LICENSE). Not modified; jarvis_arm adds a camera,\n"
        "table and objects at load time (jarvis_arm/drivers/sim.py).\n")
    print(f"SO-101 model ready: {xml_path} ({len(meshes)} meshes)")
    return xml_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--force", action="store_true", help="re-download meshes that already exist")
    a = ap.parse_args(argv)
    fetch(Path(a.dest), a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
