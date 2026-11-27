"""Try Gemini Robotics-ER grounding on photos or a webcam frame.

    python -m jarvis_arm.tools.ground_demo desk1.jpg desk2.jpg            # "all objects"
    python -m jarvis_arm.tools.ground_demo desk1.jpg -q "red cup" -q mug  # specific queries
    python -m jarvis_arm.tools.ground_demo --camera 0 -q "blue bowl"      # one webcam frame
    python -m jarvis_arm.tools.ground_demo photos/ --boxes --thinking medium

Prints each detection and writes an annotated copy per image+query to --out
(default logs/grounding_demo/). Every call is also logged in full under
logs/grounding/<id>/ (exact uploaded JPEG, prompt, raw response, timings).
Headless-safe: no windows are opened.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic"}


def _images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXT)
        else:
            out.append(p)
    return out


def _webcam_frame(index: int):
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        sys.exit(f"camera {index} not available")
    try:
        for _ in range(10):            # let auto-exposure settle
            cap.read()
            time.sleep(0.05)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        sys.exit("camera read failed")
    return img


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "all"


def main(argv: list[str] | None = None) -> int:
    import cv2

    from ..config import load_config
    from ..perception.annotate import draw_detections
    from ..perception.grounding import ALL_OBJECTS, GeminiGrounder, GroundingError

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files or folders")
    ap.add_argument("-q", "--query", action="append", help="object description (repeatable); default: all objects")
    ap.add_argument("--camera", type=int, help="grab one frame from this webcam index instead of files")
    ap.add_argument("--boxes", action="store_true", help="ask for bounding boxes instead of points")
    ap.add_argument("--thinking", choices=["minimal", "low", "medium", "high"], help="override thinking_level")
    ap.add_argument("--config", help="path to config.yaml")
    ap.add_argument("--out", default="logs/grounding_demo")
    a = ap.parse_args(argv)

    if not a.images and a.camera is None:
        ap.error("give image paths or --camera N")
    load_dotenv(find_dotenv(usecwd=True))
    cfg = load_config(a.config)
    if a.thinking:
        cfg.grounding.thinking_level = a.thinking
    grounder = GeminiGrounder(cfg.grounding, os.getenv("GEMINI_API_KEY"),
                              log_dir=Path(cfg.logging.dir) / "grounding")

    frames: list[tuple[str, object]] = []
    if a.camera is not None:
        frames.append((f"camera{a.camera}-{time.strftime('%H%M%S')}", _webcam_frame(a.camera)))
    for p in _images(a.images):
        img = cv2.imread(str(p))
        if img is None:
            print(f"!! cannot read {p} (HEIC needs converting to JPEG first)")
            continue
        frames.append((p.stem, img))

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = a.query or [ALL_OBJECTS]
    failures = 0
    for name, img in frames:
        h, w = img.shape[:2]
        for q in queries:
            shown = "all objects" if q == ALL_OBJECTS else q
            print(f"\n== {name} ({w}x{h})  query: {shown!r}  [{cfg.grounding.thinking_level}]")
            try:
                dets = grounder.ground(img, q, boxes=a.boxes)
            except GroundingError as e:
                failures += 1
                print(f"   ERROR: {e}")
                continue
            rec = grounder.last_record
            print(f"   {len(dets)} hit(s) in {rec.latency_s:.2f}s  tokens={rec.usage}")
            for i, d in enumerate(dets):
                box = f"  box={tuple(round(v) for v in d.box_px)}" if d.box_px else ""
                print(f"   {i}: {d.label:<28} at (u={d.point_px[0]:.0f}, v={d.point_px[1]:.0f}){box}")
            for wmsg in rec.warnings:
                print(f"   warning: {wmsg}")
            dst = out_dir / f"{name}__{_slug(shown)}{'__boxes' if a.boxes else ''}.jpg"
            cv2.imwrite(str(dst), draw_detections(img, dets))
            print(f"   annotated -> {dst}")
            print(f"   full log  -> {grounder.last_log_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
