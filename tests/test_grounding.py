import json
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis_arm.config import GroundingCfg
from jarvis_arm.perception.grounding import (
    ALL_OBJECTS, GeminiGrounder, GroundingError, build_prompt, parse_response,
)

cv2 = pytest.importorskip("cv2")

# Real ER 2 output captured 2026-11-26 on a synthetic 1280x720 test image.
REAL_POINTS = """```json
[
  {"point": [556, 234], "label": "red circle"},
  {"point": [458, 531], "label": "blue square"},
  {"point": [834, 406], "label": "yellow square"},
  {"point": [645, 858], "label": "green triangle"}
]
```"""
REAL_BOXES = """```json
[
  {"box_2d": [347, 469, 571, 595], "label": "the blue square"},
  {"box_2d": [527, 781, 766, 938], "label": "the green triangle"}
]
```"""


# ------------------------------------------------------------------ parsing
def test_parse_points_yx_order_and_scaling():
    dets, warns = parse_response(REAL_POINTS, 1280, 720)
    assert not warns and len(dets) == 4
    red = dets[0]
    assert red.label == "red circle"
    # drawn at (u=300, v=400) in the test image
    assert red.point_px == pytest.approx((300, 400), abs=2)
    assert red.box_px is None


def test_parse_box_2d_and_label_normalisation():
    dets, _ = parse_response(REAL_BOXES, 1280, 720)
    blue = dets[0]
    assert blue.label == "blue square"               # "the " stripped
    x1, y1, x2, y2 = blue.box_px                     # drawn at (600,250)-(760,410)
    assert (x1, y1, x2, y2) == pytest.approx((600, 250, 762, 411), abs=3)
    assert blue.point_px == pytest.approx(((x1 + x2) / 2, (y1 + y2) / 2))


def test_parse_flat_box_keys_from_docs():
    dets, _ = parse_response('[{"label": "cup", "y": 100, "x": 200, "y2": 300, "x2": 400}]', 1000, 1000)
    assert dets[0].box_px == (200, 100, 400, 300)


@pytest.mark.parametrize("text", ["[]", "```json\n[]\n```", "Sure! []"])
def test_parse_empty(text):
    assert parse_response(text, 640, 480) == ([], [])


def test_parse_skips_bad_items_with_warnings():
    text = json.dumps([
        {"point": [500, 500], "label": "ok"},
        {"point": [5000, 1], "label": "out of range"},
        {"box_2d": [500, 500, 100, 100], "label": "inverted box"},
        {"label": "no geometry"},
        "junk",
    ])
    dets, warns = parse_response(text, 100, 100)
    assert [d.label for d in dets] == ["ok"] and len(warns) == 4


@pytest.mark.parametrize("text", ["I can't see anything", "```json\n[{bad json\n```", ""])
def test_parse_garbage_raises(text):
    with pytest.raises(GroundingError):
        parse_response(text, 100, 100)


def test_prompts():
    p = build_prompt("red cup", boxes=False, max_items=10)
    assert "red cup" in p and "[y, x]" in p and "empty list" in p
    assert "box_2d" in build_prompt("red cup", boxes=True, max_items=10)
    assert "robot arm" in build_prompt(ALL_OBJECTS, boxes=False, max_items=7)


# ------------------------------------------------------------------ grounder
class FakeInteractions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def create(self, **kw):
        self.calls.append(kw)
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return SimpleNamespace(output_text=o, usage=SimpleNamespace(
            total_input_tokens=1175, total_output_tokens=96, total_thought_tokens=0))


class HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _grounder(tmp_path, outcomes, **cfg):
    fake = FakeInteractions(outcomes)
    g = GeminiGrounder(GroundingCfg(**cfg), None, log_dir=tmp_path / "g",
                       client=SimpleNamespace(interactions=fake))
    return g, fake


def test_grounder_request_shape_and_original_pixel_coords(tmp_path):
    g, fake = _grounder(tmp_path, [REAL_POINTS], max_side_px=640)
    img = np.zeros((720, 1280, 3), np.uint8)
    dets = g.ground(img, "red circle")
    kw = fake.calls[0]
    assert kw["model"] == "gemini-robotics-er-2-preview" and kw["store"] is False
    assert [p["type"] for p in kw["input"]] == ["image", "text"]          # image first
    assert kw["generation_config"]["thinking_level"] == "low"
    # Uploaded image was downscaled to 640 wide ...
    assert g.last_record.sent_wh == (640, 360)
    # ... but detections are in the ORIGINAL frame's pixels.
    assert dets[0].point_px == pytest.approx((300, 400), abs=2)


def test_grounder_logs_every_call(tmp_path):
    g, _ = _grounder(tmp_path, [REAL_POINTS])
    g.ground(np.zeros((720, 1280, 3), np.uint8), ALL_OBJECTS)
    d = g.last_log_dir
    assert (d / "sent.jpg").exists() and (d / "annotated.jpg").exists()
    rec = json.loads((d / "record.json").read_text())
    assert rec["raw_text"] == REAL_POINTS and len(rec["detections"]) == 4
    assert rec["usage"]["total_input_tokens"] == 1175
    idx = (tmp_path / "g" / "index.jsonl").read_text().splitlines()
    assert json.loads(idx[0])["n"] == 4


def test_grounder_retries_transient_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    g, fake = _grounder(tmp_path, [HttpError(503), "[]"], retries=1)
    assert g.ground(np.zeros((10, 10, 3), np.uint8), "cup") == []
    assert len(fake.calls) == 2 and g.last_record.attempts == 2


def test_grounder_does_not_retry_client_errors_and_logs_failure(tmp_path):
    g, fake = _grounder(tmp_path, [HttpError(404)], retries=3)
    with pytest.raises(GroundingError, match="404"):
        g.ground(np.zeros((10, 10, 3), np.uint8), "cup")
    assert len(fake.calls) == 1
    rec = json.loads((g.last_log_dir / "record.json").read_text())
    assert "404" in rec["error"]


def test_scan_skill_reports_vision_failure(mock_ctx, registry):
    class Broken:
        def ground(self, image, query, boxes=False):
            raise GroundingError("Gemini call failed: HTTP 503")

    mock_ctx.grounder = Broken()
    r = registry.call(mock_ctx, "scan", {})
    assert not r.ok and "vision failed" in r.message
