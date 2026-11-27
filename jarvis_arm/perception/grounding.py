"""Open-vocabulary grounding: frame + text query -> labelled points (and boxes).

The RAW frame goes to the model (nothing drawn on it - old bug).

Gemini Robotics-ER 2 notes (verified against the live API, 2026-11-26):
  * only reachable via the Interactions API on api_version v1beta
    (models.generate_content on v1 -> 404)
  * points come back as {"point": [y, x], "label": ...}, normalised 0-1000
  * boxes come back as {"box_2d": [ymin, xmin, ymax, xmax]} even when the prompt
    asks for the docs' flat {"y","x","y2","x2"} keys - we accept both
  * a query for an absent object returns []
  * output is usually wrapped in ```json fences
"""
from __future__ import annotations

import base64
import itertools
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config import GroundingCfg
from ..drivers.mocks import LinearProjection, MockWorld

log = logging.getLogger(__name__)

ALL_OBJECTS = "*"   # query sentinel: "every graspable object on the table"


@dataclass(frozen=True)
class Detection:
    label: str
    point_px: tuple[float, float]                              # (u, v) pixels
    box_px: tuple[float, float, float, float] | None = None    # x1, y1, x2, y2
    confidence: float = 1.0   # ER does not report confidence; 1.0 = "model said so"


class GroundingError(RuntimeError):
    pass


class Grounder(Protocol):
    def ground(self, image: np.ndarray, query: str, boxes: bool = False) -> list[Detection]: ...


# ----------------------------------------------------------------- prompts
_POINT_FMT = ('The answer should follow the json format: [{"point": <point>, "label": <label1>}, ...]. '
              'The points are in [y, x] format normalized to 0-1000.')
_BOX_FMT = ('The answer should follow the json format: '
            '[{"box_2d": [<y_min>, <x_min>, <y_max>, <x_max>], "label": <label1>}, ...] normalized to 0-1000.')
_LABEL_RULE = ("Each label must be a short, specific name that includes the object's colour, "
               "for example 'red cup' or 'blue bowl'.")


def build_prompt(query: str, boxes: bool, max_items: int) -> str:
    fmt = _BOX_FMT if boxes else _POINT_FMT
    verb = "Detect" if boxes else "Point to"
    if query == ALL_OBJECTS:
        what = (f"{verb} no more than {max_items} distinct objects on the table that a small robot "
                f"gripper could pick up or place things into. Ignore the table itself, the robot arm, "
                f"cables and background.")
    else:
        q = query.strip().rstrip(".?!")
        what = (f"{verb} the {q} (at most {max_items} matches). "
                f"If there is no {q} in the image, return an empty list [].")
    return f"{what} {_LABEL_RULE} {fmt}"


# ----------------------------------------------------------------- parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_ARTICLE = re.compile(r"^(the|a|an)\s+", re.I)


def _norm_label(s: Any) -> str:
    s = re.sub(r"\s+", " ", str(s)).strip().lower()
    return _ARTICLE.sub("", s) or "object"


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _in_range(*vals: float | None) -> bool:
    return all(v is not None and -5 <= v <= 1005 for v in vals)


def _clamp01k(v: float) -> float:
    return min(1000.0, max(0.0, v))


def parse_response(text: str, width: int, height: int) -> tuple[list[Detection], list[str]]:
    """Model text -> detections in PIXELS of a width x height image.
    Raises GroundingError if the text is not JSON at all; skips (and reports)
    individual malformed entries."""
    warnings: list[str] = []
    raw = (text or "").strip()
    if m := _FENCE.search(raw):
        raw = m.group(1).strip()
    if not raw.startswith(("[", "{")):
        i, j = raw.find("["), raw.rfind("]")
        if i == -1 or j <= i:
            raise GroundingError(f"no JSON in response: {text[:200]!r}")
        raw = raw[i:j + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GroundingError(f"bad JSON in response ({e}): {text[:200]!r}") from None
    if isinstance(data, dict):
        data = data.get("objects") or data.get("items") or [data]
    if not isinstance(data, list):
        raise GroundingError(f"expected a JSON list, got {type(data).__name__}")

    sx, sy = width / 1000.0, height / 1000.0
    out: list[Detection] = []
    for k, item in enumerate(data):
        if not isinstance(item, dict):
            warnings.append(f"item {k}: not an object")
            continue
        label = _norm_label(item.get("label", "object"))
        box = None
        if "box_2d" in item and isinstance(item["box_2d"], (list, tuple)) and len(item["box_2d"]) == 4:
            y1, x1, y2, x2 = (_num(v) for v in item["box_2d"])
        elif all(key in item for key in ("y", "x", "y2", "x2")):
            y1, x1, y2, x2 = (_num(item[key]) for key in ("y", "x", "y2", "x2"))
        else:
            y1 = x1 = y2 = x2 = None
        if y1 is not None or x1 is not None:
            if not _in_range(y1, x1, y2, x2) or y1 >= y2 or x1 >= x2:
                warnings.append(f"item {k} ({label}): invalid box {item.get('box_2d') or item}")
            else:
                y1, x1, y2, x2 = map(_clamp01k, (y1, x1, y2, x2))
                box = (x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        point = None
        p = item.get("point")
        if isinstance(p, (list, tuple)) and len(p) == 2:
            py, px = _num(p[0]), _num(p[1])
            if _in_range(py, px):
                point = (_clamp01k(px) * sx, _clamp01k(py) * sy)
            else:
                warnings.append(f"item {k} ({label}): point out of range {p}")
        if point is None and box is not None:
            point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if point is None:
            if not any(w.startswith(f"item {k} ") for w in warnings):
                warnings.append(f"item {k} ({label}): no usable point or box")
            continue
        out.append(Detection(label, point, box))
    return out, warnings


# ----------------------------------------------------------------- logging
@dataclass
class GroundingRecord:
    id: str
    t: float
    model: str
    query: str
    boxes: bool
    prompt: str
    image_wh: tuple[int, int]
    sent_wh: tuple[int, int]
    thinking_level: str
    latency_s: float = 0.0
    attempts: int = 0
    raw_text: str = ""
    detections: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class GroundingLog:
    """One folder per call under log_dir: sent.jpg (exact bytes uploaded),
    annotated.jpg, record.json; plus a one-line summary in index.jsonl."""

    _seq = itertools.count()

    def __init__(self, log_dir: str | Path, save_images: bool = True):
        self.dir = Path(log_dir)
        self.save_images = save_images
        self._lock = threading.Lock()

    def new_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + f"-{next(self._seq):04d}"

    def write(self, rec: GroundingRecord, sent_jpeg: bytes | None, image: np.ndarray | None,
              detections: list[Detection]) -> Path | None:
        try:
            d = self.dir / rec.id
            d.mkdir(parents=True, exist_ok=True)
            if self.save_images and sent_jpeg:
                (d / "sent.jpg").write_bytes(sent_jpeg)
                if image is not None and detections:
                    import cv2
                    from .annotate import draw_detections
                    cv2.imwrite(str(d / "annotated.jpg"), draw_detections(image, detections))
            (d / "record.json").write_text(json.dumps(asdict(rec), indent=2, default=str))
            with self._lock, open(self.dir / "index.jsonl", "a") as f:
                f.write(json.dumps({
                    "id": rec.id, "t": rec.t, "query": rec.query, "n": len(rec.detections),
                    "labels": [x["label"] for x in rec.detections], "latency_s": rec.latency_s,
                    "error": rec.error, "warnings": len(rec.warnings),
                }) + "\n")
            return d
        except OSError:
            log.exception("could not write grounding log")
            return None


# ----------------------------------------------------------------- Gemini
def _status(e: Exception) -> int | None:
    for attr in ("status_code", "code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(e, "response", None)
    return getattr(resp, "status_code", None)


def _retryable(e: Exception) -> bool:
    s = _status(e)
    if s is None:  # network errors, timeouts
        return "timeout" in type(e).__name__.lower() or "connect" in type(e).__name__.lower()
    return s == 429 or s >= 500


class GeminiGrounder:
    def __init__(self, cfg: GroundingCfg, api_key: str | None, log_dir: str | Path = "logs/grounding",
                 client: Any | None = None):
        if client is None:
            if not api_key:
                raise GroundingError("GEMINI_API_KEY not set")
            from google import genai  # vision extra
            client = genai.Client(api_key=api_key, http_options={"api_version": cfg.api_version})
        self.client = client
        self.cfg = cfg
        self.log = GroundingLog(log_dir, cfg.log_images)
        self.last_record: GroundingRecord | None = None
        self.last_log_dir: Path | None = None

    def _encode(self, image: np.ndarray) -> tuple[bytes, tuple[int, int]]:
        import cv2

        h, w = image.shape[:2]
        s = min(1.0, self.cfg.max_side_px / max(h, w))
        img = cv2.resize(image, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else image
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        if not ok:
            raise GroundingError("JPEG encode failed")
        return buf.tobytes(), (img.shape[1], img.shape[0])

    def ground(self, image: np.ndarray, query: str, boxes: bool = False) -> list[Detection]:
        if image is None or image.ndim != 3:
            raise GroundingError("expected an HxWx3 image")
        h, w = image.shape[:2]
        prompt = build_prompt(query, boxes, self.cfg.max_items)
        jpeg, sent_wh = self._encode(image)
        rec = GroundingRecord(id=self.log.new_id(), t=time.time(), model=self.cfg.model, query=query,
                              boxes=boxes, prompt=prompt, image_wh=(w, h), sent_wh=sent_wh,
                              thinking_level=self.cfg.thinking_level)
        dets: list[Detection] = []
        t0 = time.monotonic()
        try:
            resp = self._call(jpeg, prompt, rec)
            rec.raw_text = getattr(resp, "output_text", "") or ""
            rec.usage = _usage(resp)
            # Normalised coords -> pixels of the ORIGINAL frame (downscaling cancels out).
            dets, rec.warnings = parse_response(rec.raw_text, w, h)
            rec.detections = [asdict(d) for d in dets]
            return dets
        except GroundingError as e:
            rec.error = str(e)
            raise
        except Exception as e:  # noqa: BLE001 - surface as one error type to skills
            rec.error = f"{type(e).__name__}: {e}"
            raise GroundingError(f"Gemini call failed: {rec.error}") from e
        finally:
            rec.latency_s = round(time.monotonic() - t0, 3)
            self.last_record = rec
            self.last_log_dir = self.log.write(rec, jpeg, image, dets)
            log.info("ground %r -> %d hits in %.2fs%s", query, len(dets), rec.latency_s,
                     f" ERROR {rec.error}" if rec.error else "")

    def _call(self, jpeg: bytes, prompt: str, rec: GroundingRecord):
        payload = dict(
            model=self.cfg.model,
            store=False,   # don't keep frames server-side
            input=[{"type": "image", "data": base64.b64encode(jpeg).decode(), "mime_type": "image/jpeg"},
                   {"type": "text", "text": prompt}],
            generation_config={"thinking_level": self.cfg.thinking_level,
                               "temperature": self.cfg.temperature},
            timeout=self.cfg.timeout_s,
        )
        for attempt in range(self.cfg.retries + 1):
            rec.attempts = attempt + 1
            try:
                return self.client.interactions.create(**payload)
            except Exception as e:  # noqa: BLE001
                if attempt < self.cfg.retries and _retryable(e):
                    log.warning("grounding attempt %d failed (%s); retrying", attempt + 1, e)
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        raise AssertionError("unreachable")


def _usage(resp: Any) -> dict[str, Any]:
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    return {k: getattr(u, k, None) for k in ("total_input_tokens", "total_output_tokens", "total_thought_tokens")}


# ----------------------------------------------------------------- mock
class MockGrounder:
    """Finds MockWorld objects whose label shares a word with the query
    (ALL_OBJECTS returns all)."""

    def __init__(self, world: MockWorld, proj: LinearProjection):
        self.world, self.proj = world, proj

    def ground(self, image: np.ndarray, query: str, boxes: bool = False) -> list[Detection]:
        q = set(query.lower().replace(",", " ").split())
        want_all = query == ALL_OBJECTS
        out = []
        with self.world.lock:
            objs = list(self.world.objects)
        for o in objs:
            if want_all or q & set(o.label.lower().split()):
                out.append(Detection(o.label, self.proj.to_pixel(o.x, o.y), confidence=0.95))
        return out
