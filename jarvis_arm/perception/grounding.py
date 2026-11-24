"""Open-vocabulary grounding: frame + text query -> labelled points.

The RAW frame goes to the model (no boxes drawn on it - old bug).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..drivers.mocks import LinearProjection, MockWorld


@dataclass(frozen=True)
class Detection:
    label: str
    point_px: tuple[float, float]                       # (u, v) pixels
    box_px: tuple[float, float, float, float] | None = None  # x1, y1, x2, y2
    confidence: float = 1.0


class Grounder(Protocol):
    def ground(self, image: np.ndarray, query: str) -> list[Detection]: ...


class MockGrounder:
    """Finds MockWorld objects whose label shares a word with the query
    ("all objects"/"everything" returns all)."""

    def __init__(self, world: MockWorld, proj: LinearProjection):
        self.world, self.proj = world, proj

    def ground(self, image: np.ndarray, query: str) -> list[Detection]:
        q = set(query.lower().replace(",", " ").split())
        want_all = bool(q & {"all", "everything", "objects"})
        out = []
        with self.world.lock:
            objs = list(self.world.objects)
        for o in objs:
            if want_all or q & set(o.label.lower().split()):
                out.append(Detection(o.label, self.proj.to_pixel(o.x, o.y), confidence=0.95))
        return out


class GeminiGrounder:
    """TODO(step 4): Gemini Robotics-ER via google-genai.

    Plan: client.models.generate_content(model=cfg.agent.model,
    contents=[image_part, prompt], config=GenerateContentConfig(temperature=0.1,
    response_mime_type="application/json", thinking_config=ThinkingConfig(thinking_budget=0)))
    asking for [{"point": [y, x], "label": ...}] normalised 0-1000, then scale
    to pixels. Log every request/response (+ frame) to logs/grounding/.
    """

    def __init__(self, model: str, api_key: str | None):
        self.model, self.api_key = model, api_key

    def ground(self, image: np.ndarray, query: str) -> list[Detection]:
        raise NotImplementedError("GeminiGrounder is step 4 of the build order")
