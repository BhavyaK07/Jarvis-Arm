"""World model: what the arm believes is on the table."""
from __future__ import annotations

import itertools
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

Status = Literal["on_table", "held", "placed", "missing"]


@dataclass
class SceneObject:
    id: str
    label: str
    x: float
    y: float
    confidence: float
    last_seen: float = field(default_factory=time.time)
    status: Status = "on_table"
    aliases: list[str] = field(default_factory=list)   # other names the model used

    def summary(self) -> dict:
        d = asdict(self)
        d["x"], d["y"] = round(self.x, 3), round(self.y, 3)
        d["age_s"] = round(time.time() - self.last_seen, 1)
        del d["last_seen"]
        return d


_FILLER = {"small", "large", "big", "little", "object", "item", "thing", "of", "the", "a"}


def _words(label: str) -> set[str]:
    return set(label.lower().split()) - _FILLER


def labels_compatible(a: str, b: str) -> bool:
    """Same object under two names? The grounding model names things
    differently per query ("red block" on a scan, "red circle" when asked for
    one), so share-a-word counts as compatible. Unrelated names ("red cup" vs
    "blue bowl") stay separate even at the same spot (a cup placed in a bowl).
    Known gap: "red cup" placed in a "red bowl" would merge."""
    return a == b or bool(_words(a) & _words(b))


class Scene:
    """Thread-safe object list. An observation updates the nearest existing
    object within `merge_radius_m` that has a compatible label; otherwise it
    becomes a new object."""

    def __init__(self, merge_radius_m: float = 0.04):
        self._objs: dict[str, SceneObject] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.merge_radius_m = merge_radius_m

    def observe(self, label: str, x: float, y: float, confidence: float) -> SceneObject:
        with self._lock:
            best, best_d = None, self.merge_radius_m
            for o in self._objs.values():
                if o.status == "held" or not labels_compatible(o.label, label):
                    continue
                d = math.hypot(o.x - x, o.y - y)
                if d <= best_d:
                    best, best_d = o, d
            if best is not None:
                best.x, best.y, best.confidence, best.last_seen = x, y, confidence, time.time()
                best.status = "on_table" if best.status == "missing" else best.status
                if label != best.label and label not in best.aliases:
                    best.aliases.append(label)
                return best
            oid = f"obj{next(self._ids)}"
            o = SceneObject(oid, label, x, y, confidence)
            self._objs[oid] = o
            return o

    def get(self, oid: str) -> SceneObject | None:
        with self._lock:
            return self._objs.get(oid)

    def set_status(self, oid: str, status: Status, x: float | None = None, y: float | None = None) -> None:
        with self._lock:
            o = self._objs[oid]
            o.status = status
            if x is not None and y is not None:
                o.x, o.y = x, y

    def held(self) -> SceneObject | None:
        with self._lock:
            return next((o for o in self._objs.values() if o.status == "held"), None)

    def mark_unseen_missing(self, seen_ids: set[str]) -> None:
        """After a full scan: table objects not re-seen are 'missing'."""
        with self._lock:
            for o in self._objs.values():
                if o.status in ("on_table", "placed") and o.id not in seen_ids:
                    o.status = "missing"

    def all(self) -> list[SceneObject]:
        with self._lock:
            return list(self._objs.values())

    def summary(self) -> list[dict]:
        return [o.summary() for o in self.all()]
