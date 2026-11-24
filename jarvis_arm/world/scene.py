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

    def summary(self) -> dict:
        d = asdict(self)
        d["x"], d["y"] = round(self.x, 3), round(self.y, 3)
        d["age_s"] = round(time.time() - self.last_seen, 1)
        del d["last_seen"]
        return d


class Scene:
    """Thread-safe object list. Observations within `merge_radius_m` of an
    existing object with the same label update it instead of adding a new one."""

    def __init__(self, merge_radius_m: float = 0.04):
        self._objs: dict[str, SceneObject] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.merge_radius_m = merge_radius_m

    def observe(self, label: str, x: float, y: float, confidence: float) -> SceneObject:
        with self._lock:
            for o in self._objs.values():
                if o.label == label and o.status != "held" and \
                        math.hypot(o.x - x, o.y - y) <= self.merge_radius_m:
                    o.x, o.y, o.confidence, o.last_seen = x, y, confidence, time.time()
                    o.status = "on_table" if o.status == "missing" else o.status
                    return o
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
