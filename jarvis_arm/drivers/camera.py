"""Latest-frame camera: one background thread keeps the newest frame.

The camera is opened ONCE and never reopened per call (old jarvisAI bug).
Headless: no cv2.imshow anywhere.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Frame:
    image: np.ndarray   # HxWx3 BGR uint8
    t: float            # time.monotonic() at capture


class Camera(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def latest(self, max_age_s: float = 1.0) -> Frame | None: ...


class OpenCVCamera:
    """USB webcam / laptop webcam via OpenCV. Pi Camera 3 will get its own
    Picamera2 class with the same interface."""

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index, self.width, self.height = index, width, height
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import cv2  # lazy: vision extra

        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {self.index} not available")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, img = self._cap.read()
            if not ok:
                log.warning("camera read failed")
                time.sleep(0.1)
                continue
            with self._lock:
                self._frame = Frame(img, time.monotonic())

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if getattr(self, "_cap", None) is not None:
            self._cap.release()

    def latest(self, max_age_s: float = 1.0) -> Frame | None:
        with self._lock:
            f = self._frame
        if f is None or time.monotonic() - f.t > max_age_s:
            return None
        return f
