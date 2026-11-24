"""Text-to-speech with a fallback chain: ElevenLabs -> Piper -> espeak -> say -> print.

Fixes the legacy speak(): missing comma, deprecated convert_as_stream, macOS-only
fallback. Thread-safe: the task thread and main loop never talk over each other.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Callable

from ..config import TtsCfg

log = logging.getLogger(__name__)


class Speaker:
    def __init__(self, cfg: TtsCfg, elevenlabs_api_key: str | None = None, enabled: bool = True):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._key = elevenlabs_api_key
        self._el_client = None
        engines: dict[str, Callable[[str], None]] = {
            "elevenlabs": self._elevenlabs, "piper": self._piper,
            "espeak": self._espeak, "say": self._say, "print": lambda _t: None,
        }
        names = cfg.engines if enabled else ["print"]
        self._chain = [(n, engines[n]) for n in names if n in engines and self._available(n)]
        log.info("TTS engines: %s", [n for n, _ in self._chain] or ["print"])

    def _available(self, name: str) -> bool:
        if name == "elevenlabs":
            # elevenlabs.stream() plays through mpv
            return bool(self._key) and shutil.which("mpv") is not None
        if name == "piper":
            mp = self.cfg.piper.model_path
            return bool(mp) and os.path.exists(os.path.expanduser(mp)) and shutil.which("piper") is not None
        if name == "espeak":
            return bool(shutil.which("espeak-ng") or shutil.which("espeak"))
        if name == "say":
            return shutil.which("say") is not None
        return name == "print"

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        print(f"jarvis> {text}", flush=True)
        with self._lock:
            for name, fn in list(self._chain):
                try:
                    fn(text)
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("TTS engine %s failed (%s); dropping it", name, e)
                    self._chain.remove((name, fn))

    # ------------------------------------------------------------ engines
    def _elevenlabs(self, text: str) -> None:
        from elevenlabs import stream
        if self._el_client is None:
            from elevenlabs.client import ElevenLabs
            self._el_client = ElevenLabs(api_key=self._key)
        el = self.cfg.elevenlabs
        audio = self._el_client.text_to_speech.stream(
            el.voice_id, text=text, model_id=el.model_id, output_format=el.output_format)
        stream(audio)

    def _piper(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            subprocess.run(["piper", "--model", os.path.expanduser(self.cfg.piper.model_path),
                            "--output_file", path], input=text.encode(), check=True,
                           capture_output=True)
            player = "aplay" if shutil.which("aplay") else "afplay"
            subprocess.run([player, path], check=True, capture_output=True)
        finally:
            os.unlink(path)

    def _espeak(self, text: str) -> None:
        exe = shutil.which("espeak-ng") or shutil.which("espeak")
        subprocess.run([exe, "-s", "165", text], check=True, capture_output=True)

    def _say(self, text: str) -> None:
        subprocess.run(["say", "-r", "175", text], check=True)
