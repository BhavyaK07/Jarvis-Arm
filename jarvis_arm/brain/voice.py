"""Voice input: wake word -> VAD-gated recording -> Groq Whisper transcription.

Ported from legacy/jarvisAI.py with fixes:
  * transcription uses client.audio.transcriptions.create(file=..., model=...)
  * no temp files (in-memory WAV)
  * returns None when no speech was detected (old code sent silence to Whisper)
  * no macOS-only paths
TextInput is a drop-in replacement for development (--text).
"""
from __future__ import annotations

import io
import logging
import wave
from collections import deque
from typing import Protocol

import numpy as np

from ..config import VoiceCfg

log = logging.getLogger(__name__)

# Whisper's favourite hallucinations on near-silence.
_HALLUCINATIONS = {"thank you.", "thanks for watching!", "you", "bye.", "."}


class VoiceInput(Protocol):
    def wait_for_wake(self) -> None: ...
    def listen(self) -> str | None: ...
    @property
    def needs_wake_ack(self) -> bool: ...


class MicInput:
    def __init__(self, cfg: VoiceCfg, groq_api_key: str, groq_client=None):
        import sounddevice  # noqa: F401  lazy: voice extra
        import webrtcvad
        from openwakeword.model import Model

        self.cfg = cfg
        try:
            self._oww = Model(wakeword_models=[cfg.wake_word], inference_framework="onnx")
        except Exception:  # noqa: BLE001 - onnxruntime raises its own NoSuchFile
            from openwakeword.utils import download_models  # first run on a new machine
            log.info("downloading openWakeWord models (first run)")
            download_models(model_names=[cfg.wake_word])
            self._oww = Model(wakeword_models=[cfg.wake_word], inference_framework="onnx")
        self._vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        if groq_client is None:
            from groq import Groq
            groq_client = Groq(api_key=groq_api_key)
        self._groq = groq_client
        log.info("wake word model '%s' ready", cfg.wake_word)

    needs_wake_ack = True

    def wait_for_wake(self) -> None:
        import sounddevice as sd

        chunk = 1280  # 80 ms @ 16 kHz, what openWakeWord expects
        with sd.InputStream(samplerate=self.cfg.sample_rate, channels=1, dtype="int16",
                            blocksize=chunk) as stream:
            while True:
                audio, _ = stream.read(chunk)
                score = self._oww.predict(audio.flatten()).get(self.cfg.wake_word, 0.0)
                if score > self.cfg.wake_threshold:
                    self._oww.reset()
                    return

    def record(self) -> np.ndarray | None:
        """Record one utterance; None if nobody spoke before the timeout."""
        import sounddevice as sd

        fs, frame_ms = self.cfg.sample_rate, 30
        frame = int(fs * frame_ms / 1000)
        silence_needed = int(self.cfg.silence_s * 1000 / frame_ms)
        max_frames = int(self.cfg.max_utterance_s * 1000 / frame_ms)
        frames: list[np.ndarray] = []
        preroll: deque[np.ndarray] = deque(maxlen=10)   # keep ~300 ms before speech onset
        silence, started = 0, False
        with sd.InputStream(samplerate=fs, channels=1, dtype="int16", blocksize=frame) as stream:
            for _ in range(max_frames):
                chunk, _ = stream.read(frame)
                chunk = chunk.flatten()
                speech = self._vad.is_speech(chunk.tobytes(), fs)
                if speech:
                    if not started:
                        frames.extend(preroll)
                    started, silence = True, 0
                elif started:
                    silence += 1
                    if silence >= silence_needed:
                        break
                (frames if started else preroll).append(chunk)
        return np.concatenate(frames) if started else None

    def transcribe(self, pcm16: np.ndarray) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.cfg.sample_rate)
            w.writeframes(pcm16.astype(np.int16).tobytes())
        tr = self._groq.audio.transcriptions.create(
            file=("utterance.wav", buf.getvalue()),
            model=self.cfg.stt_model,
            language=self.cfg.language,
        )
        return (tr.text or "").strip()

    def listen(self) -> str | None:
        audio = self.record()
        if audio is None:
            return None
        text = self.transcribe(audio)
        if len(text) < 2 or text.lower() in _HALLUCINATIONS:
            return None
        return text


class TextInput:
    """Keyboard stand-in for the mic. Empty line = silence; Ctrl-D quits."""

    needs_wake_ack = False

    def wait_for_wake(self) -> None:
        return

    def listen(self) -> str | None:
        try:
            text = input("you> ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        return text or None
