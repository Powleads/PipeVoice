"""Google Gemini transcription (native generateContent, multimodal audio).

Genuinely free: one free Gemini key transcribes AND powers Flow-mode cleanup, so
a new user can be fully working at zero cost. Batch (transcribes on key-release),
needs internet + GEMINI_API_KEY. Talks to the REST API directly via stdlib
urllib — no extra SDK dependency to bundle.

Gemini has no Whisper-style /audio/transcriptions endpoint; audio goes in as an
inline part to generateContent with a "transcribe verbatim" instruction, and the
transcript comes back in candidates[0].content.parts[].text.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.request
import wave
from typing import Optional

import numpy as np

from .base import Engine, OnPartial, Session

SAMPLE_RATE = 16_000
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _wav_bytes(audio: np.ndarray) -> bytes:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _build_prompt(language: str, vocab: str) -> str:
    parts = [
        "Transcribe this audio verbatim.",
        "Output only the spoken words as plain text — no preamble, no quotes, no "
        "commentary, no timestamps, no speaker labels.",
        "If there is no clear speech, output nothing.",
    ]
    if language:
        parts.append(f"The speaker's language is {language}.")
    if vocab:
        parts.append(f"Likely names/terms and their spellings: {vocab}.")
    return " ".join(parts)


def _extract_text(data: dict) -> str:
    """Pull the transcript out of a generateContent response, tolerant of shape."""
    try:
        cands = data.get("candidates") or []
        if not cands:
            return ""
        parts = (cands[0].get("content") or {}).get("parts") or []
        return "".join(p.get("text", "") for p in parts).strip()
    except Exception:
        return ""


class _GeminiSession(Session):
    def __init__(self, engine: "GeminiEngine", on_partial: Optional[OnPartial]) -> None:
        self._engine = engine
        self._on_partial = on_partial

    def finish(self, audio: np.ndarray) -> str:
        if audio is None or audio.size == 0:
            return ""
        eng = self._engine
        if not eng.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        # ponytail: inline audio caps at 20MB (~10 min @16kHz mono); a push-to-talk
        # hold is seconds, so we never approach it. Use the Files API if that changes.
        b64 = base64.b64encode(_wav_bytes(audio)).decode("ascii")
        body = json.dumps({
            "contents": [{"parts": [
                {"text": eng.prompt_text},
                {"inline_data": {"mime_type": "audio/wav", "data": b64}},
            ]}],
            "generationConfig": {"temperature": 0},
        }).encode("utf-8")
        req = urllib.request.Request(
            _ENDPOINT.format(model=eng.model), data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": eng.api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=eng.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"Gemini transcription failed (HTTP {e.code}): {detail}")
        return _extract_text(data)


class GeminiEngine(Engine):
    name = "gemini"
    streaming = False

    def __init__(self, model: str = "gemini-3.1-flash-lite", language: Optional[str] = None,
                 prompt: str = "", timeout: float = 30.0) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = model or "gemini-3.1-flash-lite"
        self.language = (language or "").strip()
        self.prompt_text = _build_prompt(self.language, (prompt or "").strip())
        self.timeout = timeout

    def start_session(self, on_partial: Optional[OnPartial] = None) -> Session:
        return _GeminiSession(self, on_partial)


def _demo() -> None:
    # ponytail: offline self-check for the response parser (no network/numpy).
    assert _extract_text({"candidates": [{"content": {"parts": [{"text": " hi there "}]}}]}) == "hi there"
    assert _extract_text({"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}) == "ab"
    assert _extract_text({"candidates": []}) == ""
    assert _extract_text({}) == ""
    assert "verbatim" in _build_prompt("", "")
    assert "English" in _build_prompt("English", "")
    assert "Powleads" in _build_prompt("", "Powleads")
    print("gemini_engine self-check OK")


if __name__ == "__main__":
    _demo()
