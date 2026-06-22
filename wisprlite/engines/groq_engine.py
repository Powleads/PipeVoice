"""Groq Whisper (whisper-large-v3-turbo): real Whisper weights, ~9x cheaper than
OpenAI, 216x real-time so it feels near-instant, free dev tier. Batch
(transcribes on key-release); needs internet + GROQ_API_KEY.

Groq exposes an OpenAI-compatible /audio/transcriptions endpoint, so this is just
the OpenAI engine pointed at Groq's base URL — same Session, same call.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import OnPartial, Session
from .openai_engine import OpenAIEngine, _OpenAISession

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqEngine(OpenAIEngine):
    name = "groq"
    streaming = False

    def __init__(self, model: str = "whisper-large-v3-turbo",
                 language: Optional[str] = None, prompt: str = "") -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=GROQ_BASE_URL,
                             api_key=os.getenv("GROQ_API_KEY", "").strip())
        self.model = model or "whisper-large-v3-turbo"
        self.language = language or None
        self.prompt = (prompt or "").strip()

    def start_session(self, on_partial: Optional[OnPartial] = None) -> Session:
        return _OpenAISession(self, on_partial)
