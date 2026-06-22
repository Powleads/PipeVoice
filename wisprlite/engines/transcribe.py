"""File transcription with word-level timestamps (for the agent MCP).

Separate from the streaming dictation Session: takes a file *path* (not an
audio array), so it needs no numpy and is import-light / unit-testable.
faster-whisper is imported lazily inside the functions.
"""

from __future__ import annotations

import threading as _threading

_TRANSCRIBE_MODEL = None
_TRANSCRIBE_KEY = None
_TRANSCRIBE_LOCK = _threading.Lock()


def _segments_to_dicts(segments) -> list:
    """Shape faster-whisper segments into plain JSON-able dicts. Pure."""
    out = []
    for seg in segments:
        words = []
        for w in (getattr(seg, "words", None) or []):
            words.append({"start": round(float(w.start), 3),
                          "end": round(float(w.end), 3),
                          "word": w.word})
        out.append({"start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "text": (seg.text or "").strip(),
                    "words": words})
    return out


def _get_transcribe_model(model_size: str, device: str = "auto", compute_type: str = "int8"):
    global _TRANSCRIBE_MODEL, _TRANSCRIBE_KEY
    key = (model_size, device, compute_type)
    if _TRANSCRIBE_MODEL is None or _TRANSCRIBE_KEY != key:
        from faster_whisper import WhisperModel
        _TRANSCRIBE_MODEL = WhisperModel(model_size, device=device, compute_type=compute_type)
        _TRANSCRIBE_KEY = key
    return _TRANSCRIBE_MODEL


def transcribe_file(path: str, *, language=None, model_size: str = "base.en",
                    device: str = "auto", compute_type: str = "int8") -> dict:
    """Transcribe an audio/video file to text + word/segment timestamps.

    Serialized behind a lock (one shared warm model). faster-whisper decodes
    audio from many container formats via its bundled PyAV/ffmpeg.
    """
    with _TRANSCRIBE_LOCK:
        model = _get_transcribe_model(model_size, device, compute_type)
        segments, info = model.transcribe(
            path, language=language or None, word_timestamps=True,
            vad_filter=True, beam_size=1,
        )
        seg_dicts = _segments_to_dicts(segments)  # consume the generator inside the lock
    text = " ".join(s["text"] for s in seg_dicts).strip()
    return {"text": text,
            "language": getattr(info, "language", None),
            "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
            "segments": seg_dicts}
