"""Spoken commands: turn dictated phrases into formatting or control actions.

Runs on the FINAL transcript inside app.py:_finish. Two kinds:

  - Terminal actions (``pre`` — run on the RAW transcript, before AI cleanup so
    cleanup can't reword them): "scratch that" discards the whole utterance;
    "send it" / "press enter" type the text then hit Enter once. A trailing
    "... send it" types the prefix and presses Enter.

  - Inline substitutions (``inline`` — run AFTER cleanup so newlines survive):
    a phrase standing alone becomes a literal, e.g. "new line" -> "\n".

Matching is deliberately conservative so ordinary speech isn't mangled:
terminal actions fire only when they are essentially the whole utterance, and
inline phrases are replaced only when they stand alone (word boundaries).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# phrase -> literal text, replaced only when the phrase stands alone
INLINE = {
    "new paragraph": "\n\n",
    "new line": "\n",
    "next line": "\n",
    "newline": "\n",
    "tab key": "\t",
}

# whole-utterance phrases that throw the dictation away
DISCARD_PHRASES = {"scratch that", "cancel that", "delete that", "forget that", "never mind"}

# whole-utterance (or trailing) phrases meaning "type it, then press Enter"
ENTER_PHRASES = {"press enter", "send it", "send message", "hit enter", "enter key"}


@dataclass
class Result:
    text: str
    discard: bool = False
    press_enter: bool = False


def _norm(s: str) -> str:
    return re.sub(r"[\s.,!?]+$", "", (s or "").strip().lower())


def pre(text: str, enabled: bool = True) -> Result:
    """Terminal actions on the raw transcript (before cleanup)."""
    if not enabled or not text:
        return Result(text=text)

    whole = _norm(text)
    if whole in DISCARD_PHRASES:
        return Result(text="", discard=True)
    if whole in ENTER_PHRASES:
        return Result(text="", press_enter=True)

    # trailing "... send it" -> type the prefix, then Enter
    for phrase in ENTER_PHRASES:
        m = re.search(rf"[\s,]+{re.escape(phrase)}[\s.,!?]*$", text, flags=re.IGNORECASE)
        if m:
            return Result(text=text[: m.start()].rstrip(), press_enter=True)

    return Result(text=text)


def inline(text: str, enabled: bool = True) -> str:
    """Inline literal substitutions (after cleanup)."""
    if not enabled or not text:
        return text
    out = text
    for phrase, literal in INLINE.items():
        # eat surrounding spaces/tabs and any punctuation the recognizer attached
        # so "New line." -> "\n" and "a new line, b" -> "a\nb", not "\n." / " \n, "
        out = re.sub(rf"[ \t]*(?<!\w){re.escape(phrase)}(?!\w)[.,!?]*[ \t]*", literal, out, flags=re.IGNORECASE)
    return out
