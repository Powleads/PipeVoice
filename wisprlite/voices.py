"""Named polish 'Voices': a reusable per-utterance override preset (the same bag
app profiles apply), bound to hotkeys and/or apps. resolve() turns a name into the
non-empty overrides that App._eff reads."""
from __future__ import annotations

# the per-utterance dials a Voice may set; "" / None mean "leave as-is"
VOICE_KEYS = ("cleanup_style", "cleanup_instruction", "engine", "auto_enter", "output_mode")

STARTER_VOICES = [
    {"name": "Tidy", "cleanup_style": "tidy", "cleanup_instruction": "",
     "engine": "", "auto_enter": None, "output_mode": ""},
    {"name": "Social", "cleanup_style": "custom",
     "cleanup_instruction": ("Rewrite as a friendly, casual social-media post in the "
        "speaker's own meaning. Natural, warm, light; emojis only if they fit. Keep it concise."),
     "engine": "", "auto_enter": False, "output_mode": ""},
    {"name": "Professional", "cleanup_style": "custom",
     "cleanup_instruction": ("Rewrite as clear, professional writing (e.g. an email): correct, "
        "polished, neutral-to-formal tone, British spelling. Preserve the speaker's meaning and every "
        "specific; do not add content."),
     "engine": "", "auto_enter": None, "output_mode": ""},
    {"name": "Code / Prompt", "cleanup_style": "prompt", "cleanup_instruction": "",
     "engine": "", "auto_enter": None, "output_mode": ""},
]

def names(cfg) -> list:
    return [v.get("name", "") for v in (getattr(cfg, "voices", None) or []) if v.get("name")]

def by_name(cfg, name: str) -> dict | None:
    for v in (getattr(cfg, "voices", None) or []):
        if v.get("name") == name:
            return v
    return None

def resolve(cfg, name: str) -> dict:
    """The non-empty overrides for Voice `name`, ready to drop into App._active. {} if unknown."""
    v = by_name(cfg, name)
    if not v:
        return {}
    out = {}
    for k in VOICE_KEYS:
        val = v.get(k)
        if k == "auto_enter":
            if val is not None:
                out[k] = bool(val)
        elif val:                      # non-empty string
            out[k] = val
    return out
