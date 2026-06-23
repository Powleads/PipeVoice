import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wisprlite import voices, config

def test_resolve_filters_empty():
    cfg = config.Config()  # seeded with starters
    r = voices.resolve(cfg, "Code / Prompt")
    assert r == {"cleanup_style": "prompt"}            # only the non-empty dial
    s = voices.resolve(cfg, "Social")
    assert s["cleanup_style"] == "custom" and s["auto_enter"] is False and "engine" not in s

def test_unknown_voice():
    assert voices.resolve(config.Config(), "Nope") == {}

def test_starters_present():
    assert set(voices.names(config.Config())) >= {"Tidy", "Social", "Professional", "Code / Prompt"}

if __name__ == "__main__":
    test_resolve_filters_empty(); test_unknown_voice(); test_starters_present()
    print("OK")
