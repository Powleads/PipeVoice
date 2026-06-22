import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wisprlite.engines.transcribe import _segments_to_dicts

class W:
    def __init__(self, start, end, word): self.start, self.end, self.word = start, end, word

class S:
    def __init__(self, start, end, text, words): self.start, self.end, self.text, self.words = start, end, text, words

def test_shaping_with_words():
    segs = [S(0.0, 1.234567, " Hello ", [W(0.0, 0.5, "Hello")])]
    out = _segments_to_dicts(segs)
    assert out == [{"start": 0.0, "end": 1.235, "text": "Hello",
                    "words": [{"start": 0.0, "end": 0.5, "word": "Hello"}]}], out

def test_shaping_without_words():
    segs = [S(1.0, 2.0, "no words", None)]
    out = _segments_to_dicts(segs)
    assert out[0]["words"] == [], out

if __name__ == "__main__":
    test_shaping_with_words(); test_shaping_without_words()
    print("OK")
