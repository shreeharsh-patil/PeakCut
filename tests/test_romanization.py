"""Tests for Hindi and Urdu Romanization Engine (romanization.py)."""
from romanization import has_hindi, needs_roman, romanize_words, to_roman


def test_has_hindi_and_needs_roman():
    assert has_hindi("नमस्ते भारत") is True
    assert has_hindi("Hello World") is False
    assert needs_roman("नमस्ते") is True
    assert needs_roman("یار کیا ہے") is True
    assert needs_roman("OpenShorts Video") is False


def test_hindi_to_roman():
    assert to_roman("चूना") == "chuna"
    assert to_roman("क्या") == "kya"
    assert to_roman("नमस्ते") == "namaste"
    assert to_roman("बहुत अच्छा") == "bahut accha"


def test_urdu_to_roman():
    res = to_roman("یار کیا ہے")
    assert "yaar" in res
    assert "kya" in res
    assert "hai" in res


def test_romanize_words_preserves_timing():
    words = [
        {"w": "Hello", "s": 0.0, "e": 0.5},
        {"w": "नमस्ते", "s": 0.6, "e": 1.2},
        {"w": "World", "s": 1.3, "e": 1.8},
    ]
    out = romanize_words(words)
    assert len(out) == 3
    assert out[0]["w"] == "Hello"
    assert out[1]["w"] == "namaste"
    assert out[1]["s"] == 0.6
    assert out[1]["e"] == 1.2
    assert out[2]["w"] == "World"
