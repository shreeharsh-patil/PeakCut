import os
import subprocess
import sys
import pytest
from security_utils import normalize_media_url, assert_public_url


def test_normalize_media_url_bare_id():
    assert normalize_media_url("BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("BgeNDG5jD60?si=zSjxkinw2zPaaAPH") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("  BgeNDG5jD60?si=zSjxkinw2zPaaAPH  ") == "https://www.youtube.com/watch?v=BgeNDG5jD60"


def test_normalize_media_url_various_formats():
    assert normalize_media_url("https://www.youtube.com/watch?v=BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("https://www.youtube.com/watch?v=BgeNDG5jD60&t=42s") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("https://www.youtube.com/watch?feature=shared&v=BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("https://youtu.be/BgeNDG5jD60?si=xyz") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("youtu.be/BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("youtube.com/shorts/BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("https://youtube.com/live/BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"
    assert normalize_media_url("https://www.youtube.com/embed/BgeNDG5jD60") == "https://www.youtube.com/watch?v=BgeNDG5jD60"


def test_normalize_media_url_other_protocols():
    assert normalize_media_url("twitch.tv/ninja") == "https://twitch.tv/ninja"
    assert normalize_media_url("kick.com/streamer") == "https://kick.com/streamer"
    assert normalize_media_url("https://example.com/test.mp4") == "https://example.com/test.mp4"
    assert normalize_media_url("") == ""
    assert normalize_media_url(None) == ""


def test_assert_public_url_with_bare_id():
    # Should normalize and pass SSRF check without raising UnsafeURLError
    normalized = assert_public_url("BgeNDG5jD60?si=zSjxkinw2zPaaAPH")
    assert normalized == "https://www.youtube.com/watch?v=BgeNDG5jD60"


def test_main_cli_arguments():
    # Verify main.py accepts clip duration flags and does not crash with exit code 2 on unknown flags
    cmd = [
        sys.executable,
        "main.py",
        "--help"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "--clip-min-seconds" in res.stdout
    assert "--clip-max-seconds" in res.stdout
    assert "--target-clips" in res.stdout
