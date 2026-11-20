"""Tests for SmartCut Auto-Silence Jump Cutter (smartcut.py)."""
from pathlib import Path
from unittest.mock import patch

from engine.editing.smartcut import compute_speech_segments, detect_silences


def test_compute_speech_segments():
    # 60s total, silences from 10s-15s and 35s-40s
    total_dur = 60.0
    silences = [(10.0, 15.0), (35.0, 40.0)]

    speech = compute_speech_segments(total_dur, silences, pad_pre=0.05, pad_post=0.08)

    assert len(speech) == 3
    # First chunk ~0.0 to 10.05
    assert speech[0][0] == 0.0
    assert 10.0 <= speech[0][1] <= 10.1
    # Second chunk ~14.92 to 35.05
    assert 14.8 <= speech[1][0] <= 15.0
    assert 35.0 <= speech[1][1] <= 35.1
    # Third chunk ~39.92 to 60.0
    assert 39.8 <= speech[2][0] <= 40.0
    assert speech[2][1] == 60.0


def test_detect_silences_mock():
    mock_stderr = """
[silencedetect @ 000001] silence_start: 12.345
[silencedetect @ 000001] silence_end: 15.678 | silence_duration: 3.333
[silencedetect @ 000001] silence_start: 28.500
[silencedetect @ 000001] silence_end: 32.100 | silence_duration: 3.600
"""
    with patch("subprocess.run") as mock_run:
        mock_proc = patch("subprocess.CompletedProcess")
        mock_run.return_value.stderr = mock_stderr

        silences = detect_silences("dummy.mp4")
        assert len(silences) == 2
        assert silences[0] == (12.345, 15.678)
        assert silences[1] == (28.5, 32.1)
