"""Tests for Audio Energy & Highlight Detector (highlight_detector.py)."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from engine.editing.highlight_detector import detect_hype_moments, extract_audio_energy


def test_detect_hype_moments_synthetic(tmp_path: Path):
    # Create a synthetic 60-second RMS envelope with a clear loud peak at second 30
    sr_win = 2  # 0.5s windows -> 2 per second, 120 total windows for 60s
    total_wins = 120
    envelope = np.full(total_wins, 0.05, dtype=np.float32)

    # Spike at index 60 (30.0 seconds)
    envelope[58:64] = 0.85

    with patch("engine.editing.highlight_detector.extract_audio_energy") as mock_extract:
        mock_extract.return_value = (envelope, 60.0)

        moments = detect_hype_moments(
            "dummy_file.mp4",
            max_clips=3,
            target_clip_duration=15.0,
            pre_peak_sec=5.0,
            post_peak_sec=10.0,
            smooth_sec=1.5,
            percentile_threshold=80.0,
        )

        assert len(moments) >= 1
        peak_moment = moments[0]
        # Peak around 30s
        assert 28.0 <= peak_moment.peak_time <= 32.0
        assert peak_moment.duration == 15.0
        assert peak_moment.start <= peak_moment.peak_time <= peak_moment.end
        assert peak_moment.score > 50.0
