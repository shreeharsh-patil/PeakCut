"""Tests covering edge cases in video reframing, live stream capture, subtitle generation, and batch running."""
import os
from unittest.mock import MagicMock, patch
import pytest

from main import SmoothedCameraman, get_video_resolution
from engine.layouts.reframe_v2 import delivery_size
from engine.editing.subtitles import _escape_ass_text, generate_ass
from engine.discovery.live_clipper import resolve_live_stream
from engine.utils import batch_runner


def test_smoothed_cameraman_even_dimensions_and_centered_y():
    # 1080x1920 source reframed to square (aspect_ratio = 1.0)
    cameraman = SmoothedCameraman(
        output_width=1080, output_height=1080,
        video_width=1080, video_height=1920,
        aspect_ratio=1.0
    )
    assert cameraman.crop_width % 2 == 0
    assert cameraman.crop_height % 2 == 0
    assert cameraman.crop_width == 1080
    assert cameraman.crop_height == 1080

    x1, y1, x2, y2 = cameraman.get_crop_box()
    assert (x2 - x1) == 1080
    assert (y2 - y1) == 1080
    # y should be centered in the 1920 height: (1920 - 1080) // 2 = 420
    assert y1 == 420
    assert y2 == 1500


def test_smoothed_cameraman_odd_input_dimensions():
    # Video with odd dimensions 1921x1081 reframed to 9:16
    cameraman = SmoothedCameraman(
        output_width=1080, output_height=1920,
        video_width=1921, video_height=1081,
        aspect_ratio=9 / 16
    )
    assert cameraman.crop_width % 2 == 0
    assert cameraman.crop_height % 2 == 0

    x1, y1, x2, y2 = cameraman.get_crop_box()
    assert (x2 - x1) == cameraman.crop_width
    assert (y2 - y1) == cameraman.crop_height
    assert x1 >= 0 and x2 <= 1921
    assert y1 >= 0 and y2 <= 1081


def test_delivery_size_edge_cases():
    # Invalid / non-positive aspect ratio falls back safely
    out_w, out_h = delivery_size(1920, 1080, 0)
    assert out_w % 2 == 0
    assert out_h % 2 == 0
    assert out_w == 1080
    assert out_h == 1920

    out_w, out_h = delivery_size(1920, 1080, -1.5)
    assert out_w % 2 == 0
    assert out_h % 2 == 0


def test_escape_ass_text_cleans_newlines():
    raw = "Line 1\r\nLine 2\nLine 3\rLine 4 {override} \\backslash"
    escaped = _escape_ass_text(raw)
    assert "\n" not in escaped
    assert "\r" not in escaped
    assert "{" not in escaped
    assert "}" not in escaped
    assert "\\" not in escaped
    assert "(override)" in escaped
    assert "/backslash" in escaped


def test_generate_ass_preserves_zero_duration_words(tmp_path):
    out = tmp_path / "zero_dur.ass"
    # Transcript with identical timestamps on consecutive words
    transcript = {
        "segments": [{
            "words": [
                {"word": " word1", "start": 1.0, "end": 1.0},
                {"word": " word2", "start": 1.0, "end": 1.5},
            ]
        }]
    }
    success = generate_ass(transcript, 0.0, 5.0, str(out))
    assert success is True
    content = out.read_text(encoding="utf-8-sig")
    assert "word1" in content
    assert "word2" in content
    # Both words produce dialogue lines
    assert content.count("Dialogue:") == 2


def test_generate_ass_split_ranges_seam_alignment(tmp_path):
    out = tmp_path / "split_seam.ass"
    transcript = {
        "segments": [{
            "words": [
                {"word": "first", "start": 1.0, "end": 1.4},
                {"word": "second", "start": 1.5, "end": 2.0},
            ]
        }]
    }
    # Time range [0.5, 3.0] is in a split layout
    split_ranges = [(0.5, 3.0)]
    success = generate_ass(transcript, 0.0, 5.0, str(out), split_ranges=split_ranges)
    assert success is True
    content = out.read_text(encoding="utf-8-sig")
    assert "{\\an5}" in content
    # Reset tag preserves center seam alignment
    assert "{\\r\\an5}" in content


def test_resolve_live_stream_formats_fallback():
    with patch("yt_dlp.YoutubeDL") as mock_ydl:
        instance = MagicMock()
        # requested_formats is None, but formats has video + audio
        instance.extract_info.return_value = {
            "is_live": True,
            "title": "Stream Without Requested Formats",
            "channel": "ChannelA",
            "requested_formats": None,
            "url": None,
            "formats": [
                {"url": "https://stream.video", "vcodec": "avc1", "acodec": "none", "height": 720},
                {"url": "https://stream.audio", "vcodec": "none", "acodec": "mp4a", "abr": 128},
            ]
        }
        mock_ydl.return_value.__enter__.return_value = instance

        info = resolve_live_stream("https://youtube.com/live/abc")
        assert info["is_live"] is True
        assert len(info["stream_urls"]) == 2
        assert "https://stream.video" in info["stream_urls"]
        assert "https://stream.audio" in info["stream_urls"]


def test_batch_runner_load_urls_with_inline_comments_and_quotes(tmp_path):
    url_file = tmp_path / "urls_with_comments.txt"
    url_file.write_text("""
    # Header comment
    "https://youtu.be/vid1" # Inline comment 1
    'https://youtu.be/vid2'
    https://youtu.be/vid3 # Inline comment 2
    """, encoding="utf-8")

    class Args:
        urls = None
        file = str(url_file)

    urls = batch_runner.load_urls(Args())
    assert urls == [
        "https://youtu.be/vid1",
        "https://youtu.be/vid2",
        "https://youtu.be/vid3"
    ]


def test_get_video_resolution_fallback(monkeypatch):
    class FakeCap:
        def isOpened(self):
            return True
        def get(self, prop):
            return 0  # OpenCV returns 0
        def release(self):
            pass

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: FakeCap())

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="1920x1080\n")
        w, h = get_video_resolution("dummy.mp4")
        assert w == 1920
        assert h == 1080
