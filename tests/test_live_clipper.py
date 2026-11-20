"""Tests for Live Stream Rolling Buffer Clipper (live_clipper.py)."""
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from engine.discovery.live_clipper import (
    ClipEvent,
    LiveClipperManager,
    LiveClipperSession,
    NotLiveError,
    resolve_live_stream,
)


def test_resolve_live_stream_not_live():
    with patch("yt_dlp.YoutubeDL") as mock_ydl:
        instance = MagicMock()
        instance.extract_info.return_value = {"is_live": False, "title": "Offline Video"}
        mock_ydl.return_value.__enter__.return_value = instance

        with pytest.raises(NotLiveError):
            resolve_live_stream("https://youtube.com/watch?v=offline")


def test_resolve_live_stream_success():
    with patch("yt_dlp.YoutubeDL") as mock_ydl:
        instance = MagicMock()
        instance.extract_info.return_value = {
            "is_live": True,
            "title": "Championship Grand Final",
            "channel": "SportsLive",
            "requested_formats": [{"url": "https://stream.m3u8"}],
        }
        mock_ydl.return_value.__enter__.return_value = instance

        info = resolve_live_stream("https://twitch.tv/sportslive")
        assert info["is_live"] is True
        assert info["title"] == "Championship Grand Final"
        assert info["channel"] == "SportsLive"
        assert info["stream_urls"] == ["https://stream.m3u8"]


def test_live_clipper_session_lifecycle_and_pruning(tmp_path: Path):
    session = LiveClipperSession(
        session_id="test_sess_1",
        stream_url="https://twitch.tv/test",
        output_dir=tmp_path,
        buffer_minutes=5,
    )

    assert session.session_id == "test_sess_1"
    assert not session.is_running

    # Simulate buffer segments
    buf_dir = session.buffer_dir
    buf_dir.mkdir(parents=True, exist_ok=True)

    f1 = buf_dir / "p01_00001.ts"
    f2 = buf_dir / "p01_00002.ts"
    f3 = buf_dir / "p01_00003.ts"
    f1.write_bytes(b"chunk1")
    f2.write_bytes(b"chunk2")
    f3.write_bytes(b"chunk3")

    session.segments = [
        (f1, 0.0, 100.0),
        (f2, 100.0, 200.0),
        (f3, 200.0, 350.0),
    ]
    session.current_clock = 350.0

    # Pruning with buffer_minutes=5 (300s window): keep_from is 350 - 300 = 50s
    # f1 end is 100s (> 50s), so none should be pruned
    session._scan_and_prune_segments()
    assert len(session.segments) == 3

    # Now simulate time advancing to 600s: keep_from is 600 - 300 = 300s
    session.current_clock = 600.0
    session._scan_and_prune_segments()
    # f1 (end 100s < 300s) and f2 (end 200s < 300s) should be pruned
    assert len(session.segments) == 1
    assert not f1.exists()
    assert not f2.exists()
    assert f3.exists()


def test_live_clipper_manager(tmp_path: Path):
    manager = LiveClipperManager(base_output_dir=str(tmp_path))

    with patch.object(LiveClipperSession, "start"):
        s1 = manager.create_session("https://twitch.tv/streamer1", session_id="s1")
        assert s1.session_id == "s1"
        assert manager.get_session("s1") == s1

        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"

        assert manager.stop_session("s1")
        assert not manager.stop_session("non_existent")
