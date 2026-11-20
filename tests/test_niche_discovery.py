"""Tests for Niche & Topic Discovery Engine (niche_discovery.py)."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from engine.discovery.niche_discovery import (
    DiscoveredVideo,
    NicheDiscoveryEngine,
    SeenLedger,
    Topic,
    iso_duration_to_seconds,
)


def test_iso_duration_to_seconds():
    assert iso_duration_to_seconds("PT1M30S") == 90
    assert iso_duration_to_seconds("PT1H2M10S") == 3730
    assert iso_duration_to_seconds("PT45S") == 45
    assert iso_duration_to_seconds("PT10M") == 600
    assert iso_duration_to_seconds("") == 0
    assert iso_duration_to_seconds("INVALID") == 0


def test_seen_ledger(tmp_path: Path):
    ledger_file = tmp_path / "test_seen.json"
    ledger = SeenLedger(ledger_file)

    assert not ledger.is_seen("vid123")
    assert ledger.total_seen() == 0

    ledger.mark_seen("vid123", title="Test Video 1", metadata={"views": 5000})
    assert ledger.is_seen("vid123")
    assert ledger.total_seen() == 1

    # Reload from disk
    ledger2 = SeenLedger(ledger_file)
    assert ledger2.is_seen("vid123")
    assert ledger2.total_seen() == 1

    candidates = [
        DiscoveredVideo(
            video_id="vid123",
            title="Old Video",
            url="http://yt/vid123",
            channel_title="Test",
            duration_seconds=300,
            view_count=1000,
            published_at="",
        ),
        DiscoveredVideo(
            video_id="vid456",
            title="Fresh Video",
            url="http://yt/vid456",
            channel_title="Test",
            duration_seconds=400,
            view_count=2000,
            published_at="",
        ),
    ]

    unseen = ledger2.filter_unseen(candidates)
    assert len(unseen) == 1
    assert unseen[0].video_id == "vid456"


def test_get_topics_for_providers():
    engine = NicheDiscoveryEngine(api_key="test_key")
    trending = engine.get_topics_for_provider("youtube_trending")
    assert len(trending) >= 3
    assert any(t.name == "AI & Tech Insights" for t in trending)

    sports = engine.get_topics_for_provider("sports")
    assert len(sports) >= 2

    seeds = engine.get_topics_for_provider("seeds", custom_seeds=["Quantum Computing", "SpaceX"])
    assert len(seeds) == 2
    assert seeds[0].name == "Quantum Computing"


def test_search_youtube_api_mock(tmp_path: Path):
    ledger = SeenLedger(tmp_path / "seen.json")
    engine = NicheDiscoveryEngine(api_key="mock_key", ledger=ledger)

    search_response = {
        "items": [
            {"id": {"videoId": "vid_abc"}},
            {"id": {"videoId": "vid_xyz"}},
        ]
    }
    enrich_response = {
        "items": [
            {
                "id": "vid_abc",
                "snippet": {
                    "title": "Artificial Intelligence Podcast Episode 1",
                    "channelTitle": "AI Channel",
                    "publishedAt": "2026-11-10T00:00:00Z",
                    "thumbnails": {"high": {"url": "http://img/abc.jpg"}},
                    "description": "Deep discussion on AI",
                },
                "contentDetails": {"duration": "PT12M30S"},  # 750s
                "statistics": {"viewCount": "150000"},
            },
            {
                "id": "vid_xyz",
                "snippet": {
                    "title": "Super Short Clip",
                    "channelTitle": "Shorts",
                    "publishedAt": "2026-11-12T00:00:00Z",
                    "thumbnails": {"high": {"url": "http://img/xyz.jpg"}},
                    "description": "Too short",
                },
                "contentDetails": {"duration": "PT45S"},  # 45s (below min_sec 180)
                "statistics": {"viewCount": "500000"},
            },
        ]
    }

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm1 = MagicMock()
        mock_cm1.read.return_value = json.dumps(search_response).encode("utf-8")
        mock_resp1 = MagicMock()
        mock_resp1.__enter__.return_value = mock_cm1

        mock_cm2 = MagicMock()
        mock_cm2.read.return_value = json.dumps(enrich_response).encode("utf-8")
        mock_resp2 = MagicMock()
        mock_resp2.__enter__.return_value = mock_cm2

        mock_urlopen.side_effect = [mock_resp1, mock_resp2]

        results = engine.search_youtube_api(
            "AI discussion",
            max_results=5,
            min_sec=180,
            max_sec=1800,
        )

        assert len(results) == 1
        assert results[0].video_id == "vid_abc"
        assert results[0].duration_seconds == 750
        assert results[0].view_count == 150000
