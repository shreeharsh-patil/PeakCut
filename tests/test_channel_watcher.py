"""Tests for Multi-Platform Channel Watcher (channel_watcher.py)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from channel_watcher import ChannelWatcherService, WatchedChannel

SAMPLE_RSS_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:yt="http://www.youtube.com/xml/schemas/2015">
  <channelId>UC1234567890123456789012</channelId>
  <title>Creator Channel</title>
  <entry>
    <id>yt:video:abc_video_1</id>
    <yt:videoId>abc_video_1</yt:videoId>
    <yt:channelId>UC1234567890123456789012</yt:channelId>
    <title>Awesome New Podcast Episode 100</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=abc_video_1"/>
    <published>2026-11-15T00:00:00+00:00</published>
  </entry>
  <entry>
    <id>yt:video:def_video_2</id>
    <yt:videoId>def_video_2</yt:videoId>
    <yt:channelId>UC1234567890123456789012</yt:channelId>
    <title>Awesome Old Podcast Episode 99</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=def_video_2"/>
    <published>2026-11-14T00:00:00+00:00</published>
  </entry>
</feed>
"""


def test_channel_watcher_add_remove(tmp_path: Path):
    state_file = tmp_path / "watcher.json"
    service = ChannelWatcherService(state_path=state_file)

    ch = service.add_channel(
        platform="youtube",
        channel_identifier="@testcreator",
        creator_name="Test Creator",
        mode="vod",
    )

    assert ch.platform == "youtube"
    assert ch.channel_identifier == "@testcreator"
    assert len(service.list_channels()) == 1

    # Reload from disk
    service2 = ChannelWatcherService(state_path=state_file)
    assert len(service2.list_channels()) == 1
    assert service2.list_channels()[0]["creator_name"] == "Test Creator"

    # Remove channel
    assert service2.remove_channel("youtube:@testcreator")
    assert len(service2.list_channels()) == 0


def test_channel_watcher_rss_feed(tmp_path: Path):
    state_file = tmp_path / "watcher_rss.json"
    service = ChannelWatcherService(state_path=state_file)

    channel = service.add_channel(
        platform="youtube",
        channel_identifier="UC1234567890123456789012",
        creator_name="Test Channel",
        mode="vod",
    )
    channel.resolved_channel_id = "UC1234567890123456789012"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_cm = MagicMock()
        mock_cm.read.return_value = SAMPLE_RSS_FEED.encode("utf-8")
        mock_resp = MagicMock()
        mock_resp.__enter__.return_value = mock_cm
        mock_urlopen.return_value = mock_resp

        new_vods = service.check_youtube_rss_vods(channel)
        assert len(new_vods) == 2
        assert new_vods[0]["video_id"] == "abc_video_1"
        assert new_vods[0]["title"] == "Awesome New Podcast Episode 100"
        assert channel.last_seen_vod_id == "abc_video_1"

        # Subsequent check without new videos should stop at last_seen_vod_id
        new_vods_next = service.check_youtube_rss_vods(channel)
        assert len(new_vods_next) == 0


def test_channel_watcher_polling(tmp_path: Path):
    state_file = tmp_path / "watcher_poll.json"
    vod_events = []
    live_events = []

    def on_vod(ch, url, title):
        vod_events.append((ch.creator_name, url, title))

    def on_live(ch, url):
        live_events.append((ch.creator_name, url))

    service = ChannelWatcherService(
        state_path=state_file,
        on_new_vod=on_vod,
        on_stream_start=on_live,
    )

    channel = service.add_channel(
        platform="youtube",
        channel_identifier="UC1234567890123456789012",
        creator_name="Channel 1",
        mode="both",
    )
    channel.resolved_channel_id = "UC1234567890123456789012"

    with patch.object(service, "check_youtube_rss_vods") as mock_vods, \
         patch.object(service, "check_channel_live_status") as mock_live:
        mock_vods.return_value = [{"video_id": "v1", "title": "Vid 1", "url": "http://yt/v1"}]
        mock_live.return_value = (True, "https://youtube.com/live/stream1")

        service.poll_all_channels()

        assert len(vod_events) == 1
        assert vod_events[0] == ("Channel 1", "http://yt/v1", "Vid 1")
        assert len(live_events) == 1
        assert live_events[0] == ("Channel 1", "https://youtube.com/live/stream1")
        assert channel.last_live_state is True
