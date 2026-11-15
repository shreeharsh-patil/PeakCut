"""Autonomous Multi-Platform Channel Watcher for OpenShorts.

Features:
- Watches creator channels across YouTube, Twitch, and Kick
- Dual operational monitoring modes:
    1. 'live': Detects when the channel goes live and automatically starts rolling-buffer clipping
    2. 'vod': Monitors channel feeds (zero-quota YouTube RSS Atom feeds & API) for new long-form uploads
- Automatically submits discovered new uploads to OpenShorts' processing queue
- Persistent channel registry and state ledger (`channel_watcher.json`)
- Clean background daemon worker with thread-safe controls
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("openshorts.channel_watcher")

DEFAULT_WATCHER_STATE = Path("output/channel_watcher.json")
YOUTUBE_RSS_FEED_BASE = "https://www.youtube.com/feeds/videos.xml?channel_id="
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


@dataclass
class WatchedChannel:
    id: str  # e.g. "youtube:@hubermanlab" or "twitch:shroud"
    platform: str  # 'youtube', 'twitch', 'kick'
    channel_identifier: str  # handle, username, or UC... channel id
    creator_name: str
    mode: str = "both"  # 'live', 'vod', 'both'
    enabled: bool = True
    resolved_channel_id: str = ""
    last_live_state: bool = False
    last_seen_vod_id: str = ""
    last_checked_at: float = 0.0
    auto_publish: bool = False
    publish_account_id: str = ""
    target_layouts: List[str] = field(default_factory=lambda: ["auto"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChannelWatcherService:
    """Monitors configured channels for live streams and new VOD uploads."""

    def __init__(
        self,
        state_path: Path | str = DEFAULT_WATCHER_STATE,
        on_new_vod: Optional[Callable[[WatchedChannel, str, str], None]] = None,
        on_stream_start: Optional[Callable[[WatchedChannel, str], None]] = None,
    ):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.on_new_vod = on_new_vod
        self.on_stream_start = on_stream_start
        self.lock = threading.RLock()
        self.channels: Dict[str, WatchedChannel] = self._load()
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

    def _load(self) -> Dict[str, WatchedChannel]:
        if not self.state_path.exists():
            return {}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            return {k: WatchedChannel(**v) for k, v in raw.items()}
        except Exception as e:
            logger.warning(f"Failed to load channel watcher state {self.state_path}: {e}")
            return {}

    def _save(self) -> None:
        with self.lock:
            tmp = self.state_path.with_suffix(".tmp")
            data = {k: v.to_dict() for k, v in self.channels.items()}
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.state_path)

    def add_channel(
        self,
        platform: str,
        channel_identifier: str,
        creator_name: str = "",
        mode: str = "both",
        auto_publish: bool = False,
        publish_account_id: str = "",
        target_layouts: Optional[List[str]] = None,
    ) -> WatchedChannel:
        """Register a channel to watch."""
        clean_id = channel_identifier.strip()
        channel_key = f"{platform.lower()}:{clean_id.lower()}"

        with self.lock:
            if channel_key in self.channels:
                ch = self.channels[channel_key]
                ch.enabled = True
                ch.mode = mode
                ch.auto_publish = auto_publish
                ch.publish_account_id = publish_account_id
                if target_layouts:
                    ch.target_layouts = target_layouts
                self._save()
                return ch

            ch = WatchedChannel(
                id=channel_key,
                platform=platform.lower(),
                channel_identifier=clean_id,
                creator_name=creator_name or clean_id,
                mode=mode,
                auto_publish=auto_publish,
                publish_account_id=publish_account_id,
                target_layouts=target_layouts or ["auto"],
            )
            self.channels[channel_key] = ch
            self._save()
            logger.info(f"Added channel to watch: {channel_key} (mode: {mode})")
            return ch

    def remove_channel(self, channel_key: str) -> bool:
        """Unregister a watched channel."""
        with self.lock:
            if channel_key in self.channels:
                del self.channels[channel_key]
                self._save()
                logger.info(f"Removed watched channel: {channel_key}")
                return True
            return False

    def list_channels(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [ch.to_dict() for ch in self.channels.values()]

    def resolve_youtube_channel_id(self, identifier: str) -> str:
        """Resolve a handle (@username) or vanity name to a canonical YouTube UC... channel ID."""
        if identifier.startswith("UC") and len(identifier) == 24:
            return identifier

        # Try scraping handle page or using yt-dlp metadata
        url = f"https://www.youtube.com/{identifier}" if identifier.startswith("@") else f"https://www.youtube.com/c/{identifier}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                m = re.search(r'channel_id=([a-zA-Z0-9_-]{24})', html) or re.search(r'"channelId":"([a-zA-Z0-9_-]{24})"', html)
                if m:
                    return m.group(1)
        except Exception as e:
            logger.debug(f"Failed to scrape channel ID for {identifier}: {e}")

        # Fallback using yt-dlp extract flat
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and info.get("channel_id"):
                    return info["channel_id"]
        except Exception:
            pass

        return ""

    def check_youtube_rss_vods(self, channel: WatchedChannel) -> List[Dict[str, str]]:
        """Fetch YouTube RSS Atom feed to find newly uploaded videos with zero API quota."""
        if not channel.resolved_channel_id:
            channel.resolved_channel_id = self.resolve_youtube_channel_id(channel.channel_identifier)
            if not channel.resolved_channel_id:
                return []

        rss_url = f"{YOUTUBE_RSS_FEED_BASE}{channel.resolved_channel_id}"
        new_videos: List[Dict[str, str]] = []

        try:
            req = urllib.request.Request(rss_url, headers={"User-Agent": "OpenShorts-ChannelWatcher/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            entries = root.findall("atom:entry", ATOM_NS)

            for entry in entries:
                vid_elem = entry.find("yt:videoId", ATOM_NS)
                title_elem = entry.find("atom:title", ATOM_NS)
                if vid_elem is None or not vid_elem.text:
                    continue

                vid_id = vid_elem.text
                title = title_elem.text if title_elem is not None else ""
                url = f"https://www.youtube.com/watch?v={vid_id}"

                if vid_id == channel.last_seen_vod_id:
                    break

                new_videos.append({"video_id": vid_id, "title": title, "url": url})

            if entries:
                first_vid = entries[0].find("yt:videoId", ATOM_NS)
                if first_vid is not None and first_vid.text:
                    channel.last_seen_vod_id = first_vid.text
                    self._save()

        except Exception as e:
            logger.warning(f"Error checking RSS feed for {channel.id}: {e}")

        return new_videos

    def check_channel_live_status(self, channel: WatchedChannel) -> Tuple[bool, str]:
        """Check if a Twitch, Kick, or YouTube channel is broadcasting live."""
        if channel.platform == "twitch":
            url = f"https://www.twitch.tv/{channel.channel_identifier.lower()}"
        elif channel.platform == "kick":
            url = f"https://kick.com/{channel.channel_identifier.lower()}"
        else:
            url = f"https://www.youtube.com/{channel.channel_identifier}/live"

        try:
            import yt_dlp
            opts = {
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "extract_flat": True,
            }
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info and (info.get("is_live") or info.get("live_status") == "is_live"):
                    return True, url
        except Exception:
            pass

        return False, url

    def poll_all_channels(self) -> None:
        """Execute a full poll pass over all registered channels."""
        channels_to_check = [c for c in self.channels.values() if c.enabled]

        for channel in channels_to_check:
            channel.last_checked_at = time.time()

            # Check VODs
            if channel.mode in ("vod", "both") and channel.platform == "youtube":
                new_vods = self.check_youtube_rss_vods(channel)
                for vod in new_vods:
                    logger.info(f"[NEW VOD DETECTED] Channel '{channel.creator_name}': {vod['title']} ({vod['url']})")
                    if self.on_new_vod:
                        try:
                            self.on_new_vod(channel, vod["url"], vod["title"])
                        except Exception as e:
                            logger.error(f"Error handling new VOD callback: {e}")

            # Check Live
            if channel.mode in ("live", "both"):
                is_live, stream_url = self.check_channel_live_status(channel)
                if is_live and not channel.last_live_state:
                    logger.info(f"[CHANNEL WENT LIVE] Channel '{channel.creator_name}' is now streaming! ({stream_url})")
                    channel.last_live_state = True
                    self._save()
                    if self.on_stream_start:
                        try:
                            self.on_stream_start(channel, stream_url)
                        except Exception as e:
                            logger.error(f"Error handling live stream start callback: {e}")
                elif not is_live and channel.last_live_state:
                    logger.info(f"[CHANNEL WENT OFFLINE] Channel '{channel.creator_name}' stopped streaming.")
                    channel.last_live_state = False
                    self._save()

    def start_background_watcher(self, poll_interval_sec: float = 120.0) -> None:
        """Start the background monitoring loop."""
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()

        def _loop() -> None:
            logger.info("Channel watcher background thread started.")
            while not self.stop_event.is_set():
                try:
                    self.poll_all_channels()
                except Exception as e:
                    logger.error(f"Error in channel watcher poll loop: {e}")
                self.stop_event.wait(poll_interval_sec)
            logger.info("Channel watcher background thread stopped.")

        self.worker_thread = threading.Thread(target=_loop, daemon=True, name="OpenShorts-ChannelWatcher")
        self.worker_thread.start()

    def stop_background_watcher(self) -> None:
        self.stop_event.set()
        if self.worker_thread:
            self.worker_thread.join(timeout=5)


_GLOBAL_WATCHER: Optional[ChannelWatcherService] = None


def get_channel_watcher() -> ChannelWatcherService:
    global _GLOBAL_WATCHER
    if _GLOBAL_WATCHER is None:
        _GLOBAL_WATCHER = ChannelWatcherService()
        _GLOBAL_WATCHER.start_background_watcher()
    return _GLOBAL_WATCHER
