"""Profile & Channel Live Status Extractor for OpenShorts.

High-speed public metadata extraction for YouTube, Twitch, and Kick:
- Twitch: Fast GraphQL API (https://gql.twitch.tv/gql) with public Client-ID
- Kick: Public JSON API v2 (https://kick.com/api/v2/channels/{slug})
- YouTube: Channel /live scraping & OpenGraph tags
- 75-second caching for live status queries
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("openshorts.profiles")

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"

_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_live_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_lock = threading.Lock()


def platform_of(url: str) -> str:
    url_lower = url.lower()
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "YouTube"
    if "twitch.tv" in url_lower:
        return "Twitch"
    if "kick.com" in url_lower:
        return "Kick"
    return ""


def fetch_channel_profile(channel_url: str) -> Dict[str, Any]:
    """Retrieve public channel metadata (name, platform, follower count, avatar url)."""
    clean_url = channel_url.strip()
    if not clean_url.startswith("http"):
        clean_url = "https://" + clean_url

    clean_url = re.sub(r"/(live|videos|streams|featured)/?$", "", clean_url.split("?")[0].rstrip("/"))
    plat = platform_of(clean_url)

    now = time.time()
    with _lock:
        if clean_url in _cache:
            ts, data = _cache[clean_url]
            if now - ts < 1800:
                return data

    res = _fetch_platform_profile(clean_url, plat)
    with _lock:
        _cache[clean_url] = (now, res)
    return res


def _fetch_platform_profile(url: str, plat: str) -> Dict[str, Any]:
    if plat == "Kick":
        slug = url.split("/")[-1]
        api_url = f"https://kick.com/api/v2/channels/{slug}"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            user = data.get("user") or {}
            followers = data.get("followers_count", 0)
            return {
                "platform": "Kick",
                "name": user.get("username") or slug,
                "followers": f"{followers} followers",
                "avatar_url": user.get("profile_pic") or "",
                "slug": slug,
            }
        except Exception:
            return {"platform": "Kick", "name": slug, "followers": "", "avatar_url": "", "slug": slug}

    elif plat == "Twitch":
        slug = url.split("/")[-1].lower()
        # Query Twitch GQL
        query = {
            "query": """query { user(login: "%s") { displayName profileImageURL(width: 300) stream { id title viewersCount } } }""" % slug
        }
        try:
            req = urllib.request.Request(
                "https://gql.twitch.tv/gql",
                data=json.dumps(query).encode("utf-8"),
                headers={"Client-ID": TWITCH_CLIENT_ID, "Content-Type": "application/json", "User-Agent": UA},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            user_data = data.get("data", {}).get("user") or {}
            stream_data = user_data.get("stream")
            return {
                "platform": "Twitch",
                "name": user_data.get("displayName") or slug,
                "followers": "",
                "avatar_url": user_data.get("profileImageURL") or "",
                "slug": slug,
                "is_live": stream_data is not None,
                "viewers": stream_data.get("viewersCount", 0) if stream_data else 0,
                "stream_title": stream_data.get("title", "") if stream_data else "",
            }
        except Exception:
            return {"platform": "Twitch", "name": slug, "followers": "", "avatar_url": "", "slug": slug}

    elif plat == "YouTube":
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            title_m = re.search(r'<meta property="og:title" content="([^"]+)"', html)
            pic_m = re.search(r'<meta property="og:image" content="([^"]+)"', html)
            name = title_m.group(1) if title_m else url.split("/")[-1]
            pic = pic_m.group(1) if pic_m else ""
            return {"platform": "YouTube", "name": name, "followers": "", "avatar_url": pic, "slug": name}
        except Exception:
            return {"platform": "YouTube", "name": url.split("/")[-1], "followers": "", "avatar_url": "", "slug": ""}

    return {"platform": plat or "Video", "name": url.split("/")[-1], "followers": "", "avatar_url": "", "slug": ""}


def check_live_cached(channel_url: str) -> Dict[str, Any]:
    """Check whether a channel is currently live with 75-second multi-user caching."""
    now = time.time()
    with _lock:
        if channel_url in _live_cache:
            ts, val = _live_cache[channel_url]
            if now - ts < 75.0:
                return val

    profile = fetch_channel_profile(channel_url)
    plat = profile.get("platform")

    is_live = False
    viewers = 0
    title = ""

    if plat == "Twitch":
        is_live = profile.get("is_live", False)
        viewers = profile.get("viewers", 0)
        title = profile.get("stream_title", "")
    elif plat == "Kick":
        slug = profile.get("slug")
        try:
            req = urllib.request.Request(f"https://kick.com/api/v2/channels/{slug}", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            livestream = d.get("livestream")
            if livestream and livestream.get("is_live"):
                is_live = True
                viewers = livestream.get("viewer_count", 0)
                title = livestream.get("session_title", "")
        except Exception:
            pass
    elif plat == "YouTube":
        live_url = channel_url.rstrip("/") + "/live"
        try:
            req = urllib.request.Request(live_url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=8) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
            if '"isLiveNow":true' in html or '"liveStreamabilityRenderer"' in html:
                is_live = True
                m = re.search(r'"title":\{"runs":\[\{"text":"([^"]+)"', html)
                if m:
                    title = m.group(1)
        except Exception:
            pass

    res = {
        "is_live": is_live,
        "viewers": viewers,
        "title": title,
        "checked_at": now,
    }
    with _lock:
        _live_cache[channel_url] = (now, res)
    return res
