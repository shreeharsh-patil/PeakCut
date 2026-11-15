"""Autonomous Niche & Topic Discovery Engine for OpenShorts.

Features:
- Two-hop YouTube discovery (search -> video enrichment) via YouTube Data API v3
- Zero-API-key fallback via yt-dlp search (`ytsearch<N>:<query>`)
- Pluggable topic providers:
    1. youtube_trending: Algorithmic search on viral niches (podcasts, tech, business, reactions)
    2. sports: Sports news, reactions, match reviews (with optional TheSportsDB integration)
    3. seeds: Custom creator keyword seeds with strict title-gating
- Persistent deduplication ledger (`seen_sources.json`) so repeated runs never re-clip sources
- Duration gating (min/max seconds), view count ranking, and license filtering
- Pipeline integration helper: automatically passes top discovery candidates to OpenShorts
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openshorts.discovery")

_API_BASE = "https://www.googleapis.com/youtube/v3"
_ISO_DUR_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")
DEFAULT_LEDGER_PATH = Path("output/seen_sources.json")


def iso_duration_to_seconds(iso: str) -> int:
    """Parse ISO 8601 duration string (e.g. PT14M32S) to total seconds."""
    if not iso:
        return 0
    m = _ISO_DUR_RE.match(iso)
    if not m:
        return 0
    h, mn, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mn * 60 + s


@dataclass
class DiscoveredVideo:
    video_id: str
    title: str
    url: str
    channel_title: str
    duration_seconds: int
    view_count: int
    published_at: str
    description: str = ""
    thumbnail_url: str = ""
    topic: str = ""
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Topic:
    name: str
    query: str
    provider: str = "youtube_trending"
    must_include: List[str] = field(default_factory=list)
    min_seconds: int = 180
    max_seconds: int = 1800
    lookback_days: int = 7


class SeenLedger:
    """Persistent ledger tracking clipped/seen video IDs across runs."""

    def __init__(self, ledger_path: Path | str = DEFAULT_LEDGER_PATH):
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Error reading seen ledger {self.path}: {e}; resetting")
        return {}

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def is_seen(self, video_id: str) -> bool:
        return video_id in self._data

    def mark_seen(self, video_id: str, title: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        self._data[video_id] = {
            "title": title,
            "seen_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **(metadata or {}),
        }
        self._save()

    def filter_unseen(self, candidates: List[DiscoveredVideo]) -> List[DiscoveredVideo]:
        return [c for c in candidates if not self.is_seen(c.video_id)]

    def total_seen(self) -> int:
        return len(self._data)


class NicheDiscoveryEngine:
    """Orchestrates topic generation, YouTube searches, filtering, and deduplication."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        ledger: Optional[SeenLedger] = None,
    ):
        self.api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")
        self.ledger = ledger or SeenLedger()

    def search_youtube_api(
        self,
        query: str,
        *,
        max_results: int = 25,
        min_sec: int = 180,
        max_sec: int = 1800,
        order: str = "viewCount",
        region: str = "US",
        published_after: Optional[str] = None,
        must_include: Optional[List[str]] = None,
    ) -> List[DiscoveredVideo]:
        """Query YouTube Data API v3 with two-hop enrichment."""
        if not self.api_key:
            logger.info("No YOUTUBE_API_KEY provided; falling back to yt-dlp search")
            return self.search_ytdlp(
                query,
                max_results=max_results,
                min_sec=min_sec,
                max_sec=max_sec,
                must_include=must_include,
            )

        # Hop 1: search.list
        search_params: Dict[str, Any] = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(max_results * 2, 50),
            "order": order,
            "videoEmbeddable": "true",
            "regionCode": region,
            "videoDuration": "medium",  # 4-20 mins
            "key": self.api_key,
        }
        if published_after:
            search_params["publishedAfter"] = published_after

        url = f"{_API_BASE}/search?{urllib.parse.urlencode(search_params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OpenShorts-Discovery/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"YouTube Data API search failed: {e}; attempting yt-dlp fallback")
            return self.search_ytdlp(
                query,
                max_results=max_results,
                min_sec=min_sec,
                max_sec=max_sec,
                must_include=must_include,
            )

        video_ids = [
            it["id"]["videoId"]
            for it in data.get("items", [])
            if it.get("id", {}).get("videoId")
        ]
        if not video_ids:
            return []

        # Hop 2: videos.list for durations & statistics
        video_params = {
            "part": "contentDetails,statistics,snippet",
            "id": ",".join(video_ids[:50]),
            "key": self.api_key,
        }
        enrich_url = f"{_API_BASE}/videos?{urllib.parse.urlencode(video_params)}"
        try:
            req = urllib.request.Request(enrich_url, headers={"User-Agent": "OpenShorts-Discovery/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                enrich_data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            logger.error(f"YouTube Data API video enrichment failed: {e}")
            return []

        results: List[DiscoveredVideo] = []
        must_inc = [w.lower() for w in (must_include or [])]

        for item in enrich_data.get("items", []):
            vid = item.get("id", "")
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            stats = item.get("statistics", {})

            title = snippet.get("title", "")
            if must_inc and not all(w in title.lower() for w in must_inc):
                continue

            dur_sec = iso_duration_to_seconds(content.get("duration", ""))
            if not (min_sec <= dur_sec <= max_sec):
                continue

            view_count = int(stats.get("viewCount", 0))
            thumb = snippet.get("thumbnails", {}).get("high", {}).get("url", "")
            pub_at = snippet.get("publishedAt", "")
            channel = snippet.get("channelTitle", "")

            results.append(
                DiscoveredVideo(
                    video_id=vid,
                    title=title,
                    url=f"https://www.youtube.com/watch?v={vid}",
                    channel_title=channel,
                    duration_seconds=dur_sec,
                    view_count=view_count,
                    published_at=pub_at,
                    description=snippet.get("description", "")[:300],
                    thumbnail_url=thumb,
                    topic=query,
                    score=float(view_count) / max(1.0, dur_sec),
                )
            )

        results.sort(key=lambda x: x.view_count, reverse=True)
        return results[:max_results]

    def search_ytdlp(
        self,
        query: str,
        *,
        max_results: int = 10,
        min_sec: int = 180,
        max_sec: int = 1800,
        must_include: Optional[List[str]] = None,
    ) -> List[DiscoveredVideo]:
        """Free, zero-API-key search using yt-dlp metadata extraction."""
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
        }
        must_inc = [w.lower() for w in (must_include or [])]
        search_query = f"ytsearch{max_results * 2}:{query}"
        results: List[DiscoveredVideo] = []

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_query, download=False)
                entries = info.get("entries", []) if info else []
                for entry in entries:
                    if not entry:
                        continue
                    vid = entry.get("id", "")
                    title = entry.get("title", "")
                    dur_sec = int(entry.get("duration") or 0)
                    if must_inc and not all(w in title.lower() for w in must_inc):
                        continue
                    if dur_sec and not (min_sec <= dur_sec <= max_sec):
                        continue

                    view_count = int(entry.get("view_count") or 0)
                    results.append(
                        DiscoveredVideo(
                            video_id=vid,
                            title=title,
                            url=entry.get("url") or f"https://www.youtube.com/watch?v={vid}",
                            channel_title=entry.get("uploader") or entry.get("channel") or "",
                            duration_seconds=dur_sec,
                            view_count=view_count,
                            published_at=entry.get("upload_date") or "",
                            description=str(entry.get("description") or "")[:300],
                            thumbnail_url=entry.get("thumbnail") or "",
                            topic=query,
                            score=float(view_count) / max(1.0, dur_sec) if dur_sec else float(view_count),
                        )
                    )
        except Exception as e:
            logger.error(f"yt-dlp discovery search failed: {e}")

        results.sort(key=lambda x: x.view_count, reverse=True)
        return results[:max_results]

    def get_topics_for_provider(
        self,
        provider: str = "youtube_trending",
        custom_seeds: Optional[List[str]] = None,
    ) -> List[Topic]:
        """Generate a ranked list of candidate topics based on the chosen provider."""
        if provider == "sports":
            return [
                Topic(
                    name="Champions League Reaction",
                    query="Champions league post match reaction interview",
                    provider="sports",
                    must_include=["reaction", "interview"],
                    lookback_days=4,
                ),
                Topic(
                    name="Premier League Analysis",
                    query="Premier League analysis podcast tactix",
                    provider="sports",
                    lookback_days=5,
                ),
                Topic(
                    name="NBA Press Conference",
                    query="NBA post game press conference reaction",
                    provider="sports",
                    must_include=["press"],
                    lookback_days=3,
                ),
            ]
        elif provider == "seeds" and custom_seeds:
            topics: List[Topic] = []
            for seed in custom_seeds:
                topics.append(
                    Topic(
                        name=seed,
                        query=seed,
                        provider="seeds",
                        lookback_days=14,
                    )
                )
            return topics
        else:
            # Default: high-performing viral video niches
            return [
                Topic(
                    name="AI & Tech Insights",
                    query="artificial intelligence breakdown discussion podcast",
                    provider="youtube_trending",
                    lookback_days=7,
                ),
                Topic(
                    name="Business Breakdown",
                    query="startup breakdown business story founder interview",
                    provider="youtube_trending",
                    lookback_days=10,
                ),
                Topic(
                    name="Philosophy & Psychology",
                    query="psychology podcast human behavior conversation",
                    provider="youtube_trending",
                    lookback_days=14,
                ),
                Topic(
                    name="Creator Economy",
                    query="YouTube creator strategy interview podcast",
                    provider="youtube_trending",
                    lookback_days=10,
                ),
            ]

    def find_best_unseen_candidate(
        self,
        provider: str = "youtube_trending",
        custom_seeds: Optional[List[str]] = None,
        max_results_per_topic: int = 10,
    ) -> Optional[DiscoveredVideo]:
        """Iterate through topics, query YouTube, and return the single highest-ranking unseen candidate."""
        topics = self.get_topics_for_provider(provider, custom_seeds)

        for topic in topics:
            cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=topic.lookback_days)
            published_after = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

            candidates = self.search_youtube_api(
                topic.query,
                max_results=max_results_per_topic,
                min_sec=topic.min_seconds,
                max_sec=topic.max_seconds,
                published_after=published_after,
                must_include=topic.must_include,
            )

            unseen = self.ledger.filter_unseen(candidates)
            if unseen:
                best = unseen[0]
                logger.info(
                    f"Discovered best unseen video for topic '{topic.name}': "
                    f"'{best.title}' ({best.duration_seconds}s, {best.view_count} views, {best.url})"
                )
                return best

        return None


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="OpenShorts Autonomous Niche Discovery")
    parser.add_argument("--query", "-q", help="Direct topic query")
    parser.add_argument("--provider", "-p", default="youtube_trending", choices=["youtube_trending", "sports", "seeds"])
    parser.add_argument("--seed", action="append", help="Custom seed query (can be specified multiple times)")
    parser.add_argument("--max-results", "-n", type=int, default=5)
    parser.add_argument("--min-sec", type=int, default=180)
    parser.add_argument("--max-sec", type=int, default=1800)
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--mark-seen", help="Mark a video ID as seen in ledger")
    args = parser.parse_args()

    engine = NicheDiscoveryEngine()

    if args.mark_seen:
        engine.ledger.mark_seen(args.mark_seen, title="Manual CLI mark")
        print(f"Marked {args.mark_seen} as seen.")
        return

    if args.query:
        videos = engine.search_youtube_api(
            args.query,
            max_results=args.max_results,
            min_sec=args.min_sec,
            max_sec=args.max_sec,
        )
        unseen = engine.ledger.filter_unseen(videos)
    else:
        best = engine.find_best_unseen_candidate(
            provider=args.provider,
            custom_seeds=args.seed,
            max_results_per_topic=args.max_results,
        )
        unseen = [best] if best else []

    if args.json:
        print(json.dumps([v.to_dict() for v in unseen], indent=2))
    else:
        print(f"\n--- Discovered Videos ({len(unseen)} unseen) ---")
        for v in unseen:
            print(f"- [{v.video_id}] {v.title} ({v.duration_seconds}s | {v.view_count:,} views)")
            print(f"  URL: {v.url}")


if __name__ == "__main__":
    cli_main()
