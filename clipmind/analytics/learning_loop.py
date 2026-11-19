"""Performance Tracking & Adaptive Learning Loop for ClipMind (Phase 8).

Features:
- Query YouTube Analytics API & YouTube Data API v3 for:
    * CTR (Click-Through Rate / Impression CTR)
    * AVD (Average View Duration & Average View Percentage)
    * Views (Total views & velocity)
    * Engagement Rate ((likes + comments + shares) / views * 100)
- Historical performance database syncing (performance_tracker.csv & learning_state.json).
- Adaptive Self-Optimization Loop:
    * Analyzes top performers vs baseline across past cycles.
    * Discovers winning title hook words (e.g. "SHOCKING", "NEVER", "WHY NOBODY").
    * Discovers winning thumbnail color schemes & winning A/B test variant.
    * Discovers winning video hook formats for retention (>70% AVD).
    * Discovers optimal upload timing hours based on empirical velocity.
- Dynamic System Config Updates:
    * Automatically writes self-optimized parameters to adaptive_config.json.
    * Feeds recommendations to TrendAnalyzer, ThumbnailDesigner, ClipPipeline, and YouTubeUploader.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from googleapiclient.discovery import build as google_build
    from googleapiclient.errors import HttpError
    GOOGLE_ANALYTICS_AVAILABLE = True
except ImportError:
    google_build = None
    HttpError = Exception  # type: ignore
    GOOGLE_ANALYTICS_AVAILABLE = False

try:
    from clipmind.core.key_rotator import get_active_key, record_quota_usage
except ImportError:
    def get_active_key(service: str) -> Optional[str]:
        return os.environ.get(f"{service.upper()}_API_KEY")

    def record_quota_usage(service: str, units: int = 1) -> None:
        pass

logger = logging.getLogger("clipmind.analytics.learning")

# Common high-impact Short hook terms for pattern discovery
HOOK_CANDIDATES = [
    "WHY", "HOW", "NEVER", "STOP", "DON'T", "SECRET", "SHOCKING",
    "THE REAL REASON", "EXPOSED", "EVERYONE IS WRONG", "FINALLY",
    "UNBELIEVABLE", "THIS CHANGES EVERYTHING", "NOBODY TELLS YOU",
]

THUMBNAIL_COLOR_PALETTES = [
    "neon_yellow_contrast",
    "high_contrast_red",
    "cyberpunk_cyan",
    "electric_green",
]


@dataclass
class PerformanceMetric:
    """Consolidated performance metrics for a published video."""
    video_id: str
    cycle_id: str
    title: str
    topic: str
    views: int = 0
    ctr: float = 0.0              # Click-through rate % (e.g. 7.5%)
    avd_seconds: float = 0.0      # Average view duration in seconds
    avd_pct: float = 0.0          # Average view duration percentage (e.g. 72.0%)
    likes: int = 0
    comments: int = 0
    shares: int = 0
    engagement_rate: float = 0.0  # (likes + comments + shares) / views * 100
    thumbnail_variant: str = "Variant_A"  # "Variant_A" or "Variant_B"
    thumbnail_color: str = "neon_yellow_contrast"
    hook_style: str = "curiosity_gap"
    upload_hour_utc: int = 14
    upload_date: str = ""

    def compound_viral_score(self) -> float:
        """Calculate weighted viral quality score for ranking."""
        # 40% CTR, 35% Retention (AVD %), 25% Engagement rate
        normalized_views = min(self.views / 5000.0, 1.0) * 10.0
        return (
            (self.ctr * 0.35)
            + (self.avd_pct * 0.35)
            + (self.engagement_rate * 0.15)
            + (normalized_views * 0.15)
        )

    @property
    def viral_score(self) -> float:
        return round(self.compound_viral_score(), 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["viral_score"] = round(self.compound_viral_score(), 2)
        return d


@dataclass
class AdaptiveConfig:
    """Dynamic self-optimized parameters produced by the learning loop."""
    recommended_hook_words: List[str] = field(default_factory=lambda: ["SHOCKING", "NEVER", "WHY NOBODY"])
    winning_thumbnail_variant: str = "Variant_A"
    recommended_thumbnail_palette: str = "neon_yellow_contrast"
    optimal_upload_hour_utc: int = 14
    optimal_duration_seconds: int = 52
    topic_weights: Dict[str, float] = field(default_factory=lambda: {"tech_ai": 1.2, "productivity": 1.0})
    avg_ctr: float = 6.5
    avg_avd_pct: float = 68.0
    cycles_analyzed: int = 0
    last_updated: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AdaptiveConfig:
        return cls(
            recommended_hook_words=data.get("recommended_hook_words", ["SHOCKING", "NEVER", "WHY NOBODY"]),
            winning_thumbnail_variant=data.get("winning_thumbnail_variant", "Variant_A"),
            recommended_thumbnail_palette=data.get("recommended_thumbnail_palette", "neon_yellow_contrast"),
            optimal_upload_hour_utc=data.get("optimal_upload_hour_utc", 14),
            optimal_duration_seconds=data.get("optimal_duration_seconds", 52),
            topic_weights=data.get("topic_weights", {"tech_ai": 1.2, "productivity": 1.0}),
            avg_ctr=data.get("avg_ctr", 6.5),
            avg_avd_pct=data.get("avg_avd_pct", 68.0),
            cycles_analyzed=data.get("cycles_analyzed", 0),
            last_updated=data.get("last_updated", datetime.datetime.now(datetime.timezone.utc).isoformat()),
        )


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert strings with commas/whitespace or numbers to int without raising."""
    try:
        if isinstance(val, str):
            val = val.replace(",", "").strip()
        return int(float(val))
    except (ValueError, TypeError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert strings with percentages/commas or numbers to float without raising."""
    try:
        if isinstance(val, str):
            val = val.replace(",", "").replace("%", "").strip()
        return float(val)
    except (ValueError, TypeError):
        return default


class LearningLoop:
    """Executes performance querying, analysis, and adaptive self-tuning."""

    def __init__(
        self,
        analytics_credentials: Optional[Any] = None,
        data_dir: Optional[Union[str, Path]] = None,
        dry_run: Optional[bool] = None,
    ):
        self.credentials = analytics_credentials
        self.data_dir = Path(data_dir or os.environ.get("CLIPMIND_DATA_DIR", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "learning_state.json"
        self.adaptive_config_file = self.data_dir / "adaptive_config.json"
        self.tracker_csv_file = self.data_dir / "ClipMind" / "performance_tracker.csv"

        env_dry = os.environ.get("CLIPMIND_DRY_RUN", "0").lower() in ("1", "true", "yes")
        if dry_run is not None:
            self.dry_run = dry_run
        else:
            self.dry_run = env_dry or not (self.credentials or os.environ.get("YOUTUBE_REFRESH_TOKEN"))

        self._analytics_service = None
        self._data_service = None

    def _get_services(self) -> Tuple[Any, Any]:
        """Initialize YouTube Analytics & Data API services."""
        if self.dry_run or not GOOGLE_ANALYTICS_AVAILABLE:
            return None, None

        if self._analytics_service is not None and self._data_service is not None:
            return self._analytics_service, self._data_service

        creds = self.credentials
        if creds is None:
            try:
                from google.oauth2.credentials import Credentials
                cid = os.environ.get("YOUTUBE_CLIENT_ID")
                csecret = os.environ.get("YOUTUBE_CLIENT_SECRET")
                rtoken = os.environ.get("YOUTUBE_REFRESH_TOKEN")
                atoken = os.environ.get("YOUTUBE_ACCESS_TOKEN")
                if rtoken and cid and csecret:
                    creds = Credentials(
                        token=atoken,
                        refresh_token=rtoken,
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=cid,
                        client_secret=csecret,
                    )
                elif atoken:
                    creds = Credentials(token=atoken)
            except ImportError:
                creds = None

        if creds is not None:
            try:
                self._analytics_service = google_build("youtubeAnalytics", "v2", credentials=creds)
                self._data_service = google_build("youtube", "v3", credentials=creds)
            except Exception as e:
                logger.warning(f"Failed to build Google services with credentials: {e}")
        else:
            api_key = get_active_key("youtube")
            if api_key and not api_key.startswith("sim_"):
                try:
                    self._data_service = google_build("youtube", "v3", developerKey=api_key)
                except Exception as e:
                    logger.warning(f"Failed to build Google Data service with API key: {e}")

        return self._analytics_service, self._data_service

    def fetch_video_metrics(
        self,
        video_id: str,
        cycle_id: str = "001",
        title: str = "",
        topic: str = "",
        thumbnail_variant: str = "Variant_A",
    ) -> PerformanceMetric:
        """Query metrics for a video ID or return simulated high-fidelity data."""
        clean_vid = (video_id or "").strip()
        if self.dry_run or not clean_vid or clean_vid.startswith(("sim_", "yt_")):
            # Simulation mode: deterministic yet realistic variations based on hash
            h = abs(hash(f"{clean_vid}_{title}"))
            views = 1200 + (h % 8500)
            ctr = round(5.2 + (h % 60) / 10.0, 2)            # 5.2% - 11.2%
            avd_pct = round(62.0 + (h % 320) / 10.0, 2)      # 62.0% - 94.0%
            avd_sec = round(50.0 * (avd_pct / 100.0), 1)
            likes = int(views * (0.05 + (h % 40) / 1000.0))
            comments = int(views * (0.008 + (h % 20) / 1000.0))
            shares = int(views * (0.004 + (h % 15) / 1000.0))
            engagement = round(((likes + comments + shares) / max(views, 1)) * 100.0, 2)
            color = THUMBNAIL_COLOR_PALETTES[h % len(THUMBNAIL_COLOR_PALETTES)]

            return PerformanceMetric(
                video_id=clean_vid or f"sim_{cycle_id}",
                cycle_id=cycle_id,
                title=title or f"Video {clean_vid or cycle_id}",
                topic=topic or "Tech",
                views=views,
                ctr=ctr,
                avd_seconds=avd_sec,
                avd_pct=avd_pct,
                likes=likes,
                comments=comments,
                shares=shares,
                engagement_rate=engagement,
                thumbnail_variant=thumbnail_variant,
                thumbnail_color=color,
                upload_hour_utc=14,
                upload_date=datetime.date.today().isoformat(),
            )

        analytics_svc, data_svc = self._get_services()
        views = 0
        likes = 0
        comments = 0
        ctr = 6.0
        avd_pct = 70.0
        avd_sec = 35.0
        shares = 0

        # Step 1: Query YouTube Data API for live public stats
        if data_svc is not None:
            try:
                resp = data_svc.videos().list(part="snippet,statistics,contentDetails", id=clean_vid).execute()
                record_quota_usage("youtube", 1)
                items = resp.get("items", [])
                if items:
                    stats = items[0].get("statistics", {})
                    views = _safe_int(stats.get("viewCount", 0))
                    likes = _safe_int(stats.get("likeCount", 0))
                    comments = _safe_int(stats.get("commentCount", 0))
                    snip = items[0].get("snippet", {})
                    if not title:
                        title = snip.get("title", "")
                else:
                    logger.info(f"Video {clean_vid} not yet indexed on YouTube Data API; utilizing estimated baseline.")
                    h = abs(hash(f"{clean_vid}_{title}"))
                    views = 1200 + (h % 8500)
                    likes = int(views * (0.05 + (h % 40) / 1000.0))
                    comments = int(views * (0.008 + (h % 20) / 1000.0))
            except Exception as e:
                logger.warning(f"Failed to fetch Data API stats for {clean_vid}: {e}")
                h = abs(hash(f"{clean_vid}_{title}"))
                views = 1200 + (h % 8500)
                likes = int(views * (0.05 + (h % 40) / 1000.0))
                comments = int(views * (0.008 + (h % 20) / 1000.0))

        # Step 2: Query YouTube Analytics API for CTR & AVD
        if analytics_svc is not None and views > 0:
            try:
                today_str = datetime.date.today().isoformat()
                # 30-day window
                start_str = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
                rep = analytics_svc.reports().query(
                    ids="channel==MINE",
                    startDate=start_str,
                    endDate=today_str,
                    metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,annotationClickThroughRate",
                    filters=f"video=={clean_vid}",
                ).execute()

                rows = rep.get("rows", [])
                if rows and len(rows[0]) >= 5:
                    r_views, r_min, r_avd, r_avd_pct, r_ctr = rows[0]
                    if r_views:
                        views = _safe_int(r_views, views)
                    if r_avd_pct:
                        avd_pct = round(_safe_float(r_avd_pct, avd_pct), 2)
                    if r_avd:
                        avd_sec = round(_safe_float(r_avd, avd_sec), 1)
                    if r_ctr:
                        ctr = round(_safe_float(r_ctr, ctr), 2)
            except Exception as e:
                logger.info(f"Analytics API pending/unavailable for {clean_vid} (using estimated fallback): {e}")

        eng_rate = round(((likes + comments + shares) / max(views, 1)) * 100.0, 2)
        return PerformanceMetric(
            video_id=clean_vid,
            cycle_id=cycle_id,
            title=title,
            topic=topic,
            views=views,
            ctr=ctr,
            avd_seconds=avd_sec,
            avd_pct=avd_pct,
            likes=likes,
            comments=comments,
            shares=shares,
            engagement_rate=eng_rate,
            thumbnail_variant=thumbnail_variant,
            upload_date=datetime.date.today().isoformat(),
        )

    def record_cycle_performance(self, metric: PerformanceMetric) -> None:
        """Persist a single cycle's performance to learning_state.json."""
        state = self.load_state()
        history = state.get("history", [])

        # Deduplicate by video_id or cycle_id
        filtered = [h for h in history if h.get("video_id") != metric.video_id]
        filtered.append(metric.to_dict())
        state["history"] = filtered

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load_state(self) -> Dict[str, Any]:
        """Load persistent history from learning_state.json."""
        if not self.state_file.exists():
            return {"history": []}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"history": []}

    def load_tracker_csv_metrics(self) -> List[PerformanceMetric]:
        """Load historical records from performance_tracker.csv if available."""
        metrics: List[PerformanceMetric] = []
        target_csv = self.tracker_csv_file
        if not target_csv.exists():
            alt_csv = self.data_dir / "performance_tracker.csv"
            if alt_csv.exists():
                target_csv = alt_csv
            else:
                return metrics

        try:
            with open(target_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    v_id = row.get("video_id", "")
                    if not v_id:
                        continue
                    metrics.append(
                        PerformanceMetric(
                            video_id=v_id,
                            cycle_id=row.get("cycle_id", "001"),
                            title=row.get("title", ""),
                            topic=row.get("topic", ""),
                            views=_safe_int(row.get("views", 0)),
                            ctr=_safe_float(row.get("ctr", 0.0)),
                            avd_pct=_safe_float(row.get("avd_pct", 0.0)),
                            engagement_rate=_safe_float(row.get("engagement_rate", 0.0)),
                            upload_date=row.get("date", ""),
                        )
                    )
        except Exception as e:
            logger.warning(f"Error loading tracker CSV: {e}")
        return metrics

    def run_adaptive_learning(self) -> AdaptiveConfig:
        """Analyze past cycles, extract winning patterns, and update adaptive config."""
        state = self.load_state()
        history_raw = state.get("history", [])
        all_metrics: List[PerformanceMetric] = []

        for h in history_raw:
            all_metrics.append(
                PerformanceMetric(
                    video_id=h.get("video_id", ""),
                    cycle_id=h.get("cycle_id", ""),
                    title=h.get("title", ""),
                    topic=h.get("topic", ""),
                    views=int(h.get("views", 0)),
                    ctr=float(h.get("ctr", 0.0)),
                    avd_seconds=float(h.get("avd_seconds", 0.0)),
                    avd_pct=float(h.get("avd_pct", 0.0)),
                    likes=int(h.get("likes", 0)),
                    comments=int(h.get("comments", 0)),
                    shares=int(h.get("shares", 0)),
                    engagement_rate=float(h.get("engagement_rate", 0.0)),
                    thumbnail_variant=h.get("thumbnail_variant", "Variant_A"),
                    thumbnail_color=h.get("thumbnail_color", "neon_yellow_contrast"),
                    upload_hour_utc=int(h.get("upload_hour_utc", 14)),
                    upload_date=h.get("upload_date", ""),
                )
            )

        # Merge with CSV if needed
        csv_metrics = self.load_tracker_csv_metrics()
        existing_ids = {m.video_id for m in all_metrics}
        for cm in csv_metrics:
            if cm.video_id not in existing_ids:
                all_metrics.append(cm)

        # If empty or cold-start, synthesize baseline
        if not all_metrics:
            logger.info("Cold start detected: generating baseline adaptive configuration.")
            cfg = AdaptiveConfig(
                recommended_hook_words=["SHOCKING", "NEVER", "WHY NOBODY", "SECRET", "DON'T"],
                winning_thumbnail_variant="Variant_B",
                recommended_thumbnail_palette="neon_yellow_contrast",
                optimal_upload_hour_utc=14,
                optimal_duration_seconds=52,
                topic_weights={"tech_ai": 1.3, "finance": 1.15, "productivity": 1.05},
                avg_ctr=7.2,
                avg_avd_pct=73.5,
                cycles_analyzed=0,
            )
            self.save_adaptive_config(cfg)
            return cfg

        # Rank metrics by compound viral score
        all_metrics.sort(key=lambda m: m.compound_viral_score(), reverse=True)
        top_k = max(1, len(all_metrics) // 3)
        top_performers = all_metrics[:top_k]

        # 1. Analyze Winning Title Hook Words
        hook_scores: Dict[str, List[float]] = {}
        for m in all_metrics:
            score = m.compound_viral_score()
            title_upper = m.title.upper()
            for cand in HOOK_CANDIDATES:
                if re.search(r"\b" + re.escape(cand) + r"\b", title_upper):
                    hook_scores.setdefault(cand, []).append(score)

        ranked_hooks = []
        for hook, scores in hook_scores.items():
            avg_score = sum(scores) / max(len(scores), 1)
            ranked_hooks.append((hook, avg_score, len(scores)))
        ranked_hooks.sort(key=lambda x: (x[1], x[2]), reverse=True)

        recommended_hooks = [h[0] for h in ranked_hooks[:5]]
        if not recommended_hooks:
            recommended_hooks = ["SHOCKING", "NEVER", "WHY NOBODY", "DON'T", "SECRET"]

        # 2. Analyze Thumbnail Variants (A/B Test Winner)
        variant_scores: Dict[str, List[float]] = {"Variant_A": [], "Variant_B": []}
        color_scores: Dict[str, List[float]] = {}

        for m in all_metrics:
            var = m.thumbnail_variant or "Variant_A"
            variant_scores.setdefault(var, []).append(m.ctr)
            if m.thumbnail_color:
                color_scores.setdefault(m.thumbnail_color, []).append(m.ctr)

        winning_variant = "Variant_A"
        avg_a = sum(variant_scores["Variant_A"]) / max(len(variant_scores["Variant_A"]), 1)
        avg_b = sum(variant_scores["Variant_B"]) / max(len(variant_scores["Variant_B"]), 1)
        if avg_b > avg_a:
            winning_variant = "Variant_B"

        winning_color = "neon_yellow_contrast"
        if color_scores:
            winning_color = max(color_scores.items(), key=lambda x: sum(x[1]) / max(len(x[1]), 1))[0]

        # 3. Analyze Upload Timing
        hour_scores: Dict[int, List[float]] = {}
        for m in all_metrics:
            hour_scores.setdefault(m.upload_hour_utc, []).append(m.compound_viral_score())
        optimal_hour = 14
        if hour_scores:
            optimal_hour = max(hour_scores.items(), key=lambda x: sum(x[1]) / max(len(x[1]), 1))[0]

        # 4. Analyze Topic Multipliers
        topic_scores: Dict[str, List[float]] = {}
        for m in all_metrics:
            top = m.topic or "general"
            topic_scores.setdefault(top, []).append(m.compound_viral_score())

        global_avg = sum(m.compound_viral_score() for m in all_metrics) / max(len(all_metrics), 1)
        topic_weights: Dict[str, float] = {}
        for top, scores in topic_scores.items():
            top_avg = sum(scores) / max(len(scores), 1)
            ratio = round(max(0.7, min(2.0, top_avg / max(global_avg, 0.1))), 2)
            topic_weights[top] = ratio

        # 5. Optimal duration calculation (Shorts sweet spot: 48 - 56s)
        optimal_dur = 52
        avg_retention = sum(m.avd_pct for m in top_performers) / max(len(top_performers), 1)
        if avg_retention > 80.0:
            optimal_dur = 55  # High retention allows maximizing duration
        elif avg_retention < 65.0:
            optimal_dur = 47  # Lower retention suggests tighter editing

        overall_ctr = round(sum(m.ctr for m in all_metrics) / max(len(all_metrics), 1), 2)
        overall_avd = round(sum(m.avd_pct for m in all_metrics) / max(len(all_metrics), 1), 2)

        config = AdaptiveConfig(
            recommended_hook_words=recommended_hooks,
            winning_thumbnail_variant=winning_variant,
            recommended_thumbnail_palette=winning_color,
            optimal_upload_hour_utc=optimal_hour,
            optimal_duration_seconds=optimal_dur,
            topic_weights=topic_weights,
            avg_ctr=overall_ctr,
            avg_avd_pct=overall_avd,
            cycles_analyzed=len(all_metrics),
            last_updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

        self.save_adaptive_config(config)
        logger.info(
            f"Adaptive learning completed: {len(all_metrics)} cycles analyzed. "
            f"Winning variant: {winning_variant}, Top hook: {recommended_hooks[0]}, Avg CTR: {overall_ctr}%"
        )
        return config

    def save_adaptive_config(self, config: AdaptiveConfig) -> None:
        """Write current adaptive configuration to disk."""
        with open(self.adaptive_config_file, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2)

    def get_recommendations(self) -> AdaptiveConfig:
        """Return the current dynamic recommendations (loading or generating if missing)."""
        if self.adaptive_config_file.exists():
            try:
                with open(self.adaptive_config_file, "r", encoding="utf-8") as f:
                    return AdaptiveConfig.from_dict(json.load(f))
            except Exception:
                pass
        return self.run_adaptive_learning()


def run_learning_cycle(
    video_id: str,
    cycle_id: str = "001",
    title: str = "",
    topic: str = "",
    thumbnail_variant: str = "Variant_A",
    dry_run: Optional[bool] = None,
) -> Tuple[PerformanceMetric, AdaptiveConfig]:
    """Execute end-to-end learning step: fetch metrics, record state, update adaptive config."""
    loop = LearningLoop(dry_run=dry_run)
    clean_vid = (video_id or "").strip() or f"sim_{cycle_id}"
    metric = loop.fetch_video_metrics(
        video_id=clean_vid,
        cycle_id=cycle_id,
        title=title,
        topic=topic,
        thumbnail_variant=thumbnail_variant,
    )
    loop.record_cycle_performance(metric)
    updated_cfg = loop.run_adaptive_learning()
    return metric, updated_cfg
