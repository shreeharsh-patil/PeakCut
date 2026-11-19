"""ClipMind Configuration Management System.

Loads settings from .env file, environment variables, or local JSON config overrides.
Provides defaults for niche, audience timezone, upload frequency, directories,
and audio/video rendering parameters.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

# Auto-load dotenv if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default Project Base Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def is_serverless() -> bool:
    """Detect if running in a serverless environment (Vercel, AWS Lambda, Google Cloud Functions)."""
    return bool(
        os.getenv("VERCEL") == "1"
        or os.getenv("VERCEL_ENV")
        or os.getenv("AWS_LAMBDA_FUNCTION_NAME")
        or os.getenv("SERVERLESS")
        or os.getenv("LAMBDA_TASK_ROOT")
    )


def get_base_writable_dir() -> Path:
    """Return the base writable directory.
    
    On serverless platforms (Vercel/Lambda), the code deployment directory is read-only.
    All writable state, outputs, caches, and logs must reside in /tmp (tempfile.gettempdir()).
    """
    if is_serverless():
        base = Path(tempfile.gettempdir()) / "clipmind"
        base.mkdir(parents=True, exist_ok=True)
        return base
    return PROJECT_ROOT


class VideoSettings(BaseModel):
    """Audio and video encoding parameters for vertical Shorts."""
    width: int = 1080
    height: int = 1920
    fps: int = 30
    min_duration: float = 45.0
    max_duration: float = 58.0
    target_lufs: float = -14.0
    loudnorm_filter: str = "I=-14:TP=-1.5:LRA=11"
    bg_music_duck_db: float = -18.0
    subtitles_font: str = "Arial-Bold"
    subtitles_color: str = "#FFCC00"  # High-visibility yellow
    subtitles_stroke_color: str = "#000000"
    subtitles_stroke_width: int = 4
    subtitles_font_size: int = 54


class PathSettings(BaseModel):
    """Directory paths for persistent storage, outputs, and temporary processing."""
    project_root: Path = PROJECT_ROOT
    is_serverless: bool = Field(default_factory=is_serverless)
    writable_root: Path = Field(default_factory=get_base_writable_dir)

    data_dir: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "data") if is_serverless() else (PROJECT_ROOT / "data")
    )
    output_dir: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "output") if is_serverless() else (PROJECT_ROOT / "output")
    )
    temp_dir: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "output" / "temp") if is_serverless() else (PROJECT_ROOT / "output" / "temp")
    )
    logs_dir: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "logs") if is_serverless() else (PROJECT_ROOT / "logs")
    )
    cycle_logs_dir: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "logs" / "cycles") if is_serverless() else (PROJECT_ROOT / "logs" / "cycles")
    )
    quota_state_file: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "data" / "key_quotas.json") if is_serverless() else (PROJECT_ROOT / "data" / "key_quotas.json")
    )
    archive_dir: Path = Field(
        default_factory=lambda: (get_base_writable_dir() / "output" / "archive") if is_serverless() else (PROJECT_ROOT / "output" / "archive")
    )

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for p in [self.data_dir, self.output_dir, self.temp_dir, self.logs_dir, self.cycle_logs_dir, self.archive_dir]:
            try:
                p.mkdir(parents=True, exist_ok=True)
            except OSError:
                # If read-only filesystem error occurs, fallback to /tmp
                fallback = Path(tempfile.gettempdir()) / "clipmind" / p.name
                fallback.mkdir(parents=True, exist_ok=True)


class SchedulingSettings(BaseModel):
    """Publishing and operating loop interval configurations."""
    frequency_hours: float = 12.0
    audience_timezone: str = "America/New_York"
    # Optimal publish windows (hour ranges in target timezone)
    # E.g., US Eastern: 12-15 (12 PM - 3 PM) and 19-21 (7 PM - 9 PM)
    optimal_publish_windows: List[tuple[int, int]] = Field(
        default_factory=lambda: [(12, 15), (19, 21)]
    )
    max_retries_per_cycle: int = 3
    retry_delay_seconds: float = 15.0


class Config(BaseModel):
    """Master configuration for the ClipMind Autonomous Engine."""
    niche: str = "Tech & AI"
    audience_timezone: str = "America/New_York"
    upload_frequency_hours: float = 12.0
    dry_run: bool = False
    simulate_apis_if_missing: bool = True  # Allows pipeline to run gracefully if keys not provided yet

    # Fallback topics in case trend mining or moderation rejects topics
    fallback_topics: List[str] = Field(
        default_factory=lambda: [
            "Top 5 Mind-Blowing AI Tools Released This Week",
            "The 60-Second Secret That Changed Tech History",
            "Why Quantum Computing Will Break Everything Tomorrow",
            "The Hidden Truth Behind Artificial Superintelligence",
            "3 Insane Science Discoveries Nobody Talked About"
        ]
    )

    # API Keys Configuration
    youtube_api_keys: List[str] = Field(default_factory=list)
    openai_api_keys: List[str] = Field(default_factory=list)
    serpapi_api_keys: List[str] = Field(default_factory=list)
    elevenlabs_api_keys: List[str] = Field(default_factory=list)
    pexels_api_keys: List[str] = Field(default_factory=list)

    # Sub-configurations
    video: VideoSettings = Field(default_factory=VideoSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    scheduling: SchedulingSettings = Field(default_factory=SchedulingSettings)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def load_from_env(cls) -> Config:
        """Instantiate Config dynamically from environment variables and defaults."""
        niche = os.getenv("CLIPMIND_NICHE", os.getenv("NICHE", "Tech & AI"))
        tz = os.getenv("CLIPMIND_TIMEZONE", os.getenv("AUDIENCE_TIMEZONE", "America/New_York"))
        
        freq_str = os.getenv("CLIPMIND_FREQUENCY_HOURS", os.getenv("UPLOAD_FREQUENCY_HOURS", "12"))
        try:
            freq = float(freq_str)
        except ValueError:
            freq = 12.0

        dry_run = os.getenv("CLIPMIND_DRY_RUN", "false").lower() in ("true", "1", "yes")

        def _parse_keys(env_names: List[str]) -> List[str]:
            keys: List[str] = []
            for name in env_names:
                val = os.getenv(name, "")
                if val:
                    for part in val.replace(";", ",").split(","):
                        cleaned = part.strip().strip("'\"")
                        if cleaned and cleaned not in keys:
                            keys.append(cleaned)
            return keys

        yt_keys = _parse_keys(["YOUTUBE_API_KEY", "YOUTUBE_API_KEYS", "GOOGLE_API_KEY"])
        openai_keys = _parse_keys(["OPENAI_API_KEY", "OPENAI_API_KEYS"])
        serp_keys = _parse_keys(["SERPAPI_API_KEY", "SERPAPI_API_KEYS", "SERP_API_KEY"])
        eleven_keys = _parse_keys(["ELEVENLABS_API_KEY", "ELEVENLABS_API_KEYS", "ELEVEN_API_KEY"])
        pexels_keys = _parse_keys(["PEXELS_API_KEY", "PEXELS_API_KEYS"])

        paths = PathSettings()
        paths.ensure_directories()

        scheduling = SchedulingSettings(
            frequency_hours=freq,
            audience_timezone=tz,
        )

        cfg = cls(
            niche=niche,
            audience_timezone=tz,
            upload_frequency_hours=freq,
            dry_run=dry_run,
            youtube_api_keys=yt_keys,
            openai_api_keys=openai_keys,
            serpapi_api_keys=serp_keys,
            elevenlabs_api_keys=eleven_keys,
            pexels_api_keys=pexels_keys,
            paths=paths,
            scheduling=scheduling,
        )

        # Check for persistent override file in data_dir
        override_file = paths.data_dir / "clipmind_config.json"
        if override_file.exists():
            try:
                with open(override_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "niche" in data:
                    cfg.niche = data["niche"]
                if "audience_timezone" in data:
                    cfg.audience_timezone = data["audience_timezone"]
                    cfg.scheduling.audience_timezone = data["audience_timezone"]
                if "upload_frequency_hours" in data:
                    cfg.upload_frequency_hours = float(data["upload_frequency_hours"])
                    cfg.scheduling.frequency_hours = float(data["upload_frequency_hours"])
                if "dry_run" in data:
                    cfg.dry_run = bool(data["dry_run"])
            except Exception:
                pass

        return cfg

    def save_persistent_overrides(self) -> None:
        """Persist user selections (niche, timezone, frequency) to disk."""
        self.paths.ensure_directories()
        override_file = self.paths.data_dir / "clipmind_config.json"
        data = {
            "niche": self.niche,
            "audience_timezone": self.audience_timezone,
            "upload_frequency_hours": self.upload_frequency_hours,
            "dry_run": self.dry_run,
        }
        try:
            with open(override_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            # Fallback to temp directory if data_dir is read-only
            fallback_file = Path(tempfile.gettempdir()) / "clipmind" / "data" / "clipmind_config.json"
            fallback_file.parent.mkdir(parents=True, exist_ok=True)
            with open(fallback_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


_CONFIG_INSTANCE: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """Retrieve the singleton Config instance."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None or reload:
        _CONFIG_INSTANCE = Config.load_from_env()
    return _CONFIG_INSTANCE
