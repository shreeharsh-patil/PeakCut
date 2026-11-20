"""Phase 3 Video Generation & Processing Pipeline for ClipMind.

Implements the multi-priority vertical video production engine:
- Priority 1 (YouTube Source): Download viral long-form videos using yt-dlp, extract peak-engagement
  45-58 second segment (with a 0-3s high-retention hook).
- Priority 2 (Text-to-Video / AI Narration Fallback): Generate a viral script (3-act formula),
  synthesize AI voiceover (ElevenLabs -> Edge-TTS fallback), and composite with relevant B-roll.
- Priority 3 (Slideshow Fallback): Dynamic image slideshow with Ken Burns zoom/pan and AI voiceover.
- Video Processing via FFmpeg:
  - Vertical reframing / cropping to 9:16 (1080x1920).
  - Transcribing and burning center-frame / lower-third bold subtitles with custom ASS/SRT styling.
  - Audio loudness normalization to -14 LUFS using loudnorm (I=-14:TP=-1.5:LRA=11) and ducked
    background music at -18dB.
  - Strict 45-58 second duration enforcement.
- Dry-run & Simulation Mode: Generates synthetic vertical Short assets for automated CI/smoke testing.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core integrations
try:
    from clipmind.core.config import Config, get_config
except ImportError:
    # Graceful standalone fallback
    class Config:  # type: ignore
        def __init__(self):
            self.dry_run = False
            self.simulate_apis_if_missing = True
            self.niche = "Tech & AI"
            self.paths = None
            self.video = None

    def get_config():  # type: ignore
        return Config()

try:
    from clipmind.core.key_rotator import get_active_key, record_usage, mark_rate_limited
except ImportError:
    def get_active_key(service: str) -> Optional[str]:
        return os.environ.get(f"{service.upper()}_API_KEY")

    def record_usage(service: str, key: str, units: int = 1) -> None:
        pass

    def mark_rate_limited(service: str, key: str, duration_seconds: float = 3600.0, reason: str = "") -> None:
        pass

logger = logging.getLogger("clipmind.generation.pipeline")


# ---------------------------------------------------------------------------
# Serverless Environment & FFmpeg Binary Resolver
# ---------------------------------------------------------------------------

_CACHED_FFMPEG_PATH: Optional[str] = None
_CACHED_FFPROBE_PATH: Optional[str] = None


def is_serverless() -> bool:
    """Detect if executing inside Vercel or AWS Lambda serverless function."""
    return bool(
        os.environ.get("VERCEL") == "1"
        or os.environ.get("VERCEL_ENV")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or os.environ.get("LAMBDA_TASK_ROOT")
    )


def download_static_ffmpeg_linux(dest_path: Path) -> bool:
    """Download a standalone static Linux ffmpeg binary to /tmp/ffmpeg if absent."""
    if dest_path.exists() and os.access(dest_path, os.X_OK):
        return True

    # Lightweight static builds suitable for AWS Lambda / Amazon Linux / Vercel
    urls = [
        "https://github.com/eugeneware/ffmpeg-static/releases/download/b4.4/ffmpeg-linux-x64",
        "https://raw.githubusercontent.com/descriptinc/ffmpeg-binaries/master/ffmpeg-linux-64",
    ]
    try:
        import urllib.request
        for url in urls:
            try:
                logger.info(f"Downloading static FFmpeg for serverless Linux from {url}...")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Serverless)"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    if resp.status == 200:
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(dest_path, "wb") as f:
                            shutil.copyfileobj(resp, f)
                        dest_path.chmod(0o755)
                        logger.info(f"Static FFmpeg installed successfully at: {dest_path}")
                        return True
            except Exception as e:
                logger.debug(f"Failed downloading ffmpeg from {url}: {e}")
                continue
    except Exception as ex:
        logger.warning(f"Static FFmpeg download failed: {ex}")
    return False


def find_ffmpeg() -> str:
    """Locate the ffmpeg executable on the system across PATH and known fallback locations."""
    global _CACHED_FFMPEG_PATH
    if _CACHED_FFMPEG_PATH and os.path.exists(_CACHED_FFMPEG_PATH):
        return _CACHED_FFMPEG_PATH

    # 1. Environment variable override
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.exists(env_path):
        _CACHED_FFMPEG_PATH = env_path
        _add_to_env_path(os.path.dirname(env_path))
        return env_path

    # 2. System PATH
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg and os.path.exists(which_ffmpeg):
        _CACHED_FFMPEG_PATH = which_ffmpeg
        return which_ffmpeg

    # 3. Serverless /tmp/ffmpeg location
    tmp_ffmpeg = Path(tempfile.gettempdir()) / "ffmpeg"
    if tmp_ffmpeg.exists():
        try:
            tmp_ffmpeg.chmod(0o755)
        except Exception:
            pass
        _CACHED_FFMPEG_PATH = str(tmp_ffmpeg)
        _add_to_env_path(str(tmp_ffmpeg.parent))
        return str(tmp_ffmpeg)

    # 4. Known fallback locations on Windows / Linux
    search_patterns = [
        os.path.expanduser(r"~/.vscode/extensions/kilocode.*/bin/ffmpeg.exe"),
        r"C:\Users\*\.vscode\extensions\kilocode.*\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        os.path.expanduser(r"~/bin/ffmpeg"),
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/var/task/bin/ffmpeg",
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            candidate = matches[0]
            if os.path.isfile(candidate):
                _CACHED_FFMPEG_PATH = candidate
                _add_to_env_path(os.path.dirname(candidate))
                logger.info(f"Discovered FFmpeg at: {candidate}")
                return candidate

    # 5. In Linux serverless environment, attempt static binary download to /tmp
    if (sys.platform.startswith("linux") or is_serverless()) and not shutil.which("ffmpeg"):
        if download_static_ffmpeg_linux(tmp_ffmpeg):
            _CACHED_FFMPEG_PATH = str(tmp_ffmpeg)
            _add_to_env_path(str(tmp_ffmpeg.parent))
            return str(tmp_ffmpeg)

    return "ffmpeg"


def find_ffprobe() -> str:
    """Locate the ffprobe executable across env, PATH, or sibling of ffmpeg."""
    global _CACHED_FFPROBE_PATH
    if _CACHED_FFPROBE_PATH and os.path.exists(_CACHED_FFPROBE_PATH):
        return _CACHED_FFPROBE_PATH

    env_path = os.environ.get("FFPROBE_PATH")
    if env_path and os.path.exists(env_path):
        _CACHED_FFPROBE_PATH = env_path
        return env_path

    which_ffprobe = shutil.which("ffprobe")
    if which_ffprobe and os.path.exists(which_ffprobe):
        _CACHED_FFPROBE_PATH = which_ffprobe
        return which_ffprobe

    ffmpeg_exe = find_ffmpeg()
    if ffmpeg_exe and os.path.exists(ffmpeg_exe):
        exe_name = "ffprobe.exe" if ffmpeg_exe.lower().endswith(".exe") else "ffprobe"
        sibling = os.path.join(os.path.dirname(ffmpeg_exe), exe_name)
        if os.path.isfile(sibling):
            _CACHED_FFPROBE_PATH = sibling
            return sibling

    return "ffprobe"


def _add_to_env_path(directory: str) -> None:
    """Prepend directory to system PATH so subprocesses can resolve it."""
    if directory and directory not in os.environ.get("PATH", ""):
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")


# Initialize resolver
find_ffmpeg()
find_ffprobe()


# ---------------------------------------------------------------------------
# Enums and Data Models
# ---------------------------------------------------------------------------

class GenerationStrategy(str, Enum):
    YOUTUBE_EXTRACT = "youtube_extract"
    AI_SCRIPT_BROLL = "ai_script_broll"
    SLIDESHOW = "slideshow"
    SIMULATION = "simulation"


@dataclass
class VideoCandidate:
    """Information about a potential source video for extraction."""
    url: str
    title: str
    duration: float
    view_count: int = 0
    uploader: str = ""
    vps_score: float = 0.0
    description: str = ""
    thumbnail_url: str = ""


@dataclass
class ClipResult:
    """Final output artifact of the video generation pipeline."""
    video_path: Path
    duration: float
    aspect_ratio: str = "9:16"
    resolution: Tuple[int, int] = (1080, 1920)
    strategy_used: str = GenerationStrategy.YOUTUBE_EXTRACT.value
    hook_text: str = ""
    subtitles_path: Optional[Path] = None
    transcript: Optional[Dict[str, Any]] = None
    audio_lufs: float = -14.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": str(self.video_path),
            "duration": round(self.duration, 2),
            "aspect_ratio": self.aspect_ratio,
            "resolution": list(self.resolution),
            "strategy_used": self.strategy_used,
            "hook_text": self.hook_text,
            "subtitles_path": str(self.subtitles_path) if self.subtitles_path else None,
            "audio_lufs": self.audio_lufs,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Clip Pipeline Engine
# ---------------------------------------------------------------------------

class ClipPipeline:
    """Orchestrates Phase 3 Video Pipeline with fallback tiers and strict compliance."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.min_duration = 45.0
        self.max_duration = 58.0
        self.target_lufs = -14.0
        self.loudnorm_filter = "I=-14:TP=-1.5:LRA=11"
        self.bg_music_duck_db = -18.0
        self.ffmpeg_bin = find_ffmpeg()
        self.ffprobe_bin = find_ffprobe()

        # Output paths (Serverless / read-only filesystem safe)
        if is_serverless():
            tmp_root = Path(tempfile.gettempdir()) / "clipmind"
            self.output_dir = tmp_root / "output"
            self.temp_dir = tmp_root / "temp"
        elif hasattr(self.config, "paths") and self.config.paths:
            self.output_dir = Path(self.config.paths.output_dir)
            self.temp_dir = Path(self.config.paths.temp_dir)
        else:
            self.output_dir = Path("output")
            self.temp_dir = Path("output/temp")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            # Fallback if host filesystem is strictly read-only (e.g. AWS Lambda / Vercel)
            tmp_root = Path(tempfile.gettempdir()) / "clipmind"
            self.output_dir = tmp_root / "output"
            self.temp_dir = tmp_root / "temp"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Probe Utilities
    # -----------------------------------------------------------------------

    def get_media_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Probe media file duration, resolution, codecs, and streams via FFprobe/FFmpeg."""
        from ffmpeg_utils import probe_media

        return probe_media(
            file_path,
            ffprobe_bin=self.ffprobe_bin,
            ffmpeg_bin=self.ffmpeg_bin,
            timeout=30,
        )

    def validate_clip_output(
        self,
        file_path: Union[str, Path],
        *,
        expected_duration: Optional[float] = None,
        duration_tolerance: float = 2.0,
        expected_resolution: Optional[Tuple[int, int]] = (1080, 1920),
        require_audio: bool = False,
        decode_check: bool = True,
    ) -> Dict[str, Any]:
        """Verify that a generated clip is playable and meets resolution/duration constraints.

        Raises RuntimeError if the file is missing, empty, corrupt, or fails constraints.
        """
        from ffmpeg_utils import validate_media_output

        ok, info, reason = validate_media_output(
            file_path,
            expected_duration=expected_duration,
            duration_tolerance=duration_tolerance,
            min_duration=0.5,
            require_video=True,
            require_audio=require_audio,
            expected_resolution=expected_resolution,
            decode_check=decode_check,
            ffmpeg_bin=self.ffmpeg_bin,
            ffprobe_bin=self.ffprobe_bin,
        )
        if not ok:
            raise RuntimeError(f"Generated video validation failed for {file_path}: {reason}")
        return info

    # -----------------------------------------------------------------------
    # Subtitle Generator
    # -----------------------------------------------------------------------

    def create_styled_ass_subtitles(
        self,
        transcript_data: Union[List[Dict[str, Any]], Dict[str, Any], str],
        output_ass_path: Union[str, Path],
        duration: float,
        font_name: str = "Anton",
        font_size: int = 54,
        primary_color: str = "&H0000FFFF",  # Bright Yellow in ASS (&HAABBGGRR)
        outline_color: str = "&H00000000",  # Opaque Black
        outline_width: int = 4,
    ) -> Path:
        """Generate high-contrast ASS subtitle script for bold center-frame burning."""
        output_ass_path = Path(output_ass_path)
        output_ass_path.parent.mkdir(parents=True, exist_ok=True)
        duration = max(0.1, float(duration or 0.1))

        def _clean_ass_text(raw_text: str) -> str:
            text = unicodedata.normalize("NFC", str(raw_text or ""))
            text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
            text = text.replace("\\", "/").replace("{", "(").replace("}", ")")
            text = re.sub(r"\s+", " ", text).strip()
            return text.upper()

        # Parse segments
        segments: List[Dict[str, Any]] = []
        if isinstance(transcript_data, str):
            words = transcript_data.split()
            words_per_block = 4
            num_blocks = math.ceil(len(words) / words_per_block) if words else 0
            block_dur = max(1.5, min(3.5, duration / max(1, num_blocks)))

            for i in range(0, len(words), words_per_block):
                chunk = " ".join(words[i:i + words_per_block])
                idx = i // words_per_block
                start = round(idx * block_dur, 2)
                end = round(min(duration, (idx + 1) * block_dur), 2)
                if start < duration and end > start:
                    segments.append({"start": start, "end": end, "text": _clean_ass_text(chunk)})
        elif isinstance(transcript_data, dict):
            raw_segs = transcript_data.get("segments", [])
            for s in raw_segs:
                if isinstance(s, dict):
                    segments.append({
                        "start": float(s.get("start", 0)),
                        "end": float(s.get("end", 0)),
                        "text": _clean_ass_text(s.get("text", "")),
                    })
        elif isinstance(transcript_data, list):
            for s in transcript_data:
                if isinstance(s, dict):
                    segments.append({
                        "start": float(s.get("start", 0)),
                        "end": float(s.get("end", 0)),
                        "text": _clean_ass_text(s.get("text", "")),
                    })

        # Normalize, sort, clamp to [0, duration], and prevent overlaps
        segments.sort(key=lambda s: (s["start"], s["end"]))
        normalized_segments: List[Dict[str, Any]] = []
        prev_end = 0.0
        for seg in segments:
            text = seg.get("text", "").strip()
            if not text:
                continue
            s_t = max(prev_end, max(0.0, float(seg["start"])))
            e_t = min(duration, float(seg["end"]))
            if e_t <= s_t:
                e_t = min(duration, s_t + 0.25)
            if e_t <= s_t:
                continue
            normalized_segments.append({"start": s_t, "end": e_t, "text": text})
            prev_end = e_t

        def to_ass_time(secs: float) -> str:
            secs = max(0.0, float(secs))
            total_cs = int(round(secs * 100))
            cs = total_cs % 100
            total_s = total_cs // 100
            h = total_s // 3600
            m = (total_s % 3600) // 60
            s = total_s % 60
            return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

        # Alignment 2 is Bottom-Center. MarginV 280 keeps captions clear of Shorts/Reels UI.
        safe_font = re.sub(r"[^A-Za-z0-9 _-]", "", str(font_name or "")).strip() or "Anton"
        header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: BoldYellow,{safe_font},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,1,0,0,0,100,100,0,0,1,{outline_width},2,2,60,60,280,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        dialogue_lines = []
        for seg in normalized_segments:
            st = to_ass_time(seg["start"])
            et = to_ass_time(seg["end"])
            dialogue_lines.append(f"Dialogue: 0,{st},{et},BoldYellow,,0,0,0,,{seg['text']}")

        content = header + "\n".join(dialogue_lines) + "\n"
        with open(output_ass_path, "w", encoding="utf-8-sig") as f:
            f.write(content)

        return output_ass_path

    # -----------------------------------------------------------------------
    # Priority 1: YouTube Viral Source Extraction
    # -----------------------------------------------------------------------

    def extract_from_youtube(
        self,
        topic: str,
        youtube_url: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> ClipResult:
        """Priority 1: Download viral video via yt-dlp and extract peak 45-58s segment."""
        logger.info(f"[Priority 1] Attempting YouTube extraction for topic: '{topic}'")
        cycle_prefix = f"cycle_{cycle_id}_" if cycle_id else f"clip_{int(time.time())}_"

        # 1. Search or use provided URL
        target_url = youtube_url
        if not target_url:
            target_url = self._search_youtube_viral_candidate(topic)

        if not target_url:
            raise ValueError(f"No suitable viral YouTube video candidate found for '{topic}'")

        # 2. Download source video with yt-dlp
        download_path = self.temp_dir / f"{cycle_prefix}source.mp4"
        info = self._download_youtube_video(target_url, download_path)
        
        source_duration = float(info.get("duration", 0.0))
        if source_duration <= 0.0:
            probe = self.get_media_info(download_path)
            source_duration = probe["duration"]

        if source_duration < self.min_duration:
            raise ValueError(f"Source video too short ({source_duration:.1f}s < {self.min_duration}s)")

        # 3. Peak engagement window detection
        start_sec, end_sec, hook_text = self._detect_peak_engagement_window(
            info=info,
            video_path=download_path,
            total_duration=source_duration,
            topic=topic,
        )

        target_duration = max(self.min_duration, min(self.max_duration, end_sec - start_sec))

        # 4. Transcribe or fetch captions for the segment
        transcript_data = self._get_transcript_for_window(
            info=info,
            video_path=download_path,
            start_sec=start_sec,
            duration=target_duration,
            fallback_text=hook_text or topic,
        )

        # 5. Burn subtitles & apply 9:16 reframe + loudnorm
        output_clip_path = self.output_dir / f"{cycle_prefix}final.mp4"
        ass_sub_path = self.temp_dir / f"{cycle_prefix}subs.ass"
        self.create_styled_ass_subtitles(transcript_data, ass_sub_path, duration=target_duration)

        self._render_ffmpeg_vertical_cut(
            input_video=download_path,
            output_video=output_clip_path,
            start_sec=start_sec,
            duration=target_duration,
            ass_subtitles_path=ass_sub_path,
        )

        return ClipResult(
            video_path=output_clip_path,
            duration=target_duration,
            aspect_ratio="9:16",
            resolution=(1080, 1920),
            strategy_used=GenerationStrategy.YOUTUBE_EXTRACT.value,
            hook_text=hook_text,
            subtitles_path=ass_sub_path,
            transcript=transcript_data if isinstance(transcript_data, dict) else {"text": str(transcript_data)},
            audio_lufs=self.target_lufs,
            metadata={
                "source_url": target_url,
                "source_title": info.get("title", ""),
                "extracted_range": (start_sec, round(start_sec + target_duration, 2)),
                "view_count": info.get("view_count", 0),
            },
        )

    def _search_youtube_viral_candidate(self, topic: str) -> Optional[str]:
        """Use yt-dlp search to discover high-view candidate video."""
        try:
            import yt_dlp
        except ImportError:
            logger.warning("yt-dlp not installed, cannot search YouTube")
            return None

        query = f"ytsearch5:{topic} viral"
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                res = ydl.extract_info(query, download=False)
                entries = res.get("entries", []) if res else []
                # Filter for videos between 60s and 20 minutes with high views
                candidates = []
                for e in entries:
                    dur = e.get("duration") or 0
                    if 60 <= dur <= 1800:
                        candidates.append(e)

                if candidates:
                    # Pick highest viewed or first
                    candidates.sort(key=lambda x: x.get("view_count") or 0, reverse=True)
                    best = candidates[0]
                    vid_url = best.get("url") or f"https://www.youtube.com/watch?v={best.get('id')}"
                    logger.info(f"Selected YouTube candidate: {best.get('title')} ({vid_url})")
                    return vid_url
        except Exception as e:
            logger.warning(f"yt-dlp candidate search failed: {e}")

        return None

    def _download_youtube_video(self, url: str, target_path: Path, max_retries: int = 3) -> Dict[str, Any]:
        """Download video using yt-dlp into target_path with bounded retries and validation."""
        try:
            import yt_dlp
        except ImportError:
            raise RuntimeError("yt-dlp is not installed; cannot download YouTube video")

        try:
            from security_utils import assert_public_url, normalize_media_url
            url = normalize_media_url(url)
            assert_public_url(url)
        except ImportError:
            url = str(url or "").strip()
            if not url:
                raise ValueError("Empty YouTube URL")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        non_retryable_tokens = (
            "private video", "video unavailable", "this video has been removed",
            "sign in to confirm your age", "age-restricted", "not available in your country",
            "geo-restricted", "members-only", "copyright", "is not a valid url",
            "unsupported url", "premieres in", "this live event will begin",
        )

        ydl_opts = {
            "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(target_path.with_suffix(".%(ext)s")),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "overwrites": False,
            "continuedl": True,
            "retries": 3,
            "fragment_retries": 3,
            "socket_timeout": 30,
            "nocheckcertificate": True,
            "cachedir": False,
        }

        last_err: Optional[Exception] = None
        for attempt in range(1, max(1, int(max_retries)) + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)

                actual_file = target_path.with_suffix(".mp4")
                if not actual_file.exists():
                    for ext in (".mkv", ".webm", ".mov"):
                        alt = target_path.with_suffix(ext)
                        if alt.exists() and alt.stat().st_size > 0:
                            remux = subprocess.run(
                                [
                                    self.ffmpeg_bin, "-y", "-loglevel", "error",
                                    "-i", str(alt), "-c", "copy", "-movflags", "+faststart",
                                    str(actual_file),
                                ],
                                capture_output=True, text=True, errors="replace", timeout=300,
                            )
                            if remux.returncode == 0 and actual_file.exists():
                                alt.unlink(missing_ok=True)
                            else:
                                alt.replace(actual_file)
                            break

                if not actual_file.exists() or actual_file.stat().st_size == 0:
                    raise FileNotFoundError(f"yt-dlp downloaded file missing or empty at {actual_file}")

                media_info = self.get_media_info(actual_file)
                if not media_info.get("valid") or not media_info.get("has_video"):
                    actual_file.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Downloaded file failed validation ({media_info.get('error') or 'no video stream'})"
                    )
                return info or {}
            except Exception as e:
                last_err = e
                msg_l = str(e).lower()
                if any(tok in msg_l for tok in non_retryable_tokens) or attempt >= max_retries:
                    break
                backoff = min(10.0, 2.0 ** attempt)
                logger.warning(f"yt-dlp attempt {attempt}/{max_retries} failed ({e}); retrying in {backoff:.1f}s")
                time.sleep(backoff)

        # Clean up any partial artifacts on fatal failure
        for suffix in (".mp4", ".part", ".ytdl", ".mkv", ".webm"):
            p = target_path.with_suffix(suffix)
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass
        raise RuntimeError(f"YouTube download failed for {url}: {last_err}") from last_err

    def _detect_peak_engagement_window(
        self,
        info: Dict[str, Any],
        video_path: Path,
        total_duration: float,
        topic: str,
    ) -> Tuple[float, float, str]:
        """Calculate the 45-58s window containing the strongest retention hook (0-3s)."""
        target_span = 52.0  # Ideal target duration
        if total_duration <= target_span:
            return 0.0, total_duration, f"Shocking Truth About {topic}"

        # 1. Check if heatmap / chapters available in info
        chapters = info.get("chapters") or []
        for ch in chapters:
            title = str(ch.get("title", "")).lower()
            start = float(ch.get("start_time", 0.0))
            end = float(ch.get("end_time", 0.0))
            if any(k in title for k in ["hook", "secret", "best", "climax", "truth", "reveal"]):
                if (end - start) >= self.min_duration:
                    return start, min(total_duration, start + target_span), ch.get("title", "")

        # 2. Heuristic selection: Avoid first 5% (channel intro) and select around 15-25% mark
        safe_start = max(5.0, total_duration * 0.15)
        if safe_start + target_span > total_duration:
            safe_start = max(0.0, total_duration - target_span)

        hook_text = f"The Insane Truth About {topic} You Never Knew!"
        return round(safe_start, 2), round(safe_start + target_span, 2), hook_text

    def _get_transcript_for_window(
        self,
        info: Dict[str, Any],
        video_path: Path,
        start_sec: float,
        duration: float,
        fallback_text: str,
    ) -> Union[Dict[str, Any], str]:
        """Fetch transcript segments for the window via youtube_transcript_api or fallback."""
        video_id = info.get("id")
        if video_id:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                # Filter for segments overlapping our window
                window_end = start_sec + duration
                matched_segs = []
                for item in transcript_list:
                    item_start = item["start"]
                    item_dur = item.get("duration", 2.0)
                    item_end = item_start + item_dur
                    if item_end >= start_sec and item_start <= window_end:
                        # Normalize timestamp relative to 0.0
                        rel_start = max(0.0, item_start - start_sec)
                        rel_end = min(duration, item_end - start_sec)
                        matched_segs.append({
                            "start": round(rel_start, 2),
                            "end": round(rel_end, 2),
                            "text": item.get("text", "")
                        })
                if matched_segs:
                    return {"segments": matched_segs}
            except Exception as e:
                logger.debug(f"YouTube transcript extraction skipped/failed: {e}")

        # Fallback to high-retention structured text
        return fallback_text

    # -----------------------------------------------------------------------
    # Priority 2: Text-to-Video / AI Narration Script Fallback
    # -----------------------------------------------------------------------

    def generate_from_ai_script(
        self,
        topic: str,
        script_text: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> ClipResult:
        """Priority 2: Generate script + AI voiceover + B-roll composite."""
        logger.info(f"[Priority 2] Generating Text-to-Video AI clip for topic: '{topic}'")
        cycle_prefix = f"cycle_{cycle_id}_" if cycle_id else f"clip_{int(time.time())}_"

        # 1. Script Generation (3-act viral Shorts formula)
        if not script_text:
            script_text = self._generate_viral_script(topic)

        # 2. Voiceover Synthesis (ElevenLabs -> Edge-TTS fallback)
        voice_path = self.temp_dir / f"{cycle_prefix}narration.mp3"
        actual_voice_path, audio_duration = self._synthesize_voiceover(script_text, voice_path)

        # Ensure audio fits in 45-58 seconds
        target_duration = max(self.min_duration, min(self.max_duration, audio_duration))

        # 3. Visuals / B-roll sourcing or dynamic procedural generation
        broll_video = self._acquire_or_generate_broll(topic, duration=target_duration, cycle_prefix=cycle_prefix)

        # 4. Generate subtitles
        ass_sub_path = self.temp_dir / f"{cycle_prefix}subs.ass"
        self.create_styled_ass_subtitles(script_text, ass_sub_path, duration=target_duration)

        # 5. Composite Final Video with Audio Normalization (-14 LUFS) & Ducking
        output_clip_path = self.output_dir / f"{cycle_prefix}final.mp4"
        self._composite_video_with_audio(
            visual_source=broll_video,
            narration_audio=actual_voice_path,
            output_video=output_clip_path,
            duration=target_duration,
            ass_subtitles_path=ass_sub_path,
        )

        hook_first_line = script_text.split(".")[0].strip() if "." in script_text else script_text[:40]

        return ClipResult(
            video_path=output_clip_path,
            duration=target_duration,
            aspect_ratio="9:16",
            resolution=(1080, 1920),
            strategy_used=GenerationStrategy.AI_SCRIPT_BROLL.value,
            hook_text=hook_first_line,
            subtitles_path=ass_sub_path,
            transcript={"text": script_text},
            audio_lufs=self.target_lufs,
            metadata={
                "topic": topic,
                "script_word_count": len(script_text.split()),
                "voice_duration": audio_duration,
            },
        )

    def _generate_viral_script(self, topic: str) -> str:
        """Create a 48-54 second high-retention viral Shorts script."""
        # Check OpenAI key
        openai_key = get_active_key("openai")
        if openai_key and not openai_key.startswith("sim_"):
            try:
                import requests
                headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
                prompt = (
                    f"Write a viral YouTube Shorts script about '{topic}'. "
                    "Follow the 4-part formula strictly:\n"
                    "1. 0-3s: Shocking Curiosity Hook (Call out viewer, question or revelation).\n"
                    "2. 3-20s: Mind-blowing core fact or story beat.\n"
                    "3. 20-40s: Escalating twist or unexpected breakdown.\n"
                    "4. 40-52s: Shocking conclusion + Call to Action.\n"
                    "Total length MUST be 110 to 135 words so it reads aloud in 48-52 seconds. "
                    "Output ONLY the spoken script text, no stage directions, no speaker names."
                )
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 300,
                }
                resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=15)
                if resp.status_code == 200:
                    record_usage("openai", openai_key, units=350)
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    logger.info("OpenAI generated viral script successfully")
                    return content
                elif resp.status_code in (429, 403):
                    mark_rate_limited("openai", openai_key, reason=f"HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"OpenAI script generation error: {e}")

        # High-retention template fallback (engineered for ~50s spoken pacing)
        return (
            f"Stop scrolling! What nobody told you about {topic} will completely change how you see the world. "
            f"For years, experts maintained that {topic} was straightforward, but recent breakthroughs revealed an insane pattern. "
            f"When researchers analyzed the core mechanics, they discovered that over ninety percent of people get this completely backwards. "
            f"Here is the mind-blowing part: if you apply this one hidden principle today, the results multiply tenfold. "
            f"Do not sleep on this discovery before everyone else catches on. Tap subscribe and follow for the next breakthrough!"
        )

    def _synthesize_voiceover(self, text: str, output_path: Path) -> Tuple[Path, float]:
        """Synthesize speech via ElevenLabs, Edge-TTS, or procedural audio."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Try ElevenLabs
        eleven_key = get_active_key("elevenlabs")
        if eleven_key and not eleven_key.startswith("sim_"):
            try:
                import requests
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel (clear narration)
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                headers = {"xi-api-key": eleven_key, "Content-Type": "application/json"}
                payload = {
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    record_usage("elevenlabs", eleven_key, units=len(text))
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    probe = self.get_media_info(output_path)
                    dur = probe["duration"]
                    if dur > 0:
                        logger.info(f"ElevenLabs TTS succeeded ({dur:.1f}s)")
                        return output_path, dur
                elif resp.status_code in (401, 403, 429):
                    mark_rate_limited("elevenlabs", eleven_key, reason=f"HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"ElevenLabs TTS failed: {e}")

        # 2. Try Edge-TTS (Local, fast, crystal clear)
        try:
            import edge_tts
            voice = "en-US-ChristopherNeural"
            
            async def _run_edge():
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(output_path))

            asyncio.run(_run_edge())
            probe = self.get_media_info(output_path)
            dur = probe["duration"]
            if dur > 0:
                logger.info(f"Edge-TTS synthesized speech ({dur:.1f}s)")
                return output_path, dur
        except Exception as e:
            logger.warning(f"Edge-TTS synthesis failed: {e}")

        # 3. Fallback: Generate synthetic speech or silent audio track using FFmpeg
        dur = 50.0
        cmd = [
            self.ffmpeg_bin, "-y", "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={dur}",
            "-af", "volume=0.01",
            str(output_path)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return output_path, dur
        except Exception:
            return output_path, dur

    def _acquire_or_generate_broll(self, topic: str, duration: float, cycle_prefix: str) -> Path:
        """Fetch stock video via Pexels or dynamically render high-contrast 9:16 background."""
        target_video = self.temp_dir / f"{cycle_prefix}broll.mp4"

        # 1. Pexels API
        pexels_key = get_active_key("pexels")
        if pexels_key and not pexels_key.startswith("sim_"):
            try:
                import requests
                headers = {"Authorization": pexels_key}
                clean_query = topic.split()[0] if topic else "technology"
                url = f"https://api.pexels.com/videos/search?query={clean_query}&orientation=portrait&per_page=3"
                resp = requests.get(url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    record_usage("pexels", pexels_key, units=1)
                    data = resp.json()
                    videos = data.get("videos", [])
                    if videos:
                        video_files = videos[0].get("video_files", [])
                        # Look for 1080x1920 or portrait
                        best_link = None
                        for vf in video_files:
                            if vf.get("width") == 1080 and vf.get("height") == 1920:
                                best_link = vf.get("link")
                                break
                        if not best_link and video_files:
                            best_link = video_files[0].get("link")

                        if best_link:
                            dl_resp = requests.get(best_link, timeout=20)
                            if dl_resp.status_code == 200:
                                with open(target_video, "wb") as f:
                                    f.write(dl_resp.content)
                                logger.info("Downloaded portrait B-roll from Pexels")
                                return target_video
            except Exception as e:
                logger.warning(f"Pexels B-roll retrieval error: {e}")

        # 2. Dynamic Procedural Background Canvas via FFmpeg
        # Generates a sleek, high-tech animated vertical gradient with glowing motion
        filter_expr = (
            f"testsrc2=size=1080x1920:rate=30:duration={duration:.2f},"
            "hue=H='2*PI*t/10':s=1.2,"
            "boxblur=15:5"
        )
        cmd = [
            self.ffmpeg_bin, "-y",
            "-f", "lavfi",
            "-i", filter_expr,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-t", f"{duration:.2f}",
            str(target_video)
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return target_video
        except Exception as e:
            logger.warning(f"FFmpeg procedural b-roll failed: {e}, falling back to minimal clip")
            return self._create_simulated_video_file(target_video, duration=duration)

    # -----------------------------------------------------------------------
    # Priority 3: Dynamic Image Slideshow Fallback
    # -----------------------------------------------------------------------

    def generate_from_slideshow(
        self,
        topic: str,
        script_text: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> ClipResult:
        """Priority 3: Render image slideshow with typography + ElevenLabs/TTS voiceover."""
        logger.info(f"[Priority 3] Generating Image Slideshow for topic: '{topic}'")
        cycle_prefix = f"cycle_{cycle_id}_" if cycle_id else f"clip_{int(time.time())}_"

        if not script_text:
            script_text = self._generate_viral_script(topic)

        # Synthesize voiceover
        voice_path = self.temp_dir / f"{cycle_prefix}slideshow_voice.mp3"
        actual_voice_path, audio_duration = self._synthesize_voiceover(script_text, voice_path)
        target_duration = max(self.min_duration, min(self.max_duration, audio_duration))

        # Generate 5 thematic slide cards using Pillow
        slides = self._render_slide_deck(topic, script_text, count=5, cycle_prefix=cycle_prefix)

        # Assemble slides into MP4 video with pan/zoom (Ken Burns)
        slides_video = self.temp_dir / f"{cycle_prefix}slides_raw.mp4"
        self._assemble_slides_to_video(slides, slides_video, total_duration=target_duration)

        # Burn subtitles & normalize audio
        output_clip_path = self.output_dir / f"{cycle_prefix}final.mp4"
        ass_sub_path = self.temp_dir / f"{cycle_prefix}subs.ass"
        self.create_styled_ass_subtitles(script_text, ass_sub_path, duration=target_duration)

        self._composite_video_with_audio(
            visual_source=slides_video,
            narration_audio=actual_voice_path,
            output_video=output_clip_path,
            duration=target_duration,
            ass_subtitles_path=ass_sub_path,
        )

        return ClipResult(
            video_path=output_clip_path,
            duration=target_duration,
            aspect_ratio="9:16",
            resolution=(1080, 1920),
            strategy_used=GenerationStrategy.SLIDESHOW.value,
            hook_text=f"The Truth About {topic}",
            subtitles_path=ass_sub_path,
            transcript={"text": script_text},
            audio_lufs=self.target_lufs,
            metadata={"slides_count": len(slides), "topic": topic},
        )

    def _render_slide_deck(self, topic: str, script: str, count: int, cycle_prefix: str) -> List[Path]:
        """Render high-impact vertical graphics (1080x1920) for slideshow."""
        from PIL import Image, ImageDraw, ImageFont

        slide_paths: List[Path] = []
        font_path = Path("fonts/Anton-Regular.ttf")
        font = None
        if font_path.exists():
            try:
                font = ImageFont.truetype(str(font_path), 80)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        color_themes = [
            ((15, 15, 25), (255, 220, 0)),    # Dark Blue / Gold
            ((30, 10, 15), (255, 50, 80)),    # Dark Red / Hot Pink
            ((10, 25, 20), (0, 255, 170)),    # Dark Teal / Neon Cyan
            ((20, 15, 30), (200, 100, 255)),  # Deep Purple / Violet
            ((10, 10, 10), (255, 255, 255)),  # True Black / Crisp White
        ]

        sentences = [s.strip() for s in script.split(".") if len(s.strip()) > 10][:count]
        while len(sentences) < count:
            sentences.append(f"Mastering {topic} Today")

        for idx in range(count):
            bg_color, text_color = color_themes[idx % len(color_themes)]
            img = Image.new("RGB", (1080, 1920), bg_color)
            draw = ImageDraw.Draw(img)

            # Draw subtle geometric accent grid / frames
            draw.rectangle([40, 40, 1040, 1880], outline=text_color, width=4)
            draw.rectangle([60, 60, 1020, 1860], outline=(60, 60, 60), width=2)

            # Header badge
            draw.rounded_rectangle([340, 200, 740, 280], radius=20, fill=text_color)
            draw.text((540, 240), f"PART {idx + 1}", fill=bg_color, anchor="mm", font=font)

            # Main text
            text_lines = self._wrap_text_pillow(draw, sentences[idx].upper(), font, max_width=900)
            y_start = 800 - (len(text_lines) * 50)
            for l_idx, line in enumerate(text_lines):
                # Stroke shadow
                draw.text((542, y_start + l_idx * 110 + 2), line, fill=(0, 0, 0), anchor="mm", font=font)
                draw.text((540, y_start + l_idx * 110), line, fill=text_color, anchor="mm", font=font)

            slide_file = self.temp_dir / f"{cycle_prefix}slide_{idx + 1}.png"
            img.save(slide_file)
            slide_paths.append(slide_file)

        return slide_paths

    def _assemble_slides_to_video(self, slides: List[Path], out_video: Path, total_duration: float) -> None:
        """Combine slide images into a single video with zoom/pan."""
        slide_dur = total_duration / max(1, len(slides))
        concat_txt = self.temp_dir / "slides_concat.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for s in slides:
                # Escape path for ffmpeg concat demuxer
                safe_path = str(s).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")
                f.write(f"duration {slide_dur:.2f}\n")
            if slides:
                f.write(f"file '{str(slides[-1]).replace(chr(92), '/')}'\n")

        cmd = [
            self.ffmpeg_bin, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,fps=30",
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-t", f"{total_duration:.2f}",
            str(out_video)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace", timeout=600)
        if res.returncode != 0:
            if out_video.exists():
                out_video.unlink(missing_ok=True)
            raise RuntimeError(f"Slideshow video assembly failed (rc={res.returncode}): {(res.stderr or '')[-300:]}")
        self.validate_clip_output(
            out_video,
            expected_duration=total_duration,
            duration_tolerance=2.5,
            expected_resolution=(1080, 1920),
            require_audio=False,
            decode_check=False,
        )

    # -----------------------------------------------------------------------
    # Video & Audio FFmpeg Processing Pipeline
    # -----------------------------------------------------------------------

    def _render_ffmpeg_vertical_cut(
        self,
        input_video: Path,
        output_video: Path,
        start_sec: float,
        duration: float,
        ass_subtitles_path: Optional[Path] = None,
    ) -> None:
        """Cut, reframe to 9:16 (1080x1920), burn subtitles, and normalize audio to -14 LUFS."""
        output_video.parent.mkdir(parents=True, exist_ok=True)

        vf_filters = [
            "scale=1080:1920:force_original_aspect_ratio=increase",
            "crop=1080:1920",
            "setsar=1",
            "fps=30"
        ]

        if ass_subtitles_path and ass_subtitles_path.exists():
            clean_sub_path = str(ass_subtitles_path).replace("\\", "/").replace(":", "\\:")
            vf_filters.append(f"ass=filename='{clean_sub_path}'")

        vf_str = ",".join(vf_filters)
        af_str = f"loudnorm={self.loudnorm_filter}"

        cmd = [
            self.ffmpeg_bin, "-y",
            "-ss", f"{start_sec:.2f}",
            "-t", f"{duration:.2f}",
            "-i", str(input_video),
            "-vf", vf_str,
            "-af", af_str,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_video)
        ]

        logger.info(f"Executing FFmpeg vertical cut: start={start_sec:.2f}s, dur={duration:.2f}s")
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", timeout=600,
        )
        if res.returncode != 0:
            logger.warning(f"FFmpeg render returned code {res.returncode}: {(res.stderr or '')[-300:]}")
            self._fallback_render_direct(input_video, output_video, start_sec, duration)
        self.validate_clip_output(
            output_video,
            expected_duration=duration,
            duration_tolerance=2.5,
            expected_resolution=(1080, 1920),
            require_audio=False,
            decode_check=True,
        )

    def _composite_video_with_audio(
        self,
        visual_source: Path,
        narration_audio: Path,
        output_video: Path,
        duration: float,
        ass_subtitles_path: Optional[Path] = None,
        background_music_path: Optional[Path] = None,
    ) -> None:
        """Composite video visuals with speech audio, music ducking (-18dB), and subtitles."""
        output_video.parent.mkdir(parents=True, exist_ok=True)

        vf_filters = [
            "scale=1080:1920:force_original_aspect_ratio=increase",
            "crop=1080:1920",
            "setsar=1",
            "fps=30"
        ]

        if ass_subtitles_path and ass_subtitles_path.exists():
            clean_sub_path = str(ass_subtitles_path).replace("\\", "/").replace(":", "\\:")
            vf_filters.append(f"ass=filename='{clean_sub_path}'")

        vf_str = ",".join(vf_filters)

        if background_music_path and background_music_path.exists():
            filter_complex = (
                f"[0:v]{vf_str}[v];"
                f"[2:a]volume=0.125[bgm];"
                f"[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=2,"
                f"loudnorm={self.loudnorm_filter}[a]"
            )
            cmd = [
                self.ffmpeg_bin, "-y",
                "-stream_loop", "-1", "-i", str(visual_source),
                "-i", str(narration_audio),
                "-stream_loop", "-1", "-i", str(background_music_path),
                "-filter_complex", filter_complex,
                "-map", "[v]", "-map", "[a]",
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_video)
            ]
        else:
            cmd = [
                self.ffmpeg_bin, "-y",
                "-stream_loop", "-1", "-i", str(visual_source),
                "-i", str(narration_audio),
                "-vf", vf_str,
                "-af", f"loudnorm={self.loudnorm_filter}",
                "-map", "0:v:0", "-map", "1:a:0",
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                str(output_video)
            ]

        logger.info(f"Compositing video and audio: dur={duration:.2f}s")
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", timeout=600,
        )
        if res.returncode != 0:
            if output_video.exists():
                output_video.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg composite failed (rc={res.returncode}): {(res.stderr or '')[-300:]}")
        self.validate_clip_output(
            output_video,
            expected_duration=duration,
            duration_tolerance=2.5,
            expected_resolution=(1080, 1920),
            require_audio=True,
            decode_check=True,
        )

    def _fallback_render_direct(self, in_v: Path, out_v: Path, start: float, dur: float) -> None:
        """Simplified direct FFmpeg transcode if complex filter graphs encounter codec issues."""
        cmd = [
            self.ffmpeg_bin, "-y",
            "-ss", f"{start:.2f}",
            "-t", f"{dur:.2f}",
            "-i", str(in_v),
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_v)
        ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, errors="replace", timeout=600,
        )
        if res.returncode != 0:
            if out_v.exists():
                out_v.unlink(missing_ok=True)
            raise RuntimeError(f"Direct FFmpeg fallback render failed (rc={res.returncode}): {(res.stderr or '')[-300:]}")

    def _create_simulated_video_file(self, target_path: Path, duration: float = 50.0) -> Path:
        """Generate a lightweight, genuinely playable vertical MP4 for simulation / dry-run.

        Uses low-framerate still-image H.264 encoding (2 fps + ``-tune stillimage``)
        so a 50s 1080x1920 MP4 encodes in <1.5s even on 2-vCPU CI runners while
        remaining a fully valid, playable H.264/AAC container verified with FFprobe.
        """
        target_path = Path(target_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        duration = max(1.0, float(duration))

        # Primary: H.264/AAC with stillimage tuning (fast & universally playable).
        # Secondary: MPEG-4/AAC only if libx264 is absent in a custom FFmpeg build.
        codec_attempts = [
            ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-g", "10", "-pix_fmt", "yuv420p"],
            ["-c:v", "mpeg4", "-q:v", "8", "-pix_fmt", "yuv420p"],
        ]

        last_error = ""
        ffmpeg_missing = False
        for vcodec_args in codec_attempts:
            cmd = [
                self.ffmpeg_bin, "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c=navy:s=1080x1920:r=2:d={duration:.2f}",
                "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration:.2f}",
                *vcodec_args,
                "-c:a", "aac", "-b:a", "64k",
                "-movflags", "+faststart",
                "-shortest",
                "-t", f"{duration:.2f}",
                str(target_path),
            ]
            try:
                res = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, errors="replace", timeout=60,
                )
                if res.returncode != 0:
                    last_error = f"exit code {res.returncode}: {(res.stderr or '').strip()[-300:]}"
                    target_path.unlink(missing_ok=True)
                    continue
                self.validate_clip_output(
                    target_path,
                    expected_duration=duration,
                    duration_tolerance=1.5,
                    expected_resolution=(1080, 1920),
                    require_audio=True,
                    decode_check=True,
                )
                return target_path
            except FileNotFoundError as e:
                ffmpeg_missing = True
                last_error = str(e)
                target_path.unlink(missing_ok=True)
                break
            except subprocess.TimeoutExpired:
                last_error = f"FFmpeg timed out after 60s generating {duration:.2f}s synthetic video"
                target_path.unlink(missing_ok=True)
            except Exception as e:
                last_error = str(e)
                target_path.unlink(missing_ok=True)

        # When the ffmpeg CLI binary is not installed (e.g. GitHub Actions ubuntu-24.04),
        # generate a genuinely playable, valid 1080x1920 MP4 using OpenCV's bundled
        # FFmpeg/libavcodec VideoWriter — never write corrupt/unplayable stub bytes.
        if ffmpeg_missing:
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore

                fps = 1.0
                n_frames = max(1, int(round(duration * fps)))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(target_path), fourcc, fps, (1080, 1920))
                if not writer.isOpened():
                    raise RuntimeError("OpenCV VideoWriter could not open MP4 output")
                try:
                    frame = np.full((1920, 1080, 3), (128, 0, 0), dtype=np.uint8)
                    for _ in range(n_frames):
                        writer.write(frame)
                finally:
                    writer.release()

                self.validate_clip_output(
                    target_path,
                    expected_duration=duration,
                    duration_tolerance=1.5,
                    expected_resolution=(1080, 1920),
                    require_audio=False,
                    decode_check=True,
                )
                return target_path
            except Exception as cv_err:
                target_path.unlink(missing_ok=True)
                last_error = f"{last_error}; OpenCV MP4 fallback failed: {cv_err}"

        raise RuntimeError(
            f"Failed to generate valid simulated MP4 at {target_path} (duration={duration:.2f}s): {last_error}"
        )

    def _wrap_text_pillow(self, draw: Any, text: str, font: Any, max_width: int) -> List[str]:
        """Wrap words to fit inside specified pixel width."""
        words = text.split()
        lines: List[str] = []
        cur_line: List[str] = []

        for w in words:
            candidate = " ".join(cur_line + [w])
            length = draw.textlength(candidate, font=font)
            if length <= max_width or not cur_line:
                cur_line.append(w)
            else:
                lines.append(" ".join(cur_line))
                cur_line = [w]

        if cur_line:
            lines.append(" ".join(cur_line))
        return lines

    # -----------------------------------------------------------------------
    # Master Pipeline Entry Point
    # -----------------------------------------------------------------------

    def generate_clip(
        self,
        topic: str,
        source_type: Optional[str] = None,
        youtube_url: Optional[str] = None,
        cycle_id: Optional[str] = None,
    ) -> ClipResult:
        """Execute the multi-priority video generation loop.

        Attempts:
        1. Priority 1 (YouTube download & 45-58s viral cut)
        2. Priority 2 (Text-to-Video AI narration & B-roll)
        3. Priority 3 (Image slideshow & AI voiceover)
        4. Simulation / Dry-run fallback
        """
        logger.info(f"=== Starting ClipMind Video Pipeline for: '{topic}' ===")

        # Check explicit dry-run mode or simulation request
        if getattr(self.config, "dry_run", False) or source_type in (GenerationStrategy.SIMULATION.value, "simulation"):
            logger.info("Operating in SIMULATION / DRY_RUN mode: Generating simulated vertical Short")
            out_path = self.output_dir / f"cycle_{cycle_id or int(time.time())}_simulated.mp4"
            self._create_simulated_video_file(out_path, duration=50.0)
            media_info = self.get_media_info(out_path)
            return ClipResult(
                video_path=out_path,
                duration=round(media_info["duration"], 2),
                aspect_ratio="9:16",
                resolution=(media_info["width"], media_info["height"]),
                strategy_used=GenerationStrategy.SIMULATION.value,
                hook_text=f"The Shocking Truth About {topic}",
                audio_lufs=-14.0,
                metadata={"dry_run": True, "topic": topic},
            )

        tried: set[str] = set()

        # Explicit strategy request (never re-attempt the same tier in the fallback ladder)
        if source_type == GenerationStrategy.YOUTUBE_EXTRACT.value or youtube_url:
            tried.add(GenerationStrategy.YOUTUBE_EXTRACT.value)
            try:
                return self.extract_from_youtube(topic=topic, youtube_url=youtube_url, cycle_id=cycle_id)
            except Exception as e:
                logger.warning(f"Direct YouTube extraction failed ({e}), falling back to AI script")

        elif source_type == GenerationStrategy.AI_SCRIPT_BROLL.value:
            tried.add(GenerationStrategy.AI_SCRIPT_BROLL.value)
            try:
                return self.generate_from_ai_script(topic=topic, cycle_id=cycle_id)
            except Exception as e:
                logger.warning(f"AI script pipeline failed ({e}), falling back to slideshow")

        elif source_type == GenerationStrategy.SLIDESHOW.value:
            tried.add(GenerationStrategy.SLIDESHOW.value)
            try:
                return self.generate_from_slideshow(topic=topic, cycle_id=cycle_id)
            except Exception as e:
                logger.warning(f"Slideshow generation failed ({e}), using simulation")

        # Fallback Ladder: Tier 1 -> Tier 2 -> Tier 3 -> Simulation
        if is_serverless() and not youtube_url and not source_type:
            logger.info("Serverless environment detected: optimizing pipeline for execution time limits")
            for strat, fn in (
                (GenerationStrategy.AI_SCRIPT_BROLL.value, self.generate_from_ai_script),
                (GenerationStrategy.SLIDESHOW.value, self.generate_from_slideshow),
            ):
                if strat not in tried:
                    tried.add(strat)
                    try:
                        return fn(topic=topic, cycle_id=cycle_id)
                    except Exception as e_serv:
                        logger.warning(f"Serverless {strat} failed: {e_serv}")

        # Tier 1
        if GenerationStrategy.YOUTUBE_EXTRACT.value not in tried:
            tried.add(GenerationStrategy.YOUTUBE_EXTRACT.value)
            try:
                return self.extract_from_youtube(topic=topic, youtube_url=youtube_url, cycle_id=cycle_id)
            except Exception as e1:
                logger.warning(f"Priority 1 (YouTube source) failed: {e1}")

        # Tier 2
        if GenerationStrategy.AI_SCRIPT_BROLL.value not in tried:
            tried.add(GenerationStrategy.AI_SCRIPT_BROLL.value)
            try:
                return self.generate_from_ai_script(topic=topic, cycle_id=cycle_id)
            except Exception as e2:
                logger.warning(f"Priority 2 (AI script / narration) failed: {e2}")

        # Tier 3
        if GenerationStrategy.SLIDESHOW.value not in tried:
            tried.add(GenerationStrategy.SLIDESHOW.value)
            try:
                return self.generate_from_slideshow(topic=topic, cycle_id=cycle_id)
            except Exception as e3:
                logger.warning(f"Priority 3 (Slideshow fallback) failed: {e3}")

        # Tier 4: Robust graceful simulation fallback
        logger.info("All live rendering tiers exhausted: generating compliant simulation clip")
        out_path = self.output_dir / f"cycle_{cycle_id or int(time.time())}_fallback.mp4"
        self._create_simulated_video_file(out_path, duration=50.0)
        media_info = self.get_media_info(out_path)
        return ClipResult(
            video_path=out_path,
            duration=round(media_info["duration"], 2),
            aspect_ratio="9:16",
            resolution=(media_info["width"], media_info["height"]),
            strategy_used=GenerationStrategy.SIMULATION.value,
            hook_text=f"The Hidden Secret Behind {topic}",
            audio_lufs=-14.0,
            metadata={"fallback_simulation": True, "topic": topic},
        )


# Module-level convenience function
def generate_clip(
    topic: str,
    source_type: Optional[str] = None,
    youtube_url: Optional[str] = None,
    cycle_id: Optional[str] = None,
    config: Optional[Config] = None,
) -> ClipResult:
    """Convenience functional interface for video clip generation."""
    pipeline = ClipPipeline(config=config)
    return pipeline.generate_clip(
        topic=topic,
        source_type=source_type,
        youtube_url=youtube_url,
        cycle_id=cycle_id,
    )
