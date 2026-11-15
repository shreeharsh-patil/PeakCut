"""Media ingestion, stream probing, and external source acquisition."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("anti_reused_pipeline.ingest")


@dataclass
class MediaStreamInfo:
    """Probed technical characteristics of a media file."""
    path: str
    duration: float
    width: int
    height: int
    fps: float
    has_video: bool
    has_audio: bool
    video_codec: str = ""
    audio_codec: str = ""
    audio_sample_rate: int = 44100
    audio_channels: int = 2
    bitrate: int = 0
    format_name: str = ""


def probe_media(file_path: str) -> MediaStreamInfo:
    """Probe a media file using ffprobe and return structured stream information.

    Args:
        file_path: Path to local audio/video file.

    Returns:
        MediaStreamInfo dataclass with complete audio/video properties.

    Raises:
        FileNotFoundError: If input file does not exist.
        RuntimeError: If ffprobe execution fails or yields invalid output.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Media file not found: {file_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        file_path
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True
        )
        data = json.loads(proc.stdout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffprobe failed for {file_path}: {exc.stderr}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse ffprobe JSON output for {file_path}") from exc

    streams = data.get("streams", [])
    format_data = data.get("format", {})

    has_video = False
    has_audio = False
    width = 0
    height = 0
    fps = 30.0
    v_codec = ""
    a_codec = ""
    sample_rate = 44100
    channels = 2

    duration_str = format_data.get("duration")
    duration = float(duration_str) if duration_str else 0.0

    for s in streams:
        codec_type = s.get("codec_type")
        if codec_type == "video" and not has_video:
            has_video = True
            width = int(s.get("width", 0))
            height = int(s.get("height", 0))
            v_codec = s.get("codec_name", "")

            # Frame rate extraction
            r_frame_rate = s.get("r_frame_rate", "30/1")
            try:
                num, den = r_frame_rate.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            except Exception:
                fps = 30.0

            if duration == 0.0 and s.get("duration"):
                try:
                    duration = float(s.get("duration"))
                except ValueError:
                    pass

        elif codec_type == "audio" and not has_audio:
            has_audio = True
            a_codec = s.get("codec_name", "")
            sample_rate = int(s.get("sample_rate", 44100))
            channels = int(s.get("channels", 2))
            if duration == 0.0 and s.get("duration"):
                try:
                    duration = float(s.get("duration"))
                except ValueError:
                    pass

    bitrate = int(format_data.get("bit_rate", 0)) if format_data.get("bit_rate") else 0
    format_name = format_data.get("format_name", "")

    return MediaStreamInfo(
        path=os.path.abspath(file_path),
        duration=max(0.1, duration),
        width=width,
        height=height,
        fps=fps,
        has_video=has_video,
        has_audio=has_audio,
        video_codec=v_codec,
        audio_codec=a_codec,
        audio_sample_rate=sample_rate,
        audio_channels=channels,
        bitrate=bitrate,
        format_name=format_name
    )


def download_youtube(url: str, output_dir: str, prefix: str = "yt_download") -> str:
    """Download a YouTube video using yt-dlp.

    Args:
        url: Full YouTube or Shorts URL.
        output_dir: Directory where the downloaded media will be stored.
        prefix: Filename prefix for the output file.

    Returns:
        Absolute path to the downloaded MP4 video.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, f"{prefix}_%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--output", out_template,
        url
    ]

    logger.info("Executing yt-dlp download: %s", url)
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        raise RuntimeError(f"yt-dlp download failed: {res.stderr}")

    # Locate downloaded file
    for file_name in os.listdir(output_dir):
        if file_name.startswith(prefix) and file_name.endswith(".mp4"):
            return os.path.abspath(os.path.join(output_dir, file_name))

    raise FileNotFoundError("yt-dlp completed but output MP4 file was not found in directory.")


def resolve_media_input(input_source: str, work_dir: str) -> MediaStreamInfo:
    """Resolve an input source which may be a local path or YouTube URL.

    Args:
        input_source: File path or URL.
        work_dir: Working directory for temporary downloads.

    Returns:
        MediaStreamInfo for the resolved media.
    """
    if input_source.startswith("http://") or input_source.startswith("https://"):
        download_path = download_youtube(input_source, work_dir)
        return probe_media(download_path)
    return probe_media(input_source)
