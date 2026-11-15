"""Dynamic Captions and Text Overlay Engine.

Generates high-retention kinetic typography subtitles and styled text cards
compatible with FFmpeg on Windows.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from anti_reused_pipeline.config import CaptionConfig

logger = logging.getLogger("anti_reused_pipeline.captions")


def escape_ffmpeg_path(path: str) -> str:
    """Escape a Windows filesystem path for FFmpeg filter argument parsing.

    Args:
        path: Raw filesystem path (e.g., C:\\path\\to\\sub.srt).

    Returns:
        Escaped path safe for FFmpeg filter interpolation (e.g., C\\:/path/to/sub.srt).
    """
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def generate_styled_ass_from_srt(
    srt_path: str,
    ass_path: str,
    config: CaptionConfig,
    video_width: int = 1080,
    video_height: int = 1920
) -> str:
    """Convert an SRT file into an Advanced SubStation Alpha (.ass) file with rich styling.

    Args:
        srt_path: Input SRT file path.
        ass_path: Output ASS file path.
        config: CaptionConfig for fonts, sizes, colors, margins.
        video_width: Target video canvas width.
        video_height: Target video canvas height.

    Returns:
        Path to generated ASS file.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.font_name},{config.font_size * 2},{config.primary_color},&H000000FF,{config.outline_color},&H80000000,-1,0,0,0,100,100,0,0,1,{config.outline_width * 2},{config.shadow_depth * 2},{config.alignment},40,40,{config.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    if os.path.isfile(srt_path):
        with open(srt_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Parse simple SRT blocks
        blocks = content.strip().split("\n\n")
        for b in blocks:
            lines = [l.strip() for l in b.split("\n") if l.strip()]
            if len(lines) >= 3 and "-->" in lines[1]:
                timing = lines[1]
                text = " ".join(lines[2:])
                try:
                    st, en = timing.split("-->")
                    st = st.strip().replace(",", ".")
                    en = en.strip().replace(",", ".")

                    # Convert 00:00:00.000 to ASS format H:MM:SS.cs
                    def to_ass_time(t_str: str) -> str:
                        pts = t_str.split(":")
                        hrs = int(pts[0])
                        mins = int(pts[1])
                        secs = float(pts[2])
                        return f"{hrs}:{mins:02d}:{secs:05.2f}"

                    ass_st = to_ass_time(st)
                    ass_en = to_ass_time(en)
                    events.append(f"Dialogue: 0,{ass_st},{ass_en},Default,,0,0,0,,{text}")
                except Exception:
                    continue

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events) + "\n")

    return ass_path


def build_subtitle_filter_string(
    sub_path: str,
    config: CaptionConfig
) -> str:
    """Build an FFmpeg subtitles filter string with escaped Windows path.

    Args:
        sub_path: Path to SRT or ASS file.
        config: CaptionConfig with styling options.

    Returns:
        Filter snippet (e.g. subtitles='C\\:/path.srt':force_style='FontSize=26').
    """
    escaped = escape_ffmpeg_path(sub_path)
    if sub_path.endswith(".ass"):
        return f"ass='{escaped}'"
    return (
        f"subtitles='{escaped}':force_style='"
        f"FontName={config.font_name},"
        f"FontSize={config.font_size},"
        f"PrimaryColour=&H00FFFFFF,"
        f"OutlineColour=&H00000000,"
        f"Outline={config.outline_width},"
        f"Shadow={config.shadow_depth},"
        f"Alignment={config.alignment},"
        f"MarginV={config.margin_v}'"
    )
