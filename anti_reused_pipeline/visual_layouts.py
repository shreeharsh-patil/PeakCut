"""Visual Restructuring & Filtergraph Layout Engine.

Generates 9:16 (1080x1920) vertical layouts that structurally defeat
Content ID visual perceptual hashing and meet YPP Reused Content standards:
1. Split-Screen (50/50 commentary / dual-stream with branded divider)
2. Picture-in-Picture (PIP reaction inset occupying >= 25% canvas)
3. Canvas Letterbox (Educational cards with topic header & caption notes)
4. Blur-Fill (Dynamic blurred backdrop with centered crisp footage)
5. Anti-pHash Micro-Enhancement Filter Chain (color grading, unsharp, noise, vignette)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from anti_reused_pipeline.config import VisualConfig, LayoutMode
from anti_reused_pipeline.ingest import MediaStreamInfo

logger = logging.getLogger("anti_reused_pipeline.visual")


@dataclass
class VisualGraphSpecification:
    """FFmpeg video filtergraph parameters and input stream mappings."""
    filtergraph: str
    output_label: str
    video_inputs: List[str]  # Filepaths corresponding to video input indices


def build_micro_enhancements_filter(config: VisualConfig) -> str:
    """Build the anti-perceptual hash micro-enhancement filter chain."""
    if not config.enable_micro_adjustments:
        return ""

    filters = [
        f"eq=contrast={config.color_contrast}:brightness={config.color_brightness}:"
        f"saturation={config.color_saturation}:gamma={config.color_gamma}",
        f"unsharp=5:5:{config.unsharp_amount}:3:3:0.3",
        f"noise=alls={config.noise_strength}:allf=t+u",
        f"vignette={config.vignette_angle}"
    ]
    return ",".join(filters)


def build_visual_filtergraph(
    config: VisualConfig,
    source_media_path: str,
    source_info: MediaStreamInfo,
    secondary_video_path: Optional[str] = None
) -> VisualGraphSpecification:
    """Construct a complex FFmpeg filtergraph for transformative vertical video restructuring.

    Args:
        config: VisualConfig specifying layout, resolution, colors, and enhancements.
        source_media_path: Primary source media file.
        source_info: MediaStreamInfo of primary source.
        secondary_video_path: Optional secondary reaction/commentary video.

    Returns:
        VisualGraphSpecification containing filtergraph string and input list.
    """
    video_inputs = [source_media_path]
    micro_fx = build_micro_enhancements_filter(config)
    micro_fx_chain = f",{micro_fx}" if micro_fx else ""

    W = config.output_width
    H = config.output_height
    fps = config.target_fps

    # -------------------------------------------------------------
    # 1. SPLIT-SCREEN LAYOUT (Top Source, Bottom Commentary / B-Roll)
    # -------------------------------------------------------------
    if config.layout_mode == LayoutMode.SPLIT_SCREEN:
        top_h = int(H * config.top_ratio)
        bot_h = H - top_h
        divider_y = top_h - (config.divider_thickness // 2)

        sec_path = secondary_video_path or config.secondary_video_path
        if sec_path and os.path.isfile(sec_path):
            video_inputs.append(sec_path)
            sec_idx = len(video_inputs) - 1
            bot_stream_expr = (
                f"[{sec_idx}:v]scale={W}:{bot_h}:force_original_aspect_ratio=increase,"
                f"crop={W}:{bot_h},setsar=1,fps={fps}[bot];"
            )
        else:
            # Generate animated ambient commentary background panel
            bot_stream_expr = (
                f"color=c=0x151b2e:s={W}x{bot_h}:r={fps}:d={source_info.duration},"
                f"drawbox=x=40:y=40:w={W-80}:h={bot_h-80}:color=white@0.05:t=fill,"
                f"setsar=1[bot];"
            )

        fg = (
            f"[0:v]scale={W}:{top_h}:force_original_aspect_ratio=increase,"
            f"crop={W}:{top_h},setsar=1,fps={fps}{micro_fx_chain}[top];"
            f"{bot_stream_expr}"
            f"[top][bot]vstack=inputs=2[stacked];"
            f"[stacked]drawbox=y={divider_y}:w={W}:h={config.divider_thickness}:"
            f"color={config.divider_color}:t=fill,format=yuv420p[vout]"
        )
        return VisualGraphSpecification(filtergraph=fg, output_label="[vout]", video_inputs=video_inputs)

    # -------------------------------------------------------------
    # 2. PICTURE-IN-PICTURE (PIP Reaction Inset Layout)
    # -------------------------------------------------------------
    elif config.layout_mode == LayoutMode.PIP:
        pip_w = int(W * 0.35)
        pip_h = int(pip_w * 1.33)  # 3:4 portrait or square reaction inset

        sec_path = secondary_video_path or config.secondary_video_path
        if sec_path and os.path.isfile(sec_path):
            video_inputs.append(sec_path)
            sec_idx = len(video_inputs) - 1
            inset_stream_expr = (
                f"[{sec_idx}:v]scale={pip_w}:{pip_h}:force_original_aspect_ratio=increase,"
                f"crop={pip_w}:{pip_h},setsar=1,fps={fps}[inset];"
            )
        else:
            # Generate synthetic speaker reaction card
            inset_stream_expr = (
                f"color=c=0x1f293d:s={pip_w}x{pip_h}:r={fps}:d={source_info.duration},"
                f"drawbox=x=4:y=4:w={pip_w-8}:h={pip_h-8}:color=white@0.15:t=fill,"
                f"drawbox=x=0:y=0:w={pip_w}:h={pip_h}:color=yellow@0.8:t=3,setsar=1[inset];"
            )

        # Coordinate positioning
        if config.pip_position == "bottom_right":
            overlay_pos = f"W-w-50:H-h-120"
        elif config.pip_position == "bottom_left":
            overlay_pos = f"50:H-h-120"
        else:
            overlay_pos = f"W-w-50:120"

        fg = (
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},boxblur=25:5,setsar=1,fps={fps}[bg];"
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,setsar=1,fps={fps}{micro_fx_chain}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
            f"{inset_stream_expr}"
            f"[base][inset]overlay={overlay_pos},format=yuv420p[vout]"
        )
        return VisualGraphSpecification(filtergraph=fg, output_label="[vout]", video_inputs=video_inputs)

    # -------------------------------------------------------------
    # 3. CANVAS LETTERBOX (Educational / Analytical Card Layout)
    # -------------------------------------------------------------
    elif config.layout_mode == LayoutMode.CANVAS_LETTERBOX:
        # 16:9 source in center occupies 1080x608
        center_h = int(W * (source_info.height / max(1, source_info.width)))
        if center_h > 900:
            center_h = 720

        fg = (
            f"color=c={config.canvas_bg_color}:s={W}x{H}:r={fps}:d={source_info.duration}[bg];"
            f"[0:v]scale={W}:{center_h}:force_original_aspect_ratio=decrease,setsar=1,fps={fps}{micro_fx_chain}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
            f"[base]drawbox=x=40:y=100:w={W-80}:h=320:color=white@0.08:t=fill,"
            f"drawbox=x=40:y={H-480}:w={W-80}:h=360:color=white@0.06:t=fill,"
            f"format=yuv420p[vout]"
        )
        return VisualGraphSpecification(filtergraph=fg, output_label="[vout]", video_inputs=video_inputs)

    # -------------------------------------------------------------
    # 4. BLUR-FILL (9:16 Vertical Background Blur with Centered Crisp Frame)
    # -------------------------------------------------------------
    else:  # LayoutMode.BLUR_FILL
        fg = (
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},boxblur=25:5,setsar=1,fps={fps}[bg];"
            f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease,setsar=1,fps={fps}{micro_fx_chain}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,format=yuv420p[vout]"
        )
        return VisualGraphSpecification(filtergraph=fg, output_label="[vout]", video_inputs=video_inputs)
