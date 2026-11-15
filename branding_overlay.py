"""Watermark, Logo & Banner Overlay Engine for OpenShorts.

Features:
- Overlays creator logos, watermarks, and brand graphics onto 9:16 vertical shorts
- Smart positioning:
    - 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'top-center', 'bottom-center'
- Automatic dynamic scaling proportional to video width (e.g. 12% - 25% of 1080px width)
- Alpha opacity control (e.g. 0.2 to 1.0) using colorchannelmixer
- Safe zone padding to prevent UI collision with TikTok/IG Reels/Shorts action buttons
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

POSITIONS = {
    "top-left": ("{margin_x}", "{margin_y}"),
    "top-right": ("W-w-{margin_x}", "{margin_y}"),
    "bottom-left": ("{margin_x}", "H-h-{margin_y}"),
    "bottom-right": ("W-w-{margin_x}", "H-h-{margin_y}"),
    "top-center": ("(W-w)/2", "{margin_y}"),
    "bottom-center": ("(W-w)/2", "H-h-{margin_y}"),
}


def build_overlay_filter(
    position: str = "top-right",
    opacity: float = 0.85,
    scale_width_px: int = 180,
    margin_x: int = 40,
    margin_y: int = 80,
) -> str:
    """Build FFmpeg filtergraph string to scale, adjust opacity, and position overlay."""
    pos = POSITIONS.get(position.lower(), POSITIONS["top-right"])
    x_expr = pos[0].format(margin_x=margin_x, margin_y=margin_y)
    y_expr = pos[1].format(margin_x=margin_x, margin_y=margin_y)

    # Scale logo and adjust alpha channel
    filter_chain = (
        f"[1:v]scale={scale_width_px}:-1:flags=lanczos,format=rgba,"
        f"colorchannelmixer=aa={opacity:.2f}[logo];"
        f"[0:v][logo]overlay={x_expr}:{y_expr}"
    )
    return filter_chain


def apply_branding_overlay(
    input_video: Path | str,
    overlay_image: Path | str,
    output_video: Path | str,
    position: str = "top-right",
    opacity: float = 0.85,
    scale_width_px: int = 180,
    margin_x: int = 40,
    margin_y: int = 80,
) -> Path:
    """Burn a logo/watermark overlay onto a video file."""
    input_p = Path(input_video)
    overlay_p = Path(overlay_image)
    output_p = Path(output_video)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    filtergraph = build_overlay_filter(
        position=position,
        opacity=opacity,
        scale_width_px=scale_width_px,
        margin_x=margin_x,
        margin_y=margin_y,
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_p),
        "-i", str(overlay_p),
        "-filter_complex", filtergraph,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_p),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"Branding overlay failed: {res.stderr[-300:]}")

    return output_p


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="OpenShorts Branding & Logo Overlay")
    parser.add_argument("video", help="Input video file")
    parser.add_argument("logo", help="Overlay image (PNG with transparency recommended)")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--position", "-p", default="top-right", choices=list(POSITIONS.keys()))
    parser.add_argument("--opacity", "-o", type=float, default=0.85)
    parser.add_argument("--width", "-w", type=int, default=180)
    args = parser.parse_args()

    out = apply_branding_overlay(
        args.video,
        args.logo,
        args.output,
        position=args.position,
        opacity=args.opacity,
        scale_width_px=args.width,
    )
    print(f"Applied branding overlay to {out}")


if __name__ == "__main__":
    cli_main()
