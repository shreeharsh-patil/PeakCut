"""Visual Color Grading Engine for OpenShorts.

Features:
- Professional color grade presets tailored for high-converting 9:16 vertical video
- Presets:
    1. vibrant_pop: High-saturation punchy colors, vibrant highlights, and edge unsharp
    2. warm_cinema: Warm sunset tones, cinematic color balance, and subtle corner vignette
    3. moody_dark: Desaturated greens/blues, deep blacks, high contrast
    4. clean_contrast: Commercial punchy clarity and crisp whites
    5. cyberpunk_neon: Electric magentas and cyans with vibrant saturated highlights
- Direct FFmpeg filter string generation or standalone video transcoding
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

PRESETS: Dict[str, Dict[str, Any]] = {
    "vibrant_pop": {
        "label": "Vibrant Pop",
        "description": "Boosted saturation, rich colors, and crisp edge clarity.",
        "filter": "eq=saturation=1.28:contrast=1.12:brightness=0.02,unsharp=3:3:0.8",
    },
    "warm_cinema": {
        "label": "Warm Cinema",
        "description": "Warm golden highlights, filmic contrast, and subtle corner vignette.",
        "filter": "colorbalance=rs=0.08:gs=0.02:bs=-0.06:rm=0.06:bm=-0.04,eq=contrast=1.14,vignette=PI/4",
    },
    "moody_dark": {
        "label": "Moody Dark",
        "description": "Desaturated shadows, moody cool tones, and elevated contrast.",
        "filter": "eq=saturation=0.86:contrast=1.22:brightness=-0.03,colorbalance=bh=0.05:gh=-0.03",
    },
    "clean_contrast": {
        "label": "Clean Contrast",
        "description": "Crisp commercial clarity, punchy contrast, and subtle sharpening.",
        "filter": "eq=contrast=1.16:brightness=0.01:saturation=1.08,unsharp=5:5:0.6",
    },
    "cyberpunk_neon": {
        "label": "Cyberpunk Neon",
        "description": "Electric magentas and cyans with deep shadow contrast.",
        "filter": "colorbalance=rm=0.15:bm=0.18:gm=-0.08,eq=saturation=1.35:contrast=1.22",
    },
}


def get_color_grade_filter(preset_name: str) -> Optional[str]:
    """Return the FFmpeg video filtergraph fragment for a preset."""
    name = (preset_name or "").lower().strip()
    if name in ("none", "", "default", "normal"):
        return None
    preset = PRESETS.get(name)
    if not preset:
        raise ValueError(f"Unknown color grade preset '{preset_name}'. Choose from: {list(PRESETS.keys())}")
    return preset["filter"]


def list_color_grade_presets() -> Dict[str, Dict[str, str]]:
    """Return list of available presets and descriptions."""
    return {k: {"label": v["label"], "description": v["description"]} for k, v in PRESETS.items()}


def apply_color_grade(
    input_video: Path | str,
    output_video: Path | str,
    preset_name: str = "vibrant_pop",
) -> Path:
    """Apply visual color grading to an input video."""
    input_p = Path(input_video)
    output_p = Path(output_video)
    if not input_p.exists():
        raise FileNotFoundError(f"Input video not found: {input_p}")
    output_p.parent.mkdir(parents=True, exist_ok=True)

    vf = get_color_grade_filter(preset_name)
    if not vf:
        # None/default: stream copy
        subprocess.run(["ffmpeg", "-y", "-i", str(input_p), "-c", "copy", str(output_p)], check=True)
        return output_p

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_p),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "19",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_p),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"Color grading failed: {res.stderr[-300:]}")

    return output_p


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="OpenShorts Visual Color Grading")
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--preset", "-p", default="vibrant_pop", choices=list(PRESETS.keys()))
    args = parser.parse_args()

    out = apply_color_grade(args.input, args.output, preset_name=args.preset)
    print(f"Applied color grade '{args.preset}' to {out}")


if __name__ == "__main__":
    cli_main()
