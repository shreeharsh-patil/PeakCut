"""Audio-Energy & Hype Highlight Detector for OpenShorts.

Features:
- Detects non-speech viral moments (sports goals, crowd cheers, gaming reactions, stunts, drops)
- Computes RMS audio energy envelope across sliding windows
- Sustained hype convolution smoothing to separate true excitement from isolated mic spikes
- Percentile-based adaptive thresholding with minimum spacing suppression
- Outputs ready-to-render clip definitions compatible with OpenShorts' blurred-pad GENERAL layout
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("openshorts.highlight_detector")


@dataclass
class HighlightMoment:
    start: float
    end: float
    duration: float
    peak_time: float
    score: float
    title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_audio_energy(
    video_or_audio_path: Path | str,
    sample_rate: int = 8000,
    window_sec: float = 0.5,
) -> Tuple[np.ndarray, float]:
    """Decode audio via FFmpeg to raw mono PCM and compute RMS energy per window."""
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(video_or_audio_path),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True)
        raw_pcm = proc.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg audio extraction failed: {e.stderr.decode('utf-8', errors='ignore')}")
        return np.array([]), 0.0

    samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    total_samples = len(samples)
    duration_sec = total_samples / float(sample_rate) if sample_rate else 0.0

    samples_per_win = int(sample_rate * window_sec)
    if samples_per_win <= 0 or total_samples < samples_per_win:
        return np.array([]), duration_sec

    num_windows = total_samples // samples_per_win
    truncated = samples[: num_windows * samples_per_win].reshape(-1, samples_per_win)
    rms_envelope = np.sqrt(np.mean(truncated ** 2, axis=1) + 1e-10)

    return rms_envelope, duration_sec


def detect_hype_moments(
    media_path: Path | str,
    *,
    max_clips: int = 5,
    target_clip_duration: float = 20.0,
    pre_peak_sec: float = 7.0,
    post_peak_sec: float = 13.0,
    min_gap_sec: float = 25.0,
    smooth_sec: float = 3.0,
    percentile_threshold: float = 85.0,
) -> List[HighlightMoment]:
    """Identify the top hype moments in audio energy."""
    window_sec = 0.5
    envelope, duration = extract_audio_energy(media_path, sample_rate=8000, window_sec=window_sec)
    if envelope.size == 0 or duration < 10.0:
        return []

    # Apply convolution smoothing kernel to detect sustained energy
    kernel_size = max(1, int(round(smooth_sec / window_sec)))
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(envelope, kernel, mode="same")

    # Threshold based on percentile
    thresh = float(np.percentile(smoothed, percentile_threshold))
    peak_candidates: List[Tuple[float, float]] = []  # (time_sec, smoothed_score)

    for i in range(1, len(smoothed) - 1):
        val = smoothed[i]
        if val > thresh and val >= smoothed[i - 1] and val >= smoothed[i + 1]:
            time_sec = i * window_sec
            peak_candidates.append((time_sec, float(val)))

    # Sort peaks by score descending
    peak_candidates.sort(key=lambda x: x[1], reverse=True)

    selected_moments: List[HighlightMoment] = []
    used_times: List[float] = []

    for peak_t, score in peak_candidates:
        if any(abs(peak_t - u) < min_gap_sec for u in used_times):
            continue

        start_t = max(0.0, peak_t - pre_peak_sec)
        end_t = min(duration, start_t + target_clip_duration)
        actual_dur = end_t - start_t

        if actual_dur >= 5.0:
            norm_score = min(100.0, round(score * 300.0, 1))
            moment = HighlightMoment(
                start=round(start_t, 2),
                end=round(end_t, 2),
                duration=round(actual_dur, 2),
                peak_time=round(peak_t, 2),
                score=norm_score,
                title=f"Hype Moment @ {int(peak_t // 60)}:{int(peak_t % 60):02d}",
            )
            selected_moments.append(moment)
            used_times.append(peak_t)

            if len(selected_moments) >= max_clips:
                break

    selected_moments.sort(key=lambda m: m.start)
    return selected_moments


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="OpenShorts Audio Highlight Detector")
    parser.add_argument("media", help="Input video or audio file")
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--duration", type=float, default=20.0, help="Target clip duration")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    moments = detect_hype_moments(
        args.media,
        max_clips=args.max_clips,
        target_clip_duration=args.duration,
    )

    if args.json:
        print(json.dumps([m.to_dict() for m in moments], indent=2))
    else:
        print(f"\n--- Detected {len(moments)} Hype Moments ---")
        for m in moments:
            print(f"- [{m.start:.1f}s - {m.end:.1f}s] {m.title} (Peak: {m.peak_time:.1f}s, Score: {m.score})")


if __name__ == "__main__":
    cli_main()
