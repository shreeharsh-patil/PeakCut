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
    timeout: int = 600,
) -> Tuple[np.ndarray, float]:
    """Decode audio via FFmpeg to raw mono PCM and compute RMS energy per window.

    Streams PCM window-by-window so multi-hour videos use O(num_windows) RAM
    instead of buffering gigabytes of decoded samples into memory.
    """
    samples_per_win = int(sample_rate * window_sec)
    if sample_rate <= 0 or samples_per_win <= 0:
        return np.array([], dtype=np.float32), 0.0

    chunk_bytes = samples_per_win * 2  # 16-bit signed mono PCM
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(video_or_audio_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-",
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=max(chunk_bytes * 4, 65536),
        )
    except OSError as e:
        logger.error(f"FFmpeg audio extraction failed to start: {e}")
        return np.array([], dtype=np.float32), 0.0

    rms_values: List[float] = []
    total_samples = 0
    buf = bytearray()
    try:
        assert proc.stdout is not None
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            buf.extend(data)
            while len(buf) >= chunk_bytes:
                win_bytes = bytes(buf[:chunk_bytes])
                del buf[:chunk_bytes]
                samples = np.frombuffer(win_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                total_samples += len(samples)
                rms_values.append(float(np.sqrt(np.mean(samples ** 2) + 1e-10)))
        if len(buf) >= 2:
            rem = len(buf) - (len(buf) % 2)
            total_samples += rem // 2
        try:
            _, stderr_bytes = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.error(f"FFmpeg audio extraction timed out after {timeout}s")
            return np.array([], dtype=np.float32), 0.0
        if proc.returncode != 0 and not rms_values:
            err = (stderr_bytes or b"").decode("utf-8", errors="ignore")
            logger.error(f"FFmpeg audio extraction failed (rc={proc.returncode}): {err[-300:]}")
            return np.array([], dtype=np.float32), 0.0
    finally:
        if proc.stdout:
            try:
                proc.stdout.close()
            except OSError:
                pass
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    duration_sec = total_samples / float(sample_rate) if sample_rate else 0.0
    if not rms_values:
        return np.array([], dtype=np.float32), duration_sec
    return np.asarray(rms_values, dtype=np.float32), duration_sec


def detect_highlights(
    media_path: Path | str,
    target_clips: int = 3,
    **kwargs: Any,
) -> List[HighlightMoment]:
    """Alias for detect_hype_moments for generic highlight detection."""
    return detect_hype_moments(media_path, max_clips=target_clips, **kwargs)


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
    min_rms_floor: float = 0.008,
) -> List[HighlightMoment]:
    """Identify the top non-overlapping hype moments in audio energy."""
    window_sec = 0.5
    envelope, duration = extract_audio_energy(media_path, sample_rate=8000, window_sec=window_sec)
    if envelope.size == 0 or duration < 10.0:
        return []

    # Reject completely silent / digital-zero audio tracks instead of scoring noise floor
    if float(np.max(envelope)) < min_rms_floor:
        return []

    # Apply convolution smoothing kernel to detect sustained energy
    kernel_size = max(1, int(round(smooth_sec / window_sec)))
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smoothed = np.convolve(envelope, kernel, mode="same")

    # Threshold based on percentile and absolute noise floor
    baseline = float(np.median(smoothed))
    thresh = max(float(np.percentile(smoothed, percentile_threshold)), min_rms_floor)
    peak_candidates: List[Tuple[float, float]] = []  # (time_sec, smoothed_score)

    for i in range(1, len(smoothed) - 1):
        val = float(smoothed[i])
        if val > thresh and val >= float(smoothed[i - 1]) and val >= float(smoothed[i + 1]):
            time_sec = i * window_sec
            peak_candidates.append((time_sec, val))

    if not peak_candidates:
        return []

    # Sort peaks by score descending
    peak_candidates.sort(key=lambda x: x[1], reverse=True)
    max_peak = max(p[1] for p in peak_candidates)

    selected_moments: List[HighlightMoment] = []
    used_times: List[float] = []

    for peak_t, score in peak_candidates:
        if any(abs(peak_t - u) < min_gap_sec for u in used_times):
            continue

        clip_span = min(float(target_clip_duration), float(duration))
        start_t = max(0.0, peak_t - pre_peak_sec)
        end_t = start_t + clip_span
        if end_t > duration:
            end_t = float(duration)
            start_t = max(0.0, end_t - clip_span)
        actual_dur = end_t - start_t

        # Prevent interval overlap with already-selected moments
        if any(not (end_t <= m.start or start_t >= m.end) for m in selected_moments):
            continue

        if actual_dur >= 5.0:
            # Score from measurable prominence over baseline + absolute RMS loudness
            prominence = min(1.0, (score - baseline) / max(max_peak - baseline, 1e-4))
            loudness_factor = min(1.0, score / 0.35)
            norm_score = round(min(100.0, max(1.0, (0.55 * prominence + 0.45 * loudness_factor) * 100.0)), 1)
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
