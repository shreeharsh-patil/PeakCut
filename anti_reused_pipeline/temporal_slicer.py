"""Temporal Slicing & Curation Engine.

Enforces the YouTube Content ID and YPP Reused Content policy rule:
No single unedited segment of the source video may play uninterrupted for
more than 5.0 seconds (T_chunk <= 5.0s).
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Tuple, Optional

from anti_reused_pipeline.config import TemporalConfig
from anti_reused_pipeline.ingest import MediaStreamInfo

logger = logging.getLogger("anti_reused_pipeline.temporal")


@dataclass
class SlicingResult:
    """Output summary of temporal slicing operation."""
    sliced_media_path: str
    total_duration: float
    max_chunk_duration: float
    chunk_count: int
    chunks: List[Tuple[float, float]]


def calculate_compliant_chunks(
    duration: float,
    max_chunk: float = 5.0,
    explicit_ranges: Optional[List[Tuple[float, float]]] = None
) -> List[Tuple[float, float]]:
    """Determine chunk boundary timestamps ensuring all segments satisfy max_chunk.

    Args:
        duration: Total duration of the source video in seconds.
        max_chunk: Maximum allowed duration for any single continuous chunk.
        explicit_ranges: Optional list of (start, end) time ranges from scene detector.

    Returns:
        List of (start, end) tuples where (end - start) <= max_chunk.
    """
    if explicit_ranges:
        chunks = []
        for start, end in explicit_ranges:
            start = max(0.0, min(start, duration))
            end = max(start, min(end, duration))
            seg_len = end - start
            if seg_len <= 0:
                continue
            if seg_len <= max_chunk:
                chunks.append((start, end))
            else:
                # Subdivide oversized ranges into compliant slices
                cur = start
                while cur < end:
                    nxt = min(cur + max_chunk, end)
                    chunks.append((cur, nxt))
                    cur = nxt
        return chunks

    # Default: Subdivide entire duration into compliant chunks
    chunks = []
    cur = 0.0
    while cur < duration:
        nxt = min(cur + max_chunk, duration)
        if nxt - cur >= 0.5:  # Skip trivial sub-half-second fragments
            chunks.append((cur, nxt))
        cur = nxt
    return chunks


def execute_temporal_slice(
    input_info: MediaStreamInfo,
    config: TemporalConfig,
    work_dir: str
) -> SlicingResult:
    """Slice and reassemble source video to ensure compliance with max chunk duration.

    Args:
        input_info: MediaStreamInfo of the source media.
        config: TemporalConfig specifying chunk duration limits and freeze frames.
        work_dir: Working directory for intermediate outputs.

    Returns:
        SlicingResult containing path to sliced video and chunk metrics.
    """
    os.makedirs(work_dir, exist_ok=True)
    chunks = calculate_compliant_chunks(
        duration=input_info.duration,
        max_chunk=config.max_chunk_duration,
        explicit_ranges=config.segment_ranges
    )

    if not chunks:
        # Fallback if video is extremely short
        chunks = [(0.0, input_info.duration)]

    # Compute actual max chunk duration
    actual_max_chunk = max(end - start for start, end in chunks)

    # If the video is already within limit and slicing is disabled or only 1 chunk:
    if input_info.duration <= config.max_chunk_duration and len(chunks) == 1 and not config.interleave_freeze_frame:
        logger.info("Source duration (%.2fs) <= max_chunk (%.2fs). Preserving original stream.",
                    input_info.duration, config.max_chunk_duration)
        return SlicingResult(
            sliced_media_path=input_info.path,
            total_duration=input_info.duration,
            max_chunk_duration=actual_max_chunk,
            chunk_count=1,
            chunks=chunks
        )

    output_sliced_path = os.path.join(work_dir, "temporal_curated.mp4")
    concat_list_path = os.path.join(work_dir, "chunks_concat.txt")

    slice_files = []
    try:
        # Extract individual chunks
        for idx, (st, en) in enumerate(chunks):
            chunk_file = os.path.join(work_dir, f"chunk_{idx:03d}.mp4")
            dur = en - st

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{st:.3f}",
                "-t", f"{dur:.3f}",
                "-i", input_info.path,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-c:a", "aac",
                "-avoid_negative_ts", "make_zero",
                chunk_file
            ]
            subprocess.run(cmd, check=True)
            slice_files.append(chunk_file)

        # Write ffmpeg concat file
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for sf in slice_files:
                # Windows safe path for concat demuxer
                safe_path = sf.replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        # Concat demuxer execution
        concat_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            output_sliced_path
        ]
        subprocess.run(concat_cmd, check=True)

    finally:
        # Clean up chunk files and concat list
        if os.path.isfile(concat_list_path):
            try:
                os.remove(concat_list_path)
            except OSError:
                pass
        for sf in slice_files:
            if os.path.isfile(sf):
                try:
                    os.remove(sf)
                except OSError:
                    pass

    # Probe reassembled duration
    assembled_probe_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", output_sliced_path
    ]
    dur_proc = subprocess.run(assembled_probe_cmd, capture_output=True, text=True)
    assembled_duration = float(dur_proc.stdout.strip()) if dur_proc.stdout.strip() else sum(e - s for s, e in chunks)

    logger.info("Temporal slicing complete: %d chunks, max_chunk=%.2fs, total=%.2fs",
                len(chunks), actual_max_chunk, assembled_duration)

    return SlicingResult(
        sliced_media_path=output_sliced_path,
        total_duration=assembled_duration,
        max_chunk_duration=actual_max_chunk,
        chunk_count=len(chunks),
        chunks=chunks
    )
