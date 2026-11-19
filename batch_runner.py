#!/usr/bin/env python3
"""Batch Video Automation Runner for PeakCut / OpenShorts.

Automates the end-to-end extraction of vertical viral shorts from multiple YouTube videos.
Works seamlessly on local workstations and Google Colab (with Google Drive persistence).

Usage Examples:
    # Process multiple URLs directly:
    python batch_runner.py --urls "https://youtu.be/abc,https://youtu.be/xyz" --output "./output"

    # Process from a text file:
    python batch_runner.py --file urls.txt --target-clips 2 --format vertical

    # Google Colab with Google Drive:
    python batch_runner.py --file urls.txt --output "/content/drive/MyDrive/Shorts"
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch automate YouTube to vertical Shorts conversion."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-u", "--urls", type=str,
        help="Comma-separated or space-separated list of YouTube URLs."
    )
    group.add_argument(
        "-f", "--file", type=str,
        help="Path to a text file containing YouTube URLs (one per line)."
    )

    parser.add_argument(
        "-o", "--output", type=str, default="./output",
        help="Output directory (or Google Drive folder) where generated shorts are saved."
    )
    parser.add_argument(
        "--format", type=str, default="vertical", choices=["vertical", "horizontal", "square", "auto"],
        help="Target aspect ratio format (default: vertical 9:16)."
    )
    parser.add_argument(
        "--target-clips", type=int, default=3,
        help="Target number of clips to produce per long video (default: 3)."
    )
    parser.add_argument(
        "--clip-min-seconds", type=float, default=15.0,
        help="Minimum clip duration in seconds (default: 15.0)."
    )
    parser.add_argument(
        "--clip-max-seconds", type=float, default=60.0,
        help="Maximum clip duration in seconds (default: 60.0)."
    )
    parser.add_argument(
        "--live-record-seconds", type=int, default=900,
        help="Seconds of live stream to capture if URL is an active YouTube Live stream (default: 900s / 15m)."
    )
    parser.add_argument(
        "--whisper-model", type=str, default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Faster-Whisper model size (default: base)."
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
        help="Compute device for Whisper and AI inference (default: auto-detect)."
    )
    parser.add_argument(
        "--keep-original", action="store_true",
        help="Keep the downloaded source long video (default: False, cleans up to save disk)."
    )
    return parser.parse_args()


def load_urls(args) -> list[str]:
    raw_urls = []
    if args.urls:
        # Split by comma or whitespace
        parts = args.urls.replace(",", " ").split()
        raw_urls.extend(parts)
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            raise FileNotFoundError(f"URL file not found: {args.file}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    # Strip inline comments
                    clean_line = line.split("#")[0].strip()
                    if clean_line:
                        raw_urls.append(clean_line)

    # De-duplicate while preserving order
    seen = set()
    cleaned = []
    for u in raw_urls:
        u_clean = u.strip().strip('"\'')
        if u_clean and u_clean not in seen:
            seen.add(u_clean)
            cleaned.append(u_clean)
    return cleaned


def configure_hardware(device_arg: str, whisper_model: str):
    """Set optimal hardware acceleration environment variables."""
    has_cuda = False
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except Exception:
        pass

    if device_arg == "cuda" or (device_arg == "auto" and has_cuda):
        os.environ["WHISPER_DEVICE"] = "cuda"
        os.environ["WHISPER_COMPUTE"] = "float16"
        print("⚡ Hardware: NVIDIA CUDA GPU acceleration enabled (float16).")
    else:
        os.environ["WHISPER_DEVICE"] = "cpu"
        os.environ["WHISPER_COMPUTE"] = "int8"
        print("ℹ️ Hardware: CPU mode (int8).")

    os.environ["WHISPER_MODEL"] = whisper_model


def run_batch():
    args = parse_args()
    urls = load_urls(args)

    if not urls:
        print("⚠️ No valid URLs found to process.")
        return 0

    configure_hardware(args.device, args.whisper_model)

    output_base = Path(args.output).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎬 PeakCut / OpenShorts Batch Automation Engine")
    print(f"📦 Total Videos Queued : {len(urls)}")
    print(f"📁 Output Directory    : {output_base}")
    print(f"📐 Target Format       : {args.format}")
    print(f"✂️  Clips Per Video    : {args.target_clips}")
    print(f"⏱️  Duration Limits    : {args.clip_min_seconds}s - {args.clip_max_seconds}s")
    print("=" * 70)

    results = {
        "started_at": datetime.now().isoformat(),
        "total_queued": len(urls),
        "successful": [],
        "failed": [],
    }

    start_batch_time = time.time()

    for idx, url in enumerate(urls, 1):
        job_start = time.time()
        print(f"\n[{idx}/{len(urls)}] 🚀 Processing YouTube URL: {url}")

        # Unique sub-directory for this video's job run
        job_output_dir = output_base / f"batch_job_{idx}_{int(job_start)}"
        job_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "main.py"),
            "--url", url,
            "--format", args.format,
            "--output", str(job_output_dir),
            "--target-clips", str(args.target_clips),
            "--clip-min-seconds", str(args.clip_min_seconds),
            "--clip-max-seconds", str(args.clip_max_seconds),
            "--live-record-seconds", str(args.live_record_seconds),
        ]
        if args.keep_original:
            cmd.append("--keep-original")

        env = os.environ.copy()
        try:
            proc = subprocess.run(cmd, env=env)
            duration = round(time.time() - job_start, 2)

            if proc.returncode == 0:
                # Discover produced clips
                produced_clips = [
                    str(p.relative_to(output_base))
                    for p in job_output_dir.glob("*.mp4")
                    if not p.name.startswith("temp_")
                    and not p.name.endswith((".v1video.mp4", ".aitag.mp4"))
                ]
                print(f"✅ Video {idx} finished in {duration}s! Produced {len(produced_clips)} clip(s).")
                results["successful"].append({
                    "url": url,
                    "job_dir": str(job_output_dir),
                    "clips": produced_clips,
                    "duration_seconds": duration,
                })
            else:
                print(f"❌ Video {idx} failed with exit code {proc.returncode} in {duration}s.")
                results["failed"].append({
                    "url": url,
                    "error_code": proc.returncode,
                    "duration_seconds": duration,
                })
        except KeyboardInterrupt:
            duration = round(time.time() - job_start, 2)
            print(f"\n⚠️ Batch runner interrupted by user during video {idx}.")
            results["failed"].append({
                "url": url,
                "error": "KeyboardInterrupt",
                "duration_seconds": duration,
            })
            break
        except Exception as ex:
            duration = round(time.time() - job_start, 2)
            print(f"❌ Video {idx} exception: {ex}")
            results["failed"].append({
                "url": url,
                "error": str(ex),
                "duration_seconds": duration,
            })

    total_duration = round(time.time() - start_batch_time, 2)
    results["completed_at"] = datetime.now().isoformat()
    results["total_duration_seconds"] = total_duration

    summary_file = output_base / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("🎉 BATCH PROCESSING COMPLETE!")
    print(f"📊 Summary: {len(results['successful'])} succeeded, {len(results['failed'])} failed.")
    print(f"⏱️ Total Execution Time: {total_duration}s")
    print(f"📄 Summary Report Saved: {summary_file}")
    print("=" * 70)

    return 0 if len(results["failed"]) == 0 else 1


if __name__ == "__main__":
    sys.exit(run_batch())
