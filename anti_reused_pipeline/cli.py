"""Command-Line Interface for the Anti-Reused Video Automation Pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from anti_reused_pipeline.config import (
    PipelineConfig, VisualConfig, AudioConfig, TemporalConfig, CaptionConfig,
    LayoutMode, TTSProvider, AudioDuckingMode
)
from anti_reused_pipeline.ingest import probe_media
from anti_reused_pipeline.pipeline_engine import TransformPipeline
from anti_reused_pipeline.compliance_validator import calculate_compliance_score, audit_file_metadata
from anti_reused_pipeline.tts_engine import generate_voiceover


def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )


def cmd_process(args: argparse.Namespace) -> int:
    setup_logger(args.verbose)

    layout_mode = LayoutMode(args.layout)
    tts_prov = TTSProvider(args.tts_provider)
    ducking_mode = AudioDuckingMode(args.ducking_mode)

    audio_cfg = AudioConfig(
        voiceover_provider=tts_prov,
        voice_name=args.voice_name,
        custom_voiceover_path=args.voiceover_audio,
        voiceover_text=args.voiceover_text,
        ducking_mode=ducking_mode,
        bgm_path=args.bgm
    )

    visual_cfg = VisualConfig(
        layout_mode=layout_mode,
        secondary_video_path=args.secondary_video,
        enable_micro_adjustments=not args.no_micro_fx
    )

    temporal_cfg = TemporalConfig(
        max_chunk_duration=args.max_chunk,
        enable_slicing=not args.no_slicing
    )

    caption_cfg = CaptionConfig(
        burn_captions=not args.no_captions
    )

    pipeline_cfg = PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        audio=audio_cfg,
        visual=visual_cfg,
        temporal=temporal_cfg,
        captions=caption_cfg,
        enforce_compliance_pass=args.enforce_pass
    )

    runner = TransformPipeline(pipeline_cfg)
    try:
        result = runner.execute()
        if not result.is_success:
            print(f"\n❌ Pipeline failed: {result.error_message}", file=sys.stderr)
            return 1

        print("\n" + result.compliance.summary())
        print(f"\n✅ Output successfully rendered to: {result.output_path}")
        print(f"⏱️  Duration: {result.duration:.2f}s | Processing time: {result.processing_time_sec:.2f}s")
        return 0
    finally:
        if not args.keep_scratch:
            runner.cleanup()


def cmd_validate(args: argparse.Namespace) -> int:
    setup_logger(args.verbose)
    info = probe_media(args.file)
    layout = LayoutMode(args.layout)
    ducking = AudioDuckingMode(args.ducking_mode)

    rep = calculate_compliance_score(
        total_video_duration=info.duration,
        voiceover_duration=args.voiceover_duration or (info.duration * 0.75),
        layout_mode=layout,
        max_chunk_duration=args.max_chunk,
        ducking_mode=ducking,
        file_path_for_audit=args.file
    )

    print("\n" + rep.summary())
    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))

    return 0 if rep.is_monetizable else 2


def cmd_voiceover(args: argparse.Namespace) -> int:
    setup_logger(args.verbose)
    audio_cfg = AudioConfig(
        voiceover_provider=TTSProvider(args.tts_provider),
        voice_name=args.voice_name,
        voiceover_text=args.text
    )
    res = generate_voiceover(audio_cfg, args.out_dir)
    print(f"Generated Voiceover: {res.audio_path} ({res.duration:.2f}s)")
    if res.srt_path:
        print(f"Generated Subtitles: {res.srt_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anti-Reused Video Automation & Transformative Content Pipeline"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Process Command
    proc_p = subparsers.add_parser("process", help="Transform input video into compliant vertical 9:16 Short")
    proc_p.add_argument("-i", "--input", required=True, help="Path to input video file or YouTube URL")
    proc_p.add_argument("-o", "--output", required=True, help="Path for rendered output MP4")
    proc_p.add_argument("-l", "--layout", default="split_screen",
                        choices=["split_screen", "pip", "canvas_letterbox", "blur_fill"],
                        help="Visual restructuring layout mode")
    proc_p.add_argument("--voiceover-text", help="Script text for automated TTS voiceover commentary")
    proc_p.add_argument("--voiceover-audio", help="Path to user custom voiceover audio file")
    proc_p.add_argument("--tts-provider", default="auto", choices=["auto", "edge_tts", "sapi", "custom_audio"],
                        help="Text-to-speech engine")
    proc_p.add_argument("--voice-name", default="en-US-ChristopherNeural", help="TTS voice character")
    proc_p.add_argument("--bgm", help="Path to background music audio file")
    proc_p.add_argument("--ducking-mode", default="sidechain_duck",
                        choices=["sidechain_duck", "mute_source", "parity"],
                        help="Audio treatment: sidechain ducking, source muting, or parity")
    proc_p.add_argument("--max-chunk", type=float, default=5.0,
                        help="Maximum continuous unedited chunk duration in seconds (<= 5.0 for YPP compliance)")
    proc_p.add_argument("--secondary-video", help="Path to reaction or commentary video for split_screen or PIP")
    proc_p.add_argument("--no-slicing", action="store_true", help="Disable temporal chunk slicing")
    proc_p.add_argument("--no-captions", action="store_true", help="Disable dynamic subtitle burning")
    proc_p.add_argument("--no-micro-fx", action="store_true", help="Disable anti-perceptual hash micro filters")
    proc_p.add_argument("--enforce-pass", action="store_true", help="Abort export if compliance score < 0.80")
    proc_p.add_argument("--keep-scratch", action="store_true", help="Retain intermediate working directory files")
    proc_p.set_defaults(func=cmd_process)

    # 2. Validate Command
    val_p = subparsers.add_parser("validate", help="Audit video against YouTube compliance & fingerprinting metrics")
    val_p.add_argument("-f", "--file", required=True, help="Video file to audit")
    val_p.add_argument("-l", "--layout", default="split_screen",
                       choices=["split_screen", "pip", "canvas_letterbox", "blur_fill"])
    val_p.add_argument("--voiceover-duration", type=float, help="Voiceover duration in seconds")
    val_p.add_argument("--max-chunk", type=float, default=5.0, help="Max continuous unedited chunk")
    val_p.add_argument("--ducking-mode", default="sidechain_duck", choices=["sidechain_duck", "mute_source", "parity"])
    val_p.add_argument("--json", action="store_true", help="Emit JSON report to stdout")
    val_p.set_defaults(func=cmd_validate)

    # 3. Voiceover Command
    vo_p = subparsers.add_parser("voiceover", help="Generate standalone voiceover audio and SRT timing")
    vo_p.add_argument("-t", "--text", required=True, help="Commentary script text")
    vo_p.add_argument("-d", "--out-dir", default="./output", help="Directory to save generated voiceover")
    vo_p.add_argument("--tts-provider", default="auto", choices=["auto", "edge_tts", "sapi"])
    vo_p.add_argument("--voice-name", default="en-US-ChristopherNeural")
    vo_p.set_defaults(func=cmd_voiceover)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
