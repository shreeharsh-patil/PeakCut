"""Comprehensive unit and integration test suite for the Anti-Reused Video Pipeline."""

import os
import subprocess
import pytest
from pathlib import Path

from anti_reused_pipeline.config import (
    PipelineConfig, VisualConfig, AudioConfig, TemporalConfig, CaptionConfig,
    LayoutMode, TTSProvider, AudioDuckingMode
)
from anti_reused_pipeline.ingest import probe_media, MediaStreamInfo
from anti_reused_pipeline.temporal_slicer import calculate_compliant_chunks, execute_temporal_slice
from anti_reused_pipeline.tts_engine import synthesize_windows_sapi, build_timing_srt, generate_voiceover
from anti_reused_pipeline.audio_transform import build_audio_filtergraph
from anti_reused_pipeline.visual_layouts import build_visual_filtergraph, build_micro_enhancements_filter
from anti_reused_pipeline.captions import escape_ffmpeg_path, generate_styled_ass_from_srt
from anti_reused_pipeline.compliance_validator import (
    calculate_compliance_score, audit_file_metadata, ComplianceStatus
)
from anti_reused_pipeline.pipeline_engine import TransformPipeline


DEMO_CLIP = "D:/yt/dashboard/public/demo/clip-source.mp4"
BGM_SAMPLE = "D:/yt/shorts-autopilot/clips_auto/assets/music_bed.mp3"


def test_config_initialization():
    cfg = PipelineConfig(input_path="dummy.mp4", output_path="out.mp4")
    assert cfg.visual.output_width == 1080
    assert cfg.visual.output_height == 1920
    assert cfg.audio.ducking_mode == AudioDuckingMode.SIDECHAIN_DUCK
    assert cfg.temporal.max_chunk_duration == 5.0
    assert cfg.compliance.pass_threshold == 0.80


def test_media_probing():
    assert os.path.isfile(DEMO_CLIP), f"Demo clip not found at {DEMO_CLIP}"
    info = probe_media(DEMO_CLIP)
    assert info.duration > 0
    assert info.width > 0
    assert info.height > 0
    assert info.has_video is True


def test_calculate_compliant_chunks():
    # 12-second video with 5.0s max chunk should produce 3 chunks: [0..5], [5..10], [10..12]
    chunks = calculate_compliant_chunks(duration=12.0, max_chunk=5.0)
    assert len(chunks) == 3
    assert chunks[0] == (0.0, 5.0)
    assert chunks[1] == (5.0, 10.0)
    assert chunks[2] == (10.0, 12.0)
    for s, e in chunks:
        assert (e - s) <= 5.0


def test_compliance_scoring_pass():
    # Split screen, 80% voiceover coverage, 4.0s max chunk, sidechain ducking
    rep = calculate_compliance_score(
        total_video_duration=10.0,
        voiceover_duration=8.5,
        layout_mode=LayoutMode.SPLIT_SCREEN,
        max_chunk_duration=4.0,
        ducking_mode=AudioDuckingMode.SIDECHAIN_DUCK
    )
    # Score: 0.35 * 1.0 + 0.25 * 1.0 + 0.25 * 1.0 + 0.15 * 1.0 = 1.0
    assert rep.score >= 0.80
    assert rep.status == ComplianceStatus.PASS
    assert rep.is_monetizable is True


def test_compliance_scoring_fail():
    # Raw frame (0 layout), 0 voiceover, 15s unedited chunk, no ducking
    rep = calculate_compliance_score(
        total_video_duration=15.0,
        voiceover_duration=0.0,
        layout_mode=LayoutMode.BLUR_FILL,
        max_chunk_duration=15.0,
        ducking_mode=AudioDuckingMode.PARITY
    )
    assert rep.score < 0.60
    assert rep.status == ComplianceStatus.FAIL
    assert rep.is_monetizable is False


def test_voiceover_sapi_and_srt(tmp_path):
    wav_out = str(tmp_path / "test_sapi.wav")
    dur = synthesize_windows_sapi("Testing automated voiceover synthesis on Windows.", wav_out)
    assert dur > 0.5
    assert os.path.isfile(wav_out)

    srt_out = str(tmp_path / "test.srt")
    build_timing_srt("Testing automated voiceover synthesis on Windows.", dur, srt_out)
    assert os.path.isfile(srt_out)
    with open(srt_out, "r", encoding="utf-8") as f:
        content = f.read()
    assert "-->" in content
    assert "TESTING AUTOMATED" in content


def test_audio_filtergraph_generation():
    cfg = AudioConfig(
        ducking_mode=AudioDuckingMode.SIDECHAIN_DUCK,
        loudnorm_integrated=-14.0,
        loudnorm_true_peak=-2.0
    )
    spec = build_audio_filtergraph(
        config=cfg,
        source_media_path="source.mp4",
        voiceover_path="vo.wav",
        bgm_path=None,
        has_source_audio=True
    )
    assert "sidechaincompress=" in spec.filtergraph
    assert "loudnorm=I=-14.0:TP=-2.0:LRA=11.0" in spec.filtergraph
    assert spec.output_label == "[aout]"

    # Test Parity mode
    cfg_parity = AudioConfig(ducking_mode=AudioDuckingMode.PARITY)
    spec_parity = build_audio_filtergraph(
        config=cfg_parity,
        source_media_path="source.mp4",
        voiceover_path="vo.wav",
        has_source_audio=True
    )
    assert "sidechaincompress=" not in spec_parity.filtergraph
    assert "amix=inputs=2:weights=1.0 1.0" in spec_parity.filtergraph

    # Test Mute Source mode
    cfg_mute = AudioConfig(ducking_mode=AudioDuckingMode.MUTE_SOURCE)
    spec_mute = build_audio_filtergraph(
        config=cfg_mute,
        source_media_path="source.mp4",
        voiceover_path="vo.wav",
        has_source_audio=True
    )
    assert "sidechaincompress=" not in spec_mute.filtergraph
    assert "[vo]" in spec_mute.filtergraph or "loudnorm" in spec_mute.filtergraph


def test_visual_filtergraph_layouts():
    source_info = MediaStreamInfo(
        path="test.mp4", duration=10.0, width=1920, height=1080, fps=30.0,
        has_video=True, has_audio=True
    )
    # Split screen
    cfg_split = VisualConfig(layout_mode=LayoutMode.SPLIT_SCREEN)
    spec_split = build_visual_filtergraph(cfg_split, "test.mp4", source_info)
    assert "vstack=" in spec_split.filtergraph
    assert "drawbox=" in spec_split.filtergraph

    # PIP
    cfg_pip = VisualConfig(layout_mode=LayoutMode.PIP)
    spec_pip = build_visual_filtergraph(cfg_pip, "test.mp4", source_info)
    assert "boxblur=" in spec_pip.filtergraph
    assert "overlay=" in spec_pip.filtergraph

    # Canvas Letterbox
    cfg_letter = VisualConfig(layout_mode=LayoutMode.CANVAS_LETTERBOX)
    spec_letter = build_visual_filtergraph(cfg_letter, "test.mp4", source_info)
    assert "color=c=0x111625" in spec_letter.filtergraph


def test_subtitle_path_escaping():
    raw_path = "D:\\yt\\subtitles\\test file.srt"
    escaped = escape_ffmpeg_path(raw_path)
    assert "\\" not in escaped or "\\:" in escaped
    assert "D\\:/yt/subtitles/test file.srt" == escaped


def test_end_to_end_pipeline_execution(tmp_path):
    output_mp4 = str(tmp_path / "rendered_short.mp4")

    audio_cfg = AudioConfig(
        voiceover_provider=TTSProvider.SAPI,
        voiceover_text="This clip breaks down the most interesting moments in this video with thorough analysis.",
        ducking_mode=AudioDuckingMode.SIDECHAIN_DUCK,
        bgm_path=BGM_SAMPLE if os.path.isfile(BGM_SAMPLE) else None
    )

    visual_cfg = VisualConfig(
        layout_mode=LayoutMode.SPLIT_SCREEN,
        enable_micro_adjustments=True
    )

    temporal_cfg = TemporalConfig(
        max_chunk_duration=5.0,
        enable_slicing=True
    )

    caption_cfg = CaptionConfig(
        burn_captions=True,
        font_size=20
    )

    pipe_cfg = PipelineConfig(
        input_path=DEMO_CLIP,
        output_path=output_mp4,
        work_dir=str(tmp_path / "scratch"),
        audio=audio_cfg,
        visual=visual_cfg,
        temporal=temporal_cfg,
        captions=caption_cfg,
        strip_metadata=True,
        enforce_compliance_pass=True
    )

    runner = TransformPipeline(pipe_cfg)
    result = runner.execute()

    assert result.is_success is True, f"Pipeline failed: {result.error_message}"
    assert os.path.isfile(output_mp4), "Rendered file does not exist"
    assert os.path.getsize(output_mp4) > 10000, "Rendered file size is too small"

    # Probe output properties
    out_info = probe_media(output_mp4)
    assert out_info.width == 1080
    assert out_info.height == 1920
    assert out_info.has_video is True
    assert out_info.has_audio is True

    # Audit compliance
    assert result.compliance.score >= 0.80
    assert result.compliance.status == ComplianceStatus.PASS
    assert result.compliance.is_monetizable is True

    # Verify metadata sanitization
    meta_audit = audit_file_metadata(output_mp4)
    assert meta_audit.clean is True, f"Output file has residual metadata tags: {meta_audit.tags_found}"
