"""Anti-Reused Content & Transformative Video Automation Pipeline.

A production-grade Python system designed to repurpose video content while strictly
meeting YouTube Partner Program (YPP) reused content policies and bypassing
Content ID perceptual fingerprinting through genuine transformative value:
- Audio Primacy & Sidechain Compression Ducking (-20 dB to -26 dB)
- Structural Vertical Restructuring (Split-Screen, PIP Reaction, Canvas Letterbox, Blur-Fill)
- Temporal Curation (T_chunk <= 5.0s unedited continuous chunk rule)
- Metadata & Container Sanitization (-map_metadata -1, BT.709, CFR 30fps)
- Dynamic Kinetic Captions & Voiceover Synthesis (Edge-TTS / Windows SAPI)
- Programmatic Compliance Scoring Metric (S_compliance >= 0.80)
"""

from anti_reused_pipeline.config import (
    PipelineConfig,
    VisualConfig,
    AudioConfig,
    TemporalConfig,
    CaptionConfig,
    ComplianceThresholds,
    LayoutMode,
    TTSProvider,
    AudioDuckingMode
)
from anti_reused_pipeline.ingest import probe_media, resolve_media_input, MediaStreamInfo
from anti_reused_pipeline.temporal_slicer import execute_temporal_slice, SlicingResult
from anti_reused_pipeline.tts_engine import generate_voiceover, VoiceoverResult
from anti_reused_pipeline.audio_transform import build_audio_filtergraph, render_standalone_audio
from anti_reused_pipeline.visual_layouts import build_visual_filtergraph, build_micro_enhancements_filter
from anti_reused_pipeline.captions import generate_styled_ass_from_srt, build_subtitle_filter_string
from anti_reused_pipeline.compliance_validator import calculate_compliance_score, audit_file_metadata, ComplianceReport
from anti_reused_pipeline.pipeline_engine import TransformPipeline, PipelineResult

__version__ = "1.0.0"
__all__ = [
    "PipelineConfig",
    "VisualConfig",
    "AudioConfig",
    "TemporalConfig",
    "CaptionConfig",
    "ComplianceThresholds",
    "LayoutMode",
    "TTSProvider",
    "AudioDuckingMode",
    "probe_media",
    "resolve_media_input",
    "MediaStreamInfo",
    "execute_temporal_slice",
    "SlicingResult",
    "generate_voiceover",
    "VoiceoverResult",
    "build_audio_filtergraph",
    "render_standalone_audio",
    "build_visual_filtergraph",
    "build_micro_enhancements_filter",
    "generate_styled_ass_from_srt",
    "build_subtitle_filter_string",
    "calculate_compliance_score",
    "audit_file_metadata",
    "ComplianceReport",
    "TransformPipeline",
    "PipelineResult",
]
