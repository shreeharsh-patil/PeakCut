"""Configuration models and data classes for the Transformative Video Pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class LayoutMode(str, Enum):
    """Visual layout transformation modes."""
    SPLIT_SCREEN = "split_screen"        # 50/50 vertical commentary split
    PIP = "pip"                          # Picture-in-picture with reaction inset
    CANVAS_LETTERBOX = "canvas_letterbox"  # Educational card layout with header/footer cards
    BLUR_FILL = "blur_fill"              # 9:16 blurred background canvas with centered foreground


class TTSProvider(str, Enum):
    """Voiceover synthesis providers."""
    AUTO = "auto"          # Try edge-tts first; fallback to Windows native SAPI
    EDGE_TTS = "edge_tts"  # Microsoft Edge neural TTS
    SAPI = "sapi"          # Windows native SAPI SpVoice (offline, 0 latency)
    CUSTOM_AUDIO = "custom_audio"  # Pre-recorded audio file provided by user


class AudioDuckingMode(str, Enum):
    """Audio treatment modes."""
    SIDECHAIN_DUCK = "sidechain_duck"  # Sidechain compress source beneath voiceover
    MUTE_SOURCE = "mute_source"        # Full audio replacement (source completely muted)
    PARITY = "parity"                  # Source and voiceover mixed at standard levels


@dataclass
class AudioConfig:
    """Audio mixing, ducking, and synthesis configuration."""
    voiceover_provider: TTSProvider = TTSProvider.AUTO
    voice_name: str = "en-US-ChristopherNeural"
    custom_voiceover_path: Optional[str] = None
    voiceover_text: Optional[str] = None
    voiceover_volume: float = 1.0

    ducking_mode: AudioDuckingMode = AudioDuckingMode.SIDECHAIN_DUCK
    sidechain_threshold: float = 0.08
    sidechain_ratio: float = 10.0
    sidechain_attack_ms: float = 15.0
    sidechain_release_ms: float = 300.0
    source_attenuation_weight: float = 0.18  # Weight in amix (~ -18 to -24 dB)

    bgm_path: Optional[str] = None
    bgm_volume: float = 0.12

    # Micro-pitch / speed perturbations
    pitch_semitones: float = 0.0  # Optional slight pitch shift via rubberband
    tempo_factor: float = 1.0     # Audio playback tempo multiplier

    # Loudness normalization
    loudnorm_integrated: float = -14.0  # Target LUFS
    loudnorm_true_peak: float = -2.0    # dBTP ceiling
    loudnorm_lra: float = 11.0          # Loudness Range


@dataclass
class VisualConfig:
    """Video formatting, layout, and visual filter configuration."""
    layout_mode: LayoutMode = LayoutMode.SPLIT_SCREEN
    output_width: int = 1080
    output_height: int = 1920
    target_fps: int = 30
    secondary_video_path: Optional[str] = None  # Reaction / commentary footage

    # Split screen specifics
    divider_color: str = "white@0.85"
    divider_thickness: int = 4
    top_ratio: float = 0.50  # Split ratio (0.50 = 960x960 each)

    # PIP specifics
    pip_scale_ratio: float = 0.32  # Area occupied by inset (~25-35%)
    pip_position: str = "bottom_right"  # bottom_right, bottom_left, top_right

    # Canvas letterbox specifics
    title_text: Optional[str] = None
    subtitle_summary: Optional[str] = None
    canvas_bg_color: str = "0x111625"

    # Micro-adjustments (perceptual hash perturbation layer)
    enable_micro_adjustments: bool = True
    color_contrast: float = 1.04
    color_brightness: float = 0.01
    color_saturation: float = 1.05
    color_gamma: float = 1.02
    unsharp_amount: float = 0.60
    noise_strength: int = 5
    vignette_angle: str = "PI/5"

    # Encoding
    encoder: str = "libx264"  # libx264 or h264_nvenc
    preset: str = "fast"
    crf: int = 20


@dataclass
class TemporalConfig:
    """Temporal slicing and curation configuration."""
    max_chunk_duration: float = 5.0  # YouTube Content ID / Policy maximum unedited chunk rule
    enable_slicing: bool = True
    interleave_freeze_frame: bool = False
    freeze_frame_duration: float = 0.8
    segment_ranges: Optional[List[tuple[float, float]]] = None  # Explicit (start, end) ranges


@dataclass
class CaptionConfig:
    """Subtitle / Dynamic caption configuration."""
    burn_captions: bool = True
    font_name: str = "Arial"
    font_size: int = 24
    primary_color: str = "&H00FFFFFF"  # White in ASS format (&HAABBGGRR)
    outline_color: str = "&H00000000"  # Black outline
    highlight_color: str = "&H0000FFFF"  # Yellow highlight
    outline_width: int = 3
    shadow_depth: int = 1
    alignment: int = 2  # Bottom centered (ASS alignment 2)
    margin_v: int = 120  # Vertical margin from bottom


@dataclass
class ComplianceThresholds:
    """Thresholds for the Programmatic Compliance Scoring Metric."""
    target_voiceover_ratio: float = 0.70
    max_allowed_continuous_chunk: float = 5.0
    weight_voiceover: float = 0.35
    weight_layout: float = 0.25
    weight_temporal: float = 0.25
    weight_ducking: float = 0.15

    pass_threshold: float = 0.80
    warn_threshold: float = 0.60


@dataclass
class PipelineConfig:
    """Master pipeline execution configuration."""
    input_path: str
    output_path: str
    work_dir: Optional[str] = None

    audio: AudioConfig = field(default_factory=AudioConfig)
    visual: VisualConfig = field(default_factory=VisualConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    captions: CaptionConfig = field(default_factory=CaptionConfig)
    compliance: ComplianceThresholds = field(default_factory=ComplianceThresholds)

    strip_metadata: bool = True
    fast_start: bool = True
    enforce_compliance_pass: bool = False  # Fail export if score < 0.80

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)
