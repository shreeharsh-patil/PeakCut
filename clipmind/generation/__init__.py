"""ClipMind Video Generation Engine (Phase 3).

Provides the multi-priority video production pipeline:
- Priority 1 (YouTube source): Download viral long-form videos, extract peak-engagement 45-58s segment with 0-3s hook.
- Priority 2 (Text-to-Video): AI narration script + voiceover + b-roll.
- Priority 3 (Slideshow fallback): Image slideshow + ElevenLabs/TTS voiceover.
- FFmpeg vertical 9:16 reframing, loudnorm -14 LUFS, bold burned subtitles, 45-58s duration enforcement.
"""

from clipmind.generation.clip_pipeline import (
    ClipPipeline,
    ClipResult,
    GenerationStrategy,
    VideoCandidate,
    generate_clip,
)

__all__ = [
    "ClipPipeline",
    "ClipResult",
    "GenerationStrategy",
    "VideoCandidate",
    "generate_clip",
]
