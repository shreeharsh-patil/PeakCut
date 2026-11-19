"""ClipMind SEO Subsystem: Metadata, Title Formula, Description Templates, and Tag Mining."""

from clipmind.seo.metadata_engine import (
    DEFAULT_TIMESTAMPS,
    HOOK_WORDS,
    CURIOSITY_GAPS,
    EMOJIS,
    MetadataEngine,
    VideoMetadata,
)

__all__ = [
    "MetadataEngine",
    "VideoMetadata",
    "HOOK_WORDS",
    "CURIOSITY_GAPS",
    "EMOJIS",
    "DEFAULT_TIMESTAMPS",
]
