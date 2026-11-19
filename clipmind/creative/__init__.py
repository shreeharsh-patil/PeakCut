"""ClipMind Creative & CTR Maximization Engine (Phase 4).

Generates high-CTR A/B test thumbnail variants with:
- Bold reaction elements (stickers/emojis/badges)
- 3-5 word punchy text overlays in Anton/Impact typography
- High-contrast neon/yellow/red styling with heavy outlines and drop shadows
- Attention arrows/indicators pointing to focal points
- Watermark integration
- AI background prompt generation and Pillow-based rendering engine
"""

from clipmind.creative.thumbnail_designer import (
    ThumbnailDesigner,
    ThumbnailVariant,
    ThumbnailResult,
    generate_thumbnails,
)

__all__ = [
    "ThumbnailDesigner",
    "ThumbnailVariant",
    "ThumbnailResult",
    "generate_thumbnails",
]
