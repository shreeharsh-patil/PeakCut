"""ClipMind Distribution Module.
Provides automated YouTube publishing, algorithmic timing and scheduling,
niche playlist management, and pinned engagement comments.
"""

from clipmind.distribution.youtube_uploader import (
    YouTubeUploader,
    compute_optimal_upload_window,
    upload_short,
    OptimalWindowResult,
)

__all__ = [
    "YouTubeUploader",
    "compute_optimal_upload_window",
    "upload_short",
    "OptimalWindowResult",
]
