"""ClipMind: Autonomous YouTube Shorts Generator & Uploader.

An end-to-end autonomous engine for trend analysis, content generation,
CTR-maximized thumbnail design, SEO optimization, safety compliance,
multi-key rotation, and scheduled publishing.
"""

__version__ = "1.0.0"
__author__ = "ClipMind Team"

from clipmind.core.config import get_config, Config
from clipmind.core.key_rotator import get_active_key, KeyRotator
from clipmind.core.logger import get_logger, CycleLogger

__all__ = [
    "__version__",
    "get_config",
    "Config",
    "get_active_key",
    "KeyRotator",
    "get_logger",
    "CycleLogger",
]
