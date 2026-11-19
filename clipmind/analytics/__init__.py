"""ClipMind Analytics & Self-Optimization Module.
Provides YouTube Analytics API querying, CTR/AVD/engagement tracking,
and the adaptive learning loop that self-optimizes titles, thumbnails, and timing.
"""

from clipmind.analytics.learning_loop import (
    LearningLoop,
    PerformanceMetric,
    AdaptiveConfig,
    run_learning_cycle,
)

__all__ = [
    "LearningLoop",
    "PerformanceMetric",
    "AdaptiveConfig",
    "run_learning_cycle",
]
