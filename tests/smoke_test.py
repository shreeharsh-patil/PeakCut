#!/usr/bin/env python3
"""ClipMind Smoke Test: End-to-End Runnable Dry-Run Verification.

Executes a full simulated cycle of ClipMind without requiring paid credentials
or live network calls. Validates that all 9 phases communicate seamlessly:
- Phase 1: Trend Mining & VPS Scoring (TrendAnalyzer)
- Phase 2: Multi-Key Quota Rotation & Health Pre-flight (KeyRotator)
- Phase 3: Vertical 9:16 Video Generation & Loudnorm (ClipPipeline)
- Phase 4: High-CTR A/B Thumbnail Creation (ThumbnailDesigner)
- Phase 5: Formula Title, Timestamps & SEO Tags (MetadataEngine)
- Phase 6: Algorithmic Timing & Scheduled Upload (YouTubeUploader)
- Phase 7: Local & Drive Asset Packaging Hierarchy (DriveManager)
- Phase 8: Performance Tracking & Self-Optimization (LearningLoop)
- Phase 9: Safety Moderation & Master Cycle Loop (MasterOrchestrator)

Usage:
    python tests/smoke_test.py
    pytest tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clipmind.core.config import Config, get_config
from clipmind.core.key_rotator import KeyRotator, get_key_rotator, mask_key
from clipmind.core.logger import get_logger
from clipmind.intelligence.trend_analyzer import TrendAnalyzer
from clipmind.seo.metadata_engine import MetadataEngine
from clipmind.generation.clip_pipeline import ClipPipeline, GenerationStrategy
from clipmind.creative.thumbnail_designer import ThumbnailDesigner
from clipmind.distribution.youtube_uploader import YouTubeUploader, compute_optimal_upload_window
from clipmind.storage.drive_manager import DriveManager
from clipmind.analytics.learning_loop import LearningLoop, PerformanceMetric
from clipmind.safety.moderator import SafetyModerator
from clipmind.orchestrator import MasterOrchestrator

logger = get_logger("clipmind.smoke_test")


def run_smoke_test() -> bool:
    """Execute end-to-end dry-run verification across all ClipMind subsystems."""
    print("=" * 80)
    print("   CLIPMIND AUTONOMOUS ENGINE - END-TO-END SMOKE TEST (DRY RUN)")
    print("=" * 80)
    start_total_time = time.time()

    # -------------------------------------------------------------------------
    # 0. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[STEP 0/9] Loading System Configuration in Dry-Run Mode...")
    os.environ["CLIPMIND_DRY_RUN"] = "true"
    config = get_config(reload=True)
    config.dry_run = True
    config.simulate_apis_if_missing = True
    config.paths.ensure_directories()
    print(f"  ✓ Niche: {config.niche}")
    print(f"  ✓ Audience Timezone: {config.audience_timezone}")
    print(f"  ✓ Upload Frequency: {config.upload_frequency_hours}h")
    print(f"  ✓ Output Directory: {config.paths.output_dir}")

    # -------------------------------------------------------------------------
    # 1. KeyRotator Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 1/9] Verifying Phase 2: KeyRotator & Pre-flight Quotas...")
    rotator = get_key_rotator()
    # Add simulation keys for all core services
    for service in ["youtube", "openai", "serpapi", "elevenlabs", "pexels"]:
        rotator.add_key(service, f"sim_{service}_smoke_key")
        active_key = rotator.get_active_key(service)
        valid, msg = rotator.verify_key(service, active_key)
        print(f"  ✓ Service '{service}': Key={mask_key(active_key)} | Valid={valid} ({msg})")
        assert active_key is not None, f"Key for {service} must not be None"
        assert valid is True, f"Key verification failed for {service}"

    # -------------------------------------------------------------------------
    # 2. TrendAnalyzer Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 2/9] Verifying Phase 1: Trend Mining & VPS Topic Scoring...")
    analyzer = TrendAnalyzer()
    top_topics = analyzer.analyze(niche=config.niche, top_n=3)
    assert len(top_topics) >= 1, "Must find at least 1 top topic"
    chosen_topic = top_topics[0]
    print(f"  ✓ Selected Top Topic: '{chosen_topic.title}'")
    print(f"  ✓ Viral Potential Score (VPS): {chosen_topic.vps_score:.2f}/100.0")
    print(f"    - Search Velocity: {chosen_topic.search_velocity:.1f}")
    print(f"    - Social Mentions: {chosen_topic.social_mentions:.1f}")
    print(f"    - Competitor Views (24h): {chosen_topic.competitor_views_24h:.1f}")
    print(f"    - Recency Score: {chosen_topic.recency_score:.1f}")
    assert 0.0 <= chosen_topic.vps_score <= 100.0

    # -------------------------------------------------------------------------
    # 3. SafetyModerator Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 3/9] Verifying Phase 9: Safety Moderation & Compliance...")
    moderator = SafetyModerator()
    safe_topic, vps, mod_result = moderator.select_safe_topic([chosen_topic.title])
    print(f"  ✓ Topic Compliance: is_safe={mod_result.is_safe} | Recommendation={mod_result.recommendation}")
    assert mod_result.is_safe is True, "Topic must pass moderation"

    # -------------------------------------------------------------------------
    # 4. MetadataEngine Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 4/9] Verifying Phase 5: SEO Metadata Generation...")
    metadata_engine = MetadataEngine()
    meta = metadata_engine.generate_metadata(
        topic=safe_topic,
        niche=config.niche,
        trending_tags=["ai", "future", "tech"],
    )
    print(f"  ✓ Formula Title: {meta.title} (Length: {len(meta.title)} chars)")
    print(f"  ✓ SEO Tags Count: {len(meta.tags)} | Total Chars: {len(meta.tags_string)}")
    print(f"  ✓ Hashtags: {' '.join(meta.hashtags)}")
    print(f"  ✓ Alt Text A: {meta.thumbnail_alt_a[:80]}...")
    print(f"  ✓ Alt Text B: {meta.thumbnail_alt_b[:80]}...")
    assert len(meta.title) <= 100, "Title must be <= 100 chars"
    assert len(meta.tags_string) <= 500, "Tags string must not exceed 500 chars"
    assert "00:00 Intro" in meta.description, "Must contain standard timestamps"

    # -------------------------------------------------------------------------
    # 5. ClipPipeline Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 5/9] Verifying Phase 3: Vertical Video Generation & FFmpeg...")
    pipeline = ClipPipeline(config=config)
    clip_result = pipeline.generate_clip(
        topic=safe_topic,
        source_type="simulation",
    )
    print(f"  ✓ Clip Status: Success={clip_result.video_path.exists()} | Strategy={clip_result.strategy_used}")
    print(f"  ✓ Video Path: {clip_result.video_path}")
    print(f"  ✓ Resolution: {clip_result.resolution[0]}x{clip_result.resolution[1]} (9:16)")
    print(f"  ✓ Duration: {clip_result.duration:.2f}s (Required: 45.0s - 58.0s)")
    assert clip_result.video_path.exists(), "Video file missing on disk"
    assert 45.0 <= clip_result.duration <= 58.0, "Duration out of bounds"
    assert clip_result.resolution == (1080, 1920), "Must be 1080x1920 vertical"

    # -------------------------------------------------------------------------
    # 6. ThumbnailDesigner Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 6/9] Verifying Phase 4: A/B Thumbnail Rendering...")
    thumbnail_designer = ThumbnailDesigner(config=config)
    thumb_result = thumbnail_designer.generate_thumbnails(
        topic=safe_topic,
    )
    print(f"  ✓ Variant A Path: {thumb_result.variant_a.image_path}")
    print(f"  ✓ Variant B Path: {thumb_result.variant_b.image_path}")
    print(f"  ✓ Variant A Palette: {thumb_result.variant_a.color_palette} | Est CTR: {thumb_result.variant_a.ctr_prediction_score}%")
    print(f"  ✓ Variant B Palette: {thumb_result.variant_b.color_palette} | Est CTR: {thumb_result.variant_b.ctr_prediction_score}%")
    assert Path(thumb_result.variant_a.image_path).exists(), "Thumbnail A file missing"
    assert Path(thumb_result.variant_b.image_path).exists(), "Thumbnail B file missing"

    # -------------------------------------------------------------------------
    # 7. YouTubeUploader Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 7/9] Verifying Phase 6: Algorithmic Timing & Scheduled Upload...")
    timing_res = compute_optimal_upload_window(target_region="US")
    print(f"  ✓ Optimal Now: {timing_res.is_optimal_now}")
    if not timing_res.is_optimal_now:
        print(f"  ✓ Next Scheduled Window: {timing_res.publish_at_iso}")

    uploader = YouTubeUploader(dry_run=True)
    upload_res = uploader.upload_short(
        video_path=clip_result.video_path,
        title=meta.title,
        description=meta.description,
        tags=meta.tags,
        auto_schedule=False,
    )
    print(f"  ✓ Upload Success: {upload_res.get('success')}")
    print(f"  ✓ Video ID: {upload_res.get('video_id')}")
    print(f"  ✓ Shorts URL: {upload_res.get('url')}")
    assert upload_res.get("success") is True
    assert "video_id" in upload_res

    # -------------------------------------------------------------------------
    # 8. DriveManager & Archival Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 8/9] Verifying Phase 7: Drive Hierarchy & Asset Packaging...")
    drive_mgr = DriveManager(dry_run=True)
    cycle_id = f"smoke_{int(time.time())}"
    package_res = drive_mgr.package_cycle_assets(
        cycle_id=cycle_id,
        assets={
            "final_clip.mp4": clip_result.video_path,
            "thumbnail_A.png": thumb_result.variant_a.image_path,
            "thumbnail_B.png": thumb_result.variant_b.image_path,
        },
        metadata_dict=meta.to_dict(),
        cycle_log_text="Smoke test cycle execution trace",
        clean_temp=False,
    )
    print(f"  ✓ Packaged Local Directory: {package_res['local_dir']}")
    print(f"  ✓ Drive Folder ID: {package_res['drive_folder_id']}")
    print(f"  ✓ Packaged Assets: {list(package_res['assets'].keys())}")
    assert Path(package_res['local_dir']).exists(), "Cycle local folder missing"

    # Update performance tracker
    drive_mgr.update_performance_tracker({
        "cycle_id": cycle_id,
        "topic": safe_topic,
        "title": meta.title,
        "video_id": upload_res.get("video_id", "sim_video_001"),
        "status": "completed",
        "views": 1250,
        "ctr": 9.4,
        "avd_pct": 82.5,
        "engagement_rate": 7.1,
    })
    print("  ✓ Appended cycle record to performance_tracker.csv")

    # -------------------------------------------------------------------------
    # 9. LearningLoop Verification
    # -------------------------------------------------------------------------
    print("\n[STEP 9/9] Verifying Phase 8: Adaptive Learning & Feedback Loop...")
    loop = LearningLoop(dry_run=True)
    metric = loop.fetch_video_metrics(
        video_id=upload_res.get("video_id", "sim_video_001"),
        cycle_id=cycle_id,
        title=meta.title,
        topic=safe_topic,
    )
    loop.record_cycle_performance(metric)
    adaptive_cfg = loop.run_adaptive_learning()
    print(f"  ✓ Winning Thumbnail Variant: {adaptive_cfg.winning_thumbnail_variant}")
    print(f"  ✓ Winning Palette: {adaptive_cfg.recommended_thumbnail_palette}")
    print(f"  ✓ Optimal Upload Hour: {adaptive_cfg.optimal_upload_hour_utc}:00 UTC")
    print(f"  ✓ Recommended Hooks: {adaptive_cfg.recommended_hook_words}")

    # -------------------------------------------------------------------------
    # 10. MasterOrchestrator Complete Cycle Run
    # -------------------------------------------------------------------------
    print("\n[BONUS] Executing MasterOrchestrator.run_cycle()...")
    orchestrator = MasterOrchestrator(config=config)
    cycle_summary = orchestrator.run_cycle()
    print(f"  ✓ Master Cycle Result: {cycle_summary['status']}")
    print(f"  ✓ Master Cycle ID: {cycle_summary['cycle_id']}")
    print(f"  ✓ Master Video ID: {cycle_summary['video_id']}")
    assert cycle_summary["status"] == "COMPLETED", "Master cycle failed"

    elapsed = time.time() - start_total_time
    print("\n" + "=" * 80)
    print(f"🎉 CLIPMIND SMOKE TEST PASSED CLEANLY IN {elapsed:.2f}s!")
    print("   All 9 Phases fully operational and verified.")
    print("=" * 80)
    return True


def test_smoke_test():
    """Pytest hook for running smoke test within the pytest suite."""
    assert run_smoke_test() is True


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
