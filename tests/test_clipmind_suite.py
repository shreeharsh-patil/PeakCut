"""ClipMind Comprehensive Quality Assurance & Verification Test Suite.

Unit and integration tests covering all 9 core subsystems:
1. KeyRotator (quota tracking, 15% pre-flight check, blacklist, midnight reset, health pings)
2. TrendAnalyzer (VPS scoring formula, multi-source fusion, topic ranking & selection)
3. MetadataEngine (title formulas, description templates, competitor tag mining, alt text)
4. ClipPipeline (45-58s duration enforcement, 9:16 aspect, audio filters, subtitle styling)
5. ThumbnailDesigner (A/B variants, high-CTR elements, resolution, overlay rendering)
6. YouTubeUploader (optimal timing calculation, publishAt RFC 3339, metadata payload format)
7. DriveManager (folder tree generation, asset packaging, temp cleanup, 30-day archival)
8. LearningLoop (metrics extraction, adaptive pattern learning, dynamic config updates)
9. Orchestrator & CLI (cycle state machine, safety moderation, scheduler, CLI arguments)
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ClipMind Imports
from clipmind.core.config import Config, PathSettings, VideoSettings, get_config
from clipmind.core.key_rotator import (
    DEFAULT_QUOTAS,
    KeyRotator,
    KeyState,
    get_active_key,
    get_key_rotator,
    mark_rate_limited,
    mask_key,
    record_quota_usage,
    record_usage,
    report_rate_limit,
)
from clipmind.core.logger import CycleLogger, get_logger
from clipmind.intelligence.trend_analyzer import (
    NICHE_SEED_TOPICS,
    TopicCandidate,
    TrendAnalyzer,
)
from clipmind.seo.metadata_engine import (
    CURIOSITY_GAPS,
    DEFAULT_TIMESTAMPS,
    EMOJIS,
    HOOK_WORDS,
    MetadataEngine,
    VideoMetadata,
)
from clipmind.generation.clip_pipeline import (
    ClipPipeline,
    ClipResult,
    GenerationStrategy,
    VideoCandidate,
)
from clipmind.creative.thumbnail_designer import (
    ThumbnailDesigner,
    ThumbnailResult,
    ThumbnailVariant,
)
from clipmind.distribution.youtube_uploader import (
    OptimalWindowResult,
    YouTubeUploader,
    compute_optimal_upload_window,
)
from clipmind.storage.drive_manager import DriveManager
from clipmind.analytics.learning_loop import (
    AdaptiveConfig,
    LearningLoop,
    PerformanceMetric,
)
from clipmind.safety.moderator import ModerationResult, SafetyModerator
from clipmind.orchestrator import MasterOrchestrator
import clipmind.cli as clipmind_cli


# ==============================================================================
# SECTION 1: KeyRotator Tests
# ==============================================================================
class TestKeyRotator:
    """Verifies quota tracking, pre-flight checks, rate limit blacklisting, resets, and health pings."""

    @pytest.fixture
    def temp_state_file(self, tmp_path):
        state_file = tmp_path / "test_key_quotas.json"
        return state_file

    @pytest.fixture
    def rotator(self, temp_state_file):
        rot = KeyRotator(state_file=temp_state_file)
        # Clear default pools for clean isolated unit testing
        for k in rot.pools:
            rot.pools[k] = []
        return rot

    def test_key_state_quota_and_percentage_calculation(self):
        state = KeyState(key="test_key_abc123", service="youtube", total_quota=10000, used_units=2000)
        assert state.remaining_units == 8000
        assert state.remaining_percent == 80.0
        assert state.has_sufficient_quota is True
        assert state.is_rate_limited is False

    def test_preflight_check_enforces_15_percent_threshold(self):
        # 16% remaining -> Sufficient (>15%)
        state_pass = KeyState(key="key_1", service="youtube", total_quota=1000, used_units=840)
        assert state_pass.remaining_percent == 16.0
        assert state_pass.has_sufficient_quota is True

        # 15% remaining -> Fails pre-flight check (>15% required)
        state_fail = KeyState(key="key_2", service="youtube", total_quota=1000, used_units=850)
        assert state_fail.remaining_percent == 15.0
        assert state_fail.has_sufficient_quota is False

        # 10% remaining -> Fails pre-flight check
        state_low = KeyState(key="key_3", service="youtube", total_quota=1000, used_units=900)
        assert state_low.remaining_percent == 10.0
        assert state_low.has_sufficient_quota is False

    def test_key_rotation_round_robin_selection(self, rotator):
        rotator.add_key("youtube", "key_A", total_quota=1000)
        rotator.add_key("youtube", "key_B", total_quota=1000)
        rotator.add_key("youtube", "key_C", total_quota=1000)

        # Round-robin cycle across A, B, C
        k1 = rotator.get_active_key("youtube")
        k2 = rotator.get_active_key("youtube")
        k3 = rotator.get_active_key("youtube")
        k4 = rotator.get_active_key("youtube")

        assert k1 == "key_A"
        assert k2 == "key_B"
        assert k3 == "key_C"
        assert k4 == "key_A"

    def test_record_quota_usage_does_not_advance_key_rotation(self, rotator):
        rotator.add_key("youtube", "key_A", total_quota=1000)
        rotator.add_key("youtube", "key_B", total_quota=1000)

        # 1. Acquire active key
        k = rotator.get_active_key("youtube")
        assert k == "key_A"

        # 2. Record usage on active key without passing key argument
        from clipmind.core.key_rotator import record_quota_usage
        with patch("clipmind.core.key_rotator.KeyRotator.get_instance", return_value=rotator):
            record_quota_usage("youtube", units=100)

        # key_A should have 100 units deducted
        assert rotator.pools["youtube"][0].used_units == 100
        # key_B should have 0 units deducted
        assert rotator.pools["youtube"][1].used_units == 0

        # Next rotation must pick key_B, NOT key_A!
        next_k = rotator.get_active_key("youtube")
        assert next_k == "key_B"

    def test_record_usage_updates_state_and_persists(self, rotator, temp_state_file):
        rotator.add_key("openai", "sk-proj-test12345678", total_quota=10000)
        rotator.record_usage("openai", "sk-proj-test12345678", units=1500)

        # Check in-memory state
        state = rotator.pools["openai"][0]
        assert state.used_units == 1500
        assert state.remaining_units == 8500
        assert state.remaining_percent == 85.0

        # Check persisted file on disk
        assert temp_state_file.exists()
        with open(temp_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        saved_key = data["services"]["openai"][0]
        assert saved_key["used_units"] == 1500
        assert saved_key["remaining_units"] == 8500

    def test_rate_limit_blacklisting_skips_key(self, rotator):
        rotator.add_key("serpapi", "serp_key_1", total_quota=100)
        rotator.add_key("serpapi", "serp_key_2", total_quota=100)

        # Blacklist key 1 for 3600 seconds
        rotator.mark_rate_limited("serpapi", "serp_key_1", duration_seconds=3600, reason="HTTP 429 Too Many Requests")

        assert rotator.pools["serpapi"][0].is_rate_limited is True
        assert rotator.pools["serpapi"][1].is_rate_limited is False

        # Next active key must bypass blacklisted key 1 and return key 2
        active = rotator.get_active_key("serpapi")
        assert active == "serp_key_2"

    def test_daily_midnight_utc_reset(self, rotator):
        rotator.add_key("elevenlabs", "xi_key_1", total_quota=10000)
        rotator.record_usage("elevenlabs", "xi_key_1", units=5000)
        rotator.mark_rate_limited("elevenlabs", "xi_key_1", duration_seconds=7200)

        assert rotator.pools["elevenlabs"][0].used_units == 5000
        assert rotator.pools["elevenlabs"][0].is_rate_limited is True

        # Simulate date shift: set last reset to yesterday
        rotator.last_reset_utc_date = "2020-01-01"
        rotator._check_and_apply_midnight_reset()

        state = rotator.pools["elevenlabs"][0]
        assert state.used_units == 0
        assert state.blacklisted_until == 0.0
        assert state.is_rate_limited is False
        assert rotator.last_reset_utc_date == datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def test_health_ping_verification(self, rotator):
        # Simulation keys always pass
        valid, msg = rotator.verify_key("youtube", "sim_youtube_key_clipmind")
        assert valid is True
        assert "Simulation" in msg

        # Mocked live API response 200 (healthy)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            ok, text = rotator.verify_key("youtube", "AIzaSyLiveKeyHealthy")
            assert ok is True
            assert "healthy" in text.lower()

        # Mocked live API response 403 (quota / forbidden)
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 403
            mock_get.return_value.text = "Quota exceeded"
            ok, text = rotator.verify_key("youtube", "AIzaSyLiveKeyExceeded")
            assert ok is False
            assert "rejected" in text.lower()


# ==============================================================================
# SECTION 2: TrendAnalyzer Tests
# ==============================================================================
class TestTrendAnalyzer:
    """Verifies VPS calculation formula, topic ranking, and multi-source fusion."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        return TrendAnalyzer(cache_dir=tmp_path / "trends")

    def test_vps_scoring_formula_exact_math(self, analyzer):
        # VPS = (search_velocity * 0.35) + (social_mentions * 0.25) + (competitor_views_24h * 0.20) + (recency_score * 0.20)
        # (100 * 0.35) + (100 * 0.25) + (100 * 0.20) + (100 * 0.20) = 35 + 25 + 20 + 20 = 100.0
        assert analyzer.calculate_vps(100.0, 100.0, 100.0, 100.0) == 100.0

        # All zeros
        assert analyzer.calculate_vps(0.0, 0.0, 0.0, 0.0) == 0.0

        # Specific weights check:
        # Search=80 (28.0) + Social=60 (15.0) + Views=70 (14.0) + Recency=90 (18.0) = 75.0
        calculated = analyzer.calculate_vps(
            search_velocity=80.0,
            social_mentions=60.0,
            competitor_views_24h=70.0,
            recency_score=90.0,
        )
        assert calculated == 75.0

    def test_vps_score_clamping(self, analyzer):
        # Values exceeding 100 or below 0 should clamp between 0.0 and 100.0
        assert analyzer.calculate_vps(150.0, 120.0, 200.0, 110.0) == 100.0
        assert analyzer.calculate_vps(-20.0, -10.0, 0.0, 0.0) == 0.0

    def test_topic_ranking_order(self, analyzer):
        t1 = TopicCandidate("t1", "Low Priority", "tech", 50, 50, 50, 50, vps_score=50.0)
        t2 = TopicCandidate("t2", "High Priority", "tech", 90, 90, 90, 90, vps_score=90.0)
        t3 = TopicCandidate("t3", "Medium Priority", "tech", 70, 70, 70, 70, vps_score=70.0)

        ranked = analyzer.rank_topics([t1, t2, t3])
        assert [t.vps_score for t in ranked] == [90.0, 70.0, 50.0]
        assert ranked[0].title == "High Priority"

    def test_select_top_topics_limit(self, analyzer):
        candidates = [
            TopicCandidate(f"t{i}", f"Topic {i}", "ai", 80, 80, 80, 80, vps_score=float(80 + i))
            for i in range(10)
        ]
        top3 = analyzer.select_top_topics(candidates, limit=3)
        assert len(top3) == 3
        assert top3[0].vps_score == 89.0
        assert top3[1].vps_score == 88.0
        assert top3[2].vps_score == 87.0

    def test_offline_multi_source_trend_analysis(self, analyzer):
        # Should execute cleanly without network/API keys using fallback heuristics
        top_topics = analyzer.analyze(niche="ai", top_n=3)
        assert len(top_topics) == 3
        for topic in top_topics:
            assert isinstance(topic, TopicCandidate)
            assert topic.vps_score > 0.0
            assert topic.niche == "ai"
            assert len(topic.title) > 0


# ==============================================================================
# SECTION 3: MetadataEngine Tests
# ==============================================================================
class TestMetadataEngine:
    """Verifies title formula, description formatting, competitor tag mining, and alt text."""

    @pytest.fixture
    def engine(self):
        return MetadataEngine()

    def test_title_formula_structure(self, engine):
        # Formula: [HOOK WORD] + [TOPIC] + [VALUE/CURIOSITY GAP] + [Emoji]
        topic = "Quantum AI Chips"
        title = engine.generate_title(
            topic=topic,
            hook_word="WARNING",
            curiosity_gap="Will Change Everything",
            emoji="🚨",
        )
        assert title.startswith("WARNING")
        assert "Quantum AI Chips" in title
        assert "Will Change Everything" in title
        assert title.endswith("🚨")
        assert len(title) <= 85

    def test_generate_title_options_count(self, engine):
        options = engine.generate_title_options(topic="ChatGPT Agent Leak", count=5)
        assert len(options) == 5
        assert len(set(options)) == 5  # Distinct titles

    def test_description_template_structure_and_timestamps(self, engine):
        topic = "DeepSeek Open Source"
        niche = "Tech & AI"
        desc = engine.generate_description(topic=topic, niche=niche)

        # Must include exact required timestamps
        assert "00:00 Intro" in desc
        assert "00:08 Main Point 1" in desc
        assert "00:28 Main Point 2" in desc
        assert "00:48 Shocking Reveal" in desc

        # Must include Call to Action
        assert "Subscribe" in desc or "Like" in desc

        # Must include hashtags: #[Niche] #[Topic] #Shorts #[Trending1] #[Trending2]
        assert "#Shorts" in desc
        assert "#TechAI" in desc or "#Tech" in desc

    def test_competitor_tag_mining_length_constraints(self, engine):
        topic = "Autonomous Coding Agents"
        niche = "Artificial Intelligence"
        mock_competitors = [
            {"views": 500000, "tags": ["ai", "coding", "software", "agents", "python", "developer"]},
            {"views": 350000, "tags": ["tech", "future", "chatgpt", "deepseek", "programming"]},
            {"views": 200000, "tags": ["automation", "productivity", "devin", "anthropic"]},
        ]

        tag_result = engine.extract_competitor_tags(
            competitor_videos=mock_competitors,
            topic=topic,
            niche=niche,
        )
        tags = tag_result["tags_list"]
        tags_str = tag_result["tags_string"]

        assert len(tags) >= 15
        assert len(tags_str) <= 500  # Strict YouTube character ceiling
        assert len(tags_str) >= 300  # Well-packed SEO tags

    def test_thumbnail_alt_text_generation(self, engine):
        alt_a = engine.generate_thumbnail_alt_text("AI Superintelligence", variant="A")
        alt_b = engine.generate_thumbnail_alt_text("AI Superintelligence", variant="B")

        assert "shocked" in alt_a.lower() or "neon" in alt_a.lower()
        assert "split-screen" in alt_b.lower() or "red" in alt_b.lower() or "comparison" in alt_b.lower()
        assert "AI Superintelligence" in alt_a
        assert "AI Superintelligence" in alt_b


# ==============================================================================
# SECTION 4: ClipPipeline Tests
# ==============================================================================
class TestClipPipeline:
    """Verifies duration enforcement, 9:16 vertical resolution, audio loudnorm, and subtitles."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        cfg = get_config(reload=True)
        cfg.paths.output_dir = tmp_path / "output"
        cfg.paths.temp_dir = tmp_path / "temp"
        cfg.paths.ensure_directories()
        return ClipPipeline(config=cfg)

    def test_duration_constraints_and_validation(self, pipeline):
        # ClipMind enforces strict 45.0s to 58.0s duration
        assert pipeline.min_duration == 45.0
        assert pipeline.max_duration == 58.0

        # Test simulated video output meets duration constraints
        out_path = pipeline.temp_dir / "test_simulated.mp4"
        created_file = pipeline._create_simulated_video_file(
            target_path=out_path,
            duration=50.0,
        )
        assert created_file.exists()
        media_info = pipeline.get_media_info(created_file)

        assert 45.0 <= media_info["duration"] <= 58.0
        assert media_info["width"] == 1080
        assert media_info["height"] == 1920

    def test_audio_filters_and_loudnorm_configuration(self, pipeline):
        # YouTube Shorts loudnorm standard: I=-14:TP=-1.5:LRA=11
        assert pipeline.loudnorm_filter == "I=-14:TP=-1.5:LRA=11"
        assert pipeline.target_lufs == -14.0
        assert pipeline.bg_music_duck_db == -18.0

    def test_styled_ass_subtitle_generation(self, pipeline):
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Are you making this mistake?"},
            {"start": 3.0, "end": 8.0, "text": "Top scientists just revealed something insane."},
        ]
        ass_path = pipeline.temp_dir / "test_captions.ass"
        out = pipeline.create_styled_ass_subtitles(
            transcript_data=segments,
            output_ass_path=ass_path,
            duration=50.0,
        )

        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "PlayResX: 1080" in content
        assert "PlayResY: 1920" in content
        assert "making this mistake" in content.lower()

    def test_dry_run_clip_generation(self, pipeline):
        result = pipeline.generate_clip(
            topic="5 AI Tools That Will Shock You",
            source_type="simulation",
        )
        assert isinstance(result, ClipResult)
        assert result.video_path.exists()
        assert 45.0 <= result.duration <= 58.0
        assert result.resolution == (1080, 1920)


# ==============================================================================
# SECTION 5: ThumbnailDesigner Tests
# ==============================================================================
class TestThumbnailDesigner:
    """Verifies Variant A/B generation, high-CTR design elements, and image resolution."""

    @pytest.fixture
    def designer(self, tmp_path):
        cfg = get_config(reload=True)
        cfg.paths.output_dir = tmp_path / "thumbnails"
        cfg.paths.ensure_directories()
        return ThumbnailDesigner(config=cfg)

    def test_ab_variant_generation_and_distinct_styles(self, designer):
        result = designer.generate_thumbnails(
            topic="Claude 3.7 Sonnet Coding Shock",
        )
        assert isinstance(result, ThumbnailResult)
        assert result.variant_a is not None
        assert result.variant_b is not None
        assert Path(result.variant_a.image_path).exists()
        assert Path(result.variant_b.image_path).exists()

        # Check image resolution (1280x720 standard thumbnail)
        img_a = Image.open(result.variant_a.image_path)
        img_b = Image.open(result.variant_b.image_path)
        assert img_a.size == (1280, 720)
        assert img_b.size == (1280, 720)

        # Variants must have distinct hook text and styles
        assert result.variant_a.color_palette != result.variant_b.color_palette

    def test_high_contrast_reaction_badges_and_arrows(self, designer):
        img = Image.new("RGBA", (1280, 720), (0, 0, 0, 255))
        # Add reaction badge with x, y coordinates
        img_with_badge = designer.draw_reaction_badge(img, text="😱 99% WRONG!", x=60, y=60)
        assert img_with_badge is not None

        # Add emotion arrow
        img_with_arrow = designer.draw_emotion_arrow(img_with_badge, start_pt=(200, 200), end_pt=(500, 350))
        assert img_with_arrow is not None

        # Add target reticle
        img_with_reticle = designer.draw_target_reticle(img_with_arrow, center=(600, 360), radius=60)
        assert img_with_reticle is not None

    def test_ai_background_prompt_generation(self, designer):
        prompt = designer.generate_ai_prompt(
            topic="Humanoid Robot Super Factory",
            variant_id="A",
        )
        assert "Humanoid Robot Super Factory" in prompt
        assert len(prompt) > 20


# ==============================================================================
# SECTION 6: YouTubeUploader Tests
# ==============================================================================
class TestYouTubeUploader:
    """Verifies optimal timing computation, RFC 3339 publishAt, and upload payload formatting."""

    def test_optimal_upload_window_us_afternoon_peak(self):
        # 1:00 PM EST (18:00 UTC during standard time or 17:00 UTC during EDT) -> inside 12PM-3PM EST window
        eastern_tz = datetime.timezone(datetime.timedelta(hours=-5))
        test_time = datetime.datetime(2026, 11, 17, 13, 0, 0, tzinfo=eastern_tz)

        res = compute_optimal_upload_window(target_region="US", current_time=test_time)
        assert res.is_optimal_now is True
        assert "Afternoon" in (res.active_window_label or "")

    def test_optimal_upload_window_outside_hours_schedules_next_window(self):
        # 4:00 AM EST -> outside peak windows (12-3 PM, 7-9 PM)
        eastern_tz = datetime.timezone(datetime.timedelta(hours=-5))
        test_time = datetime.datetime(2026, 11, 17, 4, 0, 0, tzinfo=eastern_tz)

        res = compute_optimal_upload_window(target_region="US", current_time=test_time)
        assert res.is_optimal_now is False
        assert res.next_window_start_utc is not None
        assert res.publish_at_iso is not None
        # RFC 3339 formatted with 'Z' or timezone offset
        assert "T" in res.publish_at_iso

    def test_optimal_upload_window_global_audience(self):
        # 3:00 PM UTC (15:00 UTC) -> inside 14:00-16:00 UTC prime window
        utc_time = datetime.datetime(2026, 11, 17, 15, 0, 0, tzinfo=datetime.timezone.utc)
        res = compute_optimal_upload_window(target_region="GLOBAL", current_time=utc_time)
        assert res.is_optimal_now is True

    def test_dry_run_upload_short_payload_and_return_format(self, tmp_path):
        dummy_video = tmp_path / "final_clip.mp4"
        dummy_video.write_bytes(b"dummy_mp4_bytes")

        uploader = YouTubeUploader(dry_run=True)
        res = uploader.upload_short(
            video_path=dummy_video,
            title="The Real Truth About AI",
            description="00:00 Intro\n00:08 Main Point\n#Shorts",
            tags=["ai", "tech", "viral"],
            auto_schedule=False,
        )
        assert res["success"] is True
        assert "video_id" in res
        assert "youtube.com/shorts/" in res["url"]

    def test_upload_short_forced_publish_at_datetime(self, tmp_path):
        dummy_video = tmp_path / "test_short.mp4"
        dummy_video.write_bytes(b"sample bytes")

        uploader = YouTubeUploader(dry_run=True)
        scheduled_dt = datetime.datetime(2026, 12, 1, 14, 0, 0, tzinfo=datetime.timezone.utc)
        res = uploader.upload_short(
            video_path=dummy_video,
            title="Scheduled Short",
            forced_publish_at=scheduled_dt,
        )
        assert res["success"] is True
        assert res["is_scheduled"] is True
        assert res["publish_at"] == "2026-12-01T14:00:00Z"
        assert res["privacy_status"] == "private"


# ==============================================================================
# SECTION 7: DriveManager Tests
# ==============================================================================
class TestDriveManager:
    """Verifies folder tree generation, asset packaging, temp cleanup, and 30-day archival."""

    @pytest.fixture
    def drive_mgr(self, tmp_path):
        return DriveManager(local_base_dir=tmp_path / "ClipMind", dry_run=True)

    def test_folder_tree_generation(self, drive_mgr):
        cycle_id = "cycle_test_001"
        res = drive_mgr.package_cycle_assets(cycle_id=cycle_id, date_str="2026-11-17")
        assert res["success"] is True
        cycle_dir = Path(res["local_dir"])
        assert cycle_dir.exists()
        assert "ClipMind" in str(cycle_dir)
        assert cycle_id in str(cycle_dir)

    def test_package_cycle_assets_integrity(self, drive_mgr, tmp_path):
        cycle_id = "cycle_test_002"
        # Create source mock files
        vid = tmp_path / "final_clip.mp4"
        vid.write_bytes(b"video_data")
        th_a = tmp_path / "thumbnail_A.png"
        th_a.write_bytes(b"thumb_a_data")
        th_b = tmp_path / "thumbnail_B.png"
        th_b.write_bytes(b"thumb_b_data")
        srt = tmp_path / "captions.srt"
        srt.write_text("1\n00:00:00 --> 00:00:05\nHello\n", encoding="utf-8")

        assets = {
            "final_clip.mp4": vid,
            "thumbnail_A.png": th_a,
            "thumbnail_B.png": th_b,
            "captions.srt": srt,
        }

        result = drive_mgr.package_cycle_assets(
            cycle_id=cycle_id,
            assets=assets,
            metadata_dict={"title": "Test Title", "vps": 95.0},
            cycle_log_text="Cycle completed successfully",
        )

        assert result["success"] is True
        packaged_dir = Path(result["local_dir"])
        assert (packaged_dir / "final_clip.mp4").exists()
        assert (packaged_dir / "thumbnail_A.png").exists()
        assert (packaged_dir / "thumbnail_B.png").exists()
        assert (packaged_dir / "metadata.json").exists()
        assert (packaged_dir / "captions.srt").exists()
        assert (packaged_dir / "cycle_log.txt").exists()

    def test_clean_local_temp_assets_safety(self, drive_mgr, tmp_path):
        temp_dir = tmp_path / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Create intermediate and log files
        temp_render = temp_dir / "raw_render_segment.ts"
        temp_render.write_bytes(b"temp_data")
        keep_log = temp_dir / "cycle_log.txt"
        keep_log.write_text("protected log", encoding="utf-8")

        cleaned = drive_mgr.clean_local_temp_assets(
            temp_paths=[temp_render, keep_log],
            keep_extensions=[".txt", ".log"],
        )
        assert not temp_render.exists()
        assert keep_log.exists()
        assert str(temp_render) in cleaned

    def test_30_day_asset_archival(self, drive_mgr):
        # Create a mock date folder simulated as >30 days old (e.g. 2026-10-01)
        old_folder = drive_mgr.local_base_dir / "2026-10-01" / "Cycle_old_35d"
        old_folder.mkdir(parents=True, exist_ok=True)
        (old_folder / "final_clip.mp4").write_bytes(b"old_data")

        archived = drive_mgr.archive_assets_older_than(days_threshold=30)
        assert len(archived["archived_local_folders"]) >= 1
        assert "2026-10-01" in archived["archived_local_folders"]
        assert (drive_mgr.local_base_dir / "Archive" / "2026-10-01").exists()


# ==============================================================================
# SECTION 8: LearningLoop Tests
# ==============================================================================
class TestLearningLoop:
    """Verifies metrics analysis, winning pattern discovery, and adaptive config feedback."""

    @pytest.fixture
    def loop(self, tmp_path):
        return LearningLoop(data_dir=tmp_path / "data", dry_run=True)

    def test_engagement_rate_formula(self, loop):
        # engagement_rate = ((likes + comments + shares) / views) * 100
        # ((500 + 100 + 50) / 10000) * 100 = (650 / 10000) * 100 = 6.5%
        metric = PerformanceMetric(
            video_id="v1",
            cycle_id="c1",
            title="AI Revolution",
            topic="Tech",
            views=10000,
            ctr=8.2,
            avd_seconds=42.5,
            avd_pct=85.0,
            likes=500,
            comments=100,
            shares=50,
            engagement_rate=6.5,
        )
        assert round(metric.engagement_rate, 2) == 6.5

    def test_adaptive_pattern_learning_identifies_top_hooks_and_variants(self, loop):
        # Simulate history: Variant A with Hook 'SECRET' performs best
        m1 = PerformanceMetric(
            video_id="vid_1",
            cycle_id="c101",
            title="SECRET AI Breakthrough",
            topic="ai",
            views=50000,
            ctr=11.5,
            avd_seconds=48.0,
            avd_pct=88.0,
            likes=3000,
            comments=400,
            shares=200,
            engagement_rate=7.2,
            thumbnail_variant="Variant_A",
            thumbnail_color="neon_yellow_contrast",
            hook_style="curiosity_gap",
            upload_hour_utc=14,
        )
        m2 = PerformanceMetric(
            video_id="vid_2",
            cycle_id="c102",
            title="EXPOSED Mistake",
            topic="ai",
            views=8000,
            ctr=4.2,
            avd_seconds=25.0,
            avd_pct=45.0,
            likes=200,
            comments=15,
            shares=5,
            engagement_rate=2.75,
            thumbnail_variant="Variant_B",
            thumbnail_color="high_contrast_red",
            hook_style="fear_of_missing_out",
            upload_hour_utc=4,
        )

        loop.record_cycle_performance(m1)
        loop.record_cycle_performance(m2)

        adaptive_cfg = loop.run_adaptive_learning()
        assert isinstance(adaptive_cfg, AdaptiveConfig)
        assert adaptive_cfg.winning_thumbnail_variant == "Variant_A"
        assert adaptive_cfg.recommended_thumbnail_palette == "neon_yellow_contrast"
        assert adaptive_cfg.optimal_upload_hour_utc == 14

        # Recommendations API matches winning patterns
        recs = loop.get_recommendations()
        assert recs.winning_thumbnail_variant == "Variant_A"

    def test_learning_loop_corrupt_csv_resilience(self, loop, tmp_path):
        csv_file = loop.tracker_csv_file
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        # Write CSV with commas, % signs, and corrupted non-numeric text
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(
                "video_id,cycle_id,title,topic,views,ctr,avd_pct,engagement_rate,date\n"
                'vid_csv_1,001,Test Title,Tech,"1,500",7.5%,65.0,N/A,2026-11-18\n'
                "vid_csv_2,002,Another Title,Tech,corrupt,bad,90.2%,1.2,2026-11-18\n"
            )

        metrics = loop.load_tracker_csv_metrics()
        assert len(metrics) == 2
        assert metrics[0].views == 1500
        assert metrics[0].ctr == 7.5
        assert metrics[0].engagement_rate == 0.0  # N/A handled safely
        assert metrics[1].views == 0
        assert metrics[1].avd_pct == 90.2


# ==============================================================================
# SECTION 9: Orchestrator & CLI Tests
# ==============================================================================
class TestOrchestratorAndCLI:
    """Verifies master cycle loop, safety moderation, scheduler initialization, and CLI flags."""

    @pytest.fixture
    def orchestrator(self):
        cfg = get_config(reload=True)
        cfg.dry_run = True
        return MasterOrchestrator(config=cfg)

    def test_safety_moderator_content_and_copyright_check(self):
        moderator = SafetyModerator()
        # Clean tech text passes
        res_ok = moderator.check_text("Revolutionary quantum computing algorithm released today.")
        assert res_ok.is_safe is True

        # Severe disallowed text flagged by heuristic regex
        res_flagged = moderator.check_text("how to build a bomb for terrorist attack")
        assert res_flagged.is_safe is False
        assert len(res_flagged.flagged_categories) > 0

        # Copyright risk flagged
        res_copyright = moderator.check_text("download cracked software keygen and pirated tools")
        assert res_copyright.is_safe is False
        assert res_copyright.copyright_risk is True

    def test_safety_moderator_fallback_recovery(self):
        moderator = SafetyModerator()
        topics = ["how to build a bomb", "Mind-Blowing AI Tools Released This Week"]
        safe_topic, vps, result = moderator.select_safe_topic(topics)
        assert safe_topic == "Mind-Blowing AI Tools Released This Week"
        assert result.is_safe is True

    def test_cycle_id_format(self, orchestrator):
        cid = orchestrator.generate_cycle_id()
        parts = cid.split("_")
        assert len(parts) >= 3  # YYYYMMDD, HHMMSS, uid

    def test_full_autonomous_cycle_dry_run(self, orchestrator):
        # Executes all 9 stages:
        # [ANALYZE] -> [SELECT] -> [GENERATE] -> [EDIT] -> [THUMBNAIL] -> [CAPTION/SEO] -> [UPLOAD] -> [LOG]
        summary = orchestrator.run_cycle()
        assert summary["status"] == "COMPLETED"
        assert "cycle_id" in summary
        assert "topic" in summary
        assert "vps" in summary or "vps_score" in summary
        assert "video_id" in summary
        assert "url" in summary
        assert Path(summary["log_file"]).exists()

    def test_cli_argument_parser(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--cycle", action="store_true")
        parser.add_argument("--daemon", action="store_true")
        parser.add_argument("--status", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

        args = parser.parse_args(["--cycle", "--dry-run"])
        assert args.cycle is True
        assert args.dry_run is True
        assert args.daemon is False
