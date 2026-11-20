"""Regression & acceptance tests for PeakCut core video-processing fixes.

Covers:
1. Valid synthetic video duration, resolution, codec, and playability (ClipMind CI fix)
2. Invalid / corrupt / zero-byte media rejection
3. Long video streaming audio energy extraction & bounded clip count scaling
4. Silence, trailing silence, and missing speech handling
5. Clip duration boundaries, sentence-boundary snapping, overlap deduplication, and viral scoring
6. Animated subtitle synchronization, cross-block non-overlap, Unicode/Hinglish, ASS alignment & animation
7. Smart camera tracking, 1-to-1 speaker ID matching, smooth motion, static-layout fast path, edge safety
8. Output validation (H.264/AAC, aspect ratio, duration, black-frame check, NVENC->x264 runtime fallback)
9. Failed FFmpeg execution & deterministic YouTube ingestion error classification
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import active_speaker
import clip_selection
import ffmpeg_utils
import highlight_detector
import reframe_v2
import scene_detection
import security_utils
import smartcut
import subtitles
import transcribe_backends
import viral_brain
from clipmind.core.config import get_config
from clipmind.generation.clip_pipeline import ClipPipeline

HAS_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe required")


def _make_real_mp4(path: Path, duration: float = 3.0, size: str = "640x360", with_audio: bool = True, color: str = "blue"):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c={color}:s={size}:r=15:d={duration:.2f}",
    ]
    if with_audio:
        cmd.extend(["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={duration:.2f}"])
    cmd.extend([
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
    ])
    if with_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "64k"])
    cmd.extend(["-movflags", "+faststart", "-t", f"{duration:.2f}", str(path)])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=30)
    return path


# ==============================================================================
# 1. Valid synthetic video duration & resolution (Confirmed CI failure fix)
# ==============================================================================
class TestSyntheticVideoAndCIFix:
    @requires_ffmpeg
    def test_simulated_video_is_real_playable_50s_1080x1920_and_fast(self, tmp_path):
        cfg = get_config(reload=True)
        cfg.paths.output_dir = tmp_path / "out"
        cfg.paths.temp_dir = tmp_path / "tmp"
        cfg.paths.ensure_directories()
        pipeline = ClipPipeline(config=cfg)

        out = tmp_path / "sim_50s.mp4"
        t0 = time.perf_counter()
        created = pipeline._create_simulated_video_file(out, duration=50.0)
        elapsed = time.perf_counter() - t0

        assert created.exists()
        assert created.stat().st_size > 10_000
        # Must stay far below CI timeouts even on 2-vCPU runners
        assert elapsed < 10.0, f"Synthetic generation too slow: {elapsed:.2f}s"

        info = pipeline.get_media_info(created)
        assert info["valid"] is True
        assert 48.5 <= info["duration"] <= 51.5
        assert info["width"] == 1080
        assert info["height"] == 1920
        assert info["has_video"] is True
        assert info["has_audio"] is True
        assert info["video_codec"] in ("h264", "mpeg4")
        assert info["audio_codec"] == "aac"

        ok, _vinfo, reason = ffmpeg_utils.validate_media_output(
            created,
            expected_duration=50.0,
            duration_tolerance=1.5,
            expected_resolution=(1080, 1920),
            expected_aspect_ratio=9 / 16,
            require_video=True,
            require_audio=True,
            decode_check=True,
        )
        assert ok, f"Simulated video failed real decode validation: {reason}"

    def test_simulated_video_never_writes_corrupt_fake_mp4_on_ffmpeg_failure(self, tmp_path):
        cfg = get_config(reload=True)
        cfg.paths.output_dir = tmp_path / "out"
        cfg.paths.temp_dir = tmp_path / "tmp"
        cfg.paths.ensure_directories()
        pipeline = ClipPipeline(config=cfg)

        out = tmp_path / "fail_sim.mp4"
        failed_proc = MagicMock(returncode=1, stderr="mock encoder failure", stdout="")
        with patch("clipmind.generation.clip_pipeline.subprocess.run", return_value=failed_proc):
            with pytest.raises(RuntimeError, match="Failed to generate valid simulated MP4"):
                pipeline._create_simulated_video_file(out, duration=50.0)
        # Must NOT leave behind a corrupt 44-byte unplayable file
        assert not out.exists()


# ==============================================================================
# 2. Invalid / corrupt media detection & output validation
# ==============================================================================
class TestInvalidMediaAndOutputValidation:
    def test_corrupt_header_only_mp4_is_rejected(self, tmp_path):
        # The exact broken 44-byte stub previously written on timeout
        bad = tmp_path / "broken_stub.mp4"
        bad.write_bytes(
            b"\x00\x00\x00 ftypmp42\x00\x00\x00\x00isommp42\x00\x00\x00\x08free\x00\x00\x00\x10mdat\x00\x00\x00\x00"
        )
        ok, info, reason = ffmpeg_utils.validate_media_output(bad, require_video=True)
        assert ok is False
        assert info["valid"] is False
        assert info["duration"] == 0.0
        assert reason

    def test_zero_byte_and_missing_files_rejected(self, tmp_path):
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"")
        ok_empty, info_empty, reason_empty = ffmpeg_utils.validate_media_output(empty)
        assert ok_empty is False
        assert "zero bytes" in reason_empty
        assert info_empty["size_bytes"] == 0

        missing = tmp_path / "missing.mp4"
        ok_miss, info_miss, reason_miss = ffmpeg_utils.validate_media_output(missing)
        assert ok_miss is False
        assert "missing" in reason_miss
        assert info_miss["exists"] is False

    @requires_ffmpeg
    def test_validate_media_output_detects_wrong_duration_and_aspect(self, tmp_path):
        clip = _make_real_mp4(tmp_path / "landscape.mp4", duration=2.0, size="640x360", with_audio=True)

        ok, info, reason = ffmpeg_utils.validate_media_output(
            clip,
            expected_duration=2.0,
            duration_tolerance=0.5,
            expected_aspect_ratio=16 / 9,
            allowed_video_codecs=("h264",),
            allowed_audio_codecs=("aac",),
            require_audio=True,
            decode_check=True,
        )
        assert ok is True, reason
        assert info["video_codec"] == "h264"
        assert info["audio_codec"] == "aac"

        # Wrong expected duration fails
        ok_dur, _, reason_dur = ffmpeg_utils.validate_media_output(clip, expected_duration=30.0, duration_tolerance=1.0)
        assert ok_dur is False
        assert "duration mismatch" in reason_dur

        # Wrong expected aspect ratio fails
        ok_ar, _, reason_ar = ffmpeg_utils.validate_media_output(clip, expected_aspect_ratio=9 / 16)
        assert ok_ar is False
        assert "aspect ratio mismatch" in reason_ar

    @requires_ffmpeg
    def test_black_frame_detector_flags_all_black_and_accepts_color(self, tmp_path):
        black = _make_real_mp4(tmp_path / "black.mp4", duration=2.0, size="320x240", with_audio=False, color="black")
        blue = _make_real_mp4(tmp_path / "blue.mp4", duration=2.0, size="320x240", with_audio=False, color="blue")

        ok_black, ratio_black = ffmpeg_utils.check_black_frames(black, max_black_ratio=0.90)
        assert ok_black is False
        assert ratio_black >= 0.90

        ok_blue, ratio_blue = ffmpeg_utils.check_black_frames(blue, max_black_ratio=0.90)
        assert ok_blue is True
        assert ratio_blue == 0.0

    def test_nvenc_runtime_failure_falls_back_to_x264(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "h264_nvenc" in cmd:
                return MagicMock(returncode=1, stderr=b"OpenEncodeSessionEx failed: out of memory", stdout=b"")
            return MagicMock(returncode=0, stderr=b"", stdout=b"")

        monkeypatch.setattr(ffmpeg_utils.subprocess, "run", fake_run)
        cmd = ["ffmpeg", "-y", "-i", "in.mp4", *ffmpeg_utils._NVENC_ARGS[ffmpeg_utils.QUALITY], "out.mp4"]
        res = ffmpeg_utils.run_ffmpeg_with_fallback(cmd, tier=ffmpeg_utils.QUALITY)
        assert res.returncode == 0
        assert len(calls) == 2
        assert "h264_nvenc" in calls[0]
        assert "libx264" in calls[1]
        assert "h264_nvenc" not in calls[1]
        ffmpeg_utils.reset_encoder_cache()


# ==============================================================================
# 3. Long videos, silence, and missing speech
# ==============================================================================
class TestLongVideosAndSilence:
    def test_highlight_detector_rejects_pure_silence_and_noise_floor(self):
        # Tiny -90dB numerical noise must NOT generate fake hype clips
        rng = np.random.default_rng(42)
        silent_env = rng.uniform(0.0, 0.0005, size=240).astype(np.float32)
        with patch("highlight_detector.extract_audio_energy", return_value=(silent_env, 120.0)):
            assert highlight_detector.detect_hype_moments("silent.mp4") == []

    def test_highlight_detector_prevents_overlapping_clips_and_handles_end_peak(self):
        env = np.full(200, 0.03, dtype=np.float32)  # 100s at 0.5s/win
        env[186:194] = 0.85  # peak near end (~95s)
        env[40:48] = 0.80    # peak at ~22s
        with patch("highlight_detector.extract_audio_energy", return_value=(env, 100.0)):
            moments = highlight_detector.detect_hype_moments(
                "long.mp4", max_clips=3, target_clip_duration=20.0, min_gap_sec=10.0,
            )
            assert len(moments) == 2
            # End-of-video peak shifts start backwards to preserve full 20s duration
            assert moments[-1].end <= 100.0
            assert moments[-1].duration == 20.0
            # Strict non-overlap
            assert moments[0].end <= moments[1].start

    def test_smartcut_detects_trailing_silence_at_eof(self):
        mock_stderr = "[silencedetect @ 0x1] silence_start: 18.000\n"
        with patch("smartcut.subprocess.run", return_value=MagicMock(stderr=mock_stderr)):
            silences = smartcut.detect_silences("clip.mp4", total_duration=25.0)
            assert silences == [(18.0, 25.0)]

    @requires_ffmpeg
    def test_smartcut_long_filter_script_no_nameerror_and_silent_video_safe(self, tmp_path):
        # 100%-silent video should stream-copy cleanly without failing
        silent_vid = _make_real_mp4(tmp_path / "silent.mp4", duration=2.0, size="320x240", with_audio=False)
        out_vid = tmp_path / "smartcut_out.mp4"
        res = smartcut.apply_smartcut(silent_vid, out_vid)
        assert Path(res.output_path).exists()
        assert res.cuts_count == 0
        # Verify `time` module is imported in smartcut (regression for NameError when filter > 2000 chars)
        assert hasattr(smartcut, "time")

    def test_heuristic_transcript_clips_scales_long_video_and_deduplicates(self):
        main = pytest.importorskip("main")
        # Simulate a 1-hour (3600s) transcript of 600 segments
        segments = [
            {
                "start": i * 6.0,
                "end": (i + 1) * 6.0,
                "text": (
                    "Why did nobody tell you this 10x secret mistake? "
                    "In reality the lesson is simple."
                    if i % 40 == 5
                    else "And so we continue talking about regular updates here."
                ),
            }
            for i in range(600)
        ]
        clips = main._heuristic_transcript_clips({"segments": segments}, video_duration=3600.0, min_secs=18, max_secs=30)
        assert 1 <= len(clips) <= 12
        for c in clips:
            assert 18.0 <= (c["end"] - c["start"]) <= 30.0
        # Ensure zero overlap between returned clips
        for a, b in zip(clips, clips[1:]):
            assert a["end"] <= b["start"]


# ==============================================================================
# 4. Clip duration boundaries, sentence snapping, and viral scoring
# ==============================================================================
class TestClipSelectionAndViralScoring:
    def test_clipmind_constraints_stay_separate_from_general_bounds(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLIP_MIN_SECONDS", "15")
        monkeypatch.setenv("CLIP_MAX_SECONDS", "90")
        assert clip_selection.clip_duration_bounds() == (15.0, 90.0)

        cfg = get_config(reload=True)
        cfg.paths.output_dir = tmp_path / "out"
        cfg.paths.temp_dir = tmp_path / "tmp"
        cfg.paths.ensure_directories()
        pipeline = ClipPipeline(config=cfg)
        assert pipeline.min_duration == 45.0
        assert pipeline.max_duration == 58.0

    def test_snap_clip_prefers_sentence_boundaries(self):
        words = [
            {"w": "prev", "s": 8.0, "e": 9.2},
            {"w": "end.", "s": 9.3, "e": 9.8},
            {"w": "Hook", "s": 10.4, "e": 11.0},
            {"w": "word", "s": 11.1, "e": 12.0},
            {"w": "middle", "s": 20.0, "e": 22.0},
            {"w": "payoff.", "s": 28.0, "e": 29.0},
            {"w": "Next", "s": 29.8, "e": 30.4},
        ]
        # Proposed start 10.9 is closer to "word" (11.1), but sentence start "Hook" (10.4) is within search_window
        s, e = clip_selection.snap_clip_to_words(10.9, 29.5, words, video_duration=60.0)
        assert 10.0 <= s <= 10.4
        assert 29.0 <= e <= 29.5

    def test_deduplicate_overlapping_clips_keeps_highest_scored(self):
        shorts = [
            {"start": 10.0, "end": 40.0, "predicted_score": 72},
            {"start": 12.0, "end": 42.0, "predicted_score": 94},  # heavy overlap -> wins
            {"start": 100.0, "end": 130.0, "predicted_score": 81},
            {"start": 110.0, "end": 125.0, "predicted_score": 60},  # nested -> dropped
        ]
        deduped = clip_selection.deduplicate_overlapping_clips(shorts)
        assert len(deduped) == 2
        assert [c["predicted_score"] for c in deduped] == [94, 81]
        assert deduped[0]["start"] < deduped[1]["start"]

    def test_viral_brain_question_hook_fillers_and_dangling_cohesion(self):
        # Fixed bug: generate_viral_hook previously stripped '?' before checking if '?' in first_sentence
        hook = viral_brain.generate_viral_hook("Can you survive 100 hours? Let us see.")
        assert "?" in hook and "Can you survive 100 hours" in hook

        # Whole-word question check: "Island" must NOT match "is"
        s_island = viral_brain.score_hook_strength("Island weather today.")
        s_question = viral_brain.score_hook_strength("Is this the real reason?")
        assert s_question > s_island

        # Multi-word filler & punctuation awareness
        dense = viral_brain.score_density("Quantum processors cut training latency by eighty percent.")
        filler_heavy = viral_brain.score_density("Um, you know, like, I mean, sort of, basically, kind of, yeah.")
        assert dense > filler_heavy + 35

        # Cohesion penalizes dangling start/end vs complete storytelling
        dangling = viral_brain.score_cohesion("And because he said that it was going to")
        complete = viral_brain.score_cohesion("Nobody expected this result. The benchmark ran 4x faster. Here is why it worked.")
        assert complete > dangling + 25


# ==============================================================================
# 5. Animated subtitles synchronization, Unicode/Hinglish, and alignment
# ==============================================================================
class TestSubtitleSyncAndMultilingual:
    def test_ass_has_zero_cross_block_overlap_and_clamps_to_clip_end(self, tmp_path):
        out = tmp_path / "sync.ass"
        # Tightly packed words across multiple blocks, plus a word overshooting clip_end (4.0)
        transcript = {
            "segments": [{
                "start": 0.0,
                "end": 5.0,
                "words": [
                    {"word": " Alpha", "start": 0.0, "end": 0.4},
                    {"word": " Beta", "start": 0.4, "end": 0.8},
                    {"word": " Gamma", "start": 0.8, "end": 0.8},   # zero-dur end of block 1
                    {"word": " Delta", "start": 0.82, "end": 1.3},  # starts 20ms later -> used to overlap!
                    {"word": " Epsilon", "start": 3.5, "end": 4.9}, # overshoots clip_end=4.0
                ],
            }]
        }
        assert subtitles.generate_ass(transcript, 0.0, 4.0, str(out), max_chars=14) is True
        content = out.read_text(encoding="utf-8-sig")

        def _parse_ts(ts: str) -> float:
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)

        times = []
        for line in content.splitlines():
            if line.startswith("Dialogue:"):
                parts = line.split(",", 9)
                st, et = _parse_ts(parts[1]), _parse_ts(parts[2])
                assert et > st
                assert et <= 4.01
                times.append((st, et))
        assert len(times) == 5
        for (_, prev_end), (next_start, _) in zip(times, times[1:]):
            assert next_start >= prev_end - 1e-6, f"Overlapping ASS events: {prev_end} > {next_start}"

    def test_pause_triggers_new_block_so_captions_do_not_start_too_early(self, tmp_path):
        transcript = {
            "segments": [{
                "words": [
                    {"word": " Hi", "start": 0.0, "end": 0.3},
                    {"word": " again", "start": 2.5, "end": 2.9},  # 2.2s silence pause
                ]
            }]
        }
        blocks = subtitles._collect_word_blocks(transcript, 0.0, 5.0, max_chars=25, max_duration=5.0)
        assert len(blocks) == 2

    def test_hindi_english_mixed_speech_and_segment_only_fallback(self, tmp_path):
        out = tmp_path / "hinglish.ass"
        # Segment without word timestamps containing Hindi (Devanagari) + English
        transcript = {
            "language": "hi",
            "segments": [{
                "start": 0.0,
                "end": 3.0,
                "text": "यह AI trick आपको 10x fast बना देगी।",
            }],
        }
        assert subtitles.generate_ass(transcript, 0.0, 3.0, str(out), uppercase=True) is True
        content = out.read_text(encoding="utf-8-sig")
        assert "यह" in content
        assert "AI" in content
        assert "TRICK" in content
        assert "Dialogue:" in content

    def test_burn_subtitles_uses_ass_v4_alignment_numbers(self, monkeypatch, tmp_path):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = " ".join(str(c) for c in cmd)
            return MagicMock(returncode=0, stderr=b"")

        monkeypatch.setattr(subtitles.subprocess, "run", fake_run)
        srt = tmp_path / "s.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

        subtitles.burn_subtitles("in.mp4", str(srt), "out.mp4", alignment="top")
        assert "Alignment=8" in captured["cmd"]
        subtitles.burn_subtitles("in.mp4", str(srt), "out.mp4", alignment="middle")
        assert "Alignment=5" in captured["cmd"]
        subtitles.burn_subtitles("in.mp4", str(srt), "out.mp4", alignment="bottom")
        assert "Alignment=2" in captured["cmd"]


# ==============================================================================
# 6. Speaker tracking, reframing fast path, and edge-case resilience
# ==============================================================================
class TestSpeakerTrackingAndReframing:
    def test_two_close_faces_never_share_same_tracker_id(self):
        main = pytest.importorskip("main")
        tracker = main.SpeakerTracker(cooldown_frames=30)
        # Seed face at x=500
        tracker.get_target([{"box": [450, 100, 100, 100], "score": 10000}], frame_number=0, width=1920)
        # Next frame: two close faces near x=500 and x=580 (both within 15% of 1920 = 288px)
        tracker.get_target(
            [
                {"box": [450, 100, 100, 100], "score": 10000},
                {"box": [530, 100, 90, 90], "score": 8100},
            ],
            frame_number=1,
            width=1920,
        )
        ids = {kf["id"] for kf in tracker.known_faces}
        assert len(ids) == 2

    def test_detect_face_candidates_returns_empty_when_mediapipe_absent(self, monkeypatch):
        main = pytest.importorskip("main")
        monkeypatch.setattr(main, "face_detection", None)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        assert main.detect_face_candidates(frame) == []

    def test_reframe_v2_skips_ffmpeg_decode_when_all_scenes_static(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("FFmpeg decode should be skipped when no scene uses TRACK")

        monkeypatch.setattr(reframe_v2.subprocess, "Popen", boom)
        xs = reframe_v2._analyze_trajectory(
            "unused.mp4",
            scenes_boundaries=[(0, 60), (60, 150)],
            scene_strategies=["GENERAL", "SPLIT"],
            fps=30.0,
            orig_w=1920,
            orig_h=1080,
            cameraman=MagicMock(),
            tracker=MagicMock(),
        )
        assert xs == [None] * 150

    def test_active_speaker_handles_odd_byte_pcm_and_zero_fps(self):
        with patch("active_speaker.subprocess.run", return_value=MagicMock(stdout=b"\x01\x02\x03" * 3001)):
            env = active_speaker.audio_envelope("dummy.mp4", 0.0, 1.0)
            assert isinstance(env, list)
        xs = active_speaker.speaker_xs([0, 1], [(400, 400, 100, 100), (1400, 400, 100, 100)], 600, 1920, 10, fps=0.0)
        assert len(xs) == 10

    def test_scene_detection_safe_fps_handles_nan_and_zero(self):
        assert scene_detection._safe_fps(float("nan")) == 30.0
        assert scene_detection._safe_fps(0.0) == 30.0
        assert scene_detection._safe_fps(-5.0) == 30.0
        assert scene_detection._safe_fps(29.97) == 29.97


# ==============================================================================
# 7. YouTube ingestion URL normalization & deterministic error classification
# ==============================================================================
class TestYouTubeIngestionFixes:
    def test_normalize_extended_youtube_url_variants(self):
        assert security_utils.normalize_media_url(
            "Check this out: https://youtu.be/BgeNDG5jD60?t=45)."
        ) == "https://www.youtube.com/watch?v=BgeNDG5jD60"
        assert security_utils.normalize_media_url(
            "https://www.youtube-nocookie.com/embed/BgeNDG5jD60"
        ) == "https://www.youtube.com/watch?v=BgeNDG5jD60"
        assert security_utils.normalize_media_url(
            "https://m.youtube.com/shorts/BgeNDG5jD60?feature=share"
        ) == "https://www.youtube.com/watch?v=BgeNDG5jD60"

    def test_youtube_error_classification_is_deterministic(self):
        main = pytest.importorskip("main")
        assert main._classify_youtube_error(RuntimeError("Private video. Sign in if you've been granted access")) == ("private_video", True)
        assert main._classify_youtube_error(RuntimeError("This video is available to this channel's members on level... members-only")) == ("members_only", True)
        assert main._classify_youtube_error(RuntimeError("HTTP Error 429: Too Many Requests")) == ("rate_limited", False)
        assert main._classify_youtube_error(RuntimeError("Sign in to confirm your age")) == ("age_restricted", False)
        assert main._classify_youtube_error(RuntimeError("The uploader has not made this video available in your country")) == ("geo_restricted", False)
        assert main._classify_youtube_error(RuntimeError("Video unavailable")) == ("unavailable", False)

    def test_transcribe_extract_wav_cleans_temp_file_on_ffmpeg_error(self, monkeypatch, tmp_path):
        created_paths = []
        orig_mkstemp = transcribe_backends.tempfile.mkstemp

        def track_mkstemp(**kwargs):
            fd, p = orig_mkstemp(dir=str(tmp_path), **kwargs)
            created_paths.append(p)
            return fd, p

        monkeypatch.setattr(transcribe_backends.tempfile, "mkstemp", track_mkstemp)
        monkeypatch.setattr(
            transcribe_backends.subprocess, "run",
            MagicMock(side_effect=subprocess.CalledProcessError(1, ["ffmpeg"])),
        )
        with pytest.raises(subprocess.CalledProcessError):
            transcribe_backends._extract_wav("corrupt.mp4")
        assert len(created_paths) == 1
        assert not os.path.exists(created_paths[0])
