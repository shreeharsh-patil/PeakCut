"""Unit tests for PeakCut Studio Colab Autonomous Runner (colab_runner.py).

Verifies 100% zero-API local execution, Gemma model environment configuration,
and pure-Python headless execution with NO CLI dependencies.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

import colab_runner
import llm_backend


@pytest.fixture(autouse=True)
def clean_colab_env(monkeypatch):
    """Ensure environment mutations in colab runner tests do not leak into other test suites."""
    orig_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(orig_env)


def test_is_colab_detection(monkeypatch):
    monkeypatch.delenv("IS_COLAB", raising=False)
    monkeypatch.delenv("FORCE_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("COLAB_GPU", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    # By default on test runner without env, should be false
    # unless running inside colab
    is_normal = colab_runner.is_colab()

    # When IS_COLAB is set, must be True
    monkeypatch.setenv("IS_COLAB", "1")
    assert colab_runner.is_colab() is True

    # When FORCE_LOCAL_ONLY is set, must be True
    monkeypatch.delenv("IS_COLAB", raising=False)
    monkeypatch.setenv("FORCE_LOCAL_ONLY", "1")
    assert colab_runner.is_colab() is True


def test_setup_colab_environment_purges_external_apis(monkeypatch):
    # Set dummy API keys
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    monkeypatch.setenv("GEMINI_API_KEYS", "dummy-gemini-keys")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")

    with patch("colab_runner.shutil.which", return_value="/usr/bin/ollama"), \
         patch("subprocess.run") as mock_subproc, \
         patch("httpx.Client") as mock_httpx:

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__.return_value = mock_client
        mock_httpx.return_value = mock_client

        status = colab_runner.setup_colab_environment(
            model_name="gemma2:2b",
            mount_drive=False,
            whisper_model="base"
        )

        assert status["status"] == "ready"
        assert status["model"] == "gemma2:2b"

        # Verify external API keys were completely removed from environment
        assert "GEMINI_API_KEY" not in os.environ
        assert "GEMINI_API_KEYS" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ

        # Verify local model environment variables are active
        assert os.environ.get("LLM_PROVIDER") == "ollama"
        assert os.environ.get("LLM_BASE_URL") == "http://127.0.0.1:11434/v1"
        assert os.environ.get("LLM_MODEL") == "gemma2:2b"
        assert os.environ.get("FORCE_LOCAL_ONLY") == "1"


def test_parse_url_list():
    raw_text = """
    https://www.youtube.com/watch?v=video1
    # comment line
    https://www.youtube.com/watch?v=video2 # inline comment
    https://www.youtube.com/watch?v=video1
    """
    parsed = colab_runner.parse_url_list(raw_text)
    assert len(parsed) == 2
    assert parsed[0] == "https://www.youtube.com/watch?v=video1"
    assert parsed[1] == "https://www.youtube.com/watch?v=video2"

    comma_text = "https://youtu.be/a, https://youtu.be/b"
    assert colab_runner.parse_url_list(comma_text) == ["https://youtu.be/a", "https://youtu.be/b"]


def test_run_shorts_pipeline_pure_python(tmp_path, monkeypatch):
    mock_pipeline_res = {
        "success": True,
        "video_title": "test_video",
        "input_video": str(tmp_path / "test.mp4"),
        "output_dir": str(tmp_path),
        "clips": [str(tmp_path / "clip_1.mp4"), str(tmp_path / "clip_2.mp4")],
        "metadata": {
            "shorts": [
                {"video_title_for_youtube_short": "Hook 1", "viral_hook_text": "Desc 1"},
                {"video_title_for_youtube_short": "Hook 2", "viral_hook_text": "Desc 2"},
            ]
        },
        "duration": 60.0,
    }

    with patch("main.process_video_pipeline", return_value=mock_pipeline_res) as mock_main:
        results = colab_runner.run_shorts_pipeline(
            urls="https://www.youtube.com/watch?v=test",
            clips_per_video=2,
            format="vertical",
            output_dir=str(tmp_path),
            schedule_to_youtube=False
        )

        assert results["total_queued"] == 1
        assert len(results["successful"]) == 1
        assert len(results["all_clips"]) == 2
        assert mock_main.called
        assert mock_main.call_args[1]["target_clips"] == 2
        assert mock_main.call_args[1]["output_format"] == "vertical"


def test_publish_clip_pure_python(tmp_path):
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"dummy")

    with patch("youtube_uploader.upload_youtube_short", return_value={"video_id": "yt_123", "status": "scheduled"}) as mock_up:
        res = colab_runner.publish_clip(
            video_path=str(clip_file),
            title="Short Title #Shorts",
            description="Viral short",
            privacy="unlisted",
            delay_hours=1.0,
            access_token="test_token"
        )
        assert res["video_id"] == "yt_123"
        assert mock_up.called
        assert mock_up.call_args[1]["privacy_status"] == "unlisted"
        assert mock_up.call_args[1]["publish_at"] is not None


def test_peakcut_bridge_ai_status_in_local_mode(monkeypatch):
    import app as app_module
    import httpx
    import asyncio

    monkeypatch.setenv("FORCE_LOCAL_ONLY", "1")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "gemma2:2b")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)

    async def _test():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # 1. /api/status must show AI ready without external keys
            r_stat = await client.get("/api/status")
            assert r_stat.status_code == 200
            data_stat = r_stat.json()
            assert data_stat["ai"]["ready"] is True
            assert data_stat["ai"]["configured"] is True
            assert data_stat["ai"]["provider"] == "ollama"
            assert data_stat["ai"]["model"] == "gemma2:2b"

            # 2. /api/sync must also report ready
            r_sync = await client.get("/api/sync")
            assert r_sync.status_code == 200
            data_sync = r_sync.json()
            assert data_sync["ai"]["ready"] is True
            assert data_sync["ai"]["configured"] is True
            assert data_sync["ai"]["provider"] == "ollama"

            # 3. /api/ai must report ready
            r_ai = await client.get("/api/ai")
            assert r_ai.status_code == 200
            data_ai = r_ai.json()
            assert data_ai["ready"] is True
            assert data_ai["provider"] == "ollama"

    asyncio.run(_test())


def test_infer_job_stage_and_progress_dynamic():
    import peakcut_bridge

    # Downloading stage
    stage, prog = peakcut_bridge._infer_job_stage_and_progress(["📥 Downloading video from YouTube..."])
    assert "Downloading" in stage
    assert prog == 0.15

    # Transcription stage
    stage, prog = peakcut_bridge._infer_job_stage_and_progress(["🎙️  Transcribing video..."])
    assert "Transcribing" in stage
    assert 0.30 <= prog <= 0.40

    # AI clip detection
    stage, prog = peakcut_bridge._infer_job_stage_and_progress(["get_viral_clips: scoring windows with local model"])
    assert "AI" in stage or "viral" in stage
    assert 0.45 <= prog <= 0.60

    # Rendering with ready clips and suppressing YOLO warnings
    logs = [
        "🔥 Found 3 clips!",
        "🎬 Processing Clip 1: 0.0s - 30.0s",
        "Update Settings with 'yolo settings key=value' 65%",
        "CLIP_READY 0 clip_1.mp4"
    ]
    stage, prog = peakcut_bridge._infer_job_stage_and_progress(logs, ready_files={0: "clip_1.mp4"}, clips_count=3)
    assert "Rendering" in stage
    assert "1/3 ready" in stage
    assert "yolo" not in stage.lower()
    assert prog > 0.65


def test_normalize_media_url_variations():
    from security_utils import normalize_media_url, assert_public_url

    urls = [
        "watch?v=irNhWMG560k",
        "/watch?v=irNhWMG560k",
        "irNhWMG560k",
        "youtube.com/watch?v=irNhWMG560k",
        "https://www.youtube.com/watch?v=irNhWMG560k",
        "https://youtu.be/irNhWMG560k",
        "shorts/irNhWMG560k",
    ]
    for raw in urls:
        norm = normalize_media_url(raw)
        assert norm == "https://www.youtube.com/watch?v=irNhWMG560k", f"Failed for {raw}: got {norm}"
        assert assert_public_url(norm) == norm
