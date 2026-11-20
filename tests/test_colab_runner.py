"""Unit tests for PeakCut Studio Colab Autonomous Runner (colab_runner.py).

Verifies 100% zero-API local execution, Gemma model environment configuration,
and pure-Python headless execution with NO CLI dependencies.
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, mock_open

import colab_runner
from engine.intelligence import llm_backend


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


def test_setup_colab_environment_preserves_external_keys_by_default(monkeypatch):
    # Set dummy API keys — user's optional external keys must NOT be unexpectedly deleted
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")
    monkeypatch.setenv("GEMINI_API_KEYS", "dummy-gemini-keys")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")

    fake_ai = {
        "ready": True,
        "mode": "ollama",
        "binary_found": True,
        "server_running": True,
        "model_installed": True,
        "test_inference_ok": True,
        "model": "gemma2:2b",
        "error": None,
    }
    with patch("colab_runner.ensure_ollama_ready", return_value=fake_ai):
        status = colab_runner.setup_colab_environment(
            model_name="gemma2:2b",
            mount_drive=False,
            whisper_model="base",
        )

        assert status["status"] == "ready"
        assert status["model"] == "gemma2:2b"
        assert status["ollama_ready"] is True
        assert status["ai_mode"] == "ollama"

        # External API keys preserved by default while local routing is active
        assert os.environ.get("GEMINI_API_KEY") == "dummy-gemini-key"
        assert os.environ.get("OPENAI_API_KEY") == "dummy-openai-key"

        # Local model environment variables are active
        assert os.environ.get("LLM_PROVIDER") == "ollama"
        assert os.environ.get("LLM_BASE_URL") == "http://127.0.0.1:11434/v1"
        assert os.environ.get("LLM_MODEL") == "gemma2:2b"
        assert os.environ.get("FORCE_LOCAL_ONLY") == "1"

        # Explicit opt-in purge still works if requested
        colab_runner.setup_colab_environment(
            model_name="gemma2:2b",
            mount_drive=False,
            preserve_external_keys=False,
        )
        assert "GEMINI_API_KEY" not in os.environ
        assert "OPENAI_API_KEY" not in os.environ


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

    with patch("engine.services.youtube_uploader.upload_youtube_short", return_value={"video_id": "yt_123", "status": "scheduled"}) as mock_up:
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
    monkeypatch.delenv("PEAKCUT_ACCESS_TOKEN", raising=False)

    async def _test_ready():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_stat = await client.get("/api/status")
            assert r_stat.status_code == 200
            data_stat = r_stat.json()
            assert data_stat["ai"]["ready"] is True
            assert data_stat["ai"]["configured"] is True
            assert data_stat["ai"]["provider"] == "ollama"
            assert data_stat["ai"]["model"] == "gemma2:2b"
            assert data_stat["ai"]["fallback_active"] is False

            r_sync = await client.get("/api/sync")
            assert r_sync.status_code == 200
            data_sync = r_sync.json()
            assert data_sync["ai"]["ready"] is True
            assert data_sync["ai"]["provider"] == "ollama"

            r_ai = await client.get("/api/ai")
            assert r_ai.status_code == 200
            data_ai = r_ai.json()
            assert data_ai["ready"] is True
            assert data_ai["provider"] == "ollama"

    async def _test_offline():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r_stat = await client.get("/api/status")
            assert r_stat.status_code == 200
            data_stat = r_stat.json()
            # Must NEVER falsely report local AI ready when Ollama is offline
            assert data_stat["ai"]["ready"] is False
            assert data_stat["ai"]["fallback_active"] is True
            assert data_stat["ai"]["heuristic_ready"] is True

    healthy_probe = {
        "ready": True,
        "reachable": True,
        "model_installed": True,
        "installed_models": ["gemma2:2b"],
        "error": None,
    }
    offline_probe = {
        "ready": False,
        "reachable": False,
        "model_installed": False,
        "installed_models": [],
        "error": "Connection refused",
    }

    with patch("engine.intelligence.llm_backend.check_local_llm_health", return_value=healthy_probe):
        asyncio.run(_test_ready())
    with patch("engine.intelligence.llm_backend.check_local_llm_health", return_value=offline_probe):
        asyncio.run(_test_offline())


def test_infer_job_stage_and_progress_dynamic():
    from engine.studio import peakcut_bridge

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
    from engine.utils.security_utils import normalize_media_url, assert_public_url

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


def test_core_modules_annotations_python312_compatible():
    """Guard against Python 3.12 eager-annotation NameErrors hidden by Python 3.14 PEP 649."""
    import importlib
    import inspect
    import typing

    core_mods = [
        "main",
        "app",
        "engine.studio.peakcut_bridge",
        "colab_runner",
        "engine.intelligence.llm_backend",
        "engine.editing.ffmpeg_utils",
        "engine.utils.security_utils",
        "engine.utils.batch_runner",
    ]
    for mname in core_mods:
        mod = importlib.import_module(mname)
        for name, obj in inspect.getmembers(mod):
            if getattr(obj, "__module__", None) != mname:
                continue
            if inspect.isfunction(obj):
                # Accessing __annotations__ triggers PEP 649 evaluation on 3.14;
                # get_type_hints evaluates stringified annotations from __future__.
                _ = obj.__annotations__
    hints = typing.get_type_hints(importlib.import_module("main").process_video_pipeline)
    assert "output_dir" in hints
    assert "target_clips" in hints


def test_heuristic_transcript_clips_fallback_and_unicode():
    """Verify heuristic fallback works when Ollama/LLM is unreachable, including Hindi/Unicode text."""
    import main
    from engine.intelligence import viral_brain

    # Direct evaluate_viral_clip accepts optional duration
    a1 = viral_brain.evaluate_viral_clip("Wait why did nobody tell you this secret? Because it works!", 28.0)
    assert a1.overall_score > 0
    assert a1.suggested_title

    # Multilingual/Hindi titles must not be stripped to empty
    a_hi = viral_brain.evaluate_viral_clip("क्या हाल है भाई लोग! आज GTA 6 खेलते हैं।", duration=25.0)
    assert "क्या" in a_hi.suggested_title or "GTA" in a_hi.suggested_title

    transcript = {
        "segments": [
            {"start": 0.0, "end": 10.0, "text": "Why does nobody talk about this crazy gaming moment?"},
            {"start": 10.0, "end": 22.0, "text": "Watch what happens when we enter the secret room in GTA 6!"},
            {"start": 22.0, "end": 35.0, "text": "Turns out the entire mission changes completely right here."},
            {"start": 40.0, "end": 55.0, "text": "Stop doing this mistake in every match you play."},
            {"start": 55.0, "end": 68.0, "text": "Here is the real reason why pros win 10x faster."},
        ]
    }
    clips = main._heuristic_transcript_clips(transcript, video_duration=70.0, target_clips=2)
    assert len(clips) == 2
    for c in clips:
        assert c["end"] > c["start"]
        assert c["video_title_for_youtube_short"]
        assert c["viral_hook_text"]


def test_cli_parser_structure():
    parser = colab_runner.build_cli_parser()
    # Test subcommands existence
    args_clip = parser.parse_args(["clip", "https://youtube.com/watch?v=test", "--clips", "4", "--format", "square"])
    assert args_clip.subcommand == "clip"
    assert args_clip.urls == ["https://youtube.com/watch?v=test"]
    assert args_clip.clips == 4
    assert args_clip.format == "square"

    args_run = parser.parse_args(["run", "https://youtu.be/abc"])
    assert args_run.subcommand == "run"
    assert args_run.clips == 3

    args_setup = parser.parse_args(["setup", "--model", "gemma2:9b", "--no-drive"])
    assert args_setup.subcommand == "setup"
    assert args_setup.model == "gemma2:9b"
    assert args_setup.no_drive is True

    args_status = parser.parse_args(["status"])
    assert args_status.subcommand == "status"

    args_upload = parser.parse_args(["upload", "/path/to/clip.mp4", "--title", "Custom Title", "--privacy", "private"])
    assert args_upload.subcommand == "upload"
    assert args_upload.video_path == "/path/to/clip.mp4"
    assert args_upload.title == "Custom Title"
    assert args_upload.privacy == "private"


def test_cli_main_status():
    ret = colab_runner.cli_main(["status"])
    assert ret == 0


def test_cli_main_help_when_no_args(capsys):
    ret = colab_runner.cli_main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "PeakCut Studio CLI" in captured.out or "usage:" in captured.out


def test_cli_main_clip_invocation():
    with patch("colab_runner.run_shorts_pipeline", return_value={"successful": [{"id": 1}]}) as mock_pipe:
        ret = colab_runner.cli_main(["clip", "https://youtube.com/watch?v=sample", "--clips", "2", "--format", "vertical"])
        assert ret == 0
        assert mock_pipe.called
        kwargs = mock_pipe.call_args[1]
        assert kwargs["urls"] == ["https://youtube.com/watch?v=sample"]
        assert kwargs["clips_per_video"] == 2
        assert kwargs["format"] == "vertical"


def test_cli_main_upload_invocation():
    with patch("colab_runner.publish_clip", return_value={"status": "ok"}) as mock_pub:
        ret = colab_runner.cli_main(["upload", "test_clip.mp4", "--title", "Awesome Clip", "--privacy", "unlisted"])
        assert ret == 0
        assert mock_pub.called
        kwargs = mock_pub.call_args[1]
        assert kwargs["video_path"] == "test_clip.mp4"
        assert kwargs["title"] == "Awesome Clip"
        assert kwargs["privacy"] == "unlisted"


def test_cli_main_setup_invocation():
    with patch("colab_runner.setup_colab_environment", return_value={"status": "ready"}) as mock_setup, \
         patch("colab_runner.install_cli_launcher", return_value=True) as mock_install:
        ret = colab_runner.cli_main(["setup", "--model", "gemma2:2b", "--no-drive"])
        assert ret == 0
        assert mock_setup.called
        assert mock_setup.call_args[1]["model_name"] == "gemma2:2b"
        assert mock_setup.call_args[1]["mount_drive"] is False
        assert mock_install.called


def test_install_cli_launcher_linux(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_runner.sys, "platform", "linux")
    with patch("os.path.isdir", return_value=True), \
         patch("os.access", return_value=True), \
         patch("builtins.open", mock_open()), \
         patch("os.chmod"):
        success = colab_runner.install_cli_launcher()
        assert success is True


def test_compute_safe_concurrency_free_t4_and_scaling():
    # Free Colab T4 profile: 15GB VRAM, 12.7GB RAM, 2 CPUs -> conservative single-worker limits
    t4_res = {
        "has_cuda": True,
        "gpu_name": "Tesla T4",
        "vram_total_gb": 15.0,
        "vram_free_gb": 14.5,
        "ram_total_gb": 12.7,
        "ram_available_gb": 10.5,
        "cpu_count": 2,
    }
    sched = colab_runner.compute_safe_concurrency(t4_res, model_name="gemma2:2b", render_profile="balanced")
    assert sched["max_concurrent_jobs"] == 1
    assert sched["clip_workers"] == 1
    assert sched["llm_workers"] == 1
    assert sched["ollama_num_parallel"] == 1
    assert sched["asr_gpu_concurrency"] == 1
    assert sched["whisper_device"] == "cuda"
    assert sched["whisper_compute"] == "float16"
    assert sched["video_quality"] == "1080"

    # Low-resource profile drops output height to 720p
    low = colab_runner.compute_safe_concurrency(t4_res, model_name="gemma2:2b", render_profile="low_resource")
    assert low["video_quality"] == "720"
    assert low["clip_workers"] == 1

    # High-VRAM GPU (e.g., A100 40GB + 50GB RAM + 8 CPUs) scales safely above 1
    a100_res = {
        "has_cuda": True,
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "vram_total_gb": 40.0,
        "vram_free_gb": 38.0,
        "ram_total_gb": 80.0,
        "ram_available_gb": 64.0,
        "cpu_count": 8,
    }
    a100_sched = colab_runner.compute_safe_concurrency(a100_res, model_name="gemma2:2b", render_profile="balanced")
    assert a100_sched["clip_workers"] >= 2
    assert a100_sched["ollama_num_parallel"] >= 2

    # CPU-only fallback
    cpu_res = {
        "has_cuda": False,
        "gpu_name": "CPU",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "ram_total_gb": 12.7,
        "ram_available_gb": 9.0,
        "cpu_count": 2,
    }
    cpu_sched = colab_runner.compute_safe_concurrency(cpu_res)
    assert cpu_sched["whisper_device"] == "cpu"
    assert cpu_sched["whisper_compute"] == "int8"
    assert cpu_sched["ffmpeg_encoder"] == "libx264"
    assert cpu_sched["clip_workers"] == 1


def test_ensure_ollama_ready_five_step_verification_and_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKCUT_LOG_DIR", str(tmp_path))

    # 1. Missing binary with install disabled -> explicit heuristic fallback
    with patch("colab_runner.shutil.which", return_value=None):
        res = colab_runner.ensure_ollama_ready(
            model_name="gemma2:2b", install_if_missing=False, run_test_inference=True
        )
        assert res["ready"] is False
        assert res["mode"] == "heuristic_fallback"
        assert res["executable_found"] is False

    # 2. Binary exists + server healthy + model installed + test inference passes -> ready=True
    with patch("colab_runner.shutil.which", return_value="/usr/local/bin/ollama"), \
         patch("engine.intelligence.llm_backend.check_local_llm_health") as mock_health:
        mock_health.side_effect = [
            {"reachable": True, "model_installed": True, "ready": True, "installed_models": ["gemma2:2b"], "test_inference_ok": None, "error": None},
            {"reachable": True, "model_installed": True, "ready": True, "installed_models": ["gemma2:2b"], "test_inference_ok": True, "error": None},
        ]
        res = colab_runner.ensure_ollama_ready(
            model_name="gemma2:2b", install_if_missing=False, run_test_inference=True
        )
        assert res["ready"] is True
        assert res["mode"] == "ollama"
        assert res["inference_ok"] is True

    # 3. Model installed, but real test inference FAILS -> never falsely claims ready!
    with patch("colab_runner.shutil.which", return_value="/usr/local/bin/ollama"), \
         patch("engine.intelligence.llm_backend.check_local_llm_health") as mock_health:
        mock_health.side_effect = [
            {"reachable": True, "model_installed": True, "ready": True, "installed_models": ["gemma2:2b"], "test_inference_ok": None, "error": None},
            {"reachable": True, "model_installed": True, "ready": False, "installed_models": ["gemma2:2b"], "test_inference_ok": False, "error": "CUDA OOM"},
        ]
        res = colab_runner.ensure_ollama_ready(
            model_name="gemma2:2b", install_if_missing=False, run_test_inference=True
        )
        assert res["ready"] is False
        assert res["mode"] == "heuristic_fallback"
        assert "CUDA OOM" in (res["error"] or "")


def test_launch_web_ui_starts_backend_before_tunnel_and_verifies_health(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKCUT_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("PEAKCUT_ACCESS_TOKEN", raising=False)

    events = []

    class _FakeProc:
        def __init__(self, pid):
            self.pid = pid
        def poll(self):
            return None

    def _fake_popen(cmd, **kwargs):
        if "uvicorn" in cmd:
            events.append("spawn_uvicorn")
            return _FakeProc(11111)
        if "tunnel" in cmd:
            events.append("spawn_cloudflared")
            # Write tunnel URL into cloudflare.log so launcher detects it
            (tmp_path / "cloudflare.log").write_text(
                "2026 INF +--- https://peakcut-verified-test.trycloudflare.com ---+\n",
                encoding="utf-8",
            )
            return _FakeProc(22222)
        raise AssertionError(f"Unexpected command: {cmd}")

    def _fake_http(url, headers=None, timeout=3.0):
        events.append(f"get:{url}")
        return True, 200, '{"status":"ok"}'

    with patch("colab_runner.stop_managed_services", return_value={}), \
         patch("colab_runner._is_port_open", return_value=False), \
         patch("colab_runner.subprocess.Popen", side_effect=_fake_popen), \
         patch("colab_runner._http_get_ok", side_effect=_fake_http), \
         patch("colab_runner._ensure_cloudflared_binary", return_value="/usr/local/bin/cloudflared"):
        url = colab_runner.launch_web_ui(
            run_in_background=True, port=8765, use_tunnel=True, access_mode="cloudflare", require_auth=True
        )

    # Order guarantee: Uvicorn spawns -> health + required routes verified -> ONLY THEN cloudflared spawns!
    idx_uv = events.index("spawn_uvicorn")
    idx_ready = events.index("get:http://127.0.0.1:8765/health/ready")
    idx_status = events.index("get:http://127.0.0.1:8765/api/status")
    idx_jobs = events.index("get:http://127.0.0.1:8765/api/jobs")
    idx_app = events.index("get:http://127.0.0.1:8765/app")
    idx_cf = events.index("spawn_cloudflared")
    assert idx_uv < idx_ready < idx_status < idx_jobs < idx_app < idx_cf

    # Token must be generated and attached to returned dashboard URL
    tok = os.environ.get("PEAKCUT_ACCESS_TOKEN", "")
    assert len(tok) >= 16
    assert url == f"https://peakcut-verified-test.trycloudflare.com/app?token={tok}"


def test_start_dashboard_access_refuses_tunnel_when_backend_unhealthy(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKCUT_LOG_DIR", str(tmp_path))
    with patch("colab_runner._http_get_ok", return_value=(False, 0, "Connection refused")):
        with pytest.raises(RuntimeError, match="Refusing to start dashboard tunnel"):
            colab_runner.start_dashboard_access(port=8766, use_tunnel=True, access_mode="cloudflare")


def test_colab_access_token_gate_middleware(monkeypatch):
    import asyncio
    import httpx
    import app as app_module

    secret = "test_colab_session_secret_987654"
    monkeypatch.setenv("PEAKCUT_ACCESS_TOKEN", secret)

    async def _run():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=False) as client:
            # 1. Health endpoints remain accessible without token
            r_h = await client.get("/health")
            assert r_h.status_code == 200
            r_hr = await client.get("/health/ready")
            assert r_hr.status_code == 200

            # 2. Protected routes reject unauthenticated access
            r_no_auth = await client.get("/api/status")
            assert r_no_auth.status_code == 401
            invalid_token = await client.get("/api/status", params={"token": "é"})
            assert invalid_token.status_code == 401
            preflight = await client.options("/api/status", headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            })
            assert "access-control-allow-origin" not in preflight.headers
            r_app_no_auth = await client.get("/app")
            assert r_app_no_auth.status_code == 401

            # 3. Header token grants access
            r_hdr = await client.get("/api/status", headers={"X-PeakCut-Token": secret})
            assert r_hdr.status_code == 200

            # 4. Visiting /app?token=... sets HttpOnly cookie and redirects to strip token from URL
            r_login = await client.get(f"/app?token={secret}")
            assert r_login.status_code == 302
            assert r_login.headers["location"] == "/app"
            set_cookie = r_login.headers.get("set-cookie", "")
            assert "peakcut_session=" in set_cookie
            assert "httponly" in set_cookie.lower()

            # 5. Subsequent cookie-authenticated requests succeed
            r_after = await client.get("/api/status", cookies={"peakcut_session": secret})
            assert r_after.status_code == 200
            r_app_after = await client.get("/app", cookies={"peakcut_session": secret})
            assert r_app_after.status_code == 200

    asyncio.run(_run())


def test_sync_job_directory_to_storage_verified_copy_and_temp_cleanup(tmp_path):
    stage = tmp_path / "stage_job"
    drive = tmp_path / "drive_shorts"
    stage.mkdir()

    clip = stage / "clip_1.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"A" * 2048)
    meta = stage / "processing_summary.json"
    meta.write_text('{"ok": true}', encoding="utf-8")
    temp_file = stage / "temp_download.mp4"
    temp_file.write_bytes(b"temporary")

    res = colab_runner.sync_job_directory_to_storage(
        str(stage), "job_abc", drive_root=str(drive), cleanup_temp=True
    )
    assert res["ok"] is True
    dest = drive / "job_abc"
    assert (dest / "clip_1.mp4").read_bytes() == clip.read_bytes()
    assert (dest / "processing_summary.json").read_text(encoding="utf-8") == '{"ok": true}'
    assert not (dest / "temp_download.mp4").exists()
    # Temp file cleaned from staging after verified copy
    assert not temp_file.exists()


def test_run_shorts_pipeline_local_file_and_corrupt_mp4_rejected(tmp_path):
    local_vid = tmp_path / "input_demo.mp4"
    local_vid.write_bytes(b"source-placeholder")

    corrupt_clip = tmp_path / "corrupt_clip.mp4"
    corrupt_clip.write_bytes(b"not a real mp4 file")

    mock_res = {
        "success": True,
        "video_title": "input_demo",
        "clips": [str(corrupt_clip)],
        "metadata": {},
    }
    with patch("main.process_video_pipeline", return_value=mock_res) as mock_main:
        out = colab_runner.run_shorts_pipeline(
            urls=[str(local_vid)],
            clips_per_video=1,
            subtitle_preset="none",
            range_start=5.0,
            range_end=35.0,
            output_dir=str(tmp_path / "out"),
        )
        # Local file path must pass is_url=False
        assert mock_main.called
        assert mock_main.call_args[1]["is_url"] is False
        # Corrupt MP4 must be rejected by real validate_media_output instead of faking success
        assert len(out["successful"]) == 0
        assert len(out["failed"]) == 1
        # Temporary env mutations cleaned up
        assert os.environ.get("RANGE_START") is None
        assert os.environ.get("RANGE_END") is None


def test_diagnose_peakcut_all_14_checks_and_token_scrubbing(tmp_path, monkeypatch):
    monkeypatch.setenv("PEAKCUT_LOG_DIR", str(tmp_path))
    secret = "super_secret_peakcut_token_xyz123"
    monkeypatch.setenv("PEAKCUT_ACCESS_TOKEN", secret)
    (tmp_path / "backend.log").write_text(
        f"Started server with ?token={secret} ok\n", encoding="utf-8"
    )

    with patch("engine.intelligence.llm_backend.check_local_llm_health", return_value={
        "ready": False, "reachable": False, "model_installed": False,
        "installed_models": [], "base_url": "http://127.0.0.1:11434/v1", "error": "offline",
    }), patch("colab_runner._http_get_ok", return_value=(False, 0, "offline")):
        report = colab_runner.diagnose_peakcut(port=8999, run_inference_check=False)

    assert len(report["checks"]) == 14
    for chk in report["checks"]:
        assert chk["status"] in ("PASS", "WARNING", "FAIL")
        assert secret not in chk["detail"]
    # Scrubbed from log excerpts too
    assert secret not in report["log_excerpts"].get("backend", "")
    assert "***" in report["log_excerpts"].get("backend", "")


def test_peakcut_colab_notebook_structure_and_python_syntax():
    import ast
    from pathlib import Path

    nb_path = Path(colab_runner.PROJECT_ROOT) / "PeakCut_Colab.ipynb"
    assert nb_path.is_file()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
    assert len(code_cells) == 11

    all_code = []
    for idx, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        ast.parse(src, filename=f"PeakCut_Colab.ipynb::code_{idx}")
        all_code.append(src)

    joined = "\n".join(all_code)
    # Must never use broad pkill -9
    assert "pkill -9" not in joined
    # Must guard git reset --hard behind ALLOW_DISCARD_LOCAL_CHANGES
    assert "ALLOW_DISCARD_LOCAL_CHANGES" in joined
    # Must include automated diagnostics cell
    assert "diagnose_peakcut" in joined
    # Must start backend before dashboard access
    assert "start_backend_server" in joined
    assert "start_dashboard_access" in joined


def test_cli_new_subcommands_diagnose_stop_and_serve_flags():
    parser = colab_runner.build_cli_parser()

    args_diag = parser.parse_args(["diagnose", "--port", "8123", "--test-inference"])
    assert args_diag.subcommand == "diagnose"
    assert args_diag.port == 8123
    assert args_diag.test_inference is True

    args_stop = parser.parse_args(["stop"])
    assert args_stop.subcommand == "stop"

    args_serve = parser.parse_args(["serve", "--port", "8123", "--host", "0.0.0.0", "--background", "--no-tunnel"])
    assert args_serve.port == 8123
    assert args_serve.host == "0.0.0.0"
    assert args_serve.background is True
    assert args_serve.no_tunnel is True

    args_clip = parser.parse_args(["clip", "demo.mp4", "--start", "10", "--end", "45", "--no-cleanup"])
    assert args_clip.start == 10.0
    assert args_clip.end == 45.0
    assert args_clip.no_cleanup is True

    with patch("colab_runner.launch_web_ui") as mock_launch:
        rc = colab_runner.cli_main(["serve", "--port", "8123", "--background", "--no-tunnel"])
        assert rc == 0
        assert mock_launch.call_args[1]["port"] == 8123
        assert mock_launch.call_args[1]["run_in_background"] is True
        assert mock_launch.call_args[1]["use_tunnel"] is False

    with patch("colab_runner.stop_managed_services", return_value={"backend": True}) as mock_stop:
        rc = colab_runner.cli_main(["stop"])
        assert rc == 0
        assert mock_stop.called
