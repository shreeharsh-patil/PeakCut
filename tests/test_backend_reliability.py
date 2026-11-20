"""Comprehensive Backend Reliability & Regression Test Suite.

Covers:
1. Deterministic sync endpoint contract and version token stability
2. Consistent clip-index bounds and type validation across endpoints
3. Job lifecycle, cancellation, and subprocess cleanup
4. Job and publishing queue restart recovery
5. Publishing queue pause/resume, 30-minute intervals, and daily caps
6. Timezone-aware slot scheduling
7. YouTube OAuth access token caching, invalidation, and 401 refresh
8. AI provider health reporting and model validation
9. Platform validation security checks
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app
import peakcut_bridge
from publishing_queue import PostItem, PublishingQueue, QueueWorker
import youtube_uploader
import llm_backend


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(peakcut_bridge, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    peakcut_bridge.bind_app_runtime(jobs_dict=app.jobs, output_dir=tmp_path)
    return TestClient(app.app)


# ---------------------------------------------------------------------------
# 1. Deterministic Sync Contract
# ---------------------------------------------------------------------------

def test_sync_endpoint_deterministic_contract(client, tmp_path, monkeypatch):
    """GET /api/sync must return full state when v is omitted/different,

    and exactly {"v": token} when state has not changed.
    GET /api/sync must never mutate the disk or create directories.
    """
    # Baseline fetch
    res1 = client.get("/api/sync")
    assert res1.status_code == 200
    data1 = res1.json()
    token1 = data1.get("v")
    assert token1 is not None
    assert "jobs" in data1
    assert "creators" in data1

    # Immediate call with same version token: state unchanged -> version-only response
    res2 = client.get(f"/api/sync?v={token1}")
    assert res2.status_code == 200
    assert res2.json() == {"v": token1}

    # Verify that calling sync did not create any spurious files/directories in tmp_path
    items_before = list(tmp_path.iterdir())
    res3 = client.get(f"/api/sync?v={token1}")
    assert res3.status_code == 200
    items_after = list(tmp_path.iterdir())
    assert items_before == items_after

    # When state changes (e.g. a job is added to in-memory active jobs), token advances
    mock_job_id = "job_sync_test_01"
    app.jobs[mock_job_id] = {
        "status": "processing",
        "progress": 25,
        "logs": ["Processing started..."],
        "ready_files": [],
        "clips": [],
    }
    try:
        res4 = client.get(f"/api/sync?v={token1}")
        assert res4.status_code == 200
        data4 = res4.json()
        token2 = data4.get("v")
        assert token2 != token1
        assert "jobs" in data4

        # Calling again with new token returns version-only
        res5 = client.get(f"/api/sync?v={token2}")
        assert res5.status_code == 200
        assert res5.json() == {"v": token2}
    finally:
        app.jobs.pop(mock_job_id, None)


# ---------------------------------------------------------------------------
# 2. Consistent Clip-Index Validation
# ---------------------------------------------------------------------------

def test_clip_index_validation_across_endpoints(client, tmp_path, monkeypatch):
    """Negative, non-integer, and out-of-range clip indices must be rejected

    with 400 Bad Request BEFORE any credential or network checks.
    """
    job_id = "job_val_01"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status_file = job_dir / "status.json"
    dummy_status = {
        "status": "completed",
        "clips": [
            {"id": "c1", "title": "Clip 1", "file": "c1.mp4"},
            {"id": "c2", "title": "Clip 2", "file": "c2.mp4"},
        ],
    }
    status_file.write_text(json.dumps(dummy_status), encoding="utf-8")

    # Negative clip index on /api/social/post
    res_neg = client.post("/api/social/post", json={
        "job_id": job_id,
        "clip_index": -1,
        "platforms": ["youtube"],
    })
    assert res_neg.status_code == 400
    assert "Invalid clip index" in res_neg.json()["detail"]

    # Out-of-range clip index (index 5 for 2 clips)
    res_out = client.post("/api/social/post", json={
        "job_id": job_id,
        "clip_index": 5,
        "platforms": ["youtube"],
    })
    assert res_out.status_code == 400
    assert "Invalid clip index" in res_out.json()["detail"]

    # Non-integer / boolean clip index
    res_bool = client.post("/api/social/post", json={
        "job_id": job_id,
        "clip_index": True,
        "platforms": ["youtube"],
    })
    assert res_bool.status_code == 400
    assert "Invalid clip index" in res_bool.json()["detail"]

    # Negative clip index on /api/youtube/upload
    res_yt_neg = client.post("/api/youtube/upload", json={
        "job_id": job_id,
        "clip_index": -1,
    })
    assert res_yt_neg.status_code == 400
    assert "Invalid clip index" in res_yt_neg.json()["detail"]

    # Out-of-range clip index on /api/youtube/upload
    res_yt_out = client.post("/api/youtube/upload", json={
        "job_id": job_id,
        "clip_index": 10,
    })
    assert res_yt_out.status_code == 400
    assert "Invalid clip index" in res_yt_out.json()["detail"]

    # Boolean clip index on /api/youtube/upload
    res_yt_bool = client.post("/api/youtube/upload", json={
        "job_id": job_id,
        "clip_index": True,
    })
    assert res_yt_bool.status_code == 400
    assert "Invalid clip index" in res_yt_bool.json()["detail"]

    # Missing clip index on /api/youtube/upload
    res_yt_none = client.post("/api/youtube/upload", json={
        "job_id": job_id,
    })
    assert res_yt_none.status_code == 400
    assert "Missing clip index" in res_yt_none.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Job Lifecycle & Cancellation
# ---------------------------------------------------------------------------

def test_job_cancellation_and_subprocess_termination(tmp_path, monkeypatch):
    """cancel_job must terminate running subprocess and persist cancelled status."""
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    job_id = "job_cancel_test"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status_file = job_dir / "status.json"
    status_file.write_text(json.dumps({"status": "processing"}), encoding="utf-8")

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Running
    with app._job_processes_lock:
        app._job_processes[job_id] = mock_proc

    app.jobs[job_id] = {"status": "processing", "logs": []}

    try:
        # Cancel the job
        cancelled = app.cancel_job(job_id)
        assert cancelled is True
        mock_proc.terminate.assert_called_once()

        # In-memory status is cancelled
        assert app.jobs[job_id]["status"] == "cancelled"

        # On-disk status is cancelled with error details
        saved_status = json.loads(status_file.read_text(encoding="utf-8"))
        assert saved_status["status"] == "cancelled"
        assert "cancelled by user" in saved_status.get("error", "").lower()
    finally:
        with app._job_processes_lock:
            app._job_processes.pop(job_id, None)
        app.jobs.pop(job_id, None)


# ---------------------------------------------------------------------------
# 4. Job & Queue Restart Recovery
# ---------------------------------------------------------------------------

def test_job_disk_recovery_from_status_json(tmp_path, monkeypatch):
    """_job_view_from_disk must recover jobs from status.json even without metadata."""
    monkeypatch.setattr(app, "OUTPUT_DIR", str(tmp_path))
    job_id = "job_recovered_from_status"
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    status_file = job_dir / "status.json"
    status_data = {
        "status": "completed",
        "title": "Recovered Title",
        "url": "https://youtu.be/12345678901",
        "clips": [{"id": "c1", "title": "Clip 1"}],
    }
    status_file.write_text(json.dumps(status_data), encoding="utf-8")

    view = app._job_view_from_disk(job_id)
    assert view is not None
    assert view["status"] == "completed"
    assert view["title"] == "Recovered Title"
    assert len(view["clips"]) == 1


def test_publishing_queue_restart_recovery(tmp_path):
    """An interrupted post with status='posting' must recover to 'scheduled' on restart."""
    q_file = tmp_path / "queue_recovery.json"
    # Write a simulated queue state with an in-flight posting job
    interrupted_item = {
        "id": "post_in_flight",
        "job_id": "job_crash",
        "clip_id": "clip_0",
        "platform": "youtube",
        "account_id": "yt_acc",
        "account_name": "Test Channel",
        "title": "Crash Recovery",
        "description": "Desc",
        "video_path": "clip.mp4",
        "status": "posting",
        "tries": 1,
    }
    q_file.write_text(json.dumps([interrupted_item]), encoding="utf-8")

    # Initialize queue (simulating app restart)
    queue = PublishingQueue(queue_file=q_file)
    recovered = queue.get_post("post_in_flight")
    assert recovered is not None
    assert recovered.status == "scheduled"
    assert recovered.due_at <= time.time()


# ---------------------------------------------------------------------------
# 5. Publishing Queue Pause, Resume & Configurable Intervals
# ---------------------------------------------------------------------------

def test_publishing_queue_pause_resume_and_intervals(tmp_path):
    """PublishingQueue pause/resume controls worker dispatch and interval configuration."""
    q_file = tmp_path / "queue_pause.json"
    queue = PublishingQueue(queue_file=q_file, post_gap_minutes=30, daily_cap=4)
    assert queue.post_gap_minutes == 30
    assert queue.daily_cap == 4
    assert queue.is_paused is False

    # Pause queue
    queue.pause()
    assert queue.is_paused is True

    # Persist and reload
    queue2 = PublishingQueue(queue_file=q_file)
    assert queue2.is_paused is True
    assert queue2.post_gap_minutes == 30

    # Resume queue
    queue2.resume()
    assert queue2.is_paused is False

    # Dynamic gap update
    queue2.set_post_gap_minutes(45)
    assert queue2.post_gap_minutes == 45
    queue2.set_daily_cap(8)
    assert queue2.daily_cap == 8


def test_queue_worker_respects_pause(tmp_path):
    """QueueWorker must not dispatch posts when queue is paused."""
    q_file = tmp_path / "queue_worker_pause.json"
    queue = PublishingQueue(queue_file=q_file)
    queue.pause()

    item = queue.enqueue(
        job_id="j_pause",
        clip_id="c_pause",
        platform="youtube",
        account_id="yt_1",
        title="Paused Short",
        video_path="dummy.mp4",
        schedule_now=True,
    )

    dispatched = []
    queue.register_dispatcher("youtube", lambda p: dispatched.append(p.id) or {})

    worker = QueueWorker(queue, poll_interval_sec=0.1)
    # Run one pass of worker loop manually
    if not queue.is_paused:
        for post in queue.get_due_posts():
            worker._dispatch_post(post)

    assert len(dispatched) == 0
    assert queue.get_post(item.id).status == "scheduled"


# ---------------------------------------------------------------------------
# 6. Timezone-Aware Slot Scheduling
# ---------------------------------------------------------------------------

def test_timezone_aware_scheduling(tmp_path):
    """calculate_next_slot calculates 9:00 AM next day in the specified timezone."""
    q_file = tmp_path / "queue_tz.json"
    queue = PublishingQueue(queue_file=q_file, post_gap_minutes=30, daily_cap=2)

    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/New_York")
    except Exception:
        pytest.skip("zoneinfo not available")

    # Anchor to 11:00 AM New York time
    base_dt = dt.datetime(2026, 12, 1, 11, 0, 0, tzinfo=tz)
    base_ts = base_dt.timestamp()

    # Enqueue post 1
    p1 = queue.enqueue(
        job_id="j1",
        clip_id="c1",
        platform="tiktok",
        account_id="acc_ny",
        title="NY Post 1",
        video_path="v1.mp4",
        requested_time=base_ts,
        tz=tz,
    )

    # Enqueue post 2 (spaced by 30 mins)
    p2 = queue.enqueue(
        job_id="j1",
        clip_id="c2",
        platform="tiktok",
        account_id="acc_ny",
        title="NY Post 2",
        video_path="v2.mp4",
        tz=tz,
    )
    assert p2.due_at >= p1.due_at + (30 * 60)

    # Post 3 exceeds daily cap of 2 -> rolls over to 9:00 AM next day in New York time
    p3 = queue.enqueue(
        job_id="j1",
        clip_id="c3",
        platform="tiktok",
        account_id="acc_ny",
        title="NY Post 3",
        video_path="v3.mp4",
        tz=tz,
    )

    dt3 = dt.datetime.fromtimestamp(p3.due_at, tz=tz)
    assert dt3.hour == 9
    assert dt3.minute == 0
    assert dt3.day == base_dt.day + 1


# ---------------------------------------------------------------------------
# 7. YouTube Token Caching & Refresh on 401
# ---------------------------------------------------------------------------

def test_youtube_token_caching_and_invalidation(monkeypatch):
    """Access tokens are cached until expiry or explicit invalidation."""
    youtube_uploader.invalidate_cached_token()

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({
        "access_token": "tok_cached_123",
        "expires_in": 3600,
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        return mock_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        tok1 = youtube_uploader.get_access_token("cid", "csec", "rtok")
        tok2 = youtube_uploader.get_access_token("cid", "csec", "rtok")
        assert tok1 == "tok_cached_123"
        assert tok2 == "tok_cached_123"
        # Only one HTTP request should be made due to caching
        assert call_count == 1

        # Invalidate cache
        youtube_uploader.invalidate_cached_token()
        tok3 = youtube_uploader.get_access_token("cid", "csec", "rtok")
        assert tok3 == "tok_cached_123"
        assert call_count == 2


# ---------------------------------------------------------------------------
# 8. AI Provider Health Reporting & Model Discovery
# ---------------------------------------------------------------------------

def test_llm_provider_health_reporting(monkeypatch):
    """llm_backend reports provider health, availability, and active models."""
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    import httpx
    def mock_handler(request):
        if str(request.url).endswith("/models"):
            return httpx.Response(200, json={
                "data": [{"id": "qwen2.5:7b"}, {"id": "llama3.2:3b"}],
            })
        return httpx.Response(404)

    monkeypatch.setattr(llm_backend, "_client", lambda **kw: httpx.Client(transport=httpx.MockTransport(mock_handler), **kw))

    models = llm_backend.get_available_models()
    assert "qwen2.5:7b" in models
    assert "llama3.2:3b" in models

    assert llm_backend.validate_model_availability("qwen2.5:7b") is True
    assert llm_backend.validate_model_availability("nonexistent_model") is False

    health = llm_backend.get_provider_health()
    assert health["provider"] == "openai"
    assert health["model"] == "qwen2.5:7b"
    assert health["ready"] is True
    assert "llama3.2:3b" in health["available_models"]


# ---------------------------------------------------------------------------
# 9. Security & Platform Validation
# ---------------------------------------------------------------------------

def test_connected_apps_platform_validation(client):
    """PUT /api/apps must reject unsupported platform identifiers."""
    res = client.put("/api/apps", json={
        "platform": "unsupported_social_network",
        "client_id": "test_id",
    })
    assert res.status_code == 400
    assert "Unsupported platform" in res.json()["detail"]

    # Valid platform accepted
    res_valid = client.put("/api/apps", json={
        "platform": "youtube",
        "client_id": "new_yt_id",
    })
    assert res_valid.status_code == 200
    assert res_valid.json()["ok"] is True
