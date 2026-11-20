import os
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import app as app_module
from engine.services.youtube_uploader import (
    is_youtube_auth_configured,
    get_youtube_readiness_status,
)


@pytest.fixture
def client():
    return TestClient(app_module.app)


def test_youtube_readiness_endpoint_unconfigured(client, monkeypatch):
    monkeypatch.delenv("YOUTUBE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_ID", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)
    monkeypatch.setenv("YOUTUBE_DEFAULT_PRIVACY", "private")

    res = client.get("/api/youtube/readiness")
    assert res.status_code == 200
    data = res.json()
    assert data["youtube_ready"] is False
    assert data["direct_youtube"]["ready"] is False
    assert data["upload_post"]["ready"] is False
    assert data["default_privacy"] == "private"
    assert "setup_help" in data


def test_youtube_readiness_endpoint_with_direct_token(client, monkeypatch):
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "mock_access_token_abc")
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)

    res = client.get("/api/youtube/readiness")
    assert res.status_code == 200
    data = res.json()
    assert data["youtube_ready"] is True
    assert data["direct_youtube"]["ready"] is True
    assert data["direct_youtube"]["has_access_token"] is True


def test_social_post_youtube_missing_all_credentials_informative_error(client, monkeypatch):
    monkeypatch.delenv("YOUTUBE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_ID", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)

    app_module.jobs["mock_job_1"] = {
        "status": "completed",
        "result": {
            "clips": [
                {"title": "Test Clip", "video_url": "/videos/mock_job_1/clip_0.mp4"}
            ]
        }
    }

    payload = {
        "job_id": "mock_job_1",
        "clip_index": 0,
        "platforms": ["youtube"],
    }

    res = client.post("/api/social/post", json=payload)
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert "YouTube upload is not ready" in detail
    assert "configure direct YouTube OAuth" in detail
    assert "configure Upload-Post" in detail


def test_social_post_youtube_direct_fallback_with_custom_privacy(client, monkeypatch, tmp_path):
    monkeypatch.delenv("UPLOAD_POST_API_KEY", raising=False)
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "valid_mock_token")

    # Create dummy video file
    job_dir = tmp_path / "mock_job_2"
    job_dir.mkdir(parents=True)
    video_file = job_dir / "clip_0.mp4"
    video_file.write_bytes(b"dummy mp4 data")

    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))

    app_module.jobs["mock_job_2"] = {
        "status": "completed",
        "result": {
            "clips": [
                {"title": "Clip Title", "video_url": "/videos/mock_job_2/clip_0.mp4"}
            ]
        }
    }

    mock_upload_res = {
        "id": "yt_video_12345",
        "url": "https://youtube.com/shorts/yt_video_12345",
        "privacy": "unlisted",
        "title": "Clip Title #Shorts",
    }

    with patch("engine.services.youtube_uploader.upload_youtube_short", return_value=mock_upload_res) as mock_upload:
        payload = {
            "job_id": "mock_job_2",
            "clip_index": 0,
            "platforms": ["youtube"],
            "privacy_status": "unlisted",
        }

        res = client.post("/api/social/post", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["method"] == "direct_youtube"
        assert data["id"] == "yt_video_12345"
        assert data["privacy"] == "unlisted"

        # Verify privacy was passed to upload_youtube_short
        mock_upload.assert_called_once()
        _, kwargs = mock_upload.call_args
        assert kwargs["privacy_status"] == "unlisted"


def test_social_post_youtube_upload_post_respects_privacy_setting(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_POST_API_KEY", "mock_upload_post_key")

    job_dir = tmp_path / "mock_job_3"
    job_dir.mkdir(parents=True)
    video_file = job_dir / "clip_0.mp4"
    video_file.write_bytes(b"dummy mp4 data")

    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))

    app_module.jobs["mock_job_3"] = {
        "status": "completed",
        "result": {
            "clips": [
                {"title": "Clip 3", "video_url": "/videos/mock_job_3/clip_0.mp4"}
            ]
        }
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "up_post_888", "status": "queued"}

    with patch.object(app_module.httpx, "Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        payload = {
            "job_id": "mock_job_3",
            "clip_index": 0,
            "user_id": "test_user_profile",
            "platforms": ["youtube"],
            "privacy_status": "private",
        }

        res = client.post("/api/social/post", json=payload)
        assert res.status_code == 200

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args.kwargs
        data_sent = call_kwargs["data"]
        # CRITICAL TEST: Must be 'private', NEVER hardcoded 'public'!
        assert data_sent["privacyStatus"] == "private"


def test_direct_youtube_upload_endpoint(client, monkeypatch, tmp_path):
    monkeypatch.setenv("YOUTUBE_ACCESS_TOKEN", "mock_direct_token")

    dummy_video = tmp_path / "standalone_video.mp4"
    dummy_video.write_bytes(b"dummy mp4 video bytes")

    mock_res = {
        "id": "direct_yt_999",
        "url": "https://youtube.com/shorts/direct_yt_999",
        "title": "Standalone Short #Shorts",
        "privacy": "private",
    }

    with patch("engine.services.youtube_uploader.upload_youtube_short", return_value=mock_res) as mock_upload:
        payload = {
            "video_path": str(dummy_video),
            "title": "Standalone Short",
            "description": "Awesome standalone clip",
            "privacy_status": "private",
        }

        res = client.post("/api/youtube/upload", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["id"] == "direct_yt_999"
        assert data["privacy"] == "private"

        mock_upload.assert_called_once()
        _, kwargs = mock_upload.call_args
        assert kwargs["privacy_status"] == "private"
        assert kwargs["access_token"] == "mock_direct_token"


def test_upload_post_mock_mode_social_post(client, monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_POST_API_KEY", "mock")
    monkeypatch.setenv("UPLOAD_POST_PROFILE", "test_channel")

    job_dir = tmp_path / "mock_job_sandbox"
    job_dir.mkdir(parents=True)
    video_file = job_dir / "clip_0.mp4"
    video_file.write_bytes(b"dummy mp4 data")

    monkeypatch.setattr(app_module, "OUTPUT_DIR", str(tmp_path))

    app_module.jobs["mock_job_sandbox"] = {
        "status": "completed",
        "result": {
            "clips": [
                {"title": "Sandbox Clip", "video_url": "/videos/mock_job_sandbox/clip_0.mp4"}
            ]
        }
    }

    payload = {
        "job_id": "mock_job_sandbox",
        "clip_index": 0,
        "platforms": ["youtube", "tiktok"],
        "privacy_status": "unlisted",
    }

    res = client.post("/api/social/post", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["test_mode"] is True
    assert data["privacyStatus"] == "unlisted"


def test_social_user_mock_mode(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_POST_API_KEY", "mock")
    monkeypatch.setenv("UPLOAD_POST_PROFILE", "my_verified_channel")

    res = client.get("/api/social/user")
    assert res.status_code == 200
    data = res.json()
    assert len(data["profiles"]) > 0
    assert data["profiles"][0]["username"] == "my_verified_channel"
    assert "youtube" in data["profiles"][0]["connected"]


def test_config_endpoint_upload_post_state(client, monkeypatch):
    monkeypatch.setenv("UPLOAD_POST_API_KEY", "mock")
    monkeypatch.setenv("YOUTUBE_DEFAULT_PRIVACY", "unlisted")

    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert data["uploadPostConfigured"] is True
    assert data["uploadPostMock"] is True
    assert data["defaultPrivacy"] == "unlisted"


def test_upload_youtube_short_publish_at(tmp_path, monkeypatch):
    from engine.services.youtube_uploader import upload_youtube_short

    test_video = tmp_path / "clip.mp4"
    test_video.write_bytes(b"dummy_mp4_bytes")

    session_url = "https://www.googleapis.com/upload/youtube/v3/videos?upload_id=session123"

    mock_init_resp = MagicMock()
    mock_init_resp.headers = {"Location": session_url}
    mock_init_resp.__enter__.return_value = mock_init_resp

    mock_upload_resp = MagicMock()
    mock_upload_resp.read.return_value = json.dumps({
        "id": "vid_sched_123",
        "snippet": {"title": "Scheduled Viral Clip #Shorts"},
        "status": {"privacyStatus": "private", "publishAt": "2026-11-19T10:00:00Z"}
    }).encode("utf-8")
    mock_upload_resp.__enter__.return_value = mock_upload_resp

    captured_requests = []
    def fake_urlopen(req, timeout=30):
        captured_requests.append(req)
        if len(captured_requests) == 1:
            return mock_init_resp
        return mock_upload_resp

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = upload_youtube_short(
        video_path=test_video,
        title="Scheduled Viral Clip",
        publish_at="2026-11-19T10:00:00Z",
        access_token="fake_token",
        dry_run=False,
    )

    assert result["id"] == "vid_sched_123"
    assert result["publish_at"] == "2026-11-19T10:00:00Z"
    assert result["privacy"] == "private"
    assert len(captured_requests) == 2

    # Check init metadata payload
    init_req = captured_requests[0]
    payload = json.loads(init_req.data.decode("utf-8"))
    assert payload["status"]["privacyStatus"] == "private"
    assert payload["status"]["publishAt"] == "2026-11-19T10:00:00Z"


def test_social_post_negative_clip_index_rejected(client):
    app_module.jobs["mock_job_neg"] = {
        "status": "completed",
        "result": {
            "clips": [
                {"title": "Test Clip", "video_url": "/videos/mock_job_neg/clip_0.mp4"}
            ]
        }
    }
    payload = {
        "job_id": "mock_job_neg",
        "clip_index": -1,
        "platforms": ["youtube"],
    }
    res = client.post("/api/social/post", json=payload)
    assert res.status_code == 400
    assert "Invalid clip index" in res.json()["detail"]


def test_youtube_upload_negative_clip_index_rejected(client):
    app_module.jobs["mock_job_neg_yt"] = {
        "status": "completed",
        "result": {
            "clips": [
                {"title": "Test Clip", "video_url": "/videos/mock_job_neg_yt/clip_0.mp4"}
            ]
        }
    }
    payload = {
        "job_id": "mock_job_neg_yt",
        "clip_index": -1,
    }
    res = client.post("/api/youtube/upload", json=payload)
    assert res.status_code == 400
    assert "Invalid clip index" in res.json()["detail"]


def test_publishing_queue_privacy_defaults_and_remote_keys(tmp_path):
    from engine.services.publishing_queue import PublishingQueue, QueueWorker

    q_file = tmp_path / "queue_priv.json"
    queue = PublishingQueue(queue_file=q_file)

    # 1. Enqueue without privacy metadata
    item = queue.enqueue(
        job_id="job_priv",
        clip_id="clip_0",
        platform="youtube",
        account_id="yt_safe",
        title="Safe Defaults Test",
        video_path="clip.mp4",
        schedule_now=True,
    )
    assert item.metadata.get("privacy") in ("unlisted", "private")
    assert item.metadata.get("privacy_status") in ("unlisted", "private")

    # 2. Dispatcher returning remote_url / remote_id
    def custom_yt_dispatcher(post):
        return {
            "remote_id": "vid_abc_999",
            "remote_url": "https://youtube.com/shorts/vid_abc_999",
        }

    queue.register_dispatcher("youtube", custom_yt_dispatcher)
    worker = QueueWorker(queue)
    worker._dispatch_post(item)

    updated = queue.get_post(item.id)
    assert updated.status == "posted"
    assert updated.remote_id == "vid_abc_999"
    assert updated.remote_url == "https://youtube.com/shorts/vid_abc_999"




