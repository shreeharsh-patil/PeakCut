import json
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from youtube_uploader import (
    YouTubeAuthError,
    YouTubeUploadError,
    refresh_access_token,
    upload_youtube_short,
    register_youtube_dispatcher,
)
from publishing_queue import PostItem, PublishingQueue


def test_refresh_access_token_success():
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"access_token": "mock_token_123"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        token = refresh_access_token("mock_cid", "mock_sec", "mock_ref")
        assert token == "mock_token_123"


def test_refresh_access_token_missing_creds(monkeypatch):
    monkeypatch.delenv("YOUTUBE_CLIENT_ID", raising=False)
    monkeypatch.delenv("YOUTUBE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)

    with pytest.raises(YouTubeAuthError, match="Missing OAuth credentials"):
        refresh_access_token()


def test_upload_youtube_short_success(tmp_path: Path):
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"fake mp4 video stream content")

    init_resp = MagicMock()
    init_resp.headers = {"Location": "https://upload.youtube.com/resumable/12345"}
    init_resp.__enter__.return_value = init_resp

    put_resp = MagicMock()
    put_resp.read.return_value = json.dumps({"id": "v_test_9876"}).encode("utf-8")
    put_resp.__enter__.return_value = put_resp

    call_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Step 1: Init POST
            assert req.get_method() == "POST"
            # Verify #Shorts was injected
            data = json.loads(req.data.decode("utf-8"))
            assert "#Shorts" in data["snippet"]["title"]
            return init_resp
        else:
            # Step 2: PUT video data
            assert req.get_method() == "PUT"
            assert req.full_url == "https://upload.youtube.com/resumable/12345"
            return put_resp

    with patch("urllib.request.urlopen", side_effect=mock_urlopen):
        result = upload_youtube_short(
            video_path=dummy_video,
            title="My Epic Podcast Highlight",
            access_token="test_bearer_token",
            privacy_status="unlisted",
        )

        assert result["id"] == "v_test_9876"
        assert result["url"] == "https://youtube.com/shorts/v_test_9876"
        assert "#Shorts" in result["title"]
        assert result["privacy"] == "unlisted"


def test_upload_youtube_short_missing_file():
    with pytest.raises(FileNotFoundError):
        upload_youtube_short(
            video_path="non_existent_file.mp4",
            title="Clip",
            access_token="tok",
        )


def test_register_youtube_dispatcher(tmp_path: Path):
    queue = PublishingQueue(queue_file=tmp_path / "queue.json")
    register_youtube_dispatcher(queue)
    assert "youtube" in queue.post_dispatchers

    # Mock dispatcher execution
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"fake data")

    with patch("youtube_uploader.upload_youtube_short") as mock_upload:
        mock_upload.return_value = {
            "id": "yt_123",
            "url": "https://youtube.com/shorts/yt_123",
        }
        item = PostItem(
            id="p_1",
            job_id="j_1",
            clip_id="c_1",
            platform="youtube",
            account_id="acc_1",
            account_name="Channel 1",
            title="Test Post",
            description="Desc",
            video_path=str(dummy_video),
        )
        res = queue.post_dispatchers["youtube"](item)
        assert res["remote_id"] == "yt_123"
        assert res["remote_url"] == "https://youtube.com/shorts/yt_123"


def test_upload_youtube_short_dry_run(tmp_path: Path):
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"sample bytes")

    res = upload_youtube_short(
        video_path=dummy_video,
        title="Simulated Short",
        dry_run=True,
    )
    assert res["simulated"] is True
    assert res["id"].startswith("sim_")
    assert "https://youtube.com/shorts/" in res["url"]
    assert "#Shorts" in res["title"]


def test_upload_youtube_short_datetime_publish_at(tmp_path: Path):
    import datetime
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"sample bytes")

    scheduled_dt = datetime.datetime(2026, 12, 1, 15, 30, tzinfo=datetime.timezone.utc)
    res = upload_youtube_short(
        video_path=dummy_video,
        title="Future Short",
        publish_at=scheduled_dt,
        dry_run=True,
    )
    assert res["publish_at"] == "2026-12-01T15:30:00Z"
    assert res["privacy"] == "private"


def test_upload_youtube_short_network_offline(tmp_path: Path):
    dummy_video = tmp_path / "clip.mp4"
    dummy_video.write_bytes(b"sample bytes")

    import urllib.error
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network unreachable")):
        with pytest.raises(YouTubeUploadError, match="offline or unreachable"):
            upload_youtube_short(
                video_path=dummy_video,
                title="Offline Test",
                access_token="valid_token",
                dry_run=False,
            )
