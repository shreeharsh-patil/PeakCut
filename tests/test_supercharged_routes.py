"""Integration tests for Supercharged REST Endpoints and New MCP Tools."""
import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import app as app_module
from engine.studio import mcp_server
from engine.discovery.live_clipper import LiveClipperManager
from engine.discovery.niche_discovery import DiscoveredVideo


def _post_mcp(payload):
    async def _run():
        import httpx
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/mcp", json=payload)
    return asyncio.run(_run())


def _call_api(method: str, path: str, json_body: dict = None, params: dict = None):
    async def _run():
        import httpx
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            if method.upper() == "GET":
                return await client.get(path, params=params)
            else:
                return await client.post(path, json=json_body, params=params)
    return asyncio.run(_run())


class TestSuperchargedApi:
    def test_live_routes(self):
        # 1. Start live session (mocked)
        with patch.object(LiveClipperManager, "create_session") as mock_create:
            mock_sess = MagicMock()
            mock_sess.session_id = "test_sess_api"
            mock_sess.stream_url = "https://twitch.tv/test_stream"
            mock_sess.buffer_minutes = 15
            mock_create.return_value = mock_sess

            resp = _call_api("POST", "/api/live/start", json_body={
                "stream_url": "https://twitch.tv/test_stream",
                "buffer_minutes": 15,
                "session_id": "test_sess_api",
            })
            assert resp.status_code == 200
            assert resp.json()["status"] == "started"
            assert resp.json()["session_id"] == "test_sess_api"

        # 2. List sessions
        resp = _call_api("GET", "/api/live/sessions")
        assert resp.status_code == 200
        assert "sessions" in resp.json()

        # 3. Get session status by path param and sub-path
        with patch.object(LiveClipperManager, "get_session") as mock_get:
            mock_sess = MagicMock()
            mock_sess.get_status.return_value = {"session_id": "test_sess_api", "is_running": True}
            mock_sess.lock = MagicMock()
            mock_sess.total_samples = 32000
            mock_event = MagicMock()
            mock_event.clip_id = "clip_live_01"
            mock_event.duration = 10.0
            mock_sess.trigger_clip.return_value = mock_event
            mock_get.return_value = mock_sess

            resp_status1 = _call_api("GET", "/api/live/test_sess_api")
            assert resp_status1.status_code == 200
            assert resp_status1.json()["session_id"] == "test_sess_api"

            resp_status2 = _call_api("GET", "/api/live/test_sess_api/status")
            assert resp_status2.status_code == 200

            # 4. Extract clip via path param
            resp_clip = _call_api("POST", "/api/live/test_sess_api/clip", json_body={"start_sec": 0, "end_sec": 10})
            assert resp_clip.status_code == 200
            assert resp_clip.json()["clip_id"] == "clip_live_01"

            # 5. Stop session by path param
            with patch.object(LiveClipperManager, "stop_session", return_value=True):
                resp_stop = _call_api("POST", "/api/live/test_sess_api/stop")
                assert resp_stop.status_code == 200
                assert resp_stop.json()["status"] == "stopped"

    def test_discover_routes(self):
        with patch("engine.discovery.niche_discovery.NicheDiscoveryEngine.search_youtube_api") as mock_search:
            mock_search.return_value = [
                DiscoveredVideo(
                    video_id="vid_101",
                    title="Amazing Tech Talk",
                    url="https://youtube.com/watch?v=vid_101",
                    channel_title="Tech Channel",
                    duration_seconds=600,
                    view_count=50000,
                    published_at="2026-11-10T00:00:00Z",
                )
            ]
            resp = _call_api("GET", "/api/discover/search", params={"query": "AI Tech"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_found"] == 1
            assert data["unseen_candidates"][0]["video_id"] == "vid_101"

    def test_channel_watcher_routes(self):
        resp = _call_api("POST", "/api/channels/watch", json_body={
            "platform": "youtube",
            "channel_identifier": "@lexfridman",
            "creator_name": "Lex Fridman",
            "mode": "vod",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "watching"
        assert resp.json()["channel"]["platform"] == "youtube"

        list_resp = _call_api("GET", "/api/channels/list")
        assert list_resp.status_code == 200
        assert any(ch["id"] == "youtube:@lexfridman" for ch in list_resp.json()["channels"])

        unwatch_resp = _call_api("POST", "/api/channels/unwatch", json_body={
            "channel_key": "youtube:@lexfridman"
        })
        assert unwatch_resp.status_code == 200
        assert unwatch_resp.json()["status"] == "removed"

    def test_publishing_queue_routes(self):
        resp = _call_api("POST", "/api/queue/enqueue", json_body={
            "job_id": "job_99",
            "clip_id": "clip_01",
            "platform": "youtube",
            "account_id": "yt_main",
            "title": "Super Short",
            "video_path": "output/clip_01.mp4",
            "schedule_now": True,
        })
        assert resp.status_code == 200
        post = resp.json()["post"]
        assert post["job_id"] == "job_99"
        assert post["status"] == "scheduled"
        post_id = post["id"]

        # Check list
        list_resp = _call_api("GET", "/api/queue/list", params={"account_id": "yt_main"})
        assert list_resp.status_code == 200
        assert any(p["id"] == post_id for p in list_resp.json()["posts"])

        # Preview slot
        slot_resp = _call_api("GET", "/api/queue/slot", params={"account_id": "yt_main", "platform": "youtube"})
        assert slot_resp.status_code == 200
        assert "next_slot_timestamp" in slot_resp.json()

        # Cancel
        cancel_resp = _call_api("POST", "/api/queue/cancel", json_body={"post_id": post_id})
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"


class TestNewMcpTools:
    def test_tools_list_contains_new_tools(self):
        resp = _post_mcp({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        assert resp.status_code == 200
        tools = resp.json()["result"]["tools"]
        tool_names = {t["name"] for t in tools}

        expected = {
            "discover_trending_videos",
            "watch_channel",
            "start_live_clipper",
            "schedule_publish",
            "detect_audio_highlights",
        }
        for name in expected:
            assert name in tool_names, f"Missing MCP tool: {name}"

    def test_call_watch_channel_mcp(self):
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "watch_channel",
                "arguments": {
                    "platform": "twitch",
                    "channel_identifier": "ninja",
                    "mode": "live",
                },
            },
        }
        resp = _post_mcp(req)
        assert resp.status_code == 200
        data = resp.json()
        assert not data["result"].get("isError")
        text = data["result"]["content"][0]["text"]
        parsed = json.loads(text)
        assert parsed["status"] == "watching"
        assert parsed["channel"]["platform"] == "twitch"
