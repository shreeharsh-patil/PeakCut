"""Comprehensive verification test suite for all PeakCut Web UI interactions and endpoints.

Verifies all 31 endpoints audited by @analyzer-peak:
- Job submission & link preview
- Direct and chunked file upload pipeline
- Live stream recording, clipping, and stopping
- Clip editing, rerendering, bulk zip export, and deletion
- Creator workspace updates (PUT /api/creators/{cid})
- Publishing queue, accounts, and apps
- AI provider settings, model listing, key testing, and usage analytics
- Notifications and authentication stubs
"""
import asyncio
import io
import json
import pytest
import httpx
import app as app_module
import api_key_rotator


def _async_client():
    transport = httpx.ASGITransport(app=app_module.app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_ui_preview_and_job_submission():
    async def _test():
        async with _async_client() as client:
            # 1. Preview link
            r = await client.get("/api/preview", params={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"})
            assert r.status_code == 200
            data = r.json()
            assert "title" in data
            assert "uploader" in data
            assert "duration" in data
            assert "is_live" in data

            # 2. Hero job creation from URL
            job_payload = {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "options": {"style": "hype", "layout": "track", "max_clips": 3},
                "creator": "creator_test",
            }
            r_job = await client.post("/api/jobs", json=job_payload)
            assert r_job.status_code == 200
            jdata = r_job.json()
            assert "id" in jdata
            assert jdata["status"] in ("queued", "running", "done")
            jid = jdata["id"]

            # 3. GET /api/jobs/{jid}
            r_get = await client.get(f"/api/jobs/{jid}")
            assert r_get.status_code == 200
            assert r_get.json()["id"] == jid

            # 4. DELETE /api/jobs/{jid}
            r_del = await client.delete(f"/api/jobs/{jid}")
            assert r_del.status_code == 200
            assert r_del.json()["status"] == "deleted"

    asyncio.run(_test())


def test_ui_chunked_and_direct_uploads():
    async def _test():
        async with _async_client() as client:
            # 1. Direct single-part upload
            dummy_mp4 = b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2mp41"
            files = {"file": ("sample.mp4", dummy_mp4, "video/mp4")}
            data = {"options": json.dumps({"style": "bounce"}), "creator": "local"}
            r_up = await client.post("/api/upload", files=files, data=data)
            assert r_up.status_code == 200
            j = r_up.json()
            assert "id" in j

            # 2. Chunked upload flow
            # start
            r_start = await client.post(
                "/api/upload/start",
                json={"name": "big_recording.mp4", "size": 1024, "options": {}, "creator": "test"},
            )
            assert r_start.status_code == 200
            start_info = r_start.json()
            upid = start_info["id"]
            assert start_info["parts"] >= 1

            # part 0
            chunk = b"X" * 1024
            r_part = await client.put(f"/api/upload/{upid}/0", content=chunk)
            assert r_part.status_code == 200
            assert r_part.json()["ok"] is True

            # finish
            r_fin = await client.post(f"/api/upload/{upid}/finish")
            assert r_fin.status_code == 200
            fin_job = r_fin.json()
            assert "id" in fin_job

            # cancel / delete
            r_canc = await client.delete(f"/api/upload/{upid}")
            assert r_canc.status_code == 200
            assert r_canc.json()["ok"] is True

    asyncio.run(_test())


def test_ui_live_stream_controls():
    async def _test():
        async with _async_client() as client:
            # Start creator live
            r_live = await client.post("/api/creators/mock_streamer/live", json={"channel": "https://twitch.tv/ninja"})
            assert r_live.status_code == 200
            data = r_live.json()
            assert data["is_live"] is True
            jid = data["id"]

            # Trigger clip
            r_clip = await client.post(f"/api/live/{jid}/clip", json={"seconds": 15})
            assert r_clip.status_code == 200
            assert r_clip.json()["ok"] is True

            # Stop live
            r_stop = await client.post(f"/api/live/{jid}/stop")
            assert r_stop.status_code == 200
            assert r_stop.json()["ok"] is True

    asyncio.run(_test())


def test_ui_clip_editing_and_bulk_actions():
    async def _test():
        async with _async_client() as client:
            # Create a mock job folder
            from pathlib import Path
            job_id = "test_edit_job_001"
            jdir = Path("output") / job_id
            jdir.mkdir(parents=True, exist_ok=True)
            clip_file = jdir / "c_01.mp4"
            clip_file.write_bytes(b"\x00" * 100)
            status_file = jdir / "status.json"
            status_file.write_text(json.dumps({
                "id": job_id,
                "status": "done",
                "clips": [{"id": "c_01", "start": 0, "end": 30, "title": "Old Title", "hook": "Old Hook"}],
            }), encoding="utf-8")

            # 1. Remake clip
            remake_body = {
                "start": 5.0,
                "end": 25.0,
                "hook": "New Viral Hook",
                "title": "New Title",
                "style": "clean",
                "layout": "split",
            }
            r_rem = await client.post(f"/api/jobs/{job_id}/clips/c_01/render", json=remake_body)
            assert r_rem.status_code == 200
            jdata = r_rem.json()
            assert jdata["clips"][0]["hook"] == "New Viral Hook"

            # 2. Clips ZIP download
            r_zip = await client.get("/api/clips.zip", params={"ids": f"{job_id}.c_01"})
            assert r_zip.status_code == 200
            assert "application/zip" in r_zip.headers.get("content-type", "")
            assert len(r_zip.content) > 0

            # 3. Bulk delete
            r_bulk = await client.post("/api/bulk/delete", json={"videos": [job_id], "clips": [[job_id, "c_01"]]})
            assert r_bulk.status_code == 200
            assert r_bulk.json()["status"] == "deleted"

    asyncio.run(_test())


def test_ui_creator_workspace_put():
    async def _test():
        async with _async_client() as client:
            cid = "creator_put_test"
            try:
                # Initial post
                await client.post("/api/creators", json={"name": "Creator Put Test", "channel": "https://youtube.com/c/test"})

                # PUT update
                put_payload = {
                    "name": "Creator Put Updated",
                    "rate_per_1k": 4.5,
                    "ai_every_min": 30,
                    "auto_watch": True,
                }
                r_put = await client.put(f"/api/creators/{cid}", json=put_payload)
                assert r_put.status_code == 200
                c = r_put.json()
                assert c["rate_per_1k"] == 4.5
                assert c["ai_every_min"] == 30
            finally:
                await client.delete(f"/api/creators/{cid}")

    asyncio.run(_test())


def test_ui_publishing_accounts_and_apps():
    async def _test():
        async with _async_client() as client:
            # 1. Accounts list
            r_accs = await client.get("/api/accounts")
            assert r_accs.status_code == 200
            assert isinstance(r_accs.json(), list)

            # 2. Apps get & put
            r_apps = await client.get("/api/apps")
            assert r_apps.status_code == 200
            r_put_app = await client.put("/api/apps", json={"platform": "youtube", "client_id": "test_id", "client_secret": "test_sec"})
            assert r_put_app.status_code == 200

            # 3. Schedule post
            r_post = await client.post("/api/posts", json={"job": "job1", "clip": "c1", "accounts": ["yt_main"], "now": True})
            assert r_post.status_code == 200
            assert r_post.json()["ok"] is True

            # 4. Cancel & retry
            r_canc = await client.post("/api/posts/p1/cancel")
            assert r_canc.status_code == 200
            r_retry = await client.post("/api/posts/p1/retry")
            assert r_retry.status_code == 200

            # 5. Remove account
            r_del_acc = await client.delete("/api/accounts/yt_main")
            assert r_del_acc.status_code == 200

    asyncio.run(_test())


def test_ui_ai_settings_usage_and_notifications():
    async def _test():
        async with _async_client() as client:
            # 1. AI Settings GET & PUT
            r_ai = await client.get("/api/ai")
            assert r_ai.status_code == 200
            assert "providers" in r_ai.json()

            r_put_ai = await client.put("/api/ai", json={"provider": "gemini", "key": "test_key_1, test_key_2", "models": "gemini-2.5-flash"})
            assert r_put_ai.status_code == 200

            # 2. AI Models list & Test
            r_models = await client.post("/api/ai/models", json={"provider": "gemini"})
            assert r_models.status_code == 200
            assert len(r_models.json()["models"]) > 0

            r_test_ai = await client.post("/api/ai/test")
            assert r_test_ai.status_code == 200
            assert r_test_ai.json()["ok"] is True

            # 3. AI Usage analytics
            r_usage = await client.get("/api/usage")
            assert r_usage.status_code == 200
            usage_data = r_usage.json()
            assert "daily" in usage_data
            assert "by_purpose" in usage_data
            assert "recent" in usage_data

            # 4. Notifications
            r_nt = await client.get("/api/notifications")
            assert r_nt.status_code == 200
            assert "unread" in r_nt.json()

            r_nt_read = await client.post("/api/notifications/read")
            assert r_nt_read.status_code == 200
            assert r_nt_read.json()["ok"] is True

            # 5. Settings PUT
            r_put_set = await client.put("/api/settings", json={"gemini_model": "gemini-3.1-flash-lite"})
            assert r_put_set.status_code == 200

            # 6. Auth stubs
            assert (await client.post("/api/auth/logout")).status_code == 200
            assert (await client.post("/api/auth/password", json={"new": "12345678"})).status_code == 200
            assert (await client.post("/api/auth/resend-verification")).status_code == 200
            assert (await client.post("/api/account/delete", json={"confirm": "DELETE"})).status_code == 200

    asyncio.run(_test())


def test_ui_pages_and_auth_me():
    async def _test():
        async with _async_client() as client:
            # 1. GET /api/auth/me
            r_me = await client.get("/api/auth/me")
            assert r_me.status_code == 200
            assert "user" in r_me.json()
            assert r_me.json()["user"]["is_admin"] is True

            # 2. Login & Signup HTML pages
            r_login = await client.get("/login")
            assert r_login.status_code == 200
            assert "text/html" in r_login.headers.get("content-type", "")

            r_signup = await client.get("/signup")
            assert r_signup.status_code == 200

            # 3. Reset password & Admin pages
            r_reset = await client.get("/reset")
            assert r_reset.status_code == 200

            r_admin = await client.get("/admin")
            assert r_admin.status_code == 200

            # 4. Legal pages
            r_terms = await client.get("/terms")
            assert r_terms.status_code == 200

            r_privacy = await client.get("/privacy")
            assert r_privacy.status_code == 200

    asyncio.run(_test())

