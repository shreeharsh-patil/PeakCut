"""Integration tests for PeakCut Web UI and API Compatibility Bridge."""
import asyncio
from pathlib import Path

import app as app_module


def _call(method: str, path: str, json_body: dict = None, params: dict = None):
    async def _run():
        import httpx
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            if method.upper() == "GET":
                return await client.get(path, params=params)
            elif method.upper() == "DELETE":
                return await client.delete(path, params=params)
            else:
                return await client.post(path, json=json_body, params=params)
    return asyncio.run(_run())


def test_serve_peakcut_app():
    # Root path '/'
    root_resp = _call("GET", "/")
    assert root_resp.status_code == 200
    assert "text/html" in root_resp.headers.get("content-type", "")
    assert "<!doctype html>" in root_resp.text.lower() or "<html" in root_resp.text.lower()

    # '/app' and '/landing'
    resp = _call("GET", "/app")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "<!doctype html>" in resp.text.lower() or "<html" in resp.text.lower()

    landing_resp = _call("GET", "/landing")
    assert landing_resp.status_code == 200
    assert "text/html" in landing_resp.headers.get("content-type", "")


def test_peakcut_ui_feedback_updates_tour():
    # Feedback
    fb_resp = _call("POST", "/api/feedback", json_body={"message": "Love the app!"})
    assert fb_resp.status_code == 200
    assert fb_resp.json()["ok"] is True

    # Updates
    up_resp = _call("GET", "/api/updates")
    assert up_resp.status_code == 200
    assert "version" in up_resp.json()

    # Tour
    tour_resp = _call("POST", "/api/tour")
    assert tour_resp.status_code == 200
    assert tour_resp.json()["tour_done"] is True

    # Viral score API
    score_resp = _call("POST", "/api/viral-score", json_body={"transcript": "Did you know this crazy secret? Turns out it works."})
    assert score_resp.status_code == 200
    assert "overall_score" in score_resp.json()
    assert "viral_hook_text" in score_resp.json()


def test_serve_brand_asset():
    resp = _call("GET", "/brand/logo.svg")
    assert resp.status_code == 200
    assert "svg" in resp.headers.get("content-type", "")


def test_peakcut_status_endpoint():
    resp = _call("GET", "/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "user" in data
    assert "styles" in data
    assert "layouts" in data
    assert "ai" in data
    assert data["ai"]["ready"] == data["ai"]["configured"]


def test_peakcut_sync_endpoint():
    # First sync
    resp = _call("GET", "/api/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert "v" in data
    assert "jobs" in data
    assert "creators" in data
    assert data["ai"]["ready"] == data["ai"]["configured"]
    token = data["v"]

    # Repeated sync with same token should be lightweight unchanged token
    resp2 = _call("GET", "/api/sync", params={"v": token})
    assert resp2.status_code == 200
    assert resp2.json() == {"v": token}


def test_peakcut_overview_and_load_endpoints():
    overview_resp = _call("GET", "/api/overview")
    assert overview_resp.status_code == 200
    assert "creators" in overview_resp.json()
    assert "posts" in overview_resp.json()

    load_resp = _call("GET", "/api/load")
    assert load_resp.status_code == 200
    assert "cpu_percent" in load_resp.json()
    assert "ram_free_gb" in load_resp.json()


def test_peakcut_creators_and_jobs_endpoints():
    # Test creators API
    resp = _call("GET", "/api/creators")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    post_resp = _call("POST", "/api/creators", json_body={
        "name": "Test Streamer",
        "channel": "https://twitch.tv/teststreamer",
        "style": "bounce",
    })
    assert post_resp.status_code == 200
    created = post_resp.json()
    cid = created["id"]
    assert "test-streamer" in cid

    try:
        # Test jobs API
        jobs_resp = _call("GET", "/api/jobs")
        assert jobs_resp.status_code == 200
        assert isinstance(jobs_resp.json(), list)

        # Test alerts test endpoint
        alerts_resp = _call("POST", "/api/alerts/test")
        assert alerts_resp.status_code == 200
        assert "problems" in alerts_resp.json()
    finally:
        # Clean up creator
        del_resp = _call("DELETE", f"/api/creators/{cid}")
        assert del_resp.status_code == 200


def test_peakcut_accounts_apps_and_workers_metadata():
    status = _call("GET", "/api/status").json()
    assert status["workers"]["job_workers"] >= 1
    assert status["workers"]["clip_workers"] >= 4
    assert status["workers"]["llm_workers"] >= 1

    accounts = _call("GET", "/api/accounts").json()
    assert len(accounts) >= 1
    for acc in accounts:
        assert acc.get("label"), f"Account {acc} missing platform label"

    apps = _call("GET", "/api/apps").json()
    for pf in ("youtube", "instagram", "tiktok"):
        assert pf in apps
        assert apps[pf].get("id_label")
        assert apps[pf].get("secret_label")
        assert apps[pf].get("redirect")


def test_peakcut_clip_edit_and_post_uses_real_title_and_file(tmp_path):
    import json
    import shutil
    import peakcut_bridge as pb
    from publishing_queue import PublishingQueue, reset_publishing_queue, get_publishing_queue

    jid = "test_clip_resolution_job"
    job_dir = pb.OUTPUT_DIR / jid
    job_dir.mkdir(parents=True, exist_ok=True)
    clip_file = job_dir / "subtitled_demo_clip_1.mp4"
    clip_file.write_bytes(b"\x00" * 60000)

    # Deliberately omit 'id' on clip entry to match status.json produced by app.py
    status_payload = {
        "id": jid,
        "status": "done",
        "title": "Demo Video Title",
        "clips": [
            {
                "start": 0.0,
                "end": 22.5,
                "predicted_score": 91,
                "video_title_for_youtube_short": "Original Viral Title",
                "viral_hook_text": "Wait for this!",
                "video_description_for_tiktok": "Original caption #shorts",
                "video_url": f"/videos/{jid}/subtitled_demo_clip_1.mp4",
            }
        ],
    }
    (job_dir / "status.json").write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    # Isolate publishing queue so test posts don't pollute output/posts_queue.json
    test_queue = PublishingQueue(queue_file=tmp_path / "posts_queue.json")
    reset_publishing_queue(test_queue)
    try:
        # 1. Edit clip title/description via render endpoint
        remake_resp = _call(
            "POST",
            f"/api/jobs/{jid}/clips/c_01/render",
            json_body={"title": "Edited Short Title", "description": "Edited caption #viral", "hook": "NEW HOOK"},
        )
        assert remake_resp.status_code == 200
        job_after = remake_resp.json()
        assert job_after["clips"][0]["title"] == "Edited Short Title"
        assert job_after["clips"][0]["description"] == "Edited caption #viral"
        assert job_after["clips"][0]["hook"] == "NEW HOOK"

        # 2. Enqueue post and verify real file path + edited title/description are preserved
        post_resp = _call(
            "POST",
            "/api/posts",
            json_body={
                "job": jid,
                "clip": "c_01",
                "accounts": ["yt_main"],
                "now": False,
                "title": "Final Upload Title",
                "description": "Final Upload Description #Shorts",
                "privacy": "private",
            },
        )
        assert post_resp.status_code == 200
        post_data = post_resp.json()
        assert post_data["scheduled"] == 1
        enqueued = post_data["posts"][0]
        assert enqueued["title"] == "Final Upload Title"
        assert enqueued["description"] == "Final Upload Description #Shorts"
        assert Path(enqueued["video_path"]).name == "subtitled_demo_clip_1.mp4"
        assert enqueued["metadata"]["privacy"] == "private"

        # 3. Job detail should include the post pill attached to clip c_01
        job_detail = _call("GET", f"/api/jobs/{jid}").json()
        assert len(job_detail["clips"][0]["posts"]) == 1
    finally:
        reset_publishing_queue(None)
        shutil.rmtree(job_dir, ignore_errors=True)


def test_peakcut_sync_never_freezes_on_active_job():
    import peakcut_bridge as pb

    jid = "active_sync_test_job"
    app_module.jobs[jid] = {
        "status": "processing",
        "logs": ["Job started by worker.", "Found 3 clips!", "Processing Clip 1: 0s - 20s"],
        "ready_files": {0: "clip_1.mp4"},
        "output_dir": str(pb.OUTPUT_DIR / jid),
    }
    try:
        r1 = _call("GET", "/api/sync")
        assert r1.status_code == 200
        token = r1.json()["v"]

        # Even with the same sync token, an active running job must still stream fresh progress
        r2 = _call("GET", "/api/sync", params={"v": token})
        assert r2.status_code == 200
        body = r2.json()
        assert "jobs" in body
        active = next((j for j in body["jobs"] if j["id"] == jid), None)
        assert active is not None
        assert "1/3 ready" in active["stage"]
    finally:
        app_module.jobs.pop(jid, None)
        import shutil
        shutil.rmtree(pb.OUTPUT_DIR / jid, ignore_errors=True)
