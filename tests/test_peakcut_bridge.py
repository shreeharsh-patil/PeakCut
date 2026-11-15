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


def test_peakcut_sync_endpoint():
    # First sync
    resp = _call("GET", "/api/sync")
    assert resp.status_code == 200
    data = resp.json()
    assert "v" in data
    assert "jobs" in data
    assert "creators" in data
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
