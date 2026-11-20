"""Unit tests for the API Key Rotation System (api_key_rotator.py).

Verifies:
- Comma-separated key parsing, spacing, quotes, and deduplication.
- Thread-safe round-robin rotation and failover order.
- Cooldown on 429 / ResourceExhausted / QuotaExceeded errors.
- Quarantine on 401 / 403 / Invalid key errors.
- Health recovery and success reporting.
- Wrapper / decorator transparent retry on key failure.
- Integration with PeakCut API bridge settings.
"""
import time
import pytest
from engine.services import api_key_rotator
from engine.services.api_key_rotator import ApiKeyRotator, is_rotatable_error, mask_key, parse_comma_keys


def test_parse_comma_keys():
    # Various dirty string inputs
    raw = " key1,  key2, 'key3' , \"key4\", , key5\nkey6 ; key7 "
    parsed = parse_comma_keys(raw)
    assert parsed == ["key1", "key2", "key3", "key4", "key5", "key6", "key7"]

    # Deduplication
    dup = "alpha, beta, alpha, gamma, beta"
    assert parse_comma_keys(dup) == ["alpha", "beta", "gamma"]

    # Empty and None
    assert parse_comma_keys("") == []
    assert parse_comma_keys(None) == []

    # List of strings
    assert parse_comma_keys(["k1, k2", "k3", ["k4, k5"]]) == ["k1", "k2", "k3", "k4", "k5"]


def test_mask_key():
    assert mask_key("AIzaSyD-1234567890abcdef") == "AIzaSy...cdef"
    assert mask_key("shortkey") == "sh...ey"
    assert mask_key("tiny") == "ti...ny"
    assert mask_key("ab") == "***"
    assert mask_key(None) == "<none>"


def test_is_rotatable_error():
    class MockError(Exception):
        def __init__(self, msg, code=None, status=None):
            super().__init__(msg)
            self.code = code
            self.status = status

    # 429 Rate limit
    assert is_rotatable_error(MockError("Too Many Requests", code=429))
    assert is_rotatable_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded"))
    assert is_rotatable_error(Exception("Google GenAI RateLimitError: rate limit reached"))

    # 401/403 Invalid key
    assert is_rotatable_error(MockError("Unauthorized", code=401))
    assert is_rotatable_error(Exception("API_KEY_INVALID: Key not found"))

    # Non-rotatable
    assert not is_rotatable_error(ValueError("400 INVALID_ARGUMENT: input prompt too long"))
    assert not is_rotatable_error(TypeError("Unexpected type"))


def test_rotator_round_robin():
    rotator = ApiKeyRotator(default_cooldown=60.0)
    rotator.register_keys("gemini", "key1, key2, key3")

    assert rotator.get_key("gemini") == "key1"
    assert rotator.rotate_key("gemini") == "key2"
    assert rotator.rotate_key("gemini") == "key3"
    assert rotator.rotate_key("gemini") == "key1"


def test_failover_on_429():
    rotator = ApiKeyRotator(default_cooldown=10.0)
    rotator.register_keys("gemini", "keyA, keyB, keyC")

    # Initially keyA
    assert rotator.get_key("gemini") == "keyA"

    # keyA encounters 429 quota exhaustion
    err_429 = Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
    rotator.report_failure("keyA", error=err_429, service="gemini", cooldown_seconds=10.0)

    # Next key should automatically fail over to keyB
    assert rotator.get_key("gemini") == "keyB"

    # Status check
    st = rotator.status("gemini")
    assert st["healthy_keys"] == 2
    assert st["cooldown_keys"] == 1
    assert st["keys"][0]["failures"] == 1
    assert st["keys"][0]["cooldown_remaining"] > 0


def test_success_resets_health():
    rotator = ApiKeyRotator(default_cooldown=10.0)
    rotator.register_keys("gemini", "keyA, keyB")

    rotator.report_failure("keyA", error=Exception("429 limit"), cooldown_seconds=10.0)
    assert rotator.status("gemini")["keys"][0]["cooldown_remaining"] > 0

    # Report success on keyA (e.g. after testing or retry)
    rotator.report_success("keyA", service="gemini")
    assert rotator.status("gemini")["keys"][0]["cooldown_remaining"] == 0.0
    assert rotator.status("gemini")["keys"][0]["is_healthy"] is True


def test_all_keys_on_cooldown_picks_earliest():
    rotator = ApiKeyRotator(default_cooldown=10.0)
    rotator.register_keys("gemini", "key1, key2")

    # Put key1 on 5s cooldown, key2 on 20s cooldown
    rotator.report_failure("key1", error=Exception("429"), cooldown_seconds=5.0)
    rotator.report_failure("key2", error=Exception("429"), cooldown_seconds=20.0)

    # Both cooled down, should choose key1 since its cooldown is earliest
    assert rotator.get_key("gemini") == "key1"


def test_invalid_key_quarantine():
    rotator = ApiKeyRotator()
    rotator.register_keys("gemini", "bad_key, good_key")

    rotator.report_failure("bad_key", error=Exception("401 UNAUTHENTICATED: API key invalid"), service="gemini")
    st = rotator.status("gemini")
    assert st["invalid_keys"] == 1
    assert st["keys"][0]["is_invalid"] is True

    # Rotator now stays on good_key
    assert rotator.get_key("gemini") == "good_key"
    assert rotator.rotate_key("gemini") == "good_key"


def test_execute_with_retry_wrapper():
    rotator = ApiKeyRotator(default_cooldown=5.0)
    rotator.register_keys("gemini", "fail_key, success_key")

    attempts = []

    def mock_gemini_call(api_key: str = None):
        attempts.append(api_key)
        if api_key == "fail_key":
            raise Exception("429 ResourceExhausted: rate limit exceeded")
        return {"result": f"ok with {api_key}"}

    res = rotator.execute_with_retry(mock_gemini_call, service="gemini")
    assert res == {"result": "ok with success_key"}
    assert attempts == ["fail_key", "success_key"]


def test_with_key_rotation_decorator(monkeypatch):
    test_rotator = ApiKeyRotator(default_cooldown=5.0)
    test_rotator.register_keys("gemini", "key_x, key_y")
    monkeypatch.setattr(api_key_rotator, "default_rotator", test_rotator)

    invocations = []

    @api_key_rotator.with_key_rotation(service="gemini")
    def api_task(prompt: str, api_key: str = None):
        invocations.append((prompt, api_key))
        if api_key == "key_x":
            raise Exception("429 QuotaExceeded")
        return f"generated for {prompt}"

    result = api_task("hello world")
    assert result == "generated for hello world"
    assert len(invocations) == 2
    assert invocations[0][1] == "key_x"
    assert invocations[1][1] == "key_y"


def test_peakcut_bridge_settings_key_rotation():
    import app as app_module
    import asyncio
    import httpx

    async def _call():
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            # Update keys via /api/settings
            post_resp = await client.post(
                "/api/settings",
                json={
                    "gemini_api_keys": "mock_k1, mock_k2, mock_k3",
                    "gemini_model": "gemini-2.5-flash",
                },
            )
            assert post_resp.status_code == 200
            data = post_resp.json()
            assert data["changes"]["gemini_keys_registered"] == 3
            assert data["key_rotation"]["gemini"]["total_keys"] == 3

            # Check GET /api/settings
            get_resp = await client.get("/api/settings")
            assert get_resp.status_code == 200
            assert get_resp.json()["key_rotation"]["gemini"]["total_keys"] == 3

            # Rotate key via /api/keys/rotate
            rot_resp = await client.post("/api/keys/rotate?service=gemini")
            assert rot_resp.status_code == 200
            assert rot_resp.json()["status"]["active_index"] == 1

            # Check GET /api/status includes key_rotation in ai info
            st_resp = await client.get("/api/status")
            assert st_resp.status_code == 200
            ai_info = st_resp.json()["ai"]
            assert "key_rotation" in ai_info
            assert ai_info["key_rotation"]["total_keys"] == 3

    asyncio.run(_call())
