"""Text-only LLM backend for the moment picker: any OpenAI-compatible server.

Ollama, LM Studio, vLLM, llama.cpp's server, LocalAI and OpenRouter all speak
``POST {base}/chat/completions``. When ``LLM_BASE_URL`` is set (or
``LLM_PROVIDER=openai``), the transcript scoring and detail passes in
``main.get_viral_clips`` go here instead of Gemini, so a self-hosted install
can run the whole pipeline without a Google key.

What stays on Gemini, because it needs a model that can look at frames or a
video file: the layout picker (``layout_picker.py``), the on-screen content
detector (``screencast_layout.py``) and the silent-video path
(``main.get_visual_clips``). Without a Gemini key those degrade the way they
already did: the first two return "none", the third fails the job with a clear
message. Nothing in this module is imported by them.

Structured output: the prompts already spell out the exact JSON shape, so a
plain ``json_object`` mode is enough for most models. The request first asks
for ``json_schema`` (Ollama, vLLM, llama.cpp and LM Studio enforce it); a
server that rejects that field gets the same request again with
``json_object``, then with no ``response_format`` at all. Whatever comes back
is validated with the same pydantic model Gemini's ``response_schema`` uses,
so ``main.py`` sees one shape regardless of provider.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type

import httpx
from pydantic import BaseModel

DEFAULT_MODEL = "gemma2:2b"
DEFAULT_TIMEOUT = 600.0  # local models on CPU are slow; a scoring batch can take minutes

_HEALTH_CACHE: Dict[str, Any] = {"key": None, "ts": 0.0, "result": None}
_HEALTH_CACHE_TTL = 3.0


def invalidate_health_cache() -> None:
    """Clear the cached local LLM health probe (useful after pulling a model or in tests)."""
    _HEALTH_CACHE["key"] = None
    _HEALTH_CACHE["ts"] = 0.0
    _HEALTH_CACHE["result"] = None


def _model_matches(installed: List[str], requested: str) -> bool:
    """Match 'gemma2:2b' against 'gemma2:2b' or 'gemma2' against 'gemma2:latest'."""
    req = (requested or "").strip().lower()
    if not req:
        return False
    for name in installed:
        cand = (name or "").strip().lower()
        if not cand:
            continue
        if cand == req or cand == f"{req}:latest":
            return True
        if ":" not in req and cand.split(":", 1)[0] == req:
            return True
    return False


def _ollama_root_url(url: str) -> str:
    """Strip trailing '/v1' from an OpenAI-compatible Ollama base URL."""
    base = (url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        return base[:-3].rstrip("/")
    return base


def check_local_llm_health(
    model: Optional[str] = None,
    *,
    run_test_inference: bool = False,
    timeout: float = 3.0,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Perform real readiness verification against the configured local LLM / Ollama server.

    Checks:
      1. Server responds on /api/tags (Ollama) or /v1/models (OpenAI-compatible).
      2. Selected model is actually present in the server's installed model list.
      3. (Optional) A tiny test inference succeeds end-to-end.

    Never fabricates readiness when the server is down or the model is missing.
    """
    target_url = base_url()
    target_model = (model or model_name()).strip()
    cache_key = (target_url, target_model, bool(run_test_inference))
    now = time.time()
    if (
        use_cache
        and not run_test_inference
        and _HEALTH_CACHE["key"] == cache_key
        and (now - _HEALTH_CACHE["ts"]) < _HEALTH_CACHE_TTL
        and _HEALTH_CACHE["result"] is not None
    ):
        return dict(_HEALTH_CACHE["result"])

    result: Dict[str, Any] = {
        "reachable": False,
        "model_installed": False,
        "inference_ok": None if not run_test_inference else False,
        "ready": False,
        "model": target_model,
        "base_url": target_url,
        "installed_models": [],
        "error": None,
        "checked_at": now,
    }

    if not target_url:
        result["error"] = "LLM_BASE_URL not configured"
        return result

    installed: List[str] = []
    root_url = _ollama_root_url(target_url)

    try:
        with _client(timeout=timeout) as client:
            # 1a. Try Ollama native /api/tags first (gives authoritative installed models list)
            tags_ok = False
            try:
                resp = client.get(f"{root_url}/api/tags")
                if resp.status_code == 200:
                    data = resp.json() or {}
                    installed = [
                        str(m.get("name") or m.get("model") or "").strip()
                        for m in (data.get("models") or [])
                        if isinstance(m, dict)
                    ]
                    installed = [m for m in installed if m]
                    result["reachable"] = True
                    tags_ok = True
            except Exception:
                pass

            # 1b. Fall back to OpenAI-compatible GET {base}/models
            if not tags_ok:
                resp = client.get(f"{target_url}/models", headers=_headers())
                if resp.status_code == 200:
                    data = resp.json() or {}
                    installed = [
                        str(m.get("id") or "").strip()
                        for m in (data.get("data") or [])
                        if isinstance(m, dict)
                    ]
                    installed = [m for m in installed if m]
                    result["reachable"] = True
                else:
                    result["error"] = f"Server returned HTTP {resp.status_code}"

            result["installed_models"] = installed
            if result["reachable"]:
                result["model_installed"] = _model_matches(installed, target_model)
                if not result["model_installed"]:
                    result["error"] = (
                        f"Model '{target_model}' not found on server "
                        f"(installed: {', '.join(installed) if installed else 'none'})"
                    )

            # 3. Optional tiny end-to-end inference test
            if result["reachable"] and result["model_installed"] and run_test_inference:
                inf_timeout = max(float(timeout), 45.0)
                with _client(timeout=inf_timeout) as inf_client:
                    body = {
                        "model": target_model,
                        "messages": [{"role": "user", "content": 'Reply with JSON: {"ok": true}'}],
                        "max_tokens": 24,
                        "temperature": 0.0,
                        "stream": False,
                    }
                    r_inf = inf_client.post(f"{target_url}/chat/completions", json=body, headers=_headers())
                    if r_inf.status_code == 200:
                        inf_data = r_inf.json() or {}
                        choices = inf_data.get("choices") or []
                        content = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
                        if str(content).strip():
                            result["inference_ok"] = True
                        else:
                            result["inference_ok"] = False
                            result["error"] = "Test inference returned empty content"
                    else:
                        result["inference_ok"] = False
                        result["error"] = f"Test inference HTTP {r_inf.status_code}: {r_inf.text[:180]}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["ready"] = bool(
        result["reachable"]
        and result["model_installed"]
        and (result["inference_ok"] is not False)
    )

    if not run_test_inference:
        _HEALTH_CACHE["key"] = cache_key
        _HEALTH_CACHE["ts"] = now
        _HEALTH_CACHE["result"] = dict(result)
    return result


def is_colab() -> bool:
    """Return True if executing inside Google Colab environment or Colab mode is forced."""
    if os.environ.get("IS_COLAB") in ("1", "true", "True"):
        return True
    if os.environ.get("FORCE_LOCAL_ONLY") in ("1", "true", "True"):
        return True
    if os.environ.get("COLAB_GPU") is not None or os.environ.get("COLAB_RELEASE_TAG") is not None:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return os.path.exists("/content") and os.path.isdir("/content") and os.path.exists("/content/sample_data")


def provider() -> str:
    """``"openai"`` when a compatible endpoint is configured or running on Colab, else ``"gemini"``."""
    explicit = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if explicit in ("openai", "ollama", "local", "openai-compatible"):
        return "openai"
    if explicit == "gemini":
        return "gemini"
    if is_colab():
        return "openai"
    return "openai" if base_url() else "gemini"


def base_url() -> str:
    url = (os.environ.get("LLM_BASE_URL") or "").strip().rstrip("/")
    if url:
        return url
    if is_colab():
        return "http://127.0.0.1:11434/v1"
    return ""


def model_name() -> str:
    explicit = (os.environ.get("LLM_MODEL") or "").strip()
    if explicit:
        return explicit
    if is_colab() or provider() == "openai":
        return "gemma2:2b"
    return DEFAULT_MODEL


def active() -> bool:
    """True when the moment picker should call the OpenAI-compatible server."""
    return provider() == "openai" and bool(base_url())


def is_local_llm_ready() -> bool:
    """Check if the local OpenAI-compatible / Ollama server is responding with the selected model."""
    if not active():
        return False
    return bool(check_local_llm_health(timeout=2.0).get("ready", False))


def get_available_models() -> list[str]:
    """Query the OpenAI-compatible endpoint to discover available models."""
    if not active():
        return []
    url = f"{base_url()}/models"
    try:
        with _client(timeout=3.0) as client:
            resp = client.get(url, headers=_headers())
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data") or []
                if isinstance(items, list):
                    return [item.get("id") for item in items if isinstance(item, dict) and item.get("id")]
    except Exception:
        pass
    return []


def validate_model_availability(model: Optional[str] = None) -> bool:
    """Check whether the requested model is available on the configured provider."""
    target = model or model_name()
    available = get_available_models()
    if available:
        return target in available
    return is_local_llm_ready()


def get_provider_health() -> dict:
    """Provide health status, readiness, model telemetry, and connectivity details."""
    p = provider()
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS")
    is_ready = is_local_llm_ready() if p == "openai" else bool(gemini_key)
    return {
        "provider": p,
        "ready": is_ready,
        "model": model_name() if p == "openai" else os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "base_url": base_url() if p == "openai" else None,
        "available_models": get_available_models() if p == "openai" else [],
    }


def describe() -> Optional[dict]:
    """What ``/api/config`` tells the dashboard, or ``None`` when inactive."""
    if not active():
        return None
    return {"provider": "openai", "model": model_name(), "baseUrl": base_url()}


def _timeout() -> float:
    try:
        return float(os.environ.get("LLM_TIMEOUT") or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def _headers() -> dict:
    # Ollama ignores the key but the OpenAI client convention (and vLLM with
    # --api-key) wants the header present; "ollama" is the documented placeholder.
    key = (os.environ.get("LLM_API_KEY") or "ollama").strip()
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _client(**kwargs) -> httpx.Client:
    """Factory so tests can swap in ``httpx.MockTransport``."""
    kwargs.setdefault("timeout", _timeout())
    return httpx.Client(**kwargs)


def _response_formats(schema: Type[BaseModel]):
    name = getattr(schema, "__name__", "response").lower()
    yield {"type": "json_schema",
           "json_schema": {"name": name, "schema": schema.model_json_schema()}}
    yield {"type": "json_object"}
    yield None


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    return "response_format" in body or "json_schema" in body or "json_object" in body \
        or "format" in body


def generate_json(prompt: str, schema: Type[BaseModel], model: Optional[str] = None,
                  ) -> Tuple[dict, Optional[dict]]:
    """One chat completion that must come back as JSON matching ``schema``.

    Returns ``(parsed_dict, cost_analysis)`` in the exact shape
    ``main._run_gemini_stage`` returns, so the caller does not branch on the
    provider. Raises on HTTP errors, empty bodies and schema violations; the
    retry policy lives in the caller, same as for Gemini.
    """
    from engine.intelligence import gemini_worker  # local import: keeps this module free of the google SDK

    url = f"{base_url()}/chat/completions"
    model = model or model_name()
    messages = [
        {"role": "system", "content": "You answer with a single JSON object and nothing else."},
        {"role": "user", "content": prompt},
    ]
    last_rejection: Optional[str] = None
    with _client() as client:
        for fmt in _response_formats(schema):
            body = {"model": model, "messages": messages, "temperature": 0.2, "stream": False}
            if fmt is not None:
                body["response_format"] = fmt
            resp = client.post(url, json=body, headers=_headers())
            if fmt is not None and _is_format_rejection(resp):
                # The server does not know this response_format flavour; the
                # next loop iteration asks for a looser one.
                last_rejection = resp.text[:200]
                continue
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"LLM server {resp.status_code} from {url}: {resp.text[:300]}")
            data = resp.json()
            break
        else:
            raise RuntimeError(
                f"LLM server rejected every response_format variant: {last_rejection}")

    choices = data.get("choices") or []
    text = ""
    if choices:
        msg = choices[0].get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # some servers return content parts
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    parsed = gemini_worker._parse_json_response_text(text)
    # Validate against the same schema Gemini enforces server-side, so a
    # local model that drops a field fails here with a readable error
    # (retried by the caller) instead of deep inside the clip pipeline.
    validated = schema.model_validate(parsed).model_dump()

    usage = data.get("usage") or {}
    cost = {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "thinking_tokens": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "total_cost": 0.0,
        "model": model,
        "price_estimated": False,
        "local": True,
    }
    return validated, cost
