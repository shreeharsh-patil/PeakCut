"""Robust Multi-Provider API Key Rotation System for OpenShorts.

Features:
- Parses comma-separated keys from environment variables (e.g. `GEMINI_API_KEY`,
  `GEMINI_API_KEYS`, `OPENAI_API_KEY`, `OPENAI_API_KEYS`).
- Thread-safe round-robin key management with fine-grained per-key health tracking.
- Automatic cooldown on rate limit (429) / QuotaExceeded / ResourceExhausted errors.
- Automatic detection of invalid / unauthenticated keys (401/403) with permanent or
  long-term quarantine.
- Transparent retry wrapper and decorator `with_key_rotation`.
- Dynamic key updates via runtime API or settings.
- Security-first key masking for logs, status endpoints, and UI views.
"""
from __future__ import annotations

import functools
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("api_key_rotator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def mask_key(key: Optional[str]) -> str:
    """Safely mask an API key for logs and UI display."""
    if not key or not isinstance(key, str):
        return "<none>"
    s = key.strip()
    if len(s) <= 8:
        return s[:2] + "..." + s[-2:] if len(s) >= 4 else "***"
    return s[:6] + "..." + s[-4:]


def parse_comma_keys(raw: Optional[Union[str, List[Any]]]) -> List[str]:
    """Parse comma-separated strings or list of strings into clean, deduplicated keys."""
    if not raw:
        return []
    items: List[str] = []
    if isinstance(raw, (list, tuple, set)):
        for entry in raw:
            items.extend(parse_comma_keys(entry))
    elif isinstance(raw, str):
        # Split on commas, semicolons or newlines
        parts = re.split(r"[,;\n\r]+", raw)
        for part in parts:
            cleaned = part.strip().strip("'\"")
            if cleaned:
                items.append(cleaned)
    # Deduplicate preserving order
    return list(dict.fromkeys(items))


def is_rotatable_error(exc: Exception) -> bool:
    """Determine if an exception represents a rate limit, quota exhaustion, or key failure."""
    if exc is None:
        return False

    # Check status_code / code attributes (HTTP/SDK conventions)
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if code in (429, 401, 403, 402):
        return True

    status = getattr(exc, "status", None)
    if status:
        status_str = str(status).upper()
        if any(term in status_str for term in ("RESOURCE_EXHAUSTED", "UNAUTHENTICATED", "PERMISSION_DENIED")):
            return True

    # Check message text
    msg = str(exc).lower()
    patterns = (
        "429",
        "resource_exhausted",
        "resourceexhausted",
        "quota",
        "rate limit",
        "ratelimit",
        "rate_limit",
        "too many requests",
        "unauthenticated",
        "invalid api key",
        "invalid_api_key",
        "api_key_invalid",
        "key not found",
        "key invalid",
        "permission_denied",
        "permissiondenied",
        "forbidden",
        "unauthorized",
        "401",
        "403",
        "key expired",
        "billing not enabled",
    )
    return any(p in msg for p in patterns)


@dataclass
class KeyState:
    """Health and usage metrics for a single API key."""
    key: str
    service: str
    failures: int = 0
    successes: int = 0
    consecutive_failures: int = 0
    last_used: float = 0.0
    last_failed: float = 0.0
    cooldown_until: float = 0.0
    last_error: str = ""
    is_invalid: bool = False

    @property
    def is_healthy(self) -> bool:
        if self.is_invalid:
            return False
        return time.time() >= self.cooldown_until

    @property
    def cooldown_remaining(self) -> float:
        remaining = self.cooldown_until - time.time()
        return max(0.0, round(remaining, 1))

    def to_dict(self, is_active: bool = False) -> Dict[str, Any]:
        return {
            "masked": mask_key(self.key),
            "failures": self.failures,
            "successes": self.successes,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_remaining": self.cooldown_remaining,
            "is_healthy": self.is_healthy,
            "is_invalid": self.is_invalid,
            "is_active": is_active,
            "last_error": self.last_error[:120] if self.last_error else "",
        }


class ApiKeyRotator:
    """Thread-safe round-robin API key rotator with health tracking and auto-failover."""

    def __init__(
        self,
        initial_keys: Optional[Dict[str, Union[str, List[str]]]] = None,
        default_cooldown: float = 60.0,
    ):
        self._lock = threading.RLock()
        self._default_cooldown = default_cooldown
        self._keys: Dict[str, List[KeyState]] = {}
        self._active_index: Dict[str, int] = {}

        if initial_keys:
            for service, keys_val in initial_keys.items():
                self.register_keys(service, keys_val)
        else:
            self._load_from_environment()

    def _env_keys_for_service(self, service: str) -> List[str]:
        svc = service.upper().replace("-", "_")
        keys: List[str] = []
        # Check plural e.g. GEMINI_API_KEYS
        plural_val = os.environ.get(f"{svc}_API_KEYS")
        if plural_val:
            keys.extend(parse_comma_keys(plural_val))
        # Check singular e.g. GEMINI_API_KEY
        singular_val = os.environ.get(f"{svc}_API_KEY")
        if singular_val:
            keys.extend(parse_comma_keys(singular_val))
        return list(dict.fromkeys(keys))

    def _load_from_environment(self) -> None:
        """Scan standard environment variables for supported providers."""
        for service in ("gemini", "openai", "anthropic"):
            found = self._env_keys_for_service(service)
            if found:
                self.register_keys(service, found)

    def register_keys(self, service: str, keys_input: Union[str, List[str]]) -> int:
        """Register or update keys for a service, preserving existing metrics where possible."""
        svc = service.lower().strip()
        parsed = parse_comma_keys(keys_input)
        with self._lock:
            existing_map = {k.key: k for k in self._keys.get(svc, [])}
            new_list: List[KeyState] = []
            for k_str in parsed:
                if k_str in existing_map:
                    new_list.append(existing_map[k_str])
                else:
                    new_list.append(KeyState(key=k_str, service=svc))
            self._keys[svc] = new_list
            if svc not in self._active_index or self._active_index[svc] >= len(new_list):
                self._active_index[svc] = 0

            # Sync single active key back to os.environ for seamless legacy compat
            if new_list:
                active_k = new_list[self._active_index[svc]].key
                self._sync_env_active_key(svc, active_k)
                os.environ[f"{svc.upper()}_API_KEYS"] = ",".join(k.key for k in new_list)
            return len(new_list)

    def get_keys(self, service: str = "gemini") -> List[str]:
        """Return all registered raw keys for a service."""
        svc = service.lower().strip()
        with self._lock:
            if svc not in self._keys or not self._keys[svc]:
                self._load_from_environment()
            return [k.key for k in self._keys.get(svc, [])]

    def get_all_keys_string(self, service: str = "gemini") -> str:
        """Return comma-separated string of all keys for a service."""
        return ",".join(self.get_keys(service))

    def get_key(self, service: str = "gemini") -> Optional[str]:
        """Get the current active, healthy API key for a service."""
        svc = service.lower().strip()
        with self._lock:
            key_list = self._keys.get(svc, [])
            if not key_list:
                self._load_from_environment()
                key_list = self._keys.get(svc, [])
            if not key_list:
                # Direct check if user set environment variable late
                raw = os.environ.get(f"{svc.upper()}_API_KEY") or os.environ.get(f"{svc.upper()}_API_KEYS")
                if raw:
                    self.register_keys(svc, raw)
                    key_list = self._keys.get(svc, [])
            if not key_list:
                return None

            idx = self._active_index.get(svc, 0) % len(key_list)
            current = key_list[idx]

            # If current key is healthy, return it
            if current.is_healthy:
                current.last_used = time.time()
                return current.key

            # Find next healthy key
            n = len(key_list)
            for step in range(1, n):
                candidate_idx = (idx + step) % n
                candidate = key_list[candidate_idx]
                if candidate.is_healthy:
                    self._active_index[svc] = candidate_idx
                    candidate.last_used = time.time()
                    self._sync_env_active_key(svc, candidate.key)
                    logger.info(f"[{svc}] Switched active key to #{candidate_idx + 1} ({mask_key(candidate.key)})")
                    return candidate.key

            # All keys are on cooldown: pick the one that cools down earliest (excluding permanently invalid)
            valid_keys = [k for k in key_list if not k.is_invalid]
            if valid_keys:
                earliest = min(valid_keys, key=lambda k: k.cooldown_until)
                earliest_idx = key_list.index(earliest)
                self._active_index[svc] = earliest_idx
                earliest.last_used = time.time()
                self._sync_env_active_key(svc, earliest.key)
                return earliest.key

            # Fallback to current
            current.last_used = time.time()
            return current.key

    def rotate_key(self, service: str = "gemini") -> Optional[str]:
        """Explicitly advance to the next API key in the pool."""
        svc = service.lower().strip()
        with self._lock:
            key_list = self._keys.get(svc, [])
            if not key_list:
                return None

            healthy_keys = [k for k in key_list if k.is_healthy]
            if len(healthy_keys) == 1:
                idx = key_list.index(healthy_keys[0])
                self._active_index[svc] = idx
                target = healthy_keys[0]
                target.last_used = time.time()
                self._sync_env_active_key(svc, target.key)
                return target.key

            if not healthy_keys:
                valid_keys = [k for k in key_list if not k.is_invalid]
                candidates = valid_keys if valid_keys else key_list
                earliest = min(candidates, key=lambda k: k.cooldown_until)
                idx = key_list.index(earliest)
                self._active_index[svc] = idx
                earliest.last_used = time.time()
                self._sync_env_active_key(svc, earliest.key)
                return earliest.key

            cur_idx = self._active_index.get(svc, 0)
            n = len(key_list)
            for step in range(1, n + 1):
                candidate_idx = (cur_idx + step) % n
                if key_list[candidate_idx].is_healthy and candidate_idx != cur_idx:
                    self._active_index[svc] = candidate_idx
                    new_active = key_list[candidate_idx]
                    new_active.last_used = time.time()
                    self._sync_env_active_key(svc, new_active.key)
                    logger.info(f"[{svc}] Rotated to key #{candidate_idx + 1}/{n} ({mask_key(new_active.key)})")
                    return new_active.key

            return key_list[self._active_index.get(svc, 0)].key

    def report_failure(
        self,
        key: Optional[str],
        error: Optional[Exception] = None,
        service: str = "gemini",
        cooldown_seconds: Optional[float] = None,
    ) -> None:
        """Record an error for a key, applying cooldown and rotating to the next key."""
        if not key:
            return
        svc = service.lower().strip()
        with self._lock:
            key_list = self._keys.get(svc, [])
            target = next((k for k in key_list if k.key == key), None)
            if not target:
                return

            target.failures += 1
            target.consecutive_failures += 1
            target.last_failed = time.time()
            err_str = str(error) if error else "Unknown error"
            target.last_error = err_str

            # Check if this is an invalid auth error (401/403/invalid key)
            is_auth_error = any(term in err_str.lower() for term in (
                "invalid api key", "unauthenticated", "401", "api_key_invalid", "permission_denied"
            ))
            if is_auth_error and not ("quota" in err_str.lower() or "429" in err_str.lower()):
                target.is_invalid = True
                logger.error(f"[{svc}] Key {mask_key(key)} is INVALID/UNAUTHORIZED: {err_str[:120]}")
            else:
                base_cd = cooldown_seconds if cooldown_seconds is not None else self._default_cooldown
                # Exponential cooldown scaling with consecutive failures, capped at 300s
                cd = min(base_cd * min(target.consecutive_failures, 5), 300.0)
                target.cooldown_until = time.time() + cd
                logger.warning(
                    f"[{svc}] Key {mask_key(key)} failed ({err_str[:100]}). Cooldown {cd:g}s applied."
                )

            # Auto-rotate away from this failing key
            self.rotate_key(svc)

    def report_success(self, key: Optional[str], service: str = "gemini") -> None:
        """Reset consecutive failures and record successful API invocation."""
        if not key:
            return
        svc = service.lower().strip()
        with self._lock:
            key_list = self._keys.get(svc, [])
            target = next((k for k in key_list if k.key == key), None)
            if target:
                target.successes += 1
                target.consecutive_failures = 0
                target.cooldown_until = 0.0

    def _sync_env_active_key(self, service: str, active_key: str) -> None:
        """Update environment variables so external subprocesses/tools find the active key."""
        if service == "gemini":
            os.environ["GEMINI_API_KEY"] = active_key
        elif service == "openai":
            os.environ["OPENAI_API_KEY"] = active_key
        elif service == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = active_key

    def status(self, service: str = "gemini") -> Dict[str, Any]:
        """Detailed status summary for a service."""
        svc = service.lower().strip()
        with self._lock:
            key_list = self._keys.get(svc, [])
            if not key_list:
                self._load_from_environment()
                key_list = self._keys.get(svc, [])

            cur_idx = self._active_index.get(svc, 0)
            healthy = sum(1 for k in key_list if k.is_healthy)
            cooldown = sum(1 for k in key_list if not k.is_healthy and not k.is_invalid)
            invalid = sum(1 for k in key_list if k.is_invalid)

            active_masked = "<none>"
            if key_list and 0 <= cur_idx < len(key_list):
                active_masked = mask_key(key_list[cur_idx].key)

            return {
                "service": svc,
                "total_keys": len(key_list),
                "active_index": cur_idx if key_list else 0,
                "active_key": active_masked,
                "healthy_keys": healthy,
                "cooldown_keys": cooldown,
                "invalid_keys": invalid,
                "keys": [k.to_dict(is_active=(i == cur_idx)) for i, k in enumerate(key_list)],
            }

    def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        service: str = "gemini",
        max_retries: Optional[int] = None,
        cooldown_seconds: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute callable with automatic key rotation on 429/quota/auth failures."""
        svc = service.lower().strip()
        key_list = self.get_keys(svc)
        retries_limit = max_retries if max_retries is not None else max(len(key_list), 3)

        last_exc: Optional[Exception] = None
        for attempt in range(1, retries_limit + 1):
            key = self.get_key(svc)
            # Inject key into kwargs if fn accepts api_key
            call_kwargs = dict(kwargs)
            if "api_key" in call_kwargs:
                call_kwargs["api_key"] = key
            else:
                try:
                    import inspect
                    sig = inspect.signature(fn)
                    if "api_key" in sig.parameters:
                        call_kwargs["api_key"] = key
                except Exception:
                    pass

            try:
                res = fn(*args, **call_kwargs)
                self.report_success(key, svc)
                return res
            except Exception as e:
                last_exc = e
                if is_rotatable_error(e) and attempt < retries_limit:
                    logger.warning(
                        f"[{svc}] API call failed on attempt {attempt}/{retries_limit} "
                        f"with key {mask_key(key)}: {e}. Rotating to next key..."
                    )
                    self.report_failure(key, error=e, service=svc, cooldown_seconds=cooldown_seconds)
                    continue
                # Non-rotatable or exhausted retries
                raise e
        if last_exc:
            raise last_exc


# Global singleton instance
default_rotator = ApiKeyRotator()
get_rotator = lambda: default_rotator


# Module-level convenience functions
def get_gemini_key() -> Optional[str]:
    """Retrieve the active Gemini API key from the global rotator."""
    return default_rotator.get_key("gemini")


def rotate_gemini_key() -> Optional[str]:
    """Force rotate to the next Gemini API key."""
    return default_rotator.rotate_key("gemini")


def report_gemini_failure(key: Optional[str], error: Optional[Exception] = None, cooldown: float = 60.0) -> None:
    """Report an error on a Gemini API key."""
    default_rotator.report_failure(key, error=error, service="gemini", cooldown_seconds=cooldown)


def report_gemini_success(key: Optional[str]) -> None:
    """Report a successful API invocation for a Gemini key."""
    default_rotator.report_success(key, service="gemini")


def get_status(service: str = "gemini") -> Dict[str, Any]:
    """Retrieve status dictionary for a service."""
    return default_rotator.status(service)


def register_keys(service: str, keys: Union[str, List[str]]) -> int:
    """Register keys into the global rotator."""
    return default_rotator.register_keys(service, keys)


def get_all_keys(service: str = "gemini") -> List[str]:
    """Get all registered raw keys for a service."""
    return default_rotator.get_keys(service)


def get_all_keys_string(service: str = "gemini") -> str:
    """Get comma-separated string of all keys for a service."""
    return default_rotator.get_all_keys_string(service)


def create_gemini_client(api_key: Optional[str] = None):
    """Factory creating a google.genai.Client using the active rotated key."""
    from google import genai
    chosen_key = api_key or get_gemini_key()
    if not chosen_key:
        raise ValueError("No GEMINI_API_KEY configured.")
    # If a comma-separated string was passed, register and pick the active one
    if "," in chosen_key:
        register_keys("gemini", chosen_key)
        chosen_key = get_gemini_key()
    return genai.Client(api_key=chosen_key)


def with_key_rotation(
    service: str = "gemini",
    max_retries: Optional[int] = None,
    cooldown_seconds: Optional[float] = None,
):
    """Decorator wrapping functions with automatic key rotation and retry."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return default_rotator.execute_with_retry(
                fn,
                *args,
                service=service,
                max_retries=max_retries,
                cooldown_seconds=cooldown_seconds,
                **kwargs,
            )
        return wrapper
    return decorator
