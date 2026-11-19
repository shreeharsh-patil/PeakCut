"""Phase 2 API Key Rotation Engine for ClipMind.

Manages API key pools for YouTube, OpenAI, SerpAPI, ElevenLabs, and Pexels.
Features:
- Quota tracking persisted in a JSON state file (data/key_quotas.json).
- Daily midnight UTC reset of quota counters.
- Pre-flight check enforcing >15% quota remaining before issuing a key.
- Dynamic blacklisting on HTTP 429 (Rate Limit) and 403 (Forbidden/Quota Exceeded).
- Active health pings verifying key validity against live provider endpoints.
- Thread-safe round-robin selection.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from clipmind.core.config import get_config

logger = logging.getLogger("clipmind.key_rotator")


# Standard default quotas per provider (daily units)
DEFAULT_QUOTAS = {
    "youtube": 10000,    # 10,000 units/day standard quota
    "openai": 1000000,   # 1M tokens/day or arbitrary budget unit
    "serpapi": 100,      # 100 searches/month or searches/day
    "elevenlabs": 10000, # 10,000 characters/day
    "pexels": 200,       # 200 requests/hour or day
}


def mask_key(key: Optional[str]) -> str:
    """Mask key for secure logging and UI display."""
    if not key:
        return "<none>"
    s = key.strip()
    if len(s) <= 8:
        return s[:2] + "..." + s[-2:] if len(s) >= 4 else "***"
    return s[:6] + "..." + s[-4:]


@dataclass
class KeyState:
    """State and quota metadata for an individual API key."""
    key: str
    service: str
    total_quota: int
    used_units: int = 0
    blacklisted_until: float = 0.0
    last_used: float = 0.0
    is_valid: bool = True
    last_error: str = ""
    success_count: int = 0
    failure_count: int = 0

    @property
    def remaining_units(self) -> int:
        return max(0, self.total_quota - self.used_units)

    @property
    def remaining_percent(self) -> float:
        if self.total_quota <= 0:
            return 100.0
        return max(0.0, ((self.total_quota - self.used_units) / self.total_quota) * 100.0)

    @property
    def is_rate_limited(self) -> bool:
        return time.time() < self.blacklisted_until

    @property
    def has_sufficient_quota(self) -> bool:
        """Pre-flight check: must have > 15% quota remaining."""
        return self.remaining_percent > 15.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_masked": mask_key(self.key),
            "service": self.service,
            "total_quota": self.total_quota,
            "used_units": self.used_units,
            "remaining_units": self.remaining_units,
            "remaining_percent": round(self.remaining_percent, 2),
            "is_rate_limited": self.is_rate_limited,
            "has_sufficient_quota": self.has_sufficient_quota,
            "is_valid": self.is_valid,
            "last_error": self.last_error,
            "blacklisted_until": self.blacklisted_until,
            "last_used": self.last_used,
        }


class KeyRotator:
    """Thread-safe Multi-Provider API Key Rotation Engine."""

    _instance: Optional[KeyRotator] = None
    _lock = threading.RLock()

    def __init__(self, state_file: Optional[Path] = None):
        self.config = get_config()
        self.state_file = state_file or self.config.paths.quota_state_file
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            import tempfile
            self.state_file = Path(tempfile.gettempdir()) / "clipmind" / "data" / "key_quotas.json"
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.pools: Dict[str, List[KeyState]] = {
            "youtube": [],
            "openai": [],
            "serpapi": [],
            "elevenlabs": [],
            "pexels": [],
        }
        self.current_index: Dict[str, int] = {k: 0 for k in self.pools}
        self.last_reset_utc_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        self._load_pools_from_config()
        self._load_state()
        self._check_and_apply_midnight_reset()

    @classmethod
    def get_instance(cls) -> KeyRotator:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _load_pools_from_config(self) -> None:
        """Populate initial key lists from configuration."""
        key_mapping = {
            "youtube": self.config.youtube_api_keys,
            "openai": self.config.openai_api_keys,
            "serpapi": self.config.serpapi_api_keys,
            "elevenlabs": self.config.elevenlabs_api_keys,
            "pexels": self.config.pexels_api_keys,
        }
        for service, keys in key_mapping.items():
            for key in keys:
                if key and not any(k.key == key for k in self.pools[service]):
                    self.pools[service].append(
                        KeyState(
                            key=key,
                            service=service,
                            total_quota=DEFAULT_QUOTAS.get(service, 1000),
                        )
                    )

    def _load_state(self) -> None:
        """Load persisted quota and blacklist state from disk."""
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.last_reset_utc_date = data.get("last_reset_utc_date", self.last_reset_utc_date)
            saved_services = data.get("services", {})
            for service, key_states in saved_services.items():
                if service not in self.pools:
                    self.pools[service] = []
                for s_data in key_states:
                    k_str = s_data.get("key")
                    # Match by key or masked key
                    existing = next((k for k in self.pools[service] if k.key == k_str), None)
                    if existing:
                        existing.used_units = s_data.get("used_units", 0)
                        existing.blacklisted_until = s_data.get("blacklisted_until", 0.0)
                        existing.last_used = s_data.get("last_used", 0.0)
                        existing.is_valid = s_data.get("is_valid", True)
                        existing.last_error = s_data.get("last_error", "")
                    elif k_str:
                        # Key was loaded previously
                        self.pools[service].append(
                            KeyState(
                                key=k_str,
                                service=service,
                                total_quota=s_data.get("total_quota", DEFAULT_QUOTAS.get(service, 1000)),
                                used_units=s_data.get("used_units", 0),
                                blacklisted_until=s_data.get("blacklisted_until", 0.0),
                                last_used=s_data.get("last_used", 0.0),
                                is_valid=s_data.get("is_valid", True),
                                last_error=s_data.get("last_error", ""),
                            )
                        )
        except Exception as e:
            logger.warning(f"Could not load key quota state: {e}")

    def save_state(self) -> None:
        """Persist key quotas and state to disk atomically."""
        with self._lock:
            try:
                data = {
                    "last_reset_utc_date": self.last_reset_utc_date,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "services": {}
                }
                for service, states in self.pools.items():
                    data["services"][service] = [
                        {
                            "key": s.key,
                            "service": s.service,
                            "total_quota": s.total_quota,
                            "used_units": s.used_units,
                            "remaining_units": s.remaining_units,
                            "remaining_percent": round(s.remaining_percent, 2),
                            "blacklisted_until": s.blacklisted_until,
                            "last_used": s.last_used,
                            "is_valid": s.is_valid,
                            "last_error": s.last_error,
                        }
                        for s in states
                    ]
                temp_file = self.state_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                try:
                    temp_file.replace(self.state_file)
                except OSError:
                    try:
                        if self.state_file.exists():
                            self.state_file.unlink()
                        temp_file.rename(self.state_file)
                    except OSError:
                        with open(self.state_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        try:
                            temp_file.unlink(missing_ok=True)
                        except OSError:
                            pass
            except Exception as e:
                logger.error(f"Failed to save key quotas: {e}")

    def _check_and_apply_midnight_reset(self) -> None:
        """Daily midnight UTC reset: clears used units and transient blacklists."""
        with self._lock:
            current_utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if current_utc_date > self.last_reset_utc_date:
                logger.info(f"Midnight UTC reset triggered: {self.last_reset_utc_date} -> {current_utc_date}")
                for states in self.pools.values():
                    for s in states:
                        s.used_units = 0
                        # Clear rate-limit blacklist on daily reset
                        s.blacklisted_until = 0.0
                        s.failure_count = 0
                self.last_reset_utc_date = current_utc_date
                self.save_state()

    def add_key(self, service: str, key: str, total_quota: Optional[int] = None) -> None:
        """Dynamically register a new API key into the pool."""
        service = service.lower().strip()
        key = key.strip()
        if not key:
            return
        with self._lock:
            if service not in self.pools:
                self.pools[service] = []
            if not any(k.key == key for k in self.pools[service]):
                quota = total_quota or DEFAULT_QUOTAS.get(service, 1000)
                self.pools[service].append(
                    KeyState(key=key, service=service, total_quota=quota)
                )
                self.save_state()
                logger.info(f"Registered new key for {service}: {mask_key(key)}")

    def get_active_key(self, service: str) -> Optional[str]:
        """Obtain the next eligible active key for a service with pre-flight check (>15% remaining)."""
        service = service.lower().strip()
        with self._lock:
            self._check_and_apply_midnight_reset()

            pool = self.pools.get(service, [])
            if not pool:
                if self.config.simulate_apis_if_missing:
                    # Provide an annotated simulation key so workflows can proceed gracefully
                    sim_key = f"sim_{service}_key_clipmind"
                    return sim_key
                return None

            # Filter candidates:
            # 1. Not currently blacklisted (e.g. 429/403 cooldown)
            # 2. Key marked valid
            # 3. Pre-flight check: > 15% quota remaining
            candidates = [
                k for k in pool
                if not k.is_rate_limited and k.is_valid and k.has_sufficient_quota
            ]

            if not candidates:
                # Fallback: check if any key is at least not blacklisted
                lesser_candidates = [k for k in pool if not k.is_rate_limited and k.is_valid]
                if lesser_candidates:
                    # Sort by remaining units descending
                    best = max(lesser_candidates, key=lambda x: x.remaining_units)
                    logger.warning(
                        f"All keys for {service} are below 15% threshold! "
                        f"Selected best remaining: {mask_key(best.key)} ({best.remaining_percent:.1f}%)"
                    )
                    best.last_used = time.time()
                    self.save_state()
                    return best.key

                logger.error(f"No available active keys for {service}! All exhausted or blacklisted.")
                if self.config.simulate_apis_if_missing:
                    return f"sim_{service}_fallback_key"
                return None

            # Round-robin selection among candidates
            idx = self.current_index.get(service, 0) % len(candidates)
            chosen = candidates[idx]
            self.current_index[service] = (idx + 1) % len(candidates)
            chosen.last_used = time.time()
            self.save_state()
            return chosen.key

    def record_usage(self, service: str, key: str, units: int = 1) -> None:
        """Deduct units from a key's quota and record usage."""
        service = service.lower().strip()
        with self._lock:
            pool = self.pools.get(service, [])
            for k in pool:
                if k.key == key:
                    k.used_units += units
                    k.last_used = time.time()
                    k.success_count += 1
                    break
            self.save_state()

    def mark_rate_limited(
        self,
        service: str,
        key: str,
        duration_seconds: float = 3600.0,
        reason: str = "Rate Limit 429 / Quota 403"
    ) -> None:
        """Blacklist a key for the current cycle or specified cooldown."""
        service = service.lower().strip()
        with self._lock:
            pool = self.pools.get(service, [])
            for k in pool:
                if k.key == key:
                    k.blacklisted_until = time.time() + duration_seconds
                    k.last_error = reason
                    k.failure_count += 1
                    logger.warning(
                        f"Key {mask_key(key)} for {service} blacklisted for {duration_seconds}s. Reason: {reason}"
                    )
                    break
            self.save_state()

    def mark_invalid(self, service: str, key: str, reason: str = "401 Unauthorized / Invalid Key") -> None:
        """Mark a key permanently invalid until re-checked."""
        service = service.lower().strip()
        with self._lock:
            pool = self.pools.get(service, [])
            for k in pool:
                if k.key == key:
                    k.is_valid = False
                    k.last_error = reason
                    logger.error(f"Key {mask_key(key)} for {service} marked INVALID: {reason}")
                    break
            self.save_state()

    def verify_key(self, service: str, key: str) -> Tuple[bool, str]:
        """Perform a live health ping to verify key authenticity and reachability."""
        service = service.lower().strip()
        if not key or key.startswith("sim_"):
            return True, "Simulation/Mock key"

        try:
            if service == "youtube":
                url = "https://www.googleapis.com/youtube/v3/videos"
                params = {"part": "id", "chart": "mostPopular", "maxResults": 1, "key": key}
                resp = requests.get(url, params=params, timeout=8)
                if resp.status_code == 200:
                    return True, "YouTube Data API healthy"
                elif resp.status_code in (400, 401, 403, 429):
                    return False, f"YouTube API rejected ({resp.status_code}): {resp.text[:120]}"
                return True, f"Response {resp.status_code}"

            elif service == "openai":
                url = "https://api.openai.com/v1/models"
                headers = {"Authorization": f"Bearer {key}"}
                resp = requests.get(url, headers=headers, timeout=8)
                if resp.status_code == 200:
                    return True, "OpenAI API healthy"
                elif resp.status_code in (401, 403, 429):
                    return False, f"OpenAI auth/rate failure ({resp.status_code}): {resp.text[:120]}"
                return True, f"Response {resp.status_code}"

            elif service == "serpapi":
                url = "https://serpapi.com/account"
                params = {"api_key": key}
                resp = requests.get(url, params=params, timeout=8)
                if resp.status_code == 200:
                    return True, "SerpAPI healthy"
                elif resp.status_code in (401, 403, 429):
                    return False, f"SerpAPI auth/rate failure ({resp.status_code})"
                return True, f"Response {resp.status_code}"

            elif service == "elevenlabs":
                url = "https://api.elevenlabs.io/v1/user"
                headers = {"xi-api-key": key}
                resp = requests.get(url, headers=headers, timeout=8)
                if resp.status_code == 200:
                    return True, "ElevenLabs API healthy"
                elif resp.status_code in (401, 403, 429):
                    return False, f"ElevenLabs auth/rate failure ({resp.status_code})"
                return True, f"Response {resp.status_code}"

            elif service == "pexels":
                url = "https://api.pexels.com/v1/curated"
                headers = {"Authorization": key}
                params = {"per_page": 1}
                resp = requests.get(url, headers=headers, params=params, timeout=8)
                if resp.status_code == 200:
                    return True, "Pexels API healthy"
                return False, f"Pexels response {resp.status_code}"

            return True, "Unknown service ping passed"
        except Exception as e:
            return False, f"Ping connection failed: {str(e)}"

    def verify_all_keys(self) -> Dict[str, List[Dict[str, Any]]]:
        """Verify all registered keys across all pools and update validity."""
        results: Dict[str, List[Dict[str, Any]]] = {}
        with self._lock:
            for service, states in self.pools.items():
                results[service] = []
                for s in states:
                    valid, message = self.verify_key(service, s.key)
                    s.is_valid = valid
                    if not valid:
                        s.last_error = message
                    results[service].append({
                        "key_masked": mask_key(s.key),
                        "valid": valid,
                        "status_message": message,
                        "remaining_percent": round(s.remaining_percent, 1),
                    })
            self.save_state()
        return results

    def get_pool_status(self) -> Dict[str, Any]:
        """Return a comprehensive health, quota, and blacklist report for UI / CLI."""
        with self._lock:
            self._check_and_apply_midnight_reset()
            status: Dict[str, Any] = {
                "last_reset_utc_date": self.last_reset_utc_date,
                "services": {}
            }
            for service, states in self.pools.items():
                status["services"][service] = {
                    "total_keys": len(states),
                    "healthy_keys": sum(1 for s in states if s.is_valid and not s.is_rate_limited and s.has_sufficient_quota),
                    "keys": [s.to_dict() for s in states]
                }
            return status


# Module-level convenience functions
def get_key_rotator() -> KeyRotator:
    return KeyRotator.get_instance()


def get_active_key(service: str) -> Optional[str]:
    """Retrieve the next healthy active API key for the requested service."""
    return KeyRotator.get_instance().get_active_key(service)


def record_usage(service: str, key: str, units: int = 1) -> None:
    """Record units used against a specific key."""
    KeyRotator.get_instance().record_usage(service, key, units)


def record_quota_usage(service: str, units: int = 1, key: Optional[str] = None) -> None:
    """Record quota usage on the specified key or the most recently used active key."""
    rotator = KeyRotator.get_instance()
    service = service.lower().strip()
    with rotator._lock:
        target_key = key
        pool = rotator.pools.get(service, [])
        if not target_key and pool:
            # Pick the key that was most recently used/acquired rather than rotating
            mru = max(pool, key=lambda k: k.last_used)
            target_key = mru.key
        if target_key:
            rotator.record_usage(service, target_key, units)


def mark_rate_limited(service: str, key: str, duration_seconds: float = 3600.0, reason: str = "") -> None:
    """Mark a key as rate limited for the current cycle or cooldown window."""
    KeyRotator.get_instance().mark_rate_limited(service, key, duration_seconds, reason)


def report_rate_limit(service: str, key: Optional[str] = None, duration_seconds: float = 3600.0, reason: str = "") -> None:
    """Report rate limit on a key or the most recently used active key."""
    rotator = KeyRotator.get_instance()
    service = service.lower().strip()
    with rotator._lock:
        target_key = key
        pool = rotator.pools.get(service, [])
        if (not target_key or target_key == "active_key") and pool:
            mru = max(pool, key=lambda k: k.last_used)
            target_key = mru.key
        if target_key:
            rotator.mark_rate_limited(service, target_key, duration_seconds, reason or "Rate limit reported")
