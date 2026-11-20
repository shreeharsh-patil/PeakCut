"""Creator Workspaces Engine for OpenShorts.

Features:
- Independent workspace per creator with custom clipping rules, channel profiles, and styling presets
- Persistent storage under `output/creators/{cid}/creator.json` and append-only event stream `log.jsonl`
- ROI / Billing estimation based on views and custom `rate_per_1k`
- Live stream preferences (auto-watch, rolling buffer minutes, trigger phrases)
- Autopost rules (linked publishing accounts, minimum score threshold, posting gap, daily cap)
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("openshorts.creators")

BASE_CREATORS_DIR = Path("output/creators")

def get_base_creators_dir() -> Path:
    import os
    return Path(os.environ.get("CREATORS_DIR", "output/creators"))

DEFAULT_CREATOR = {
    "name": "",
    "channel": "",
    "channels": {},  # "youtube": "...", "twitch": "...", "kick": "..."
    "style": "bounce",
    "layout": "split",  # split, track, screencast, blur
    "min_len": 15,
    "max_len": 60,
    "max_clips": 6,
    "min_score": 80,
    "auto_watch": False,
    "buffer_min": 20,
    "live_per_hour": 4,
    "ai_every_min": 3,
    "ai_only_on_spike": True,
    "auto_jump_pct": 60,
    "trigger_words": "clip that, clip it",
    "autopost": False,
    "accounts": [],
    "post_min_score": 75,
    "post_gap_min": 45,
    "daily_cap": 6,
    "privacy": "public",
    "post_footer": "#Shorts",
    "rate_per_1k": 0.0,
}


class CreatorWorkspaceManager:
    """Manages creator workspace records, configurations, and logs."""

    _instance: Optional[CreatorWorkspaceManager] = None

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir) if base_dir is not None else get_base_creators_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_all()

    @classmethod
    def get_instance(cls) -> CreatorWorkspaceManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls, instance: Optional[CreatorWorkspaceManager] = None) -> None:
        cls._instance = instance

    def _slug(self, name: str) -> str:
        s = re.sub(r"[^a-zA-Z0-9_-]", "-", name.lower()).strip("-")
        return s[:40] or "creator"

    def _load_all(self) -> None:
        with self.lock:
            for c_dir in self.base_dir.iterdir():
                if c_dir.is_dir():
                    c_file = c_dir / "creator.json"
                    if c_file.exists():
                        try:
                            data = json.loads(c_file.read_text(encoding="utf-8"))
                            cid = data.get("id") or c_dir.name
                            self._cache[cid] = {**DEFAULT_CREATOR, **data, "id": cid}
                        except Exception as e:
                            logger.warning(f"Failed to load creator {c_file}: {e}")

    def list_creators(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self._cache.values())

    def get_creator(self, cid: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self._cache.get(cid)

    def save_creator(self, cid: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            creator_id = cid or self._slug(data.get("name") or "new-creator")
            if not cid and creator_id in self._cache:
                creator_id = f"{creator_id}-{int(time.time()) % 1000}"

            c_dir = self.base_dir / creator_id
            c_dir.mkdir(parents=True, exist_ok=True)

            merged = {**DEFAULT_CREATOR, **self._cache.get(creator_id, {}), **data, "id": creator_id}

            c_file = c_dir / "creator.json"
            tmp = c_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
            tmp.replace(c_file)

            self._cache[creator_id] = merged
            self.log_event(creator_id, f"Workspace settings updated for {merged.get('name')}")
            return merged

    def delete_creator(self, cid: str) -> bool:
        with self.lock:
            if cid in self._cache:
                del self._cache[cid]
                c_dir = self.base_dir / cid
                if c_dir.exists():
                    import shutil
                    shutil.rmtree(c_dir, ignore_errors=True)
                return True
            return False

    def log_event(self, cid: str, message: str) -> None:
        """Append an event to the creator's activity stream."""
        c_dir = self.base_dir / cid
        if not c_dir.exists():
            return
        log_file = c_dir / "log.jsonl"
        entry = {"t": time.time(), "msg": message}
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def read_logs(self, cid: str, max_entries: int = 30) -> List[Dict[str, Any]]:
        log_file = self.base_dir / cid / "log.jsonl"
        if not log_file.exists():
            return []
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            entries = []
            for ln in lines[-max_entries:]:
                if ln.strip():
                    entries.append(json.loads(ln))
            return entries
        except Exception:
            return []
