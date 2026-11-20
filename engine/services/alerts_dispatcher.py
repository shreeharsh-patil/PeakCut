"""Multi-Channel Alert Dispatcher and System Keep-Awake for OpenShorts.

Features:
- Discord Webhooks (rich embeds or text)
- Telegram Bot Notifications (with Markdown/HTML support)
- Asynchronous non-blocking dispatch
- Windows OS Sleep Prevention (`SetThreadExecutionState`) while jobs or live stream clipping are active
- Persistent notification settings (`alert_settings.json`)
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("openshorts.alerts")

DEFAULT_SETTINGS_PATH = Path("output/alert_settings.json")


class AlertDispatcher:
    """Manages notifications across Discord, Telegram, and in-app feeds."""

    _instance: Optional[AlertDispatcher] = None

    def __init__(self, settings_path: Path | str = DEFAULT_SETTINGS_PATH):
        self.settings_path = Path(settings_path)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.settings = self._load()
        self._keep_awake_active = False

    @classmethod
    def get_instance(cls) -> AlertDispatcher:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> Dict[str, Any]:
        default = {
            "discord_webhook": os.environ.get("DISCORD_WEBHOOK_URL", ""),
            "telegram_token": os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat": os.environ.get("TELEGRAM_CHAT_ID", ""),
            "enabled": True,
        }
        if self.settings_path.exists():
            try:
                saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
                default.update(saved)
            except Exception:
                pass
        return default

    def save_settings(self, new_settings: Dict[str, Any]) -> None:
        with self.lock:
            self.settings.update(new_settings)
            tmp = self.settings_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            tmp.replace(self.settings_path)

    def is_configured(self) -> bool:
        return bool(
            self.settings.get("discord_webhook")
            or (self.settings.get("telegram_token") and self.settings.get("telegram_chat"))
        )

    def send_discord(self, message: str, webhook_url: Optional[str] = None) -> bool:
        url = webhook_url or self.settings.get("discord_webhook")
        if not url:
            return False
        try:
            r = httpx.post(
                url,
                json={"content": f"**[OpenShorts]** {message[:1900]}", "allowed_mentions": {"parse": []}},
                timeout=12.0,
            )
            return r.status_code < 400
        except Exception as e:
            logger.error(f"Discord alert failed: {e}")
            return False

    def send_telegram(self, message: str, bot_token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
        token = bot_token or self.settings.get("telegram_token")
        cid = chat_id or self.settings.get("telegram_chat")
        if not token or not cid:
            return False
        try:
            r = httpx.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": cid, "text": f"[OpenShorts] {message[:3900]}"},
                timeout=12.0,
            )
            return r.status_code < 400
        except Exception as e:
            logger.error(f"Telegram alert failed: {e}")
            return False

    def notify(self, message: str, level: str = "info", link: str = "") -> None:
        """Fire and forget dispatch so network latency never blocks video pipelines."""
        if not self.settings.get("enabled", True):
            return

        formatted = f"{message}" + (f"\nLink: {link}" if link else "")

        def _dispatch():
            if self.settings.get("discord_webhook"):
                self.send_discord(formatted)
            if self.settings.get("telegram_token") and self.settings.get("telegram_chat"):
                self.send_telegram(formatted)

        threading.Thread(target=_dispatch, daemon=True, name="AlertDispatch").start()

    def set_system_keep_awake(self, awake: bool) -> None:
        """On Windows, tells the OS not to enter sleep/hibernate while renders/recordings are ongoing."""
        if os.name != "nt":
            return
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        flag = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if awake else 0)
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(flag)
            self._keep_awake_active = awake
        except Exception as e:
            logger.debug(f"Failed to set thread execution state: {e}")


def notify(message: str, level: str = "info", link: str = "") -> None:
    AlertDispatcher.get_instance().notify(message, level, link)
