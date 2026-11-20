"""Tests for Multi-Channel Alert Dispatcher (alerts_dispatcher.py)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from engine.services.alerts_dispatcher import AlertDispatcher


def test_alert_dispatcher_settings(tmp_path: Path):
    settings_file = tmp_path / "alerts.json"
    dispatcher = AlertDispatcher(settings_path=settings_file)

    assert not dispatcher.is_configured()

    dispatcher.save_settings({
        "discord_webhook": "https://discord.com/api/webhooks/test",
        "telegram_token": "bot123456",
        "telegram_chat": "chat987",
    })

    assert dispatcher.is_configured()

    # Reload from disk
    dispatcher2 = AlertDispatcher(settings_path=settings_file)
    assert dispatcher2.is_configured()
    assert dispatcher2.settings["discord_webhook"] == "https://discord.com/api/webhooks/test"


def test_discord_and_telegram_dispatch(tmp_path: Path):
    dispatcher = AlertDispatcher(settings_path=tmp_path / "alerts.json")

    with patch("httpx.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        ok_discord = dispatcher.send_discord("Hello Discord", webhook_url="https://discord.com/api/webhooks/test")
        assert ok_discord is True
        assert mock_post.called

        mock_post.reset_mock()
        ok_tg = dispatcher.send_telegram("Hello Telegram", bot_token="token123", chat_id="chat123")
        assert ok_tg is True
        assert mock_post.called
