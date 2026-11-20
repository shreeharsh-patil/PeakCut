"""Tests for Creator Workspaces (creator_workspaces.py)."""
from pathlib import Path

from engine.services.creator_workspaces import CreatorWorkspaceManager


def test_creator_workspaces_crud(tmp_path: Path):
    manager = CreatorWorkspaceManager(base_dir=tmp_path)

    assert len(manager.list_creators()) == 0

    c1 = manager.save_creator(
        cid=None,
        data={
            "name": "Shroud Gaming",
            "channel": "https://twitch.tv/shroud",
            "channels": {"twitch": "https://twitch.tv/shroud"},
            "style": "bounce",
            "layout": "split",
            "auto_watch": True,
            "rate_per_1k": 0.50,
        },
    )

    cid = c1["id"]
    assert cid.startswith("shroud-gaming")
    assert c1["auto_watch"] is True
    assert len(manager.list_creators()) == 1

    # Retrieve
    retrieved = manager.get_creator(cid)
    assert retrieved is not None
    assert retrieved["name"] == "Shroud Gaming"

    # Event logging
    manager.log_event(cid, "Stream started")
    logs = manager.read_logs(cid)
    assert len(logs) >= 2  # created/updated + stream started
    assert any("Stream started" in l["msg"] for l in logs)

    # Delete
    assert manager.delete_creator(cid) is True
    assert manager.get_creator(cid) is None
    assert len(manager.list_creators()) == 0
