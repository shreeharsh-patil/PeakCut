"""FastAPI Router for Supercharged Capabilities in OpenShorts.

Includes:
- Live stream rolling-buffer clipping endpoints (`/api/live/*`)
- Autonomous topic & niche discovery endpoints (`/api/discover/*`)
- Multi-platform channel watcher endpoints (`/api/channels/*`)
- Smart publishing queue & scheduler endpoints (`/api/queue/*`)
- Non-speech audio energy highlight detection (`/api/highlights/*`)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from channel_watcher import get_channel_watcher, WatchedChannel
from highlight_detector import detect_hype_moments
from live_clipper import LiveClipperManager, resolve_live_stream
from niche_discovery import NicheDiscoveryEngine, Topic
from publishing_queue import get_publishing_queue, PostItem

logger = logging.getLogger("openshorts.routes")

router = APIRouter(tags=["Supercharged Features"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LiveStartRequest(BaseModel):
    stream_url: str
    buffer_minutes: int = Field(default=15, ge=3, le=120)
    session_id: Optional[str] = None


class LiveStopRequest(BaseModel):
    session_id: str


class LiveClipRequest(BaseModel):
    session_id: str
    start_sec: float
    end_sec: float
    title: Optional[str] = None


class DiscoverEnqueueRequest(BaseModel):
    provider: str = Field(default="youtube_trending")
    custom_seeds: Optional[List[str]] = None
    target_clips: Optional[int] = 3
    layouts: Optional[List[str]] = Field(default_factory=lambda: ["auto"])
    auto_hook: bool = True


class WatchChannelRequest(BaseModel):
    platform: str = Field(description="'youtube', 'twitch', or 'kick'")
    channel_identifier: str = Field(description="Handle (@...), Channel ID (UC...), or username")
    creator_name: Optional[str] = None
    mode: str = Field(default="both", description="'live', 'vod', or 'both'")
    auto_publish: bool = False
    publish_account_id: Optional[str] = None
    target_layouts: Optional[List[str]] = None


class UnwatchChannelRequest(BaseModel):
    channel_key: str


class QueueEnqueueRequest(BaseModel):
    job_id: str
    clip_id: str
    platform: str = Field(description="'youtube', 'tiktok', or 'instagram'")
    account_id: str
    title: str
    video_path: str
    description: str = ""
    account_name: Optional[str] = None
    schedule_now: bool = False
    requested_time: Optional[float] = None


class QueuePostActionRequest(BaseModel):
    post_id: str


class HighlightDetectRequest(BaseModel):
    media_path: str
    max_clips: int = Field(default=5, ge=1, le=20)
    duration: float = Field(default=20.0, ge=5.0, le=90.0)


# ---------------------------------------------------------------------------
# 1. Live Stream Clipping Routes
# ---------------------------------------------------------------------------

@router.post("/api/live/start")
async def start_live_session(req: LiveStartRequest):
    """Start rolling-buffer recording and spike monitoring for a live stream."""
    manager = LiveClipperManager.get_instance()
    try:
        session = manager.create_session(
            stream_url=req.stream_url,
            buffer_minutes=req.buffer_minutes,
            session_id=req.session_id,
        )
        return {
            "status": "started",
            "session_id": session.session_id,
            "stream_url": session.stream_url,
            "buffer_minutes": session.buffer_minutes,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to start live session: {e}")


@router.post("/api/live/stop")
async def stop_live_session(req: LiveStopRequest):
    """Stop an active live stream recording session."""
    manager = LiveClipperManager.get_instance()
    success = manager.stop_session(req.session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")
    return {"status": "stopped", "session_id": req.session_id}


@router.get("/api/live/sessions")
async def list_live_sessions():
    """List all active and past live stream clipping sessions."""
    manager = LiveClipperManager.get_instance()
    return {"sessions": manager.list_sessions()}


@router.get("/api/live/status/{session_id}")
async def get_live_status(session_id: str):
    """Retrieve detailed real-time telemetry, buffer info, and extracted clips for a session."""
    manager = LiveClipperManager.get_instance()
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session.get_status()


@router.post("/api/live/clip")
async def extract_live_clip(req: LiveClipRequest):
    """Manually extract a clip of the requested time range from the rolling buffer."""
    manager = LiveClipperManager.get_instance()
    session = manager.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{req.session_id}' not found")

    event = session.trigger_clip(
        start_sec=req.start_sec,
        end_sec=req.end_sec,
        trigger_type="manual",
        title=req.title or f"Manual Clip @ {req.start_sec:.1f}s",
    )
    return {"status": "extracting", "clip_id": event.clip_id, "duration": event.duration}


# ---------------------------------------------------------------------------
# 2. Autonomous Niche & Topic Discovery Routes
# ---------------------------------------------------------------------------

@router.get("/api/discover/search")
async def search_discovery_topics(
    query: str = Query(..., description="Search topic query"),
    max_results: int = Query(10, ge=1, le=50),
    min_sec: int = Query(180, ge=30),
    max_sec: int = Query(1800, le=7200),
):
    """Search for candidate long-form videos for a specific topic query."""
    engine = NicheDiscoveryEngine()
    videos = engine.search_youtube_api(
        query=query,
        max_results=max_results,
        min_sec=min_sec,
        max_sec=max_sec,
    )
    unseen = engine.ledger.filter_unseen(videos)
    return {
        "query": query,
        "total_found": len(videos),
        "unseen_candidates": [v.to_dict() for v in unseen],
    }


@router.get("/api/discover/best")
async def discover_best_unseen(
    provider: str = Query("youtube_trending", description="'youtube_trending', 'sports', or 'seeds'"),
    custom_seeds: Optional[str] = Query(None, description="Comma-separated seed queries"),
):
    """Return the single top-ranking unseen source video across configured topics."""
    engine = NicheDiscoveryEngine()
    seeds = [s.strip() for s in custom_seeds.split(",")] if custom_seeds else None
    best = engine.find_best_unseen_candidate(provider=provider, custom_seeds=seeds)
    if not best:
        return {"candidate": None, "note": "No unseen candidates found for criteria."}
    return {"candidate": best.to_dict()}


@router.post("/api/discover/enqueue")
async def discover_and_enqueue(req: DiscoverEnqueueRequest, request: Request):
    """Autonomous discover -> enqueue pipeline: finds top unseen video and submits to OpenShorts."""
    engine = NicheDiscoveryEngine()
    best = engine.find_best_unseen_candidate(provider=req.provider, custom_seeds=req.custom_seeds)
    if not best:
        raise HTTPException(status_code=404, detail="No unseen candidate video found.")

    engine.ledger.mark_seen(
        best.video_id,
        title=best.title,
        metadata={"url": best.url, "channel": best.channel_title, "views": best.view_count},
    )

    # Submit directly to OpenShorts' in-process /api/process endpoint
    body = {
        "url": best.url,
        "acknowledged": True,
        "target_clips": req.target_clips,
        "layouts": req.layouts or ["auto"],
        "auto_hook": req.auto_hook,
    }

    try:
        transport = httpx.ASGITransport(app=request.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://internal") as client:
            headers = {}
            for h in ("authorization", "x-api-key", "x-gemini-key"):
                if h in request.headers:
                    headers[h] = request.headers[h]
            resp = await client.post("/api/process", json=body, headers=headers)
            if resp.status_code >= 400:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            job_data = resp.json()
            return {
                "discovered_video": best.to_dict(),
                "job": job_data,
                "status": "enqueued",
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit discovery job: {e}")


# ---------------------------------------------------------------------------
# 3. Autonomous Channel Watcher Routes
# ---------------------------------------------------------------------------

@router.post("/api/channels/watch")
async def watch_channel(req: WatchChannelRequest):
    """Register a creator channel to watch for live streams and new VOD uploads."""
    watcher = get_channel_watcher()
    channel = watcher.add_channel(
        platform=req.platform,
        channel_identifier=req.channel_identifier,
        creator_name=req.creator_name or "",
        mode=req.mode,
        auto_publish=req.auto_publish,
        publish_account_id=req.publish_account_id or "",
        target_layouts=req.target_layouts,
    )
    return {"status": "watching", "channel": channel.to_dict()}


@router.post("/api/channels/unwatch")
async def unwatch_channel(req: UnwatchChannelRequest):
    """Remove a watched channel."""
    watcher = get_channel_watcher()
    removed = watcher.remove_channel(req.channel_key)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Channel '{req.channel_key}' not found.")
    return {"status": "removed", "channel_key": req.channel_key}


@router.get("/api/channels/list")
async def list_watched_channels():
    """List all registered watched channels and their latest statuses."""
    watcher = get_channel_watcher()
    return {"channels": watcher.list_channels()}


@router.post("/api/channels/poll")
async def poll_channels_now():
    """Trigger an immediate poll of all watched channels."""
    watcher = get_channel_watcher()
    watcher.poll_all_channels()
    return {"status": "polled", "channels": watcher.list_channels()}


# ---------------------------------------------------------------------------
# 4. Multi-Platform Publishing Queue Routes
# ---------------------------------------------------------------------------

@router.post("/api/queue/enqueue")
async def enqueue_post(req: QueueEnqueueRequest):
    """Enqueue a finished clip for scheduled, spaced publication across platforms."""
    queue = get_publishing_queue()
    item = queue.enqueue(
        job_id=req.job_id,
        clip_id=req.clip_id,
        platform=req.platform,
        account_id=req.account_id,
        title=req.title,
        video_path=req.video_path,
        description=req.description,
        account_name=req.account_name or req.account_id,
        schedule_now=req.schedule_now,
        requested_time=req.requested_time,
    )
    return {"status": "scheduled", "post": item.to_dict()}


@router.get("/api/queue/list")
async def list_queue_posts(
    account_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
):
    """List posts in the publishing queue filtered by account or status."""
    queue = get_publishing_queue()
    return {"posts": queue.list_posts(account_id=account_id, status=status)}


@router.post("/api/queue/cancel")
async def cancel_queue_post(req: QueuePostActionRequest):
    """Cancel a scheduled or failed post."""
    queue = get_publishing_queue()
    cancelled = queue.cancel_post(req.post_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Cannot cancel post (not found or already posted).")
    return {"status": "cancelled", "post_id": req.post_id}


@router.post("/api/queue/retry")
async def retry_queue_post(req: QueuePostActionRequest):
    """Retry a failed or cancelled post immediately."""
    queue = get_publishing_queue()
    retried = queue.retry_post(req.post_id)
    if not retried:
        raise HTTPException(status_code=400, detail="Cannot retry post (not found or not in failed/cancelled state).")
    return {"status": "retrying", "post_id": req.post_id}


@router.get("/api/queue/slot")
async def preview_next_slot(account_id: str, platform: str):
    """Preview the next free scheduled slot for an account respecting spacing and daily cap."""
    queue = get_publishing_queue()
    slot_ts = queue.calculate_next_slot(account_id, platform)
    return {
        "account_id": account_id,
        "platform": platform,
        "next_slot_timestamp": slot_ts,
        "next_slot_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(slot_ts)),
    }


# ---------------------------------------------------------------------------
# 5. Non-Speech Audio Energy Highlight Detection Routes
# ---------------------------------------------------------------------------

@router.post("/api/highlights/detect")
async def detect_audio_highlights(req: HighlightDetectRequest):
    """Detect non-speech excitement peaks (sports goals, esports cheers, reactions)."""
    if not os.path.exists(req.media_path):
        raise HTTPException(status_code=404, detail=f"Media file '{req.media_path}' not found.")

    moments = detect_hype_moments(
        req.media_path,
        max_clips=req.max_clips,
        target_clip_duration=req.duration,
    )
    return {
        "media_path": req.media_path,
        "moments_count": len(moments),
        "moments": [m.to_dict() for m in moments],
    }
