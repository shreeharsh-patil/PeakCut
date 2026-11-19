"""YouTube Shorts Uploader & Publishing Engine for ClipMind (Phase 6).

Handles:
- Resumable video upload with rich snippet metadata (title, description, tags, category, public status, madeForKids=False).
- Algorithmic timing & scheduling: calculates optimal engagement windows
  (US: 12PM-3PM EST, 7PM-9PM EST; Global: 2PM-4PM UTC) and schedules via publishAt.
- Post-upload actions: automated niche playlist discovery/creation and video insertion.
- Pinned engagement comment publishing to maximize comment velocity.
- API Key / OAuth rotation and quota tracking integration with ClipMind core.
- Robust dry-run / simulation mode for safe headless testing.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import zoneinfo
except ImportError:
    zoneinfo = None  # type: ignore

# Optional imports for Google API client
try:
    from googleapiclient.discovery import build as google_build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload
    from google.oauth2.credentials import Credentials
    GOOGLE_API_AVAILABLE = True
except ImportError:
    google_build = None
    HttpError = Exception  # type: ignore
    MediaFileUpload = None
    Credentials = None
    GOOGLE_API_AVAILABLE = False

# ClipMind core integrations (graceful fallback if running isolated)
try:
    from clipmind.core.key_rotator import get_active_key, record_quota_usage, report_rate_limit
except ImportError:
    def get_active_key(service: str) -> Optional[str]:
        return os.environ.get(f"{service.upper()}_API_KEY") or os.environ.get(f"{service.upper()}_KEY")

    def record_quota_usage(service: str, units: int = 1) -> None:
        pass

    def report_rate_limit(service: str, key: str) -> None:
        pass

logger = logging.getLogger("clipmind.distribution.youtube")

# YouTube API Quota costs
QUOTA_VIDEO_UPLOAD = 1600
QUOTA_PLAYLIST_WRITE = 50
QUOTA_COMMENT_WRITE = 50
QUOTA_READ = 1


@dataclass
class OptimalWindowResult:
    """Represents the algorithmic upload window calculation."""
    is_optimal_now: bool
    current_time_utc: datetime.datetime
    target_region: str
    active_window_label: Optional[str] = None
    next_window_start_utc: Optional[datetime.datetime] = None
    next_window_end_utc: Optional[datetime.datetime] = None
    publish_at_iso: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_optimal_now": self.is_optimal_now,
            "current_time_utc": self.current_time_utc.isoformat(),
            "target_region": self.target_region,
            "active_window_label": self.active_window_label,
            "next_window_start_utc": self.next_window_start_utc.isoformat() if self.next_window_start_utc else None,
            "next_window_end_utc": self.next_window_end_utc.isoformat() if self.next_window_end_utc else None,
            "publish_at_iso": self.publish_at_iso,
        }


def _get_eastern_tz() -> datetime.tzinfo:
    """Return US Eastern timezone with DST support or static UTC-5 fallback."""
    if zoneinfo is not None:
        try:
            return zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            pass
    # Fallback to standard Eastern Standard Time (UTC-5)
    return datetime.timezone(datetime.timedelta(hours=-5))


def compute_optimal_upload_window(
    target_region: str = "US",
    current_time: Optional[datetime.datetime] = None,
) -> OptimalWindowResult:
    """Compute whether current time is in an optimal window or determine next window.

    Optimal Publishing Windows:
    - US Audience:
        * Afternoon Peak: 12:00 PM - 3:00 PM EST (12:00 - 15:00)
        * Evening Peak:   7:00 PM - 9:00 PM EST (19:00 - 21:00)
    - Global Audience:
        * International Prime: 2:00 PM - 4:00 PM UTC (14:00 - 16:00 UTC)

    If current_time is outside optimal windows, computes next window start time
    and formats RFC 3339 / ISO 8601 publishAt for YouTube scheduled upload.
    """
    region = target_region.upper()
    now_utc = current_time or datetime.datetime.now(datetime.timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=datetime.timezone.utc)
    else:
        now_utc = now_utc.astimezone(datetime.timezone.utc)

    if region == "US":
        tz_eastern = _get_eastern_tz()
        now_local = now_utc.astimezone(tz_eastern)
        today = now_local.date()

        # Window 1: 12:00 PM - 3:00 PM EST
        # Window 2: 7:00 PM - 9:00 PM EST
        windows_local = [
            (
                "US Afternoon Peak (12PM-3PM EST)",
                datetime.datetime.combine(today, datetime.time(12, 0), tzinfo=tz_eastern),
                datetime.datetime.combine(today, datetime.time(15, 0), tzinfo=tz_eastern),
            ),
            (
                "US Evening Peak (7PM-9PM EST)",
                datetime.datetime.combine(today, datetime.time(19, 0), tzinfo=tz_eastern),
                datetime.datetime.combine(today, datetime.time(21, 0), tzinfo=tz_eastern),
            ),
        ]

        # Check if inside any today window
        for label, start, end in windows_local:
            if start <= now_local <= end:
                return OptimalWindowResult(
                    is_optimal_now=True,
                    current_time_utc=now_utc,
                    target_region="US",
                    active_window_label=label,
                    publish_at_iso=None,
                )

        # Outside window: find next upcoming window
        upcoming: List[Tuple[str, datetime.datetime, datetime.datetime]] = []
        for label, start, end in windows_local:
            if start > now_local:
                upcoming.append((label, start, end))

        if not upcoming:
            # All today's windows passed, schedule for tomorrow's first window
            tomorrow = today + datetime.timedelta(days=1)
            next_start = datetime.datetime.combine(tomorrow, datetime.time(12, 0), tzinfo=tz_eastern)
            next_end = datetime.datetime.combine(tomorrow, datetime.time(15, 0), tzinfo=tz_eastern)
            next_label = "US Afternoon Peak (12PM-3PM EST)"
        else:
            next_label, next_start, next_end = upcoming[0]

        next_start_utc = next_start.astimezone(datetime.timezone.utc)
        next_end_utc = next_end.astimezone(datetime.timezone.utc)
        publish_at_iso = next_start_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        return OptimalWindowResult(
            is_optimal_now=False,
            current_time_utc=now_utc,
            target_region="US",
            active_window_label=next_label,
            next_window_start_utc=next_start_utc,
            next_window_end_utc=next_end_utc,
            publish_at_iso=publish_at_iso,
        )

    else:
        # Global: 2:00 PM - 4:00 PM UTC
        today_utc = now_utc.date()
        win_start = datetime.datetime.combine(today_utc, datetime.time(14, 0), tzinfo=datetime.timezone.utc)
        win_end = datetime.datetime.combine(today_utc, datetime.time(16, 0), tzinfo=datetime.timezone.utc)
        label = "Global International Prime (2PM-4PM UTC)"

        if win_start <= now_utc <= win_end:
            return OptimalWindowResult(
                is_optimal_now=True,
                current_time_utc=now_utc,
                target_region="GLOBAL",
                active_window_label=label,
                publish_at_iso=None,
            )

        if now_utc < win_start:
            next_start_utc = win_start
            next_end_utc = win_end
        else:
            tomorrow_utc = today_utc + datetime.timedelta(days=1)
            next_start_utc = datetime.datetime.combine(tomorrow_utc, datetime.time(14, 0), tzinfo=datetime.timezone.utc)
            next_end_utc = datetime.datetime.combine(tomorrow_utc, datetime.time(16, 0), tzinfo=datetime.timezone.utc)

        publish_at_iso = next_start_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        return OptimalWindowResult(
            is_optimal_now=False,
            current_time_utc=now_utc,
            target_region="GLOBAL",
            active_window_label=label,
            next_window_start_utc=next_start_utc,
            next_window_end_utc=next_end_utc,
            publish_at_iso=publish_at_iso,
        )


class YouTubeUploader:
    """Automated YouTube Uploader with scheduling, playlists, and engagement comments."""

    def __init__(
        self,
        credentials: Optional[Any] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        refresh_token: Optional[str] = None,
        access_token: Optional[str] = None,
        dry_run: Optional[bool] = None,
    ):
        self.client_id = client_id or os.environ.get("YOUTUBE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET")
        self.refresh_token = refresh_token or os.environ.get("YOUTUBE_REFRESH_TOKEN")
        self.access_token = access_token or os.environ.get("YOUTUBE_ACCESS_TOKEN")
        self.credentials = credentials
        self._service = None

        # Determine dry run mode
        env_dry = os.environ.get("CLIPMIND_DRY_RUN", "0").lower() in ("1", "true", "yes")
        if dry_run is not None:
            self.dry_run = dry_run
        else:
            self.dry_run = env_dry or not (
                self.credentials or self.access_token or (self.refresh_token and self.client_id and self.client_secret)
            )

        if self.dry_run:
            logger.info("YouTubeUploader initialized in SIMULATION / DRY-RUN mode.")
        else:
            logger.info("YouTubeUploader initialized with live YouTube credentials.")

    def _get_service(self) -> Any:
        """Initialize or return authenticated Google API service client."""
        if self.dry_run:
            return None

        if self._service is not None:
            return self._service

        if not GOOGLE_API_AVAILABLE:
            raise RuntimeError("google-api-python-client is not installed; live upload unavailable.")

        creds = self.credentials
        if creds is None:
            if self.refresh_token and self.client_id and self.client_secret:
                creds = Credentials(
                    token=self.access_token,
                    refresh_token=self.refresh_token,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    scopes=[
                        "https://www.googleapis.com/auth/youtube.upload",
                        "https://www.googleapis.com/auth/youtube",
                        "https://www.googleapis.com/auth/youtube.force-ssl",
                    ],
                )
            elif self.access_token:
                creds = Credentials(token=self.access_token)
            else:
                api_key = get_active_key("youtube")
                if api_key:
                    self._service = google_build("youtube", "v3", developerKey=api_key)
                    return self._service
                raise ValueError("Missing YouTube credentials (refresh token, access token, or API key).")

        self._service = google_build("youtube", "v3", credentials=creds)
        return self._service

    def upload_short(
        self,
        video_path: Union[str, Path],
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
        category_id: str = "22",
        privacy_status: str = "public",
        target_region: str = "US",
        auto_schedule: bool = True,
        playlist_name: Optional[str] = None,
        pinned_comment: Optional[str] = None,
        forced_publish_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a video as a YouTube Short.

        Args:
            video_path: Path to MP4 video file.
            title: Title (truncated to 100 chars, ensuring #Shorts tag).
            description: Description body (hook, timestamps, hashtags, CTA).
            tags: List of SEO tags (up to ~500 chars).
            category_id: YouTube category ID ("22" for People & Blogs, "24" for Entertainment).
            privacy_status: "public", "unlisted", or "private".
            target_region: "US" or "GLOBAL" for algorithmic timing.
            auto_schedule: If True and outside optimal window, sets status='private' with publishAt.
            playlist_name: Optional playlist to discover/create and append video.
            pinned_comment: Optional engagement comment to publish.
            forced_publish_at: Explicit RFC 3339 string to override auto-scheduling.

        Returns:
            Dictionary with video_id, url, timing_info, playlist_id, comment_id, and status.
        """
        vpath = Path(video_path)
        if not self.dry_run and not vpath.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Ensure #Shorts is present in title or description
        clean_title = title.strip()
        if "#Shorts" not in clean_title and "#shorts" not in clean_title:
            if len(clean_title) <= 92:
                clean_title = f"{clean_title} #Shorts"
        clean_title = clean_title[:100]

        clean_description = description.strip()
        if "#Shorts" not in clean_description and "#shorts" not in clean_description:
            clean_description = f"{clean_description}\n\n#Shorts #ClipMind".strip()
        clean_description = clean_description[:5000]

        cleaned_tags = tags or ["Shorts", "ClipMind", "Viral"]
        # Format snippet
        snippet_body: Dict[str, Any] = {
            "title": clean_title,
            "description": clean_description,
            "tags": cleaned_tags,
            "categoryId": str(category_id),
            "defaultLanguage": "en",
        }

        # Calculate optimal timing
        timing_info = compute_optimal_upload_window(target_region=target_region)
        final_publish_at: Optional[str] = None
        final_privacy = privacy_status.lower()

        if forced_publish_at:
            if isinstance(forced_publish_at, (datetime.datetime, datetime.date)):
                if isinstance(forced_publish_at, datetime.datetime) and forced_publish_at.tzinfo is None:
                    forced_publish_at = forced_publish_at.replace(tzinfo=datetime.timezone.utc)
                forced_publish_at = forced_publish_at.isoformat()
            if isinstance(forced_publish_at, str):
                forced_publish_at = forced_publish_at.replace("+00:00", "Z")
                if not (
                    forced_publish_at.endswith("Z") or "+" in forced_publish_at[10:] or "-" in forced_publish_at[10:]
                ):
                    forced_publish_at = f"{forced_publish_at}Z"
            final_publish_at = forced_publish_at
            final_privacy = "private"  # Scheduled videos in YT must be private
            is_scheduled = True
        elif auto_schedule and not timing_info.is_optimal_now:
            final_publish_at = timing_info.publish_at_iso
            final_privacy = "private"  # Scheduled videos in YT must be private
            is_scheduled = True
        else:
            is_scheduled = False

        status_body: Dict[str, Any] = {
            "privacyStatus": final_privacy,
            "selfDeclaredMadeForKids": False,
        }
        if final_publish_at:
            status_body["publishAt"] = final_publish_at

        # Execute upload (or simulate)
        if self.dry_run:
            simulated_id = f"sim_{int(time.time())}_{abs(hash(clean_title)) % 10000}"
            short_url = f"https://youtube.com/shorts/{simulated_id}"
            logger.info(
                f"[SIMULATION] Uploaded video '{clean_title}' -> {short_url} "
                f"(scheduled={is_scheduled}, publishAt={final_publish_at}, privacy={final_privacy})"
            )

            playlist_res = None
            if playlist_name:
                playlist_res = self.get_or_create_playlist(playlist_name)
                logger.info(f"[SIMULATION] Added video {simulated_id} to playlist '{playlist_name}'")

            comment_res = None
            if pinned_comment:
                comment_res = self.post_engagement_comment(simulated_id, pinned_comment)
                logger.info(f"[SIMULATION] Pinned comment posted: '{pinned_comment[:50]}...'")

            return {
                "success": True,
                "video_id": simulated_id,
                "url": short_url,
                "title": clean_title,
                "is_scheduled": is_scheduled,
                "publish_at": final_publish_at,
                "privacy_status": final_privacy,
                "playlist_id": playlist_res.get("playlist_id") if playlist_res else None,
                "comment_id": comment_res.get("comment_id") if comment_res else None,
                "timing": timing_info.to_dict(),
                "simulated": True,
            }

        # Live Upload
        service = self._get_service()
        body = {
            "snippet": snippet_body,
            "status": status_body,
        }

        media = MediaFileUpload(
            str(vpath),
            mimetype="video/mp4",
            resumable=True,
            chunksize=10 * 1024 * 1024,
        )

        try:
            request = service.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )
            response = None
            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    logger.debug(f"Upload progress: {progress}%")

            video_id = response.get("id")
            if not video_id:
                raise RuntimeError(f"YouTube upload failed: missing video id in response: {response}")

            record_quota_usage("youtube", QUOTA_VIDEO_UPLOAD)
            short_url = f"https://youtube.com/shorts/{video_id}"
            logger.info(f"✅ Successfully uploaded YouTube Short: {short_url}")

            # Post-upload: playlist
            playlist_id = None
            if playlist_name:
                try:
                    pl_data = self.get_or_create_playlist(playlist_name)
                    playlist_id = pl_data.get("playlist_id")
                    if playlist_id:
                        self.add_to_playlist(video_id, playlist_id)
                except Exception as e:
                    logger.warning(f"Playlist assignment failed: {e}")

            # Post-upload: pinned comment
            comment_id = None
            if pinned_comment:
                try:
                    c_data = self.post_engagement_comment(video_id, pinned_comment)
                    comment_id = c_data.get("comment_id")
                except Exception as e:
                    logger.warning(f"Posting engagement comment failed: {e}")

            return {
                "success": True,
                "video_id": video_id,
                "url": short_url,
                "title": clean_title,
                "is_scheduled": is_scheduled,
                "publish_at": final_publish_at,
                "privacy_status": final_privacy,
                "playlist_id": playlist_id,
                "comment_id": comment_id,
                "timing": timing_info.to_dict(),
                "simulated": False,
                "raw_response": response,
            }

        except HttpError as err:
            logger.error(f"YouTube API HttpError during upload: {err}")
            status_code = getattr(err, "status_code", None) or getattr(getattr(err, "resp", None), "status", None)
            if status_code in (403, 429):
                report_rate_limit("youtube", reason=f"HTTP {status_code} quota limit exceeded")
            raise
        except (ConnectionError, TimeoutError, OSError) as err:
            logger.error(f"YouTube upload network/connection error (offline or unreachable): {err}")
            raise

    def get_or_create_playlist(
        self,
        title: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Find existing playlist by title or create a new public niche playlist."""
        if self.dry_run:
            mock_pid = f"PL_sim_{abs(hash(title)) % 100000}"
            return {"playlist_id": mock_pid, "title": title, "created": False, "simulated": True}

        service = self._get_service()
        clean_title = title.strip()
        desc = description or f"Best {clean_title} curated by ClipMind."

        # Search existing playlists
        try:
            req = service.playlists().list(
                part="snippet,id",
                mine=True,
                maxResults=50,
            )
            resp = req.execute()
            record_quota_usage("youtube", QUOTA_READ)

            for item in resp.get("items", []):
                snippet = item.get("snippet", {})
                if snippet.get("title", "").strip().lower() == clean_title.lower():
                    logger.info(f"Found existing playlist '{clean_title}': {item['id']}")
                    return {"playlist_id": item["id"], "title": clean_title, "created": False}

            # Not found -> create new playlist
            create_req = service.playlists().insert(
                part="snippet,status",
                body={
                    "snippet": {
                        "title": clean_title,
                        "description": desc,
                    },
                    "status": {
                        "privacyStatus": "public",
                    },
                },
            )
            create_resp = create_req.execute()
            record_quota_usage("youtube", QUOTA_PLAYLIST_WRITE)
            new_pid = create_resp.get("id")
            logger.info(f"Created new playlist '{clean_title}': {new_pid}")
            return {"playlist_id": new_pid, "title": clean_title, "created": True}

        except (HttpError, ConnectionError, TimeoutError, OSError, Exception) as e:
            logger.error(f"Error managing playlist '{title}': {e}")
            raise

    def add_to_playlist(self, video_id: str, playlist_id: str) -> Dict[str, Any]:
        """Add a video to a specific playlist."""
        if self.dry_run:
            return {"success": True, "playlist_id": playlist_id, "video_id": video_id, "simulated": True}

        service = self._get_service()
        try:
            req = service.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": video_id,
                        },
                    },
                },
            )
            res = req.execute()
            record_quota_usage("youtube", QUOTA_PLAYLIST_WRITE)
            return {"success": True, "item_id": res.get("id"), "playlist_id": playlist_id}
        except (HttpError, ConnectionError, TimeoutError, OSError, Exception) as e:
            logger.warning(f"Failed to add video {video_id} to playlist {playlist_id}: {e}")
            raise

    def post_engagement_comment(
        self,
        video_id: str,
        comment_text: str,
    ) -> Dict[str, Any]:
        """Post a top-level pinned engagement comment on the video to prompt viewer discussion."""
        if self.dry_run:
            return {
                "success": True,
                "comment_id": f"comment_sim_{abs(hash(comment_text)) % 10000}",
                "video_id": video_id,
                "text": comment_text,
                "simulated": True,
            }

        service = self._get_service()
        try:
            req = service.commentThreads().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": video_id,
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": comment_text,
                            },
                        },
                    },
                },
            )
            res = req.execute()
            record_quota_usage("youtube", QUOTA_COMMENT_WRITE)
            comment_id = res.get("id")
            logger.info(f"Posted engagement comment on video {video_id}: {comment_id}")
            return {
                "success": True,
                "comment_id": comment_id,
                "video_id": video_id,
                "text": comment_text,
            }
        except (HttpError, ConnectionError, TimeoutError, OSError, Exception) as e:
            logger.warning(f"Failed to post engagement comment on video {video_id}: {e}")
            raise


# Module level shortcut
def upload_short(
    video_path: Union[str, Path],
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    category_id: str = "22",
    privacy_status: str = "public",
    target_region: str = "US",
    auto_schedule: bool = True,
    playlist_name: Optional[str] = None,
    pinned_comment: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convenience function to upload a Short using YouTubeUploader."""
    uploader = YouTubeUploader(dry_run=dry_run)
    return uploader.upload_short(
        video_path=video_path,
        title=title,
        description=description,
        tags=tags,
        category_id=category_id,
        privacy_status=privacy_status,
        target_region=target_region,
        auto_schedule=auto_schedule,
        playlist_name=playlist_name,
        pinned_comment=pinned_comment,
    )
