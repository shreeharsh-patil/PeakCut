"""YouTube Shorts Uploader & Automated Publishing Dispatcher for PeakCut.

Provides:
- Resumable video upload to YouTube Data API v3 without heavy external SDK dependencies.
- Automatic Shorts detection, titling with #Shorts, and metadata formatting.
- Headless token refresh via OAuth 2.0 (YOUTUBE_REFRESH_TOKEN, YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET).
- Seamless integration with PeakCut's PublishingQueue (`register_youtube_dispatcher`).
- Standalone CLI for direct video uploading.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger("peakcut.youtube_uploader")

YOUTUBE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
YOUTUBE_UPLOAD_ENDPOINT = "https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status"
DEFAULT_CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB chunks for resumable upload


class YouTubeAuthError(Exception):
    """Raised when YouTube authentication or token refresh fails."""
    pass


class YouTubeUploadError(Exception):
    """Raised when YouTube video upload fails."""
    pass


def refresh_access_token(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> str:
    """Obtain a fresh OAuth2 access token using a refresh token."""
    cid = client_id or os.environ.get("YOUTUBE_CLIENT_ID")
    csecret = client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET")
    rtoken = refresh_token or os.environ.get("YOUTUBE_REFRESH_TOKEN")

    if not (cid and csecret and rtoken):
        raise YouTubeAuthError(
            "Missing OAuth credentials. Provide YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, "
            "and YOUTUBE_REFRESH_TOKEN environment variables."
        )

    data = urllib.parse.urlencode({
        "client_id": cid,
        "client_secret": csecret,
        "refresh_token": rtoken,
        "grant_type": "refresh_token",
    }).encode("utf-8")

    req = urllib.request.Request(
        YOUTUBE_TOKEN_ENDPOINT,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            access_token = payload.get("access_token")
            if not access_token:
                raise YouTubeAuthError(f"No access_token returned by Google: {payload}")
            return access_token
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise YouTubeAuthError(f"Google OAuth token refresh failed ({e.code}): {body}") from e
    except Exception as e:
        raise YouTubeAuthError(f"Token refresh request failed: {e}") from e


def get_authenticated_token() -> str:
    """Return an active access token from direct environment or via refresh token."""
    # Check for direct access token (useful for short-lived test sessions or Colab OAuth)
    direct_token = os.environ.get("YOUTUBE_ACCESS_TOKEN", "").strip()
    if direct_token:
        return direct_token

    return refresh_access_token()


def is_youtube_auth_configured(
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    refresh_token: Optional[str] = None,
    access_token: Optional[str] = None,
) -> bool:
    """Return True if sufficient credentials exist for direct YouTube upload."""
    token = access_token or os.environ.get("YOUTUBE_ACCESS_TOKEN", "").strip()
    if token:
        return True
    cid = client_id or os.environ.get("YOUTUBE_CLIENT_ID", "").strip()
    csec = client_secret or os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip()
    rtoken = refresh_token or os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip()
    return bool(cid and csec and rtoken)


def get_youtube_readiness_status() -> Dict[str, Any]:
    """Inspect environment and return direct YouTube upload readiness metadata."""
    has_token = bool(os.environ.get("YOUTUBE_ACCESS_TOKEN", "").strip())
    cid = bool(os.environ.get("YOUTUBE_CLIENT_ID", "").strip())
    csec = bool(os.environ.get("YOUTUBE_CLIENT_SECRET", "").strip())
    rtoken = bool(os.environ.get("YOUTUBE_REFRESH_TOKEN", "").strip())
    is_ready = has_token or (cid and csec and rtoken)
    default_privacy = os.environ.get("YOUTUBE_DEFAULT_PRIVACY", "private").lower().strip()
    if default_privacy not in ("private", "unlisted", "public"):
        default_privacy = "private"

    return {
        "ready": is_ready,
        "direct_auth_ready": is_ready,
        "has_access_token": has_token,
        "has_client_id": cid,
        "has_client_secret": csec,
        "has_refresh_token": rtoken,
        "default_privacy": default_privacy,
    }


def upload_youtube_short(
    video_path: str | Path,
    title: str,
    description: str = "",
    tags: Optional[List[str]] = None,
    category_id: str = "22",  # 22 = People & Blogs, 24 = Entertainment
    privacy_status: Optional[str] = None,  # 'public', 'unlisted', or 'private'
    publish_at: Optional[Any] = None,  # ISO 8601 / RFC 3339 string or datetime for scheduled publishing
    access_token: Optional[str] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    """Upload a video file as a YouTube Short using Google's resumable upload protocol."""
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Determine simulation / dry-run mode
    if dry_run is None:
        dry_run = (
            os.environ.get("YOUTUBE_DRY_RUN", "0").lower() in ("1", "true", "yes")
            or os.environ.get("CLIPMIND_DRY_RUN", "0").lower() in ("1", "true", "yes")
        )

    # Resolve confirmed privacy status safely
    if not privacy_status:
        privacy_status = os.environ.get("YOUTUBE_DEFAULT_PRIVACY", "unlisted")
    privacy_status = str(privacy_status).strip().lower()
    if privacy_status not in ("public", "unlisted", "private"):
        privacy_status = "private"

    # Ensure #Shorts is present in title or description to trigger YouTube Shorts recognition
    if "#Shorts" not in title and "#shorts" not in title:
        title = f"{title.strip()} #Shorts"

    if "#Shorts" not in description and "#shorts" not in description:
        description = f"{description.strip()}\n\n#Shorts #Viral #PeakCut".strip()

    # Normalize publish_at formatting
    formatted_publish_at: Optional[str] = None
    if publish_at:
        import datetime
        if isinstance(publish_at, (datetime.datetime, datetime.date)):
            if isinstance(publish_at, datetime.datetime) and publish_at.tzinfo is None:
                publish_at = publish_at.replace(tzinfo=datetime.timezone.utc)
            formatted_publish_at = publish_at.isoformat()
        else:
            formatted_publish_at = str(publish_at).strip()

        if formatted_publish_at:
            formatted_publish_at = formatted_publish_at.replace("+00:00", "Z")
            if not (
                formatted_publish_at.endswith("Z")
                or "+" in formatted_publish_at[10:]
                or "-" in formatted_publish_at[10:]
            ):
                formatted_publish_at = f"{formatted_publish_at}Z"

    if dry_run:
        import uuid
        sim_id = f"sim_{uuid.uuid4().hex[:11]}"
        short_url = f"https://youtube.com/shorts/{sim_id}"
        logger.info(
            f"[SIMULATION] Uploaded YouTube Short: '{title}' -> {short_url} "
            f"(scheduled={bool(formatted_publish_at)}, publishAt={formatted_publish_at})"
        )
        return {
            "id": sim_id,
            "url": short_url,
            "title": title,
            "privacy": "private" if formatted_publish_at else privacy_status,
            "publish_at": formatted_publish_at,
            "simulated": True,
        }

    token = access_token or get_authenticated_token()
    file_size = path.stat().st_size

    status_dict: Dict[str, Any] = {
        "selfDeclaredMadeForKids": False,
    }
    if formatted_publish_at:
        # YouTube requires scheduled videos to have status='private' alongside publishAt
        status_dict["privacyStatus"] = "private"
        status_dict["publishAt"] = formatted_publish_at
    else:
        status_dict["privacyStatus"] = privacy_status.lower()

    metadata = {
        "snippet": {
            "title": title[:100],  # YouTube title limit is 100 characters
            "description": description[:5000],
            "tags": tags or ["Shorts", "Viral", "PeakCut"],
            "categoryId": str(category_id),
        },
        "status": status_dict,
    }

    # Step 1: Initialize Resumable Upload Session
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    init_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size),
    }

    init_req = urllib.request.Request(
        YOUTUBE_UPLOAD_ENDPOINT,
        data=metadata_bytes,
        headers=init_headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(init_req, timeout=30) as resp:
            upload_url = resp.headers.get("Location")
            if not upload_url:
                raise YouTubeUploadError("YouTube API did not return an upload Location header.")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise YouTubeUploadError(f"YouTube upload initialization failed ({e.code}): {err_body}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise YouTubeUploadError(f"YouTube upload connection failed (offline or unreachable): {e}") from e

    # Step 2: Stream video bytes to the resumable upload URL
    logger.info(f"Uploading {file_size} bytes to YouTube resumable session...")
    with open(path, "rb") as video_file:
        video_data = video_file.read()

    upload_headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "video/mp4",
        "Content-Length": str(file_size),
    }

    upload_req = urllib.request.Request(
        upload_url,
        data=video_data,
        headers=upload_headers,
        method="PUT",
    )

    try:
        with urllib.request.urlopen(upload_req, timeout=120) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            video_id = resp_data.get("id")
            if not video_id:
                raise YouTubeUploadError(f"YouTube response missing video id: {resp_data}")

            short_url = f"https://youtube.com/shorts/{video_id}"
            logger.info(f"✅ Successfully uploaded YouTube Short: {short_url}")

            # Record quota if ClipMind key rotator is present
            try:
                from clipmind.core.key_rotator import record_quota_usage
                record_quota_usage("youtube", 1600)
            except Exception:
                pass

            return {
                "id": video_id,
                "url": short_url,
                "title": title,
                "privacy": "private" if formatted_publish_at else privacy_status,
                "publish_at": formatted_publish_at,
                "raw_response": resp_data,
            }
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        raise YouTubeUploadError(f"YouTube video upload stream failed ({e.code}): {err_body}") from e
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise YouTubeUploadError(f"YouTube video stream upload failed (offline or disconnected): {e}") from e


def register_youtube_dispatcher(queue: Any) -> None:
    """Register YouTube upload handler with PeakCut's PublishingQueue."""
    def _dispatcher(post_item: Any) -> Dict[str, Any]:
        privacy = "unlisted"
        publish_at = None
        dry_run = None
        if hasattr(post_item, "metadata") and isinstance(post_item.metadata, dict):
            privacy = post_item.metadata.get("privacy_status") or post_item.metadata.get("privacy") or os.environ.get("YOUTUBE_DEFAULT_PRIVACY", "unlisted")
            publish_at = post_item.metadata.get("publish_at")
            dry_run = post_item.metadata.get("dry_run")

        if dry_run is None and not is_youtube_auth_configured():
            logger.warning("No direct YouTube credentials configured; falling back to dry-run mode for PublishingQueue item.")
            dry_run = True

        result = upload_youtube_short(
            video_path=post_item.video_path,
            title=post_item.title,
            description=post_item.description,
            privacy_status=privacy,
            publish_at=publish_at,
            dry_run=dry_run,
        )
        return {
            "remote_id": result["id"],
            "remote_url": result["url"],
            "privacy": result.get("privacy", privacy),
            "publish_at": result.get("publish_at"),
            "simulated": result.get("simulated", False),
        }

    queue.register_dispatcher("youtube", _dispatcher)
    logger.info("Registered YouTube Shorts dispatcher with PublishingQueue.")


def main():
    parser = argparse.ArgumentParser(description="PeakCut YouTube Shorts Uploader CLI")
    parser.add_argument("-v", "--video", required=True, help="Path to the MP4 video file to upload")
    parser.add_argument("-t", "--title", required=True, help="Title of the YouTube Short")
    parser.add_argument("-d", "--description", default="", help="Video description")
    parser.add_argument("-p", "--privacy", default="unlisted", choices=["public", "unlisted", "private"],
                        help="Video privacy status (default: unlisted)")
    parser.add_argument("--publish-at", default=None,
                        help="ISO 8601 / RFC 3339 timestamp for scheduled release (e.g. 2026-11-18T23:00:00Z)")
    args = parser.parse_args()

    try:
        res = upload_youtube_short(
            video_path=args.video,
            title=args.title,
            description=args.description,
            privacy_status=args.privacy,
            publish_at=args.publish_at,
        )
        print(f"🎉 YouTube Short Uploaded Successfully!")
        print(f"ID:  {res['id']}")
        print(f"URL: {res['url']}")
        if res.get("publish_at"):
            print(f"⏰ Scheduled for Release at: {res['publish_at']}")
    except Exception as e:
        print(f"❌ Error uploading video: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
