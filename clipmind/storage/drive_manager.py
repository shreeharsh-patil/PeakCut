"""Google Drive Integration & Asset Archival Engine for ClipMind (Phase 7).

Features:
- Automated Google Drive folder hierarchy:
    ClipMind/
      ├── {date}/
      │   └── Cycle_{id}/
      │       ├── final_clip.mp4
      │       ├── thumbnail_A.png
      │       ├── thumbnail_B.png
      │       ├── metadata.json
      │       ├── captions.srt
      │       └── cycle_log.txt
      ├── Archive/
      └── performance_tracker.csv
- Asset packaging and resumable upload to Google Drive.
- Safe local temp asset cleanup after successful cycle packaging.
- Automated 30-day asset archival (moves folders > 30 days old to Archive/).
- Full local mirroring and simulation mode when Drive credentials are not present.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from googleapiclient.discovery import build as google_build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    google_build = None
    HttpError = Exception  # type: ignore
    MediaFileUpload = None
    MediaIoBaseUpload = None
    service_account = None
    Credentials = None
    GOOGLE_DRIVE_AVAILABLE = False

logger = logging.getLogger("clipmind.storage.drive")

STANDARD_CYCLE_ASSETS = [
    "final_clip.mp4",
    "thumbnail_A.png",
    "thumbnail_B.png",
    "metadata.json",
    "captions.srt",
    "cycle_log.txt",
]

TRACKER_CSV_COLUMNS = [
    "cycle_id",
    "date",
    "timestamp",
    "topic",
    "title",
    "video_id",
    "status",
    "views",
    "ctr",
    "avd_pct",
    "engagement_rate",
    "drive_folder_id",
    "drive_folder_url",
]


class DriveManager:
    """Manages cloud storage hierarchies on Google Drive and local mirrors."""

    def __init__(
        self,
        credentials_path: Optional[Union[str, Path]] = None,
        credentials: Optional[Any] = None,
        local_base_dir: Optional[Union[str, Path]] = None,
        dry_run: Optional[bool] = None,
    ):
        self.credentials_path = credentials_path or os.environ.get("GOOGLE_DRIVE_CREDENTIALS")
        self.credentials = credentials
        self.local_base_dir = Path(local_base_dir or os.environ.get("CLIPMIND_STORAGE_DIR", "data/ClipMind"))
        self.local_base_dir.mkdir(parents=True, exist_ok=True)

        env_dry = os.environ.get("CLIPMIND_DRY_RUN", "0").lower() in ("1", "true", "yes")
        if dry_run is not None:
            self.dry_run = dry_run
        else:
            self.dry_run = env_dry or not (
                self.credentials or (self.credentials_path and Path(self.credentials_path).exists())
            )

        self._service = None
        self._root_folder_id = None
        self._archive_folder_id = None

        if self.dry_run:
            logger.info(f"DriveManager operating in LOCAL / SIMULATION mode at {self.local_base_dir}")
        else:
            logger.info("DriveManager operating with active Google Drive API client.")

    def _get_service(self) -> Any:
        """Initialize Google Drive v3 API client."""
        if self.dry_run:
            return None

        if self._service is not None:
            return self._service

        if not GOOGLE_DRIVE_AVAILABLE:
            raise RuntimeError("google-api-python-client is not installed; Google Drive API unavailable.")

        creds = self.credentials
        if creds is None:
            if self.credentials_path and Path(self.credentials_path).exists():
                path_str = str(self.credentials_path)
                try:
                    creds = service_account.Credentials.from_service_account_file(
                        path_str,
                        scopes=["https://www.googleapis.com/auth/drive"],
                    )
                except Exception:
                    # Try OAuth2 user credentials format
                    creds = Credentials.from_authorized_user_file(
                        path_str,
                        scopes=["https://www.googleapis.com/auth/drive"],
                    )
            else:
                raise ValueError("No valid Google Drive credentials or service account path provided.")

        self._service = google_build("drive", "v3", credentials=creds)
        return self._service

    def get_or_create_drive_folder(self, name: str, parent_id: Optional[str] = None) -> str:
        """Find or create a folder in Google Drive."""
        if self.dry_run:
            return f"gdrive_folder_{name.replace(' ', '_')}_{abs(hash(name)) % 10000}"

        service = self._get_service()
        query = f"mimeType = 'application/vnd.google-apps.folder' and name = '{name}' and trashed = false"
        if parent_id:
            query += f" and '{parent_id}' in parents"

        try:
            resp = service.files().list(
                q=query,
                spaces="drive",
                fields="files(id, name)",
                pageSize=1,
            ).execute()
            files = resp.get("files", [])
            if files:
                return files[0]["id"]

            file_metadata = {
                "name": name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_id:
                file_metadata["parents"] = [parent_id]

            folder = service.files().create(body=file_metadata, fields="id").execute()
            return folder.get("id")
        except HttpError as e:
            logger.error(f"Failed to find or create Drive folder '{name}': {e}")
            raise

    def get_root_and_archive_folders(self) -> Tuple[str, str]:
        """Ensure ClipMind root and Archive folders exist."""
        if self._root_folder_id and self._archive_folder_id:
            return self._root_folder_id, self._archive_folder_id

        root_id = self.get_or_create_drive_folder("ClipMind")
        archive_id = self.get_or_create_drive_folder("Archive", parent_id=root_id)
        self._root_folder_id = root_id
        self._archive_folder_id = archive_id
        return root_id, archive_id

    def package_cycle_assets(
        self,
        cycle_id: str,
        date_str: Optional[str] = None,
        assets: Optional[Dict[str, Union[str, Path]]] = None,
        metadata_dict: Optional[Dict[str, Any]] = None,
        cycle_log_text: Optional[str] = None,
        clean_temp: bool = False,
        temp_files_to_clean: Optional[List[Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        """Package all cycle assets into ClipMind/{date}/Cycle_{id}/ and sync to Drive.

        Args:
            cycle_id: Identifier of the generation cycle (e.g., "001" or "20261117_01").
            date_str: Folder date (defaults to today 'YYYY-MM-DD').
            assets: Dict mapping asset names ('final_clip.mp4', 'thumbnail_A.png', etc.) to local file paths.
            metadata_dict: Dict to be saved as metadata.json in the cycle folder.
            cycle_log_text: Text log to be saved as cycle_log.txt.
            clean_temp: Whether to clean up temporary rendering files after packaging.
            temp_files_to_clean: List of temp file paths to clean safely.

        Returns:
            Dict containing local path, drive folder id, asset map, and upload status.
        """
        date_tag = date_str or datetime.date.today().isoformat()
        cycle_folder_name = f"Cycle_{cycle_id}"

        # Setup local hierarchy: local_base_dir / date / Cycle_id
        cycle_local_dir = self.local_base_dir / date_tag / cycle_folder_name
        cycle_local_dir.mkdir(parents=True, exist_ok=True)

        asset_map = assets.copy() if assets else {}

        # Save metadata.json if provided
        if metadata_dict is not None:
            meta_path = cycle_local_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata_dict, f, indent=2)
            asset_map["metadata.json"] = meta_path

        # Save cycle_log.txt if provided
        if cycle_log_text is not None:
            log_path = cycle_local_dir / "cycle_log.txt"
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(cycle_log_text)
            asset_map["cycle_log.txt"] = log_path

        # Copy any external asset paths to the local cycle directory
        copied_assets: Dict[str, Path] = {}
        for asset_name, src_path in asset_map.items():
            src = Path(src_path)
            dest = cycle_local_dir / asset_name
            if src.exists() and src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
                copied_assets[asset_name] = dest
            elif dest.exists():
                copied_assets[asset_name] = dest
            elif src.exists():
                copied_assets[asset_name] = src

        # Ensure Drive hierarchy: ClipMind -> {date} -> Cycle_{id}
        root_id, _ = self.get_root_and_archive_folders()
        date_folder_id = self.get_or_create_drive_folder(date_tag, parent_id=root_id)
        cycle_folder_id = self.get_or_create_drive_folder(cycle_folder_name, parent_id=date_folder_id)

        # Upload files to Drive
        uploaded_records: Dict[str, str] = {}
        if not self.dry_run:
            service = self._get_service()
            for asset_name, file_path in copied_assets.items():
                if file_path.exists():
                    try:
                        media = MediaFileUpload(str(file_path), resumable=True)
                        file_metadata = {
                            "name": asset_name,
                            "parents": [cycle_folder_id],
                        }
                        uploaded_file = service.files().create(
                            body=file_metadata,
                            media_body=media,
                            fields="id, name, webViewLink",
                        ).execute()
                        uploaded_records[asset_name] = uploaded_file.get("id")
                        logger.info(f"Uploaded asset '{asset_name}' to Drive: {uploaded_file.get('id')}")
                    except Exception as e:
                        logger.warning(f"Failed to upload asset '{asset_name}' to Drive: {e}")
        else:
            for asset_name in copied_assets:
                uploaded_records[asset_name] = f"sim_file_{asset_name}_{abs(hash(asset_name)) % 1000}"

        drive_url = f"https://drive.google.com/drive/folders/{cycle_folder_id}"

        # Clean local temporary assets if requested
        cleaned_files = []
        if clean_temp and temp_files_to_clean:
            cleaned_files = self.clean_local_temp_assets(temp_files_to_clean)

        result = {
            "success": True,
            "cycle_id": cycle_id,
            "date": date_tag,
            "local_dir": str(cycle_local_dir),
            "drive_folder_id": cycle_folder_id,
            "drive_url": drive_url,
            "assets": {k: str(v) for k, v in copied_assets.items()},
            "uploaded_drive_files": uploaded_records,
            "cleaned_temp_files": cleaned_files,
            "simulated": self.dry_run,
        }
        return result

    def update_performance_tracker(self, record: Dict[str, Any]) -> Path:
        """Append or update a cycle record in performance_tracker.csv."""
        tracker_file = self.local_base_dir / "performance_tracker.csv"
        file_exists = tracker_file.exists()

        row_data = {
            "cycle_id": record.get("cycle_id", ""),
            "date": record.get("date", datetime.date.today().isoformat()),
            "timestamp": record.get("timestamp", datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "topic": record.get("topic", ""),
            "title": record.get("title", ""),
            "video_id": record.get("video_id", ""),
            "status": record.get("status", "completed"),
            "views": record.get("views", 0),
            "ctr": record.get("ctr", 0.0),
            "avd_pct": record.get("avd_pct", 0.0),
            "engagement_rate": record.get("engagement_rate", 0.0),
            "drive_folder_id": record.get("drive_folder_id", ""),
            "drive_folder_url": record.get("drive_url", ""),
        }

        with open(tracker_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACKER_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)

        logger.info(f"Appended cycle {row_data['cycle_id']} to {tracker_file}")

        # If live Drive is available, sync tracker to Drive root
        if not self.dry_run:
            try:
                service = self._get_service()
                root_id, _ = self.get_root_and_archive_folders()
                # Check if tracker already exists on Drive
                query = f"name = 'performance_tracker.csv' and '{root_id}' in parents and trashed = false"
                res = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
                items = res.get("files", [])
                media = MediaFileUpload(str(tracker_file), mimetype="text/csv", resumable=True)

                if items:
                    file_id = items[0]["id"]
                    service.files().update(fileId=file_id, media_body=media).execute()
                    logger.info(f"Updated performance_tracker.csv on Drive ({file_id})")
                else:
                    service.files().create(
                        body={"name": "performance_tracker.csv", "parents": [root_id]},
                        media_body=media,
                    ).execute()
                    logger.info("Created performance_tracker.csv on Drive root.")
            except Exception as e:
                logger.warning(f"Could not sync performance_tracker.csv to Drive: {e}")

        return tracker_file

    def clean_local_temp_assets(
        self,
        temp_paths: List[Union[str, Path]],
        keep_extensions: Optional[List[str]] = None,
    ) -> List[str]:
        """Safely delete intermediate render/temp files while preserving critical logs."""
        cleaned: List[str] = []
        extensions_to_keep = set(ext.lower() for ext in (keep_extensions or [".log", ".txt", ".json"]))

        for item in temp_paths:
            path = Path(item)
            if not path.exists():
                continue
            if path.suffix.lower() in extensions_to_keep:
                logger.debug(f"Preserving protected file: {path}")
                continue

            try:
                if path.is_file():
                    path.unlink()
                    cleaned.append(str(path))
                elif path.is_dir() and "temp" in path.name.lower():
                    shutil.rmtree(path)
                    cleaned.append(str(path))
            except Exception as e:
                logger.warning(f"Error removing temp asset {path}: {e}")

        logger.info(f"Cleaned {len(cleaned)} temporary local files.")
        return cleaned

    def archive_assets_older_than(self, days_threshold: int = 30) -> Dict[str, Any]:
        """Move cycle folders older than `days_threshold` to Archive/ locally and on Drive.

        Args:
            days_threshold: Age in days before archiving (default: 30).

        Returns:
            Dict containing archived local folders and Drive folders.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        archived_local: List[str] = []
        archived_drive: List[str] = []

        # 1. Local archival
        local_archive_dir = self.local_base_dir / "Archive"
        local_archive_dir.mkdir(parents=True, exist_ok=True)

        for date_dir in self.local_base_dir.iterdir():
            if not date_dir.is_dir() or date_dir.name == "Archive":
                continue

            try:
                # Try parsing folder date YYYY-MM-DD
                folder_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d").replace(
                    tzinfo=datetime.timezone.utc
                )
                age_days = (now - folder_date).days
                if age_days > days_threshold:
                    dest = local_archive_dir / date_dir.name
                    if not dest.exists():
                        shutil.move(str(date_dir), str(dest))
                        archived_local.append(date_dir.name)
                        logger.info(f"Locally archived {date_dir.name} (age: {age_days} days)")
            except ValueError:
                # Folder is not in YYYY-MM-DD format, check filesystem modified time
                mtime = datetime.datetime.fromtimestamp(date_dir.stat().st_mtime, tz=datetime.timezone.utc)
                if (now - mtime).days > days_threshold:
                    dest = local_archive_dir / date_dir.name
                    if not dest.exists():
                        shutil.move(str(date_dir), str(dest))
                        archived_local.append(date_dir.name)

        # 2. Google Drive Archival
        if not self.dry_run:
            try:
                service = self._get_service()
                root_id, archive_id = self.get_root_and_archive_folders()
                # Query all date folders inside ClipMind root
                query = f"mimeType = 'application/vnd.google-apps.folder' and '{root_id}' in parents and trashed = false"
                res = service.files().list(q=query, spaces="drive", fields="files(id, name, createdTime)").execute()

                for folder in res.get("files", []):
                    name = folder.get("name")
                    folder_id = folder.get("id")
                    if name in ("Archive",):
                        continue

                    should_archive = False
                    try:
                        f_date = datetime.datetime.strptime(name, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
                        if (now - f_date).days > days_threshold:
                            should_archive = True
                    except ValueError:
                        created = folder.get("createdTime")
                        if created:
                            dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                            if (now - dt).days > days_threshold:
                                should_archive = True

                    if should_archive:
                        # Move folder to Archive by modifying parents
                        service.files().update(
                            fileId=folder_id,
                            addParents=archive_id,
                            removeParents=root_id,
                            fields="id, parents",
                        ).execute()
                        archived_drive.append(name)
                        logger.info(f"Archived Drive folder {name} ({folder_id}) into Archive/")
            except Exception as e:
                logger.error(f"Error archiving Drive folders: {e}")
        else:
            archived_drive = [f"sim_archived_{f}" for f in archived_local]

        return {
            "threshold_days": days_threshold,
            "archived_local_count": len(archived_local),
            "archived_local_folders": archived_local,
            "archived_drive_count": len(archived_drive),
            "archived_drive_folders": archived_drive,
        }


# Module level convenience functions
def package_cycle_assets(
    cycle_id: str,
    date_str: Optional[str] = None,
    assets: Optional[Dict[str, Union[str, Path]]] = None,
    metadata_dict: Optional[Dict[str, Any]] = None,
    cycle_log_text: Optional[str] = None,
    clean_temp: bool = False,
    temp_files: Optional[List[Union[str, Path]]] = None,
    dry_run: Optional[bool] = None,
) -> Dict[str, Any]:
    manager = DriveManager(dry_run=dry_run)
    return manager.package_cycle_assets(
        cycle_id=cycle_id,
        date_str=date_str,
        assets=assets,
        metadata_dict=metadata_dict,
        cycle_log_text=cycle_log_text,
        clean_temp=clean_temp,
        temp_files_to_clean=temp_files,
    )


def archive_old_assets(days_threshold: int = 30, dry_run: Optional[bool] = None) -> Dict[str, Any]:
    manager = DriveManager(dry_run=dry_run)
    return manager.archive_assets_older_than(days_threshold=days_threshold)
