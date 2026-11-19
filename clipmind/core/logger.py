"""Structured Cycle Logger for ClipMind.

Writes per-cycle JSON audit logs to `logs/cycles/cycle_{cycle_id}.json`
recording:
- cycle_id
- topic
- VPS (Viral Potential Score)
- video_id
- rotated_keys used during the run
- errors encountered
- execution status and timestamps per stage
- console formatting using Rich
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from clipmind.core.config import get_config

# Configure base logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    from rich.console import Console
    from rich.theme import Theme
    custom_theme = Theme({
        "info": "cyan",
        "warning": "yellow",
        "error": "bold red",
        "success": "bold green",
        "stage": "bold magenta",
    })
    console = Console(theme=custom_theme)
except ImportError:
    console = None  # type: ignore


@dataclass
class StepLog:
    """Log details for an individual pipeline step."""
    name: str
    status: str = "PENDING"  # PENDING, RUNNING, SUCCESS, FAILED, SKIPPED
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class CycleLogger:
    """Tracks and records end-to-end execution of a single ClipMind cycle."""

    def __init__(self, cycle_id: str, topic: str = "", vps: float = 0.0):
        self.config = get_config()
        self.cycle_id = cycle_id
        self.topic = topic
        self.vps = vps
        self.video_id: Optional[str] = None
        self.status: str = "STARTED"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.total_duration_seconds: float = 0.0
        self._start_time = time.time()

        self.rotated_keys: Dict[str, List[str]] = {}
        self.errors: List[Dict[str, Any]] = []
        self.steps: Dict[str, StepLog] = {}
        self._current_step_start: Dict[str, float] = {}

        try:
            self.config.paths.cycle_logs_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = self.config.paths.cycle_logs_dir / f"cycle_{self.cycle_id}.json"
        except OSError:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "clipmind" / "logs" / "cycles"
            temp_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = temp_dir / f"cycle_{self.cycle_id}.json"
        self.save()

    def set_topic(self, topic: str, vps: float = 0.0) -> None:
        """Update selected topic and VPS score."""
        self.topic = topic
        self.vps = vps
        self.info(f"Topic selected: '{topic}' (VPS: {vps:.2f})")
        self.save()

    def start_step(self, step_name: str, description: str = "") -> None:
        """Record the beginning of a pipeline phase."""
        now_str = datetime.now(timezone.utc).isoformat()
        self._current_step_start[step_name] = time.time()
        self.steps[step_name] = StepLog(
            name=step_name,
            status="RUNNING",
            started_at=now_str,
            details={"description": description} if description else {}
        )
        if console:
            console.print(f"[stage]>>> [{step_name}][/stage] {description}")
        else:
            logging.info(f">>> [{step_name}] {description}")
        self.save()

    def finish_step(
        self,
        step_name: str,
        status: str = "SUCCESS",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Record the completion of a pipeline phase."""
        now_str = datetime.now(timezone.utc).isoformat()
        step = self.steps.get(step_name)
        if not step:
            step = StepLog(name=step_name)
            self.steps[step_name] = step

        start_t = self._current_step_start.pop(step_name, None)
        duration = round(time.time() - start_t, 2) if start_t else 0.0

        step.status = status
        step.finished_at = now_str
        step.duration_seconds = duration
        if details:
            step.details.update(details)
        if error:
            step.error = error

        msg = f"<<< [{step_name}] {status} in {duration}s"
        if error:
            msg += f" - Error: {error}"

        if console:
            color = "success" if status == "SUCCESS" else ("error" if status == "FAILED" else "warning")
            console.print(f"[{color}]{msg}[/{color}]")
        else:
            logging.info(msg)
        self.save()

    def record_key_usage(self, service: str, masked_key: str) -> None:
        """Track which API keys were leveraged in this cycle."""
        if service not in self.rotated_keys:
            self.rotated_keys[service] = []
        if masked_key not in self.rotated_keys[service]:
            self.rotated_keys[service].append(masked_key)
            self.save()

    def record_error(self, step_name: str, error_message: str, exc: Optional[Exception] = None) -> None:
        """Capture an error or retry event."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step_name,
            "message": error_message,
            "exception_type": type(exc).__name__ if exc else None,
        }
        self.errors.append(entry)
        if console:
            console.print(f"[error]ERROR in [{step_name}]: {error_message}[/error]")
        else:
            logging.error(f"ERROR in [{step_name}]: {error_message}")
        self.save()

    def complete(self, video_id: Optional[str] = None, status: str = "COMPLETED") -> None:
        """Finalize the cycle record."""
        self.video_id = video_id
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.total_duration_seconds = round(time.time() - self._start_time, 2)
        if console:
            console.print(
                f"[success]Cycle {self.cycle_id} finished with status '{status}' "
                f"in {self.total_duration_seconds}s (Video ID: {video_id or 'N/A'})[/success]"
            )
        else:
            logging.info(
                f"Cycle {self.cycle_id} finished with status '{status}' "
                f"in {self.total_duration_seconds}s (Video ID: {video_id or 'N/A'})"
            )
        self.save()

    def fail(self, error_message: str) -> None:
        """Mark cycle as failed."""
        self.record_error("CYCLE_ABORT", error_message)
        self.complete(status="FAILED")

    def info(self, message: str) -> None:
        logging.info(f"[{self.cycle_id}] {message}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert cycle state to serializable dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "topic": self.topic,
            "VPS": self.vps,
            "video_id": self.video_id,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "total_duration_seconds": self.total_duration_seconds,
            "rotated_keys": self.rotated_keys,
            "errors": self.errors,
            "steps": {k: asdict(v) for k, v in self.steps.items()},
        }

    def save(self) -> None:
        """Write JSON log file to disk."""
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to write cycle log {self.log_file}: {e}")


def get_logger(name: str = "clipmind") -> logging.Logger:
    """Return a standard python logger."""
    return logging.getLogger(name)


def get_cycle_logger(cycle_id: str, topic: str = "", vps: float = 0.0) -> CycleLogger:
    """Factory for CycleLogger instances."""
    return CycleLogger(cycle_id, topic, vps)
