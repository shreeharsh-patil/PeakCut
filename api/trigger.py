"""Manual Trigger Endpoint for ClipMind (/api/trigger).

Allows manual on-demand execution of an autonomous YouTube Shorts cycle.
Supports:
- Optional query params: ?niche=Tech&dry_run=true&secret=...
- Authentication via TRIGGER_SECRET or CRON_SECRET if configured.
- Immediate response with complete cycle execution record.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from clipmind.core.config import get_config
from clipmind.orchestrator import MasterOrchestrator

app = FastAPI(title="ClipMind Manual Trigger")


class TriggerRequest(BaseModel):
    niche: Optional[str] = None
    dry_run: Optional[bool] = None
    secret: Optional[str] = None


@app.get("/api/trigger")
@app.post("/api/trigger")
async def trigger_cycle(
    request: Request,
    niche: Optional[str] = Query(None),
    dry_run: Optional[bool] = Query(None),
    secret: Optional[str] = Query(None),
):
    """Trigger an autonomous ClipMind cycle immediately."""
    # Check authorization if secret configured
    expected_secret = os.getenv("TRIGGER_SECRET") or os.getenv("CRON_SECRET")
    auth_header = request.headers.get("authorization")

    token = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split("Bearer ", 1)[1].strip()
    elif secret:
        token = secret.strip()

    if expected_secret and token != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid trigger secret")

    # Parse body if JSON provided
    body_data = None
    if request.method == "POST":
        try:
            body_data = await request.json()
        except Exception:
            body_data = None

    config = get_config()

    # Apply overrides
    target_niche = niche or (body_data.get("niche") if body_data else None)
    if target_niche:
        config.niche = target_niche

    is_dry = dry_run if dry_run is not None else (body_data.get("dry_run") if body_data else None)
    if is_dry is not None:
        config.dry_run = is_dry

    start_t = time.time()
    orchestrator = MasterOrchestrator(config=config)
    result = orchestrator.run_cycle()
    duration = round(time.time() - start_t, 2)

    status_code = 200 if result.get("status") == "COMPLETED" else 500

    return JSONResponse(
        status_code=status_code,
        content={
            "triggered_by": "manual_api",
            "cycle_result": result,
            "duration_seconds": duration,
        }
    )


# Vercel entry point
handler = app
