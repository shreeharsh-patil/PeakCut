"""Vercel Cron Trigger Endpoint for ClipMind (/api/cron).

Invoked by Vercel Cron on the configured 12-hour schedule: "0 */12 * * *".
Features:
- Verifies CRON_SECRET if configured (via Authorization: Bearer <secret> or query param).
- Executes an autonomous YouTube Shorts production & publishing cycle.
- Returns JSON execution telemetry and cycle status.
"""

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from clipmind.orchestrator import MasterOrchestrator
from clipmind.core.config import get_config

app = FastAPI(title="ClipMind Vercel Cron")


def verify_cron_authorization(
    authorization: str | None = Header(None),
    secret: str | None = Query(None),
) -> bool:
    """Validate CRON_SECRET if configured in the environment."""
    expected_secret = os.getenv("CRON_SECRET")
    if not expected_secret:
        # If no secret configured, allow trigger (e.g. initial dev setup)
        return True

    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split("Bearer ", 1)[1].strip()
    elif secret:
        token = secret.strip()

    if token != expected_secret:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid CRON_SECRET")
    return True


@app.get("/api/cron")
@app.post("/api/cron")
async def run_cron(request: Request):
    """Execute autonomous YouTube Shorts cycle from Vercel Cron schedule."""
    # Check authorization
    auth_header = request.headers.get("authorization")
    secret_param = request.query_params.get("secret")
    verify_cron_authorization(authorization=auth_header, secret=secret_param)

    config = get_config()
    start_t = time.time()
    
    # Initialize and execute cycle
    orchestrator = MasterOrchestrator(config=config)
    result = orchestrator.run_cycle()
    duration = round(time.time() - start_t, 2)

    status_code = 200 if result.get("status") == "COMPLETED" else 500

    return JSONResponse(
        status_code=status_code,
        content={
            "triggered_by": "vercel_cron",
            "cycle_result": result,
            "duration_seconds": duration,
        }
    )


# Vercel entry point
handler = app
