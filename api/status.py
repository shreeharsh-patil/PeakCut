"""Vercel Status Endpoint for ClipMind (/api/status).

Returns:
- Engine version & serverless platform metadata
- Current niche, timezone, and upload frequency
- API key pools, quota usage, rate-limit state, and health check results
- Recent cycle JSON records
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from clipmind import __version__
from clipmind.core.config import get_config
from clipmind.core.key_rotator import get_key_rotator

app = FastAPI(title="ClipMind Status API")


@app.get("/api/status")
@app.get("/")
def get_status():
    config = get_config()
    rotator = get_key_rotator()
    pool_status = rotator.get_pool_status()

    # Read recent cycle logs
    recent_cycles = []
    log_dir = config.paths.cycle_logs_dir
    if log_dir.exists():
        for f in sorted(log_dir.glob("cycle_*.json"), key=os.path.getmtime, reverse=True)[:5]:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    recent_cycles.append(json.load(fp))
            except Exception:
                pass

    return JSONResponse(content={
        "service": "ClipMind",
        "version": __version__,
        "platform": "vercel_serverless",
        "niche": config.niche,
        "audience_timezone": config.audience_timezone,
        "upload_frequency_hours": config.upload_frequency_hours,
        "dry_run": config.dry_run,
        "key_pools": pool_status,
        "recent_cycles": recent_cycles,
    })


# Vercel entry point
handler = app
