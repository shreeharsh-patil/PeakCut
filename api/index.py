"""FastAPI Serverless Application for ClipMind on Vercel.

Serves:
- /: Web dashboard displaying engine status, active niche, and recent cycles
- /health: Health probe endpoint
- /api/status: JSON status report (quotas, keys, scheduler settings)
- /api/quotas: Key quotas and rate limit report
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for Vercel Serverless environment
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from clipmind.core.config import get_config
from clipmind.core.key_rotator import get_key_rotator
from clipmind import __version__

app = FastAPI(
    title="ClipMind Vercel Engine",
    description="Serverless Autonomous YouTube Shorts Generator & Uploader",
    version=__version__,
)


@app.get("/health")
def health_check():
    """Health check endpoint for Vercel monitoring."""
    return {
        "status": "healthy",
        "service": "clipmind",
        "version": __version__,
        "serverless": True,
    }


@app.get("/api/status")
@app.get("/status")
def status_endpoint():
    """Detailed JSON report of engine configuration, key pools, and recent cycle logs."""
    config = get_config()
    rotator = get_key_rotator()
    pool_status = rotator.get_pool_status()

    # Discover recent cycle logs
    recent_cycles = []
    log_dir = config.paths.cycle_logs_dir
    if log_dir.exists():
        import json
        for log_file in sorted(log_dir.glob("cycle_*.json"), key=os.path.getmtime, reverse=True)[:5]:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    recent_cycles.append(json.load(f))
            except Exception:
                pass

    return {
        "service": "ClipMind",
        "version": __version__,
        "environment": "vercel_serverless",
        "niche": config.niche,
        "audience_timezone": config.audience_timezone,
        "upload_frequency_hours": config.upload_frequency_hours,
        "key_pools": pool_status,
        "recent_cycles": recent_cycles,
    }


@app.get("/api/quotas")
def quotas_endpoint():
    """Key quotas and health status."""
    rotator = get_key_rotator()
    return rotator.get_pool_status()


@app.get("/", response_class=HTMLResponse)
def dashboard_html(request: Request):
    """Clean visual status dashboard for ClipMind."""
    config = get_config()
    rotator = get_key_rotator()
    status_data = rotator.get_pool_status()
    services = status_data.get("services", {})

    key_rows = ""
    for s_name, s_info in services.items():
        keys = s_info.get("keys", [])
        if not keys:
            key_rows += f"""
            <tr>
                <td style="padding:10px; border-bottom:1px solid #333; font-weight:bold;">{s_name.upper()}</td>
                <td style="padding:10px; border-bottom:1px solid #333; color:#888;">&lt;none configured&gt;</td>
                <td style="padding:10px; border-bottom:1px solid #333;">N/A</td>
                <td style="padding:10px; border-bottom:1px solid #333; color:#aaa;">Simulation active</td>
            </tr>
            """
        for k in keys:
            valid_color = "#4ade80" if k.get("is_valid") else "#f87171"
            rem_pct = k.get("remaining_percent", 100.0)
            key_rows += f"""
            <tr>
                <td style="padding:10px; border-bottom:1px solid #333; font-weight:bold;">{s_name.upper()}</td>
                <td style="padding:10px; border-bottom:1px solid #333; font-family:monospace;">{k.get('key_masked')}</td>
                <td style="padding:10px; border-bottom:1px solid #333;">{rem_pct:.1f}% ({k.get('remaining_units')} left)</td>
                <td style="padding:10px; border-bottom:1px solid #333; color:{valid_color};">{'HEALTHY' if k.get('is_valid') else 'ATTENTION'}</td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>ClipMind Engine Status</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 40px 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
            h1 {{ margin-top: 0; color: #38bdf8; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: bold; background: #0284c7; color: white; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
            th {{ text-align: left; padding: 10px; border-bottom: 2px solid #475569; color: #94a3b8; font-size: 14px; }}
            .btn {{ display: inline-block; padding: 10px 18px; border-radius: 8px; background: #2563eb; color: white; text-decoration: none; font-weight: 500; margin-right: 10px; }}
            .btn:hover {{ background: #1d4ed8; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>🎬 ClipMind Autonomous Engine <span class="badge">Vercel Serverless v{__version__}</span></h1>
                <p style="color: #94a3b8;">Autonomous YouTube Shorts Generator & Uploader running on 12-hour schedule via Vercel Cron.</p>
                <div style="display: flex; gap: 24px; margin-top: 16px;">
                    <div><strong>Niche:</strong> <span style="color:#38bdf8;">{config.niche}</span></div>
                    <div><strong>Target Timezone:</strong> <span style="color:#38bdf8;">{config.audience_timezone}</span></div>
                    <div><strong>Upload Frequency:</strong> <span style="color:#38bdf8;">Every {config.upload_frequency_hours} hours</span></div>
                </div>
            </div>

            <div class="card">
                <h2>API Key Rotation Pools & Quota Health</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Service</th>
                            <th>Active Key</th>
                            <th>Quota Remaining</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {key_rows}
                    </tbody>
                </table>
            </div>

            <div class="card">
                <h2>Serverless Endpoints</h2>
                <p>Trigger or monitor ClipMind via serverless API hooks:</p>
                <p>
                    <a class="btn" href="/api/status" target="_blank">View Status JSON</a>
                    <a class="btn" href="/health" target="_blank">Health Check</a>
                    <a class="btn" style="background:#059669;" href="/api/cron" target="_blank">Vercel Cron (/api/cron)</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# Vercel Serverless Function entry point
handler = app
