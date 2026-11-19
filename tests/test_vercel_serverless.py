"""Test Suite for ClipMind Vercel Serverless Integration.

Verifies:
1. vercel.json syntax, cron configuration (12-hour schedule '0 */12 * * *'), and function limits.
2. Serverless environment detection (VERCEL=1) and automatic routing of all writable paths to /tmp.
3. Serverless API endpoints using FastAPI TestClient:
   - api/index.py: /health, /api/status, /api/quotas, / (HTML dashboard)
   - api/status.py: /api/status JSON report
   - api/cron.py: /api/cron authentication and cycle execution
   - api/trigger.py: /api/trigger manual execution and secret authentication
4. Clean end-to-end cycle execution under simulated Vercel serverless environment.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clipmind.core.config import Config, PathSettings, get_config, is_serverless
from clipmind.core.key_rotator import KeyRotator, get_key_rotator
from clipmind.generation.clip_pipeline import ClipPipeline
from clipmind.creative.thumbnail_designer import ThumbnailDesigner
from clipmind.orchestrator import MasterOrchestrator

import api.index as api_index
import api.cron as api_cron
import api.status as api_status
import api.trigger as api_trigger


# ==============================================================================
# SECTION 1: vercel.json Configuration Verification
# ==============================================================================
def test_vercel_json_syntax_and_crons():
    """Verify vercel.json has valid structure, 12-hour cron schedule, and function limits."""
    vercel_path = PROJECT_ROOT / "vercel.json"
    assert vercel_path.exists(), "vercel.json must exist in project root"

    with open(vercel_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Cron configuration
    assert "crons" in data, "vercel.json must contain 'crons' key"
    crons = data["crons"]
    assert len(crons) >= 1, "At least one cron job must be defined"

    cron_entry = next((c for c in crons if c.get("path") == "/api/cron"), None)
    assert cron_entry is not None, "Cron job for /api/cron must be configured"
    assert cron_entry.get("schedule") == "0 */12 * * *", "Schedule must be 12-hour ('0 */12 * * *')"

    # 2. Function configuration
    assert "functions" in data, "vercel.json must configure 'functions'"
    func_config = data["functions"].get("api/**/*.py")
    assert func_config is not None, "Functions config for 'api/**/*.py' must exist"
    assert func_config.get("maxDuration") >= 60, "maxDuration must be configured for video processing"
    assert func_config.get("memory") >= 512, "memory must be configured"


# ==============================================================================
# SECTION 2: Serverless Environment Detection & /tmp Re-routing
# ==============================================================================
def test_serverless_detection_and_tmp_paths():
    """Verify that under VERCEL=1, paths are re-routed to tempfile.gettempdir()."""
    with patch.dict(os.environ, {"VERCEL": "1", "CLIPMIND_DRY_RUN": "true"}):
        assert is_serverless() is True

        cfg = Config.load_from_env()
        temp_dir = tempfile.gettempdir()

        # All writable directories must be rooted in /tmp
        assert temp_dir in str(cfg.paths.data_dir)
        assert temp_dir in str(cfg.paths.output_dir)
        assert temp_dir in str(cfg.paths.logs_dir)
        assert temp_dir in str(cfg.paths.temp_dir)
        assert temp_dir in str(cfg.paths.cycle_logs_dir)
        assert temp_dir in str(cfg.paths.quota_state_file)
        assert temp_dir in str(cfg.paths.archive_dir)


def test_pipeline_and_designer_use_tmp_in_serverless():
    """Verify ClipPipeline and ThumbnailDesigner write to /tmp under Vercel."""
    with patch.dict(os.environ, {"VERCEL": "1", "CLIPMIND_DRY_RUN": "true"}):
        cfg = get_config(reload=True)
        pipeline = ClipPipeline(config=cfg)
        designer = ThumbnailDesigner(config=cfg)

        temp_dir = tempfile.gettempdir()
        assert temp_dir in str(pipeline.output_dir)
        assert temp_dir in str(pipeline.temp_dir)
        assert temp_dir in str(designer.output_dir)


# ==============================================================================
# SECTION 3: Serverless API Endpoints Tests
# ==============================================================================
def test_api_index_endpoints():
    """Test /health, /api/status, /api/quotas, and dashboard HTML on api.index."""
    client = TestClient(api_index.app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    data_h = res_health.json()
    assert data_h["status"] == "healthy"
    assert data_h["service"] == "clipmind"
    assert data_h["serverless"] is True

    # 2. API Status
    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_s = res_status.json()
    assert data_s["service"] == "ClipMind"
    assert "key_pools" in data_s

    # 3. Quotas
    res_quotas = client.get("/api/quotas")
    assert res_quotas.status_code == 200
    assert "services" in res_quotas.json()

    # 4. Web Dashboard HTML
    res_html = client.get("/")
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers.get("content-type", "")
    assert "ClipMind" in res_html.text


def test_api_status_endpoint():
    """Test dedicated /api/status endpoint in api.status."""
    client = TestClient(api_status.app)
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["service"] == "ClipMind"
    assert data["platform"] == "vercel_serverless"
    assert "key_pools" in data


def test_api_cron_authorization_and_execution():
    """Test /api/cron secret authentication and execution."""
    client = TestClient(api_cron.app)

    # 1. Test unauthorized when CRON_SECRET is configured
    with patch.dict(os.environ, {"CRON_SECRET": "test_secure_cron_token_123"}):
        res_unauth = client.get("/api/cron")
        assert res_unauth.status_code == 401

        # Wrong token
        res_wrong = client.get("/api/cron", headers={"Authorization": "Bearer invalid_secret"})
        assert res_wrong.status_code == 401

        # Correct token via Header (dry run)
        with patch.dict(os.environ, {"CLIPMIND_DRY_RUN": "true"}):
            res_auth = client.get("/api/cron", headers={"Authorization": "Bearer test_secure_cron_token_123"})
            assert res_auth.status_code == 200
            data = res_auth.json()
            assert data["triggered_by"] == "vercel_cron"
            assert data["cycle_result"]["status"] == "COMPLETED"


def test_api_trigger_authorization_and_execution():
    """Test /api/trigger manual trigger and parameter overrides."""
    client = TestClient(api_trigger.app)

    # 1. Test unauthorized when TRIGGER_SECRET is set
    with patch.dict(os.environ, {"TRIGGER_SECRET": "test_trigger_token_456"}):
        res_unauth = client.get("/api/trigger")
        assert res_unauth.status_code == 401

        # 2. Test authorized execution via GET query param (dry run)
        with patch.dict(os.environ, {"CLIPMIND_DRY_RUN": "true"}):
            res_ok = client.get(
                "/api/trigger",
                params={"secret": "test_trigger_token_456", "niche": "AI Tools", "dry_run": "true"}
            )
            assert res_ok.status_code == 200
            data = res_ok.json()
            assert data["triggered_by"] == "manual_api"
            assert data["cycle_result"]["status"] == "COMPLETED"


# ==============================================================================
# SECTION 4: Full Serverless Cycle Execution
# ==============================================================================
def test_end_to_end_serverless_cycle():
    """Simulate a complete Vercel serverless execution cycle."""
    with patch.dict(os.environ, {"VERCEL": "1", "CLIPMIND_DRY_RUN": "true"}):
        cfg = get_config(reload=True)
        assert cfg.dry_run is True

        orchestrator = MasterOrchestrator(config=cfg)
        summary = orchestrator.run_cycle()

        assert summary["status"] == "COMPLETED"
        assert "cycle_id" in summary
        assert "video_id" in summary
        # Log file must exist inside temp directory
        assert Path(summary["log_file"]).exists()
        assert tempfile.gettempdir() in str(summary["log_file"])
