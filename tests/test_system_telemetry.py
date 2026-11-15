"""Tests for Hardware & System Telemetry (system_telemetry.py)."""
from system_telemetry import SystemTelemetry


def test_system_telemetry_get():
    telemetry = SystemTelemetry.get_instance()
    data = telemetry.get_telemetry()

    assert "cpu_percent" in data
    assert isinstance(data["cpu_percent"], float)
    assert "ram_free_gb" in data
    assert data["ram_free_gb"] > 0
    assert "disk_free_gb" in data
    assert data["disk_free_gb"] > 0
    assert "gpu" in data
    assert "busy" in data
    assert "disk_full" in data
