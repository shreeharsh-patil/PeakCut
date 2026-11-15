"""Real-time OS and Hardware Telemetry for OpenShorts.

Zero-dependency system monitor for Windows and Linux:
- CPU utilization via kernel32.GetSystemTimes on Windows or /proc/stat on Linux
- RAM metrics via kernel32.GlobalMemoryStatusEx or /proc/meminfo
- Disk space via shutil.disk_usage
- GPU telemetry via PyTorch or nvidia-smi
- Background smoothing sampler ensuring transient spikes do not cause false throttling
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Optional, Tuple

BUSY_CPU_PERCENT = float(os.environ.get("OPENSHORTS_BUSY_CPU", "85.0"))
MIN_FREE_RAM_GB = float(os.environ.get("OPENSHORTS_MIN_FREE_RAM_GB", "1.5"))
MIN_FREE_DISK_GB = float(os.environ.get("OPENSHORTS_MIN_FREE_DISK_GB", "3.0"))


class SystemTelemetry:
    """Collects real-time hardware metrics with a rolling smoothed window."""

    _instance: Optional[SystemTelemetry] = None

    def __init__(self, data_dir: str = "output"):
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.is_running = False
        self._cpu_usage = 0.0
        self._ram_free_gb = 99.0
        self._ram_total_gb = 99.0
        self._disk_free_gb = 99.0
        self._disk_total_gb = 99.0
        self._gpu_info = self._probe_gpu()
        self._last_disk_check = 0.0

    @classmethod
    def get_instance(cls) -> SystemTelemetry:
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.start()
        return cls._instance

    def start(self) -> None:
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            threading.Thread(target=self._sampler_loop, daemon=True, name="TelemetrySampler").start()

    def _probe_gpu(self) -> Dict[str, Any]:
        """Detect GPU availability and details."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                }
        except Exception:
            pass

        # Fallback probe via nvidia-smi
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0 and out.stdout.strip():
                parts = out.stdout.strip().split(",")
                return {
                    "available": True,
                    "name": parts[0].strip(),
                    "memory_raw": parts[1].strip() if len(parts) > 1 else "",
                }
        except Exception:
            pass

        return {"available": False, "name": "None (CPU Only)"}

    def _get_cpu_times(self) -> Tuple[int, int]:
        """Returns (busy_time, total_time)."""
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            idle = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            to_int = lambda f: (f.dwHighDateTime << 32) | f.dwLowDateTime
            total = to_int(kernel) + to_int(user)
            busy = total - to_int(idle)
            return busy, total
        else:
            with open("/proc/stat", encoding="ascii") as f:
                vals = [int(x) for x in f.readline().split()[1:9]]
            total = sum(vals)
            busy = total - vals[3] - vals[4]  # subtract idle & iowait
            return busy, total

    def _get_ram_info(self) -> Tuple[float, float]:
        """Returns (free_gb, total_gb)."""
        if os.name == "nt":
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullAvailPhys / 1e9, stat.ullTotalPhys / 1e9
        else:
            total, free = 99.0, 99.0
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) / 1e6
                    elif line.startswith("MemAvailable:"):
                        free = int(line.split()[1]) / 1e6
            return free, total

    def _sampler_loop(self) -> None:
        try:
            last_busy, last_total = self._get_cpu_times()
        except Exception:
            return

        while self.is_running:
            time.sleep(1.0)
            try:
                busy, total = self._get_cpu_times()
                delta_busy = busy - last_busy
                delta_total = total - last_total
                last_busy, last_total = busy, total

                if delta_total > 0:
                    instant_cpu = (delta_busy / delta_total) * 100.0
                    # Exponential smoothing: 0.6 * previous + 0.4 * current
                    self._cpu_usage = 0.6 * self._cpu_usage + 0.4 * instant_cpu

                free_ram, total_ram = self._get_ram_info()
                self._ram_free_gb = free_ram
                self._ram_total_gb = total_ram

                now = time.time()
                if now - self._last_disk_check > 15.0:
                    usage = shutil.disk_usage(self.data_dir)
                    self._disk_free_gb = usage.free / 1e9
                    self._disk_total_gb = usage.total / 1e9
                    self._last_disk_check = now
            except Exception:
                pass

    def get_telemetry(self) -> Dict[str, Any]:
        """Return snapshot of system metrics."""
        return {
            "cpu_percent": round(self._cpu_usage, 1),
            "ram_free_gb": round(self._ram_free_gb, 2),
            "ram_total_gb": round(self._ram_total_gb, 2),
            "disk_free_gb": round(self._disk_free_gb, 2),
            "disk_total_gb": round(self._disk_total_gb, 2),
            "gpu": self._gpu_info,
            "busy": self.is_busy(),
            "disk_full": self.is_disk_full(),
        }

    def is_busy(self) -> bool:
        return self._cpu_usage >= BUSY_CPU_PERCENT or self._ram_free_gb < MIN_FREE_RAM_GB

    def is_disk_full(self) -> bool:
        return self._disk_free_gb < MIN_FREE_DISK_GB
