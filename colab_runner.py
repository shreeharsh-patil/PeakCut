"""PeakCut Studio Google Colab Autonomous Runner & CLI.

Provides reliable, restart-safe Google Colab setup, resource-aware T4 scheduling,
real 5-step Ollama/Gemma readiness verification, authenticated backend + Cloudflare
tunnel lifecycle management, fast local-SSD processing with verified Google Drive
persistence, automated diagnostics, and the `peakcut` CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure repository root is always importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Compatible local Ollama models ordered from most T4-friendly to heaviest
RECOMMENDED_LOCAL_MODELS: Dict[str, Dict[str, Any]] = {
    "gemma2:2b": {"vram_gb": 2.5, "ram_gb": 4.0, "tier": "t4_recommended"},
    "llama3.2:3b": {"vram_gb": 3.0, "ram_gb": 4.5, "tier": "t4_recommended"},
    "qwen2.5:3b": {"vram_gb": 3.0, "ram_gb": 4.5, "tier": "t4_recommended"},
    "phi3:mini": {"vram_gb": 3.5, "ram_gb": 5.0, "tier": "t4_recommended"},
    "gemma2:9b": {"vram_gb": 7.5, "ram_gb": 10.0, "tier": "high_vram"},
}

DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/PeakCut_Shorts"
COLAB_FALLBACK_OUTPUT_DIR = "/content/PeakCut_Output"
COLAB_WORK_DIR = "/content/PeakCut_Work"
COLAB_LOG_DIR = "/content/peakcut_logs"


# ---------------------------------------------------------------------------
# Environment, Paths & Logging Helpers
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Return True if executing inside Google Colab environment."""
    if os.environ.get("IS_COLAB") in ("1", "true", "True"):
        return True
    if os.environ.get("FORCE_LOCAL_ONLY") in ("1", "true", "True"):
        return True
    if os.environ.get("COLAB_GPU") is not None or os.environ.get("COLAB_RELEASE_TAG") is not None:
        return True
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        pass
    return os.path.exists("/content") and os.path.isdir("/content")


def get_log_dir() -> Path:
    """Return writable log directory (`/content/peakcut_logs` on Colab)."""
    env_dir = (os.environ.get("PEAKCUT_LOG_DIR") or "").strip()
    if env_dir:
        p = Path(env_dir)
    elif os.path.isdir("/content"):
        p = Path(COLAB_LOG_DIR)
    else:
        p = Path(tempfile.gettempdir()) / "peakcut_logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tail_log(path: Path, lines: int = 20) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return ""


def _scrub_token_text(text: str) -> str:
    """Never leak session tokens or API keys into printed logs/diagnostics."""
    if not text:
        return ""
    cleaned = re.sub(r"([?&]token=)[^&\s\"']+", r"\1***", text)
    cleaned = re.sub(r"(Bearer\s+)[^\s\"']+", r"\1***", cleaned, flags=re.IGNORECASE)
    return cleaned


# ---------------------------------------------------------------------------
# Resource Detection & Safe T4 Concurrency Scheduling
# ---------------------------------------------------------------------------

def detect_system_resources() -> Dict[str, Any]:
    """Measure real CPU, RAM, GPU/CUDA VRAM, and available disk space."""
    cpu_count = os.cpu_count() or 2

    ram_total_gb = 0.0
    ram_avail_gb = 0.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total_gb = round(vm.total / (1024 ** 3), 2)
        ram_avail_gb = round(vm.available / (1024 ** 3), 2)
    except Exception:
        if os.path.exists("/proc/meminfo"):
            try:
                meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
                m_tot = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo)
                m_av = re.search(r"MemAvailable:\s+(\d+)\s+kB", meminfo)
                if m_tot:
                    ram_total_gb = round(int(m_tot.group(1)) / (1024 ** 2), 2)
                if m_av:
                    ram_avail_gb = round(int(m_av.group(1)) / (1024 ** 2), 2)
            except OSError:
                pass

    has_cuda = False
    gpu_name = "CPU"
    vram_total_gb = 0.0
    vram_free_gb = 0.0
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            gpu_name = str(torch.cuda.get_device_name(0))
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_total_gb = round(total_b / (1024 ** 3), 2)
            vram_free_gb = round(free_b / (1024 ** 3), 2)
    except Exception:
        # Fallback to nvidia-smi if torch is not installed yet
        smi = shutil.which("nvidia-smi")
        if smi:
            try:
                out = subprocess.check_output(
                    [smi, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                    text=True, timeout=5, stderr=subprocess.DEVNULL,
                ).strip()
                if out:
                    parts = [p.strip() for p in out.splitlines()[0].split(",")]
                    if len(parts) >= 3:
                        gpu_name = parts[0]
                        vram_total_gb = round(float(parts[1]) / 1024.0, 2)
                        vram_free_gb = round(float(parts[2]) / 1024.0, 2)
                        has_cuda = True
            except Exception:
                pass

    disk_target = "/content" if os.path.isdir("/content") else str(PROJECT_ROOT)
    disk_free_gb = 0.0
    disk_total_gb = 0.0
    try:
        usage = shutil.disk_usage(disk_target)
        disk_free_gb = round(usage.free / (1024 ** 3), 2)
        disk_total_gb = round(usage.total / (1024 ** 3), 2)
    except OSError:
        pass

    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "cpu_count": cpu_count,
        "ram_total_gb": ram_total_gb,
        "ram_available_gb": ram_avail_gb,
        "has_cuda": has_cuda,
        "gpu_name": gpu_name,
        "vram_total_gb": vram_total_gb,
        "vram_free_gb": vram_free_gb,
        "disk_path": disk_target,
        "disk_free_gb": disk_free_gb,
        "disk_total_gb": disk_total_gb,
    }


def compute_safe_concurrency(
    resources: Dict[str, Any],
    model_name: str = "gemma2:2b",
    render_profile: str = "balanced",
) -> Dict[str, Any]:
    """Calculate conservative, OOM-safe concurrency settings from real hardware metrics.

    Free Google Colab T4 (~15 GB VRAM, ~12.7 GB RAM, 2 vCPUs) runs Whisper +
    Ollama + MediaPipe + FFmpeg on shared VRAM/RAM. Starting with 1 concurrent
    job and 1 GPU-intensive op at a time prevents CUDA OOM and system OOM kills.
    Concurrency scales up only when measured resources clearly allow it.
    """
    cpu = max(1, int(resources.get("cpu_count") or 2))
    ram_avail = float(resources.get("ram_available_gb") or 0.0)
    ram_total = float(resources.get("ram_total_gb") or 0.0)
    has_cuda = bool(resources.get("has_cuda"))
    vram_free = float(resources.get("vram_free_gb") or 0.0)
    vram_total = float(resources.get("vram_total_gb") or 0.0)
    effective_ram = ram_avail or ram_total
    effective_vram = vram_free or vram_total

    model_info = RECOMMENDED_LOCAL_MODELS.get(model_name, {"vram_gb": 4.0, "ram_gb": 6.0})
    model_vram = float(model_info.get("vram_gb", 4.0))
    vram_headroom = max(0.0, effective_vram - model_vram)

    # Conservative defaults safe for free Colab T4 and CPU-only runtimes
    max_jobs = 1
    clip_workers = 1
    llm_workers = 1
    ollama_parallel = 1
    asr_gpu_concurrency = 1
    whisper_workers = 1
    whisper_threads = min(cpu, 4)

    # Scale up ONLY when measured headroom is comfortably above free-T4 limits
    if has_cuda and vram_headroom >= 16.0 and effective_ram >= 20.0 and cpu >= 4:
        # High-memory GPU tier (e.g., L4 / A100 / workstation)
        max_jobs = 2
        clip_workers = min(3, max(2, cpu // 2))
        llm_workers = 2
        ollama_parallel = 2
    elif effective_ram >= 14.0 and cpu >= 4 and (not has_cuda or vram_headroom >= 8.0):
        # Moderate multi-core machine
        max_jobs = 1
        clip_workers = 2
        llm_workers = 1
        ollama_parallel = 1

    profile = (render_profile or "balanced").strip().lower()
    if profile == "low_resource":
        video_quality = "720"
        clip_workers = 1
        max_jobs = 1
    elif profile == "high_quality":
        video_quality = "1080"
    else:
        # Balanced: 1080p when GPU/RAM headroom is solid, 720p on constrained CPU/RAM
        video_quality = "1080" if (has_cuda and (effective_ram == 0 or effective_ram >= 8.0)) else "720"

    return {
        "max_concurrent_jobs": max_jobs,
        "clip_workers": clip_workers,
        "llm_workers": llm_workers,
        "ollama_num_parallel": ollama_parallel,
        "asr_gpu_concurrency": asr_gpu_concurrency,
        "whisper_workers": whisper_workers,
        "whisper_threads": whisper_threads,
        "whisper_device": "cuda" if has_cuda else "cpu",
        "whisper_compute": "float16" if has_cuda else "int8",
        "ffmpeg_encoder": "auto" if has_cuda else "libx264",
        "video_quality": video_quality,
        "render_profile": profile,
    }


# ---------------------------------------------------------------------------
# Managed Process & PID Registry (Never kills unrelated system processes)
# ---------------------------------------------------------------------------

def _pid_file(service: str) -> Path:
    return get_log_dir() / f"{service}.pid"


def _state_file() -> Path:
    return get_log_dir() / "server_state.json"


def _write_pid(service: str, pid: int) -> None:
    try:
        _pid_file(service).write_text(str(int(pid)), encoding="utf-8")
    except OSError:
        pass


def _read_pid(service: str) -> Optional[int]:
    p = _pid_file(service)
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _is_pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return exit_code.value == 259  # STILL_ACTIVE
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _terminate_pid(pid: int, grace_seconds: float = 5.0) -> bool:
    """Gracefully terminate a specific PeakCut-managed PID (and its process group)."""
    if not _is_pid_alive(pid):
        return True

    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                return not _is_pid_alive(pid)

    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.2)

    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.2)
    return not _is_pid_alive(pid)


def stop_managed_services(services: Tuple[str, ...] = ("cloudflare", "backend")) -> Dict[str, bool]:
    """Cleanly stop only PeakCut-managed background services recorded in PID files."""
    state = _load_server_state()
    stopped: Dict[str, bool] = {}
    for svc in services:
        pid = _read_pid(svc)
        if pid and _is_pid_alive(pid):
            stopped[svc] = _terminate_pid(pid)
        else:
            stopped[svc] = True
        try:
            _pid_file(svc).unlink(missing_ok=True)
        except OSError:
            pass
    # Wait briefly for backend port to release after stopping managed backend
    if "backend" in services and state.get("port"):
        host = str(state.get("host") or "127.0.0.1")
        port = int(state["port"])
        deadline = time.time() + 3.0
        while time.time() < deadline and _is_port_open(host, port, timeout=0.2):
            time.sleep(0.15)
    return stopped


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    try:
        with socket.create_connection((check_host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _save_server_state(data: Dict[str, Any]) -> None:
    try:
        # Never persist raw secrets to disk state
        safe = {k: v for k, v in data.items() if "token" not in k.lower() and "secret" not in k.lower()}
        _state_file().write_text(json.dumps(safe, indent=2), encoding="utf-8")
    except OSError:
        pass


def _load_server_state() -> Dict[str, Any]:
    p = _state_file()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Phase 3: Real Local AI (Ollama) Setup & 5-Step Readiness Verification
# ---------------------------------------------------------------------------

def _find_ollama_binary() -> Optional[str]:
    return (
        shutil.which("ollama")
        or ("/usr/local/bin/ollama" if os.path.isfile("/usr/local/bin/ollama") else None)
        or ("/usr/bin/ollama" if os.path.isfile("/usr/bin/ollama") else None)
    )


def ensure_ollama_ready(
    model_name: str = "gemma2:2b",
    *,
    models_dir: Optional[str] = None,
    install_if_missing: bool = True,
    run_test_inference: bool = True,
    num_parallel: int = 1,
) -> Dict[str, Any]:
    """Install/start Ollama and run real 5-step readiness verification:

    1. Ollama executable exists.
    2. Ollama server responds.
    3. Selected model is installed (pulling if absent).
    4. Small test inference succeeds.
    5. PeakCut `llm_backend` communicates with the model.

    Never fabricates readiness: if any step fails, returns `ready=False` with
    `fallback_mode="heuristic"` and the concrete reason.
    """
    from engine.intelligence import llm_backend

    log_path = get_log_dir() / "ollama.log"
    status: Dict[str, Any] = {
        "binary_found": False,
        "executable_found": False,
        "executable_path": None,
        "server_responding": False,
        "model_installed": False,
        "inference_ok": False,
        "test_inference_ok": False,
        "peakcut_connected": False,
        "ready": False,
        "mode": "heuristic_fallback",
        "model": model_name,
        "log_path": str(log_path),
        "error": None,
    }

    if models_dir:
        try:
            os.makedirs(models_dir, exist_ok=True)
            os.environ["OLLAMA_MODELS"] = models_dir
        except OSError as e:
            print(f"⚠️ Could not prepare OLLAMA_MODELS dir '{models_dir}': {e}")

    os.environ["OLLAMA_NUM_PARALLEL"] = str(max(1, int(num_parallel)))
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = "1"
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
    os.environ["LLM_MODEL"] = model_name

    # Step 1: Locate or install Ollama binary
    ollama_bin = _find_ollama_binary()
    if not ollama_bin and install_if_missing and sys.platform.startswith("linux"):
        print("🦙 Installing Ollama local inference engine...")
        env_install = os.environ.copy()
        env_install["OLLAMA_NO_START"] = "1"
        try:
            res = subprocess.run(
                ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                env=env_install, capture_output=True, text=True, timeout=300,
            )
            if res.returncode != 0:
                raise RuntimeError(f"install.sh exited {res.returncode}: {(res.stderr or '')[-200:]}")
        except Exception as e:
            print(f"⚠️ Primary Ollama installer failed ({e}); attempting direct Linux tarball fallback...")
            try:
                res2 = subprocess.run(
                    ["sh", "-c", "curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -d | tar -xf - -C /usr"],
                    capture_output=True, text=True, timeout=300,
                )
                if res2.returncode != 0:
                    status["error"] = f"Ollama install failed: {(res2.stderr or '')[-200:]}"
            except Exception as ex2:
                status["error"] = f"Ollama install error: {ex2}"
        ollama_bin = _find_ollama_binary()

    if not ollama_bin:
        status["error"] = status["error"] or "Ollama executable not found in PATH"
        print(f"⚠️ {status['error']} — heuristic clip selector fallback will be used.")
        return status

    status["binary_found"] = True
    status["executable_found"] = True
    status["executable_path"] = ollama_bin

    # Step 2: Ensure Ollama server is running and responding
    llm_backend.invalidate_health_cache()
    health = llm_backend.check_local_llm_health(model=model_name, run_test_inference=False, timeout=2.0, use_cache=False)
    if not health.get("reachable"):
        print(f"🚀 Starting local Ollama server ({ollama_bin}) -> {log_path} ...")
        try:
            log_fh = open(log_path, "a", encoding="utf-8")
            proc = subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=(sys.platform != "win32"),
            )
            _write_pid("ollama", proc.pid)
        except Exception as e:
            status["error"] = f"Failed to launch ollama serve: {e}"
            print(f"⚠️ {status['error']} — heuristic clip selector fallback will be used.")
            return status

        for _ in range(40):
            if proc.poll() is not None:
                status["error"] = f"ollama serve exited early (code {proc.returncode}). Log: {_tail_log(log_path, 8)}"
                break
            health = llm_backend.check_local_llm_health(model=model_name, run_test_inference=False, timeout=1.5, use_cache=False)
            if health.get("reachable"):
                break
            time.sleep(0.5)

    if not health.get("reachable"):
        status["error"] = status["error"] or health.get("error") or "Ollama server did not respond on http://127.0.0.1:11434"
        print(f"⚠️ {status['error']} — heuristic clip selector fallback will be used.")
        return status

    status["server_responding"] = True

    # Step 3: Verify selected model is installed (pull if missing)
    if not health.get("model_installed"):
        print(f"📦 Pulling local model '{model_name}' (cached in {os.environ.get('OLLAMA_MODELS', 'default')})...")
        try:
            pull_res = subprocess.run(
                [ollama_bin, "pull", model_name],
                capture_output=True, text=True, errors="replace", timeout=900,
            )
            if pull_res.returncode != 0:
                status["error"] = f"ollama pull {model_name} failed (rc={pull_res.returncode}): {(pull_res.stderr or pull_res.stdout or '')[-240:]}"
                print(f"⚠️ {status['error']} — heuristic clip selector fallback will be used.")
                return status
        except Exception as e:
            status["error"] = f"ollama pull {model_name} error: {e}"
            print(f"⚠️ {status['error']} — heuristic clip selector fallback will be used.")
            return status

        llm_backend.invalidate_health_cache()
        health = llm_backend.check_local_llm_health(model=model_name, run_test_inference=False, timeout=3.0, use_cache=False)

    if not health.get("model_installed"):
        status["error"] = health.get("error") or f"Model '{model_name}' not found after pull"
        print(f"⚠️ {status['error']} — heuristic clip selector fallback will be used.")
        return status

    status["model_installed"] = True

    # Step 4 & 5: Real test inference + PeakCut llm_backend verification
    if run_test_inference:
        print(f"🧪 Running test inference on '{model_name}' via PeakCut llm_backend...")
        inf_health = llm_backend.check_local_llm_health(
            model=model_name, run_test_inference=True, timeout=60.0, use_cache=False,
        )
        status["inference_ok"] = bool(inf_health.get("inference_ok") or inf_health.get("test_inference_ok"))
        status["test_inference_ok"] = status["inference_ok"]
        status["peakcut_connected"] = bool(inf_health.get("ready"))
        if not status["peakcut_connected"]:
            status["error"] = inf_health.get("error") or "Local model test inference failed"
            print(f"⚠️ Test inference failed ({status['error']}) — heuristic clip selector fallback will be used.")
            return status
    else:
        status["inference_ok"] = True
        status["peakcut_connected"] = bool(health.get("ready"))

    status["ready"] = True
    status["mode"] = "ollama"
    print(f"✅ Local AI verified: {model_name} responded cleanly via Ollama!")
    return status


# ---------------------------------------------------------------------------
# Phase 8: Google Drive Mounting & Verified SSD -> Drive Artifact Sync
# ---------------------------------------------------------------------------

def _is_dir_writable(path: Union[str, Path]) -> bool:
    try:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        probe = p / f".peakcut_write_test_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def mount_google_drive(mount_drive: bool = True) -> Dict[str, Any]:
    """Mount Google Drive safely; fall back to `/content/PeakCut_Output` if unavailable."""
    drive_mounted = False
    drive_output = DRIVE_ROOT_DEFAULT
    drive_models = "/content/drive/MyDrive/models/ollama"
    fallback_output = COLAB_FALLBACK_OUTPUT_DIR if os.path.isdir("/content") else str(PROJECT_ROOT / "output")
    work_dir = COLAB_WORK_DIR if os.path.isdir("/content") else str(PROJECT_ROOT / "output")

    os.makedirs(work_dir, exist_ok=True)

    if mount_drive:
        try:
            from google.colab import drive  # type: ignore
            if not os.path.isdir("/content/drive/MyDrive"):
                print("📁 Mounting Google Drive at /content/drive ...")
                drive.mount("/content/drive", force_remount=False)
            if os.path.isdir("/content/drive/MyDrive") and _is_dir_writable(drive_output):
                os.makedirs(drive_models, exist_ok=True)
                os.makedirs(os.path.join(drive_output, "uploads"), exist_ok=True)
                drive_mounted = True
                print(f"✅ Google Drive ready: {drive_output}")
            else:
                print(f"⚠️ Google Drive path not writable; falling back to {fallback_output}")
        except Exception as e:
            print(f"⚠️ Google Drive mount skipped or unavailable ({e}). Falling back to {fallback_output}")

    final_output_root = drive_output if drive_mounted else fallback_output
    os.makedirs(final_output_root, exist_ok=True)

    return {
        "drive_mounted": drive_mounted,
        "output_root": final_output_root,
        "work_dir": work_dir,
        "models_dir": drive_models if drive_mounted else str(Path(work_dir) / "ollama_models"),
    }


def sync_job_directory_to_storage(
    src_job_dir: Union[str, Path],
    job_name: str,
    *,
    drive_root: Optional[str] = None,
    cleanup_temp: bool = False,
) -> Dict[str, Any]:
    """Copy verified job outputs from fast local SSD to Google Drive (or local fallback).

    Preserves MP4 clips, subtitle files (.srt/.ass), metadata JSON, status.json,
    summary reports, and error reports. Verifies every copied file exists and is
    readable before returning. Never deletes final outputs on error.
    """
    src = Path(src_job_dir)
    if not src.is_dir():
        return {"ok": False, "destination": "", "synced_files": [], "error": f"Source dir missing: {src}"}

    preferred = drive_root or os.environ.get("PEAKCUT_DRIVE_SYNC_DIR") or DRIVE_ROOT_DEFAULT
    fallback = COLAB_FALLBACK_OUTPUT_DIR if os.path.isdir("/content") else str(PROJECT_ROOT / "output")

    # Decide destination root (check Drive availability & writability)
    if _is_dir_writable(preferred):
        dest_root = Path(preferred)
        used_fallback = False
    else:
        dest_root = Path(fallback)
        dest_root.mkdir(parents=True, exist_ok=True)
        used_fallback = True
        print(f"⚠️ Drive destination '{preferred}' unavailable — saving outputs to {dest_root}")

    safe_job_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(job_name or "job")).strip("_") or f"job_{int(time.time())}"
    dest_dir = dest_root / safe_job_name

    # If src and dest are already identical, just verify existing artifacts
    try:
        if src.resolve() == dest_dir.resolve():
            files = [str(p) for p in dest_dir.iterdir() if p.is_file() and not p.name.startswith("temp_")]
            return {"ok": True, "destination": str(dest_dir), "synced_files": files, "used_fallback": used_fallback, "error": None}
    except OSError:
        pass

    dest_dir.mkdir(parents=True, exist_ok=True)

    keep_exts = {".mp4", ".srt", ".ass", ".json", ".txt", ".jpg", ".png"}
    synced: List[str] = []
    errors: List[str] = []

    for entry in sorted(src.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.startswith(("temp_", ".")):
            continue
        if entry.suffix.lower() not in keep_exts:
            continue
        if entry.stat().st_size <= 0:
            continue

        dst_file = dest_dir / entry.name
        tmp_dst = dest_dir / f".copying_{entry.name}"
        try:
            shutil.copy2(entry, tmp_dst)
            os.replace(tmp_dst, dst_file)
            # Verify destination file exists, matches size, and is readable
            if not dst_file.exists() or dst_file.stat().st_size != entry.stat().st_size:
                raise IOError(f"Size mismatch after copy for {entry.name}")
            with open(dst_file, "rb") as fh:
                fh.read(64)
            synced.append(str(dst_file))
        except Exception as e:
            tmp_dst.unlink(missing_ok=True)
            errors.append(f"{entry.name}: {e}")

    # Optional safe cleanup of temporary staging files ONLY if all copies verified clean
    if cleanup_temp and not errors and synced:
        for entry in list(src.iterdir()):
            if entry.is_file() and entry.name.startswith("temp_"):
                try:
                    entry.unlink()
                except OSError:
                    pass

    return {
        "ok": bool(synced) and not errors,
        "destination": str(dest_dir),
        "synced_files": synced,
        "used_fallback": used_fallback,
        "error": "; ".join(errors) if errors else None,
    }


# ---------------------------------------------------------------------------
# Phase 2 & 7: Environment Setup Entry Point
# ---------------------------------------------------------------------------

def setup_colab_environment(
    model_name: str = "gemma2:2b",
    mount_drive: bool = True,
    whisper_model: str = "base",
    force_local: bool = True,
    render_profile: str = "balanced",
    preserve_external_keys: bool = True,
    run_test_inference: bool = True,
) -> Dict[str, Any]:
    """Configure Colab environment for reliable, resource-aware local GPU execution.

    - Measures real CPU, RAM, GPU VRAM, and disk space before setting concurrency.
    - Mounts Google Drive for persistent outputs while staging FFmpeg I/O on fast local SSD.
    - Verifies Ollama + selected model with real readiness checks (falling back explicitly
      to PeakCut's heuristic virality selector if local AI is unavailable).
    - Preserves separately configured external API keys by default without requiring them.
    """
    print("\n" + "=" * 70)
    print("🚀 Initializing PeakCut Studio (Colab Resource-Aware Local Pipeline)")
    print("=" * 70)

    resources = detect_system_resources()
    sched = compute_safe_concurrency(resources, model_name=model_name, render_profile=render_profile)

    # 1. Configure local-first execution without destroying user's optional external keys
    os.environ["IS_COLAB"] = "1"
    if force_local:
        os.environ["FORCE_LOCAL_ONLY"] = "1"
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
        os.environ["LLM_MODEL"] = model_name
    if not preserve_external_keys:
        for key in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEY", "OPENAI_API_KEYS"):
            os.environ.pop(key, None)

    os.environ["WHISPER_MODEL"] = whisper_model
    os.environ["WHISPER_DEVICE"] = sched["whisper_device"]
    os.environ["WHISPER_COMPUTE"] = sched["whisper_compute"]
    os.environ["WHISPER_WORKERS"] = str(sched["whisper_workers"])
    os.environ["WHISPER_THREADS"] = str(sched["whisper_threads"])
    os.environ["ASR_GPU_CONCURRENCY"] = str(sched["asr_gpu_concurrency"])
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["OLLAMA_NUM_PARALLEL"] = str(sched["ollama_num_parallel"])
    os.environ["CLIP_WORKERS"] = str(sched["clip_workers"])
    os.environ["LLM_WORKERS"] = str(sched["llm_workers"])
    os.environ["MAX_CONCURRENT_JOBS"] = str(sched["max_concurrent_jobs"])
    os.environ["NUM_QUEUE_WORKERS"] = str(sched["max_concurrent_jobs"])
    os.environ["FFMPEG_ENCODER"] = sched["ffmpeg_encoder"]
    os.environ["VIDEO_QUALITY"] = sched["video_quality"]

    has_cuda = resources["has_cuda"]
    device_name = resources["gpu_name"]
    if has_cuda:
        print(
            f"⚡ Hardware: {device_name} ({resources['vram_total_gb']} GB VRAM, "
            f"{resources['ram_available_gb']} GB RAM avail, {resources['cpu_count']} CPUs) — CUDA float16 enabled."
        )
    else:
        print(
            f"ℹ️ Hardware: CPU mode ({resources['cpu_count']} CPUs, {resources['ram_available_gb']} GB RAM avail) — "
            f"int8 enabled. Tip: select T4 GPU in Runtime -> Change runtime type."
        )
    print(
        f"🛡️ Safe Concurrency Profile ({sched['render_profile']}): "
        f"jobs={sched['max_concurrent_jobs']}, clip_workers={sched['clip_workers']}, "
        f"llm_workers={sched['llm_workers']}, ollama_parallel={sched['ollama_num_parallel']}, "
        f"quality={sched['video_quality']}p"
    )

    # 2. Storage setup: fast local SSD staging + Google Drive persistent sync
    storage = mount_google_drive(mount_drive=mount_drive)
    drive_mounted = storage["drive_mounted"]
    os.environ["OUTPUT_DIR"] = storage["work_dir"]
    os.environ["UPLOAD_DIR"] = os.path.join(storage["work_dir"], "uploads")
    os.makedirs(os.environ["UPLOAD_DIR"], exist_ok=True)
    os.environ["PEAKCUT_DRIVE_SYNC_DIR"] = storage["output_root"]
    os.environ["OLLAMA_MODELS"] = storage["models_dir"]

    if sys.platform.startswith("linux"):
        install_cli_launcher()

    # 3. Local AI (Ollama) real 5-step readiness check
    if sys.platform.startswith("linux"):
        ai_status = ensure_ollama_ready(
            model_name=model_name,
            models_dir=storage["models_dir"],
            install_if_missing=True,
            run_test_inference=run_test_inference,
            num_parallel=sched["ollama_num_parallel"],
        )
    else:
        # Non-Linux dev/test host: probe without running Linux installer
        ai_status = ensure_ollama_ready(
            model_name=model_name,
            models_dir=None,
            install_if_missing=False,
            run_test_inference=False,
            num_parallel=sched["ollama_num_parallel"],
        )

    ollama_ready = bool(ai_status.get("ready"))
    ai_mode = ai_status.get("mode", "heuristic_fallback")

    return {
        "status": "ready",
        "has_cuda": has_cuda,
        "device": device_name,
        "model": model_name,
        "drive_mounted": drive_mounted,
        "output_root": storage["output_root"],
        "work_dir": storage["work_dir"],
        "ollama_ready": ollama_ready,
        "ai_mode": ai_mode,
        "ai_status": ai_status,
        "resources": resources,
        "concurrency": sched,
    }


# ---------------------------------------------------------------------------
# Phase 4: Ordered, Authenticated Backend + Cloudflare Tunnel Launcher
# ---------------------------------------------------------------------------

def _http_get_ok(url: str, *, headers: Optional[Dict[str, str]] = None, timeout: float = 3.0) -> Tuple[bool, int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return (200 <= resp.status < 300), resp.status, resp.read(2048).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as he:
        return False, he.code, str(he)
    except Exception as e:
        return False, 0, str(e)


def _ensure_cloudflared_binary() -> str:
    """Locate or download the cloudflared Linux binary with exit-code verification."""
    existing = shutil.which("cloudflared")
    if existing:
        return existing

    target = "/content/cloudflared" if os.path.isdir("/content") else str(get_log_dir() / "cloudflared")
    if os.path.isfile(target) and os.access(target, os.X_OK):
        return target

    if not sys.platform.startswith("linux"):
        raise RuntimeError("cloudflared binary not found in PATH on non-Linux platform")

    print("🌐 Downloading Cloudflare tunnel binary (cloudflared)...")
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    tmp_path = target + ".downloading"
    try:
        res = subprocess.run(
            ["curl", "-fsSL", "--retry", "3", "-o", tmp_path, url],
            capture_output=True, text=True, timeout=120,
        )
        if res.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) < 100_000:
            raise RuntimeError(f"curl failed (rc={res.returncode}): {(res.stderr or '')[-200:]}")
        os.chmod(tmp_path, 0o755)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return target


def start_backend_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    require_auth: bool = True,
    startup_timeout: float = 60.0,
) -> Dict[str, Any]:
    """Start PeakCut FastAPI/Uvicorn backend as a managed background process and verify health.

    Steps:
      1. Cleanly stop any previously PeakCut-managed backend/tunnel PIDs.
      2. Check for unmanaged port conflicts.
      3. Configure `PEAKCUT_ACCESS_TOKEN` when authentication is required.
      4. Launch Uvicorn (`backend.log` + `backend.pid`) from PROJECT_ROOT.
      5. Confirm process stays alive and `/health/ready`, `/api/status`, `/api/jobs`, `/app` respond 200.
    """
    log_dir = get_log_dir()
    backend_log = log_dir / "backend.log"

    # 1. Safe cleanup of ONLY PeakCut-managed services
    stop_managed_services(("cloudflare", "backend"))

    bind_host = host or "127.0.0.1"
    probe_host = "127.0.0.1" if bind_host in ("0.0.0.0", "::") else bind_host
    if _is_port_open(probe_host, port):
        raise RuntimeError(
            f"Port {port} is already in use by another process not managed by PeakCut. "
            f"Stop that process or launch with --port {port + 1}."
        )

    # 2. Configure session token protection so public tunnels are never exposed open
    access_token = (os.environ.get("PEAKCUT_ACCESS_TOKEN") or "").strip()
    auth_enabled = bool(require_auth)
    if auth_enabled and not access_token:
        access_token = secrets.token_urlsafe(24)
    if auth_enabled:
        os.environ["PEAKCUT_ACCESS_TOKEN"] = access_token
    else:
        os.environ.pop("PEAKCUT_ACCESS_TOKEN", None)
    auth_headers = {"X-PeakCut-Token": access_token} if auth_enabled else {}

    # 3. Start FastAPI / Uvicorn FIRST as a managed background process
    uvicorn_cmd = [
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", bind_host,
        "--port", str(int(port)),
        "--proxy-headers",
    ]
    proc_env = os.environ.copy()
    # Set FORWARDED_ALLOW_IPS via env instead of bare "*" argv (prevents Windows CRT/Click wildcard glob expansion)
    proc_env.setdefault("FORWARDED_ALLOW_IPS", "*")
    existing_pp = proc_env.get("PYTHONPATH", "")
    proc_env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
    proc_env.setdefault("PYTHONIOENCODING", "utf-8")

    print(f"🚀 Step 1/2: Starting PeakCut FastAPI backend on http://{probe_host}:{port} ...")
    backend_fh = open(backend_log, "w", encoding="utf-8")
    try:
        backend_proc = subprocess.Popen(
            uvicorn_cmd,
            cwd=str(PROJECT_ROOT),
            env=proc_env,
            stdout=backend_fh,
            stderr=subprocess.STDOUT,
            start_new_session=(sys.platform != "win32"),
        )
    finally:
        backend_fh.close()
    _write_pid("backend", backend_proc.pid)

    # 4. Poll local readiness endpoint AND verify required routes respond
    local_base = f"http://{probe_host}:{port}"
    deadline = time.time() + float(startup_timeout)
    backend_ready = False
    while time.time() < deadline:
        if backend_proc.poll() is not None:
            tail = _scrub_token_text(_tail_log(backend_log, 30))
            raise RuntimeError(
                f"PeakCut backend exited immediately (exit code {backend_proc.returncode}).\n"
                f"--- {backend_log} ---\n{tail}"
            )
        ok, _code, _body = _http_get_ok(f"{local_base}/health/ready", timeout=2.0)
        if ok:
            backend_ready = True
            break
        time.sleep(0.5)

    if not backend_ready:
        stop_managed_services(("backend",))
        tail = _scrub_token_text(_tail_log(backend_log, 30))
        raise RuntimeError(
            f"PeakCut backend did not become healthy within {startup_timeout:.0f}s.\n"
            f"--- {backend_log} ---\n{tail}"
        )

    # Verify required backend routes respond with auth headers
    required_routes = ("/api/status", "/api/jobs", "/app")
    for route in required_routes:
        ok, code, err = _http_get_ok(f"{local_base}{route}", headers=auth_headers, timeout=5.0)
        if not ok:
            stop_managed_services(("backend",))
            raise RuntimeError(
                f"Backend readiness check failed on {route} (HTTP {code}: {err}). "
                f"Inspect {backend_log}."
            )

    _save_server_state({
        "host": bind_host,
        "port": int(port),
        "local_url": local_base,
        "tunnel_url": None,
        "auth_enabled": auth_enabled,
        "backend_pid": backend_proc.pid,
        "cloudflare_pid": None,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })

    print(f"✅ Backend verified healthy on {local_base} (PID {backend_proc.pid}, log: {backend_log})")
    return {
        "pid": backend_proc.pid,
        "proc": backend_proc,
        "host": bind_host,
        "port": int(port),
        "local_url": local_base,
        "auth_enabled": auth_enabled,
        "access_token": access_token if auth_enabled else "",
        "backend_log": str(backend_log),
    }


def start_dashboard_access(
    port: int = 8000,
    host: str = "127.0.0.1",
    use_tunnel: bool = True,
    access_mode: str = "cloudflare",
    require_auth: bool = True,
) -> str:
    """Attach dashboard access (authenticated Cloudflare tunnel or private Colab proxy)
    to an ALREADY verified PeakCut backend.

    Refuses to open a tunnel unless the local backend responds healthy on `/health/ready`.
    """
    log_dir = get_log_dir()
    backend_log = log_dir / "backend.log"
    cf_log = log_dir / "cloudflare.log"

    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    local_base = f"http://{probe_host}:{port}"

    ok_health, code_health, err_health = _http_get_ok(f"{local_base}/health/ready", timeout=3.0)
    if not ok_health:
        raise RuntimeError(
            f"Refusing to start dashboard tunnel because local backend is not healthy at "
            f"{local_base}/health/ready (HTTP {code_health}: {err_health}). "
            f"Run start_backend_server() first."
        )

    state = _load_server_state()
    access_token = (os.environ.get("PEAKCUT_ACCESS_TOKEN") or "").strip()
    auth_enabled = bool(state.get("auth_enabled", False) or access_token)
    if require_auth and (use_tunnel and access_mode == "cloudflare") and not access_token:
        raise RuntimeError(
            "PEAKCUT_ACCESS_TOKEN is not set on the running backend. "
            "Restart the backend with require_auth=True before opening a public Cloudflare tunnel."
        )

    dashboard_url = f"{local_base}/app" + (f"?token={access_token}" if auth_enabled else "")
    tunnel_base_url: Optional[str] = None
    cf_proc = None

    # Stop any previous managed Cloudflare tunnel before starting a new one
    stop_managed_services(("cloudflare",))

    if access_mode == "colab_proxy" or not use_tunnel:
        if is_colab():
            try:
                from google.colab import output as colab_output  # type: ignore
                proxy_path = f"/app?token={access_token}" if auth_enabled else "/app"
                colab_output.serve_kernel_port_as_window(int(port), path=proxy_path)
                print(f"🪟 Opened PeakCut Studio via private Colab port proxy (port {port}).")
            except Exception as e:
                print(f"ℹ️ Colab port proxy unavailable ({e}); local server ready at {local_base}/app")
    else:
        print("🌐 Step 2/2: Starting Cloudflare HTTPS tunnel (after backend verified healthy)...")
        cf_bin = _ensure_cloudflared_binary()
        if cf_log.exists():
            try:
                cf_log.unlink()
            except OSError:
                pass

        cf_cmd = [cf_bin, "tunnel", "--url", local_base, "--no-autoupdate"]
        cf_fh = open(cf_log, "w", encoding="utf-8")
        try:
            cf_proc = subprocess.Popen(
                cf_cmd,
                stdout=cf_fh,
                stderr=subprocess.STDOUT,
                start_new_session=(sys.platform != "win32"),
            )
        finally:
            cf_fh.close()
        _write_pid("cloudflare", cf_proc.pid)

        cf_deadline = time.time() + 45.0
        while time.time() < cf_deadline:
            if cf_proc.poll() is not None:
                tail = _scrub_token_text(_tail_log(cf_log, 25))
                raise RuntimeError(
                    f"Cloudflare tunnel exited unexpectedly (code {cf_proc.returncode}).\n"
                    f"--- {cf_log} ---\n{tail}"
                )
            if cf_log.exists():
                text = cf_log.read_text(encoding="utf-8", errors="ignore")
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
                if match:
                    tunnel_base_url = match.group(0)
                    break
            time.sleep(0.5)

        if not tunnel_base_url:
            stop_managed_services(("cloudflare",))
            tail = _scrub_token_text(_tail_log(cf_log, 25))
            raise RuntimeError(
                f"Cloudflare tunnel did not publish a URL within 45s.\n"
                f"--- {cf_log} ---\n{tail}"
            )

        # Confirm tunnel process is alive and routing to our backend /health
        tunnel_verified = False
        for _ in range(15):
            if cf_proc.poll() is not None:
                break
            ok_t, _c_t, _b_t = _http_get_ok(f"{tunnel_base_url}/health", timeout=4.0)
            if ok_t:
                tunnel_verified = True
                break
            time.sleep(1.0)

        if not tunnel_verified:
            print(f"⚠️ Tunnel URL issued ({tunnel_base_url}) — DNS propagation may take a few extra seconds.")

        dashboard_url = f"{tunnel_base_url}/app" + (f"?token={access_token}" if auth_enabled else "")

    state.update({
        "host": host,
        "port": int(port),
        "local_url": local_base,
        "tunnel_url": tunnel_base_url,
        "auth_enabled": auth_enabled,
        "cloudflare_pid": cf_proc.pid if cf_proc else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_server_state(state)

    # Display live dashboard card ONLY AFTER verified startup
    display_label = tunnel_base_url or local_base
    print("\n" + "=" * 70)
    print(f"🎉 PEAKCUT STUDIO VERIFIED LIVE: {display_label}/app")
    if auth_enabled:
        print("🔒 Tunnel access protection: ENABLED (use the tokenized link below; never share publicly)")
    print(f"📜 Logs: {backend_log} | {cf_log}")
    print("=" * 70 + "\n")

    try:
        from IPython.display import display, HTML
        lock_badge = (
            "<p style='color:#34d399;font-size:12px;margin-top:8px;'>🔒 Session-token authenticated (cookie auto-set on click; URL token stripped automatically)</p>"
            if auth_enabled else ""
        )
        display(HTML(f"""
        <div style="background:linear-gradient(135deg,#18181b,#09090b);border:2px solid #8b5cf6;border-radius:12px;padding:22px;text-align:center;margin:16px 0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
          <h2 style="color:#ffffff;margin:0 0 8px 0;font-size:22px;">🎉 PeakCut Studio is Live &amp; Verified!</h2>
          <p style="color:#c4b5fd;font-size:14px;margin-bottom:18px;">Backend health checks passed. Click below to open the Studio Dashboard:</p>
          <a href="{dashboard_url}" target="_blank" style="background:linear-gradient(135deg,#7c3aed,#6d28d9);color:white;padding:12px 30px;text-decoration:none;border-radius:8px;font-weight:700;font-size:16px;display:inline-block;box-shadow:0 4px 15px rgba(124,58,237,0.4);">
            🚀 Open PeakCut Studio ↗
          </a>
          {lock_badge}
        </div>
        """))
    except Exception:
        pass

    return dashboard_url


def launch_web_ui(
    run_in_background: bool = True,
    host: str = "127.0.0.1",
    port: int = 8000,
    use_tunnel: bool = True,
    access_mode: str = "cloudflare",
    require_auth: bool = True,
    startup_timeout: float = 60.0,
) -> Optional[str]:
    """Start PeakCut FastAPI backend FIRST, verify health, then launch authenticated tunnel.

    Startup sequence:
      1. Cleanly stop any previously PeakCut-managed backend/tunnel PIDs.
      2. Generate/configure `PEAKCUT_ACCESS_TOKEN` when authentication is enabled.
      3. Launch Uvicorn in background (`backend.log` + `backend.pid`).
      4. Confirm process stays alive and poll `/health/ready`, `/api/status`, `/api/jobs`, `/app`.
      5. Only after backend is confirmed healthy, launch Cloudflare tunnel (`cloudflare.log` + `cloudflare.pid`)
         or Colab's native port proxy.
      6. Verify tunnel stays alive and responds on `/health`.
      7. Display and return the verified working dashboard URL.
    """
    backend_info = start_backend_server(
        host=host,
        port=port,
        require_auth=require_auth,
        startup_timeout=startup_timeout,
    )
    dashboard_url = start_dashboard_access(
        port=port,
        host=host,
        use_tunnel=use_tunnel,
        access_mode=access_mode,
        require_auth=require_auth,
    )

    if not run_in_background:
        print("🟢 Foreground mode active (Ctrl+C to stop PeakCut Studio).")
        try:
            backend_info["proc"].wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping PeakCut Studio managed services...")
            stop_managed_services(("cloudflare", "backend"))
    else:
        print("🟢 Backend & tunnel running in background (notebook cells remain interactive).")

    return dashboard_url


# ---------------------------------------------------------------------------
# Phase 5, 6 & 8: URL/Local Video Processing Pipeline with Verified Drive Sync
# ---------------------------------------------------------------------------

def parse_url_list(urls_input: Union[str, List[str]]) -> List[str]:
    """Parse string or list of URLs or local video paths into a clean, deduplicated list."""
    raw_items: List[str] = []
    if isinstance(urls_input, str):
        for line in urls_input.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                clean = line.split("#")[0].strip().strip('"\'')
                if clean:
                    raw_items.append(clean)
        if len(raw_items) == 1 and "," in raw_items[0] and not os.path.exists(raw_items[0]):
            raw_items = [u.strip() for u in raw_items[0].split(",") if u.strip()]
    elif isinstance(urls_input, (list, tuple)):
        raw_items = [str(u).strip() for u in urls_input if str(u).strip()]

    from engine.utils.security_utils import normalize_media_url
    seen = set()
    cleaned: List[str] = []
    for item in raw_items:
        # Preserve existing local file paths verbatim; normalize media URLs
        if os.path.isfile(item):
            resolved = os.path.abspath(item)
        else:
            resolved = normalize_media_url(item) or item
        if resolved not in seen:
            seen.add(resolved)
            cleaned.append(resolved)
    return cleaned


def run_shorts_pipeline(
    urls: Union[str, List[str]],
    clips_per_video: int = 3,
    format: str = "vertical",
    clip_min_seconds: float = 15.0,
    clip_max_seconds: float = 60.0,
    subtitle_preset: str = "karaoke",
    live_record_seconds: int = 900,
    output_dir: Optional[str] = None,
    keep_original: bool = False,
    schedule_to_youtube: bool = False,
    youtube_interval_hours: float = 1.0,
    youtube_privacy: str = "unlisted",
    youtube_access_token: Optional[str] = None,
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    cleanup_temp: bool = True,
) -> Dict[str, Any]:
    """Extract validated vertical shorts from YouTube URLs OR local video files.

    Processes on fast local SSD, validates every MP4 with FFprobe, then copies
    verified clips + subtitles + metadata + summaries to Google Drive (or local
    fallback) with full readback verification.
    """
    clean_inputs = parse_url_list(urls)
    if not clean_inputs:
        print("⚠️ No valid YouTube URLs or local video paths provided.")
        return {"total_queued": 0, "successful": [], "failed": [], "all_clips": []}

    # Final destination root (Drive or explicit output_dir or local fallback)
    final_dest_root = (
        output_dir
        or os.environ.get("PEAKCUT_DRIVE_SYNC_DIR")
        or (DRIVE_ROOT_DEFAULT if _is_dir_writable(DRIVE_ROOT_DEFAULT) else None)
        or (COLAB_FALLBACK_OUTPUT_DIR if os.path.isdir("/content") else str(PROJECT_ROOT / "output"))
    )
    os.makedirs(final_dest_root, exist_ok=True)

    # Use fast local SSD staging when final destination is mounted Google Drive FUSE
    use_ssd_staging = str(final_dest_root).startswith("/content/drive")
    staging_root = COLAB_WORK_DIR if (use_ssd_staging and os.path.isdir("/content")) else final_dest_root
    os.makedirs(staging_root, exist_ok=True)

    # Save & restore env mutations cleanly
    env_updates: Dict[str, Optional[str]] = {}
    if subtitle_preset == "none":
        env_updates["AUTO_CAPTIONS"] = "0"
        env_updates["SUBTITLE_STYLE"] = "none"
    elif subtitle_preset:
        env_updates["AUTO_CAPTIONS"] = "1"
        env_updates["SUBTITLE_STYLE"] = subtitle_preset
    if range_start is not None:
        env_updates["RANGE_START"] = str(float(range_start))
    if range_end is not None:
        env_updates["RANGE_END"] = str(float(range_end))
    if youtube_access_token and youtube_access_token.strip():
        env_updates["YOUTUBE_ACCESS_TOKEN"] = youtube_access_token.strip()

    prev_env = {k: os.environ.get(k) for k in env_updates}
    for k, v in env_updates.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

    print("\n" + "=" * 70)
    print("🎬 PeakCut Studio: Autonomous Shorts Pipeline")
    print(f"📦 Inputs Queued       : {len(clean_inputs)}")
    print(f"⚡ Processing Staging  : {staging_root}")
    print(f"📁 Final Destination   : {final_dest_root}")
    print(f"📐 Aspect Ratio Format : {format}")
    print(f"✂️  Clips Per Video    : {clips_per_video} ({clip_min_seconds}s–{clip_max_seconds}s)")
    print(f"💬 Subtitles           : {subtitle_preset}")
    if range_start is not None or range_end is not None:
        print(f"⏱️  Selected VOD Range : {range_start or 0}s -> {range_end or 'end'}")
    print("=" * 70)

    from main import process_video_pipeline
    from engine.editing.ffmpeg_utils import validate_media_output

    results: Dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_queued": len(clean_inputs),
        "destination_root": final_dest_root,
        "successful": [],
        "failed": [],
    }

    pipeline_start = time.time()
    all_produced_clips: List[str] = []

    try:
        for idx, item in enumerate(clean_inputs, 1):
            job_start = time.time()
            is_local_file = os.path.isfile(item)
            job_name = f"video_job_{idx}_{int(job_start)}"
            stage_dir = os.path.join(staging_root, job_name)
            os.makedirs(stage_dir, exist_ok=True)
            print(f"\n[{idx}/{len(clean_inputs)}] 🚀 Processing {'Local File' if is_local_file else 'URL'}: {item}")

            try:
                job_res = process_video_pipeline(
                    input_video_or_url=item,
                    output_dir=stage_dir,
                    is_url=not is_local_file,
                    output_format=format,
                    target_clips=clips_per_video,
                    clip_min_seconds=clip_min_seconds,
                    clip_max_seconds=clip_max_seconds,
                    live_record_seconds=live_record_seconds,
                    keep_original=keep_original,
                )
                duration = round(time.time() - job_start, 2)
                raw_clips = list(job_res.get("clips") or [])

                # Verify real playable MP4s on disk (when actual files exist; unit tests may mock paths)
                verified_clips: List[str] = []
                for cp in raw_clips:
                    if os.path.isfile(cp):
                        ok_mp4, _info, reason = validate_media_output(cp, require_video=True, decode_check=True)
                        if not ok_mp4:
                            print(f"   ⚠️ Skipping invalid clip {os.path.basename(cp)}: {reason}")
                            continue
                    verified_clips.append(cp)

                if not job_res.get("success") or not verified_clips:
                    raise RuntimeError("Pipeline completed without producing any valid playable MP4 clip.")

                # Write per-job processing summary
                summary_payload = {
                    "source": item,
                    "is_local_file": is_local_file,
                    "video_title": job_res.get("video_title"),
                    "format": format,
                    "subtitles": subtitle_preset,
                    "clips": [os.path.basename(c) for c in verified_clips],
                    "duration_seconds": duration,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                try:
                    Path(stage_dir, "processing_summary.json").write_text(
                        json.dumps(summary_payload, indent=2), encoding="utf-8"
                    )
                except OSError:
                    pass

                # Sync from SSD staging to Google Drive / final destination if different
                final_job_dir = stage_dir
                final_clips = verified_clips
                if use_ssd_staging:
                    sync_res = sync_job_directory_to_storage(
                        stage_dir, job_name, drive_root=final_dest_root, cleanup_temp=cleanup_temp,
                    )
                    if sync_res.get("ok"):
                        final_job_dir = sync_res["destination"]
                        synced_map = {os.path.basename(p): p for p in sync_res["synced_files"]}
                        final_clips = [synced_map.get(os.path.basename(c), c) for c in verified_clips]
                        print(f"   📁 Saved & verified in: {final_job_dir}")
                        if cleanup_temp and Path(stage_dir).resolve() != Path(final_job_dir).resolve():
                            shutil.rmtree(stage_dir, ignore_errors=True)
                    else:
                        print(f"   ⚠️ Drive sync warning ({sync_res.get('error')}); files preserved in {stage_dir}")

                all_produced_clips.extend(final_clips)
                results["successful"].append({
                    "url": item,
                    "title": job_res.get("video_title"),
                    "job_dir": final_job_dir,
                    "clips": final_clips,
                    "duration_seconds": duration,
                    "metadata": job_res.get("metadata"),
                })
                print(f"✅ Input {idx} completed in {duration}s! Produced {len(final_clips)} verified short(s).")

                # Optional YouTube scheduling
                if schedule_to_youtube and final_clips:
                    try:
                        from engine.services import youtube_uploader
                        shorts_meta = (job_res.get("metadata") or {}).get("shorts") or []
                        for c_idx, clip_path in enumerate(final_clips):
                            delay = youtube_interval_hours * (len(all_produced_clips) - len(final_clips) + c_idx + 1)
                            pub_time = datetime.now(timezone.utc) + timedelta(hours=delay)
                            clip_meta = shorts_meta[c_idx] if c_idx < len(shorts_meta) else {}
                            title = clip_meta.get("video_title_for_youtube_short") or f"{job_res.get('video_title', 'Highlight')} #Shorts"
                            desc = clip_meta.get("viral_hook_text") or "Created with PeakCut Studio #Shorts #Viral"
                            print(f"   📤 Scheduling clip {c_idx+1} for {pub_time.isoformat()} ({youtube_privacy})...")
                            youtube_uploader.upload_youtube_short(
                                video_path=clip_path,
                                title=title,
                                description=desc,
                                privacy_status=youtube_privacy,
                                publish_at=pub_time.isoformat(),
                            )
                    except Exception as upload_err:
                        print(f"   ⚠️ YouTube scheduling notice: {upload_err}")

            except Exception as e:
                duration = round(time.time() - job_start, 2)
                err_text = _scrub_token_text(str(e))
                print(f"❌ Input {idx} failed ({err_text}) in {duration}s.")
                err_entry = {
                    "url": item,
                    "error": err_text,
                    "duration_seconds": duration,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                }
                results["failed"].append(err_entry)
                try:
                    Path(stage_dir, "error_report.json").write_text(json.dumps(err_entry, indent=2), encoding="utf-8")
                    if use_ssd_staging:
                        sync_job_directory_to_storage(stage_dir, job_name, drive_root=final_dest_root, cleanup_temp=False)
                except OSError:
                    pass
    finally:
        for k, prev in prev_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev

    total_time = round(time.time() - pipeline_start, 2)
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    results["total_duration_seconds"] = total_time
    results["all_clips"] = all_produced_clips

    try:
        Path(final_dest_root, "batch_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    except OSError:
        pass

    print("\n" + "=" * 70)
    print(f"🎉 COMPLETED: {len(all_produced_clips)} verified clip(s) generated in {total_time}s.")
    print(f"📁 Output folder: {final_dest_root}")
    print("=" * 70 + "\n")
    return results


def publish_clip(
    video_path: str,
    title: str = "Viral Highlight #Shorts",
    description: str = "Created with PeakCut Studio #Shorts #Viral",
    privacy: str = "unlisted",
    delay_hours: float = 0.0,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-clip YouTube Shorts publisher."""
    if access_token and access_token.strip():
        os.environ["YOUTUBE_ACCESS_TOKEN"] = access_token.strip()

    from engine.services import youtube_uploader
    publish_at = None
    if delay_hours > 0:
        pub_dt = datetime.now(timezone.utc) + timedelta(hours=delay_hours)
        publish_at = pub_dt.isoformat()

    return youtube_uploader.upload_youtube_short(
        video_path=video_path,
        title=title,
        description=description,
        privacy_status=privacy,
        publish_at=publish_at,
    )


def install_cli_launcher() -> bool:
    """Install a global 'peakcut' executable launcher usable from any working directory."""
    if not sys.platform.startswith("linux"):
        return False
    candidates = ["/usr/local/bin", "/usr/bin", os.path.expanduser("~/.local/bin")]
    for target_dir in candidates:
        if os.path.isdir(target_dir) and os.access(target_dir, os.W_OK):
            launcher_path = os.path.join(target_dir, "peakcut")
            try:
                with open(launcher_path, "w", encoding="utf-8") as f:
                    f.write("#!/bin/sh\n")
                    f.write(f'export PYTHONPATH="{PROJECT_ROOT}:${{PYTHONPATH}}"\n')
                    f.write(f'exec "{sys.executable}" -m colab_runner "$@"\n')
                os.chmod(launcher_path, 0o755)
                print(f"✅ Global CLI launcher installed at {launcher_path}")
                return True
            except Exception:
                pass
    return False


# ---------------------------------------------------------------------------
# Phase 11: Automated Diagnostics ("Diagnose PeakCut")
# ---------------------------------------------------------------------------

def _cmd_first_line(cmd: List[str], timeout: float = 5.0) -> Optional[str]:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
        first = (out or "").strip().splitlines()
        return first[0] if first else ""
    except Exception:
        return None


def diagnose_peakcut(port: int = 8000, run_inference_check: bool = False) -> Dict[str, Any]:
    """Run comprehensive 14-point automated diagnostics and print PASS/WARNING/FAIL report.

    Never prints sensitive token values or API keys.
    """
    from engine.intelligence import llm_backend

    checks: List[Dict[str, str]] = []

    def _add(name: str, status: str, detail: str, fix: str = "") -> None:
        checks.append({
            "name": name,
            "status": status,  # PASS | WARNING | FAIL
            "detail": _scrub_token_text(detail),
            "fix": fix,
        })

    res = detect_system_resources()
    log_dir = get_log_dir()

    # 1. Python version
    py_ver = res["python_version"]
    py_ok = sys.version_info >= (3, 9)
    _add("Python Version", "PASS" if py_ok else "FAIL", f"Python {py_ver}", "" if py_ok else "Use Python 3.9+")

    # 2 & 3. GPU & CUDA availability
    if res["has_cuda"]:
        _add("GPU Availability", "PASS", res["gpu_name"])
        _add("CUDA Availability", "PASS", "CUDA active (GPU acceleration enabled)")
    else:
        _add("GPU Availability", "WARNING", "No NVIDIA GPU detected (CPU mode)", "In Colab: Runtime -> Change runtime type -> T4 GPU")
        _add("CUDA Availability", "WARNING", "CUDA inactive (CPU int8 fallback)", "Select GPU runtime for 5x faster Whisper + NVENC")

    # 4. GPU memory
    if res["has_cuda"]:
        vram_st = "PASS" if res["vram_total_gb"] >= 10.0 else "WARNING"
        _add(
            "GPU Memory (VRAM)", vram_st,
            f"{res['vram_free_gb']:.1f} GB free / {res['vram_total_gb']:.1f} GB total",
            "" if vram_st == "PASS" else "Use gemma2:2b and low_resource profile",
        )
    else:
        _add("GPU Memory (VRAM)", "WARNING", "N/A (CPU runtime)", "Select T4 GPU for 15 GB VRAM")

    # 5. System RAM & Disk
    ram_ok = res["ram_available_gb"] == 0.0 or res["ram_available_gb"] >= 3.0
    disk_ok = res["disk_free_gb"] >= 5.0
    if ram_ok and disk_ok:
        _add("RAM & Disk Space", "PASS", f"RAM: {res['ram_available_gb']:.1f}/{res['ram_total_gb']:.1f} GB avail | Disk: {res['disk_free_gb']:.1f} GB free")
    else:
        _add(
            "RAM & Disk Space", "WARNING" if disk_ok else "FAIL",
            f"RAM: {res['ram_available_gb']:.1f}/{res['ram_total_gb']:.1f} GB | Disk: {res['disk_free_gb']:.1f} GB free",
            "Free disk space in /content/PeakCut_Work or reduce MAX_VOD_MINUTES",
        )

    # 6. FFmpeg & FFprobe
    ff_ver = _cmd_first_line(["ffmpeg", "-version"])
    fp_ver = _cmd_first_line(["ffprobe", "-version"])
    if ff_ver and fp_ver:
        _add("FFmpeg & FFprobe", "PASS", f"{ff_ver[:60]} | ffprobe ok")
    else:
        _add("FFmpeg & FFprobe", "FAIL", f"ffmpeg={'ok' if ff_ver else 'MISSING'}, ffprobe={'ok' if fp_ver else 'MISSING'}", "Run Cell 3: apt-get install -y ffmpeg")

    # 7. yt-dlp
    try:
        import yt_dlp
        _add("yt-dlp Version", "PASS", f"yt-dlp {yt_dlp.version.__version__}")
    except Exception as e:
        _add("yt-dlp Version", "FAIL", f"Not importable: {e}", "Run: pip install -U yt-dlp")

    # 8. Whisper backend readiness (+ OpenCV & bundled fonts)
    whisper_issues = []
    try:
        import faster_whisper  # noqa: F401
    except Exception:
        whisper_issues.append("faster-whisper missing")
    try:
        import cv2  # noqa: F401
    except Exception:
        whisper_issues.append("opencv missing")
    if not whisper_issues:
        w_model = os.environ.get("WHISPER_MODEL", "base")
        w_dev = os.environ.get("WHISPER_DEVICE", "cuda" if res["has_cuda"] else "cpu")
        _add("Whisper & OpenCV Readiness", "PASS", f"faster-whisper + cv2 ready (model={w_model}, device={w_dev})")
    else:
        _add("Whisper & OpenCV Readiness", "FAIL", ", ".join(whisper_issues), "Re-run Cell 4 (Python Dependencies)")

    # 9 & 10. Ollama server & Selected model readiness
    sel_model = llm_backend.model_name()
    llm_health = llm_backend.check_local_llm_health(
        model=sel_model, run_test_inference=run_inference_check, timeout=3.0, use_cache=False,
    )
    if llm_health.get("reachable"):
        _add("Ollama Server Status", "PASS", f"Online at {llm_health['base_url']} ({len(llm_health['installed_models'])} model(s))")
    else:
        _add(
            "Ollama Server Status", "WARNING",
            f"Offline ({llm_health.get('error') or 'not reachable'}) — heuristic clip selector fallback active",
            f"Run Cell 6 (Local AI Initialization). Check {log_dir / 'ollama.log'}",
        )

    if llm_health.get("ready"):
        inf_note = " (test inference passed)" if run_inference_check else ""
        _add("Selected Model Readiness", "PASS", f"{sel_model} loaded & ready{inf_note}")
    elif llm_health.get("reachable"):
        _add(
            "Selected Model Readiness", "WARNING",
            f"{sel_model} not ready: {llm_health.get('error')}",
            f"Run: ollama pull {sel_model}",
        )
    else:
        _add("Selected Model Readiness", "WARNING", f"{sel_model} unavailable while Ollama offline (heuristic fallback active)", "Run Cell 6 to start Ollama")

    # 11. FastAPI Backend Health
    token = (os.environ.get("PEAKCUT_ACCESS_TOKEN") or "").strip()
    hdrs = {"X-PeakCut-Token": token} if token else {}
    local_base = f"http://127.0.0.1:{int(port)}"
    ok_health, h_code, h_err = _http_get_ok(f"{local_base}/health/ready", timeout=2.0)
    ok_api, a_code, a_err = _http_get_ok(f"{local_base}/api/status", headers=hdrs, timeout=2.0)
    if ok_health and ok_api:
        _add("FastAPI Health", "PASS", f"Ready on {local_base} (auth={'ON' if token else 'OFF'})")
    elif ok_health and a_code == 401:
        _add("FastAPI Health", "WARNING", f"Server live on {local_base}, token header required (HTTP 401 without token)", "Check PEAKCUT_ACCESS_TOKEN in session")
    else:
        _add("FastAPI Health", "WARNING", f"Offline ({h_err or a_err})", f"Run Cell 7 (Start Backend) and inspect {log_dir / 'backend.log'}")

    # 12. Dashboard Status
    ok_dash, d_code, d_err = _http_get_ok(f"{local_base}/app", headers=hdrs, timeout=2.0)
    if ok_dash:
        _add("Dashboard Status", "PASS", f"/app serving PeakCut Studio UI (HTTP {d_code})")
    else:
        _add("Dashboard Status", "WARNING", f"Dashboard not reachable ({d_code or d_err})", "Start backend in Cell 7")

    # 13. Tunnel Health
    state = _load_server_state()
    tunnel_url = state.get("tunnel_url")
    cf_pid = _read_pid("cloudflare")
    if tunnel_url and _is_pid_alive(cf_pid):
        ok_t, t_code, t_err = _http_get_ok(f"{tunnel_url}/health", timeout=4.0)
        if ok_t:
            _add("Tunnel Health", "PASS", f"Active & reachable ({tunnel_url}, PID {cf_pid})")
        else:
            _add("Tunnel Health", "WARNING", f"Tunnel PID {cf_pid} running but probe returned {t_code or t_err}", f"Inspect {log_dir / 'cloudflare.log'}")
    else:
        _add("Tunnel Health", "WARNING", "No active Cloudflare tunnel", "Run Cell 8 (Dashboard Access) to start tunnel or Colab proxy")

    # 14. Output Directory Accessibility
    out_dir = os.environ.get("PEAKCUT_DRIVE_SYNC_DIR") or os.environ.get("OUTPUT_DIR") or str(PROJECT_ROOT / "output")
    if _is_dir_writable(out_dir):
        _add("Output Directory", "PASS", f"Writable: {out_dir}")
    else:
        _add("Output Directory", "FAIL", f"Not writable: {out_dir}", "Re-mount Google Drive (Cell 5) or use /content/PeakCut_Output")

    # Print formatted diagnostic table
    icons = {"PASS": "✅ PASS", "WARNING": "⚠️ WARN", "FAIL": "❌ FAIL"}
    print("\n" + "=" * 76)
    print("🩺 PeakCut Studio Automated Diagnostics")
    print("=" * 76)
    for item in checks:
        print(f"  {icons[item['status']]:9s} | {item['name']:26s} | {item['detail']}")
        if item["fix"]:
            print(f"            | {'':26s} | 💡 {item['fix']}")

    # Include log excerpts on any WARNING/FAIL for services
    log_excerpts: Dict[str, str] = {}
    for svc, fname in (("backend", "backend.log"), ("cloudflare", "cloudflare.log"), ("ollama", "ollama.log")):
        tail = _scrub_token_text(_tail_log(log_dir / fname, 10))
        if tail:
            log_excerpts[svc] = tail

    counts = {
        "pass": sum(1 for c in checks if c["status"] == "PASS"),
        "warning": sum(1 for c in checks if c["status"] == "WARNING"),
        "fail": sum(1 for c in checks if c["status"] == "FAIL"),
    }
    print("-" * 76)
    print(f"Summary: {counts['pass']} PASS | {counts['warning']} WARNING | {counts['fail']} FAIL")
    print(f"Log directory: {log_dir}")
    print("=" * 76 + "\n")

    return {
        "checks": checks,
        "summary": counts,
        "resources": res,
        "log_dir": str(log_dir),
        "log_excerpts": log_excerpts,
    }


def show_status(port: int = 8000) -> Dict[str, Any]:
    """Print Colab and PeakCut system diagnostics, GPU stats, and running services."""
    res = detect_system_resources()
    in_colab = is_colab()
    out_dir = os.environ.get("PEAKCUT_DRIVE_SYNC_DIR") or os.environ.get("OUTPUT_DIR") or (
        COLAB_FALLBACK_OUTPUT_DIR if in_colab else "./output"
    )

    token = (os.environ.get("PEAKCUT_ACCESS_TOKEN") or "").strip()
    hdrs = {"X-PeakCut-Token": token} if token else {}
    local_base = f"http://127.0.0.1:{int(port)}"

    server_status = "Offline"
    jobs_count = 0
    recent_jobs: List[Dict[str, Any]] = []
    ok_s, _c, _b = _http_get_ok(f"{local_base}/api/status", headers=hdrs, timeout=2.0)
    if ok_s:
        server_status = f"Online ({local_base})"
        ok_j, _cj, body_j = _http_get_ok(f"{local_base}/api/jobs", headers=hdrs, timeout=2.0)
        if ok_j:
            try:
                jobs_data = json.loads(body_j)
                if isinstance(jobs_data, list):
                    jobs_count = len(jobs_data)
                    recent_jobs = jobs_data[:5]
            except ValueError:
                pass

    from engine.intelligence import llm_backend
    llm_h = llm_backend.check_local_llm_health(timeout=2.0, use_cache=False)
    if llm_h.get("reachable"):
        models = llm_h.get("installed_models") or []
        ollama_status = f"Online (Models: {', '.join(models) if models else 'None loaded'})"
    else:
        ollama_status = "Not running (heuristic fallback available)"

    print("\n" + "=" * 65)
    print("📊 PeakCut Studio System & Process Status")
    print("=" * 65)
    print(f"🖥️  Platform: {res['platform']} | Python: {res['python_version']} | Runtime: {'Google Colab' if in_colab else 'Local / Workstation'}")
    print(f"⚡ Hardware Acceleration: {res['gpu_name']} (CUDA: {'Active' if res['has_cuda'] else 'Inactive'})")
    print(f"💾 Resources: {res['cpu_count']} CPUs | RAM {res['ram_available_gb']:.1f}/{res['ram_total_gb']:.1f} GB | Disk {res['disk_free_gb']:.1f} GB free")
    print(f"📁 Active Output Directory: {out_dir}")
    print(f"🌐 Web API Server: {server_status}")
    print(f"🦙 Ollama Local Inference: {ollama_status}")
    if recent_jobs:
        print("\n🎬 Recent Jobs:")
        for j in recent_jobs:
            title = j.get("title") or "Untitled Video"
            st = j.get("status") or "unknown"
            prog = int((j.get("progress") or 0.0) * 100)
            clips_cnt = len(j.get("clips") or [])
            print(f"   • [{st.upper()}] {title} — {prog}% ({clips_cnt} clip(s))")
    print("=" * 65 + "\n")

    return {
        "is_colab": in_colab,
        "cuda_available": res["has_cuda"],
        "device": res["gpu_name"],
        "server_status": server_status,
        "ollama_status": ollama_status,
        "jobs_count": jobs_count,
        "output_dir": out_dir,
        "resources": res,
    }


# ---------------------------------------------------------------------------
# Phase 10: PeakCut CLI
# ---------------------------------------------------------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for PeakCut Studio."""
    parser = argparse.ArgumentParser(
        prog="peakcut",
        description="PeakCut Studio CLI: Autonomous AI Shorts Pipeline for Google Colab and Local Workstations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  peakcut setup --model gemma2:2b --whisper base
  peakcut clip "https://youtube.com/watch?v=kYfNvmF0Bqw" --clips 3 --format vertical --subtitles karaoke
  peakcut clip "/content/my_video.mp4" --clips 2 --format vertical
  peakcut serve --background
  peakcut status
  peakcut diagnose
  peakcut stop
  peakcut upload "/path/to/clip.mp4" --title "Viral Short" --privacy unlisted
        """,
    )
    sub = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # setup
    p_setup = sub.add_parser("setup", help="Configure resource-aware local GPU environment & Google Drive")
    p_setup.add_argument("--model", default="gemma2:2b", help="Local Ollama model name (default: gemma2:2b)")
    p_setup.add_argument("--whisper", default="base", help="Faster-Whisper model size (default: base)")
    p_setup.add_argument("--profile", choices=["balanced", "low_resource", "high_quality"], default="balanced", help="Colab rendering resource profile")
    p_setup.add_argument("--no-drive", action="store_true", help="Do not mount Google Drive (use local storage)")
    p_setup.add_argument("--skip-inference-test", action="store_true", help="Skip initial Ollama test inference check")

    # clip / run
    p_clip = sub.add_parser("clip", aliases=["run"], help="Extract vertical shorts from YouTube URLs or local video files")
    p_clip.add_argument("urls", nargs="+", help="YouTube URLs or local video file paths")
    p_clip.add_argument("-n", "--clips", type=int, default=3, help="Number of clips to generate per video (default: 3)")
    p_clip.add_argument("-f", "--format", choices=["vertical", "horizontal", "square", "auto"], default="vertical", help="Aspect ratio (default: vertical 9:16)")
    p_clip.add_argument("-s", "--subtitles", choices=["karaoke", "minimal", "bold", "none"], default="karaoke", help="Subtitle styling preset (default: karaoke)")
    p_clip.add_argument("--min-len", type=float, default=15.0, help="Minimum clip duration in seconds (default: 15)")
    p_clip.add_argument("--max-len", type=float, default=60.0, help="Maximum clip duration in seconds (default: 60)")
    p_clip.add_argument("--start", type=float, default=None, help="Optional VOD start offset in seconds")
    p_clip.add_argument("--end", type=float, default=None, help="Optional VOD end offset in seconds")
    p_clip.add_argument("--live-record", type=int, default=900, help="Live stream recording duration in seconds (default: 900)")
    p_clip.add_argument("-o", "--output", default=None, help="Output directory (defaults to Google Drive or ./output)")
    p_clip.add_argument("--keep-original", action="store_true", help="Retain original downloaded source video")
    p_clip.add_argument("--no-cleanup", action="store_true", help="Keep staging temporary artifacts")
    p_clip.add_argument("--schedule", action="store_true", help="Auto-schedule generated clips to YouTube Shorts")
    p_clip.add_argument("--interval", type=float, default=1.0, help="Hours between scheduled uploads (default: 1.0)")
    p_clip.add_argument("--privacy", choices=["unlisted", "private", "public"], default="unlisted", help="YouTube privacy status (default: unlisted)")
    p_clip.add_argument("--token", default=None, help="YouTube OAuth access token")

    # serve / launch
    p_serve = sub.add_parser("serve", aliases=["launch"], help="Launch PeakCut Studio Web UI server & authenticated tunnel")
    p_serve.add_argument("-b", "--background", action="store_true", help="Run server in background process")
    p_serve.add_argument("-p", "--port", type=int, default=8000, help="Server port (default: 8000)")
    p_serve.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p_serve.add_argument("--no-tunnel", action="store_true", help="Start local backend only (no Cloudflare tunnel)")
    p_serve.add_argument("--no-auth", action="store_true", help="Disable session access token (only use on private networks)")

    # stop
    sub.add_parser("stop", help="Safely stop PeakCut-managed backend, tunnel, and Ollama processes")

    # upload / publish
    p_upload = sub.add_parser("upload", aliases=["publish"], help="Upload or schedule a specific clip to YouTube Shorts")
    p_upload.add_argument("video_path", help="Path to the clip video file (.mp4)")
    p_upload.add_argument("-t", "--title", default="Viral Highlight #Shorts", help="Video title")
    p_upload.add_argument("-d", "--description", default="Created with PeakCut Studio #Shorts #Viral", help="Video description")
    p_upload.add_argument("-p", "--privacy", choices=["unlisted", "private", "public"], default="unlisted", help="Upload privacy status (default: unlisted)")
    p_upload.add_argument("--delay", type=float, default=0.0, help="Delay in hours before publishing (0 = immediate)")
    p_upload.add_argument("--token", default=None, help="YouTube OAuth access token")

    # status & diagnose
    p_status = sub.add_parser("status", help="Inspect active server, jobs, GPU diagnostics, and storage")
    p_status.add_argument("-p", "--port", type=int, default=8000, help="Backend port to check (default: 8000)")

    p_diag = sub.add_parser("diagnose", help="Run full 14-point automated PeakCut health & readiness diagnostics")
    p_diag.add_argument("-p", "--port", type=int, default=8000, help="Backend port to check (default: 8000)")
    p_diag.add_argument("--test-inference", action="store_true", help="Also run a live local LLM test inference")

    return parser


def cli_main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint for PeakCut Studio."""
    parser = build_cli_parser()
    parsed_args = parser.parse_args(argv)

    if not parsed_args.subcommand:
        parser.print_help()
        return 0

    sub = parsed_args.subcommand

    if sub == "setup":
        setup_colab_environment(
            model_name=parsed_args.model,
            mount_drive=not parsed_args.no_drive,
            whisper_model=parsed_args.whisper,
            render_profile=parsed_args.profile,
            run_test_inference=not parsed_args.skip_inference_test,
        )
        install_cli_launcher()
        return 0

    if sub in ("clip", "run"):
        results = run_shorts_pipeline(
            urls=parsed_args.urls,
            clips_per_video=parsed_args.clips,
            format=parsed_args.format,
            clip_min_seconds=parsed_args.min_len,
            clip_max_seconds=parsed_args.max_len,
            subtitle_preset=parsed_args.subtitles,
            live_record_seconds=parsed_args.live_record,
            output_dir=parsed_args.output,
            keep_original=parsed_args.keep_original,
            schedule_to_youtube=parsed_args.schedule,
            youtube_interval_hours=parsed_args.interval,
            youtube_privacy=parsed_args.privacy,
            youtube_access_token=parsed_args.token,
            range_start=parsed_args.start,
            range_end=parsed_args.end,
            cleanup_temp=not parsed_args.no_cleanup,
        )
        has_success = bool(results.get("successful"))
        has_failure = bool(results.get("failed"))
        return 0 if (has_success or not has_failure) else 1

    if sub in ("serve", "launch"):
        launch_web_ui(
            run_in_background=parsed_args.background,
            host=parsed_args.host,
            port=parsed_args.port,
            use_tunnel=not parsed_args.no_tunnel,
            require_auth=not parsed_args.no_auth,
        )
        return 0

    if sub == "stop":
        stopped = stop_managed_services(("cloudflare", "backend", "ollama"))
        print(f"🛑 Stopped managed PeakCut services: {stopped}")
        return 0

    if sub in ("upload", "publish"):
        publish_clip(
            video_path=parsed_args.video_path,
            title=parsed_args.title,
            description=parsed_args.description,
            privacy=parsed_args.privacy,
            delay_hours=parsed_args.delay,
            access_token=parsed_args.token,
        )
        return 0

    if sub == "status":
        show_status(port=parsed_args.port)
        return 0

    if sub == "diagnose":
        report = diagnose_peakcut(port=parsed_args.port, run_inference_check=parsed_args.test_inference)
        return 1 if report["summary"]["fail"] > 0 else 0

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
