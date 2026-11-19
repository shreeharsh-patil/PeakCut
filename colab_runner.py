"""PeakCut Studio Google Colab Autonomous Runner.

Enforces 100% Zero-API local execution using local Gemma on NVIDIA GPU with
persistent Google Drive weight caching and pure-Python notebook forms (NO CLI).
"""

from __future__ import annotations

import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure root workspace is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def setup_colab_environment(
    model_name: str = "gemma2:2b",
    mount_drive: bool = True,
    whisper_model: str = "base",
    force_local: bool = True,
) -> Dict[str, Any]:
    """Configure Colab environment for 100% local GPU execution with ZERO external APIs.

    - Mounts Google Drive for persistent model storage and output shorts.
    - Caches Gemma model weights in Google Drive (downloads once, persists forever).
    - Configures NVIDIA GPU acceleration for Whisper and Gemma.
    - Purges external API keys to guarantee zero network API usage / rate limits.
    """
    print("\n" + "=" * 70)
    print("🚀 Initializing PeakCut Studio: 100% Local GPU Pipeline")
    print("=" * 70)

    # 1. Enforce Zero External APIs
    os.environ["IS_COLAB"] = "1"
    os.environ["FORCE_LOCAL_ONLY"] = "1"
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_BASE_URL"] = "http://127.0.0.1:11434/v1"
    os.environ["LLM_MODEL"] = model_name
    os.environ["WHISPER_MODEL"] = whisper_model
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["OLLAMA_NUM_PARALLEL"] = "4"
    os.environ["CLIP_WORKERS"] = str(max(4, min(os.cpu_count() or 4, 8)))
    os.environ["LLM_WORKERS"] = "3"
    os.environ["MAX_CONCURRENT_JOBS"] = "4"
    os.environ["NUM_QUEUE_WORKERS"] = "4"
    os.environ["FFMPEG_ENCODER"] = "auto"
    os.environ["WHISPER_WORKERS"] = "2"
    os.environ["WHISPER_THREADS"] = "4"

    for key in ["GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEY", "OPENAI_API_KEYS"]:
        os.environ.pop(key, None)

    # 2. Hardware acceleration detection
    has_cuda = False
    device_name = "CPU"
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            device_name = torch.cuda.get_device_name(0)
    except Exception:
        pass

    if has_cuda:
        os.environ["WHISPER_DEVICE"] = "cuda"
        os.environ["WHISPER_COMPUTE"] = "float16"
        print(f"⚡ Hardware: NVIDIA GPU detected ({device_name}) — CUDA float16 enabled.")
    else:
        os.environ["WHISPER_DEVICE"] = "cpu"
        os.environ["WHISPER_COMPUTE"] = "int8"
        print("ℹ️ Hardware: Running on CPU mode (int8). Tip: switch runtime to T4 GPU for 5x speed.")

    # 3. Mount Google Drive for persistent storage
    drive_mounted = False
    drive_model_dir = "/content/drive/MyDrive/models/ollama"
    drive_output_dir = "/content/drive/MyDrive/PeakCut_Shorts"

    if mount_drive:
        try:
            from google.colab import drive
            print("📁 Connecting Google Drive for persistent storage...")
            drive.mount("/content/drive")
            drive_upload_dir = "/content/drive/MyDrive/PeakCut_Shorts/uploads"
            os.makedirs(drive_model_dir, exist_ok=True)
            os.makedirs(drive_output_dir, exist_ok=True)
            os.makedirs(drive_upload_dir, exist_ok=True)
            os.environ["OLLAMA_MODELS"] = drive_model_dir
            os.environ["OUTPUT_DIR"] = drive_output_dir
            os.environ["UPLOAD_DIR"] = drive_upload_dir
            drive_mounted = True
            print(f"✅ Google Drive connected! Models: {drive_model_dir} | Output: {drive_output_dir}")
        except Exception as e:
            print(f"⚠️ Google Drive mount skipped or unavailable ({e}). Using local storage.")
            os.environ["OUTPUT_DIR"] = "/content/output"
    else:
        os.environ["OUTPUT_DIR"] = "/content/output"

    # 4. System dependencies (Linux / Colab)
    if sys.platform.startswith("linux"):
        print("📦 Verifying system packages (ffmpeg, fonts, zstd)...")
        try:
            subprocess.run(
                "apt-get update -qq && apt-get install -y -qq ffmpeg libsm6 libxext6 libgl1 fonts-freefont-ttf fonts-dejavu-core zstd curl",
                shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except Exception:
            pass
        install_cli_launcher()

    # 5. Local Ollama Engine Setup on GPU
    ollama_ready = False
    if sys.platform.startswith("linux"):
        ollama_bin = shutil.which("ollama") or (
            "/usr/local/bin/ollama" if os.path.exists("/usr/local/bin/ollama") else
            "/usr/bin/ollama" if os.path.exists("/usr/bin/ollama") else None
        )

        if not ollama_bin:
            print("🦙 Installing Ollama local inference engine...")
            try:
                # OLLAMA_NO_START=1 skips systemd service config (which fails in containers/Colab)
                subprocess.run("OLLAMA_NO_START=1 curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
            except Exception as e:
                print(f"⚠️ Primary Ollama install notice: {e}. Trying direct binary archive...")
                try:
                    subprocess.run("curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -d | tar -xf - -C /usr", shell=True, check=True)
                except Exception as ex2:
                    print(f"⚠️ Direct Ollama binary extract notice: {ex2}")

            # Re-check binary location
            ollama_bin = shutil.which("ollama") or (
                "/usr/local/bin/ollama" if os.path.exists("/usr/local/bin/ollama") else
                "/usr/bin/ollama" if os.path.exists("/usr/bin/ollama") else None
            )

        if ollama_bin:
            # Check if server is already responding
            try:
                import httpx
                with httpx.Client(timeout=1.5) as client:
                    res = client.get("http://127.0.0.1:11434/api/tags")
                    ollama_ready = (res.status_code == 200)
            except Exception:
                ollama_ready = False

            if not ollama_ready:
                print(f"🚀 Starting local Ollama server ({ollama_bin})...")
                subprocess.Popen([ollama_bin, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                for _ in range(30):
                    try:
                        import httpx
                        with httpx.Client(timeout=1.0) as client:
                            res = client.get("http://127.0.0.1:11434/api/tags")
                            if res.status_code == 200:
                                ollama_ready = True
                                break
                    except Exception:
                        pass
                    time.sleep(0.5)

            if ollama_ready:
                print(f"📦 Loading {model_name} onto NVIDIA GPU (from Drive cache if available)...")
                try:
                    subprocess.run([ollama_bin, "pull", model_name], check=True)
                    print(f"✅ Local model {model_name} is active on GPU with ZERO external API calls!")
                except Exception as e:
                    print(f"⚠️ Could not pull {model_name} ({e}). Virality heuristic engine active as fallback.")
            else:
                print("ℹ️ Ollama server starting in background. Heuristic Virality Engine available as fallback.")
        else:
            print("ℹ️ Local Ollama binary not found. PeakCut's heuristic Virality Engine active (zero API keys required).")

    # Render rich HTML status if in notebook
    try:
        from IPython.display import display, HTML
        storage_label = f"Google Drive ({drive_output_dir})" if drive_mounted else "Local (/content/output)"
        display(HTML(f"""
        <div style="background: linear-gradient(135deg, #18181b, #09090b); border: 2px solid #10b981; border-radius: 12px; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #ffffff; margin: 15px 0;">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
                <span style="font-size: 26px; margin-right: 12px;">🎉</span>
                <h3 style="margin: 0; font-size: 20px; color: #10b981;">PeakCut 100% Local GPU Environment Ready!</h3>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 14px;">
                <div style="background: #27272a; padding: 10px 14px; border-radius: 8px;">
                    <span style="color: #a1a1aa; font-size: 12px;">HARDWARE ACCELERATION</span><br>
                    <strong style="color: #38bdf8;">⚡ {device_name} (CUDA float16)</strong>
                </div>
                <div style="background: #27272a; padding: 10px 14px; border-radius: 8px;">
                    <span style="color: #a1a1aa; font-size: 12px;">LOCAL AI MODEL</span><br>
                    <strong style="color: #a855f7;">🦙 {model_name} (Zero API Keys)</strong>
                </div>
                <div style="background: #27272a; padding: 10px 14px; border-radius: 8px;">
                    <span style="color: #a1a1aa; font-size: 12px;">PERSISTENT STORAGE</span><br>
                    <strong style="color: #4ade80;">📁 {storage_label}</strong>
                </div>
            </div>
        </div>
        """))
    except Exception:
        pass

    return {
        "status": "ready",
        "has_cuda": has_cuda,
        "device": device_name,
        "model": model_name,
        "drive_mounted": drive_mounted,
        "ollama_ready": ollama_ready,
    }


def parse_url_list(urls_input: Union[str, List[str]]) -> List[str]:
    """Parse string or list of URLs into clean, deduplicated list."""
    raw_urls = []
    if isinstance(urls_input, str):
        for line in urls_input.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                clean = line.split("#")[0].strip().strip('"\'')
                if clean:
                    raw_urls.append(clean)
        # Also split by comma if in a single line
        if len(raw_urls) == 1 and "," in raw_urls[0]:
            raw_urls = [u.strip() for u in raw_urls[0].split(",") if u.strip()]
    elif isinstance(urls_input, (list, tuple)):
        raw_urls = [str(u).strip() for u in urls_input if str(u).strip()]

    # Preserve order while deduplicating
    from security_utils import normalize_media_url
    seen = set()
    cleaned = []
    for u in raw_urls:
        norm = normalize_media_url(u) or u
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
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
) -> Dict[str, Any]:
    """Pure-Python execution for extracting vertical shorts from YouTube videos or streams.

    NO CLI commands required! Everything runs directly in-process with live notebook feedback.
    """
    clean_urls = parse_url_list(urls)
    if not clean_urls:
        print("⚠️ No valid YouTube URLs provided.")
        return {"total_queued": 0, "successful": [], "failed": []}

    target_output = output_dir or os.environ.get("OUTPUT_DIR") or "/content/drive/MyDrive/PeakCut_Shorts"
    if not os.path.exists(target_output):
        try:
            os.makedirs(target_output, exist_ok=True)
        except Exception:
            target_output = "./output"
            os.makedirs(target_output, exist_ok=True)

    if subtitle_preset and subtitle_preset != "none":
        os.environ["SUBTITLE_STYLE"] = subtitle_preset

    if youtube_access_token and youtube_access_token.strip():
        os.environ["YOUTUBE_ACCESS_TOKEN"] = youtube_access_token.strip()

    print("\n" + "=" * 70)
    print("🎬 PeakCut Studio: Autonomous Shorts Pipeline")
    print(f"📦 Total Videos Queued : {len(clean_urls)}")
    print(f"📁 Output Destination  : {target_output}")
    print(f"📐 Aspect Ratio Format : {format} (9:16)")
    print(f"✂️  Clips Per Video    : {clips_per_video}")
    print(f"⏱️  Duration Bounds    : {clip_min_seconds}s - {clip_max_seconds}s")
    if schedule_to_youtube:
        print(f"📤 YouTube Publishing  : Auto-Schedule (staggered by {youtube_interval_hours} hr, {youtube_privacy})")
    print("=" * 70)

    from main import process_video_pipeline

    results = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_queued": len(clean_urls),
        "successful": [],
        "failed": [],
    }

    pipeline_start = time.time()
    all_produced_clips = []

    for idx, url in enumerate(clean_urls, 1):
        job_start = time.time()
        print(f"\n[{idx}/{len(clean_urls)}] 🚀 Processing YouTube URL: {url}")
        job_dir = os.path.join(target_output, f"video_job_{idx}_{int(job_start)}")
        os.makedirs(job_dir, exist_ok=True)

        try:
            job_res = process_video_pipeline(
                input_video_or_url=url,
                output_dir=job_dir,
                is_url=True,
                output_format=format,
                target_clips=clips_per_video,
                clip_min_seconds=clip_min_seconds,
                clip_max_seconds=clip_max_seconds,
                live_record_seconds=live_record_seconds,
                keep_original=keep_original,
            )
            duration = round(time.time() - job_start, 2)
            clips = job_res.get("clips") or []
            all_produced_clips.extend(clips)

            entry = {
                "url": url,
                "title": job_res.get("video_title"),
                "job_dir": job_dir,
                "clips": clips,
                "duration_seconds": duration,
                "metadata": job_res.get("metadata"),
            }
            results["successful"].append(entry)
            print(f"✅ Video {idx} completed in {duration}s! Produced {len(clips)} vertical short(s).")

            # YouTube Auto-Scheduling if enabled
            if schedule_to_youtube and clips:
                try:
                    import youtube_uploader
                    for c_idx, clip_path in enumerate(clips):
                        delay = youtube_interval_hours * (len(all_produced_clips) - len(clips) + c_idx + 1)
                        pub_time = datetime.now(timezone.utc) + timedelta(hours=delay)
                        clip_meta = {}
                        if job_res.get("metadata") and job_res["metadata"].get("shorts"):
                            shorts_list = job_res["metadata"]["shorts"]
                            if c_idx < len(shorts_list):
                                clip_meta = shorts_list[c_idx]

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
            print(f"❌ Video {idx} failed ({e}) in {duration}s.")
            results["failed"].append({
                "url": url,
                "error": str(e),
                "duration_seconds": duration,
            })

    total_time = round(time.time() - pipeline_start, 2)
    results["completed_at"] = datetime.now(timezone.utc).isoformat()
    results["total_duration_seconds"] = total_time
    results["all_clips"] = all_produced_clips

    summary_file = os.path.join(target_output, "batch_summary.json")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    # Rich HTML display in Jupyter / Colab
    try:
        from IPython.display import display, HTML
        clips_html = ""
        for clip in all_produced_clips:
            basename = os.path.basename(clip)
            clips_html += f"""
            <div style="background: #27272a; padding: 12px 16px; border-radius: 8px; margin-top: 8px; border-left: 4px solid #8b5cf6;">
                <strong style="color: #ffffff; font-size: 14px;">🎬 {basename}</strong><br>
                <span style="color: #a1a1aa; font-size: 12px;">📁 Path: {clip}</span>
            </div>
            """

        display(HTML(f"""
        <div style="background: linear-gradient(135deg, #18181b, #09090b); border: 2px solid #8b5cf6; border-radius: 12px; padding: 22px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 16px 0;">
            <h3 style="color: #ffffff; margin: 0 0 8px 0; font-size: 20px;">🎉 Batch Processing Completed!</h3>
            <p style="color: #c4b5fd; font-size: 14px; margin-bottom: 14px;">
                Successfully extracted <b>{len(all_produced_clips)}</b> vertical short(s) across <b>{len(results['successful'])}</b> video(s) in {total_time}s.
            </p>
            <div style="margin-top: 10px;">
                {clips_html if clips_html else "<p style='color: #71717a;'>No clips rendered.</p>"}
            </div>
            <p style="color: #10b981; font-size: 13px; margin-top: 14px;">
                ✅ Saved directly in: <b>{target_output}</b>
            </p>
        </div>
        """))
    except Exception:
        pass

    print("\n" + "=" * 70)
    print(f"🎉 COMPLETED: {len(all_produced_clips)} clips generated in {total_time}s.")
    print(f"📁 Files saved in: {target_output}")
    print("=" * 70 + "\n")
    return results


def launch_web_ui(run_in_background: bool = False) -> Optional[str]:
    """Launch the PeakCut Studio FastAPI server & Cloudflare tunnel without CLI."""
    import re
    subprocess.run("pkill -9 -f cloudflared; pkill -9 -f uvicorn", shell=True)

    # Ensure cloudflared binary is available
    cloudflared_path = "/content/cloudflared"
    if not os.path.exists(cloudflared_path):
        print("🌐 Downloading Cloudflare tunnel binary...")
        subprocess.run(
            f"wget -q -nc -O {cloudflared_path} https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x {cloudflared_path}",
            shell=True, check=True
        )

    log_path = "/content/cloudflared.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    print("🚀 Starting Cloudflare HTTPS tunnel...")
    subprocess.Popen(f"{cloudflared_path} tunnel --url http://127.0.0.1:8000 --logfile {log_path}", shell=True)

    public_url = None
    for _ in range(60):
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
                if match:
                    public_url = match.group(0)
                    break
        time.sleep(0.5)

    if public_url:
        print(f"\n🎉 PEAKCUT STUDIO IS LIVE AT: {public_url}\n")
        try:
            from IPython.display import display, HTML
            display(HTML(f"""
            <div style="background: linear-gradient(135deg, #18181b, #09090b); border: 2px solid #8b5cf6; border-radius: 12px; padding: 22px; text-align: center; margin: 16px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
              <h2 style="color: #ffffff; margin: 0 0 8px 0; font-size: 22px;">🎉 PeakCut Studio is Live!</h2>
              <p style="color: #c4b5fd; font-size: 14px; margin-bottom: 18px;">Click below to open the full web dashboard in your browser:</p>
              <a href="{public_url}" target="_blank" style="background: linear-gradient(135deg, #7c3aed, #6d28d9); color: white; padding: 12px 30px; text-decoration: none; border-radius: 8px; font-weight: 700; font-size: 16px; display: inline-block; box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);">
                🚀 Open PeakCut Studio ↗
              </a>
              <p style="color: #71717a; font-size: 12px; margin-top: 14px;">
                Public Link: <a href="{public_url}" target="_blank" style="color: #a78bfa; text-decoration: underline;">{public_url}</a>
              </p>
            </div>
            """))
        except Exception:
            pass
    else:
        print("⚠️ Cloudflare tunnel starting... check /content/cloudflared.log for details.")

    start_cmd = "python -m uvicorn app:app --host 0.0.0.0 --port 8000"
    if run_in_background:
        print("🟢 Running server in background mode.")
        subprocess.Popen(f"{start_cmd} > /content/uvicorn.log 2>&1", shell=True)
    else:
        print("🟢 Running server in foreground mode (keeps cell active).")
        try:
            subprocess.run(start_cmd, shell=True)
        except KeyboardInterrupt:
            print("\n🛑 PeakCut Studio stopped by user.")

    return public_url


def publish_clip(
    video_path: str,
    title: str = "Viral Highlight #Shorts",
    description: str = "Created with PeakCut Studio #Shorts #Viral",
    privacy: str = "unlisted",
    delay_hours: float = 0.0,
    access_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Single-clip YouTube Shorts publisher (pure Python, zero CLI)."""
    if access_token and access_token.strip():
        os.environ["YOUTUBE_ACCESS_TOKEN"] = access_token.strip()

    import youtube_uploader
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
    """Install a global 'peakcut' executable launcher in system PATH (Linux / Colab)."""
    if not sys.platform.startswith("linux"):
        return False
    candidates = ["/usr/local/bin", "/usr/bin", os.path.expanduser("~/.local/bin")]
    for target_dir in candidates:
        if os.path.isdir(target_dir) and os.access(target_dir, os.W_OK):
            launcher_path = os.path.join(target_dir, "peakcut")
            try:
                with open(launcher_path, "w", encoding="utf-8") as f:
                    f.write("#!/bin/sh\n")
                    f.write(f'exec "{sys.executable}" -m colab_runner "$@"\n')
                os.chmod(launcher_path, 0o755)
                print(f"✅ Global CLI launcher installed at {launcher_path}")
                return True
            except Exception:
                pass
    return False


def show_status() -> Dict[str, Any]:
    """Print Colab and PeakCut system diagnostics, GPU stats, and running services."""
    print("\n" + "=" * 65)
    print("📊 PeakCut Studio System & Process Status")
    print("=" * 65)

    # 1. Environment & Colab status
    in_colab = is_colab()
    print(f"🖥️  Platform: {sys.platform} | Python: {sys.version.split()[0]} | Runtime: {'Google Colab' if in_colab else 'Local / Workstation'}")

    # 2. Hardware / GPU detection
    cuda_avail = False
    device_name = "CPU"
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            device_name = f"{torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPU(s))"
    except Exception:
        pass
    print(f"⚡ Hardware Acceleration: {device_name} (CUDA: {'Active' if cuda_avail else 'Inactive'})")

    # 3. Storage targets
    out_dir = os.environ.get("OUTPUT_DIR", "/content/output" if in_colab else "./output")
    print(f"📁 Active Output Directory: {out_dir}")

    # 4. Service status (Uvicorn / FastAPI)
    server_status = "Offline"
    jobs_count = 0
    recent_jobs = []
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8000/api/status", timeout=2.0) as resp:
            if resp.status == 200:
                server_status = "Online (http://127.0.0.1:8000)"
        with urllib.request.urlopen("http://127.0.0.1:8000/api/jobs", timeout=2.0) as resp:
            if resp.status == 200:
                jobs_data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                jobs_count = len(jobs_data)
                recent_jobs = jobs_data[:5]
    except Exception:
        pass
    print(f"🌐 Web API Server: {server_status}")

    # 5. Ollama local model service
    ollama_status = "Not running"
    ollama_models = []
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2.0) as resp:
            if resp.status == 200:
                tags = json.loads(resp.read().decode("utf-8", errors="ignore"))
                ollama_models = [m.get("name") for m in tags.get("models", [])]
                ollama_status = f"Online (Models: {', '.join(ollama_models) if ollama_models else 'None loaded'})"
    except Exception:
        pass
    print(f"🦙 Ollama Local Inference: {ollama_status}")

    # 6. Jobs summary if online
    if recent_jobs:
        print("\n🎬 Recent Jobs:")
        for j in recent_jobs:
            title = j.get("title") or "Untitled Video"
            status = j.get("status") or "unknown"
            progress = int((j.get("progress") or 0.0) * 100)
            clips_cnt = len(j.get("clips") or [])
            print(f"   • [{status.upper()}] {title} — {progress}% ({clips_cnt} clip(s))")

    print("=" * 65 + "\n")
    return {
        "is_colab": in_colab,
        "cuda_available": cuda_avail,
        "device": device_name,
        "server_status": server_status,
        "ollama_status": ollama_status,
        "jobs_count": jobs_count,
        "output_dir": out_dir,
    }


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for PeakCut Studio."""
    parser = argparse.ArgumentParser(
        prog="peakcut",
        description="PeakCut Studio CLI: Autonomous AI Shorts Pipeline for Google Colab and Local Workstations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  peakcut setup --model gemma2:2b --whisper base
  peakcut clip "https://youtube.com/watch?v=kYfNvmF0Bqw" --clips 3 --format vertical
  peakcut serve --background
  peakcut status
  peakcut upload "/path/to/clip.mp4" --title "Viral Short" --privacy unlisted
        """,
    )
    sub = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # setup
    p_setup = sub.add_parser("setup", help="Configure 100%% zero-API local GPU environment & Google Drive")
    p_setup.add_argument("--model", default="gemma2:2b", help="Local Gemma/Ollama model name (default: gemma2:2b)")
    p_setup.add_argument("--whisper", default="base", help="Faster-Whisper model size (default: base)")
    p_setup.add_argument("--no-drive", action="store_true", help="Do not mount Google Drive (use local storage)")

    # clip / run
    p_clip = sub.add_parser("clip", aliases=["run"], help="Extract vertical shorts from YouTube URLs or video files")
    p_clip.add_argument("urls", nargs="+", help="YouTube URLs, playlist links, or local video file paths")
    p_clip.add_argument("-n", "--clips", type=int, default=3, help="Number of clips to generate per video (default: 3)")
    p_clip.add_argument("-f", "--format", choices=["vertical", "horizontal", "square", "auto"], default="vertical", help="Aspect ratio (default: vertical 9:16)")
    p_clip.add_argument("-s", "--subtitles", choices=["karaoke", "minimal", "bold", "none"], default="karaoke", help="Subtitle styling preset (default: karaoke)")
    p_clip.add_argument("--min-len", type=float, default=15.0, help="Minimum clip duration in seconds (default: 15)")
    p_clip.add_argument("--max-len", type=float, default=60.0, help="Maximum clip duration in seconds (default: 60)")
    p_clip.add_argument("--live-record", type=int, default=900, help="Live stream recording duration in seconds (default: 900)")
    p_clip.add_argument("-o", "--output", default=None, help="Output directory (defaults to Google Drive or ./output)")
    p_clip.add_argument("--keep-original", action="store_true", help="Retain original downloaded source video")
    p_clip.add_argument("--schedule", action="store_true", help="Auto-schedule generated clips to YouTube Shorts")
    p_clip.add_argument("--interval", type=float, default=1.0, help="Hours between scheduled uploads (default: 1.0)")
    p_clip.add_argument("--privacy", choices=["unlisted", "private", "public"], default="unlisted", help="YouTube privacy status (default: unlisted)")
    p_clip.add_argument("--token", default=None, help="YouTube OAuth access token")

    # serve / launch
    p_serve = sub.add_parser("serve", aliases=["launch"], help="Launch PeakCut Studio Web UI server & Cloudflare tunnel")
    p_serve.add_argument("-b", "--background", action="store_true", help="Run server in background process")
    p_serve.add_argument("-p", "--port", type=int, default=8000, help="Server port (default: 8000)")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")

    # upload / publish
    p_upload = sub.add_parser("upload", aliases=["publish"], help="Upload or schedule a specific clip to YouTube Shorts")
    p_upload.add_argument("video_path", help="Path to the clip video file (.mp4)")
    p_upload.add_argument("-t", "--title", default="Viral Highlight #Shorts", help="Video title")
    p_upload.add_argument("-d", "--description", default="Created with PeakCut Studio #Shorts #Viral", help="Video description")
    p_upload.add_argument("-p", "--privacy", choices=["unlisted", "private", "public"], default="unlisted", help="Upload privacy status (default: unlisted)")
    p_upload.add_argument("--delay", type=float, default=0.0, help="Delay in hours before publishing (0 = immediate)")
    p_upload.add_argument("--token", default=None, help="YouTube OAuth access token")

    # status
    sub.add_parser("status", help="Inspect active server, jobs, GPU diagnostics, and storage")

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
        )
        install_cli_launcher()
        return 0

    elif sub in ("clip", "run"):
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
        )
        has_success = bool(results.get("successful"))
        has_failure = bool(results.get("failed"))
        return 0 if (has_success or not has_failure) else 1

    elif sub in ("serve", "launch"):
        launch_web_ui(run_in_background=parsed_args.background)
        return 0

    elif sub in ("upload", "publish"):
        publish_clip(
            video_path=parsed_args.video_path,
            title=parsed_args.title,
            description=parsed_args.description,
            privacy=parsed_args.privacy,
            delay_hours=parsed_args.delay,
            access_token=parsed_args.token,
        )
        return 0

    elif sub == "status":
        show_status()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
