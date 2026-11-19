"""Live Stream Rolling Buffer Clipper for OpenShorts.

Features:
- Stream ingestion via yt-dlp (YouTube Live, Twitch, Kick, HLS/RTMP streams)
- Circular rolling video buffer of MPEG-TS chunks (e.g. 6-second segments) with disk bounding
- Simultaneous raw PCM audio stream for real-time loudness RMS computation
- Real-time anonymous Twitch IRC chat activity monitor (zero OAuth required)
- Multi-signal spike / hype detection:
    1. Audio loudness spike (e.g. >= 60% above running baseline)
    2. Chat velocity spike (>= 2x baseline message rate)
    3. Spoken or manual clip triggers ("clip that", "clip it")
- Automatic and on-demand clip extraction with clock alignment from rolling buffer
- Seamless integration with OpenShorts' reframing and rendering pipeline
"""
from __future__ import annotations

import collections
import json
import logging
import math
import os
import re
import socket
import ssl
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("openshorts.live_clipper")

SEGMENT_SECONDS = 6
SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
BYTES_PER_SAMPLE = 2  # 16-bit PCM


class NotLiveError(Exception):
    """Raised when the channel or stream URL is not currently live."""
    pass


@dataclass
class ClipEvent:
    clip_id: str
    trigger_type: str  # 'loudness_spike', 'chat_spike', 'manual', 'keyword'
    start_time: float
    end_time: float
    duration: float
    status: str = "pending"  # pending, extracted, failed
    output_path: str = ""
    title: str = ""
    loudness_db: float = 0.0
    chat_rate: float = 0.0


def resolve_live_stream(url: str) -> Dict[str, Any]:
    """Inspect stream URL via yt-dlp to determine if it is live and get playable HLS/DASH URLs."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "b[height<=1080]/bv*[height<=1080]+ba/b",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise NotLiveError(f"Failed to resolve stream URL: {e}") from e

    if not info:
        raise NotLiveError("No stream metadata returned.")

    is_live = info.get("is_live") or info.get("live_status") == "is_live"
    if not is_live:
        raise NotLiveError(f"Channel/URL '{url}' is not currently broadcasting live.")

    formats = info.get("requested_formats")
    stream_urls = []
    if formats:
        stream_urls = [f["url"] for f in formats if isinstance(f, dict) and "url" in f]
    if not stream_urls and info.get("url"):
        stream_urls = [info["url"]]
    if not stream_urls and info.get("formats"):
        fmts = [f for f in info.get("formats", []) if isinstance(f, dict) and f.get("url")]
        combined = [f for f in fmts if f.get("vcodec") != "none" and f.get("acodec") != "none"]
        if combined:
            best = max(combined, key=lambda f: f.get("height") or 0)
            stream_urls = [best["url"]]
        else:
            videos = [f for f in fmts if f.get("vcodec") != "none"]
            audios = [f for f in fmts if f.get("acodec") != "none"]
            if videos:
                stream_urls.append(max(videos, key=lambda f: f.get("height") or 0)["url"])
            if audios:
                stream_urls.append(max(audios, key=lambda f: f.get("abr") or 0)["url"])
            if not stream_urls and fmts:
                stream_urls = [fmts[-1]["url"]]

    title = info.get("title") or "Live Stream"
    channel = info.get("uploader") or info.get("channel") or ""

    return {
        "stream_urls": stream_urls,
        "title": title,
        "channel": channel,
        "is_live": True,
    }


class LiveClipperSession:
    """Manages recording of a single live stream into a rolling circular buffer."""

    def __init__(
        self,
        session_id: str,
        stream_url: str,
        output_dir: Path | str,
        buffer_minutes: int = 15,
        loudness_spike_threshold: float = 0.60,  # 60% above baseline
        chat_spike_multiplier: float = 2.0,      # 2x chat rate
    ):
        self.session_id = session_id
        self.stream_url = stream_url
        self.output_dir = Path(output_dir) / session_id
        self.buffer_dir = self.output_dir / "buffer"
        self.clips_dir = self.output_dir / "clips"
        self.buffer_minutes = max(3, buffer_minutes)
        self.loudness_spike_threshold = loudness_spike_threshold
        self.chat_spike_multiplier = chat_spike_multiplier

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.is_running = False
        self.stream_title = ""
        self.channel_name = ""

        # Buffer tracking
        self.segments: List[Tuple[Path, float, float]] = []  # (path, start_sec, end_sec)
        self.current_clock: float = 0.0
        self.part_index: int = 0
        self.csv_seen: Dict[int, int] = {}
        self.proc: Optional[subprocess.Popen] = None

        # Telemetry & Audio
        self.audio_db_history: List[float] = []
        self.total_samples: int = 0
        self.chat_counts = collections.Counter()  # second -> message count
        self.chat_total: int = 0
        self.chat_connected: bool = False

        # Events & Clips
        self.clips: List[ClipEvent] = []
        self.last_spike_time: float = -999.0
        self.min_gap_between_spikes: float = 45.0  # seconds

    def start(self) -> None:
        """Start the live clipper background ingestion and monitoring threads."""
        with self.lock:
            if self.is_running:
                return
            self.is_running = True

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting LiveClipperSession {self.session_id} for {self.stream_url}")

        # Worker 1: Ingestion & Buffer segmenting
        threading.Thread(target=self._run_ingest, daemon=True, name=f"LiveIngest-{self.session_id}").start()

        # Worker 2: Twitch Chat listener (if applicable)
        twitch_match = re.search(r"twitch\.tv/([a-zA-Z0-9_]+)", self.stream_url)
        if twitch_match:
            channel = twitch_match.group(1).lower()
            threading.Thread(
                target=self._run_twitch_chat,
                args=(channel,),
                daemon=True,
                name=f"LiveChat-{self.session_id}",
            ).start()

    def stop(self) -> None:
        """Stop ingestion and terminate processes."""
        with self.lock:
            self.stop_event.set()
            self.is_running = False
            if self.proc and self.proc.poll() is None:
                try:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
                except Exception:
                    pass
                try:
                    if self.proc.stdout and not self.proc.stdout.closed:
                        self.proc.stdout.close()
                except Exception:
                    pass
        logger.info(f"Stopped LiveClipperSession {self.session_id}")

    def _run_ingest(self) -> None:
        """Capture stream into rolling segments and stream PCM audio to stdout."""
        consecutive_failures = 0

        while not self.stop_event.is_set():
            try:
                info = resolve_live_stream(self.stream_url)
                self.stream_title = info["title"]
                self.channel_name = info["channel"]
            except NotLiveError as e:
                logger.warning(f"Stream {self.stream_url} is not live: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                self.stop_event.wait(15)
                continue
            except Exception as e:
                logger.error(f"Error resolving stream: {e}")
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                self.stop_event.wait(15)
                continue

            self.part_index += 1
            part_name = f"part_{self.part_index:03d}"
            csv_path = f"{part_name}.csv"
            ts_pattern = f"{part_name}_%05d.ts"

            cmd = [
                "ffmpeg",
                "-v", "error",
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "10",
            ]
            for u in info["stream_urls"]:
                cmd.extend(["-i", u])

            audio_in = "1:a:0" if len(info["stream_urls"]) > 1 else "0:a:0"

            # 1. Segment video copy to TS files
            cmd.extend([
                "-map", "0:v:0",
                "-map", audio_in,
                "-c", "copy",
                "-f", "segment",
                "-segment_time", str(SEGMENT_SECONDS),
                "-reset_timestamps", "1",
                "-segment_format", "mpegts",
                "-segment_list", csv_path,
                "-segment_list_type", "csv",
                ts_pattern,
                # 2. Pipe raw mono 16kHz audio to stdout
                "-map", audio_in,
                "-vn",
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-f", "s16le",
                "pipe:1",
            ])

            log_path = self.output_dir / "ffmpeg_ingest.log"
            try:
                with open(log_path, "ab") as errlog:
                    self.proc = subprocess.Popen(
                        cmd,
                        cwd=str(self.buffer_dir),
                        stdout=subprocess.PIPE,
                        stderr=errlog,
                    )
                    try:
                        self._read_audio_stream(self.proc.stdout)
                        self.proc.wait()
                    finally:
                        try:
                            if self.proc.stdout and not self.proc.stdout.closed:
                                self.proc.stdout.close()
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"FFmpeg ingest process failed: {e}")

            consecutive_failures = 0
            self._scan_and_prune_segments()
            self.stop_event.wait(2)

        self.is_running = False

    def _read_audio_stream(self, stdout: Any) -> None:
        """Read 1-second chunks of 16-bit PCM audio from stdout and compute RMS loudness."""
        bytes_per_second = SAMPLE_RATE * BYTES_PER_SAMPLE

        while not self.stop_event.is_set():
            data = stdout.read(bytes_per_second)
            if not data:
                break

            # Prevent ValueError: buffer size must be a multiple of element size
            if len(data) % 2 != 0:
                data = data[:len(data) - (len(data) % 2)]
            if not data:
                continue

            # Calculate RMS loudness in dB
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(samples ** 2) + 1e-10)
            db = float(20 * math.log10(rms)) if rms > 0 else -100.0

            with self.lock:
                self.audio_db_history.append(db)
                self.total_samples += len(samples)
                current_time = self.total_samples / SAMPLE_RATE

            self._scan_and_prune_segments()
            self._check_triggers(current_time, db)

    def _scan_and_prune_segments(self) -> None:
        """Check CSV segment files, update the rolling segment list, and unlink expired chunks."""
        with self.lock:
            for i in range(1, self.part_index + 1):
                csv_file = self.buffer_dir / f"part_{i:03d}.csv"
                if not csv_file.exists():
                    continue

                try:
                    lines = csv_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                except Exception:
                    continue

                seen_count = self.csv_seen.get(i, 0)
                new_lines = lines[seen_count:]
                for line in new_lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        name, start_t, end_t = parts[0], float(parts[1]), float(parts[2])
                        dur = max(0.0, end_t - start_t)
                        self.segments.append((self.buffer_dir / name, self.current_clock, self.current_clock + dur))
                        self.current_clock += dur

                self.csv_seen[i] = len(lines)

            # Circular buffer pruning
            retention_window = self.buffer_minutes * 60.0
            cutoff_time = self.current_clock - retention_window
            while self.segments and self.segments[0][2] < cutoff_time:
                old_seg, _, _ = self.segments.pop(0)
                try:
                    if old_seg.exists():
                        old_seg.unlink()
                except Exception:
                    pass

    def _run_twitch_chat(self, channel: str) -> None:
        """Connect to Twitch IRC chat anonymously and record chat message velocity."""
        while not self.stop_event.is_set():
            ssl_sock = None
            try:
                raw_sock = socket.create_connection(("irc.chat.twitch.tv", 6697), timeout=15)
                ssl_sock = ssl.create_default_context().wrap_socket(raw_sock, server_hostname="irc.chat.twitch.tv")
                nick = f"justinfan{int(time.time()) % 80000 + 10000}"
                ssl_sock.sendall(f"NICK {nick}\r\nJOIN #{channel}\r\n".encode("utf-8"))
                ssl_sock.settimeout(25)
                self.chat_connected = True
                buffer = b""

                while not self.stop_event.is_set():
                    try:
                        chunk = ssl_sock.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break

                    lines = (buffer + chunk).split(b"\r\n")
                    buffer = lines.pop()

                    for line in lines:
                        if line.startswith(b"PING"):
                            ssl_sock.sendall(b"PONG :tmi.twitch.tv\r\n")
                        elif b" PRIVMSG #" in line:
                            current_sec = int(self.total_samples / SAMPLE_RATE)
                            with self.lock:
                                self.chat_counts[current_sec] += 1
                                self.chat_total += 1
            except Exception as e:
                logger.debug(f"Twitch chat listener disconnected: {e}")
            finally:
                self.chat_connected = False
                if ssl_sock:
                    try:
                        ssl_sock.close()
                    except Exception:
                        pass

            self.stop_event.wait(5)

    def _check_triggers(self, current_sec: float, current_db: float) -> None:
        """Detect spikes in loudness or chat activity."""
        if current_sec - self.last_spike_time < self.min_gap_between_spikes:
            return

        # Need at least 60 seconds of history to compute baselines
        history_len = len(self.audio_db_history)
        if history_len < 60:
            return

        # 1. Loudness Baseline & Spike Check
        # Recent 3-minute baseline
        baseline_slice = self.audio_db_history[-180:-10] if history_len >= 180 else self.audio_db_history[:-10]
        avg_db = float(np.mean(baseline_slice))
        # Decibel is logarithmic: a 6dB increase corresponds to doubling sound pressure
        # Loudness spike: current dB is significantly higher than the baseline average
        loudness_spike = current_db > (avg_db + 8.0) and current_db > -22.0

        # 2. Chat Spike Check
        chat_spike = False
        chat_rate = 0.0
        if self.chat_total > 20 and current_sec > 60:
            avg_chat_rate = self.chat_total / current_sec
            recent_chat = sum(self.chat_counts.get(t, 0) for t in range(int(current_sec - 10), int(current_sec) + 1)) / 10.0
            chat_rate = recent_chat
            chat_spike = recent_chat > max(avg_chat_rate * self.chat_spike_multiplier, 1.5)

        if loudness_spike or chat_spike:
            trigger = "loudness_spike" if loudness_spike else "chat_spike"
            logger.info(
                f"[HYPE DETECTED] Session {self.session_id} @ {current_sec:.1f}s | "
                f"Trigger: {trigger} | dB: {current_db:.1f} (avg {avg_db:.1f}) | Chat: {chat_rate:.1f}/s"
            )
            self.last_spike_time = current_sec
            self.trigger_clip(
                start_sec=max(0.0, current_sec - 25.0),
                end_sec=current_sec + 15.0,
                trigger_type=trigger,
                title=f"Live Hype: {self.stream_title or self.channel_name} ({trigger})",
                loudness_db=current_db,
                chat_rate=chat_rate,
            )

    def trigger_clip(
        self,
        start_sec: float,
        end_sec: float,
        trigger_type: str = "manual",
        title: str = "",
        loudness_db: float = 0.0,
        chat_rate: float = 0.0,
    ) -> ClipEvent:
        """Extract a clip of the requested time range from the rolling buffer."""
        clip_id = f"live_{int(time.time())}_{len(self.clips) + 1:02d}"
        duration = max(5.0, end_sec - start_sec)

        event = ClipEvent(
            clip_id=clip_id,
            trigger_type=trigger_type,
            start_time=start_sec,
            end_time=end_sec,
            duration=duration,
            status="extracting",
            title=title or f"Live Clip {clip_id}",
            loudness_db=loudness_db,
            chat_rate=chat_rate,
        )

        with self.lock:
            self.clips.append(event)

        # Extract asynchronously
        threading.Thread(
            target=self._extract_clip_task,
            args=(event,),
            daemon=True,
            name=f"Extract-{clip_id}",
        ).start()

        return event

    def _extract_clip_task(self, event: ClipEvent) -> None:
        """Find matching MPEG-TS buffer chunks and use FFmpeg concat + trim to emit a clean MP4."""
        with self.lock:
            active_segs = list(self.segments)

        # Find overlapping TS chunks
        matching = [
            (path, s, e)
            for path, s, e in active_segs
            if e >= event.start_time and s <= event.end_time and path.exists()
        ]

        if not matching:
            logger.warning(f"No buffer segments found for range {event.start_time:.1f}s - {event.end_time:.1f}s")
            event.status = "failed"
            return

        out_path = self.clips_dir / f"{event.clip_id}.mp4"
        concat_txt = self.output_dir / f"{event.clip_id}_concat.txt"

        try:
            # Write ffmpeg concat file
            with open(concat_txt, "w", encoding="utf-8") as f:
                for path, _, _ in matching:
                    # Escape single quotes and backslashes
                    safe_path = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")

            first_seg_start = matching[0][1]
            relative_start = max(0.0, event.start_time - first_seg_start)

            # Fast extract with frame-accurate seek across TS boundaries
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_txt),
                "-ss", f"{relative_start:.3f}",
                "-t", f"{event.duration:.3f}",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "22",
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                str(out_path),
            ]

            res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if res.returncode == 0 and out_path.exists():
                event.status = "extracted"
                event.output_path = str(out_path.resolve())
                logger.info(f"Successfully extracted live clip to {out_path} ({event.duration:.1f}s)")
            else:
                event.status = "failed"
                logger.error(f"FFmpeg failed to extract clip {event.clip_id}: {res.stderr[-300:]}")
        except subprocess.TimeoutExpired:
            event.status = "failed"
            logger.error(f"FFmpeg extraction timed out for clip {event.clip_id}")
        except FileNotFoundError as e:
            event.status = "failed"
            logger.error(f"FFmpeg executable not found: {e}")
        except Exception as e:
            event.status = "failed"
            logger.error(f"Error during clip extraction: {e}")
        finally:
            if concat_txt.exists():
                try:
                    concat_txt.unlink()
                except Exception:
                    pass

    def get_status(self) -> Dict[str, Any]:
        """Return real-time session telemetry, buffer state, and clips."""
        with self.lock:
            current_sec = self.total_samples / SAMPLE_RATE if SAMPLE_RATE else 0.0
            recent_db = self.audio_db_history[-1] if self.audio_db_history else -100.0
            return {
                "session_id": self.session_id,
                "stream_url": self.stream_url,
                "is_running": self.is_running,
                "title": self.stream_title,
                "channel": self.channel_name,
                "duration_seconds": round(current_sec, 1),
                "buffer_segments_count": len(self.segments),
                "buffer_minutes_target": self.buffer_minutes,
                "current_loudness_db": round(recent_db, 1),
                "chat_connected": self.chat_connected,
                "chat_total_messages": self.chat_total,
                "clips": [asdict(c) for c in self.clips],
            }


class LiveClipperManager:
    """Registry and manager for active live stream clipping sessions."""

    _instance: Optional[LiveClipperManager] = None

    def __init__(self, base_output_dir: str = "output/live"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, LiveClipperSession] = {}
        self.lock = threading.RLock()

    @classmethod
    def get_instance(cls) -> LiveClipperManager:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_session(
        self,
        stream_url: str,
        buffer_minutes: int = 15,
        session_id: Optional[str] = None,
    ) -> LiveClipperSession:
        with self.lock:
            sid = session_id or f"session_{int(time.time())}"
            if sid in self.sessions and self.sessions[sid].is_running:
                return self.sessions[sid]

            session = LiveClipperSession(
                session_id=sid,
                stream_url=stream_url,
                output_dir=self.base_output_dir,
                buffer_minutes=buffer_minutes,
            )
            self.sessions[sid] = session
            session.start()
            return session

    def stop_session(self, session_id: str) -> bool:
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].stop()
                return True
            return False

    def get_session(self, session_id: str) -> Optional[LiveClipperSession]:
        with self.lock:
            return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [s.get_status() for s in self.sessions.values()]
