"""Central video-encoder selection for every ffmpeg encode call site.

FFMPEG_ENCODER env values:
  x264  (default) — CPU libx264, exact pre-GPU behavior
  nvenc           — force h264_nvenc; probed once and falls back to x264
                    (with a warning) if the GPU/driver is unavailable
  auto            — h264_nvenc when the probe succeeds, else x264

Only the codec/quality args live here; surrounding args (-movflags, -pix_fmt,
audio codecs, filters) stay at each call site.
"""
import json
import math
import os
import re
import shutil
import subprocess
import threading

# Quality tiers pinning the historical libx264 settings.
QUALITY = "quality"            # was: -preset medium -crf 18
QUALITY_FAST = "quality_fast"  # was: -preset fast -crf 18
DELIVERY = "delivery"          # was: -preset fast -crf 22

_X264_ARGS = {
    QUALITY: ["-c:v", "libx264", "-preset", "medium", "-crf", "18"],
    QUALITY_FAST: ["-c:v", "libx264", "-preset", "fast", "-crf", "18"],
    DELIVERY: ["-c:v", "libx264", "-preset", "fast", "-crf", "22"],
}

# NVENC -cq is not 1:1 with x264 CRF: benchmarked on the prod GPU (RTX 4000
# Ada), cq ≈ crf + 7 lands in the same file-size ballpark, with vbr + AQ for
# quality. Presets p1-p7: p5 ≈ "medium", p4 ≈ "fast".
# -pix_fmt yuv420p is REQUIRED: with RGB input (the bgr24 rawvideo pipe from
# OpenCV) nvenc otherwise emits H.264 in gbrp/GBR colorspace, which ffmpeg
# reads fine but web players render as a magenta/green mess.
_NVENC_ARGS = {
    QUALITY: ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
              "-rc", "vbr", "-cq", "25", "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1", "-pix_fmt", "yuv420p"],
    QUALITY_FAST: ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                   "-rc", "vbr", "-cq", "25", "-b:v", "0", "-spatial-aq", "1",
                   "-pix_fmt", "yuv420p"],
    DELIVERY: ["-c:v", "h264_nvenc", "-preset", "p4",
               "-rc", "vbr", "-cq", "29", "-b:v", "0", "-spatial-aq", "1",
               "-pix_fmt", "yuv420p"],
}

# Output args that drop container/stream metadata carried over from the source
# — most notably YouTube's "produced by Google Inc." stream handler, which
# otherwise survives every re-encode (ffmpeg copies input metadata by default)
# and rides into the published clip. The per-stream specifiers are required:
# global -map_metadata -1 alone leaves the audio handler_name intact on a
# stream copy. Empty audio/video specifiers are harmless when a clip lacks that
# stream (ffmpeg ignores them, verified). Spliced in before the output filename
# at each final-artifact producer; kept out of video_encode_args() so that
# stays purely codec/quality args.
METADATA_SCRUB = ["-map_metadata", "-1", "-map_chapters", "-1",
                  "-map_metadata:s:v", "-1", "-map_metadata:s:a", "-1"]

# Loudness normalisation for the delivered clip.
#
# Without this the clip inherits whatever the source was mastered at, so a
# user's clips land anywhere: measured across real delivered clips on
# 26-jul-2026, from -13.8 LUFS on a loud upload down to -28 LUFS on a quiet
# talk. TikTok, Reels and Shorts all normalise playback to roughly -14 LUFS,
# which means the quiet ones just sound thin next to everything else in the
# feed — the loud ones aren't rewarded, the quiet ones are punished.
#
# I=-14 matches the platforms' target, LRA=11 is the usual allowance for speech.
# Applied at the clip cut, where the audio is being encoded to AAC anyway, so it
# costs nothing extra. AUDIO_NORMALIZE=0 turns it off.
#
# TP=-2.0, not the -1.5 that matches the platforms' own advice, because the
# ceiling is enforced BEFORE the AAC encode and the encoder then adds
# inter-sample peaks on top. Measured over 14 corpus clips (31-jul-2026):
#
#   TP=-1.5   peak reached +0.2 dBTP, 1 clip clipping,  8 above -1.0
#   TP=-2.0   peak reached -0.3 dBTP, 0 clipping,       5 above -1.0
#   TP=-3.0   peak reached -0.9 dBTP, 0 clipping,       1 above -1.0
#
# -3.0 also costs level: only 8 of 14 stayed inside -15..-13 LUFS versus 12 at
# -2.0, and level is what the listener notices. Two other fixes were tried and
# do NOT work, so don't reach for them again: an `alimiter` after loudnorm
# (limits sample peaks, not inter-sample, and measured WORSE at +0.7), and
# two-pass loudnorm with linear=true (+0.4, still clipping). The overshoot is
# the codec's, so the only lever is headroom.
LOUDNORM_FILTER = "loudnorm=I=-14:TP=-2.0:LRA=11"


# AI Act art. 50(2): a provider whose system generates synthetic audio or video
# has to mark the output in a machine-readable way. The regulation asks for the
# marking, not for a particular standard, and the cheap one that survives every
# player and every upload is a container tag: `ffprobe -show_format` reads it
# back, and TikTok/Reels/YouTube ignore it. Stamped by a stream copy, so it is
# a remux (no quality loss, ~0.2 s) and never a re-encode.
#
# Deliberately narrow: it goes on outputs where a machine actually synthesised
# voice or a person (dubbing, AI actors), not on an ordinary clip, whose audio
# and pixels are the user's own footage. Marking everything would make the tag
# mean nothing.
AI_DISCLOSURE = "AI-generated content produced with OpenShorts (openshorts.app)"


def mark_ai_generated(path, detail=""):
    """Stamp AI Act art. 50(2) machine-readable tags on a finished file.

    Returns True when the file now carries the tags. Never raises: a missing
    tag must not lose the user the video they just paid minutes for.
    """
    note = f"{AI_DISCLOSURE}: {detail}" if detail else AI_DISCLOSURE
    tmp = f"{path}.aitag.mp4"
    # `comment` and not a custom `ai_generated` key: mp4 only carries the
    # standard iTunes-style tags, and ffmpeg drops anything else without
    # warning (verified with ffprobe -show_entries format_tags).
    # -map 0 because ffmpeg's default picks one stream per type: a dubbed file
    # that ever ships two audio tracks would come back with one.
    cmd = ["ffmpeg", "-y", "-i", path, "-map", "0", "-c", "copy",
           "-metadata", f"comment={note}",
           "-movflags", "+faststart", tmp]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0 or not os.path.exists(tmp) or os.path.getsize(tmp) == 0:
            print(f"[ai-tag] skipped for {os.path.basename(path)}: {r.stderr[-300:]}")
            if os.path.exists(tmp):
                os.remove(tmp)
            return False
        os.replace(tmp, path)
        return True
    except Exception as e:
        print(f"[ai-tag] skipped for {os.path.basename(path)}: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False


def audio_encode_args():
    """AAC encode args for a delivered clip, with loudness normalisation."""
    args = ["-c:a", "aac"]
    if os.environ.get("AUDIO_NORMALIZE", "1").strip() != "0":
        args = ["-af", LOUDNORM_FILTER] + args
    return args

_probe_lock = threading.Lock()
_nvenc_ok = None  # None = not probed yet
_announced = False


def _probe_nvenc():
    """One tiny lavfi encode to prove h264_nvenc works end-to-end.

    NVENC rejects frames smaller than ~145px, so the probe uses 256x256.
    Any failure (no ffmpeg binary, no GPU, no driver libs) means False.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
        "-c:v", "h264_nvenc", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False


def nvenc_available():
    """Probe h264_nvenc once and cache the verdict (thread-safe)."""
    global _nvenc_ok
    if _nvenc_ok is None:
        with _probe_lock:
            if _nvenc_ok is None:
                _nvenc_ok = _probe_nvenc()
    return _nvenc_ok


def reset_encoder_cache():
    """Test hook: forget the cached probe result."""
    global _nvenc_ok, _announced
    with _probe_lock:
        _nvenc_ok = None
        _announced = False


def video_encode_args(tier=QUALITY):
    """Return the codec/quality args for one encode, honoring FFMPEG_ENCODER."""
    global _announced
    if tier not in _X264_ARGS:
        raise ValueError(f"Unknown encode tier: {tier!r}")

    mode = os.environ.get("FFMPEG_ENCODER", "x264").strip().lower()
    use_nvenc = False
    if mode in ("nvenc", "auto"):
        use_nvenc = nvenc_available()
        if mode == "nvenc" and not use_nvenc:
            print("⚠️ [Encoder] FFMPEG_ENCODER=nvenc but h264_nvenc is not "
                  "usable here — falling back to libx264")

    if not _announced:
        _announced = True
        print(f"🎞️ [Encoder] video encoder: {'h264_nvenc' if use_nvenc else 'libx264'} "
              f"(FFMPEG_ENCODER={mode})")

    return list((_NVENC_ARGS if use_nvenc else _X264_ARGS)[tier])


def escape_filter_value(value):
    r"""Escape a path/value for use inside a quoted FFmpeg filter argument.

    Windows absolute paths are why this exists: ``:`` separates filter options,
    so an interpolated ``C:/x/y.txt`` makes the parser look for an option named
    ``/x/y.txt`` and the whole filtergraph fails to build.

    NOTE: an apostrophe in the path cannot be made safe here. ffmpeg's
    filtergraph parser is not a shell -- the shell idiom ``'\''`` was tried on
    29-jul-2026 and is worse than doing nothing: it drops the apostrophe AND
    swallows the following option, so ``ass='...Earth'\''s.ass':fontsdir='...'``
    resolved to a filename of "...Earths.ass:fontsdir=..." and failed to open.

    The only reliable answer is to keep apostrophes OUT of any path that is
    interpolated into a filter. Callers generate their own filenames, so they
    control this: use a neutral name, never one derived from a video title.
    """
    return value.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")


def _parse_rate(raw):
    """Parse an ffprobe rational frame rate like '30000/1001' or '30/1'."""
    if not raw:
        return 0.0
    s = str(raw).strip()
    if "/" in s:
        num, den = s.split("/", 1)
        try:
            n, d = float(num), float(den)
            if d > 0 and n > 0:
                val = n / d
                return val if math.isfinite(val) else 0.0
        except ValueError:
            return 0.0
        return 0.0
    try:
        val = float(s)
        return val if math.isfinite(val) and val > 0 else 0.0
    except ValueError:
        return 0.0


def _empty_media_info(path, error=None):
    exists = bool(path and os.path.isfile(str(path)))
    size = 0
    if exists:
        try:
            size = os.path.getsize(str(path))
        except OSError:
            size = 0
    return {
        "valid": False,
        "exists": exists,
        "size_bytes": size,
        "duration": 0.0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "video_codec": None,
        "audio_codec": None,
        "has_video": False,
        "has_audio": False,
        "pix_fmt": None,
        "audio_sample_rate": 0,
        "video_duration": 0.0,
        "audio_duration": 0.0,
        "error": error,
    }


def probe_media(path, ffprobe_bin=None, ffmpeg_bin=None, timeout=30):
    """Probe media metadata reliably using ffprobe JSON (with ffmpeg -i fallback).

    Never fabricates metadata: missing, empty, or corrupt files return
    ``valid=False`` with ``duration=0.0`` and a descriptive ``error``.
    """
    file_path = str(path)
    if not file_path or not os.path.isfile(file_path):
        return _empty_media_info(file_path, error="file does not exist")

    try:
        size_bytes = os.path.getsize(file_path)
    except OSError as e:
        return _empty_media_info(file_path, error=f"cannot stat file: {e}")
    if size_bytes <= 0:
        return _empty_media_info(file_path, error="file is empty (0 bytes)")

    probe_exe = ffprobe_bin or shutil.which("ffprobe") or "ffprobe"
    ffprobe_err = None
    try:
        proc = subprocess.run(
            [
                probe_exe, "-v", "error",
                "-print_format", "json",
                "-show_format", "-show_streams",
                file_path,
            ],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            data = json.loads(proc.stdout)
            fmt = data.get("format") or {}
            streams = data.get("streams") or []

            v_stream = next((s for s in streams
                             if s.get("codec_type") == "video"
                             and not (s.get("disposition") or {}).get("attached_pic")), None)
            a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

            def _to_pos_float(val):
                try:
                    f = float(val)
                    return f if math.isfinite(f) and f > 0 else 0.0
                except (TypeError, ValueError):
                    return 0.0

            fmt_dur = _to_pos_float(fmt.get("duration"))
            v_dur = _to_pos_float(v_stream.get("duration")) if v_stream else 0.0
            a_dur = _to_pos_float(a_stream.get("duration")) if a_stream else 0.0
            duration = fmt_dur or max(v_dur, a_dur)

            width = int(v_stream.get("width") or 0) if v_stream else 0
            height = int(v_stream.get("height") or 0) if v_stream else 0
            fps = 0.0
            if v_stream:
                fps = (
                    _parse_rate(v_stream.get("avg_frame_rate"))
                    or _parse_rate(v_stream.get("r_frame_rate"))
                )
            audio_sr = 0
            if a_stream:
                try:
                    audio_sr = int(a_stream.get("sample_rate") or 0)
                except (TypeError, ValueError):
                    audio_sr = 0

            has_video = bool(v_stream and width > 0 and height > 0)
            has_audio = bool(a_stream)
            valid = bool(duration > 0.0 and (has_video or has_audio))
            return {
                "valid": valid,
                "exists": True,
                "size_bytes": size_bytes,
                "duration": round(duration, 4),
                "width": width,
                "height": height,
                "fps": round(fps, 4) if fps else 0.0,
                "video_codec": v_stream.get("codec_name") if v_stream else None,
                "audio_codec": a_stream.get("codec_name") if a_stream else None,
                "has_video": has_video,
                "has_audio": has_audio,
                "pix_fmt": v_stream.get("pix_fmt") if v_stream else None,
                "audio_sample_rate": audio_sr,
                "video_duration": round(v_dur, 4),
                "audio_duration": round(a_dur, 4),
                "error": None if valid else "no playable streams or zero duration",
            }
        ffprobe_err = (proc.stderr or "").strip()[-240:] or f"ffprobe exit code {proc.returncode}"
    except FileNotFoundError:
        ffprobe_err = "ffprobe not found"
    except subprocess.TimeoutExpired:
        return _empty_media_info(file_path, error=f"ffprobe timed out after {timeout}s")
    except Exception as e:
        ffprobe_err = str(e)

    # Fallback: parse `ffmpeg -hide_banner -i` only if ffprobe wasn't usable
    ff_exe = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    try:
        res = subprocess.run(
            [ff_exe, "-hide_banner", "-i", file_path],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        stderr = res.stderr or ""
        dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", stderr)
        duration = 0.0
        if dur_match:
            duration = (
                float(dur_match.group(1)) * 3600.0
                + float(dur_match.group(2)) * 60.0
                + float(dur_match.group(3))
            )
        video_line = next((line for line in stderr.splitlines()
                           if "Stream #" in line and "Video:" in line and "(attached pic)" not in line), "")
        res_match = re.search(r"Video:\s*([a-zA-Z0-9_]+).*?,\s*(\d+)x(\d+)", video_line)
        width, height, vcodec = 0, 0, None
        if res_match:
            vcodec = res_match.group(1).lower()
            width = int(res_match.group(2))
            height = int(res_match.group(3))
        fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", video_line)
        fps = float(fps_match.group(1)) if fps_match else 0.0
        audio_match = re.search(r"Stream.*Audio:\s*([a-zA-Z0-9_]+)", stderr)
        acodec = audio_match.group(1).lower() if audio_match else None
        has_video = bool(width > 0 and height > 0)
        has_audio = bool(acodec)
        valid = bool(duration > 0.0 and (has_video or has_audio))
        return {
            "valid": valid,
            "exists": True,
            "size_bytes": size_bytes,
            "duration": round(duration, 4),
            "width": width,
            "height": height,
            "fps": round(fps, 4) if fps else 0.0,
            "video_codec": vcodec,
            "audio_codec": acodec,
            "has_video": has_video,
            "has_audio": has_audio,
            "pix_fmt": None,
            "audio_sample_rate": 0,
            "video_duration": 0.0,
            "audio_duration": 0.0,
            "error": None if valid else (ffprobe_err or "unreadable media"),
        }
    except FileNotFoundError:
        # Both ffprobe and ffmpeg CLI binaries are unavailable (e.g. minimal CI):
        # fall back to real media decoding via OpenCV if installed. Never fabricates
        # metadata — requires actually opening the container and decoding frame 0.
        try:
            import cv2  # type: ignore

            cap = cv2.VideoCapture(file_path)
            try:
                if cap.isOpened():
                    ok_frame, frame = cap.read()
                    if not ok_frame or frame is None:
                        return _empty_media_info(file_path, error="OpenCV could not decode a video frame")
                    h, w = frame.shape[:2]
                    raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    fps = raw_fps if (math.isfinite(raw_fps) and 0.1 <= raw_fps <= 240.0) else 0.0
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if math.isfinite(frame_count) and frame_count > 0 and fps > 0 else 0.0
                    if w > 0 and h > 0 and duration > 0.0:
                        return {
                            "valid": True,
                            "exists": True,
                            "size_bytes": size_bytes,
                            "duration": round(duration, 4),
                            "width": w,
                            "height": h,
                            "fps": round(fps, 4),
                            "video_codec": None,
                            "audio_codec": None,
                            "has_video": True,
                            "has_audio": False,
                            "pix_fmt": None,
                            "audio_sample_rate": 0,
                            "video_duration": round(duration, 4),
                            "audio_duration": 0.0,
                            "error": None,
                        }
            finally:
                cap.release()
        except Exception as e:
            return _empty_media_info(file_path, error=f"Video probe binaries unavailable and OpenCV probe failed: {e}")
        return _empty_media_info(file_path, error="Video probe binaries unavailable and OpenCV found no timed video")
    except Exception as e:
        return _empty_media_info(file_path, error=ffprobe_err or str(e))


def validate_media_output(
    path,
    *,
    expected_duration=None,
    duration_tolerance=2.0,
    min_duration=0.1,
    max_duration=None,
    require_video=True,
    require_audio=False,
    expected_resolution=None,
    expected_aspect_ratio=None,
    aspect_tolerance=0.05,
    allowed_video_codecs=None,
    allowed_audio_codecs=None,
    max_av_desync=1.5,
    decode_check=False,
    ffmpeg_bin=None,
    ffprobe_bin=None,
    timeout=60,
):
    """Validate that an exported media file is genuinely playable and meets constraints.

    Returns ``(ok: bool, info: dict, reason: str)``.
    """
    info = probe_media(path, ffprobe_bin=ffprobe_bin, ffmpeg_bin=ffmpeg_bin, timeout=min(timeout, 30))
    if not info["exists"]:
        return False, info, f"output file missing: {path}"
    if info["size_bytes"] <= 0:
        return False, info, "output file is zero bytes"
    if not info["valid"]:
        return False, info, f"invalid or unreadable media: {info.get('error') or 'probe failed'}"

    if require_video:
        if not info["has_video"] or info["width"] <= 0 or info["height"] <= 0:
            return False, info, "missing valid video stream"
        if info["fps"] <= 0:
            return False, info, "invalid or missing video frame rate"

    if require_audio and not info["has_audio"]:
        return False, info, "missing audio stream"

    dur = info["duration"]
    if min_duration is not None and dur < float(min_duration):
        return False, info, f"duration {dur:.2f}s below minimum {float(min_duration):.2f}s"
    if max_duration is not None and dur > float(max_duration):
        return False, info, f"duration {dur:.2f}s exceeds maximum {float(max_duration):.2f}s"
    if expected_duration is not None:
        diff = abs(dur - float(expected_duration))
        if diff > float(duration_tolerance):
            return False, info, (
                f"duration mismatch: got {dur:.2f}s, expected {float(expected_duration):.2f}s "
                f"(tolerance ±{float(duration_tolerance):.2f}s)"
            )

    if expected_resolution is not None:
        exp_w, exp_h = int(expected_resolution[0]), int(expected_resolution[1])
        if info["width"] != exp_w or info["height"] != exp_h:
            return False, info, (
                f"resolution mismatch: got {info['width']}x{info['height']}, "
                f"expected {exp_w}x{exp_h}"
            )

    if expected_aspect_ratio is not None and info["height"] > 0:
        actual_ar = info["width"] / float(info["height"])
        if abs(actual_ar - float(expected_aspect_ratio)) > float(aspect_tolerance):
            return False, info, (
                f"aspect ratio mismatch: got {actual_ar:.3f} ({info['width']}x{info['height']}), "
                f"expected {float(expected_aspect_ratio):.3f}"
            )

    if allowed_video_codecs and info["has_video"]:
        allowed_v = {c.lower() for c in allowed_video_codecs}
        if (info["video_codec"] or "").lower() not in allowed_v:
            return False, info, (
                f"unexpected video codec {info['video_codec']!r}; allowed: {sorted(allowed_v)}"
            )

    if allowed_audio_codecs and info["has_audio"]:
        allowed_a = {c.lower() for c in allowed_audio_codecs}
        if (info["audio_codec"] or "").lower() not in allowed_a:
            return False, info, (
                f"unexpected audio codec {info['audio_codec']!r}; allowed: {sorted(allowed_a)}"
            )

    if (
        max_av_desync is not None
        and info["video_duration"] > 0
        and info["audio_duration"] > 0
    ):
        desync = abs(info["video_duration"] - info["audio_duration"])
        if desync > float(max_av_desync):
            return False, info, (
                f"audio/video duration desync {desync:.2f}s "
                f"(video={info['video_duration']:.2f}s, audio={info['audio_duration']:.2f}s)"
            )

    if decode_check:
        ff_exe = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
        maps = []
        if info["has_video"]:
            maps.extend(["-map", "0:V:0"])
        if info["has_audio"]:
            maps.extend(["-map", "0:a:0"])
        # Check both ends of an export without decoding the entire video again.
        samples = [(0, 10)]
        if dur > 11:
            samples.append((dur - 2, 2))
        try:
            for start, length in samples:
                dec = subprocess.run(
                    [ff_exe, "-nostdin", "-v", "error", "-xerror", "-ss", str(start),
                     "-i", str(path), *maps, "-t", str(length), "-abort_on", "empty_output_stream",
                     "-f", "null", "-"],
                    capture_output=True, text=True, errors="replace", timeout=timeout,
                )
                if dec.returncode != 0 or (dec.stderr and dec.stderr.strip()):
                    err_tail = (dec.stderr or "").strip()[-240:]
                    return False, info, f"media decode error at {start:.2f}s (rc={dec.returncode}): {err_tail}"
        except FileNotFoundError:
            if info["has_audio"] or require_audio:
                return False, info, "media decode verification requires FFmpeg for audio"
            try:
                import cv2  # type: ignore

                cap = cv2.VideoCapture(str(path))
                try:
                    ok_f, frame = cap.read() if cap.isOpened() else (False, None)
                finally:
                    cap.release()
                if not ok_f or frame is None:
                    return False, info, "media decode verification failed: OpenCV could not decode frame 0"
            except Exception as e:
                return False, info, f"media decode verification failed (no ffmpeg): {e}"
        except subprocess.TimeoutExpired:
            return False, info, f"media decode verification timed out after {timeout}s"
        except Exception as e:
            return False, info, f"media decode verification failed: {e}"

    return True, info, "ok"


def check_black_frames(path, duration=None, max_black_ratio=0.95, ffmpeg_bin=None, timeout=60):
    """Detect whether a video is unexpectedly almost entirely black frames.

    Returns ``(ok: bool, black_ratio: float)`` where ``ok=False`` indicates
    black frames exceed ``max_black_ratio`` or analysis failed. The ratio is
    ``None`` when the analysis could not run.
    """
    ff_exe = ffmpeg_bin or shutil.which("ffmpeg") or "ffmpeg"
    if duration is None:
        duration = probe_media(path, ffmpeg_bin=ff_exe).get("duration", 0.0)
    if duration <= 0:
        return False, 1.0
    try:
        res = subprocess.run(
            [
                ff_exe, "-v", "info", "-i", str(path),
                "-vf", "blackdetect=d=0.5:pix_th=0.10",
                "-an", "-f", "null", "-",
            ],
            capture_output=True, text=True, errors="replace", timeout=timeout,
        )
        if res.returncode != 0:
            return False, None
        black_total = 0.0
        for m in re.finditer(r"black_duration:\s*([0-9.]+)", res.stderr or ""):
            black_total += float(m.group(1))
        ratio = min(1.0, black_total / float(duration))
        return ratio <= float(max_black_ratio), round(ratio, 4)
    except (OSError, subprocess.TimeoutExpired):
        return False, None


def _swap_nvenc_to_x264(cmd, tier=QUALITY):
    """Replace h264_nvenc codec args in an ffmpeg argv with CPU libx264 args."""
    if "h264_nvenc" not in cmd:
        return None
    nvenc_flags_with_val = {
        "-preset", "-tune", "-rc", "-cq", "-b:v", "-spatial-aq", "-temporal-aq",
    }
    out = []
    i = 0
    replaced = False
    while i < len(cmd):
        tok = cmd[i]
        if tok == "-c:v" and i + 1 < len(cmd) and cmd[i + 1] == "h264_nvenc":
            out.extend(_X264_ARGS.get(tier, _X264_ARGS[QUALITY]))
            replaced = True
            i += 2
            continue
        if replaced and tok in nvenc_flags_with_val and i + 1 < len(cmd):
            i += 2
            continue
        out.append(tok)
        i += 1
    return out if replaced else None


def run_ffmpeg_with_fallback(cmd, *, timeout=1800, tier=QUALITY, check=False):
    """Run an ffmpeg command, safely falling back from h264_nvenc to libx264 on GPU failure."""
    global _nvenc_ok
    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout)
    if res.returncode != 0 and "h264_nvenc" in cmd:
        swapped = _swap_nvenc_to_x264(cmd, tier=tier)
        if swapped:
            err_tail = (res.stderr or b"").decode("utf-8", errors="replace")[-200:]
            print(f"⚠️ [Encoder] h264_nvenc encode failed ({err_tail}) — retrying with libx264")
            with _probe_lock:
                _nvenc_ok = False
            res = subprocess.run(
                swapped, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout
            )
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)
    return res
