"""Voiceover and Text-to-Speech (TTS) Synthesis Engine.

Provides dual-engine synthesis:
1. Microsoft Edge Neural TTS (edge-tts) for lifelike multi-voice narration.
2. Windows Native SAPI SpVoice (via pywin32) for zero-latency, offline Windows execution.
3. User-supplied pre-recorded voiceover audio.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple

from anti_reused_pipeline.config import AudioConfig, TTSProvider

logger = logging.getLogger("anti_reused_pipeline.tts")


@dataclass
class VoiceoverResult:
    """Result of voiceover generation."""
    audio_path: str
    duration: float
    transcript_text: str
    srt_path: Optional[str] = None
    provider_used: str = "none"


def synthesize_windows_sapi(text: str, output_wav: str) -> float:
    """Generate speech using Windows native SAPI SpVoice.

    Args:
        text: Text to speak.
        output_wav: Path to target WAV file.

    Returns:
        Duration of generated WAV in seconds.
    """
    import win32com.client

    # 3 = SSFMCreateForWrite (Audio stream mode)
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    speaker = win32com.client.Dispatch("SAPI.SpVoice")

    # Set rate (speed) between -10 and 10; default 0
    speaker.Rate = 0
    speaker.Volume = 100

    stream.Open(output_wav, 3, False)
    speaker.AudioOutputStream = stream
    speaker.Speak(text)
    stream.Close()

    # Probe duration via ffprobe
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", output_wav
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = float(res.stdout.strip()) if res.stdout.strip() else 0.0
    return max(0.1, dur)


async def _synthesize_edge_tts_async(text: str, voice: str, output_mp3: str) -> None:
    """Internal coroutine for edge-tts Communicate."""
    import edge_tts
    comm = edge_tts.Communicate(text, voice)
    await comm.save(output_mp3)


def synthesize_edge_tts(text: str, voice: str, output_mp3: str) -> float:
    """Generate neural speech using edge-tts.

    Args:
        text: Text to synthesize.
        voice: Neural voice name (e.g. en-US-ChristopherNeural).
        output_mp3: Path to target MP3 file.

    Returns:
        Duration of generated MP3 in seconds.
    """
    asyncio.run(_synthesize_edge_tts_async(text, voice, output_mp3))

    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", output_mp3
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dur = float(res.stdout.strip()) if res.stdout.strip() else 0.0
    return max(0.1, dur)


def build_timing_srt(text: str, total_duration: float, srt_output_path: str) -> str:
    """Generate an evenly distributed, sentence/phrase-level SRT file.

    Args:
        text: Full transcript text.
        total_duration: Audio duration in seconds.
        srt_output_path: Output SRT filepath.

    Returns:
        Path to written SRT file.
    """
    # Split text into bite-sized phrases (4-8 words per subtitle card)
    words = text.strip().split()
    if not words:
        words = ["..."]

    chunk_size = 5
    phrases = []
    for i in range(0, len(words), chunk_size):
        phrases.append(" ".join(words[i:i + chunk_size]))

    num_phrases = len(phrases)
    time_per_phrase = total_duration / max(1, num_phrases)

    srt_entries = []
    for idx, phrase in enumerate(phrases):
        start_sec = idx * time_per_phrase
        end_sec = min((idx + 1) * time_per_phrase, total_duration)

        def fmt_time(sec: float) -> str:
            hrs = int(sec // 3600)
            mins = int((sec % 3600) // 60)
            s = int(sec % 60)
            ms = int((sec - int(sec)) * 1000)
            return f"{hrs:02d}:{mins:02d}:{s:02d},{ms:03d}"

        srt_entries.append(
            f"{idx + 1}\n{fmt_time(start_sec)} --> {fmt_time(end_sec)}\n{phrase.upper()}\n"
        )

    with open(srt_output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries) + "\n")

    return srt_output_path


def generate_voiceover(
    config: AudioConfig,
    work_dir: str,
    target_duration: Optional[float] = None
) -> VoiceoverResult:
    """Produce voiceover audio and synchronized subtitle timing.

    Args:
        config: AudioConfig with provider, voice, text, or custom path.
        work_dir: Temp directory for generated assets.
        target_duration: Video duration target for length alignment.

    Returns:
        VoiceoverResult containing audio path, duration, and SRT path.
    """
    os.makedirs(work_dir, exist_ok=True)

    # Case 1: Custom audio provided
    if config.custom_voiceover_path and os.path.isfile(config.custom_voiceover_path):
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", config.custom_voiceover_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = float(res.stdout.strip()) if res.stdout.strip() else 1.0
        text = config.voiceover_text or "Original Voiceover Commentary"
        srt_file = os.path.join(work_dir, "voiceover_subs.srt")
        build_timing_srt(text, dur, srt_file)
        return VoiceoverResult(
            audio_path=os.path.abspath(config.custom_voiceover_path),
            duration=dur,
            transcript_text=text,
            srt_path=srt_file,
            provider_used="custom_audio"
        )

    text = config.voiceover_text or "This video breaks down key analysis and transformative commentary on the source material."
    srt_file = os.path.join(work_dir, "voiceover_subs.srt")

    # Case 2: Auto or Edge-TTS requested
    if config.voiceover_provider in (TTSProvider.AUTO, TTSProvider.EDGE_TTS):
        mp3_out = os.path.join(work_dir, "vo_edge.mp3")
        try:
            logger.info("Attempting edge-tts synthesis: voice=%s", config.voice_name)
            dur = synthesize_edge_tts(text, config.voice_name, mp3_out)
            build_timing_srt(text, dur, srt_file)
            return VoiceoverResult(
                audio_path=mp3_out,
                duration=dur,
                transcript_text=text,
                srt_path=srt_file,
                provider_used="edge_tts"
            )
        except Exception as exc:
            if config.voiceover_provider == TTSProvider.EDGE_TTS:
                raise RuntimeError(f"Edge-TTS failed: {exc}") from exc
            logger.warning("Edge-TTS unavailable (%s). Falling back to Windows Native SAPI.", exc)

    # Case 3: Windows SAPI (Native / Offline fallback)
    wav_out = os.path.join(work_dir, "vo_sapi.wav")
    try:
        dur = synthesize_windows_sapi(text, wav_out)
        build_timing_srt(text, dur, srt_file)
        return VoiceoverResult(
            audio_path=wav_out,
            duration=dur,
            transcript_text=text,
            srt_path=srt_file,
            provider_used="windows_sapi"
        )
    except Exception as exc:
        raise RuntimeError(f"Windows SAPI voiceover synthesis failed: {exc}") from exc
