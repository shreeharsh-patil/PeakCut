"""Audio Transformation & Sidechain Ducking Filtergraph Engine.

Implements the Audio Primacy & Ducking Rule defined by policy:
- Sidechain ducking of source audio beneath commentary track (-20 dB to -26 dB)
- Complete source mute option for high-risk / music copyrighted sources
- Background music (BGM) mixing layer
- Strict EBU R128 / YouTube loudness normalization (-14 LUFS, -2.0 dBTP, LRA 11)
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from anti_reused_pipeline.config import AudioConfig, AudioDuckingMode

logger = logging.getLogger("anti_reused_pipeline.audio")


@dataclass
class AudioGraphSpecification:
    """FFmpeg audio filtergraph parameters and input stream mappings."""
    filtergraph: str
    output_label: str
    audio_inputs: List[str]  # Filepaths corresponding to audio input indices


def build_audio_filtergraph(
    config: AudioConfig,
    source_media_path: str,
    voiceover_path: Optional[str] = None,
    bgm_path: Optional[str] = None,
    source_audio_index: int = 0,
    has_source_audio: bool = True
) -> AudioGraphSpecification:
    """Construct an FFmpeg complex filtergraph for multi-track ducking and normalization.

    Args:
        config: AudioConfig specifying ducking mode, thresholds, and loudness targets.
        source_media_path: Main source video/audio file.
        voiceover_path: Primary commentary / TTS voiceover audio file.
        bgm_path: Optional background music track.
        source_audio_index: Input index of the source media (default 0).
        has_source_audio: Whether the source media actually contains an audio stream.

    Returns:
        AudioGraphSpecification with filtergraph string, output label, and input paths.
    """
    audio_inputs = [source_media_path]
    vo_idx = -1
    bgm_idx = -1

    if voiceover_path:
        audio_inputs.append(voiceover_path)
        vo_idx = len(audio_inputs) - 1

    effective_bgm = bgm_path or config.bgm_path
    if effective_bgm and os.path.isfile(effective_bgm):
        audio_inputs.append(effective_bgm)
        bgm_idx = len(audio_inputs) - 1

    # Micro pitch/tempo filter prefix if requested
    tempo_filter = ""
    if abs(config.tempo_factor - 1.0) > 0.001:
        tempo_filter = f",atempo={config.tempo_factor:.3f}"

    loudnorm_str = f"loudnorm=I={config.loudnorm_integrated}:TP={config.loudnorm_true_peak}:LRA={config.loudnorm_lra}"

    # Scenario 1: Source audio muted OR source lacks audio track -> Voiceover dominance
    if config.ducking_mode == AudioDuckingMode.MUTE_SOURCE or not has_source_audio:
        logger.info("Building MUTE_SOURCE audio filtergraph (voiceover + optional bgm)")
        if vo_idx >= 0 and bgm_idx >= 0:
            # Voiceover + BGM
            fg = (
                f"[{vo_idx}:a]volume={config.voiceover_volume}{tempo_filter}[vo];"
                f"[{bgm_idx}:a]volume={config.bgm_volume}[bgm];"
                f"[bgm][vo]amix=inputs=2:weights=0.15 1.0:duration=first[mixed];"
                f"[mixed]{loudnorm_str}[aout]"
            )
        elif vo_idx >= 0:
            # Voiceover only
            fg = (
                f"[{vo_idx}:a]volume={config.voiceover_volume}{tempo_filter},{loudnorm_str}[aout]"
            )
        elif bgm_idx >= 0:
            # BGM only
            fg = (
                f"[{bgm_idx}:a]volume={config.bgm_volume},{loudnorm_str}[aout]"
            )
        else:
            # Generate silent audio bed if nothing provided
            fg = f"aevalsrc=0:d=10,{loudnorm_str}[aout]"

        return AudioGraphSpecification(filtergraph=fg, output_label="[aout]", audio_inputs=audio_inputs)

    # Scenario 2: Source audio present AND Voiceover present
    if vo_idx >= 0:
        if config.ducking_mode == AudioDuckingMode.PARITY:
            logger.info("Building PARITY audio filtergraph (source mixed at parity with voiceover)")
            if bgm_idx >= 0:
                fg = (
                    f"[{vo_idx}:a]volume={config.voiceover_volume}{tempo_filter}[vo];"
                    f"[{bgm_idx}:a]volume={config.bgm_volume}[bgm];"
                    f"[{source_audio_index}:a][bgm]amix=inputs=2:weights=1.0 0.15:duration=first[backing];"
                    f"[backing][vo]amix=inputs=2:weights=1.0 1.0[mixed];"
                    f"[mixed]{loudnorm_str}[aout]"
                )
            else:
                fg = (
                    f"[{vo_idx}:a]volume={config.voiceover_volume}{tempo_filter}[vo];"
                    f"[{source_audio_index}:a][vo]amix=inputs=2:weights=1.0 1.0[mixed];"
                    f"[mixed]{loudnorm_str}[aout]"
                )
            return AudioGraphSpecification(filtergraph=fg, output_label="[aout]", audio_inputs=audio_inputs)

        # Default: Automated Sidechain Ducking
        logger.info("Building SIDECHAIN_DUCK audio filtergraph (source ducked beneath voiceover)")
        th = config.sidechain_threshold
        ra = config.sidechain_ratio
        at = config.sidechain_attack_ms
        rel = config.sidechain_release_ms
        src_wt = config.source_attenuation_weight

        if bgm_idx >= 0:
            # Source + VO + BGM
            fg = (
                f"[{vo_idx}:a]volume={config.voiceover_volume}{tempo_filter},asplit=2[sc][vo];"
                f"[{source_audio_index}:a][sc]sidechaincompress=threshold={th}:ratio={ra}:attack={at}:release={rel}[ducked];"
                f"[{bgm_idx}:a]volume={config.bgm_volume}[bgm];"
                f"[ducked][bgm]amix=inputs=2:weights={src_wt} 0.12:duration=first[backing];"
                f"[backing][vo]amix=inputs=2:weights=0.3 1.0[mixed];"
                f"[mixed]{loudnorm_str}[aout]"
            )
        else:
            # Source + VO
            fg = (
                f"[{vo_idx}:a]volume={config.voiceover_volume}{tempo_filter},asplit=2[sc][vo];"
                f"[{source_audio_index}:a][sc]sidechaincompress=threshold={th}:ratio={ra}:attack={at}:release={rel}[ducked];"
                f"[ducked][vo]amix=inputs=2:weights={src_wt} 1.0[mixed];"
                f"[mixed]{loudnorm_str}[aout]"
            )
        return AudioGraphSpecification(filtergraph=fg, output_label="[aout]", audio_inputs=audio_inputs)

    # Scenario 3: Parity / Source only with loudness normalization
    logger.info("Building standard loudness-normalized source audio filtergraph")
    fg = f"[{source_audio_index}:a]{loudnorm_str}[aout]"
    return AudioGraphSpecification(filtergraph=fg, output_label="[aout]", audio_inputs=audio_inputs)


def render_standalone_audio(
    config: AudioConfig,
    source_media_path: str,
    output_audio_path: str,
    voiceover_path: Optional[str] = None,
    bgm_path: Optional[str] = None
) -> str:
    """Render the transformed audio mix directly to an output audio file (e.g. M4A or MP3).

    Args:
        config: AudioConfig.
        source_media_path: Source file.
        output_audio_path: Target audio path.
        voiceover_path: Voiceover audio path.
        bgm_path: BGM audio path.

    Returns:
        Path to rendered audio file.
    """
    spec = build_audio_filtergraph(
        config=config,
        source_media_path=source_media_path,
        voiceover_path=voiceover_path,
        bgm_path=bgm_path
    )

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for inp in spec.audio_inputs:
        cmd.extend(["-i", inp])

    cmd.extend([
        "-filter_complex", spec.filtergraph,
        "-map", spec.output_label,
        "-c:a", "aac",
        "-b:a", "192k",
        output_audio_path
    ])

    subprocess.run(cmd, check=True)
    return output_audio_path
