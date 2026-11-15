"""Master Pipeline Orchestrator for Transformative Video Automation.

Coordinates:
1. Ingestion & media stream probing
2. Temporal slicing (enforcing <= 5.0s continuous chunk limit)
3. Voiceover synthesis (Edge-TTS or Windows SAPI) & dynamic captions
4. Audio sidechain ducking, BGM layering, and loudness normalization
5. 9:16 vertical visual restructuring (Split-Screen, PIP, Canvas Letterbox, Blur-Fill)
6. Anti-perceptual hash micro-adjustments (color grading, unsharp, noise, vignette)
7. Container and stream metadata sanitization
8. Automated compliance validation (S_compliance score)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from anti_reused_pipeline.config import PipelineConfig, LayoutMode, AudioDuckingMode
from anti_reused_pipeline.ingest import resolve_media_input, probe_media, MediaStreamInfo
from anti_reused_pipeline.temporal_slicer import execute_temporal_slice, SlicingResult
from anti_reused_pipeline.tts_engine import generate_voiceover, VoiceoverResult
from anti_reused_pipeline.audio_transform import build_audio_filtergraph
from anti_reused_pipeline.visual_layouts import build_visual_filtergraph
from anti_reused_pipeline.captions import generate_styled_ass_from_srt, build_subtitle_filter_string, escape_ffmpeg_path
from anti_reused_pipeline.compliance_validator import calculate_compliance_score, ComplianceReport

logger = logging.getLogger("anti_reused_pipeline.engine")


@dataclass
class PipelineResult:
    """Execution output from master transformation pipeline."""
    output_path: str
    duration: float
    compliance: ComplianceReport
    processing_time_sec: float
    slicing_result: Optional[SlicingResult]
    voiceover_result: Optional[VoiceoverResult]
    is_success: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_path": self.output_path,
            "duration": self.duration,
            "compliance": self.compliance.to_dict(),
            "processing_time_sec": self.processing_time_sec,
            "is_success": self.is_success,
            "error_message": self.error_message
        }


class TransformPipeline:
    """Master pipeline runner."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._work_dir = config.work_dir or tempfile.mkdtemp(prefix="trans_pipe_")
        os.makedirs(self._work_dir, exist_ok=True)

    def execute(self) -> PipelineResult:
        """Run the end-to-end transformation pipeline.

        Returns:
            PipelineResult containing output metadata, metrics, and compliance report.
        """
        start_time = time.time()
        logger.info("Starting TransformPipeline execution for %s", self.config.input_path)

        try:
            # Step 1: Ingest & Probe
            source_info = resolve_media_input(self.config.input_path, self._work_dir)
            logger.info("Ingested media: %.2fs, %dx%d, %s",
                        source_info.duration, source_info.width, source_info.height, source_info.format_name)

            # Step 2: Temporal Slicing (enforce <= 5.0s unedited chunk rule)
            if self.config.temporal.enable_slicing:
                slice_res = execute_temporal_slice(source_info, self.config.temporal, self._work_dir)
                curated_video_path = slice_res.sliced_media_path
                curated_info = probe_media(curated_video_path)
            else:
                slice_res = SlicingResult(
                    sliced_media_path=source_info.path,
                    total_duration=source_info.duration,
                    max_chunk_duration=source_info.duration,
                    chunk_count=1,
                    chunks=[(0.0, source_info.duration)]
                )
                curated_video_path = source_info.path
                curated_info = source_info

            # Step 3: Voiceover & Dynamic Captions Synthesis
            vo_res = generate_voiceover(
                self.config.audio,
                self._work_dir,
                target_duration=curated_info.duration
            )
            logger.info("Voiceover ready: %.2fs (Provider: %s)", vo_res.duration, vo_res.provider_used)

            # Subtitle preparation (SRT to styled ASS)
            ass_path = None
            if self.config.captions.burn_captions and vo_res.srt_path:
                ass_path = os.path.join(self._work_dir, "captions_styled.ass")
                generate_styled_ass_from_srt(
                    srt_path=vo_res.srt_path,
                    ass_path=ass_path,
                    config=self.config.captions,
                    video_width=self.config.visual.output_width,
                    video_height=self.config.visual.output_height
                )

            # Step 4: Render Transformed Audio Mix (Sidechain Ducking + BGM + Loudnorm)
            intermediate_audio_path = os.path.join(self._work_dir, "mixed_audio.m4a")
            audio_spec = build_audio_filtergraph(
                config=self.config.audio,
                source_media_path=curated_video_path,
                voiceover_path=vo_res.audio_path,
                bgm_path=self.config.audio.bgm_path,
                has_source_audio=curated_info.has_audio
            )

            audio_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            for a_inp in audio_spec.audio_inputs:
                audio_cmd.extend(["-i", a_inp])
            audio_cmd.extend([
                "-filter_complex", audio_spec.filtergraph,
                "-map", audio_spec.output_label,
                "-c:a", "aac",
                "-b:a", "192k",
                intermediate_audio_path
            ])
            logger.info("Rendering transformed audio mix...")
            subprocess.run(audio_cmd, check=True)

            # Step 5: Render Visual Layout & Burn Captions
            # Build visual filtergraph
            visual_spec = build_visual_filtergraph(
                config=self.config.visual,
                source_media_path=curated_video_path,
                source_info=curated_info,
                secondary_video_path=self.config.visual.secondary_video_path
            )

            # Append subtitle burning filter to visual chain if requested
            video_fg = visual_spec.filtergraph
            if self.config.captions.burn_captions and ass_path and os.path.isfile(ass_path):
                sub_filter = build_subtitle_filter_string(ass_path, self.config.captions)
                # Replace output label with sub-burned output
                video_fg = video_fg.replace(
                    f"{visual_spec.output_label}",
                    f"[pre_sub];[pre_sub]{sub_filter}{visual_spec.output_label}"
                )

            # Destination directory check
            out_file = os.path.abspath(self.config.output_path)
            os.makedirs(os.path.dirname(out_file), exist_ok=True)

            # Master assembly command
            video_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
            for v_inp in visual_spec.video_inputs:
                video_cmd.extend(["-i", v_inp])

            # Include the pre-rendered audio mix
            video_cmd.extend(["-i", intermediate_audio_path])
            audio_stream_idx = len(visual_spec.video_inputs)

            video_cmd.extend([
                "-filter_complex", video_fg,
                "-map", visual_spec.output_label,
                "-map", f"{audio_stream_idx}:a",
                "-c:v", self.config.visual.encoder,
                "-preset", self.config.visual.preset,
                "-crf", str(self.config.visual.crf),
                "-pix_fmt", "yuv420p",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-color_range", "tv",
                "-r", str(self.config.visual.target_fps),
                "-c:a", "copy",
                "-shortest"
            ])

            # Metadata Sanitization
            if self.config.strip_metadata:
                video_cmd.extend([
                    "-map_metadata", "-1",
                    "-map_chapters", "-1",
                    "-map_metadata:s:v", "-1",
                    "-map_metadata:s:a", "-1",
                    "-fflags", "+bitexact",
                    "-flags:v", "+bitexact",
                    "-flags:a", "+bitexact"
                ])

            if self.config.fast_start:
                video_cmd.extend(["-movflags", "+faststart"])

            video_cmd.append(out_file)

            logger.info("Executing final video transcode to %s", out_file)
            subprocess.run(video_cmd, check=True)

            # Step 6: Programmatic Compliance Audit
            final_info = probe_media(out_file)
            compliance_rep = calculate_compliance_score(
                total_video_duration=final_info.duration,
                voiceover_duration=vo_res.duration,
                layout_mode=self.config.visual.layout_mode,
                max_chunk_duration=slice_res.max_chunk_duration,
                ducking_mode=self.config.audio.ducking_mode,
                file_path_for_audit=out_file,
                thresholds=self.config.compliance
            )

            # Enforce gate if configured
            if self.config.enforce_compliance_pass and not compliance_rep.is_monetizable:
                raise RuntimeError(
                    f"Compliance gate failed! Score: {compliance_rep.score:.2f} < 0.80. "
                    f"Reasons: {'; '.join(compliance_rep.recommendations)}"
                )

            total_sec = time.time() - start_time
            logger.info("Transformation finished successfully in %.2fs (Compliance Score: %.2f)",
                        total_sec, compliance_rep.score)

            return PipelineResult(
                output_path=out_file,
                duration=final_info.duration,
                compliance=compliance_rep,
                processing_time_sec=total_sec,
                slicing_result=slice_res,
                voiceover_result=vo_res,
                is_success=True
            )

        except Exception as exc:
            total_sec = time.time() - start_time
            logger.exception("Pipeline failed: %s", exc)
            return PipelineResult(
                output_path=self.config.output_path,
                duration=0.0,
                compliance=calculate_compliance_score(0, 0, self.config.visual.layout_mode, 99, self.config.audio.ducking_mode),
                processing_time_sec=total_sec,
                slicing_result=None,
                voiceover_result=None,
                is_success=False,
                error_message=str(exc)
            )

    def cleanup(self) -> None:
        """Remove intermediate scratch directory."""
        if self._work_dir and os.path.isdir(self._work_dir):
            try:
                shutil.rmtree(self._work_dir, ignore_errors=True)
            except OSError:
                pass
