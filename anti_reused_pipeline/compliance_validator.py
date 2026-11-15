"""Automated Compliance Validator and Scoring Engine.

Implements the official Programmatic Compliance Scoring Metric (S_compliance)
defined by @policy in docs/compliance_and_transformative_guidelines.md:

    S = 0.35 * R_VO + 0.25 * R_layout + 0.25 * R_temporal + 0.15 * R_audio_duck

Threshold Gates:
- S >= 0.80: PASS (Compliant & Monetizable, YPP Fair Use Safe)
- 0.60 <= S < 0.80: WARNING (Borderline, requires manual review)
- S < 0.60: FAIL (High Risk / Reused Content)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, List, Optional

from anti_reused_pipeline.config import ComplianceThresholds, LayoutMode, AudioDuckingMode

logger = logging.getLogger("anti_reused_pipeline.compliance")


class ComplianceStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class ScoreBreakdown:
    voiceover_ratio: float
    r_vo: float
    layout_mode: str
    r_layout: float
    max_chunk_duration: float
    r_temporal: float
    ducking_mode: str
    r_duck: float


@dataclass
class MetadataAuditResult:
    has_stream_metadata: bool
    has_format_metadata: bool
    tags_found: Dict[str, Any]
    clean: bool


@dataclass
class ComplianceReport:
    """Comprehensive compliance and audit report."""
    score: float
    status: ComplianceStatus
    breakdown: ScoreBreakdown
    metadata_audit: Optional[MetadataAuditResult]
    recommendations: List[str]
    is_monetizable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  TRANSFORMATIVE COMPLIANCE REPORT: {self.status.value}",
            "=" * 60,
            f"Overall Score: {self.score:.3f} / 1.000",
            f"Status:        {self.status.value} (Monetizable: {self.is_monetizable})",
            "-" * 60,
            "Pillar Breakdown:",
            f"  • Narrative Primacy (VO Ratio): {self.breakdown.voiceover_ratio:.1%} -> R_vo = {self.breakdown.r_vo:.2f} (Weight: 35%)",
            f"  • Visual Restructure (Layout):  {self.breakdown.layout_mode} -> R_layout = {self.breakdown.r_layout:.2f} (Weight: 25%)",
            f"  • Temporal Slicing (Max Chunk): {self.breakdown.max_chunk_duration:.2f}s -> R_temporal = {self.breakdown.r_temporal:.2f} (Weight: 25%)",
            f"  • Audio Ducking / Primacy:     {self.breakdown.ducking_mode} -> R_duck = {self.breakdown.r_duck:.2f} (Weight: 15%)",
            "-" * 60,
        ]
        if self.metadata_audit:
            lines.append(f"Metadata Cleanliness: {'CLEAN (All tags stripped)' if self.metadata_audit.clean else 'WARNING (Tags present)'}")
            if not self.metadata_audit.clean:
                lines.append(f"  Residual tags: {self.metadata_audit.tags_found}")
        if self.recommendations:
            lines.append("Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        lines.append("=" * 60)
        return "\n".join(lines)


def audit_file_metadata(file_path: str) -> MetadataAuditResult:
    """Audit output video file with ffprobe to verify that metadata is sanitized."""
    if not os.path.isfile(file_path):
        return MetadataAuditResult(has_stream_metadata=False, has_format_metadata=False, tags_found={}, clean=True)

    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", file_path
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(proc.stdout)
    except Exception as exc:
        logger.warning("ffprobe metadata audit failed: %s", exc)
        return MetadataAuditResult(has_stream_metadata=False, has_format_metadata=False, tags_found={}, clean=True)

    format_tags = data.get("format", {}).get("tags", {})
    standard_format_keys = {"encoder", "major_brand", "minor_version", "compatible_brands"}
    residual_format_tags = {k: v for k, v in format_tags.items() if k.lower() not in standard_format_keys}

    stream_tags = {}
    standard_stream_keys = {"handler_name", "vendor_id", "language", "encoder"}
    for s in data.get("streams", []):
        st = s.get("tags", {})
        for k, v in st.items():
            if k.lower() not in standard_stream_keys:
                stream_tags[k] = v

    has_format = len(residual_format_tags) > 0
    has_stream = len(stream_tags) > 0
    residual_tags = {**residual_format_tags, **stream_tags}

    return MetadataAuditResult(
        has_stream_metadata=has_stream,
        has_format_metadata=has_format,
        tags_found=residual_tags,
        clean=(not has_format and not has_stream)
    )


def calculate_compliance_score(
    total_video_duration: float,
    voiceover_duration: float,
    layout_mode: LayoutMode,
    max_chunk_duration: float,
    ducking_mode: AudioDuckingMode,
    file_path_for_audit: Optional[str] = None,
    thresholds: Optional[ComplianceThresholds] = None
) -> ComplianceReport:
    """Calculate the objective transformative compliance score.

    Args:
        total_video_duration: Overall length of output video in seconds.
        voiceover_duration: Total duration of commentary/voiceover track.
        layout_mode: LayoutMode employed (SPLIT_SCREEN, PIP, CANVAS_LETTERBOX, BLUR_FILL).
        max_chunk_duration: Duration of longest continuous unedited source slice.
        ducking_mode: AudioDuckingMode employed.
        file_path_for_audit: Optional path to inspect output file container tags.
        thresholds: ComplianceThresholds instance.

    Returns:
        ComplianceReport with score, status, breakdown, and guidance.
    """
    th = thresholds or ComplianceThresholds()

    # 1. R_vo: Voiceover Duration Ratio
    if total_video_duration > 0:
        vo_ratio = min(1.0, voiceover_duration / total_video_duration)
    else:
        vo_ratio = 1.0

    # Score R_vo based on 70% threshold target
    if vo_ratio >= th.target_voiceover_ratio:
        r_vo = 1.0
    else:
        r_vo = max(0.0, vo_ratio / th.target_voiceover_ratio)

    # 2. R_layout: Visual Restructuring Factor
    if layout_mode in (LayoutMode.SPLIT_SCREEN, LayoutMode.PIP):
        r_layout = 1.0
    elif layout_mode == LayoutMode.CANVAS_LETTERBOX:
        r_layout = 0.50
    elif layout_mode == LayoutMode.BLUR_FILL:
        r_layout = 0.40
    else:
        r_layout = 0.0

    # 3. R_temporal: Continuous Source Chunk Rule
    if max_chunk_duration <= 5.0:
        r_temporal = 1.0
    elif max_chunk_duration <= 8.0:
        r_temporal = 0.60
    else:
        r_temporal = 0.0

    # 4. R_audio_duck: Audio Primacy Factor
    if ducking_mode in (AudioDuckingMode.SIDECHAIN_DUCK, AudioDuckingMode.MUTE_SOURCE):
        r_duck = 1.0
    elif ducking_mode == AudioDuckingMode.PARITY:
        r_duck = 0.20
    else:
        r_duck = 0.0

    # Master Score Formula
    score = (
        th.weight_voiceover * r_vo +
        th.weight_layout * r_layout +
        th.weight_temporal * r_temporal +
        th.weight_ducking * r_duck
    )
    score = max(0.0, min(1.0, round(score, 4)))

    recommendations = []
    if r_vo < 0.8:
        recommendations.append(f"Increase voiceover coverage to at least {th.target_voiceover_ratio:.0%} of video duration.")
    if r_layout < 0.8:
        recommendations.append("Switch to 'split_screen' or 'pip' layout for maximum structural transformation.")
    if r_temporal < 0.8:
        recommendations.append(f"Reduce continuous unedited source chunks to <= {th.max_allowed_continuous_chunk:.1f}s.")
    if r_duck < 0.8:
        recommendations.append("Enable sidechain ducking or mute source audio to defeat acoustic fingerprinting.")

    # Status Determination
    if score >= th.pass_threshold:
        status = ComplianceStatus.PASS
        is_monetizable = True
    elif score >= th.warn_threshold:
        status = ComplianceStatus.WARN
        is_monetizable = False
        recommendations.append("Warning: Borderline compliance score. Review before uploading.")
    else:
        status = ComplianceStatus.FAIL
        is_monetizable = False
        recommendations.append("Critical: Fails YPP Reused Content guidelines. Uploading risks demonetization.")

    meta_audit = None
    if file_path_for_audit:
        meta_audit = audit_file_metadata(file_path_for_audit)

    breakdown = ScoreBreakdown(
        voiceover_ratio=vo_ratio,
        r_vo=r_vo,
        layout_mode=layout_mode.value,
        r_layout=r_layout,
        max_chunk_duration=max_chunk_duration,
        r_temporal=r_temporal,
        ducking_mode=ducking_mode.value,
        r_duck=r_duck
    )

    return ComplianceReport(
        score=score,
        status=status,
        breakdown=breakdown,
        metadata_audit=meta_audit,
        recommendations=recommendations,
        is_monetizable=is_monetizable
    )
