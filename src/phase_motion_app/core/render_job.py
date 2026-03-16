"""This file defines the render request and codec plan types so the shell, worker, and finalization code share one concrete export contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.models import DiagnosticLevel, JobIntent


@dataclass(frozen=True)
class RenderPaths:
    """This model groups all filesystem paths for one render attempt."""

    source_path: Path
    output_directory: Path
    output_stem: str
    scratch_directory: Path
    diagnostics_directory: Path

    @property
    def staged_mp4_path(self) -> Path:
        return self.output_directory / f"{self.output_stem}.staged.mp4"

    @property
    def staged_sidecar_path(self) -> Path:
        return self.output_directory / f"{self.output_stem}.staged.json"

    @property
    def final_mp4_path(self) -> Path:
        return self.output_directory / f"{self.output_stem}.mp4"

    @property
    def final_sidecar_path(self) -> Path:
        return self.output_directory / f"{self.output_stem}.json"

    @property
    def failed_evidence_directory(self) -> Path:
        return self.diagnostics_directory / "failed-evidence"

    @property
    def jsonl_log_path(self) -> Path:
        return self.diagnostics_directory / "run.jsonl"


@dataclass(frozen=True)
class RenderRequest:
    """This model carries the authoritative input and output settings for one render attempt."""

    job_id: str
    intent: JobIntent
    paths: RenderPaths
    expected_source_fingerprint_sha256: str
    diagnostic_level: DiagnosticLevel
    diagnostics_cap_bytes: int
    retention_budget_bytes: int
    drift_assessment: DriftAssessment


@dataclass(frozen=True)
class CodecPlan:
    """This model records the chosen encoder path and what ffprobe should later validate on the staged MP4."""

    ffmpeg_codec: str
    expected_codec_name: str
    expected_profile: str
    pixel_format: str
    warnings: tuple[str, ...] = ()
    color_tags: dict[str, str] | None = None
