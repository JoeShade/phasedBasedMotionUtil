"""This file defines the core data models that the UI, worker, and sidecar code share so they agree on settings, outputs, and reproducibility boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ResourcePolicy(str, Enum):
    """This enum mirrors the three operator-facing scheduler choices from the design document."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class DiagnosticLevel(str, Enum):
    """This enum keeps diagnostics levels explicit and stable across UI, logs, and metadata."""

    OFF = "off"
    BASIC = "basic"
    DETAILED = "detailed"
    TRACE = "trace"


class ZoneShape(str, Enum):
    """This enum enforces the v1 exclusion-zone geometry limits from the design doc."""

    RECTANGLE = "rectangle"
    CIRCLE = "circle"


class ZoneMode(str, Enum):
    """This enum marks whether one stored mask zone preserves or amplifies its covered region."""

    EXCLUDE = "exclude"
    INCLUDE = "include"


class AnalysisBandMode(str, Enum):
    """This enum keeps the quantitative-analysis band-selection modes stable across UI, worker, and sidecar serialization."""

    AUTO = "auto"
    MANUAL_SINGLE = "manual_single"
    MANUAL_MULTI = "manual_multi"


@dataclass(frozen=True)
class Resolution:
    """This small value object keeps width and height paired so downscale-only checks stay simple."""

    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Resolution":
        return cls(width=int(data["width"]), height=int(data["height"]))


@dataclass(frozen=True)
class ExclusionZone:
    """This model stores source-frame mask geometry. The historical name stays in place so old sidecars and tests remain compatible."""

    zone_id: str
    shape: ZoneShape
    x: float
    y: float
    mode: ZoneMode = ZoneMode.EXCLUDE
    width: float | None = None
    height: float | None = None
    radius: float | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "shape": self.shape.value,
            "mode": self.mode.value,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "radius": self.radius,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExclusionZone":
        return cls(
            zone_id=str(data["zone_id"]),
            shape=ZoneShape(data["shape"]),
            mode=ZoneMode(data.get("mode", ZoneMode.EXCLUDE.value)),
            x=float(data["x"]),
            y=float(data["y"]),
            width=None if data.get("width") is None else float(data["width"]),
            height=None if data.get("height") is None else float(data["height"]),
            radius=None if data.get("radius") is None else float(data["radius"]),
            label=data.get("label"),
        )


@dataclass(frozen=True)
class PhaseSettings:
    """This model captures the user intent that changes the signal-processing output."""

    magnification: float
    low_hz: float
    high_hz: float
    pyramid_type: str
    sigma: float
    attenuate_other_frequencies: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "magnification": self.magnification,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "pyramid_type": self.pyramid_type,
            "sigma": self.sigma,
            "attenuate_other_frequencies": self.attenuate_other_frequencies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhaseSettings":
        return cls(
            magnification=float(data["magnification"]),
            low_hz=float(data["low_hz"]),
            high_hz=float(data["high_hz"]),
            pyramid_type=str(data["pyramid_type"]),
            sigma=float(data["sigma"]),
            attenuate_other_frequencies=bool(data["attenuate_other_frequencies"]),
        )


@dataclass(frozen=True)
class AnalysisBand:
    """This model stores one explicit quantitative-analysis band so manual-band settings and generated heatmap bands share one shape."""

    band_id: str
    low_hz: float
    high_hz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisBand":
        return cls(
            band_id=str(data["band_id"]),
            low_hz=float(data["low_hz"]),
            high_hz=float(data["high_hz"]),
        )


@dataclass(frozen=True)
class AnalysisSettings:
    """This model captures the small curated set of quantitative-analysis controls from the NVH extension design."""

    enabled: bool = True
    roi: ExclusionZone | None = None
    minimum_cell_support_fraction: float = 0.35
    roi_quality_cutoff: float = 0.45
    low_confidence_threshold: float = 0.35
    auto_band_count: int = 5
    band_mode: AnalysisBandMode = AnalysisBandMode.AUTO
    manual_bands: tuple[AnalysisBand, ...] = field(default_factory=tuple)
    export_advanced_files: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "roi": None if self.roi is None else self.roi.to_dict(),
            "minimum_cell_support_fraction": self.minimum_cell_support_fraction,
            "roi_quality_cutoff": self.roi_quality_cutoff,
            "low_confidence_threshold": self.low_confidence_threshold,
            "auto_band_count": self.auto_band_count,
            "band_mode": self.band_mode.value,
            "manual_bands": [band.to_dict() for band in self.manual_bands],
            "export_advanced_files": self.export_advanced_files,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisSettings":
        return cls(
            enabled=bool(data.get("enabled", True)),
            roi=(
                None
                if data.get("roi") is None
                else ExclusionZone.from_dict(data["roi"])
            ),
            minimum_cell_support_fraction=float(
                data.get("minimum_cell_support_fraction", 0.35)
            ),
            roi_quality_cutoff=float(data.get("roi_quality_cutoff", 0.45)),
            low_confidence_threshold=float(data.get("low_confidence_threshold", 0.35)),
            auto_band_count=int(data.get("auto_band_count", 5)),
            band_mode=AnalysisBandMode(
                data.get("band_mode", AnalysisBandMode.AUTO.value)
            ),
            manual_bands=tuple(
                AnalysisBand.from_dict(band_data)
                for band_data in data.get("manual_bands", [])
            ),
            export_advanced_files=bool(data.get("export_advanced_files", True)),
        )


@dataclass(frozen=True)
class JobIntent:
    """This model holds only the reproducible settings that the user meant to run again in the future."""

    phase: PhaseSettings
    processing_resolution: Resolution
    output_resolution: Resolution
    resource_policy: ResourcePolicy
    exclusion_zones: tuple[ExclusionZone, ...] = field(default_factory=tuple)
    mask_feather_px: float = 4.0
    analysis: AnalysisSettings = field(default_factory=AnalysisSettings)
    output_container: str = "mp4"
    requested_output_codec: str = "prefer_hevc_main10"
    hardware_acceleration_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.to_dict(),
            "processing_resolution": self.processing_resolution.to_dict(),
            "output_resolution": self.output_resolution.to_dict(),
            "resource_policy": self.resource_policy.value,
            "exclusion_zones": [zone.to_dict() for zone in self.exclusion_zones],
            "mask_feather_px": self.mask_feather_px,
            "analysis": self.analysis.to_dict(),
            "output_container": self.output_container,
            "requested_output_codec": self.requested_output_codec,
            "hardware_acceleration_enabled": self.hardware_acceleration_enabled,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobIntent":
        return cls(
            phase=PhaseSettings.from_dict(data["phase"]),
            processing_resolution=Resolution.from_dict(data["processing_resolution"]),
            output_resolution=Resolution.from_dict(data["output_resolution"]),
            resource_policy=ResourcePolicy(data["resource_policy"]),
            exclusion_zones=tuple(
                ExclusionZone.from_dict(zone_data)
                for zone_data in data.get("exclusion_zones", [])
            ),
            mask_feather_px=float(data.get("mask_feather_px", 4.0)),
            analysis=AnalysisSettings.from_dict(data.get("analysis", {})),
            output_container=str(data.get("output_container", "mp4")),
            requested_output_codec=str(
                data.get("requested_output_codec", "prefer_hevc_main10")
            ),
            hardware_acceleration_enabled=bool(
                data.get("hardware_acceleration_enabled", False)
            ),
        )


@dataclass(frozen=True)
class ObservedEnvironment:
    """This model stores what the machine and scheduler actually did so reruns can review context without reusing it as intent."""

    app_version: str
    engine_version: str
    platform: str
    diagnostic_level: DiagnosticLevel
    diagnostics_cap_bytes: int
    temp_root: str | None = None
    ffmpeg_version: str | None = None
    ffprobe_version: str | None = None
    scheduler_clamp_threads: int | None = None
    effective_thread_limits: dict[str, int] = field(default_factory=dict)
    acceleration_backend: str | None = None
    acceleration_device_name: str | None = None
    hardware_acceleration_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "app_version": self.app_version,
            "engine_version": self.engine_version,
            "platform": self.platform,
            "diagnostic_level": self.diagnostic_level.value,
            "diagnostics_cap_bytes": self.diagnostics_cap_bytes,
            "temp_root": self.temp_root,
            "ffmpeg_version": self.ffmpeg_version,
            "ffprobe_version": self.ffprobe_version,
            "scheduler_clamp_threads": self.scheduler_clamp_threads,
            "effective_thread_limits": self.effective_thread_limits,
            "acceleration_backend": self.acceleration_backend,
            "acceleration_device_name": self.acceleration_device_name,
            "hardware_acceleration_active": self.hardware_acceleration_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservedEnvironment":
        return cls(
            app_version=str(data["app_version"]),
            engine_version=str(data["engine_version"]),
            platform=str(data["platform"]),
            diagnostic_level=DiagnosticLevel(data["diagnostic_level"]),
            diagnostics_cap_bytes=int(data["diagnostics_cap_bytes"]),
            temp_root=data.get("temp_root"),
            ffmpeg_version=data.get("ffmpeg_version"),
            ffprobe_version=data.get("ffprobe_version"),
            scheduler_clamp_threads=(
                None
                if data.get("scheduler_clamp_threads") is None
                else int(data["scheduler_clamp_threads"])
            ),
            effective_thread_limits={
                str(key): int(value)
                for key, value in data.get("effective_thread_limits", {}).items()
            },
            acceleration_backend=data.get("acceleration_backend"),
            acceleration_device_name=data.get("acceleration_device_name"),
            hardware_acceleration_active=bool(
                data.get("hardware_acceleration_active", False)
            ),
        )


@dataclass(frozen=True)
class SourceRecord:
    """This model captures the authoritative source identity that later renders and operator attestations refer back to."""

    path: str
    fingerprint_sha256: str
    size_bytes: int
    modified_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "fingerprint_sha256": self.fingerprint_sha256,
            "size_bytes": self.size_bytes,
            "modified_utc": self.modified_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRecord":
        return cls(
            path=str(data["path"]),
            fingerprint_sha256=str(data["fingerprint_sha256"]),
            size_bytes=int(data["size_bytes"]),
            modified_utc=str(data["modified_utc"]),
        )


@dataclass(frozen=True)
class DriftAcknowledgement:
    """This model records the operator attestation that drift was reviewed for one specific source state."""

    acknowledged: bool
    reviewed_source_fingerprint_sha256: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "acknowledged": self.acknowledged,
            "reviewed_source_fingerprint_sha256": self.reviewed_source_fingerprint_sha256,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DriftAcknowledgement":
        return cls(
            acknowledged=bool(data["acknowledged"]),
            reviewed_source_fingerprint_sha256=str(
                data["reviewed_source_fingerprint_sha256"]
            ),
            note=data.get("note"),
        )


@dataclass(frozen=True)
class PreflightSummary:
    """This model stores what pre-flight found so successful exports still carry the admission logic that justified the run."""

    source_fps: float
    source_is_cfr: bool
    nyquist_limit_hz: float
    working_fps: float | None = None
    working_source_resolution: Resolution | None = None
    normalization_steps: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)
    hardware_acceleration_requested: bool = False
    hardware_acceleration_active: bool = False
    acceleration_backend: str | None = None
    acceleration_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source_fps": self.source_fps,
            "source_is_cfr": self.source_is_cfr,
            "nyquist_limit_hz": self.nyquist_limit_hz,
            "warnings": list(self.warnings),
            "blockers": list(self.blockers),
            "hardware_acceleration_requested": self.hardware_acceleration_requested,
            "hardware_acceleration_active": self.hardware_acceleration_active,
        }
        if self.working_fps is not None:
            data["working_fps"] = self.working_fps
        if self.working_source_resolution is not None:
            data["working_source_resolution"] = self.working_source_resolution.to_dict()
        if self.normalization_steps:
            data["normalization_steps"] = list(self.normalization_steps)
        if self.acceleration_backend is not None:
            data["acceleration_backend"] = self.acceleration_backend
        if self.acceleration_status is not None:
            data["acceleration_status"] = self.acceleration_status
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreflightSummary":
        return cls(
            source_fps=float(data["source_fps"]),
            source_is_cfr=bool(data["source_is_cfr"]),
            working_fps=(
                None if data.get("working_fps") is None else float(data["working_fps"])
            ),
            working_source_resolution=(
                None
                if data.get("working_source_resolution") is None
                else Resolution.from_dict(data["working_source_resolution"])
            ),
            normalization_steps=tuple(
                str(item) for item in data.get("normalization_steps", [])
            ),
            nyquist_limit_hz=float(data["nyquist_limit_hz"]),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            blockers=tuple(str(item) for item in data.get("blockers", [])),
            hardware_acceleration_requested=bool(
                data.get("hardware_acceleration_requested", False)
            ),
            hardware_acceleration_active=bool(
                data.get("hardware_acceleration_active", False)
            ),
            acceleration_backend=data.get("acceleration_backend"),
            acceleration_status=data.get("acceleration_status"),
        )


@dataclass(frozen=True)
class JobResults:
    """This model holds observed run facts and output evidence that must never be silently reloaded as future intent."""

    render_timestamp_utc: str
    source: SourceRecord
    preflight: PreflightSummary
    warnings: tuple[str, ...] = field(default_factory=tuple)
    fallbacks: tuple[str, ...] = field(default_factory=tuple)
    artifact_paths: dict[str, str] = field(default_factory=dict)
    diagnostics_summary: dict[str, Any] = field(default_factory=dict)
    output_details: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    drift_acknowledgement: DriftAcknowledgement | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "render_timestamp_utc": self.render_timestamp_utc,
            "source": self.source.to_dict(),
            "preflight": self.preflight.to_dict(),
            "warnings": list(self.warnings),
            "fallbacks": list(self.fallbacks),
            "artifact_paths": self.artifact_paths,
            "diagnostics_summary": self.diagnostics_summary,
        }
        if self.output_details:
            data["output_details"] = self.output_details
        if self.analysis:
            data["analysis"] = self.analysis
        if self.drift_acknowledgement is not None:
            data["drift_acknowledgement"] = self.drift_acknowledgement.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobResults":
        drift_acknowledgement = data.get("drift_acknowledgement")
        return cls(
            render_timestamp_utc=str(data["render_timestamp_utc"]),
            source=SourceRecord.from_dict(data["source"]),
            preflight=PreflightSummary.from_dict(data["preflight"]),
            warnings=tuple(str(item) for item in data.get("warnings", [])),
            fallbacks=tuple(str(item) for item in data.get("fallbacks", [])),
            artifact_paths={
                str(key): str(value)
                for key, value in data.get("artifact_paths", {}).items()
            },
            diagnostics_summary=dict(data.get("diagnostics_summary", {})),
            output_details=dict(data.get("output_details", {})),
            analysis=dict(data.get("analysis", {})),
            drift_acknowledgement=(
                None
                if drift_acknowledgement is None
                else DriftAcknowledgement.from_dict(drift_acknowledgement)
            ),
        )


@dataclass(frozen=True)
class SidecarDocument:
    """This top-level model keeps the three normative domains separate so reloading can stay disciplined."""

    schema_version: str
    intent: JobIntent
    observed_environment: ObservedEnvironment
    results: JobResults

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "intent": self.intent.to_dict(),
            "observed_environment": self.observed_environment.to_dict(),
            "results": self.results.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SidecarDocument":
        return cls(
            schema_version=str(data["schema_version"]),
            intent=JobIntent.from_dict(data["intent"]),
            observed_environment=ObservedEnvironment.from_dict(
                data["observed_environment"]
            ),
            results=JobResults.from_dict(data["results"]),
        )

# ######################################################################################################################
#
#
#                                         AAAAAAAA
#                                       AAAA    AAAAA              AAAAAAAA
#                                     AAA          AAA           AAAA    AAA
#                                     AA            AA          AAA       AAA
#                                     AA            AAAAAAAAAA  AAA       AAAAAAAAAA
#                                     AAA                  AAA  AAA               AA
#                                      AAA                AAA    AAAAA            AA
#                                       AAAAA            AAA        AAA           AA
#                                          AAA          AAA                       AA
#                                          AAA         AAA                        AA
#                                          AA         AAA                         AA
#                                          AA        AAA                          AA
#                                         AAA       AAAAAAAAA                     AA
#                                         AAA       AAAAAAAAA                     AA
#                                         AA                   AAAAAAAAAAAAAA     AA
#                                         AA  AAAAAAAAAAAAAAAAAAAAAAAA    AAAAAAA AA
#                                        AAAAAAAAAAA                           AA AA
#                                                                            AAA  AA
#                                                                          AAAA   AA
#                                                                       AAAA      AA
#                                                                    AAAAA        AA
#                                                                AAAAA            AA
#                                                             AAAAA               AA
#                                                         AAAAAA                  AA
#                                                     AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
#
#
# ######################################################################################################################
#
#                                                 Copyright (c) JoeShade
#                               Licensed under the GNU Affero General Public License v3.0
#
# ######################################################################################################################
#
#                                         +44 (0) 7356 042702 | joe@jshade.co.uk
#
# ######################################################################################################################
