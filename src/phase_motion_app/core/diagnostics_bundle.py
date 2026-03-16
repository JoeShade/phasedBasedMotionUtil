"""This file owns diagnostics bundle assembly so each run leaves a capped, structured record that explains what happened without flooding disk with uncontrolled debug output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phase_motion_app.core.diagnostics import (
    DiagnosticsArtifact,
    DiagnosticsArtifactClass,
    apply_diagnostics_cap,
)


@dataclass(frozen=True)
class OptionalDiagnosticsArtifact:
    """This model describes a planned optional diagnostics artifact that may be suppressed by the active cap."""

    name: str
    size_bytes: int
    artifact_class: DiagnosticsArtifactClass


@dataclass(frozen=True)
class DiagnosticsBundleInput:
    """This model captures the structured run facts that the diagnostics bundle must preserve."""

    job_id: str
    status: str
    diagnostics_directory: Path
    diagnostics_cap_bytes: int
    jsonl_log_path: Path
    settings_snapshot: dict[str, Any]
    source_metadata: dict[str, Any]
    preflight_report: dict[str, Any]
    scheduler_decisions: dict[str, Any]
    memory_estimate: dict[str, Any]
    mask_geometry: list[dict[str, Any]]
    stage_timings: dict[str, float]
    watchdog_evidence: dict[str, Any]
    artifact_paths: dict[str, str]
    terminal_details: dict[str, Any]
    intermediate_storage_policy: str
    optional_artifacts: tuple[OptionalDiagnosticsArtifact, ...] = ()


@dataclass(frozen=True)
class DiagnosticsBundleResult:
    """This result reports where the bundle was written and which optional artifacts were suppressed."""

    bundle_path: Path
    suppressed_artifacts: tuple[str, ...] = ()


def write_diagnostics_bundle(bundle: DiagnosticsBundleInput) -> DiagnosticsBundleResult:
    """Write the minimal structured diagnostics bundle and record any cap-triggered suppression."""

    bundle.diagnostics_directory.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "job_id": bundle.job_id,
        "status": bundle.status,
        "diagnostics_cap_bytes": bundle.diagnostics_cap_bytes,
        "jsonl_log_path": str(bundle.jsonl_log_path),
        "settings_snapshot": bundle.settings_snapshot,
        "source_metadata": bundle.source_metadata,
        "preflight_report": bundle.preflight_report,
        "scheduler_decisions": bundle.scheduler_decisions,
        "memory_estimate": bundle.memory_estimate,
        "mask_geometry": bundle.mask_geometry,
        "stage_timings": bundle.stage_timings,
        "watchdog_evidence": bundle.watchdog_evidence,
        "artifact_paths": bundle.artifact_paths,
        "terminal_details": bundle.terminal_details,
        "intermediate_storage_policy": bundle.intermediate_storage_policy,
        "suppressed_artifacts": [],
    }

    manifest_bytes = len(json.dumps(manifest_payload, separators=(",", ":")).encode("utf-8"))
    preflight_bytes = len(
        json.dumps(bundle.preflight_report, separators=(",", ":")).encode("utf-8")
    )
    jsonl_size = (
        bundle.jsonl_log_path.stat().st_size if bundle.jsonl_log_path.exists() else 0
    )
    planned_artifacts = [
        DiagnosticsArtifact(
            name="diagnostics_bundle.json",
            size_bytes=manifest_bytes,
            artifact_class=DiagnosticsArtifactClass.ARTIFACT_MANIFEST,
        ),
        DiagnosticsArtifact(
            name="preflight_report",
            size_bytes=preflight_bytes,
            artifact_class=DiagnosticsArtifactClass.PREFLIGHT_REPORT,
        ),
        DiagnosticsArtifact(
            name="run.jsonl",
            size_bytes=jsonl_size,
            artifact_class=DiagnosticsArtifactClass.CORE_JSONL_LOG,
        ),
        DiagnosticsArtifact(
            name="terminal_details",
            size_bytes=len(
                json.dumps(bundle.terminal_details, separators=(",", ":")).encode("utf-8")
            ),
            artifact_class=DiagnosticsArtifactClass.TERMINAL_FAILURE_CLASSIFICATION,
        ),
        DiagnosticsArtifact(
            name="watchdog_evidence",
            size_bytes=len(
                json.dumps(bundle.watchdog_evidence, separators=(",", ":")).encode("utf-8")
            ),
            artifact_class=DiagnosticsArtifactClass.WATCHDOG_MINIMUM,
        ),
        DiagnosticsArtifact(
            name="settings_snapshot",
            size_bytes=len(
                json.dumps(bundle.settings_snapshot, separators=(",", ":")).encode("utf-8")
            ),
            artifact_class=DiagnosticsArtifactClass.SIDECAR_RECORD,
        ),
    ]
    planned_artifacts.extend(
        DiagnosticsArtifact(
            name=artifact.name,
            size_bytes=artifact.size_bytes,
            artifact_class=artifact.artifact_class,
        )
        for artifact in bundle.optional_artifacts
    )
    retention_plan = apply_diagnostics_cap(
        planned_artifacts,
        cap_bytes=bundle.diagnostics_cap_bytes,
    )
    suppressed = tuple(artifact.name for artifact in retention_plan.dropped)
    manifest_payload["suppressed_artifacts"] = list(suppressed)

    bundle_path = bundle.diagnostics_directory / "diagnostics_bundle.json"
    bundle_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return DiagnosticsBundleResult(
        bundle_path=bundle_path,
        suppressed_artifacts=suppressed,
    )
