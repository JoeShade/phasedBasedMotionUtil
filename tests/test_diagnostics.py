"""This file tests diagnostics retention planning so the cap fallback order never drops the minimum evidence needed to explain a run."""

from __future__ import annotations

from phase_motion_app.core.diagnostics import (
    DiagnosticsArtifact,
    DiagnosticsArtifactClass,
    apply_diagnostics_cap,
)


def test_diagnostics_cap_drops_artifacts_in_documented_priority_order() -> None:
    artifacts = [
        DiagnosticsArtifact(
            name="visual.png",
            size_bytes=40,
            artifact_class=DiagnosticsArtifactClass.VISUAL_DEBUG_EXPORT,
        ),
        DiagnosticsArtifact(
            name="trace.jsonl",
            size_bytes=30,
            artifact_class=DiagnosticsArtifactClass.TRACE_CHUNK_DETAIL,
        ),
        DiagnosticsArtifact(
            name="timing.json",
            size_bytes=20,
            artifact_class=DiagnosticsArtifactClass.DETAILED_TIMING,
        ),
        DiagnosticsArtifact(
            name="aux.log",
            size_bytes=10,
            artifact_class=DiagnosticsArtifactClass.AUXILIARY_DUPLICATE_LOG,
        ),
        DiagnosticsArtifact(
            name="core.jsonl",
            size_bytes=5,
            artifact_class=DiagnosticsArtifactClass.CORE_JSONL_LOG,
        ),
        DiagnosticsArtifact(
            name="preflight.json",
            size_bytes=5,
            artifact_class=DiagnosticsArtifactClass.PREFLIGHT_REPORT,
        ),
        DiagnosticsArtifact(
            name="sidecar.json",
            size_bytes=5,
            artifact_class=DiagnosticsArtifactClass.SIDECAR_RECORD,
        ),
        DiagnosticsArtifact(
            name="manifest.json",
            size_bytes=5,
            artifact_class=DiagnosticsArtifactClass.ARTIFACT_MANIFEST,
        ),
        DiagnosticsArtifact(
            name="watchdog.json",
            size_bytes=5,
            artifact_class=DiagnosticsArtifactClass.WATCHDOG_MINIMUM,
        ),
        DiagnosticsArtifact(
            name="failure.txt",
            size_bytes=5,
            artifact_class=DiagnosticsArtifactClass.TERMINAL_FAILURE_CLASSIFICATION,
        ),
    ]

    plan = apply_diagnostics_cap(artifacts, cap_bytes=35)

    assert [artifact.name for artifact in plan.dropped] == [
        "visual.png",
        "trace.jsonl",
        "timing.json",
        "aux.log",
    ]
    assert plan.suppression_record_required is True
    assert {artifact.name for artifact in plan.kept} == {
        "core.jsonl",
        "preflight.json",
        "sidecar.json",
        "manifest.json",
        "watchdog.json",
        "failure.txt",
    }
