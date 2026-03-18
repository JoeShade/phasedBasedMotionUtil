"""This file owns diagnostics retention planning so large debug artifacts can be trimmed without losing the minimum evidence required to explain a run."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DiagnosticsArtifactClass(str, Enum):
    """This enum encodes the strict diagnostics drop order from the design doc."""

    VISUAL_DEBUG_EXPORT = "visual_debug_export"
    TRACE_CHUNK_DETAIL = "trace_chunk_detail"
    DETAILED_TIMING = "detailed_timing"
    AUXILIARY_DUPLICATE_LOG = "auxiliary_duplicate_log"
    TERMINAL_FAILURE_CLASSIFICATION = "terminal_failure_classification"
    CORE_JSONL_LOG = "core_jsonl_log"
    PREFLIGHT_REPORT = "preflight_report"
    SIDECAR_RECORD = "sidecar_record"
    ARTIFACT_MANIFEST = "artifact_manifest"
    WATCHDOG_MINIMUM = "watchdog_minimum"


_DROP_PRIORITY = {
    DiagnosticsArtifactClass.VISUAL_DEBUG_EXPORT: 1,
    DiagnosticsArtifactClass.TRACE_CHUNK_DETAIL: 2,
    DiagnosticsArtifactClass.DETAILED_TIMING: 3,
    DiagnosticsArtifactClass.AUXILIARY_DUPLICATE_LOG: 4,
}

_MANDATORY_CLASSES = {
    DiagnosticsArtifactClass.TERMINAL_FAILURE_CLASSIFICATION,
    DiagnosticsArtifactClass.CORE_JSONL_LOG,
    DiagnosticsArtifactClass.PREFLIGHT_REPORT,
    DiagnosticsArtifactClass.SIDECAR_RECORD,
    DiagnosticsArtifactClass.ARTIFACT_MANIFEST,
    DiagnosticsArtifactClass.WATCHDOG_MINIMUM,
}


@dataclass(frozen=True)
class DiagnosticsArtifact:
    """This model describes one diagnostics artifact and the class that controls whether it may be dropped."""

    name: str
    size_bytes: int
    artifact_class: DiagnosticsArtifactClass


@dataclass(frozen=True)
class DiagnosticsRetentionPlan:
    """This result object explains what was kept, what was dropped, and whether suppression had to be recorded."""

    kept: tuple[DiagnosticsArtifact, ...]
    dropped: tuple[DiagnosticsArtifact, ...]
    suppression_record_required: bool


def apply_diagnostics_cap(
    artifacts: list[DiagnosticsArtifact], cap_bytes: int
) -> DiagnosticsRetentionPlan:
    """Drop diagnostics artifacts in the required order while preserving the irreducible run record."""

    total_bytes = sum(artifact.size_bytes for artifact in artifacts)
    if total_bytes <= cap_bytes:
        return DiagnosticsRetentionPlan(
            kept=tuple(artifacts),
            dropped=(),
            suppression_record_required=False,
        )

    kept = list(artifacts)
    dropped: list[DiagnosticsArtifact] = []
    for priority in sorted(set(_DROP_PRIORITY.values())):
        if sum(artifact.size_bytes for artifact in kept) <= cap_bytes:
            break
        droppable = [
            artifact
            for artifact in kept
            if _DROP_PRIORITY.get(artifact.artifact_class) == priority
        ]
        for artifact in droppable:
            if sum(item.size_bytes for item in kept) <= cap_bytes:
                break
            kept.remove(artifact)
            dropped.append(artifact)

    mandatory_still_present = all(
        any(artifact.artifact_class is required for artifact in kept)
        for required in _MANDATORY_CLASSES
    )
    if not mandatory_still_present:
        raise RuntimeError("Diagnostics cap planning attempted to drop mandatory evidence.")

    return DiagnosticsRetentionPlan(
        kept=tuple(kept),
        dropped=tuple(dropped),
        suppression_record_required=bool(dropped),
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
