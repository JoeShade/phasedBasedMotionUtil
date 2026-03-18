"""This file tests diagnostics bundle assembly so capped run records remain structured and preserve the mandatory evidence set."""

from __future__ import annotations

import json
from pathlib import Path

from phase_motion_app.core.diagnostics import DiagnosticsArtifactClass
from phase_motion_app.core.diagnostics_bundle import (
    DiagnosticsBundleInput,
    OptionalDiagnosticsArtifact,
    write_diagnostics_bundle,
)


def _bundle_input(tmp_path: Path, *, diagnostics_cap_bytes: int) -> DiagnosticsBundleInput:
    jsonl_path = tmp_path / "run.jsonl"
    jsonl_path.write_text('{"event":"started"}\n', encoding="utf-8")
    return DiagnosticsBundleInput(
        job_id="job-bundle-1",
        status="failed",
        diagnostics_directory=tmp_path / "diagnostics",
        diagnostics_cap_bytes=diagnostics_cap_bytes,
        jsonl_log_path=jsonl_path,
        settings_snapshot={"intent": {"resource_policy": "balanced"}},
        source_metadata={"path": "A:/videos/source.mp4"},
        preflight_report={"warnings": [], "blockers": []},
        scheduler_decisions={"chunk_frames": 32, "thread_limit": 1},
        memory_estimate={"ram_required_bytes": 1048576},
        mask_geometry=[{"zone_id": "zone-1"}],
        stage_timings={"decode": 1.2},
        watchdog_evidence={"last_message_type": "failure"},
        artifact_paths={"diagnostics": "A:/diagnostics/job-bundle-1"},
        terminal_details={"classification": "encode_failure", "stage": "encode"},
        intermediate_storage_policy="in_memory_float32",
    )


def test_diagnostics_bundle_writer_creates_structured_bundle(tmp_path: Path) -> None:
    result = write_diagnostics_bundle(_bundle_input(tmp_path, diagnostics_cap_bytes=1024 * 1024))
    payload = json.loads(result.bundle_path.read_text(encoding="utf-8"))

    assert result.bundle_path.exists()
    assert payload["job_id"] == "job-bundle-1"
    assert payload["diagnostics_cap_bytes"] == 1024 * 1024
    assert payload["terminal_details"]["classification"] == "encode_failure"
    assert payload["suppressed_artifacts"] == []


def test_diagnostics_bundle_writer_records_cap_suppression(tmp_path: Path) -> None:
    bundle = _bundle_input(tmp_path, diagnostics_cap_bytes=1024)
    bundle = DiagnosticsBundleInput(
        **{
            **bundle.__dict__,
            "optional_artifacts": (
                OptionalDiagnosticsArtifact(
                    name="overlay.png",
                    size_bytes=2048,
                    artifact_class=DiagnosticsArtifactClass.VISUAL_DEBUG_EXPORT,
                ),
            ),
        }
    )

    result = write_diagnostics_bundle(bundle)
    payload = json.loads(result.bundle_path.read_text(encoding="utf-8"))

    assert "overlay.png" in result.suppressed_artifacts
    assert "overlay.png" in payload["suppressed_artifacts"]

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
