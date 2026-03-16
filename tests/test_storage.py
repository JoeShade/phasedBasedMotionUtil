"""This file tests staged-output validation and finalization so failed pair commits never masquerade as completed exports."""

from __future__ import annotations

from pathlib import Path

import phase_motion_app.core.storage as storage
from phase_motion_app.core.storage import (
    FinalizationResult,
    OutputExpectation,
    StagedMp4Observation,
    finalize_output_pair,
    validate_staged_mp4,
)


def _expectation() -> OutputExpectation:
    return OutputExpectation(
        width=640,
        height=360,
        codec="hevc",
        profile="Main 10",
        pixel_format="yuv420p10le",
        expected_frame_count=120,
        expected_fps=60.0,
    )


def _observation(**overrides: object) -> StagedMp4Observation:
    base = StagedMp4Observation(
        encoder_exit_code=0,
        file_exists=True,
        file_closed=True,
        size_bytes=1024,
        probe_ok=True,
        video_stream_count=1,
        audio_stream_count=0,
        width=640,
        height=360,
        codec="hevc",
        profile="Main 10",
        pixel_format="yuv420p10le",
        frame_count=120,
        duration_seconds=2.0,
    )
    return StagedMp4Observation(**{**base.__dict__, **overrides})


def test_staged_mp4_validation_accepts_matching_output() -> None:
    result = validate_staged_mp4(_observation(), _expectation())
    assert result.is_valid is True
    assert result.errors == ()


def test_staged_mp4_validation_rejects_audio_and_dimension_mismatch() -> None:
    result = validate_staged_mp4(
        _observation(audio_stream_count=1, width=800),
        _expectation(),
    )

    assert result.is_valid is False
    assert any("must not contain audio" in error for error in result.errors)
    assert any("dimensions do not match" in error for error in result.errors)


def test_staged_mp4_validation_accepts_duration_evidence_when_frame_count_is_missing() -> None:
    result = validate_staged_mp4(
        _observation(frame_count=0, duration_seconds=2.0),
        _expectation(),
    )

    assert result.is_valid is True


def test_finalization_moves_mp4_and_sidecar_into_final_locations(tmp_path: Path) -> None:
    staged_mp4 = tmp_path / "stage" / "output.mp4"
    staged_sidecar = tmp_path / "stage" / "output.json"
    final_mp4 = tmp_path / "final" / "output.mp4"
    final_sidecar = tmp_path / "final" / "output.json"
    failed_dir = tmp_path / "failed"
    staged_mp4.parent.mkdir(parents=True)
    staged_mp4.write_text("mp4", encoding="utf-8")
    staged_sidecar.write_text("json", encoding="utf-8")

    result = finalize_output_pair(
        staged_mp4=staged_mp4,
        staged_sidecar=staged_sidecar,
        final_mp4=final_mp4,
        final_sidecar=final_sidecar,
        failed_evidence_dir=failed_dir,
    )

    assert result == FinalizationResult(
        status="completed",
        final_mp4_path=final_mp4,
        final_sidecar_path=final_sidecar,
    )
    assert final_mp4.exists()
    assert final_sidecar.exists()


def test_finalization_quarantines_lone_mp4_when_sidecar_commit_fails(tmp_path: Path) -> None:
    staged_mp4 = tmp_path / "stage" / "output.mp4"
    staged_sidecar = tmp_path / "stage" / "output.json"
    final_mp4 = tmp_path / "final" / "output.mp4"
    final_sidecar = tmp_path / "final" / "output.json"
    failed_dir = tmp_path / "failed"
    staged_mp4.parent.mkdir(parents=True)
    staged_mp4.write_text("mp4", encoding="utf-8")
    staged_sidecar.write_text("json", encoding="utf-8")

    def flaky_move(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination == final_sidecar:
            raise OSError("sidecar finalization failed")
        source.replace(destination)

    result = finalize_output_pair(
        staged_mp4=staged_mp4,
        staged_sidecar=staged_sidecar,
        final_mp4=final_mp4,
        final_sidecar=final_sidecar,
        failed_evidence_dir=failed_dir,
        move_file=flaky_move,
    )

    assert result.status == "failed"
    assert result.classification == "lone_mp4_quarantined"
    assert result.quarantine_mp4_path == failed_dir / "output.mp4"
    assert result.quarantine_mp4_path.exists()
    assert not final_mp4.exists()


def test_finalization_marks_visible_mp4_as_incomplete_when_quarantine_fails(tmp_path: Path) -> None:
    staged_mp4 = tmp_path / "stage" / "output.mp4"
    staged_sidecar = tmp_path / "stage" / "output.json"
    final_mp4 = tmp_path / "final" / "output.mp4"
    final_sidecar = tmp_path / "final" / "output.json"
    failed_dir = tmp_path / "failed"
    staged_mp4.parent.mkdir(parents=True)
    staged_mp4.write_text("mp4", encoding="utf-8")
    staged_sidecar.write_text("json", encoding="utf-8")

    def always_fail_after_mp4(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination == final_mp4:
            source.replace(destination)
            return
        raise OSError("follow-up move failed")

    result = finalize_output_pair(
        staged_mp4=staged_mp4,
        staged_sidecar=staged_sidecar,
        final_mp4=final_mp4,
        final_sidecar=final_sidecar,
        failed_evidence_dir=failed_dir,
        move_file=always_fail_after_mp4,
    )

    assert result.status == "failed"
    assert result.classification == "lone_mp4_incomplete_failed_evidence"
    assert result.incomplete_visible_mp4_path == final_mp4
    assert final_mp4.exists()


def test_finalization_rejects_cross_volume_staging(
    tmp_path: Path, monkeypatch
) -> None:
    staged_mp4 = tmp_path / "stage" / "output.mp4"
    staged_sidecar = tmp_path / "stage" / "output.json"
    final_mp4 = tmp_path / "final" / "output.mp4"
    final_sidecar = tmp_path / "final" / "output.json"
    failed_dir = tmp_path / "failed"

    staged_mp4.parent.mkdir(parents=True)
    staged_mp4.write_text("mp4", encoding="utf-8")
    staged_sidecar.write_text("json", encoding="utf-8")

    def fake_device_id(path: Path) -> int:
        return 1 if "stage" in str(path) else 2

    monkeypatch.setattr(storage, "_path_device_id", fake_device_id)

    result = finalize_output_pair(
        staged_mp4=staged_mp4,
        staged_sidecar=staged_sidecar,
        final_mp4=final_mp4,
        final_sidecar=final_sidecar,
        failed_evidence_dir=failed_dir,
    )

    assert result.status == "failed"
    assert result.classification == "output_volume_staging_mismatch"
