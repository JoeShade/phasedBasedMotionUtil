"""This file owns staged-output validation and finalization so a visible MP4 is never mistaken for a successful export until the matching sidecar is committed too."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OutputExpectation:
    """This model describes the staged MP4 properties that must match before final commit is allowed."""

    width: int
    height: int
    codec: str
    profile: str
    pixel_format: str
    expected_frame_count: int
    expected_fps: float


@dataclass(frozen=True)
class StagedMp4Observation:
    """This model carries the facts gathered from the encoder exit code and ffprobe validation step."""

    encoder_exit_code: int
    file_exists: bool
    file_closed: bool
    size_bytes: int
    probe_ok: bool
    video_stream_count: int
    audio_stream_count: int
    width: int
    height: int
    codec: str
    profile: str
    pixel_format: str
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class StagedMp4Validation:
    """This result object reports whether staged MP4 validation passed and why it failed if not."""

    is_valid: bool
    errors: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class FinalizationResult:
    """This result object makes the paired-output completion rule explicit for the UI and logs."""

    status: str
    classification: str | None = None
    final_mp4_path: Path | None = None
    final_sidecar_path: Path | None = None
    quarantine_mp4_path: Path | None = None
    incomplete_visible_mp4_path: Path | None = None


def _move_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def _nearest_existing_path(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _path_device_id(path: Path) -> int:
    return _nearest_existing_path(path).stat().st_dev


def _same_volume(left: Path, right: Path) -> bool:
    try:
        return _path_device_id(left) == _path_device_id(right)
    except OSError:
        return left.resolve(strict=False).anchor.lower() == right.resolve(
            strict=False
        ).anchor.lower()


def validate_staged_mp4(
    observation: StagedMp4Observation,
    expectation: OutputExpectation,
) -> StagedMp4Validation:
    """Apply the staged MP4 validation contract before final rename is allowed."""

    errors: list[str] = []

    if observation.encoder_exit_code != 0:
        errors.append("Encoder exit code was not zero.")
    if not observation.file_exists:
        errors.append("Staged MP4 does not exist.")
    if not observation.file_closed:
        errors.append("Staged MP4 is not closed.")
    if observation.size_bytes <= 0:
        errors.append("Staged MP4 has zero size.")
    if not observation.probe_ok:
        errors.append("ffprobe could not parse the staged MP4.")
    if observation.video_stream_count != 1:
        errors.append("Staged MP4 must contain exactly one video stream.")
    if observation.audio_stream_count != 0:
        errors.append("Staged MP4 must not contain audio.")
    if observation.width != expectation.width or observation.height != expectation.height:
        errors.append("Staged MP4 dimensions do not match the expected output size.")
    if observation.codec != expectation.codec:
        errors.append("Staged MP4 codec does not match the selected output path.")
    if observation.profile != expectation.profile:
        errors.append("Staged MP4 profile does not match the selected output path.")
    if observation.pixel_format != expectation.pixel_format:
        errors.append("Staged MP4 pixel format does not match the selected output path.")

    frame_evidence_available = (
        observation.frame_count > 0 and expectation.expected_frame_count > 0
    )
    frame_count_matches = (
        frame_evidence_available
        and abs(observation.frame_count - expectation.expected_frame_count) <= 1
    )

    duration_evidence_available = (
        observation.duration_seconds > 0 and expectation.expected_fps > 0
    )
    if duration_evidence_available:
        expected_duration = expectation.expected_frame_count / expectation.expected_fps
        frame_period = 1.0 / expectation.expected_fps
        duration_matches = (
            abs(observation.duration_seconds - expected_duration) <= frame_period
        )
    else:
        duration_matches = False

    if frame_evidence_available and duration_evidence_available:
        if not frame_count_matches and not duration_matches:
            errors.append(
                "Staged MP4 frame-count and duration evidence both failed the CFR consistency check."
            )
    elif frame_evidence_available:
        if not frame_count_matches:
            errors.append("Staged MP4 frame count differs beyond the one-frame tolerance.")
    elif duration_evidence_available:
        if not duration_matches:
            errors.append("Staged MP4 duration is not CFR-consistent within one frame period.")
    else:
        errors.append("Staged MP4 did not expose frame-count or duration evidence for CFR validation.")

    return StagedMp4Validation(is_valid=not errors, errors=tuple(errors))


def finalize_output_pair(
    *,
    staged_mp4: Path,
    staged_sidecar: Path,
    final_mp4: Path,
    final_sidecar: Path,
    failed_evidence_dir: Path,
    move_file=_move_file,
) -> FinalizationResult:
    """Commit the final MP4 + sidecar pair or quarantine a lone MP4 if the second half of the pair fails."""

    if not _same_volume(staged_mp4, final_mp4):
        return FinalizationResult(
            status="failed",
            classification="output_volume_staging_mismatch",
        )
    if not _same_volume(staged_sidecar, final_sidecar):
        return FinalizationResult(
            status="failed",
            classification="sidecar_volume_staging_mismatch",
        )

    move_file(staged_mp4, final_mp4)

    try:
        move_file(staged_sidecar, final_sidecar)
    except Exception:
        quarantine_mp4 = failed_evidence_dir / final_mp4.name
        try:
            if not _same_volume(final_mp4, quarantine_mp4):
                raise OSError("Quarantine path must be on the same volume as the final MP4.")
            move_file(final_mp4, quarantine_mp4)
        except Exception:
            return FinalizationResult(
                status="failed",
                classification="lone_mp4_incomplete_failed_evidence",
                incomplete_visible_mp4_path=final_mp4,
            )
        return FinalizationResult(
            status="failed",
            classification="lone_mp4_quarantined",
            quarantine_mp4_path=quarantine_mp4,
        )

    return FinalizationResult(
        status="completed",
        final_mp4_path=final_mp4,
        final_sidecar_path=final_sidecar,
    )
