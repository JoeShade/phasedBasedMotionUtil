"""This file tests the numeric phase engine on synthetic clips so the worker can rely on a real amplification core rather than a render placeholder."""

from __future__ import annotations

import math

import numpy as np

from phase_motion_app.core.phase_engine import StreamingPhaseAmplifier, amplify_motion_rgb


def _make_static_clip() -> np.ndarray:
    frames = np.zeros((12, 24, 24, 3), dtype=np.float32)
    frames[:, 8:16, 8:16, :] = 0.8
    return frames


def _make_edge_motion_clip() -> np.ndarray:
    frame_count = 24
    height = 32
    width = 32
    frames = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    base_profile = np.linspace(0.0, 1.0, width, dtype=np.float32)
    base_x = np.arange(width, dtype=np.float32)
    for index in range(frame_count):
        offset = 0.75 * math.sin(index * 2.0 * math.pi / 6.0)
        shifted = np.interp(
            base_x - offset,
            base_x,
            base_profile,
            left=base_profile[0],
            right=base_profile[-1],
        )
        frames[index] = shifted[None, :, None]
    return frames


def _make_blob_motion_clip() -> np.ndarray:
    frame_count = 24
    height = 48
    width = 48
    center_y = 24.0
    base_x = 24.0
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    for index in range(frame_count):
        offset = 1.0 * math.sin(index * 2.0 * math.pi / 6.0)
        gaussian = np.exp(
            -(
                ((grid_x - (base_x + offset)) ** 2)
                + ((grid_y - center_y) ** 2)
            )
            / (2.0 * 4.0**2)
        ).astype(np.float32)
        frames[index] = gaussian[..., None]
    return frames


def _estimate_motion_amplitude_x(frames: np.ndarray) -> float:
    width = frames.shape[2]
    positions = []
    x = np.arange(width, dtype=np.float32)
    for frame in frames:
        profile = frame[..., 0].mean(axis=0)
        total = float(profile.sum())
        if total <= 1e-6:
            positions.append(0.0)
            continue
        positions.append(float((profile * x).sum() / total))
    centered = np.asarray(positions, dtype=np.float32) - np.mean(positions)
    return float(np.max(np.abs(centered)))


def test_phase_engine_keeps_static_clip_stable() -> None:
    clip = _make_static_clip()
    amplified = amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=20.0,
    )

    assert np.allclose(amplified, clip, atol=1e-3)


def test_phase_engine_changes_band_motion_without_leaving_bounds() -> None:
    clip = _make_edge_motion_clip()

    amplified = amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
    )

    assert np.mean(np.abs(amplified - clip)) > 1e-3
    assert amplified.min() >= 0.0
    assert amplified.max() <= 1.0


def test_phase_engine_accepts_float32_input_without_changing_shape() -> None:
    clip = _make_static_clip().astype(np.float32, copy=False)

    amplified = amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=2.0,
    )

    assert amplified.shape == clip.shape
    assert amplified.dtype == np.float32


def test_phase_engine_reports_internal_progress_milestones() -> None:
    clip = _make_edge_motion_clip()
    milestones: list[str] = []

    amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=2.0,
        progress_callback=milestones.append,
    )

    assert milestones
    assert any(item.endswith("_done") for item in milestones)
    assert any("temporal_band" in item for item in milestones)


def test_phase_engine_increases_visible_motion_amplitude() -> None:
    clip = _make_blob_motion_clip()

    amplified = amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
    )

    source_amplitude = _estimate_motion_amplitude_x(clip)
    amplified_amplitude = _estimate_motion_amplitude_x(amplified)

    assert amplified_amplitude > source_amplitude * 1.2


def test_streaming_phase_amplifier_keeps_static_clip_stable() -> None:
    clip = _make_static_clip()
    reference_luma = (
        0.2126 * clip[..., 0] + 0.7152 * clip[..., 1] + 0.0722 * clip[..., 2]
    ).mean(axis=0).astype(np.float32)
    amplifier = StreamingPhaseAmplifier(
        reference_luma=reference_luma,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=8.0,
    )

    first = amplifier.process_chunk(clip[:6])
    second = amplifier.process_chunk(clip[6:])
    amplified = np.concatenate((first, second), axis=0)

    assert np.allclose(amplified, clip, atol=2e-2)


def test_streaming_phase_amplifier_increases_motion_amplitude_across_chunks() -> None:
    clip = _make_blob_motion_clip()
    reference_luma = (
        0.2126 * clip[..., 0] + 0.7152 * clip[..., 1] + 0.0722 * clip[..., 2]
    ).mean(axis=0).astype(np.float32)
    amplifier = StreamingPhaseAmplifier(
        reference_luma=reference_luma,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
    )

    chunks = (
        amplifier.process_chunk(clip[:7]),
        amplifier.process_chunk(clip[7:15]),
        amplifier.process_chunk(clip[15:]),
    )
    amplified = np.concatenate(chunks, axis=0)

    source_amplitude = _estimate_motion_amplitude_x(clip)
    amplified_amplitude = _estimate_motion_amplitude_x(amplified)

    assert amplified.shape == clip.shape
    assert amplified.dtype == np.float32
    assert amplified.min() >= 0.0
    assert amplified.max() <= 1.0
    assert amplified_amplitude > source_amplitude * 1.1
