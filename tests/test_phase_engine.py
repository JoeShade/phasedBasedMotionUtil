"""This file tests the numeric phase engine on synthetic clips so the worker can rely on a real amplification core rather than a render placeholder."""

from __future__ import annotations

import math

import numpy as np
import pytest

import phase_motion_app.core.phase_engine as phase_engine_module
from phase_motion_app.core.acceleration import (
    build_processing_backend,
    detect_acceleration_capability,
)
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


def test_phase_engine_parallel_helper_partitions_match_serial_output() -> None:
    clip = _make_blob_motion_clip()

    serial = amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
        warp_worker_count=1,
        motion_worker_count=1,
    )
    parallel = amplify_motion_rgb(
        clip,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
        warp_worker_count=3,
        motion_worker_count=2,
    )

    assert np.allclose(parallel, serial, atol=1e-5)


def test_cached_scalar_field_resize_plan_matches_uncached_resize() -> None:
    field = np.arange(16, dtype=np.float32).reshape(4, 4)
    resize_plan = phase_engine_module._build_scalar_field_resize_plan(
        source_height=4,
        source_width=4,
        target_height=9,
        target_width=7,
    )

    uncached = phase_engine_module._resize_scalar_field_bilinear(
        field,
        target_height=9,
        target_width=7,
    )
    cached = phase_engine_module._resize_scalar_field_bilinear(
        field,
        target_height=9,
        target_width=7,
        plan=resize_plan,
    )

    assert uncached.shape == (9, 7)
    assert np.allclose(cached, uncached, atol=1e-6)


def test_streaming_phase_amplifier_parallel_workers_preserve_chunk_order() -> None:
    clip = _make_blob_motion_clip()
    reference_luma = (
        0.2126 * clip[..., 0] + 0.7152 * clip[..., 1] + 0.0722 * clip[..., 2]
    ).mean(axis=0).astype(np.float32)
    serial = StreamingPhaseAmplifier(
        reference_luma=reference_luma,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
        warp_worker_count=1,
        motion_worker_count=1,
    )
    parallel = StreamingPhaseAmplifier(
        reference_luma=reference_luma,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
        warp_worker_count=3,
        motion_worker_count=2,
    )

    try:
        serial_output = np.concatenate(
            (
                serial.process_chunk(clip[:7]),
                serial.process_chunk(clip[7:15]),
                serial.process_chunk(clip[15:]),
            ),
            axis=0,
        )
        parallel_output = np.concatenate(
            (
                parallel.process_chunk(clip[:7]),
                parallel.process_chunk(clip[7:15]),
                parallel.process_chunk(clip[15:]),
            ),
            axis=0,
        )
    finally:
        serial.close()
        parallel.close()

    assert np.allclose(parallel_output, serial_output, atol=1e-5)


def test_confidence_normalization_suppresses_weak_cells_without_reordering_strength() -> None:
    confidence = np.array(
        [
            [0.0, 0.05, 0.2],
            [0.5, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    normalized = phase_engine_module._normalize_confidence(confidence)

    assert normalized.shape == confidence.shape
    assert normalized.dtype == np.float32
    assert float(normalized[0, 0]) == 0.0
    assert float(normalized[0, 1]) < 0.1
    assert float(normalized[0, 2]) < float(normalized[1, 0]) < float(normalized[1, 1])
    assert float(normalized[1, 2]) == 1.0


def test_analysis_confidence_normalization_preserves_mid_confidence_cells_more_than_render_gate() -> None:
    confidence = np.array(
        [
            [0.0, 0.05, 0.2],
            [0.5, 0.8, 1.0],
        ],
        dtype=np.float32,
    )

    render_normalized = phase_engine_module._normalize_confidence(confidence)
    analysis_normalized = phase_engine_module._normalize_analysis_confidence(confidence)

    assert analysis_normalized.shape == confidence.shape
    assert analysis_normalized.dtype == np.float32
    assert float(analysis_normalized[0, 0]) == 0.0
    assert float(analysis_normalized[0, 1]) > float(render_normalized[0, 1])
    assert float(analysis_normalized[0, 2]) > float(render_normalized[0, 2])
    assert float(analysis_normalized[1, 2]) == 1.0


def test_streaming_phase_amplifier_matches_accelerated_backend_when_available() -> None:
    capability = detect_acceleration_capability()
    if not capability.usable:
        pytest.skip("Optional CuPy backend is unavailable in this test environment.")

    _, backend = build_processing_backend(True)
    clip = _make_blob_motion_clip()[:8]
    reference_luma = (
        0.2126 * clip[..., 0] + 0.7152 * clip[..., 1] + 0.0722 * clip[..., 2]
    ).mean(axis=0).astype(np.float32)
    cpu_amplifier = StreamingPhaseAmplifier(
        reference_luma=reference_luma,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
    )
    accelerated_amplifier = StreamingPhaseAmplifier(
        reference_luma=reference_luma,
        fps=12.0,
        low_hz=1.0,
        high_hz=3.0,
        magnification=4.0,
        backend=backend,
    )

    try:
        cpu_output = cpu_amplifier.process_chunk(clip)
        accelerated_output = accelerated_amplifier.process_chunk(
            backend.asarray(clip, dtype=backend.xp.float32, copy=False)
        )
    finally:
        cpu_amplifier.close()
        accelerated_amplifier.close()

    assert np.allclose(backend.to_host(accelerated_output), cpu_output, atol=1e-4)

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
