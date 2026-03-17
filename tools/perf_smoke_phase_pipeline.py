"""This developer-only script profiles one synthetic phase-processing chunk so policy changes can be compared locally without turning timing checks into a flaky test gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import psutil
import numpy as np

from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.models import (
    AnalysisSettings,
    DiagnosticLevel,
    JobIntent,
    PhaseSettings,
    Resolution,
    ResourcePolicy,
)
from phase_motion_app.core.phase_engine import (
    StreamingPhaseAmplifier,
    _build_motion_reference,
    _build_warp_geometry,
    _estimate_local_phase_shifts_against_reference,
    _normalize_confidence,
    _rgb_to_luma,
    _warp_rgb_frames,
)
from phase_motion_app.core.preflight import (
    ResourceBudget,
    SourceMetadata,
    choose_scheduler_inputs,
)
from phase_motion_app.core.quantitative_analysis import (
    BackgroundStreamingQuantitativeAnalyzer,
    StreamingQuantitativeAnalyzer,
)


def _synthetic_clip(*, frame_count: int, width: int, height: int) -> np.ndarray:
    """Build one deterministic moving blob clip so repeated local runs compare the same workload."""

    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = np.zeros((frame_count, height, width, 3), dtype=np.float32)
    center_x = width / 2.0
    center_y = height / 2.0
    sigma = max(min(width, height) / 14.0, 4.0)
    for frame_index in range(frame_count):
        offset = 1.75 * math.sin(frame_index * 2.0 * math.pi / 8.0)
        gaussian = np.exp(
            -(
                ((grid_x - (center_x + offset)) ** 2)
                + ((grid_y - center_y) ** 2)
            )
            / (2.0 * sigma**2)
        ).astype(np.float32)
        frames[frame_index] = gaussian[..., None]
    return frames


def _intent(*, resolution: Resolution, policy: ResourcePolicy) -> JobIntent:
    return JobIntent(
        phase=PhaseSettings(
            magnification=4.0,
            low_hz=1.0,
            high_hz=4.0,
            pyramid_type="complex_steerable",
            sigma=1.0,
            attenuate_other_frequencies=True,
        ),
        processing_resolution=resolution,
        output_resolution=resolution,
        resource_policy=policy,
    )


def _budget() -> ResourceBudget:
    memory = psutil.virtual_memory()
    free_disk = 32 * 1024 * 1024 * 1024
    return ResourceBudget(
        available_scratch_bytes=free_disk,
        scratch_floor_bytes=512 * 1024 * 1024,
        available_output_volume_bytes=free_disk,
        available_ram_bytes=int(memory.available),
        reserved_ui_headroom_bytes=512 * 1024 * 1024,
        retention_budget_bytes=free_disk,
        retained_evidence_bytes=0,
    )


def _source(*, resolution: Resolution, frame_count: int, fps: float) -> SourceMetadata:
    return SourceMetadata(
        fps=fps,
        duration_seconds=frame_count / fps,
        frame_count=frame_count,
        width=resolution.width,
        height=resolution.height,
        is_cfr=True,
        bit_depth=8,
    )


def _measure_policy(
    *,
    policy: ResourcePolicy,
    frames: np.ndarray,
    fps: float,
    repetitions: int,
) -> dict[str, object]:
    """Measure motion, warp, end-to-end chunk time, and analysis handoff for one policy-sized runtime plan."""

    resolution = Resolution(width=frames.shape[2], height=frames.shape[1])
    scheduler = choose_scheduler_inputs(
        intent=_intent(resolution=resolution, policy=policy),
        source=_source(resolution=resolution, frame_count=frames.shape[0], fps=fps),
        budgets=_budget(),
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    reference_luma = _rgb_to_luma(frames).mean(axis=0, dtype=np.float32).astype(
        np.float32,
        copy=False,
    )
    reference = _build_motion_reference(reference_luma, sigma_scale=1.0)
    geometry = _build_warp_geometry(
        height=resolution.height,
        width=resolution.width,
        source_height=len(reference.layout.row_starts),
        source_width=len(reference.layout.column_starts),
    )
    motion_times: list[float] = []
    warp_times: list[float] = []
    process_times: list[float] = []
    analysis_submit_times: list[float] = []

    for _ in range(repetitions):
        amplifier = StreamingPhaseAmplifier(
            reference_luma=reference_luma,
            fps=fps,
            low_hz=1.0,
            high_hz=4.0,
            magnification=4.0,
            warp_worker_count=scheduler.warp_worker_count,
            motion_worker_count=scheduler.motion_worker_count,
        )
        try:
            luma = _rgb_to_luma(frames)
            start = time.perf_counter()
            displacement_x, displacement_y, confidence = _estimate_local_phase_shifts_against_reference(
                luma,
                reference,
                worker_pool=amplifier._worker_pool,
                worker_count=scheduler.motion_worker_count,
                progress_callback=None,
            )
            motion_times.append(time.perf_counter() - start)

            confidence_scale = _normalize_confidence(confidence)
            displacement_x *= confidence_scale[None, :, :] * np.float32(4.0)
            displacement_y *= confidence_scale[None, :, :] * np.float32(4.0)
            start = time.perf_counter()
            _warp_rgb_frames(
                frames,
                displacement_x_grid=displacement_x,
                displacement_y_grid=displacement_y,
                layout=reference.layout,
                geometry=geometry,
                max_displacement_px=np.float32(4.0),
                worker_pool=amplifier._worker_pool,
                worker_count=scheduler.warp_worker_count,
                progress_callback=None,
                progress_prefix="warp",
            )
            warp_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            amplifier.process_chunk(frames)
            process_times.append(time.perf_counter() - start)
        finally:
            amplifier.close()

        analyzer = StreamingQuantitativeAnalyzer(
            settings=AnalysisSettings(export_advanced_files=False),
            processing_resolution=resolution,
            fps=fps,
            low_hz=1.0,
            high_hz=4.0,
            reference_luma=reference_luma,
            exclusion_zones=(),
            drift_assessment=DriftAssessment(),
        )
        if scheduler.analyzer_mode.value == "background_thread":
            background = BackgroundStreamingQuantitativeAnalyzer(
                analyzer,
                queue_depth=scheduler.analysis_queue_depth,
            )
            try:
                start = time.perf_counter()
                background.add_chunk(frames)
                analysis_submit_times.append(time.perf_counter() - start)
            finally:
                background.close()
        else:
            start = time.perf_counter()
            analyzer.add_chunk(frames)
            analysis_submit_times.append(time.perf_counter() - start)

    return {
        "resource_policy": policy.value,
        "scheduler": {
            "chunk_frames": scheduler.chunk_frames,
            "chunk_cap_frames": scheduler.chunk_cap_frames,
            "internal_queue_depth": scheduler.internal_queue_depth,
            "compute_worker_count": scheduler.compute_worker_count,
            "warp_worker_count": scheduler.warp_worker_count,
            "motion_worker_count": scheduler.motion_worker_count,
            "analyzer_mode": scheduler.analyzer_mode.value,
            "analysis_queue_depth": scheduler.analysis_queue_depth,
        },
        "timings_seconds": {
            "motion_estimation_mean": statistics.fmean(motion_times),
            "warp_mean": statistics.fmean(warp_times),
            "process_chunk_mean": statistics.fmean(process_times),
            "analysis_submit_mean": statistics.fmean(analysis_submit_times),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one synthetic phase-processing chunk under the repository's current resource-policy presets.",
    )
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["conservative", "balanced", "aggressive"],
        choices=[policy.value for policy in ResourcePolicy],
    )
    args = parser.parse_args()

    frames = _synthetic_clip(
        frame_count=args.frames,
        width=args.width,
        height=args.height,
    )
    results = [
        _measure_policy(
            policy=ResourcePolicy(policy_name),
            frames=frames,
            fps=float(args.fps),
            repetitions=max(1, args.repetitions),
        )
        for policy_name in args.policies
    ]
    print(json.dumps({"workload": vars(args), "results": results}, indent=2))


if __name__ == "__main__":
    main()
