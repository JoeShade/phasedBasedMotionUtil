"""This file owns bounded ingest-time source-motion analysis so the shell can suggest a sane starting frequency band without running the full render pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_motion_app.core.ffprobe import FfprobeMediaInfo
from phase_motion_app.core.source_normalization import (
    build_ffmpeg_normalization_filters,
    build_source_normalization_plan,
)
from phase_motion_app.core.toolchain import resolve_toolchain


@dataclass(frozen=True)
class FrequencyBandSuggestion:
    """This model carries the ingest-time band suggestion and how strongly the proxy analysis supported it."""

    low_hz: float
    high_hz: float
    peak_hz: float
    analysis_fps: float
    confidence: float


@dataclass(frozen=True)
class _SpectrumPeakCandidate:
    """This model keeps one candidate peak explicit so selection stays testable and readable."""

    index: int
    frequency_hz: float
    amplitude: float
    prominence: float
    support_fraction: float
    score: float


_INGEST_ANALYSIS_FPS_CAP = 18.0
_INGEST_ANALYSIS_WIDTH = 176
_INGEST_ANALYSIS_MAX_VIDEO_SECONDS = 18.0
_INGEST_ANALYSIS_MIN_VIDEO_SECONDS = 8.0
_TRACE_SMOOTH_KERNEL = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
_TRACE_SMOOTH_KERNEL /= _TRACE_SMOOTH_KERNEL.sum()


def suggest_frequency_band_from_motion_trace(
    motion_trace: np.ndarray,
    *,
    fps: float,
    min_hz: float = 0.1,
    max_hz: float | None = None,
) -> FrequencyBandSuggestion | None:
    """Turn one motion trace into a conservative starting band using the same robust spectrum logic as ingest-time proxy analysis."""

    if motion_trace.ndim != 1 or motion_trace.shape[0] < 32 or fps <= 0:
        return None

    return _suggest_frequency_band_from_traces(
        traces=(motion_trace.astype(np.float32, copy=False),),
        trace_weights=np.asarray([1.0], dtype=np.float32),
        fps=fps,
        min_hz=min_hz,
        max_hz=max_hz,
    )


def analyze_source_frequency_band(
    source_path: Path,
    probe: FfprobeMediaInfo,
) -> FrequencyBandSuggestion:
    """Decode a bounded grayscale proxy clip and derive a conservative baseline band suggestion from a richer multi-cell motion spectrum."""

    normalization_plan = build_source_normalization_plan(probe)
    analysis_fps = min(normalization_plan.working_fps, _INGEST_ANALYSIS_FPS_CAP)
    analysis_width = _INGEST_ANALYSIS_WIDTH
    scaled_height = (
        normalization_plan.working_resolution.height
        * analysis_width
        / normalization_plan.working_resolution.width
    )
    analysis_height = max(2, int(round(scaled_height / 2.0) * 2))
    sample_duration_seconds = min(
        max(_INGEST_ANALYSIS_MIN_VIDEO_SECONDS, probe.duration_seconds),
        _INGEST_ANALYSIS_MAX_VIDEO_SECONDS,
    )
    ffmpeg = str(resolve_toolchain().ffmpeg)
    filter_chain = build_ffmpeg_normalization_filters(
        normalization_plan,
        output_fps=analysis_fps,
    )
    filter_chain.extend(
        (
            f"scale={analysis_width}:-2:flags=lanczos",
            "format=gray",
        )
    )
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(source_path),
        "-an",
        "-t",
        f"{sample_duration_seconds:.3f}",
        "-vf",
        ",".join(filter_chain),
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    raw = subprocess.check_output(command)
    frame_pixels = analysis_width * analysis_height
    frame_count = len(raw) // frame_pixels
    if frame_count < 32:
        return _fallback_band(probe, analysis_fps)

    frames = np.frombuffer(raw[: frame_count * frame_pixels], dtype=np.uint8).reshape(
        frame_count,
        analysis_height,
        analysis_width,
    )
    grayscale = frames.astype(np.float32) / 255.0

    # Keep a small border trim so hard edges and black bars do not dominate.
    row_start = int(analysis_height * 0.08)
    row_end = max(row_start + 2, int(analysis_height * 0.92))
    column_start = int(analysis_width * 0.08)
    column_end = max(column_start + 2, int(analysis_width * 0.92))
    cropped = grayscale[:, row_start:row_end, column_start:column_end]
    if cropped.shape[1] < 4 or cropped.shape[2] < 4:
        return _fallback_band(probe, analysis_fps)
    even_height = cropped.shape[1] - (cropped.shape[1] % 2)
    even_width = cropped.shape[2] - (cropped.shape[2] % 2)
    cropped = cropped[:, :even_height, :even_width]

    pooled = 0.25 * (
        cropped[:, ::2, ::2]
        + cropped[:, 1::2, ::2]
        + cropped[:, ::2, 1::2]
        + cropped[:, 1::2, 1::2]
    )
    traces, weights = _build_proxy_motion_traces(pooled)
    suggestion = _suggest_frequency_band_from_traces(
        traces=traces,
        trace_weights=weights,
        fps=analysis_fps,
        min_hz=max(0.15, 2.0 / max(probe.duration_seconds, 1.0)),
    )
    if suggestion is None or suggestion.confidence < 0.15:
        return _fallback_band(probe, analysis_fps)
    return suggestion


def _build_proxy_motion_traces(
    grayscale_frames: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Build a small set of signed motion traces from gradient-projected local cells so ingest can see real motion rather than a single whole-frame difference curve."""

    frame_count, height, width = grayscale_frames.shape
    if frame_count < 32 or height < 8 or width < 8:
        return (), np.zeros(0, dtype=np.float32)

    reference = grayscale_frames.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    grad_y, grad_x = np.gradient(reference)
    row_edges = np.linspace(0, height, int(np.clip(round(height / 18.0), 4, 6)) + 1)
    column_edges = np.linspace(0, width, int(np.clip(round(width / 18.0), 5, 8)) + 1)
    row_edges = np.round(row_edges).astype(np.int32)
    column_edges = np.round(column_edges).astype(np.int32)

    traces: list[np.ndarray] = []
    weights: list[float] = []
    for row_index in range(len(row_edges) - 1):
        for column_index in range(len(column_edges) - 1):
            row_start = int(row_edges[row_index])
            row_end = int(row_edges[row_index + 1])
            column_start = int(column_edges[column_index])
            column_end = int(column_edges[column_index + 1])
            if (row_end - row_start) < 6 or (column_end - column_start) < 6:
                continue
            cell_reference = reference[row_start:row_end, column_start:column_end]
            cell_grad_x = grad_x[row_start:row_end, column_start:column_end]
            cell_grad_y = grad_y[row_start:row_end, column_start:column_end]
            cell_frames = grayscale_frames[:, row_start:row_end, column_start:column_end]
            cell_diff = cell_frames - cell_reference[None, :, :]
            texture_strength = float(
                np.sqrt(np.mean(np.square(cell_grad_x)) + np.mean(np.square(cell_grad_y)))
            )
            if texture_strength <= 1e-4:
                continue
            for component_gradient in (cell_grad_x, cell_grad_y):
                gradient_energy = float(np.mean(np.square(component_gradient)))
                if gradient_energy <= 1e-5:
                    continue
                trace = np.mean(
                    cell_diff * component_gradient[None, :, :],
                    axis=(1, 2),
                    dtype=np.float32,
                ) / gradient_energy
                trace_rms = float(np.sqrt(np.mean(np.square(trace.astype(np.float32, copy=False)))))
                if trace_rms <= 1e-5:
                    continue
                traces.append(trace.astype(np.float32, copy=False))
                weights.append(texture_strength * trace_rms)

    # Keep one whole-frame trace as a cheap guard against all local cells missing
    # a broad moving subject.
    global_diff = grayscale_frames - reference[None, :, :]
    global_trace = np.mean(global_diff * grad_x[None, :, :], axis=(1, 2), dtype=np.float32)
    global_rms = float(np.sqrt(np.mean(np.square(global_trace.astype(np.float32, copy=False)))))
    if global_rms > 1e-5:
        traces.append(global_trace.astype(np.float32, copy=False))
        weights.append(global_rms)

    return tuple(traces), np.asarray(weights, dtype=np.float32)


def _suggest_frequency_band_from_traces(
    *,
    traces: tuple[np.ndarray, ...],
    trace_weights: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float | None = None,
) -> FrequencyBandSuggestion | None:
    """Aggregate one or more motion traces into a single robust spectrum and suggest a band from the strongest supported peak."""

    if not traces or fps <= 0:
        return None

    prepared_traces: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    spectrum_weights: list[float] = []
    frequencies: np.ndarray | None = None
    for trace, weight in zip(traces, trace_weights, strict=False):
        prepared = _prepare_trace_for_spectrum(trace)
        if prepared is None:
            continue
        spectrum = np.abs(np.fft.rfft(prepared)).astype(np.float32, copy=False)
        if frequencies is None:
            frequencies = np.fft.rfftfreq(prepared.shape[0], d=1.0 / fps)
        prepared_traces.append(prepared)
        spectra.append(spectrum)
        spectrum_weights.append(max(float(weight), 1e-6))

    if not spectra or frequencies is None:
        return None

    weighted_spectra = np.stack(spectra, axis=0)
    weight_array = np.asarray(spectrum_weights, dtype=np.float32)
    aggregate = np.average(weighted_spectra, axis=0, weights=weight_array).astype(
        np.float32,
        copy=False,
    )
    nyquist = fps / 2.0
    search_max_hz = min(max_hz or (nyquist * 0.85), nyquist * 0.85)
    search = (frequencies >= min_hz) & (frequencies <= search_max_hz)
    if np.count_nonzero(search) < 3:
        return None

    search_frequencies = frequencies[search]
    search_amplitudes = aggregate[search]
    search_cell_spectra = weighted_spectra[:, search]
    candidate = _select_peak_candidate(
        frequencies=search_frequencies,
        amplitudes=search_amplitudes,
        cell_spectra=search_cell_spectra,
    )
    if candidate is None:
        return None

    low_hz, high_hz = _estimate_band_edges_from_spectrum(
        frequencies=search_frequencies,
        amplitudes=search_amplitudes,
        peak_index=candidate.index,
        min_hz=min_hz,
        max_hz=search_max_hz,
    )
    confidence_floor = (
        float(np.median(search_amplitudes))
        if search_amplitudes.size > 0
        else 0.0
    )
    confidence = float(
        np.clip(
            0.45 * candidate.support_fraction
            + 0.35 * (candidate.prominence / max(candidate.amplitude, 1e-6))
            + 0.20 * (
                (candidate.amplitude - confidence_floor)
                / max(candidate.amplitude, 1e-6)
            ),
            0.0,
            1.0,
        )
    )
    return FrequencyBandSuggestion(
        low_hz=_round_band_edge(low_hz),
        high_hz=_round_band_edge(high_hz),
        peak_hz=candidate.frequency_hz,
        analysis_fps=fps,
        confidence=confidence,
    )


def _prepare_trace_for_spectrum(trace: np.ndarray) -> np.ndarray | None:
    """Smooth and center one trace so ingest-time spectra ignore DC drift and single-frame noise spikes."""

    if trace.ndim != 1 or trace.shape[0] < 32:
        return None
    smoothed = np.convolve(
        trace.astype(np.float32, copy=False),
        _TRACE_SMOOTH_KERNEL,
        mode="same",
    )
    centered = smoothed - smoothed.mean(dtype=np.float32)
    if np.max(np.abs(centered)) <= 1e-6:
        return None
    return centered.astype(np.float32, copy=False)


def _select_peak_candidate(
    *,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    cell_spectra: np.ndarray,
) -> _SpectrumPeakCandidate | None:
    """Score candidate peaks with neighborhood prominence, cell support, and harmonic evidence so slow fundamentals are not discarded just because a harmonic is sharper."""

    if amplitudes.size == 0 or np.max(amplitudes) <= 1e-6:
        return None

    peak_indices = _find_local_peak_indices(amplitudes)
    global_max = float(np.max(amplitudes))
    candidates: list[_SpectrumPeakCandidate] = []
    for peak_index in peak_indices:
        amplitude = float(amplitudes[peak_index])
        prominence = _estimate_peak_prominence(amplitudes, peak_index)
        support_fraction = _estimate_support_fraction(cell_spectra, peak_index)
        harmonic_bonus = _estimate_harmonic_bonus(
            frequencies=frequencies,
            amplitudes=amplitudes,
            peak_indices=peak_indices,
            peak_index=peak_index,
        )
        score = (
            0.55 * (amplitude / max(global_max, 1e-6))
            + 0.20 * support_fraction
            + 0.15 * (prominence / max(amplitude, 1e-6))
            + 0.10 * harmonic_bonus
        )
        candidates.append(
            _SpectrumPeakCandidate(
                index=int(peak_index),
                frequency_hz=float(frequencies[peak_index]),
                amplitude=amplitude,
                prominence=prominence,
                support_fraction=support_fraction,
                score=float(score),
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.score)


def _find_local_peak_indices(amplitudes: np.ndarray) -> tuple[int, ...]:
    """Find simple local maxima without external dependencies so baseline-band analysis works in the packaged environment."""

    if amplitudes.size == 0:
        return ()
    if amplitudes.size == 1:
        return (0,)

    peak_indices: list[int] = []
    for index in range(1, amplitudes.size - 1):
        center = float(amplitudes[index])
        if center >= float(amplitudes[index - 1]) and center > float(amplitudes[index + 1]):
            peak_indices.append(index)
    if not peak_indices:
        peak_indices.append(int(np.argmax(amplitudes)))
    return tuple(peak_indices)


def _estimate_peak_prominence(amplitudes: np.ndarray, peak_index: int) -> float:
    """Estimate one peak's neighborhood prominence with a small local window instead of a full signal-processing dependency."""

    window_radius = min(6, max(2, amplitudes.size // 24))
    left = amplitudes[max(0, peak_index - window_radius) : peak_index]
    right = amplitudes[peak_index + 1 : min(amplitudes.size, peak_index + window_radius + 1)]
    baseline_candidates = [float(np.median(amplitudes))]
    if left.size > 0:
        baseline_candidates.append(float(np.percentile(left, 75.0)))
    if right.size > 0:
        baseline_candidates.append(float(np.percentile(right, 75.0)))
    baseline = max(baseline_candidates)
    return max(float(amplitudes[peak_index]) - baseline, 0.0)


def _estimate_support_fraction(cell_spectra: np.ndarray, peak_index: int) -> float:
    """Measure how many proxy traces support one candidate peak so one noisy cell cannot drive the whole suggestion."""

    if cell_spectra.size == 0:
        return 0.0
    window_start = max(0, peak_index - 1)
    window_stop = min(cell_spectra.shape[1], peak_index + 2)
    peak_values = np.max(cell_spectra[:, window_start:window_stop], axis=1)
    cell_baseline = np.median(cell_spectra, axis=1)
    cell_max = np.max(cell_spectra, axis=1)
    supporting = peak_values >= np.maximum(cell_baseline * np.float32(1.8), cell_max * np.float32(0.45))
    weights = np.clip(cell_max, 1e-6, None)
    return float(np.average(supporting.astype(np.float32), weights=weights))


def _estimate_harmonic_bonus(
    *,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    peak_indices: tuple[int, ...],
    peak_index: int,
) -> float:
    """Reward lower-frequency candidates when stronger harmonics align with them, which helps breathing-style motion avoid being outranked by sharper harmonics."""

    if len(peak_indices) <= 1:
        return 0.0
    candidate_frequency = float(frequencies[peak_index])
    candidate_amplitude = float(amplitudes[peak_index])
    if candidate_frequency <= 0.0 or candidate_amplitude <= 1e-6:
        return 0.0
    bin_width = (
        float(frequencies[1] - frequencies[0])
        if frequencies.size > 1
        else 0.1
    )
    bonus = 0.0
    for factor, weight in ((2.0, 0.30), (3.0, 0.12)):
        target_frequency = candidate_frequency * factor
        tolerance = max(bin_width * 2.0, target_frequency * 0.08)
        matching_peaks = [
            index
            for index in peak_indices
            if abs(float(frequencies[index]) - target_frequency) <= tolerance
        ]
        if not matching_peaks:
            continue
        harmonic_amplitude = max(float(amplitudes[index]) for index in matching_peaks)
        bonus += weight * min(1.0, harmonic_amplitude / candidate_amplitude)
    return float(np.clip(bonus, 0.0, 1.0))


def _fallback_band(
    probe: FfprobeMediaInfo, analysis_fps: float
) -> FrequencyBandSuggestion:
    duration_floor_hz = max(0.1, 2.0 / max(probe.duration_seconds, 1.0))
    low_hz = _round_band_edge(max(0.15, duration_floor_hz))
    high_hz = _round_band_edge(
        min((analysis_fps / 2.0) * 0.7, max(low_hz + 0.35, low_hz * 2.5))
    )
    return FrequencyBandSuggestion(
        low_hz=low_hz,
        high_hz=high_hz,
        peak_hz=low_hz,
        analysis_fps=analysis_fps,
        confidence=0.0,
    )


def _estimate_band_edges_from_spectrum(
    *,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    peak_index: int,
    min_hz: float,
    max_hz: float,
) -> tuple[float, float]:
    peak_hz = float(frequencies[peak_index])
    peak_amplitude = float(amplitudes[peak_index])
    low_index = peak_index
    while low_index > 0 and float(amplitudes[low_index]) >= peak_amplitude * 0.50:
        low_index -= 1
    high_index = peak_index
    while high_index < len(amplitudes) - 1 and float(amplitudes[high_index]) >= peak_amplitude * 0.50:
        high_index += 1

    low_hz = max(min_hz, float(frequencies[low_index]))
    high_hz = min(max_hz, float(frequencies[high_index]))
    bin_width = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.1
    min_width = max(0.20, bin_width * 3.0)
    max_width = max(min_width, min(max_hz - min_hz, max(0.75, peak_hz * 0.9)))

    if peak_hz < 0.5:
        max_width = min(max_width, 0.45)
    elif peak_hz > max_hz * 0.65:
        max_width = max(max_width, min(max_hz - min_hz, 0.9))

    if high_hz - low_hz < min_width:
        low_hz = max(min_hz, peak_hz - (min_width / 2.0))
        high_hz = min(max_hz, peak_hz + (min_width / 2.0))
    if high_hz - low_hz > max_width:
        low_hz = max(min_hz, peak_hz - (max_width / 2.0))
        high_hz = min(max_hz, peak_hz + (max_width / 2.0))
    return low_hz, high_hz


def _round_band_edge(value: float) -> float:
    return round(value / 0.05) * 0.05
