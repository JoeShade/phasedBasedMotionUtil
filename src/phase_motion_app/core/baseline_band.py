"""This file owns bounded ingest-time source-motion analysis so the shell can suggest a sane starting frequency band without running the full render pipeline."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phase_motion_app.core.ffprobe import FfprobeMediaInfo
from phase_motion_app.core.phase_engine import (
    _build_motion_grid_layout,
    _estimate_local_phase_shifts,
)
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
class _ProxyMotionTrace:
    """This model keeps one proxy trace's spatial context explicit so ingest can prefer interior subject motion over border-heavy drift."""

    trace: np.ndarray
    weight: float
    center_row_fraction: float
    center_col_fraction: float
    is_global: bool = False


@dataclass(frozen=True)
class _SpectrumPeakCandidate:
    """This model keeps one candidate peak explicit so selection stays testable and readable."""

    index: int
    frequency_hz: float
    amplitude: float
    prominence: float
    support_fraction: float
    sharpness: float
    periodicity_strength: float
    harmonic_bonus: float
    ubiquity_penalty: float
    edge_dominance: float
    score: float


@dataclass(frozen=True)
class _ProxyBandScore:
    """This model keeps one ingest-time proxy band score explicit so engine-style vibration promotion stays explainable and testable."""

    low_hz: float
    high_hz: float
    score: float


_INGEST_ANALYSIS_FPS_CAP = 18.0
_INGEST_ANALYSIS_WIDTH = 176
_INGEST_ANALYSIS_MAX_VIDEO_SECONDS = 18.0
_INGEST_ANALYSIS_MIN_VIDEO_SECONDS = 8.0
_PHASE_PROXY_SECONDS = 6.0
_PHASE_PROXY_MIN_FRAMES = 64
_PHASE_PROXY_PROMOTION_SCORE_RATIO = 1.08
_PHASE_PROXY_VIBRATION_BANDS = (
    (0.50, 1.50),
    (0.75, 2.50),
    (1.00, 5.00),
    (1.50, 5.00),
    (2.00, 6.00),
)
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

    suggestion, _candidate = _suggest_frequency_band_from_traces_with_candidate(
        traces=(
            _ProxyMotionTrace(
                trace=motion_trace.astype(np.float32, copy=False),
                weight=1.0,
                center_row_fraction=0.5,
                center_col_fraction=0.5,
            ),
        ),
        fps=fps,
        min_hz=min_hz,
        max_hz=max_hz,
    )
    return suggestion


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
    traces = _build_proxy_motion_traces(pooled)
    suggestion, selected_candidate = _suggest_frequency_band_from_traces_with_candidate(
        traces=traces,
        fps=analysis_fps,
        min_hz=max(0.15, 2.0 / max(probe.duration_seconds, 1.0)),
    )
    if suggestion is None or suggestion.confidence < 0.15:
        return _fallback_band(probe, analysis_fps)
    return _promote_phase_proxy_band_if_warranted(
        baseline_suggestion=suggestion,
        baseline_candidate=selected_candidate,
        grayscale_frames=cropped,
        fps=analysis_fps,
        min_hz=max(0.15, 2.0 / max(probe.duration_seconds, 1.0)),
    )


def _build_proxy_motion_traces(
    grayscale_frames: np.ndarray,
) -> tuple[_ProxyMotionTrace, ...]:
    """Build a small set of signed motion traces from gradient-projected local cells so ingest can see real motion rather than a single whole-frame difference curve."""

    frame_count, height, width = grayscale_frames.shape
    if frame_count < 32 or height < 8 or width < 8:
        return ()

    reference = grayscale_frames.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    grad_y, grad_x = np.gradient(reference)
    row_edges = np.linspace(0, height, int(np.clip(round(height / 18.0), 4, 6)) + 1)
    column_edges = np.linspace(0, width, int(np.clip(round(width / 18.0), 5, 8)) + 1)
    row_edges = np.round(row_edges).astype(np.int32)
    column_edges = np.round(column_edges).astype(np.int32)

    traces: list[_ProxyMotionTrace] = []
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
            center_row_fraction = ((row_start + row_end) / 2.0) / height
            center_col_fraction = ((column_start + column_end) / 2.0) / width
            center_weight = _estimate_proxy_center_weight(
                center_row_fraction=center_row_fraction,
                center_col_fraction=center_col_fraction,
            )
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
                traces.append(
                    _ProxyMotionTrace(
                        trace=trace.astype(np.float32, copy=False),
                        weight=texture_strength * trace_rms * center_weight,
                        center_row_fraction=center_row_fraction,
                        center_col_fraction=center_col_fraction,
                    )
                )

    # Keep one whole-frame trace as a cheap guard against all local cells missing
    # a broad moving subject, but keep it light so it does not drown out
    # interior local motion with whole-frame sway.
    global_diff = grayscale_frames - reference[None, :, :]
    global_trace = np.mean(global_diff * grad_x[None, :, :], axis=(1, 2), dtype=np.float32)
    global_rms = float(np.sqrt(np.mean(np.square(global_trace.astype(np.float32, copy=False)))))
    if global_rms > 1e-5:
        traces.append(
            _ProxyMotionTrace(
                trace=global_trace.astype(np.float32, copy=False),
                weight=global_rms * np.float32(0.15),
                center_row_fraction=0.5,
                center_col_fraction=0.5,
                is_global=True,
            )
        )

    return tuple(traces)


def _suggest_frequency_band_from_traces(
    *,
    traces: tuple[_ProxyMotionTrace, ...],
    fps: float,
    min_hz: float,
    max_hz: float | None = None,
) -> FrequencyBandSuggestion | None:
    """Aggregate one or more motion traces into a single robust spectrum and suggest a band from the strongest supported peak."""

    suggestion, _candidate = _suggest_frequency_band_from_traces_with_candidate(
        traces=traces,
        fps=fps,
        min_hz=min_hz,
        max_hz=max_hz,
    )
    return suggestion


def _suggest_frequency_band_from_traces_with_candidate(
    *,
    traces: tuple[_ProxyMotionTrace, ...],
    fps: float,
    min_hz: float,
    max_hz: float | None = None,
) -> tuple[FrequencyBandSuggestion | None, _SpectrumPeakCandidate | None]:
    """Return the trace-derived suggestion plus the selected peak so ingest can optionally run one heavier promotion pass for suspicious low shelves."""

    if not traces or fps <= 0:
        return None, None

    prepared_traces: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    differentiated_spectra: list[np.ndarray] = []
    spectrum_weights: list[float] = []
    trace_positions: list[tuple[float, float]] = []
    global_trace_flags: list[bool] = []
    frequencies: np.ndarray | None = None
    for proxy_trace in traces:
        prepared = _prepare_trace_for_spectrum(proxy_trace.trace)
        if prepared is None:
            continue
        spectrum = np.abs(np.fft.rfft(prepared)).astype(np.float32, copy=False)
        differentiated = np.diff(prepared, prepend=prepared[:1]).astype(np.float32, copy=False)
        differentiated -= differentiated.mean(dtype=np.float32)
        differentiated_spectrum = np.abs(np.fft.rfft(differentiated)).astype(
            np.float32,
            copy=False,
        )
        if frequencies is None:
            frequencies = np.fft.rfftfreq(prepared.shape[0], d=1.0 / fps)
        prepared_traces.append(prepared)
        spectra.append(spectrum)
        differentiated_spectra.append(differentiated_spectrum)
        spectrum_weights.append(max(float(proxy_trace.weight), 1e-6))
        trace_positions.append(
            (proxy_trace.center_row_fraction, proxy_trace.center_col_fraction)
        )
        global_trace_flags.append(proxy_trace.is_global)

    if not spectra or frequencies is None:
        return None, None

    weighted_spectra = np.stack(spectra, axis=0)
    weighted_differentiated_spectra = np.stack(differentiated_spectra, axis=0)
    weight_array = np.asarray(spectrum_weights, dtype=np.float32)
    aggregate = np.average(weighted_spectra, axis=0, weights=weight_array).astype(
        np.float32,
        copy=False,
    )
    differentiated_aggregate = np.average(
        weighted_differentiated_spectra,
        axis=0,
        weights=weight_array,
    ).astype(np.float32, copy=False)
    nyquist = fps / 2.0
    search_max_hz = min(max_hz or (nyquist * 0.85), nyquist * 0.85)
    search = (frequencies >= min_hz) & (frequencies <= search_max_hz)
    if np.count_nonzero(search) < 3:
        return None, None

    search_frequencies = frequencies[search]
    search_amplitudes = aggregate[search]
    search_periodicity_amplitudes = differentiated_aggregate[search]
    search_cell_spectra = weighted_spectra[:, search]
    candidates = _build_peak_candidates(
        frequencies=search_frequencies,
        amplitudes=search_amplitudes,
        cell_spectra=search_cell_spectra,
        periodicity_amplitudes=search_periodicity_amplitudes,
        trace_weights=weight_array,
        trace_positions=np.asarray(trace_positions, dtype=np.float32),
        global_trace_flags=np.asarray(global_trace_flags, dtype=bool),
    )
    candidate = _select_peak_candidate(candidates)
    if candidate is None:
        return None, None

    low_hz, high_hz = _estimate_band_edges_from_spectrum(
        frequencies=search_frequencies,
        amplitudes=search_amplitudes,
        peak_index=candidate.index,
        min_hz=min_hz,
        max_hz=search_max_hz,
    )
    low_hz, high_hz = _tighten_periodic_candidate_band(
        low_hz=low_hz,
        high_hz=high_hz,
        candidate=candidate,
        min_hz=min_hz,
        max_hz=search_max_hz,
    )
    low_hz, high_hz = _extend_band_with_related_higher_frequency_peak(
        low_hz=low_hz,
        high_hz=high_hz,
        selected_candidate=candidate,
        candidates=candidates,
        frequencies=search_frequencies,
        amplitudes=search_amplitudes,
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
            )
            + 0.08 * candidate.periodicity_strength
            + 0.05 * candidate.sharpness
            + 0.08 * (1.0 - candidate.edge_dominance)
            - 0.10 * candidate.ubiquity_penalty,
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
    ), candidate


def _promote_phase_proxy_band_if_warranted(
    *,
    baseline_suggestion: FrequencyBandSuggestion,
    baseline_candidate: _SpectrumPeakCandidate | None,
    grayscale_frames: np.ndarray,
    fps: float,
    min_hz: float,
) -> FrequencyBandSuggestion:
    """Run one heavier proxy-motion check only when the cheap trace picker landed on a suspicious low shelf."""

    if baseline_candidate is None or not _is_suspicious_low_frequency_candidate(
        baseline_candidate
    ):
        return baseline_suggestion

    search_max_hz = min(fps * 0.45, 6.0)
    if search_max_hz <= 1.0:
        return baseline_suggestion
    candidate_scores = _score_phase_proxy_bands(
        grayscale_frames=grayscale_frames,
        fps=fps,
        min_hz=min_hz,
        max_hz=search_max_hz,
        baseline_suggestion=baseline_suggestion,
    )
    if not candidate_scores:
        return baseline_suggestion

    baseline_score = max(
        (
            score.score
            for score in candidate_scores
            if abs(score.low_hz - baseline_suggestion.low_hz) <= 1e-6
            and abs(score.high_hz - baseline_suggestion.high_hz) <= 1e-6
        ),
        default=0.0,
    )
    higher_frequency_candidates = [
        score for score in candidate_scores if score.low_hz >= 0.95
    ]
    if not higher_frequency_candidates:
        return baseline_suggestion
    promoted = max(
        higher_frequency_candidates,
        key=lambda score: (score.score, score.high_hz, -score.low_hz),
    )
    required_score = max(
        baseline_score * _PHASE_PROXY_PROMOTION_SCORE_RATIO,
        baseline_score + 0.02,
    )
    if promoted.score < required_score:
        return baseline_suggestion

    # When the cheap path locked onto broad sway, the local motion proxy can
    # legitimately say "the strongest compact oscillation actually lives in a
    # faster family". Promote that whole family instead of just nudging the
    # upper edge by one bin.
    promoted_peak = min(
        promoted.high_hz,
        max(promoted.low_hz, (promoted.low_hz + promoted.high_hz) / 2.0),
    )
    promoted_confidence = float(
        np.clip(
            max(
                baseline_suggestion.confidence,
                baseline_suggestion.confidence + 0.12 + (promoted.score - baseline_score),
            ),
            0.0,
            1.0,
        )
    )
    return FrequencyBandSuggestion(
        low_hz=_round_band_edge(promoted.low_hz),
        high_hz=_round_band_edge(promoted.high_hz),
        peak_hz=promoted_peak,
        analysis_fps=fps,
        confidence=promoted_confidence,
    )


def _is_suspicious_low_frequency_candidate(candidate: _SpectrumPeakCandidate) -> bool:
    """Detect the cheap-path failure mode where broad low-frequency sway is more likely than the intended subject vibration."""

    if candidate.frequency_hz >= 0.85:
        return False
    if candidate.ubiquity_penalty >= 0.12:
        return True
    if candidate.edge_dominance >= 0.55:
        return True
    return candidate.support_fraction >= 0.60 and candidate.sharpness < 0.18


def _score_phase_proxy_bands(
    *,
    grayscale_frames: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
    baseline_suggestion: FrequencyBandSuggestion,
) -> tuple[_ProxyBandScore, ...]:
    """Score a few heavier local-motion proxy bands so ingest can rescue engine-style vibration without turning every clip into a high-band guess."""

    frame_budget = min(
        grayscale_frames.shape[0],
        max(_PHASE_PROXY_MIN_FRAMES, int(round(fps * _PHASE_PROXY_SECONDS))),
    )
    if (
        frame_budget < _PHASE_PROXY_MIN_FRAMES
        or grayscale_frames.shape[1] < 24
        or grayscale_frames.shape[2] < 24
    ):
        return ()
    phase_frames = grayscale_frames[:frame_budget].astype(np.float32, copy=False)
    layout = _build_motion_grid_layout(
        height=phase_frames.shape[1],
        width=phase_frames.shape[2],
        sigma_scale=1.0,
    )
    displacement_x, displacement_y, confidence = _estimate_local_phase_shifts(
        phase_frames,
        layout,
        progress_callback=None,
    )
    confidence_flat = confidence.reshape(-1)
    if float(np.max(confidence_flat)) <= 1e-6:
        return ()
    x_traces = displacement_x.reshape(displacement_x.shape[0], -1)
    y_traces = displacement_y.reshape(displacement_y.shape[0], -1)
    use_x_trace = x_traces.std(axis=0) >= y_traces.std(axis=0)
    traces = np.where(use_x_trace[None, :], x_traces, y_traces).astype(
        np.float32,
        copy=False,
    )
    traces -= traces.mean(axis=0, keepdims=True, dtype=np.float32)
    spectra = np.square(np.abs(np.fft.rfft(traces, axis=0))).astype(
        np.float32,
        copy=False,
    )
    frequencies = np.fft.rfftfreq(traces.shape[0], d=1.0 / fps).astype(
        np.float32,
        copy=False,
    )
    total_energy = np.sum(spectra, axis=0).astype(np.float32, copy=False) + np.float32(1e-8)
    row_coords = np.repeat(np.arange(len(layout.row_starts)), len(layout.column_starts))
    column_coords = np.tile(np.arange(len(layout.column_starts)), len(layout.row_starts))
    center_weights = np.array(
        [
            _estimate_proxy_center_weight(
                center_row_fraction=(row_index + 0.5) / max(len(layout.row_starts), 1),
                center_col_fraction=(column_index + 0.5) / max(len(layout.column_starts), 1),
            )
            for row_index, column_index in zip(row_coords, column_coords, strict=True)
        ],
        dtype=np.float32,
    )
    scores: list[_ProxyBandScore] = []
    for low_hz, high_hz in _phase_proxy_candidate_bands(
        min_hz=min_hz,
        max_hz=max_hz,
        baseline_suggestion=baseline_suggestion,
    ):
        band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
        if int(np.count_nonzero(band_mask)) < 2:
            continue
        band_energy = np.sum(spectra[band_mask], axis=0).astype(np.float32, copy=False)
        normalized_energy = band_energy / total_energy
        cell_score = (
            np.sqrt(np.clip(normalized_energy, 0.0, 1.0)).astype(np.float32, copy=False)
            * np.sqrt(np.clip(confidence_flat, 0.0, 1.0)).astype(np.float32, copy=False)
            * center_weights
        )
        grid = cell_score.reshape(len(layout.row_starts), len(layout.column_starts))
        score = _max_local_patch_score(grid)
        scores.append(
            _ProxyBandScore(
                low_hz=low_hz,
                high_hz=high_hz,
                score=score,
            )
        )
    return tuple(scores)


def _phase_proxy_candidate_bands(
    *,
    min_hz: float,
    max_hz: float,
    baseline_suggestion: FrequencyBandSuggestion,
) -> tuple[tuple[float, float], ...]:
    """Build a tiny candidate set so the heavier proxy stays bounded and focused on the realistic decision points."""

    unique_candidates: dict[tuple[float, float], None] = {}
    for low_hz, high_hz in (
        (baseline_suggestion.low_hz, baseline_suggestion.high_hz),
        *_PHASE_PROXY_VIBRATION_BANDS,
    ):
        clipped_low = float(np.clip(low_hz, min_hz, max_hz))
        clipped_high = float(np.clip(high_hz, min_hz, max_hz))
        if clipped_high <= clipped_low + 0.10:
            continue
        rounded = (_round_band_edge(clipped_low), _round_band_edge(clipped_high))
        unique_candidates[rounded] = None
    return tuple(unique_candidates.keys())


def _max_local_patch_score(grid: np.ndarray) -> float:
    """Score the strongest small patch instead of whole-frame support so localized vibration can outrank broad sway when it is genuinely stronger in one compact region."""

    if grid.size == 0:
        return 0.0
    smoothed = np.zeros_like(grid, dtype=np.float32)
    for row_index in range(grid.shape[0]):
        row_start = max(0, row_index - 1)
        row_stop = min(grid.shape[0], row_index + 2)
        for column_index in range(grid.shape[1]):
            column_start = max(0, column_index - 1)
            column_stop = min(grid.shape[1], column_index + 2)
            smoothed[row_index, column_index] = float(
                np.mean(grid[row_start:row_stop, column_start:column_stop])
            )
    return float(np.max(smoothed))


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


def _build_peak_candidates(
    *,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    cell_spectra: np.ndarray,
    periodicity_amplitudes: np.ndarray,
    trace_weights: np.ndarray,
    trace_positions: np.ndarray,
    global_trace_flags: np.ndarray,
) -> list[_SpectrumPeakCandidate]:
    """Score candidate peaks with neighborhood prominence, periodicity, and spatial support so ingest does not lock onto border-heavy motion when the subject movement is more central."""

    if amplitudes.size == 0 or np.max(amplitudes) <= 1e-6:
        return []

    peak_indices = _find_local_peak_indices(amplitudes)
    global_max = float(np.max(amplitudes))
    periodicity_max = float(np.max(periodicity_amplitudes)) if periodicity_amplitudes.size > 0 else 0.0
    candidates: list[_SpectrumPeakCandidate] = []
    for peak_index in peak_indices:
        amplitude = float(amplitudes[peak_index])
        frequency_hz = float(frequencies[peak_index])
        prominence = _estimate_peak_prominence(amplitudes, peak_index)
        support_fraction = _estimate_support_fraction(cell_spectra, peak_index)
        sharpness = _estimate_peak_sharpness(amplitudes, peak_index)
        edge_dominance = _estimate_peak_edge_dominance(
            cell_spectra=cell_spectra,
            peak_index=peak_index,
            trace_weights=trace_weights,
            trace_positions=trace_positions,
            global_trace_flags=global_trace_flags,
        )
        periodicity_strength = (
            float(periodicity_amplitudes[peak_index]) / max(periodicity_max, 1e-6)
            if periodicity_max > 1e-6
            else 0.0
        )
        ubiquity_penalty = _estimate_low_frequency_sway_penalty(
            frequency_hz=frequency_hz,
            support_fraction=support_fraction,
        )
        harmonic_bonus = _estimate_harmonic_bonus(
            frequencies=frequencies,
            amplitudes=amplitudes,
            peak_indices=peak_indices,
            peak_index=peak_index,
        )
        base_score = (
            0.55 * (amplitude / max(global_max, 1e-6))
            + 0.20 * support_fraction
            + 0.15 * (prominence / max(amplitude, 1e-6))
            + 0.10 * harmonic_bonus
        )
        score = (
            base_score
            + 0.08 * periodicity_strength
            + 0.08 * sharpness
            + 0.10 * (1.0 - edge_dominance)
            - 0.18 * ubiquity_penalty
        )
        candidates.append(
            _SpectrumPeakCandidate(
                index=int(peak_index),
                frequency_hz=frequency_hz,
                amplitude=amplitude,
                prominence=prominence,
                support_fraction=support_fraction,
                sharpness=sharpness,
                periodicity_strength=periodicity_strength,
                harmonic_bonus=harmonic_bonus,
                ubiquity_penalty=ubiquity_penalty,
                edge_dominance=edge_dominance,
                score=float(score),
            )
        )
    return candidates


def _select_peak_candidate(
    candidates: list[_SpectrumPeakCandidate],
) -> _SpectrumPeakCandidate | None:
    """Choose one scored candidate so higher-frequency near-ties can be handled separately from scoring."""

    if not candidates:
        return None
    return _prefer_sharper_higher_frequency_candidate(candidates)


def _prefer_sharper_higher_frequency_candidate(
    candidates: list[_SpectrumPeakCandidate],
) -> _SpectrumPeakCandidate:
    """Break near-ties in favor of a sharper faster candidate when the lower winner looks like frame-wide sway rather than the target oscillation."""

    best = max(candidates, key=lambda candidate: candidate.score)
    if best.frequency_hz >= 0.5:
        return best

    # The engine clip failure mode was a broad slow shelf winning by support alone.
    # If a clearly sharper higher-frequency candidate is essentially tied, prefer it.
    alternatives = [
        candidate
        for candidate in candidates
        if candidate.frequency_hz >= max(0.5, best.frequency_hz * 1.4)
        and candidate.score >= (best.score - 0.03)
        and candidate.sharpness >= (best.sharpness + 0.05)
        and candidate.periodicity_strength >= (best.periodicity_strength + 0.08)
        and candidate.edge_dominance <= (best.edge_dominance - 0.05)
    ]
    if not alternatives:
        return best
    return max(
        alternatives,
        key=lambda candidate: (
            candidate.score,
            candidate.sharpness,
            candidate.periodicity_strength,
            candidate.frequency_hz,
        ),
    )


def _extend_band_with_related_higher_frequency_peak(
    *,
    low_hz: float,
    high_hz: float,
    selected_candidate: _SpectrumPeakCandidate,
    candidates: list[_SpectrumPeakCandidate],
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    min_hz: float,
    max_hz: float,
) -> tuple[float, float]:
    """Widen the upper edge when a low-frequency winner has a credible higher-frequency family behind it, which helps engine-style motion avoid being truncated to the slow fundamental alone."""

    if selected_candidate.frequency_hz >= 0.5:
        return low_hz, high_hz
    if selected_candidate.edge_dominance > 0.60:
        return low_hz, high_hz
    if (
        selected_candidate.ubiquity_penalty < 0.10
        and selected_candidate.support_fraction < 0.75
    ):
        return low_hz, high_hz

    upper_family_limit_hz = max(0.85, selected_candidate.frequency_hz * 2.8)
    extension_candidates = [
        candidate
        for candidate in candidates
        if candidate.frequency_hz >= max(0.5, selected_candidate.frequency_hz * 1.4)
        and candidate.frequency_hz <= upper_family_limit_hz
        and candidate.score >= (selected_candidate.score - 0.20)
        and candidate.support_fraction >= 0.30
        and candidate.periodicity_strength >= max(0.35, selected_candidate.periodicity_strength)
        and candidate.edge_dominance <= (selected_candidate.edge_dominance + 0.10)
    ]
    if not extension_candidates:
        return low_hz, high_hz

    extension_candidate = max(
        extension_candidates,
        key=lambda candidate: (
            candidate.frequency_hz,
            candidate.score,
            candidate.periodicity_strength,
            candidate.sharpness,
        ),
    )
    extension_low, extension_high = _estimate_band_edges_from_spectrum(
        frequencies=frequencies,
        amplitudes=amplitudes,
        peak_index=extension_candidate.index,
        min_hz=min_hz,
        max_hz=max_hz,
    )
    extension_low, extension_high = _tighten_periodic_candidate_band(
        low_hz=extension_low,
        high_hz=extension_high,
        candidate=extension_candidate,
        min_hz=min_hz,
        max_hz=max_hz,
    )
    return low_hz, max(high_hz, extension_high)


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


def _estimate_peak_sharpness(amplitudes: np.ndarray, peak_index: int) -> float:
    """Estimate how much one peak rises above its immediate shoulder so broad shelves do not outrank tighter periodic peaks by support alone."""

    neighborhood = amplitudes[max(0, peak_index - 2) : min(amplitudes.size, peak_index + 3)]
    if neighborhood.size == 0:
        return 0.0
    peak_amplitude = float(amplitudes[peak_index])
    shoulder_mean = float(np.mean(neighborhood))
    return float(np.clip(1.0 - (shoulder_mean / max(peak_amplitude, 1e-6)), 0.0, 1.0))


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


def _estimate_peak_edge_dominance(
    *,
    cell_spectra: np.ndarray,
    peak_index: int,
    trace_weights: np.ndarray,
    trace_positions: np.ndarray,
    global_trace_flags: np.ndarray,
) -> float:
    """Estimate how much one candidate band lives on the outer frame so ingest can avoid edge-led picks when interior motion is stronger enough."""

    if cell_spectra.size == 0 or trace_weights.size == 0 or trace_positions.size == 0:
        return 0.0
    window_start = max(0, peak_index - 1)
    window_stop = min(cell_spectra.shape[1], peak_index + 2)
    peak_values = np.max(cell_spectra[:, window_start:window_stop], axis=1).astype(
        np.float32,
        copy=False,
    )
    weighted_values = peak_values * np.clip(trace_weights.astype(np.float32, copy=False), 1e-6, None)
    local_mask = ~global_trace_flags
    if np.any(local_mask):
        weighted_values = weighted_values[local_mask]
        trace_positions = trace_positions[local_mask]
    if weighted_values.size == 0:
        return 0.0
    edge_mask = (
        (trace_positions[:, 0] < np.float32(0.18))
        | (trace_positions[:, 0] > np.float32(0.82))
        | (trace_positions[:, 1] < np.float32(0.18))
        | (trace_positions[:, 1] > np.float32(0.82))
    )
    if not np.any(edge_mask):
        return 0.0
    return float(
        np.sum(weighted_values[edge_mask], dtype=np.float32)
        / max(np.sum(weighted_values, dtype=np.float32), np.float32(1e-6))
    )


def _estimate_low_frequency_sway_penalty(*, frequency_hz: float, support_fraction: float) -> float:
    """Softly penalize very low-frequency peaks when almost every cell agrees, because that pattern usually means broad sway or camera drift rather than the intended localized motion."""

    support_excess = np.clip((support_fraction - 0.80) / 0.15, 0.0, 1.0)
    low_frequency_weight = np.clip((0.65 - frequency_hz) / 0.30, 0.0, 1.0)
    return float(support_excess * low_frequency_weight)


def _estimate_proxy_center_weight(
    *,
    center_row_fraction: float,
    center_col_fraction: float,
) -> float:
    """Downweight border cells softly so strong outer-frame edges do not dominate the ingest suggestion for centered subjects."""

    normalized_distance = np.sqrt(
        ((center_row_fraction - 0.5) / 0.5) ** 2
        + ((center_col_fraction - 0.5) / 0.5) ** 2
    )
    return float(np.clip(1.10 - (0.70 * normalized_distance), 0.20, 1.0))


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


def _tighten_periodic_candidate_band(
    *,
    low_hz: float,
    high_hz: float,
    candidate: _SpectrumPeakCandidate,
    min_hz: float,
    max_hz: float,
) -> tuple[float, float]:
    """Tighten promoted periodic bands so a better peak does not widen back out and re-include the slow shelf we were trying to avoid."""

    current_width = high_hz - low_hz
    target_half_width: float | None = None
    if candidate.frequency_hz >= 0.5:
        if candidate.periodicity_strength < 0.40 and candidate.sharpness < 0.20:
            return low_hz, high_hz
        target_half_width = min(0.18, max(0.12, candidate.frequency_hz * 0.22))
    elif (
        candidate.edge_dominance <= 0.55
        and candidate.support_fraction >= 0.55
        and candidate.sharpness >= 0.12
    ):
        # When a slower candidate is backed by interior support rather than
        # whole-frame sway, keep the band focused around that subject motion
        # instead of widening all the way back out to the search floor.
        target_half_width = min(0.14, max(0.10, candidate.frequency_hz * 0.34))

    if target_half_width is None:
        return low_hz, high_hz
    target_width = target_half_width * 2.0
    if current_width <= target_width:
        return low_hz, high_hz

    tightened_low = max(min_hz, candidate.frequency_hz - target_half_width)
    tightened_high = min(max_hz, candidate.frequency_hz + target_half_width)
    return tightened_low, tightened_high


def _round_band_edge(value: float) -> float:
    return round(value / 0.05) * 0.05
