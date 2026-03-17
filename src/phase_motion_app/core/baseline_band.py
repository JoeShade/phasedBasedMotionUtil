"""This file owns lightweight source-motion analysis so the shell can suggest a sane starting frequency band on import without running the full phase engine."""

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
    """This model carries the lightweight import-time band suggestion and how strongly the quick analysis supported it."""

    low_hz: float
    high_hz: float
    peak_hz: float
    analysis_fps: float
    confidence: float


def suggest_frequency_band_from_motion_trace(
    motion_trace: np.ndarray,
    *,
    fps: float,
    min_hz: float = 0.1,
    max_hz: float | None = None,
) -> FrequencyBandSuggestion | None:
    """Turn a cheap 1-D motion trace into a conservative starting band for the GUI."""

    if motion_trace.ndim != 1 or motion_trace.shape[0] < 32 or fps <= 0:
        return None

    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel /= kernel.sum()
    smoothed = np.convolve(
        motion_trace.astype(np.float32, copy=False),
        kernel,
        mode="same",
    )
    centered = smoothed - smoothed.mean(dtype=np.float32)
    if np.max(np.abs(centered)) <= 1e-6:
        return None

    frequencies = np.fft.rfftfreq(centered.shape[0], d=1.0 / fps)
    amplitudes = np.abs(np.fft.rfft(centered)).astype(np.float32, copy=False)
    nyquist = fps / 2.0
    search_max_hz = min(max_hz or (nyquist * 0.85), nyquist * 0.85)
    search = (frequencies >= min_hz) & (frequencies <= search_max_hz)
    if np.count_nonzero(search) < 3:
        return None

    search_amplitudes = amplitudes[search]
    search_frequencies = frequencies[search]

    # Improved peak detection: find all peaks above threshold
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(search_amplitudes, height=np.median(search_amplitudes) * 1.2)
    if len(peaks) == 0:
        peak_index = int(np.argmax(search_amplitudes))
    else:
        # Pick the highest peak
        peak_index = peaks[np.argmax(properties['peak_heights'])]

    peak_hz = float(search_frequencies[peak_index])
    peak_amplitude = float(search_amplitudes[peak_index])
    noise_floor = float(np.median(search_amplitudes))
    confidence = float(
        np.clip((peak_amplitude - noise_floor) / max(peak_amplitude, 1e-6), 0.0, 1.0)
    )

    # Adaptive band width: expand if peak is broad or multiple peaks
    band_peaks = [peak_index]
    if len(peaks) > 1:
        # If multiple significant peaks, include them in band
        band_peaks = list(peaks)
        min_band = float(search_frequencies[min(band_peaks)])
        max_band = float(search_frequencies[max(band_peaks)])
        low_hz = max(min_hz, min_band - 0.1)
        high_hz = min(search_max_hz, max_band + 0.1)
    else:
        low_hz, high_hz = _estimate_band_edges_from_spectrum(
            frequencies=search_frequencies,
            amplitudes=search_amplitudes,
            peak_index=peak_index,
            min_hz=min_hz,
            max_hz=search_max_hz,
        )

    # Clamp for low/high frequency cases
    if peak_hz < 0.5:
        low_hz = max(min_hz, peak_hz - 0.1)
        high_hz = min(search_max_hz, peak_hz + 0.5)
    elif peak_hz > nyquist * 0.7:
        low_hz = max(min_hz, peak_hz - 0.5)
        high_hz = min(search_max_hz, peak_hz + 0.1)

    return FrequencyBandSuggestion(
        low_hz=_round_band_edge(low_hz),
        high_hz=_round_band_edge(high_hz),
        peak_hz=peak_hz,
        analysis_fps=fps,
        confidence=confidence,
    )


def analyze_source_frequency_band(
    source_path: Path,
    probe: FfprobeMediaInfo,
) -> FrequencyBandSuggestion:
    """Decode a small grayscale proxy clip and derive a conservative baseline band suggestion."""

    normalization_plan = build_source_normalization_plan(probe)
    analysis_fps = min(normalization_plan.working_fps, 15.0)
    analysis_width = 160
    scaled_height = (
        normalization_plan.working_resolution.height
        * analysis_width
        / normalization_plan.working_resolution.width
    )
    analysis_height = max(2, int(round(scaled_height / 2.0) * 2))
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

    # Keep a small border trim so edge junk does not dominate, but avoid the
    # older center-heavy crop that missed off-center motion sources.
    row_start = int(analysis_height * 0.1)
    row_end = max(row_start + 2, int(analysis_height * 0.9))
    column_start = int(analysis_width * 0.1)
    column_end = max(column_start + 2, int(analysis_width * 0.9))
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
    motion_trace = np.mean(np.abs(np.diff(pooled, axis=0)), axis=(1, 2)).astype(
        np.float32,
        copy=False,
    )
    suggestion = suggest_frequency_band_from_motion_trace(
        motion_trace,
        fps=analysis_fps,
        min_hz=max(0.1, 2.0 / max(probe.duration_seconds, 1.0)),
    )
    if suggestion is None or suggestion.confidence < 0.15:
        return _fallback_band(probe, analysis_fps)
    return suggestion


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
    while low_index > 0 and float(amplitudes[low_index]) >= peak_amplitude * 0.45:
        low_index -= 1
    high_index = peak_index
    while high_index < len(amplitudes) - 1 and float(amplitudes[high_index]) >= peak_amplitude * 0.45:
        high_index += 1

    low_hz = max(min_hz, float(frequencies[low_index]))
    high_hz = min(max_hz, float(frequencies[high_index]))
    bin_width = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.1
    min_width = max(0.25, bin_width * 2.0)
    max_width = max(min_width, min(max_hz - min_hz, max(0.9, peak_hz * 1.2)))

    if high_hz - low_hz < min_width:
        low_hz = max(min_hz, peak_hz - (min_width / 2.0))
        high_hz = min(max_hz, peak_hz + (min_width / 2.0))
    if high_hz - low_hz > max_width:
        low_hz = max(min_hz, peak_hz - (max_width / 2.0))
        high_hz = min(max_hz, peak_hz + (max_width / 2.0))
    return low_hz, high_hz


def _round_band_edge(value: float) -> float:
    return round(value / 0.05) * 0.05
