"""This file owns drift warning policy so visible or estimated drift can block render until the operator acknowledges the reviewed source state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from phase_motion_app.core.models import DriftAcknowledgement


@dataclass(frozen=True)
class DriftEstimate:
    """This model carries one shell-side global drift estimate from the reviewed first/last frames."""

    delta_x_px: float
    delta_y_px: float
    magnitude_px: float
    peak_ratio: float


@dataclass(frozen=True)
class DriftAssessment:
    """This model tracks whether the current reviewed source state has an active drift warning and whether the operator acknowledged it."""

    visible_drift_confirmed: bool = False
    estimated_global_drift_px: float | None = None
    advisory_threshold_px: float = 2.0
    acknowledged: bool = False

    @property
    def warning_active(self) -> bool:
        estimated_warning = (
            self.estimated_global_drift_px is not None
            and self.estimated_global_drift_px >= self.advisory_threshold_px
        )
        return self.visible_drift_confirmed or estimated_warning

    @property
    def can_render(self) -> bool:
        return not self.warning_active or self.acknowledged


def estimate_global_drift(
    first_frame: object,
    last_frame: object,
    *,
    max_dimension: int = 256,
    min_texture_std: float = 1.0,
    min_peak_ratio: float = 1.5,
) -> DriftEstimate | None:
    """Estimate simple global translation from the reviewed first/last frames when the signal is reliable enough."""

    first_gray = _frame_to_grayscale(first_frame)
    last_gray = _frame_to_grayscale(last_frame)
    if first_gray.shape != last_gray.shape:
        raise ValueError("First and last frame dimensions must match for drift estimation.")

    first_small, scale_x, scale_y = _downsample_for_drift(
        first_gray,
        max_dimension=max_dimension,
    )
    last_small, _, _ = _downsample_for_drift(
        last_gray,
        max_dimension=max_dimension,
    )

    first_edges = _gradient_magnitude(first_small)
    last_edges = _gradient_magnitude(last_small)
    first_edges = first_edges - float(first_edges.mean())
    last_edges = last_edges - float(last_edges.mean())
    if float(first_edges.std()) < min_texture_std or float(last_edges.std()) < min_texture_std:
        return None

    peak_row, peak_col, peak_ratio = _phase_correlation_peak(first_edges, last_edges)
    if peak_ratio < min_peak_ratio:
        return None

    shift_x = _unwrap_shift(peak_col, first_edges.shape[1]) * scale_x
    shift_y = _unwrap_shift(peak_row, first_edges.shape[0]) * scale_y
    return DriftEstimate(
        delta_x_px=float(shift_x),
        delta_y_px=float(shift_y),
        magnitude_px=float(math.hypot(shift_x, shift_y)),
        peak_ratio=float(peak_ratio),
    )


def build_drift_acknowledgement(
    assessment: DriftAssessment,
    *,
    source_fingerprint_sha256: str,
    note: str | None = None,
) -> DriftAcknowledgement | None:
    """Create the results-side operator attestation only when a warning was active and the operator acknowledged it."""

    if not assessment.warning_active:
        return None
    if not assessment.acknowledged:
        raise ValueError("Drift warning is active and requires explicit acknowledgement.")
    return DriftAcknowledgement(
        acknowledged=True,
        reviewed_source_fingerprint_sha256=source_fingerprint_sha256,
        note=note,
    )


def _frame_to_grayscale(frame: object) -> np.ndarray:
    width = int(getattr(frame, "width"))
    height = int(getattr(frame, "height"))
    rgb24 = getattr(frame, "rgb24")
    expected_bytes = width * height * 3
    if len(rgb24) != expected_bytes:
        raise ValueError("Frame RGB byte count did not match width and height.")

    rgb = np.frombuffer(rgb24, dtype=np.uint8).reshape(height, width, 3).astype(np.float32)
    return (
        rgb[:, :, 0] * np.float32(0.2126)
        + rgb[:, :, 1] * np.float32(0.7152)
        + rgb[:, :, 2] * np.float32(0.0722)
    )


def _downsample_for_drift(
    grayscale: np.ndarray,
    *,
    max_dimension: int,
) -> tuple[np.ndarray, float, float]:
    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive.")

    step = max(1, math.ceil(max(grayscale.shape) / max_dimension))
    sampled = grayscale[::step, ::step]
    return (
        sampled.astype(np.float32, copy=False),
        grayscale.shape[1] / sampled.shape[1],
        grayscale.shape[0] / sampled.shape[0],
    )


def _gradient_magnitude(grayscale: np.ndarray) -> np.ndarray:
    gradient_x = np.zeros_like(grayscale)
    gradient_y = np.zeros_like(grayscale)
    gradient_x[:, 1:-1] = grayscale[:, 2:] - grayscale[:, :-2]
    gradient_y[1:-1, :] = grayscale[2:, :] - grayscale[:-2, :]
    return np.hypot(gradient_x, gradient_y)


def _phase_correlation_peak(
    first_frame: np.ndarray,
    last_frame: np.ndarray,
) -> tuple[float, float, float]:
    window = np.outer(
        np.hanning(first_frame.shape[0]),
        np.hanning(first_frame.shape[1]),
    ).astype(np.float32)
    first_fft = np.fft.fft2(first_frame * window)
    last_fft = np.fft.fft2(last_frame * window)
    cross_power = first_fft * np.conj(last_fft)
    magnitude = np.abs(cross_power)
    cross_power /= np.where(magnitude < 1e-9, 1.0, magnitude)
    correlation = np.abs(np.fft.ifft2(cross_power))

    peak_row, peak_col = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    peak_ratio = _measure_peak_ratio(correlation, peak_row=peak_row, peak_col=peak_col)

    row_offset = _quadratic_peak_offset(
        correlation[(peak_row - 1) % correlation.shape[0], peak_col],
        correlation[peak_row, peak_col],
        correlation[(peak_row + 1) % correlation.shape[0], peak_col],
    )
    col_offset = _quadratic_peak_offset(
        correlation[peak_row, (peak_col - 1) % correlation.shape[1]],
        correlation[peak_row, peak_col],
        correlation[peak_row, (peak_col + 1) % correlation.shape[1]],
    )
    return peak_row + row_offset, peak_col + col_offset, peak_ratio


def _measure_peak_ratio(
    correlation: np.ndarray,
    *,
    peak_row: int,
    peak_col: int,
    exclusion_radius: int = 2,
) -> float:
    masked = correlation.copy()
    row_start = max(0, peak_row - exclusion_radius)
    row_stop = min(masked.shape[0], peak_row + exclusion_radius + 1)
    col_start = max(0, peak_col - exclusion_radius)
    col_stop = min(masked.shape[1], peak_col + exclusion_radius + 1)
    masked[row_start:row_stop, col_start:col_stop] = 0.0
    second_peak = float(masked.max())
    peak_value = float(correlation[peak_row, peak_col])
    return peak_value / (second_peak + 1e-9)


def _quadratic_peak_offset(previous_value: float, center_value: float, next_value: float) -> float:
    denominator = previous_value - (2.0 * center_value) + next_value
    if abs(denominator) < 1e-9:
        return 0.0
    return 0.5 * (previous_value - next_value) / denominator


def _unwrap_shift(coordinate: float, axis_size: int) -> float:
    half_axis = axis_size / 2.0
    if coordinate > half_axis:
        return coordinate - axis_size
    return coordinate

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
