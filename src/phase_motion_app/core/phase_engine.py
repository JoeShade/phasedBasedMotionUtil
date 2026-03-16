"""This file owns the numeric motion-amplification core so the worker can run a real phase-derived pipeline without dragging numeric code into the PyQt shell."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _MotionGridLayout:
    """This model keeps the coarse motion-estimation grid explicit and reproducible."""

    tile_size: int
    row_starts: tuple[int, ...]
    column_starts: tuple[int, ...]


@dataclass(frozen=True)
class _MotionReferenceTile:
    """This model keeps the per-tile reference FFT stable so chunked processing does not rebuild it repeatedly."""

    reference_fft: np.ndarray | None
    texture_strength: float


@dataclass(frozen=True)
class _MotionReference:
    """This model captures the static reference information that streaming chunk processing reuses."""

    height: int
    width: int
    layout: _MotionGridLayout
    window: np.ndarray
    tiles: tuple[tuple[_MotionReferenceTile, ...], ...]


class _StreamingBandpassFilter:
    """This helper applies a simple stateful temporal band-pass so chunk windows can be processed sequentially."""

    def __init__(self, *, grid_shape: tuple[int, int], fps: float, low_hz: float, high_hz: float) -> None:
        self._alpha_low = _lowpass_alpha(fps=fps, cutoff_hz=low_hz)
        self._alpha_high = _lowpass_alpha(fps=fps, cutoff_hz=high_hz)
        self._low_state = np.zeros(grid_shape, dtype=np.float32)
        self._high_state = np.zeros(grid_shape, dtype=np.float32)

    def filter_chunk(
        self,
        signal: np.ndarray,
        *,
        progress_callback: Callable[[str], None] | None = None,
        progress_prefix: str = "temporal",
    ) -> np.ndarray:
        filtered = np.empty_like(signal, dtype=np.float32)
        frame_count = signal.shape[0]
        progress_step = max(1, frame_count // 8)
        for frame_index in range(frame_count):
            sample = signal[frame_index]
            self._high_state = self._high_state + self._alpha_high * (
                sample - self._high_state
            )
            self._low_state = self._low_state + self._alpha_low * (
                sample - self._low_state
            )
            filtered[frame_index] = self._high_state - self._low_state
            if progress_callback is not None and (
                frame_index == frame_count - 1 or frame_index % progress_step == 0
            ):
                progress_callback(
                    f"{progress_prefix}_frame_{frame_index + 1}_of_{frame_count}"
                )
        return filtered


class StreamingPhaseAmplifier:
    """This helper keeps only one chunk window in memory while preserving a global reference frame and temporal filter state across chunks."""

    def __init__(
        self,
        *,
        reference_luma: np.ndarray,
        fps: float,
        low_hz: float,
        high_hz: float,
        magnification: float,
        sigma: float = 1.0,
        attenuate_other_frequencies: bool = True,
    ) -> None:
        if reference_luma.ndim != 2:
            raise ValueError("reference_luma must have shape [height, width].")
        if magnification <= 0:
            raise ValueError("magnification must be positive.")

        self._reference = _build_motion_reference(
            reference_luma.astype(np.float32, copy=False),
            sigma_scale=max(sigma, 0.1),
        )
        grid_shape = (
            len(self._reference.layout.row_starts),
            len(self._reference.layout.column_starts),
        )
        self._x_filter = _StreamingBandpassFilter(
            grid_shape=grid_shape,
            fps=fps,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        self._y_filter = _StreamingBandpassFilter(
            grid_shape=grid_shape,
            fps=fps,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        self._magnification = np.float32(magnification)
        self._attenuate_other_frequencies = attenuate_other_frequencies

    def process_chunk(
        self,
        frames_rgb: np.ndarray,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> np.ndarray:
        """Amplify one bounded chunk while preserving the running temporal filter state."""

        if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
            raise ValueError("frames_rgb must have shape [time, height, width, 3].")
        if frames_rgb.shape[0] < 1:
            return frames_rgb.astype(np.float32, copy=True)

        if (
            frames_rgb.shape[1] != self._reference.height
            or frames_rgb.shape[2] != self._reference.width
        ):
            raise ValueError("frames_rgb must match the reference geometry.")

        working = frames_rgb.astype(np.float32, copy=False)
        luma = _rgb_to_luma(working)
        displacement_x, displacement_y, confidence = _estimate_local_phase_shifts_against_reference(
            luma,
            self._reference,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback("motion_grid_done")

        bandpassed_x = self._x_filter.filter_chunk(
            displacement_x,
            progress_callback=progress_callback,
            progress_prefix="x_temporal_band",
        )
        if progress_callback is not None:
            progress_callback("x_temporal_band_done")

        bandpassed_y = self._y_filter.filter_chunk(
            displacement_y,
            progress_callback=progress_callback,
            progress_prefix="y_temporal_band",
        )
        if progress_callback is not None:
            progress_callback("y_temporal_band_done")

        if self._attenuate_other_frequencies:
            phase_motion_x = bandpassed_x
            phase_motion_y = bandpassed_y
        else:
            phase_motion_x = bandpassed_x + np.float32(0.25) * (
                displacement_x - displacement_x.mean(axis=0, dtype=np.float32)
            )
            phase_motion_y = bandpassed_y + np.float32(0.25) * (
                displacement_y - displacement_y.mean(axis=0, dtype=np.float32)
            )

        confidence_scale = _normalize_confidence(confidence)
        motion_gain = np.float32(self._magnification * 1.4)
        displacement_x_grid = phase_motion_x * confidence_scale[None, :, :] * motion_gain
        displacement_y_grid = phase_motion_y * confidence_scale[None, :, :] * motion_gain
        max_displacement_px = np.float32(
            max(1.0, min(min(luma.shape[1], luma.shape[2]) / 24.0, 6.0))
        )
        amplified_rgb = _warp_rgb_frames(
            working,
            displacement_x_grid=displacement_x_grid,
            displacement_y_grid=displacement_y_grid,
            layout=self._reference.layout,
            max_displacement_px=max_displacement_px,
            progress_callback=progress_callback,
            progress_prefix="warp",
        )
        if progress_callback is not None:
            progress_callback("warp_done")
        return np.clip(amplified_rgb, 0.0, 1.0).astype(np.float32, copy=False)


def amplify_motion_rgb(
    frames_rgb: np.ndarray,
    *,
    fps: float,
    low_hz: float,
    high_hz: float,
    magnification: float,
    sigma: float = 1.0,
    attenuate_other_frequencies: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Amplify motion by phase-correlating local windows, filtering those motions over time, and warping the original RGB clip with the amplified displacement field."""

    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError("frames_rgb must have shape [time, height, width, 3].")
    if frames_rgb.shape[0] < 2 or magnification <= 0:
        return frames_rgb.copy()

    working = frames_rgb.astype(np.float32, copy=False)
    luma = _rgb_to_luma(working)

    layout = _build_motion_grid_layout(
        height=luma.shape[1],
        width=luma.shape[2],
        sigma_scale=max(sigma, 0.1),
    )
    displacement_x, displacement_y, confidence = _estimate_local_phase_shifts(
        luma,
        layout,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback("motion_grid_done")

    bandpassed_x = _temporal_bandpass(
        displacement_x,
        fps=fps,
        low_hz=low_hz,
        high_hz=high_hz,
        progress_callback=progress_callback,
        progress_prefix="x_temporal_band",
    )
    if progress_callback is not None:
        progress_callback("x_temporal_band_done")

    bandpassed_y = _temporal_bandpass(
        displacement_y,
        fps=fps,
        low_hz=low_hz,
        high_hz=high_hz,
        progress_callback=progress_callback,
        progress_prefix="y_temporal_band",
    )
    if progress_callback is not None:
        progress_callback("y_temporal_band_done")

    if attenuate_other_frequencies:
        phase_motion_x = bandpassed_x
        phase_motion_y = bandpassed_y
    else:
        phase_motion_x = bandpassed_x + np.float32(0.25) * (
            displacement_x - displacement_x.mean(axis=0, dtype=np.float32)
        )
        phase_motion_y = bandpassed_y + np.float32(0.25) * (
            displacement_y - displacement_y.mean(axis=0, dtype=np.float32)
        )

    confidence_scale = _normalize_confidence(confidence)
    # Local phase-correlation shifts undershoot subtle subpixel motion unless the
    # amplified field gets a small extra lift before resampling.
    motion_gain = np.float32(magnification * 1.4)
    displacement_x_grid = phase_motion_x * confidence_scale[None, :, :] * motion_gain
    displacement_y_grid = phase_motion_y * confidence_scale[None, :, :] * motion_gain

    max_displacement_px = np.float32(
        max(1.0, min(min(luma.shape[1], luma.shape[2]) / 24.0, 6.0))
    )
    amplified_rgb = _warp_rgb_frames(
        working,
        displacement_x_grid=displacement_x_grid,
        displacement_y_grid=displacement_y_grid,
        layout=layout,
        max_displacement_px=max_displacement_px,
        progress_callback=progress_callback,
        progress_prefix="warp",
    )
    if progress_callback is not None:
        progress_callback("warp_done")
    return np.clip(amplified_rgb, 0.0, 1.0).astype(np.float32, copy=False)


def _build_motion_grid_layout(
    *, height: int, width: int, sigma_scale: float
) -> _MotionGridLayout:
    shortest_side = min(height, width)
    tile_size = max(16, int(round(shortest_side / 6.0 * max(sigma_scale, 0.75))))
    tile_size = min(tile_size, shortest_side)
    if tile_size % 2 != 0:
        tile_size += -1 if tile_size == shortest_side else 1
    tile_size = max(8, tile_size)
    step = max(4, tile_size // 2)
    row_starts = _build_axis_starts(length=height, tile_size=tile_size, step=step)
    column_starts = _build_axis_starts(length=width, tile_size=tile_size, step=step)
    return _MotionGridLayout(
        tile_size=tile_size,
        row_starts=tuple(row_starts),
        column_starts=tuple(column_starts),
    )


def _build_axis_starts(*, length: int, tile_size: int, step: int) -> list[int]:
    if tile_size >= length:
        return [0]
    starts = list(range(0, length - tile_size + 1, step))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _estimate_local_phase_shifts(
    luma: np.ndarray,
    layout: _MotionGridLayout,
    *,
    progress_callback: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_frame = luma.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    reference = _build_motion_reference_from_layout(reference_frame, layout=layout)
    return _estimate_local_phase_shifts_against_reference(
        luma,
        reference,
        progress_callback=progress_callback,
    )


def _build_motion_reference(
    reference_frame: np.ndarray,
    *,
    sigma_scale: float,
) -> _MotionReference:
    layout = _build_motion_grid_layout(
        height=reference_frame.shape[0],
        width=reference_frame.shape[1],
        sigma_scale=sigma_scale,
    )
    return _build_motion_reference_from_layout(reference_frame, layout=layout)


def _build_motion_reference_from_layout(
    reference_frame: np.ndarray,
    *,
    layout: _MotionGridLayout,
) -> _MotionReference:
    window = _hann_window(layout.tile_size)
    rows: list[tuple[_MotionReferenceTile, ...]] = []
    for row_start in layout.row_starts:
        row_end = row_start + layout.tile_size
        tiles: list[_MotionReferenceTile] = []
        for column_start in layout.column_starts:
            column_end = column_start + layout.tile_size
            reference_tile = reference_frame[row_start:row_end, column_start:column_end]
            texture_strength = float(reference_tile.std())
            if texture_strength < 1e-4:
                tiles.append(
                    _MotionReferenceTile(reference_fft=None, texture_strength=0.0)
                )
                continue
            reference_fft = np.fft.fft2(reference_tile * window).astype(
                np.complex64,
                copy=False,
            )
            tiles.append(
                _MotionReferenceTile(
                    reference_fft=reference_fft,
                    texture_strength=texture_strength,
                )
            )
        rows.append(tuple(tiles))
    return _MotionReference(
        height=reference_frame.shape[0],
        width=reference_frame.shape[1],
        layout=layout,
        window=window,
        tiles=tuple(rows),
    )


def _estimate_local_phase_shifts_against_reference(
    luma: np.ndarray,
    reference: _MotionReference,
    *,
    progress_callback: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_count = luma.shape[0]
    row_count = len(reference.layout.row_starts)
    column_count = len(reference.layout.column_starts)
    displacement_x = np.zeros((frame_count, row_count, column_count), dtype=np.float32)
    displacement_y = np.zeros_like(displacement_x)
    confidence = np.zeros((row_count, column_count), dtype=np.float32)
    tile_count = max(1, row_count * column_count)
    processed_tiles = 0

    for row_index, row_start in enumerate(reference.layout.row_starts):
        row_end = row_start + reference.layout.tile_size
        for column_index, column_start in enumerate(reference.layout.column_starts):
            column_end = column_start + reference.layout.tile_size
            reference_tile = reference.tiles[row_index][column_index]
            if reference_tile.reference_fft is None or reference_tile.texture_strength < 1e-4:
                processed_tiles += 1
                if progress_callback is not None:
                    progress_callback(
                        f"motion_grid_tile_{processed_tiles}_of_{tile_count}"
                    )
                continue

            tile_stack = (
                luma[:, row_start:row_end, column_start:column_end]
                * reference.window[None, :, :]
            ).astype(np.float32, copy=False)
            tile_fft = np.fft.fft2(tile_stack, axes=(1, 2)).astype(
                np.complex64,
                copy=False,
            )
            cross_power = reference_tile.reference_fft[None, :, :] * np.conjugate(
                tile_fft
            )
            cross_power /= np.maximum(np.abs(cross_power), np.float32(1e-6))
            correlation = np.abs(np.fft.ifft2(cross_power, axes=(1, 2))).astype(
                np.float32,
                copy=False,
            )

            peak_quality = _extract_tile_motion(
                correlation,
                displacement_x[:, row_index, column_index],
                displacement_y[:, row_index, column_index],
            )
            confidence[row_index, column_index] = np.float32(
                reference_tile.texture_strength * peak_quality
            )

            processed_tiles += 1
            if progress_callback is not None:
                progress_callback(f"motion_grid_tile_{processed_tiles}_of_{tile_count}")

    return displacement_x, displacement_y, confidence


def _rgb_to_luma(frames_rgb: np.ndarray) -> np.ndarray:
    red_weight = np.float32(0.2126)
    green_weight = np.float32(0.7152)
    blue_weight = np.float32(0.0722)
    return (
        red_weight * frames_rgb[..., 0]
        + green_weight * frames_rgb[..., 1]
        + blue_weight * frames_rgb[..., 2]
    ).astype(np.float32, copy=False)


def _extract_tile_motion(
    correlation: np.ndarray,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
) -> float:
    frame_count, height, width = correlation.shape
    flat_indices = np.argmax(correlation.reshape(frame_count, -1), axis=1)
    peak_rows = flat_indices // width
    peak_columns = flat_indices % width
    peak_qualities: list[float] = []

    for frame_index in range(frame_count):
        peak_row = int(peak_rows[frame_index])
        peak_column = int(peak_columns[frame_index])
        surface = correlation[frame_index]
        displacement_x[frame_index] = _wrap_peak_coordinate(
            coordinate=float(peak_column)
            + _subpixel_peak_offset(
                center=float(surface[peak_row, peak_column]),
                negative_neighbor=float(surface[peak_row, (peak_column - 1) % width]),
                positive_neighbor=float(surface[peak_row, (peak_column + 1) % width]),
            ),
            length=width,
        )
        displacement_y[frame_index] = _wrap_peak_coordinate(
            coordinate=float(peak_row)
            + _subpixel_peak_offset(
                center=float(surface[peak_row, peak_column]),
                negative_neighbor=float(surface[(peak_row - 1) % height, peak_column]),
                positive_neighbor=float(surface[(peak_row + 1) % height, peak_column]),
            ),
            length=height,
        )
        peak_value = float(surface[peak_row, peak_column])
        peak_qualities.append(peak_value / (float(surface.mean()) + 1e-6))

    return float(np.clip(np.median(peak_qualities) / 4.0, 0.0, 1.0))


def _subpixel_peak_offset(
    *, center: float, negative_neighbor: float, positive_neighbor: float
) -> float:
    denominator = negative_neighbor - 2.0 * center + positive_neighbor
    if abs(denominator) < 1e-6:
        return 0.0
    offset = 0.5 * (negative_neighbor - positive_neighbor) / denominator
    return float(np.clip(offset, -1.0, 1.0))


def _wrap_peak_coordinate(*, coordinate: float, length: int) -> float:
    if coordinate > length / 2.0:
        return coordinate - float(length)
    return coordinate


def _hann_window(size: int) -> np.ndarray:
    base = np.hanning(size).astype(np.float32)
    return np.outer(base, base).astype(np.float32, copy=False)


def _normalize_confidence(confidence: np.ndarray) -> np.ndarray:
    max_confidence = float(confidence.max())
    if max_confidence <= 1e-6:
        return np.zeros_like(confidence, dtype=np.float32)
    normalized = np.clip(confidence / np.float32(max_confidence), 0.0, 1.0)
    return np.sqrt(normalized).astype(np.float32, copy=False)


def _temporal_bandpass(
    signal: np.ndarray,
    *,
    fps: float,
    low_hz: float,
    high_hz: float,
    progress_callback: Callable[[str], None] | None = None,
    progress_prefix: str = "temporal",
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(signal.shape[0], d=1.0 / fps)
    passband = (frequencies >= low_hz) & (frequencies <= high_hz)
    filtered = np.empty_like(signal, dtype=np.float32)
    row_band_height = _choose_row_band_height(
        signal.shape[0],
        signal.shape[1],
        signal.shape[2],
    )
    band_count = max(1, math.ceil(signal.shape[1] / row_band_height))
    passband_shape = (passband.shape[0], 1, 1)
    for band_index, row_start in enumerate(range(0, signal.shape[1], row_band_height), start=1):
        row_end = min(row_start + row_band_height, signal.shape[1])
        spectrum = np.fft.rfft(signal[:, row_start:row_end, :], axis=0)
        filtered_spectrum = spectrum * passband.reshape(passband_shape)
        filtered[:, row_start:row_end, :] = np.fft.irfft(
            filtered_spectrum,
            n=signal.shape[0],
            axis=0,
        ).real.astype(np.float32, copy=False)
        if progress_callback is not None:
            progress_callback(f"{progress_prefix}_band_{band_index}_of_{band_count}")
    return filtered


def _lowpass_alpha(*, fps: float, cutoff_hz: float) -> np.float32:
    if fps <= 0 or cutoff_hz <= 0:
        return np.float32(1.0)
    dt = 1.0 / fps
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return np.float32(dt / (rc + dt))


def _warp_rgb_frames(
    frames_rgb: np.ndarray,
    *,
    displacement_x_grid: np.ndarray,
    displacement_y_grid: np.ndarray,
    layout: _MotionGridLayout,
    max_displacement_px: np.float32,
    progress_callback: Callable[[str], None] | None,
    progress_prefix: str,
) -> np.ndarray:
    frame_count, height, width, _ = frames_rgb.shape
    output = np.empty_like(frames_rgb, dtype=np.float32)
    base_x = np.broadcast_to(
        np.arange(width, dtype=np.float32)[None, :],
        (height, width),
    )
    base_y = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None],
        (height, width),
    )
    progress_step = max(1, frame_count // 24)
    for frame_index in range(frame_count):
        full_x = np.clip(
            _resize_scalar_field_bilinear(
                displacement_x_grid[frame_index],
                target_height=height,
                target_width=width,
            ),
            -max_displacement_px,
            max_displacement_px,
        )
        full_y = np.clip(
            _resize_scalar_field_bilinear(
                displacement_y_grid[frame_index],
                target_height=height,
                target_width=width,
            ),
            -max_displacement_px,
            max_displacement_px,
        )
        sample_x = np.clip(base_x + full_x, 0.0, width - 1.001)
        sample_y = np.clip(base_y + full_y, 0.0, height - 1.001)
        x0 = np.floor(sample_x).astype(np.int32)
        y0 = np.floor(sample_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)
        wx = (sample_x - x0).astype(np.float32)
        wy = (sample_y - y0).astype(np.float32)

        frame = frames_rgb[frame_index]
        top_left = frame[y0, x0]
        top_right = frame[y0, x1]
        bottom_left = frame[y1, x0]
        bottom_right = frame[y1, x1]

        top = top_left * (1.0 - wx)[..., None] + top_right * wx[..., None]
        bottom = bottom_left * (1.0 - wx)[..., None] + bottom_right * wx[..., None]
        output[frame_index] = top * (1.0 - wy)[..., None] + bottom * wy[..., None]

        if progress_callback is not None and (
            frame_index == frame_count - 1 or frame_index % progress_step == 0
        ):
            progress_callback(f"{progress_prefix}_frame_{frame_index + 1}_of_{frame_count}")
    return output


def _resize_scalar_field_bilinear(
    field: np.ndarray, *, target_height: int, target_width: int
) -> np.ndarray:
    source_height, source_width = field.shape
    if source_height == target_height and source_width == target_width:
        return field.astype(np.float32, copy=False)

    x = np.linspace(0.0, source_width - 1, target_width)
    y = np.linspace(0.0, source_height - 1, target_height)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    rows0 = field[y0, :]
    rows1 = field[y1, :]
    interp_y = rows0 * (1.0 - wy)[:, None] + rows1 * wy[:, None]
    cols0 = interp_y[:, x0]
    cols1 = interp_y[:, x1]
    return cols0 * (1.0 - wx)[None, :] + cols1 * wx[None, :]


def _choose_row_band_height(frame_count: int, height: int, width: int) -> int:
    target_bytes = 128 * 1024 * 1024
    frequency_bins = max(1, frame_count // 2 + 1)
    bytes_per_row_band = max(width * frequency_bins * 16, 1)
    return max(1, min(height, target_bytes // bytes_per_row_band))
