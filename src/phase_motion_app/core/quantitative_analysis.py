"""This file owns the render-time NVH quantitative-analysis pipeline so ROI spectra, quality scoring, band heatmaps, and exported artifacts come from the internal motion-analysis backend instead of the encoded MP4."""

from __future__ import annotations

import binascii
import csv
import json
import math
import queue
import struct
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.models import (
    AnalysisBandMode,
    AnalysisSettings,
    ExclusionZone,
    Resolution,
    ZoneMode,
    ZoneShape,
)
from phase_motion_app.core.phase_engine import (
    _MotionGridLayout,
    _StreamingBandpassFilter,
    _build_motion_reference_from_layout,
    _estimate_local_phase_shifts_against_reference,
    _normalize_analysis_confidence,
    _rgb_to_luma,
)


@dataclass(frozen=True)
class QuantitativeAnalysisExport:
    """This result carries the sidecar-ready analysis summary plus the written artifact paths for one render."""

    summary: dict[str, Any]
    artifact_paths: dict[str, str]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BaseCell:
    """This model keeps one dense heatmap-cell footprint explicit so ROI grouping and artifact export stay deterministic."""

    cell_id: str
    row_index: int
    column_index: int
    start_x: int
    start_y: int
    width: int
    height: int
    center_x: float
    center_y: float
    roi_fraction: float
    usable_fraction: float
    excluded_fraction: float
    roi_cell_id: str | None


@dataclass(frozen=True)
class _RoiCellRecord:
    """This model captures one exported ROI cell trace, spectrum, and quality state so later writers reuse the same facts."""

    cell_id: str
    trace: np.ndarray
    spectrum: np.ndarray
    texture_adequacy: float
    trace_stability: float
    component_agreement: float
    band_relevance: float
    mask_impact: float
    drift_impact: float
    composite_quality: float
    valid: bool
    hard_fail: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _PeakRecord:
    """This model keeps peak metadata compact and explicit for support checks, ranking, and sidecar export."""

    frequency_hz: float
    amplitude: float
    support_fraction: float
    ranking_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "frequency_hz": self.frequency_hz,
            "amplitude": self.amplitude,
            "support_fraction": self.support_fraction,
            "ranking_score": self.ranking_score,
        }


@dataclass(frozen=True)
class _GeneratedBand:
    """This model stores one generated heatmap band with its provenance so filenames and sidecar records stay predictable."""

    band_id: str
    low_hz: float
    high_hz: float
    mode: str
    source_peak_hz: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "band_id": self.band_id,
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "mode": self.mode,
            "source_peak_hz": self.source_peak_hz,
        }


_ANALYSIS_QUEUE_SENTINEL = object()
_ANALYSIS_INTERNAL_BATCH_FRAMES = 8


class _StreamingMotionFieldAnalyzer:
    """This helper runs the same local motion backend as the render engine on an independent grid that feeds ROI spectra and heatmaps."""

    def __init__(
        self,
        *,
        reference_luma: np.ndarray,
        fps: float,
        low_hz: float,
        high_hz: float,
        layout: _MotionGridLayout,
        attenuate_other_frequencies: bool,
    ) -> None:
        self._reference = _build_motion_reference_from_layout(reference_luma, layout=layout)
        grid_shape = (len(layout.row_starts), len(layout.column_starts))
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
        self._attenuate_other_frequencies = attenuate_other_frequencies

    def analyze_chunk(self, frames_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return band-limited X/Y motion traces plus the chunk confidence map for one bounded frame batch."""

        luma = _rgb_to_luma(frames_rgb.astype(np.float32, copy=False))
        displacement_x, displacement_y, confidence = _estimate_local_phase_shifts_against_reference(
            luma,
            self._reference,
            progress_callback=None,
        )
        bandpassed_x = self._x_filter.filter_chunk(displacement_x)
        bandpassed_y = self._y_filter.filter_chunk(displacement_y)
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
        # Quantitative analysis needs a gentler transfer than the render warp so
        # coherent medium-confidence motion still shows up in the exported
        # traces, ROI quality, and heatmaps.
        confidence_scale = _normalize_analysis_confidence(confidence)
        return (
            phase_motion_x * confidence_scale[None, :, :],
            phase_motion_y * confidence_scale[None, :, :],
            confidence_scale,
        )


class BackgroundStreamingQuantitativeAnalyzer:
    """This helper moves chunk collection onto one bounded background thread so render-time analysis does not sit directly on the phase-processing critical path."""

    def __init__(
        self,
        analyzer: "StreamingQuantitativeAnalyzer",
        *,
        queue_depth: int,
        cancel_check: Callable[[], bool] | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_depth)))
        self._cancel_check = cancel_check
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="quantitative-analysis-background",
            daemon=True,
        )
        self._thread.start()

    def add_chunk(self, frames_rgb: np.ndarray) -> None:
        """Queue one chunk copy for background analysis so the caller keeps ownership of the hot-path working buffers."""

        if frames_rgb.shape[0] == 0:
            return
        self._raise_if_failed()
        queued_chunk = frames_rgb.astype(np.float32, copy=True)
        while True:
            if self._closed.is_set():
                self._raise_if_failed()
                raise RuntimeError("Background quantitative analysis is already closed.")
            if self._cancel_check is not None and self._cancel_check():
                raise RuntimeError("Background quantitative analysis was cancelled.")
            try:
                self._queue.put(queued_chunk, timeout=0.05)
                return
            except queue.Full:
                self._raise_if_failed()
                continue

    def finalize(self, output_directory: Path) -> QuantitativeAnalysisExport:
        """Drain queued work before exporting so the final artifact set still reflects every accepted chunk in order."""

        self._close_worker()
        self._raise_if_failed()
        return self._analyzer.finalize(output_directory)

    def close(self) -> None:
        """Stop the helper thread without writing artifacts so cancellation and worker teardown do not leak threads."""

        self._close_worker()

    def _close_worker(self) -> None:
        if self._closed.is_set():
            if self._thread.is_alive():
                self._thread.join(timeout=1.0)
            return
        self._closed.set()
        while self._thread.is_alive():
            try:
                self._queue.put(_ANALYSIS_QUEUE_SENTINEL, timeout=0.05)
                break
            except queue.Full:
                self._raise_if_failed()
                continue
        self._thread.join(timeout=10.0)

    def _run(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._closed.is_set():
                        return
                    continue
                if item is _ANALYSIS_QUEUE_SENTINEL:
                    return
                self._analyzer.add_chunk(item)
        except BaseException as exc:
            with self._failure_lock:
                self._failure = exc
            self._closed.set()

    def _raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is None:
            return
        raise RuntimeError("Background quantitative analysis failed.") from failure


class StreamingQuantitativeAnalyzer:
    """This class accumulates render-time motion traces and exports the full NVH quantitative-analysis artifact set after encode succeeds."""

    def __init__(
        self,
        *,
        settings: AnalysisSettings,
        processing_resolution: Resolution,
        fps: float,
        low_hz: float,
        high_hz: float,
        reference_luma: np.ndarray,
        exclusion_zones: tuple[ExclusionZone, ...],
        drift_assessment: DriftAssessment,
    ) -> None:
        self._settings = settings
        self._processing_resolution = processing_resolution
        self._fps = fps
        self._low_hz = low_hz
        self._high_hz = high_hz
        self._drift_assessment = drift_assessment
        self._roi = _normalize_roi_zone(settings.roi)
        self._roi_mode = "whole_frame" if self._roi is None else "manual"
        self._roi_label = "Whole-frame ROI" if self._roi is None else "Manual ROI"
        self._layout = _build_dense_layout(processing_resolution)
        self._base_cells = _build_base_cells(
            layout=self._layout,
            processing_resolution=processing_resolution,
            roi=self._roi,
            exclusion_zones=exclusion_zones,
        )
        self._roi_cell_map = _group_roi_cells(
            base_cells=self._base_cells,
            roi=self._roi,
            processing_resolution=processing_resolution,
        )
        self._motion_analyzer = _StreamingMotionFieldAnalyzer(
            reference_luma=reference_luma,
            fps=fps,
            low_hz=low_hz,
            high_hz=high_hz,
            layout=self._layout,
            attenuate_other_frequencies=True,
        )
        self._x_chunks: list[np.ndarray] = []
        self._y_chunks: list[np.ndarray] = []
        self._confidence_sum = np.zeros(len(self._base_cells), dtype=np.float64)
        self._confidence_frame_weight = 0
        self._frame_count = 0
        self._representative_still_frame_rgb: np.ndarray | None = None
        self._pending_frames_rgb: np.ndarray | None = None
        self._fallback_still_frame_rgb = np.repeat(
            np.clip(reference_luma[:, :, None], 0.0, 1.0),
            3,
            axis=2,
        ).astype(np.float32, copy=False)

    def add_chunk(self, frames_rgb: np.ndarray) -> None:
        """Accumulate one bounded processing chunk so the final analysis still runs alongside the authoritative render path."""

        if frames_rgb.shape[0] == 0:
            return
        if self._representative_still_frame_rgb is None and frames_rgb.shape[0] > 0:
            self._representative_still_frame_rgb = frames_rgb[0].astype(
                np.float32,
                copy=True,
            )
        incoming = frames_rgb.astype(np.float32, copy=False)
        if self._pending_frames_rgb is not None and self._pending_frames_rgb.shape[0] > 0:
            # Keep one tiny carry-over buffer so analysis uses one fixed internal
            # batch cadence even when the render pipeline changes its chunk size.
            incoming = np.concatenate((self._pending_frames_rgb, incoming), axis=0)
            self._pending_frames_rgb = None

        processed_frame_count = 0
        while (incoming.shape[0] - processed_frame_count) >= _ANALYSIS_INTERNAL_BATCH_FRAMES:
            batch = incoming[
                processed_frame_count : processed_frame_count + _ANALYSIS_INTERNAL_BATCH_FRAMES
            ]
            self._accumulate_processed_chunk(batch)
            processed_frame_count += _ANALYSIS_INTERNAL_BATCH_FRAMES

        if processed_frame_count < incoming.shape[0]:
            self._pending_frames_rgb = incoming[processed_frame_count:].astype(
                np.float32,
                copy=True,
            )

    def finalize(self, output_directory: Path) -> QuantitativeAnalysisExport:
        """Compute the ROI-level outputs and write the fixed artifact filenames once the render has otherwise completed."""

        output_directory.mkdir(parents=True, exist_ok=True)
        if self._pending_frames_rgb is not None and self._pending_frames_rgb.shape[0] > 0:
            self._accumulate_processed_chunk(self._pending_frames_rgb)
            self._pending_frames_rgb = None
        if self._frame_count == 0 or not self._base_cells:
            return build_empty_analysis_export(
                roi=self._roi,
                roi_mode=self._roi_mode,
                roi_label=self._roi_label,
                warning="No motion samples were accumulated for quantitative analysis.",
                output_directory=output_directory,
            )

        x_traces = np.concatenate(self._x_chunks, axis=0)
        y_traces = np.concatenate(self._y_chunks, axis=0)
        confidence_mean = (
            self._confidence_sum / max(self._confidence_frame_weight, 1)
        ).astype(np.float32, copy=False)
        frequencies = _frequency_axis(self._frame_count, self._fps)

        base_traces, base_spectra, base_metrics = _build_base_traces(
            x_traces=x_traces,
            y_traces=y_traces,
            frequencies=frequencies,
            low_hz=self._low_hz,
        )
        roi_cell_records = _build_roi_cell_records(
            roi_cell_map=self._roi_cell_map,
            base_cells=self._base_cells,
            base_traces=base_traces,
            base_spectra=base_spectra,
            base_metrics=base_metrics,
            confidence_mean=confidence_mean,
            frequencies=frequencies,
            low_hz=self._low_hz,
            roi_quality_cutoff=self._settings.roi_quality_cutoff,
            drift_assessment=self._drift_assessment,
        )
        roi_spectrum = _aggregate_roi_spectrum(roi_cell_records, base_spectra, self._roi_cell_map)
        provisional_quality = _compute_roi_quality(
            roi_cell_records=roi_cell_records,
            reported_peaks=(),
            drift_assessment=self._drift_assessment,
        )
        reported_peaks, suppressed_peak_reasons = _detect_supported_peaks(
            roi_spectrum=roi_spectrum,
            roi_cell_records=roi_cell_records,
            frequencies=frequencies,
            minimum_support_fraction=self._settings.minimum_cell_support_fraction,
            roi_quality_score=provisional_quality["overall_quality_score"],
        )
        roi_quality = _compute_roi_quality(
            roi_cell_records=roi_cell_records,
            reported_peaks=reported_peaks,
            drift_assessment=self._drift_assessment,
        )
        if roi_quality["overall_quality_score"] < self._settings.roi_quality_cutoff:
            suppressed_peak_reasons = (
                *suppressed_peak_reasons,
                "ROI quality stayed below the reporting threshold, so peak reporting was suppressed.",
            )
            reported_peaks = ()

        bands, merge_steps = _resolve_bands(
            settings=self._settings,
            roi_spectrum=roi_spectrum,
            frequencies=frequencies,
            reported_peaks=reported_peaks,
            low_hz=self._low_hz,
            high_hz=self._high_hz,
        )
        heatmaps, heatmap_scale = _build_heatmaps(
            base_cells=self._base_cells,
            base_spectra=base_spectra,
            confidence_mean=confidence_mean,
            frequencies=frequencies,
            bands=bands,
            low_confidence_threshold=self._settings.low_confidence_threshold,
        )
        roi_trace = _aggregate_roi_trace(roi_cell_records)
        artifact_paths = _write_analysis_artifacts(
            output_directory=output_directory,
            roi_mode=self._roi_mode,
            roi_label=self._roi_label,
            roi_quality=roi_quality,
            roi_trace=roi_trace,
            roi_spectrum=roi_spectrum,
            frequencies=frequencies,
            reported_peaks=reported_peaks,
            roi_cell_records=roi_cell_records,
            bands=bands,
            heatmaps=heatmaps,
            heatmap_scale=heatmap_scale,
            fps=self._fps,
            analysis_mode=self._settings.band_mode.value,
            export_advanced_files=self._settings.export_advanced_files,
            representative_still_frame_rgb=(
                self._fallback_still_frame_rgb
                if self._representative_still_frame_rgb is None
                else self._representative_still_frame_rgb
            ),
        )
        summary = {
            "enabled": True,
            "status": "completed",
            "roi_mode": self._roi_mode,
            "roi_label": self._roi_label,
            "roi_geometry": None if self._roi is None else self._roi.to_dict(),
            "roi_quality_score": roi_quality["overall_quality_score"],
            "confidence_label": roi_quality["confidence_label"],
            "reported_peaks": [peak.to_dict() for peak in reported_peaks],
            "bands": [band.to_dict() for band in bands],
            "artifact_paths": artifact_paths,
            "warnings": list(roi_quality["warnings"]),
            "cell_rejection_stats": {
                "valid_cell_count": roi_quality["valid_cell_count"],
                "rejected_cell_count": roi_quality["rejected_cell_count"],
                "rejection_penalty_contribution": roi_quality["rejection_penalty_contribution"],
            },
            "auto_band_merge_steps": list(merge_steps),
            "suppressed_peak_reasons": list(suppressed_peak_reasons),
            "heatmap_scale": heatmap_scale,
        }
        metadata_path = output_directory / "analysis_metadata.json"
        metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        artifact_paths["analysis_metadata"] = str(metadata_path)
        return QuantitativeAnalysisExport(
            summary=summary,
            artifact_paths=artifact_paths,
            warnings=tuple(roi_quality["warnings"]),
        )

    def _accumulate_processed_chunk(self, frames_rgb: np.ndarray) -> None:
        """Process one fixed internal analysis batch so heatmaps depend on source motion, not render chunk boundaries."""

        motion_x, motion_y, confidence = self._motion_analyzer.analyze_chunk(frames_rgb)
        frame_count = motion_x.shape[0]
        self._x_chunks.append(motion_x.reshape(frame_count, -1))
        self._y_chunks.append(motion_y.reshape(frame_count, -1))
        self._confidence_sum += (
            confidence.reshape(-1).astype(np.float64, copy=False) * frame_count
        )
        self._confidence_frame_weight += frame_count
        self._frame_count += frame_count


def build_disabled_analysis_export(settings: AnalysisSettings) -> QuantitativeAnalysisExport:
    """Return the sidecar summary used when the operator disables quantitative analysis for a run."""

    summary = {
        "enabled": False,
        "status": "disabled",
        "roi_mode": "whole_frame" if settings.roi is None else "manual",
        "roi_label": "Whole-frame ROI" if settings.roi is None else "Manual ROI",
        "roi_geometry": None if settings.roi is None else settings.roi.to_dict(),
        "reported_peaks": [],
        "bands": [],
        "artifact_paths": {},
        "warnings": [],
        "cell_rejection_stats": {
            "valid_cell_count": 0,
            "rejected_cell_count": 0,
            "rejection_penalty_contribution": 0.0,
        },
        "auto_band_merge_steps": [],
        "suppressed_peak_reasons": [],
        "heatmap_scale": {
            "normalization_method": "robust_percentile",
            "lower_percentile": 5.0,
            "upper_percentile": 95.0,
            "display_min": 0.0,
            "display_max": 1.0,
            "clipped_cell_count": 0,
        },
    }
    return QuantitativeAnalysisExport(summary=summary, artifact_paths={})


def build_empty_analysis_export(
    *,
    roi: ExclusionZone | None,
    roi_mode: str,
    roi_label: str,
    warning: str,
    output_directory: Path,
) -> QuantitativeAnalysisExport:
    """Write the minimal analysis metadata when analysis ran but could not accumulate any usable motion data."""

    artifact_paths: dict[str, str] = {}
    summary = {
        "enabled": True,
        "status": "warning",
        "roi_mode": roi_mode,
        "roi_label": roi_label,
        "roi_geometry": None if roi is None else roi.to_dict(),
        "roi_quality_score": 0.0,
        "confidence_label": "Unavailable",
        "reported_peaks": [],
        "bands": [],
        "artifact_paths": artifact_paths,
        "warnings": [warning],
        "cell_rejection_stats": {
            "valid_cell_count": 0,
            "rejected_cell_count": 0,
            "rejection_penalty_contribution": 1.0,
        },
        "auto_band_merge_steps": [],
        "suppressed_peak_reasons": [],
        "heatmap_scale": {
            "normalization_method": "robust_percentile",
            "lower_percentile": 5.0,
            "upper_percentile": 95.0,
            "display_min": 0.0,
            "display_max": 1.0,
            "clipped_cell_count": 0,
        },
    }
    metadata_path = output_directory / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    artifact_paths["analysis_metadata"] = str(metadata_path)
    return QuantitativeAnalysisExport(
        summary=summary,
        artifact_paths=artifact_paths,
        warnings=(warning,),
    )


def _normalize_roi_zone(roi: ExclusionZone | None) -> ExclusionZone | None:
    if roi is None:
        return None
    return ExclusionZone(
        zone_id=roi.zone_id,
        shape=roi.shape,
        x=roi.x,
        y=roi.y,
        mode=ZoneMode.INCLUDE,
        width=roi.width,
        height=roi.height,
        radius=roi.radius,
        label=roi.label or "Analysis ROI",
    )


def _build_dense_layout(resolution: Resolution) -> _MotionGridLayout:
    rows = int(np.clip(round(resolution.height / 72.0), 4, 14))
    columns = int(np.clip(round(resolution.width / 72.0), 4, 14))
    cell_height = resolution.height / rows
    cell_width = resolution.width / columns
    tile_size = int(round(min(cell_height, cell_width) * 1.7))
    tile_size = max(12, min(tile_size, min(resolution.width, resolution.height)))
    if tile_size % 2 != 0:
        tile_size += -1 if tile_size == min(resolution.width, resolution.height) else 1
    return _MotionGridLayout(
        tile_size=tile_size,
        row_starts=tuple(_evenly_spaced_starts(resolution.height, tile_size, rows)),
        column_starts=tuple(_evenly_spaced_starts(resolution.width, tile_size, columns)),
    )


def _evenly_spaced_starts(length: int, tile_size: int, count: int) -> list[int]:
    if tile_size >= length or count <= 1:
        return [0]
    max_start = max(length - tile_size, 0)
    starts = sorted({int(round(value)) for value in np.linspace(0, max_start, count)})
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return starts


def _roi_bounds(roi: ExclusionZone | None, resolution: Resolution) -> tuple[float, float, float, float]:
    if roi is None:
        return (0.0, 0.0, float(resolution.width), float(resolution.height))
    if roi.shape is ZoneShape.RECTANGLE:
        return (
            roi.x,
            roi.y,
            roi.x + (roi.width or 0.0),
            roi.y + (roi.height or 0.0),
        )
    radius = roi.radius or 0.0
    return (roi.x - radius, roi.y - radius, roi.x + radius, roi.y + radius)


def _group_roi_cells(
    *,
    base_cells: list[_BaseCell],
    roi: ExclusionZone | None,
    processing_resolution: Resolution,
) -> dict[str, list[int]]:
    roi_bounds = _roi_bounds(roi, processing_resolution)
    grouped: dict[str, list[int]] = {}
    for index, cell in enumerate(base_cells):
        if cell.roi_fraction <= 0:
            continue
        roi_cell_id = cell.roi_cell_id or _roi_cell_id(
            center_x=cell.center_x,
            center_y=cell.center_y,
            roi_bounds=roi_bounds,
        )
        grouped.setdefault(roi_cell_id, []).append(index)
    return grouped


def _roi_cell_id(
    *,
    center_x: float,
    center_y: float,
    roi_bounds: tuple[float, float, float, float],
) -> str:
    left, top, right, bottom = roi_bounds
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    columns = int(np.clip(round(width / 160.0), 2, 6))
    rows = int(np.clip(round(height / 160.0), 2, 6))
    column_index = min(columns - 1, max(0, int((center_x - left) / width * columns)))
    row_index = min(rows - 1, max(0, int((center_y - top) / height * rows)))
    return f"cell_r{row_index + 1:02d}_c{column_index + 1:02d}"


def _build_base_cells(
    *,
    layout: _MotionGridLayout,
    processing_resolution: Resolution,
    roi: ExclusionZone | None,
    exclusion_zones: tuple[ExclusionZone, ...],
) -> list[_BaseCell]:
    roi_bounds = _roi_bounds(roi, processing_resolution)
    cells: list[_BaseCell] = []
    for row_index, row_start in enumerate(layout.row_starts):
        for column_index, column_start in enumerate(layout.column_starts):
            width = min(layout.tile_size, processing_resolution.width - column_start)
            height = min(layout.tile_size, processing_resolution.height - row_start)
            roi_fraction, usable_fraction = _sample_cell_coverage(
                start_x=column_start,
                start_y=row_start,
                width=width,
                height=height,
                roi=roi,
                exclusion_zones=exclusion_zones,
                processing_resolution=processing_resolution,
            )
            excluded_fraction = (
                1.0
                if roi_fraction <= 1e-6
                else float(np.clip((roi_fraction - usable_fraction) / roi_fraction, 0.0, 1.0))
            )
            center_x = column_start + (width / 2.0)
            center_y = row_start + (height / 2.0)
            cells.append(
                _BaseCell(
                    cell_id=f"hm_r{row_index + 1:02d}_c{column_index + 1:02d}",
                    row_index=row_index,
                    column_index=column_index,
                    start_x=column_start,
                    start_y=row_start,
                    width=width,
                    height=height,
                    center_x=center_x,
                    center_y=center_y,
                    roi_fraction=roi_fraction,
                    usable_fraction=usable_fraction,
                    excluded_fraction=excluded_fraction,
                    roi_cell_id=_roi_cell_id(
                        center_x=center_x,
                        center_y=center_y,
                        roi_bounds=roi_bounds,
                    ),
                )
            )
    return cells


def _sample_cell_coverage(
    *,
    start_x: int,
    start_y: int,
    width: int,
    height: int,
    roi: ExclusionZone | None,
    exclusion_zones: tuple[ExclusionZone, ...],
    processing_resolution: Resolution,
) -> tuple[float, float]:
    sample_x = np.linspace(start_x + 0.5, start_x + max(width - 0.5, 0.5), 5)
    sample_y = np.linspace(start_y + 0.5, start_y + max(height - 0.5, 0.5), 5)
    roi_hits = 0
    usable_hits = 0
    for y in sample_y:
        for x in sample_x:
            inside_nominal = _point_in_nominal_roi(
                x=x,
                y=y,
                roi=roi,
                exclusion_zones=exclusion_zones,
                processing_resolution=processing_resolution,
            )
            if inside_nominal:
                roi_hits += 1
                if _point_in_effective_roi(
                    x=x,
                    y=y,
                    roi=roi,
                    exclusion_zones=exclusion_zones,
                    processing_resolution=processing_resolution,
                ):
                    usable_hits += 1
    total = max(len(sample_x) * len(sample_y), 1)
    return roi_hits / total, usable_hits / total


def _point_in_nominal_roi(
    *,
    x: float,
    y: float,
    roi: ExclusionZone | None,
    exclusion_zones: tuple[ExclusionZone, ...],
    processing_resolution: Resolution,
) -> bool:
    include_zones = [zone for zone in exclusion_zones if zone.mode is ZoneMode.INCLUDE]
    if include_zones:
        include_allowed = any(_point_in_zone(x, y, zone) for zone in include_zones)
    else:
        include_allowed = 0.0 <= x <= processing_resolution.width and 0.0 <= y <= processing_resolution.height
    if not include_allowed:
        return False
    if roi is not None and not _point_in_zone(x, y, roi):
        return False
    return True


def _point_in_effective_roi(
    *,
    x: float,
    y: float,
    roi: ExclusionZone | None,
    exclusion_zones: tuple[ExclusionZone, ...],
    processing_resolution: Resolution,
) -> bool:
    if not _point_in_nominal_roi(
        x=x,
        y=y,
        roi=roi,
        exclusion_zones=exclusion_zones,
        processing_resolution=processing_resolution,
    ):
        return False
    exclude_zones = [zone for zone in exclusion_zones if zone.mode is ZoneMode.EXCLUDE]
    return not any(_point_in_zone(x, y, zone) for zone in exclude_zones)


def _point_in_zone(x: float, y: float, zone: ExclusionZone) -> bool:
    if zone.shape is ZoneShape.RECTANGLE:
        return (
            zone.x <= x <= zone.x + (zone.width or 0.0)
            and zone.y <= y <= zone.y + (zone.height or 0.0)
        )
    radius = zone.radius or 0.0
    return math.hypot(x - zone.x, y - zone.y) <= radius


def _frequency_axis(frame_count: int, fps: float) -> np.ndarray:
    if frame_count < 1:
        return np.zeros(1, dtype=np.float32)
    return np.fft.rfftfreq(frame_count, d=1.0 / fps).astype(np.float32, copy=False)


def _build_base_traces(
    *,
    x_traces: np.ndarray,
    y_traces: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cell_count = x_traces.shape[1]
    combined = np.zeros_like(x_traces, dtype=np.float32)
    spectra = np.zeros((len(frequencies), cell_count), dtype=np.float32)
    component_agreement = np.zeros(cell_count, dtype=np.float32)
    band_relevance = np.zeros(cell_count, dtype=np.float32)
    trace_stability = np.zeros(cell_count, dtype=np.float32)
    band_mask = frequencies >= low_hz
    drift_mask = (frequencies > 0) & (
        frequencies < max(low_hz * 0.5, float(frequencies[1]) if len(frequencies) > 1 else low_hz)
    )

    for index in range(cell_count):
        x_trace = x_traces[:, index].astype(np.float32, copy=False)
        y_trace = y_traces[:, index].astype(np.float32, copy=False)
        x_energy = _trace_rms(x_trace)
        y_energy = _trace_rms(y_trace)
        dominant_norm = max(math.hypot(x_energy, y_energy), 1e-6)
        projected = (
            np.float32(x_energy / dominant_norm) * x_trace
            + np.float32(y_energy / dominant_norm) * y_trace
        )
        components = (x_trace, y_trace, projected)
        weights = np.array(
            [
                _trace_rms(component) * _component_stability(component, frequencies, drift_mask)
                for component in components
            ],
            dtype=np.float32,
        )
        if float(weights.sum()) <= 1e-6:
            combined_trace = projected
        else:
            normalized_weights = weights / np.float32(weights.sum())
            combined_trace = (
                normalized_weights[0] * components[0]
                + normalized_weights[1] * components[1]
                + normalized_weights[2] * components[2]
            ).astype(np.float32, copy=False)
        combined[:, index] = combined_trace
        spectrum = _amplitude_spectrum(combined_trace)
        spectra[:, index] = spectrum
        component_agreement[index] = np.float32(
            (
                abs(_safe_correlation(x_trace, projected))
                + abs(_safe_correlation(y_trace, projected))
            )
            / 2.0
        )
        total_energy = float(np.sum(np.square(spectrum)))
        band_energy = float(np.sum(np.square(spectrum[band_mask])))
        drift_energy = float(np.sum(np.square(spectrum[drift_mask])))
        band_relevance[index] = np.float32(
            0.0 if total_energy <= 1e-6 else np.clip(band_energy / total_energy, 0.0, 1.0)
        )
        trace_stability[index] = np.float32(
            np.clip(1.0 - (drift_energy / max(band_energy + drift_energy, 1e-6)), 0.0, 1.0)
        )

    return combined, spectra, {
        "component_agreement": component_agreement,
        "band_relevance": band_relevance,
        "trace_stability": trace_stability,
    }


def _build_roi_cell_records(
    *,
    roi_cell_map: dict[str, list[int]],
    base_cells: list[_BaseCell],
    base_traces: np.ndarray,
    base_spectra: np.ndarray,
    base_metrics: dict[str, np.ndarray],
    confidence_mean: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    roi_quality_cutoff: float,
    drift_assessment: DriftAssessment,
) -> tuple[_RoiCellRecord, ...]:
    records: list[_RoiCellRecord] = []
    for cell_id, indices in sorted(roi_cell_map.items()):
        weights = np.array(
            [
                float(
                    np.clip(
                        base_cells[index].usable_fraction * max(confidence_mean[index], 0.05),
                        0.0,
                        1.0,
                    )
                )
                for index in indices
            ],
            dtype=np.float32,
        )
        hard_fail = False
        reasons: list[str] = []
        if len(indices) == 0:
            hard_fail = True
            reasons.append("no_usable_base_cells")
        if float(weights.sum()) <= 1e-6:
            hard_fail = True
            reasons.append("no_weighted_signal")

        if hard_fail:
            trace = np.zeros(base_traces.shape[0], dtype=np.float32)
        else:
            normalized_weights = weights / np.float32(weights.sum())
            trace = np.sum(
                base_traces[:, indices] * normalized_weights[None, :],
                axis=1,
                dtype=np.float32,
            ).astype(np.float32, copy=False)

        spectrum = _amplitude_spectrum(trace)
        texture_adequacy = _safe_weighted_average(confidence_mean[indices], weights)
        trace_stability = _safe_weighted_average(
            base_metrics["trace_stability"][indices],
            weights,
        )
        component_agreement = _safe_weighted_average(
            base_metrics["component_agreement"][indices],
            weights,
        )
        band_relevance = _safe_weighted_average(
            base_metrics["band_relevance"][indices],
            weights,
        )
        mask_impact = _safe_weighted_average(
            np.array([base_cells[index].usable_fraction for index in indices], dtype=np.float32),
            weights,
        )
        drift_impact = _cell_drift_impact(
            trace=trace,
            frequencies=frequencies,
            low_hz=low_hz,
            drift_assessment=drift_assessment,
        )
        composite_quality = float(
            np.clip(
                (
                    0.22 * texture_adequacy
                    + 0.18 * trace_stability
                    + 0.16 * component_agreement
                    + 0.18 * band_relevance
                    + 0.12 * mask_impact
                    + 0.14 * drift_impact
                ),
                0.0,
                1.0,
            )
        )
        if texture_adequacy < 0.08:
            hard_fail = True
            reasons.append("insufficient_texture")
        if mask_impact < 0.35:
            hard_fail = True
            reasons.append("masked_out")
        if _trace_rms(trace) <= 1e-5:
            hard_fail = True
            reasons.append("no_usable_motion_signal")
        reject_threshold = max(0.30, min(0.55, roi_quality_cutoff * 0.8))
        valid = (not hard_fail) and composite_quality >= reject_threshold
        if not valid and not hard_fail:
            reasons.append("quality_below_threshold")
        quality_factor = 1.0 if valid else 0.5
        records.append(
            _RoiCellRecord(
                cell_id=cell_id,
                trace=trace,
                spectrum=spectrum,
                texture_adequacy=texture_adequacy * quality_factor,
                trace_stability=trace_stability * quality_factor,
                component_agreement=component_agreement * quality_factor,
                band_relevance=band_relevance * quality_factor,
                mask_impact=mask_impact * quality_factor,
                drift_impact=drift_impact * quality_factor,
                composite_quality=composite_quality * quality_factor,
                valid=valid,
                hard_fail=hard_fail,
                rejection_reasons=tuple(reasons),
            )
        )
    return tuple(records)


def _safe_weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or float(weights.sum()) <= 1e-6:
        return 0.0
    return float(np.average(values, weights=weights))


def _aggregate_roi_spectrum(
    roi_cell_records: tuple[_RoiCellRecord, ...],
    base_spectra: np.ndarray,
    roi_cell_map: dict[str, list[int]],
) -> np.ndarray:
    valid_spectra = [record.spectrum for record in roi_cell_records if record.valid]
    if valid_spectra:
        return np.median(np.stack(valid_spectra, axis=0), axis=0).astype(np.float32, copy=False)
    if roi_cell_records:
        return np.median(
            np.stack([record.spectrum for record in roi_cell_records], axis=0),
            axis=0,
        ).astype(np.float32, copy=False)
    if roi_cell_map:
        fallback = [base_spectra[:, indices[0]] for indices in roi_cell_map.values() if indices]
        if fallback:
            return np.median(np.stack(fallback, axis=0), axis=0).astype(np.float32, copy=False)
    return np.zeros(base_spectra.shape[0], dtype=np.float32)


def _aggregate_roi_trace(roi_cell_records: tuple[_RoiCellRecord, ...]) -> np.ndarray:
    valid_traces = [record.trace for record in roi_cell_records if record.valid]
    if valid_traces:
        return np.median(np.stack(valid_traces, axis=0), axis=0).astype(np.float32, copy=False)
    if roi_cell_records:
        return np.median(
            np.stack([record.trace for record in roi_cell_records], axis=0),
            axis=0,
        ).astype(np.float32, copy=False)
    return np.zeros(1, dtype=np.float32)


def _amplitude_spectrum(trace: np.ndarray) -> np.ndarray:
    if trace.size == 0:
        return np.zeros(1, dtype=np.float32)
    centered = trace.astype(np.float32, copy=False) - np.float32(trace.mean())
    spectrum = np.fft.rfft(centered)
    amplitudes = np.abs(spectrum).astype(np.float32, copy=False)
    amplitudes /= np.float32(max(centered.size, 1))
    return amplitudes


def _trace_rms(trace: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(trace.astype(np.float32, copy=False)))))


def _component_stability(trace: np.ndarray, frequencies: np.ndarray, drift_mask: np.ndarray) -> float:
    spectrum = _amplitude_spectrum(trace)
    total_energy = float(np.sum(np.square(spectrum[1:])))
    drift_energy = float(np.sum(np.square(spectrum[drift_mask])))
    if total_energy <= 1e-6:
        return 0.0
    return float(np.clip(1.0 - (drift_energy / total_energy), 0.0, 1.0))


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_std = float(left.std())
    right_std = float(right.std())
    if left_std <= 1e-6 or right_std <= 1e-6:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _cell_drift_impact(
    *,
    trace: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    drift_assessment: DriftAssessment,
) -> float:
    spectrum = _amplitude_spectrum(trace)
    drift_mask = (frequencies > 0) & (
        frequencies < max(low_hz * 0.5, float(frequencies[1]) if len(frequencies) > 1 else low_hz)
    )
    band_mask = frequencies >= low_hz
    drift_energy = float(np.sum(np.square(spectrum[drift_mask])))
    band_energy = float(np.sum(np.square(spectrum[band_mask])))
    ratio = 0.0 if band_energy <= 1e-6 else np.clip(drift_energy / band_energy, 0.0, 1.0)
    global_penalty = 0.0
    if drift_assessment.warning_active:
        global_penalty = 0.20 if drift_assessment.acknowledged else 0.35
    return float(np.clip(1.0 - ratio - global_penalty, 0.0, 1.0))


def _compute_roi_quality(
    *,
    roi_cell_records: tuple[_RoiCellRecord, ...],
    reported_peaks: tuple[_PeakRecord, ...],
    drift_assessment: DriftAssessment,
) -> dict[str, Any]:
    if not roi_cell_records:
        return {
            "overall_quality_score": 0.0,
            "confidence_label": "Unavailable",
            "warnings": ("No ROI cells were available for quantitative analysis.",),
            "valid_cell_count": 0,
            "rejected_cell_count": 0,
            "rejection_penalty_contribution": 1.0,
            "sub_scores": {
                "texture_adequacy": 0.0,
                "valid_cell_fraction": 0.0,
                "inter_cell_agreement": 0.0,
                "peak_consistency": 0.0,
                "mask_impact": 0.0,
                "drift_impact": 0.0,
            },
        }
    valid_cell_count = sum(1 for record in roi_cell_records if record.valid)
    rejected_cell_count = len(roi_cell_records) - valid_cell_count
    valid_fraction = valid_cell_count / len(roi_cell_records)
    texture = float(np.mean([record.texture_adequacy for record in roi_cell_records]))
    inter_cell_agreement = _inter_cell_agreement(roi_cell_records)
    peak_consistency = (
        float(np.mean([peak.support_fraction for peak in reported_peaks]))
        if reported_peaks
        else 0.0
    )
    mask_impact = float(np.mean([record.mask_impact for record in roi_cell_records]))
    drift_impact = float(np.mean([record.drift_impact for record in roi_cell_records]))
    overall = float(
        np.clip(
            (
                0.18 * texture
                + 0.20 * valid_fraction
                + 0.20 * inter_cell_agreement
                + 0.17 * peak_consistency
                + 0.10 * mask_impact
                + 0.15 * drift_impact
            ),
            0.0,
            1.0,
        )
    )
    warnings: list[str] = []
    if overall < 0.45:
        warnings.append("ROI quality stayed below the reporting threshold.")
    if drift_assessment.warning_active:
        warnings.append("Drift review flagged the source state during the run.")
    return {
        "overall_quality_score": overall,
        "confidence_label": _confidence_label(overall),
        "warnings": tuple(warnings),
        "valid_cell_count": valid_cell_count,
        "rejected_cell_count": rejected_cell_count,
        "rejection_penalty_contribution": float(np.clip(1.0 - valid_fraction, 0.0, 1.0)),
        "sub_scores": {
            "texture_adequacy": texture,
            "valid_cell_fraction": valid_fraction,
            "inter_cell_agreement": inter_cell_agreement,
            "peak_consistency": peak_consistency,
            "mask_impact": mask_impact,
            "drift_impact": drift_impact,
        },
    }


def _inter_cell_agreement(roi_cell_records: tuple[_RoiCellRecord, ...]) -> float:
    valid_records = [record for record in roi_cell_records if record.valid]
    if len(valid_records) <= 1:
        return valid_records[0].component_agreement if valid_records else 0.0
    agreements: list[float] = []
    for left_index in range(len(valid_records)):
        for right_index in range(left_index + 1, len(valid_records)):
            agreements.append(
                abs(
                    _safe_correlation(
                        valid_records[left_index].spectrum,
                        valid_records[right_index].spectrum,
                    )
                )
            )
    return float(np.mean(agreements)) if agreements else 0.0


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "High"
    if score >= 0.60:
        return "Moderate"
    if score >= 0.40:
        return "Low"
    return "Poor"


def _detect_supported_peaks(
    *,
    roi_spectrum: np.ndarray,
    roi_cell_records: tuple[_RoiCellRecord, ...],
    frequencies: np.ndarray,
    minimum_support_fraction: float,
    roi_quality_score: float,
) -> tuple[tuple[_PeakRecord, ...], tuple[str, ...]]:
    if len(frequencies) < 3 or roi_spectrum.size != len(frequencies):
        return (), ()
    candidate_indices = _find_peak_indices(roi_spectrum, frequencies)
    if not candidate_indices:
        return (), ("No supported spectral peaks were found in the ROI spectrum.",)
    valid_cells = [record for record in roi_cell_records if record.valid]
    if not valid_cells:
        return (), ("No valid ROI cells remained after quality filtering.",)

    max_amplitude = float(max(roi_spectrum[index] for index in candidate_indices))
    reported: list[_PeakRecord] = []
    suppressed: list[str] = []
    for index in candidate_indices:
        support = _peak_support_fraction(index=index, records=valid_cells)
        amplitude = float(roi_spectrum[index])
        if support < minimum_support_fraction:
            suppressed.append(
                f"Peak near {frequencies[index]:.3f} Hz was suppressed because only {support:.2f} of valid cells supported it."
            )
            continue
        ranking = float(
            np.clip(
                0.55 * (0.0 if max_amplitude <= 1e-6 else amplitude / max_amplitude)
                + 0.35 * support
                + 0.10 * roi_quality_score,
                0.0,
                1.0,
            )
        )
        reported.append(
            _PeakRecord(
                frequency_hz=float(frequencies[index]),
                amplitude=amplitude,
                support_fraction=support,
                ranking_score=ranking,
            )
        )
    reported.sort(key=lambda item: item.ranking_score, reverse=True)
    return tuple(reported), tuple(suppressed)


def _find_peak_indices(spectrum: np.ndarray, frequencies: np.ndarray) -> list[int]:
    positive_indices = [
        index
        for index in range(1, len(spectrum) - 1)
        if frequencies[index] > 0
        and spectrum[index] >= spectrum[index - 1]
        and spectrum[index] >= spectrum[index + 1]
    ]
    if not positive_indices:
        return []
    search_amplitudes = np.array([spectrum[index] for index in positive_indices], dtype=np.float32)
    floor = float(np.median(search_amplitudes))
    threshold = floor + float(search_amplitudes.std()) * 0.5
    filtered = [index for index in positive_indices if float(spectrum[index]) >= threshold]
    if filtered:
        positive_indices = filtered
    positive_indices.sort(key=lambda index: float(spectrum[index]), reverse=True)
    return positive_indices[:10]


def _peak_support_fraction(index: int, records: list[_RoiCellRecord]) -> float:
    support = 0
    for record in records:
        start = max(1, index - 1)
        stop = min(len(record.spectrum) - 1, index + 2)
        local = record.spectrum[start:stop]
        if local.size == 0:
            continue
        if int(np.argmax(local)) + start == index and float(record.spectrum[index]) > 0:
            support += 1
    return support / max(len(records), 1)


def _resolve_bands(
    *,
    settings: AnalysisSettings,
    roi_spectrum: np.ndarray,
    frequencies: np.ndarray,
    reported_peaks: tuple[_PeakRecord, ...],
    low_hz: float,
    high_hz: float,
) -> tuple[tuple[_GeneratedBand, ...], tuple[str, ...]]:
    if settings.band_mode is AnalysisBandMode.MANUAL_SINGLE:
        manual_bands = settings.manual_bands[:1]
        return (
            tuple(
                _clamp_generated_band_to_requested_range(
                    _GeneratedBand(
                        band_id=band.band_id,
                        low_hz=band.low_hz,
                        high_hz=band.high_hz,
                        mode=settings.band_mode.value,
                        source_peak_hz=None,
                    ),
                    low_hz=low_hz,
                    high_hz=high_hz,
                )
                for band in manual_bands
            ),
            (),
        )
    if settings.band_mode is AnalysisBandMode.MANUAL_MULTI:
        return (
            tuple(
                _clamp_generated_band_to_requested_range(
                    _GeneratedBand(
                        band_id=band.band_id,
                        low_hz=band.low_hz,
                        high_hz=band.high_hz,
                        mode=settings.band_mode.value,
                        source_peak_hz=None,
                    ),
                    low_hz=low_hz,
                    high_hz=high_hz,
                )
                for band in settings.manual_bands[:5]
            ),
            (),
        )

    candidate_frequencies = [
        peak.frequency_hz
        for peak in reported_peaks
        if low_hz <= peak.frequency_hz <= high_hz
    ]
    if not candidate_frequencies:
        peak_indices = _find_peak_indices(roi_spectrum, frequencies)
        candidate_frequencies = [
            float(frequencies[index])
            for index in peak_indices
            if low_hz <= float(frequencies[index]) <= high_hz
        ]
    bands: list[_GeneratedBand] = []
    merge_steps: list[str] = []
    for band_number, peak_hz in enumerate(candidate_frequencies[: settings.auto_band_count], start=1):
        low_edge, high_edge = _estimate_band_edges(
            peak_hz=peak_hz,
            roi_spectrum=roi_spectrum,
            frequencies=frequencies,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        bands.append(
            _clamp_generated_band_to_requested_range(
                _GeneratedBand(
                    band_id=f"band{band_number:02d}",
                    low_hz=low_edge,
                    high_hz=high_edge,
                    mode=settings.band_mode.value,
                    source_peak_hz=peak_hz,
                ),
                low_hz=low_hz,
                high_hz=high_hz,
            )
        )
    if not bands:
        span = max(high_hz - low_hz, 0.5)
        center = low_hz + (span / 2.0)
        half_width = min(0.5, span / 2.0)
        bands.append(
            _clamp_generated_band_to_requested_range(
                _GeneratedBand(
                    band_id="band01",
                    low_hz=max(low_hz, center - half_width),
                    high_hz=min(high_hz, center + half_width),
                    mode=settings.band_mode.value,
                    source_peak_hz=center,
                ),
                low_hz=low_hz,
                high_hz=high_hz,
            )
        )
    bands = _split_auto_bands_at_clear_valleys(
        bands=bands,
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
    )
    bands = [
        _clamp_generated_band_to_requested_range(
            band,
            low_hz=low_hz,
            high_hz=high_hz,
        )
        for band in bands
    ]
    merged: list[_GeneratedBand] = []
    for band in sorted(bands, key=lambda item: item.low_hz):
        if not merged:
            merged.append(band)
            continue
        previous = merged[-1]
        gap = band.low_hz - previous.high_hz
        boundary_hz = _clear_peak_separation_boundary(
            left_band=previous,
            right_band=band,
            roi_spectrum=roi_spectrum,
            frequencies=frequencies,
        )
        if gap <= max(0.10, (previous.high_hz - previous.low_hz) * 0.15) and boundary_hz is None:
            merged[-1] = _GeneratedBand(
                band_id=previous.band_id,
                low_hz=min(previous.low_hz, band.low_hz),
                high_hz=max(previous.high_hz, band.high_hz),
                mode=previous.mode,
                source_peak_hz=previous.source_peak_hz,
            )
            merge_steps.append(
                f"Merged overlapping auto-bands around {previous.source_peak_hz or previous.low_hz:.3f} Hz and {band.source_peak_hz or band.low_hz:.3f} Hz."
            )
            continue
        merged.append(band)
    return tuple(merged[: settings.auto_band_count]), tuple(merge_steps)


def _clamp_generated_band_to_requested_range(
    band: _GeneratedBand,
    *,
    low_hz: float,
    high_hz: float,
) -> _GeneratedBand:
    """Clamp one generated band to the selected phase window so sidecar metadata and heatmap exports never describe impossible ranges."""

    if high_hz < low_hz:
        high_hz = low_hz
    clamped_low = float(np.clip(band.low_hz, low_hz, high_hz))
    clamped_high = float(np.clip(band.high_hz, low_hz, high_hz))
    if clamped_high < clamped_low:
        center = band.source_peak_hz
        if center is None:
            center = low_hz
        center = float(np.clip(center, low_hz, high_hz))
        clamped_low = center
        clamped_high = center
    return _GeneratedBand(
        band_id=band.band_id,
        low_hz=clamped_low,
        high_hz=clamped_high,
        mode=band.mode,
        source_peak_hz=band.source_peak_hz,
    )


def _split_auto_bands_at_clear_valleys(
    *,
    bands: list[_GeneratedBand],
    roi_spectrum: np.ndarray,
    frequencies: np.ndarray,
) -> list[_GeneratedBand]:
    if len(bands) <= 1:
        return bands
    bin_width = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.0
    split_padding = max(bin_width * 0.25, 0.0)
    adjusted: list[_GeneratedBand] = [band for band in sorted(bands, key=lambda item: item.low_hz)]
    for index in range(len(adjusted) - 1):
        left_band = adjusted[index]
        right_band = adjusted[index + 1]
        boundary_hz = _clear_peak_separation_boundary(
            left_band=left_band,
            right_band=right_band,
            roi_spectrum=roi_spectrum,
            frequencies=frequencies,
        )
        if boundary_hz is None:
            continue
        left_high = min(left_band.high_hz, boundary_hz - split_padding)
        right_low = max(right_band.low_hz, boundary_hz + split_padding)
        if left_high <= left_band.low_hz:
            left_high = min(left_band.high_hz, boundary_hz)
        if right_low >= right_band.high_hz:
            right_low = max(right_band.low_hz, boundary_hz)
        adjusted[index] = _GeneratedBand(
            band_id=left_band.band_id,
            low_hz=left_band.low_hz,
            high_hz=max(left_high, left_band.low_hz),
            mode=left_band.mode,
            source_peak_hz=left_band.source_peak_hz,
        )
        adjusted[index + 1] = _GeneratedBand(
            band_id=right_band.band_id,
            low_hz=min(right_low, right_band.high_hz),
            high_hz=right_band.high_hz,
            mode=right_band.mode,
            source_peak_hz=right_band.source_peak_hz,
        )
    return adjusted


def _clear_peak_separation_boundary(
    *,
    left_band: _GeneratedBand,
    right_band: _GeneratedBand,
    roi_spectrum: np.ndarray,
    frequencies: np.ndarray,
) -> float | None:
    if left_band.source_peak_hz is None or right_band.source_peak_hz is None:
        return None
    left_index = int(np.argmin(np.abs(frequencies - left_band.source_peak_hz)))
    right_index = int(np.argmin(np.abs(frequencies - right_band.source_peak_hz)))
    if right_index <= left_index + 1:
        return None
    valley_index = left_index + int(np.argmin(roi_spectrum[left_index : right_index + 1]))
    if valley_index <= left_index or valley_index >= right_index:
        return None
    smaller_peak_amplitude = min(
        float(roi_spectrum[left_index]),
        float(roi_spectrum[right_index]),
    )
    if smaller_peak_amplitude <= 1e-6:
        return None
    valley_amplitude = float(roi_spectrum[valley_index])
    if valley_amplitude > smaller_peak_amplitude * 0.55:
        return None
    return float(frequencies[valley_index])


def _estimate_band_edges(
    *,
    peak_hz: float,
    roi_spectrum: np.ndarray,
    frequencies: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> tuple[float, float]:
    peak_index = int(np.argmin(np.abs(frequencies - peak_hz)))
    peak_amplitude = float(roi_spectrum[peak_index])
    bin_width = float(frequencies[1] - frequencies[0]) if len(frequencies) > 1 else 0.25
    min_width = max(0.25, bin_width * 2.0)
    max_width = max(min_width, min(2.5, (high_hz - low_hz) / 2.0))
    low_index = peak_index
    while low_index > 1 and float(roi_spectrum[low_index]) >= peak_amplitude * 0.45:
        low_index -= 1
    high_index = peak_index
    while high_index < len(roi_spectrum) - 2 and float(roi_spectrum[high_index]) >= peak_amplitude * 0.45:
        high_index += 1
    estimated_low = max(low_hz, float(frequencies[low_index]))
    estimated_high = min(high_hz, float(frequencies[high_index]))
    width = estimated_high - estimated_low
    if width < min_width:
        center = float(frequencies[peak_index])
        estimated_low = max(low_hz, center - (min_width / 2.0))
        estimated_high = min(high_hz, center + (min_width / 2.0))
    if estimated_high - estimated_low > max_width:
        center = float(frequencies[peak_index])
        estimated_low = max(low_hz, center - (max_width / 2.0))
        estimated_high = min(high_hz, center + (max_width / 2.0))
    return estimated_low, estimated_high


def _build_heatmaps(
    *,
    base_cells: list[_BaseCell],
    base_spectra: np.ndarray,
    confidence_mean: np.ndarray,
    frequencies: np.ndarray,
    bands: tuple[_GeneratedBand, ...],
    low_confidence_threshold: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    total_energy = np.sum(np.square(base_spectra), axis=0)
    grid_rows = max((cell.row_index for cell in base_cells), default=-1) + 1
    grid_columns = max((cell.column_index for cell in base_cells), default=-1) + 1
    heatmaps: dict[str, dict[str, Any]] = {}
    all_valid_values: list[float] = []
    all_valid_band_energy: list[float] = []
    clipped_cell_count = 0
    weak_signal_floor = _heatmap_weak_signal_floor(
        total_energy=total_energy,
        base_cells=base_cells,
        confidence_mean=confidence_mean,
        low_confidence_threshold=low_confidence_threshold,
    )

    for band in bands:
        grid = np.full((grid_rows, grid_columns), np.nan, dtype=np.float32)
        band_energy_grid = np.full((grid_rows, grid_columns), np.nan, dtype=np.float32)
        confidence_grid = np.zeros((grid_rows, grid_columns), dtype=np.float32)
        low_confidence_mask = np.zeros((grid_rows, grid_columns), dtype=bool)
        reason_summary = {
            "outside_roi": 0,
            "low_confidence": 0,
            "masked_out": 0,
            "weak_signal": 0,
        }
        band_mask = (frequencies >= band.low_hz) & (frequencies <= band.high_hz)
        for index, cell in enumerate(base_cells):
            if cell.roi_fraction <= 0:
                reason_summary["outside_roi"] += 1
                continue
            confidence_grid[cell.row_index, cell.column_index] = np.float32(
                max(float(confidence_mean[index]), 0.0)
            )
            band_energy = float(np.sum(np.square(base_spectra[band_mask, index])))
            normalized_energy = 0.0 if float(total_energy[index]) <= 1e-6 else band_energy / float(total_energy[index])
            grid[cell.row_index, cell.column_index] = np.float32(normalized_energy)
            band_energy_grid[cell.row_index, cell.column_index] = np.float32(band_energy)
            if cell.usable_fraction < 0.35:
                low_confidence_mask[cell.row_index, cell.column_index] = True
                reason_summary["masked_out"] += 1
                continue
            if float(confidence_mean[index]) < low_confidence_threshold:
                low_confidence_mask[cell.row_index, cell.column_index] = True
                reason_summary["low_confidence"] += 1
                continue
            # Normalized band-energy heatmaps can make tiny motion look hot when
            # nearly all of a very weak signal lands inside one band. Demote
            # those cells to monochrome instead of treating them like solid hits.
            if float(total_energy[index]) < weak_signal_floor:
                low_confidence_mask[cell.row_index, cell.column_index] = True
                reason_summary["weak_signal"] += 1
                continue
            all_valid_values.append(normalized_energy)
            all_valid_band_energy.append(band_energy)
        heatmaps[band.band_id] = {
            "grid": grid,
            "band_energy_grid": band_energy_grid,
            "confidence_grid": confidence_grid,
            "low_confidence_mask": low_confidence_mask,
            "low_confidence_count": int(low_confidence_mask.sum()),
            "reason_summary": reason_summary,
        }

    if all_valid_values:
        display_min = float(np.percentile(all_valid_values, 5.0))
        display_max = float(np.percentile(all_valid_values, 95.0))
        if display_max <= display_min:
            display_max = display_min + 1e-6
        for item in heatmaps.values():
            grid = item["grid"]
            clipped_cell_count += int(
                np.sum(
                    np.logical_and(
                        np.isfinite(grid),
                        np.logical_not(item["low_confidence_mask"]),
                    )
                    & ((grid < display_min) | (grid > display_max))
                )
            )
    else:
        display_min = 0.0
        display_max = 1.0

    if all_valid_band_energy:
        band_energy_display_min = float(np.percentile(all_valid_band_energy, 5.0))
        band_energy_display_max = float(np.percentile(all_valid_band_energy, 95.0))
        if band_energy_display_max <= band_energy_display_min:
            band_energy_display_max = band_energy_display_min + 1e-12
    else:
        band_energy_display_min = 0.0
        band_energy_display_max = 1.0

    band_energy_scale = max(band_energy_display_max - band_energy_display_min, 1e-12)
    for item in heatmaps.values():
        band_energy_grid = item["band_energy_grid"]
        confidence_grid = item["confidence_grid"]
        salience_grid = np.zeros_like(band_energy_grid, dtype=np.float32)
        finite_mask = np.isfinite(band_energy_grid)
        if np.any(finite_mask):
            normalized_band_energy = np.clip(
                (band_energy_grid[finite_mask] - np.float32(band_energy_display_min))
                / np.float32(band_energy_scale),
                0.0,
                1.0,
            )
            # Keep the normalized heatmap values as the primary data, but fade
            # cells that are only marginally trustworthy so the display stays
            # focused on coherent motion instead of lighting or texture noise.
            confidence_visibility = np.sqrt(
                np.clip(confidence_grid[finite_mask], 0.0, 1.0)
            ).astype(
                np.float32,
                copy=False,
            )
            salience_grid[finite_mask] = (
                np.sqrt(normalized_band_energy).astype(
                    np.float32,
                    copy=False,
                )
                * confidence_visibility
            ).astype(
                np.float32,
                copy=False,
            )
        item["band_salience_grid"] = salience_grid

    return heatmaps, {
        "normalization_method": "robust_percentile",
        "lower_percentile": 5.0,
        "upper_percentile": 95.0,
        "display_min": display_min,
        "display_max": display_max,
        "clipped_cell_count": clipped_cell_count,
    }


def _heatmap_weak_signal_floor(
    *,
    total_energy: np.ndarray,
    base_cells: list[_BaseCell],
    confidence_mean: np.ndarray,
    low_confidence_threshold: float,
) -> float:
    eligible_energy = [
        float(total_energy[index])
        for index, cell in enumerate(base_cells)
        if cell.roi_fraction > 0.0
        and cell.usable_fraction >= 0.35
        and float(confidence_mean[index]) >= low_confidence_threshold
        and float(total_energy[index]) > 1e-8
    ]
    if len(eligible_energy) < 4:
        return 0.0
    reference_energy = float(np.percentile(eligible_energy, 75.0))
    return max(reference_energy * 0.05, 1e-8)


def _write_analysis_artifacts(
    *,
    output_directory: Path,
    roi_mode: str,
    roi_label: str,
    roi_quality: dict[str, Any],
    roi_trace: np.ndarray,
    roi_spectrum: np.ndarray,
    frequencies: np.ndarray,
    reported_peaks: tuple[_PeakRecord, ...],
    roi_cell_records: tuple[_RoiCellRecord, ...],
    bands: tuple[_GeneratedBand, ...],
    heatmaps: dict[str, dict[str, Any]],
    heatmap_scale: dict[str, Any],
    fps: float,
    analysis_mode: str,
    export_advanced_files: bool,
    representative_still_frame_rgb: np.ndarray,
) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}

    metrics_path = output_directory / "roi_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["roi_label", roi_label])
        writer.writerow(["roi_mode", roi_mode])
        writer.writerow(["analysis_mode", analysis_mode])
        writer.writerow(["roi_quality_score", f"{roi_quality['overall_quality_score']:.6f}"])
        writer.writerow(["confidence_label", roi_quality["confidence_label"]])
        for key, value in roi_quality["sub_scores"].items():
            writer.writerow([key, f"{value:.6f}"])
        writer.writerow(["valid_cell_count", roi_quality["valid_cell_count"]])
        writer.writerow(["rejected_cell_count", roi_quality["rejected_cell_count"]])
        writer.writerow(
            [
                "rejection_penalty_contribution",
                f"{roi_quality['rejection_penalty_contribution']:.6f}",
            ]
        )
        writer.writerow(["reported_peak_count", len(reported_peaks)])
        writer.writerow(
            [
                "top_peak_frequencies_hz",
                ";".join(f"{peak.frequency_hz:.6f}" for peak in reported_peaks[:5]),
            ]
        )
        writer.writerow(
            [
                "support_fractions",
                ";".join(f"{peak.support_fraction:.6f}" for peak in reported_peaks[:5]),
            ]
        )
    artifact_paths["roi_metrics"] = str(metrics_path)

    spectrum_path = output_directory / "roi_spectrum.json"
    spectrum_path.write_text(
        json.dumps(
            {
                "roi_label": roi_label,
                "roi_mode": roi_mode,
                "frequency_hz": [float(value) for value in frequencies],
                "amplitude": [float(value) for value in roi_spectrum],
                "reported_peaks": [peak.to_dict() for peak in reported_peaks],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_paths["roi_spectrum"] = str(spectrum_path)

    trace_path = output_directory / "roi_trace.csv"
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_index", "time_seconds", "trace_value"])
        for frame_index, value in enumerate(roi_trace):
            writer.writerow([frame_index, f"{frame_index / fps:.6f}", f"{float(value):.8f}"])
    artifact_paths["roi_trace"] = str(trace_path)

    traces_path = output_directory / "cell_traces.csv"
    with traces_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["cell_id", "frame_index", "time_seconds", "trace_value", "valid", "hard_fail", "rejection_reasons"]
        )
        for record in roi_cell_records:
            reasons = ";".join(record.rejection_reasons)
            for frame_index, value in enumerate(record.trace):
                writer.writerow(
                    [
                        record.cell_id,
                        frame_index,
                        f"{frame_index / fps:.6f}",
                        f"{float(value):.8f}",
                        int(record.valid),
                        int(record.hard_fail),
                        reasons,
                    ]
                )
    artifact_paths["cell_traces"] = str(traces_path)

    spectra_path = output_directory / "cell_spectra.csv"
    with spectra_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["cell_id", "frequency_hz", "spectral_value", "valid", "hard_fail", "rejection_reasons"]
        )
        for record in roi_cell_records:
            reasons = ";".join(record.rejection_reasons)
            for frequency_hz, value in zip(frequencies, record.spectrum, strict=False):
                writer.writerow(
                    [
                        record.cell_id,
                        f"{float(frequency_hz):.6f}",
                        f"{float(value):.8f}",
                        int(record.valid),
                        int(record.hard_fail),
                        reasons,
                    ]
                )
    artifact_paths["cell_spectra"] = str(spectra_path)

    for band in bands:
        heatmap = heatmaps[band.band_id]
        csv_path = output_directory / f"heatmap_{band.band_id}.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for row in heatmap["grid"]:
                writer.writerow(
                    ["" if not np.isfinite(value) else f"{float(value):.8f}" for value in row]
                )
        artifact_paths[f"heatmap_{band.band_id}_csv"] = str(csv_path)

        png_path = output_directory / f"heatmap_{band.band_id}.png"
        image = _render_heatmap_figure(
            representative_still_frame_rgb=representative_still_frame_rgb,
            grid=heatmap["grid"],
            band_salience_grid=heatmap["band_salience_grid"],
            low_confidence_mask=heatmap["low_confidence_mask"],
            display_min=float(heatmap_scale["display_min"]),
            display_max=float(heatmap_scale["display_max"]),
            band=band,
            roi_label=roi_label,
            low_confidence_count=int(heatmap["low_confidence_count"]),
        )
        _write_png(png_path, image)
        artifact_paths[f"heatmap_{band.band_id}_png"] = str(png_path)

    return artifact_paths


_BITMAP_FONT_3X5: dict[str, tuple[str, ...]] = {
    " ": ("   ", "   ", "   ", "   ", "   "),
    "-": ("   ", "   ", "###", "   ", "   "),
    ".": ("   ", "   ", "   ", "   ", " # "),
    ":": ("   ", " # ", "   ", " # ", "   "),
    "0": ("###", "# #", "# #", "# #", "###"),
    "1": (" # ", "## ", " # ", " # ", "###"),
    "2": ("###", "  #", "###", "#  ", "###"),
    "3": ("###", "  #", " ##", "  #", "###"),
    "4": ("# #", "# #", "###", "  #", "  #"),
    "5": ("###", "#  ", "###", "  #", "###"),
    "6": ("###", "#  ", "###", "# #", "###"),
    "7": ("###", "  #", "  #", "  #", "  #"),
    "8": ("###", "# #", "###", "# #", "###"),
    "9": ("###", "# #", "###", "  #", "###"),
    "A": (" # ", "# #", "###", "# #", "# #"),
    "B": ("## ", "# #", "## ", "# #", "## "),
    "C": (" ##", "#  ", "#  ", "#  ", " ##"),
    "D": ("## ", "# #", "# #", "# #", "## "),
    "E": ("###", "#  ", "## ", "#  ", "###"),
    "F": ("###", "#  ", "## ", "#  ", "#  "),
    "G": (" ##", "#  ", "# #", "# #", " ##"),
    "H": ("# #", "# #", "###", "# #", "# #"),
    "I": ("###", " # ", " # ", " # ", "###"),
    "J": ("  #", "  #", "  #", "# #", "###"),
    "K": ("# #", "# #", "## ", "# #", "# #"),
    "L": ("#  ", "#  ", "#  ", "#  ", "###"),
    "M": ("# #", "###", "###", "# #", "# #"),
    "N": ("# #", "###", "###", "###", "# #"),
    "O": ("###", "# #", "# #", "# #", "###"),
    "P": ("## ", "# #", "## ", "#  ", "#  "),
    "Q": ("###", "# #", "# #", "###", "  #"),
    "R": ("## ", "# #", "## ", "# #", "# #"),
    "S": ("###", "#  ", "###", "  #", "###"),
    "T": ("###", " # ", " # ", " # ", " # "),
    "U": ("# #", "# #", "# #", "# #", "###"),
    "V": ("# #", "# #", "# #", "# #", " # "),
    "W": ("# #", "# #", "###", "###", "# #"),
    "X": ("# #", "# #", " # ", "# #", "# #"),
    "Y": ("# #", "# #", " # ", " # ", " # "),
    "Z": ("###", "  #", " # ", "#  ", "###"),
}


def _render_heatmap_figure(
    *,
    representative_still_frame_rgb: np.ndarray,
    grid: np.ndarray,
    band_salience_grid: np.ndarray,
    low_confidence_mask: np.ndarray,
    display_min: float,
    display_max: float,
    band: _GeneratedBand,
    roi_label: str,
    low_confidence_count: int,
) -> np.ndarray:
    source_height, source_width = representative_still_frame_rgb.shape[:2]
    display_height, display_width = _analysis_display_resolution(
        height=source_height,
        width=source_width,
    )
    grayscale_underlay = _build_grayscale_underlay(
        representative_still_frame_rgb,
        target_height=display_height,
        target_width=display_width,
    )
    overlay_rgb_small, overlay_alpha_small = _build_heatmap_overlay_grid(
        grid=grid,
        band_salience_grid=band_salience_grid,
        low_confidence_mask=low_confidence_mask,
        display_min=display_min,
        display_max=display_max,
    )
    overlay_rgb = _resize_bilinear(
        overlay_rgb_small.astype(np.float32) / np.float32(255.0),
        target_height=display_height,
        target_width=display_width,
    )
    overlay_alpha = _resize_bilinear(
        overlay_alpha_small.astype(np.float32),
        target_height=display_height,
        target_width=display_width,
    )
    # Portrait clips often arrive pillarboxed inside the working frame. Keep
    # the overlay inside the visible picture area so hot cells near the edge do
    # not smear into black bars after interpolation.
    overlay_alpha *= _build_visible_content_mask(
        representative_still_frame_rgb,
        target_height=display_height,
        target_width=display_width,
    )
    composite = (
        (grayscale_underlay.astype(np.float32) / np.float32(255.0))
        * (np.float32(1.0) - overlay_alpha[:, :, None])
        + overlay_rgb * overlay_alpha[:, :, None]
    )
    composite_pixels = np.clip(np.round(composite * 255.0), 0, 255).astype(np.uint8)

    margin = 16
    text_scale = 3
    line_height = (5 * text_scale) + text_scale
    key_label = "NORMALIZED BAND ENERGY"
    caption_text = "HEATMAP SUPERIMPOSED OVER THE FIRST FRAME"
    band_text = (
        f"BAND {band.band_id.upper()} {band.low_hz:.2f}-{band.high_hz:.2f} HZ ROI {roi_label.upper()}"
    )
    scale_text = (
        f"DISPLAY SCALE {display_min:.3f} TO {display_max:.3f} NORMALIZED ENERGY"
    )
    salience_text = "STRONGER BAND ACTIVITY SHOWN WITH HIGHER OPACITY"
    low_confidence_text = (
        None
        if low_confidence_count <= 0
        else f"LOW CONFIDENCE CELLS SHOWN IN MONOCHROME COUNT {low_confidence_count}"
    )
    text_width = display_width + (margin * 2)
    caption_lines = _wrap_bitmap_text(caption_text, max_width=text_width, scale=text_scale)
    caption_lines.extend(
        _wrap_bitmap_text(band_text, max_width=text_width, scale=text_scale)
    )
    caption_lines.extend(
        _wrap_bitmap_text(scale_text, max_width=text_width, scale=text_scale)
    )
    caption_lines.extend(
        _wrap_bitmap_text(salience_text, max_width=text_width, scale=text_scale)
    )
    if low_confidence_text is not None:
        caption_lines.extend(
            _wrap_bitmap_text(low_confidence_text, max_width=text_width, scale=text_scale)
        )

    bar_width = max(220, min(display_width - 32, 520))
    bar_height = 24
    key_block_height = line_height + 6 + bar_height + 6 + line_height
    caption_height = max(line_height * len(caption_lines), line_height)
    canvas_height = (
        margin
        + display_height
        + margin
        + key_block_height
        + margin
        + caption_height
        + margin
    )
    canvas_width = display_width + (margin * 2)
    canvas = np.full((canvas_height, canvas_width, 3), 248, dtype=np.uint8)

    image_top = margin
    image_left = margin
    canvas[
        image_top : image_top + display_height,
        image_left : image_left + display_width,
    ] = composite_pixels
    canvas[image_top - 1 : image_top + display_height + 1, image_left - 1] = 90
    canvas[image_top - 1 : image_top + display_height + 1, image_left + display_width] = 90
    canvas[image_top - 1, image_left - 1 : image_left + display_width + 1] = 90
    canvas[image_top + display_height, image_left - 1 : image_left + display_width + 1] = 90

    key_top = image_top + display_height + margin
    _draw_bitmap_text(
        canvas,
        key_label,
        top=key_top,
        left=max(margin, (canvas_width - _measure_bitmap_text(key_label, text_scale)) // 2),
        color=np.array([28, 28, 28], dtype=np.uint8),
        scale=text_scale,
    )

    bar_left = max(margin, (canvas_width - bar_width) // 2)
    bar_top = key_top + line_height + 6
    for column_index in range(bar_width):
        normalized = 0.0 if bar_width <= 1 else column_index / (bar_width - 1)
        canvas[
            bar_top : bar_top + bar_height,
            bar_left + column_index,
        ] = _heatmap_color(float(normalized))
    canvas[bar_top - 1 : bar_top + bar_height + 1, bar_left - 1] = 70
    canvas[bar_top - 1 : bar_top + bar_height + 1, bar_left + bar_width] = 70
    canvas[bar_top - 1, bar_left - 1 : bar_left + bar_width + 1] = 70
    canvas[bar_top + bar_height, bar_left - 1 : bar_left + bar_width + 1] = 70

    label_top = bar_top + bar_height + 6
    scale_labels = (
        (f"{display_min:.3f}", bar_left),
        (f"{((display_min + display_max) / 2.0):.3f}", bar_left + (bar_width // 2)),
        (f"{display_max:.3f}", bar_left + bar_width),
    )
    for text, anchor in scale_labels:
        label_width = _measure_bitmap_text(text, text_scale)
        _draw_bitmap_text(
            canvas,
            text,
            top=label_top,
            left=max(
                margin,
                min(canvas_width - margin - label_width, anchor - (label_width // 2)),
            ),
            color=np.array([42, 42, 42], dtype=np.uint8),
            scale=text_scale,
        )

    caption_top = key_top + key_block_height + margin
    for line_index, line in enumerate(caption_lines):
        _draw_bitmap_text(
            canvas,
            line,
            top=caption_top + (line_index * line_height),
            left=max(margin, (canvas_width - _measure_bitmap_text(line, text_scale)) // 2),
            color=np.array([36, 36, 36], dtype=np.uint8),
            scale=text_scale,
        )

    return canvas


def _analysis_display_resolution(*, height: int, width: int) -> tuple[int, int]:
    long_side = max(height, width, 1)
    target_long_side = min(max(long_side, 480), 840)
    scale = target_long_side / long_side
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def _build_grayscale_underlay(
    frame_rgb: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    luma = (
        np.float32(0.2126) * frame_rgb[:, :, 0]
        + np.float32(0.7152) * frame_rgb[:, :, 1]
        + np.float32(0.0722) * frame_rgb[:, :, 2]
    ).astype(np.float32, copy=False)
    low, high = np.percentile(luma, [1.0, 99.0])
    if high <= low:
        high = low + 1e-6
    normalized = np.clip((luma - low) / (high - low), 0.0, 1.0)
    resized = _resize_bilinear(
        normalized,
        target_height=target_height,
        target_width=target_width,
    )
    gray = np.clip(np.round(resized * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _build_visible_content_mask(
    frame_rgb: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Build a hard mask for the visibly active picture area so heatmap overlays stay out of encoded black bars."""

    top, bottom, left, right = _visible_content_bounds(frame_rgb)
    mask = np.zeros((target_height, target_width), dtype=np.float32)
    source_height, source_width = frame_rgb.shape[:2]
    top_target = max(0, min(target_height, int(np.floor(top * target_height / max(source_height, 1)))))
    bottom_target = max(
        top_target + 1,
        min(target_height, int(np.ceil(bottom * target_height / max(source_height, 1)))),
    )
    left_target = max(0, min(target_width, int(np.floor(left * target_width / max(source_width, 1)))))
    right_target = max(
        left_target + 1,
        min(target_width, int(np.ceil(right * target_width / max(source_width, 1)))),
    )
    mask[top_target:bottom_target, left_target:right_target] = np.float32(1.0)
    return mask


def _visible_content_bounds(frame_rgb: np.ndarray) -> tuple[int, int, int, int]:
    """Detect obvious pillarbox or letterbox bars from the representative still frame so overlay alpha only covers the visible picture area."""

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError("frame_rgb must have shape [height, width, 3].")
    luma = (
        np.float32(0.2126) * frame_rgb[:, :, 0]
        + np.float32(0.7152) * frame_rgb[:, :, 1]
        + np.float32(0.0722) * frame_rgb[:, :, 2]
    ).astype(np.float32, copy=False)
    height, width = luma.shape
    column_profile = np.maximum(
        np.percentile(luma, 99.0, axis=0),
        luma.std(axis=0) * np.float32(3.0),
    ).astype(np.float32, copy=False)
    row_profile = np.maximum(
        np.percentile(luma, 99.0, axis=1),
        luma.std(axis=1) * np.float32(3.0),
    ).astype(np.float32, copy=False)
    left, right = _edge_profile_active_bounds(column_profile)
    top, bottom = _edge_profile_active_bounds(row_profile)
    if right - left < max(8, width // 5):
        left, right = 0, width
    if bottom - top < max(8, height // 5):
        top, bottom = 0, height
    return top, bottom, left, right


def _edge_profile_active_bounds(profile: np.ndarray) -> tuple[int, int]:
    """Find the active span in one edge profile using a small moving average so isolated dark edge columns do not get mistaken for bars."""

    length = int(profile.shape[0])
    if length <= 0:
        return 0, 0
    threshold = max(
        0.01,
        float(np.percentile(profile, 90.0)) * 0.08,
    )
    window = max(2, min(12, length // 40))

    start = 0
    while start + window <= length:
        if float(np.mean(profile[start : start + window])) >= threshold:
            break
        start += 1

    stop = length
    while stop - window >= 0:
        if float(np.mean(profile[stop - window : stop])) >= threshold:
            break
        stop -= 1

    if stop <= start:
        return 0, length
    return start, stop


def _build_heatmap_overlay_grid(
    *,
    grid: np.ndarray,
    band_salience_grid: np.ndarray,
    low_confidence_mask: np.ndarray,
    display_min: float,
    display_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = grid.shape
    overlay_rgb = np.zeros((rows, columns, 3), dtype=np.uint8)
    overlay_alpha = np.zeros((rows, columns), dtype=np.float32)
    scale = max(display_max - display_min, 1e-6)
    for row_index in range(rows):
        for column_index in range(columns):
            value = grid[row_index, column_index]
            if not np.isfinite(value):
                continue
            normalized = float(
                np.clip((float(value) - display_min) / scale, 0.0, 1.0)
            )
            salience = float(
                np.clip(band_salience_grid[row_index, column_index], 0.0, 1.0)
            )
            visibility = 0.20 + (0.80 * salience)
            if low_confidence_mask[row_index, column_index]:
                gray = int(np.clip(np.round(normalized * 255.0), 0, 255))
                overlay_rgb[row_index, column_index] = np.array(
                    [gray, gray, gray],
                    dtype=np.uint8,
                )
                overlay_alpha[row_index, column_index] = np.float32(
                    (0.12 + (0.14 * normalized)) * visibility
                )
            else:
                overlay_rgb[row_index, column_index] = _heatmap_color(normalized)
                overlay_alpha[row_index, column_index] = np.float32(
                    (0.14 + (0.56 * normalized)) * visibility
                )
    return overlay_rgb, np.clip(overlay_alpha, 0.0, 0.82)


def _resize_bilinear(
    image: np.ndarray,
    *,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    if image.ndim == 2:
        return _resize_bilinear(image[:, :, None], target_height=target_height, target_width=target_width)[
            :, :, 0
        ]
    source_height, source_width, channels = image.shape
    if source_height == target_height and source_width == target_width:
        return image.copy()
    y = np.linspace(0.0, max(source_height - 1, 0), target_height, dtype=np.float32)
    x = np.linspace(0.0, max(source_width - 1, 0), target_width, dtype=np.float32)
    y0 = np.floor(y).astype(np.int32)
    x0 = np.floor(x).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, max(source_height - 1, 0))
    x1 = np.clip(x0 + 1, 0, max(source_width - 1, 0))
    y_weight = (y - y0).astype(np.float32)
    x_weight = (x - x0).astype(np.float32)

    top_left = image[y0[:, None], x0[None, :]].astype(np.float32, copy=False)
    top_right = image[y0[:, None], x1[None, :]].astype(np.float32, copy=False)
    bottom_left = image[y1[:, None], x0[None, :]].astype(np.float32, copy=False)
    bottom_right = image[y1[:, None], x1[None, :]].astype(np.float32, copy=False)

    top = top_left + (top_right - top_left) * x_weight[None, :, None]
    bottom = bottom_left + (bottom_right - bottom_left) * x_weight[None, :, None]
    resized = top + (bottom - top) * y_weight[:, None, None]
    return resized.reshape(target_height, target_width, channels)


def _wrap_bitmap_text(text: str, *, max_width: int, scale: int) -> list[str]:
    sanitized = _sanitize_bitmap_text(text)
    max_chars = max(8, max_width // max((3 * scale) + scale, 1))
    words = sanitized.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    return lines


def _sanitize_bitmap_text(text: str) -> str:
    sanitized = []
    for character in text.upper():
        if character in _BITMAP_FONT_3X5:
            sanitized.append(character)
        else:
            sanitized.append(" ")
    return " ".join("".join(sanitized).split())


def _measure_bitmap_text(text: str, scale: int) -> int:
    sanitized = _sanitize_bitmap_text(text)
    if not sanitized:
        return 0
    return (len(sanitized) * ((3 * scale) + scale)) - scale


def _draw_bitmap_text(
    canvas: np.ndarray,
    text: str,
    *,
    top: int,
    left: int,
    color: np.ndarray,
    scale: int,
) -> None:
    sanitized = _sanitize_bitmap_text(text)
    cursor_left = left
    for character in sanitized:
        glyph = _BITMAP_FONT_3X5.get(character, _BITMAP_FONT_3X5[" "])
        for row_index, row in enumerate(glyph):
            for column_index, marker in enumerate(row):
                if marker != "#":
                    continue
                row_start = top + (row_index * scale)
                row_stop = row_start + scale
                column_start = cursor_left + (column_index * scale)
                column_stop = column_start + scale
                if row_start < 0 or column_start < 0:
                    continue
                canvas[row_start:row_stop, column_start:column_stop] = color
        cursor_left += (3 * scale) + scale


def _heatmap_color(normalized: float) -> np.ndarray:
    anchors = (
        (0.0, np.array([12, 38, 84], dtype=np.float32)),
        (0.5, np.array([58, 160, 214], dtype=np.float32)),
        (1.0, np.array([240, 96, 52], dtype=np.float32)),
    )
    for left, right in zip(anchors[:-1], anchors[1:], strict=False):
        left_pos, left_color = left
        right_pos, right_color = right
        if normalized <= right_pos:
            amount = 0.0 if right_pos == left_pos else (normalized - left_pos) / (right_pos - left_pos)
            mixed = left_color + (right_color - left_color) * np.float32(amount)
            return np.clip(np.round(mixed), 0, 255).astype(np.uint8)
    return anchors[-1][1].astype(np.uint8)


def _write_png(path: Path, pixels: np.ndarray) -> None:
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("PNG writer expects RGB pixels.")
    raw_rows = b"".join(
        b"\x00" + pixels[row_index].astype(np.uint8, copy=False).tobytes()
        for row_index in range(height)
    )
    compressed = zlib.compress(raw_rows)

    def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
        crc = binascii.crc32(chunk_type + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)

    png_bytes = b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _chunk(b"IDAT", compressed),
            _chunk(b"IEND", b""),
        )
    )
    path.write_bytes(png_bytes)
