"""This file tests the quantitative-analysis engine directly so ROI spectra, heatmaps, and artifact export stay verifiable without going through the full worker harness every time."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import zlib

import numpy as np
import phase_motion_app.core.quantitative_analysis as quantitative_analysis_module

from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.models import AnalysisSettings, Resolution
from phase_motion_app.core.models import ExclusionZone, ZoneMode, ZoneShape
from phase_motion_app.core.quantitative_analysis import StreamingQuantitativeAnalyzer


def _synthetic_motion_frames() -> np.ndarray:
    frames = np.zeros((16, 32, 32, 3), dtype=np.float32)
    for frame_index in range(frames.shape[0]):
        offset = frame_index % 4
        frames[frame_index, 6:26, 4 + offset : 12 + offset, :] = 0.85
        frames[frame_index, 10:22, 18 - offset : 26 - offset, :] = 0.45
        frames[frame_index, :, :, 1] += np.linspace(0.05, 0.25, 32, dtype=np.float32)[
            None, :
        ]
    return np.clip(frames, 0.0, 1.0)


def _read_png_rgb(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    offset = 8
    width = 0
    height = 0
    idat_payload = bytearray()
    while offset < len(payload):
        chunk_length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_data = payload[offset + 8 : offset + 8 + chunk_length]
        offset += 12 + chunk_length
        if chunk_type == b"IHDR":
            width, height = struct.unpack(">II", chunk_data[:8])
        elif chunk_type == b"IDAT":
            idat_payload.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    raw_rows = zlib.decompress(bytes(idat_payload))
    stride = (width * 3) + 1
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for row_index in range(height):
        row = raw_rows[row_index * stride : (row_index + 1) * stride]
        assert row[0] == 0
        pixels[row_index] = np.frombuffer(row[1:], dtype=np.uint8).reshape(width, 3)
    return pixels


def test_quantitative_analysis_writes_required_artifacts(tmp_path: Path) -> None:
    frames = _synthetic_motion_frames()
    reference_luma = frames.mean(axis=0, dtype=np.float32)[..., 1]
    analyzer = StreamingQuantitativeAnalyzer(
        settings=AnalysisSettings(export_advanced_files=False),
        processing_resolution=Resolution(32, 32),
        fps=12.0,
        low_hz=1.0,
        high_hz=4.0,
        reference_luma=reference_luma,
        exclusion_zones=(),
        drift_assessment=DriftAssessment(),
    )

    analyzer.add_chunk(frames[:8])
    analyzer.add_chunk(frames[8:])
    export = analyzer.finalize(tmp_path)

    assert export.summary["enabled"] is True
    assert export.summary["status"] == "completed"
    assert (tmp_path / "roi_metrics.csv").exists()
    assert (tmp_path / "roi_spectrum.json").exists()
    assert (tmp_path / "roi_trace.csv").exists()
    assert (tmp_path / "analysis_metadata.json").exists()
    assert (tmp_path / "cell_traces.csv").exists()
    assert (tmp_path / "cell_spectra.csv").exists()
    assert any(path.endswith(".png") for path in export.artifact_paths.values())
    metadata = json.loads((tmp_path / "analysis_metadata.json").read_text(encoding="utf-8"))
    assert metadata["roi_label"] == "Whole-frame ROI"
    assert metadata["heatmap_scale"]["normalization_method"] == "robust_percentile"
    heatmap_png_path = Path(
        next(
            path
            for key, path in export.artifact_paths.items()
            if key.startswith("heatmap_") and key.endswith("_png")
        )
    )
    heatmap_pixels = _read_png_rgb(heatmap_png_path)
    upper_region = heatmap_pixels[: int(heatmap_pixels.shape[0] * 0.6)]
    footer_region = heatmap_pixels[int(heatmap_pixels.shape[0] * 0.7) :]
    grayscale_pixels = (
        (upper_region[:, :, 0] == upper_region[:, :, 1])
        & (upper_region[:, :, 1] == upper_region[:, :, 2])
    )
    colored_pixels = (
        (upper_region[:, :, 0] != upper_region[:, :, 1])
        | (upper_region[:, :, 1] != upper_region[:, :, 2])
    )
    assert heatmap_pixels.shape[0] > 200
    assert heatmap_pixels.shape[1] > 200
    assert bool(np.any(grayscale_pixels))
    assert bool(np.any(colored_pixels))
    assert np.unique(footer_region.reshape(-1, 3), axis=0).shape[0] > 10


def test_quantitative_analysis_whole_frame_fallback_respects_processing_mask() -> None:
    analyzer = StreamingQuantitativeAnalyzer(
        settings=AnalysisSettings(export_advanced_files=False),
        processing_resolution=Resolution(64, 64),
        fps=12.0,
        low_hz=1.0,
        high_hz=4.0,
        reference_luma=np.zeros((64, 64), dtype=np.float32),
        exclusion_zones=(
            ExclusionZone(
                zone_id="include-left",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.INCLUDE,
                x=0.0,
                y=0.0,
                width=32.0,
                height=64.0,
            ),
            ExclusionZone(
                zone_id="exclude-center",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.EXCLUDE,
                x=8.0,
                y=16.0,
                width=16.0,
                height=24.0,
            ),
        ),
        drift_assessment=DriftAssessment(),
    )

    right_side_cells = [cell for cell in analyzer._base_cells if cell.start_x >= 36]

    assert right_side_cells
    assert all(
        cell.roi_fraction == 0.0 and cell.usable_fraction == 0.0
        for cell in right_side_cells
    )
    assert any(
        cell.roi_fraction > 0.0 and cell.usable_fraction < cell.roi_fraction
        for cell in analyzer._base_cells
    )
    assert any(cell.usable_fraction > 0.0 for cell in analyzer._base_cells)


def test_heatmap_marks_weak_signal_cells_low_confidence() -> None:
    base_cells = [
        quantitative_analysis_module._BaseCell(
            cell_id="hm_r01_c01",
            row_index=0,
            column_index=0,
            start_x=0,
            start_y=0,
            width=8,
            height=8,
            center_x=4.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id="cell_r01_c01",
        ),
        quantitative_analysis_module._BaseCell(
            cell_id="hm_r01_c02",
            row_index=0,
            column_index=1,
            start_x=8,
            start_y=0,
            width=8,
            height=8,
            center_x=12.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id="cell_r01_c02",
        ),
        quantitative_analysis_module._BaseCell(
            cell_id="hm_r01_c03",
            row_index=0,
            column_index=2,
            start_x=16,
            start_y=0,
            width=8,
            height=8,
            center_x=20.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id="cell_r01_c03",
        ),
        quantitative_analysis_module._BaseCell(
            cell_id="hm_r01_c04",
            row_index=0,
            column_index=3,
            start_x=24,
            start_y=0,
            width=8,
            height=8,
            center_x=28.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id="cell_r01_c04",
        ),
    ]
    base_spectra = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.001, 0.80, 0.90, 1.00],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    frequencies = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    heatmaps, _scale = quantitative_analysis_module._build_heatmaps(
        base_cells=base_cells,
        base_spectra=base_spectra,
        confidence_mean=np.array([0.95, 0.95, 0.95, 0.95], dtype=np.float32),
        frequencies=frequencies,
        bands=(
            quantitative_analysis_module._GeneratedBand(
                band_id="band01",
                low_hz=0.5,
                high_hz=1.5,
                mode="auto",
                source_peak_hz=1.0,
            ),
        ),
        low_confidence_threshold=0.35,
    )

    heatmap = heatmaps["band01"]

    assert bool(heatmap["low_confidence_mask"][0, 0]) is True
    assert bool(heatmap["low_confidence_mask"][0, 1]) is False
    assert heatmap["low_confidence_count"] == 1
    assert heatmap["reason_summary"]["weak_signal"] == 1


def test_heatmap_overlay_fades_weak_absolute_band_activity() -> None:
    base_cells = [
        quantitative_analysis_module._BaseCell(
            cell_id="hm_r01_c01",
            row_index=0,
            column_index=0,
            start_x=0,
            start_y=0,
            width=8,
            height=8,
            center_x=4.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id="cell_r01_c01",
        ),
        quantitative_analysis_module._BaseCell(
            cell_id="hm_r01_c02",
            row_index=0,
            column_index=1,
            start_x=8,
            start_y=0,
            width=8,
            height=8,
            center_x=12.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id="cell_r01_c02",
        ),
    ]
    base_spectra = np.array(
        [
            [0.0, 0.0],
            [0.010, 0.500],
            [0.003, 0.400],
        ],
        dtype=np.float32,
    )
    heatmaps, _scale = quantitative_analysis_module._build_heatmaps(
        base_cells=base_cells,
        base_spectra=base_spectra,
        confidence_mean=np.array([0.95, 0.95], dtype=np.float32),
        frequencies=np.array([0.0, 1.0, 2.0], dtype=np.float32),
        bands=(
            quantitative_analysis_module._GeneratedBand(
                band_id="band01",
                low_hz=0.5,
                high_hz=1.5,
                mode="auto",
                source_peak_hz=1.0,
            ),
        ),
        low_confidence_threshold=0.35,
    )

    heatmap = heatmaps["band01"]

    assert float(heatmap["grid"][0, 0]) > float(heatmap["grid"][0, 1])
    assert float(heatmap["band_salience_grid"][0, 1]) > float(
        heatmap["band_salience_grid"][0, 0]
    )

    _overlay_rgb, overlay_alpha = quantitative_analysis_module._build_heatmap_overlay_grid(
        grid=heatmap["grid"],
        band_salience_grid=heatmap["band_salience_grid"],
        low_confidence_mask=heatmap["low_confidence_mask"],
        display_min=0.0,
        display_max=1.0,
    )

    assert float(overlay_alpha[0, 1]) > float(overlay_alpha[0, 0])


def test_auto_band_resolution_keeps_valley_separated_peaks_distinct() -> None:
    frequencies = np.arange(0.6, 1.5, 0.1, dtype=np.float32)
    roi_spectrum = np.array(
        [0.00, 0.14, 0.30, 0.22, 0.14, 0.20, 0.32, 0.14, 0.00],
        dtype=np.float32,
    )

    bands, merge_steps = quantitative_analysis_module._resolve_bands(
        settings=AnalysisSettings(auto_band_count=5),
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
        reported_peaks=(
            quantitative_analysis_module._PeakRecord(
                frequency_hz=0.8,
                amplitude=0.30,
                support_fraction=0.8,
                ranking_score=0.9,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=1.2,
                amplitude=0.32,
                support_fraction=0.7,
                ranking_score=0.85,
            ),
        ),
        low_hz=0.6,
        high_hz=1.4,
    )

    assert len(bands) == 2
    assert not merge_steps
    assert bands[0].high_hz < bands[1].low_hz


def test_auto_band_resolution_merges_peaks_without_clear_valley() -> None:
    frequencies = np.arange(0.6, 1.5, 0.1, dtype=np.float32)
    roi_spectrum = np.array(
        [0.00, 0.26, 0.30, 0.28, 0.26, 0.29, 0.32, 0.26, 0.00],
        dtype=np.float32,
    )

    bands, merge_steps = quantitative_analysis_module._resolve_bands(
        settings=AnalysisSettings(auto_band_count=5),
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
        reported_peaks=(
            quantitative_analysis_module._PeakRecord(
                frequency_hz=0.8,
                amplitude=0.30,
                support_fraction=0.8,
                ranking_score=0.9,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=1.2,
                amplitude=0.32,
                support_fraction=0.7,
                ranking_score=0.85,
            ),
        ),
        low_hz=0.6,
        high_hz=1.4,
    )

    assert len(bands) == 1
    assert merge_steps


def test_roi_cell_records_ignore_zero_weight_averages() -> None:
    records = quantitative_analysis_module._build_roi_cell_records(
        roi_cell_map={"cell_r01_c01": [0]},
        base_cells=[
            quantitative_analysis_module._BaseCell(
                cell_id="hm_r01_c01",
                row_index=0,
                column_index=0,
                start_x=0,
                start_y=0,
                width=8,
                height=8,
                center_x=4.0,
                center_y=4.0,
                roi_fraction=1.0,
                usable_fraction=0.0,
                excluded_fraction=1.0,
                roi_cell_id="cell_r01_c01",
            )
        ],
        base_traces=np.zeros((8, 1), dtype=np.float32),
        base_spectra=np.zeros((5, 1), dtype=np.float32),
        base_metrics={
            "trace_stability": np.zeros(1, dtype=np.float32),
            "component_agreement": np.zeros(1, dtype=np.float32),
            "band_relevance": np.zeros(1, dtype=np.float32),
        },
        confidence_mean=np.zeros(1, dtype=np.float32),
        frequencies=np.linspace(0.0, 4.0, 5, dtype=np.float32),
        low_hz=1.0,
        roi_quality_cutoff=0.45,
        drift_assessment=DriftAssessment(),
    )

    assert len(records) == 1
    assert records[0].hard_fail is True
    assert "no_weighted_signal" in records[0].rejection_reasons
