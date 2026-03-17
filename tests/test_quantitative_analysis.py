"""This file tests the quantitative-analysis engine directly so ROI spectra, heatmaps, and artifact export stay verifiable without going through the full worker harness every time."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import struct
import time
import zlib

import numpy as np
import phase_motion_app.core.quantitative_analysis as quantitative_analysis_module

from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.models import AnalysisSettings, Resolution
from phase_motion_app.core.models import ExclusionZone, ZoneMode, ZoneShape
from phase_motion_app.core.quantitative_analysis import StreamingQuantitativeAnalyzer
from phase_motion_app.core.quantitative_analysis import (
    BackgroundStreamingQuantitativeAnalyzer,
)


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


def test_background_quantitative_analysis_finalizes_after_bounded_handoff(
    tmp_path: Path,
) -> None:
    frames = _synthetic_motion_frames()
    reference_luma = frames.mean(axis=0, dtype=np.float32)[..., 1]
    background = BackgroundStreamingQuantitativeAnalyzer(
        StreamingQuantitativeAnalyzer(
            settings=AnalysisSettings(export_advanced_files=False),
            processing_resolution=Resolution(32, 32),
            fps=12.0,
            low_hz=1.0,
            high_hz=4.0,
            reference_luma=reference_luma,
            exclusion_zones=(),
            drift_assessment=DriftAssessment(),
        ),
        queue_depth=1,
    )

    try:
        background.add_chunk(frames[:8])
        background.add_chunk(frames[8:])
        export = background.finalize(tmp_path)
    finally:
        background.close()

    assert export.summary["status"] == "completed"
    assert (tmp_path / "roi_metrics.csv").exists()
    assert (tmp_path / "roi_spectrum.json").exists()


def test_background_quantitative_analysis_surfaces_worker_failure(tmp_path: Path) -> None:
    class _BrokenAnalyzer:
        def add_chunk(self, _frames_rgb: np.ndarray) -> None:
            raise RuntimeError("analysis exploded")

        def finalize(self, _output_directory: Path):
            raise AssertionError("finalize should not run after collection failure")

    background = BackgroundStreamingQuantitativeAnalyzer(
        _BrokenAnalyzer(),
        queue_depth=1,
    )

    try:
        background.add_chunk(np.zeros((2, 4, 4, 3), dtype=np.float32))
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                background.finalize(tmp_path)
            except RuntimeError as exc:
                assert "Background quantitative analysis failed." in str(exc)
                return
            time.sleep(0.05)
        raise AssertionError("background failure did not surface in time")
    finally:
        background.close()


def test_quantitative_analysis_confidence_weight_is_chunk_partition_invariant(
    tmp_path: Path,
) -> None:
    analyzer_a = StreamingQuantitativeAnalyzer(
        settings=AnalysisSettings(export_advanced_files=False),
        processing_resolution=Resolution(32, 32),
        fps=12.0,
        low_hz=1.0,
        high_hz=4.0,
        reference_luma=np.zeros((32, 32), dtype=np.float32),
        exclusion_zones=(),
        drift_assessment=DriftAssessment(),
    )
    analyzer_b = StreamingQuantitativeAnalyzer(
        settings=AnalysisSettings(export_advanced_files=False),
        processing_resolution=Resolution(32, 32),
        fps=12.0,
        low_hz=1.0,
        high_hz=4.0,
        reference_luma=np.zeros((32, 32), dtype=np.float32),
        exclusion_zones=(),
        drift_assessment=DriftAssessment(),
    )

    row_count = len(analyzer_a._layout.row_starts)
    column_count = len(analyzer_a._layout.column_starts)

    def fake_analyze_chunk(frames_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame_count = frames_rgb.shape[0]
        motion_shape = (frame_count, row_count, column_count)
        confidence = np.full(
            (row_count, column_count),
            float(frame_count),
            dtype=np.float32,
        )
        return (
            np.zeros(motion_shape, dtype=np.float32),
            np.zeros(motion_shape, dtype=np.float32),
            confidence,
        )

    analyzer_a._motion_analyzer.analyze_chunk = fake_analyze_chunk  # type: ignore[method-assign]
    analyzer_b._motion_analyzer.analyze_chunk = fake_analyze_chunk  # type: ignore[method-assign]

    analyzer_a.add_chunk(np.zeros((6, 32, 32, 3), dtype=np.float32))
    analyzer_a.add_chunk(np.zeros((4, 32, 32, 3), dtype=np.float32))

    analyzer_b.add_chunk(np.zeros((4, 32, 32, 3), dtype=np.float32))
    analyzer_b.add_chunk(np.zeros((4, 32, 32, 3), dtype=np.float32))
    analyzer_b.add_chunk(np.zeros((2, 32, 32, 3), dtype=np.float32))

    analyzer_a.finalize(tmp_path / "confidence-a")
    analyzer_b.finalize(tmp_path / "confidence-b")

    confidence_a = analyzer_a._confidence_sum / analyzer_a._confidence_frame_weight
    confidence_b = analyzer_b._confidence_sum / analyzer_b._confidence_frame_weight

    assert analyzer_a._frame_count == analyzer_b._frame_count == 10
    assert analyzer_a._confidence_frame_weight == analyzer_b._confidence_frame_weight == 10
    assert np.allclose(confidence_a, confidence_b, atol=1e-6)


def test_quantitative_analysis_export_is_invariant_to_render_chunk_boundaries(
    tmp_path: Path,
) -> None:
    frames = np.zeros((26, 32, 32, 3), dtype=np.float32)
    np.random.seed(3)
    for frame_index in range(frames.shape[0]):
        phase = frame_index / float(frames.shape[0] - 1)
        frames[
            frame_index,
            4:20,
            int(4 + phase * 8) : int(12 + phase * 8),
            :,
        ] = 0.8
        if frame_index >= 18:
            frames[frame_index] += (
                np.random.rand(32, 32, 3).astype(np.float32) - 0.5
            ) * 0.2
    frames = np.clip(frames, 0.0, 1.0)
    reference_luma = frames.mean(axis=0, dtype=np.float32)[..., 1]

    def run_export(subdir: str, splits: tuple[int, ...]) -> tuple[dict, list[list[str]]]:
        output_dir = tmp_path / subdir
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
        start = 0
        for count in splits:
            analyzer.add_chunk(frames[start : start + count])
            start += count
        export = analyzer.finalize(output_dir)
        heatmap_csv_path = Path(
            next(
                path
                for key, path in export.artifact_paths.items()
                if key.startswith("heatmap_") and key.endswith("_csv")
            )
        )
        with heatmap_csv_path.open("r", encoding="utf-8", newline="") as handle:
            heatmap_rows = list(csv.reader(handle))
        summary = dict(export.summary)
        summary["artifact_paths"] = {}
        return summary, heatmap_rows

    split_a_summary, split_a_heatmap = run_export("split-a", (20, 6))
    split_b_summary, split_b_heatmap = run_export("split-b", (8, 8, 8, 2))
    split_c_summary, split_c_heatmap = run_export("split-c", (24, 2))

    assert split_a_summary == split_b_summary == split_c_summary
    assert split_a_heatmap == split_b_heatmap == split_c_heatmap


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


def test_heatmap_overlay_fades_lower_confidence_cells_even_when_band_energy_matches() -> None:
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
    ]
    base_spectra = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.50, 0.50, 0.10],
            [0.10, 0.10, 0.02],
        ],
        dtype=np.float32,
    )
    heatmaps, _scale = quantitative_analysis_module._build_heatmaps(
        base_cells=base_cells,
        base_spectra=base_spectra,
        confidence_mean=np.array([0.40, 0.95, 0.95], dtype=np.float32),
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

    assert float(heatmap["grid"][0, 0]) == float(heatmap["grid"][0, 1])
    assert float(heatmap["band_salience_grid"][0, 2]) < float(
        heatmap["band_salience_grid"][0, 0]
    )
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


def test_heatmap_salience_penalizes_broadband_cells_with_weaker_band_dominance() -> None:
    salience = quantitative_analysis_module._build_heatmap_salience_grid(
        grid=np.array([[0.80, 0.25]], dtype=np.float32),
        band_energy_grid=np.array([[0.60, 0.80]], dtype=np.float32),
        confidence_grid=np.array([[0.95, 0.95]], dtype=np.float32),
        quality_grid=np.array([[0.95, 0.95]], dtype=np.float32),
        band_energy_display_min=0.0,
        band_energy_scale=1.0,
    )

    assert float(salience[0, 0]) > float(salience[0, 1])


def test_heatmap_salience_prefers_supported_high_quality_patch_over_edge_spike() -> None:
    salience = quantitative_analysis_module._build_heatmap_salience_grid(
        grid=np.array(
            [
                [0.72, 0.05, 0.05],
                [0.08, 0.66, 0.64],
                [0.05, 0.62, 0.60],
            ],
            dtype=np.float32,
        ),
        band_energy_grid=np.array(
            [
                [0.95, 0.05, 0.05],
                [0.08, 0.78, 0.76],
                [0.05, 0.74, 0.72],
            ],
            dtype=np.float32,
        ),
        confidence_grid=np.full((3, 3), 0.95, dtype=np.float32),
        quality_grid=np.array(
            [
                [0.20, 0.40, 0.40],
                [0.40, 0.95, 0.95],
                [0.40, 0.95, 0.95],
            ],
            dtype=np.float32,
        ),
        band_energy_display_min=0.0,
        band_energy_scale=1.0,
    )

    assert float(salience[1, 1]) > float(salience[0, 0])
    assert float(salience[2, 1]) > float(salience[0, 0])


def test_heatmap_effective_confidence_threshold_relaxes_for_quiet_clip() -> None:
    base_cells = [
        quantitative_analysis_module._BaseCell(
            cell_id=f"hm_r01_c0{index + 1}",
            row_index=0,
            column_index=index,
            start_x=index * 8,
            start_y=0,
            width=8,
            height=8,
            center_x=(index * 8) + 4.0,
            center_y=4.0,
            roi_fraction=1.0,
            usable_fraction=1.0,
            excluded_fraction=0.0,
            roi_cell_id=f"cell_r01_c0{index + 1}",
        )
        for index in range(4)
    ]

    threshold = quantitative_analysis_module._heatmap_effective_confidence_threshold(
        base_cells=base_cells,
        confidence_mean=np.array([0.12, 0.15, 0.18, 0.20], dtype=np.float32),
        low_confidence_threshold=0.35,
    )

    assert 0.08 <= threshold < 0.20


def test_visible_content_mask_suppresses_pillarbox_bars() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.float32)
    frame[:, 10:38, :] = 0.6

    mask = quantitative_analysis_module._build_visible_content_mask(
        frame,
        target_height=32,
        target_width=48,
    )

    assert mask.shape == (32, 48)
    assert float(mask[:, :8].max()) == 0.0
    assert float(mask[:, -8:].max()) == 0.0
    assert float(mask[:, 14:34].min()) == 1.0


def test_auto_band_resolution_adds_cover_band_for_wide_multi_peak_range() -> None:
    frequencies = np.arange(1.0, 5.2, 0.2, dtype=np.float32)
    roi_spectrum = np.zeros_like(frequencies)
    for peak_hz, amplitude in ((1.6, 0.24), (2.8, 0.22), (4.0, 0.20)):
        index = int(np.argmin(np.abs(frequencies - peak_hz)))
        roi_spectrum[index] = amplitude

    bands, _merge_steps = quantitative_analysis_module._resolve_bands(
        settings=AnalysisSettings(auto_band_count=4),
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
        reported_peaks=(
            quantitative_analysis_module._PeakRecord(
                frequency_hz=1.6,
                amplitude=0.24,
                support_fraction=0.8,
                ranking_score=0.9,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=2.8,
                amplitude=0.22,
                support_fraction=0.75,
                ranking_score=0.82,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=4.0,
                amplitude=0.20,
                support_fraction=0.70,
                ranking_score=0.76,
            ),
        ),
        low_hz=1.0,
        high_hz=5.0,
    )

    assert bands
    assert any(band.low_hz == 1.0 and band.high_hz == 5.0 for band in bands)


def test_auto_band_resolution_promotes_localized_upper_edge_peak() -> None:
    frequencies = np.arange(1.0, 6.1, 0.1, dtype=np.float32)
    roi_spectrum = np.zeros_like(frequencies)
    for peak_hz, amplitude in ((1.1, 0.30), (2.5, 0.26), (3.5, 0.24), (5.6, 0.23)):
        index = int(np.argmin(np.abs(frequencies - peak_hz)))
        roi_spectrum[index] = amplitude

    def build_record(
        cell_id: str,
        *,
        shared_scale: float,
        localized_scale: float,
    ) -> quantitative_analysis_module._RoiCellRecord:
        spectrum = np.full_like(frequencies, 1.0e-6)
        for peak_hz, amplitude in ((1.1, 0.30), (2.5, 0.26), (3.5, 0.24)):
            index = int(np.argmin(np.abs(frequencies - peak_hz)))
            spectrum[index] = amplitude * shared_scale
        localized_index = int(np.argmin(np.abs(frequencies - 5.6)))
        spectrum[localized_index] = 0.23 * localized_scale
        return quantitative_analysis_module._RoiCellRecord(
            cell_id=cell_id,
            trace=np.zeros(16, dtype=np.float32),
            spectrum=spectrum,
            texture_adequacy=1.0,
            trace_stability=1.0,
            component_agreement=1.0,
            band_relevance=1.0,
            mask_impact=0.0,
            drift_impact=0.0,
            composite_quality=1.0,
            valid=True,
            hard_fail=False,
            rejection_reasons=(),
        )

    bands, _merge_steps = quantitative_analysis_module._resolve_bands(
        settings=AnalysisSettings(auto_band_count=4),
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
        reported_peaks=(
            quantitative_analysis_module._PeakRecord(
                frequency_hz=1.1,
                amplitude=0.30,
                support_fraction=0.85,
                ranking_score=0.86,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=2.5,
                amplitude=0.26,
                support_fraction=0.80,
                ranking_score=0.80,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=3.5,
                amplitude=0.24,
                support_fraction=0.78,
                ranking_score=0.76,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=5.6,
                amplitude=0.23,
                support_fraction=0.62,
                ranking_score=0.72,
            ),
        ),
        low_hz=1.0,
        high_hz=5.0,
        roi_cell_records=(
            build_record("cell-a", shared_scale=1.0, localized_scale=14.0),
            build_record("cell-b", shared_scale=1.0, localized_scale=9.0),
            build_record("cell-c", shared_scale=1.1, localized_scale=0.2),
            build_record("cell-d", shared_scale=1.0, localized_scale=0.1),
            build_record("cell-e", shared_scale=0.9, localized_scale=0.1),
            build_record("cell-f", shared_scale=0.8, localized_scale=0.05),
        ),
    )

    assert bands
    assert abs(bands[0].source_peak_hz - 5.6) < 0.25
    assert any(band.source_peak_hz is not None and abs(band.source_peak_hz - 5.6) < 0.25 for band in bands)


def test_estimate_band_edges_keeps_more_upper_span_for_floor_adjacent_low_peak() -> None:
    frequencies = np.arange(0.15, 0.50, 0.05, dtype=np.float32)
    roi_spectrum = np.array([0.18, 0.26, 0.20, 0.16, 0.10, 0.08, 0.05], dtype=np.float32)

    low_edge, high_edge = quantitative_analysis_module._estimate_band_edges(
        peak_hz=0.20,
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
        low_hz=0.15,
        high_hz=0.45,
    )

    assert low_edge == 0.15
    assert high_edge >= 0.34


def test_dense_layout_stays_localized_at_405x720() -> None:
    layout = quantitative_analysis_module._build_dense_layout(Resolution(405, 720))

    assert len(layout.row_starts) >= 12
    assert len(layout.column_starts) >= 7
    assert layout.tile_size <= 96


def test_choose_quantitative_analysis_resolution_uses_richer_source_detail_when_bounded() -> None:
    chosen = quantitative_analysis_module.choose_quantitative_analysis_resolution(
        source_resolution=Resolution(608, 1080),
        processing_resolution=Resolution(405, 720),
    )

    assert chosen == Resolution(608, 1080)


def test_choose_quantitative_analysis_resolution_caps_large_source_but_stays_above_processing() -> None:
    chosen = quantitative_analysis_module.choose_quantitative_analysis_resolution(
        source_resolution=Resolution(1920, 1080),
        processing_resolution=Resolution(640, 360),
    )

    assert chosen.width >= 640
    assert chosen.height >= 360
    assert (chosen.width * chosen.height) <= 700_000
    assert chosen.width % 2 == 0
    assert chosen.height % 2 == 0


def test_quantitative_analysis_scales_exclusion_zones_into_analysis_resolution() -> None:
    analyzer = StreamingQuantitativeAnalyzer(
        settings=AnalysisSettings(export_advanced_files=False),
        processing_resolution=Resolution(50, 50),
        source_resolution=Resolution(100, 100),
        fps=12.0,
        low_hz=1.0,
        high_hz=4.0,
        reference_luma=np.zeros((50, 50), dtype=np.float32),
        exclusion_zones=(
            ExclusionZone(
                zone_id="exclude-bottom",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.EXCLUDE,
                x=0.0,
                y=50.0,
                width=100.0,
                height=50.0,
            ),
        ),
        drift_assessment=DriftAssessment(),
    )

    lower_cells = [cell for cell in analyzer._base_cells if cell.center_y >= 37.5]
    upper_cells = [cell for cell in analyzer._base_cells if cell.center_y <= 12.5]

    assert lower_cells
    assert upper_cells
    assert all(cell.usable_fraction == 0.0 for cell in lower_cells)
    assert all(cell.usable_fraction > 0.0 for cell in upper_cells)


def test_quantitative_analysis_finalize_passes_roi_cell_records_into_band_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
    captured: dict[str, int] = {}
    original_resolve_bands = quantitative_analysis_module._resolve_bands

    def spy_resolve_bands(*, roi_cell_records=(), **kwargs):
        captured["roi_cell_count"] = len(roi_cell_records)
        return original_resolve_bands(roi_cell_records=roi_cell_records, **kwargs)

    monkeypatch.setattr(quantitative_analysis_module, "_resolve_bands", spy_resolve_bands)
    analyzer.add_chunk(frames[:8])
    analyzer.add_chunk(frames[8:])
    analyzer.finalize(tmp_path)

    assert captured["roi_cell_count"] > 0


def test_resolve_bands_clamps_out_of_range_auto_peaks_to_requested_window() -> None:
    frequencies = np.arange(0.0, 1.6, 0.1, dtype=np.float32)
    roi_spectrum = np.array(
        [0.0, 0.02, 0.04, 0.30, 0.22, 0.08, 0.05, 0.18, 0.03, 0.20, 0.02, 0.18, 0.01, 0.16, 0.01, 0.12],
        dtype=np.float32,
    )

    bands, _merge_steps = quantitative_analysis_module._resolve_bands(
        settings=AnalysisSettings(auto_band_count=5),
        roi_spectrum=roi_spectrum,
        frequencies=frequencies,
        reported_peaks=(
            quantitative_analysis_module._PeakRecord(
                frequency_hz=0.30,
                amplitude=0.30,
                support_fraction=0.7,
                ranking_score=0.9,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=0.70,
                amplitude=0.18,
                support_fraction=0.5,
                ranking_score=0.5,
            ),
            quantitative_analysis_module._PeakRecord(
                frequency_hz=1.20,
                amplitude=0.20,
                support_fraction=0.5,
                ranking_score=0.5,
            ),
        ),
        low_hz=0.15,
        high_hz=0.45,
    )

    assert bands
    assert all(0.15 <= band.low_hz <= 0.45 for band in bands)
    assert all(0.15 <= band.high_hz <= 0.45 for band in bands)
    assert all(band.low_hz <= band.high_hz for band in bands)


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
