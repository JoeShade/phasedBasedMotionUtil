"""This file tests small shell-level behaviors that affect reproducibility without dragging render work into the GUI process."""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QGroupBox

import phase_motion_app.app.main_window as main_window_module
from phase_motion_app.app.main_window import MainWindow, PreflightWarningDialog
from phase_motion_app.core.acceleration import AccelerationCapability, AccelerationDecision
from phase_motion_app.core.baseline_band import FrequencyBandSuggestion
from phase_motion_app.core.ffprobe import FfprobeMediaInfo
from phase_motion_app.core.job_state import SourceSnapshot, UiState
from phase_motion_app.core.models import (
    AnalysisBand,
    AnalysisBandMode,
    AnalysisSettings,
    JobIntent,
    ExclusionZone,
    MAX_ANALYSIS_BANDS,
    PhaseSettings,
    Resolution,
    ResourcePolicy,
    ZoneMode,
    ZoneShape,
)
from phase_motion_app.core.preflight import (
    PreflightIssue,
    PreflightReport,
    PreflightSeverity,
)
from phase_motion_app.core.render_supervisor import (
    RenderEvent,
    RenderPollResult,
    RenderSupervisorSnapshot,
)


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


class FakeSupervisor:
    """This fake supervisor lets the Qt shell tests drive terminal states without spawning a real worker."""

    def __init__(self, poll_results: list[RenderPollResult]) -> None:
        self._poll_results = list(poll_results)
        self._last_result = poll_results[-1]
        self.started = False
        self.cancelled = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def poll(self) -> RenderPollResult:
        if self._poll_results:
            self._last_result = self._poll_results.pop(0)
        return self._last_result

    def close(self) -> None:
        self.closed = True


def _source_info() -> FfprobeMediaInfo:
    return FfprobeMediaInfo(
        width=1280,
        height=720,
        fps=30.0,
        avg_fps=30.0,
        is_cfr=True,
        duration_seconds=2.0,
        frame_count=60,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        profile="High",
        pixel_format="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
    )


def _ready_window(
    tmp_path: Path,
    *,
    render_supervisor_factory=None,
) -> MainWindow:
    window = MainWindow(
        render_supervisor_factory=render_supervisor_factory,
        state_path=tmp_path / "settings.json",
    )
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-bytes")
    snapshot = SourceSnapshot(
        path=str(source_path),
        size_bytes=source_path.stat().st_size,
        modified_ns=source_path.stat().st_mtime_ns,
    )
    window._current_source_path = source_path
    window.source_path_edit.setText(str(source_path))
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    window.output_folder_edit.setText(str(output_dir))
    window._current_fingerprint = "a" * 64
    window._source_probe_info = _source_info()
    window._last_known_snapshot = snapshot
    window._controller.load_source(snapshot)
    window._controller.set_settings_complete(True)
    window._controller.mark_fingerprint_complete(snapshot)
    window._update_settings_state()
    return window


def test_main_window_preserves_fractional_phase_settings_on_intent_reload(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    intent = JobIntent(
        phase=PhaseSettings(
            magnification=20.5,
            low_hz=5.125,
            high_hz=12.75,
            pyramid_type="complex_steerable",
            sigma=1.0,
            attenuate_other_frequencies=True,
        ),
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
    )

    try:
        window._apply_intent(intent)
        reloaded_intent = window._build_intent()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert reloaded_intent.phase.magnification == 20.5
    assert reloaded_intent.phase.low_hz == 5.125
    assert reloaded_intent.phase.high_hz == 12.75
    assert reloaded_intent.output_resolution == reloaded_intent.processing_resolution


def test_main_window_roundtrips_analysis_settings_in_intent_reload(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    intent = JobIntent(
        phase=PhaseSettings(
            magnification=20.0,
            low_hz=5.0,
            high_hz=12.0,
            pyramid_type="complex_steerable",
            sigma=1.0,
            attenuate_other_frequencies=True,
        ),
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
        analysis=AnalysisSettings(
            enabled=True,
            minimum_cell_support_fraction=0.4,
            roi_quality_cutoff=0.5,
            low_confidence_threshold=0.3,
            auto_band_count=4,
            band_mode=AnalysisBandMode.MANUAL_SINGLE,
            manual_bands=(AnalysisBand("band01", 6.0, 8.0),),
            export_advanced_files=False,
        ),
    )

    try:
        window._apply_intent(intent)
        reloaded_intent = window._build_intent()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert reloaded_intent.analysis.band_mode is AnalysisBandMode.MANUAL_SINGLE
    assert reloaded_intent.analysis.manual_bands[0].low_hz == 6.0
    assert reloaded_intent.analysis.auto_band_count == 4
    assert reloaded_intent.analysis.export_advanced_files is False
    assert reloaded_intent.output_resolution == reloaded_intent.processing_resolution


def test_main_window_hardware_acceleration_checkbox_round_trips_through_intent(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    intent = JobIntent(
        phase=PhaseSettings(
            magnification=20.0,
            low_hz=5.0,
            high_hz=12.0,
            pyramid_type="complex_steerable",
            sigma=1.0,
            attenuate_other_frequencies=True,
        ),
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
        hardware_acceleration_enabled=True,
    )

    try:
        window._apply_intent(intent)
        reloaded_intent = window._build_intent()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window.hardware_acceleration_checkbox.isChecked() is True
    assert reloaded_intent.hardware_acceleration_enabled is True


def test_main_window_hardware_acceleration_status_text_reports_availability_and_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    states = {
        True: AccelerationDecision(
            requested=True,
            active=False,
            status="gpu_requested_cpu_fallback",
            detail=(
                "Hardware acceleration was requested, but the optional CuPy backend "
                "is not installed. CPU fallback will be used."
            ),
            backend_name="cupy",
        ),
        False: AccelerationDecision(
            requested=False,
            active=False,
            status="cpu_selected",
            detail=(
                "Hardware acceleration is available but disabled. The authoritative "
                "CPU path will be used."
            ),
            backend_name="cupy",
            device_name="Example GPU",
        ),
    }
    monkeypatch.setattr(
        main_window_module,
        "detect_acceleration_capability",
        lambda: AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=True,
            status="available",
            detail="CuPy and a compatible CUDA device are available.",
            installed_version="13.0",
            device_name="Example GPU",
        ),
    )
    monkeypatch.setattr(
        main_window_module,
        "resolve_acceleration_request",
        lambda requested, capability=None: states[requested],
    )
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        disabled_text = window.hardware_acceleration_status_label.text()
        window.hardware_acceleration_checkbox.setChecked(True)
        fallback_text = window.hardware_acceleration_status_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "Available: CuPy on Example GPU" in disabled_text
    assert "Enable the checkbox" in disabled_text
    assert "Unavailable: CuPy on Example GPU" in fallback_text
    assert "CPU fallback" in fallback_text


def test_main_window_analysis_validation_blocks_invalid_manual_band(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        window.analysis_enabled_checkbox.setChecked(True)
        window.analysis_band_mode_combo.setCurrentIndex(
            window.analysis_band_mode_combo.findData(AnalysisBandMode.MANUAL_SINGLE)
        )
        enabled_check, low_spin, high_spin = window._manual_band_controls[0]
        enabled_check.setChecked(True)
        low_spin.setValue(13.0)
        high_spin.setValue(15.0)
        window._update_settings_state()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window._controller.state is not UiState.READY


def test_main_window_automatic_analysis_roi_summary_mentions_processing_mask(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        window._analysis_roi = None
        window._exclusion_zones = (
            ExclusionZone(
                zone_id="include-main",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.INCLUDE,
                x=10.0,
                y=10.0,
                width=90.0,
                height=60.0,
            ),
            ExclusionZone(
                zone_id="exclude-fan",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.EXCLUDE,
                x=40.0,
                y=20.0,
                width=20.0,
                height=20.0,
            ),
        )
        summary = window._analysis_roi_summary()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "processing inclusion zones" in summary
    assert "exclusion zones" in summary


def test_main_window_applies_advanced_analysis_dialog_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    captured: dict[str, object] = {}

    class FakeAnalysisAdvancedDialog:
        def __init__(
            self,
            *,
            band_mode,
            minimum_cell_support_fraction,
            roi_quality_cutoff,
            low_confidence_threshold,
            auto_band_count,
            export_advanced_files,
            manual_band_values,
            parent=None,
        ) -> None:
            captured["band_mode"] = band_mode
            captured["minimum_cell_support_fraction"] = minimum_cell_support_fraction
            captured["parent"] = parent

        def exec(self) -> bool:
            return True

        def result_data(self):
            return main_window_module.AnalysisAdvancedDialogResult(
                minimum_cell_support_fraction=0.55,
                roi_quality_cutoff=0.65,
                low_confidence_threshold=0.25,
                auto_band_count=3,
                export_advanced_files=False,
                manual_band_values=(
                    (True, 6.0, 7.5),
                    (True, 8.0, 9.5),
                    (False, 7.0, 10.0),
                    (False, 8.0, 11.0),
                    (False, 9.0, 12.0),
                    (False, 10.0, 13.0),
                    (False, 11.0, 14.0),
                    (False, 12.0, 15.0),
                    (False, 13.0, 16.0),
                    (False, 14.0, 17.0),
                ),
            )

    monkeypatch.setattr(
        main_window_module,
        "AnalysisAdvancedDialog",
        FakeAnalysisAdvancedDialog,
    )

    try:
        window.analysis_band_mode_combo.setCurrentIndex(
            window.analysis_band_mode_combo.findData(AnalysisBandMode.MANUAL_MULTI)
        )
        window._open_analysis_advanced_dialog()
        intent = window._build_intent()
        summary_text = window.analysis_advanced_summary_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert captured["band_mode"] is AnalysisBandMode.MANUAL_MULTI
    assert captured["parent"] is window
    assert intent.analysis.minimum_cell_support_fraction == 0.55
    assert intent.analysis.roi_quality_cutoff == 0.65
    assert intent.analysis.low_confidence_threshold == 0.25
    assert intent.analysis.auto_band_count == 3
    assert intent.analysis.export_advanced_files is False
    assert len(intent.analysis.manual_bands) == 2
    assert intent.analysis.manual_bands[0].low_hz == 6.0
    assert intent.analysis.manual_bands[1].high_hz == 9.5
    assert "2 manual bands" in summary_text


def test_main_window_accepts_ten_analysis_bands(tmp_path: Path) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        window.analysis_band_mode_combo.setCurrentIndex(
            window.analysis_band_mode_combo.findData(AnalysisBandMode.MANUAL_MULTI)
        )
        window.high_hz_spin.setValue(20.0)
        window.analysis_auto_band_count_spin.setValue(MAX_ANALYSIS_BANDS)
        assert len(window._manual_band_controls) == MAX_ANALYSIS_BANDS
        for band_index, (enabled_check, low_spin, high_spin) in enumerate(
            window._manual_band_controls,
            start=1,
        ):
            enabled_check.setChecked(True)
            low_spin.setValue(5.0 + band_index)
            high_spin.setValue(5.5 + band_index)
        intent = window._build_intent()
        is_valid = window._analysis_settings_valid(intent)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert intent.analysis.auto_band_count == MAX_ANALYSIS_BANDS
    assert len(intent.analysis.manual_bands) == MAX_ANALYSIS_BANDS
    assert intent.analysis.manual_bands[-1].band_id == f"band{MAX_ANALYSIS_BANDS:02d}"
    assert intent.analysis.manual_bands[-1].low_hz == 15.0
    assert is_valid is True


def test_main_window_defaults_to_repo_local_input_and_output_roots(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        output_folder = Path(window.output_folder_edit.text())
        input_folder = window._default_input_directory
        hidden_manual_band_container_visible = (
            window._analysis_manual_band_container.isVisible()
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert output_folder == tmp_path / "output"
    assert input_folder == tmp_path / "input"
    assert hidden_manual_band_container_visible is False
    assert output_folder.exists()
    assert input_folder.exists()


def test_main_window_builds_source_metadata_with_rotation_blockers(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    probe = FfprobeMediaInfo(
        width=1920,
        height=1080,
        fps=30.0,
        avg_fps=30.0,
        is_cfr=True,
        duration_seconds=2.0,
        frame_count=60,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        profile="High",
        pixel_format="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
        rotation_degrees=-90.0,
        sample_aspect_ratio=1.0,
    )

    try:
        source = window._build_source_metadata_from_probe(probe)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert source.has_unsupported_rotation is True


def test_main_window_builds_source_metadata_with_normalized_working_geometry(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    probe = FfprobeMediaInfo(
        width=720,
        height=480,
        fps=30.0,
        avg_fps=29.97,
        is_cfr=False,
        duration_seconds=2.0,
        frame_count=60,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        profile="High",
        pixel_format="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
        sample_aspect_ratio=8.0 / 9.0,
    )

    try:
        source = window._build_source_metadata_from_probe(probe)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert source.width == 640
    assert source.height == 480
    assert source.is_cfr is True
    assert source.source_is_cfr is False
    assert source.requires_cfr_normalization is True
    assert source.requires_square_pixel_normalization is True


def test_main_window_keeps_default_output_folder_when_source_is_chosen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    source_path = tmp_path / "input" / "source.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-bytes")
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(source_path), "Video files"),
    )
    window._start_source_probe = lambda: None
    window._start_fingerprint = lambda: None
    window._start_frame_extraction = lambda: None

    try:
        window._choose_source()
        output_folder = Path(window.output_folder_edit.text())
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert output_folder == tmp_path / "output"


def test_main_window_can_browse_for_output_folder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    selected_output = tmp_path / "custom-output"
    selected_output.mkdir()
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *args, **kwargs: str(selected_output),
    )

    try:
        window._choose_output_folder()
        output_folder = Path(window.output_folder_edit.text())
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert output_folder == selected_output


def test_main_window_rebuilds_resolution_options_for_normalized_source_geometry(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    probe = FfprobeMediaInfo(
        width=720,
        height=480,
        fps=30.0,
        avg_fps=29.97,
        is_cfr=False,
        duration_seconds=2.0,
        frame_count=60,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        profile="High",
        pixel_format="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
        sample_aspect_ratio=8.0 / 9.0,
    )
    window._start_baseline_band_analysis = lambda _info: None

    try:
        window._on_source_probe_complete(probe)
        processing_index = window._find_resolution_index(
            window.processing_resolution_combo,
            Resolution(853, 480),
        )
        window.processing_resolution_combo.setCurrentIndex(processing_index)
        processing_options = [
            window.processing_resolution_combo.itemText(index)
            for index in range(window.processing_resolution_combo.count())
        ]
        metadata_text = window.source_metadata_label.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert processing_options == ["640 x 480", "480 x 360"]
    assert "working 640 x 480" in metadata_text
    assert "auto CFR 29.970 fps" in metadata_text


def test_main_window_output_resolution_options_stay_even_for_codec_path(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    probe = FfprobeMediaInfo(
        width=853,
        height=480,
        fps=30.0,
        avg_fps=30.0,
        is_cfr=True,
        duration_seconds=2.0,
        frame_count=60,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        profile="High",
        pixel_format="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
    )
    window._start_baseline_band_analysis = lambda _info: None

    try:
        window._on_source_probe_complete(probe)
        processing_index = window._find_resolution_index(
            window.processing_resolution_combo,
            Resolution(852, 480),
        )
        window.processing_resolution_combo.setCurrentIndex(processing_index)
        processing_options = [
            window.processing_resolution_combo.itemText(index)
            for index in range(window.processing_resolution_combo.count())
        ]
        output_options = [
            window.output_resolution_combo.itemText(index)
            for index in range(window.output_resolution_combo.count())
        ]
        mirrored_output = window.output_resolution_combo.currentData()
        output_combo_enabled = window.output_resolution_combo.isEnabled()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert processing_options == ["852 x 480", "640 x 360"]
    assert output_options == ["852 x 480"]
    assert mirrored_output == Resolution(852, 480)
    assert output_combo_enabled is False


def test_main_window_output_resolution_mirrors_processing_selection(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    probe = FfprobeMediaInfo(
        width=1280,
        height=720,
        fps=30.0,
        avg_fps=30.0,
        is_cfr=True,
        duration_seconds=2.0,
        frame_count=60,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        profile="High",
        pixel_format="yuv420p",
        color_primaries="bt709",
        color_transfer="bt709",
        color_space="bt709",
        color_range="tv",
    )
    window._start_baseline_band_analysis = lambda _info: None

    try:
        window._on_source_probe_complete(probe)
        processing_index = window._find_resolution_index(
            window.processing_resolution_combo,
            Resolution(640, 360),
        )
        window.processing_resolution_combo.setCurrentIndex(processing_index)
        mirrored_output = window.output_resolution_combo.currentData()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert mirrored_output == Resolution(640, 360)


def test_main_window_restarts_probe_and_fingerprint_when_source_goes_stale(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    source_path = window._current_source_path
    assert source_path is not None
    calls = {"probe": 0, "fingerprint": 0, "frames": 0}
    window._start_source_probe = lambda: calls.__setitem__("probe", calls["probe"] + 1)
    window._start_fingerprint = lambda: calls.__setitem__(
        "fingerprint", calls["fingerprint"] + 1
    )
    window._start_frame_extraction = lambda: calls.__setitem__(
        "frames", calls["frames"] + 1
    )

    try:
        time.sleep(0.01)
        source_path.write_bytes(b"changed-source-bytes")
        window._poll_source_staleness()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window._controller.state is UiState.FINGERPRINT_PENDING
    assert window._current_fingerprint is None
    assert window._source_probe_info is None
    assert calls == {"probe": 1, "fingerprint": 1, "frames": 1}


def test_main_window_invalidates_ready_state_when_source_goes_missing(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    source_path = window._current_source_path
    assert source_path is not None
    calls = {"probe": 0, "fingerprint": 0, "frames": 0}
    window._start_source_probe = lambda: calls.__setitem__("probe", calls["probe"] + 1)
    window._start_fingerprint = lambda: calls.__setitem__(
        "fingerprint", calls["fingerprint"] + 1
    )
    window._start_frame_extraction = lambda: calls.__setitem__(
        "frames", calls["frames"] + 1
    )

    try:
        source_path.unlink()
        window._poll_source_staleness()

        assert window._controller.state is UiState.LOADED
        assert window._current_fingerprint is None
        assert window._source_probe_info is None
        assert window.source_metadata_label.text() == "Selected source file is missing."
        assert window.start_render_button.isEnabled() is False

        source_path.write_bytes(b"source-returned")
        window._poll_source_staleness()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window._controller.state is UiState.FINGERPRINT_PENDING
    assert calls == {"probe": 1, "fingerprint": 1, "frames": 1}


def test_main_window_dry_run_rejects_missing_source_even_before_stale_poll(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    source_path = window._current_source_path
    assert source_path is not None

    try:
        source_path.unlink()
        with pytest.raises(RuntimeError, match="no longer available"):
            window._build_shell_preflight_report()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()


def test_main_window_missing_source_poll_does_not_repeat_log_noise(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    source_path = window._current_source_path
    assert source_path is not None

    try:
        source_path.unlink()
        window._poll_source_staleness()
        first_log = window.event_log.toPlainText()
        window._poll_source_staleness()
        second_log = window.event_log.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    marker = "Selected source file is no longer available."
    assert first_log.count(marker) == 1
    assert second_log.count(marker) == 1


def test_main_window_start_render_warns_and_skips_supervisor_when_source_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    warnings: list[str] = []
    supervisor_calls = 0

    def fake_warning(_parent, _title, message) -> None:
        warnings.append(str(message))

    def fake_supervisor_factory(_request):
        nonlocal supervisor_calls
        supervisor_calls += 1
        return FakeSupervisor([])

    monkeypatch.setattr(main_window_module.QMessageBox, "warning", fake_warning)
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=fake_supervisor_factory,
    )
    source_path = window._current_source_path
    assert source_path is not None

    try:
        source_path.unlink()
        window._start_render()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert warnings == [
        "Selected source file is no longer available. Reload the source before starting a render."
    ]
    assert supervisor_calls == 0
    assert window._controller.state is UiState.READY


def test_main_window_ignores_persisted_output_folder_for_startup_default(
    tmp_path: Path,
) -> None:
    app = _app()
    persisted_output = tmp_path / "persisted-output"
    persisted_output.mkdir()
    state = main_window_module.PersistedAppState(
        preferences=main_window_module.default_preferences(tmp_path),
        last_used=main_window_module.LastUsedSettings(
            intent=JobIntent(
                phase=PhaseSettings(
                    magnification=12.0,
                    low_hz=3.0,
                    high_hz=8.0,
                    pyramid_type="complex_steerable",
                    sigma=1.0,
                    attenuate_other_frequencies=True,
                ),
                processing_resolution=Resolution(1280, 720),
                output_resolution=Resolution(1280, 720),
                resource_policy=ResourcePolicy.BALANCED,
            ),
            output_directory=str(persisted_output),
            output_stem="result",
            diagnostic_level=main_window_module.DiagnosticLevel.BASIC,
        ),
    )
    main_window_module.save_app_state(tmp_path / "settings.json", state)
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        output_folder = Path(window.output_folder_edit.text())
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert output_folder == tmp_path / "output"


def test_main_window_migrates_legacy_temp_root_to_repo_local_temp(tmp_path: Path) -> None:
    app = _app()
    state = main_window_module.PersistedAppState(
        preferences=main_window_module.default_preferences(tmp_path),
    )
    state = state.__class__(
        preferences=state.preferences.__class__(
            temp_root=str(tmp_path / "scratch"),
            diagnostics_root=state.preferences.diagnostics_root,
            diagnostic_level=state.preferences.diagnostic_level,
            diagnostics_cap_mb=state.preferences.diagnostics_cap_mb,
            retention_budget_gb=state.preferences.retention_budget_gb,
        ),
        last_used=state.last_used,
        version=state.version,
    )
    main_window_module.save_app_state(tmp_path / "settings.json", state)
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        temp_root = Path(window._preferences.temp_root)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert temp_root == tmp_path / "temp"
    assert temp_root.exists()
    assert os.environ["TEMP"] == str(temp_root)


def test_main_window_builds_render_request_in_timestamped_output_subfolder(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    fixed_time = datetime(2026, 3, 16, 18, 42, 7)
    monkeypatch.setattr(
        main_window_module,
        "_timestamped_render_directory",
        lambda base_output_directory, timestamp=None: (
            base_output_directory / fixed_time.strftime("%Y-%m-%d_%H-%M-%S")
        ),
    )

    try:
        request = window._build_render_request()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert request.paths.output_directory == tmp_path / "output" / "2026-03-16_18-42-07"
    assert request.paths.output_directory.exists()
    assert request.paths.final_mp4_path == request.paths.output_directory / "render.mp4"


def test_timestamped_render_directory_adds_suffix_when_second_collides(tmp_path: Path) -> None:
    first = tmp_path / "output" / "2026-03-16_18-42-07"
    first.mkdir(parents=True)

    second = main_window_module._timestamped_render_directory(
        tmp_path / "output",
        timestamp=datetime(2026, 3, 16, 18, 42, 7),
    )

    assert second == tmp_path / "output" / "2026-03-16_18-42-07_02"


def test_main_window_start_render_reaches_complete_and_prepare_new_run(tmp_path: Path) -> None:
    app = _app()
    complete_snapshot = RenderSupervisorSnapshot(
        phase="complete",
        stage="finalize",
        frames_completed=60,
        total_frames=60,
        terminal_message_type="job_completed",
    )
    fake_supervisor = FakeSupervisor(
        [
            RenderPollResult(
                snapshot=RenderSupervisorSnapshot(
                    phase="rendering",
                    stage="encode",
                    frames_completed=30,
                    total_frames=60,
                ),
                events=(),
            ),
            RenderPollResult(snapshot=complete_snapshot, events=()),
        ]
    )
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: fake_supervisor,
    )

    try:
        window._confirm_preflight_warnings = lambda _report: True
        window._start_render()
        assert fake_supervisor.started is True
        assert window._controller.state is UiState.PREFLIGHT_CHECK
        assert window.state_label.text() == "Preflight Check"

        window._poll_render_supervisor()
        assert window._controller.state is UiState.RENDERING
        assert window.state_label.text() == "Rendering"

        window._poll_render_supervisor()
        assert window._controller.state is UiState.COMPLETE
        assert window.state_label.text() == "Complete"
        assert fake_supervisor.closed is True
        assert window.eta_label.text() == "-"

        window._prepare_new_run()
        assert window._controller.state is UiState.READY
        assert window.state_label.text() == "Ready"
        assert window.eta_label.text() == "-"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()


def test_main_window_cancel_render_calls_supervisor_cancel(tmp_path: Path) -> None:
    app = _app()
    fake_supervisor = FakeSupervisor(
        [
            RenderPollResult(
                snapshot=RenderSupervisorSnapshot(
                    phase="rendering",
                    stage="decode",
                    frames_completed=1,
                    total_frames=60,
                ),
                events=(),
            )
        ]
    )
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: fake_supervisor,
    )

    try:
        window._confirm_preflight_warnings = lambda _report: True
        window._start_render()
        window._cancel_render()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert fake_supervisor.cancelled is True


def test_main_window_dry_run_populates_preflight_report(tmp_path: Path) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        window._run_dry_run_validation()
        report_text = window.preflight_report_view.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "Nyquist limit" in report_text
    assert "Selected band" in report_text
    assert "Source timing: CFR" in report_text
    assert (
        f"Resource policy: {window.resource_policy_combo.currentData().value}"
        in report_text
    )
    assert "Active scratch required:" in report_text
    assert "RAM required:" in report_text
    assert "MB" in report_text
    assert "Warnings:" in report_text


def test_main_window_applies_suggested_band_when_user_has_not_edited(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-bytes")
    window._current_source_path = source_path
    suggestion = FrequencyBandSuggestion(
        low_hz=0.15,
        high_hz=0.45,
        peak_hz=0.2,
        analysis_fps=15.0,
        confidence=0.5,
    )

    try:
        window._on_baseline_band_complete(str(source_path), suggestion)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window.low_hz_spin.value() == 0.15
    assert window.high_hz_spin.value() == 0.45
    assert (
        window.suggested_band_label.text()
        == "Suggested band: 0.15 Hz to 0.45 Hz (peak 0.20 Hz)"
    )


def test_main_window_keeps_user_band_when_suggestion_arrives_late(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"source-bytes")
    window._current_source_path = source_path
    window.low_hz_spin.setValue(1.2)
    window.high_hz_spin.setValue(2.4)
    window._band_user_edited = True
    suggestion = FrequencyBandSuggestion(
        low_hz=0.15,
        high_hz=0.45,
        peak_hz=0.2,
        analysis_fps=15.0,
        confidence=0.5,
    )

    try:
        window._on_baseline_band_complete(str(source_path), suggestion)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window.low_hz_spin.value() == 1.2
    assert window.high_hz_spin.value() == 2.4
    assert window.apply_suggested_band_button.isEnabled() is True


def test_main_window_updates_first_frame_preview(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")
    rgb24 = bytes(
        [
            255,
            0,
            0,
            255,
            0,
            0,
            255,
            0,
            0,
            255,
            0,
            0,
        ]
    )
    frame = type("FakeFrame", (), {"rgb24": rgb24, "width": 2, "height": 2})()

    try:
        window._on_frame_pair_complete(frame, frame)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window.first_frame_preview_label.pixmap() is not None


def test_main_window_keeps_top_content_scrollable_and_bottom_widgets_pinned(
    tmp_path: Path,
) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        layout = window.centralWidget().layout()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert layout.indexOf(window.content_scroll_area) == 0
    assert layout.indexOf(window.event_log) > layout.indexOf(window.content_scroll_area)
    assert window.content_scroll_area.widget() is window.scroll_content


def test_main_window_exposes_watermark_style_anchors(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        central = window.centralWidget()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert central is not None
    assert central.objectName() == "appShell"
    assert window.scroll_content.objectName() == "scrollContent"
    assert window.content_scroll_area.viewport().objectName() == "contentViewport"


def test_main_window_keeps_hardware_controls_in_core_settings(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        group_titles = {group.title() for group in window.findChildren(QGroupBox)}
        tooltip_text = window.output_resolution_combo.toolTip()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "Performance" not in group_titles
    assert not hasattr(window, "upscale_note")
    assert (
        tooltip_text
        == "Output resolution mirrors processing resolution for the current pipeline."
    )


def test_main_window_formats_preflight_report_with_grouped_megabytes(tmp_path: Path) -> None:
    app = _app()
    window = MainWindow(state_path=tmp_path / "settings.json")

    try:
        window._set_preflight_report_text(
            details={
                "source_fps": 29.97,
                "source_is_cfr": True,
                "nyquist_limit_hz": 14.985,
                "selected_low_hz": 0.15,
                "selected_high_hz": 0.9,
                "resource_policy": "balanced",
                "active_scratch_required_bytes": 1234 * 1024 * 1024,
                "ram_required_bytes": 5678 * 1024 * 1024,
                "output_staging_required_bytes": 90 * 1024 * 1024,
            },
            warnings=[],
            blockers=[],
        )
        report_text = window.preflight_report_view.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "Active scratch required: 1,234 MB" in report_text
    assert "RAM required: 5,678 MB" in report_text
    assert "Output staging required: 90 MB" in report_text


def test_main_window_blocks_render_until_preflight_warnings_are_acknowledged(
    tmp_path: Path,
) -> None:
    app = _app()
    fake_supervisor = FakeSupervisor(
        [
            RenderPollResult(
                snapshot=RenderSupervisorSnapshot(
                    phase="rendering",
                    stage="decode",
                    frames_completed=1,
                    total_frames=60,
                ),
                events=(),
            )
        ]
    )
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: fake_supervisor,
    )
    warning_report = PreflightReport(
        issues=(
            PreflightIssue(
                severity=PreflightSeverity.WARNING,
                code="eight_bit_input_warning",
                message="8-bit input may show banding or quantization artifacts at higher magnification.",
            ),
        ),
        nyquist_limit_hz=15.0,
        active_scratch_required_bytes=123 * 1024 * 1024,
        ram_required_bytes=456 * 1024 * 1024,
        output_staging_required_bytes=78 * 1024 * 1024,
    )

    try:
        window._build_shell_preflight_report = lambda: warning_report
        window._confirm_preflight_warnings = lambda _report: False
        window._start_render()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert fake_supervisor.started is False
    assert window._controller.state is UiState.READY


def test_main_window_starts_render_after_preflight_warning_acknowledgement(
    tmp_path: Path,
) -> None:
    app = _app()
    fake_supervisor = FakeSupervisor(
        [
            RenderPollResult(
                snapshot=RenderSupervisorSnapshot(
                    phase="rendering",
                    stage="decode",
                    frames_completed=1,
                    total_frames=60,
                ),
                events=(),
            )
        ]
    )
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: fake_supervisor,
    )
    warning_report = PreflightReport(
        issues=(
            PreflightIssue(
                severity=PreflightSeverity.WARNING,
                code="eight_bit_input_warning",
                message="8-bit input may show banding or quantization artifacts at higher magnification.",
            ),
        ),
        nyquist_limit_hz=15.0,
        active_scratch_required_bytes=123 * 1024 * 1024,
        ram_required_bytes=456 * 1024 * 1024,
        output_staging_required_bytes=78 * 1024 * 1024,
    )

    try:
        window._build_shell_preflight_report = lambda: warning_report
        window._confirm_preflight_warnings = lambda _report: True
        window._start_render()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert fake_supervisor.started is True
    assert window._controller.state is UiState.PREFLIGHT_CHECK


def test_main_window_stays_in_preflight_until_worker_reports_rendering(
    tmp_path: Path,
) -> None:
    app = _app()
    fake_supervisor = FakeSupervisor(
        [
            RenderPollResult(
                snapshot=RenderSupervisorSnapshot(
                    phase="preflight",
                    stage="preflight",
                ),
                events=(),
            )
        ]
    )
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: fake_supervisor,
    )

    try:
        window._confirm_preflight_warnings = lambda _report: True
        window._start_render()
        window._poll_render_supervisor()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window._controller.state is UiState.PREFLIGHT_CHECK


def test_main_window_restores_ready_state_when_worker_launch_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    messages: list[str] = []

    def fake_critical(_parent, _title, message) -> None:
        messages.append(str(message))

    monkeypatch.setattr(main_window_module.QMessageBox, "critical", fake_critical)
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: (_ for _ in ()).throw(
            RuntimeError("worker bootstrap failed")
        ),
    )

    try:
        window._confirm_preflight_warnings = lambda _report: True
        window._start_render()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert messages == ["worker bootstrap failed"]
    assert window._controller.state is UiState.READY
    assert window.render_stage_label.text() == "Idle"
    assert window.render_progress_label.text() == "No active render."


def test_main_window_failed_outcome_dialog_uses_diagnostics_and_job_purge_actions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    request = window._build_render_request()
    request.paths.diagnostics_directory.mkdir(parents=True, exist_ok=True)
    request.paths.scratch_directory.mkdir(parents=True, exist_ok=True)
    window._active_render_request = request
    captured: dict[str, object] = {}

    class FakeDialog:
        def __init__(self, *, outcome, open_output, open_video, parent=None) -> None:
            captured["outcome"] = outcome
            captured["open_output"] = open_output
            captured["open_video"] = open_video

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "TerminalOutcomeDialog", FakeDialog)

    try:
        window._show_terminal_outcome_dialog(
            RenderSupervisorSnapshot(phase="failed")
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    outcome = captured["outcome"]
    assert outcome.primary_action_label == "Open Diagnostics"
    assert outcome.secondary_action_label == "Purge Failed-run Files"
    assert outcome.primary_action_enabled is True
    assert outcome.secondary_action_enabled is True
    assert captured["open_output"] == window._open_diagnostics_folder
    assert captured["open_video"] == window._purge_job_temp_files


def test_main_window_terminal_summary_includes_failure_detail() -> None:
    app = _app()
    window = MainWindow()
    snapshot = RenderSupervisorSnapshot(
        phase="failed",
        failure_classification="internal_processing_exception",
        failure_detail="RuntimeError: synthetic failure",
    )

    try:
        summary = window._format_terminal_summary(snapshot)
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "internal_processing_exception" in summary
    assert "RuntimeError: synthetic failure" in summary


def test_main_window_sync_render_metrics_resets_when_stage_counter_restarts(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        window._render_started_at = time.monotonic() - 2.0
        window._progress_metric_stage = "decode"
        window._progress_metric_total_frames = 860
        window._last_progress_frame_count = 860
        window._last_progress_at = time.monotonic() - 1.0
        window._mean_seconds_per_frame = 0.5
        window.mean_frame_label.setText("0.500 s")
        window.eta_label.setText("10.0 s")

        window._sync_render_metrics(
            RenderSupervisorSnapshot(
                phase="rendering",
                stage="phase_processing",
                frames_completed=0,
                total_frames=860,
            )
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window._last_progress_frame_count == 0
    assert window._mean_seconds_per_frame is None
    assert window.mean_frame_label.text() == "-"
    assert window.eta_label.text() == "Estimating..."


def test_main_window_sync_render_metrics_updates_eta_from_stage_progress(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        now = time.monotonic()
        window._render_started_at = now - 5.0
        window._progress_metric_stage = "phase_processing"
        window._progress_metric_total_frames = 60
        window._last_progress_frame_count = 10
        window._last_progress_at = now - 2.0
        window._mean_seconds_per_frame = None

        window._sync_render_metrics(
            RenderSupervisorSnapshot(
                phase="rendering",
                stage="phase_processing",
                frames_completed=20,
                total_frames=60,
            )
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window.mean_frame_label.text() in {"0.200 s", "0.201 s"}
    assert window.eta_label.text() == "8.0 s"


def test_main_window_phase_processing_eta_uses_token_progress_before_batch_completion(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        now = time.monotonic()
        window._render_started_at = now - 5.0
        window._progress_metric_stage = "phase_processing"
        window._progress_metric_total_frames = 100
        window._last_progress_frame_count = 20.0
        window._last_progress_at = now - 2.0
        window._mean_seconds_per_frame = None

        window._consume_render_event_for_metrics(
            RenderEvent(
                message_type="progress_update",
                payload={
                    "stage": "phase_processing",
                    "detail": "decode_chunk_ready",
                    "progress_token": "decode_chunk:20",
                    "decode_frames_completed": 20,
                },
                received_at=now - 1.5,
            )
        )
        window._consume_render_event_for_metrics(
            RenderEvent(
                message_type="progress_update",
                payload={
                    "stage": "phase_processing",
                    "detail": "decode_chunk_ready",
                    "progress_token": "decode_chunk:40",
                    "decode_frames_completed": 40,
                },
                received_at=now - 1.0,
            )
        )
        window._consume_render_event_for_metrics(
            RenderEvent(
                message_type="progress_update",
                payload={
                    "stage": "phase_processing",
                    "detail": "warp_frame_10_of_20",
                    "progress_token": "phase_processing:20:9",
                },
                received_at=now,
            )
        )

        window._sync_render_metrics(
            RenderSupervisorSnapshot(
                phase="rendering",
                stage="phase_processing",
                frames_completed=20,
                total_frames=100,
            )
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window.mean_frame_label.text() == "0.120 s"
    assert window.eta_label.text() == "7.6 s"


def test_main_window_phase_processing_token_progress_stays_monotonic_across_overlap(
    tmp_path: Path,
) -> None:
    app = _app()
    window = _ready_window(tmp_path)

    try:
        window._phase_nominal_chunk_frame_count = 20
        window._phase_virtual_frames_completed = 40.0
        window._consume_render_event_for_metrics(
            RenderEvent(
                message_type="progress_update",
                payload={
                    "stage": "phase_processing",
                    "detail": "motion_grid_tile_2_of_4",
                    "progress_token": "phase_processing:20:5",
                },
                received_at=time.monotonic(),
            )
        )
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert window._phase_virtual_frames_completed == 40.0


def test_preflight_warning_dialog_keeps_warning_details_visible() -> None:
    app = _app()
    dialog = PreflightWarningDialog(
        active_scratch_text="14,693 MB",
        ram_text="9,498 MB",
        output_staging_text="198 MB",
        warning_messages=[
            "Selected frequency band is narrow and may not be well supported by the recording.",
            "8-bit input may show banding or quantization artifacts at higher magnification.",
        ],
    )

    try:
        details_text = dialog.warning_details_view.toPlainText()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert "Selected frequency band is narrow" in details_text
    assert "8-bit input may show banding" in details_text


def test_main_window_heartbeat_events_flash_indicator_without_log_spam(
    tmp_path: Path,
) -> None:
    app = _app()
    received_at = time.monotonic()
    fake_supervisor = FakeSupervisor(
        [
            RenderPollResult(
                snapshot=RenderSupervisorSnapshot(
                    phase="rendering",
                    stage="decode",
                    frames_completed=1,
                    total_frames=60,
                ),
                events=(
                    RenderEvent(
                        message_type="heartbeat",
                        payload={},
                        received_at=received_at,
                    ),
                    RenderEvent(
                        message_type="stage_started",
                        payload={"stage": "decode", "total_frames": 60},
                        received_at=received_at,
                    ),
                ),
            )
        ]
    )
    window = _ready_window(
        tmp_path,
        render_supervisor_factory=lambda _request: fake_supervisor,
    )

    try:
        window._confirm_preflight_warnings = lambda _report: True
        window._start_render()
        window._poll_render_supervisor()
        log_text = window.event_log.toPlainText()
        dot_style = window.heartbeat_dot_label.styleSheet()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert "heartbeat" not in log_text.lower()
    assert "Stage started: decode." in log_text
    assert "#ff2f2f" in dot_style


def test_main_window_purges_only_failed_run_temp_files(tmp_path: Path) -> None:
    app = _app()
    window = _ready_window(tmp_path)
    diagnostics_root = tmp_path / "diagnostics"
    scratch_root = tmp_path / "scratch"
    failed_diag = diagnostics_root / "job-failed"
    completed_diag = diagnostics_root / "job-complete"
    failed_scratch = scratch_root / "job-failed"
    completed_scratch = scratch_root / "job-complete"
    failed_diag.mkdir(parents=True)
    completed_diag.mkdir(parents=True)
    failed_scratch.mkdir(parents=True)
    completed_scratch.mkdir(parents=True)
    (failed_diag / "diagnostics_bundle.json").write_text(
        '{"status":"failed"}',
        encoding="utf-8",
    )
    (completed_diag / "diagnostics_bundle.json").write_text(
        '{"status":"completed"}',
        encoding="utf-8",
    )
    (failed_scratch / "payload.bin").write_bytes(b"x")
    (completed_scratch / "payload.bin").write_bytes(b"x")
    window._preferences = window._preferences.__class__(
        temp_root=str(scratch_root),
        diagnostics_root=str(diagnostics_root),
        diagnostic_level=window._preferences.diagnostic_level,
        diagnostics_cap_mb=window._preferences.diagnostics_cap_mb,
        retention_budget_gb=window._preferences.retention_budget_gb,
    )

    try:
        window._purge_failed_run_temp_files()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
        app.quit()

    assert not failed_diag.exists()
    assert not failed_scratch.exists()
    assert completed_diag.exists()
    assert completed_scratch.exists()

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
