"""This file owns the PyQt shell window so source selection, drift review, dry-run validation, worker supervision, and cleanup controls stay in the GUI while heavy render work stays out of process."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil
from PyQt6.QtCore import QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from phase_motion_app.app.drift_editor import AnalysisRoiEditorDialog, DriftEditorDialog
from phase_motion_app.app.terminal_outcome import TerminalOutcomeData, TerminalOutcomeDialog
from phase_motion_app.core.acceleration import (
    detect_acceleration_capability,
    resolve_acceleration_request,
)
from phase_motion_app.core.baseline_band import (
    FrequencyBandSuggestion,
    analyze_source_frequency_band,
)
from phase_motion_app.core.drift import DriftAssessment, estimate_global_drift
from phase_motion_app.core.ffprobe import FfprobeMediaInfo, FfprobeRunner
from phase_motion_app.core.job_state import (
    InvalidTransitionError,
    SingleJobController,
    SourceSnapshot,
    UiState,
    detect_stale_source,
)
from phase_motion_app.core.masking import (
    summarize_automatic_analysis_roi,
    validate_exclusion_zones,
)
from phase_motion_app.core.media_tools import extract_first_frame, extract_last_frame
from phase_motion_app.core.models import (
    AnalysisBand,
    AnalysisBandMode,
    AnalysisSettings,
    DiagnosticLevel,
    JobIntent,
    PhaseSettings,
    Resolution,
    ResourcePolicy,
)
from phase_motion_app.core.preflight import (
    choose_scheduler_inputs,
    PreflightInputs,
    PreflightReport,
    ResourceBudget,
    SourceMetadata,
    run_preflight,
)
from phase_motion_app.core.retention import (
    build_retained_entry,
    measure_retained_roots_bytes,
    plan_oldest_first_purge,
    purge_retained_entries,
)
from phase_motion_app.core.render_job import RenderPaths, RenderRequest
from phase_motion_app.core.render_supervisor import (
    RenderEvent,
    RenderSupervisor,
    RenderSupervisorSnapshot,
    WorkerLaunchPlan,
)
from phase_motion_app.core.sidecar import SidecarValidationError, load_reusable_intent
from phase_motion_app.core.source_normalization import (
    SourceNormalizationPlan,
    build_source_normalization_plan,
)
from phase_motion_app.core.settings_store import (
    LastUsedSettings,
    PersistedAppState,
    default_preferences,
    load_app_state,
    migrate_legacy_temp_root,
    save_app_state,
)
from phase_motion_app.core.watchdog import WatchdogThresholds
from phase_motion_app.worker.bootstrap import RenderWorkerConfig, render_worker_entry


def _detect_checkout_root() -> Path | None:
    """Return the source checkout root when running from a repo, else None."""

    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").exists():
        return candidate
    return None


def _configure_process_temp_root(temp_root: Path) -> None:
    """Keep generic temp-file APIs under the repo/runtime temp root so the app does not spill onto the system drive."""

    temp_root.mkdir(parents=True, exist_ok=True)
    temp_root_text = str(temp_root)
    tempfile.tempdir = temp_root_text
    for variable_name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[variable_name] = temp_root_text


def _timestamped_render_directory(
    base_output_directory: Path,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Build a collision-safe per-render subdirectory under the selected output root."""

    moment = datetime.now() if timestamp is None else timestamp
    folder_name = moment.strftime("%Y-%m-%d_%H-%M-%S")
    candidate = base_output_directory / folder_name
    suffix = 2
    while candidate.exists():
        candidate = base_output_directory / f"{folder_name}_{suffix:02d}"
        suffix += 1
    return candidate


def _resolution_options_for_source(
    source_resolution: Resolution,
    presets: tuple[Resolution, ...],
) -> tuple[Resolution, ...]:
    """Fit the fixed UI resolution presets to the current source aspect ratio."""

    seen: set[tuple[int, int]] = set()
    options: list[Resolution] = []
    for preset in presets:
        scale = min(
            preset.width / source_resolution.width,
            preset.height / source_resolution.height,
            1.0,
        )
        width = max(1, int(round(source_resolution.width * scale)))
        height = max(1, int(round(source_resolution.height * scale)))
        key = (width, height)
        if key in seen:
            continue
        seen.add(key)
        options.append(Resolution(width=width, height=height))
    return tuple(options)


def _codec_safe_output_resolution(
    resolution: Resolution,
    *,
    processing_resolution: Resolution,
) -> Resolution | None:
    """Clamp one processing/output geometry to the current 4:2:0 encoder requirements without ever upscaling."""

    width = min(resolution.width, processing_resolution.width)
    height = min(resolution.height, processing_resolution.height)
    if width > 1 and width % 2 != 0:
        width -= 1
    if height > 1 and height % 2 != 0:
        height -= 1
    if width <= 0 or height <= 0:
        return None
    return Resolution(width=width, height=height)


class FingerprintWorker(QThread):
    """This thread computes the full-file SHA-256 fingerprint so the shell can stay responsive during the authoritative source identity step."""

    progress = pyqtSignal(int)
    completed = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(self, source_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_path = source_path

    def run(self) -> None:
        try:
            file_size = max(self.source_path.stat().st_size, 1)
            digest = hashlib.sha256()
            bytes_read = 0
            with self.source_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    bytes_read += len(chunk)
                    self.progress.emit(int((bytes_read / file_size) * 100))
            self.completed.emit(digest.hexdigest(), self.source_path.stat().st_size)
        except Exception as exc:  # pragma: no cover - defensive UI path.
            self.failed.emit(str(exc))


class SourceProbeWorker(QThread):
    """This thread runs the fast ffprobe-style source probe so source dimensions and timing metadata appear without blocking the shell."""

    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_path = source_path

    def run(self) -> None:
        try:
            info = FfprobeRunner().run(self.source_path)
            self.completed.emit(info)
        except Exception as exc:  # pragma: no cover - defensive UI path.
            self.failed.emit(str(exc))


class FramePairWorker(QThread):
    """This thread extracts first and last decoded frames for the drift-check editor without blocking the main event loop."""

    completed = pyqtSignal(object, object, object)
    failed = pyqtSignal(str)

    def __init__(self, source_path: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source_path = source_path

    def run(self) -> None:
        try:
            first = extract_first_frame(self.source_path)
            last = extract_last_frame(self.source_path)
            # Drift estimation is advisory-only. If it cannot produce a reliable
            # number, the first/last review imagery should still load normally.
            try:
                drift_estimate = estimate_global_drift(first, last)
            except Exception:
                drift_estimate = None
            self.completed.emit(first, last, drift_estimate)
        except Exception as exc:  # pragma: no cover - defensive UI path.
            self.failed.emit(str(exc))


class BaselineBandWorker(QThread):
    """This thread runs a light motion-spectrum pass so source import can offer a baseline frequency band without blocking the shell."""

    completed = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        source_path: Path,
        probe_info: FfprobeMediaInfo,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_path = source_path
        self.probe_info = probe_info

    def run(self) -> None:
        try:
            suggestion = analyze_source_frequency_band(self.source_path, self.probe_info)
            self.completed.emit(str(self.source_path), suggestion)
        except Exception as exc:  # pragma: no cover - defensive UI path.
            self.failed.emit(str(self.source_path), str(exc))


class AspectPreviewLabel(QLabel):
    """This label keeps the first-frame preview at a stable aspect ratio instead of letting the form layout squash it into a thin strip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(120)
        self.setMaximumHeight(220)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._source_pixmap: QPixmap | None = None

    def set_preview_image(self, image: QImage | None, placeholder: str) -> None:
        if image is None:
            self._source_pixmap = None
            self.clear()
            self.setText(placeholder)
            return

        self._source_pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming convention.
        self._refresh_pixmap()
        super().resizeEvent(event)

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt naming convention.
        if width <= 0:
            return self.minimumHeight()
        return min(self.maximumHeight(), max(self.minimumHeight(), int(width * 9 / 16)))

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt naming convention.
        return True

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class PreflightWarningDialog(QDialog):
    """This dialog keeps warning details visible instead of hiding them behind a collapsible details pane."""

    def __init__(
        self,
        *,
        active_scratch_text: str,
        ram_text: str,
        output_staging_text: str,
        warning_messages: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Acknowledge Pre-flight Warnings")
        self.resize(520, 280)

        layout = QVBoxLayout(self)
        summary_label = QLabel(
            "Shell pre-flight found warnings. You must acknowledge them before rendering."
        )
        summary_label.setWordWrap(True)
        metrics_label = QLabel(
            "\n".join(
                (
                    f"Active scratch required: {active_scratch_text}",
                    f"RAM required: {ram_text}",
                    f"Output staging required: {output_staging_text}",
                )
            )
        )
        metrics_label.setWordWrap(True)
        self.warning_details_view = QPlainTextEdit()
        self.warning_details_view.setReadOnly(True)
        self.warning_details_view.setMinimumHeight(120)
        self.warning_details_view.setPlainText(
            "\n".join(f"- {message}" for message in warning_messages)
        )
        buttons = QDialogButtonBox(self)
        buttons.addButton(
            "Acknowledge and Render",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(summary_label)
        layout.addWidget(metrics_label)
        layout.addWidget(self.warning_details_view)
        layout.addWidget(buttons)


@dataclass(frozen=True)
class AnalysisAdvancedDialogResult:
    """This result object returns the advanced analysis controls without pushing diagnostics settings through the same UI path."""

    minimum_cell_support_fraction: float
    roi_quality_cutoff: float
    low_confidence_threshold: float
    auto_band_count: int
    export_advanced_files: bool
    manual_band_values: tuple[tuple[bool, float, float], ...]


class AnalysisAdvancedDialog(QDialog):
    """This dialog keeps the detailed quantitative-analysis knobs out of the main instrument surface while preserving the existing validation model."""

    def __init__(
        self,
        *,
        band_mode: AnalysisBandMode,
        minimum_cell_support_fraction: float,
        roi_quality_cutoff: float,
        low_confidence_threshold: float,
        auto_band_count: int,
        export_advanced_files: bool,
        manual_band_values: tuple[tuple[bool, float, float], ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._band_mode = band_mode
        self.setWindowTitle("Advanced Analysis")
        self.resize(560, 420)

        layout = QVBoxLayout(self)
        summary_label = QLabel(
            "Band mode is chosen on the main screen. Use this popout for advanced analysis thresholds, export options, and manual band values."
        )
        summary_label.setWordWrap(True)
        mode_label = QLabel(f"Current band mode: {self._band_mode.value.replace('_', ' ')}")
        mode_label.setWordWrap(True)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.analysis_support_spin = QDoubleSpinBox()
        self.analysis_support_spin.setDecimals(2)
        self.analysis_support_spin.setRange(0.05, 1.0)
        self.analysis_support_spin.setSingleStep(0.05)
        self.analysis_support_spin.setValue(minimum_cell_support_fraction)
        self.analysis_quality_cutoff_spin = QDoubleSpinBox()
        self.analysis_quality_cutoff_spin.setDecimals(2)
        self.analysis_quality_cutoff_spin.setRange(0.0, 1.0)
        self.analysis_quality_cutoff_spin.setSingleStep(0.05)
        self.analysis_quality_cutoff_spin.setValue(roi_quality_cutoff)
        self.analysis_low_confidence_spin = QDoubleSpinBox()
        self.analysis_low_confidence_spin.setDecimals(2)
        self.analysis_low_confidence_spin.setRange(0.0, 1.0)
        self.analysis_low_confidence_spin.setSingleStep(0.05)
        self.analysis_low_confidence_spin.setValue(low_confidence_threshold)
        self.analysis_auto_band_count_spin = QSpinBox()
        self.analysis_auto_band_count_spin.setRange(1, 5)
        self.analysis_auto_band_count_spin.setValue(auto_band_count)
        self.analysis_export_advanced_checkbox = QCheckBox(
            "Export advanced/internal analysis files"
        )
        self.analysis_export_advanced_checkbox.setChecked(export_advanced_files)

        self._manual_band_controls: list[tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox]] = []
        self._manual_band_container = QWidget(self)
        manual_band_layout = QVBoxLayout(self._manual_band_container)
        manual_band_layout.setContentsMargins(0, 0, 0, 0)
        manual_band_layout.setSpacing(4)
        band_defaults = manual_band_values or tuple(
            (index == 0, 5.0 + index, 12.0 + index) for index in range(5)
        )
        for band_index in range(5):
            enabled, low_hz, high_hz = band_defaults[band_index]
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            enabled_check = QCheckBox(f"Band {band_index + 1}")
            enabled_check.setChecked(enabled)
            low_spin = QDoubleSpinBox()
            low_spin.setDecimals(3)
            low_spin.setRange(0.001, 1000.0)
            low_spin.setValue(low_hz)
            low_spin.setSuffix(" Hz")
            high_spin = QDoubleSpinBox()
            high_spin.setDecimals(3)
            high_spin.setRange(0.001, 1000.0)
            high_spin.setValue(high_hz)
            high_spin.setSuffix(" Hz")
            row_layout.addWidget(enabled_check)
            row_layout.addWidget(QLabel("Low"))
            row_layout.addWidget(low_spin)
            row_layout.addWidget(QLabel("High"))
            row_layout.addWidget(high_spin)
            manual_band_layout.addWidget(row)
            self._manual_band_controls.append((enabled_check, low_spin, high_spin))
            enabled_check.toggled.connect(self._sync_band_controls)

        form.addRow("Minimum cell support", self.analysis_support_spin)
        form.addRow("ROI quality cutoff", self.analysis_quality_cutoff_spin)
        form.addRow("Low-confidence threshold", self.analysis_low_confidence_spin)
        form.addRow("Auto-band count", self.analysis_auto_band_count_spin)
        form.addRow("", self.analysis_export_advanced_checkbox)
        form.addRow("Manual bands", self._manual_band_container)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(summary_label)
        layout.addWidget(mode_label)
        layout.addLayout(form)
        layout.addWidget(buttons)
        self._sync_band_controls()

    def _sync_band_controls(self) -> None:
        auto_mode = self._band_mode is AnalysisBandMode.AUTO
        self.analysis_auto_band_count_spin.setEnabled(auto_mode)
        for index, (enabled_check, low_spin, high_spin) in enumerate(self._manual_band_controls):
            band_enabled = not auto_mode and (
                self._band_mode is AnalysisBandMode.MANUAL_MULTI
                or (
                    self._band_mode is AnalysisBandMode.MANUAL_SINGLE
                    and index == 0
                )
            )
            enabled_check.setEnabled(band_enabled)
            low_spin.setEnabled(band_enabled and enabled_check.isChecked())
            high_spin.setEnabled(band_enabled and enabled_check.isChecked())

    def result_data(self) -> AnalysisAdvancedDialogResult:
        return AnalysisAdvancedDialogResult(
            minimum_cell_support_fraction=float(self.analysis_support_spin.value()),
            roi_quality_cutoff=float(self.analysis_quality_cutoff_spin.value()),
            low_confidence_threshold=float(self.analysis_low_confidence_spin.value()),
            auto_band_count=int(self.analysis_auto_band_count_spin.value()),
            export_advanced_files=self.analysis_export_advanced_checkbox.isChecked(),
            manual_band_values=tuple(
                (
                    enabled_check.isChecked(),
                    float(low_spin.value()),
                    float(high_spin.value()),
                )
                for enabled_check, low_spin, high_spin in self._manual_band_controls
            ),
        )


class MainWindow(QMainWindow):
    """This window is a thin shell over the tested core services and intentionally avoids running heavy work in the GUI thread."""

    def __init__(
        self,
        *,
        render_supervisor_factory: Any | None = None,
        state_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Phase-based Motion Amplification Utility")
        self._controller = SingleJobController()
        self._render_supervisor_factory = (
            render_supervisor_factory or self._default_render_supervisor_factory
        )
        self._state_root = Path.home() / ".phase_motion_app"
        default_runtime_root = _detect_checkout_root() or self._state_root
        self._runtime_root = state_path.parent if state_path is not None else default_runtime_root
        self._state_path = state_path or (self._state_root / "settings.json")
        self._preferences = default_preferences(self._runtime_root)
        _configure_process_temp_root(Path(self._preferences.temp_root))
        self._default_input_directory = self._runtime_root / "input"
        self._default_output_directory = self._runtime_root / "output"
        for directory in (
            self._default_input_directory,
            self._default_output_directory,
            Path(self._preferences.temp_root),
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._current_source_path: Path | None = None
        self._current_fingerprint: str | None = None
        self._source_probe_info: FfprobeMediaInfo | None = None
        self._fingerprint_worker: FingerprintWorker | None = None
        self._probe_worker: SourceProbeWorker | None = None
        self._frame_pair_worker: FramePairWorker | None = None
        self._baseline_band_worker: BaselineBandWorker | None = None
        self._last_known_snapshot: SourceSnapshot | None = None
        self._first_frame_image: QImage | None = None
        self._last_frame_image: QImage | None = None
        self._suggested_frequency_band: FrequencyBandSuggestion | None = None
        self._band_user_edited = False
        self._applying_suggested_band = False
        self._exclusion_zones = ()
        self._mask_feather_px = 4.0
        self._analysis_roi = None
        self._drift_assessment = DriftAssessment()
        self._render_supervisor: Any | None = None
        self._active_render_request: RenderRequest | None = None
        self._last_terminal_snapshot: RenderSupervisorSnapshot | None = None
        self._render_started_at: float | None = None
        self._progress_metric_stage: str | None = None
        self._progress_metric_total_frames: int | None = None
        self._progress_metric_started_at: float | None = None
        self._last_progress_frame_count: float | None = None
        self._last_progress_at: float | None = None
        self._mean_seconds_per_frame: float | None = None
        self._phase_virtual_frames_completed: float | None = None
        self._phase_last_decode_counter: int | None = None
        self._phase_nominal_chunk_frame_count: int | None = None
        self._heartbeat_flash_until: float | None = None
        self._resolution_presets = (
            Resolution(1920, 1080),
            Resolution(1280, 720),
            Resolution(640, 360),
        )
        self._resolution_options = self._resolution_presets

        self._build_ui()
        self._load_persisted_state()
        self._wire_signals()
        self._update_analysis_controls()
        self._update_performance_status()
        self._refresh_state()

        self._stale_timer = QTimer(self)
        self._stale_timer.setInterval(2000)
        self._stale_timer.timeout.connect(self._poll_source_staleness)
        self._stale_timer.start()

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(60)
        self._render_timer.timeout.connect(self._poll_render_supervisor)

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("appShell")
        central.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.content_scroll_area = QScrollArea(self)
        self.content_scroll_area.setWidgetResizable(True)
        self.content_scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.content_scroll_area.viewport().setObjectName("contentViewport")
        self.content_scroll_area.viewport().setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )
        scroll_content = QWidget()
        scroll_content.setObjectName("scrollContent")
        scroll_content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.scroll_content = scroll_content
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)
        self.content_scroll_area.setWidget(scroll_content)

        source_group = QGroupBox("Source and Output")
        source_layout = QFormLayout(source_group)
        source_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        source_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setReadOnly(True)
        self.source_browse_button = QPushButton("Browse Source")
        self.metadata_load_button = QPushButton("Load from Export Metadata")
        source_buttons = QWidget()
        source_buttons_layout = QHBoxLayout(source_buttons)
        source_buttons_layout.setContentsMargins(0, 0, 0, 0)
        source_buttons_layout.addWidget(self.source_browse_button)
        source_buttons_layout.addWidget(self.metadata_load_button)
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setText(str(self._default_output_directory))
        self.output_browse_button = QPushButton("Browse Output")
        output_folder_row = QWidget()
        output_folder_layout = QHBoxLayout(output_folder_row)
        output_folder_layout.setContentsMargins(0, 0, 0, 0)
        output_folder_layout.addWidget(self.output_folder_edit)
        output_folder_layout.addWidget(self.output_browse_button)
        self.output_name_edit = QLineEdit("render")
        self.source_metadata_label = QLabel("No source metadata yet.")
        self.source_metadata_label.setWordWrap(True)
        self.first_frame_preview_label = AspectPreviewLabel()
        self.first_frame_preview_label.setObjectName("sourcePreview")
        self.first_frame_preview_label.set_preview_image(
            None,
            "First frame preview pending.",
        )
        source_layout.addRow("Source video", self.source_path_edit)
        source_layout.addRow("", source_buttons)
        source_layout.addRow("Output folder", output_folder_row)
        source_layout.addRow("Output name", self.output_name_edit)
        source_layout.addRow("Source metadata", self.source_metadata_label)
        source_layout.addRow("First frame", self.first_frame_preview_label)

        settings_group = QGroupBox("Core Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.magnification_spin = QDoubleSpinBox()
        self.magnification_spin.setDecimals(2)
        self.magnification_spin.setRange(0.01, 1000.0)
        self.magnification_spin.setSingleStep(0.5)
        self.magnification_spin.setValue(20.0)
        self.low_hz_spin = QDoubleSpinBox()
        self.low_hz_spin.setDecimals(3)
        self.low_hz_spin.setRange(0.001, 1000.0)
        self.low_hz_spin.setSingleStep(0.25)
        self.low_hz_spin.setValue(5.0)
        self.high_hz_spin = QDoubleSpinBox()
        self.high_hz_spin.setDecimals(3)
        self.high_hz_spin.setRange(0.001, 1000.0)
        self.high_hz_spin.setSingleStep(0.25)
        self.high_hz_spin.setValue(12.0)
        self.suggested_band_label = QLabel("Suggested band: waiting for source analysis.")
        self.suggested_band_label.setWordWrap(True)
        self.apply_suggested_band_button = QPushButton("Use Suggested Band")
        self.apply_suggested_band_button.setEnabled(False)
        self.resource_policy_combo = QComboBox()
        for policy in ResourcePolicy:
            self.resource_policy_combo.addItem(policy.value.title(), policy)
        self.hardware_acceleration_checkbox = QCheckBox(
            "Enable hardware acceleration"
        )
        self.hardware_acceleration_status_label = QLabel("")
        self.hardware_acceleration_status_label.setWordWrap(True)
        self.processing_resolution_combo = QComboBox()
        self.output_resolution_combo = QComboBox()
        self.output_resolution_combo.setEnabled(False)
        self.output_resolution_combo.setToolTip(
            "Output resolution mirrors processing resolution for the current pipeline."
        )
        self._rebuild_processing_resolution_options()
        self.performance_summary_label = QLabel(
            "Tied render resolution avoids a second resize pass. Hardware acceleration targets dense warp, resize, and FFT-based motion estimation. Pre-flight reports CPU fallback when acceleration cannot run."
        )
        self.performance_summary_label.setWordWrap(True)
        self.exclusion_editor_button = QPushButton("Drift Check / Mask Zones")
        self.exclusion_editor_button.setEnabled(False)
        settings_layout.addRow("Magnification", self.magnification_spin)
        settings_layout.addRow("Low cutoff (Hz)", self.low_hz_spin)
        settings_layout.addRow("High cutoff (Hz)", self.high_hz_spin)
        settings_layout.addRow("Suggested band", self.suggested_band_label)
        settings_layout.addRow("", self.apply_suggested_band_button)
        settings_layout.addRow("Resource policy", self.resource_policy_combo)
        settings_layout.addRow("", self.hardware_acceleration_checkbox)
        settings_layout.addRow("Acceleration support", self.hardware_acceleration_status_label)
        settings_layout.addRow("Processing resolution", self.processing_resolution_combo)
        settings_layout.addRow("Output resolution", self.output_resolution_combo)
        settings_layout.addRow("Performance notes", self.performance_summary_label)
        settings_layout.addRow("Mask editor", self.exclusion_editor_button)

        analysis_group = QGroupBox("Analysis")
        analysis_layout = QFormLayout(analysis_group)
        analysis_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.analysis_enabled_checkbox = QCheckBox("Enable quantitative analysis")
        self.analysis_enabled_checkbox.setChecked(True)
        self.analysis_roi_label = QLabel(summarize_automatic_analysis_roi(()))
        self.analysis_roi_label.setWordWrap(True)
        self.analysis_roi_button = QPushButton("Edit Analysis ROI")
        self.analysis_roi_button.setEnabled(False)
        self.analysis_band_mode_combo = QComboBox()
        self.analysis_band_mode_combo.addItem("Auto-bands", AnalysisBandMode.AUTO)
        self.analysis_band_mode_combo.addItem(
            "Manual single band", AnalysisBandMode.MANUAL_SINGLE
        )
        self.analysis_band_mode_combo.addItem(
            "Manual multiple bands", AnalysisBandMode.MANUAL_MULTI
        )
        self.analysis_band_summary_label = QLabel("Auto-bands will be generated from the ROI spectrum.")
        self.analysis_band_summary_label.setWordWrap(True)
        self.analysis_advanced_summary_label = QLabel(
            "Auto-bands up to 5, support 0.35, ROI quality 0.45, low-confidence 0.35."
        )
        self.analysis_advanced_summary_label.setWordWrap(True)
        self.analysis_advanced_button = QPushButton("Advanced Analysis...")
        analysis_layout.addRow("", self.analysis_enabled_checkbox)
        analysis_layout.addRow("ROI", self.analysis_roi_label)
        analysis_layout.addRow("", self.analysis_roi_button)
        analysis_layout.addRow("Band mode", self.analysis_band_mode_combo)
        analysis_layout.addRow("Band summary", self.analysis_band_summary_label)
        analysis_layout.addRow("Advanced summary", self.analysis_advanced_summary_label)
        analysis_layout.addRow("", self.analysis_advanced_button)

        advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout(advanced_group)
        advanced_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.retention_budget_gb_spin = QSpinBox()
        self.retention_budget_gb_spin.setRange(1, 1000)
        self.retention_budget_gb_spin.setValue(50)

        self.analysis_support_spin = QDoubleSpinBox()
        self.analysis_support_spin.setDecimals(2)
        self.analysis_support_spin.setRange(0.05, 1.0)
        self.analysis_support_spin.setSingleStep(0.05)
        self.analysis_support_spin.setValue(0.35)
        self.analysis_quality_cutoff_spin = QDoubleSpinBox()
        self.analysis_quality_cutoff_spin.setDecimals(2)
        self.analysis_quality_cutoff_spin.setRange(0.0, 1.0)
        self.analysis_quality_cutoff_spin.setSingleStep(0.05)
        self.analysis_quality_cutoff_spin.setValue(0.45)
        self.analysis_low_confidence_spin = QDoubleSpinBox()
        self.analysis_low_confidence_spin.setDecimals(2)
        self.analysis_low_confidence_spin.setRange(0.0, 1.0)
        self.analysis_low_confidence_spin.setSingleStep(0.05)
        self.analysis_low_confidence_spin.setValue(0.35)
        self.analysis_auto_band_count_spin = QSpinBox()
        self.analysis_auto_band_count_spin.setRange(1, 5)
        self.analysis_auto_band_count_spin.setValue(5)
        self.analysis_export_advanced_checkbox = QCheckBox("Export advanced/internal analysis files")
        self.analysis_export_advanced_checkbox.setChecked(True)
        self._manual_band_controls: list[tuple[QCheckBox, QDoubleSpinBox, QDoubleSpinBox]] = []
        self._analysis_manual_band_container = QWidget(self)
        # This container exists only to keep the manual-band controls alive
        # between popout dialog sessions. It must stay hidden on the main shell.
        self._analysis_manual_band_container.hide()
        manual_band_layout = QVBoxLayout(self._analysis_manual_band_container)
        manual_band_layout.setContentsMargins(0, 0, 0, 0)
        manual_band_layout.setSpacing(4)
        for band_index in range(5):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            enabled_check = QCheckBox(f"Band {band_index + 1}")
            enabled_check.setChecked(band_index == 0)
            low_spin = QDoubleSpinBox()
            low_spin.setDecimals(3)
            low_spin.setRange(0.001, 1000.0)
            low_spin.setValue(5.0 + band_index)
            low_spin.setSuffix(" Hz")
            high_spin = QDoubleSpinBox()
            high_spin.setDecimals(3)
            high_spin.setRange(0.001, 1000.0)
            high_spin.setValue(12.0 + band_index)
            high_spin.setSuffix(" Hz")
            row_layout.addWidget(enabled_check)
            row_layout.addWidget(QLabel("Low"))
            row_layout.addWidget(low_spin)
            row_layout.addWidget(QLabel("High"))
            row_layout.addWidget(high_spin)
            manual_band_layout.addWidget(row)
            self._manual_band_controls.append((enabled_check, low_spin, high_spin))
        self.dry_run_button = QPushButton("Dry-run Validation")
        advanced_layout.addRow("Retention budget (GB)", self.retention_budget_gb_spin)
        advanced_layout.addRow("Validation", self.dry_run_button)

        diagnostics_group = QGroupBox("Diagnostics")
        diagnostics_layout = QFormLayout(diagnostics_group)
        diagnostics_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.diagnostic_level_combo = QComboBox()
        self.diagnostic_level_combo.addItems(["Off", "Basic", "Detailed", "Trace"])
        self.diagnostics_cap_mb_spin = QSpinBox()
        self.diagnostics_cap_mb_spin.setRange(64, 4096)
        self.diagnostics_cap_mb_spin.setValue(1024)
        diagnostics_layout.addRow("Diagnostics level", self.diagnostic_level_combo)
        diagnostics_layout.addRow("Diagnostics cap (MB)", self.diagnostics_cap_mb_spin)

        status_group = QGroupBox("Status")
        status_layout = QFormLayout(status_group)
        status_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        self.state_label = QLabel("-")
        self.fingerprint_label = QLabel("No source loaded.")
        self.fingerprint_progress_label = QLabel("")
        self.drift_gate_label = QLabel("No drift review required yet.")
        self.drift_gate_label.setWordWrap(True)
        self.render_stage_label = QLabel("Idle")
        self.render_progress_label = QLabel("No active render.")
        self.elapsed_label = QLabel("-")
        self.eta_label = QLabel("-")
        self.eta_label.setToolTip(
            "ETA is estimated in the GUI from the current stage's recent frame rate."
        )
        self.mean_frame_label = QLabel("-")
        self.terminal_status_label = QLabel("No run yet.")
        self.terminal_status_label.setWordWrap(True)
        status_layout.addRow("Workflow state", self.state_label)
        status_layout.addRow("Fingerprint", self.fingerprint_label)
        status_layout.addRow("Fingerprint progress", self.fingerprint_progress_label)
        status_layout.addRow("Drift gate", self.drift_gate_label)
        status_layout.addRow("Current stage", self.render_stage_label)
        status_layout.addRow("Render progress", self.render_progress_label)
        status_layout.addRow("Elapsed", self.elapsed_label)
        status_layout.addRow("ETA", self.eta_label)
        status_layout.addRow("Mean time / frame", self.mean_frame_label)
        status_layout.addRow("Latest outcome", self.terminal_status_label)

        report_group = QGroupBox("Pre-flight Report")
        report_layout = QVBoxLayout(report_group)
        self.preflight_report_view = QPlainTextEdit()
        self.preflight_report_view.setReadOnly(True)
        self.preflight_report_view.setMinimumHeight(160)
        report_layout.addWidget(self.preflight_report_view)

        actions = QWidget()
        actions.setObjectName("actionStrip")
        actions.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self.start_render_button = QPushButton("Start Render")
        self.start_render_button.setEnabled(False)
        self.cancel_render_button = QPushButton("Cancel Render")
        self.cancel_render_button.setEnabled(False)
        self.prepare_new_run_button = QPushButton("Prepare New Run")
        self.prepare_new_run_button.setEnabled(False)
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setEnabled(False)
        self.open_sidecar_button = QPushButton("Open Sidecar")
        self.open_sidecar_button.setEnabled(False)
        self.open_diagnostics_button = QPushButton("Open Diagnostics")
        self.open_diagnostics_button.setEnabled(False)
        self.open_temp_button = QPushButton("Open Temp Folder")
        self.open_temp_button.setEnabled(False)
        self.purge_temp_button = QPushButton("Purge Temp Files for this Job")
        self.purge_temp_button.setEnabled(False)
        self.purge_failed_runs_button = QPushButton("Purge Failed-run Temp Files")
        self.purge_failed_runs_button.setEnabled(False)
        actions_layout.addWidget(self.start_render_button)
        actions_layout.addWidget(self.cancel_render_button)
        actions_layout.addWidget(self.prepare_new_run_button)
        actions_layout.addWidget(self.open_output_button)
        actions_layout.addWidget(self.open_sidecar_button)
        actions_layout.addWidget(self.open_diagnostics_button)
        actions_layout.addWidget(self.open_temp_button)
        actions_layout.addWidget(self.purge_temp_button)
        actions_layout.addWidget(self.purge_failed_runs_button)
        actions_layout.addStretch(1)
        self.heartbeat_text_label = QLabel("Heartbeat")
        self.heartbeat_text_label.setObjectName("heartbeatText")
        self.heartbeat_dot_label = QLabel("●")
        self.heartbeat_dot_label.setObjectName("heartbeatDot")
        actions_layout.addWidget(self.heartbeat_text_label)
        actions_layout.addWidget(self.heartbeat_dot_label)

        self.event_log = QPlainTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setMaximumHeight(120)
        self.event_log.setMinimumHeight(76)

        scroll_layout.addWidget(source_group)
        scroll_layout.addWidget(settings_group)
        scroll_layout.addWidget(analysis_group)
        scroll_layout.addWidget(advanced_group)
        scroll_layout.addWidget(diagnostics_group)
        scroll_layout.addWidget(status_group)
        scroll_layout.addWidget(report_group)
        scroll_layout.addStretch(1)

        root.addWidget(self.content_scroll_area, stretch=1)
        root.addWidget(actions)
        root.addWidget(self.event_log)
        self.setCentralWidget(central)

    def _wire_signals(self) -> None:
        self.source_browse_button.clicked.connect(self._choose_source)
        self.output_browse_button.clicked.connect(self._choose_output_folder)
        self.metadata_load_button.clicked.connect(self._load_from_sidecar)
        self.processing_resolution_combo.currentIndexChanged.connect(
            self._rebuild_output_resolution_options
        )
        self.processing_resolution_combo.currentIndexChanged.connect(self._update_settings_state)
        self.magnification_spin.valueChanged.connect(self._update_settings_state)
        self.low_hz_spin.valueChanged.connect(self._mark_band_edited)
        self.high_hz_spin.valueChanged.connect(self._mark_band_edited)
        self.low_hz_spin.valueChanged.connect(self._update_settings_state)
        self.high_hz_spin.valueChanged.connect(self._update_settings_state)
        self.hardware_acceleration_checkbox.toggled.connect(self._update_settings_state)
        self.hardware_acceleration_checkbox.toggled.connect(
            self._update_performance_status
        )
        self.analysis_enabled_checkbox.toggled.connect(self._update_settings_state)
        self.analysis_enabled_checkbox.toggled.connect(self._update_analysis_controls)
        self.analysis_band_mode_combo.currentIndexChanged.connect(self._update_settings_state)
        self.analysis_band_mode_combo.currentIndexChanged.connect(self._update_analysis_controls)
        self.analysis_roi_button.clicked.connect(self._open_analysis_roi_editor)
        self.analysis_advanced_button.clicked.connect(self._open_analysis_advanced_dialog)
        self.analysis_support_spin.valueChanged.connect(self._update_settings_state)
        self.analysis_quality_cutoff_spin.valueChanged.connect(self._update_settings_state)
        self.analysis_low_confidence_spin.valueChanged.connect(self._update_settings_state)
        self.analysis_auto_band_count_spin.valueChanged.connect(self._update_settings_state)
        self.analysis_export_advanced_checkbox.toggled.connect(self._update_settings_state)
        for enabled_check, low_spin, high_spin in self._manual_band_controls:
            enabled_check.toggled.connect(self._update_settings_state)
            enabled_check.toggled.connect(self._update_analysis_controls)
            low_spin.valueChanged.connect(self._update_settings_state)
            high_spin.valueChanged.connect(self._update_settings_state)
        self.resource_policy_combo.currentIndexChanged.connect(self._update_settings_state)
        self.apply_suggested_band_button.clicked.connect(self._apply_suggested_band)
        self.exclusion_editor_button.clicked.connect(self._open_drift_editor)
        self.dry_run_button.clicked.connect(self._run_dry_run_validation)
        self.start_render_button.clicked.connect(self._start_render)
        self.cancel_render_button.clicked.connect(self._cancel_render)
        self.prepare_new_run_button.clicked.connect(self._prepare_new_run)
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.open_sidecar_button.clicked.connect(self._open_sidecar)
        self.open_diagnostics_button.clicked.connect(self._open_diagnostics_folder)
        self.open_temp_button.clicked.connect(self._open_temp_folder)
        self.purge_temp_button.clicked.connect(self._purge_job_temp_files)
        self.purge_failed_runs_button.clicked.connect(self._purge_failed_run_temp_files)

    def _append_log(self, message: str) -> None:
        self.event_log.appendPlainText(message)

    def _current_snapshot(self) -> SourceSnapshot | None:
        if self._current_source_path is None or not self._current_source_path.exists():
            return None
        stat = self._current_source_path.stat()
        return SourceSnapshot(
            path=str(self._current_source_path),
            size_bytes=stat.st_size,
            modified_ns=stat.st_mtime_ns,
        )

    def _invalidate_missing_source(self) -> None:
        """Drop authoritative source state when the selected file disappears so the shell cannot stay falsely Ready."""

        already_invalidated = (
            self._last_known_snapshot is None
            and self._current_fingerprint is None
            and self._source_probe_info is None
        )
        self._last_known_snapshot = None
        self._current_fingerprint = None
        self._source_probe_info = None
        self._first_frame_image = None
        self._last_frame_image = None
        self._suggested_frequency_band = None
        self._band_user_edited = False
        self._drift_assessment = DriftAssessment()
        self.source_metadata_label.setText("Selected source file is missing.")
        self.suggested_band_label.setText(
            "Suggested band unavailable until the source file is available again."
        )
        self.apply_suggested_band_button.setEnabled(False)
        self._set_first_frame_preview(None, "First frame preview unavailable.")
        self.preflight_report_view.setPlainText("")
        self._controller.mark_fingerprint_failed()
        self._update_settings_state()
        if not already_invalidated:
            self._append_log(
                "Selected source file is no longer available. Readiness was cleared until the source is reloaded or becomes available again."
            )

    def _choose_source(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Choose source video",
            str(
                self._current_source_path.parent
                if self._current_source_path is not None
                else self._default_input_directory
            ),
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)",
        )
        if not path_text:
            return

        self._current_source_path = Path(path_text)
        self.source_path_edit.setText(path_text)
        if not self.output_folder_edit.text().strip():
            self.output_folder_edit.setText(str(self._default_output_directory))
        self._current_fingerprint = None
        self._source_probe_info = None
        self._first_frame_image = None
        self._last_frame_image = None
        self._suggested_frequency_band = None
        self._band_user_edited = False
        self._exclusion_zones = ()
        self._mask_feather_px = 4.0
        self._analysis_roi = None
        self._drift_assessment = DriftAssessment()
        self._last_terminal_snapshot = None
        self.source_metadata_label.setText("Probe pending...")
        self.suggested_band_label.setText("Suggested band: analyzing source motion...")
        self.apply_suggested_band_button.setEnabled(False)
        self._set_first_frame_preview(None, "First frame preview pending...")
        self.preflight_report_view.setPlainText("")
        snapshot = self._current_snapshot()
        if snapshot is None:
            return
        self._last_known_snapshot = snapshot
        self._controller.load_source(snapshot)
        self._append_log(f"Loaded source: {path_text}")
        self._start_source_probe()
        self._start_fingerprint()
        self._start_frame_extraction()
        self._update_settings_state()

    def _choose_output_folder(self) -> None:
        current_output = self.output_folder_edit.text().strip()
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose output folder",
            current_output or str(self._default_output_directory),
        )
        if not selected:
            return
        self.output_folder_edit.setText(selected)

    def _start_frame_extraction(self) -> None:
        if self._current_source_path is None:
            return
        if self._frame_pair_worker is not None and self._frame_pair_worker.isRunning():
            self._frame_pair_worker.quit()
            self._frame_pair_worker.wait(100)

        worker = FramePairWorker(self._current_source_path, self)
        worker.completed.connect(self._on_frame_pair_complete)
        worker.failed.connect(self._on_frame_pair_failed)
        self._frame_pair_worker = worker
        worker.start()

    def _start_source_probe(self) -> None:
        if self._current_source_path is None:
            return
        if self._probe_worker is not None and self._probe_worker.isRunning():
            self._probe_worker.quit()
            self._probe_worker.wait(100)

        worker = SourceProbeWorker(self._current_source_path, self)
        worker.completed.connect(self._on_source_probe_complete)
        worker.failed.connect(self._on_source_probe_failed)
        self._probe_worker = worker
        worker.start()

    def _start_baseline_band_analysis(self, probe_info: FfprobeMediaInfo) -> None:
        if self._current_source_path is None:
            return
        if self._baseline_band_worker is not None and self._baseline_band_worker.isRunning():
            self._baseline_band_worker.quit()
            self._baseline_band_worker.wait(100)

        self.suggested_band_label.setText("Suggested band: analyzing source motion...")
        self.apply_suggested_band_button.setEnabled(False)
        worker = BaselineBandWorker(self._current_source_path, probe_info, self)
        worker.completed.connect(self._on_baseline_band_complete)
        worker.failed.connect(self._on_baseline_band_failed)
        self._baseline_band_worker = worker
        worker.start()

    def _start_fingerprint(self) -> None:
        if self._current_source_path is None:
            return
        if self._fingerprint_worker is not None and self._fingerprint_worker.isRunning():
            self._fingerprint_worker.quit()
            self._fingerprint_worker.wait(100)

        self.fingerprint_label.setText("Fingerprint pending")
        self.fingerprint_progress_label.setText("0%")
        worker = FingerprintWorker(self._current_source_path, self)
        worker.progress.connect(self._on_fingerprint_progress)
        worker.completed.connect(self._on_fingerprint_complete)
        worker.failed.connect(self._on_fingerprint_failed)
        self._fingerprint_worker = worker
        worker.start()

    def _on_source_probe_complete(self, info: object) -> None:
        self._source_probe_info = info
        normalization_plan = build_source_normalization_plan(info)
        metadata_parts = [
            f"{info.width} x {info.height}",
            f"{info.fps:.3f} fps",
            "CFR" if info.is_cfr else "VFR",
            f"{info.bit_depth}-bit",
        ]
        if (
            isinstance(info.rotation_degrees, (int, float))
            and abs(info.rotation_degrees) % 360.0 > 1e-6
        ):
            metadata_parts.append(f"rotation {info.rotation_degrees:.0f}°")
        if abs(info.sample_aspect_ratio - 1.0) > 1e-6:
            metadata_parts.append(f"SAR {info.sample_aspect_ratio:.3f}")
        if normalization_plan.working_resolution != normalization_plan.native_resolution:
            metadata_parts.append(
                "working "
                f"{normalization_plan.working_resolution.width} x "
                f"{normalization_plan.working_resolution.height}"
            )
        if normalization_plan.requires_cfr_normalization:
            metadata_parts.append(f"auto CFR {normalization_plan.working_fps:.3f} fps")
        self.source_metadata_label.setText(", ".join(metadata_parts))
        self._rebuild_processing_resolution_options(normalization_plan.working_resolution)
        self._append_log("Fast probe metadata loaded.")
        for message in normalization_plan.normalization_messages:
            self._append_log(message)
        self._start_baseline_band_analysis(info)
        self._update_settings_state()

    def _on_source_probe_failed(self, message: str) -> None:
        self._source_probe_info = None
        self.source_metadata_label.setText("Probe failed.")
        self.suggested_band_label.setText("Suggested band unavailable because probe failed.")
        self.apply_suggested_band_button.setEnabled(False)
        self._append_log(f"Source probe failed: {message}")
        self._update_settings_state()

    def _on_frame_pair_complete(
        self,
        first_frame: object,
        last_frame: object,
        drift_estimate: object = None,
    ) -> None:
        self._first_frame_image = self._rgb_frame_to_qimage(first_frame)
        self._last_frame_image = self._rgb_frame_to_qimage(last_frame)
        estimated_drift_px = None
        if drift_estimate is not None:
            estimated_drift_px = float(drift_estimate.magnitude_px)
        self._drift_assessment = DriftAssessment(
            visible_drift_confirmed=self._drift_assessment.visible_drift_confirmed,
            estimated_global_drift_px=estimated_drift_px,
            advisory_threshold_px=self._drift_assessment.advisory_threshold_px,
            acknowledged=self._drift_assessment.acknowledged,
        )
        self._set_first_frame_preview(self._first_frame_image)
        self._append_log("First/last frame context loaded for drift review.")
        if estimated_drift_px is None:
            self._append_log("Estimated global drift was unavailable for the reviewed frame pair.")
        else:
            self._append_log(f"Estimated global drift: {estimated_drift_px:.2f}px.")
        self._update_settings_state()

    def _on_frame_pair_failed(self, message: str) -> None:
        self._first_frame_image = None
        self._last_frame_image = None
        self._drift_assessment = DriftAssessment(
            visible_drift_confirmed=self._drift_assessment.visible_drift_confirmed,
            estimated_global_drift_px=None,
            advisory_threshold_px=self._drift_assessment.advisory_threshold_px,
            acknowledged=self._drift_assessment.acknowledged,
        )
        self._set_first_frame_preview(None, "First frame preview unavailable.")
        self._append_log(f"Frame extraction failed: {message}")
        self._update_settings_state()

    def _on_baseline_band_complete(
        self,
        source_path: str,
        suggestion: object,
    ) -> None:
        if self._current_source_path is None or str(self._current_source_path) != source_path:
            return
        self._suggested_frequency_band = suggestion
        if not isinstance(suggestion, FrequencyBandSuggestion):
            self.suggested_band_label.setText("Suggested band unavailable.")
            self.apply_suggested_band_button.setEnabled(False)
            return
        self.suggested_band_label.setText(
            "Suggested band: "
            f"{suggestion.low_hz:.2f} Hz to {suggestion.high_hz:.2f} Hz "
            f"(peak {suggestion.peak_hz:.2f} Hz)"
        )
        self.apply_suggested_band_button.setEnabled(True)
        self._append_log(
            "Baseline source analysis suggested "
            f"{suggestion.low_hz:.2f} Hz to {suggestion.high_hz:.2f} Hz."
        )
        if not self._band_user_edited:
            self._apply_suggested_band(log_action=False)
            self._append_log("Applied the suggested baseline frequency band to the current settings.")

    def _on_baseline_band_failed(self, source_path: str, message: str) -> None:
        if self._current_source_path is None or str(self._current_source_path) != source_path:
            return
        self._suggested_frequency_band = None
        self.suggested_band_label.setText("Suggested band unavailable.")
        self.apply_suggested_band_button.setEnabled(False)
        self._append_log(f"Baseline band analysis failed: {message}")

    def _mark_band_edited(self) -> None:
        if not self._applying_suggested_band and self._current_source_path is not None:
            self._band_user_edited = True

    def _apply_suggested_band(self, *, log_action: bool = True) -> None:
        suggestion = self._suggested_frequency_band
        if not isinstance(suggestion, FrequencyBandSuggestion):
            return
        self._applying_suggested_band = True
        try:
            self.low_hz_spin.setValue(suggestion.low_hz)
            self.high_hz_spin.setValue(suggestion.high_hz)
        finally:
            self._applying_suggested_band = False
        self._band_user_edited = False
        if log_action:
            self._append_log(
                "Applied the suggested baseline frequency band: "
                f"{suggestion.low_hz:.2f} Hz to {suggestion.high_hz:.2f} Hz."
            )

    def _set_first_frame_preview(
        self,
        image: QImage | None,
        placeholder: str = "First frame preview pending.",
    ) -> None:
        self.first_frame_preview_label.set_preview_image(image, placeholder)

    def _on_fingerprint_progress(self, percent: int) -> None:
        self.fingerprint_progress_label.setText(f"{percent}%")

    def _on_fingerprint_complete(self, digest: str, _: int) -> None:
        self._current_fingerprint = digest
        snapshot = self._current_snapshot()
        if snapshot is not None:
            self._last_known_snapshot = snapshot
            self._controller.mark_fingerprint_complete(snapshot)
        self.fingerprint_label.setText(f"Ready ({digest[:12]}...)")
        self.fingerprint_progress_label.setText("100%")
        self._append_log("Canonical SHA-256 fingerprint completed.")
        self._refresh_state()

    def _on_fingerprint_failed(self, message: str) -> None:
        self._controller.mark_fingerprint_failed()
        self.fingerprint_label.setText("Fingerprint failed")
        self.fingerprint_progress_label.setText("")
        self._append_log(f"Fingerprint failed: {message}")
        self._refresh_state()

    def _rebuild_output_resolution_options(self) -> None:
        processing = self.processing_resolution_combo.currentData()
        self.output_resolution_combo.blockSignals(True)
        self.output_resolution_combo.clear()
        if processing is not None:
            mirrored_resolution = _codec_safe_output_resolution(
                processing,
                processing_resolution=processing,
            )
            if mirrored_resolution is not None:
                label = f"{mirrored_resolution.width} x {mirrored_resolution.height}"
                self.output_resolution_combo.addItem(label, mirrored_resolution)
                self.output_resolution_combo.setCurrentIndex(0)
        self.output_resolution_combo.blockSignals(False)

    def _rebuild_processing_resolution_options(
        self,
        source_resolution: Resolution | None = None,
    ) -> None:
        previous_processing = self.processing_resolution_combo.currentData()
        self._resolution_options = (
            self._resolution_presets
            if source_resolution is None
            else _resolution_options_for_source(source_resolution, self._resolution_presets)
        )
        self.processing_resolution_combo.blockSignals(True)
        self.processing_resolution_combo.clear()
        seen: set[tuple[int, int]] = set()
        for resolution in self._resolution_options:
            safe_resolution = _codec_safe_output_resolution(
                resolution,
                processing_resolution=resolution,
            )
            if safe_resolution is None:
                continue
            key = (safe_resolution.width, safe_resolution.height)
            if key in seen:
                continue
            seen.add(key)
            label = f"{safe_resolution.width} x {safe_resolution.height}"
            self.processing_resolution_combo.addItem(label, safe_resolution)
        if previous_processing is not None:
            previous_index = self._find_resolution_index(
                self.processing_resolution_combo,
                previous_processing,
            )
            if previous_index >= 0:
                self.processing_resolution_combo.setCurrentIndex(previous_index)
            elif self.processing_resolution_combo.count() > 0:
                fallback_index = 1 if self.processing_resolution_combo.count() > 1 else 0
                self.processing_resolution_combo.setCurrentIndex(fallback_index)
        elif self.processing_resolution_combo.count() > 0:
            fallback_index = 1 if self.processing_resolution_combo.count() > 1 else 0
            self.processing_resolution_combo.setCurrentIndex(fallback_index)
        self.processing_resolution_combo.blockSignals(False)
        self._rebuild_output_resolution_options()

    def _source_normalization_plan(self) -> SourceNormalizationPlan | None:
        if self._source_probe_info is None:
            return None
        return build_source_normalization_plan(self._source_probe_info)

    def _current_source_resolution(self) -> Resolution | None:
        plan = self._source_normalization_plan()
        if plan is not None:
            return plan.working_resolution
        return None

    def _build_manual_analysis_bands(self) -> tuple[AnalysisBand, ...]:
        manual_bands: list[AnalysisBand] = []
        band_mode = self.analysis_band_mode_combo.currentData()
        for band_index, (enabled_check, low_spin, high_spin) in enumerate(
            self._manual_band_controls,
            start=1,
        ):
            if not enabled_check.isChecked():
                continue
            if band_mode is AnalysisBandMode.MANUAL_SINGLE and band_index > 1:
                continue
            manual_bands.append(
                AnalysisBand(
                    band_id=f"band{band_index:02d}",
                    low_hz=float(low_spin.value()),
                    high_hz=float(high_spin.value()),
                )
            )
        return tuple(manual_bands)

    def _current_manual_band_values(self) -> tuple[tuple[bool, float, float], ...]:
        return tuple(
            (
                enabled_check.isChecked(),
                float(low_spin.value()),
                float(high_spin.value()),
            )
            for enabled_check, low_spin, high_spin in self._manual_band_controls
        )

    def _build_analysis_settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            enabled=self.analysis_enabled_checkbox.isChecked(),
            roi=self._analysis_roi,
            minimum_cell_support_fraction=float(self.analysis_support_spin.value()),
            roi_quality_cutoff=float(self.analysis_quality_cutoff_spin.value()),
            low_confidence_threshold=float(self.analysis_low_confidence_spin.value()),
            auto_band_count=int(self.analysis_auto_band_count_spin.value()),
            band_mode=self.analysis_band_mode_combo.currentData(),
            manual_bands=self._build_manual_analysis_bands(),
            export_advanced_files=self.analysis_export_advanced_checkbox.isChecked(),
        )

    def _build_intent(self) -> JobIntent:
        processing_resolution = self.processing_resolution_combo.currentData()
        return JobIntent(
            phase=PhaseSettings(
                magnification=float(self.magnification_spin.value()),
                low_hz=float(self.low_hz_spin.value()),
                high_hz=float(self.high_hz_spin.value()),
                pyramid_type="complex_steerable",
                sigma=1.0,
                attenuate_other_frequencies=True,
            ),
            processing_resolution=processing_resolution,
            output_resolution=processing_resolution,
            resource_policy=self.resource_policy_combo.currentData(),
            exclusion_zones=self._exclusion_zones,
            mask_feather_px=self._mask_feather_px,
            analysis=self._build_analysis_settings(),
            hardware_acceleration_enabled=self.hardware_acceleration_checkbox.isChecked(),
        )

    def _find_resolution_index(
        self, combo: QComboBox, resolution: Resolution
    ) -> int:
        for index in range(combo.count()):
            candidate = combo.itemData(index)
            if (
                isinstance(candidate, Resolution)
                and candidate.width == resolution.width
                and candidate.height == resolution.height
            ):
                return index
        return -1

    def _analysis_settings_valid(self, intent: JobIntent) -> bool:
        if not intent.analysis.enabled:
            return True
        if not (0.0 <= intent.analysis.roi_quality_cutoff <= 1.0):
            return False
        if not (0.0 <= intent.analysis.low_confidence_threshold <= 1.0):
            return False
        if not (0.0 < intent.analysis.minimum_cell_support_fraction <= 1.0):
            return False
        if not (1 <= intent.analysis.auto_band_count <= 5):
            return False
        if intent.analysis.band_mode is AnalysisBandMode.AUTO:
            return True
        if not intent.analysis.manual_bands:
            return False
        if (
            intent.analysis.band_mode is AnalysisBandMode.MANUAL_SINGLE
            and len(intent.analysis.manual_bands) != 1
        ):
            return False
        if (
            intent.analysis.band_mode is AnalysisBandMode.MANUAL_MULTI
            and len(intent.analysis.manual_bands) > 5
        ):
            return False
        for band in intent.analysis.manual_bands:
            if band.high_hz <= band.low_hz:
                return False
            if band.low_hz < intent.phase.low_hz or band.high_hz > intent.phase.high_hz:
                return False
        return True

    def _analysis_roi_summary(self) -> str:
        if self._analysis_roi is None:
            return summarize_automatic_analysis_roi(self._exclusion_zones)
        if self._analysis_roi.shape.name == "RECTANGLE":
            return (
                f"Manual rectangle ROI at ({self._analysis_roi.x:.1f}, {self._analysis_roi.y:.1f}) "
                f"with {(self._analysis_roi.width or 0.0):.1f} x {(self._analysis_roi.height or 0.0):.1f} px."
            )
        return (
            f"Manual circle ROI centered at ({self._analysis_roi.x:.1f}, {self._analysis_roi.y:.1f}) "
            f"with radius {(self._analysis_roi.radius or 0.0):.1f} px."
        )

    def _update_analysis_controls(self) -> None:
        analysis_enabled = self.analysis_enabled_checkbox.isChecked()
        band_mode = self.analysis_band_mode_combo.currentData()
        render_active = self._render_supervisor is not None and not self._last_terminal_snapshot
        self.analysis_roi_button.setEnabled(
            analysis_enabled and self._source_probe_info is not None and not render_active
        )
        self.analysis_advanced_button.setEnabled(not render_active)
        self.analysis_roi_label.setText(self._analysis_roi_summary())
        self.analysis_band_summary_label.setText(self._analysis_band_summary())
        self.analysis_advanced_summary_label.setText(self._analysis_advanced_summary())
        for index, (enabled_check, low_spin, high_spin) in enumerate(self._manual_band_controls):
            band_enabled = analysis_enabled and (
                band_mode is AnalysisBandMode.MANUAL_MULTI
                or (
                    band_mode is AnalysisBandMode.MANUAL_SINGLE
                    and index == 0
                )
            )
            enabled_check.setEnabled(band_enabled)
            low_spin.setEnabled(band_enabled and enabled_check.isChecked())
            high_spin.setEnabled(band_enabled and enabled_check.isChecked())

    def _analysis_band_summary(self) -> str:
        band_mode = self.analysis_band_mode_combo.currentData()
        manual_bands = self._build_manual_analysis_bands()
        if band_mode is AnalysisBandMode.AUTO:
            return (
                f"Auto-bands will use the ROI spectrum and generate up to "
                f"{self.analysis_auto_band_count_spin.value()} bands."
            )
        if not manual_bands:
            return "Manual mode requires at least one band inside the selected render range."
        return "Manual bands: " + ", ".join(
            f"{band.low_hz:.2f}-{band.high_hz:.2f} Hz" for band in manual_bands
        )

    def _analysis_advanced_summary(self) -> str:
        band_mode = self.analysis_band_mode_combo.currentData()
        if band_mode is AnalysisBandMode.AUTO:
            band_text = f"Auto-bands up to {self.analysis_auto_band_count_spin.value()}"
        else:
            manual_bands = self._build_manual_analysis_bands()
            band_text = (
                f"{len(manual_bands)} manual band"
                f"{'' if len(manual_bands) == 1 else 's'}"
            )
        export_text = (
            "advanced export on"
            if self.analysis_export_advanced_checkbox.isChecked()
            else "advanced export off"
        )
        return (
            f"{band_text}, support {self.analysis_support_spin.value():.2f}, "
            f"ROI quality {self.analysis_quality_cutoff_spin.value():.2f}, "
            f"low-confidence {self.analysis_low_confidence_spin.value():.2f}, "
            f"{export_text}."
        )

    def _update_performance_status(self) -> None:
        capability = detect_acceleration_capability()
        decision = resolve_acceleration_request(
            self.hardware_acceleration_checkbox.isChecked(),
            capability=capability,
        )
        backend_name = capability.backend_name or decision.backend_name or "cupy"
        backend_label = "CuPy" if backend_name.lower() == "cupy" else backend_name
        device_name = decision.device_name or capability.device_name
        backend_text = (
            f"{backend_label} on {device_name}" if device_name else backend_label
        )
        if decision.active:
            status_text = f"Active: {backend_text}. {decision.detail}"
        elif decision.requested:
            status_text = f"Unavailable: {backend_text}. {decision.detail}"
        elif capability.usable:
            status_text = (
                f"Available: {backend_text}. Enable the checkbox to use hardware acceleration."
            )
        else:
            status_text = f"Unavailable: {backend_text}. {capability.detail}"
        self.hardware_acceleration_status_label.setText(status_text)

    def _update_settings_state(self) -> None:
        processing_resolution = self.processing_resolution_combo.currentData()
        output_resolution = self.output_resolution_combo.currentData()
        if processing_resolution is None or output_resolution is None:
            self._controller.set_settings_complete(False)
            self._refresh_state()
            return

        intent = self._build_intent()
        probe_ready = self._source_probe_info is not None
        drift_gate_satisfied = self._drift_assessment.can_render
        zones_valid = self._zones_valid_for_current_source()
        settings_complete = (
            probe_ready
            and zones_valid
            and drift_gate_satisfied
            and intent.mask_feather_px > 0
            and intent.phase.high_hz > intent.phase.low_hz
            and self._analysis_settings_valid(intent)
            and intent.output_resolution == intent.processing_resolution
        )
        self._controller.set_settings_complete(settings_complete)
        self._update_analysis_controls()
        self._update_performance_status()
        self._refresh_state()

    def _zones_valid_for_current_source(self) -> bool:
        source_resolution = self._current_source_resolution()
        if source_resolution is None:
            return not self._exclusion_zones
        issues = validate_exclusion_zones(
            self._exclusion_zones,
            source_resolution,
        )
        return not issues

    def _refresh_state(self) -> None:
        state = self._controller.state
        render_active = self._render_supervisor is not None and not self._last_terminal_snapshot
        terminal_state = state in {UiState.COMPLETE, UiState.FAILED, UiState.CANCELLED}

        self.state_label.setText(state.value.replace("_", " ").title())
        self._refresh_heartbeat_indicator()
        self.start_render_button.setEnabled(state is UiState.READY and not render_active)
        self.cancel_render_button.setEnabled(render_active)
        self.prepare_new_run_button.setEnabled(terminal_state)
        self.exclusion_editor_button.setEnabled(
            self._source_probe_info is not None and not render_active
        )
        self.analysis_enabled_checkbox.setEnabled(not render_active)
        self.analysis_band_mode_combo.setEnabled(not render_active)
        self.analysis_advanced_button.setEnabled(not render_active)
        self.diagnostic_level_combo.setEnabled(not render_active)
        self.diagnostics_cap_mb_spin.setEnabled(not render_active)
        self.retention_budget_gb_spin.setEnabled(not render_active)
        self.source_browse_button.setEnabled(not render_active)
        self.metadata_load_button.setEnabled(not render_active)
        self.dry_run_button.setEnabled(
            self._source_probe_info is not None
            and self._current_source_path is not None
            and not render_active
        )
        self.exclusion_editor_button.setText(
            f"Drift Check / Mask Zones ({len(self._exclusion_zones)})"
        )
        self._update_analysis_controls()

        if self._source_probe_info is None:
            self.drift_gate_label.setText("Waiting for fast probe metadata.")
        elif not self._zones_valid_for_current_source():
            self.drift_gate_label.setText(
                "Current mask geometry does not fit the loaded source frame."
            )
        elif self._drift_assessment.warning_active and not self._drift_assessment.acknowledged:
            self.drift_gate_label.setText(
                "Drift warning active. Render stays blocked until the reviewed source state is acknowledged."
            )
        elif self._drift_assessment.warning_active:
            self.drift_gate_label.setText(
                "Drift warning acknowledged for the current reviewed source state."
            )
        else:
            self.drift_gate_label.setText("No active drift warning.")

        request = self._active_render_request
        self.open_output_button.setEnabled(
            request is not None and request.paths.output_directory.exists()
        )
        self.open_sidecar_button.setEnabled(
            request is not None and request.paths.final_sidecar_path.exists()
        )
        self.open_diagnostics_button.setEnabled(
            request is not None and request.paths.diagnostics_directory.exists()
        )
        self.open_temp_button.setEnabled(
            request is not None and request.paths.scratch_directory.exists()
        )
        self.purge_temp_button.setEnabled(
            request is not None
            and not render_active
            and (
                request.paths.scratch_directory.exists()
                or request.paths.diagnostics_directory.exists()
            )
        )
        self.purge_failed_runs_button.setEnabled(
            not render_active
            and (
                Path(self._preferences.temp_root).exists()
                or Path(self._preferences.diagnostics_root).exists()
            )
        )

    def _poll_source_staleness(self) -> None:
        if self._render_supervisor is not None:
            return
        if self._current_source_path is None:
            return
        if not self._current_source_path.exists():
            self._invalidate_missing_source()
            return
        current_snapshot = self._current_snapshot()
        if current_snapshot is None:
            return
        if self._last_known_snapshot is None:
            self._last_known_snapshot = current_snapshot
            self._controller.load_source(current_snapshot)
            self._append_log(
                "Source file became available again. Probe, fingerprint, and drift review were restarted for the current path."
            )
            self._start_source_probe()
            self._start_fingerprint()
            self._start_frame_extraction()
            self._update_settings_state()
            return
        if detect_stale_source(self._last_known_snapshot, current_snapshot):
            self._last_known_snapshot = current_snapshot
            self._current_fingerprint = None
            self._source_probe_info = None
            self._first_frame_image = None
            self._last_frame_image = None
            self._suggested_frequency_band = None
            self._band_user_edited = False
            self._drift_assessment = DriftAssessment()
            self.source_metadata_label.setText("Probe pending...")
            self.suggested_band_label.setText("Suggested band: analyzing source motion...")
            self.apply_suggested_band_button.setEnabled(False)
            self._set_first_frame_preview(None, "First frame preview pending...")
            self.preflight_report_view.setPlainText("")
            self._controller.load_source(current_snapshot)
            self._append_log(
                "Source file size or modified time changed. Probe, fingerprint, and drift review were restarted for the updated source."
            )
            self._start_source_probe()
            self._start_fingerprint()
            self._start_frame_extraction()
            self._update_settings_state()
            return

        self._controller.mark_source_changed(current_snapshot)
        self._refresh_state()

    def _load_from_sidecar(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Choose export metadata",
            self.output_folder_edit.text().strip() or str(self._default_output_directory),
            "JSON files (*.json);;All files (*)",
        )
        if not path_text:
            return
        try:
            payload = json.loads(Path(path_text).read_text(encoding="utf-8"))
            result = load_reusable_intent(payload)
        except json.JSONDecodeError as exc:
            QMessageBox.critical(self, "Metadata Load Failed", str(exc))
            return
        except SidecarValidationError as exc:
            QMessageBox.critical(self, "Metadata Load Failed", "\n".join(exc.errors))
            return

        self._apply_intent(result.intent)
        for warning in result.warnings:
            self._append_log(warning)
        self._append_log(f"Loaded reusable intent from {path_text}")
        self._update_settings_state()

    def _apply_intent(self, intent: JobIntent) -> None:
        self.magnification_spin.setValue(intent.phase.magnification)
        self.low_hz_spin.setValue(intent.phase.low_hz)
        self.high_hz_spin.setValue(intent.phase.high_hz)
        self.resource_policy_combo.setCurrentIndex(
            self.resource_policy_combo.findData(intent.resource_policy)
        )
        processing_index = self._find_resolution_index(
            self.processing_resolution_combo, intent.processing_resolution
        )
        if processing_index >= 0:
            self.processing_resolution_combo.setCurrentIndex(processing_index)
        self._rebuild_output_resolution_options()
        self._exclusion_zones = intent.exclusion_zones
        self._mask_feather_px = intent.mask_feather_px
        self._analysis_roi = intent.analysis.roi
        self.hardware_acceleration_checkbox.setChecked(
            intent.hardware_acceleration_enabled
        )
        self.analysis_enabled_checkbox.setChecked(intent.analysis.enabled)
        self.analysis_band_mode_combo.setCurrentIndex(
            self.analysis_band_mode_combo.findData(intent.analysis.band_mode)
        )
        self.analysis_support_spin.setValue(intent.analysis.minimum_cell_support_fraction)
        self.analysis_quality_cutoff_spin.setValue(intent.analysis.roi_quality_cutoff)
        self.analysis_low_confidence_spin.setValue(intent.analysis.low_confidence_threshold)
        self.analysis_auto_band_count_spin.setValue(intent.analysis.auto_band_count)
        self.analysis_export_advanced_checkbox.setChecked(intent.analysis.export_advanced_files)
        manual_band_map = {band.band_id: band for band in intent.analysis.manual_bands}
        for band_index, (enabled_check, low_spin, high_spin) in enumerate(
            self._manual_band_controls,
            start=1,
        ):
            band_id = f"band{band_index:02d}"
            band = manual_band_map.get(band_id)
            enabled_check.setChecked(band is not None)
            if band is not None:
                low_spin.setValue(band.low_hz)
                high_spin.setValue(band.high_hz)
        self._drift_assessment = DriftAssessment()
        self._update_analysis_controls()

    def _open_drift_editor(self) -> None:
        source_resolution = self._current_source_resolution()
        if source_resolution is None:
            QMessageBox.information(
                self,
                "Drift Check",
                "Fast probe metadata is still required before the editor can map zones into source-frame coordinates.",
            )
            return

        dialog = DriftEditorDialog(
            source_resolution=source_resolution,
            zones=self._exclusion_zones,
            mask_feather_px=self._mask_feather_px,
            drift_assessment=self._drift_assessment,
            first_frame=self._first_frame_image,
            last_frame=self._last_frame_image,
            parent=self,
        )
        if dialog.exec():
            result = dialog.result_data()
            self._exclusion_zones = result.zones
            self._mask_feather_px = result.mask_feather_px
            self._drift_assessment = result.drift_assessment
            self._append_log(
                f"Updated drift review: {len(result.zones)} mask zones, feather {result.mask_feather_px:.1f}px."
            )
            if (
                result.drift_assessment.warning_active
                and not result.drift_assessment.acknowledged
            ):
                self._append_log("Drift warning remains active without acknowledgement.")
            self._update_settings_state()

    def _open_analysis_roi_editor(self) -> None:
        source_resolution = self._current_source_resolution()
        if source_resolution is None:
            QMessageBox.information(
                self,
                "Analysis ROI",
                "Fast probe metadata is still required before the ROI editor can map geometry into source-frame coordinates.",
            )
            return

        dialog = AnalysisRoiEditorDialog(
            source_resolution=source_resolution,
            roi=self._analysis_roi,
            processing_zones=self._exclusion_zones,
            first_frame=self._first_frame_image,
            last_frame=self._last_frame_image,
            parent=self,
        )
        if dialog.exec():
            self._analysis_roi = dialog.result_data().roi
            self._append_log(f"Updated analysis ROI: {self._analysis_roi_summary()}.")
            self._update_settings_state()

    def _open_analysis_advanced_dialog(self) -> None:
        dialog = AnalysisAdvancedDialog(
            band_mode=self.analysis_band_mode_combo.currentData(),
            minimum_cell_support_fraction=float(self.analysis_support_spin.value()),
            roi_quality_cutoff=float(self.analysis_quality_cutoff_spin.value()),
            low_confidence_threshold=float(self.analysis_low_confidence_spin.value()),
            auto_band_count=int(self.analysis_auto_band_count_spin.value()),
            export_advanced_files=self.analysis_export_advanced_checkbox.isChecked(),
            manual_band_values=self._current_manual_band_values(),
            parent=self,
        )
        if dialog.exec():
            result = dialog.result_data()
            self.analysis_support_spin.setValue(result.minimum_cell_support_fraction)
            self.analysis_quality_cutoff_spin.setValue(result.roi_quality_cutoff)
            self.analysis_low_confidence_spin.setValue(result.low_confidence_threshold)
            self.analysis_auto_band_count_spin.setValue(result.auto_band_count)
            self.analysis_export_advanced_checkbox.setChecked(result.export_advanced_files)
            for (enabled_check, low_spin, high_spin), (
                enabled,
                low_hz,
                high_hz,
            ) in zip(
                self._manual_band_controls,
                result.manual_band_values,
                strict=True,
            ):
                enabled_check.setChecked(enabled)
                low_spin.setValue(low_hz)
                high_spin.setValue(high_hz)
            self._append_log("Updated advanced analysis settings.")
            self._update_settings_state()

    def _run_dry_run_validation(self) -> None:
        try:
            report = self._build_shell_preflight_report()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Dry-run Validation", str(exc))
            return

        self._set_preflight_report_text(
            details=self._preflight_report_details(report),
            warnings=[issue.message for issue in report.warnings],
            blockers=[issue.message for issue in report.blockers],
        )
        self._append_log(
            "Dry-run validation completed in the shell. The worker will still run authoritative pre-flight before render."
        )

    def _build_shell_preflight_report(self):
        if self._current_source_path is None or self._source_probe_info is None:
            raise RuntimeError("Load a source and wait for fast probe metadata before dry-run validation.")
        if not self._current_source_path.exists():
            raise RuntimeError(
                "Selected source file is no longer available. Reload the source before dry-run validation."
            )
        if self._current_fingerprint is None:
            raise RuntimeError("Canonical fingerprinting must complete before dry-run validation.")

        intent = self._build_intent()
        source_metadata = self._build_source_metadata_from_probe(self._source_probe_info)
        output_directory = Path(
            self.output_folder_edit.text().strip() or str(self._default_output_directory)
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        scratch_root = Path(self._preferences.temp_root)
        diagnostics_root = Path(self._preferences.diagnostics_root)
        scratch_root.mkdir(parents=True, exist_ok=True)
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        output_usage = psutil.disk_usage(str(output_directory))
        scratch_usage = psutil.disk_usage(str(scratch_root))
        memory = psutil.virtual_memory()
        retained_evidence_bytes = measure_retained_roots_bytes(
            (diagnostics_root, scratch_root)
        )

        # This shell-side report is a quick operator preview. The worker still
        # re-runs the same categories authoritatively against the live source.
        budgets = ResourceBudget(
            available_scratch_bytes=int(scratch_usage.free),
            scratch_floor_bytes=512 * 1024 * 1024,
            available_output_volume_bytes=int(output_usage.free),
            available_ram_bytes=int(memory.available),
            reserved_ui_headroom_bytes=512 * 1024 * 1024,
            retention_budget_bytes=int(self.retention_budget_gb_spin.value())
            * 1024
            * 1024
            * 1024,
            retained_evidence_bytes=retained_evidence_bytes,
        )
        return run_preflight(
            PreflightInputs(
                intent=intent,
                source=source_metadata,
                budgets=budgets,
                scheduler=choose_scheduler_inputs(
                    intent=intent,
                    source=source_metadata,
                    budgets=budgets,
                    diagnostic_level=self._diagnostic_level(),
                ),
                diagnostic_level=self._diagnostic_level(),
            )
        )

    def _preflight_report_details(self, report: PreflightReport) -> dict[str, Any]:
        plan = self._source_normalization_plan()
        return {
            "source_fps": None if plan is None else plan.native_fps,
            "working_fps": report.nyquist_limit_hz * 2.0,
            "source_is_cfr": None if self._source_probe_info is None else self._source_probe_info.is_cfr,
            "normalization_steps": [] if plan is None else list(plan.normalization_steps),
            "working_source_width": None if plan is None else plan.working_resolution.width,
            "working_source_height": None if plan is None else plan.working_resolution.height,
            "nyquist_limit_hz": report.nyquist_limit_hz,
            "selected_low_hz": self.low_hz_spin.value(),
            "selected_high_hz": self.high_hz_spin.value(),
            "resource_policy": self.resource_policy_combo.currentData().value,
            "active_scratch_required_bytes": report.active_scratch_required_bytes,
            "ram_required_bytes": report.ram_required_bytes,
            "output_staging_required_bytes": report.output_staging_required_bytes,
            "hardware_acceleration_requested": report.hardware_acceleration_requested,
            "hardware_acceleration_active": report.hardware_acceleration_active,
            "acceleration_backend": report.acceleration_backend,
            "acceleration_status": report.acceleration_status,
        }

    def _start_render(self) -> None:
        if self._render_supervisor is not None:
            return
        try:
            request = self._build_render_request()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Start Render", str(exc))
            return

        try:
            shell_preflight = self._build_shell_preflight_report()
        except RuntimeError as exc:
            QMessageBox.warning(self, "Start Render", str(exc))
            return

        self._set_preflight_report_text(
            details=self._preflight_report_details(shell_preflight),
            warnings=[issue.message for issue in shell_preflight.warnings],
            blockers=[issue.message for issue in shell_preflight.blockers],
        )
        if shell_preflight.blockers:
            QMessageBox.warning(
                self,
                "Start Render Blocked",
                "Shell pre-flight found blockers. Review the pre-flight report before rendering.",
            )
            return
        if shell_preflight.warnings and not self._confirm_preflight_warnings(
            shell_preflight
        ):
            self._append_log(
                "Render launch cancelled because pre-flight warnings were not acknowledged."
            )
            return

        try:
            self._controller.start_preflight()
        except InvalidTransitionError as exc:
            QMessageBox.warning(self, "Start Render", str(exc))
            return

        self._active_render_request = request
        self._last_terminal_snapshot = None
        self._render_started_at = time.monotonic()
        self._progress_metric_stage = None
        self._progress_metric_total_frames = None
        self._progress_metric_started_at = None
        self._last_progress_frame_count = None
        self._last_progress_at = None
        self._mean_seconds_per_frame = None
        self._reset_phase_processing_eta_state()
        self._heartbeat_flash_until = None
        self.render_stage_label.setText("Starting worker...")
        self.render_progress_label.setText("Waiting for handshake...")
        self.elapsed_label.setText("0.0 s")
        self.eta_label.setText("Estimating...")
        self.mean_frame_label.setText("-")
        self.terminal_status_label.setText("Run in progress.")

        try:
            self._render_supervisor = self._render_supervisor_factory(request)
            self._render_supervisor.start()
        except Exception as exc:  # pragma: no cover - defensive UI path.
            self._render_supervisor = None
            self._active_render_request = None
            self._render_started_at = None
            self._progress_metric_stage = None
            self._progress_metric_total_frames = None
            self._progress_metric_started_at = None
            self._last_progress_frame_count = None
            self._last_progress_at = None
            self._mean_seconds_per_frame = None
            self._reset_phase_processing_eta_state()
            self._heartbeat_flash_until = None
            self.render_stage_label.setText("Idle")
            self.render_progress_label.setText("No active render.")
            self.elapsed_label.setText("-")
            self.eta_label.setText("-")
            self.mean_frame_label.setText("-")
            self.terminal_status_label.setText("No run yet.")
            try:
                self._controller.abort_preflight()
            except InvalidTransitionError:
                pass
            self._refresh_state()
            QMessageBox.critical(self, "Start Render Failed", str(exc))
            return

        self._render_timer.start()
        self._append_log(f"Render launched for job {request.job_id}.")
        self._refresh_state()

    def _confirm_preflight_warnings(self, report: PreflightReport) -> bool:
        warning_messages = [issue.message for issue in report.warnings]
        if not warning_messages:
            return True

        dialog = PreflightWarningDialog(
            active_scratch_text=self._format_megabytes(
                report.active_scratch_required_bytes
            ),
            ram_text=self._format_megabytes(report.ram_required_bytes),
            output_staging_text=self._format_megabytes(
                report.output_staging_required_bytes
            ),
            warning_messages=warning_messages,
            parent=self,
        )
        return dialog.exec() == int(QDialog.DialogCode.Accepted)

    def _cancel_render(self) -> None:
        if self._render_supervisor is None:
            return
        self._append_log("Cancellation requested.")
        self._render_supervisor.cancel()
        self._refresh_state()

    def _prepare_new_run(self) -> None:
        try:
            self._controller.reset_after_terminal()
        except InvalidTransitionError:
            return
        self._last_terminal_snapshot = None
        self.terminal_status_label.setText(
            "Ready for a fresh attempt with the current settings."
        )
        self.render_stage_label.setText("Idle")
        self.render_progress_label.setText("No active render.")
        self.elapsed_label.setText("-")
        self.eta_label.setText("-")
        self.mean_frame_label.setText("-")
        self._progress_metric_stage = None
        self._progress_metric_total_frames = None
        self._progress_metric_started_at = None
        self._last_progress_frame_count = None
        self._last_progress_at = None
        self._mean_seconds_per_frame = None
        self._reset_phase_processing_eta_state()
        self._heartbeat_flash_until = None
        self._append_log(
            "Terminal run state cleared. The current source and settings remain loaded for a new attempt."
        )
        self._refresh_state()

    def _poll_render_supervisor(self) -> None:
        if self._render_supervisor is None:
            return
        result = self._render_supervisor.poll()
        snapshot = result.snapshot
        for event in result.events:
            if event.message_type == "heartbeat":
                self._note_heartbeat(event.received_at)
                continue
            self._consume_render_event_for_metrics(event)
            message = self._format_render_event(event)
            if message:
                self._append_log(message)

        self._sync_render_metrics(snapshot)
        self._apply_render_state(snapshot)
        self._refresh_render_status(snapshot)
        self._refresh_state()

        if snapshot.is_terminal:
            self._render_timer.stop()
            self._last_terminal_snapshot = snapshot
            self._append_log(self._format_terminal_summary(snapshot))
            self._render_supervisor.close()
            self._render_supervisor = None
            self._enforce_retention_budget()
            if self.isVisible():
                self._show_terminal_outcome_dialog(snapshot)
            self._refresh_state()

    def _sync_render_metrics(self, snapshot: RenderSupervisorSnapshot) -> None:
        now = time.monotonic()
        if self._render_started_at is not None:
            self.elapsed_label.setText(f"{now - self._render_started_at:.1f} s")

        if snapshot.phase in {"starting", "preflight"}:
            self.eta_label.setText("Estimating...")
            return
        if snapshot.phase != "rendering":
            self.eta_label.setText("-")
            return

        current_stage = snapshot.stage or "render"
        current_total_frames = snapshot.total_frames
        progress_frames_completed = self._effective_progress_frame_count(snapshot)
        if (
            current_stage != self._progress_metric_stage
            or current_total_frames != self._progress_metric_total_frames
        ):
            # The worker reports stage-local counters. Reset the shell-side rate
            # estimate whenever the stage changes so decode timing does not leak
            # into phase-processing or encode ETA text.
            self._progress_metric_stage = current_stage
            self._progress_metric_total_frames = current_total_frames
            self._progress_metric_started_at = now
            self._last_progress_frame_count = progress_frames_completed
            self._last_progress_at = (
                now if progress_frames_completed is not None else None
            )
            self._mean_seconds_per_frame = None
            self.mean_frame_label.setText("-")
            self._refresh_eta_label(snapshot)
            return

        if progress_frames_completed is None:
            self._refresh_eta_label(snapshot)
            return
        if self._last_progress_frame_count is None:
            self._last_progress_frame_count = progress_frames_completed
            self._last_progress_at = now
            self._refresh_eta_label(snapshot)
            return
        if progress_frames_completed < self._last_progress_frame_count:
            # Stage-local frame counters can reset when decode hands off to
            # phase processing or later stages. Treat that as a fresh baseline
            # instead of pinning the UI to the previous stage's terminal count.
            self._last_progress_frame_count = progress_frames_completed
            self._last_progress_at = now
            self._progress_metric_started_at = now
            self._mean_seconds_per_frame = None
            self.mean_frame_label.setText("-")
            self._refresh_eta_label(snapshot)
            return
        if (
            progress_frames_completed <= self._last_progress_frame_count
            or self._last_progress_at is None
        ):
            self._refresh_eta_label(snapshot)
            return

        if (
            current_stage == "phase_processing"
            and self._progress_metric_started_at is not None
            and progress_frames_completed > 0
        ):
            # Phase-processing emits many token-only sub-step updates before the
            # next encode batch lands. Using a stage-average rate there is much
            # steadier than re-timing every token-to-token jump.
            seconds_per_frame = max(
                now - self._progress_metric_started_at,
                1e-6,
            ) / progress_frames_completed
        else:
            delta_frames = progress_frames_completed - self._last_progress_frame_count
            delta_seconds = max(now - self._last_progress_at, 1e-6)
            seconds_per_frame = delta_seconds / delta_frames
        if self._mean_seconds_per_frame is None:
            self._mean_seconds_per_frame = seconds_per_frame
        else:
            smoothing = 0.85 if current_stage == "phase_processing" else 0.7
            self._mean_seconds_per_frame = (
                self._mean_seconds_per_frame * smoothing
                + seconds_per_frame * (1.0 - smoothing)
            )
        self._last_progress_frame_count = progress_frames_completed
        self._last_progress_at = now
        self.mean_frame_label.setText(f"{self._mean_seconds_per_frame:.3f} s")
        self._refresh_eta_label(snapshot)

    def _refresh_eta_label(self, snapshot: RenderSupervisorSnapshot) -> None:
        """Update the GUI-only ETA from the current stage's recent frame rate without asking the worker for extra state."""

        if snapshot.phase != "rendering":
            self.eta_label.setText("-")
            return
        progress_frames_completed = self._effective_progress_frame_count(snapshot)
        if progress_frames_completed is None or snapshot.total_frames is None:
            self.eta_label.setText("Estimating...")
            return
        if progress_frames_completed >= snapshot.total_frames:
            self.eta_label.setText("0.0 s")
            return
        if self._mean_seconds_per_frame is None or progress_frames_completed <= 0:
            self.eta_label.setText("Estimating...")
            return
        remaining_frames = max(snapshot.total_frames - progress_frames_completed, 0.0)
        eta_seconds = remaining_frames * self._mean_seconds_per_frame
        self.eta_label.setText(self._format_eta_seconds(eta_seconds))

    def _effective_progress_frame_count(
        self,
        snapshot: RenderSupervisorSnapshot,
    ) -> float | None:
        """Return the shell-side best estimate of completed frames for ETA purposes, including phase-stage sub-step progress."""

        completed = (
            None
            if snapshot.frames_completed is None
            else float(snapshot.frames_completed)
        )
        if snapshot.phase != "rendering" or snapshot.stage != "phase_processing":
            return completed
        if self._phase_virtual_frames_completed is None:
            return completed
        if completed is None:
            return self._phase_virtual_frames_completed
        return max(completed, self._phase_virtual_frames_completed)

    def _consume_render_event_for_metrics(self, event: RenderEvent) -> None:
        """Fold shell-visible progress traffic into ETA state so the GUI can react to worker sub-stage motion without changing the IPC contract."""

        if event.message_type == "stage_started":
            self._reset_phase_processing_eta_state()
            return
        if event.message_type != "progress_update":
            return
        payload = event.payload
        if str(payload.get("stage") or "") != "phase_processing":
            return

        decode_counter = payload.get("decode_frames_completed")
        if isinstance(decode_counter, int):
            if (
                self._phase_last_decode_counter is None
                or decode_counter < self._phase_last_decode_counter
            ):
                self._phase_nominal_chunk_frame_count = decode_counter
            else:
                chunk_frame_count = decode_counter - self._phase_last_decode_counter
                if chunk_frame_count > 0:
                    self._phase_nominal_chunk_frame_count = chunk_frame_count
            self._phase_last_decode_counter = decode_counter

        frames_completed = payload.get("frames_completed")
        if isinstance(frames_completed, int):
            actual_completed = float(frames_completed)
            if self._phase_virtual_frames_completed is None:
                self._phase_virtual_frames_completed = actual_completed
            else:
                self._phase_virtual_frames_completed = max(
                    self._phase_virtual_frames_completed,
                    actual_completed,
                )
            return

        detail = payload.get("detail")
        if not isinstance(detail, str):
            return
        chunk_progress_fraction = self._phase_processing_detail_fraction(detail)
        if chunk_progress_fraction is None:
            return
        progress_token = payload.get("progress_token")
        encoded_baseline = self._phase_processing_encoded_baseline(progress_token)
        if encoded_baseline is None or self._phase_nominal_chunk_frame_count is None:
            return
        candidate_progress = float(
            encoded_baseline
            + (self._phase_nominal_chunk_frame_count * chunk_progress_fraction)
        )
        if self._phase_virtual_frames_completed is None:
            self._phase_virtual_frames_completed = candidate_progress
            return
        self._phase_virtual_frames_completed = max(
            self._phase_virtual_frames_completed,
            candidate_progress,
        )

    def _phase_processing_encoded_baseline(
        self,
        progress_token: object,
    ) -> int | None:
        """Extract the last fully encoded frame count from the phase-processing token so token-only updates can still advance the ETA model."""

        if not isinstance(progress_token, str):
            return None
        token_parts = progress_token.split(":")
        if len(token_parts) != 3 or token_parts[0] != "phase_processing":
            return None
        try:
            return int(token_parts[1])
        except ValueError:
            return None

    def _phase_processing_detail_fraction(self, detail: str) -> float | None:
        """Map one phase-engine progress string onto an approximate chunk fraction so the GUI ETA moves during compute-heavy work instead of waiting for encode-only counters."""

        motion_weight = 0.34
        temporal_weight = 0.16
        warp_weight = 0.34

        if detail == "motion_grid_done":
            return motion_weight
        if detail == "x_temporal_band_done":
            return motion_weight + temporal_weight
        if detail == "y_temporal_band_done":
            return motion_weight + (temporal_weight * 2.0)
        if detail == "warp_done":
            return 1.0

        fraction = self._progress_detail_fraction(detail)
        if fraction is None:
            return None

        if detail.startswith("motion_grid_tile_") or detail.startswith(
            "motion_grid_partition_"
        ):
            return motion_weight * fraction
        if detail.startswith("x_temporal_band_"):
            return motion_weight + (temporal_weight * fraction)
        if detail.startswith("y_temporal_band_"):
            return motion_weight + temporal_weight + (temporal_weight * fraction)
        if detail.startswith("warp_"):
            return motion_weight + (temporal_weight * 2.0) + (warp_weight * fraction)
        return None

    def _progress_detail_fraction(self, detail: str) -> float | None:
        """Parse one `_X_of_Y` progress detail into a stable fraction."""

        match = re.search(r"_(\d+)_of_(\d+)$", detail)
        if match is None:
            return None
        total = int(match.group(2))
        if total <= 0:
            return None
        completed = int(match.group(1))
        return float(max(0.0, min(completed / total, 1.0)))

    def _reset_phase_processing_eta_state(self) -> None:
        """Clear chunk-local phase ETA state when a new render starts or the worker leaves phase processing."""

        self._phase_virtual_frames_completed = None
        self._phase_last_decode_counter = None
        self._phase_nominal_chunk_frame_count = None

    def _format_eta_seconds(self, seconds: float) -> str:
        """Render ETA text compactly so longer runs do not leave the shell showing a wall of seconds."""

        if seconds < 60.0:
            return f"{seconds:.1f} s"
        total_seconds = max(0, int(round(seconds)))
        minutes, second_value = divmod(total_seconds, 60)
        if minutes < 60:
            return f"{minutes:d}m {second_value:02d}s"
        hours, minute_value = divmod(minutes, 60)
        return f"{hours:d}h {minute_value:02d}m {second_value:02d}s"

    def _apply_render_state(self, snapshot: RenderSupervisorSnapshot) -> None:
        if (
            snapshot.phase == "rendering"
            and self._controller.state is UiState.PREFLIGHT_CHECK
        ):
            try:
                self._controller.begin_rendering()
            except InvalidTransitionError:
                pass

        if snapshot.phase == "complete":
            if self._controller.state in {UiState.PREFLIGHT_CHECK, UiState.RENDERING}:
                self._controller.mark_complete()
            return

        if snapshot.phase == "failed":
            if self._controller.state in {UiState.PREFLIGHT_CHECK, UiState.RENDERING}:
                self._controller.mark_failed()
            return

        if snapshot.phase == "cancelled":
            if self._controller.state in {UiState.PREFLIGHT_CHECK, UiState.RENDERING}:
                self._controller.mark_cancelled()

    def _refresh_render_status(self, snapshot: RenderSupervisorSnapshot) -> None:
        self.render_stage_label.setText(snapshot.stage or snapshot.phase.title())
        if snapshot.frames_completed is not None and snapshot.total_frames:
            self.render_progress_label.setText(
                f"{snapshot.frames_completed} / {snapshot.total_frames} frames"
            )
        elif snapshot.progress_token is not None:
            self.render_progress_label.setText(
                f"Progress token {snapshot.progress_token}"
            )
        else:
            self.render_progress_label.setText("Waiting for progress.")

        if (
            snapshot.preflight_details
            or snapshot.preflight_warnings
            or snapshot.preflight_blockers
        ):
            self._set_preflight_report_text(
                details=snapshot.preflight_details,
                warnings=list(snapshot.preflight_warnings),
                blockers=list(snapshot.preflight_blockers),
            )

        if snapshot.watchdog_status == "warning" and snapshot.watchdog_message:
            self.terminal_status_label.setText(snapshot.watchdog_message)
        elif snapshot.phase == "complete":
            self.terminal_status_label.setText("Render Complete")
        elif snapshot.phase == "failed":
            self.terminal_status_label.setText(self._format_terminal_summary(snapshot))
        elif snapshot.phase == "cancelled":
            self.terminal_status_label.setText("Render cancelled.")
        else:
            self.terminal_status_label.setText("Run in progress.")

    def _set_preflight_report_text(
        self,
        *,
        details: dict[str, Any],
        warnings: list[str],
        blockers: list[str],
    ) -> None:
        normalization_steps = details.get("normalization_steps") or []
        normalization_status = (
            "Direct source path"
            if not normalization_steps
            else ", ".join(str(step) for step in normalization_steps)
        )
        lines = [
            f"Source FPS: {self._format_measurement(details.get('source_fps'), unit=' fps')}",
            f"Working FPS: {self._format_measurement(details.get('working_fps'), unit=' fps')}",
            (
                "Source timing: "
                + (
                    "CFR"
                    if details.get("source_is_cfr") is True
                    else "VFR"
                    if details.get("source_is_cfr") is False
                    else "-"
                )
            ),
            f"Normalization: {normalization_status}",
            (
                "Working source: "
                f"{details.get('working_source_width', '-')}"
                " x "
                f"{details.get('working_source_height', '-')}"
            ),
            f"Nyquist limit: {self._format_measurement(details.get('nyquist_limit_hz'), unit=' Hz')}",
            (
                "Selected band: "
                f"{self._format_measurement(details.get('selected_low_hz'), unit=' Hz')} to "
                f"{self._format_measurement(details.get('selected_high_hz'), unit=' Hz')}"
            ),
            f"Resource policy: {details.get('resource_policy', '-')}",
            (
                "Hardware acceleration requested: "
                + (
                    "yes"
                    if details.get("hardware_acceleration_requested") is True
                    else "no"
                )
            ),
            (
                "Hardware acceleration active: "
                + (
                    "yes"
                    if details.get("hardware_acceleration_active") is True
                    else "no"
                )
            ),
            (
                "Acceleration backend: "
                f"{details.get('acceleration_backend') or 'cpu'}"
            ),
            (
                "Acceleration status: "
                f"{details.get('acceleration_status') or '-'}"
            ),
            f"Active scratch required: {self._format_megabytes(details.get('active_scratch_required_bytes'))}",
            f"RAM required: {self._format_megabytes(details.get('ram_required_bytes'))}",
            f"Output staging required: {self._format_megabytes(details.get('output_staging_required_bytes'))}",
            "",
            "Warnings:",
        ]
        if warnings:
            lines.extend(f"- {message}" for message in warnings)
        else:
            lines.append("- none")
        lines.append("")
        lines.append("Blockers:")
        if blockers:
            lines.extend(f"- {message}" for message in blockers)
        else:
            lines.append("- none")
        self.preflight_report_view.setPlainText("\n".join(lines))

    @staticmethod
    def _format_measurement(value: Any, *, unit: str = "") -> str:
        if not isinstance(value, (int, float)):
            return "-"
        return f"{value:,.3f}{unit}"

    @staticmethod
    def _format_megabytes(value: Any) -> str:
        if not isinstance(value, (int, float)):
            return "-"
        mebibytes = int((float(value) + (1024 * 1024) - 1) // (1024 * 1024))
        return f"{mebibytes:,} MB"

    def _note_heartbeat(self, received_at: float) -> None:
        self._heartbeat_flash_until = received_at + 0.35

    def _refresh_heartbeat_indicator(self) -> None:
        render_active = self._render_supervisor is not None and not self._last_terminal_snapshot
        if not render_active:
            self.heartbeat_dot_label.setStyleSheet("color: #5a5a5a;")
            return

        now = time.monotonic()
        if self._heartbeat_flash_until is not None and now <= self._heartbeat_flash_until:
            self.heartbeat_dot_label.setStyleSheet("color: #ff2f2f;")
            return
        self.heartbeat_dot_label.setStyleSheet("color: #6a1a1a;")

    def _format_render_event(self, event: RenderEvent) -> str:
        payload = event.payload
        if event.message_type == "preflight_started":
            return "Worker pre-flight started."
        if event.message_type == "preflight_report":
            return (
                f"Worker pre-flight reported {len(payload.get('warnings', []))} warnings and "
                f"{len(payload.get('blockers', []))} blockers."
            )
        if event.message_type == "stage_started":
            return f"Stage started: {payload.get('stage', 'unknown')}."
        if event.message_type == "progress_update":
            if (
                payload.get("frames_completed") is not None
                and payload.get("total_frames") is not None
            ):
                return (
                    f"Progress: {payload['frames_completed']} / {payload['total_frames']} frames "
                    f"during {payload.get('stage', 'active stage')}."
                )
            return f"Progress token advanced to {payload.get('progress_token')}."
        if event.message_type == "warning":
            messages = payload.get("messages", [])
            if messages:
                return f"Warning: {messages[0]}"
        if event.message_type == "artifact_paths":
            return "Final artifact paths reported by the worker."
        if event.message_type == "failure":
            detail = payload.get("detail")
            message = (
                f"Worker reported failure '{payload.get('classification', 'unknown_failure')}' "
                f"at stage '{payload.get('stage', 'unknown')}'."
            )
            if detail:
                return f"{message} Detail: {detail}"
            return message
        if event.message_type == "job_completed":
            return "Worker reported completed output pair."
        if event.message_type == "job_cancelled":
            return "Worker reported cancellation."
        return event.message_type

    def _format_terminal_summary(self, snapshot: RenderSupervisorSnapshot) -> str:
        if snapshot.phase == "complete":
            return "Render completed successfully."
        if snapshot.phase == "cancelled":
            return "Render cancelled before completion."
        summary = (
            "Render failed with "
            f"{snapshot.failure_classification or snapshot.watchdog_classification or 'unknown_failure'}."
        )
        if snapshot.failure_detail:
            return f"{summary} {snapshot.failure_detail}"
        return summary

    def _build_render_request(self) -> RenderRequest:
        if self._current_source_path is None:
            raise RuntimeError("Choose a source video before starting a render.")
        if not self._current_source_path.exists():
            raise RuntimeError(
                "Selected source file is no longer available. Reload the source before starting a render."
            )
        if self._current_fingerprint is None:
            raise RuntimeError(
                "Canonical fingerprinting must complete before render can start."
            )
        if self._source_probe_info is None:
            raise RuntimeError(
                "Fast probe metadata is required before render can start."
            )
        if self._controller.state is not UiState.READY:
            raise RuntimeError(
                "The current source and settings are not in the Ready state."
            )

        output_stem = self.output_name_edit.text().strip() or "render"
        base_output_directory = Path(
            self.output_folder_edit.text().strip() or str(self._default_output_directory)
        )
        base_output_directory.mkdir(parents=True, exist_ok=True)
        output_directory = _timestamped_render_directory(base_output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        job_id = f"render-{uuid.uuid4().hex[:12]}"
        scratch_directory = Path(self._preferences.temp_root) / job_id
        diagnostics_directory = Path(self._preferences.diagnostics_root) / job_id
        return RenderRequest(
            job_id=job_id,
            intent=self._build_intent(),
            paths=RenderPaths(
                source_path=self._current_source_path,
                output_directory=output_directory,
                output_stem=output_stem,
                scratch_directory=scratch_directory,
                diagnostics_directory=diagnostics_directory,
            ),
            expected_source_fingerprint_sha256=self._current_fingerprint,
            diagnostic_level=self._diagnostic_level(),
            diagnostics_cap_bytes=int(self.diagnostics_cap_mb_spin.value()) * 1024 * 1024,
            retention_budget_bytes=int(self.retention_budget_gb_spin.value())
            * 1024
            * 1024
            * 1024,
            drift_assessment=self._drift_assessment,
        )

    def _default_render_supervisor_factory(self, request: RenderRequest) -> RenderSupervisor:
        def build_config(
            host: str,
            port: int,
            session_token: str,
            job_id: str,
            role: str,
        ) -> RenderWorkerConfig:
            return RenderWorkerConfig(
                host=host,
                port=port,
                session_token=session_token,
                job_id=job_id,
                role=role,
                request=request,
            )

        return RenderSupervisor(
            job_id=request.job_id,
            launch_plan=WorkerLaunchPlan(
                role="render",
                target=render_worker_entry,
                config_factory=build_config,
            ),
            thresholds=WatchdogThresholds(
                soft_timeout_seconds=4.0,
                hard_timeout_seconds=12.0,
                stall_timeout_seconds=30.0,
            ),
        )

    def _diagnostic_level(self) -> DiagnosticLevel:
        return DiagnosticLevel(self.diagnostic_level_combo.currentText().lower())

    def _load_persisted_state(self) -> None:
        state = load_app_state(self._state_path)
        if state is None:
            return

        self._preferences = migrate_legacy_temp_root(state.preferences, self._runtime_root)
        _configure_process_temp_root(Path(self._preferences.temp_root))
        self.diagnostic_level_combo.setCurrentText(
            self._preferences.diagnostic_level.value.title()
        )
        self.diagnostics_cap_mb_spin.setValue(self._preferences.diagnostics_cap_mb)
        self.retention_budget_gb_spin.setValue(self._preferences.retention_budget_gb)

        if state.last_used is not None:
            self._apply_intent(state.last_used.intent)
            self.output_name_edit.setText(state.last_used.output_stem)
            self.diagnostic_level_combo.setCurrentText(
                state.last_used.diagnostic_level.value.title()
            )

    def _persist_state(self) -> None:
        self._preferences = self._preferences.__class__(
            temp_root=self._preferences.temp_root,
            diagnostics_root=self._preferences.diagnostics_root,
            diagnostic_level=self._diagnostic_level(),
            diagnostics_cap_mb=int(self.diagnostics_cap_mb_spin.value()),
            retention_budget_gb=int(self.retention_budget_gb_spin.value()),
        )
        last_used = LastUsedSettings(
            intent=self._build_intent(),
            output_directory=self.output_folder_edit.text().strip() or None,
            output_stem=self.output_name_edit.text().strip() or "render",
            diagnostic_level=self._diagnostic_level(),
        )
        save_app_state(
            self._state_path,
            PersistedAppState(
                preferences=self._preferences,
                last_used=last_used,
            ),
        )

    def _build_source_metadata_from_probe(self, probe: FfprobeMediaInfo) -> SourceMetadata:
        normalization_plan = build_source_normalization_plan(probe)
        tags = {
            (probe.color_primaries or "").lower(),
            (probe.color_transfer or "").lower(),
            (probe.color_space or "").lower(),
        }
        has_hdr = any(
            marker in {"bt2020", "smpte2084", "arib-std-b67"} for marker in tags
        )
        explicit_rec709 = all(
            tag in {"", "bt709", "tv", "pc", "gbr"}
            for tag in tags | {(probe.color_range or "").lower()}
        )
        return SourceMetadata(
            fps=normalization_plan.working_fps,
            duration_seconds=probe.duration_seconds,
            frame_count=normalization_plan.working_frame_count,
            width=normalization_plan.working_resolution.width,
            height=normalization_plan.working_resolution.height,
            is_cfr=True,
            bit_depth=probe.bit_depth,
            pixel_aspect_ratio=1.0,
            source_fps=normalization_plan.native_fps,
            source_is_cfr=normalization_plan.source_is_cfr,
            source_pixel_aspect_ratio=normalization_plan.source_pixel_aspect_ratio,
            requires_cfr_normalization=normalization_plan.requires_cfr_normalization,
            requires_square_pixel_normalization=normalization_plan.requires_square_pixel_normalization,
            normalization_steps=normalization_plan.normalization_steps,
            has_unsupported_rotation=(
                isinstance(probe.rotation_degrees, (int, float))
                and abs(probe.rotation_degrees) % 360.0 > 1e-6
            ),
            explicit_rec709_compatible=explicit_rec709,
            heuristic_sdr_allowed=not has_hdr and not explicit_rec709,
            has_hdr_markers=has_hdr,
            contradictory_color_metadata=False,
            decoded_format_supported=True,
        )

    def _open_output_folder(self) -> None:
        if self._active_render_request is None:
            return
        self._open_path(self._active_render_request.paths.output_directory)

    def _show_terminal_outcome_dialog(self, snapshot: RenderSupervisorSnapshot) -> None:
        if self._active_render_request is None:
            return
        summary = self.terminal_status_label.text()
        title = {
            "complete": "Render Complete",
            "failed": "Render Failed",
            "cancelled": "Render Cancelled",
        }.get(snapshot.phase, "Render Outcome")
        outcome = TerminalOutcomeData(
            title=title,
            summary=summary,
            output_directory=self._active_render_request.paths.output_directory,
            output_video_path=self._active_render_request.paths.final_mp4_path,
        )
        primary_action = self._open_output_folder
        secondary_action = self._open_output_video
        if snapshot.phase == "failed":
            outcome = TerminalOutcomeData(
                title=title,
                summary=summary,
                output_directory=self._active_render_request.paths.diagnostics_directory,
                output_video_path=self._active_render_request.paths.final_mp4_path,
                primary_action_label="Open Diagnostics",
                secondary_action_label="Purge Failed-run Files",
                primary_action_enabled=self._active_render_request.paths.diagnostics_directory.exists(),
                secondary_action_enabled=(
                    self._active_render_request.paths.diagnostics_directory.exists()
                    or self._active_render_request.paths.scratch_directory.exists()
                ),
            )
            primary_action = self._open_diagnostics_folder
            secondary_action = self._purge_job_temp_files
        dialog = TerminalOutcomeDialog(
            outcome=outcome,
            open_output=primary_action,
            open_video=secondary_action,
            parent=self,
        )
        dialog.exec()

    def _open_output_video(self) -> None:
        if self._active_render_request is None:
            return
        video_path = self._active_render_request.paths.final_mp4_path
        if video_path.exists():
            self._open_path(video_path)

    def _open_sidecar(self) -> None:
        if self._active_render_request is None:
            return
        sidecar_path = self._active_render_request.paths.final_sidecar_path
        if sidecar_path.exists():
            self._open_path(sidecar_path)

    def _open_diagnostics_folder(self) -> None:
        if self._active_render_request is None:
            return
        self._open_path(self._active_render_request.paths.diagnostics_directory)

    def _open_temp_folder(self) -> None:
        if self._active_render_request is None:
            return
        self._open_path(self._active_render_request.paths.scratch_directory)

    def _open_path(self, path: Path) -> None:
        if not path.exists():
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _purge_job_temp_files(self) -> None:
        if self._active_render_request is None:
            return
        if self._render_supervisor is not None:
            QMessageBox.information(
                self,
                "Purge Temp Files",
                "Stop the active render before purging job temp files.",
            )
            return

        removed_any = False
        for directory in (
            self._active_render_request.paths.scratch_directory,
            self._active_render_request.paths.diagnostics_directory,
        ):
            if directory.exists():
                # This purge intentionally removes run-scoped temp and diagnostics
                # roots only. It does not touch the final MP4 or final sidecar.
                shutil.rmtree(directory, ignore_errors=True)
                removed_any = True
        if removed_any:
            self._append_log(
                "Purged run-scoped temp and diagnostics directories for the last job."
            )
        self._refresh_state()

    def _purge_failed_run_temp_files(self) -> None:
        if self._render_supervisor is not None:
            QMessageBox.information(
                self,
                "Purge Failed-run Temp Files",
                "Stop the active render before purging retained failed-run material.",
            )
            return

        diagnostics_root = Path(self._preferences.diagnostics_root)
        scratch_root = Path(self._preferences.temp_root)
        current_job_id = (
            None if self._active_render_request is None else self._active_render_request.job_id
        )
        entries = []
        if diagnostics_root.exists():
            for job_dir in diagnostics_root.iterdir():
                if not job_dir.is_dir() or job_dir.name == current_job_id:
                    continue
                bundle_path = job_dir / "diagnostics_bundle.json"
                status = "failed"
                if bundle_path.exists():
                    try:
                        status = json.loads(bundle_path.read_text(encoding="utf-8")).get(
                            "status",
                            "failed",
                        )
                    except json.JSONDecodeError:
                        status = "failed"
                if status == "completed":
                    continue
                entries.append(build_retained_entry(job_dir))
                matching_scratch = scratch_root / job_dir.name
                if matching_scratch.exists():
                    entries.append(build_retained_entry(matching_scratch))
        removed = purge_retained_entries(entries)
        if removed:
            self._append_log(
                f"Purged {len(removed)} failed-run temp or diagnostics trees."
            )
        self._refresh_state()

    def _enforce_retention_budget(self) -> None:
        diagnostics_root = Path(self._preferences.diagnostics_root)
        scratch_root = Path(self._preferences.temp_root)
        preserve_job_id = None
        if (
            self._last_terminal_snapshot is not None
            and self._last_terminal_snapshot.phase in {"failed", "cancelled"}
            and self._active_render_request is not None
        ):
            preserve_job_id = self._active_render_request.job_id

        entries = []
        for root in (diagnostics_root, scratch_root):
            if not root.exists():
                continue
            for child in root.iterdir():
                if child.is_dir():
                    entries.append(
                        build_retained_entry(
                            child,
                            preserve=child.name == preserve_job_id,
                        )
                    )
        purge = plan_oldest_first_purge(
            entries,
            budget_bytes=int(self.retention_budget_gb_spin.value()) * 1024 * 1024 * 1024,
        )
        removed = purge_retained_entries(list(purge))
        if removed:
            self._append_log(
                f"Retention budget enforcement purged {len(removed)} retained artifact roots."
            )

    def _rgb_frame_to_qimage(self, frame: object) -> QImage:
        # The frame extractor hands back packed RGB bytes. The GUI copies that
        # into a QImage so the worker-side media buffer can be released safely.
        image = QImage(
            frame.rgb24,
            frame.width,
            frame.height,
            frame.width * 3,
            QImage.Format.Format_RGB888,
        )
        return image.copy()

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt naming convention.
        if self._first_frame_image is not None:
            self._set_first_frame_preview(self._first_frame_image)
        super().resizeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # pragma: no cover - Qt close path.
        self._persist_state()
        for worker in (
            self._baseline_band_worker,
            self._frame_pair_worker,
            self._probe_worker,
            self._fingerprint_worker,
        ):
            if worker is not None and worker.isRunning():
                worker.quit()
                worker.wait(100)
        if self._render_supervisor is not None:
            # Closing the shell should not orphan a worker. The shell requests
            # cooperative cancellation and then closes its supervision handles.
            self._render_supervisor.cancel()
            self._render_supervisor.close()
            self._render_supervisor = None
        super().closeEvent(event)

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
