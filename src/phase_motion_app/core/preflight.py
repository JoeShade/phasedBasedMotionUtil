"""This file owns mandatory pre-flight checks so unsafe jobs are blocked before a worker spends time or disk on an untrustworthy render."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from phase_motion_app.core.acceleration import (
    detect_acceleration_capability,
    resolve_acceleration_request,
)
from phase_motion_app.core.models import DiagnosticLevel, JobIntent, Resolution, ResourcePolicy
from phase_motion_app.core.quantitative_analysis import (
    choose_quantitative_analysis_resolution,
)


class PreflightSeverity(str, Enum):
    """This enum keeps warnings separate from hard blockers so the UI can be explicit about what must stop a run."""

    WARNING = "warning"
    BLOCKER = "blocker"


class AnalyzerExecutionMode(str, Enum):
    """This enum makes the analysis collection handoff explicit so policy presets document whether chunks stay inline or move onto a bounded helper thread."""

    INLINE = "inline"
    BACKGROUND_THREAD = "background_thread"


@dataclass(frozen=True)
class PreflightIssue:
    """This model carries one human-readable pre-flight finding."""

    severity: PreflightSeverity
    code: str
    message: str


@dataclass(frozen=True)
class SourceMetadata:
    """This model captures the source facts that pre-flight needs to judge whether the clip is safe to process."""

    fps: float
    duration_seconds: float
    frame_count: int
    width: int
    height: int
    is_cfr: bool
    bit_depth: int
    pixel_aspect_ratio: float = 1.0
    source_fps: float | None = None
    source_is_cfr: bool | None = None
    source_pixel_aspect_ratio: float = 1.0
    requires_cfr_normalization: bool = False
    requires_square_pixel_normalization: bool = False
    normalization_steps: tuple[str, ...] = ()
    has_unsupported_rotation: bool = False
    has_variable_display_transform: bool = False
    explicit_rec709_compatible: bool = True
    heuristic_sdr_allowed: bool = False
    has_hdr_markers: bool = False
    contradictory_color_metadata: bool = False
    decoded_format_supported: bool = True


@dataclass(frozen=True)
class ResourceBudget:
    """This model separates active scratch, retention budget, and RAM because the design doc forbids treating them as the same bucket."""

    available_scratch_bytes: int
    scratch_floor_bytes: int
    available_output_volume_bytes: int
    available_ram_bytes: int
    reserved_ui_headroom_bytes: int
    retention_budget_bytes: int
    retained_evidence_bytes: int


@dataclass(frozen=True)
class SchedulerInputs:
    """This model carries the real bounded worker execution plan so pre-flight math, diagnostics, and runtime behavior stay aligned."""

    chunk_frames: int = 32
    chunk_cap_frames: int = 32
    chunk_target_ram_fraction: float = 0.5
    thread_limit: int = 4
    precision_bytes: int = 4
    native_buffer_multiplier: float = 6.0
    internal_queue_depth: int = 1
    compute_worker_count: int = 1
    warp_worker_count: int = 1
    motion_worker_count: int = 1
    analyzer_mode: AnalyzerExecutionMode = AnalyzerExecutionMode.INLINE
    analysis_queue_depth: int = 0


@dataclass(frozen=True)
class PreflightInputs:
    """This model collects the user intent, source facts, and machine budgets needed to admit or reject a run."""

    intent: JobIntent
    source: SourceMetadata
    budgets: ResourceBudget
    scheduler: SchedulerInputs = field(default_factory=SchedulerInputs)
    diagnostic_level: DiagnosticLevel = DiagnosticLevel.BASIC


@dataclass(frozen=True)
class PreflightReport:
    """This model summarizes admission math and findings so the result can be shown in UI, logs, and sidecars."""

    issues: tuple[PreflightIssue, ...]
    nyquist_limit_hz: float
    active_scratch_required_bytes: int
    ram_required_bytes: int
    output_staging_required_bytes: int
    hardware_acceleration_requested: bool = False
    hardware_acceleration_active: bool = False
    acceleration_backend: str | None = None
    acceleration_status: str | None = None

    @property
    def blockers(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is PreflightSeverity.BLOCKER
        )

    @property
    def warnings(self) -> tuple[PreflightIssue, ...]:
        return tuple(
            issue for issue in self.issues if issue.severity is PreflightSeverity.WARNING
        )

    @property
    def can_render(self) -> bool:
        return not self.blockers


def _estimate_diagnostics_bytes(diagnostic_level: DiagnosticLevel) -> int:
    """Keep diagnostics-size assumptions in one place so scratch and retention logic use the same bucket definition."""

    return {
        DiagnosticLevel.OFF: 0,
        DiagnosticLevel.BASIC: 32 * 1024 * 1024,
        DiagnosticLevel.DETAILED: 128 * 1024 * 1024,
        DiagnosticLevel.TRACE: 256 * 1024 * 1024,
    }[diagnostic_level]


def choose_scheduler_inputs(
    *,
    intent: JobIntent,
    source: SourceMetadata,
    budgets: ResourceBudget,
    diagnostic_level: DiagnosticLevel,
) -> SchedulerInputs:
    """Choose a bounded chunk size from the measured machine budget instead of assuming the whole clip fits in RAM."""

    import psutil

    precision_bytes = 4
    admissible_ram_bytes = max(
        budgets.available_ram_bytes - budgets.reserved_ui_headroom_bytes,
        64 * 1024 * 1024,
    )
    cpu_count = psutil.cpu_count(logical=True) or 4
    available_parallel_cores = max(cpu_count - 1, 1)

    if intent.resource_policy == ResourcePolicy.AGGRESSIVE:
        compute_worker_count = max(2, min(8, available_parallel_cores))
        warp_worker_count = compute_worker_count
        # Motion-grid helper fan-out is intentionally held at one worker for
        # now. The warp stage delivered most of the safe speedup, while motion
        # estimation is the newest numeric helper path and should stay
        # conservative until it has more real-clip validation.
        motion_worker_count = 1
        internal_queue_depth = 3
        analyzer_mode = AnalyzerExecutionMode.BACKGROUND_THREAD
        analysis_queue_depth = 2
        chunk_cap_frames = min(
            256,
            max(96, int(admissible_ram_bytes // (128 * 1024 * 1024))),
        )
        chunk_target_ram_fraction = 0.65
        native_buffer_multiplier = 3.0
    elif intent.resource_policy == ResourcePolicy.BALANCED:
        compute_worker_count = max(2, min(4, max(available_parallel_cores // 2, 1)))
        warp_worker_count = compute_worker_count
        motion_worker_count = 1
        internal_queue_depth = 2
        analyzer_mode = AnalyzerExecutionMode.BACKGROUND_THREAD
        analysis_queue_depth = 1
        chunk_cap_frames = 48
        chunk_target_ram_fraction = 0.5
        native_buffer_multiplier = 2.5
    else:
        compute_worker_count = 1
        warp_worker_count = 1
        motion_worker_count = 1
        internal_queue_depth = 1
        analyzer_mode = AnalyzerExecutionMode.BACKGROUND_THREAD
        analysis_queue_depth = 1
        chunk_cap_frames = 24
        chunk_target_ram_fraction = 0.35
        native_buffer_multiplier = 2.0

    thread_limit = max(
        compute_worker_count,
        warp_worker_count,
        motion_worker_count,
    )
    probe_inputs = PreflightInputs(
        intent=intent,
        source=source,
        budgets=budgets,
        scheduler=SchedulerInputs(
            chunk_frames=1,
            chunk_cap_frames=chunk_cap_frames,
            chunk_target_ram_fraction=chunk_target_ram_fraction,
            thread_limit=thread_limit,
            precision_bytes=precision_bytes,
            native_buffer_multiplier=native_buffer_multiplier,
            internal_queue_depth=internal_queue_depth,
            compute_worker_count=compute_worker_count,
            warp_worker_count=warp_worker_count,
            motion_worker_count=motion_worker_count,
            analyzer_mode=analyzer_mode,
            analysis_queue_depth=analysis_queue_depth,
        ),
        diagnostic_level=diagnostic_level,
    )
    fixed_overhead_bytes = estimate_streaming_fixed_ram_bytes(probe_inputs)
    per_frame_bytes = estimate_streaming_per_frame_ram_bytes(probe_inputs)
    chunk_budget_bytes = max(
        int(admissible_ram_bytes * chunk_target_ram_fraction) - fixed_overhead_bytes,
        per_frame_bytes,
    )
    chunk_frames = max(1, chunk_budget_bytes // max(per_frame_bytes, 1))
    chunk_frames = min(
        chunk_frames,
        max(source.frame_count, 1),
        chunk_cap_frames,
    )
    chunk_frames = _cap_chunk_frames_for_available_device_memory(
        requested=intent.hardware_acceleration_enabled,
        inputs=probe_inputs,
        chunk_frames=max(1, int(chunk_frames)),
    )
    chunk_frames = _cap_chunk_frames_for_analysis_resolution(
        inputs=probe_inputs,
        chunk_frames=max(1, int(chunk_frames)),
    )
    return SchedulerInputs(
        chunk_frames=max(1, int(chunk_frames)),
        chunk_cap_frames=chunk_cap_frames,
        chunk_target_ram_fraction=chunk_target_ram_fraction,
        thread_limit=thread_limit,
        precision_bytes=precision_bytes,
        native_buffer_multiplier=native_buffer_multiplier,
        internal_queue_depth=internal_queue_depth,
        compute_worker_count=compute_worker_count,
        warp_worker_count=warp_worker_count,
        motion_worker_count=motion_worker_count,
        analyzer_mode=analyzer_mode,
        analysis_queue_depth=analysis_queue_depth,
    )


def _cap_chunk_frames_for_available_device_memory(
    *,
    requested: bool,
    inputs: PreflightInputs,
    chunk_frames: int,
) -> int:
    """Clamp GPU chunk sizing against free device memory so CPU-oriented RAM plans do not overcommit the optional CuPy path."""

    if not requested:
        return chunk_frames
    capability = detect_acceleration_capability()
    acceleration = resolve_acceleration_request(True, capability=capability)
    if not acceleration.active or capability.device_free_bytes is None:
        return chunk_frames
    device_total_bytes = (
        capability.device_total_bytes
        if capability.device_total_bytes is not None
        else capability.device_free_bytes
    )
    reserved_device_headroom_bytes = max(
        512 * 1024 * 1024,
        int(device_total_bytes * 0.15),
    )
    admissible_device_bytes = max(
        capability.device_free_bytes - reserved_device_headroom_bytes,
        256 * 1024 * 1024,
    )
    conservative_gpu_fraction = min(inputs.scheduler.chunk_target_ram_fraction, 0.40)
    fixed_overhead_bytes = estimate_streaming_fixed_ram_bytes(inputs)
    per_frame_bytes = estimate_streaming_per_frame_ram_bytes(inputs)
    chunk_budget_bytes = max(
        int(admissible_device_bytes * conservative_gpu_fraction) - fixed_overhead_bytes,
        per_frame_bytes,
    )
    device_chunk_frames = max(1, chunk_budget_bytes // max(per_frame_bytes, 1))
    return min(
        chunk_frames,
        max(1, int(device_chunk_frames)),
        max(inputs.source.frame_count, 1),
        inputs.scheduler.chunk_cap_frames,
    )


def _analysis_resolution_for_inputs(inputs: PreflightInputs) -> Resolution:
    if not inputs.intent.analysis.enabled:
        return inputs.intent.processing_resolution
    return choose_quantitative_analysis_resolution(
        source_resolution=Resolution(
            width=inputs.source.width,
            height=inputs.source.height,
        ),
        processing_resolution=inputs.intent.processing_resolution,
    )


def _cap_chunk_frames_for_analysis_resolution(
    *,
    inputs: PreflightInputs,
    chunk_frames: int,
) -> int:
    """Clamp render chunks when analysis runs at a richer resolution so reference and analysis batches do not grow beyond a safe size."""

    analysis_resolution = _analysis_resolution_for_inputs(inputs)
    analysis_pixels = analysis_resolution.width * analysis_resolution.height
    processing_pixels = (
        inputs.intent.processing_resolution.width
        * inputs.intent.processing_resolution.height
    )
    if analysis_pixels <= processing_pixels:
        return chunk_frames
    per_frame_analysis_bytes = (
        analysis_pixels * 3 * inputs.scheduler.precision_bytes
    )
    analysis_chunk_cap_bytes = 192 * 1024 * 1024
    analysis_chunk_frames = max(
        1,
        analysis_chunk_cap_bytes // max(per_frame_analysis_bytes, 1),
    )
    return min(
        chunk_frames,
        max(1, int(analysis_chunk_frames)),
        max(inputs.source.frame_count, 1),
        inputs.scheduler.chunk_cap_frames,
    )


def estimate_output_staging_bytes(intent: JobIntent, source: SourceMetadata) -> int:
    """Estimate the encoded staging footprint conservatively so finalization has room on the output volume."""

    output_pixels = intent.output_resolution.width * intent.output_resolution.height
    # This is not a codec predictor. It is a conservative staging allowance so
    # finalization does not begin on a nearly full target volume.
    return max(output_pixels * max(source.frame_count, 1), 64 * 1024 * 1024)


def estimate_active_scratch_bytes(inputs: PreflightInputs) -> int:
    """Estimate the active scratch requirement with safety bias rather than speed bias."""

    proc_pixels = (
        inputs.intent.processing_resolution.width * inputs.intent.processing_resolution.height
    )
    per_frame_rgb_bytes = proc_pixels * 3 * inputs.scheduler.precision_bytes
    chunk_bytes = per_frame_rgb_bytes * inputs.scheduler.chunk_frames
    diagnostics_bytes = _estimate_diagnostics_bytes(inputs.diagnostic_level)
    mask_bytes = proc_pixels
    output_staging = estimate_output_staging_bytes(inputs.intent, inputs.source)
    normalized_source_bytes = 0
    if (
        inputs.source.requires_cfr_normalization
        or inputs.source.requires_square_pixel_normalization
    ):
        # The normalized working source is staged in scratch before decode when
        # cadence or pixel geometry has to be regularized. This estimate is
        # deliberately conservative so scratch admission does not silently
        # ignore that derived evidence file.
        normalized_source_bytes = (
            max(inputs.source.frame_count, 1)
            * max(inputs.source.width, 1)
            * max(inputs.source.height, 1)
            * 3
        )
    return int(
        chunk_bytes * inputs.scheduler.native_buffer_multiplier
        + diagnostics_bytes
        + mask_bytes
        + output_staging
        + normalized_source_bytes
    )


def estimate_streaming_per_frame_ram_bytes(inputs: PreflightInputs) -> int:
    """Estimate the per-frame RAM footprint of the bounded streaming render path."""

    source_pixels = max(inputs.source.width * inputs.source.height, 1)
    proc_pixels = max(
        inputs.intent.processing_resolution.width * inputs.intent.processing_resolution.height,
        1,
    )
    output_pixels = max(
        inputs.intent.output_resolution.width * inputs.intent.output_resolution.height,
        1,
    )
    precision_bytes = inputs.scheduler.precision_bytes
    raw_decode_bytes = source_pixels * 3
    source_rgb_bytes = source_pixels * 3 * precision_bytes
    processing_rgb_bytes = proc_pixels * 3 * precision_bytes
    output_rgb_bytes = output_pixels * 3 * precision_bytes
    luma_bytes = proc_pixels * precision_bytes
    motion_grid_bytes = max(proc_pixels // 8, 1) * precision_bytes
    return int(
        raw_decode_bytes
        + source_rgb_bytes
        + (processing_rgb_bytes * 2)
        + (output_rgb_bytes * 3)
        + (luma_bytes * 2)
        + motion_grid_bytes
    )


def estimate_streaming_fixed_ram_bytes(inputs: PreflightInputs) -> int:
    """Estimate the fixed non-batch RAM overhead of the bounded streaming render path."""

    proc_pixels = max(
        inputs.intent.processing_resolution.width * inputs.intent.processing_resolution.height,
        1,
    )
    output_pixels = max(
        inputs.intent.output_resolution.width * inputs.intent.output_resolution.height,
        1,
    )
    diagnostics_bytes = _estimate_diagnostics_bytes(inputs.diagnostic_level)
    reference_bytes = proc_pixels * inputs.scheduler.precision_bytes
    mask_bytes = output_pixels * 4
    filter_state_bytes = max(proc_pixels // 4, 1) * inputs.scheduler.precision_bytes
    python_headroom_bytes = 96 * 1024 * 1024
    return int(
        diagnostics_bytes
        + reference_bytes
        + mask_bytes
        + filter_state_bytes
        + python_headroom_bytes
    )


def estimate_ram_required_bytes(inputs: PreflightInputs) -> int:
    """Estimate RAM admission conservatively for the bounded batch pipeline."""

    effective_chunk_frames = max(
        1,
        min(inputs.scheduler.chunk_frames, max(inputs.source.frame_count, 1)),
    )
    per_frame_bytes = estimate_streaming_per_frame_ram_bytes(inputs)
    fixed_overhead_bytes = estimate_streaming_fixed_ram_bytes(inputs)
    thread_overhead = per_frame_bytes * max(inputs.scheduler.thread_limit - 1, 0)
    return int(
        fixed_overhead_bytes
        + (per_frame_bytes * effective_chunk_frames)
        + thread_overhead
    )


def _issue(severity: PreflightSeverity, code: str, message: str) -> PreflightIssue:
    return PreflightIssue(severity=severity, code=code, message=message)


def run_preflight(inputs: PreflightInputs) -> PreflightReport:
    """Run the mandatory pre-flight checks required by the design document."""

    issues: list[PreflightIssue] = []
    nyquist_limit_hz = inputs.source.fps / 2.0
    acceleration = resolve_acceleration_request(
        inputs.intent.hardware_acceleration_enabled
    )

    if inputs.intent.output_container.lower() != "mp4":
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "output_container_invalid",
                "Output container must be MP4 in v1.",
            )
        )

    if inputs.intent.output_resolution != inputs.intent.processing_resolution:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "output_resolution_mismatch",
                "Processing and output resolution must match for the current render pipeline.",
            )
        )

    if (
        inputs.intent.output_resolution.width % 2 != 0
        or inputs.intent.output_resolution.height % 2 != 0
    ):
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "output_resolution_even_required",
                "The current MP4 encoder path requires even output width and height.",
            )
        )

    if inputs.intent.phase.high_hz <= inputs.intent.phase.low_hz:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "invalid_frequency_band",
                "High cutoff must be greater than low cutoff.",
            )
        )

    if inputs.intent.phase.high_hz >= nyquist_limit_hz:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "frequency_above_nyquist",
                "Selected high cutoff is at or above the Nyquist limit for this source.",
            )
        )
    elif inputs.intent.phase.high_hz >= nyquist_limit_hz * 0.9:
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "frequency_near_nyquist",
                "Selected high cutoff is close to the Nyquist limit and may be poorly supported.",
            )
        )

    # The design doc asks for a warning when the clip is too short for the low
    # cutoff to be meaningful, but it does not define the exact threshold. The
    # small safe choice here is two cycles of the low-frequency component.
    if inputs.source.duration_seconds * inputs.intent.phase.low_hz < 2.0:
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "low_frequency_under_supported_duration",
                "Clip duration is short relative to the selected low-frequency cutoff.",
            )
        )

    if (
        inputs.intent.phase.high_hz - inputs.intent.phase.low_hz
        < max(0.5, inputs.intent.phase.high_hz * 0.1)
    ):
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "frequency_band_narrow",
                "Selected frequency band is narrow and may not be well supported by the recording.",
            )
        )

    if inputs.source.requires_cfr_normalization:
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "variable_frame_rate_normalized",
                "Variable frame rate input will be normalized automatically to a CFR working stream before processing.",
            )
        )

    if inputs.source.has_variable_display_transform:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "display_transform_unsupported",
                "Variable display transforms are not supported in v1.",
            )
        )

    if inputs.source.has_unsupported_rotation:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "rotation_unsupported",
                "Unsupported rotation or orientation metadata must be normalized before processing.",
            )
        )

    if inputs.source.requires_square_pixel_normalization:
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "pixel_aspect_ratio_normalized",
                "Non-square pixel geometry will be normalized automatically to square-pixel working frames before processing.",
            )
        )

    if inputs.source.has_hdr_markers:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "hdr_unsupported",
                "HDR or wide-gamut input is outside the supported SDR workflow.",
            )
        )

    if inputs.source.contradictory_color_metadata:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "color_metadata_contradictory",
                "Input color metadata is contradictory and cannot be mapped safely.",
            )
        )

    if not inputs.source.decoded_format_supported:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "decoded_format_unsupported",
                "Decoded source format is not compatible with the supported SDR workflow.",
            )
        )

    if not inputs.source.explicit_rec709_compatible:
        if inputs.source.heuristic_sdr_allowed:
            issues.append(
                _issue(
                    PreflightSeverity.WARNING,
                    "heuristic_sdr_acceptance",
                    "Input was accepted through the heuristic SDR path and that assumption must be recorded.",
                )
            )
        else:
            issues.append(
                _issue(
                    PreflightSeverity.BLOCKER,
                    "color_workflow_unsupported",
                    "Input color interpretation is too ambiguous for the supported SDR policy.",
                )
            )

    if inputs.source.bit_depth <= 8:
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "eight_bit_input_warning",
                "8-bit input may show banding or quantization artifacts at higher magnification.",
            )
        )

    if inputs.intent.hardware_acceleration_enabled and not acceleration.active:
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "hardware_acceleration_fallback",
                acceleration.detail,
            )
        )

    active_scratch_required = estimate_active_scratch_bytes(inputs)
    output_staging_required = estimate_output_staging_bytes(inputs.intent, inputs.source)
    ram_required = estimate_ram_required_bytes(inputs)

    scratch_remaining = (
        inputs.budgets.available_scratch_bytes - active_scratch_required
    )
    if scratch_remaining < inputs.budgets.scratch_floor_bytes:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "active_scratch_reservation_failed",
                "Active scratch space cannot be reserved while preserving the required free-space floor.",
            )
        )

    if inputs.budgets.available_output_volume_bytes < output_staging_required:
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "output_volume_staging_insufficient",
                "The output volume does not have enough space for staged finalization.",
            )
        )

    retained_run_allowance = _estimate_diagnostics_bytes(inputs.diagnostic_level)
    if (
        inputs.budgets.retained_evidence_bytes + retained_run_allowance
        > inputs.budgets.retention_budget_bytes
    ):
        issues.append(
            _issue(
                PreflightSeverity.WARNING,
                "retention_budget_likely_exceeded",
                "Retained diagnostics or failed-run evidence may exceed the configured retention budget.",
            )
        )

    if (
        inputs.budgets.available_ram_bytes - inputs.budgets.reserved_ui_headroom_bytes
        < ram_required
    ):
        issues.append(
            _issue(
                PreflightSeverity.BLOCKER,
                "ram_admission_failed",
                "No safe RAM admission plan is available for the requested processing settings.",
            )
        )

    return PreflightReport(
        issues=tuple(issues),
        nyquist_limit_hz=nyquist_limit_hz,
        active_scratch_required_bytes=active_scratch_required,
        ram_required_bytes=ram_required,
        output_staging_required_bytes=output_staging_required,
        hardware_acceleration_requested=inputs.intent.hardware_acceleration_enabled,
        hardware_acceleration_active=acceleration.active,
        acceleration_backend=acceleration.backend_name,
        acceleration_status=acceleration.detail,
    )

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
