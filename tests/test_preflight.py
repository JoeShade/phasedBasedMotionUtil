"""This file tests the mandatory pre-flight gates so invalid frequency choices, storage shortages, and unsupported sources are blocked before render."""

from __future__ import annotations

import phase_motion_app.core.preflight as preflight_module

from phase_motion_app.core.acceleration import AccelerationCapability, AccelerationDecision
from phase_motion_app.core.models import (
    AnalysisSettings,
    JobIntent,
    PhaseSettings,
    Resolution,
    ResourcePolicy,
)
from phase_motion_app.core.preflight import (
    AnalyzerExecutionMode,
    DiagnosticLevel,
    PreflightInputs,
    ResourceBudget,
    SchedulerInputs,
    SourceMetadata,
    choose_scheduler_inputs,
    run_preflight,
)


def _intent() -> JobIntent:
    return JobIntent(
        phase=PhaseSettings(
            magnification=15.0,
            low_hz=5.0,
            high_hz=12.0,
            pyramid_type="complex_steerable",
            sigma=1.0,
            attenuate_other_frequencies=True,
        ),
        processing_resolution=Resolution(width=640, height=360),
        output_resolution=Resolution(width=640, height=360),
        resource_policy=ResourcePolicy.BALANCED,
    )


def _source(**overrides: object) -> SourceMetadata:
    base = SourceMetadata(
        fps=60.0,
        duration_seconds=5.0,
        frame_count=300,
        width=1920,
        height=1080,
        is_cfr=True,
        bit_depth=10,
    )
    return SourceMetadata(**{**base.__dict__, **overrides})


def _budgets(**overrides: object) -> ResourceBudget:
    base = ResourceBudget(
        available_scratch_bytes=10 * 1024 * 1024 * 1024,
        scratch_floor_bytes=512 * 1024 * 1024,
        available_output_volume_bytes=10 * 1024 * 1024 * 1024,
        available_ram_bytes=16 * 1024 * 1024 * 1024,
        reserved_ui_headroom_bytes=512 * 1024 * 1024,
        retention_budget_bytes=50 * 1024 * 1024 * 1024,
        retained_evidence_bytes=10 * 1024 * 1024 * 1024,
    )
    return ResourceBudget(**{**base.__dict__, **overrides})


def _inputs(**overrides: object) -> PreflightInputs:
    base = PreflightInputs(
        intent=_intent(),
        source=_source(),
        budgets=_budgets(),
        scheduler=SchedulerInputs(chunk_frames=16, thread_limit=4),
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    return PreflightInputs(**{**base.__dict__, **overrides})


def test_preflight_accepts_safe_configuration() -> None:
    report = run_preflight(_inputs())
    assert report.can_render is True
    assert report.blockers == ()


def test_preflight_blocks_invalid_frequency_band() -> None:
    intent = _intent()
    intent = JobIntent(
        phase=PhaseSettings(
            magnification=intent.phase.magnification,
            low_hz=10.0,
            high_hz=10.0,
            pyramid_type=intent.phase.pyramid_type,
            sigma=intent.phase.sigma,
            attenuate_other_frequencies=intent.phase.attenuate_other_frequencies,
        ),
        processing_resolution=intent.processing_resolution,
        output_resolution=intent.output_resolution,
        resource_policy=intent.resource_policy,
    )

    report = run_preflight(_inputs(intent=intent))

    assert any(issue.code == "invalid_frequency_band" for issue in report.blockers)


def test_preflight_warns_when_vfr_input_will_be_normalized_to_cfr() -> None:
    report = run_preflight(
        _inputs(
            source=_source(
                source_is_cfr=False,
                requires_cfr_normalization=True,
            )
        )
    )

    assert report.can_render is True
    assert any(issue.code == "variable_frame_rate_normalized" for issue in report.warnings)


def test_preflight_blocks_unsupported_rotation_metadata() -> None:
    report = run_preflight(_inputs(source=_source(has_unsupported_rotation=True)))

    assert any(issue.code == "rotation_unsupported" for issue in report.blockers)


def test_preflight_warns_when_non_square_pixel_aspect_ratio_will_be_normalized() -> None:
    report = run_preflight(
        _inputs(
            source=_source(
                source_pixel_aspect_ratio=4.0 / 3.0,
                requires_square_pixel_normalization=True,
            )
        )
    )

    assert report.can_render is True
    assert any(issue.code == "pixel_aspect_ratio_normalized" for issue in report.warnings)


def test_preflight_warns_when_high_cutoff_is_near_nyquist() -> None:
    intent = _intent()
    intent = JobIntent(
        phase=PhaseSettings(
            magnification=intent.phase.magnification,
            low_hz=intent.phase.low_hz,
            high_hz=28.0,
            pyramid_type=intent.phase.pyramid_type,
            sigma=intent.phase.sigma,
            attenuate_other_frequencies=intent.phase.attenuate_other_frequencies,
        ),
        processing_resolution=intent.processing_resolution,
        output_resolution=intent.output_resolution,
        resource_policy=intent.resource_policy,
    )

    report = run_preflight(_inputs(intent=intent))

    assert report.can_render is True
    assert any(issue.code == "frequency_near_nyquist" for issue in report.warnings)


def test_preflight_blocks_when_high_cutoff_reaches_nyquist() -> None:
    intent = _intent()
    intent = JobIntent(
        phase=PhaseSettings(
            magnification=intent.phase.magnification,
            low_hz=intent.phase.low_hz,
            high_hz=30.0,
            pyramid_type=intent.phase.pyramid_type,
            sigma=intent.phase.sigma,
            attenuate_other_frequencies=intent.phase.attenuate_other_frequencies,
        ),
        processing_resolution=intent.processing_resolution,
        output_resolution=intent.output_resolution,
        resource_policy=intent.resource_policy,
    )

    report = run_preflight(_inputs(intent=intent))

    assert any(issue.code == "frequency_above_nyquist" for issue in report.blockers)


def test_preflight_keeps_retention_budget_separate_from_active_scratch_gate() -> None:
    budgets = _budgets(
        retention_budget_bytes=2 * 1024 * 1024 * 1024,
        retained_evidence_bytes=2050 * 1024 * 1024,
    )

    report = run_preflight(_inputs(budgets=budgets))

    assert report.can_render is True
    assert any(issue.code == "retention_budget_likely_exceeded" for issue in report.warnings)
    assert all(issue.code != "active_scratch_reservation_failed" for issue in report.blockers)


def test_preflight_does_not_treat_output_staging_as_retention_evidence() -> None:
    budgets = _budgets(
        retention_budget_bytes=(10 * 1024 * 1024 * 1024) + (40 * 1024 * 1024),
        retained_evidence_bytes=10 * 1024 * 1024 * 1024,
    )

    report = run_preflight(_inputs(budgets=budgets))

    assert all(
        issue.code != "retention_budget_likely_exceeded" for issue in report.warnings
    )


def test_preflight_blocks_when_active_scratch_cannot_be_reserved() -> None:
    budgets = _budgets(
        available_scratch_bytes=800 * 1024 * 1024,
        scratch_floor_bytes=512 * 1024 * 1024,
    )

    report = run_preflight(_inputs(budgets=budgets))

    assert any(issue.code == "active_scratch_reservation_failed" for issue in report.blockers)


def test_preflight_blocks_processing_output_resolution_mismatch() -> None:
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=640, height=360),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
    )

    report = run_preflight(_inputs(intent=intent))

    assert any(issue.code == "output_resolution_mismatch" for issue in report.blockers)


def test_preflight_blocks_odd_output_resolution_for_current_encoder_path() -> None:
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=853, height=480),
        output_resolution=Resolution(width=853, height=480),
        resource_policy=ResourcePolicy.BALANCED,
    )

    report = run_preflight(_inputs(intent=intent))

    assert any(
        issue.code == "output_resolution_even_required" for issue in report.blockers
    )


def test_preflight_warns_when_hardware_acceleration_is_requested_but_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        preflight_module,
        "resolve_acceleration_request",
        lambda _requested: AccelerationDecision(
            requested=True,
            active=False,
            status="gpu_requested_cpu_fallback",
            detail=(
                "Hardware acceleration was requested, but the optional CuPy backend "
                "is not installed. CPU fallback will be used."
            ),
            backend_name="cupy",
        ),
    )
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=640, height=360),
        output_resolution=Resolution(width=640, height=360),
        resource_policy=ResourcePolicy.BALANCED,
        hardware_acceleration_enabled=True,
    )

    report = run_preflight(_inputs(intent=intent))

    assert report.can_render is True
    assert report.hardware_acceleration_requested is True
    assert report.hardware_acceleration_active is False
    assert report.acceleration_backend == "cupy"
    assert "CPU fallback" in (report.acceleration_status or "")
    assert any(
        issue.code == "hardware_acceleration_fallback" for issue in report.warnings
    )


def test_preflight_reports_gpu_status_when_hardware_acceleration_is_available(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        preflight_module,
        "resolve_acceleration_request",
        lambda _requested: AccelerationDecision(
            requested=True,
            active=True,
            status="gpu_active",
            detail="Hardware acceleration is enabled. Using CuPy on Example GPU.",
            backend_name="cupy",
            device_name="Example GPU",
        ),
    )
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=640, height=360),
        output_resolution=Resolution(width=640, height=360),
        resource_policy=ResourcePolicy.BALANCED,
        hardware_acceleration_enabled=True,
    )

    report = run_preflight(_inputs(intent=intent))

    assert report.can_render is True
    assert report.hardware_acceleration_requested is True
    assert report.hardware_acceleration_active is True
    assert report.acceleration_backend == "cupy"
    assert report.acceleration_status == (
        "Hardware acceleration is enabled. Using CuPy on Example GPU."
    )
    assert all(
        issue.code != "hardware_acceleration_fallback" for issue in report.warnings
    )


def test_preflight_blocks_when_output_volume_cannot_stage_finalization() -> None:
    budgets = _budgets(available_output_volume_bytes=32 * 1024 * 1024)

    report = run_preflight(_inputs(budgets=budgets))

    assert any(
        issue.code == "output_volume_staging_insufficient"
        for issue in report.blockers
    )


def test_preflight_blocks_when_ram_admission_fails() -> None:
    budgets = _budgets(
        available_ram_bytes=128 * 1024 * 1024,
        reserved_ui_headroom_bytes=64 * 1024 * 1024,
    )

    report = run_preflight(_inputs(budgets=budgets))

    assert any(issue.code == "ram_admission_failed" for issue in report.blockers)


def test_preflight_accepts_long_clip_when_chunk_plan_fits_ram_budget() -> None:
    source = _source(
        duration_seconds=30.0,
        frame_count=900,
        width=1280,
        height=720,
    )
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
    )
    budgets = _budgets(
        available_ram_bytes=24 * 1024 * 1024 * 1024,
        reserved_ui_headroom_bytes=512 * 1024 * 1024,
    )

    scheduler = choose_scheduler_inputs(
        intent=intent,
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    report = run_preflight(
        _inputs(
            source=source,
            intent=intent,
            budgets=budgets,
            scheduler=scheduler,
        )
    )

    assert report.can_render is True
    assert scheduler.chunk_frames < source.frame_count
    assert all(issue.code != "ram_admission_failed" for issue in report.blockers)


def test_choose_scheduler_inputs_shrinks_chunk_frames_when_ram_is_tighter() -> None:
    source = _source(frame_count=600, width=1280, height=720)
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
    )
    generous = choose_scheduler_inputs(
        intent=intent,
        source=source,
        budgets=_budgets(available_ram_bytes=16 * 1024 * 1024 * 1024),
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    tight = choose_scheduler_inputs(
        intent=intent,
        source=source,
        budgets=_budgets(available_ram_bytes=2 * 1024 * 1024 * 1024),
        diagnostic_level=DiagnosticLevel.BASIC,
    )

    assert tight.chunk_frames >= 1
    assert generous.chunk_frames >= tight.chunk_frames


def test_choose_scheduler_inputs_turns_resource_policies_into_distinct_runtime_plans() -> None:
    source = _source(frame_count=600, width=1280, height=720)
    conservative_intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.CONSERVATIVE,
    )
    balanced_intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.BALANCED,
    )
    aggressive_intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=1280, height=720),
        output_resolution=Resolution(width=1280, height=720),
        resource_policy=ResourcePolicy.AGGRESSIVE,
    )

    conservative = choose_scheduler_inputs(
        intent=conservative_intent,
        source=source,
        budgets=_budgets(),
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    balanced = choose_scheduler_inputs(
        intent=balanced_intent,
        source=source,
        budgets=_budgets(),
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    aggressive = choose_scheduler_inputs(
        intent=aggressive_intent,
        source=source,
        budgets=_budgets(),
        diagnostic_level=DiagnosticLevel.BASIC,
    )

    assert conservative.internal_queue_depth == 1
    assert balanced.internal_queue_depth == 2
    assert aggressive.internal_queue_depth == 3
    assert conservative.chunk_cap_frames < balanced.chunk_cap_frames <= aggressive.chunk_cap_frames
    assert conservative.compute_worker_count <= balanced.compute_worker_count <= aggressive.compute_worker_count
    assert conservative.warp_worker_count <= balanced.warp_worker_count <= aggressive.warp_worker_count
    assert conservative.motion_worker_count == 1
    assert balanced.motion_worker_count == 1
    assert aggressive.motion_worker_count == 1
    assert conservative.analyzer_mode is AnalyzerExecutionMode.BACKGROUND_THREAD
    assert balanced.analyzer_mode is AnalyzerExecutionMode.BACKGROUND_THREAD
    assert aggressive.analyzer_mode is AnalyzerExecutionMode.BACKGROUND_THREAD
    assert conservative.analysis_queue_depth <= balanced.analysis_queue_depth <= aggressive.analysis_queue_depth


def test_choose_scheduler_inputs_aggressive_policy_materially_increases_parallelism() -> None:
    source = _source(frame_count=900, width=1920, height=1080)
    budgets = _budgets(available_ram_bytes=24 * 1024 * 1024 * 1024)
    conservative = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=1920, height=1080),
            output_resolution=Resolution(width=1920, height=1080),
            resource_policy=ResourcePolicy.CONSERVATIVE,
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    aggressive = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=1920, height=1080),
            output_resolution=Resolution(width=1920, height=1080),
            resource_policy=ResourcePolicy.AGGRESSIVE,
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )

    assert aggressive.compute_worker_count > conservative.compute_worker_count
    assert aggressive.warp_worker_count > conservative.warp_worker_count
    assert aggressive.internal_queue_depth > conservative.internal_queue_depth
    assert aggressive.chunk_cap_frames > conservative.chunk_cap_frames


def test_choose_scheduler_inputs_keeps_motion_estimation_serial_pending_validation() -> None:
    source = _source(frame_count=900, width=1920, height=1080)
    budgets = _budgets(available_ram_bytes=24 * 1024 * 1024 * 1024)

    conservative = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=1920, height=1080),
            output_resolution=Resolution(width=1920, height=1080),
            resource_policy=ResourcePolicy.CONSERVATIVE,
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    aggressive = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=1920, height=1080),
            output_resolution=Resolution(width=1920, height=1080),
            resource_policy=ResourcePolicy.AGGRESSIVE,
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )

    assert conservative.motion_worker_count == 1
    assert aggressive.motion_worker_count == 1


def test_choose_scheduler_inputs_caps_gpu_chunk_frames_against_free_device_memory(
    monkeypatch,
) -> None:
    source = _source(frame_count=900, width=1920, height=1080)
    budgets = _budgets(available_ram_bytes=24 * 1024 * 1024 * 1024)
    intent = JobIntent(
        phase=_intent().phase,
        processing_resolution=Resolution(width=1920, height=1080),
        output_resolution=Resolution(width=1920, height=1080),
        resource_policy=ResourcePolicy.AGGRESSIVE,
        hardware_acceleration_enabled=True,
    )
    monkeypatch.setattr(
        preflight_module,
        "detect_acceleration_capability",
        lambda: AccelerationCapability(
            backend_name="cupy",
            importable=True,
            usable=True,
            status="available",
            detail="CuPy and a compatible CUDA device are available.",
            installed_version="13.6.0",
            device_name="Example GPU",
            device_total_bytes=8 * 1024 * 1024 * 1024,
            device_free_bytes=6 * 1024 * 1024 * 1024,
        ),
    )

    gpu_scheduler = choose_scheduler_inputs(
        intent=intent,
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )
    cpu_scheduler = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=1920, height=1080),
            output_resolution=Resolution(width=1920, height=1080),
            resource_policy=ResourcePolicy.AGGRESSIVE,
            hardware_acceleration_enabled=False,
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.BASIC,
    )

    assert gpu_scheduler.chunk_frames >= 1
    assert gpu_scheduler.chunk_frames < cpu_scheduler.chunk_frames


def test_choose_scheduler_inputs_caps_chunk_frames_when_analysis_uses_richer_resolution() -> None:
    source = _source(frame_count=900, width=1280, height=720)
    budgets = _budgets(available_ram_bytes=16 * 1024 * 1024 * 1024)
    analysis_enabled = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=640, height=360),
            output_resolution=Resolution(width=640, height=360),
            resource_policy=ResourcePolicy.AGGRESSIVE,
            analysis=AnalysisSettings(enabled=True),
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.TRACE,
    )
    analysis_disabled = choose_scheduler_inputs(
        intent=JobIntent(
            phase=_intent().phase,
            processing_resolution=Resolution(width=640, height=360),
            output_resolution=Resolution(width=640, height=360),
            resource_policy=ResourcePolicy.AGGRESSIVE,
            analysis=AnalysisSettings(enabled=False),
        ),
        source=source,
        budgets=budgets,
        diagnostic_level=DiagnosticLevel.TRACE,
    )

    assert analysis_enabled.chunk_frames >= 1
    assert analysis_enabled.chunk_frames < analysis_disabled.chunk_frames

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
