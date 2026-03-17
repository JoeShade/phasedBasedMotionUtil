"""This file owns the real render worker entrypoint so heavy decode, phase processing, encode, diagnostics, and finalization stay outside the PyQt process."""

from __future__ import annotations

import hashlib
import gc
import json
import os
import platform
import socket
import subprocess
import threading
import time
import traceback
from datetime import datetime, timezone
from multiprocessing.synchronize import Event as EventType
from pathlib import Path
from typing import Any

import psutil
import numpy as np

from phase_motion_app import __version__
from phase_motion_app.core.diagnostics_bundle import (
    DiagnosticsBundleInput,
    write_diagnostics_bundle,
)
from phase_motion_app.core.drift import build_drift_acknowledgement
from phase_motion_app.core.ffprobe import FfprobeRunner
from phase_motion_app.core.image_ops import resize_rgb_frames_bilinear
from phase_motion_app.core.ipc import (
    JsonLineConnection,
    SessionConfig,
    build_hello_ack,
    build_message,
)
from phase_motion_app.core.jsonl_log import JsonlLogger
from phase_motion_app.core.masking import rasterize_output_domain_mask, validate_exclusion_zones
from phase_motion_app.core.media_tools import (
    RawvideoDecodeProcess,
    RawvideoEncodeProcess,
    WorkingSourceTranscodeProcess,
)
from phase_motion_app.core.models import ObservedEnvironment, PreflightSummary, SourceRecord
from phase_motion_app.core.phase_engine import StreamingPhaseAmplifier
from phase_motion_app.core.quantitative_analysis import (
    StreamingQuantitativeAnalyzer,
    build_disabled_analysis_export,
    build_empty_analysis_export,
)
from phase_motion_app.core.preflight import (
    PreflightSeverity,
    ResourceBudget,
    SchedulerInputs,
    SourceMetadata,
    PreflightInputs,
    choose_scheduler_inputs,
    run_preflight,
)
from phase_motion_app.core.render_job import CodecPlan, RenderPaths, RenderRequest
from phase_motion_app.core.retention import measure_retained_roots_bytes
from phase_motion_app.core.sidecar import SCHEMA_VERSION
from phase_motion_app.core.source_normalization import (
    SourceNormalizationPlan,
    build_source_normalization_plan,
)
from phase_motion_app.core.storage import (
    FinalizationResult,
    OutputExpectation,
    StagedMp4Observation,
    finalize_output_pair,
    validate_staged_mp4,
)
from phase_motion_app.core.toolchain import resolve_toolchain
from phase_motion_app.worker.bootstrap import RenderWorkerConfig


class _ThreadSafeConnection:
    """This helper keeps heartbeat traffic and stage traffic from interleaving on the same socket."""

    def __init__(self, connection: JsonLineConnection) -> None:
        self.connection = connection
        self._lock = threading.Lock()

    def send(self, message: dict) -> None:
        with self._lock:
            self.connection.send(message)


class _SharedSequence:
    """This helper hands out strictly increasing sequence numbers across worker threads."""

    def __init__(self) -> None:
        self._value = 0
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            self._value += 1
            return self._value


def render_worker_process_main(config: RenderWorkerConfig, cancel_event: EventType) -> None:
    """Run one real render worker attempt using the documented socket protocol and out-of-process media toolchain."""

    request = config.request
    request.paths.output_directory.mkdir(parents=True, exist_ok=True)
    request.paths.scratch_directory.mkdir(parents=True, exist_ok=True)
    request.paths.diagnostics_directory.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(request.paths.jsonl_log_path)

    sock = socket.create_connection((config.host, config.port), timeout=5.0)
    connection = JsonLineConnection(sock)
    session = SessionConfig(
        session_token=config.session_token,
        job_id=config.job_id,
        role=config.role,
    )

    hello = connection.read(timeout_seconds=5.0)
    if (
        hello.get("message_type") != "hello"
        or hello.get("protocol_version") != session.protocol_version
        or hello.get("session_token") != config.session_token
        or hello.get("job_id") != config.job_id
        or hello.get("role") != config.role
    ):
        sock.close()
        raise SystemExit(2)

    safe_connection = _ThreadSafeConnection(connection)
    safe_connection.send(build_hello_ack(session, os.getpid()))

    stop_heartbeat = threading.Event()
    sequence = _SharedSequence()
    stage_timings: dict[str, float] = {}
    current_stage = "bootstrap"
    current_stage_started_at = time.monotonic()
    terminal_message_type: str | None = None
    last_emitted_message_type: str | None = None
    last_progress_token: int | None = None
    failure_classification: str | None = None
    failure_stage: str | None = None
    failure_detail: str | None = None
    failure_exception_type: str | None = None
    fingerprint: str | None = None
    probe = None
    working_probe = None
    source_metadata = None
    normalization_plan: SourceNormalizationPlan | None = None
    normalized_source_path: Path | None = None
    scheduler = None
    budgets = None
    preflight = None
    diagnostics_bundle_path = request.paths.diagnostics_directory / "diagnostics_bundle.json"
    analysis_export = build_disabled_analysis_export(request.intent.analysis)
    analysis_collection_warning: str | None = None

    def emit(message_type: str, payload: dict | None = None) -> None:
        nonlocal last_emitted_message_type, last_progress_token
        last_emitted_message_type = message_type
        if payload is not None and isinstance(payload.get("progress_token"), int):
            last_progress_token = payload["progress_token"]
        safe_connection.send(
            build_message(
                config=session,
                seq=sequence.next(),
                message_type=message_type,
                monotonic_time_ns=time.monotonic_ns(),
                payload=payload,
            )
        )

    def log(level: str, event_type: str, stage: str, message: str, payload: dict | None = None) -> None:
        logger.log(
            level=level,
            event_type=event_type,
            job_id=request.job_id,
            stage=stage,
            message=message,
            payload=payload,
        )

    def record_stage_transition(next_stage: str) -> None:
        nonlocal current_stage, current_stage_started_at
        now = time.monotonic()
        stage_timings[current_stage] = stage_timings.get(current_stage, 0.0) + max(
            now - current_stage_started_at,
            0.0,
        )
        current_stage = next_stage
        current_stage_started_at = now

    def note_terminal(
        message_type: str,
        *,
        classification: str | None = None,
        stage: str | None = None,
        detail: str | None = None,
        exception_type: str | None = None,
    ) -> None:
        nonlocal terminal_message_type, failure_classification, failure_stage
        nonlocal failure_detail, failure_exception_type
        terminal_message_type = message_type
        if classification is not None:
            failure_classification = classification
        if stage is not None:
            failure_stage = stage
        if detail is not None:
            failure_detail = detail
        if exception_type is not None:
            failure_exception_type = exception_type

    def emit_failure(
        classification: str,
        stage: str,
        *,
        detail: str | None = None,
        exception_type: str | None = None,
    ) -> None:
        note_terminal(
            "failure",
            classification=classification,
            stage=stage,
            detail=detail,
            exception_type=exception_type,
        )
        emit(
            "failure",
            {
                "classification": classification,
                "stage": stage,
                **({} if detail is None else {"detail": detail}),
                **({} if exception_type is None else {"exception_type": exception_type}),
            },
        )

    def heartbeat_loop() -> None:
        while not stop_heartbeat.is_set():
            emit("heartbeat", {})
            stop_heartbeat.wait(0.5)

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        name="render-worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        emit("job_started", {"role": config.role})
        log("info", "job_started", "bootstrap", "Render worker started.")

        if cancel_event.is_set():
            note_terminal("job_cancelled")
            emit("job_cancelled", {})
            raise SystemExit(0)

        record_stage_transition("preflight")
        emit("preflight_started", {})
        log("info", "preflight_started", "preflight", "Authoritative worker pre-flight started.")

        fingerprint = _sha256_file(request.paths.source_path)
        if fingerprint != request.expected_source_fingerprint_sha256:
            emit_failure("stale_source_detected", "preflight")
            raise SystemExit(1)

        probe = FfprobeRunner().run(request.paths.source_path)
        normalization_plan = build_source_normalization_plan(probe)
        source_metadata = _build_source_metadata(probe, normalization_plan)
        budgets = _build_resource_budget(
            request.paths,
            request.retention_budget_bytes,
        )
        scheduler = choose_scheduler_inputs(
            intent=request.intent,
            source=source_metadata,
            budgets=budgets,
            diagnostic_level=request.diagnostic_level,
        )
        preflight = run_preflight(
            PreflightInputs(
                intent=request.intent,
                source=source_metadata,
                budgets=budgets,
                scheduler=scheduler,
                diagnostic_level=request.diagnostic_level,
            )
        )
        emit(
            "preflight_report",
            {
                "warnings": [issue.message for issue in preflight.warnings],
                "blockers": [issue.message for issue in preflight.blockers],
                "source_fps": normalization_plan.native_fps,
                "working_fps": normalization_plan.working_fps,
                "source_is_cfr": probe.is_cfr,
                "normalization_steps": list(normalization_plan.normalization_steps),
                "working_source_width": normalization_plan.working_resolution.width,
                "working_source_height": normalization_plan.working_resolution.height,
                "nyquist_limit_hz": preflight.nyquist_limit_hz,
                "selected_low_hz": request.intent.phase.low_hz,
                "selected_high_hz": request.intent.phase.high_hz,
                "resource_policy": request.intent.resource_policy.value,
                "active_scratch_required_bytes": preflight.active_scratch_required_bytes,
                "ram_required_bytes": preflight.ram_required_bytes,
                "output_staging_required_bytes": preflight.output_staging_required_bytes,
            },
        )
        log(
            "info",
            "preflight_report",
            "preflight",
            "Worker pre-flight completed.",
            {
                "warnings": [issue.code for issue in preflight.warnings],
                "blockers": [issue.code for issue in preflight.blockers],
            },
        )
        if not preflight.can_render:
            emit_failure(preflight.blockers[0].code, "preflight")
            raise SystemExit(1)

        source_resolution = normalization_plan.working_resolution
        mask_issues = validate_exclusion_zones(request.intent.exclusion_zones, source_resolution)
        if mask_issues:
            emit_failure(mask_issues[0].code, "preflight")
            raise SystemExit(1)

        working_source_path = request.paths.source_path
        working_probe = probe
        working_plan = normalization_plan
        if normalization_plan.requires_working_source_staging:
            normalized_source_path = request.paths.scratch_directory / "working-source.mkv"
            record_stage_transition("normalize_source")
            emit(
                "stage_started",
                {
                    "stage": "normalize_source",
                    "total_frames": normalization_plan.working_frame_count,
                },
            )
            log(
                "info",
                "stage_started",
                "normalize_source",
                "Staging normalized working source.",
                {
                    "normalization_steps": list(normalization_plan.normalization_steps),
                    "working_path": str(normalized_source_path),
                },
            )
            normalizer = WorkingSourceTranscodeProcess(
                source_path=request.paths.source_path,
                output_path=normalized_source_path,
                normalization_plan=normalization_plan,
            )
            normalization_exit_code: int | None = None
            last_normalized_counter: int | None = None
            try:
                while normalizer.process.poll() is None:
                    if cancel_event.is_set():
                        note_terminal("job_cancelled")
                        emit("job_cancelled", {})
                        raise SystemExit(0)
                    counter = normalizer.take_progress_counter()
                    if counter is not None and counter != last_normalized_counter:
                        last_normalized_counter = counter
                        emit(
                            "progress_update",
                            {
                                "progress_token": counter,
                                "stage": "normalize_source",
                                "frames_completed": counter,
                                "total_frames": normalization_plan.working_frame_count,
                            },
                        )
                    time.sleep(0.05)
                normalization_exit_code = normalizer.finish()
            finally:
                if normalization_exit_code is None:
                    normalizer.cancel()
            if normalization_exit_code != 0 or not normalized_source_path.exists():
                emit_failure("source_normalization_failed", "normalize_source")
                raise SystemExit(1)
            working_source_path = normalized_source_path
            working_probe = FfprobeRunner().run(working_source_path)
            working_plan = build_source_normalization_plan(working_probe)
            if working_plan.requires_working_source_staging:
                emit_failure("source_normalization_failed", "normalize_source")
                raise SystemExit(1)

        record_stage_transition("decode")
        emit("stage_started", {"stage": "decode", "total_frames": working_probe.frame_count})
        log(
            "info",
            "stage_started",
            "decode",
            "Scanning the working clip to build the global phase reference.",
            {"chunk_frames": scheduler.chunk_frames},
        )
        reference_decoder = RawvideoDecodeProcess(
            source_path=working_source_path,
            output_resolution=request.intent.processing_resolution,
            normalization_plan=working_plan,
        )
        reference_exit_code: int | None = None
        reference_luma_sum = np.zeros(
            (
                request.intent.processing_resolution.height,
                request.intent.processing_resolution.width,
            ),
            dtype=np.float64,
        )
        reference_frame_count = 0
        try:
            while True:
                if cancel_event.is_set():
                    note_terminal("job_cancelled")
                    emit("job_cancelled", {})
                    raise SystemExit(0)
                chunk = reference_decoder.read_frames(scheduler.chunk_frames)
                if not chunk:
                    break
                processing_chunk = _bytes_to_float_frames(
                    chunk,
                    width=request.intent.processing_resolution.width,
                    height=request.intent.processing_resolution.height,
                )
                reference_luma_sum += _frames_to_luma(processing_chunk).sum(
                    axis=0,
                    dtype=np.float64,
                )
                reference_frame_count += processing_chunk.shape[0]
                child_counter = reference_decoder.take_progress_counter() or reference_frame_count
                emit(
                    "progress_update",
                    {
                        "progress_token": f"decode:{child_counter}",
                        "stage": "decode",
                        "frames_completed": reference_frame_count,
                        "total_frames": working_probe.frame_count,
                    },
                )
                del processing_chunk
            reference_exit_code = reference_decoder.close()
        finally:
            if reference_exit_code is None:
                reference_decoder.cancel()
        if reference_exit_code != 0 or reference_frame_count == 0:
            emit_failure("decode_failure", "decode")
            raise SystemExit(1)

        reference_luma = (
            reference_luma_sum / max(reference_frame_count, 1)
        ).astype(np.float32, copy=False)

        codec_plan = _choose_codec_plan()
        emit("warning", {"messages": list(codec_plan.warnings)})
        for warning in codec_plan.warnings:
            log("warning", "codec_warning", "encode", warning)

        mask = rasterize_output_domain_mask(
            zones=request.intent.exclusion_zones,
            source_resolution=source_resolution,
            output_resolution=request.intent.output_resolution,
            feather_px=request.intent.mask_feather_px,
        )
        mask_array = np.asarray(mask, dtype=np.float32)
        phase_amplifier = StreamingPhaseAmplifier(
            reference_luma=reference_luma,
            fps=working_plan.working_fps,
            low_hz=request.intent.phase.low_hz,
            high_hz=request.intent.phase.high_hz,
            magnification=request.intent.phase.magnification,
            sigma=request.intent.phase.sigma,
            attenuate_other_frequencies=request.intent.phase.attenuate_other_frequencies,
        )
        analysis_analyzer = None
        if request.intent.analysis.enabled:
            try:
                analysis_analyzer = StreamingQuantitativeAnalyzer(
                    settings=request.intent.analysis,
                    processing_resolution=request.intent.processing_resolution,
                    fps=working_plan.working_fps,
                    low_hz=request.intent.phase.low_hz,
                    high_hz=request.intent.phase.high_hz,
                    reference_luma=reference_luma,
                    exclusion_zones=request.intent.exclusion_zones,
                    drift_assessment=request.drift_assessment,
                )
            except Exception as exc:
                analysis_collection_warning = (
                    "Quantitative analysis could not be initialized and was disabled for "
                    f"this render: {exc}"
                )
                log(
                    "warning",
                    "analysis_initialization_failed",
                    "phase_processing",
                    analysis_collection_warning,
                )
                emit("warning", {"messages": [analysis_collection_warning]})

        record_stage_transition("phase_processing")
        emit(
            "stage_started",
            {"stage": "phase_processing", "total_frames": working_probe.frame_count},
        )
        log(
            "info",
            "stage_started",
            "phase_processing",
            "Processing bounded frame batches and streaming them to the encoder.",
            {"chunk_frames": scheduler.chunk_frames},
        )
        decoder = RawvideoDecodeProcess(
            source_path=working_source_path,
            output_resolution=source_resolution,
            normalization_plan=working_plan,
        )
        encoder = RawvideoEncodeProcess(
            staged_output_path=request.paths.staged_mp4_path,
            resolution=request.intent.output_resolution,
            fps=working_plan.working_fps,
            codec=codec_plan.ffmpeg_codec,
            pixel_format=codec_plan.pixel_format,
            color_tags=codec_plan.color_tags or _default_color_tags(),
        )
        decode_exit: int | None = None
        encoder_exit_code: int | None = None
        processed_frame_count = 0
        phase_progress_counter = 0
        try:
            while True:
                if cancel_event.is_set():
                    note_terminal("job_cancelled")
                    emit("job_cancelled", {})
                    raise SystemExit(0)
                chunk = decoder.read_frames(scheduler.chunk_frames)
                if not chunk:
                    break
                source_chunk = _bytes_to_float_frames(
                    chunk,
                    width=source_resolution.width,
                    height=source_resolution.height,
                )

                def report_phase_progress(detail: str) -> None:
                    nonlocal phase_progress_counter
                    phase_progress_counter += 1
                    emit(
                        "progress_update",
                        {
                            "progress_token": f"phase_processing:{processed_frame_count}:{phase_progress_counter}",
                            "stage": "phase_processing",
                            "detail": detail,
                        },
                    )

                processing_chunk = resize_rgb_frames_bilinear(
                    source_chunk,
                    request.intent.processing_resolution,
                )
                if analysis_analyzer is not None:
                    try:
                        analysis_analyzer.add_chunk(processing_chunk)
                    except Exception as exc:
                        analysis_collection_warning = (
                            "Quantitative analysis collection failed during phase processing "
                            f"and was disabled for the rest of the render: {exc}"
                        )
                        log(
                            "warning",
                            "analysis_collection_failed",
                            "phase_processing",
                            analysis_collection_warning,
                        )
                        emit("warning", {"messages": [analysis_collection_warning]})
                        analysis_analyzer = None
                amplified_processing = phase_amplifier.process_chunk(
                    processing_chunk,
                    progress_callback=report_phase_progress,
                )
                amplified_output = resize_rgb_frames_bilinear(
                    amplified_processing,
                    request.intent.output_resolution,
                )
                passthrough_output = resize_rgb_frames_bilinear(
                    source_chunk,
                    request.intent.output_resolution,
                )
                composed_output = passthrough_output * mask_array[None, :, :, None] + (
                    amplified_output * (1.0 - mask_array[None, :, :, None])
                )
                chunk_frame_count = source_chunk.shape[0]
                chunk_start_frame_count = processed_frame_count
                progress_emit_step = max(1, chunk_frame_count // 8)
                for frame in composed_output:
                    if cancel_event.is_set():
                        note_terminal("job_cancelled")
                        emit("job_cancelled", {})
                        raise SystemExit(0)
                    try:
                        encoder.write_frame(_float_frame_to_rgb24(frame))
                    except (BrokenPipeError, OSError) as exc:
                        encode_detail = _describe_encode_stream_failure(encoder, exc)
                        log(
                            "error",
                            "encode_stream_failure",
                            "phase_processing",
                            "The ffmpeg encoder stopped accepting frames during phase processing.",
                            {"error": encode_detail},
                        )
                        emit_failure(
                            "encode_failure",
                            "phase_processing",
                            detail=encode_detail,
                            exception_type=type(exc).__name__,
                        )
                        raise SystemExit(1)
                    processed_frame_count += 1
                    frame_index_in_chunk = processed_frame_count - chunk_start_frame_count
                    if (
                        frame_index_in_chunk == chunk_frame_count
                        or frame_index_in_chunk % progress_emit_step == 0
                    ):
                        decode_counter = decoder.take_progress_counter() or (
                            processed_frame_count
                            + max(chunk_frame_count - frame_index_in_chunk, 0)
                        )
                        encode_counter = encoder.take_progress_counter() or processed_frame_count
                        emit(
                            "progress_update",
                            {
                                "progress_token": f"batch:{processed_frame_count}",
                                "stage": "phase_processing",
                                "frames_completed": processed_frame_count,
                                "total_frames": working_probe.frame_count,
                                "decode_frames_completed": decode_counter,
                                "encoded_frames_completed": encode_counter,
                            },
                        )
                del composed_output
                del passthrough_output
                del amplified_output
                del amplified_processing
                del processing_chunk
                del source_chunk
                gc.collect()

            decode_exit = decoder.close()
            record_stage_transition("encode")
            emit("stage_started", {"stage": "encode", "total_frames": working_probe.frame_count})
            log("info", "stage_started", "encode", "Waiting for encoder drain and exit.")
            encoder_exit_code = encoder.finish()
        finally:
            if decode_exit is None:
                decoder.cancel()
            if encoder_exit_code is None:
                encoder.cancel()

        if decode_exit != 0 or processed_frame_count == 0:
            emit_failure("decode_failure", "phase_processing")
            raise SystemExit(1)

        try:
            output_probe = (
                FfprobeRunner().run(request.paths.staged_mp4_path)
                if request.paths.staged_mp4_path.exists()
                else None
            )
            output_probe_ok = output_probe is not None
        except Exception:
            output_probe = None
            output_probe_ok = False
        validation = validate_staged_mp4(
            StagedMp4Observation(
                encoder_exit_code=encoder_exit_code,
                file_exists=request.paths.staged_mp4_path.exists(),
                file_closed=True,
                size_bytes=request.paths.staged_mp4_path.stat().st_size
                if request.paths.staged_mp4_path.exists()
                else 0,
                probe_ok=output_probe_ok,
                video_stream_count=0 if output_probe is None else 1,
                audio_stream_count=0 if output_probe is None else output_probe.audio_stream_count,
                width=0 if output_probe is None else output_probe.width,
                height=0 if output_probe is None else output_probe.height,
                codec="" if output_probe is None else output_probe.codec_name,
                profile="" if output_probe is None else output_probe.profile,
                pixel_format="" if output_probe is None else output_probe.pixel_format,
                frame_count=0 if output_probe is None else output_probe.frame_count,
                duration_seconds=0.0 if output_probe is None else output_probe.duration_seconds,
            ),
            OutputExpectation(
                width=request.intent.output_resolution.width,
                height=request.intent.output_resolution.height,
                codec=codec_plan.expected_codec_name,
                profile=codec_plan.expected_profile,
                pixel_format=codec_plan.pixel_format,
                expected_frame_count=working_probe.frame_count,
                expected_fps=working_plan.working_fps,
            ),
        )
        if not validation.is_valid:
            log(
                "error",
                "staged_output_invalid",
                "finalize",
                "Staged MP4 validation failed.",
                {"errors": list(validation.errors)},
            )
            emit_failure("encode_failure", "finalize")
            raise SystemExit(1)

        record_stage_transition("finalize")
        emit("stage_started", {"stage": "finalize", "total_frames": working_probe.frame_count})
        log("info", "stage_started", "finalize", "Writing sidecar and finalizing output pair.")
        if request.intent.analysis.enabled:
            if analysis_analyzer is not None:
                try:
                    analysis_export = analysis_analyzer.finalize(request.paths.output_directory)
                except Exception as exc:
                    analysis_warning = (
                        "Quantitative analysis export failed after render completion: "
                        f"{exc}"
                    )
                    log(
                        "warning",
                        "analysis_export_failed",
                        "finalize",
                        analysis_warning,
                    )
                    emit("warning", {"messages": [analysis_warning]})
                    analysis_export = build_empty_analysis_export(
                        roi=request.intent.analysis.roi,
                        roi_mode=(
                            "whole_frame"
                            if request.intent.analysis.roi is None
                            else "manual"
                        ),
                        roi_label=(
                            "Whole-frame ROI"
                            if request.intent.analysis.roi is None
                            else "Manual ROI"
                        ),
                        warning=analysis_warning,
                        output_directory=request.paths.output_directory,
                    )
            elif analysis_collection_warning is not None:
                analysis_export = build_empty_analysis_export(
                    roi=request.intent.analysis.roi,
                    roi_mode=(
                        "whole_frame"
                        if request.intent.analysis.roi is None
                        else "manual"
                    ),
                    roi_label=(
                        "Whole-frame ROI"
                        if request.intent.analysis.roi is None
                        else "Manual ROI"
                    ),
                    warning=analysis_collection_warning,
                    output_directory=request.paths.output_directory,
                )
        sidecar_payload = _build_sidecar_payload(
            request=request,
            probe=probe,
            working_probe=working_probe,
            fingerprint=fingerprint,
            preflight=preflight,
            codec_plan=codec_plan,
            normalization_plan=normalization_plan,
            normalized_source_path=normalized_source_path,
            final_validation_errors=validation.errors,
            analysis_export=analysis_export,
            analysis_collection_warning=analysis_collection_warning,
        )
        request.paths.staged_sidecar_path.write_text(
            json.dumps(sidecar_payload, indent=2),
            encoding="utf-8",
        )

        finalization = finalize_output_pair(
            staged_mp4=request.paths.staged_mp4_path,
            staged_sidecar=request.paths.staged_sidecar_path,
            final_mp4=request.paths.final_mp4_path,
            final_sidecar=request.paths.final_sidecar_path,
            failed_evidence_dir=request.paths.failed_evidence_directory,
        )
        _handle_finalization_result(finalization, emit, log)
        if finalization.status != "completed":
            note_terminal(
                "failure",
                classification=finalization.classification or "finalization_failed",
                stage="finalize",
            )
            raise SystemExit(1)

        emit(
            "artifact_paths",
            {
                "mp4": str(request.paths.final_mp4_path),
                "sidecar": str(request.paths.final_sidecar_path),
                "diagnostics": str(request.paths.diagnostics_directory),
                "diagnostics_bundle": str(diagnostics_bundle_path),
                **analysis_export.artifact_paths,
            },
        )
        note_terminal("job_completed")
        emit("job_completed", {})
        raise SystemExit(0)

    except SystemExit:
        raise
    except MemoryError as exc:
        note_terminal(
            "failure",
            classification=_classify_worker_exception(exc),
            stage=current_stage,
        )
        log(
            "error",
            _classify_worker_exception(exc),
            current_stage,
            "Render worker hit a classified memory allocation failure.",
            {
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        emit_failure(
            _classify_worker_exception(exc),
            current_stage,
            detail=_format_worker_exception_detail(exc),
            exception_type=type(exc).__name__,
        )
        raise SystemExit(1)
    except Exception as exc:  # pragma: no cover - defensive worker path.
        classification = _classify_worker_exception(exc)
        note_terminal(
            "failure",
            classification=classification,
            stage=current_stage,
            detail=_format_worker_exception_detail(exc),
            exception_type=type(exc).__name__,
        )
        log(
            "error",
            classification,
            current_stage,
            "Unhandled render worker exception.",
            {
                "error": str(exc),
                "exception_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            },
        )
        emit_failure(
            classification,
            current_stage,
            detail=_format_worker_exception_detail(exc),
            exception_type=type(exc).__name__,
        )
        raise SystemExit(1)
    finally:
        stage_timings[current_stage] = stage_timings.get(current_stage, 0.0) + max(
            time.monotonic() - current_stage_started_at,
            0.0,
        )
        _write_bundle_if_possible(
            request=request,
            diagnostics_bundle_path=diagnostics_bundle_path,
            fingerprint=fingerprint,
            probe=probe,
            working_probe=working_probe,
            source_metadata=source_metadata,
            normalization_plan=normalization_plan,
            normalized_source_path=normalized_source_path,
            preflight=preflight,
            scheduler=scheduler,
            budgets=budgets,
            stage_timings=stage_timings,
            terminal_message_type=terminal_message_type,
            last_emitted_message_type=last_emitted_message_type,
            last_progress_token=last_progress_token,
            failure_classification=failure_classification,
            failure_stage=failure_stage,
            failure_detail=failure_detail,
            failure_exception_type=failure_exception_type,
            analysis_artifact_paths=analysis_export.artifact_paths,
        )
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)
        sock.close()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_worker_exception(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "out_of_memory"
    return "internal_processing_exception"


def _format_worker_exception_detail(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _describe_encode_stream_failure(
    encoder: RawvideoEncodeProcess,
    exc: BaseException,
) -> str:
    detail = _format_worker_exception_detail(exc)
    exit_code = encoder.process.poll()
    ffmpeg_detail = encoder.latest_stderr_summary()
    if exit_code is not None:
        detail = f"{detail}. Encoder exit code {exit_code}."
    if ffmpeg_detail:
        return f"{detail} ffmpeg: {ffmpeg_detail}"
    return detail


def _write_bundle_if_possible(
    *,
    request: RenderRequest,
    diagnostics_bundle_path: Path,
    fingerprint: str | None,
    probe,
    working_probe,
    source_metadata: SourceMetadata | None,
    normalization_plan: SourceNormalizationPlan | None,
    normalized_source_path: Path | None,
    preflight,
    scheduler: SchedulerInputs | None,
    budgets: ResourceBudget | None,
    stage_timings: dict[str, float],
    terminal_message_type: str | None,
    last_emitted_message_type: str | None,
    last_progress_token: int | None,
    failure_classification: str | None,
    failure_stage: str | None,
    failure_detail: str | None,
    failure_exception_type: str | None,
    analysis_artifact_paths: dict[str, str],
) -> None:
    settings_snapshot = {
        "intent": request.intent.to_dict(),
        "diagnostic_level": request.diagnostic_level.value,
    }
    source_payload = {"path": str(request.paths.source_path)}
    if fingerprint is not None:
        source_payload["fingerprint_sha256"] = fingerprint
    if probe is not None:
        source_payload.update(
            {
                "native_width": probe.width,
                "native_height": probe.height,
                "native_fps": probe.avg_fps or probe.fps or 1.0,
                "native_frame_count": probe.frame_count,
                "bit_depth": probe.bit_depth,
                "codec_name": probe.codec_name,
                "pixel_format": probe.pixel_format,
            }
        )
    if working_probe is not None:
        source_payload.update(
            {
                "working_width": working_probe.width,
                "working_height": working_probe.height,
                "working_fps": working_probe.avg_fps or working_probe.fps or 1.0,
                "working_frame_count": working_probe.frame_count,
            }
        )
    if normalization_plan is not None:
        source_payload["normalization"] = {
            "steps": list(normalization_plan.normalization_steps),
            "requires_cfr_normalization": normalization_plan.requires_cfr_normalization,
            "requires_square_pixel_normalization": normalization_plan.requires_square_pixel_normalization,
        }
    if normalized_source_path is not None:
        source_payload["normalized_source_path"] = str(normalized_source_path)
    if source_metadata is not None:
        source_payload["supported_sdr_path"] = {
            "explicit_rec709_compatible": source_metadata.explicit_rec709_compatible,
            "heuristic_sdr_allowed": source_metadata.heuristic_sdr_allowed,
            "has_hdr_markers": source_metadata.has_hdr_markers,
        }

    preflight_report = {
        "warnings": [],
        "blockers": [],
    }
    if preflight is not None:
        preflight_report.update(
            {
                "warnings": [issue.message for issue in preflight.warnings],
                "blockers": [issue.message for issue in preflight.blockers],
                "source_fps": None if normalization_plan is None else normalization_plan.native_fps,
                "working_fps": None if normalization_plan is None else normalization_plan.working_fps,
                "source_is_cfr": None if normalization_plan is None else normalization_plan.source_is_cfr,
                "normalization_steps": []
                if normalization_plan is None
                else list(normalization_plan.normalization_steps),
                "nyquist_limit_hz": preflight.nyquist_limit_hz,
                "active_scratch_required_bytes": preflight.active_scratch_required_bytes,
                "ram_required_bytes": preflight.ram_required_bytes,
                "output_staging_required_bytes": preflight.output_staging_required_bytes,
            }
        )
    scheduler_payload = {
        "chunk_frames": None if scheduler is None else scheduler.chunk_frames,
        "thread_limit": None if scheduler is None else scheduler.thread_limit,
        "precision_bytes": None if scheduler is None else scheduler.precision_bytes,
        "decode_pass_count": 2,
        "pipeline_mode": "two_pass_streaming_batches",
        "ffmpeg_thread_args": ["-threads", "1"],
        "effective_thread_limits": {"python_worker": 1},
    }
    memory_payload = {
        "available_ram_bytes": None if budgets is None else budgets.available_ram_bytes,
        "reserved_ui_headroom_bytes": None
        if budgets is None
        else budgets.reserved_ui_headroom_bytes,
        "estimated_ram_required_bytes": None
        if preflight is None
        else preflight.ram_required_bytes,
    }
    status = {
        "job_completed": "completed",
        "job_cancelled": "cancelled",
    }.get(terminal_message_type, "failed")
    write_diagnostics_bundle(
        DiagnosticsBundleInput(
            job_id=request.job_id,
            status=status,
            diagnostics_directory=request.paths.diagnostics_directory,
            diagnostics_cap_bytes=request.diagnostics_cap_bytes,
            jsonl_log_path=request.paths.jsonl_log_path,
            settings_snapshot=settings_snapshot,
            source_metadata=source_payload,
            preflight_report=preflight_report,
            scheduler_decisions=scheduler_payload,
            memory_estimate=memory_payload,
            mask_geometry=[zone.to_dict() for zone in request.intent.exclusion_zones],
            stage_timings=stage_timings,
            watchdog_evidence={
                "worker_pid": os.getpid(),
                "terminal_message_type": terminal_message_type,
                "last_emitted_message_type": last_emitted_message_type,
                "last_progress_token": last_progress_token,
            },
            artifact_paths={
                "mp4": str(request.paths.final_mp4_path),
                "sidecar": str(request.paths.final_sidecar_path),
                "diagnostics": str(request.paths.diagnostics_directory),
                "diagnostics_bundle": str(diagnostics_bundle_path),
                **analysis_artifact_paths,
            },
            terminal_details={
                "message_type": terminal_message_type,
                "classification": failure_classification,
                "stage": failure_stage,
                "detail": failure_detail,
                "exception_type": failure_exception_type,
            },
            intermediate_storage_policy="two_pass_streaming_rgb24_batches",
        )
    )


def _build_source_metadata(
    probe: Any,
    normalization_plan: SourceNormalizationPlan,
) -> SourceMetadata:
    hdr_markers = {
        (probe.color_primaries or "").lower(),
        (probe.color_transfer or "").lower(),
        (probe.color_space or "").lower(),
    }
    has_hdr = any(
        marker in {"bt2020", "smpte2084", "arib-std-b67"} for marker in hdr_markers
    )
    explicit_rec709 = all(
        tag in {"", "bt709", "tv", "pc", "gbr"}
        for tag in hdr_markers | {(probe.color_range or "").lower()}
    )
    heuristic = not has_hdr and not explicit_rec709
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
        heuristic_sdr_allowed=heuristic,
        has_hdr_markers=has_hdr,
        contradictory_color_metadata=False,
        decoded_format_supported=True,
    )


def _build_resource_budget(paths: RenderPaths, retention_budget_bytes: int) -> ResourceBudget:
    scratch_usage = psutil.disk_usage(str(paths.scratch_directory))
    output_usage = psutil.disk_usage(str(paths.output_directory))
    memory = psutil.virtual_memory()
    retained_evidence_bytes = measure_retained_roots_bytes(
        (paths.diagnostics_directory.parent, paths.scratch_directory.parent),
        exclude_paths=(paths.diagnostics_directory, paths.scratch_directory),
    )
    return ResourceBudget(
        available_scratch_bytes=int(scratch_usage.free),
        scratch_floor_bytes=512 * 1024 * 1024,
        available_output_volume_bytes=int(output_usage.free),
        available_ram_bytes=int(memory.available),
        reserved_ui_headroom_bytes=512 * 1024 * 1024,
        retention_budget_bytes=retention_budget_bytes,
        retained_evidence_bytes=retained_evidence_bytes,
    )


def _bytes_to_float_frames(frame_bytes: list[bytes], *, width: int, height: int) -> np.ndarray:
    array = np.frombuffer(b"".join(frame_bytes), dtype=np.uint8)
    frames = array.reshape(len(frame_bytes), height, width, 3)
    return (frames.astype(np.float32) / 255.0).copy()


def _frames_to_luma(frames_rgb: np.ndarray) -> np.ndarray:
    red_weight = np.float32(0.2126)
    green_weight = np.float32(0.7152)
    blue_weight = np.float32(0.0722)
    return (
        red_weight * frames_rgb[..., 0]
        + green_weight * frames_rgb[..., 1]
        + blue_weight * frames_rgb[..., 2]
    ).astype(np.float32, copy=False)


def _float_frame_to_rgb24(frame: np.ndarray) -> bytes:
    frame_uint8 = np.clip(np.round(frame * 255.0), 0, 255).astype(np.uint8)
    return frame_uint8.tobytes()


def _default_color_tags() -> dict[str, str]:
    return {
        "color_primaries": "bt709",
        "color_transfer": "bt709",
        "color_space": "bt709",
        "color_range": "tv",
    }


def _choose_codec_plan() -> CodecPlan:
    tools = resolve_toolchain()
    try:
        completed = subprocess.run(
            [str(tools.ffmpeg), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        encoder_list = completed.stdout
    except (OSError, subprocess.TimeoutExpired):
        encoder_list = ""
    if "libx265" in encoder_list:
        return CodecPlan(
            ffmpeg_codec="libx265",
            expected_codec_name="hevc",
            expected_profile="Main 10",
            pixel_format="yuv420p10le",
            color_tags=_default_color_tags(),
        )
    return CodecPlan(
        ffmpeg_codec="libx264",
        expected_codec_name="h264",
        expected_profile="High",
        pixel_format="yuv420p",
        warnings=(
            "Falling back to H.264 High because HEVC Main10 encoding is not available.",
        ),
        color_tags=_default_color_tags(),
    )


def _query_tool_version(executable: Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(executable), "-version"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    version_output = completed.stdout or completed.stderr
    first_line = version_output.splitlines()[0].strip() if version_output else ""
    return first_line or None


def _build_sidecar_payload(
    *,
    request: RenderRequest,
    probe,
    working_probe,
    fingerprint: str,
    preflight,
    codec_plan: CodecPlan,
    normalization_plan: SourceNormalizationPlan | None,
    normalized_source_path: Path | None,
    final_validation_errors: tuple[str, ...],
    analysis_export,
    analysis_collection_warning: str | None,
) -> dict:
    toolchain = resolve_toolchain()
    observed_environment = ObservedEnvironment(
        app_version=__version__,
        engine_version=__version__,
        platform=platform.platform(),
        diagnostic_level=request.diagnostic_level,
        diagnostics_cap_bytes=request.diagnostics_cap_bytes,
        temp_root=str(request.paths.scratch_directory.parent),
        ffmpeg_version=_query_tool_version(toolchain.ffmpeg),
        ffprobe_version=_query_tool_version(toolchain.ffprobe),
        scheduler_clamp_threads=1,
        effective_thread_limits={"python_worker": 1},
    )
    preflight_summary = PreflightSummary(
        source_fps=probe.avg_fps or probe.fps or 1.0,
        source_is_cfr=probe.is_cfr,
        working_fps=(
            None
            if normalization_plan is None
            else normalization_plan.working_fps
        ),
        working_source_resolution=(
            None
            if normalization_plan is None
            else normalization_plan.working_resolution
        ),
        normalization_steps=()
        if normalization_plan is None
        else normalization_plan.normalization_steps,
        nyquist_limit_hz=(
            (probe.avg_fps or probe.fps or 1.0) / 2.0
            if normalization_plan is None
            else normalization_plan.working_fps / 2.0
        ),
        warnings=tuple(issue.message for issue in preflight.warnings),
        blockers=tuple(issue.message for issue in preflight.blockers),
    )
    source_record = SourceRecord(
        path=str(request.paths.source_path),
        fingerprint_sha256=fingerprint,
        size_bytes=request.paths.source_path.stat().st_size,
        modified_utc=datetime.fromtimestamp(
            request.paths.source_path.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat(),
    )
    drift_ack = build_drift_acknowledgement(
        request.drift_assessment,
        source_fingerprint_sha256=fingerprint,
        note="Reviewed in the drift-check editor.",
    )
    warnings = [*codec_plan.warnings, *analysis_export.warnings]
    if analysis_collection_warning is not None and analysis_collection_warning not in warnings:
        warnings.append(analysis_collection_warning)
    results = {
        "render_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_record.to_dict(),
        "preflight": preflight_summary.to_dict(),
        "warnings": warnings,
        "fallbacks": list(codec_plan.warnings),
        "artifact_paths": {
            "mp4": str(request.paths.final_mp4_path),
            "sidecar": str(request.paths.final_sidecar_path),
            "diagnostics": str(request.paths.diagnostics_directory),
            **analysis_export.artifact_paths,
        },
        "diagnostics_summary": {
            "jsonl_log": str(request.paths.jsonl_log_path),
            "bundle_path": str(request.paths.diagnostics_directory / "diagnostics_bundle.json"),
            "validation_errors": list(final_validation_errors),
            "normalized_source_path": (
                None if normalized_source_path is None else str(normalized_source_path)
            ),
        },
        "output_details": {
            "codec": codec_plan.expected_codec_name,
            "profile": codec_plan.expected_profile,
            "pixel_format": codec_plan.pixel_format,
            "color_tags": codec_plan.color_tags,
            "working_source": {
                "width": working_probe.width,
                "height": working_probe.height,
                "fps": working_probe.avg_fps or working_probe.fps or 1.0,
            },
        },
        "analysis": analysis_export.summary,
    }
    if drift_ack is not None:
        results["drift_acknowledgement"] = drift_ack.to_dict()
    return {
        "schema_version": SCHEMA_VERSION,
        "intent": request.intent.to_dict(),
        "observed_environment": observed_environment.to_dict(),
        "results": results,
    }


def _handle_finalization_result(finalization: FinalizationResult, emit, log) -> None:
    if finalization.status == "completed":
        log("info", "finalized", "finalize", "Final MP4 and sidecar pair committed.")
        return
    if finalization.classification == "lone_mp4_quarantined":
        log(
            "error",
            "lone_mp4_quarantined",
            "finalize",
            "Sidecar finalization failed after the MP4 became visible; the MP4 was quarantined.",
            {"quarantine_mp4_path": str(finalization.quarantine_mp4_path)},
        )
        emit(
            "failure",
            {
                "classification": "lone_mp4_quarantined",
                "stage": "finalize",
                "quarantine_mp4_path": str(finalization.quarantine_mp4_path),
            },
        )
        return
    log(
        "error",
        "lone_mp4_incomplete_failed_evidence",
        "finalize",
        "Sidecar finalization failed and quarantine also failed; the visible MP4 must be treated as incomplete failed evidence.",
        {"incomplete_visible_mp4_path": str(finalization.incomplete_visible_mp4_path)},
    )
    emit(
        "failure",
        {
            "classification": finalization.classification or "finalization_failed",
            "stage": "finalize",
            "incomplete_visible_mp4_path": str(finalization.incomplete_visible_mp4_path),
        },
    )
