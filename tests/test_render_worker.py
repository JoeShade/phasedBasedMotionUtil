"""This file runs an end-to-end worker test so the real render path is exercised through spawn, IPC, decode, amplification, encode, sidecar writing, and finalization."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import socket
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import phase_motion_app.worker.render as render_module
from phase_motion_app.core.acceleration import AccelerationDecision
from phase_motion_app.core.ipc import (
    JsonLineConnection,
    SessionConfig,
    open_loopback_server,
    perform_shell_handshake,
    validate_session_message,
)
from phase_motion_app.core.models import DiagnosticLevel, JobIntent, PhaseSettings, Resolution, ResourcePolicy
from phase_motion_app.core.render_job import RenderPaths, RenderRequest
from phase_motion_app.core.sidecar import validate_sidecar_file
from phase_motion_app.core.toolchain import resolve_toolchain
from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.preflight import AnalyzerExecutionMode, SchedulerInputs
from phase_motion_app.worker.bootstrap import RenderWorkerConfig
from phase_motion_app.worker.render import render_worker_process_main


def _spawn_context() -> multiprocessing.context.SpawnContext:
    return multiprocessing.get_context("spawn")


def _create_motion_test_video(path: Path, *, frame_count: int = 12) -> None:
    tools = resolve_toolchain()
    width = 16
    height = 16
    frames: list[bytes] = []
    for index in range(frame_count):
        offset = index % 4
        frame = bytearray()
        for _y in range(height):
            for x in range(width):
                value = max(0, min(255, 32 + (x + offset) * 12))
                frame.extend((value, value, value))
        frames.append(bytes(frame))
    subprocess.run(
        [
            str(tools.ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "12",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            str(path),
        ],
        input=b"".join(frames),
        check=True,
        capture_output=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _request(
    tmp_path: Path,
    *,
    hardware_acceleration_enabled: bool = False,
    processing_resolution: Resolution = Resolution(16, 16),
    output_resolution: Resolution = Resolution(16, 16),
) -> RenderRequest:
    source_path = tmp_path / "request-source.mkv"
    source_path.write_bytes(b"worker-test-source")
    return RenderRequest(
        job_id="job-request",
        intent=JobIntent(
            phase=PhaseSettings(
                magnification=3.0,
                low_hz=1.0,
                high_hz=4.0,
                pyramid_type="complex_steerable",
                sigma=1.0,
                attenuate_other_frequencies=True,
            ),
            processing_resolution=processing_resolution,
            output_resolution=output_resolution,
            resource_policy=ResourcePolicy.CONSERVATIVE,
            mask_feather_px=4.0,
            hardware_acceleration_enabled=hardware_acceleration_enabled,
        ),
        paths=RenderPaths(
            source_path=source_path,
            output_directory=tmp_path / "output",
            output_stem="result",
            scratch_directory=tmp_path / "scratch",
            diagnostics_directory=tmp_path / "diagnostics",
        ),
        expected_source_fingerprint_sha256=_sha256(source_path),
        diagnostic_level=DiagnosticLevel.BASIC,
        diagnostics_cap_bytes=128 * 1024 * 1024,
        retention_budget_bytes=1,
        drift_assessment=DriftAssessment(),
    )


def _drain_messages(
    connection: JsonLineConnection,
    session: SessionConfig,
    *,
    timeout_seconds: float = 30.0,
) -> list[dict]:
    messages: list[dict] = []
    previous_seq: int | None = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            message = connection.read(timeout_seconds=0.5)
        except EOFError:
            break
        except socket.timeout:
            continue
        validate_session_message(message, session, previous_seq)
        previous_seq = message["seq"]
        messages.append(message)
        if message["message_type"] in {"job_completed", "failure", "job_cancelled"}:
            break
    return messages


def test_render_worker_produces_final_mp4_and_sidecar(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mkv"
    _create_motion_test_video(source_path)
    request = RenderRequest(
        job_id="job-render-1",
        intent=JobIntent(
            phase=PhaseSettings(
                magnification=3.0,
                low_hz=1.0,
                high_hz=4.0,
                pyramid_type="complex_steerable",
                sigma=1.0,
                attenuate_other_frequencies=True,
            ),
            processing_resolution=Resolution(16, 16),
            output_resolution=Resolution(16, 16),
            resource_policy=ResourcePolicy.CONSERVATIVE,
            mask_feather_px=4.0,
        ),
        paths=RenderPaths(
            source_path=source_path,
            output_directory=tmp_path / "output",
            output_stem="result",
            scratch_directory=tmp_path / "scratch",
            diagnostics_directory=tmp_path / "diagnostics",
        ),
        expected_source_fingerprint_sha256=_sha256(source_path),
        diagnostic_level=DiagnosticLevel.BASIC,
        diagnostics_cap_bytes=128 * 1024 * 1024,
        retention_budget_bytes=1,
        drift_assessment=DriftAssessment(),
    )
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    session = SessionConfig(session_token="token-render-1", job_id=request.job_id, role="render")
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=render_worker_process_main,
        args=(
            RenderWorkerConfig(
                host=host,
                port=port,
                session_token=session.session_token,
                job_id=session.job_id,
                role=session.role,
                request=request,
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        perform_shell_handshake(connection, session)
        messages = _drain_messages(connection, session)
    finally:
        process.join(timeout=60.0)
        server.close()

    assert process.exitcode == 0
    preflight_reports = [
        message for message in messages if message["message_type"] == "preflight_report"
    ]
    assert preflight_reports
    assert any(
        "retention budget" in warning.lower()
        for warning in preflight_reports[0]["payload"]["warnings"]
    )
    assert any(
        message["message_type"] == "progress_update"
        and message["payload"].get("stage") == "phase_processing"
        for message in messages
    )
    assert any(message["message_type"] == "job_completed" for message in messages)
    assert request.paths.final_mp4_path.exists()
    assert request.paths.final_sidecar_path.exists()
    assert (request.paths.diagnostics_directory / "diagnostics_bundle.json").exists()
    assert (request.paths.output_directory / "roi_metrics.csv").exists()
    validation = validate_sidecar_file(request.paths.final_sidecar_path)
    assert validation.is_valid is True
    sidecar_payload = json.loads(request.paths.final_sidecar_path.read_text(encoding="utf-8"))
    assert (
        sidecar_payload["observed_environment"]["diagnostics_cap_bytes"]
        == request.diagnostics_cap_bytes
    )
    assert sidecar_payload["results"]["preflight"]["working_fps"] == 12.0
    assert sidecar_payload["results"]["preflight"]["working_source_resolution"] == {
        "width": 16,
        "height": 16,
    }
    assert sidecar_payload["results"]["output_details"]["working_source"] == {
        "width": 16,
        "height": 16,
        "fps": 12.0,
    }
    assert sidecar_payload["results"]["analysis"]["enabled"] is True
    assert sidecar_payload["results"]["analysis"]["status"] == "completed"
    assert "roi_metrics" in sidecar_payload["results"]["artifact_paths"]
    bundle_payload = json.loads(
        (request.paths.diagnostics_directory / "diagnostics_bundle.json").read_text(
            encoding="utf-8"
        )
    )
    assert bundle_payload["diagnostics_cap_bytes"] == request.diagnostics_cap_bytes
    assert bundle_payload["scheduler_decisions"]["decode_pass_count"] == 2
    assert bundle_payload["scheduler_decisions"]["pipeline_mode"] == "two_pass_bounded_threaded_pipeline"
    assert bundle_payload["scheduler_decisions"]["internal_queue_depth"] == 1
    assert bundle_payload["scheduler_decisions"]["compute_worker_count"] == 1
    assert bundle_payload["scheduler_decisions"]["warp_worker_count"] == 1
    assert bundle_payload["scheduler_decisions"]["motion_worker_count"] == 1
    assert bundle_payload["scheduler_decisions"]["analyzer_mode"] == "background_thread"
    assert bundle_payload["intermediate_storage_policy"] == "two_pass_bounded_rgb24_pipeline"
    assert "roi_metrics" in bundle_payload["artifact_paths"]
    assert sidecar_payload["observed_environment"]["scheduler_clamp_threads"] == 1
    assert sidecar_payload["observed_environment"]["effective_thread_limits"][
        "python_compute_coordinator_threads"
    ] == 1
    assert sidecar_payload["observed_environment"]["effective_thread_limits"][
        "python_warp_worker_threads"
    ] == 1


def test_choose_codec_plan_falls_back_without_hevc_encoder(monkeypatch) -> None:
    monkeypatch.setattr(
        render_module,
        "resolve_toolchain",
        lambda: SimpleNamespace(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
    )
    monkeypatch.setattr(
        render_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="Encoders:\n V....D libx264\n"),
    )

    plan = render_module._choose_codec_plan()

    assert plan.ffmpeg_codec == "libx264"
    assert plan.expected_codec_name == "h264"
    assert plan.warnings


def test_query_tool_version_returns_first_output_line(monkeypatch) -> None:
    monkeypatch.setattr(
        render_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="ffmpeg version 6.1.1-static\nbuilt with clang\n",
            stderr="",
        ),
    )

    version = render_module._query_tool_version(Path("ffmpeg"))

    assert version == "ffmpeg version 6.1.1-static"


def test_worker_exception_classifier_maps_memory_error_to_out_of_memory() -> None:
    assert render_module._classify_worker_exception(MemoryError("oom")) == "out_of_memory"
    assert (
        render_module._classify_worker_exception(RuntimeError("boom"))
        == "internal_processing_exception"
    )


def test_format_worker_exception_detail_includes_type_and_message() -> None:
    assert (
        render_module._format_worker_exception_detail(RuntimeError("phase blew up"))
        == "RuntimeError: phase blew up"
    )
    assert render_module._format_worker_exception_detail(RuntimeError()) == "RuntimeError"


def test_describe_encode_stream_failure_includes_encoder_stderr() -> None:
    class _FakeProcess:
        def poll(self) -> int | None:
            return 1

    class _FakeEncoder:
        process = _FakeProcess()

        @staticmethod
        def latest_stderr_summary() -> str | None:
            return "width not divisible by 2 (853x480)"

    detail = render_module._describe_encode_stream_failure(
        _FakeEncoder(),
        BrokenPipeError(32, "Broken pipe"),
    )

    assert "BrokenPipeError: [Errno 32] Broken pipe" in detail
    assert "Encoder exit code 1" in detail
    assert "width not divisible by 2" in detail


def test_describe_staged_output_validation_failure_joins_errors() -> None:
    detail = render_module._describe_staged_output_validation_failure(
        (
            "Frame count was short.",
            "Duration was short.",
        )
    )

    assert "Staged MP4 validation failed:" in detail
    assert "Frame count was short." in detail
    assert "Duration was short." in detail


def test_scheduler_payload_reports_real_runtime_plan_fields() -> None:
    payload = render_module._scheduler_payload_from_inputs(
        SchedulerInputs(
            chunk_frames=64,
            chunk_cap_frames=96,
            chunk_target_ram_fraction=0.65,
            thread_limit=8,
            precision_bytes=4,
            native_buffer_multiplier=3.0,
            internal_queue_depth=3,
            compute_worker_count=6,
            warp_worker_count=6,
            motion_worker_count=3,
            analyzer_mode=AnalyzerExecutionMode.BACKGROUND_THREAD,
            analysis_queue_depth=2,
        )
    )

    assert payload["chunk_frames"] == 64
    assert payload["chunk_cap_frames"] == 96
    assert payload["internal_queue_depth"] == 3
    assert payload["compute_worker_count"] == 6
    assert payload["warp_worker_count"] == 6
    assert payload["motion_worker_count"] == 3
    assert payload["analyzer_mode"] == "background_thread"
    assert payload["analyzer_queue_depth"] == 2
    assert payload["effective_thread_limits"]["python_warp_worker_threads"] == 6


def test_render_worker_reuses_processing_domain_frames_for_output_when_resolutions_match(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    amplified_processing = np.zeros((2, 16, 16, 3), dtype=np.float32)
    processing_chunk = np.ones((2, 16, 16, 3), dtype=np.float32)

    amplified_output, passthrough_output = render_module._prepare_output_domain_chunks(
        amplified_processing=amplified_processing,
        processing_chunk=processing_chunk,
        request=request,
    )

    assert amplified_output is amplified_processing
    assert passthrough_output is processing_chunk


def test_validate_encoded_chunk_byte_length_rejects_short_payload() -> None:
    chunk = render_module._EncodedChunk(
        chunk_index=1,
        frame_count=2,
        frame_bytes=b"\x00" * 5,
        decode_counter=2,
    )

    with pytest.raises(ValueError, match="byte length did not match"):
        render_module._validate_encoded_chunk_byte_length(chunk, 3)


def test_render_worker_selects_accelerated_backend_when_requested_and_available(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_backend = SimpleNamespace(name="cupy", active=True)
    monkeypatch.setattr(
        render_module,
        "build_processing_backend",
        lambda requested: (
            AccelerationDecision(
                requested=requested,
                active=True,
                status="gpu_active",
                detail="Hardware acceleration is enabled. Using CuPy on Example GPU.",
                backend_name="cupy",
                device_name="Example GPU",
            ),
            fake_backend,
        ),
    )

    decision, backend = render_module._resolve_processing_backend_for_request(
        _request(tmp_path, hardware_acceleration_enabled=True)
    )

    assert decision.active is True
    assert backend is fake_backend


def test_render_worker_uses_cpu_backend_when_acceleration_falls_back(
    tmp_path: Path,
    monkeypatch,
) -> None:
    fake_backend = SimpleNamespace(name="cpu", active=False)
    monkeypatch.setattr(
        render_module,
        "build_processing_backend",
        lambda requested: (
            AccelerationDecision(
                requested=requested,
                active=False,
                status="gpu_requested_cpu_fallback",
                detail="Hardware acceleration was requested, but CuPy is unavailable. CPU fallback will be used.",
                backend_name="cupy",
            ),
            fake_backend,
        ),
    )

    decision, backend = render_module._resolve_processing_backend_for_request(
        _request(tmp_path, hardware_acceleration_enabled=True)
    )

    assert decision.active is False
    assert "CPU fallback" in decision.detail
    assert backend is fake_backend


def test_render_worker_rejects_mismatched_processing_and_output_resolution_request(
    tmp_path: Path,
) -> None:
    request = _request(
        tmp_path,
        processing_resolution=Resolution(16, 16),
        output_resolution=Resolution(12, 12),
    )

    with pytest.raises(ValueError, match="must match"):
        render_module._validate_matching_render_resolutions(request)


def test_render_worker_reports_incremental_phase_processing_frame_progress(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mkv"
    _create_motion_test_video(source_path)
    request = RenderRequest(
        job_id="job-render-progress",
        intent=JobIntent(
            phase=PhaseSettings(
                magnification=3.0,
                low_hz=1.0,
                high_hz=4.0,
                pyramid_type="complex_steerable",
                sigma=1.0,
                attenuate_other_frequencies=True,
            ),
            processing_resolution=Resolution(16, 16),
            output_resolution=Resolution(16, 16),
            resource_policy=ResourcePolicy.CONSERVATIVE,
            mask_feather_px=4.0,
        ),
        paths=RenderPaths(
            source_path=source_path,
            output_directory=tmp_path / "output",
            output_stem="result",
            scratch_directory=tmp_path / "scratch",
            diagnostics_directory=tmp_path / "diagnostics",
        ),
        expected_source_fingerprint_sha256=_sha256(source_path),
        diagnostic_level=DiagnosticLevel.BASIC,
        diagnostics_cap_bytes=128 * 1024 * 1024,
        retention_budget_bytes=1,
        drift_assessment=DriftAssessment(),
    )
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    session = SessionConfig(
        session_token="token-render-progress",
        job_id=request.job_id,
        role="render",
    )
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=render_worker_process_main,
        args=(
            RenderWorkerConfig(
                host=host,
                port=port,
                session_token=session.session_token,
                job_id=session.job_id,
                role=session.role,
                request=request,
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        perform_shell_handshake(connection, session)
        messages = _drain_messages(connection, session)
    finally:
        process.join(timeout=60.0)
        server.close()

    frame_progress = [
        message["payload"]["frames_completed"]
        for message in messages
        if message["message_type"] == "progress_update"
        and message["payload"].get("stage") == "phase_processing"
        and isinstance(message["payload"].get("frames_completed"), int)
    ]

    assert process.exitcode == 0
    assert len(frame_progress) >= 2
    assert frame_progress == sorted(frame_progress)
    assert frame_progress[-1] == 12


def test_render_worker_handles_partial_final_pipeline_chunk(tmp_path: Path) -> None:
    source_path = tmp_path / "source-tail.mkv"
    _create_motion_test_video(source_path, frame_count=26)
    request = RenderRequest(
        job_id="job-render-tail",
        intent=JobIntent(
            phase=PhaseSettings(
                magnification=3.0,
                low_hz=1.0,
                high_hz=4.0,
                pyramid_type="complex_steerable",
                sigma=1.0,
                attenuate_other_frequencies=True,
            ),
            processing_resolution=Resolution(16, 16),
            output_resolution=Resolution(16, 16),
            resource_policy=ResourcePolicy.CONSERVATIVE,
            mask_feather_px=4.0,
        ),
        paths=RenderPaths(
            source_path=source_path,
            output_directory=tmp_path / "output",
            output_stem="result",
            scratch_directory=tmp_path / "scratch",
            diagnostics_directory=tmp_path / "diagnostics",
        ),
        expected_source_fingerprint_sha256=_sha256(source_path),
        diagnostic_level=DiagnosticLevel.BASIC,
        diagnostics_cap_bytes=128 * 1024 * 1024,
        retention_budget_bytes=1,
        drift_assessment=DriftAssessment(),
    )
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    session = SessionConfig(
        session_token="token-render-tail",
        job_id=request.job_id,
        role="render",
    )
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=render_worker_process_main,
        args=(
            RenderWorkerConfig(
                host=host,
                port=port,
                session_token=session.session_token,
                job_id=session.job_id,
                role=session.role,
                request=request,
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        perform_shell_handshake(connection, session)
        messages = _drain_messages(connection, session)
    finally:
        process.join(timeout=60.0)
        server.close()

    assert process.exitcode == 0
    assert any(message["message_type"] == "job_completed" for message in messages)
    assert request.paths.final_mp4_path.exists()

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
