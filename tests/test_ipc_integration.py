"""This file runs integration-style tests around worker launch, handshake, and shutdown so the socket protocol and spawn-based worker lifecycle stay honest."""

from __future__ import annotations

import multiprocessing
import socket
import time

import pytest

from phase_motion_app.core.ipc import (
    JsonLineConnection,
    ProtocolError,
    SessionConfig,
    open_loopback_server,
    perform_shell_handshake,
    validate_session_message,
)
from phase_motion_app.core.watchdog import WatchdogState, WatchdogThresholds
from phase_motion_app.worker.main import WorkerBehavior, WorkerLaunchConfig, worker_process_main


def _spawn_context() -> multiprocessing.context.SpawnContext:
    return multiprocessing.get_context("spawn")


def _thresholds() -> WatchdogThresholds:
    return WatchdogThresholds(
        soft_timeout_seconds=0.2,
        hard_timeout_seconds=0.4,
        stall_timeout_seconds=0.4,
    )


def _drain_messages(
    connection: JsonLineConnection,
    session: SessionConfig,
    *,
    timeout_seconds: float = 2.0,
) -> list[dict]:
    messages: list[dict] = []
    previous_seq: int | None = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            message = connection.read(timeout_seconds=0.1)
        except (EOFError, socket.timeout):
            break
        validate_session_message(message, session, previous_seq)
        previous_seq = message["seq"]
        messages.append(message)
        if message["message_type"] in {"job_completed", "failure", "job_cancelled"}:
            break
    return messages


def test_worker_launch_and_handshake() -> None:
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    session = SessionConfig(session_token="token-1", job_id="job-1", role="render")
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=worker_process_main,
        args=(
            WorkerLaunchConfig(
                host=host,
                port=port,
                session_token=session.session_token,
                job_id=session.job_id,
                role=session.role,
                behavior=WorkerBehavior(mode="success", progress_steps=2, exit_code=0),
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        ack = perform_shell_handshake(connection, session)
        messages = _drain_messages(connection, session)
    finally:
        process.join(timeout=5.0)
        server.close()

    assert ack["message_type"] == "hello_ack"
    assert process.exitcode == 0
    assert any(message["message_type"] == "job_started" for message in messages)
    assert any(message["message_type"] == "job_completed" for message in messages)


def test_worker_silent_exit_is_detected_as_failure() -> None:
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    session = SessionConfig(session_token="token-2", job_id="job-2", role="render")
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=worker_process_main,
        args=(
            WorkerLaunchConfig(
                host=host,
                port=port,
                session_token=session.session_token,
                job_id=session.job_id,
                role=session.role,
                behavior=WorkerBehavior(mode="silent_exit", exit_code=0),
            ),
            cancel_event,
        ),
    )
    process.start()

    watchdog = WatchdogState()
    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        perform_shell_handshake(connection, session)
        # A clean EOF after the ack is still a failure because the worker never
        # sent an authoritative terminal protocol message.
        try:
            connection.read(timeout_seconds=0.2)
        except EOFError:
            pass
    finally:
        process.join(timeout=5.0)
        server.close()

    decision = watchdog.evaluate(
        now=time.monotonic(),
        process_alive=process.is_alive(),
        exitcode=process.exitcode,
        thresholds=_thresholds(),
    )

    assert process.exitcode == 0
    assert decision.classification == "worker_protocol_failure"


def test_worker_rejects_protocol_version_mismatch_during_handshake() -> None:
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=worker_process_main,
        args=(
            WorkerLaunchConfig(
                host=host,
                port=port,
                session_token="token-4",
                job_id="job-4",
                role="render",
                behavior=WorkerBehavior(mode="success", progress_steps=1, exit_code=0),
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        connection.send(
            {
                "message_type": "hello",
                "protocol_version": 99,
                "session_token": "token-4",
                "job_id": "job-4",
                "role": "render",
            }
        )
        with pytest.raises(EOFError):
            connection.read(timeout_seconds=0.5)
    finally:
        process.join(timeout=5.0)
        server.close()

    assert process.exitcode == 2


def test_worker_rejects_role_mismatch_during_handshake() -> None:
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=worker_process_main,
        args=(
            WorkerLaunchConfig(
                host=host,
                port=port,
                session_token="token-5",
                job_id="job-5",
                role="render",
                behavior=WorkerBehavior(mode="success", progress_steps=1, exit_code=0),
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        connection.send(
            {
                "message_type": "hello",
                "protocol_version": 1,
                "session_token": "token-5",
                "job_id": "job-5",
                "role": "probe",
            }
        )
        with pytest.raises(EOFError):
            connection.read(timeout_seconds=0.5)
    finally:
        process.join(timeout=5.0)
        server.close()

    assert process.exitcode == 2


def test_json_line_connection_rejects_oversized_messages() -> None:
    left, right = socket.socketpair()
    connection = JsonLineConnection(left, max_message_bytes=32)
    try:
        right.sendall(b"{\"payload\":\"" + (b"x" * 64) + b"\"}\n")
        with pytest.raises(ProtocolError):
            connection.read(timeout_seconds=0.5)
    finally:
        left.close()
        right.close()


def test_worker_cancellation_path_reports_job_cancelled() -> None:
    ctx = _spawn_context()
    cancel_event = ctx.Event()
    session = SessionConfig(session_token="token-6", job_id="job-6", role="render")
    server = open_loopback_server()
    host, port = server.getsockname()
    process = ctx.Process(
        target=worker_process_main,
        args=(
            WorkerLaunchConfig(
                host=host,
                port=port,
                session_token=session.session_token,
                job_id=session.job_id,
                role=session.role,
                behavior=WorkerBehavior(mode="cancel_wait", exit_code=0),
            ),
            cancel_event,
        ),
    )
    process.start()

    try:
        connection_socket, _ = server.accept()
        connection = JsonLineConnection(connection_socket)
        perform_shell_handshake(connection, session)
        cancel_event.set()
        messages = _drain_messages(connection, session)
    finally:
        process.join(timeout=5.0)
        server.close()

    assert process.exitcode == 0
    assert messages[-1]["message_type"] == "job_cancelled"


def test_ipc_validation_rejects_non_increasing_sequence_numbers() -> None:
    session = SessionConfig(session_token="token-3", job_id="job-3", role="render")

    with_value_1 = {
        "protocol_version": 1,
        "session_token": "token-3",
        "seq": 1,
        "message_type": "job_started",
        "job_id": "job-3",
        "monotonic_time_ns": 10,
        "payload": {},
    }
    with_value_1_repeat = dict(with_value_1)

    validate_session_message(with_value_1, session, previous_seq=None)

    try:
        validate_session_message(with_value_1_repeat, session, previous_seq=1)
    except ProtocolError:
        pass
    else:
        raise AssertionError("Non-increasing sequence numbers must be rejected.")

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
