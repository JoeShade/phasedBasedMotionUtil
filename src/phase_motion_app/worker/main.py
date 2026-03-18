"""This file provides a small test worker scaffold so the IPC and supervision contract can be exercised without invoking the real render engine."""

from __future__ import annotations

import os
import socket
import threading
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event as EventType

from phase_motion_app.core.ipc import (
    JsonLineConnection,
    SessionConfig,
    build_hello_ack,
    build_message,
)


@dataclass(frozen=True)
class WorkerBehavior:
    """This model lets tests exercise success, cancellation, and silent-death paths without a real render engine yet."""

    mode: str = "success"
    heartbeat_interval_seconds: float = 0.05
    progress_steps: int = 2
    exit_code: int = 0
    terminal_settle_seconds: float = 0.05
    failure_detail: str | None = None


@dataclass(frozen=True)
class WorkerLaunchConfig:
    """This model contains the shell endpoint and session details the worker needs to connect back to the parent."""

    host: str
    port: int
    session_token: str
    job_id: str
    role: str
    behavior: WorkerBehavior


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


def worker_process_main(config: WorkerLaunchConfig, cancel_event: EventType) -> None:
    """Run one worker process using the documented shell-issued hello and worker hello_ack handshake."""

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

    if config.behavior.mode == "silent_exit":
        sock.close()
        raise SystemExit(config.behavior.exit_code)

    stop_heartbeat = threading.Event()
    sequence = _SharedSequence()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.is_set():
            safe_connection.send(
                build_message(
                    config=session,
                    seq=sequence.next(),
                    message_type="heartbeat",
                    monotonic_time_ns=time.monotonic_ns(),
                )
            )
            stop_heartbeat.wait(config.behavior.heartbeat_interval_seconds)

    safe_connection.send(
        build_message(
            config=session,
            seq=sequence.next(),
            message_type="job_started",
            monotonic_time_ns=time.monotonic_ns(),
            payload={"role": config.role},
        )
    )

    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        name="worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    def stop_heartbeat_loop() -> None:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)

    def settle_and_exit(exit_code: int) -> None:
        time.sleep(config.behavior.terminal_settle_seconds)
        sock.close()
        raise SystemExit(exit_code)

    safe_connection.send(
        build_message(
            config=session,
            seq=sequence.next(),
            message_type="stage_started",
            monotonic_time_ns=time.monotonic_ns(),
            payload={"stage": "render", "total_frames": config.behavior.progress_steps},
        )
    )

    if config.behavior.mode == "cancel_wait":
        while not cancel_event.is_set():
            time.sleep(0.02)
        stop_heartbeat_loop()
        safe_connection.send(
            build_message(
                config=session,
                seq=sequence.next(),
                message_type="job_cancelled",
                monotonic_time_ns=time.monotonic_ns(),
                payload={},
            )
        )
        settle_and_exit(0)

    if config.behavior.mode == "failure":
        stop_heartbeat_loop()
        safe_connection.send(
            build_message(
                config=session,
                seq=sequence.next(),
                message_type="failure",
                monotonic_time_ns=time.monotonic_ns(),
                payload={
                    "classification": "internal_processing_exception",
                    **(
                        {}
                        if config.behavior.failure_detail is None
                        else {"detail": config.behavior.failure_detail}
                    ),
                },
            )
        )
        settle_and_exit(config.behavior.exit_code or 1)

    for progress_token in range(1, config.behavior.progress_steps + 1):
        time.sleep(0.02)
        safe_connection.send(
            build_message(
                config=session,
                seq=sequence.next(),
                message_type="progress_update",
                monotonic_time_ns=time.monotonic_ns(),
                payload={
                    "progress_token": progress_token,
                    "frames_completed": progress_token,
                    "total_frames": config.behavior.progress_steps,
                },
            )
        )

    stop_heartbeat_loop()
    safe_connection.send(
        build_message(
            config=session,
            seq=sequence.next(),
            message_type="job_completed",
            monotonic_time_ns=time.monotonic_ns(),
            payload={"artifact_paths": {}},
        )
    )
    settle_and_exit(config.behavior.exit_code)

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
