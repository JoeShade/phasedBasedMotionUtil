"""This file tests shell-side render supervision so worker launch, IPC polling, cancellation, and failure classification stay correct before Qt wiring is added."""

from __future__ import annotations

import multiprocessing
import os
import socket
import time
from dataclasses import dataclass

from phase_motion_app.core.ipc import JsonLineConnection, SessionConfig, build_hello_ack
from phase_motion_app.core.render_supervisor import (
    RenderSupervisor,
    WorkerLaunchPlan,
)
from phase_motion_app.core.watchdog import WatchdogThresholds
from phase_motion_app.worker.main import WorkerBehavior, WorkerLaunchConfig, worker_process_main


def _spawn_context() -> multiprocessing.context.SpawnContext:
    return multiprocessing.get_context("spawn")


def _thresholds() -> WatchdogThresholds:
    return WatchdogThresholds(
        soft_timeout_seconds=0.2,
        hard_timeout_seconds=0.5,
        stall_timeout_seconds=0.5,
    )


def _launch_plan(behavior: WorkerBehavior) -> WorkerLaunchPlan:
    def build_config(
        host: str,
        port: int,
        session_token: str,
        job_id: str,
        role: str,
    ) -> WorkerLaunchConfig:
        return WorkerLaunchConfig(
            host=host,
            port=port,
            session_token=session_token,
            job_id=job_id,
            role=role,
            behavior=behavior,
        )

    return WorkerLaunchPlan(
        role="render",
        target=worker_process_main,
        config_factory=build_config,
    )


def _poll_until_terminal(
    supervisor: RenderSupervisor,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = supervisor.poll()
        if result.snapshot.is_terminal:
            return
        time.sleep(0.02)
    raise AssertionError("Worker session did not reach a terminal state.")


def _poll_until_connected(
    supervisor: RenderSupervisor,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = supervisor.poll()
        if result.snapshot.worker_pid is not None:
            return
        time.sleep(0.02)
    raise AssertionError("Worker session did not complete its handshake.")


@dataclass(frozen=True)
class _StubbornWorkerConfig:
    host: str
    port: int
    session_token: str
    job_id: str
    role: str


def _stubborn_worker_main(config: _StubbornWorkerConfig, _cancel_event) -> None:
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

    connection.send(build_hello_ack(session, os.getpid()))
    while True:
        time.sleep(0.1)


def test_render_supervisor_completes_successful_worker() -> None:
    supervisor = RenderSupervisor(
        job_id="job-supervisor-success",
        launch_plan=_launch_plan(WorkerBehavior(mode="success", progress_steps=2, exit_code=0)),
        thresholds=_thresholds(),
        spawn_context=_spawn_context(),
    )

    try:
        supervisor.start()
        _poll_until_terminal(supervisor)
        snapshot = supervisor.snapshot()
    finally:
        supervisor.close()

    assert snapshot.phase == "complete"
    assert snapshot.terminal_message_type == "job_completed"
    assert snapshot.failure_classification is None


def test_render_supervisor_classifies_silent_exit_as_protocol_failure() -> None:
    supervisor = RenderSupervisor(
        job_id="job-supervisor-silent",
        launch_plan=_launch_plan(WorkerBehavior(mode="silent_exit", exit_code=0)),
        thresholds=_thresholds(),
        spawn_context=_spawn_context(),
    )

    try:
        supervisor.start()
        _poll_until_terminal(supervisor)
        snapshot = supervisor.snapshot()
    finally:
        supervisor.close()

    assert snapshot.phase == "failed"
    assert snapshot.failure_classification == "worker_protocol_failure"


def test_render_supervisor_surfaces_terminal_failure_payload() -> None:
    supervisor = RenderSupervisor(
        job_id="job-supervisor-failure",
        launch_plan=_launch_plan(WorkerBehavior(mode="failure", exit_code=1)),
        thresholds=_thresholds(),
        spawn_context=_spawn_context(),
    )

    try:
        supervisor.start()
        _poll_until_terminal(supervisor)
        snapshot = supervisor.snapshot()
    finally:
        supervisor.close()

    assert snapshot.phase == "failed"
    assert snapshot.failure_classification == "internal_processing_exception"
    assert snapshot.terminal_message_type == "failure"


def test_render_supervisor_rejects_claimed_success_with_nonzero_exit() -> None:
    supervisor = RenderSupervisor(
        job_id="job-supervisor-nonzero-success",
        launch_plan=_launch_plan(WorkerBehavior(mode="success", progress_steps=2, exit_code=7)),
        thresholds=_thresholds(),
        spawn_context=_spawn_context(),
    )

    try:
        supervisor.start()
        _poll_until_terminal(supervisor)
        snapshot = supervisor.snapshot()
    finally:
        supervisor.close()

    assert snapshot.phase == "failed"
    assert snapshot.terminal_message_type == "job_completed"
    assert snapshot.failure_classification == "worker_claimed_success_then_nonzero_exit"


def test_render_supervisor_cancels_active_worker() -> None:
    supervisor = RenderSupervisor(
        job_id="job-supervisor-cancel",
        launch_plan=_launch_plan(WorkerBehavior(mode="cancel_wait", exit_code=0)),
        thresholds=_thresholds(),
        spawn_context=_spawn_context(),
    )

    try:
        supervisor.start()
        time.sleep(0.05)
        supervisor.cancel()
        _poll_until_terminal(supervisor)
        snapshot = supervisor.snapshot()
    finally:
        supervisor.close()

    assert snapshot.phase == "cancelled"
    assert snapshot.terminal_message_type == "job_cancelled"
    assert snapshot.cancellation_requested is True


def test_render_supervisor_close_terminates_unresponsive_worker() -> None:
    def build_config(
        host: str,
        port: int,
        session_token: str,
        job_id: str,
        role: str,
    ) -> _StubbornWorkerConfig:
        return _StubbornWorkerConfig(
            host=host,
            port=port,
            session_token=session_token,
            job_id=job_id,
            role=role,
        )

    supervisor = RenderSupervisor(
        job_id="job-supervisor-close",
        launch_plan=WorkerLaunchPlan(
            role="render",
            target=_stubborn_worker_main,
            config_factory=build_config,
        ),
        thresholds=_thresholds(),
        spawn_context=_spawn_context(),
    )

    supervisor.start()
    _poll_until_connected(supervisor)
    supervisor.close()

    assert supervisor._process is not None
    assert supervisor._process.is_alive() is False
