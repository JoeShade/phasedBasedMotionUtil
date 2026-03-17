"""This file owns shell-side worker supervision so the PyQt process can launch one render worker, validate IPC, apply watchdog policy, and classify failures without doing heavy processing itself."""

from __future__ import annotations

import multiprocessing
import socket
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from phase_motion_app.core.ipc import (
    JsonLineConnection,
    ProtocolError,
    SessionConfig,
    open_loopback_server,
    validate_hello_ack,
    validate_session_message,
)
from phase_motion_app.core.watchdog import WatchdogDecision, WatchdogState, WatchdogThresholds


@dataclass(frozen=True)
class WorkerLaunchPlan:
    """This model tells the shell how to build and spawn exactly one worker attempt for one job."""

    role: str
    target: Callable[[Any, Any], None]
    config_factory: Callable[[str, int, str, str, str], Any]


@dataclass(frozen=True)
class RenderEvent:
    """This event object gives the shell a stable record of each newly received IPC message."""

    message_type: str
    payload: dict[str, Any]
    received_at: float


@dataclass(frozen=True)
class RenderSupervisorSnapshot:
    """This snapshot keeps the latest shell-side view of one worker session and its classified outcome."""

    phase: str
    stage: str | None = None
    worker_pid: int | None = None
    progress_token: int | str | None = None
    frames_completed: int | None = None
    total_frames: int | None = None
    preflight_warnings: tuple[str, ...] = ()
    preflight_blockers: tuple[str, ...] = ()
    preflight_details: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    artifact_paths: dict[str, str] = field(default_factory=dict)
    terminal_message_type: str | None = None
    failure_classification: str | None = None
    failure_stage: str | None = None
    failure_detail: str | None = None
    failure_exception_type: str | None = None
    watchdog_status: str = "running"
    watchdog_classification: str | None = None
    watchdog_message: str | None = None
    protocol_error: str | None = None
    exitcode: int | None = None
    cancellation_requested: bool = False

    @property
    def is_terminal(self) -> bool:
        return self.phase in {"complete", "failed", "cancelled"}


@dataclass(frozen=True)
class RenderPollResult:
    """This result groups new IPC traffic with the current classified session snapshot."""

    snapshot: RenderSupervisorSnapshot
    events: tuple[RenderEvent, ...] = ()


class RenderSupervisor:
    """This controller owns one worker process, its loopback socket, and the local watchdog timers."""

    def __init__(
        self,
        *,
        job_id: str,
        launch_plan: WorkerLaunchPlan,
        thresholds: WatchdogThresholds,
        spawn_context: multiprocessing.context.BaseContext | None = None,
    ) -> None:
        self._job_id = job_id
        self._launch_plan = launch_plan
        self._thresholds = thresholds
        self._spawn_context = spawn_context or multiprocessing.get_context("spawn")
        self._server: socket.socket | None = None
        self._connection_socket: socket.socket | None = None
        self._connection: JsonLineConnection | None = None
        self._process: multiprocessing.Process | None = None
        self._cancel_event: Any | None = None
        self._session: SessionConfig | None = None
        self._watchdog = WatchdogState()
        self._previous_seq: int | None = None
        self._hello_sent = False
        self._handshake_complete = False
        self._closed = False
        self._protocol_error: str | None = None
        self._phase = "idle"
        self._stage: str | None = None
        self._worker_pid: int | None = None
        self._progress_token: int | str | None = None
        self._frames_completed: int | None = None
        self._total_frames: int | None = None
        self._preflight_warnings: tuple[str, ...] = ()
        self._preflight_blockers: tuple[str, ...] = ()
        self._preflight_details: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._artifact_paths: dict[str, str] = {}
        self._failure_classification: str | None = None
        self._failure_stage: str | None = None
        self._failure_detail: str | None = None
        self._failure_exception_type: str | None = None
        self._watchdog_decision = WatchdogDecision(status="running")
        self._cancellation_requested = False

    @property
    def is_active(self) -> bool:
        return self._process is not None and not self.snapshot().is_terminal

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("Render supervisor has already started a worker.")

        # The shell opens the loopback listener first so the worker can connect
        # back using the documented hello/hello_ack protocol.
        server = open_loopback_server()
        server.settimeout(0.0)
        host, port = server.getsockname()
        session = SessionConfig(
            session_token=uuid.uuid4().hex,
            job_id=self._job_id,
            role=self._launch_plan.role,
        )
        cancel_event = self._spawn_context.Event()
        worker_config = self._launch_plan.config_factory(
            host,
            port,
            session.session_token,
            session.job_id,
            session.role,
        )
        process = self._spawn_context.Process(
            target=self._launch_plan.target,
            args=(worker_config, cancel_event),
        )
        process.start()

        self._server = server
        self._session = session
        self._cancel_event = cancel_event
        self._process = process
        self._phase = "starting"

    def cancel(self) -> None:
        if self._cancel_event is None or self.snapshot().is_terminal:
            return
        self._cancellation_requested = True
        self._cancel_event.set()

    def poll(self, *, now: float | None = None) -> RenderPollResult:
        if self._closed:
            return RenderPollResult(snapshot=self.snapshot())
        if self._process is None:
            raise RuntimeError("Render supervisor must be started before polling.")

        now = time.monotonic() if now is None else now
        events: list[RenderEvent] = []

        self._accept_connection_if_available()
        self._advance_handshake(now)
        events.extend(self._drain_messages(now))
        self._watchdog_decision = self._evaluate_watchdog(now)
        self._apply_terminal_decision()
        self._join_if_dead()

        return RenderPollResult(snapshot=self.snapshot(), events=tuple(events))

    def snapshot(self) -> RenderSupervisorSnapshot:
        exitcode = None if self._process is None else self._process.exitcode
        terminal_message = self._watchdog.terminal_message_type
        return RenderSupervisorSnapshot(
            phase=self._phase,
            stage=self._stage,
            worker_pid=self._worker_pid,
            progress_token=self._progress_token,
            frames_completed=self._frames_completed,
            total_frames=self._total_frames,
            preflight_warnings=self._preflight_warnings,
            preflight_blockers=self._preflight_blockers,
            preflight_details=dict(self._preflight_details),
            warnings=tuple(self._warnings),
            artifact_paths=dict(self._artifact_paths),
            terminal_message_type=terminal_message,
            failure_classification=self._failure_classification,
            failure_stage=self._failure_stage,
            failure_detail=self._failure_detail,
            failure_exception_type=self._failure_exception_type,
            watchdog_status=self._watchdog_decision.status,
            watchdog_classification=self._watchdog_decision.classification,
            watchdog_message=self._watchdog_decision.message,
            protocol_error=self._protocol_error,
            exitcode=exitcode,
            cancellation_requested=self._cancellation_requested,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._cancel_event is not None:
            self._cancel_event.set()

        if self._connection_socket is not None:
            try:
                self._connection_socket.close()
            except OSError:
                pass
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._process is not None:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=1.0)

    def _accept_connection_if_available(self) -> None:
        if self._server is None or self._connection is not None:
            return
        try:
            connection_socket, _ = self._server.accept()
        except (BlockingIOError, socket.timeout):
            return
        connection_socket.settimeout(0.0)
        self._connection_socket = connection_socket
        self._connection = JsonLineConnection(connection_socket)

    def _advance_handshake(self, now: float) -> None:
        if self._connection is None or self._session is None or self._handshake_complete:
            return

        if not self._hello_sent:
            self._connection.send(
                {
                    "message_type": "hello",
                    "protocol_version": self._session.protocol_version,
                    "session_token": self._session.session_token,
                    "job_id": self._session.job_id,
                    "role": self._session.role,
                }
            )
            self._hello_sent = True

        try:
            ack = self._connection.read(timeout_seconds=0.0)
        except (BlockingIOError, socket.timeout):
            return
        except EOFError:
            return
        except ProtocolError as exc:
            self._record_protocol_error(str(exc))
            return

        try:
            validate_hello_ack(ack, self._session)
        except ProtocolError as exc:
            self._record_protocol_error(str(exc))
            return

        self._handshake_complete = True
        self._worker_pid = ack["pid"]
        # The watchdog needs a shell-local receipt timestamp as soon as the
        # session is bound so a worker that goes silent immediately after the
        # handshake still times out instead of waiting forever.
        self._watchdog.record_telemetry(received_at=now)

    def _drain_messages(self, now: float) -> list[RenderEvent]:
        if self._connection is None or self._session is None or not self._handshake_complete:
            return []

        events: list[RenderEvent] = []
        while True:
            try:
                message = self._connection.read(timeout_seconds=0.0)
            except (BlockingIOError, socket.timeout):
                break
            except EOFError:
                break
            except ProtocolError as exc:
                self._record_protocol_error(str(exc))
                break

            try:
                validate_session_message(message, self._session, self._previous_seq)
            except ProtocolError as exc:
                self._record_protocol_error(str(exc))
                break

            self._previous_seq = message["seq"]
            message_type = message["message_type"]
            payload = message["payload"]
            progress_token = payload.get("progress_token")
            self._watchdog.record_message(
                message_type=message_type,
                received_at=now,
                progress_token=progress_token,
            )
            if message_type == "progress_update":
                decode_counter = payload.get("decode_frames_completed")
                if isinstance(decode_counter, int):
                    self._watchdog.record_child_progress(
                        channel="decode",
                        counter=decode_counter,
                        received_at=now,
                    )
                encode_counter = payload.get("encoded_frames_completed")
                if isinstance(encode_counter, int):
                    self._watchdog.record_child_progress(
                        channel="encode",
                        counter=encode_counter,
                        received_at=now,
                    )
            self._apply_message(message_type, payload)
            events.append(RenderEvent(message_type=message_type, payload=payload, received_at=now))
        return events

    def _apply_message(self, message_type: str, payload: dict[str, Any]) -> None:
        if message_type == "preflight_started":
            self._phase = "preflight"
            self._stage = "preflight"
            return
        if message_type == "preflight_report":
            self._preflight_warnings = tuple(str(item) for item in payload.get("warnings", []))
            self._preflight_blockers = tuple(str(item) for item in payload.get("blockers", []))
            self._preflight_details = {
                str(key): value
                for key, value in payload.items()
                if key not in {"warnings", "blockers"}
            }
            return
        if message_type == "stage_started":
            self._phase = "rendering"
            next_stage = str(payload.get("stage") or "render")
            stage_changed = next_stage != self._stage
            self._stage = next_stage
            if stage_changed:
                self._frames_completed = 0
                self._progress_token = None
            if isinstance(payload.get("total_frames"), int):
                self._total_frames = payload["total_frames"]
            return
        if message_type == "progress_update":
            self._progress_token = payload.get("progress_token")
            if isinstance(payload.get("frames_completed"), int):
                self._frames_completed = payload["frames_completed"]
            if isinstance(payload.get("total_frames"), int):
                self._total_frames = payload["total_frames"]
            return
        if message_type == "warning":
            messages = payload.get("messages", [])
            self._warnings.extend(str(item) for item in messages)
            return
        if message_type == "artifact_paths":
            self._artifact_paths.update(
                {str(key): str(value) for key, value in payload.items()}
            )
            return
        if message_type == "failure":
            self._failure_classification = payload.get("classification")
            self._failure_stage = payload.get("stage")
            detail = payload.get("detail")
            exception_type = payload.get("exception_type")
            self._failure_detail = None if detail is None else str(detail)
            self._failure_exception_type = (
                None if exception_type is None else str(exception_type)
            )
            return

    def _evaluate_watchdog(self, now: float) -> WatchdogDecision:
        if self._protocol_error is not None:
            return WatchdogDecision(
                status="failed",
                classification="ipc_protocol_error",
                message=self._protocol_error,
            )
        process_alive = self._process is not None and self._process.is_alive()
        exitcode = None if self._process is None else self._process.exitcode
        return self._watchdog.evaluate(
            now=now,
            process_alive=bool(process_alive),
            exitcode=exitcode,
            thresholds=self._thresholds,
            cancellation_requested=self._cancellation_requested,
        )

    def _apply_terminal_decision(self) -> None:
        decision = self._watchdog_decision
        if decision.status == "complete":
            self._phase = "complete"
            return
        if decision.status == "cancelled":
            self._phase = "cancelled"
            return
        if decision.status == "failed":
            self._phase = "failed"
            if self._failure_classification is None:
                self._failure_classification = decision.classification
            return
        if self._phase == "starting" and self._handshake_complete:
            self._phase = "connected"

    def _record_protocol_error(self, message: str) -> None:
        self._protocol_error = message
        if self._cancel_event is not None:
            self._cancel_event.set()

    def _join_if_dead(self) -> None:
        if self._process is None or self._process.is_alive():
            return
        self._process.join(timeout=0.0)
