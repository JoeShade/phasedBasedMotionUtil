"""This file defines the shell/worker IPC contract so the GUI and worker can exchange small structured messages without depending on queue-only telemetry."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1

TERMINAL_MESSAGE_TYPES = {"job_completed", "failure", "job_cancelled"}
NON_TERMINAL_MESSAGE_TYPES = {
    "job_started",
    "preflight_started",
    "preflight_report",
    "stage_started",
    "progress_update",
    "heartbeat",
    "warning",
    "artifact_paths",
}
ALL_MESSAGE_TYPES = TERMINAL_MESSAGE_TYPES | NON_TERMINAL_MESSAGE_TYPES


class ProtocolError(RuntimeError):
    """This exception marks malformed or mismatched IPC traffic so the shell can classify it explicitly."""


@dataclass(frozen=True)
class SessionConfig:
    """This model describes one shell/worker session."""

    session_token: str
    job_id: str
    role: str
    protocol_version: int = PROTOCOL_VERSION


class JsonLineConnection:
    """This small wrapper sends and receives bounded newline-delimited JSON messages over one socket."""

    def __init__(self, sock: socket.socket, max_message_bytes: int = 1024 * 1024) -> None:
        self.sock = sock
        self.max_message_bytes = max_message_bytes
        self._buffer = bytearray()

    def send(self, message: dict[str, Any]) -> None:
        encoded = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        self.sock.sendall(encoded)

    def read(self, timeout_seconds: float | None = None) -> dict[str, Any]:
        previous_timeout = self.sock.gettimeout()
        self.sock.settimeout(timeout_seconds)
        try:
            while b"\n" not in self._buffer:
                chunk = self.sock.recv(4096)
                if not chunk:
                    if self._buffer:
                        raise ProtocolError("Socket closed in the middle of a JSON line.")
                    raise EOFError("Socket closed.")
                self._buffer.extend(chunk)
                if len(self._buffer) > self.max_message_bytes:
                    raise ProtocolError("JSON line exceeded the maximum allowed size.")
            line, _, remainder = self._buffer.partition(b"\n")
            if len(line) > self.max_message_bytes:
                raise ProtocolError("JSON line exceeded the maximum allowed size.")
            self._buffer = bytearray(remainder)
            return json.loads(line.decode("utf-8"))
        finally:
            self.sock.settimeout(previous_timeout)


def open_loopback_server() -> socket.socket:
    """Open a loopback-only listening socket for one worker session."""

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    return server


def build_hello(config: SessionConfig) -> dict[str, Any]:
    """Build the shell-issued hello message required to start a session."""

    return {
        "message_type": "hello",
        "protocol_version": config.protocol_version,
        "session_token": config.session_token,
        "job_id": config.job_id,
        "role": config.role,
    }


def build_hello_ack(config: SessionConfig, pid: int) -> dict[str, Any]:
    """Build the worker hello acknowledgement."""

    return {
        "message_type": "hello_ack",
        "protocol_version": config.protocol_version,
        "session_token": config.session_token,
        "job_id": config.job_id,
        "role": config.role,
        "pid": pid,
    }


def build_message(
    *,
    config: SessionConfig,
    seq: int,
    message_type: str,
    monotonic_time_ns: int,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one routine IPC message."""

    if message_type not in ALL_MESSAGE_TYPES:
        raise ProtocolError(f"Unsupported IPC message type: {message_type}")
    return {
        "protocol_version": config.protocol_version,
        "session_token": config.session_token,
        "seq": seq,
        "message_type": message_type,
        "job_id": config.job_id,
        "monotonic_time_ns": monotonic_time_ns,
        "payload": payload or {},
    }


def validate_hello_ack(message: dict[str, Any], expected: SessionConfig) -> None:
    """Validate the worker hello acknowledgement against the expected session."""

    if message.get("message_type") != "hello_ack":
        raise ProtocolError("Expected hello_ack from worker.")
    if message.get("protocol_version") != expected.protocol_version:
        raise ProtocolError("Worker returned an unexpected protocol version.")
    if message.get("session_token") != expected.session_token:
        raise ProtocolError("Worker echoed the wrong session token.")
    if message.get("job_id") != expected.job_id:
        raise ProtocolError("Worker echoed the wrong job id.")
    if message.get("role") != expected.role:
        raise ProtocolError("Worker role did not match the shell expectation.")
    if not isinstance(message.get("pid"), int):
        raise ProtocolError("Worker hello_ack did not include a valid pid.")


def validate_session_message(
    message: dict[str, Any], expected: SessionConfig, previous_seq: int | None
) -> None:
    """Validate a routine session message and its strict sequence ordering."""

    if message.get("protocol_version") != expected.protocol_version:
        raise ProtocolError("Message protocol version does not match the session.")
    if message.get("session_token") != expected.session_token:
        raise ProtocolError("Message session token does not match the session.")
    if message.get("job_id") != expected.job_id:
        raise ProtocolError("Message job id does not match the session.")
    if message.get("message_type") not in ALL_MESSAGE_TYPES:
        raise ProtocolError("Message type is not part of the protocol contract.")
    if not isinstance(message.get("seq"), int):
        raise ProtocolError("Message seq is missing or invalid.")
    if previous_seq is not None and message["seq"] <= previous_seq:
        raise ProtocolError("Message seq must increase strictly within one session.")
    if not isinstance(message.get("monotonic_time_ns"), int):
        raise ProtocolError("Message monotonic_time_ns is missing or invalid.")
    if not isinstance(message.get("payload"), dict):
        raise ProtocolError("Message payload must be a JSON object.")


def perform_shell_handshake(
    connection: JsonLineConnection,
    config: SessionConfig,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Perform the shell side of the mandatory hello/hello_ack exchange."""

    connection.send(build_hello(config))
    ack = connection.read(timeout_seconds)
    validate_hello_ack(ack, config)
    return ack

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
