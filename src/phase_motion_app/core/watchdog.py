"""This file owns worker liveness evaluation so heartbeat silence, missing progress, and terminal-message disagreements are classified explicitly."""

from __future__ import annotations

from dataclasses import dataclass

from phase_motion_app.core.ipc import TERMINAL_MESSAGE_TYPES


@dataclass(frozen=True)
class WatchdogThresholds:
    """These configurable thresholds keep the policy explicit because the design doc does not hard-code numeric timeout values."""

    soft_timeout_seconds: float
    hard_timeout_seconds: float
    stall_timeout_seconds: float


@dataclass(frozen=True)
class WatchdogDecision:
    """This result object tells the caller whether the worker is healthy, warning, or terminally failed."""

    status: str
    classification: str | None = None
    message: str | None = None


class WatchdogState:
    """This state tracker applies the design doc's rule that only qualifying forward progress can clear a stall timer."""

    def __init__(self) -> None:
        self.last_heartbeat_at: float | None = None
        self.last_telemetry_at: float | None = None
        self.last_progress_at: float | None = None
        self.last_progress_token: int | str | None = None
        self.last_child_counters: dict[str, int] = {}
        self.terminal_message_type: str | None = None

    def record_telemetry(self, *, received_at: float) -> None:
        """Remember the latest shell-side receipt time even when the traffic was only the session handshake."""

        self.last_telemetry_at = received_at

    def record_message(
        self,
        *,
        message_type: str,
        received_at: float,
        progress_token: int | str | None = None,
    ) -> None:
        """Record one received shell-side message."""

        self.record_telemetry(received_at=received_at)

        if message_type == "heartbeat":
            self.last_heartbeat_at = received_at
            return

        if message_type == "stage_started":
            self.last_progress_at = received_at
            return

        if message_type == "progress_update":
            if progress_token is not None and progress_token != self.last_progress_token:
                self.last_progress_token = progress_token
                self.last_progress_at = received_at
            return

        if message_type in TERMINAL_MESSAGE_TYPES:
            self.terminal_message_type = message_type
            self.last_progress_at = received_at

    def record_child_progress(
        self,
        *,
        counter: int,
        received_at: float,
        channel: str = "child",
    ) -> None:
        """Record authoritative child progress such as ffmpeg machine-readable counters."""

        previous = self.last_child_counters.get(channel)
        if previous is None or counter > previous:
            self.last_child_counters[channel] = counter
            self.last_progress_at = received_at

    def evaluate(
        self,
        *,
        now: float,
        process_alive: bool,
        exitcode: int | None,
        thresholds: WatchdogThresholds,
        cancellation_requested: bool = False,
    ) -> WatchdogDecision:
        """Evaluate the current liveness state using shell-local time only."""

        if not process_alive:
            return self._evaluate_dead_process(
                exitcode=exitcode,
                cancellation_requested=cancellation_requested,
            )

        if self.last_progress_at is not None:
            if now - self.last_progress_at > thresholds.stall_timeout_seconds:
                return WatchdogDecision(
                    status="failed",
                    classification="worker_stalled",
                    message="Worker is alive but has not produced qualifying forward progress.",
                )

        silence_anchor = self.last_heartbeat_at
        if silence_anchor is None:
            silence_anchor = self.last_telemetry_at
        if silence_anchor is not None:
            silence = now - silence_anchor
            if silence > thresholds.hard_timeout_seconds:
                return WatchdogDecision(
                    status="failed",
                    classification="worker_unresponsive",
                    message="Worker heartbeat silence exceeded the hard timeout.",
                )
            if silence > thresholds.soft_timeout_seconds:
                return WatchdogDecision(
                    status="warning",
                    classification="worker_unresponsive_warning",
                    message="Worker heartbeat silence exceeded the soft timeout.",
                )

        return WatchdogDecision(status="running")

    def _evaluate_dead_process(
        self,
        *,
        exitcode: int | None,
        cancellation_requested: bool,
    ) -> WatchdogDecision:
        if self.terminal_message_type == "job_completed" and exitcode == 0:
            return WatchdogDecision(status="complete")
        if self.terminal_message_type == "job_completed" and exitcode != 0:
            return WatchdogDecision(
                status="failed",
                classification="worker_claimed_success_then_nonzero_exit",
                message="Worker reported success but exited with a non-zero code.",
            )
        if self.terminal_message_type == "failure":
            return WatchdogDecision(
                status="failed",
                classification="worker_reported_failure",
                message="Worker reported terminal failure.",
            )
        if self.terminal_message_type == "job_cancelled":
            if exitcode not in (None, 0):
                return WatchdogDecision(
                    status="failed",
                    classification="worker_reported_cancel_then_nonzero_exit",
                    message="Worker reported cancellation but exited with a non-zero code.",
                )
            return WatchdogDecision(
                status="cancelled" if cancellation_requested else "failed",
                classification=None if cancellation_requested else "unexpected_job_cancelled",
                message=None
                if cancellation_requested
                else "Worker reported cancellation without a shell-side cancellation request.",
            )
        if exitcode == 0:
            return WatchdogDecision(
                status="failed",
                classification="worker_protocol_failure",
                message="Worker exited cleanly without an authoritative terminal protocol message.",
            )
        return WatchdogDecision(
            status="failed",
            classification="worker_terminated_unexpectedly",
            message="Worker died before reporting a terminal state.",
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
