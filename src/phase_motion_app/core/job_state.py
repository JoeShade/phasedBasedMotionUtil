"""This file owns the shell-facing setup and run state rules so the UI can stay responsive while still enforcing the design document's strict readiness semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class UiState(str, Enum):
    """This enum mirrors the instrument workflow states defined in the design doc."""

    IDLE = "idle"
    LOADED = "loaded"
    FINGERPRINT_PENDING = "fingerprint_pending"
    READY = "ready"
    PREFLIGHT_CHECK = "preflight_check"
    RENDERING = "rendering"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class SourceSnapshot:
    """This model carries the cheap source identity fields that the shell can monitor without rehashing the file every moment."""

    path: str
    size_bytes: int
    modified_ns: int


def detect_stale_source(reference: SourceSnapshot, current: SourceSnapshot) -> bool:
    """Detect obvious source changes using the shell-safe size and mtime checks required by the design doc."""

    return (
        reference.path != current.path
        or reference.size_bytes != current.size_bytes
        or reference.modified_ns != current.modified_ns
    )


def derive_setup_state(
    *,
    source_loaded: bool,
    settings_complete: bool,
    fingerprint_pending: bool,
    fingerprint_complete: bool,
    source_stale: bool,
) -> UiState:
    """Derive the setup state before pre-flight begins."""

    if not source_loaded:
        return UiState.IDLE
    if fingerprint_pending:
        return UiState.FINGERPRINT_PENDING
    if settings_complete and fingerprint_complete and not source_stale:
        return UiState.READY
    return UiState.LOADED


class InvalidTransitionError(RuntimeError):
    """This exception makes illegal state jumps explicit instead of silently mutating the controller."""


class SingleJobController:
    """This controller enforces one active job at a time and keeps the setup state separate from active worker states."""

    def __init__(self) -> None:
        self.state = UiState.IDLE
        self._source_loaded = False
        self._settings_complete = False
        self._fingerprint_pending = False
        self._fingerprint_complete = False
        self._source_stale = False
        self._reference_source: SourceSnapshot | None = None

    def _refresh_setup_state(self) -> None:
        if self.state in {
            UiState.PREFLIGHT_CHECK,
            UiState.RENDERING,
            UiState.COMPLETE,
            UiState.FAILED,
            UiState.CANCELLED,
        }:
            return
        self.state = derive_setup_state(
            source_loaded=self._source_loaded,
            settings_complete=self._settings_complete,
            fingerprint_pending=self._fingerprint_pending,
            fingerprint_complete=self._fingerprint_complete,
            source_stale=self._source_stale,
        )

    def load_source(self, snapshot: SourceSnapshot) -> None:
        if self.state in {UiState.COMPLETE, UiState.FAILED, UiState.CANCELLED}:
            self.state = UiState.IDLE
        self._source_loaded = True
        self._fingerprint_pending = True
        self._fingerprint_complete = False
        self._source_stale = False
        self._reference_source = snapshot
        self._refresh_setup_state()

    def set_settings_complete(self, complete: bool) -> None:
        self._settings_complete = complete
        self._refresh_setup_state()

    def mark_fingerprint_complete(self, snapshot: SourceSnapshot) -> None:
        self._reference_source = snapshot
        self._fingerprint_pending = False
        self._fingerprint_complete = True
        self._source_stale = False
        self._refresh_setup_state()

    def mark_fingerprint_failed(self) -> None:
        self._fingerprint_pending = False
        self._fingerprint_complete = False
        self._refresh_setup_state()

    def mark_source_changed(self, current: SourceSnapshot) -> None:
        if self._reference_source is None:
            return
        self._source_stale = detect_stale_source(self._reference_source, current)
        # A stale source invalidates authoritative readiness. The shell drops back
        # to Loaded until a new fingerprint run starts, rather than pretending the
        # source is still authoritative.
        if self._source_stale:
            self._fingerprint_complete = False
            self._fingerprint_pending = False
        self._refresh_setup_state()

    def start_preflight(self) -> None:
        if self.state is not UiState.READY:
            raise InvalidTransitionError("Pre-flight can start only from the Ready state.")
        self.state = UiState.PREFLIGHT_CHECK

    def begin_rendering(self) -> None:
        if self.state is not UiState.PREFLIGHT_CHECK:
            raise InvalidTransitionError(
                "Rendering can start only after pre-flight has been entered."
            )
        self.state = UiState.RENDERING

    def abort_preflight(self) -> None:
        """Return from a failed launch or aborted pre-flight bootstrap to the setup flow."""

        if self.state is not UiState.PREFLIGHT_CHECK:
            raise InvalidTransitionError(
                "Aborting pre-flight is allowed only while pre-flight is active."
            )
        self.state = UiState.LOADED
        self._refresh_setup_state()

    def mark_complete(self) -> None:
        if self.state is not UiState.RENDERING:
            raise InvalidTransitionError("Only an active render can complete.")
        self.state = UiState.COMPLETE

    def mark_failed(self) -> None:
        if self.state not in {UiState.PREFLIGHT_CHECK, UiState.RENDERING}:
            raise InvalidTransitionError(
                "Only pre-flight or an active render can fail into the Failed state."
            )
        self.state = UiState.FAILED

    def mark_cancelled(self) -> None:
        if self.state not in {UiState.PREFLIGHT_CHECK, UiState.RENDERING}:
            raise InvalidTransitionError(
                "Only pre-flight or an active render can transition to Cancelled."
            )
        self.state = UiState.CANCELLED

    def reset_after_terminal(self) -> None:
        """Return from a terminal run outcome to the setup flow so the user can launch a fresh attempt."""

        if self.state not in {UiState.COMPLETE, UiState.FAILED, UiState.CANCELLED}:
            raise InvalidTransitionError(
                "Resetting for a new attempt is allowed only after a terminal run outcome."
            )
        self.state = UiState.LOADED
        self._refresh_setup_state()

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
