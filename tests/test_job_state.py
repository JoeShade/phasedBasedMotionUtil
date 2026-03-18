"""This file tests the setup and run state machine so readiness, source invalidation, and terminal transitions match the design document."""

from __future__ import annotations

import pytest

from phase_motion_app.core.job_state import (
    InvalidTransitionError,
    SingleJobController,
    SourceSnapshot,
    UiState,
    derive_setup_state,
    detect_stale_source,
)


def _snapshot(*, size_bytes: int = 100, modified_ns: int = 1) -> SourceSnapshot:
    return SourceSnapshot(
        path="A:/videos/source.mp4",
        size_bytes=size_bytes,
        modified_ns=modified_ns,
    )


def test_ready_requires_completed_fingerprint_and_non_stale_source() -> None:
    assert (
        derive_setup_state(
            source_loaded=True,
            settings_complete=True,
            fingerprint_pending=False,
            fingerprint_complete=False,
            source_stale=False,
        )
        is UiState.LOADED
    )
    assert (
        derive_setup_state(
            source_loaded=True,
            settings_complete=True,
            fingerprint_pending=True,
            fingerprint_complete=False,
            source_stale=False,
        )
        is UiState.FINGERPRINT_PENDING
    )
    assert (
        derive_setup_state(
            source_loaded=True,
            settings_complete=True,
            fingerprint_pending=False,
            fingerprint_complete=True,
            source_stale=False,
        )
        is UiState.READY
    )


def test_stale_source_detection_uses_size_and_mtime() -> None:
    reference = _snapshot()
    assert detect_stale_source(reference, _snapshot(size_bytes=101)) is True
    assert detect_stale_source(reference, _snapshot(modified_ns=2)) is True
    assert detect_stale_source(reference, _snapshot()) is False


def test_controller_transitions_follow_single_job_flow() -> None:
    controller = SingleJobController()

    controller.load_source(_snapshot())
    assert controller.state is UiState.FINGERPRINT_PENDING

    controller.set_settings_complete(True)
    assert controller.state is UiState.FINGERPRINT_PENDING

    controller.mark_fingerprint_complete(_snapshot())
    assert controller.state is UiState.READY

    controller.start_preflight()
    assert controller.state is UiState.PREFLIGHT_CHECK

    controller.begin_rendering()
    assert controller.state is UiState.RENDERING

    controller.mark_complete()
    assert controller.state is UiState.COMPLETE


def test_controller_drops_back_to_loaded_when_source_becomes_stale() -> None:
    controller = SingleJobController()
    controller.load_source(_snapshot())
    controller.set_settings_complete(True)
    controller.mark_fingerprint_complete(_snapshot())

    controller.mark_source_changed(_snapshot(size_bytes=200))

    assert controller.state is UiState.LOADED


def test_invalid_state_transition_raises() -> None:
    controller = SingleJobController()

    with pytest.raises(InvalidTransitionError):
        controller.start_preflight()

    controller.load_source(_snapshot())
    with pytest.raises(InvalidTransitionError):
        controller.begin_rendering()


def test_controller_can_return_from_complete_to_ready_for_a_new_attempt() -> None:
    controller = SingleJobController()

    controller.load_source(_snapshot())
    controller.set_settings_complete(True)
    controller.mark_fingerprint_complete(_snapshot())
    controller.start_preflight()
    controller.begin_rendering()
    controller.mark_complete()

    controller.reset_after_terminal()

    assert controller.state is UiState.READY


def test_controller_can_abort_preflight_back_to_ready_before_rendering() -> None:
    controller = SingleJobController()

    controller.load_source(_snapshot())
    controller.set_settings_complete(True)
    controller.mark_fingerprint_complete(_snapshot())
    controller.start_preflight()

    controller.abort_preflight()

    assert controller.state is UiState.READY

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
