"""This file tests worker liveness classification so silence, stalls, and protocol disagreements become explicit failure reasons instead of generic errors."""

from __future__ import annotations

from phase_motion_app.core.watchdog import WatchdogState, WatchdogThresholds


def _thresholds() -> WatchdogThresholds:
    return WatchdogThresholds(
        soft_timeout_seconds=2.0,
        hard_timeout_seconds=4.0,
        stall_timeout_seconds=3.0,
    )


def test_watchdog_heartbeats_do_not_clear_stall_without_progress() -> None:
    watchdog = WatchdogState()
    watchdog.record_message(message_type="stage_started", received_at=0.0)
    watchdog.record_message(message_type="heartbeat", received_at=1.0)
    watchdog.record_message(message_type="heartbeat", received_at=2.0)

    decision = watchdog.evaluate(
        now=3.5,
        process_alive=True,
        exitcode=None,
        thresholds=_thresholds(),
    )

    assert decision.status == "failed"
    assert decision.classification == "worker_stalled"


def test_watchdog_child_progress_clears_stall_timer() -> None:
    watchdog = WatchdogState()
    watchdog.record_message(message_type="stage_started", received_at=0.0)
    watchdog.record_message(message_type="heartbeat", received_at=1.0)
    watchdog.record_child_progress(counter=50, received_at=2.5)
    watchdog.record_message(message_type="heartbeat", received_at=3.5)

    decision = watchdog.evaluate(
        now=4.0,
        process_alive=True,
        exitcode=None,
        thresholds=_thresholds(),
    )

    assert decision.status == "running"


def test_watchdog_reports_soft_then_hard_unresponsive_classification() -> None:
    watchdog = WatchdogState()
    watchdog.record_message(message_type="heartbeat", received_at=0.0)

    warning = watchdog.evaluate(
        now=2.5,
        process_alive=True,
        exitcode=None,
        thresholds=_thresholds(),
    )
    failure = watchdog.evaluate(
        now=4.5,
        process_alive=True,
        exitcode=None,
        thresholds=_thresholds(),
    )

    assert warning.status == "warning"
    assert warning.classification == "worker_unresponsive_warning"
    assert failure.status == "failed"
    assert failure.classification == "worker_unresponsive"


def test_watchdog_classifies_success_message_with_nonzero_exit_as_failure() -> None:
    watchdog = WatchdogState()
    watchdog.record_message(message_type="job_completed", received_at=2.0)

    decision = watchdog.evaluate(
        now=3.0,
        process_alive=False,
        exitcode=7,
        thresholds=_thresholds(),
    )

    assert decision.status == "failed"
    assert decision.classification == "worker_claimed_success_then_nonzero_exit"


def test_watchdog_classifies_zero_exit_without_terminal_message_as_protocol_failure() -> None:
    watchdog = WatchdogState()

    decision = watchdog.evaluate(
        now=1.0,
        process_alive=False,
        exitcode=0,
        thresholds=_thresholds(),
    )

    assert decision.status == "failed"
    assert decision.classification == "worker_protocol_failure"


def test_watchdog_rejects_cancel_message_with_nonzero_exit() -> None:
    watchdog = WatchdogState()
    watchdog.record_message(message_type="job_cancelled", received_at=2.0)

    decision = watchdog.evaluate(
        now=3.0,
        process_alive=False,
        exitcode=9,
        thresholds=_thresholds(),
        cancellation_requested=True,
    )

    assert decision.status == "failed"
    assert decision.classification == "worker_reported_cancel_then_nonzero_exit"
