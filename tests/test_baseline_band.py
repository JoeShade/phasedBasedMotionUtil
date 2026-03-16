"""This file tests the lightweight source-band suggestion so import-time guidance stays cheap and still lands around the dominant motion frequency."""

from __future__ import annotations

import math

import numpy as np

from phase_motion_app.core.baseline_band import suggest_frequency_band_from_motion_trace


def test_suggest_frequency_band_tracks_dominant_low_frequency() -> None:
    fps = 15.0
    samples = np.arange(0.0, 30.0, 1.0 / fps, dtype=np.float32)
    motion_trace = (
        0.2
        + 0.04 * np.sin(2.0 * math.pi * 0.4 * samples)
        + 0.01 * np.sin(2.0 * math.pi * 1.2 * samples)
    ).astype(np.float32)

    suggestion = suggest_frequency_band_from_motion_trace(motion_trace, fps=fps)

    assert suggestion is not None
    assert suggestion.low_hz < 0.4 < suggestion.high_hz
    assert 0.0 <= suggestion.confidence <= 1.0


def test_suggest_frequency_band_returns_none_for_flat_trace() -> None:
    motion_trace = np.zeros(64, dtype=np.float32)

    suggestion = suggest_frequency_band_from_motion_trace(motion_trace, fps=15.0)

    assert suggestion is None
