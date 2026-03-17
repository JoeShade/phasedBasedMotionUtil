"""This file tests the lightweight source-band suggestion so import-time guidance stays cheap and still lands around the dominant motion frequency."""

from __future__ import annotations

import math

import numpy as np

from phase_motion_app.core.baseline_band import (
    _ProxyBandScore,
    _SpectrumPeakCandidate,
    _prefer_sharper_higher_frequency_candidate,
    _promote_phase_proxy_band_if_warranted,
    _tighten_periodic_candidate_band,
    suggest_frequency_band_from_motion_trace,
)


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


def test_suggest_frequency_band_supports_high_frequency_candidates() -> None:
    fps = 24.0
    samples = np.arange(0.0, 20.0, 1.0 / fps, dtype=np.float32)
    motion_trace = (
        0.3
        + 0.08 * np.sin(2.0 * math.pi * 5.6 * samples)
        + 0.01 * np.sin(2.0 * math.pi * 0.6 * samples)
    ).astype(np.float32)

    suggestion = suggest_frequency_band_from_motion_trace(motion_trace, fps=fps)

    assert suggestion is not None
    assert suggestion.low_hz < 5.6 < suggestion.high_hz
    assert suggestion.peak_hz > 5.0


def test_suggest_frequency_band_uses_narrower_shoulder_based_edges() -> None:
    fps = 15.0
    samples = np.arange(0.0, 30.0, 1.0 / fps, dtype=np.float32)
    motion_trace = (
        0.2
        + 0.06 * np.sin(2.0 * math.pi * 1.1 * samples)
        + 0.004 * np.sin(2.0 * math.pi * 3.8 * samples)
    ).astype(np.float32)

    suggestion = suggest_frequency_band_from_motion_trace(motion_trace, fps=fps)

    assert suggestion is not None
    assert suggestion.low_hz < 1.1 < suggestion.high_hz
    assert suggestion.high_hz < 1.8


def test_peak_selection_prefers_sharper_higher_frequency_near_tie() -> None:
    lower_sway_candidate = _SpectrumPeakCandidate(
        index=3,
        frequency_hz=0.33,
        amplitude=1.0,
        prominence=0.02,
        support_fraction=0.92,
        sharpness=0.12,
        periodicity_strength=0.30,
        harmonic_bonus=0.28,
        ubiquity_penalty=0.70,
        edge_dominance=0.72,
        score=0.566,
    )
    sharper_periodic_candidate = _SpectrumPeakCandidate(
        index=7,
        frequency_hz=0.56,
        amplitude=0.82,
        prominence=0.04,
        support_fraction=0.58,
        sharpness=0.22,
        periodicity_strength=0.43,
        harmonic_bonus=0.05,
        ubiquity_penalty=0.0,
        edge_dominance=0.48,
        score=0.549,
    )

    selected = _prefer_sharper_higher_frequency_candidate(
        [lower_sway_candidate, sharper_periodic_candidate]
    )

    assert selected is sharper_periodic_candidate


def test_peak_selection_keeps_clear_low_frequency_winner() -> None:
    dominant_low_candidate = _SpectrumPeakCandidate(
        index=3,
        frequency_hz=0.33,
        amplitude=1.0,
        prominence=0.03,
        support_fraction=0.66,
        sharpness=0.16,
        periodicity_strength=0.85,
        harmonic_bonus=0.10,
        ubiquity_penalty=0.0,
        edge_dominance=0.42,
        score=0.612,
    )
    weaker_higher_candidate = _SpectrumPeakCandidate(
        index=7,
        frequency_hz=0.56,
        amplitude=0.78,
        prominence=0.03,
        support_fraction=0.48,
        sharpness=0.23,
        periodicity_strength=0.74,
        harmonic_bonus=0.02,
        ubiquity_penalty=0.0,
        edge_dominance=0.55,
        score=0.484,
    )

    selected = _prefer_sharper_higher_frequency_candidate(
        [dominant_low_candidate, weaker_higher_candidate]
    )

    assert selected is dominant_low_candidate


def test_tighten_periodic_candidate_band_reins_in_promoted_peak_width() -> None:
    promoted_candidate = _SpectrumPeakCandidate(
        index=7,
        frequency_hz=0.56,
        amplitude=0.82,
        prominence=0.04,
        support_fraction=0.58,
        sharpness=0.22,
        periodicity_strength=0.43,
        harmonic_bonus=0.05,
        ubiquity_penalty=0.0,
        edge_dominance=0.30,
        score=0.549,
    )

    low_hz, high_hz = _tighten_periodic_candidate_band(
        low_hz=0.15,
        high_hz=0.90,
        candidate=promoted_candidate,
        min_hz=0.15,
        max_hz=2.0,
    )

    assert 0.40 < low_hz < 0.50
    assert 0.65 < high_hz < 0.75


def test_peak_selection_does_not_promote_edge_dominated_higher_frequency_near_tie() -> None:
    centered_low_candidate = _SpectrumPeakCandidate(
        index=3,
        frequency_hz=0.33,
        amplitude=1.0,
        prominence=0.03,
        support_fraction=0.90,
        sharpness=0.18,
        periodicity_strength=0.35,
        harmonic_bonus=0.08,
        ubiquity_penalty=0.20,
        edge_dominance=0.40,
        score=0.620,
    )
    edge_heavy_higher_candidate = _SpectrumPeakCandidate(
        index=7,
        frequency_hz=0.56,
        amplitude=0.90,
        prominence=0.05,
        support_fraction=0.60,
        sharpness=0.28,
        periodicity_strength=0.50,
        harmonic_bonus=0.04,
        ubiquity_penalty=0.0,
        edge_dominance=0.62,
        score=0.600,
    )

    selected = _prefer_sharper_higher_frequency_candidate(
        [centered_low_candidate, edge_heavy_higher_candidate]
    )

    assert selected is centered_low_candidate


def test_suggest_frequency_band_keeps_engine_style_higher_family_in_band() -> None:
    fps = 18.0
    samples = np.arange(0.0, 24.0, 1.0 / fps, dtype=np.float32)
    motion_trace = (
        0.03 * np.sin(2.0 * math.pi * 0.33 * samples)
        + 0.024 * np.sin(2.0 * math.pi * 0.56 * samples)
        + 0.020 * np.sin(2.0 * math.pi * 0.72 * samples)
    ).astype(np.float32)

    suggestion = suggest_frequency_band_from_motion_trace(motion_trace, fps=fps)

    assert suggestion is not None
    assert suggestion.low_hz <= 0.35
    assert suggestion.high_hz >= 0.65


def test_phase_proxy_promotion_upgrades_suspicious_low_engine_band(monkeypatch) -> None:
    baseline_candidate = _SpectrumPeakCandidate(
        index=3,
        frequency_hz=0.33,
        amplitude=1.0,
        prominence=0.02,
        support_fraction=0.90,
        sharpness=0.12,
        periodicity_strength=0.30,
        harmonic_bonus=0.18,
        ubiquity_penalty=0.45,
        edge_dominance=0.60,
        score=0.56,
    )
    baseline_suggestion = suggest_frequency_band_from_motion_trace(
        np.sin(np.linspace(0.0, 6.0 * math.pi, 96, dtype=np.float32)),
        fps=18.0,
    )
    assert baseline_suggestion is not None
    baseline_suggestion = baseline_suggestion.__class__(
        low_hz=0.2,
        high_hz=1.0,
        peak_hz=0.33,
        analysis_fps=18.0,
        confidence=0.55,
    )

    monkeypatch.setattr(
        "phase_motion_app.core.baseline_band._score_phase_proxy_bands",
        lambda **kwargs: (
            _ProxyBandScore(low_hz=0.2, high_hz=1.0, score=0.12),
            _ProxyBandScore(low_hz=1.0, high_hz=5.0, score=0.18),
        ),
    )

    promoted = _promote_phase_proxy_band_if_warranted(
        baseline_suggestion=baseline_suggestion,
        baseline_candidate=baseline_candidate,
        grayscale_frames=np.zeros((96, 48, 48), dtype=np.float32),
        fps=18.0,
        min_hz=0.15,
    )

    assert promoted.low_hz == 1.0
    assert promoted.high_hz == 5.0
    assert promoted.confidence > baseline_suggestion.confidence


def test_phase_proxy_promotion_keeps_clear_low_breathing_band(monkeypatch) -> None:
    baseline_candidate = _SpectrumPeakCandidate(
        index=3,
        frequency_hz=0.33,
        amplitude=1.0,
        prominence=0.05,
        support_fraction=0.66,
        sharpness=0.22,
        periodicity_strength=0.78,
        harmonic_bonus=0.10,
        ubiquity_penalty=0.0,
        edge_dominance=0.38,
        score=0.62,
    )
    baseline_suggestion = suggest_frequency_band_from_motion_trace(
        np.sin(np.linspace(0.0, 6.0 * math.pi, 96, dtype=np.float32)),
        fps=18.0,
    )
    assert baseline_suggestion is not None
    baseline_suggestion = baseline_suggestion.__class__(
        low_hz=0.15,
        high_hz=0.45,
        peak_hz=0.33,
        analysis_fps=18.0,
        confidence=0.60,
    )

    monkeypatch.setattr(
        "phase_motion_app.core.baseline_band._score_phase_proxy_bands",
        lambda **kwargs: (
            _ProxyBandScore(low_hz=0.15, high_hz=0.45, score=0.14),
            _ProxyBandScore(low_hz=1.0, high_hz=5.0, score=0.20),
        ),
    )

    promoted = _promote_phase_proxy_band_if_warranted(
        baseline_suggestion=baseline_suggestion,
        baseline_candidate=baseline_candidate,
        grayscale_frames=np.zeros((96, 48, 48), dtype=np.float32),
        fps=18.0,
        min_hz=0.15,
    )

    assert promoted == baseline_suggestion
