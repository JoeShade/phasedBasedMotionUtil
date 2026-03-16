"""This file tests app-state persistence so last-used settings survive restarts without crossing the sidecar intent boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase_motion_app.core.models import DiagnosticLevel, JobIntent, PhaseSettings, Resolution, ResourcePolicy
from phase_motion_app.core.settings_store import (
    PersistedAppState,
    LastUsedSettings,
    default_preferences,
    load_app_state,
    save_app_state,
)


def _intent() -> JobIntent:
    return JobIntent(
        phase=PhaseSettings(
            magnification=12.0,
            low_hz=3.0,
            high_hz=8.0,
            pyramid_type="complex_steerable",
            sigma=1.0,
            attenuate_other_frequencies=True,
        ),
        processing_resolution=Resolution(1280, 720),
        output_resolution=Resolution(640, 360),
        resource_policy=ResourcePolicy.BALANCED,
    )


def test_settings_store_round_trip_preserves_intent_and_global_preferences(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "settings.json"
    state = PersistedAppState(
        preferences=default_preferences(tmp_path / "runtime"),
        last_used=LastUsedSettings(
            intent=_intent(),
            output_directory="A:/output",
            output_stem="result",
            diagnostic_level=DiagnosticLevel.DETAILED,
        ),
    )

    save_app_state(state_path, state)
    loaded = load_app_state(state_path)

    assert loaded is not None
    assert loaded.last_used is not None
    assert loaded.last_used.intent.output_resolution == Resolution(640, 360)
    assert loaded.last_used.output_stem == "result"
    assert loaded.preferences.retention_budget_gb == 50


def test_settings_store_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_app_state(tmp_path / "missing.json") is None


def test_settings_store_rejects_unsupported_version(tmp_path: Path) -> None:
    state_path = tmp_path / "settings.json"
    state_path.write_text(
        '{"version":99,"preferences":{"temp_root":"x","diagnostics_root":"y","diagnostic_level":"basic","diagnostics_cap_mb":512,"retention_budget_gb":50}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_app_state(state_path)


def test_default_preferences_build_portable_friendly_roots(tmp_path: Path) -> None:
    preferences = default_preferences(tmp_path / "portable")

    assert preferences.temp_root.endswith("scratch")
    assert preferences.diagnostics_root.endswith("diagnostics")
