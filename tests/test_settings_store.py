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
    migrate_legacy_temp_root,
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
        output_resolution=Resolution(1280, 720),
        resource_policy=ResourcePolicy.BALANCED,
        hardware_acceleration_enabled=True,
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
    assert loaded.last_used.intent.output_resolution == Resolution(1280, 720)
    assert loaded.last_used.intent.hardware_acceleration_enabled is True
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

    assert preferences.temp_root.endswith("temp")
    assert preferences.diagnostics_root.endswith("diagnostics")


def test_migrate_legacy_temp_root_rehomes_runtime_scratch(tmp_path: Path) -> None:
    legacy_preferences = default_preferences(tmp_path)
    legacy_preferences = legacy_preferences.__class__(
        temp_root=str(tmp_path / "scratch"),
        diagnostics_root=legacy_preferences.diagnostics_root,
        diagnostic_level=legacy_preferences.diagnostic_level,
        diagnostics_cap_mb=legacy_preferences.diagnostics_cap_mb,
        retention_budget_gb=legacy_preferences.retention_budget_gb,
    )

    migrated = migrate_legacy_temp_root(legacy_preferences, tmp_path)

    assert migrated.temp_root == str(tmp_path / "temp")
    assert migrated.diagnostics_root == legacy_preferences.diagnostics_root

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
