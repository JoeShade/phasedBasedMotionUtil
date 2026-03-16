"""This file owns lightweight app-state persistence so convenience settings survive restarts without being confused with sidecar reproducibility intent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from phase_motion_app.core.models import DiagnosticLevel, JobIntent

STATE_VERSION = 1


@dataclass(frozen=True)
class GlobalPreferences:
    """This model stores machine-local preferences that shape shell behavior but are not reusable sidecar intent."""

    temp_root: str
    diagnostics_root: str
    diagnostic_level: DiagnosticLevel = DiagnosticLevel.BASIC
    diagnostics_cap_mb: int = 1024
    retention_budget_gb: int = 50

    def to_dict(self) -> dict:
        return {
            "temp_root": self.temp_root,
            "diagnostics_root": self.diagnostics_root,
            "diagnostic_level": self.diagnostic_level.value,
            "diagnostics_cap_mb": self.diagnostics_cap_mb,
            "retention_budget_gb": self.retention_budget_gb,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalPreferences":
        return cls(
            temp_root=str(data["temp_root"]),
            diagnostics_root=str(data["diagnostics_root"]),
            diagnostic_level=DiagnosticLevel(str(data["diagnostic_level"])),
            diagnostics_cap_mb=int(data["diagnostics_cap_mb"]),
            retention_budget_gb=int(data["retention_budget_gb"]),
        )


@dataclass(frozen=True)
class LastUsedSettings:
    """This model stores convenience defaults for the next run without conflating them with the source or prior results."""

    intent: JobIntent
    output_directory: str | None = None
    output_stem: str = "render"
    diagnostic_level: DiagnosticLevel = DiagnosticLevel.BASIC

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "output_directory": self.output_directory,
            "output_stem": self.output_stem,
            "diagnostic_level": self.diagnostic_level.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LastUsedSettings":
        return cls(
            intent=JobIntent.from_dict(dict(data["intent"])),
            output_directory=data.get("output_directory"),
            output_stem=str(data.get("output_stem", "render")),
            diagnostic_level=DiagnosticLevel(str(data.get("diagnostic_level", "basic"))),
        )


@dataclass(frozen=True)
class PersistedAppState:
    """This top-level model keeps global preferences separate from last-used run settings for clarity and safe reloads."""

    preferences: GlobalPreferences
    last_used: LastUsedSettings | None = None
    version: int = STATE_VERSION

    def to_dict(self) -> dict:
        payload = {
            "version": self.version,
            "preferences": self.preferences.to_dict(),
        }
        if self.last_used is not None:
            payload["last_used"] = self.last_used.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "PersistedAppState":
        version = int(data["version"])
        if version != STATE_VERSION:
            raise ValueError(f"Unsupported app-state version: {version}")
        last_used_data = data.get("last_used")
        return cls(
            version=version,
            preferences=GlobalPreferences.from_dict(dict(data["preferences"])),
            last_used=None
            if last_used_data is None
            else LastUsedSettings.from_dict(dict(last_used_data)),
        )


def default_preferences(base_root: Path | None = None) -> GlobalPreferences:
    """Build portable-friendly defaults rooted under the local app data directory."""

    root = base_root or (Path.home() / ".phase_motion_app")
    return GlobalPreferences(
        temp_root=str(root / "scratch"),
        diagnostics_root=str(root / "diagnostics"),
        diagnostic_level=DiagnosticLevel.BASIC,
        diagnostics_cap_mb=1024,
        retention_budget_gb=50,
    )


def load_app_state(path: Path) -> PersistedAppState | None:
    """Load persisted shell state if present, leaving missing files as a normal cold-start case."""

    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PersistedAppState.from_dict(dict(payload))


def save_app_state(path: Path, state: PersistedAppState) -> None:
    """Write convenience shell state atomically enough for a single-user desktop utility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
