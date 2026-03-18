"""This file owns sidecar schema validation and safe metadata reload so future reruns restore only reusable intent and never machine facts or operator attestations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from phase_motion_app.core.models import JobIntent, SidecarDocument

SCHEMA_VERSION = "1.3.0"
_SEMVER_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")
_SUPPORTED_OLDER_SCHEMA_VERSIONS = {
    "1.0.0": "schema_version 1.0.0 was accepted through the explicit compatibility path.",
    "1.1.0": "schema_version 1.1.0 was accepted through the explicit compatibility path.",
    "1.2.0": "schema_version 1.2.0 was accepted through the explicit compatibility path.",
}

# The design doc defines the domains and required safety checks, but it does not
# publish a full JSON field catalog. The safest small interpretation is to lock
# the top-level boundary and the fields needed for reload, provenance, and
# pre-flight review while still allowing future extension inside each domain.
SIDECAR_SCHEMA_V1: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "intent", "observed_environment", "results"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {
            "type": "string",
            "pattern": r"^\d+\.\d+\.\d+$",
        },
        "intent": {
            "type": "object",
            "required": [
                "phase",
                "processing_resolution",
                "output_resolution",
                "resource_policy",
                "exclusion_zones",
                "mask_feather_px",
                "output_container",
                "requested_output_codec",
            ],
            "additionalProperties": False,
            "properties": {
                "phase": {
                    "type": "object",
                    "required": [
                        "magnification",
                        "low_hz",
                        "high_hz",
                        "pyramid_type",
                        "sigma",
                        "attenuate_other_frequencies",
                    ],
                    "properties": {
                        "magnification": {"type": "number", "exclusiveMinimum": 0},
                        "low_hz": {"type": "number", "exclusiveMinimum": 0},
                        "high_hz": {"type": "number", "exclusiveMinimum": 0},
                        "pyramid_type": {"type": "string", "minLength": 1},
                        "sigma": {"type": "number", "minimum": 0},
                        "attenuate_other_frequencies": {"type": "boolean"},
                    },
                    "additionalProperties": False,
                },
                "processing_resolution": {"$ref": "#/$defs/resolution"},
                "output_resolution": {"$ref": "#/$defs/resolution"},
                "resource_policy": {
                    "type": "string",
                    "enum": ["conservative", "balanced", "aggressive"],
                },
                "exclusion_zones": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/exclusion_zone"},
                },
                "mask_feather_px": {"type": "number", "exclusiveMinimum": 0},
                "analysis": {
                    "type": "object",
                    "required": [
                        "enabled",
                        "minimum_cell_support_fraction",
                        "roi_quality_cutoff",
                        "low_confidence_threshold",
                        "auto_band_count",
                        "band_mode",
                        "manual_bands",
                        "export_advanced_files",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "roi": {
                            "anyOf": [
                                {"type": "null"},
                                {"$ref": "#/$defs/exclusion_zone"},
                            ]
                        },
                        "minimum_cell_support_fraction": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 1,
                        },
                        "roi_quality_cutoff": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "low_confidence_threshold": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "auto_band_count": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                        },
                        "band_mode": {
                            "type": "string",
                            "enum": ["auto", "manual_single", "manual_multi"],
                        },
                        "manual_bands": {
                            "type": "array",
                            "maxItems": 5,
                            "items": {"$ref": "#/$defs/analysis_band"},
                        },
                        "export_advanced_files": {"type": "boolean"},
                    },
                },
                "output_container": {"type": "string", "const": "mp4"},
                "requested_output_codec": {"type": "string", "minLength": 1},
            },
        },
        "observed_environment": {
            "type": "object",
            "required": [
                "app_version",
                "engine_version",
                "platform",
                "diagnostic_level",
                "diagnostics_cap_bytes",
            ],
            "additionalProperties": False,
            "properties": {
                "app_version": {"type": "string", "minLength": 1},
                "engine_version": {"type": "string", "minLength": 1},
                "platform": {"type": "string", "minLength": 1},
                "diagnostic_level": {
                    "type": "string",
                    "enum": ["off", "basic", "detailed", "trace"],
                },
                "diagnostics_cap_bytes": {"type": "integer", "minimum": 0},
                "temp_root": {"type": ["string", "null"]},
                "ffmpeg_version": {"type": ["string", "null"]},
                "ffprobe_version": {"type": ["string", "null"]},
                "scheduler_clamp_threads": {"type": ["integer", "null"], "minimum": 1},
                "effective_thread_limits": {
                    "type": "object",
                    "additionalProperties": {"type": "integer", "minimum": 1},
                },
            },
        },
        "results": {
            "type": "object",
            "required": [
                "render_timestamp_utc",
                "source",
                "preflight",
                "warnings",
                "fallbacks",
                "artifact_paths",
                "diagnostics_summary",
            ],
            "additionalProperties": False,
            "properties": {
                "render_timestamp_utc": {"type": "string", "minLength": 1},
                "source": {
                    "type": "object",
                    "required": [
                        "path",
                        "fingerprint_sha256",
                        "size_bytes",
                        "modified_utc",
                    ],
                    "properties": {
                        "path": {"type": "string", "minLength": 1},
                        "fingerprint_sha256": {
                            "type": "string",
                            "pattern": "^[a-fA-F0-9]{64}$",
                        },
                        "size_bytes": {"type": "integer", "minimum": 0},
                        "modified_utc": {"type": "string", "minLength": 1},
                    },
                    "additionalProperties": False,
                },
                "preflight": {
                    "type": "object",
                    "required": [
                        "source_fps",
                        "source_is_cfr",
                        "nyquist_limit_hz",
                        "warnings",
                        "blockers",
                    ],
                    "properties": {
                        "source_fps": {"type": "number", "exclusiveMinimum": 0},
                        "source_is_cfr": {"type": "boolean"},
                        "working_fps": {"type": "number", "exclusiveMinimum": 0},
                        "working_source_resolution": {"$ref": "#/$defs/resolution"},
                        "normalization_steps": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "nyquist_limit_hz": {"type": "number", "exclusiveMinimum": 0},
                        "warnings": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "blockers": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "additionalProperties": False,
                },
                "warnings": {"type": "array", "items": {"type": "string"}},
                "fallbacks": {"type": "array", "items": {"type": "string"}},
                "artifact_paths": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "diagnostics_summary": {"type": "object"},
                "output_details": {
                    "type": "object",
                    "required": ["codec", "profile", "pixel_format", "color_tags"],
                    "additionalProperties": False,
                    "properties": {
                        "codec": {"type": "string", "minLength": 1},
                        "profile": {"type": ["string", "null"]},
                        "pixel_format": {"type": "string", "minLength": 1},
                        "working_source": {
                            "type": "object",
                            "required": ["width", "height", "fps"],
                            "additionalProperties": False,
                            "properties": {
                                "width": {"type": "integer", "minimum": 1},
                                "height": {"type": "integer", "minimum": 1},
                                "fps": {"type": "number", "exclusiveMinimum": 0},
                            },
                        },
                        "color_tags": {
                            "type": "object",
                            "required": [
                                "color_primaries",
                                "color_transfer",
                                "color_space",
                                "color_range",
                            ],
                            "additionalProperties": False,
                            "properties": {
                                "color_primaries": {"type": "string", "minLength": 1},
                                "color_transfer": {"type": "string", "minLength": 1},
                                "color_space": {"type": "string", "minLength": 1},
                                "color_range": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
                "analysis": {
                    "type": "object",
                    "required": [
                        "enabled",
                        "status",
                        "roi_mode",
                        "roi_label",
                        "reported_peaks",
                        "bands",
                        "artifact_paths",
                        "warnings",
                        "cell_rejection_stats",
                        "auto_band_merge_steps",
                        "suppressed_peak_reasons",
                    ],
                    "additionalProperties": False,
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "status": {
                            "type": "string",
                            "enum": ["disabled", "completed", "warning"],
                        },
                        "roi_mode": {
                            "type": "string",
                            "enum": ["manual", "whole_frame"],
                        },
                        "roi_label": {"type": "string", "minLength": 1},
                        "roi_geometry": {
                            "anyOf": [
                                {"type": "null"},
                                {"$ref": "#/$defs/exclusion_zone"},
                            ]
                        },
                        "analysis_resolution": {
                            "anyOf": [
                                {"type": "null"},
                                {"$ref": "#/$defs/resolution"},
                            ]
                        },
                        "roi_quality_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "confidence_label": {"type": "string", "minLength": 1},
                        "reported_peaks": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/reported_peak"},
                        },
                        "bands": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/generated_band"},
                        },
                        "artifact_paths": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "cell_rejection_stats": {"type": "object"},
                        "auto_band_merge_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "suppressed_peak_reasons": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "heatmap_scale": {
                            "type": "object",
                            "required": [
                                "normalization_method",
                                "lower_percentile",
                                "upper_percentile",
                                "display_min",
                                "display_max",
                                "clipped_cell_count",
                            ],
                            "additionalProperties": False,
                            "properties": {
                                "normalization_method": {"type": "string", "minLength": 1},
                                "lower_percentile": {"type": "number", "minimum": 0, "maximum": 100},
                                "upper_percentile": {"type": "number", "minimum": 0, "maximum": 100},
                                "display_min": {"type": "number", "minimum": 0},
                                "display_max": {"type": "number", "minimum": 0},
                                "clipped_cell_count": {"type": "integer", "minimum": 0},
                            },
                        },
                    },
                },
                "drift_acknowledgement": {
                    "type": "object",
                    "required": [
                        "acknowledged",
                        "reviewed_source_fingerprint_sha256",
                    ],
                    "properties": {
                        "acknowledged": {"type": "boolean"},
                        "reviewed_source_fingerprint_sha256": {
                            "type": "string",
                            "pattern": "^[a-fA-F0-9]{64}$",
                        },
                        "note": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
        },
    },
    "$defs": {
        "resolution": {
            "type": "object",
            "required": ["width", "height"],
            "additionalProperties": False,
            "properties": {
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
            },
        },
        "exclusion_zone": {
            "type": "object",
            "required": ["zone_id", "shape", "x", "y"],
            "additionalProperties": False,
            "properties": {
                "zone_id": {"type": "string", "minLength": 1},
                "shape": {"type": "string", "enum": ["rectangle", "circle"]},
                "mode": {"type": "string", "enum": ["exclude", "include"]},
                "x": {"type": "number", "minimum": 0},
                "y": {"type": "number", "minimum": 0},
                "width": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "height": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "radius": {"type": ["number", "null"], "exclusiveMinimum": 0},
                "label": {"type": ["string", "null"]},
            },
            "allOf": [
                {
                    "if": {"properties": {"shape": {"const": "rectangle"}}},
                    "then": {"required": ["width", "height"]},
                },
                {
                    "if": {"properties": {"shape": {"const": "circle"}}},
                    "then": {"required": ["radius"]},
                },
            ],
        },
        "analysis_band": {
            "type": "object",
            "required": ["band_id", "low_hz", "high_hz"],
            "additionalProperties": False,
            "properties": {
                "band_id": {"type": "string", "minLength": 1},
                "low_hz": {"type": "number", "exclusiveMinimum": 0},
                "high_hz": {"type": "number", "exclusiveMinimum": 0},
            },
        },
        "reported_peak": {
            "type": "object",
            "required": [
                "frequency_hz",
                "amplitude",
                "support_fraction",
                "ranking_score",
            ],
            "additionalProperties": False,
            "properties": {
                "frequency_hz": {"type": "number", "exclusiveMinimum": 0},
                "amplitude": {"type": "number", "minimum": 0},
                "support_fraction": {"type": "number", "minimum": 0, "maximum": 1},
                "ranking_score": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "generated_band": {
            "type": "object",
            "required": ["band_id", "low_hz", "high_hz", "mode"],
            "additionalProperties": False,
            "properties": {
                "band_id": {"type": "string", "minLength": 1},
                "low_hz": {"type": "number", "exclusiveMinimum": 0},
                "high_hz": {"type": "number", "exclusiveMinimum": 0},
                "mode": {
                    "type": "string",
                    "enum": ["auto", "manual_single", "manual_multi"],
                },
                "source_peak_hz": {"type": ["number", "null"], "exclusiveMinimum": 0},
            },
        },
    },
}

_VALIDATOR = Draft202012Validator(SIDECAR_SCHEMA_V1)


class SidecarValidationError(ValueError):
    """This exception groups validation failures so the UI can explain why a sidecar was rejected."""

    def __init__(self, errors: list[str], warnings: list[str] | None = None) -> None:
        super().__init__("Invalid sidecar metadata")
        self.errors = tuple(errors)
        self.warnings = tuple(warnings or [])


@dataclass(frozen=True)
class ValidationResult:
    """This result object keeps validation status explicit so the caller can decide whether to block or warn."""

    status: str
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        return self.status in {"valid", "valid_with_warnings"}


@dataclass(frozen=True)
class ReusableIntentLoadResult:
    """This result object returns the restored intent and the fields that were deliberately ignored during reload."""

    intent: JobIntent
    warnings: tuple[str, ...]


def _format_error(error: Any) -> str:
    path_text = ".".join(str(part) for part in error.absolute_path)
    if path_text:
        return f"{path_text}: {error.message}"
    return str(error.message)


def _semantic_validation_errors(data: dict[str, Any]) -> list[str]:
    """Validate the design-level sidecar rules that require cross-field checks after schema shape validation."""

    errors: list[str] = []
    intent = data["intent"]
    phase = intent["phase"]
    processing_resolution = intent["processing_resolution"]
    output_resolution = intent["output_resolution"]

    if phase["high_hz"] <= phase["low_hz"]:
        errors.append(
            "intent.phase.high_hz: high_hz must be greater than low_hz for a valid exported intent."
        )

    analysis = intent.get("analysis")
    if analysis is not None:
        manual_bands = analysis.get("manual_bands", [])
        if len(manual_bands) > 5:
            errors.append(
                "intent.analysis.manual_bands: at most five manual bands are allowed."
            )
        for index, band in enumerate(manual_bands):
            if band["high_hz"] <= band["low_hz"]:
                errors.append(
                    f"intent.analysis.manual_bands.{index}.high_hz: high_hz must be greater than low_hz."
                )
        band_mode = analysis.get("band_mode")
        if band_mode == "manual_single" and len(manual_bands) < 1:
            errors.append(
                "intent.analysis.manual_bands: manual_single mode requires one manual band."
            )
        if band_mode == "manual_multi" and len(manual_bands) < 1:
            errors.append(
                "intent.analysis.manual_bands: manual_multi mode requires at least one manual band."
            )

    if (
        output_resolution["width"] > processing_resolution["width"]
        or output_resolution["height"] > processing_resolution["height"]
    ):
        errors.append(
            "intent.output_resolution: output resolution must not exceed processing resolution."
        )

    drift_acknowledgement = data["results"].get("drift_acknowledgement")
    if drift_acknowledgement is not None:
        source_fingerprint = data["results"]["source"]["fingerprint_sha256"]
        if (
            drift_acknowledgement["reviewed_source_fingerprint_sha256"]
            != source_fingerprint
        ):
            errors.append(
                "results.drift_acknowledgement.reviewed_source_fingerprint_sha256: drift acknowledgement must match the rendered source fingerprint."
            )

    results = data["results"]
    preflight = results["preflight"]
    diagnostics_summary = results.get("diagnostics_summary", {})
    normalized_source_path = diagnostics_summary.get("normalized_source_path")
    normalization_steps = preflight.get("normalization_steps") or []
    normalization_metadata_required = (
        preflight.get("source_is_cfr") is False
        or normalized_source_path not in (None, "")
        or bool(normalization_steps)
    )
    if normalization_metadata_required:
        if preflight.get("working_fps") is None:
            errors.append(
                "results.preflight.working_fps: normalization metadata must record the working FPS when a derived working source was used."
            )
        if preflight.get("working_source_resolution") is None:
            errors.append(
                "results.preflight.working_source_resolution: normalization metadata must record the working source resolution when a derived working source was used."
            )
        if not normalization_steps:
            errors.append(
                "results.preflight.normalization_steps: normalization metadata must record the applied normalization steps when a derived working source was used."
            )

    return errors


def _parse_semver(version_text: str) -> tuple[int, int, int] | None:
    match = _SEMVER_PATTERN.fullmatch(version_text)
    if match is None:
        return None
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def _validate_schema_compatibility(version_text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    current_version = _parse_semver(SCHEMA_VERSION)
    parsed_version = _parse_semver(version_text)
    if current_version is None or parsed_version is None:
        return (
            ("schema_version: must use semantic versioning in major.minor.patch form.",),
            (),
        )

    if version_text == SCHEMA_VERSION:
        return (), ()

    if version_text in _SUPPORTED_OLDER_SCHEMA_VERSIONS:
        return (), (_SUPPORTED_OLDER_SCHEMA_VERSIONS[version_text],)

    if parsed_version[0] != current_version[0]:
        return (
            (
                f"schema_version: unsupported major version {version_text}; expected "
                f"{current_version[0]}.x sidecars only.",
            ),
            (),
        )

    if parsed_version > current_version:
        return (
            (
                f"schema_version: newer sidecar version {version_text} is not supported by "
                f"this build ({SCHEMA_VERSION}).",
            ),
            (),
        )

    return (
        (
            f"schema_version: older sidecar version {version_text} is not in the supported "
            "compatibility set.",
        ),
        (),
    )


def validate_sidecar_data(data: dict[str, Any]) -> ValidationResult:
    """Validate the sidecar against the v1 schema and return explicit status instead of raising by default."""

    errors = sorted(_VALIDATOR.iter_errors(data), key=lambda item: list(item.absolute_path))
    if errors:
        return ValidationResult(
            status="invalid",
            errors=tuple(_format_error(error) for error in errors),
        )

    compatibility_errors, compatibility_warnings = _validate_schema_compatibility(
        str(data["schema_version"])
    )
    if compatibility_errors:
        return ValidationResult(
            status="invalid",
            errors=compatibility_errors,
            warnings=compatibility_warnings,
        )

    semantic_errors = _semantic_validation_errors(data)
    if semantic_errors:
        return ValidationResult(
            status="invalid",
            errors=tuple(semantic_errors),
            warnings=compatibility_warnings,
        )

    status = "valid_with_warnings" if compatibility_warnings else "valid"
    return ValidationResult(status=status, warnings=compatibility_warnings)


def validate_sidecar_file(path: str | Path) -> ValidationResult:
    """Load and validate one sidecar file from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_sidecar_data(data)


def load_sidecar_document(path: str | Path) -> SidecarDocument:
    """Load one fully validated sidecar document from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validation = validate_sidecar_data(data)
    if not validation.is_valid:
        raise SidecarValidationError(list(validation.errors), list(validation.warnings))
    return SidecarDocument.from_dict(data)


def load_reusable_intent(data: dict[str, Any]) -> ReusableIntentLoadResult:
    """Restore only reusable intent so machine facts, diagnostics settings, and drift attestations never leak into a new run."""

    validation = validate_sidecar_data(data)
    if not validation.is_valid:
        raise SidecarValidationError(list(validation.errors), list(validation.warnings))

    warnings = [
        *validation.warnings,
        "Only the sidecar intent was restored.",
        "observed_environment was preserved for review and not applied as future run settings.",
    ]

    if data.get("results", {}).get("drift_acknowledgement") is not None:
        warnings.append(
            "results.drift_acknowledgement was not reused because operator attestations are tied to a reviewed source state."
        )

    return ReusableIntentLoadResult(
        intent=JobIntent.from_dict(data["intent"]),
        warnings=tuple(warnings),
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
