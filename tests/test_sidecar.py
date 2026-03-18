"""This file tests the sidecar schema boundary so metadata import stays strict, reproducible, and safe before the GUI starts trusting it."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from phase_motion_app.core.models import ResourcePolicy, ZoneMode
from phase_motion_app.core.sidecar import (
    SCHEMA_VERSION,
    SidecarValidationError,
    load_sidecar_document,
    load_reusable_intent,
    validate_sidecar_data,
)


def _valid_sidecar() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "intent": {
            "phase": {
                "magnification": 20.0,
                "low_hz": 5.0,
                "high_hz": 14.0,
                "pyramid_type": "complex_steerable",
                "sigma": 1.0,
                "attenuate_other_frequencies": True,
            },
            "processing_resolution": {"width": 1280, "height": 720},
            "output_resolution": {"width": 640, "height": 360},
            "resource_policy": "balanced",
            "exclusion_zones": [
                {
                    "zone_id": "zone-1",
                    "shape": "rectangle",
                    "mode": "exclude",
                    "x": 120.0,
                    "y": 140.0,
                    "width": 240.0,
                    "height": 180.0,
                    "label": "background fan",
                }
            ],
            "mask_feather_px": 4.0,
            "output_container": "mp4",
            "requested_output_codec": "prefer_hevc_main10",
        },
        "observed_environment": {
            "app_version": "0.1.0",
            "engine_version": "0.1.0",
            "platform": "Windows-11",
            "diagnostic_level": "trace",
            "diagnostics_cap_bytes": 1073741824,
            "temp_root": "A:/temp",
            "ffmpeg_version": "6.1",
            "ffprobe_version": "6.1",
            "scheduler_clamp_threads": 7,
            "effective_thread_limits": {"ffmpeg": 4, "opencv": 2},
        },
        "results": {
            "render_timestamp_utc": "2026-03-16T05:20:00Z",
            "source": {
                "path": "A:/videos/source.mp4",
                "fingerprint_sha256": "a" * 64,
                "size_bytes": 1234567,
                "modified_utc": "2026-03-16T04:55:00Z",
            },
            "preflight": {
                "source_fps": 59.94,
                "source_is_cfr": True,
                "nyquist_limit_hz": 29.97,
                "warnings": ["High band is close to Nyquist."],
                "blockers": [],
            },
            "warnings": ["Using heuristic SDR acceptance path."],
            "fallbacks": ["Stored output as H.264 High."],
            "artifact_paths": {"mp4": "A:/out/output.mp4", "sidecar": "A:/out/output.json"},
            "diagnostics_summary": {"bundle_written": True},
            "output_details": {
                "codec": "h264",
                "profile": "High",
                "pixel_format": "yuv420p",
                "color_tags": {
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "color_space": "bt709",
                    "color_range": "tv",
                },
            },
            "drift_acknowledgement": {
                "acknowledged": True,
                "reviewed_source_fingerprint_sha256": "a" * 64,
                "note": "slight tripod creep near frame edge",
            },
        },
    }


def test_sidecar_schema_validation_accepts_valid_document() -> None:
    validation = validate_sidecar_data(_valid_sidecar())
    assert validation.status == "valid"
    assert validation.errors == ()


def test_sidecar_schema_validation_accepts_normalization_metadata(tmp_path) -> None:
    sidecar = _valid_sidecar()
    sidecar["results"]["preflight"]["source_is_cfr"] = False
    sidecar["results"]["diagnostics_summary"]["normalized_source_path"] = (
        "A:/scratch/working-source.mkv"
    )
    sidecar["results"]["preflight"]["working_fps"] = 59.94
    sidecar["results"]["preflight"]["working_source_resolution"] = {
        "width": 1440,
        "height": 1080,
    }
    sidecar["results"]["preflight"]["normalization_steps"] = [
        "square_pixels_1440x1080",
    ]
    sidecar["results"]["output_details"]["working_source"] = {
        "width": 1440,
        "height": 1080,
        "fps": 59.94,
    }

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "valid"
    sidecar_path = tmp_path / "normalization-sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    document = load_sidecar_document(sidecar_path)
    assert document.results.preflight.working_fps == 59.94
    assert document.results.preflight.working_source_resolution.width == 1440


def test_sidecar_schema_validation_rejects_missing_normalization_metadata() -> None:
    sidecar = _valid_sidecar()
    sidecar["results"]["preflight"]["source_is_cfr"] = False
    sidecar["results"]["diagnostics_summary"]["normalized_source_path"] = (
        "A:/scratch/working-source.mkv"
    )

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("working_fps" in error for error in validation.errors)
    assert any("working_source_resolution" in error for error in validation.errors)
    assert any("normalization_steps" in error for error in validation.errors)


def test_sidecar_schema_validation_accepts_quantitative_analysis_metadata() -> None:
    sidecar = _valid_sidecar()
    sidecar["intent"]["analysis"] = {
        "enabled": True,
        "roi": None,
        "minimum_cell_support_fraction": 0.35,
        "roi_quality_cutoff": 0.45,
        "low_confidence_threshold": 0.35,
        "auto_band_count": 5,
        "band_mode": "auto",
        "manual_bands": [],
        "export_advanced_files": True,
    }
    sidecar["results"]["analysis"] = {
        "enabled": True,
        "status": "completed",
        "roi_mode": "whole_frame",
        "roi_label": "Whole-frame ROI",
        "reported_peaks": [],
        "bands": [],
        "artifact_paths": {"roi_metrics": "A:/out/roi_metrics.csv"},
        "warnings": [],
        "cell_rejection_stats": {
            "valid_cell_count": 4,
            "rejected_cell_count": 1,
            "rejection_penalty_contribution": 0.2,
        },
        "auto_band_merge_steps": [],
        "suppressed_peak_reasons": [],
        "heatmap_scale": {
            "normalization_method": "robust_percentile",
            "lower_percentile": 5.0,
            "upper_percentile": 95.0,
            "display_min": 0.0,
            "display_max": 0.8,
            "clipped_cell_count": 2,
        },
    }

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "valid"


def test_sidecar_schema_validation_rejects_malformed_output_details() -> None:
    sidecar = _valid_sidecar()
    sidecar["results"]["output_details"]["color_tags"]["color_range"] = 7

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("output_details.color_tags.color_range" in error for error in validation.errors)


def test_sidecar_schema_validation_rejects_missing_required_domains() -> None:
    sidecar = _valid_sidecar()
    del sidecar["results"]

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("results" in error for error in validation.errors)


def test_sidecar_schema_validation_rejects_invalid_exclusion_zone_coordinates() -> None:
    sidecar = _valid_sidecar()
    sidecar["intent"]["exclusion_zones"][0]["x"] = -1

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("exclusion_zones.0.x" in error for error in validation.errors)


def test_sidecar_schema_validation_rejects_unknown_intent_fields() -> None:
    sidecar = _valid_sidecar()
    sidecar["intent"]["drift_acknowledgement"] = {"acknowledged": True}

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("Additional properties" in error for error in validation.errors)


def test_sidecar_schema_validation_rejects_upscaled_output_intent() -> None:
    sidecar = _valid_sidecar()
    sidecar["intent"]["output_resolution"] = {"width": 1920, "height": 1080}

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("output_resolution" in error for error in validation.errors)


def test_sidecar_schema_validation_rejects_mismatched_drift_ack_source() -> None:
    sidecar = _valid_sidecar()
    sidecar["results"]["drift_acknowledgement"][
        "reviewed_source_fingerprint_sha256"
    ] = "b" * 64

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("drift_acknowledgement" in error for error in validation.errors)


def test_sidecar_schema_validation_rejects_unsupported_schema_version() -> None:
    sidecar = _valid_sidecar()
    sidecar["schema_version"] = "2.0.0"

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("schema_version" in error for error in validation.errors)


def test_sidecar_schema_validation_accepts_supported_older_version_with_warning() -> None:
    sidecar = _valid_sidecar()
    sidecar["schema_version"] = "1.0.0"

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "valid_with_warnings"
    assert validation.errors == ()
    assert any("compatibility path" in warning for warning in validation.warnings)


def test_load_reusable_intent_restores_only_intent() -> None:
    result = load_reusable_intent(_valid_sidecar())

    assert result.intent.resource_policy is ResourcePolicy.BALANCED
    assert result.intent.output_container == "mp4"
    assert result.intent.mask_feather_px == 4.0
    assert result.intent.exclusion_zones[0].label == "background fan"
    assert result.intent.exclusion_zones[0].mode is ZoneMode.EXCLUDE
    assert any("Only the sidecar intent was restored." == warning for warning in result.warnings)


def test_load_reusable_intent_does_not_import_observed_environment_runtime_choices() -> None:
    result = load_reusable_intent(_valid_sidecar())

    assert result.intent.requested_output_codec == "prefer_hevc_main10"
    assert all("scheduler_clamp_threads" not in warning for warning in result.warnings)
    assert any("observed_environment" in warning for warning in result.warnings)


def test_drift_acknowledgement_is_excluded_from_reusable_intent() -> None:
    result = load_reusable_intent(_valid_sidecar())

    assert not hasattr(result.intent, "drift_acknowledgement")
    assert any("drift_acknowledgement" in warning for warning in result.warnings)


def test_invalid_sidecar_raises_when_loading_reusable_intent() -> None:
    sidecar = deepcopy(_valid_sidecar())
    sidecar["results"]["preflight"]["source_is_cfr"] = "maybe"

    with pytest.raises(SidecarValidationError) as exc_info:
        load_reusable_intent(sidecar)

    assert any("source_is_cfr" in error for error in exc_info.value.errors)


def test_load_reusable_intent_preserves_compatibility_warning() -> None:
    sidecar = _valid_sidecar()
    sidecar["schema_version"] = "1.0.0"

    result = load_reusable_intent(sidecar)

    assert any("compatibility path" in warning for warning in result.warnings)


def test_sidecar_schema_validation_accepts_current_include_mask_zone() -> None:
    sidecar = _valid_sidecar()
    sidecar["intent"]["exclusion_zones"][0]["mode"] = "include"

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "valid"
    result = load_reusable_intent(sidecar)
    assert result.intent.exclusion_zones[0].mode is ZoneMode.INCLUDE


def test_sidecar_schema_validation_accepts_older_version_without_zone_mode() -> None:
    sidecar = _valid_sidecar()
    sidecar["schema_version"] = "1.1.0"
    del sidecar["intent"]["exclusion_zones"][0]["mode"]

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "valid_with_warnings"
    assert any("compatibility path" in warning for warning in validation.warnings)


def test_sidecar_schema_validation_rejects_newer_minor_version() -> None:
    sidecar = _valid_sidecar()
    sidecar["schema_version"] = "1.4.0"

    validation = validate_sidecar_data(sidecar)

    assert validation.status == "invalid"
    assert any("newer sidecar version" in error for error in validation.errors)


def test_load_sidecar_document_preserves_results_output_details(tmp_path) -> None:
    sidecar_path = tmp_path / "sidecar.json"
    sidecar_path.write_text(json.dumps(_valid_sidecar()), encoding="utf-8")

    document = load_sidecar_document(sidecar_path)

    assert document.results.output_details["codec"] == "h264"
    assert document.to_dict()["results"]["output_details"]["profile"] == "High"

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
