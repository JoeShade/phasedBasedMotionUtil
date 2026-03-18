"""This file tests drift warning policy so reviewed drift risk blocks render until the operator explicitly acknowledges it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phase_motion_app.core.drift import (
    DriftAssessment,
    estimate_global_drift,
    build_drift_acknowledgement,
)
from phase_motion_app.core.media_tools import RgbFrame


def test_estimated_drift_above_threshold_requires_acknowledgement() -> None:
    assessment = DriftAssessment(
        estimated_global_drift_px=3.0,
        advisory_threshold_px=2.0,
        acknowledged=False,
    )

    assert assessment.warning_active is True
    assert assessment.can_render is False


def test_visible_drift_can_render_after_acknowledgement() -> None:
    assessment = DriftAssessment(
        visible_drift_confirmed=True,
        acknowledged=True,
    )

    assert assessment.warning_active is True
    assert assessment.can_render is True


def test_build_drift_acknowledgement_requires_active_warning_ack() -> None:
    assessment = DriftAssessment(visible_drift_confirmed=True, acknowledged=False)

    with pytest.raises(ValueError):
        build_drift_acknowledgement(
            assessment,
            source_fingerprint_sha256="a" * 64,
        )


def test_build_drift_acknowledgement_returns_results_side_attestation() -> None:
    assessment = DriftAssessment(visible_drift_confirmed=True, acknowledged=True)

    attestation = build_drift_acknowledgement(
        assessment,
        source_fingerprint_sha256="b" * 64,
        note="reviewed overlay drift warning",
    )

    assert attestation is not None
    assert attestation.reviewed_source_fingerprint_sha256 == "b" * 64
    assert attestation.note == "reviewed overlay drift warning"


def test_estimate_global_drift_detects_synthetic_translation() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, (128, 160)).astype(np.float32)
    for _ in range(5):
        base = (
            base
            + np.roll(base, 1, axis=0)
            + np.roll(base, -1, axis=0)
            + np.roll(base, 1, axis=1)
            + np.roll(base, -1, axis=1)
        ) / 5.0
    base = ((base - base.min()) / (base.max() - base.min()) * 255.0).astype(np.uint8)
    shifted = np.roll(np.roll(base, -3, axis=0), 5, axis=1)

    estimate = estimate_global_drift(
        _rgb_frame_from_grayscale(base),
        _rgb_frame_from_grayscale(shifted),
    )

    assert estimate is not None
    assert estimate.magnitude_px == pytest.approx(math.hypot(5.0, 3.0), abs=0.5)
    assert estimate.peak_ratio > 2.0


def test_estimate_global_drift_returns_none_for_unreliable_frames() -> None:
    first = np.zeros((96, 128), dtype=np.uint8)
    last = np.zeros((96, 128), dtype=np.uint8)

    estimate = estimate_global_drift(
        _rgb_frame_from_grayscale(first),
        _rgb_frame_from_grayscale(last),
    )

    assert estimate is None


def _rgb_frame_from_grayscale(grayscale: np.ndarray) -> RgbFrame:
    rgb = np.repeat(grayscale[:, :, None], 3, axis=2)
    return RgbFrame(
        width=int(grayscale.shape[1]),
        height=int(grayscale.shape[0]),
        rgb24=rgb.tobytes(),
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
