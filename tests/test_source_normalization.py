"""This file tests the shared source-normalization plan so shell review and worker render derive the same working cadence and geometry."""

from __future__ import annotations

import pytest

from phase_motion_app.core.ffprobe import FfprobeMediaInfo
from phase_motion_app.core.models import Resolution
from phase_motion_app.core.source_normalization import (
    build_ffmpeg_normalization_filters,
    build_source_normalization_plan,
)


def test_build_source_normalization_plan_marks_vfr_and_non_square_paths() -> None:
    probe = FfprobeMediaInfo(
        width=720,
        height=480,
        fps=30.0,
        avg_fps=29.97,
        is_cfr=False,
        duration_seconds=10.0,
        frame_count=300,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        sample_aspect_ratio=8.0 / 9.0,
    )

    plan = build_source_normalization_plan(probe)

    assert plan.native_resolution == Resolution(720, 480)
    assert plan.working_resolution == Resolution(640, 480)
    assert plan.working_fps == pytest.approx(29.97)
    assert plan.working_frame_count == 300
    assert plan.requires_cfr_normalization is True
    assert plan.requires_square_pixel_normalization is True
    assert plan.normalization_steps == ("cfr_29.970fps", "square_pixels_640x480")


def test_build_ffmpeg_normalization_filters_applies_working_fps_and_geometry() -> None:
    probe = FfprobeMediaInfo(
        width=720,
        height=480,
        fps=30.0,
        avg_fps=29.97,
        is_cfr=False,
        duration_seconds=10.0,
        frame_count=300,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        sample_aspect_ratio=8.0 / 9.0,
    )

    plan = build_source_normalization_plan(probe)
    filters = build_ffmpeg_normalization_filters(
        plan,
        output_resolution=Resolution(320, 240),
    )

    assert filters[0] == "fps=29.970000"
    assert "scale=640:480:flags=lanczos" in filters
    assert "setsar=1" in filters
    assert filters[-1] == "scale=320:240:flags=lanczos"

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
