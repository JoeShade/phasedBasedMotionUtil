"""This file tests clip resizing helpers so same-resolution paths do not waste memory by rebuilding whole clips unnecessarily."""

from __future__ import annotations

import numpy as np

from phase_motion_app.core.image_ops import resize_rgb_frame_bilinear, resize_rgb_frames_bilinear
from phase_motion_app.core.models import Resolution


def test_resize_rgb_frames_bilinear_returns_same_float32_array_when_size_matches() -> None:
    frames = np.zeros((4, 8, 8, 3), dtype=np.float32)

    resized = resize_rgb_frames_bilinear(frames, Resolution(width=8, height=8))

    assert resized is frames


def test_resize_rgb_frames_bilinear_matches_single_frame_resizer() -> None:
    frames = np.linspace(0.0, 1.0, 2 * 5 * 7 * 3, dtype=np.float32).reshape(2, 5, 7, 3)

    resized = resize_rgb_frames_bilinear(frames, Resolution(width=4, height=3))
    expected = np.stack(
        [
            resize_rgb_frame_bilinear(frame, Resolution(width=4, height=3))
            for frame in frames
        ],
        axis=0,
    ).astype(np.float32)

    assert resized.shape == (2, 3, 4, 3)
    assert resized.dtype == np.float32
    assert np.allclose(resized, expected)

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
