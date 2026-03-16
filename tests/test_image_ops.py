"""This file tests clip resizing helpers so same-resolution paths do not waste memory by rebuilding whole clips unnecessarily."""

from __future__ import annotations

import numpy as np

from phase_motion_app.core.image_ops import resize_rgb_frames_bilinear
from phase_motion_app.core.models import Resolution


def test_resize_rgb_frames_bilinear_returns_same_float32_array_when_size_matches() -> None:
    frames = np.zeros((4, 8, 8, 3), dtype=np.float32)

    resized = resize_rgb_frames_bilinear(frames, Resolution(width=8, height=8))

    assert resized is frames
