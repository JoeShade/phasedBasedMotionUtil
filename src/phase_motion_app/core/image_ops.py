"""This file owns simple float-domain image resizing so the worker can keep one decode path and derive processing/output domains deterministically."""

from __future__ import annotations

import numpy as np

from phase_motion_app.core.acceleration import AccelerationDecision, CpuProcessingBackend
from phase_motion_app.core.models import Resolution

_CPU_BACKEND = CpuProcessingBackend(
    AccelerationDecision(
        requested=False,
        active=False,
        status="cpu_selected",
        detail="Hardware acceleration is disabled. The authoritative CPU path will be used.",
    )
)


def resize_rgb_frame_bilinear(
    frame: np.ndarray,
    target: Resolution,
    *,
    backend: object | None = None,
) -> np.ndarray:
    """Resize one RGB float frame with bilinear interpolation."""

    selected_backend = _CPU_BACKEND if backend is None else backend
    xp = selected_backend.xp
    working = selected_backend.asarray(frame, dtype=xp.float32, copy=False)
    source_height, source_width, channels = working.shape
    if source_width == target.width and source_height == target.height:
        return working.astype(xp.float32, copy=False)

    x = xp.linspace(0.0, source_width - 1, target.width, dtype=xp.float32)
    y = xp.linspace(0.0, source_height - 1, target.height, dtype=xp.float32)
    x0 = xp.floor(x).astype(xp.int32)
    y0 = xp.floor(y).astype(xp.int32)
    x1 = xp.clip(x0 + 1, 0, source_width - 1)
    y1 = xp.clip(y0 + 1, 0, source_height - 1)
    wx = (x - x0).astype(xp.float32)
    wy = (y - y0).astype(xp.float32)

    rows0 = working[y0, :, :]
    rows1 = working[y1, :, :]
    interp_y = rows0 * (1.0 - wy)[:, None, None] + rows1 * wy[:, None, None]
    cols0 = interp_y[:, x0, :]
    cols1 = interp_y[:, x1, :]
    return cols0 * (1.0 - wx)[None, :, None] + cols1 * wx[None, :, None]


def resize_rgb_frames_bilinear(
    frames: np.ndarray,
    target: Resolution,
    *,
    backend: object | None = None,
) -> np.ndarray:
    """Resize a full clip frame-by-frame with the same deterministic resizer."""

    selected_backend = _CPU_BACKEND if backend is None else backend
    xp = selected_backend.xp
    working = selected_backend.asarray(frames, dtype=xp.float32, copy=False)
    if (
        working.ndim == 4
        and working.shape[1] == target.height
        and working.shape[2] == target.width
    ):
        return working.astype(xp.float32, copy=False)

    source_height = working.shape[1]
    source_width = working.shape[2]
    x = xp.linspace(0.0, source_width - 1, target.width, dtype=xp.float32)
    y = xp.linspace(0.0, source_height - 1, target.height, dtype=xp.float32)
    x0 = xp.floor(x).astype(xp.int32)
    y0 = xp.floor(y).astype(xp.int32)
    x1 = xp.clip(x0 + 1, 0, source_width - 1)
    y1 = xp.clip(y0 + 1, 0, source_height - 1)
    wx = (x - x0).astype(xp.float32)
    wy = (y - y0).astype(xp.float32)

    # The resize grid is the same for every frame in the chunk, so build it
    # once and apply it across the whole stack instead of looping in Python.
    rows0 = working[:, y0, :, :]
    rows1 = working[:, y1, :, :]
    interp_y = rows0 * (1.0 - wy)[None, :, None, None] + rows1 * wy[None, :, None, None]
    cols0 = interp_y[:, :, x0, :]
    cols1 = interp_y[:, :, x1, :]
    resized = cols0 * (1.0 - wx)[None, None, :, None] + cols1 * wx[None, None, :, None]
    return resized.astype(xp.float32, copy=False)

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
