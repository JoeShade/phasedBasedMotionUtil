"""This file owns simple float-domain image resizing so the worker can keep one decode path and derive processing/output domains deterministically."""

from __future__ import annotations

import numpy as np

from phase_motion_app.core.models import Resolution


def resize_rgb_frame_bilinear(frame: np.ndarray, target: Resolution) -> np.ndarray:
    """Resize one RGB float frame with bilinear interpolation."""

    source_height, source_width, channels = frame.shape
    if source_width == target.width and source_height == target.height:
        return frame.copy()

    x = np.linspace(0.0, source_width - 1, target.width)
    y = np.linspace(0.0, source_height - 1, target.height)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    rows0 = frame[y0, :, :]
    rows1 = frame[y1, :, :]
    interp_y = rows0 * (1.0 - wy)[:, None, None] + rows1 * wy[:, None, None]
    cols0 = interp_y[:, x0, :]
    cols1 = interp_y[:, x1, :]
    return cols0 * (1.0 - wx)[None, :, None] + cols1 * wx[None, :, None]


def resize_rgb_frames_bilinear(frames: np.ndarray, target: Resolution) -> np.ndarray:
    """Resize a full clip frame-by-frame with the same deterministic resizer."""

    if (
        frames.ndim == 4
        and frames.shape[1] == target.height
        and frames.shape[2] == target.width
    ):
        return frames.astype(np.float32, copy=False)

    source_height = frames.shape[1]
    source_width = frames.shape[2]
    x = np.linspace(0.0, source_width - 1, target.width)
    y = np.linspace(0.0, source_height - 1, target.height)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    # The resize grid is the same for every frame in the chunk, so build it
    # once and apply it across the whole stack instead of looping in Python.
    rows0 = frames[:, y0, :, :]
    rows1 = frames[:, y1, :, :]
    interp_y = rows0 * (1.0 - wy)[None, :, None, None] + rows1 * wy[None, :, None, None]
    cols0 = interp_y[:, :, x0, :]
    cols1 = interp_y[:, :, x1, :]
    resized = cols0 * (1.0 - wx)[None, None, :, None] + cols1 * wx[None, None, :, None]
    return resized.astype(np.float32, copy=False)
