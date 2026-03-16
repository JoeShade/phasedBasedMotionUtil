"""This file resolves packaged ffmpeg and ffprobe paths so probe, extraction, decode, validation, and encode all use an explicit local toolchain."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise


@dataclass(frozen=True)
class ToolchainPaths:
    """This model carries the resolved executable paths so callers do not have to rediscover them repeatedly."""

    ffmpeg: Path
    ffprobe: Path


def resolve_toolchain() -> ToolchainPaths:
    """Resolve the packaged ffmpeg and ffprobe binaries, allowing explicit environment overrides for later packaging work."""

    env_ffmpeg = os.environ.get("PHASE_MOTION_FFMPEG")
    env_ffprobe = os.environ.get("PHASE_MOTION_FFPROBE")
    if env_ffmpeg and env_ffprobe:
        return ToolchainPaths(ffmpeg=Path(env_ffmpeg), ffprobe=Path(env_ffprobe))

    ffmpeg_exe, ffprobe_exe = get_or_fetch_platform_executables_else_raise()
    return ToolchainPaths(ffmpeg=Path(ffmpeg_exe), ffprobe=Path(ffprobe_exe))
