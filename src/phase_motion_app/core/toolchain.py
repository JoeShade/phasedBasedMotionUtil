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
