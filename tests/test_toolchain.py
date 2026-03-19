"""This file tests media-toolchain resolution so explicit overrides fail clearly when misconfigured and packaged fallback remains available."""

from __future__ import annotations

from pathlib import Path

import pytest

import phase_motion_app.core.toolchain as toolchain


def test_resolve_toolchain_prefers_explicit_environment_pair(monkeypatch) -> None:
    monkeypatch.setenv("PHASE_MOTION_FFMPEG", "~/bin/ffmpeg")
    monkeypatch.setenv("PHASE_MOTION_FFPROBE", "~/bin/ffprobe")

    paths = toolchain.resolve_toolchain()

    assert paths.ffmpeg == Path("~/bin/ffmpeg").expanduser()
    assert paths.ffprobe == Path("~/bin/ffprobe").expanduser()


def test_resolve_toolchain_rejects_partial_environment_override(monkeypatch) -> None:
    monkeypatch.setenv("PHASE_MOTION_FFMPEG", "A:/tools/ffmpeg.exe")
    monkeypatch.delenv("PHASE_MOTION_FFPROBE", raising=False)
    monkeypatch.setattr(
        toolchain,
        "get_or_fetch_platform_executables_else_raise",
        lambda: (_ for _ in ()).throw(AssertionError("packaged toolchain fallback should not run")),
    )

    with pytest.raises(ValueError, match="must both be set together"):
        toolchain.resolve_toolchain()


def test_resolve_toolchain_uses_packaged_toolchain_when_overrides_are_absent(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PHASE_MOTION_FFMPEG", raising=False)
    monkeypatch.delenv("PHASE_MOTION_FFPROBE", raising=False)
    monkeypatch.setattr(
        toolchain,
        "get_or_fetch_platform_executables_else_raise",
        lambda: ("A:/packaged/ffmpeg.exe", "A:/packaged/ffprobe.exe"),
    )

    paths = toolchain.resolve_toolchain()

    assert paths.ffmpeg == Path("A:/packaged/ffmpeg.exe")
    assert paths.ffprobe == Path("A:/packaged/ffprobe.exe")

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
