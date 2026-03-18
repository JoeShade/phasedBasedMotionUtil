"""This file wraps ffprobe so probe data can be gathered out of process and turned into conservative source metadata for pre-flight."""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from phase_motion_app.core.toolchain import resolve_toolchain


def _parse_fraction(value: str | None, default: float = 0.0) -> float:
    if value in {None, "", "N/A"}:
        return default
    separator = "/" if "/" in value else ":" if ":" in value else None
    if separator is None:
        return _parse_float(value, default)
    numerator_text, denominator_text = value.split(separator, 1)
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except (TypeError, ValueError):
        return default
    if denominator == 0:
        return default
    return numerator / denominator


def _parse_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(float(value)) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_ratio(value: str | None, default: float = 1.0) -> float:
    if value in {None, "", "0:1", "1:0", "N/A"}:
        return default
    separator = ":" if ":" in value else "/" if "/" in value else None
    if separator is None:
        return _parse_float(value, default)
    numerator_text, denominator_text = value.split(separator, 1)
    try:
        numerator = float(numerator_text)
        denominator = float(denominator_text)
    except ValueError:
        return default
    if denominator == 0:
        return default
    return numerator / denominator


def _parse_rotation_degrees(video_stream: dict) -> float | None:
    for side_data in video_stream.get("side_data_list", []):
        if side_data.get("side_data_type") == "Display Matrix":
            rotation = side_data.get("rotation")
            if rotation is None:
                return None
            try:
                return float(rotation)
            except (TypeError, ValueError):
                return None
    tags = video_stream.get("tags", {})
    rotation_text = tags.get("rotate")
    if rotation_text is None:
        return None
    try:
        return float(rotation_text)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class FfprobeMediaInfo:
    """This model carries the source facts that matter most for admission and logging."""

    width: int
    height: int
    fps: float
    avg_fps: float
    is_cfr: bool
    duration_seconds: float
    frame_count: int
    bit_depth: int
    audio_stream_count: int
    codec_name: str
    profile: str = ""
    pixel_format: str = ""
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_space: str | None = None
    color_range: str | None = None
    rotation_degrees: float | None = None
    sample_aspect_ratio: float = 1.0


@dataclass(frozen=True)
class FfprobeRunner:
    """This runner executes ffprobe or a compatible wrapper command and parses its JSON output."""

    command_prefix: tuple[str, ...] | None = None

    def run(self, source_path: str | Path) -> FfprobeMediaInfo:
        command_prefix = self.command_prefix
        if command_prefix is None:
            command_prefix = (str(resolve_toolchain().ffprobe),)
        command = [
            *command_prefix,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(source_path),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        return parse_ffprobe_json(payload)


def parse_ffprobe_json(payload: dict) -> FfprobeMediaInfo:
    """Parse ffprobe JSON using a conservative CFR rule: if the rates disagree or are missing, treat the source as not safely CFR."""

    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("ffprobe payload did not contain a video stream.")
    audio_stream_count = sum(
        1
        for stream in payload.get("streams", [])
        if stream.get("codec_type") == "audio"
    )
    r_frame_rate = _parse_fraction(video_stream.get("r_frame_rate", "0/0"))
    avg_frame_rate = _parse_fraction(video_stream.get("avg_frame_rate", "0/0"))
    is_cfr = (
        r_frame_rate > 0
        and avg_frame_rate > 0
        and abs(r_frame_rate - avg_frame_rate) < 1e-6
    )
    bit_depth_text = (
        video_stream.get("bits_per_raw_sample")
        or video_stream.get("bits_per_sample")
        or "8"
    )
    frame_count_text = video_stream.get("nb_frames") or "0"
    duration_text = (
        video_stream.get("duration")
        or payload.get("format", {}).get("duration")
        or "0"
    )
    duration_seconds = _parse_float(duration_text, 0.0)
    frame_count = _parse_int(frame_count_text, 0)
    inferred_fps = avg_frame_rate or r_frame_rate
    sample_aspect_ratio = _parse_ratio(video_stream.get("sample_aspect_ratio"), 1.0)
    rotation_degrees = _parse_rotation_degrees(video_stream)
    if frame_count <= 0 and duration_seconds > 0 and inferred_fps > 0:
        # ffprobe often omits nb_frames for some containers. Infer a conservative
        # count so storage and RAM admission logic does not silently under-budget.
        frame_count = max(1, math.ceil(duration_seconds * inferred_fps - 1e-9))
    if duration_seconds <= 0 and frame_count > 0 and inferred_fps > 0:
        duration_seconds = frame_count / inferred_fps
    bit_depth = _parse_int(bit_depth_text, 8)
    if bit_depth <= 0:
        bit_depth = 8

    return FfprobeMediaInfo(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=r_frame_rate,
        avg_fps=avg_frame_rate,
        is_cfr=is_cfr,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
        bit_depth=bit_depth,
        audio_stream_count=audio_stream_count,
        codec_name=str(video_stream.get("codec_name", "")),
        profile=str(video_stream.get("profile", "")),
        pixel_format=str(video_stream.get("pix_fmt", "")),
        color_primaries=video_stream.get("color_primaries"),
        color_transfer=video_stream.get("color_transfer"),
        color_space=video_stream.get("color_space"),
        color_range=video_stream.get("color_range"),
        rotation_degrees=rotation_degrees,
        sample_aspect_ratio=sample_aspect_ratio,
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
