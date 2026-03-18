"""This file tests ffprobe parsing and command execution so CFR checks use explicit out-of-process probe results rather than guessed metadata."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

from phase_motion_app.core.ffprobe import FfprobeRunner, parse_ffprobe_json


def test_parse_ffprobe_json_marks_vfr_when_rates_disagree() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30000/1001",
                "avg_frame_rate": "24000/1001",
                "nb_frames": "100",
                "duration": "4.170833",
                "codec_name": "h264",
                "bits_per_raw_sample": "8",
            }
        ],
        "format": {"duration": "4.170833"},
    }

    info = parse_ffprobe_json(payload)

    assert info.is_cfr is False
    assert info.audio_stream_count == 0


def test_parse_ffprobe_json_infers_frame_count_when_nb_frames_is_missing() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1280,
                "height": 720,
                "r_frame_rate": "60/1",
                "avg_frame_rate": "60/1",
                "duration": "2.0",
                "codec_name": "hevc",
                "bits_per_raw_sample": "N/A",
            }
        ],
        "format": {"duration": "2.0"},
    }

    info = parse_ffprobe_json(payload)

    assert info.frame_count == 120
    assert info.bit_depth == 8


def test_ffprobe_runner_executes_external_probe_command(tmp_path: Path) -> None:
    script_path = tmp_path / "fake_ffprobe.py"
    script_path.write_text(
        "import json\n"
        "payload = {\n"
        "  'streams': [\n"
        "    {\n"
        "      'codec_type': 'video',\n"
        "      'width': 1280,\n"
        "      'height': 720,\n"
        "      'r_frame_rate': '60/1',\n"
        "      'avg_frame_rate': '60/1',\n"
        "      'nb_frames': '120',\n"
        "      'duration': '2.0',\n"
        "      'codec_name': 'hevc',\n"
        "      'bits_per_raw_sample': '10'\n"
        "    },\n"
        "    {\n"
        "      'codec_type': 'audio'\n"
        "    }\n"
        "  ],\n"
        "  'format': {'duration': '2.0'}\n"
        "}\n"
        "print(json.dumps(payload))\n",
        encoding="utf-8",
    )

    runner = FfprobeRunner(command_prefix=(sys.executable, str(script_path)))
    info = runner.run(tmp_path / "input.mp4")

    assert info.width == 1280
    assert info.height == 720
    assert info.is_cfr is True
    assert info.bit_depth == 10
    assert info.audio_stream_count == 1


def test_parse_ffprobe_json_captures_rotation_and_sample_aspect_ratio() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 1920,
                "height": 1080,
                "r_frame_rate": "30/1",
                "avg_frame_rate": "30/1",
                "nb_frames": "300",
                "duration": "10.0",
                "codec_name": "h264",
                "bits_per_raw_sample": "8",
                "sample_aspect_ratio": "4:3",
                "side_data_list": [
                    {
                        "side_data_type": "Display Matrix",
                        "rotation": -90,
                    }
                ],
            }
        ],
        "format": {"duration": "10.0"},
    }

    info = parse_ffprobe_json(payload)

    assert info.rotation_degrees == -90.0
    assert info.sample_aspect_ratio == pytest.approx(4.0 / 3.0)


def test_parse_ffprobe_json_treats_malformed_rate_fields_as_non_cfr() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 640,
                "height": 360,
                "r_frame_rate": "N/A",
                "avg_frame_rate": "not-a-rate",
                "nb_frames": "0",
                "duration": "1.5",
                "codec_name": "h264",
                "bits_per_raw_sample": "8",
            }
        ],
        "format": {"duration": "1.5"},
    }

    info = parse_ffprobe_json(payload)

    assert info.fps == 0.0
    assert info.avg_fps == 0.0
    assert info.is_cfr is False


def test_parse_ffprobe_json_requires_a_video_stream() -> None:
    with pytest.raises(ValueError, match="video stream"):
        parse_ffprobe_json({"streams": [{"codec_type": "audio"}], "format": {}})

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
