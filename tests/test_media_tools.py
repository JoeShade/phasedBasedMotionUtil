"""This file tests the explicit ffmpeg/ffprobe toolchain and raw frame extraction so later render work is grounded in real media I/O."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import phase_motion_app.core.media_tools as media_tools
from phase_motion_app.core.ffprobe import FfprobeRunner
from phase_motion_app.core.media_tools import (
    RawvideoDecodeProcess,
    RawvideoEncodeProcess,
    WorkingSourceTranscodeProcess,
    extract_frame_by_index,
    extract_first_frame,
    extract_last_frame,
)
from phase_motion_app.core.models import Resolution
from phase_motion_app.core.ffprobe import FfprobeMediaInfo
from phase_motion_app.core.toolchain import resolve_toolchain


def _create_two_frame_test_video(path: Path) -> None:
    tools = resolve_toolchain()
    width = 4
    height = 4
    red_frame = bytes([255, 0, 0] * width * height)
    blue_frame = bytes([0, 0, 255] * width * height)
    subprocess.run(
        [
            str(tools.ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "2",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            str(path),
        ],
        input=red_frame + blue_frame,
        check=True,
        capture_output=True,
    )


def _create_three_frame_test_video(path: Path) -> None:
    tools = resolve_toolchain()
    width = 4
    height = 4
    red_frame = bytes([255, 0, 0] * width * height)
    green_frame = bytes([0, 255, 0] * width * height)
    blue_frame = bytes([0, 0, 255] * width * height)
    subprocess.run(
        [
            str(tools.ffmpeg),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "3",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "ultrafast",
            str(path),
        ],
        input=red_frame + green_frame + blue_frame,
        check=True,
        capture_output=True,
    )


def test_resolved_toolchain_paths_exist() -> None:
    tools = resolve_toolchain()
    assert tools.ffmpeg.exists()
    assert tools.ffprobe.exists()


def test_ffprobe_runner_defaults_to_resolved_toolchain(tmp_path: Path) -> None:
    video_path = tmp_path / "probe-test.mkv"
    _create_two_frame_test_video(video_path)

    info = FfprobeRunner().run(video_path)

    assert info.width == 4
    assert info.height == 4
    assert info.frame_count == 2


def test_extract_first_and_last_frames_return_expected_pixels(tmp_path: Path) -> None:
    video_path = tmp_path / "frames-test.mkv"
    _create_two_frame_test_video(video_path)

    first = extract_first_frame(video_path)
    last = extract_last_frame(video_path, target_resolution=Resolution(4, 4))

    assert first.rgb24[:3] == bytes([255, 0, 0])
    assert last.rgb24[:3] == bytes([0, 0, 255])


def test_extract_frame_by_index_disables_ffmpeg_autorotate(monkeypatch) -> None:
    captured_command: list[str] = []

    monkeypatch.setattr(
        media_tools,
        "resolve_toolchain",
        lambda: SimpleNamespace(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
    )

    def fake_run(command, **kwargs):
        captured_command[:] = list(command)
        return SimpleNamespace(stdout=bytes([0] * (4 * 4 * 3)))

    monkeypatch.setattr(media_tools.subprocess, "run", fake_run)

    frame = extract_frame_by_index(
        "source.mp4",
        frame_index=0,
        media_info=FfprobeMediaInfo(
            width=4,
            height=4,
            fps=30.0,
            avg_fps=30.0,
            is_cfr=True,
            duration_seconds=1.0,
            frame_count=1,
            bit_depth=8,
            audio_stream_count=0,
            codec_name="h264",
        ),
    )

    assert frame.width == 4
    assert frame.height == 4
    assert "-noautorotate" in captured_command
    assert captured_command.index("-noautorotate") < captured_command.index("-i")


def test_extract_frame_by_index_applies_cfr_and_square_pixel_normalization_filters(
    monkeypatch,
) -> None:
    captured_command: list[str] = []

    monkeypatch.setattr(
        media_tools,
        "resolve_toolchain",
        lambda: SimpleNamespace(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
    )

    def fake_run(command, **kwargs):
        captured_command[:] = list(command)
        return SimpleNamespace(stdout=bytes([0] * (64 * 48 * 3)))

    monkeypatch.setattr(media_tools.subprocess, "run", fake_run)

    extract_frame_by_index(
        "source.mp4",
        frame_index=0,
        target_resolution=Resolution(64, 48),
        media_info=FfprobeMediaInfo(
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
        ),
    )

    filter_index = captured_command.index("-vf") + 1
    filter_chain = captured_command[filter_index]
    assert "fps=29.970000" in filter_chain
    assert "scale=640:480:flags=lanczos" in filter_chain
    assert "setsar=1" in filter_chain
    assert "scale=64:48:flags=lanczos" in filter_chain


def test_extract_last_frame_retries_backward_when_normalized_boundary_frame_is_missing(
    monkeypatch,
) -> None:
    info = FfprobeMediaInfo(
        width=720,
        height=480,
        fps=30.0,
        avg_fps=29.97,
        is_cfr=False,
        duration_seconds=28.773211,
        frame_count=861,
        bit_depth=8,
        audio_stream_count=0,
        codec_name="h264",
        sample_aspect_ratio=8.0 / 9.0,
    )
    attempts: list[int] = []
    expected = media_tools.RgbFrame(
        width=640,
        height=480,
        rgb24=bytes([0] * (640 * 480 * 3)),
    )

    monkeypatch.setattr(media_tools.FfprobeRunner, "run", lambda self, _source: info)

    def fake_extract_frame_by_index(source_path, *, frame_index, **kwargs):
        attempts.append(frame_index)
        if frame_index == 860:
            raise RuntimeError("Decoded frame size did not match the expected RGB frame size.")
        return expected

    monkeypatch.setattr(media_tools, "extract_frame_by_index", fake_extract_frame_by_index)

    frame = extract_last_frame("source.mp4")

    assert attempts[:2] == [860, 859]
    assert frame == expected


def test_rawvideo_helpers_capture_ffmpeg_progress_counters(tmp_path: Path) -> None:
    source_path = tmp_path / "progress-source.mkv"
    staged_output_path = tmp_path / "progress-output.mp4"
    _create_two_frame_test_video(source_path)

    decoder = RawvideoDecodeProcess(
        source_path=source_path,
        output_resolution=Resolution(4, 4),
    )
    decoded_frames = decoder.read_frames(2)
    decode_exit = decoder.close()

    encoder = RawvideoEncodeProcess(
        staged_output_path=staged_output_path,
        resolution=Resolution(4, 4),
        fps=2.0,
        codec="libx264",
        pixel_format="yuv420p",
        color_tags={
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
        },
    )
    for frame in decoded_frames:
        encoder.write_frame(frame)
    encode_exit = encoder.finish()

    assert decode_exit == 0
    assert encoder.take_progress_counter() is not None
    assert encoder.take_progress_counter() >= 1
    assert decoder.take_progress_counter() is not None
    assert decoder.take_progress_counter() >= 1
    assert encode_exit == 0


def test_rawvideo_decode_process_can_return_one_packed_chunk(tmp_path: Path) -> None:
    source_path = tmp_path / "packed-source.mkv"
    _create_two_frame_test_video(source_path)

    decoder = RawvideoDecodeProcess(
        source_path=source_path,
        output_resolution=Resolution(4, 4),
    )
    packed = decoder.read_chunk_bytes(2)
    decode_exit = decoder.close()

    assert decode_exit == 0
    assert len(packed) == 2 * 4 * 4 * 3
    assert packed[:3] == bytes([255, 0, 0])
    assert packed[-3:] == bytes([0, 0, 255])


def test_rawvideo_encode_process_accepts_packed_multi_frame_write(tmp_path: Path) -> None:
    source_path = tmp_path / "packed-write-source.mkv"
    staged_output_path = tmp_path / "packed-write-output.mp4"
    _create_two_frame_test_video(source_path)

    decoder = RawvideoDecodeProcess(
        source_path=source_path,
        output_resolution=Resolution(4, 4),
    )
    packed = decoder.read_chunk_bytes(2)
    decode_exit = decoder.close()

    encoder = RawvideoEncodeProcess(
        staged_output_path=staged_output_path,
        resolution=Resolution(4, 4),
        fps=2.0,
        codec="libx264",
        pixel_format="yuv420p",
        color_tags={
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
        },
    )
    encoder.write_frames(packed)
    encode_exit = encoder.finish()

    info = FfprobeRunner().run(staged_output_path)

    assert decode_exit == 0
    assert encode_exit == 0
    assert info.frame_count == 2


def test_rawvideo_decode_process_handles_partial_final_chunk_without_buffer_error(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "partial-tail-source.mkv"
    _create_three_frame_test_video(source_path)

    decoder = RawvideoDecodeProcess(
        source_path=source_path,
        output_resolution=Resolution(4, 4),
    )
    first_chunk = decoder.read_chunk_bytes(2)
    second_chunk = decoder.read_chunk_bytes(2)
    decode_exit = decoder.close()

    assert decode_exit == 0
    assert len(first_chunk) == 2 * 4 * 4 * 3
    assert len(second_chunk) == 1 * 4 * 4 * 3
    assert second_chunk[:3] == bytes([0, 0, 255])


class _FakePipe:
    def __init__(self, events: list[str], name: str) -> None:
        self._events = events
        self._name = name

    def read(self, _size: int = -1) -> bytes:
        return b""

    def readline(self) -> bytes:
        return b""

    def write(self, _data: bytes) -> int:
        self._events.append(f"{self._name}_write")
        return 0

    def close(self) -> None:
        self._events.append(f"{self._name}_close")


class _FakeLinePipe(_FakePipe):
    def __init__(self, events: list[str], name: str, lines: list[bytes]) -> None:
        super().__init__(events, name)
        self._lines = list(lines)

    def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _FakeProcess:
    def __init__(self, events: list[str], wait_outcomes: list[object]) -> None:
        self.stdin = _FakePipe(events, "stdin")
        self.stdout = _FakePipe(events, "stdout")
        self.stderr = _FakePipe(events, "stderr")
        self._events = events
        self._wait_outcomes = list(wait_outcomes)

    def wait(self, timeout: float | None = None) -> int:
        self._events.append(f"wait:{timeout}")
        outcome = self._wait_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return int(outcome)

    def terminate(self) -> None:
        self._events.append("terminate")

    def kill(self) -> None:
        self._events.append("kill")

    def poll(self) -> int | None:
        return None


def _patch_fake_popen(monkeypatch, fake_process: _FakeProcess) -> None:
    monkeypatch.setattr(
        media_tools,
        "resolve_toolchain",
        lambda: SimpleNamespace(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
    )
    monkeypatch.setattr(media_tools.subprocess, "Popen", lambda *args, **kwargs: fake_process)


def test_decode_close_escalates_after_bounded_wait(monkeypatch) -> None:
    events: list[str] = []
    fake_process = _FakeProcess(
        events,
        [
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=10),
            0,
        ],
    )
    _patch_fake_popen(monkeypatch, fake_process)

    decoder = RawvideoDecodeProcess(
        source_path="source.mp4",
        output_resolution=Resolution(4, 4),
    )
    exit_code = decoder.close()

    assert exit_code == 0
    assert events.index("stdout_close") < events.index("terminate")


def test_rawvideo_decode_process_disables_ffmpeg_autorotate(monkeypatch) -> None:
    events: list[str] = []
    fake_process = _FakeProcess(events, [0])
    captured_command: list[str] = []

    monkeypatch.setattr(
        media_tools,
        "resolve_toolchain",
        lambda: SimpleNamespace(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
    )

    def fake_popen(command, **kwargs):
        captured_command[:] = list(command)
        return fake_process

    monkeypatch.setattr(media_tools.subprocess, "Popen", fake_popen)

    decoder = RawvideoDecodeProcess(
        source_path="source.mp4",
        output_resolution=Resolution(4, 4),
    )
    decoder.close()

    assert "-noautorotate" in captured_command
    assert captured_command.index("-noautorotate") < captured_command.index("-i")


def test_working_source_transcode_process_builds_ffv1_normalization_command(
    monkeypatch,
) -> None:
    events: list[str] = []
    fake_process = _FakeProcess(events, [0])
    captured_command: list[str] = []

    monkeypatch.setattr(
        media_tools,
        "resolve_toolchain",
        lambda: SimpleNamespace(ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe")),
    )

    def fake_popen(command, **kwargs):
        captured_command[:] = list(command)
        return fake_process

    monkeypatch.setattr(media_tools.subprocess, "Popen", fake_popen)

    process = WorkingSourceTranscodeProcess(
        source_path="source.mp4",
        output_path="scratch/working-source.mkv",
        normalization_plan=media_tools.build_source_normalization_plan(
            FfprobeMediaInfo(
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
        ),
    )
    process.finish()

    assert "-c:v" in captured_command
    assert captured_command[captured_command.index("-c:v") + 1] == "ffv1"
    assert "-noautorotate" in captured_command
    filter_index = captured_command.index("-vf") + 1
    filter_chain = captured_command[filter_index]
    assert "fps=29.970000" in filter_chain
    assert "scale=640:480:flags=lanczos" in filter_chain
    assert "setsar=1" in filter_chain


def test_encode_cancel_closes_stdin_before_terminate_and_kill(monkeypatch) -> None:
    events: list[str] = []
    fake_process = _FakeProcess(
        events,
        [
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5),
            subprocess.TimeoutExpired(cmd="ffmpeg", timeout=2),
            9,
        ],
    )
    _patch_fake_popen(monkeypatch, fake_process)

    encoder = RawvideoEncodeProcess(
        staged_output_path="output.mp4",
        resolution=Resolution(4, 4),
        fps=2.0,
        codec="libx264",
        pixel_format="yuv420p",
        color_tags={
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
        },
    )
    exit_code = encoder.cancel()

    assert exit_code == 9
    assert events.index("stdin_close") < events.index("terminate")
    assert events.index("terminate") < events.index("kill")


def test_rawvideo_encode_process_retains_recent_non_progress_stderr(monkeypatch) -> None:
    events: list[str] = []
    fake_process = _FakeProcess(events, [0])
    fake_process.stderr = _FakeLinePipe(
        events,
        "stderr",
        [
            b"frame=1\n",
            b"[libx264 @ 000001] width not divisible by 2 (853x480)\n",
            b"Error initializing output stream 0:0 -- Error while opening encoder\n",
            b"",
        ],
    )
    _patch_fake_popen(monkeypatch, fake_process)

    encoder = RawvideoEncodeProcess(
        staged_output_path="output.mp4",
        resolution=Resolution(853, 480),
        fps=2.0,
        codec="libx264",
        pixel_format="yuv420p",
        color_tags={
            "color_primaries": "bt709",
            "color_transfer": "bt709",
            "color_space": "bt709",
            "color_range": "tv",
        },
    )
    encoder._stderr_thread.join(timeout=1.0)
    summary = encoder.latest_stderr_summary()
    encoder.cancel()

    assert summary is not None
    assert "width not divisible by 2" in summary
    assert "Error initializing output stream" in summary

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
