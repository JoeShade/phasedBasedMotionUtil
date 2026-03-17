"""This file owns ffmpeg-based frame extraction and rawvideo streaming helpers so the editor and worker can use one explicit media toolchain."""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from phase_motion_app.core.ffprobe import FfprobeMediaInfo, FfprobeRunner
from phase_motion_app.core.models import Resolution
from phase_motion_app.core.source_normalization import (
    SourceNormalizationPlan,
    build_ffmpeg_normalization_filters,
    build_source_normalization_plan,
)
from phase_motion_app.core.toolchain import resolve_toolchain


@dataclass(frozen=True)
class RgbFrame:
    """This model carries one decoded RGB frame in packed byte form so the UI can build a QImage without reaching into ffmpeg plumbing."""

    width: int
    height: int
    rgb24: bytes


def extract_first_frame(
    source_path: str | Path, *, target_resolution: Resolution | None = None
) -> RgbFrame:
    """Extract the first decoded video frame for drift review."""

    media_info = FfprobeRunner().run(source_path)
    normalization_plan = build_source_normalization_plan(media_info)
    return extract_frame_by_index(
        source_path,
        frame_index=0,
        media_info=media_info,
        normalization_plan=normalization_plan,
        target_resolution=target_resolution,
    )


def extract_last_frame(
    source_path: str | Path, *, target_resolution: Resolution | None = None
) -> RgbFrame:
    """Extract the last decoded video frame for drift review."""

    media_info = FfprobeRunner().run(source_path)
    normalization_plan = build_source_normalization_plan(media_info)
    frame_index = max(normalization_plan.working_frame_count - 1, 0)
    last_error: RuntimeError | None = None
    # VFR-to-CFR normalization can land one frame short of the probe-side count
    # because the final fps-filter output count is timestamp-derived. Step back a
    # few frames before giving up so last-frame review stays usable.
    for backtrack in range(min(frame_index, 8) + 1):
        try:
            return extract_frame_by_index(
                source_path,
                frame_index=max(frame_index - backtrack, 0),
                media_info=media_info,
                normalization_plan=normalization_plan,
                target_resolution=target_resolution,
            )
        except RuntimeError as exc:
            if str(exc) != "Decoded frame size did not match the expected RGB frame size.":
                raise
            last_error = exc
            if frame_index - backtrack <= 0:
                raise
    assert last_error is not None
    raise last_error


def extract_frame_by_index(
    source_path: str | Path,
    *,
    frame_index: int,
    media_info: FfprobeMediaInfo | None = None,
    normalization_plan: SourceNormalizationPlan | None = None,
    target_resolution: Resolution | None = None,
) -> RgbFrame:
    """Extract one specific frame index through ffmpeg so the worker and editor can inspect authoritative decoded pixels."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")

    media_info = media_info or FfprobeRunner().run(source_path)
    normalization_plan = normalization_plan or build_source_normalization_plan(media_info)
    resolution = target_resolution or normalization_plan.working_resolution
    frame_size = resolution.width * resolution.height * 3
    tools = resolve_toolchain()
    filter_chain = build_ffmpeg_normalization_filters(
        normalization_plan,
        output_resolution=resolution,
    )
    filter_chain.append(f"select=eq(n\\,{frame_index})")
    command = [
        str(tools.ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-noautorotate",
        "-i",
        str(source_path),
        "-vf",
        ",".join(filter_chain),
        "-an",
        "-sn",
        "-dn",
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    if len(completed.stdout) != frame_size:
        raise RuntimeError("Decoded frame size did not match the expected RGB frame size.")
    return RgbFrame(
        width=resolution.width,
        height=resolution.height,
        rgb24=completed.stdout,
    )


class RawvideoDecodeProcess:
    """This helper keeps one long-lived decode subprocess open so the worker can read bounded rawvideo chunks under backpressure."""

    def __init__(
        self,
        *,
        source_path: str | Path,
        output_resolution: Resolution,
        normalization_plan: SourceNormalizationPlan | None = None,
    ) -> None:
        self.source_path = Path(source_path)
        self.output_resolution = output_resolution
        self.normalization_plan = normalization_plan
        self.frame_size_bytes = (
            self.output_resolution.width * self.output_resolution.height * 3
        )
        tools = resolve_toolchain()
        if normalization_plan is None:
            filter_chain = [
                f"scale={self.output_resolution.width}:{self.output_resolution.height}:flags=lanczos"
            ]
        else:
            filter_chain = build_ffmpeg_normalization_filters(
                normalization_plan,
                output_resolution=self.output_resolution,
            )
        command = [
            str(tools.ffmpeg),
            "-v",
            "error",
            "-stats_period",
            "0.25",
            "-progress",
            "pipe:2",
            "-nostdin",
            "-noautorotate",
            "-i",
            str(self.source_path),
            "-an",
            "-sn",
            "-dn",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        if filter_chain:
            command[command.index("-an"):command.index("-an")] = [
                "-vf",
                ",".join(filter_chain),
            ]
        # The worker owns backpressure here by reading a bounded number of bytes
        # from stdout at a time instead of asking ffmpeg to spill frames into temp
        # image files or an unbounded queue.
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._last_progress_counter: int | None = None
        self._stderr_thread = threading.Thread(
            target=self._drain_progress_stream,
            name="ffmpeg-decode-progress",
            daemon=True,
        )
        self._stderr_thread.start()

    def read_frames(self, max_frames: int) -> list[bytes]:
        """Read at most one bounded chunk of RGB frames from the long-lived decoder."""

        chunk = self.read_chunk_bytes(max_frames)
        if not chunk:
            return []
        frame_count = len(chunk) // self.frame_size_bytes
        return [
            bytes(
                chunk[
                    frame_index
                    * self.frame_size_bytes : (frame_index + 1)
                    * self.frame_size_bytes
                ]
            )
            for frame_index in range(frame_count)
        ]

    def read_chunk_bytes(self, max_frames: int) -> bytearray:
        """Read one bounded chunk as one packed byte buffer so callers can avoid per-frame Python joins."""

        if max_frames <= 0:
            return bytearray()
        assert self.process.stdout is not None
        target_bytes = self.frame_size_bytes * max_frames
        buffer = bytearray(target_bytes)
        view = memoryview(buffer)
        total_bytes = 0
        while total_bytes < target_bytes:
            reader = getattr(self.process.stdout, "readinto", None)
            if callable(reader):
                read_count = reader(view[total_bytes:])
            else:
                chunk = self.process.stdout.read(target_bytes - total_bytes)
                read_count = len(chunk)
                if read_count:
                    view[total_bytes : total_bytes + read_count] = chunk
            if not read_count:
                break
            total_bytes += read_count
        complete_bytes = (total_bytes // self.frame_size_bytes) * self.frame_size_bytes
        # Drop the exported view before returning a shortened tail chunk. Python
        # forbids resizing a bytearray while a memoryview still references it.
        view.release()
        if complete_bytes <= 0:
            return bytearray()
        if complete_bytes == target_bytes:
            return buffer
        return bytearray(buffer[:complete_bytes])

    def take_progress_counter(self) -> int | None:
        """Return the latest child-reported frame counter when ffmpeg has emitted one."""

        return self._last_progress_counter

    def _drain_progress_stream(self) -> None:
        assert self.process.stderr is not None
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            counter = _parse_progress_counter(line)
            if counter is not None:
                self._last_progress_counter = counter

    def close(self) -> int:
        """Close the decode process and return its exit code."""

        return self._shutdown(wait_timeout_seconds=10.0)

    def cancel(self) -> int:
        """Stop the decode child promptly when the worker abandons the active read path."""

        return self._shutdown(wait_timeout_seconds=2.0)

    def _shutdown(self, *, wait_timeout_seconds: float) -> int:
        if self.process.stdout is not None:
            _close_pipe(self.process.stdout)
        exit_code = _wait_for_exit_with_escalation(
            self.process,
            wait_timeout_seconds=wait_timeout_seconds,
            kill_timeout_seconds=2.0,
        )
        self._stderr_thread.join(timeout=1.0)
        if self.process.stderr is not None:
            _close_pipe(self.process.stderr)
        return exit_code


class RawvideoEncodeProcess:
    """This helper streams RGB frames into ffmpeg and keeps the staged MP4 closed only after the encoder exits cleanly."""

    def __init__(
        self,
        *,
        staged_output_path: str | Path,
        resolution: Resolution,
        fps: float,
        codec: str,
        pixel_format: str,
        color_tags: dict[str, str],
    ) -> None:
        self.staged_output_path = Path(staged_output_path)
        self.resolution = resolution
        self.fps = fps
        tools = resolve_toolchain()
        command = [
            str(tools.ffmpeg),
            "-y",
            "-v",
            "error",
            "-stats_period",
            "0.25",
            "-progress",
            "pipe:2",
            "-nostdin",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{resolution.width}x{resolution.height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            codec,
            "-pix_fmt",
            pixel_format,
            "-color_primaries",
            color_tags["color_primaries"],
            "-color_trc",
            color_tags["color_transfer"],
            "-colorspace",
            color_tags["color_space"],
            "-color_range",
            color_tags["color_range"],
            str(self.staged_output_path),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._last_progress_counter: int | None = None
        self._recent_stderr_lines: list[str] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_progress_stream,
            name="ffmpeg-encode-progress",
            daemon=True,
        )
        self._stderr_thread.start()

    def write_frame(self, frame_rgb24: bytes) -> None:
        """Write one packed RGB frame to the encoder."""

        self.write_frames(frame_rgb24)

    def write_frames(self, frames_rgb24: bytes | bytearray | memoryview) -> None:
        """Write one packed RGB byte run so callers can batch multiple frames per Python call."""

        assert self.process.stdin is not None
        self.process.stdin.write(frames_rgb24)

    def take_progress_counter(self) -> int | None:
        """Return the latest child-reported encoded-frame counter when ffmpeg has emitted one."""

        return self._last_progress_counter

    def latest_stderr_summary(self) -> str | None:
        """Return the last few non-progress stderr lines so worker failures can surface ffmpeg's actual reason."""

        if not self._recent_stderr_lines:
            return None
        return " | ".join(self._recent_stderr_lines)

    def _drain_progress_stream(self) -> None:
        assert self.process.stderr is not None
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            counter = _parse_progress_counter_text(text)
            if counter is not None:
                self._last_progress_counter = counter
                continue
            self._remember_stderr_line(text)

    def _remember_stderr_line(self, text: str) -> None:
        if self._recent_stderr_lines and self._recent_stderr_lines[-1] == text:
            return
        self._recent_stderr_lines.append(text)
        if len(self._recent_stderr_lines) > 6:
            self._recent_stderr_lines.pop(0)

    def finish(self) -> int:
        """Close encoder input and wait for the final exit code."""

        return self._shutdown(wait_timeout_seconds=30.0)

    def cancel(self) -> int:
        """Close encoder input first, then allow a short bounded drain before escalation."""

        return self._shutdown(wait_timeout_seconds=5.0)

    def _shutdown(self, *, wait_timeout_seconds: float) -> int:
        if self.process.stdin is not None:
            _close_pipe(self.process.stdin)
        if self.process.stdout is not None:
            _close_pipe(self.process.stdout)
        exit_code = _wait_for_exit_with_escalation(
            self.process,
            wait_timeout_seconds=wait_timeout_seconds,
            kill_timeout_seconds=2.0,
        )
        self._stderr_thread.join(timeout=1.0)
        if self.process.stderr is not None:
            _close_pipe(self.process.stderr)
        return exit_code


class WorkingSourceTranscodeProcess:
    """This helper stages a deterministic normalized working source into scratch so cadence and pixel geometry are fixed before render decode."""

    def __init__(
        self,
        *,
        source_path: str | Path,
        output_path: str | Path,
        normalization_plan: SourceNormalizationPlan,
    ) -> None:
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.normalization_plan = normalization_plan
        tools = resolve_toolchain()
        filter_chain = build_ffmpeg_normalization_filters(
            normalization_plan,
            output_resolution=normalization_plan.working_resolution,
        )
        command = [
            str(tools.ffmpeg),
            "-y",
            "-v",
            "error",
            "-stats_period",
            "0.25",
            "-progress",
            "pipe:2",
            "-nostdin",
            "-noautorotate",
            "-i",
            str(self.source_path),
            "-an",
            "-sn",
            "-dn",
            "-map_metadata",
            "-1",
            "-c:v",
            "ffv1",
            "-level",
            "3",
            "-pix_fmt",
            "rgb24",
            str(self.output_path),
        ]
        if filter_chain:
            command[command.index("-an"):command.index("-an")] = [
                "-vf",
                ",".join(filter_chain),
            ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._last_progress_counter: int | None = None
        self._stderr_thread = threading.Thread(
            target=self._drain_progress_stream,
            name="ffmpeg-normalize-progress",
            daemon=True,
        )
        self._stderr_thread.start()

    def take_progress_counter(self) -> int | None:
        """Return the latest child-reported normalized frame counter when ffmpeg has emitted one."""

        return self._last_progress_counter

    def _drain_progress_stream(self) -> None:
        assert self.process.stderr is not None
        while True:
            line = self.process.stderr.readline()
            if not line:
                break
            counter = _parse_progress_counter(line)
            if counter is not None:
                self._last_progress_counter = counter

    def finish(self) -> int:
        """Wait for the staged working source to be fully written."""

        return self._shutdown(wait_timeout_seconds=60.0)

    def cancel(self) -> int:
        """Stop normalization promptly if the worker abandons the run."""

        return self._shutdown(wait_timeout_seconds=5.0)

    def _shutdown(self, *, wait_timeout_seconds: float) -> int:
        if self.process.stdout is not None:
            _close_pipe(self.process.stdout)
        exit_code = _wait_for_exit_with_escalation(
            self.process,
            wait_timeout_seconds=wait_timeout_seconds,
            kill_timeout_seconds=2.0,
        )
        self._stderr_thread.join(timeout=1.0)
        if self.process.stderr is not None:
            _close_pipe(self.process.stderr)
        return exit_code


def _close_pipe(pipe) -> None:
    try:
        pipe.close()
    except (BrokenPipeError, OSError, ValueError):
        pass


def _wait_for_exit_with_escalation(
    process: subprocess.Popen,
    *,
    wait_timeout_seconds: float,
    kill_timeout_seconds: float,
) -> int:
    try:
        return process.wait(timeout=wait_timeout_seconds)
    except subprocess.TimeoutExpired:
        # Closing the relevant pipe is the first step. If the child still does
        # not exit within the bounded grace window, escalate explicitly.
        process.terminate()

    try:
        return process.wait(timeout=kill_timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait(timeout=kill_timeout_seconds)


def _parse_progress_counter(line: bytes) -> int | None:
    """Parse one ffmpeg progress line and ignore unrelated stderr text safely."""

    text = line.decode("utf-8", errors="ignore").strip()
    return _parse_progress_counter_text(text)


def _parse_progress_counter_text(text: str) -> int | None:
    """Parse one decoded ffmpeg progress line and ignore unrelated stderr text safely."""

    if not text.startswith("frame="):
        return None
    _, value = text.split("=", 1)
    try:
        return int(value)
    except ValueError:
        return None
