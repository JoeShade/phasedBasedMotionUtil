"""This file defines the deterministic source-normalization contract so the shell, worker, and media helpers agree on working cadence and square-pixel geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass

from phase_motion_app.core.ffprobe import FfprobeMediaInfo
from phase_motion_app.core.models import Resolution


@dataclass(frozen=True)
class SourceNormalizationPlan:
    """This model records how the native source maps to the working representation used for review and render."""

    native_resolution: Resolution
    working_resolution: Resolution
    native_fps: float
    working_fps: float
    native_frame_count: int
    working_frame_count: int
    source_is_cfr: bool
    source_pixel_aspect_ratio: float
    requires_cfr_normalization: bool
    requires_square_pixel_normalization: bool

    @property
    def requires_working_source_staging(self) -> bool:
        return self.requires_cfr_normalization or self.requires_square_pixel_normalization

    @property
    def normalization_steps(self) -> tuple[str, ...]:
        steps: list[str] = []
        if self.requires_cfr_normalization:
            steps.append(f"cfr_{self.working_fps:.3f}fps")
        if self.requires_square_pixel_normalization:
            steps.append(
                f"square_pixels_{self.working_resolution.width}x{self.working_resolution.height}"
            )
        return tuple(steps)

    @property
    def normalization_messages(self) -> tuple[str, ...]:
        messages: list[str] = []
        if self.requires_cfr_normalization:
            messages.append(
                "Variable frame rate input will be normalized automatically to "
                f"{self.working_fps:.3f} fps before drift review and render."
            )
        if self.requires_square_pixel_normalization:
            messages.append(
                "Non-square pixel geometry will be normalized automatically to "
                f"{self.working_resolution.width} x {self.working_resolution.height} "
                "square-pixel working frames before drift review and render."
            )
        return tuple(messages)


def build_source_normalization_plan(
    probe: FfprobeMediaInfo,
) -> SourceNormalizationPlan:
    """Build the deterministic working-source plan from authoritative probe facts."""

    native_resolution = Resolution(probe.width, probe.height)
    working_resolution = _square_pixel_resolution(
        native_resolution,
        sample_aspect_ratio=probe.sample_aspect_ratio,
    )
    native_fps = probe.avg_fps or probe.fps or 1.0
    working_fps = native_fps
    working_frame_count = probe.frame_count
    if working_frame_count <= 0 and probe.duration_seconds > 0 and working_fps > 0:
        working_frame_count = max(1, math.ceil(probe.duration_seconds * working_fps - 1e-9))
    return SourceNormalizationPlan(
        native_resolution=native_resolution,
        working_resolution=working_resolution,
        native_fps=native_fps,
        working_fps=working_fps,
        native_frame_count=probe.frame_count,
        working_frame_count=working_frame_count,
        source_is_cfr=probe.is_cfr,
        source_pixel_aspect_ratio=probe.sample_aspect_ratio,
        requires_cfr_normalization=not probe.is_cfr,
        requires_square_pixel_normalization=abs(probe.sample_aspect_ratio - 1.0) > 1e-3,
    )


def build_ffmpeg_normalization_filters(
    plan: SourceNormalizationPlan,
    *,
    output_resolution: Resolution | None = None,
    output_fps: float | None = None,
) -> list[str]:
    """Build the deterministic ffmpeg filter chain that produces the working representation."""

    filters: list[str] = []
    effective_fps = output_fps
    if effective_fps is None and plan.requires_cfr_normalization:
        effective_fps = plan.working_fps
    if effective_fps is not None:
        filters.append(f"fps={effective_fps:.6f}")
    if plan.requires_square_pixel_normalization:
        filters.append(
            f"scale={plan.working_resolution.width}:{plan.working_resolution.height}:flags=lanczos"
        )
        filters.append("setsar=1")
    if output_resolution is not None and output_resolution != plan.working_resolution:
        filters.append(
            f"scale={output_resolution.width}:{output_resolution.height}:flags=lanczos"
        )
    return filters


def _square_pixel_resolution(
    resolution: Resolution,
    *,
    sample_aspect_ratio: float,
) -> Resolution:
    if abs(sample_aspect_ratio - 1.0) <= 1e-3:
        return resolution
    normalized_width = max(1, int(round(resolution.width * sample_aspect_ratio)))
    return Resolution(width=normalized_width, height=resolution.height)
