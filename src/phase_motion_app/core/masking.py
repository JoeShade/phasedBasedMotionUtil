"""This file owns mask-zone geometry checks and output-domain rasterization so the editor and render path share one deterministic masking contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from phase_motion_app.core.models import ExclusionZone, Resolution, ZoneMode, ZoneShape


@dataclass(frozen=True)
class MaskValidationIssue:
    """This model carries one geometry problem so the shell can explain why a zone cannot be used."""

    zone_id: str
    code: str
    message: str


def validate_exclusion_zones(
    zones: tuple[ExclusionZone, ...], source_resolution: Resolution
) -> tuple[MaskValidationIssue, ...]:
    """Validate zone geometry against the current source frame. The legacy function name stays for compatibility with older call sites."""

    issues: list[MaskValidationIssue] = []
    for zone in zones:
        if zone.x < 0 or zone.y < 0:
            issues.append(
                MaskValidationIssue(
                    zone_id=zone.zone_id,
                    code="negative_origin",
                    message="Zone origin must stay inside the source frame.",
                )
            )
            continue

        if zone.shape is ZoneShape.RECTANGLE:
            if zone.width is None or zone.height is None:
                issues.append(
                    MaskValidationIssue(
                        zone_id=zone.zone_id,
                        code="rectangle_dimensions_missing",
                        message="Rectangle zones require width and height.",
                    )
                )
                continue
            if zone.width <= 0 or zone.height <= 0:
                issues.append(
                    MaskValidationIssue(
                        zone_id=zone.zone_id,
                        code="rectangle_dimensions_invalid",
                        message="Rectangle width and height must be positive.",
                    )
                )
                continue
            if zone.x + zone.width > source_resolution.width or zone.y + zone.height > source_resolution.height:
                issues.append(
                    MaskValidationIssue(
                        zone_id=zone.zone_id,
                        code="rectangle_out_of_bounds",
                        message="Rectangle zone extends beyond the source frame.",
                    )
                )
                continue

        if zone.shape is ZoneShape.CIRCLE:
            if zone.radius is None:
                issues.append(
                    MaskValidationIssue(
                        zone_id=zone.zone_id,
                        code="circle_radius_missing",
                        message="Circle zones require a radius.",
                    )
                )
                continue
            if zone.radius <= 0:
                issues.append(
                    MaskValidationIssue(
                        zone_id=zone.zone_id,
                        code="circle_radius_invalid",
                        message="Circle radius must be positive.",
                    )
                )
                continue
            if (
                zone.x - zone.radius < 0
                or zone.y - zone.radius < 0
                or zone.x + zone.radius > source_resolution.width
                or zone.y + zone.radius > source_resolution.height
            ):
                issues.append(
                    MaskValidationIssue(
                        zone_id=zone.zone_id,
                        code="circle_out_of_bounds",
                        message="Circle zone extends beyond the source frame.",
                    )
                )
                continue
    return tuple(issues)


def summarize_automatic_analysis_roi(zones: tuple[ExclusionZone, ...]) -> str:
    """Describe the no-manual-ROI fallback so the shell can explain that analysis follows the same processing mask as render output."""

    has_include = any(zone.mode is ZoneMode.INCLUDE for zone in zones)
    has_exclude = any(zone.mode is ZoneMode.EXCLUDE for zone in zones)
    if has_include and has_exclude:
        return (
            "Whole-frame ROI (automatic; constrained by processing inclusion zones "
            "and reduced by exclusion zones)"
        )
    if has_include:
        return "Whole-frame ROI (automatic; limited to processing inclusion zones)"
    if has_exclude:
        return "Whole-frame ROI (automatic; excludes processing mask zones)"
    return "Whole-frame ROI (automatic)"


def explain_automatic_analysis_roi(zones: tuple[ExclusionZone, ...]) -> str:
    """Return one plain-language sentence for dialogs that need to explain the automatic ROI scope."""

    has_include = any(zone.mode is ZoneMode.INCLUDE for zone in zones)
    has_exclude = any(zone.mode is ZoneMode.EXCLUDE for zone in zones)
    if has_include and has_exclude:
        return (
            "The analysis will use the full frame, constrained by processing inclusion "
            "zones and then reduced by exclusion zones."
        )
    if has_include:
        return "The analysis will use the full frame, limited to processing inclusion zones."
    if has_exclude:
        return "The analysis will use the full frame with processing exclusion zones removed."
    return "The analysis will use the full frame."


def scale_zone_to_domain(
    zone: ExclusionZone, source_resolution: Resolution, target_resolution: Resolution
) -> ExclusionZone:
    """Scale one source-frame zone into another pixel domain so mask rasterization can happen once in the final output domain."""

    scale_x = target_resolution.width / source_resolution.width
    scale_y = target_resolution.height / source_resolution.height
    if zone.shape is ZoneShape.RECTANGLE:
        return ExclusionZone(
            zone_id=zone.zone_id,
            shape=zone.shape,
            x=zone.x * scale_x,
            y=zone.y * scale_y,
            mode=zone.mode,
            width=(zone.width or 0.0) * scale_x,
            height=(zone.height or 0.0) * scale_y,
            label=zone.label,
        )
    return ExclusionZone(
        zone_id=zone.zone_id,
        shape=zone.shape,
        x=zone.x * scale_x,
        y=zone.y * scale_y,
        mode=zone.mode,
        radius=(zone.radius or 0.0) * ((scale_x + scale_y) / 2.0),
        label=zone.label,
    )


def rasterize_output_domain_mask(
    *,
    zones: tuple[ExclusionZone, ...],
    source_resolution: Resolution,
    output_resolution: Resolution,
    feather_px: float,
) -> list[list[float]]:
    """Rasterize a feathered passthrough mask in the final output domain because compositing must happen there exactly once."""

    if feather_px <= 0:
        raise ValueError("Mask feather must be positive because hard-edge blending is not allowed.")

    scaled_zones = tuple(
        scale_zone_to_domain(zone, source_resolution, output_resolution) for zone in zones
    )
    inclusion_zones = tuple(zone for zone in scaled_zones if zone.mode is ZoneMode.INCLUDE)
    exclusion_zones = tuple(zone for zone in scaled_zones if zone.mode is ZoneMode.EXCLUDE)
    mask: list[list[float]] = []
    for y in range(output_resolution.height):
        row: list[float] = []
        for x in range(output_resolution.width):
            alpha = 1.0 if inclusion_zones else 0.0
            pixel_x = x + 0.5
            pixel_y = y + 0.5
            if inclusion_zones:
                inclusion_alpha = 0.0
                for zone in inclusion_zones:
                    inclusion_alpha = max(
                        inclusion_alpha,
                        _zone_alpha(pixel_x, pixel_y, zone, feather_px),
                    )
                    if inclusion_alpha >= 1.0:
                        break
                alpha = 1.0 - inclusion_alpha
            for zone in exclusion_zones:
                alpha = max(alpha, _zone_alpha(pixel_x, pixel_y, zone, feather_px))
                if alpha >= 1.0:
                    break
            row.append(min(1.0, max(0.0, alpha)))
        mask.append(row)
    return mask


def composite_output_domain(
    *,
    amplified_plane: list[list[float]],
    passthrough_plane: list[list[float]],
    exclusion_mask: list[list[float]],
) -> list[list[float]]:
    """Blend one scalar plane in output space so the later numeric pipeline can reuse the same semantics for full image tensors."""

    output: list[list[float]] = []
    for row_index, mask_row in enumerate(exclusion_mask):
        blended_row: list[float] = []
        for column_index, alpha in enumerate(mask_row):
            amplified_value = amplified_plane[row_index][column_index]
            passthrough_value = passthrough_plane[row_index][column_index]
            blended_row.append(
                passthrough_value * alpha + amplified_value * (1.0 - alpha)
            )
        output.append(blended_row)
    return output


def _rectangle_alpha(
    pixel_x: float, pixel_y: float, zone: ExclusionZone, feather_px: float
) -> float:
    left = zone.x
    top = zone.y
    right = zone.x + (zone.width or 0.0)
    bottom = zone.y + (zone.height or 0.0)
    inside = left <= pixel_x <= right and top <= pixel_y <= bottom
    if inside:
        return 1.0
    dx = max(left - pixel_x, 0.0, pixel_x - right)
    dy = max(top - pixel_y, 0.0, pixel_y - bottom)
    distance = math.hypot(dx, dy)
    return max(0.0, 1.0 - (distance / feather_px))


def _circle_alpha(
    pixel_x: float, pixel_y: float, zone: ExclusionZone, feather_px: float
) -> float:
    distance_to_center = math.hypot(pixel_x - zone.x, pixel_y - zone.y)
    radius = zone.radius or 0.0
    if distance_to_center <= radius:
        return 1.0
    return max(0.0, 1.0 - ((distance_to_center - radius) / feather_px))


def _zone_alpha(
    pixel_x: float, pixel_y: float, zone: ExclusionZone, feather_px: float
) -> float:
    if zone.shape is ZoneShape.RECTANGLE:
        return _rectangle_alpha(pixel_x, pixel_y, zone, feather_px)
    return _circle_alpha(pixel_x, pixel_y, zone, feather_px)
