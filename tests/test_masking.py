"""This file tests exclusion-zone geometry and mask rasterization so the editor and render path share one source-to-output masking contract."""

from __future__ import annotations

from phase_motion_app.core.masking import (
    composite_output_domain,
    rasterize_output_domain_mask,
    scale_zone_to_domain,
    validate_exclusion_zones,
)
from phase_motion_app.core.models import ExclusionZone, Resolution, ZoneMode, ZoneShape


def test_validate_exclusion_zones_rejects_out_of_bounds_rectangle() -> None:
    issues = validate_exclusion_zones(
        (
            ExclusionZone(
                zone_id="z1",
                shape=ZoneShape.RECTANGLE,
                x=90,
                y=20,
                width=20,
                height=30,
            ),
        ),
        Resolution(width=100, height=100),
    )

    assert len(issues) == 1
    assert issues[0].code == "rectangle_out_of_bounds"


def test_scale_zone_to_output_domain_uses_source_frame_coordinates() -> None:
    scaled = scale_zone_to_domain(
        ExclusionZone(
            zone_id="z1",
            shape=ZoneShape.RECTANGLE,
            x=200,
            y=100,
            width=400,
            height=200,
        ),
        source_resolution=Resolution(width=2000, height=1000),
        target_resolution=Resolution(width=1000, height=500),
    )

    assert scaled.x == 100
    assert scaled.y == 50
    assert scaled.width == 200
    assert scaled.height == 100


def test_rasterize_output_domain_mask_builds_feathered_alpha() -> None:
    mask = rasterize_output_domain_mask(
        zones=(
            ExclusionZone(
                zone_id="z1",
                shape=ZoneShape.RECTANGLE,
                x=2,
                y=2,
                width=4,
                height=4,
            ),
        ),
        source_resolution=Resolution(width=10, height=10),
        output_resolution=Resolution(width=10, height=10),
        feather_px=2.0,
    )

    assert mask[3][3] == 1.0
    assert 0.0 < mask[1][3] < 1.0
    assert mask[0][0] == 0.0


def test_rasterize_output_domain_mask_inclusion_zone_limits_amplification_area() -> None:
    mask = rasterize_output_domain_mask(
        zones=(
            ExclusionZone(
                zone_id="z1",
                shape=ZoneShape.RECTANGLE,
                x=2,
                y=2,
                mode=ZoneMode.INCLUDE,
                width=4,
                height=4,
            ),
        ),
        source_resolution=Resolution(width=10, height=10),
        output_resolution=Resolution(width=10, height=10),
        feather_px=2.0,
    )

    assert mask[3][3] == 0.0
    assert 0.0 < mask[1][3] < 1.0
    assert mask[0][0] == 1.0


def test_rasterize_output_domain_mask_exclusion_overrides_inclusion() -> None:
    mask = rasterize_output_domain_mask(
        zones=(
            ExclusionZone(
                zone_id="include-1",
                shape=ZoneShape.RECTANGLE,
                x=1,
                y=1,
                mode=ZoneMode.INCLUDE,
                width=6,
                height=6,
            ),
            ExclusionZone(
                zone_id="exclude-1",
                shape=ZoneShape.RECTANGLE,
                x=3,
                y=3,
                mode=ZoneMode.EXCLUDE,
                width=2,
                height=2,
            ),
        ),
        source_resolution=Resolution(width=10, height=10),
        output_resolution=Resolution(width=10, height=10),
        feather_px=1.0,
    )

    assert mask[1][1] == 0.0
    assert mask[3][3] == 1.0


def test_composite_output_domain_uses_output_space_mask() -> None:
    mask = [
        [0.0, 1.0],
        [0.5, 0.0],
    ]
    amplified = [
        [10.0, 10.0],
        [10.0, 10.0],
    ]
    passthrough = [
        [1.0, 1.0],
        [1.0, 1.0],
    ]

    output = composite_output_domain(
        amplified_plane=amplified,
        passthrough_plane=passthrough,
        exclusion_mask=mask,
    )

    assert output == [
        [10.0, 1.0],
        [5.5, 10.0],
    ]
