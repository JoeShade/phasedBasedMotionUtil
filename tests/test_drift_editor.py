"""This file tests small mask-editor geometry behaviors so rectangle editing can continue off-canvas without producing invalid zones."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtWidgets import QApplication, QWidget

from phase_motion_app.app.drift_editor import (
    AnalysisRoiEditorDialog,
    CanvasTool,
    DriftCanvas,
    DriftEditorDialog,
)
from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.models import ExclusionZone, Resolution, ZoneMode, ZoneShape


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_drift_canvas_clips_offscreen_rectangle_creation() -> None:
    app = _app()
    canvas = DriftCanvas(source_resolution=Resolution(100, 50), zones=())

    try:
        zone = canvas._build_rectangle_zone(
            zone_id="zone-1",
            start=QPointF(-20.0, -10.0),
            current=QPointF(30.0, 25.0),
        )
    finally:
        canvas.deleteLater()
        app.processEvents()
        app.quit()

    assert zone is not None
    assert zone.x == 0.0
    assert zone.y == 0.0
    assert zone.width == 30.0
    assert zone.height == 25.0


def test_drift_canvas_maps_points_outside_view_during_rectangle_drag() -> None:
    app = _app()
    canvas = DriftCanvas(source_resolution=Resolution(100, 50), zones=())
    canvas.resize(640, 420)

    try:
        source_point = canvas._to_source_point(QPointF(0.0, 0.0), allow_outside=True)
    finally:
        canvas.deleteLater()
        app.processEvents()
        app.quit()

    assert source_point is not None
    assert source_point.x() < 0.0 or source_point.y() < 0.0


def test_drift_canvas_uses_creation_mode_for_new_zones() -> None:
    app = _app()
    canvas = DriftCanvas(source_resolution=Resolution(100, 50), zones=())

    try:
        canvas.set_creation_mode(ZoneMode.INCLUDE)
        zone = canvas._build_rectangle_zone(
            zone_id="zone-1",
            start=QPointF(5.0, 5.0),
            current=QPointF(30.0, 25.0),
        )
    finally:
        canvas.deleteLater()
        app.processEvents()
        app.quit()

    assert zone is not None
    assert zone.mode is ZoneMode.INCLUDE


def test_drift_editor_exposes_theme_style_anchors() -> None:
    app = _app()
    dialog = DriftEditorDialog(
        source_resolution=Resolution(100, 50),
        zones=(),
        mask_feather_px=4.0,
        drift_assessment=DriftAssessment(),
    )

    try:
        object_name = dialog.objectName()
        control_name = dialog.findChild(QWidget, "maskEditorControls")
        review_name = dialog.findChild(QWidget, "maskEditorReview")
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert object_name == "maskEditorShell"
    assert control_name is not None
    assert review_name is not None


def test_drift_editor_shows_estimated_drift_and_requires_ack_when_warning_active() -> None:
    app = _app()
    dialog = DriftEditorDialog(
        source_resolution=Resolution(100, 50),
        zones=(),
        mask_feather_px=4.0,
        drift_assessment=DriftAssessment(estimated_global_drift_px=2.5),
    )

    try:
        estimate_text = dialog.estimated_drift_label.text()
        ack_enabled = dialog.ack_checkbox.isEnabled()
        warning_text = dialog.warning_label.text()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert estimate_text == "2.50 px (threshold 2.00 px)"
    assert ack_enabled is True
    assert "Render must stay blocked" in warning_text


def test_drift_editor_can_switch_selected_zone_to_include_mode() -> None:
    app = _app()
    dialog = DriftEditorDialog(
        source_resolution=Resolution(100, 50),
        zones=(
            ExclusionZone(
                zone_id="zone-1",
                shape=ZoneShape.RECTANGLE,
                x=10.0,
                y=10.0,
                width=20.0,
                height=10.0,
            ),
        ),
        mask_feather_px=4.0,
        drift_assessment=DriftAssessment(),
    )

    try:
        dialog.canvas._selected_zone_id = "zone-1"
        dialog._sync_zone_mode_combo()
        dialog.zone_mode_combo.setCurrentIndex(
            dialog.zone_mode_combo.findData(ZoneMode.INCLUDE)
        )
        updated_mode = dialog.canvas.zones[0].mode
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert updated_mode is ZoneMode.INCLUDE


def test_drift_editor_marks_active_drawing_tool_buttons_checked() -> None:
    app = _app()
    dialog = DriftEditorDialog(
        source_resolution=Resolution(100, 50),
        zones=(),
        mask_feather_px=4.0,
        drift_assessment=DriftAssessment(),
    )

    try:
        dialog.add_rectangle_button.click()
        rectangle_checked = dialog.add_rectangle_button.isChecked()
        circle_checked_before = dialog.add_circle_button.isChecked()
        dialog.add_circle_button.click()
        rectangle_checked_after = dialog.add_rectangle_button.isChecked()
        circle_checked_after = dialog.add_circle_button.isChecked()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert rectangle_checked is True
    assert circle_checked_before is False
    assert rectangle_checked_after is False
    assert circle_checked_after is True


def test_drift_canvas_uses_crosshair_cursor_over_image_when_add_rectangle_active() -> None:
    app = _app()
    canvas = DriftCanvas(source_resolution=Resolution(100, 50), zones=())
    canvas.resize(640, 420)

    try:
        target_rect, _ = canvas._target_rect_and_scale()
        canvas.set_tool(CanvasTool.ADD_RECTANGLE)
        canvas._refresh_hover_cursor(target_rect.center())
        active_shape = canvas.cursor().shape()
        canvas._refresh_hover_cursor(QPointF(0.0, 0.0))
        inactive_shape = canvas.cursor().shape()
    finally:
        canvas.deleteLater()
        app.processEvents()
        app.quit()

    assert active_shape is Qt.CursorShape.CrossCursor
    assert inactive_shape is not Qt.CursorShape.CrossCursor


def test_analysis_roi_editor_defaults_to_whole_frame_fallback() -> None:
    app = _app()
    dialog = AnalysisRoiEditorDialog(
        source_resolution=Resolution(100, 50),
        roi=None,
    )

    try:
        summary_text = dialog.summary_label.text()
        result = dialog.result_data()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert "Whole-frame" in summary_text
    assert result.roi is None


def test_analysis_roi_editor_explains_processing_mask_fallback() -> None:
    app = _app()
    dialog = AnalysisRoiEditorDialog(
        source_resolution=Resolution(100, 50),
        roi=None,
        processing_zones=(
            ExclusionZone(
                zone_id="include-main",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.INCLUDE,
                x=10.0,
                y=5.0,
                width=50.0,
                height=30.0,
            ),
            ExclusionZone(
                zone_id="exclude-background",
                shape=ZoneShape.RECTANGLE,
                mode=ZoneMode.EXCLUDE,
                x=15.0,
                y=10.0,
                width=10.0,
                height=10.0,
            ),
        ),
    )

    try:
        summary_text = dialog.summary_label.text()
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert "processing inclusion zones" in summary_text
    assert "reduced by exclusion zones" in summary_text

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
