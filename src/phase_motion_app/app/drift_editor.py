"""This file owns the PyQt drift-check and mask-zone editor so operators can review camera stability and define static include/exclude regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from phase_motion_app.core.drift import DriftAssessment
from phase_motion_app.core.masking import validate_exclusion_zones
from phase_motion_app.core.models import ExclusionZone, Resolution, ZoneMode, ZoneShape


class CanvasTool(str, Enum):
    """This enum keeps editor tool state explicit so canvas mouse behavior stays predictable."""

    SELECT = "select"
    ADD_RECTANGLE = "add_rectangle"
    ADD_CIRCLE = "add_circle"


class FrameInspectionMode(str, Enum):
    """This enum mirrors the first/last/overlay inspection modes from the design doc."""

    FIRST = "first"
    LAST = "last"
    OVERLAY = "overlay"


@dataclass(frozen=True)
class DriftEditorResult:
    """This result object returns the edited zones and drift review state back to the main shell."""

    zones: tuple[ExclusionZone, ...]
    mask_feather_px: float
    drift_assessment: DriftAssessment


class DriftCanvas(QWidget):
    """This widget shows inspection imagery and lets the operator edit static mask zones in source-frame coordinates."""

    zones_changed = pyqtSignal(tuple)
    selection_changed = pyqtSignal(object)

    def __init__(
        self,
        *,
        source_resolution: Resolution,
        zones: tuple[ExclusionZone, ...],
        first_frame: QImage | None = None,
        last_frame: QImage | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.source_resolution = source_resolution
        self.first_frame = first_frame
        self.last_frame = last_frame
        self._zones = list(zones)
        self._selected_zone_id: str | None = None
        self._tool = CanvasTool.SELECT
        self._mode = FrameInspectionMode.OVERLAY
        self._creation_mode = ZoneMode.EXCLUDE
        self._overlay_opacity = 0.5
        self._blink_enabled = False
        self._blink_phase = False
        self._fit_to_view = True
        self._zoom_factor = 1.0
        self._drag_mode: str | None = None
        self._drag_start_source: QPointF | None = None
        self._drag_origin_zone: ExclusionZone | None = None
        self._draft_zone: ExclusionZone | None = None
        self._next_zone_index = len(zones) + 1
        self.setMinimumSize(640, 420)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    @property
    def zones(self) -> tuple[ExclusionZone, ...]:
        return tuple(self._zones)

    @property
    def selected_zone_id(self) -> str | None:
        return self._selected_zone_id

    def set_tool(self, tool: CanvasTool) -> None:
        self._tool = tool
        self._refresh_hover_cursor()

    def set_mode(self, mode: FrameInspectionMode) -> None:
        self._mode = mode
        self.update()

    @property
    def creation_mode(self) -> ZoneMode:
        return self._creation_mode

    def set_creation_mode(self, mode: ZoneMode) -> None:
        self._creation_mode = mode

    def selected_zone(self) -> ExclusionZone | None:
        if self._selected_zone_id is None:
            return None
        for zone in self._zones:
            if zone.zone_id == self._selected_zone_id:
                return zone
        return None

    def set_selected_zone_mode(self, mode: ZoneMode) -> None:
        selected_zone = self.selected_zone()
        if selected_zone is None or selected_zone.mode is mode:
            return
        self._replace_zone(
            ExclusionZone(
                zone_id=selected_zone.zone_id,
                shape=selected_zone.shape,
                x=selected_zone.x,
                y=selected_zone.y,
                mode=mode,
                width=selected_zone.width,
                height=selected_zone.height,
                radius=selected_zone.radius,
                label=selected_zone.label,
            )
        )

    def set_overlay_opacity(self, opacity: float) -> None:
        self._overlay_opacity = max(0.0, min(1.0, opacity))
        self.update()

    def set_blink_enabled(self, enabled: bool) -> None:
        self._blink_enabled = enabled
        self.update()

    def set_blink_phase(self, phase: bool) -> None:
        self._blink_phase = phase
        self.update()

    def set_fit_view(self) -> None:
        self._fit_to_view = True
        self.update()

    def set_one_to_one_view(self) -> None:
        self._fit_to_view = False
        self._zoom_factor = 1.0
        self.update()

    def delete_selected_zone(self) -> None:
        if self._selected_zone_id is None:
            return
        self._zones = [zone for zone in self._zones if zone.zone_id != self._selected_zone_id]
        self._selected_zone_id = None
        self.zones_changed.emit(self.zones)
        self.selection_changed.emit(self._selected_zone_id)
        self.update()

    def clear_zones(self) -> None:
        self._zones.clear()
        self._selected_zone_id = None
        self.zones_changed.emit(self.zones)
        self.selection_changed.emit(self._selected_zone_id)
        self.update()

    def paintEvent(self, _: object) -> None:  # noqa: N802 - Qt naming convention.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        target_rect, scale = self._target_rect_and_scale()
        self._paint_backdrop(painter, target_rect)
        painter.setPen(QColor(210, 210, 210))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target_rect)

        for zone in self._zones:
            self._paint_zone(painter, zone, target_rect, scale)
        if self._draft_zone is not None:
            self._paint_zone(painter, self._draft_zone, target_rect, scale, draft=True)

    def mousePressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming convention.
        mouse_event = event
        if mouse_event.button() != Qt.MouseButton.LeftButton:
            return
        self.setFocus()

        if self._tool is CanvasTool.ADD_RECTANGLE:
            source_point = self._to_source_point(
                mouse_event.position(),
                allow_outside=True,
            )
            if source_point is None:
                return
            self._drag_mode = "create_rectangle"
            self._drag_start_source = source_point
            self._draft_zone = self._build_rectangle_zone(
                zone_id=f"zone-{self._next_zone_index}",
                start=source_point,
                current=source_point,
            )
            self.update()
            return

        if self._tool is CanvasTool.ADD_CIRCLE:
            source_point = self._to_source_point(mouse_event.position())
            if source_point is None:
                return
            self._drag_mode = "create_circle"
            self._drag_start_source = source_point
            self._draft_zone = ExclusionZone(
                zone_id=f"zone-{self._next_zone_index}",
                shape=ZoneShape.CIRCLE,
                x=source_point.x(),
                y=source_point.y(),
                mode=self._creation_mode,
                radius=1.0,
            )
            self.update()
            return

        source_point = self._to_source_point(mouse_event.position())
        if source_point is None:
            return
        hit = self._hit_test(source_point)
        if hit is None:
            self._selected_zone_id = None
            self.selection_changed.emit(self._selected_zone_id)
            self.update()
            return

        zone, part = hit
        self._selected_zone_id = zone.zone_id
        self.selection_changed.emit(self._selected_zone_id)
        self._drag_start_source = source_point
        self._drag_origin_zone = zone
        self._drag_mode = "resize" if part == "handle" else "move"
        self.update()

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802 - Qt naming convention.
        mouse_event = event
        self._refresh_hover_cursor(mouse_event.position())
        if self._drag_mode is None:
            return
        source_point = self._to_source_point(
            mouse_event.position(),
            allow_outside=self._drag_mode in {"create_rectangle", "move", "resize"},
        )
        if source_point is None:
            return

        if self._drag_mode == "create_rectangle" and self._drag_start_source is not None:
            self._draft_zone = self._build_rectangle_zone(
                zone_id=f"zone-{self._next_zone_index}",
                start=self._drag_start_source,
                current=source_point,
            )
            self.update()
            return

        if self._drag_mode == "create_circle" and self._drag_start_source is not None:
            radius = max(
                1.0,
                math.hypot(
                    source_point.x() - self._drag_start_source.x(),
                    source_point.y() - self._drag_start_source.y(),
                ),
            )
            self._draft_zone = ExclusionZone(
                zone_id=f"zone-{self._next_zone_index}",
                shape=ZoneShape.CIRCLE,
                x=self._drag_start_source.x(),
                y=self._drag_start_source.y(),
                mode=self._creation_mode,
                radius=min(radius, self._max_circle_radius(self._drag_start_source)),
            )
            self.update()
            return

        if self._drag_origin_zone is None or self._drag_start_source is None:
            return

        if self._drag_mode == "move":
            dx = source_point.x() - self._drag_start_source.x()
            dy = source_point.y() - self._drag_start_source.y()
            updated = self._move_zone(self._drag_origin_zone, dx, dy)
            self._replace_zone(updated)
            return

        if self._drag_mode == "resize":
            updated = self._resize_zone(self._drag_origin_zone, source_point)
            self._replace_zone(updated)
            return

    def mouseReleaseEvent(self, _: object) -> None:  # noqa: N802 - Qt naming convention.
        if self._drag_mode in {"create_rectangle", "create_circle"} and self._draft_zone is not None:
            issues = validate_exclusion_zones((self._draft_zone,), self.source_resolution)
            if not issues:
                self._zones.append(self._draft_zone)
                self._selected_zone_id = self._draft_zone.zone_id
                self._next_zone_index += 1
                self.zones_changed.emit(self.zones)
                self.selection_changed.emit(self._selected_zone_id)
        self._draft_zone = None
        self._drag_mode = None
        self._drag_start_source = None
        self._drag_origin_zone = None
        self._refresh_hover_cursor()
        self.update()

    def keyPressEvent(self, event: object) -> None:  # noqa: N802 - Qt naming convention.
        key_event = event
        if key_event.key() == Qt.Key.Key_Delete:
            self.delete_selected_zone()
            return
        super().keyPressEvent(key_event)

    def leaveEvent(self, event: object) -> None:  # noqa: N802 - Qt naming convention.
        self.unsetCursor()
        super().leaveEvent(event)

    def _target_rect_and_scale(self) -> tuple[QRectF, float]:
        content_rect = self.rect().adjusted(16, 16, -16, -16)
        if self._fit_to_view:
            scale = min(
                content_rect.width() / self.source_resolution.width,
                content_rect.height() / self.source_resolution.height,
            )
        else:
            scale = self._zoom_factor
        width = self.source_resolution.width * scale
        height = self.source_resolution.height * scale
        x = content_rect.x() + (content_rect.width() - width) / 2.0
        y = content_rect.y() + (content_rect.height() - height) / 2.0
        return QRectF(x, y, width, height), scale

    def _hover_cursor_shape(
        self,
        widget_point: QPointF | None,
    ) -> Qt.CursorShape | None:
        if self._tool is not CanvasTool.ADD_RECTANGLE or widget_point is None:
            return None
        target_rect, _ = self._target_rect_and_scale()
        if target_rect.contains(widget_point):
            return Qt.CursorShape.CrossCursor
        return None

    def _refresh_hover_cursor(self, widget_point: QPointF | None = None) -> None:
        if widget_point is None:
            widget_point = QPointF(self.mapFromGlobal(QCursor.pos()))
        cursor_shape = self._hover_cursor_shape(widget_point)
        if cursor_shape is None:
            self.unsetCursor()
            return
        self.setCursor(cursor_shape)

    def _paint_backdrop(self, painter: QPainter, target_rect: QRectF) -> None:
        current_image = self._current_image()
        if current_image is not None:
            painter.drawImage(target_rect, current_image)
            if (
                not self._blink_enabled
                and self._mode is FrameInspectionMode.OVERLAY
                and self.first_frame is not None
                and self.last_frame is not None
            ):
                painter.save()
                painter.setOpacity(self._overlay_opacity)
                painter.drawImage(target_rect, self.last_frame)
                painter.restore()
            return

        # This placeholder keeps the editor usable before explicit first/last
        # frame extraction is wired in. Geometry editing still uses the correct
        # source-frame coordinate space from probe metadata.
        painter.fillRect(target_rect, QColor(52, 52, 52))
        painter.setPen(QColor(94, 94, 94))
        cell = 32
        for x in range(int(target_rect.left()), int(target_rect.right()), cell):
            painter.drawLine(x, int(target_rect.top()), x, int(target_rect.bottom()))
        for y in range(int(target_rect.top()), int(target_rect.bottom()), cell):
            painter.drawLine(int(target_rect.left()), y, int(target_rect.right()), y)
        painter.setPen(QColor(245, 245, 245))
        painter.drawText(
            target_rect,
            Qt.AlignmentFlag.AlignCenter,
            "First / Last frame imagery is not connected yet.\n"
            "Zone geometry still uses the probed source frame size.",
        )

    def _current_image(self) -> QImage | None:
        if self._blink_enabled:
            return self.last_frame if self._blink_phase else self.first_frame
        if self._mode is FrameInspectionMode.FIRST:
            return self.first_frame
        if self._mode is FrameInspectionMode.LAST:
            return self.last_frame
        return self.first_frame

    def _paint_zone(
        self,
        painter: QPainter,
        zone: ExclusionZone,
        target_rect: QRectF,
        scale: float,
        *,
        draft: bool = False,
    ) -> None:
        if zone.mode is ZoneMode.INCLUDE:
            fill = QColor(45, 180, 90, 50 if draft else 75)
            outline = QColor(88, 240, 134)
            handle = QColor(220, 255, 232)
        else:
            fill = QColor(220, 45, 45, 50 if draft else 75)
            outline = QColor(255, 70, 70)
            handle = QColor(255, 220, 220)
        pen = QPen(outline, 2)
        painter.setPen(pen)
        painter.setBrush(fill)

        if zone.shape is ZoneShape.RECTANGLE:
            rect = QRectF(
                target_rect.left() + zone.x * scale,
                target_rect.top() + zone.y * scale,
                (zone.width or 0.0) * scale,
                (zone.height or 0.0) * scale,
            )
            painter.drawRect(rect)
            if zone.zone_id == self._selected_zone_id:
                self._paint_handle(painter, rect.bottomRight(), handle)
            return

        center = QPointF(
            target_rect.left() + zone.x * scale,
            target_rect.top() + zone.y * scale,
        )
        radius = (zone.radius or 0.0) * scale
        painter.drawEllipse(center, radius, radius)
        if zone.zone_id == self._selected_zone_id:
            self._paint_handle(
                painter,
                QPointF(center.x() + radius, center.y()),
                handle,
            )

    def _paint_handle(self, painter: QPainter, point: QPointF, color: QColor) -> None:
        painter.fillRect(QRectF(point.x() - 4, point.y() - 4, 8, 8), color)

    def _to_source_point(
        self,
        widget_point: QPointF,
        *,
        allow_outside: bool = False,
    ) -> QPointF | None:
        target_rect, scale = self._target_rect_and_scale()
        if not allow_outside and not target_rect.contains(widget_point):
            return None
        source_x = (widget_point.x() - target_rect.left()) / scale
        source_y = (widget_point.y() - target_rect.top()) / scale
        return QPointF(source_x, source_y)

    def _hit_test(self, source_point: QPointF) -> tuple[ExclusionZone, str] | None:
        handle_tolerance = 8.0
        for zone in reversed(self._zones):
            if self._point_hits_handle(source_point, zone, handle_tolerance):
                return zone, "handle"
            if self._point_hits_zone(source_point, zone):
                return zone, "body"
        return None

    def _point_hits_zone(self, source_point: QPointF, zone: ExclusionZone) -> bool:
        if zone.shape is ZoneShape.RECTANGLE:
            return (
                zone.x <= source_point.x() <= zone.x + (zone.width or 0.0)
                and zone.y <= source_point.y() <= zone.y + (zone.height or 0.0)
            )
        return math.hypot(source_point.x() - zone.x, source_point.y() - zone.y) <= (zone.radius or 0.0)

    def _point_hits_handle(
        self, source_point: QPointF, zone: ExclusionZone, tolerance: float
    ) -> bool:
        if zone.shape is ZoneShape.RECTANGLE:
            handle_x = zone.x + (zone.width or 0.0)
            handle_y = zone.y + (zone.height or 0.0)
        else:
            handle_x = zone.x + (zone.radius or 0.0)
            handle_y = zone.y
        return abs(source_point.x() - handle_x) <= tolerance and abs(source_point.y() - handle_y) <= tolerance

    def _build_rectangle_zone(
        self, *, zone_id: str, start: QPointF, current: QPointF
    ) -> ExclusionZone | None:
        left = min(start.x(), current.x())
        right = max(start.x(), current.x())
        top = min(start.y(), current.y())
        bottom = max(start.y(), current.y())
        clipped_left = max(0.0, left)
        clipped_top = max(0.0, top)
        clipped_right = min(float(self.source_resolution.width), right)
        clipped_bottom = min(float(self.source_resolution.height), bottom)
        width = clipped_right - clipped_left
        height = clipped_bottom - clipped_top
        if width < 1.0 or height < 1.0:
            return None
        return ExclusionZone(
            zone_id=zone_id,
            shape=ZoneShape.RECTANGLE,
            x=clipped_left,
            y=clipped_top,
            mode=self._creation_mode,
            width=width,
            height=height,
        )

    def _move_zone(self, zone: ExclusionZone, dx: float, dy: float) -> ExclusionZone:
        if zone.shape is ZoneShape.RECTANGLE:
            x = min(
                max(0.0, zone.x + dx),
                self.source_resolution.width - (zone.width or 0.0),
            )
            y = min(
                max(0.0, zone.y + dy),
                self.source_resolution.height - (zone.height or 0.0),
            )
            return ExclusionZone(
                zone_id=zone.zone_id,
                shape=zone.shape,
                x=x,
                y=y,
                mode=zone.mode,
                width=zone.width,
                height=zone.height,
                label=zone.label,
            )
        radius = zone.radius or 0.0
        x = min(max(radius, zone.x + dx), self.source_resolution.width - radius)
        y = min(max(radius, zone.y + dy), self.source_resolution.height - radius)
        return ExclusionZone(
            zone_id=zone.zone_id,
            shape=zone.shape,
            x=x,
            y=y,
            mode=zone.mode,
            radius=radius,
            label=zone.label,
        )

    def _resize_zone(self, zone: ExclusionZone, source_point: QPointF) -> ExclusionZone:
        if zone.shape is ZoneShape.RECTANGLE:
            width = min(
                max(1.0, source_point.x() - zone.x),
                self.source_resolution.width - zone.x,
            )
            height = min(
                max(1.0, source_point.y() - zone.y),
                self.source_resolution.height - zone.y,
            )
            return ExclusionZone(
                zone_id=zone.zone_id,
                shape=zone.shape,
                x=zone.x,
                y=zone.y,
                mode=zone.mode,
                width=width,
                height=height,
                label=zone.label,
            )
        radius = min(
            max(1.0, math.hypot(source_point.x() - zone.x, source_point.y() - zone.y)),
            self._max_circle_radius(QPointF(zone.x, zone.y)),
        )
        return ExclusionZone(
            zone_id=zone.zone_id,
            shape=zone.shape,
            x=zone.x,
            y=zone.y,
            mode=zone.mode,
            radius=radius,
            label=zone.label,
        )

    def _max_circle_radius(self, center: QPointF) -> float:
        return min(
            center.x(),
            center.y(),
            self.source_resolution.width - center.x(),
            self.source_resolution.height - center.y(),
        )

    def _replace_zone(self, updated: ExclusionZone) -> None:
        self._zones = [
            updated if zone.zone_id == updated.zone_id else zone for zone in self._zones
        ]
        self.zones_changed.emit(self.zones)
        self.update()


class DriftEditorDialog(QDialog):
    """This dialog wraps the canvas and drift warning controls so the main shell can keep drift review separate from render execution."""

    def __init__(
        self,
        *,
        source_resolution: Resolution,
        zones: tuple[ExclusionZone, ...],
        mask_feather_px: float,
        drift_assessment: DriftAssessment,
        first_frame: QImage | None = None,
        last_frame: QImage | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Drift Check and Mask Zones")
        self.resize(1100, 760)
        self.setObjectName("maskEditorShell")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._drift_assessment = drift_assessment

        self.canvas = DriftCanvas(
            source_resolution=source_resolution,
            zones=zones,
            first_frame=first_frame,
            last_frame=last_frame,
        )

        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(500)
        self._blink_timer.timeout.connect(self._toggle_blink_phase)

        layout = QVBoxLayout(self)
        controls = QWidget()
        controls.setObjectName("maskEditorControls")
        controls.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        controls_layout = QGridLayout(controls)

        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItem("First frame", FrameInspectionMode.FIRST)
        self.view_mode_combo.addItem("Last frame", FrameInspectionMode.LAST)
        self.view_mode_combo.addItem("First vs Last overlay", FrameInspectionMode.OVERLAY)
        self.overlay_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlay_slider.setRange(0, 100)
        self.overlay_slider.setValue(50)
        self.blink_checkbox = QCheckBox("Blink")
        self.add_rectangle_button = QPushButton("Add Rectangle")
        self.add_circle_button = QPushButton("Add Circle")
        self.select_button = QPushButton("Select / Move")
        for button in (self.add_rectangle_button, self.add_circle_button):
            button.setCheckable(True)
            button.setProperty("drawingToolButton", True)
        self.delete_button = QPushButton("Delete Selected")
        self.clear_button = QPushButton("Clear All")
        self.fit_button = QPushButton("Fit")
        self.one_to_one_button = QPushButton("1:1")
        self.zone_mode_combo = QComboBox()
        self.zone_mode_combo.addItem("Exclude (Red)", ZoneMode.EXCLUDE)
        self.zone_mode_combo.addItem("Include (Green)", ZoneMode.INCLUDE)

        controls_layout.addWidget(QLabel("View"), 0, 0)
        controls_layout.addWidget(self.view_mode_combo, 0, 1)
        controls_layout.addWidget(QLabel("Overlay opacity"), 0, 2)
        controls_layout.addWidget(self.overlay_slider, 0, 3)
        controls_layout.addWidget(self.blink_checkbox, 0, 4)
        controls_layout.addWidget(self.select_button, 1, 0)
        controls_layout.addWidget(self.add_rectangle_button, 1, 1)
        controls_layout.addWidget(self.add_circle_button, 1, 2)
        controls_layout.addWidget(self.delete_button, 1, 3)
        controls_layout.addWidget(self.clear_button, 1, 4)
        controls_layout.addWidget(self.fit_button, 2, 0)
        controls_layout.addWidget(self.one_to_one_button, 2, 1)

        review_panel = QWidget()
        review_panel.setObjectName("maskEditorReview")
        review_panel.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        review_layout = QFormLayout(review_panel)
        self.zone_count_label = QLabel(str(len(zones)))
        self.selection_label = QLabel("None")
        self.mask_feather_spin = QDoubleSpinBox()
        self.mask_feather_spin.setRange(0.1, 100.0)
        self.mask_feather_spin.setValue(mask_feather_px)
        self.mask_feather_spin.setSuffix(" px")
        self.estimated_drift_label = QLabel(
            self._format_estimated_drift(drift_assessment)
        )
        self.visible_drift_checkbox = QCheckBox("Visible drift observed during review")
        self.visible_drift_checkbox.setChecked(drift_assessment.visible_drift_confirmed)
        self.ack_checkbox = QCheckBox("I acknowledge drift risk for this reviewed source state")
        self.ack_checkbox.setChecked(drift_assessment.acknowledged)
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: rgb(200, 70, 70);")
        self.warning_label.setWordWrap(True)
        review_layout.addRow("Zones", self.zone_count_label)
        review_layout.addRow("Selected", self.selection_label)
        review_layout.addRow("Zone mode", self.zone_mode_combo)
        review_layout.addRow("Mask feather", self.mask_feather_spin)
        review_layout.addRow("Estimated drift", self.estimated_drift_label)
        review_layout.addRow("", self.visible_drift_checkbox)
        review_layout.addRow("", self.ack_checkbox)
        review_layout.addRow("", self.warning_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )

        layout.addWidget(controls)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(review_panel)
        layout.addWidget(buttons)

        self.view_mode_combo.currentIndexChanged.connect(self._update_view_mode)
        self.overlay_slider.valueChanged.connect(
            lambda value: self.canvas.set_overlay_opacity(value / 100.0)
        )
        self.blink_checkbox.toggled.connect(self._set_blink_enabled)
        self.select_button.clicked.connect(lambda: self._set_canvas_tool(CanvasTool.SELECT))
        self.add_rectangle_button.clicked.connect(
            lambda: self._set_canvas_tool(CanvasTool.ADD_RECTANGLE)
        )
        self.add_circle_button.clicked.connect(
            lambda: self._set_canvas_tool(CanvasTool.ADD_CIRCLE)
        )
        self.delete_button.clicked.connect(self.canvas.delete_selected_zone)
        self.clear_button.clicked.connect(self.canvas.clear_zones)
        self.fit_button.clicked.connect(self.canvas.set_fit_view)
        self.one_to_one_button.clicked.connect(self.canvas.set_one_to_one_view)
        self.zone_mode_combo.currentIndexChanged.connect(self._set_zone_mode)
        self.canvas.zones_changed.connect(self._on_zones_changed)
        self.canvas.selection_changed.connect(self._on_selection_changed)
        self.visible_drift_checkbox.toggled.connect(self._update_warning_state)
        self.ack_checkbox.toggled.connect(self._update_warning_state)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        self._set_canvas_tool(CanvasTool.SELECT)
        self._sync_zone_mode_combo()
        self._update_view_mode()
        self._update_warning_state()

    def result_data(self) -> DriftEditorResult:
        """Collect the edited values for the caller after the dialog closes."""

        assessment = DriftAssessment(
            visible_drift_confirmed=self.visible_drift_checkbox.isChecked(),
            estimated_global_drift_px=self._drift_assessment.estimated_global_drift_px,
            advisory_threshold_px=self._drift_assessment.advisory_threshold_px,
            acknowledged=self.ack_checkbox.isChecked(),
        )
        return DriftEditorResult(
            zones=self.canvas.zones,
            mask_feather_px=self.mask_feather_spin.value(),
            drift_assessment=assessment,
        )

    def _update_view_mode(self) -> None:
        self.canvas.set_mode(self.view_mode_combo.currentData())

    def _set_canvas_tool(self, tool: CanvasTool) -> None:
        self.canvas.set_tool(tool)
        self.add_rectangle_button.setChecked(tool is CanvasTool.ADD_RECTANGLE)
        self.add_circle_button.setChecked(tool is CanvasTool.ADD_CIRCLE)

    def _set_blink_enabled(self, enabled: bool) -> None:
        self.canvas.set_blink_enabled(enabled)
        if enabled:
            self._blink_timer.start()
        else:
            self._blink_timer.stop()
            self.canvas.set_blink_phase(False)

    def _toggle_blink_phase(self) -> None:
        self.canvas.set_blink_phase(not self.canvas._blink_phase)

    def _on_zones_changed(self, zones: tuple[ExclusionZone, ...]) -> None:
        self.zone_count_label.setText(str(len(zones)))
        self._sync_zone_mode_combo()

    def _on_selection_changed(self, selected_zone_id: str | None) -> None:
        self.selection_label.setText(selected_zone_id or "None")
        self._sync_zone_mode_combo()

    def _set_zone_mode(self) -> None:
        mode = self.zone_mode_combo.currentData()
        if not isinstance(mode, ZoneMode):
            return
        self.canvas.set_creation_mode(mode)
        self.canvas.set_selected_zone_mode(mode)

    def _sync_zone_mode_combo(self) -> None:
        selected_zone = self.canvas.selected_zone()
        mode = self.canvas.creation_mode if selected_zone is None else selected_zone.mode
        index = self.zone_mode_combo.findData(mode)
        if index < 0:
            return
        self.zone_mode_combo.blockSignals(True)
        self.zone_mode_combo.setCurrentIndex(index)
        self.zone_mode_combo.blockSignals(False)

    def _update_warning_state(self) -> None:
        warning_active = self.visible_drift_checkbox.isChecked() or self._drift_assessment.warning_active
        self.ack_checkbox.setEnabled(warning_active)
        if warning_active and not self.ack_checkbox.isChecked():
            self.warning_label.setText(
                "Drift warning is active. Render must stay blocked until this reviewed source state is acknowledged."
            )
            return
        if warning_active:
            self.warning_label.setText(
                "Drift warning acknowledged. The reviewed source state may proceed with operator-attested risk."
            )
            return
        self.warning_label.setText("")

    @staticmethod
    def _format_estimated_drift(assessment: DriftAssessment) -> str:
        if assessment.estimated_global_drift_px is None:
            return "Unavailable for this source slice"
        return (
            f"{assessment.estimated_global_drift_px:.2f} px "
            f"(threshold {assessment.advisory_threshold_px:.2f} px)"
        )
