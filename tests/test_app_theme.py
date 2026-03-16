"""This file tests shared shell theming so key visual assets stay wired into the stylesheet."""

from __future__ import annotations

from phase_motion_app.app.main import _build_stylesheet


def test_stylesheet_uses_explicit_chevrons_and_mask_editor_override() -> None:
    stylesheet = _build_stylesheet()

    assert "chevron-up-light.svg" in stylesheet
    assert "chevron-down-light.svg" in stylesheet
    assert "QDialog#maskEditorShell" in stylesheet
    assert "background-color: #2d2d2d;" in stylesheet
    assert 'background-image: none;' in stylesheet
