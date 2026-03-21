"""This file tests shared shell theming so key visual assets stay wired into the stylesheet."""

from __future__ import annotations

from types import SimpleNamespace

import phase_motion_app.app.windows_shell as windows_shell_module
from phase_motion_app.app.main import _build_stylesheet
from phase_motion_app.app.windows_shell import (
    WINDOWS_APP_USER_MODEL_ID,
    WINDOWS_RELAUNCH_DISPLAY_NAME,
    _window_property_values,
    configure_windows_process_identity,
)


def test_stylesheet_uses_explicit_chevrons_and_mask_editor_override() -> None:
    stylesheet = _build_stylesheet()

    assert "chevron-up-light.svg" in stylesheet
    assert "chevron-down-light.svg" in stylesheet
    assert "QDialog#maskEditorShell" in stylesheet
    assert "background-color: #2d2d2d;" in stylesheet
    assert 'background-image: none;' in stylesheet


def test_windows_shell_identity_sets_explicit_app_user_model_id(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def _set_app_id(value: str) -> None:
        captured["value"] = value

    monkeypatch.setattr(windows_shell_module.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(
        windows_shell_module,
        "ctypes",
        SimpleNamespace(
            windll=SimpleNamespace(
                shell32=SimpleNamespace(
                    SetCurrentProcessExplicitAppUserModelID=_set_app_id
                )
            )
        ),
    )

    configure_windows_process_identity()

    assert captured["value"] == WINDOWS_APP_USER_MODEL_ID


def test_windows_window_property_values_include_relaunch_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        windows_shell_module,
        "_current_process_command_line",
        lambda: '"pythonw.exe" -m phase_motion_app.app.main',
    )
    monkeypatch.setattr(
        windows_shell_module,
        "_relaunch_icon_resource",
        lambda: "A:\\Documents\\phasedBasedMotionUtil\\assets\\programIcon.ico,0",
    )

    values = _window_property_values()

    assert values == (
        (2, '"pythonw.exe" -m phase_motion_app.app.main'),
        (4, WINDOWS_RELAUNCH_DISPLAY_NAME),
        (3, "A:\\Documents\\phasedBasedMotionUtil\\assets\\programIcon.ico,0"),
        (5, WINDOWS_APP_USER_MODEL_ID),
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
