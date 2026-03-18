"""This file provides the package entrypoint and starts the PyQt shell without pulling heavy engine logic into the GUI process."""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from phase_motion_app.app.main_window import MainWindow


def _build_stylesheet() -> str:
    asset_root = Path(__file__).resolve().parents[3] / "assets"
    watermark_path = asset_root / "grayWatermark.png"
    chevron_up_path = asset_root / "chevron-up-light.svg"
    chevron_down_path = asset_root / "chevron-down-light.svg"
    chevron_down_rule = """
QComboBox::down-arrow {
    width: 10px;
    height: 10px;
}
"""
    if chevron_down_path.exists():
        chevron_down_rule = f"""
QComboBox::down-arrow {{
    image: url("{chevron_down_path.as_posix()}");
    width: 10px;
    height: 10px;
}}
"""
    spin_chevron_rules = """
QAbstractSpinBox::up-arrow,
QAbstractSpinBox::down-arrow {
    width: 10px;
    height: 10px;
}
"""
    if chevron_up_path.exists() and chevron_down_path.exists():
        spin_chevron_rules = f"""
QAbstractSpinBox::up-arrow {{
    image: url("{chevron_up_path.as_posix()}");
    width: 10px;
    height: 10px;
}}
QAbstractSpinBox::down-arrow {{
    image: url("{chevron_down_path.as_posix()}");
    width: 10px;
    height: 10px;
}}
"""
    shell_background = """
QWidget#appShell {
    background-color: #000000;
}
QDialog#maskEditorShell {
    background-color: #2d2d2d;
    background-image: none;
}
"""
    if watermark_path.exists():
        shell_background = f"""
QWidget#appShell {{
    background-color: #000000;
    background-image: url("{watermark_path.as_posix()}");
    background-position: center;
    background-repeat: no-repeat;
}}
QDialog#maskEditorShell {{
    background-color: #2d2d2d;
    background-image: none;
}}
"""

    return shell_background + """
QScrollArea,
QWidget#contentViewport,
QWidget#scrollContent,
QWidget#actionStrip,
QWidget#maskEditorControls,
QWidget#maskEditorReview {
    background-color: transparent;
}
QLabel#heartbeatText,
QLabel#heartbeatDot {
    background-color: transparent;
    border: none;
    color: #f5f5f5;
}
QLabel#heartbeatDot {
    font-size: 18px;
    min-width: 16px;
}
QPushButton[drawingToolButton="true"]:checked {
    background-color: #4a4a4a;
    border: 1px solid #8a8a8a;
}
QLineEdit,
QLineEdit:read-only,
QPlainTextEdit,
QComboBox,
QAbstractSpinBox,
QPushButton,
QLabel#sourcePreview {
    background-color: #111111;
    color: #f5f5f5;
    border: 1px solid #454545;
}
QLineEdit:disabled,
QPlainTextEdit:disabled,
QComboBox:disabled,
QAbstractSpinBox:disabled,
QPushButton:disabled,
QLabel#sourcePreview:disabled {
    background-color: #151515;
    color: #8a8a8a;
    border: 1px solid #2f2f2f;
}
QComboBox::drop-down,
QAbstractSpinBox::up-button,
QAbstractSpinBox::down-button {
    background-color: #2d2d2d;
    border-left: 1px solid #656565;
    width: 22px;
}
QComboBox::drop-down {
    border-top-right-radius: 0px;
    border-bottom-right-radius: 0px;
}
QAbstractSpinBox::up-button {
    border-bottom: 1px solid #656565;
}
QComboBox QAbstractItemView {
    background-color: #111111;
    color: #f5f5f5;
}
QSlider::groove:horizontal {
    background-color: #2d2d2d;
    border: 1px solid #454545;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background-color: #c0c0c0;
    border: 1px solid #707070;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
""" + chevron_down_rule + spin_chevron_rules


def _apply_dark_palette(app: QApplication) -> None:
    """Keep native widget sizing while forcing a white-on-black application palette."""

    palette = app.palette()
    black = QColor(0, 0, 0)
    panel_black = QColor(17, 17, 17)
    button_black = QColor(21, 21, 21)
    alternate_black = QColor(28, 28, 28)
    white = QColor(245, 245, 245)
    muted = QColor(138, 138, 138)
    highlight = QColor(58, 94, 138)
    border = QColor(69, 69, 69)

    for group in (
        QPalette.ColorGroup.Active,
        QPalette.ColorGroup.Inactive,
        QPalette.ColorGroup.Disabled,
    ):
        palette.setColor(group, QPalette.ColorRole.Window, black)
        palette.setColor(group, QPalette.ColorRole.Base, panel_black)
        palette.setColor(group, QPalette.ColorRole.AlternateBase, alternate_black)
        palette.setColor(group, QPalette.ColorRole.ToolTipBase, panel_black)
        palette.setColor(group, QPalette.ColorRole.Button, button_black)
        palette.setColor(group, QPalette.ColorRole.WindowText, white if group != QPalette.ColorGroup.Disabled else muted)
        palette.setColor(group, QPalette.ColorRole.Text, white if group != QPalette.ColorGroup.Disabled else muted)
        palette.setColor(group, QPalette.ColorRole.ToolTipText, white)
        palette.setColor(group, QPalette.ColorRole.ButtonText, white if group != QPalette.ColorGroup.Disabled else muted)
        palette.setColor(group, QPalette.ColorRole.PlaceholderText, muted)
        palette.setColor(group, QPalette.ColorRole.Highlight, highlight)
        palette.setColor(group, QPalette.ColorRole.HighlightedText, white)
        palette.setColor(group, QPalette.ColorRole.Light, border)
        palette.setColor(group, QPalette.ColorRole.Midlight, border)
        palette.setColor(group, QPalette.ColorRole.Mid, border)
        palette.setColor(group, QPalette.ColorRole.Dark, QColor(32, 32, 32))
        palette.setColor(group, QPalette.ColorRole.Shadow, QColor(0, 0, 0))

    app.setPalette(palette)
    app.setStyleSheet(_build_stylesheet())


def main() -> int:
    app = QApplication(sys.argv)
    _apply_dark_palette(app)
    window = MainWindow()
    window.resize(860, 760)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

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
