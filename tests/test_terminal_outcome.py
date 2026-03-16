"""This file locks down the post-run dialog so it stays focused on the final output only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QPushButton

from phase_motion_app.app.terminal_outcome import TerminalOutcomeData, TerminalOutcomeDialog


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    return app


def test_terminal_outcome_dialog_only_exposes_output_actions(tmp_path: Path) -> None:
    app = _app()
    output_directory = tmp_path / "output"
    output_directory.mkdir()
    output_video_path = output_directory / "render.mp4"
    output_video_path.write_bytes(b"mp4")

    dialog = TerminalOutcomeDialog(
        outcome=TerminalOutcomeData(
            title="Render Complete",
            summary="Render Complete",
            output_directory=output_directory,
            output_video_path=output_video_path,
        ),
        open_output=lambda: None,
        open_video=lambda: None,
    )

    try:
        button_texts = {button.text() for button in dialog.findChildren(QPushButton)}
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert button_texts == {"Open Output Folder", "Open Video", "Close"}


def test_terminal_outcome_dialog_supports_failure_specific_actions(tmp_path: Path) -> None:
    app = _app()
    diagnostics_directory = tmp_path / "diagnostics"
    diagnostics_directory.mkdir()

    dialog = TerminalOutcomeDialog(
        outcome=TerminalOutcomeData(
            title="Render Failed",
            summary="Render failed: out_of_memory.",
            output_directory=diagnostics_directory,
            primary_action_label="Open Diagnostics",
            secondary_action_label="Purge Failed-run Files",
            primary_action_enabled=True,
            secondary_action_enabled=True,
        ),
        open_output=lambda: None,
        open_video=lambda: None,
    )

    try:
        button_texts = {button.text() for button in dialog.findChildren(QPushButton)}
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()
        app.quit()

    assert button_texts == {"Open Diagnostics", "Purge Failed-run Files", "Close"}
