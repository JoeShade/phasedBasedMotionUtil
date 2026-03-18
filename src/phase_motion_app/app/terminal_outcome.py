"""This file owns the terminal outcome dialog so completed and failed runs can present focused post-run actions without mixing them into the setup controls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


@dataclass(frozen=True)
class TerminalOutcomeData:
    """This model carries the paths and summary text shown in the terminal outcome dialog."""

    title: str
    summary: str
    output_directory: Path | None = None
    output_video_path: Path | None = None
    primary_action_label: str = "Open Output Folder"
    secondary_action_label: str = "Open Video"
    primary_action_enabled: bool | None = None
    secondary_action_enabled: bool | None = None


class TerminalOutcomeDialog(QDialog):
    """This dialog keeps the post-run actions focused so the operator can inspect or clean up the result immediately."""

    def __init__(
        self,
        *,
        outcome: TerminalOutcomeData,
        open_output,
        open_video,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(outcome.title)

        layout = QVBoxLayout(self)
        summary = QLabel(outcome.summary)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        actions = QHBoxLayout()
        output_button = QPushButton(outcome.primary_action_label)
        video_button = QPushButton(outcome.secondary_action_label)
        output_button.setEnabled(
            (
                outcome.output_directory is not None
                and outcome.output_directory.exists()
            )
            if outcome.primary_action_enabled is None
            else outcome.primary_action_enabled
        )
        video_button.setEnabled(
            (
                outcome.output_video_path is not None
                and outcome.output_video_path.exists()
            )
            if outcome.secondary_action_enabled is None
            else outcome.secondary_action_enabled
        )
        output_button.clicked.connect(open_output)
        video_button.clicked.connect(open_video)
        for button in (output_button, video_button):
            actions.addWidget(button)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

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
