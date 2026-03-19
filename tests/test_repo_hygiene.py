"""This file tests repository-level hygiene rules so version drift, footer regressions, and removed migration artifacts are caught automatically."""

from __future__ import annotations

import re
from pathlib import Path

import phase_motion_app

REPO_ROOT = Path(__file__).resolve().parents[1]
FOOTER_SIGNATURES = (
    "Copyright (c) JoeShade",
    "Licensed under the GNU Affero General Public License v3.0",
)


def _applicable_source_files() -> list[Path]:
    files: list[Path] = []
    for relative_root in ("src", "tests", "tools"):
        files.extend(sorted((REPO_ROOT / relative_root).rglob("*.py")))
    files.append(REPO_ROOT / "run.bat")
    return files


def test_package_version_matches_pyproject() -> None:
    pyproject_text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.MULTILINE)

    assert match is not None
    assert phase_motion_app.__version__ == match.group(1)


def test_applicable_source_files_preserve_standard_footer() -> None:
    missing_footer = []
    for path in _applicable_source_files():
        text = path.read_text(encoding="utf-8")
        if not all(signature in text for signature in FOOTER_SIGNATURES):
            missing_footer.append(str(path.relative_to(REPO_ROOT)))

    assert missing_footer == []


def test_legacy_footer_source_file_remains_deleted() -> None:
    assert (REPO_ROOT / "source-code-footer.txt").exists() is False

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
