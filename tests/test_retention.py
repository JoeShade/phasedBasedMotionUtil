"""This file tests retained-evidence cleanup planning so oldest-first purging respects the global budget and preserves the current failed job when required."""

from __future__ import annotations

from pathlib import Path

from phase_motion_app.core.retention import (
    RetainedEntry,
    build_retained_entry,
    measure_retained_roots_bytes,
    plan_oldest_first_purge,
    purge_retained_entries,
)


def _write_tree(path: Path, size_bytes: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "payload.bin").write_bytes(b"x" * size_bytes)


def test_retention_plan_purges_oldest_entries_first(tmp_path: Path) -> None:
    old_path = tmp_path / "old"
    new_path = tmp_path / "new"
    _write_tree(old_path, 128)
    _write_tree(new_path, 128)
    old_entry = build_retained_entry(old_path)
    new_entry = build_retained_entry(new_path)
    entries = [
        RetainedEntry(
            path=old_entry.path,
            size_bytes=old_entry.size_bytes,
            modified_ns=old_entry.modified_ns - 10,
        ),
        RetainedEntry(
            path=new_entry.path,
            size_bytes=new_entry.size_bytes,
            modified_ns=new_entry.modified_ns,
        ),
    ]

    purge = plan_oldest_first_purge(entries, budget_bytes=128)

    assert purge == (entries[0],)


def test_retention_plan_preserves_marked_current_failed_job(tmp_path: Path) -> None:
    preserved = tmp_path / "preserved"
    expendable = tmp_path / "expendable"
    _write_tree(preserved, 128)
    _write_tree(expendable, 128)

    purge = plan_oldest_first_purge(
        [
            build_retained_entry(preserved, preserve=True),
            build_retained_entry(expendable),
        ],
        budget_bytes=128,
    )

    assert len(purge) == 1
    assert purge[0].path == expendable


def test_retention_purge_removes_selected_entries(tmp_path: Path) -> None:
    target = tmp_path / "target"
    _write_tree(target, 64)

    removed = purge_retained_entries([build_retained_entry(target)])

    assert removed == (target,)
    assert not target.exists()


def test_measure_retained_roots_bytes_sums_roots_without_counting_excluded_job(
    tmp_path: Path,
) -> None:
    diagnostics_root = tmp_path / "diagnostics"
    scratch_root = tmp_path / "scratch"
    current_diag = diagnostics_root / "job-current"
    old_diag = diagnostics_root / "job-old"
    current_scratch = scratch_root / "job-current"
    old_scratch = scratch_root / "job-old"

    _write_tree(current_diag, 10)
    _write_tree(old_diag, 20)
    _write_tree(current_scratch, 30)
    _write_tree(old_scratch, 40)

    total = measure_retained_roots_bytes(
        (diagnostics_root, scratch_root),
        exclude_paths=(current_diag, current_scratch),
    )

    assert total == 60

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
