"""This file owns retained-evidence cleanup planning so oversized diagnostics and failed-run material can be purged without confusing that policy with active scratch admission."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetainedEntry:
    """This model describes one retained artifact root that counts against the global retention budget."""

    path: Path
    size_bytes: int
    modified_ns: int
    preserve: bool = False


def measure_path_bytes(path: Path) -> int:
    """Measure one file tree so retention planning can compare real disk cost across jobs."""

    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def measure_retained_roots_bytes(
    roots: tuple[Path, ...],
    *,
    exclude_paths: tuple[Path, ...] = (),
) -> int:
    """Measure retained roots under the configured app roots, excluding any active job directories."""

    excluded = {path.resolve(strict=False) for path in exclude_paths}
    seen: set[Path] = set()
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            resolved_child = child.resolve(strict=False)
            if resolved_child in excluded or resolved_child in seen:
                continue
            total += measure_path_bytes(child)
            seen.add(resolved_child)
    return total


def build_retained_entry(path: Path, *, preserve: bool = False) -> RetainedEntry:
    """Build one retained-entry record from the filesystem."""

    stat = path.stat()
    return RetainedEntry(
        path=path,
        size_bytes=measure_path_bytes(path),
        modified_ns=stat.st_mtime_ns,
        preserve=preserve,
    )


def plan_oldest_first_purge(
    entries: list[RetainedEntry],
    *,
    budget_bytes: int,
) -> tuple[RetainedEntry, ...]:
    """Select the oldest non-preserved retained entries until the total falls back under budget."""

    total = sum(entry.size_bytes for entry in entries)
    if total <= budget_bytes:
        return ()

    purge: list[RetainedEntry] = []
    for entry in sorted(entries, key=lambda item: item.modified_ns):
        if total <= budget_bytes:
            break
        if entry.preserve:
            continue
        purge.append(entry)
        total -= entry.size_bytes
    return tuple(purge)


def purge_retained_entries(entries: list[RetainedEntry]) -> tuple[Path, ...]:
    """Delete the selected retained entries from disk."""

    removed: list[Path] = []
    for entry in entries:
        if not entry.path.exists():
            continue
        if entry.path.is_dir():
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            entry.path.unlink(missing_ok=True)
        removed.append(entry.path)
    return tuple(removed)

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
