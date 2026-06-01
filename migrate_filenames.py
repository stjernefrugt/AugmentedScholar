#!/usr/bin/env python3
"""Rename library files to match the current pyiso4-based naming convention.

Scans a library directory for ``*.json`` sidecar files, reconstructs the
canonical filename stem using the current :func:`~src.naming.build_stem`
function, and renames any file whose stem has changed.  Associated ``.pdf``
and ``_SI.pdf`` files are renamed alongside their sidecar.

Runs in **dry-run mode by default** — pass ``--apply`` to make changes.

Usage::

    # Preview what would be renamed
    python migrate_filenames.py --library-dir ./library

    # Apply the renames
    python migrate_filenames.py --library-dir ./library --apply
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.models import Author, PaperMetadata
from src.naming import build_stem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_metadata(json_path: Path) -> PaperMetadata | None:
    """Parse a JSON sidecar into a PaperMetadata, or return None on failure."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read %s: %s", json_path.name, exc)
        return None

    title: str | None = data.get("title")
    year_raw = data.get("year")
    journal: str = data.get("journal") or "Unknown Journal"
    author_names: list[str] = data.get("authors") or []

    if not title or not year_raw:
        logger.warning("Skipping %s — missing title or year", json_path.name)
        return None

    try:
        year = int(year_raw)
    except (ValueError, TypeError):
        logger.warning("Skipping %s — invalid year %r", json_path.name, year_raw)
        return None

    return PaperMetadata(
        title=title,
        authors=[Author(name=n) for n in author_names],
        journal=journal,
        year=year,
        doi=data.get("doi"),
    )


def _associated_files(json_path: Path) -> list[tuple[Path, str]]:
    """Return (existing_path, suffix) pairs for files linked to *json_path*.

    Checks for the main PDF and SI PDF alongside the sidecar.
    """
    stem = json_path.stem
    parent = json_path.parent
    candidates = [
        (parent / f"{stem}.pdf", ".pdf"),
        (parent / f"{stem}_SI.pdf", "_SI.pdf"),
    ]
    return [(p, sfx) for p, sfx in candidates if p.exists()]


# ---------------------------------------------------------------------------
# Core migration logic
# ---------------------------------------------------------------------------


def _plan_renames(
    library_dir: Path,
) -> list[tuple[Path, Path]]:
    """Return a list of (old_path, new_path) pairs for all files needing rename.

    Does not touch the filesystem.
    """
    renames: list[tuple[Path, Path]] = []

    for json_path in sorted(library_dir.glob("*.json")):
        meta = _load_metadata(json_path)
        if meta is None:
            continue

        new_stem = build_stem(meta)
        old_stem = json_path.stem

        if new_stem == old_stem:
            continue

        new_json = json_path.with_name(f"{new_stem}.json")
        renames.append((json_path, new_json))

        for old_file, suffix in _associated_files(json_path):
            new_file = json_path.parent / f"{new_stem}{suffix}"
            renames.append((old_file, new_file))

    return renames


def _check_conflicts(
    renames: list[tuple[Path, Path]],
) -> list[tuple[Path, Path]]:
    """Return subset of *renames* where the destination already exists."""
    sources = {old for old, _ in renames}
    return [
        (old, new)
        for old, new in renames
        if new.exists() and old != new and new not in sources
    ]


def migrate(library_dir: Path, *, apply: bool) -> None:
    """Plan and optionally apply filename migrations.

    Args:
        library_dir: Root directory containing JSON sidecars and PDFs.
        apply: If ``True``, renames are performed.  If ``False``, only a
            preview is printed.
    """
    if not library_dir.is_dir():
        logger.error("Library directory not found: %s", library_dir)
        sys.exit(1)

    renames = _plan_renames(library_dir)

    if not renames:
        print("No renames needed — all filenames already match the current convention.")
        return

    conflicts = _check_conflicts(renames)
    if conflicts:
        print(f"\nWARNING: {len(conflicts)} conflict(s) — destination already exists:")
        for old, new in conflicts:
            print(f"  CONFLICT  {old.name} → {new.name}")
        print("\nResolve conflicts manually before running with --apply.\nAborting.\n")
        sys.exit(1)

    # Group by old JSON stem for readable output
    stem_groups: dict[str, list[tuple[Path, Path]]] = {}
    for old, new in renames:
        key = old.stem.removesuffix("_SI")
        stem_groups.setdefault(key, []).append((old, new))

    action = "Renaming" if apply else "Would rename"
    print(f"\n{action} {len(stem_groups)} paper(s) ({len(renames)} file(s)):\n")

    for old_stem in sorted(stem_groups):
        for old, new in stem_groups[old_stem]:
            print(f"  {old.name}")
            print(f"    → {new.name}")
        print()

    if not apply:
        print("Dry run — pass --apply to perform these renames.\n")
        return

    renamed = 0
    failed = 0
    for old, new in renames:
        try:
            old.rename(new)
            renamed += 1
        except OSError as exc:
            logger.error("Failed to rename %s → %s: %s", old.name, new.name, exc)
            failed += 1

    print(f"Done. Renamed {renamed} file(s).", end="")
    if failed:
        print(f"  {failed} failed (see log above).", end="")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the migration."""
    parser = argparse.ArgumentParser(
        description=(
            "Rename library files to match the current pyiso4 naming convention."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--library-dir",
        default="./library",
        metavar="DIR",
        help="Library root directory (default: ./library)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames (default: dry-run preview only)",
    )
    args = parser.parse_args()

    migrate(Path(args.library_dir), apply=args.apply)


if __name__ == "__main__":
    main()
