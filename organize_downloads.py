#!/usr/bin/env python3
"""Match manually downloaded PDFs to the missing-PDFs manifest and move them.

Scans *downloads_dir* for ``*.pdf`` files and tries to pair each one with
an entry in *missing_pdfs.json* using three strategies (tried in order):

1. **Exact stem** — the downloaded filename (without ``.pdf``) matches the
   expected library filename stem exactly.
2. **Fuzzy stem** — ``difflib.SequenceMatcher`` ratio between the normalised
   download stem and the expected stem is ≥ ``--min-ratio``.
3. **Fuzzy title** — same ratio test against the paper *title* from the
   manifest entry.

Each manifest destination is claimed by at most one source file (the
highest-confidence match wins).

Usage examples::

    # Dry run (default) — print what would happen
    python organize_downloads.py

    # Apply the moves
    python organize_downloads.py --apply

    # Suppress per-file confirmation prompt
    python organize_downloads.py --apply --no-confirm

    # Custom directories
    python organize_downloads.py \\
        --downloads-dir ~/Desktop \\
        --library-dir ./library2 \\
        --apply
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from src.pdf_handler import PDFHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip extension, collapse punctuation/whitespace to spaces.

    Args:
        text: Filename stem or paper title.

    Returns:
        Normalised string suitable for fuzzy comparison.
    """
    text = text.lower()
    # Remove file extension if present
    text = re.sub(r"\.\w{1,5}$", "", text)
    # Collapse punctuation and whitespace runs to a single space
    text = re.sub(r"[^\w]+", " ", text)
    return text.strip()


def _ratio(a: str, b: str) -> float:
    """Return SequenceMatcher similarity ratio for normalised *a* and *b*."""
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
# Match planning
# ---------------------------------------------------------------------------


def _plan_moves(
    downloads: list[Path],
    manifest: list[dict],
    min_ratio: float,
) -> list[tuple[Path, Path, float, str]]:
    """Compute the best download→library moves above *min_ratio*.

    Args:
        downloads: PDF files found in the downloads directory.
        manifest: Parsed entries from ``missing_pdfs.json``.
        min_ratio: Minimum similarity score to accept a match.

    Returns:
        List of ``(src, dest, score, match_method)`` tuples, one per match,
        sorted best-score-first.  Each *dest* appears at most once.
    """
    # Pre-compute normalised keys for all manifest entries
    entries: list[tuple[dict, str, str]] = []
    for entry in manifest:
        expected = Path(entry.get("expected_pdf_path", ""))
        stem_norm = _normalize(expected.stem) if expected.stem else ""
        title_norm = _normalize(entry.get("title") or "")
        entries.append((entry, stem_norm, title_norm))

    # For each download, find the best manifest match
    candidates: list[tuple[float, str, Path, Path]] = []  # (score, method, src, dest)
    for dl in downloads:
        dl_norm = _normalize(dl.stem)
        best_score = 0.0
        best_entry: dict | None = None
        best_method = ""

        for entry, stem_norm, title_norm in entries:
            expected = Path(entry.get("expected_pdf_path", ""))
            if not expected.name:
                continue

            # Strategy 1: exact stem
            if dl.stem == expected.stem:
                score, method = 1.0, "exact"
            else:
                # Strategy 2: fuzzy stem
                fs = _ratio(dl_norm, stem_norm) if stem_norm else 0.0
                # Strategy 3: fuzzy title
                ft = _ratio(dl_norm, title_norm) if title_norm else 0.0
                if fs >= ft:
                    score, method = fs, "fuzzy-stem"
                else:
                    score, method = ft, "fuzzy-title"

            if score > best_score:
                best_score = score
                best_entry = entry
                best_method = method

        if best_entry is not None and best_score >= min_ratio:
            dest = Path(best_entry["expected_pdf_path"])
            candidates.append((best_score, best_method, dl, dest))

    # Sort by descending score; claim each dest only once (greedy)
    candidates.sort(key=lambda c: c[0], reverse=True)
    claimed_dests: set[str] = set()
    moves: list[tuple[Path, Path, float, str]] = []
    for score, method, src, dest in candidates:
        dest_key = str(dest)
        if dest_key in claimed_dests:
            continue
        claimed_dests.add(dest_key)
        moves.append((src, dest, score, method))

    return sorted(moves, key=lambda m: m[2], reverse=True)


# ---------------------------------------------------------------------------
# Move execution
# ---------------------------------------------------------------------------


def _apply_move(src: Path, dest: Path, handler: PDFHandler) -> bool:
    """Move *src* to *dest* using PDFHandler; fall back to copy+delete cross-device.

    Args:
        src: Source PDF (in downloads dir).
        dest: Target path (inside library dir).
        handler: Configured :class:`~src.pdf_handler.PDFHandler` instance.

    Returns:
        ``True`` on success, ``False`` on failure.
    """
    if dest.exists():
        logger.info("SKIP — already exists: %s", dest.name)
        return False

    try:
        handler.rename(src, dest.name)
        return True
    except OSError as exc:
        if "Invalid cross-device link" in str(exc) or exc.errno == 18:
            # Cross-device: copy then delete original
            logger.warning("Cross-device move detected; copying instead of renaming.")
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                src.unlink()
                logger.info("Copied and removed original: %s → %s", src.name, dest.name)
                return True
            except OSError as copy_exc:
                logger.error("Copy failed for '%s': %s", src.name, copy_exc)
                print(
                    f"  ERROR: could not copy '{src}' to '{dest}': {copy_exc}\n"
                    f"  Tip: copy the file manually then re-run with --apply."
                )
                return False
        logger.error("Move failed for '%s': %s", src.name, exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the organiser."""
    parser = argparse.ArgumentParser(
        description="Match manually downloaded PDFs to the manifest and move them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--downloads-dir",
        default="~/Downloads",
        metavar="DIR",
        help="Directory to scan for downloaded PDFs (default: ~/Downloads)",
    )
    parser.add_argument(
        "--library-dir",
        default="./library",
        metavar="DIR",
        help="Library root directory (default: ./library)",
    )
    parser.add_argument(
        "--manifest",
        default="./library/missing_pdfs.json",
        metavar="FILE",
        help="Path to missing_pdfs.json (default: ./library/missing_pdfs.json)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files (default: dry run only)",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip per-file confirmation prompt when --apply is set",
    )
    parser.add_argument(
        "--min-ratio",
        type=float,
        default=0.6,
        metavar="RATIO",
        help="Minimum fuzzy-match similarity score 0–1 (default: 0.6)",
    )
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir).expanduser().resolve()
    library_dir = Path(args.library_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not downloads_dir.is_dir():
        print(f"Downloads directory not found: {downloads_dir}", file=sys.stderr)
        sys.exit(1)

    if not manifest_path.exists():
        print(
            f"Manifest not found: {manifest_path}\n"
            "Run `python run_expansion.py` to generate it.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        manifest: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read manifest: {exc}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(manifest, list):
        print("Manifest must be a JSON array.", file=sys.stderr)
        sys.exit(1)

    downloads = sorted(downloads_dir.glob("*.pdf"))
    if not downloads:
        print(f"No PDF files found in {downloads_dir}")
        sys.exit(0)

    # Filter manifest to only entries still missing their PDF
    pending = [
        e
        for e in manifest
        if e.get("expected_pdf_path") and not Path(e["expected_pdf_path"]).exists()
    ]

    print(
        f"\nDownloads dir : {downloads_dir}  ({len(downloads)} PDF files)\n"
        f"Library dir   : {library_dir}\n"
        f"Manifest      : {manifest_path}  ({len(pending)} missing entries)\n"
        f"Min ratio     : {args.min_ratio}\n"
        f"Mode          : {'APPLY' if args.apply else 'DRY RUN'}\n"
    )

    moves = _plan_moves(downloads, pending, args.min_ratio)

    if not moves:
        print("No matches found above the similarity threshold.")
        return

    divider = "-" * 70
    print(divider)
    print(f"{'Score':>6}  {'Method':<13}  {'Download':<30}  →  Target")
    print(divider)
    for src, dest, score, method in moves:
        short_src = src.name if len(src.name) <= 30 else src.name[:27] + "…"
        print(f"  {score:.2f}  {method:<13}  {short_src:<30}  →  {dest.name}")
    print(divider)
    print(f"  {len(moves)} match(es) found")

    unmatched = [d for d in downloads if d not in {m[0] for m in moves}]
    if unmatched:
        print(f"\n  {len(unmatched)} file(s) not matched (score < {args.min_ratio}):")
        for u in unmatched:
            print(f"    {u.name}")

    if not args.apply:
        print("\n  Run with --apply to perform the moves.\n")
        return

    print()
    handler = PDFHandler(library_dir)
    moved = 0
    skipped = 0

    for src, dest, _score, _method in moves:
        label = f"  {src.name[:40]:<40}  →  {dest.name}"
        if dest.exists():
            print(f"  SKIP (exists) {dest.name}")
            skipped += 1
            continue

        if not args.no_confirm:
            ans = input(f"{label}\n  Move this file? [y/N] ").strip().lower()
            if ans not in {"y", "yes"}:
                print("  Skipped.")
                skipped += 1
                continue
        else:
            print(label)

        if _apply_move(src, dest, handler):
            moved += 1
        else:
            skipped += 1

    print(f"\n  Moved   : {moved}")
    print(f"  Skipped : {skipped}\n")


if __name__ == "__main__":
    main()
