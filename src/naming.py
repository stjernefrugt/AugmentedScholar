"""File naming utilities for the AugmentedScholar library.

Naming convention::

    {first_author}_{last_author}_{year}_{journal_abbr}.pdf
    {first_author}_{last_author}_{year}_{journal_abbr}_SI.pdf

Journal abbreviations follow ISO 4 (LTWA) via pyiso4, with dots removed
and words hyphen-joined to produce filesystem-safe tokens such as
``nat-commun`` or ``phys-rev-lett``.
"""

from __future__ import annotations

import re

from pyiso4.ltwa import Abbreviate

from .models import PaperMetadata

# Initialised once at module load — parses the bundled LTWA data file.
_abbr = Abbreviate.create()


def abbreviate_journal(journal: str) -> str:
    """Return a dot-free, hyphen-delimited ISO 4 abbreviation for a journal name.

    Uses the LTWA (List of Title Word Abbreviations) via ``pyiso4``.  Dots are
    stripped and whitespace is replaced with hyphens so the result is safe for
    use in filenames.

    Args:
        journal: Full journal name as returned by an external API.

    Returns:
        Lowercase hyphen-delimited abbreviation (e.g. ``'nat-commun'``,
        ``'phys-rev-lett'``).

    Examples:
        >>> abbreviate_journal("Nature Communications")
        'nat-commun'
        >>> abbreviate_journal("Physical Review Letters")
        'phys-rev-lett'
        >>> abbreviate_journal("Journal of the American Chemical Society")
        'j-am-chem-soc'
    """
    raw: str = _abbr(journal.strip())
    no_dots = raw.replace(".", "")
    lowered = no_dots.lower()
    hyphenated = re.sub(r"[\s&/()]+", "-", lowered)
    return re.sub(r"-+", "-", hyphenated).strip("-") or "unknown-journal"


def build_stem(metadata: PaperMetadata, *, is_si: bool = False) -> str:
    """Build the filename stem (without extension) for a paper.

    Args:
        metadata: Parsed paper metadata.
        is_si: If ``True``, appends ``_SI`` to the stem.

    Returns:
        Underscore-delimited stem string following the convention::

            {first_author}_{last_author}_{year}_{journal_abbr}[_SI]
    """
    first = metadata.first_author_lastname
    last = metadata.last_author_lastname
    year = str(metadata.year)
    journal = abbreviate_journal(metadata.journal)

    stem = f"{first}_{last}_{year}_{journal}"
    if is_si:
        stem += "_SI"
    return stem


def generate_filename(metadata: PaperMetadata, *, is_si: bool = False) -> str:
    """Generate the canonical PDF filename for a paper.

    Args:
        metadata: Parsed paper metadata.
        is_si: If ``True``, generates the supplementary-information filename.

    Returns:
        Full filename string with ``.pdf`` extension.

    Examples:
        >>> from src.models import Author, PaperMetadata
        >>> meta = PaperMetadata(
        ...     title="Example",
        ...     authors=[Author(name="Alice Smith"), Author(name="Bob Jones")],
        ...     journal="Nature",
        ...     year=2024,
        ... )
        >>> generate_filename(meta)
        'smith_jones_2024_nature.pdf'
        >>> generate_filename(meta, is_si=True)
        'smith_jones_2024_nature_SI.pdf'
    """
    return f"{build_stem(metadata, is_si=is_si)}.pdf"
