"""Unit tests for src/naming.py."""

from __future__ import annotations

from src.models import Author, PaperMetadata
from src.naming import abbreviate_journal, build_stem, generate_filename


def _meta(
    authors: list[str],
    journal: str,
    year: int = 2024,
) -> PaperMetadata:
    return PaperMetadata(
        title="Example Paper",
        authors=[Author(name=n) for n in authors],
        journal=journal,
        year=year,
    )


class TestAbbreviateJournal:
    def test_single_word_unchanged(self) -> None:
        # pyiso4 keeps single-word titles that are not in LTWA as-is
        assert abbreviate_journal("Nature") == "nature"

    def test_nature_communications(self) -> None:
        assert abbreviate_journal("Nature Communications") == "nat-commun"

    def test_physical_review_letters(self) -> None:
        assert abbreviate_journal("Physical Review Letters") == "phys-rev-lett"

    def test_jacs(self) -> None:
        assert (
            abbreviate_journal("Journal of the American Chemical Society")
            == "j-am-chem-soc"
        )

    def test_pnas(self) -> None:
        assert (
            abbreviate_journal("Proceedings of the National Academy of Sciences")
            == "proc-natl-acad-sci"
        )

    def test_two_word_fallback(self) -> None:
        assert abbreviate_journal("Some Journal") == "some-j"

    def test_journal_of_biochemistry(self) -> None:
        assert abbreviate_journal("Journal of Biochemistry") == "j-biochem"

    def test_no_dots_in_output(self) -> None:
        result = abbreviate_journal("Physical Review Letters")
        assert "." not in result

    def test_lowercase_output(self) -> None:
        result = abbreviate_journal("Nature Communications")
        assert result == result.lower()

    def test_leading_trailing_whitespace(self) -> None:
        assert abbreviate_journal("  Nature  ") == "nature"


class TestBuildStem:
    def test_standard_two_author(self) -> None:
        meta = _meta(["Alice Smith", "Bob Jones"], "Nature", 2024)
        assert build_stem(meta) == "smith_jones_2024_nature"

    def test_si_suffix(self) -> None:
        meta = _meta(["Alice Smith", "Bob Jones"], "Nature", 2024)
        assert build_stem(meta, is_si=True) == "smith_jones_2024_nature_SI"

    def test_single_author(self) -> None:
        meta = _meta(["Alice Smith"], "Science", 2020)
        assert build_stem(meta) == "smith_smith_2020_science"

    def test_unicode_author(self) -> None:
        meta = _meta(["José García", "Anna Müller"], "Cell", 2022)
        assert build_stem(meta) == "garcia_muller_2022_cell"

    def test_journal_abbreviation_applied(self) -> None:
        meta = _meta(["Alice Smith", "Bob Jones"], "Physical Review Letters", 2023)
        assert build_stem(meta) == "smith_jones_2023_phys-rev-lett"

    def test_year_is_string_in_stem(self) -> None:
        meta = _meta(["Alice Smith", "Bob Jones"], "Nature", 1999)
        assert "1999" in build_stem(meta)

    def test_no_dots_in_stem(self) -> None:
        meta = _meta(
            ["Alice Smith", "Bob Jones"],
            "Journal of the American Chemical Society",
            2022,
        )
        assert "." not in build_stem(meta)


class TestGenerateFilename:
    def test_main_pdf(self) -> None:
        meta = _meta(["Alice Smith", "Bob Jones"], "Nature", 2024)
        assert generate_filename(meta) == "smith_jones_2024_nature.pdf"

    def test_si_pdf(self) -> None:
        meta = _meta(["Alice Smith", "Bob Jones"], "Nature", 2024)
        assert generate_filename(meta, is_si=True) == "smith_jones_2024_nature_SI.pdf"

    def test_extension_always_pdf(self) -> None:
        meta = _meta(["A B"], "Nature", 2024)
        assert generate_filename(meta).endswith(".pdf")
