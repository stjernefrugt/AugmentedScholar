"""Unit tests for src/ingest.py.

All external I/O (HTTP, filesystem) is either mocked or exercised via
temporary directories so no network access is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.ingest import IngestionPipeline
from src.models import Author, PaperMetadata

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_metadata() -> PaperMetadata:
    """Return a realistic PaperMetadata instance for use across tests."""
    return PaperMetadata(
        title="Highly accurate protein structure prediction with AlphaFold",
        authors=[
            Author(name="John Jumper"),
            Author(name="Demis Hassabis"),
        ],
        journal="Nature",
        year=2021,
        doi="10.1038/s41586-021-03819-2",
        abstract="A description of AlphaFold.",
        citation_count=12000,
        keywords=["protein structure", "deep learning"],
    )


@pytest.fixture()
def mock_fetcher(sample_metadata: PaperMetadata) -> MagicMock:
    """Return a MetadataFetcher mock that always returns sample_metadata."""
    fetcher = MagicMock()
    fetcher.fetch_by_doi.return_value = sample_metadata
    return fetcher


@pytest.fixture()
def mock_pdf_handler(tmp_path: Path) -> MagicMock:
    """Return a PDFHandler mock whose download() creates a real empty file."""

    def fake_download(url: str, filename: str) -> Path:
        dest = tmp_path / filename
        dest.write_bytes(b"%PDF-1.4 fake")
        return dest

    handler = MagicMock()
    handler.download.side_effect = fake_download
    return handler


@pytest.fixture()
def pipeline(
    tmp_path: Path,
    mock_fetcher: MagicMock,
    mock_pdf_handler: MagicMock,
) -> IngestionPipeline:
    """Return a fully wired IngestionPipeline backed by mocks."""
    return IngestionPipeline(
        library_dir=tmp_path,
        fetcher=mock_fetcher,
        pdf_handler=mock_pdf_handler,
    )


# ---------------------------------------------------------------------------
# Tests — metadata path and filenames
# ---------------------------------------------------------------------------


class TestRunMetadataOutput:
    def test_metadata_json_is_written(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        assert result.metadata_path.exists()
        assert result.metadata_path.suffix == ".json"

    def test_metadata_json_content(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        payload = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        expected_title = "Highly accurate protein structure prediction with AlphaFold"
        assert payload["title"] == expected_title
        assert payload["year"] == 2021
        assert payload["doi"] == "10.1038/s41586-021-03819-2"
        assert "John Jumper" in payload["authors"]

    def test_metadata_filename_follows_convention(
        self, pipeline: IngestionPipeline, tmp_path: Path
    ) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        # jumper_hassabis_2021_nature.json
        assert result.metadata_path.name == "jumper_hassabis_2021_nature.json"

    def test_result_metadata_matches_fetcher_output(
        self,
        pipeline: IngestionPipeline,
        sample_metadata: PaperMetadata,
    ) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        assert result.metadata == sample_metadata


# ---------------------------------------------------------------------------
# Tests — BibTeX generation
# ---------------------------------------------------------------------------


class TestBibtex:
    def test_bibtex_is_article_type(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        assert result.bibtex.startswith("@article{")

    def test_bibtex_citation_key(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        assert "jumper_hassabis_2021_nature" in result.bibtex

    def test_bibtex_contains_doi(self, pipeline: IngestionPipeline) -> None:
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        assert "10.1038/s41586-021-03819-2" in result.bibtex

    def test_bibtex_no_doi_field_when_none(
        self,
        tmp_path: Path,
        mock_pdf_handler: MagicMock,
    ) -> None:
        no_doi_meta = PaperMetadata(
            title="No DOI Paper",
            authors=[Author(name="A B"), Author(name="C D")],
            journal="Science",
            year=2020,
        )
        fetcher = MagicMock()
        fetcher.fetch_by_doi.return_value = no_doi_meta
        pipeline = IngestionPipeline(
            library_dir=tmp_path,
            fetcher=fetcher,
            pdf_handler=mock_pdf_handler,
        )
        result = pipeline.run(doi="")
        assert "doi" not in result.bibtex


# ---------------------------------------------------------------------------
# Tests — PDF download control
# ---------------------------------------------------------------------------


class TestPdfDownload:
    def test_no_pdf_download_when_no_url(
        self,
        pipeline: IngestionPipeline,
        mock_pdf_handler: MagicMock,
    ) -> None:
        # sample_metadata has no pdf_url; no explicit url passed
        result = pipeline.run(doi="10.1038/s41586-021-03819-2")
        mock_pdf_handler.download.assert_not_called()
        assert result.pdf_path is None

    def test_pdf_downloaded_when_url_provided(
        self,
        pipeline: IngestionPipeline,
        mock_pdf_handler: MagicMock,
        tmp_path: Path,
    ) -> None:
        result = pipeline.run(
            doi="10.1038/s41586-021-03819-2",
            pdf_url="https://example.com/paper.pdf",
        )
        mock_pdf_handler.download.assert_called_once()
        assert result.pdf_path is not None
        assert result.pdf_path.name == "jumper_hassabis_2021_nature.pdf"

    def test_si_downloaded_when_si_url_provided(
        self,
        pipeline: IngestionPipeline,
        mock_pdf_handler: MagicMock,
    ) -> None:
        result = pipeline.run(
            doi="10.1038/s41586-021-03819-2",
            si_url="https://example.com/si.pdf",
        )
        assert result.si_path is not None
        assert result.si_path.name == "jumper_hassabis_2021_nature_SI.pdf"

    def test_explicit_empty_pdf_url_skips_download(
        self,
        tmp_path: Path,
        mock_pdf_handler: MagicMock,
    ) -> None:
        # Metadata has an oa pdf_url; passing "" should override and skip
        meta_with_url = PaperMetadata(
            title="T",
            authors=[Author(name="A B"), Author(name="C D")],
            journal="Nature",
            year=2024,
            pdf_url="https://example.com/paper.pdf",
        )
        fetcher = MagicMock()
        fetcher.fetch_by_doi.return_value = meta_with_url
        pipeline = IngestionPipeline(
            library_dir=tmp_path,
            fetcher=fetcher,
            pdf_handler=mock_pdf_handler,
        )
        result = pipeline.run(doi="10.fake/doi", pdf_url="")
        mock_pdf_handler.download.assert_not_called()
        assert result.pdf_path is None

    def test_pdf_url_falls_back_to_metadata(
        self,
        tmp_path: Path,
        mock_pdf_handler: MagicMock,
    ) -> None:
        meta_with_url = PaperMetadata(
            title="T",
            authors=[Author(name="A B"), Author(name="C D")],
            journal="Nature",
            year=2024,
            pdf_url="https://example.com/oa.pdf",
        )
        fetcher = MagicMock()
        fetcher.fetch_by_doi.return_value = meta_with_url
        pipeline = IngestionPipeline(
            library_dir=tmp_path,
            fetcher=fetcher,
            pdf_handler=mock_pdf_handler,
        )
        result = pipeline.run(doi="10.fake/doi")
        mock_pdf_handler.download.assert_called_once_with(
            "https://example.com/oa.pdf", "b_d_2024_nature.pdf"
        )
        assert result.pdf_path is not None

    def test_pdf_download_failure_is_non_fatal(
        self,
        pipeline: IngestionPipeline,
        mock_pdf_handler: MagicMock,
    ) -> None:
        """A 403 / network error on the PDF must not abort metadata ingestion."""
        mock_pdf_handler.download.side_effect = RuntimeError("403 Forbidden")
        result = pipeline.run(
            doi="10.1038/s41586-021-03819-2",
            pdf_url="https://example.com/paywalled.pdf",
        )
        # pdf_path is None but the run succeeded and metadata JSON was written
        assert result.pdf_path is None
        assert result.metadata_path.exists()
