"""Unit tests for src/enrichment.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.enrichment import enrich_library
from src.models import Author, PaperMetadata


def _make_fetcher(metadata: PaperMetadata) -> MagicMock:
    fetcher = MagicMock()
    fetcher.fetch_by_doi.return_value = metadata
    return fetcher


def _write_sidecar(path: Path, **fields: object) -> None:
    path.write_text(json.dumps(fields, ensure_ascii=False), encoding="utf-8")


def _rich_meta(abstract: str = "A great paper.") -> PaperMetadata:
    return PaperMetadata(
        title="T",
        authors=[Author(name="A B")],
        journal="Nature",
        year=2020,
        doi="10.fake/doi",
        abstract=abstract,
        citation_count=42,
        keywords=["kw1"],
    )


class TestEnrichLibrary:
    def test_fills_abstract_from_fetcher(self, tmp_path: Path) -> None:
        _write_sidecar(
            tmp_path / "paper.json",
            title="T",
            doi="10.fake/doi",
            abstract=None,
        )
        enriched, _ = enrich_library(tmp_path, _make_fetcher(_rich_meta()))
        data = json.loads((tmp_path / "paper.json").read_text())
        assert data["abstract"] == "A great paper."
        assert enriched == 1

    def test_skips_paper_with_existing_abstract(self, tmp_path: Path) -> None:
        _write_sidecar(
            tmp_path / "paper.json",
            title="T",
            doi="10.fake/doi",
            abstract="Already here.",
        )
        enriched, _ = enrich_library(tmp_path, _make_fetcher(_rich_meta()))
        assert enriched == 0

    def test_force_overwrites_existing_abstract(self, tmp_path: Path) -> None:
        _write_sidecar(
            tmp_path / "paper.json",
            title="T",
            doi="10.fake/doi",
            abstract="Old abstract.",
        )
        enriched, _ = enrich_library(
            tmp_path, _make_fetcher(_rich_meta("New abstract.")), force=True
        )
        data = json.loads((tmp_path / "paper.json").read_text())
        assert data["abstract"] == "New abstract."
        assert enriched == 1

    def test_skips_paper_without_doi(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path / "paper.json", title="T", doi=None)
        enriched, _ = enrich_library(tmp_path, _make_fetcher(_rich_meta()))
        assert enriched == 0

    def test_fetch_failure_counted_as_failed(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path / "paper.json", title="T", doi="10.bad/doi")
        fetcher = MagicMock()
        fetcher.fetch_by_doi.side_effect = RuntimeError("404")
        _, failed = enrich_library(tmp_path, fetcher)
        assert failed == 1

    def test_fills_citation_count(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path / "paper.json", title="T", doi="10.fake/doi")
        enrich_library(tmp_path, _make_fetcher(_rich_meta()))
        data = json.loads((tmp_path / "paper.json").read_text())
        assert data["citation_count"] == 42

    def test_fills_pdf_url_when_absent(self, tmp_path: Path) -> None:
        _write_sidecar(tmp_path / "paper.json", title="T", doi="10.fake/doi")
        meta = PaperMetadata(
            title="T",
            authors=[Author(name="A B")],
            journal="Nature",
            year=2020,
            doi="10.fake/doi",
            pdf_url="https://example.com/paper.pdf",
        )
        enrich_library(tmp_path, _make_fetcher(meta))
        data = json.loads((tmp_path / "paper.json").read_text())
        assert data["pdf_url"] == "https://example.com/paper.pdf"

    def test_does_not_overwrite_existing_pdf_url(self, tmp_path: Path) -> None:
        _write_sidecar(
            tmp_path / "paper.json",
            title="T",
            doi="10.fake/doi",
            pdf_url="https://existing.com/paper.pdf",
        )
        meta = PaperMetadata(
            title="T",
            authors=[Author(name="A B")],
            journal="Nature",
            year=2020,
            doi="10.fake/doi",
            pdf_url="https://new.com/paper.pdf",
        )
        enrich_library(tmp_path, _make_fetcher(meta))
        data = json.loads((tmp_path / "paper.json").read_text())
        assert data["pdf_url"] == "https://existing.com/paper.pdf"

    def test_multiple_files_processed(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_sidecar(tmp_path / f"paper{i}.json", doi=f"10.fake/{i}")
        enriched, _ = enrich_library(tmp_path, _make_fetcher(_rich_meta()))
        assert enriched == 3

    @pytest.mark.parametrize("delay", [0.0])
    def test_delay_parameter_accepted(self, tmp_path: Path, delay: float) -> None:
        _write_sidecar(tmp_path / "p.json", doi="10.fake/x")
        enrich_library(tmp_path, _make_fetcher(_rich_meta()), delay=delay)
