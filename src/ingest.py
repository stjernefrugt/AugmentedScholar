"""Main ingestion pipeline for the AugmentedScholar system.

Orchestrates the full ingestion sequence for a single paper:

1. Fetch bibliographic metadata (DOI → API).
2. Derive canonical filenames using the naming convention.
3. Optionally download the main PDF and/or supplementary information PDF.
4. Write a JSON metadata sidecar file.
5. Generate a BibTeX citation entry.
6. Return a structured :class:`~src.models.IngestionResult`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .metadata_fetcher import MetadataFetcher
from .models import IngestionResult, PaperMetadata
from .naming import build_stem, generate_filename
from .pdf_handler import PDFHandler
from .unpaywall_client import UnpaywallClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates end-to-end ingestion of a single research paper.

    Args:
        library_dir: Root directory for PDFs and metadata JSON files.
        fetcher: Optional :class:`~src.metadata_fetcher.MetadataFetcher`
            instance.  A default instance is created if not provided.
        pdf_handler: Optional :class:`~src.pdf_handler.PDFHandler` instance.
            A default instance targeting *library_dir* is created if not
            provided.

    Examples:
        >>> pipeline = IngestionPipeline(Path("/data/papers"))
        >>> result = pipeline.run(
        ...     doi="10.1038/s41586-021-03819-2",
        ...     pdf_url="https://example.com/paper.pdf",
        ... )
        >>> print(result.metadata_path)
    """

    def __init__(
        self,
        library_dir: Path,
        fetcher: MetadataFetcher | None = None,
        pdf_handler: PDFHandler | None = None,
        unpaywall_client: UnpaywallClient | None = None,
    ) -> None:
        self.library_dir = library_dir
        self._fetcher: MetadataFetcher = fetcher or MetadataFetcher()
        self._pdf_handler: PDFHandler = pdf_handler or PDFHandler(library_dir)
        self._unpaywall: UnpaywallClient | None = unpaywall_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        doi: str,
        pdf_url: str | None = None,
        si_url: str | None = None,
    ) -> IngestionResult:
        """Ingest a single paper identified by *doi*.

        Args:
            doi: DOI string, with or without the ``https://doi.org/`` prefix.
            pdf_url: Direct URL to the main PDF.  If ``None``, the pipeline
                falls back to any open-access URL returned by the metadata
                API.  Pass an empty string to explicitly skip downloading.
            si_url: Direct URL to the supplementary information PDF.  If
                ``None``, SI download is skipped.

        Returns:
            Populated :class:`~src.models.IngestionResult`.

        Raises:
            RuntimeError: If metadata cannot be fetched from any backend.
            httpx.HTTPStatusError: If a PDF URL returns a non-2xx response.
            ValueError: If a PDF URL returns a non-PDF content type.
        """
        logger.info("Starting ingestion for DOI: %s", doi)

        # --- 1. Metadata ------------------------------------------------
        metadata = self._fetcher.fetch_by_doi(doi)
        logger.debug(
            "Fetched metadata: title=%r year=%d", metadata.title, metadata.year
        )

        # --- 2. Resolve PDF URL -----------------------------------------
        resolved_pdf_url = self._resolve_pdf_url(pdf_url, metadata, doi)

        # --- 3. Derive filenames ----------------------------------------
        stem = build_stem(metadata)
        main_filename = generate_filename(metadata)
        si_filename = generate_filename(metadata, is_si=True)

        # --- 4. Download PDFs -------------------------------------------
        pdf_path = self._maybe_download(resolved_pdf_url, main_filename)
        si_path = self._maybe_download(si_url, si_filename)

        # --- 5. Write JSON sidecar --------------------------------------
        metadata_path = self._write_metadata(metadata, stem)

        # --- 6. Generate BibTeX -----------------------------------------
        bibtex = self._generate_bibtex(metadata, stem)

        logger.info("Ingestion complete for DOI: %s", doi)
        return IngestionResult(
            metadata=metadata,
            pdf_path=pdf_path,
            si_path=si_path,
            metadata_path=metadata_path,
            bibtex=bibtex,
        )

    def ingest_from_metadata(
        self,
        metadata: PaperMetadata,
        pdf_url: str | None = None,
        si_url: str | None = None,
    ) -> IngestionResult:
        """Ingest a paper from pre-fetched metadata, bypassing the API fetch.

        Use this when metadata has already been retrieved (e.g. from a
        citation traversal response) to avoid a redundant API round-trip.

        Args:
            metadata: Already-populated paper metadata.
            pdf_url: Direct URL to the main PDF.  Overrides any open-access
                URL stored in *metadata*.  Pass ``''`` to skip download.
            si_url: Direct URL to the supplementary information PDF.

        Returns:
            Populated :class:`~src.models.IngestionResult`.
        """
        logger.info("Ingesting from metadata: %r (%d)", metadata.title, metadata.year)
        resolved_pdf_url = self._resolve_pdf_url(pdf_url, metadata, metadata.doi)
        stem = build_stem(metadata)
        pdf_path = self._maybe_download(resolved_pdf_url, generate_filename(metadata))
        si_path = self._maybe_download(si_url, generate_filename(metadata, is_si=True))
        metadata_path = self._write_metadata(metadata, stem)
        bibtex = self._generate_bibtex(metadata, stem)
        return IngestionResult(
            metadata=metadata,
            pdf_path=pdf_path,
            si_path=si_path,
            metadata_path=metadata_path,
            bibtex=bibtex,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_pdf_url(
        self,
        explicit_url: str | None,
        metadata: PaperMetadata,
        doi: str | None = None,
    ) -> str | None:
        """Determine the effective PDF URL to use.

        Resolution order:
        1. *explicit_url* — caller-supplied override (empty string = skip).
        2. ``metadata.pdf_url`` — open-access URL from the metadata API.
        3. Unpaywall — queried by DOI when no URL was found above.

        Args:
            explicit_url: Caller-supplied URL override.
            metadata: Metadata that may contain an open-access URL.
            doi: DOI to pass to Unpaywall when other sources yield nothing.

        Returns:
            URL string to download, or ``None`` if no download should occur.
        """
        if explicit_url is not None:
            return explicit_url if explicit_url else None
        if metadata.pdf_url:
            return metadata.pdf_url
        if doi and self._unpaywall:
            return self._unpaywall.get_pdf_url(doi)
        return None

    def _maybe_download(self, url: str | None, filename: str) -> Path | None:
        """Download *url* to *filename* if *url* is not ``None``.

        Args:
            url: URL to download, or ``None`` to skip.
            filename: Target filename within :attr:`library_dir`.

        Returns:
            Path to the downloaded file, or ``None`` if skipped.
        """
        if not url:
            return None
        try:
            return self._pdf_handler.download(url, filename)
        except Exception as exc:
            logger.warning(
                "PDF download failed for '%s' (%s) — metadata will still be saved.",
                url,
                exc,
            )
            return None

    def _write_metadata(self, metadata: PaperMetadata, stem: str) -> Path:
        """Serialise *metadata* to a JSON sidecar file.

        The file is written to :attr:`library_dir` / ``{stem}.json``.

        Args:
            metadata: Paper metadata to serialise.
            stem: Filename stem (no extension) for the output file.

        Returns:
            Absolute path of the written JSON file.

        Raises:
            OSError: If the file cannot be written.
        """
        dest = self.library_dir / f"{stem}.json"
        payload = {
            "title": metadata.title,
            "authors": [a.name for a in metadata.authors],
            "journal": metadata.journal,
            "year": metadata.year,
            "doi": metadata.doi,
            "abstract": metadata.abstract,
            "summary": metadata.summary,
            "keywords": metadata.keywords,
            "citation_count": metadata.citation_count,
        }
        try:
            self.library_dir.mkdir(parents=True, exist_ok=True)
            dest.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:
            raise OSError(f"Failed to write metadata JSON to '{dest}': {exc}") from exc

        logger.debug("Wrote metadata JSON: %s", dest)
        return dest

    @staticmethod
    def _generate_bibtex(metadata: PaperMetadata, stem: str) -> str:
        """Generate a BibTeX ``@article`` entry for *metadata*.

        Args:
            metadata: Paper metadata.
            stem: Used as the BibTeX citation key.

        Returns:
            BibTeX entry string.
        """
        author_field = " and ".join(a.name for a in metadata.authors)
        doi_line = f"  doi = {{{metadata.doi}}},\n" if metadata.doi else ""
        return (
            f"@article{{{stem},\n"
            f"  title = {{{metadata.title}}},\n"
            f"  author = {{{author_field}}},\n"
            f"  journal = {{{metadata.journal}}},\n"
            f"  year = {{{metadata.year}}},\n"
            f"{doi_line}"
            f"}}"
        )
