"""Bibliographic Expansion Engine — BFS citation traversal.

Traverses three tiers of the citation ecosystem around an author:

* **Tier 0** — the author's own publications.
* **Tier 1** — papers the author's work cites (references).
* **Tier 2** — papers that cite the author's work (cited-by).

Higher tiers are reachable via ``max_depth > 1``, but can retrieve
thousands of papers; proceed with care.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

from .citation_models import CitationRecord, ExpansionResult, RelationshipType
from .graph import CitationGraph
from .gscholar_client import GoogleScholarClient
from .ingest import IngestionPipeline
from .models import Author, PaperMetadata
from .naming import generate_filename
from .ss_client import SemanticScholarClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_doi(paper: dict[str, Any]) -> str | None:
    """Extract a normalised DOI from a Semantic Scholar paper dict.

    Args:
        paper: Raw paper dict as returned by the SS Graph API.

    Returns:
        Stripped DOI string without URL prefix, or ``None`` if absent.
    """
    ids: dict[str, Any] = paper.get("externalIds") or {}
    doi: str | None = ids.get("DOI")
    return doi.strip() if doi else None


def _parse_ss_paper(paper: dict[str, Any]) -> PaperMetadata | None:
    """Convert a Semantic Scholar paper dict into a PaperMetadata.

    Args:
        paper: Raw paper dict from the SS Graph API.

    Returns:
        :class:`~src.models.PaperMetadata` if required fields are present,
        otherwise ``None``.
    """
    title: str | None = paper.get("title")
    year: int | None = paper.get("year")
    if not title or not year:
        return None

    authors = [
        Author(name=a.get("name", "Unknown")) for a in (paper.get("authors") or [])
    ]
    journal_info: dict[str, Any] = paper.get("journal") or {}
    journal: str = journal_info.get("name") or "Unknown Journal"

    oa_pdf: dict[str, Any] | None = paper.get("openAccessPdf")
    pdf_url: str | None = oa_pdf.get("url") if oa_pdf else None

    return PaperMetadata(
        title=title,
        authors=authors,
        journal=journal,
        year=int(year),
        doi=_extract_doi(paper),
        abstract=paper.get("abstract"),
        citation_count=int(paper.get("citationCount") or 0),
        pdf_url=pdf_url,
    )


# ---------------------------------------------------------------------------
# CitationExplorer
# ---------------------------------------------------------------------------

# BFS queue item: (paper_dict, relationship_type, source_doi, current_depth)
_QueueItem = tuple[dict[str, Any], RelationshipType, str | None, int]


class CitationExplorer:
    """Traverse the citation ecosystem around an author's body of work.

    Performs a breadth-first expansion from a Semantic Scholar author ID.
    Deduplication is handled via a ``visited`` DOI set seeded from the
    optional ``known_dois`` argument (e.g. populated from ChromaDB).

    Args:
        library_dir: Root directory for PDFs and metadata JSON files.
        pipeline: Configured :class:`~src.ingest.IngestionPipeline`.
        graph: :class:`~src.graph.CitationGraph` to update in place.
        ss_client: Semantic Scholar client (created automatically if
            omitted).
        max_depth: Maximum traversal depth.  ``0`` = author's own papers
            only, ``1`` = + direct references and citations (default),
            ``2`` = + one further hop from each Tier 1/2 paper.
        known_dois: Pre-populated set of DOIs to treat as already
            ingested and skip.  Useful for linking against ChromaDB state.
    """

    def __init__(
        self,
        library_dir: Path,
        pipeline: IngestionPipeline,
        graph: CitationGraph,
        ss_client: SemanticScholarClient | None = None,
        max_depth: int = 1,
        known_dois: set[str] | None = None,
    ) -> None:
        self.library_dir = library_dir
        self._pipeline = pipeline
        self._graph = graph
        self._ss = ss_client or SemanticScholarClient()
        self.max_depth = max_depth
        self._seed_dois: set[str] = set(known_dois or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand_from_author(self, author_id: str) -> ExpansionResult:
        """Run a full citation expansion from an author's Semantic Scholar ID.

        Args:
            author_id: Semantic Scholar integer author ID (e.g. ``'1741101'``).

        Returns:
            :class:`~src.citation_models.ExpansionResult` summarising all
            records discovered during the run.
        """
        logger.info(
            "Starting expansion for author %s (max_depth=%d)",
            author_id,
            self.max_depth,
        )

        records: list[CitationRecord] = []
        errors: list[str] = []
        visited: set[str] = set(self._seed_dois)

        queue: deque[_QueueItem] = deque()
        tier0 = self._ss.get_author_papers(author_id)
        logger.info("Found %d Tier 0 papers for author %s", len(tier0), author_id)

        for paper in tier0:
            queue.append((paper, RelationshipType.IS_AUTHOR_OF, None, 0))

        while queue:
            paper, rel_type, source_doi, depth = queue.popleft()
            doi = _extract_doi(paper)
            ss_id: str | None = paper.get("paperId")

            if not doi or doi in visited:
                continue

            visited.add(doi)
            record = self._process_paper(paper, rel_type, source_doi, depth, errors)
            if record:
                records.append(record)
                if source_doi:
                    self._graph.add_edge(source_doi, doi, rel_type)

            if ss_id and depth < self.max_depth:
                self._enqueue_neighbors(ss_id, doi, depth, visited, queue)

        new_count = sum(1 for r in records if r.newly_ingested)
        skipped = max(0, len(visited) - len(self._seed_dois) - new_count)
        logger.info(
            "Expansion complete: %d records, %d new, %d skipped, %d errors",
            len(records),
            new_count,
            skipped,
            len(errors),
        )
        return ExpansionResult(
            author_id=author_id,
            records=records,
            new_ingestions=new_count,
            skipped=skipped,
            errors=errors,
        )

    def expand_from_google_scholar(
        self,
        url_or_id: str,
        gscholar_client: GoogleScholarClient | None = None,
    ) -> ExpansionResult:
        """Hybrid expansion: Google Scholar for Tier 0, Semantic Scholar for Tier 1/2.

        Google Scholar typically indexes a more complete author publication list
        than Semantic Scholar.  This method uses GS to seed Tier 0, then resolves
        each paper to a Semantic Scholar ID (by title search) so that Tier 1/2
        citation traversal can proceed via the SS API.

        Args:
            url_or_id: Google Scholar author ID or full profile URL
                (e.g. ``'https://scholar.google.com/citations?user=AbCdEfGhIjK'``).
            gscholar_client: Optional pre-configured
                :class:`~src.gscholar_client.GoogleScholarClient`.

        Returns:
            :class:`~src.citation_models.ExpansionResult` from the full run.
        """
        gs = gscholar_client or GoogleScholarClient()
        gs_papers = gs.get_author_papers(url_or_id)
        logger.info("Google Scholar returned %d Tier 0 papers", len(gs_papers))

        records: list[CitationRecord] = []
        errors: list[str] = []
        visited: set[str] = set(self._seed_dois)
        queue: deque[_QueueItem] = deque()

        for i, meta in enumerate(gs_papers):
            ss_paper: dict[str, Any] | None = None
            if self.max_depth > 0:
                # SS ID is only needed for Tier 1/2 traversal; skip at depth 0.
                # Throttle to avoid 429 — one request per _delay seconds.
                if i > 0:
                    time.sleep(self._ss._delay)
                ss_paper = self._resolve_to_ss_paper(meta, errors)

            if ss_paper is not None:
                doi = _extract_doi(ss_paper) or meta.doi
                if doi:
                    ss_paper.setdefault("externalIds", {})["DOI"] = doi
                queue.append((ss_paper, RelationshipType.IS_AUTHOR_OF, None, 0))
            elif meta.doi:
                # depth 0 or no SS match — ingest from GS metadata directly
                stub = self._metadata_to_ss_stub(meta)
                queue.append((stub, RelationshipType.IS_AUTHOR_OF, None, 0))

        # Reuse the standard BFS loop
        while queue:
            paper, rel_type, source_doi, depth = queue.popleft()
            doi = _extract_doi(paper)
            ss_id: str | None = paper.get("paperId")

            if not doi or doi in visited:
                continue

            visited.add(doi)
            record = self._process_paper(paper, rel_type, source_doi, depth, errors)
            if record:
                records.append(record)
                if source_doi:
                    self._graph.add_edge(source_doi, doi, rel_type)

            if ss_id and depth < self.max_depth:
                self._enqueue_neighbors(ss_id, doi, depth, visited, queue)

        new_count = sum(1 for r in records if r.newly_ingested)
        skipped = max(0, len(visited) - len(self._seed_dois) - new_count)
        return ExpansionResult(
            author_id=url_or_id,
            records=records,
            new_ingestions=new_count,
            skipped=skipped,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_to_ss_paper(
        self,
        meta: PaperMetadata,
        errors: list[str],
    ) -> dict[str, Any] | None:
        """Search Semantic Scholar for the SS paper dict matching *meta*.

        Args:
            meta: Paper metadata from Google Scholar.
            errors: Mutable list to append warnings to.

        Returns:
            SS paper dict with a ``paperId`` field, or ``None`` if not found.
        """
        result = self._ss.search_paper_by_title(meta.title)
        if result and result.get("paperId"):
            return result
        logger.debug("No SS match for '%s' — will ingest without SS ID", meta.title)
        return None

    @staticmethod
    def _metadata_to_ss_stub(meta: PaperMetadata) -> dict[str, Any]:
        """Build a minimal SS-shaped dict from a PaperMetadata for BFS ingestion.

        Used when a paper has a DOI but no Semantic Scholar record.

        Args:
            meta: Paper metadata to convert.

        Returns:
            Dict compatible with :func:`_parse_ss_paper`.
        """
        return {
            "paperId": None,
            "title": meta.title,
            "year": meta.year,
            "authors": [{"name": a.name} for a in meta.authors],
            "journal": {"name": meta.journal},
            "abstract": meta.abstract,
            "citationCount": meta.citation_count,
            "externalIds": {"DOI": meta.doi} if meta.doi else {},
            "openAccessPdf": {"url": meta.pdf_url} if meta.pdf_url else None,
        }

    def _enqueue_neighbors(
        self,
        ss_id: str,
        doi: str,
        depth: int,
        visited: set[str],
        queue: deque[_QueueItem],
    ) -> None:
        """Fetch references and citations for a paper and enqueue them.

        Args:
            ss_id: Semantic Scholar paper ID.
            doi: DOI of the current paper (becomes ``source_doi`` for children).
            depth: Current traversal depth.
            visited: Set of already-visited DOIs (read-only here).
            queue: BFS queue to append new items to.
        """
        next_depth = depth + 1
        for ref in self._ss.get_paper_references(ss_id):
            ref_doi = _extract_doi(ref)
            if ref_doi and ref_doi not in visited:
                queue.append((ref, RelationshipType.CITES, doi, next_depth))

        for cite in self._ss.get_paper_citations(ss_id):
            cite_doi = _extract_doi(cite)
            if cite_doi and cite_doi not in visited:
                queue.append((cite, RelationshipType.IS_CITED_BY, doi, next_depth))

    def _process_paper(
        self,
        paper: dict[str, Any],
        rel_type: RelationshipType,
        source_doi: str | None,
        tier: int,
        errors: list[str],
    ) -> CitationRecord | None:
        """Parse, ingest, and graph-register a single discovered paper.

        Args:
            paper: Raw Semantic Scholar paper dict.
            rel_type: Relationship to the root author.
            source_doi: DOI of the linking paper.
            tier: Traversal depth at which this paper was found.
            errors: Mutable list to append error descriptions to.

        Returns:
            :class:`~src.citation_models.CitationRecord` on success,
            ``None`` on parse or ingestion failure.
        """
        metadata = _parse_ss_paper(paper)
        if metadata is None:
            errors.append(
                f"Unparseable paper (missing title/year): {paper.get('paperId', '?')}"
            )
            return None

        filename = generate_filename(metadata)
        already_exists = (self.library_dir / filename).exists()
        newly_ingested = False

        if not already_exists:
            try:
                self._pipeline.ingest_from_metadata(
                    metadata=metadata,
                    pdf_url=metadata.pdf_url,
                )
                newly_ingested = True
            except Exception as exc:
                label = metadata.doi or metadata.title
                errors.append(f"{label}: {exc}")
                return None

        self._graph.add_paper(metadata, tier=tier)

        return CitationRecord(
            metadata=metadata,
            relationship_type=rel_type,
            source_doi=source_doi,
            tier=tier,
            newly_ingested=newly_ingested,
        )
