"""Post-ingestion metadata enrichment for the AugmentedScholar library.

Scans a library directory for JSON sidecar files that are missing abstracts
and fills them in via a DOI lookup against Semantic Scholar / OpenAlex.

Typical use-case: a ``--depth 0`` run populates metadata from Google Scholar,
which does not return full abstracts.  Running ``enrich_library`` afterwards
fetches the missing fields without re-downloading PDFs or rebuilding filenames.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .metadata_fetcher import MetadataFetcher

logger = logging.getLogger(__name__)


def enrich_library(
    library_dir: Path,
    fetcher: MetadataFetcher,
    *,
    delay: float = 1.0,
    force: bool = False,
) -> tuple[int, int]:
    """Enrich JSON sidecars that are missing abstracts via DOI lookup.

    For each ``*.json`` file in *library_dir* that has a ``doi`` field but
    no ``abstract`` (or ``force=True``), fetches full metadata and merges
    the following fields back into the sidecar:

    * ``abstract``
    * ``citation_count``
    * ``keywords``
    * ``pdf_url`` (only if currently absent)

    Args:
        library_dir: Root directory containing JSON sidecar files.
        fetcher: Configured :class:`~src.metadata_fetcher.MetadataFetcher`.
        delay: Seconds to sleep between API requests (default 1.0 s).
        force: If ``True``, re-fetch even papers that already have abstracts.

    Returns:
        ``(enriched, failed)`` counts.
    """
    json_files = sorted(library_dir.glob("*.json"))
    enriched = 0
    failed = 0

    for json_path in json_files:
        try:
            data: dict[str, object] = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read '%s': %s", json_path, exc)
            continue

        doi = data.get("doi")
        if not doi:
            logger.debug("No DOI in %s — skipping enrichment", json_path.name)
            continue

        has_abstract = bool(data.get("abstract"))
        if has_abstract and not force:
            continue

        logger.info("Enriching %s (doi=%s)", json_path.name, doi)
        time.sleep(delay)

        try:
            meta = fetcher.fetch_by_doi(str(doi))
        except Exception as exc:
            logger.warning("Enrichment failed for '%s': %s", doi, exc)
            failed += 1
            continue

        changed = False
        if meta.abstract and (not has_abstract or force):
            data["abstract"] = meta.abstract
            changed = True
        if meta.citation_count:
            data["citation_count"] = meta.citation_count
            changed = True
        if meta.keywords:
            data["keywords"] = meta.keywords
            changed = True
        if meta.pdf_url and not data.get("pdf_url"):
            data["pdf_url"] = meta.pdf_url
            changed = True

        if changed:
            try:
                json_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                enriched += 1
                logger.debug("Updated %s", json_path.name)
            except OSError as exc:
                logger.warning("Could not write '%s': %s", json_path, exc)
                failed += 1

    logger.info("Enrichment complete: %d updated, %d failed", enriched, failed)
    return enriched, failed
