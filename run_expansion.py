#!/usr/bin/env python3
"""Entry point for the Bibliographic Expansion Engine.

Usage examples::

    # Find your Semantic Scholar author ID by name
    python run_expansion.py --find-author "Your Name"

    # Expand using Semantic Scholar author ID
    python run_expansion.py --author-id 1741101 --library-dir ./library

    # Expand using Google Scholar profile URL (more complete Tier 0)
    python run_expansion.py \\
        --gscholar-url "https://scholar.google.com/citations?user=AbCdEfGhIjK" \\
        --library-dir ./library

    # Full two-hop expansion with API key and existing ChromaDB DOI list
    python run_expansion.py \\
        --author-id 1741101 \\
        --depth 2 \\
        --api-key YOUR_KEY \\
        --known-dois library/known_dois.txt \\
        --output-json library/expansion_result.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.citation_explorer import CitationExplorer
from src.enrichment import enrich_library
from src.graph import CitationGraph
from src.gscholar_client import GoogleScholarClient
from src.ingest import IngestionPipeline
from src.metadata_fetcher import MetadataFetcher
from src.ss_client import SemanticScholarClient
from src.unpaywall_client import UnpaywallClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _find_author(name: str, api_key: str | None) -> None:
    """Print matching Semantic Scholar authors and exit.

    Args:
        name: Author display name to search for.
        api_key: Optional Semantic Scholar API key.
    """
    with SemanticScholarClient(api_key=api_key) as client:
        results = client.search_author(name)

    if not results:
        print(f"No authors found for '{name}'.")
        sys.exit(1)

    print(f"\nFound {len(results)} result(s) for '{name}':\n")
    for author in results[:10]:
        print(
            f"  ID: {str(author.get('authorId', '?')):<12}"
            f"  Papers: {str(author.get('paperCount', '?')):<6}"
            f"  Citations: {str(author.get('citationCount', '?')):<8}"
            f"  {author.get('name', '?')}"
        )
    print("\nPass the ID to --author-id to run an expansion.\n")


def _write_missing_pdfs_manifest(library_dir: Path, manifest_path: Path) -> int:
    """Scan *library_dir* for JSON sidecars without a matching PDF and write manifest.

    For each ``*.json`` sidecar that has no corresponding ``*.pdf``, emits a
    record containing the paper's bibliographic info and a
    ``https://doi.org/`` URL for Playwright-assisted downloading.

    Args:
        library_dir: Root directory containing sidecar JSON files.
        manifest_path: Destination path for the manifest JSON file.

    Returns:
        Number of missing-PDF entries written.
    """
    import json as _json

    entries = []
    for json_path in sorted(library_dir.glob("*.json")):
        if json_path.name == manifest_path.name:
            continue
        pdf_path = json_path.with_suffix(".pdf")
        if pdf_path.exists():
            continue
        try:
            data = _json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        doi = data.get("doi")
        entries.append(
            {
                "title": data.get("title", ""),
                "authors": data.get("authors", []),
                "year": data.get("year"),
                "journal": data.get("journal", ""),
                "doi": doi,
                "doi_url": f"https://doi.org/{doi}" if doi else None,
                "metadata_path": str(json_path),
                "expected_pdf_path": str(pdf_path),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        _json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(entries)


def _load_known_dois(path: Path) -> set[str]:
    """Load a newline-delimited file of DOIs to skip during ingestion.

    Args:
        path: Path to the text file.

    Returns:
        Set of stripped DOI strings.  Returns an empty set on read error.
    """
    try:
        return {
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    except OSError as exc:
        logger.warning("Could not read known-dois file '%s': %s", path, exc)
        return set()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the expansion pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Bibliographic Expansion Engine — traverse your citation ecosystem."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--author-id",
        help="Semantic Scholar author ID (integer string)",
    )
    parser.add_argument(
        "--gscholar-url",
        metavar="URL",
        help=(
            "Google Scholar profile URL or bare author ID "
            "(e.g. 'https://scholar.google.com/citations?user=AbCdEfGhIjK'). "
            "Uses Google Scholar for Tier 0 (more complete), "
            "Semantic Scholar for Tier 1/2."
        ),
    )
    parser.add_argument(
        "--scholar-proxy",
        metavar="PROXY",
        help=(
            "Proxy for Google Scholar requests. "
            "Use 'free' to rotate through free public proxies, "
            "or supply an explicit URL (e.g. 'http://host:port'). "
            "Needed when Google blocks direct scraping."
        ),
    )
    parser.add_argument(
        "--find-author",
        metavar="NAME",
        help="Search for your Semantic Scholar author ID by name and exit",
    )
    parser.add_argument(
        "--library-dir",
        default="./library",
        help="PDF/metadata library root directory (default: ./library)",
    )
    parser.add_argument(
        "--graph-path",
        default="./library/citation_graph.json",
        help="Path for saving the citation graph JSON",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help=(
            "Traversal depth: "
            "0 = author's papers only, "
            "1 = + direct refs/cites (default), "
            "2 = + one further hop"
        ),
    )
    parser.add_argument(
        "--api-key",
        help="Semantic Scholar API key (increases rate limit)",
    )
    parser.add_argument(
        "--known-dois",
        metavar="FILE",
        help="Newline-delimited file of DOIs already in ChromaDB to skip",
    )
    parser.add_argument(
        "--output-json",
        metavar="FILE",
        help="Write the expansion result summary to this JSON file",
    )
    parser.add_argument(
        "--unpaywall-email",
        metavar="EMAIL",
        help=(
            "Your email address for the Unpaywall API. "
            "When provided, papers with no open-access URL are looked up via "
            "Unpaywall (free, legal OA resolver) before being added to the "
            "missing-PDFs manifest."
        ),
    )
    parser.add_argument(
        "--missing-pdfs",
        metavar="FILE",
        default="./library/missing_pdfs.json",
        help=(
            "Write a JSON manifest of papers with no PDF to this path "
            "(default: ./library/missing_pdfs.json). "
            "Use with download_pdfs.py for Playwright-assisted downloading."
        ),
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help=(
            "After expansion, fetch full metadata (abstract, citation count) "
            "for any papers missing an abstract via DOI lookup. "
            "Automatically enabled when --depth 0 is used with --gscholar-url."
        ),
    )
    parser.add_argument(
        "--enrich-delay",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Seconds between enrichment API requests (default: 1.0)",
    )
    args = parser.parse_args()

    if args.find_author:
        _find_author(args.find_author, args.api_key)
        return

    if not args.author_id and not args.gscholar_url:
        parser.error(
            "Provide --author-id (Semantic Scholar) or "
            "--gscholar-url (Google Scholar profile URL)."
        )

    library_dir = Path(args.library_dir)
    graph_path = Path(args.graph_path)

    known_dois: set[str] = set()
    if args.known_dois:
        known_dois = _load_known_dois(Path(args.known_dois))
        logger.info("Loaded %d known DOIs from %s", len(known_dois), args.known_dois)

    graph = CitationGraph(persist_path=graph_path)
    if graph_path.exists():
        graph.load()
        logger.info("Loaded existing graph from %s", graph_path)

    ss_client = SemanticScholarClient(api_key=args.api_key or None)
    unpaywall = (
        UnpaywallClient(email=args.unpaywall_email) if args.unpaywall_email else None
    )
    pipeline = IngestionPipeline(library_dir=library_dir, unpaywall_client=unpaywall)

    explorer = CitationExplorer(
        library_dir=library_dir,
        pipeline=pipeline,
        graph=graph,
        ss_client=ss_client,
        max_depth=args.depth,
        known_dois=known_dois,
    )

    if args.gscholar_url:
        gs_client = GoogleScholarClient(proxy=getattr(args, "scholar_proxy", None))
        result = explorer.expand_from_google_scholar(
            args.gscholar_url, gscholar_client=gs_client
        )
        author_label = args.gscholar_url
    else:
        result = explorer.expand_from_author(args.author_id)
        author_label = args.author_id

    graph.save()

    divider = "=" * 52
    print(f"\n{divider}")
    print(f"  Expansion complete — {author_label}")
    print(divider)
    print(f"  Total records  : {len(result.records)}")
    print(f"  New ingestions : {result.new_ingestions}")
    print(f"  Skipped (dup)  : {result.skipped}")
    print(f"  Errors         : {len(result.errors)}")
    print(f"  Graph nodes    : {graph.graph.number_of_nodes()}")
    print(f"  Graph edges    : {graph.graph.number_of_edges()}")
    print(f"{divider}\n")

    # Enrichment pass (auto-enabled for GS depth-0 runs)
    run_enrich = args.enrich or (args.gscholar_url and args.depth == 0)
    if run_enrich:
        logger.info("Running metadata enrichment pass...")
        fetcher = MetadataFetcher()
        enriched, failed = enrich_library(library_dir, fetcher, delay=args.enrich_delay)
        print(f"  Enriched       : {enriched} papers")
        if failed:
            logger.warning("Enrichment failed for %d paper(s)", failed)

    # Write missing-PDFs manifest
    manifest_path = Path(args.missing_pdfs)
    missing_count = _write_missing_pdfs_manifest(library_dir, manifest_path)
    if missing_count:
        print(f"  Missing PDFs   : {missing_count} — manifest: {manifest_path}")
        print(f"  Run: python download_pdfs.py --manifest {manifest_path}\n")
    else:
        print("  Missing PDFs   : 0\n")

    if args.output_json:
        try:
            Path(args.output_json).write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            logger.info("Expansion result written to %s", args.output_json)
        except OSError as exc:
            logger.error("Failed to write output JSON: %s", exc)

    if result.errors:
        logger.warning("%d error(s) during expansion:", len(result.errors))
        for err in result.errors:
            logger.warning("  - %s", err)


if __name__ == "__main__":
    main()
