"""Analytics layer over CitationGraph: centrality metrics and gap analysis.

Use :class:`GraphBuilder` to enrich an existing :class:`~src.graph.CitationGraph`
with betweenness / eigenvector centrality scores and to surface structural
gaps (missing PDFs, missing abstracts, isolated nodes).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from .graph import CitationGraph

logger = logging.getLogger(__name__)


@dataclass
class GapReport:
    """Summary of structural gaps in the citation network.

    Attributes:
        isolated_nodes: Node keys with no citation edges whatsoever.
        high_centrality_no_pdf: Papers above the median betweenness
            centrality whose full-text PDF is not locally available.
            Sorted descending by betweenness score.
        no_abstract: DOIs of papers whose JSON sidecar has no abstract.
    """

    isolated_nodes: list[str] = field(default_factory=list)
    high_centrality_no_pdf: list[dict[str, Any]] = field(default_factory=list)
    no_abstract: list[str] = field(default_factory=list)

    @property
    def total_gaps(self) -> int:
        """Total count of items needing attention across all categories."""
        return (
            len(self.isolated_nodes)
            + len(self.high_centrality_no_pdf)
            + len(self.no_abstract)
        )


class GraphBuilder:
    """Adds centrality metrics and gap analysis to a :class:`~src.graph.CitationGraph`.

    Args:
        citation_graph: Loaded :class:`~src.graph.CitationGraph` instance.
        library_dir: Library root directory.  Required for PDF / abstract
            gap checks; gap analysis is skipped without it.

    Examples:
        >>> from src.graph import CitationGraph
        >>> cg = CitationGraph(persist_path=Path("library/citation_graph.json"))
        >>> cg.load()
        >>> gb = GraphBuilder(cg, library_dir=Path("library"))
        >>> scores = gb.apply_centrality()
        >>> report = gb.gap_analysis()
    """

    def __init__(
        self,
        citation_graph: CitationGraph,
        library_dir: Path | None = None,
    ) -> None:
        self._cg = citation_graph
        self.graph: nx.DiGraph = citation_graph.graph
        self.library_dir = library_dir

    # ------------------------------------------------------------------
    # Centrality
    # ------------------------------------------------------------------

    def compute_centrality(self) -> dict[str, dict[str, float]]:
        """Compute betweenness and eigenvector centrality for all nodes.

        Betweenness uses a k=min(n,200) approximation for performance on
        large graphs.  Eigenvector centrality runs on the undirected view
        (more stable) and falls back to degree centrality if power iteration
        does not converge.

        Returns:
            Mapping of ``node_id → {"betweenness": float, "eigenvector": float}``.
        """
        n = self.graph.number_of_nodes()
        if n == 0:
            return {}

        k = min(n, 200)
        betweenness: dict[str, float] = nx.betweenness_centrality(
            self.graph, normalized=True, k=k, seed=42
        )

        undirected = self.graph.to_undirected()
        try:
            eigenvector: dict[str, float] = nx.eigenvector_centrality(
                undirected, max_iter=1000, tol=1e-6
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "Eigenvector centrality failed to converge — using degree centrality"
            )
            eigenvector = nx.degree_centrality(undirected)

        return {
            node: {
                "betweenness": betweenness.get(node, 0.0),
                "eigenvector": eigenvector.get(node, 0.0),
            }
            for node in self.graph.nodes()
        }

    def apply_centrality(self) -> dict[str, dict[str, float]]:
        """Compute centrality and write scores back as node attributes.

        This mutates the underlying :attr:`graph` so the scores survive a
        subsequent :meth:`~src.graph.CitationGraph.save` call.

        Returns:
            Same mapping as :meth:`compute_centrality`.
        """
        scores = self.compute_centrality()
        for node, metrics in scores.items():
            self.graph.nodes[node]["betweenness"] = metrics["betweenness"]
            self.graph.nodes[node]["eigenvector"] = metrics["eigenvector"]
        logger.info("Centrality applied to %d nodes", len(scores))
        return scores

    # ------------------------------------------------------------------
    # Gap analysis
    # ------------------------------------------------------------------

    def gap_analysis(self) -> GapReport:
        """Identify papers that need attention in the citation network.

        Checks for:

        * **Isolated nodes** — papers with no citation edges at all.
        * **High-centrality, no PDF** — bridge papers (above-median
          betweenness) whose full text is not locally available.
        * **No abstract** — papers whose JSON sidecar lacks an abstract.

        Returns:
            Populated :class:`GapReport`.
        """
        isolated = [n for n, d in self.graph.degree() if d == 0]

        betweenness: dict[str, float] = {
            n: self.graph.nodes[n].get("betweenness", 0.0) for n in self.graph.nodes()
        }
        values = sorted(betweenness.values())
        median_bc = values[len(values) // 2] if values else 0.0

        high_central_no_pdf: list[dict[str, Any]] = []
        no_abstract: list[str] = []

        if self.library_dir:
            for json_path in sorted(self.library_dir.glob("*.json")):
                try:
                    data: dict[str, Any] = json.loads(
                        json_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue

                if "title" not in data:  # skip graph JSON, manifest files
                    continue

                doi: str | None = data.get("doi")
                if not doi:
                    continue

                if not data.get("abstract"):
                    no_abstract.append(doi)

                bc = betweenness.get(doi, 0.0)
                pdf_missing = not json_path.with_suffix(".pdf").exists()
                if bc >= median_bc and bc > 0 and pdf_missing:
                    high_central_no_pdf.append(
                        {
                            "doi": doi,
                            "title": data.get("title", "?"),
                            "year": data.get("year"),
                            "journal": data.get("journal", "?"),
                            "betweenness": round(bc, 4),
                            "doi_url": f"https://doi.org/{doi}",
                        }
                    )

        high_central_no_pdf.sort(key=lambda x: x["betweenness"], reverse=True)
        return GapReport(
            isolated_nodes=isolated,
            high_centrality_no_pdf=high_central_no_pdf,
            no_abstract=no_abstract,
        )
