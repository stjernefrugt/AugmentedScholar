"""Tests for src/graph_builder.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import networkx as nx
import pytest

from src.graph_builder import GapReport, GraphBuilder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_graph(*dois: str) -> MagicMock:
    """Return a fake CitationGraph whose .graph is a DiGraph with *dois* nodes."""
    g = nx.DiGraph()
    for doi in dois:
        g.add_node(doi, title=f"Paper {doi}", year=2020, citation_count=10)
    mock_cg = MagicMock()
    mock_cg.graph = g
    return mock_cg


# ---------------------------------------------------------------------------
# GapReport
# ---------------------------------------------------------------------------


class TestGapReport:
    def test_total_gaps_sums_all_fields(self) -> None:
        report = GapReport(
            isolated_nodes=["a", "b"],
            high_centrality_no_pdf=[{"doi": "c"}],
            no_abstract=["d", "e", "f"],
        )
        assert report.total_gaps == 6

    def test_total_gaps_empty(self) -> None:
        assert GapReport().total_gaps == 0


# ---------------------------------------------------------------------------
# GraphBuilder.compute_centrality
# ---------------------------------------------------------------------------


class TestComputeCentrality:
    def test_empty_graph_returns_empty_dict(self) -> None:
        gb = GraphBuilder(_make_graph())
        assert gb.compute_centrality() == {}

    def test_single_node_returns_zero_betweenness(self) -> None:
        gb = GraphBuilder(_make_graph("doi:1"))
        scores = gb.compute_centrality()
        assert "doi:1" in scores
        assert scores["doi:1"]["betweenness"] == pytest.approx(0.0)
        # Single node normalises to eigenvector = 1.0
        assert scores["doi:1"]["eigenvector"] == pytest.approx(1.0)

    def test_all_nodes_present_in_scores(self) -> None:
        cg = _make_graph("a", "b", "c")
        cg.graph.add_edge("a", "b")
        cg.graph.add_edge("b", "c")
        gb = GraphBuilder(cg)
        scores = gb.compute_centrality()
        assert set(scores.keys()) == {"a", "b", "c"}
        for v in scores.values():
            assert "betweenness" in v
            assert "eigenvector" in v

    def test_bridge_node_has_highest_betweenness(self) -> None:
        # a → b → c: b is the bridge
        cg = _make_graph("a", "b", "c")
        cg.graph.add_edge("a", "b")
        cg.graph.add_edge("b", "c")
        gb = GraphBuilder(cg)
        scores = gb.compute_centrality()
        assert scores["b"]["betweenness"] >= scores["a"]["betweenness"]
        assert scores["b"]["betweenness"] >= scores["c"]["betweenness"]


# ---------------------------------------------------------------------------
# GraphBuilder.apply_centrality
# ---------------------------------------------------------------------------


class TestApplyCentrality:
    def test_scores_written_to_node_attributes(self) -> None:
        cg = _make_graph("doi:1", "doi:2")
        cg.graph.add_edge("doi:1", "doi:2")
        gb = GraphBuilder(cg)
        gb.apply_centrality()
        for node in cg.graph.nodes():
            assert "betweenness" in cg.graph.nodes[node]
            assert "eigenvector" in cg.graph.nodes[node]

    def test_returns_same_as_compute_centrality(self) -> None:
        cg = _make_graph("x", "y")
        cg.graph.add_edge("x", "y")
        gb = GraphBuilder(cg)
        returned = gb.apply_centrality()
        direct = gb.compute_centrality()
        assert returned.keys() == direct.keys()


# ---------------------------------------------------------------------------
# GraphBuilder.gap_analysis
# ---------------------------------------------------------------------------


class TestGapAnalysis:
    def test_isolated_nodes_detected(self) -> None:
        cg = _make_graph("iso", "connected_a", "connected_b")
        cg.graph.add_edge("connected_a", "connected_b")
        gb = GraphBuilder(cg)
        report = gb.gap_analysis()
        assert "iso" in report.isolated_nodes
        assert "connected_a" not in report.isolated_nodes

    def test_no_library_dir_skips_pdf_and_abstract(self) -> None:
        cg = _make_graph("a", "b")
        gb = GraphBuilder(cg, library_dir=None)
        report = gb.gap_analysis()
        assert report.high_centrality_no_pdf == []
        assert report.no_abstract == []

    def test_missing_abstract_detected(self, tmp_path: Path) -> None:
        cg = _make_graph("10.1234/test")
        gb = GraphBuilder(cg)
        gb.apply_centrality()

        sidecar = {
            "title": "A Test Paper",
            "doi": "10.1234/test",
            "year": 2020,
            "journal": "Nature",
            # No "abstract" key
        }
        (tmp_path / "test.json").write_text(json.dumps(sidecar), encoding="utf-8")

        gb2 = GraphBuilder(cg, library_dir=tmp_path)
        report = gb2.gap_analysis()
        assert "10.1234/test" in report.no_abstract

    def test_paper_with_abstract_not_in_no_abstract(self, tmp_path: Path) -> None:
        cg = _make_graph("10.9999/abs")
        gb = GraphBuilder(cg)
        gb.apply_centrality()

        sidecar = {
            "title": "A Paper With Abstract",
            "doi": "10.9999/abs",
            "year": 2021,
            "journal": "Science",
            "abstract": "This paper has an abstract.",
        }
        (tmp_path / "abs.json").write_text(json.dumps(sidecar), encoding="utf-8")

        gb2 = GraphBuilder(cg, library_dir=tmp_path)
        report = gb2.gap_analysis()
        assert "10.9999/abs" not in report.no_abstract

    def test_high_centrality_no_pdf(self, tmp_path: Path) -> None:
        # Build a graph where one node has high betweenness (bridge)
        cg = _make_graph("a", "bridge", "c")
        cg.graph.add_edge("a", "bridge")
        cg.graph.add_edge("bridge", "c")
        gb = GraphBuilder(cg)
        gb.apply_centrality()

        sidecar = {
            "title": "Bridge Paper",
            "doi": "bridge",
            "year": 2020,
            "journal": "Nature",
        }
        (tmp_path / "bridge.json").write_text(json.dumps(sidecar), encoding="utf-8")
        # No bridge.pdf → should appear in high_centrality_no_pdf

        gb2 = GraphBuilder(cg, library_dir=tmp_path)
        report = gb2.gap_analysis()
        dois = [e["doi"] for e in report.high_centrality_no_pdf]
        assert "bridge" in dois

    def test_pdf_present_excluded_from_high_centrality(self, tmp_path: Path) -> None:
        cg = _make_graph("a", "bridge", "c")
        cg.graph.add_edge("a", "bridge")
        cg.graph.add_edge("bridge", "c")
        gb = GraphBuilder(cg)
        gb.apply_centrality()

        sidecar = {
            "title": "Bridge Paper",
            "doi": "bridge",
            "year": 2020,
            "journal": "Nature",
        }
        (tmp_path / "bridge.json").write_text(json.dumps(sidecar), encoding="utf-8")
        (tmp_path / "bridge.pdf").write_bytes(b"%PDF")  # PDF exists

        gb2 = GraphBuilder(cg, library_dir=tmp_path)
        report = gb2.gap_analysis()
        dois = [e["doi"] for e in report.high_centrality_no_pdf]
        assert "bridge" not in dois
