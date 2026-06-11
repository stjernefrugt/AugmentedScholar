"""Tests for src/viz_engine.py."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from src.viz_engine import (
    build_plotly_3d,
    compute_layout_3d,
    load_paper_metadata,
    temporal_colormap,
)

# ---------------------------------------------------------------------------
# temporal_colormap
# ---------------------------------------------------------------------------


class TestTemporalColormap:
    def test_empty_returns_empty(self) -> None:
        assert temporal_colormap([]) == []

    def test_length_matches_input(self) -> None:
        years = [2000, 2010, 2020]
        colours = temporal_colormap(years)
        assert len(colours) == len(years)

    def test_all_hex_strings(self) -> None:
        for colour in temporal_colormap([1990, 2000, 2015, 2024]):
            assert colour.startswith("#")
            assert len(colour) == 7

    def test_oldest_year_is_blue(self) -> None:
        # Single colour with one year → t=0.5 by definition; test with span
        colours = temporal_colormap([2000, 2024])
        # Oldest (2000) should be bluer (low red component)
        old_r = int(colours[0][1:3], 16)
        new_r = int(colours[1][1:3], 16)
        assert old_r < new_r

    def test_newest_year_is_redder(self) -> None:
        colours = temporal_colormap([2000, 2024])
        new_b = int(colours[1][5:7], 16)
        old_b = int(colours[0][5:7], 16)
        # Newest should have lower blue component
        assert new_b < old_b

    def test_single_year_returns_midpoint_colour(self) -> None:
        # When min == max, t=0.5 → mid-grey-purple mix
        colours = temporal_colormap([2015])
        assert len(colours) == 1
        # r and b should be equal at t=0.5 (110 each)
        r = int(colours[0][1:3], 16)
        b = int(colours[0][5:7], 16)
        assert r == b


# ---------------------------------------------------------------------------
# compute_layout_3d
# ---------------------------------------------------------------------------


class TestComputeLayout3d:
    def _make_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(["a", "b", "c"])
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        return g

    def test_empty_graph_returns_empty(self) -> None:
        assert compute_layout_3d(nx.DiGraph()) == {}

    def test_all_nodes_have_positions(self) -> None:
        g = self._make_graph()
        pos = compute_layout_3d(g)
        assert set(pos.keys()) == {"a", "b", "c"}

    def test_positions_are_3d_tuples(self) -> None:
        pos = compute_layout_3d(self._make_graph())
        for coords in pos.values():
            assert len(coords) == 3
            assert all(isinstance(v, float) for v in coords)

    def test_max_nodes_cap_respected(self) -> None:
        g = nx.DiGraph()
        for i in range(10):
            g.add_node(str(i))
        pos = compute_layout_3d(g, max_nodes=5)
        assert len(pos) <= 5

    def test_deterministic_with_seed(self) -> None:
        g = self._make_graph()
        pos1 = compute_layout_3d(g, seed=0)
        pos2 = compute_layout_3d(g, seed=0)
        for node in pos1:
            assert pos1[node] == pytest.approx(pos2[node])


# ---------------------------------------------------------------------------
# build_plotly_3d
# ---------------------------------------------------------------------------


class TestBuildPlotly3d:
    def _make_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for doi, year, cit in [("10.1/a", 2020, 5), ("10.1/b", 2015, 100)]:
            g.add_node(
                doi,
                title="T",
                authors=["Alice Smith"],
                journal="J",
                year=year,
                citation_count=cit,
            )
        g.add_edge("10.1/a", "10.1/b")
        return g

    def test_returns_figure(self) -> None:
        import plotly.graph_objects as go

        fig = build_plotly_3d(self._make_graph())
        assert isinstance(fig, go.Figure)

    def test_edge_trace_plus_one_node_trace_without_categories(self) -> None:
        # Without node_categories → single "other" node trace + edge trace
        fig = build_plotly_3d(self._make_graph())
        assert len(fig.data) == 2

    def test_node_trace_customdata_contains_dois(self) -> None:
        g = self._make_graph()
        fig = build_plotly_3d(g)
        # Single node trace is at index 1 when no categories provided
        node_trace = fig.data[1]
        assert set(node_trace.customdata) == {"10.1/a", "10.1/b"}

    def test_accepts_precomputed_positions(self) -> None:
        g = self._make_graph()
        pos = {"10.1/a": (0.0, 0.0, 0.0), "10.1/b": (1.0, 1.0, 1.0)}
        fig = build_plotly_3d(g, positions=pos)
        assert fig is not None

    def test_empty_graph_returns_figure(self) -> None:
        import plotly.graph_objects as go

        fig = build_plotly_3d(nx.DiGraph())
        assert isinstance(fig, go.Figure)

    def test_category_traces_split_by_type(self) -> None:
        g = self._make_graph()
        cats = {"10.1/a": "own", "10.1/b": "cited"}
        fig = build_plotly_3d(g, node_categories=cats)
        # Edge trace + at least 2 node traces (own, cited)
        assert len(fig.data) >= 3
        # All DOIs present across node traces
        all_custom: list[str] = []
        for trace in fig.data[1:]:
            all_custom.extend(list(trace.customdata or []))
        assert set(all_custom) == {"10.1/a", "10.1/b"}

    def test_own_trace_uses_diamond_symbol(self) -> None:
        g = self._make_graph()
        cats = {"10.1/a": "own", "10.1/b": "cited"}
        fig = build_plotly_3d(g, node_categories=cats)
        own_trace = next(t for t in fig.data if getattr(t, "name", "") == "Own papers")
        assert own_trace.marker.symbol == "diamond"

    def test_colorbar_on_first_node_trace(self) -> None:
        fig = build_plotly_3d(self._make_graph())
        # First node trace (index 1) should have showscale=True
        assert fig.data[1].marker.showscale is True

    def test_numeric_color_values_used_not_hex(self) -> None:
        fig = build_plotly_3d(self._make_graph())
        colors = fig.data[1].marker.color
        # Colors must be numeric years, not hex strings
        assert all(isinstance(c, int) for c in colors)


# ---------------------------------------------------------------------------
# load_paper_metadata
# ---------------------------------------------------------------------------


class TestLoadPaperMetadata:
    def test_returns_matching_sidecar(self, tmp_path: Path) -> None:
        data = {"title": "Test Paper", "doi": "10.1234/test", "year": 2021}
        (tmp_path / "test_paper.json").write_text(json.dumps(data), encoding="utf-8")
        result = load_paper_metadata("10.1234/test", tmp_path)
        assert result is not None
        assert result["title"] == "Test Paper"

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        result = load_paper_metadata("10.9999/missing", tmp_path)
        assert result is None

    def test_skips_corrupt_json(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("{not valid json", encoding="utf-8")
        result = load_paper_metadata("10.1234/any", tmp_path)
        assert result is None

    def test_returns_first_match(self, tmp_path: Path) -> None:
        data = {"title": "Paper A", "doi": "10.5/same"}
        (tmp_path / "a.json").write_text(json.dumps(data), encoding="utf-8")
        result = load_paper_metadata("10.5/same", tmp_path)
        assert result is not None
        assert result["title"] == "Paper A"
