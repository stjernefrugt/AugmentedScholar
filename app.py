"""Streamlit dashboard — AugmentedScholar Network Analysis Engine.

Three tabs:
  * Citation Map   — interactive 3D citation network; click a node for details.
  * Gap Analysis   — isolated nodes, missing PDFs, missing abstracts.
  * Analytics      — centrality leaderboard table.

Launch::

    streamlit run app.py -- --library-dir ./library \\
        --graph-path ./library/citation_graph.json
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CLI args parsed before Streamlit intercepts sys.argv
# ---------------------------------------------------------------------------
import argparse as _argparse
import sys
from pathlib import Path

import streamlit as st

_parser = _argparse.ArgumentParser(add_help=False)
_parser.add_argument("--library-dir", default="./library")
_parser.add_argument("--graph-path", default="./library/citation_graph.json")
_args, _ = _parser.parse_known_args()

LIBRARY_DIR = Path(_args.library_dir)
GRAPH_PATH = Path(_args.graph_path)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_graph_builder() -> tuple[object, object]:  # type: ignore[type-arg]
    """Load CitationGraph + GraphBuilder (cached for the session lifetime)."""
    from src.graph import CitationGraph
    from src.graph_builder import GraphBuilder

    cg = CitationGraph(persist_path=GRAPH_PATH)
    if GRAPH_PATH.exists():
        cg.load()
    gb = GraphBuilder(cg, library_dir=LIBRARY_DIR if LIBRARY_DIR.is_dir() else None)
    gb.apply_centrality()
    return cg, gb


@st.cache_data(ttl=300)
def _get_positions(mtime: float) -> dict[str, tuple[float, float, float]]:
    """Compute 3D layout positions; invalidated when graph file is modified."""
    _ = mtime  # cache key only
    cg, _ = _load_graph_builder()
    from src.viz_engine import compute_layout_3d

    return compute_layout_3d(cg.graph)  # type: ignore[arg-type]


def _get_figure() -> object:
    mtime = GRAPH_PATH.stat().st_mtime if GRAPH_PATH.exists() else 0.0
    positions = _get_positions(mtime)
    cg, _ = _load_graph_builder()
    from src.viz_engine import build_plotly_3d

    return build_plotly_3d(cg.graph, positions=positions)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Paper card
# ---------------------------------------------------------------------------


def _paper_card(doi: str) -> None:
    """Render a metadata summary card for *doi* in a Streamlit expander."""
    from src.viz_engine import load_paper_metadata

    data = load_paper_metadata(doi, LIBRARY_DIR)
    if data is None:
        st.info(f"No sidecar found for {doi}")
        return

    with st.expander(f"**{data.get('title', doi)}**", expanded=True):
        cols = st.columns([2, 1])
        with cols[0]:
            authors = data.get("authors", [])
            if authors:
                st.caption(", ".join(str(a) for a in authors[:5]))
        with cols[1]:
            st.caption(
                f"{data.get('journal', '')}  {data.get('year', '')}  "
                f"·  {data.get('citation_count', 0)} cit."
            )
        abstract = data.get("abstract", "")
        if abstract:
            st.write(abstract[:600] + ("…" if len(abstract) > 600 else ""))
        keywords = data.get("keywords", [])
        if keywords:
            st.caption("Keywords: " + ", ".join(str(k) for k in keywords[:10]))
        if doi:
            st.markdown(f"[Open DOI](https://doi.org/{doi})")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _tab_citation_map() -> None:
    st.subheader("3D Citation Network")
    cg, _ = _load_graph_builder()
    n_nodes = cg.graph.number_of_nodes()  # type: ignore[union-attr]
    n_edges = cg.graph.number_of_edges()  # type: ignore[union-attr]
    st.caption(f"{n_nodes} papers · {n_edges} citation edges")

    if n_nodes == 0:
        st.warning("Graph is empty — run `run_expansion.py` first.")
        return

    fig = _get_figure()
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="citation_map",
    )

    points = getattr(getattr(event, "selection", None), "points", [])
    if points:
        doi = points[0].get("customdata", "")
        if doi:
            _paper_card(str(doi))


def _tab_gap_analysis() -> None:
    st.subheader("Gap Analysis")
    _, gb = _load_graph_builder()

    with st.spinner("Analysing gaps…"):
        report = gb.gap_analysis()  # type: ignore[union-attr]

    st.metric("Total items needing attention", report.total_gaps)
    col1, col2, col3 = st.columns(3)
    col1.metric("Isolated nodes", len(report.isolated_nodes))
    col2.metric("High-centrality, no PDF", len(report.high_centrality_no_pdf))
    col3.metric("Missing abstracts", len(report.no_abstract))

    if report.high_centrality_no_pdf:
        st.markdown("#### High-Centrality Papers Without PDF")
        import pandas as pd

        df = pd.DataFrame(report.high_centrality_no_pdf)
        st.dataframe(df, use_container_width=True, hide_index=True)

    if report.isolated_nodes:
        st.markdown("#### Isolated Nodes (no citation edges)")
        for doi in report.isolated_nodes[:50]:
            st.code(doi, language=None)

    if report.no_abstract:
        st.markdown("#### Papers Missing Abstract")
        for doi in report.no_abstract[:50]:
            st.code(doi, language=None)


def _tab_analytics() -> None:
    st.subheader("Centrality Leaderboard")
    cg, _ = _load_graph_builder()
    import pandas as pd

    rows = []
    for node, attrs in cg.graph.nodes(data=True):  # type: ignore[union-attr]
        rows.append(
            {
                "doi": node,
                "title": attrs.get("title", node)[:80],
                "year": attrs.get("year"),
                "betweenness": round(attrs.get("betweenness", 0.0), 5),
                "eigenvector": round(attrs.get("eigenvector", 0.0), 5),
                "citations": attrs.get("citation_count", 0),
            }
        )
    if not rows:
        st.info("No nodes in graph yet.")
        return

    df = pd.DataFrame(rows).sort_values("betweenness", ascending=False)
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="AugmentedScholar",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("AugmentedScholar — Network Analysis")

    tab1, tab2, tab3 = st.tabs(["Citation Map", "Gap Analysis", "Analytics"])
    with tab1:
        _tab_citation_map()
    with tab2:
        _tab_gap_analysis()
    with tab3:
        _tab_analytics()


if __name__ == "__main__" or "streamlit" in sys.modules:
    main()
