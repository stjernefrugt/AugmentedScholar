"""Streamlit dashboard — AugmentedScholar Network Analysis Engine.

Four tabs:
  * Citation Map   — interactive 3D citation network; click a node to open its DOI.
  * Papers         — reverse-chronological paper list with formatted citations.
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
import html as _html
import re
import sys
from pathlib import Path
from typing import Any

import networkx as nx
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


def _classify_nodes(graph: nx.DiGraph) -> dict[str, set[str]]:
    """Classify nodes into 'own', 'cited', and 'citing' sets.

    - **own**: tier-0 — the author's own papers.
    - **cited**: reachable from own via ``"cites"`` edges (the author's references).
    - **citing**: reachable from own via ``"is_cited_by"`` edges (papers that cite the
      author).
    """
    own: set[str] = {n for n, d in graph.nodes(data=True) if d.get("tier", 1) == 0}

    cited: set[str] = set()
    frontier = list(own)
    while frontier:
        node = frontier.pop()
        for _, nbr, data in graph.out_edges(node, data=True):
            if data.get("relationship") == "cites" and nbr not in cited | own:
                cited.add(nbr)
                frontier.append(nbr)

    citing: set[str] = set()
    frontier = list(own)
    while frontier:
        node = frontier.pop()
        for _, nbr, data in graph.out_edges(node, data=True):
            if data.get("relationship") == "is_cited_by" and nbr not in citing | own:
                citing.add(nbr)
                frontier.append(nbr)

    return {"own": own, "cited": cited, "citing": citing}


def _get_figure(
    visible_nodes: set[str] | None = None,
    node_categories: dict[str, str] | None = None,
) -> object:
    mtime = GRAPH_PATH.stat().st_mtime if GRAPH_PATH.exists() else 0.0
    all_positions = _get_positions(mtime)
    positions = (
        {n: p for n, p in all_positions.items() if n in visible_nodes}
        if visible_nodes is not None
        else all_positions
    )
    cg, _ = _load_graph_builder()
    from src.viz_engine import build_plotly_3d

    return build_plotly_3d(  # type: ignore[arg-type]
        cg.graph, positions=positions, node_categories=node_categories
    )


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

_CITATION_STYLES = ["APA", "MLA", "Chicago", "Vancouver", "BibTeX"]


def _author_last_first(name: str) -> str:
    """'Alice Smith' → 'Smith, A.'  (handles 'Smith, Alice' too)."""
    name = name.strip()
    if "," in name:
        last, *rest = name.split(",", 1)
        first_parts = rest[0].strip().split() if rest else []
    else:
        parts = name.split()
        if not parts:
            return name
        last = parts[-1]
        first_parts = parts[:-1]
    initials = " ".join(p[0] + "." for p in first_parts if p)
    return f"{last.strip()}, {initials}" if initials else last.strip()


def _author_display(name: str) -> str:
    """Return name with normalised whitespace."""
    return " ".join(name.strip().split())


def _bib_key(doi: str, authors: list[str], year: int | str | None) -> str:
    first = (
        authors[0].strip().split(",")[0].split()[-1].lower() if authors else "unknown"
    )
    return re.sub(r"[^a-z0-9]", "", first) + str(year or "nd")


def _format_citation(doi: str, attrs: dict[str, Any], style: str) -> str:
    """Return a formatted citation string for *style*."""
    title: str = attrs.get("title") or doi
    authors: list[str] = attrs.get("authors") or []
    year = attrs.get("year")
    journal: str = attrs.get("journal") or ""
    url = f"https://doi.org/{doi}" if doi else ""

    if style == "APA":
        if authors:
            fmt = [_author_last_first(a) for a in authors]
            if len(fmt) <= 7:
                author_str = (
                    fmt[0] if len(fmt) == 1 else ", ".join(fmt[:-1]) + ", & " + fmt[-1]
                )
            else:
                author_str = ", ".join(fmt[:6]) + ", ... " + fmt[-1]
        else:
            author_str = ""
        year_str = f"({year})." if year else "(n.d.)."
        journal_part = f" *{journal}*." if journal else ""
        doi_part = f" {url}" if url else ""
        lead = " ".join(p for p in [author_str, year_str] if p)
        return f"{lead} {title}.{journal_part}{doi_part}"

    if style == "MLA":
        if authors:
            p = authors[0].strip().split()
            first_fmt = (
                f"{p[-1]}, {' '.join(p[:-1])}" if len(p) >= 2 else authors[0].strip()
            )
            author_str = first_fmt + ", et al." if len(authors) > 1 else first_fmt + "."
        else:
            author_str = ""
        journal_part = f" *{journal}*," if journal else ""
        year_part = f" {year}," if year else ""
        doi_part = f" {url}." if url else "."
        return f'{author_str} "{title}."{journal_part}{year_part}{doi_part}'

    if style == "Chicago":
        if authors:
            p = authors[0].strip().split()
            first_fmt = (
                f"{p[-1]}, {' '.join(p[:-1])}" if len(p) >= 2 else authors[0].strip()
            )
            rest = [_author_display(a) for a in authors[1:]]
            author_str = (
                first_fmt + ", and " + ", ".join(rest) + "."
                if rest
                else first_fmt + "."
            )
        else:
            author_str = ""
        journal_part = f" *{journal}*" if journal else ""
        year_part = f" ({year})" if year else ""
        doi_part = f". {url}" if url else ""
        return f'{author_str} "{title}."{journal_part}{year_part}{doi_part}.'

    if style == "Vancouver":
        if authors:
            fmt = [_author_last_first(a) for a in authors[:6]]
            author_str = ", ".join(fmt) + (", et al" if len(authors) > 6 else "")
        else:
            author_str = ""
        journal_part = f" {journal}." if journal else ""
        year_part = f" {year};" if year else ""
        doi_part = f" doi:{doi}" if doi else ""
        return f"{author_str}.{journal_part}{year_part}{doi_part}"

    if style == "BibTeX":
        key = _bib_key(doi, authors, year)
        bib_authors = " and ".join(
            (
                f"{a.strip().split()[-1]}, {' '.join(a.strip().split()[:-1])}"
                if len(a.strip().split()) > 1
                else a.strip()
            )
            for a in authors
        )
        lines = [f"@article{{{key},", f"  title     = {{{title}}},"]
        if bib_authors:
            lines.append(f"  author    = {{{bib_authors}}},")
        if journal:
            lines.append(f"  journal   = {{{journal}}},")
        if year:
            lines.append(f"  year      = {{{year}}},")
        if doi:
            lines.append(f"  doi       = {{{doi}}},")
        lines.append("}")
        return "\n".join(lines)

    return f"{title} ({year}). {url}"


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


def _render_paper_list(
    graph: nx.DiGraph,
    list_nodes: set[str],
    selected_doi: str,
) -> None:
    """Scrollable left-panel paper list; highlights the selected node."""
    if not list_nodes:
        st.caption("No papers — select a category above.")
        return

    rows = sorted(
        [(doi, graph.nodes[doi]) for doi in list_nodes],
        key=lambda x: x[1].get("year") or 0,
        reverse=True,
    )
    st.caption(f"{len(rows)} paper(s)")

    with st.container(height=620, border=False):
        for doi, attrs in rows:
            raw_title = attrs.get("title") or doi
            title = _html.escape(raw_title[:68] + ("…" if len(raw_title) > 68 else ""))
            year = attrs.get("year", "")
            authors: list[str] = attrs.get("authors") or []
            first = _html.escape(authors[0].split()[-1] if authors else "")
            meta = f"{first}{' · ' if first else ''}{year}"

            if doi == selected_doi:
                st.markdown(
                    f'<div style="background:rgba(91,155,213,0.18);'
                    f"border-left:3px solid #5b9bd5;"
                    f'padding:6px 8px;border-radius:3px;margin:1px 0">'
                    f'<span style="font-size:12px;font-weight:600">{title}</span><br>'
                    f'<span style="font-size:11px;color:#aaa">{meta}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="border-left:3px solid transparent;'
                    f'padding:5px 8px;margin:1px 0">'
                    f'<span style="font-size:12px">{title}</span><br>'
                    f'<span style="font-size:11px;color:#aaa">{meta}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _tab_citation_map() -> None:
    st.subheader("3D Citation Network")
    cg, _ = _load_graph_builder()
    graph: nx.DiGraph = cg.graph  # type: ignore[assignment]
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        st.warning("Graph is empty — run `run_expansion.py` first.")
        return

    # ---- Category toggles (full-width row) --------------------------------
    classification = _classify_nodes(graph)
    n_own = len(classification["own"])
    n_cited = len(classification["cited"])
    n_citing = len(classification["citing"])

    c1, c2, c3, _ = st.columns([2, 2, 2, 4])
    show_own = c1.checkbox(f"Own ({n_own})", value=True, key="show_own")
    show_cited = c2.checkbox(f"Cited ({n_cited})", value=True, key="show_cited")
    show_citing = c3.checkbox(f"Citing ({n_citing})", value=True, key="show_citing")

    # Papers shown in left panel (strictly from selected categories)
    list_nodes: set[str] = set()
    if show_own:
        list_nodes |= classification["own"]
    if show_cited:
        list_nodes |= classification["cited"]
    if show_citing:
        list_nodes |= classification["citing"]

    # 3D plot also renders uncategorised nodes that don't belong to any set
    all_nodes: set[str] = set(graph.nodes())
    uncategorised = all_nodes - (
        classification["own"] | classification["cited"] | classification["citing"]
    )
    visible = list_nodes | uncategorised

    # Current highlighted DOI (set by previous click interaction)
    selected_doi: str = st.session_state.get("_selected_doi", "")

    # ---- Two-column layout: paper list | 3D plot --------------------------
    left, right = st.columns([1, 3], gap="small")

    with left:
        _render_paper_list(graph, list_nodes, selected_doi)

    with right:
        st.caption(f"{len(visible)}/{n_nodes} papers · {n_edges} citation edges")

        node_cat: dict[str, str] = {}
        for cat_name in ("citing", "cited", "own"):
            for node in classification[cat_name]:
                node_cat[node] = cat_name

        filter_arg = visible if visible != all_nodes else None
        fig = _get_figure(filter_arg, node_categories=node_cat)
        event = st.plotly_chart(
            fig,
            width="stretch",
            on_select="rerun",
            key="citation_map",
        )

        points = getattr(getattr(event, "selection", None), "points", [])
        if points:
            doi = str(points[0].get("customdata", ""))
            if doi:
                # Sync left-panel highlight — rerun so the list sees the new value
                if doi != selected_doi:
                    st.session_state["_selected_doi"] = doi
                    st.rerun()
                # Open DOI in a new browser tab (once per selection)
                if doi != st.session_state.get("_last_doi_opened"):
                    st.session_state["_last_doi_opened"] = doi
                    import streamlit.components.v1 as components

                    components.html(
                        f"<script>"
                        f'window.open("https://doi.org/{doi}", "_blank");'
                        f"</script>",
                        height=0,
                    )
                _paper_card(doi)


def _tab_papers() -> None:
    st.subheader("Papers")
    cg, _ = _load_graph_builder()
    graph: nx.DiGraph = cg.graph  # type: ignore[assignment]

    if graph.number_of_nodes() == 0:
        st.info("No papers yet — run `run_expansion.py` first.")
        return

    classification = _classify_nodes(graph)

    c1, c2, c3, _, c5 = st.columns([1, 1, 1, 1, 3])
    show_own = c1.checkbox("Own", value=True, key="papers_own")
    show_cited = c2.checkbox("Cited", value=False, key="papers_cited")
    show_citing = c3.checkbox("Citing", value=False, key="papers_citing")
    style: str = c5.selectbox(  # type: ignore[assignment]
        "Citation style", _CITATION_STYLES, key="papers_style"
    )

    visible: set[str] = set()
    if show_own:
        visible |= classification["own"]
    if show_cited:
        visible |= classification["cited"]
    if show_citing:
        visible |= classification["citing"]

    rows = sorted(
        [(doi, graph.nodes[doi]) for doi in visible],
        key=lambda x: x[1].get("year") or 0,
        reverse=True,
    )

    st.caption(f"{len(rows)} paper(s)")

    if not rows:
        st.info("No papers selected — tick at least one category above.")
        return

    if style == "BibTeX":
        all_bib = "\n\n".join(
            _format_citation(doi, attrs, "BibTeX") for doi, attrs in rows
        )
        st.code(all_bib, language="bibtex")
    else:
        for doi, attrs in rows:
            st.markdown(_format_citation(doi, attrs, style))
            st.divider()


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

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Citation Map", "Papers", "Gap Analysis", "Analytics"]
    )
    with tab1:
        _tab_citation_map()
    with tab2:
        _tab_papers()
    with tab3:
        _tab_gap_analysis()
    with tab4:
        _tab_analytics()


if __name__ == "__main__" or "streamlit" in sys.modules:
    main()
