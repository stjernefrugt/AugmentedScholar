"""Streamlit dashboard — AugmentedScholar Network Analysis Engine.

Six tabs:
  * Citation Map   — interactive 3D citation network; click a node to open its DOI.
  * Papers         — reverse-chronological paper list with formatted citations.
  * Gap Analysis   — isolated nodes, missing PDFs, missing abstracts.
  * Analytics      — centrality leaderboard table with clickable DOIs.
  * Missing Papers — papers without PDFs, sorted by citation count.
  * Semantic Map   — t-SNE clustering of paper abstracts.

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
import json
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


@st.cache_data(ttl=300)
def _load_sidecar_texts(mtime: float) -> dict[str, str]:
    """Single-pass load of abstract+keywords text from all JSON sidecars.

    Returns {doi: text}.  Much faster than load_paper_metadata per DOI
    (O(n) total vs O(n²)).  Skips list-valued JSON files (missing_pdfs.json).
    """
    _ = mtime  # cache key only
    result: dict[str, str] = {}
    for json_path in LIBRARY_DIR.glob("*.json"):
        try:
            data: Any = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or "title" not in data:
            continue
        doi = data.get("doi")
        if not doi:
            continue
        abstract: str = data.get("abstract") or ""
        keywords: list[str] = data.get("keywords") or []
        text = abstract
        if keywords:
            text += " " + " ".join(str(k) for k in keywords)
        text = text.strip()
        if text:
            result[doi] = text
    return result


@st.cache_data(ttl=300)
def _compute_tsne_layout(
    mtime: float,
    k: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, int], dict[int, list[str]]]:
    """TF-IDF + t-SNE + K-Means for all papers with abstracts.

    Returns:
        positions: {doi: (x, y)}
        cluster_labels: {doi: cluster_id}  (-1 = no abstract text)
        top_terms: {cluster_id: [term1, term2, term3]}
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE

    texts = _load_sidecar_texts(mtime)
    cg, _ = _load_graph_builder()
    graph_dois = list(cg.graph.nodes())  # type: ignore[union-attr]

    dois_with_text = [d for d in graph_dois if d in texts]
    dois_no_text = [d for d in graph_dois if d not in texts]

    positions: dict[str, tuple[float, float]] = {}
    cluster_labels: dict[str, int] = {}
    top_terms: dict[int, list[str]] = {}

    for doi in dois_no_text:
        positions[doi] = (0.0, 0.0)
        cluster_labels[doi] = -1

    n = len(dois_with_text)
    if n == 0:
        return positions, cluster_labels, top_terms

    if n == 1:
        positions[dois_with_text[0]] = (0.0, 0.0)
        cluster_labels[dois_with_text[0]] = 0
        top_terms[0] = []
        return positions, cluster_labels, top_terms

    corpus = [texts[doi] for doi in dois_with_text]
    vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
    X = vectorizer.fit_transform(corpus)
    X_dense: np.ndarray = X.toarray()

    if n >= 3:
        perplexity = min(30.0, float(n - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
            learning_rate="auto",
            init="pca",
        )
        coords: np.ndarray = reducer.fit_transform(X_dense)
    else:
        n_components = min(2, n - 1)
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(X_dense)
        if n_components == 1:
            coords = np.column_stack([reduced, np.zeros(n)])
        else:
            coords = reduced

    effective_k = max(1, min(k, n))
    km = KMeans(n_clusters=effective_k, random_state=42, n_init=10)
    cluster_ids: np.ndarray = km.fit_predict(X_dense)

    for i, doi in enumerate(dois_with_text):
        positions[doi] = (float(coords[i, 0]), float(coords[i, 1]))
        cluster_labels[doi] = int(cluster_ids[i])

    feature_names = vectorizer.get_feature_names_out()
    for cid in range(effective_k):
        mask = cluster_ids == cid
        if not mask.any():
            top_terms[cid] = []
            continue
        centroid = X_dense[mask].mean(axis=0)
        top_idx = centroid.argsort()[::-1][:3]
        top_terms[cid] = [str(feature_names[i]) for i in top_idx]

    return positions, cluster_labels, top_terms


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------


def _classify_nodes(graph: nx.DiGraph) -> dict[str, set[str]]:
    """Classify nodes into 'own', 'cited', and 'citing' sets."""
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
    return " ".join(name.strip().split())


def _bib_key(doi: str, authors: list[str], year: int | str | None) -> str:
    first = (
        authors[0].strip().split(",")[0].split()[-1].lower() if authors else "unknown"
    )
    return re.sub(r"[^a-z0-9]", "", first) + str(year or "nd")


def _format_citation(doi: str, attrs: dict[str, Any], style: str) -> str:
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
# Visual helpers
# ---------------------------------------------------------------------------


def _inject_css() -> None:
    """Inject dark-theme CSS to match the 3D visualisation palette."""
    st.markdown(
        """
<style>
/* ---- Page & sidebar ---- */
.stApp { background-color: #0d0d0d; color: #e0e0e0; }
section[data-testid="stSidebar"] { background-color: #111111; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    border-bottom: 1px solid #2a2a2a;
    background-color: #0d0d0d;
}
.stTabs [data-baseweb="tab"] {
    background-color: #1a1a1a;
    border-radius: 6px 6px 0 0;
    color: #999;
    padding: 6px 20px;
    font-size: 13px;
    border: 1px solid #2a2a2a;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #ffffff !important;
    border-color: #2a5080 !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ddd; background-color: #222; }

/* ---- Metrics ---- */
[data-testid="stMetricValue"] { color: #5b9bd5; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #888; }
[data-testid="stMetricDelta"] { font-size: 0.85rem; }

/* ---- Headings ---- */
h1 { color: #ffffff; font-size: 1.5rem !important; }
h2, h3 { color: #e0e0e0; }

/* ---- DataFrames ---- */
[data-testid="stDataFrame"] {
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    overflow: hidden;
}

/* ---- Expanders ---- */
[data-testid="stExpander"] summary {
    background-color: #1a1a1a;
    border-radius: 4px;
    color: #ddd;
}

/* ---- Buttons ---- */
.stButton > button {
    background-color: #1e3a5f;
    color: #fff;
    border: 1px solid #2a5080;
    border-radius: 4px;
}
.stButton > button:hover { background-color: #2a5080; }

/* ---- Link buttons ---- */
.stLinkButton a {
    background-color: #1a2a3a !important;
    color: #5b9bd5 !important;
    border: 1px solid #2a4a6a !important;
    border-radius: 4px !important;
    font-size: 12px !important;
    padding: 2px 10px !important;
    text-decoration: none !important;
}
.stLinkButton a:hover { background-color: #1e3a5f !important; }

/* ---- Captions ---- */
.stCaption { color: #777 !important; font-size: 12px; }

/* ---- Code blocks ---- */
.stCode { background-color: #1a1a1a !important; }

/* ---- Spinners ---- */
[data-testid="stSpinner"] { color: #5b9bd5; }

/* ---- Checkboxes ---- */
.stCheckbox label { color: #ccc; font-size: 13px; }

/* ---- Dividers ---- */
hr { border-color: #2a2a2a; }
</style>""",
        unsafe_allow_html=True,
    )


def _render_doi_list(dois: list[str], limit: int = 60) -> None:
    """Render a 3-column grid of DOI link-buttons."""
    subset = dois[:limit]
    cols_per_row = 3
    for i in range(0, len(subset), cols_per_row):
        chunk = subset[i : i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, doi in zip(cols, chunk, strict=False):
            short = doi if len(doi) <= 34 else doi[:31] + "…"
            col.link_button(
                short, url=f"https://doi.org/{doi}", use_container_width=True
            )
    if len(dois) > limit:
        st.caption(f"… and {len(dois) - limit} more")


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
            st.link_button("Open DOI ↗", f"https://doi.org/{doi}")


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

    classification = _classify_nodes(graph)
    n_own = len(classification["own"])
    n_cited = len(classification["cited"])
    n_citing = len(classification["citing"])

    c1, c2, c3, _ = st.columns([2, 2, 2, 4])
    show_own = c1.checkbox(f"Own ({n_own})", value=True, key="show_own")
    show_cited = c2.checkbox(f"Cited ({n_cited})", value=True, key="show_cited")
    show_citing = c3.checkbox(f"Citing ({n_citing})", value=True, key="show_citing")

    visible: set[str] = set()
    if show_own:
        visible |= classification["own"]
    if show_cited:
        visible |= classification["cited"]
    if show_citing:
        visible |= classification["citing"]

    all_nodes: set[str] = set(graph.nodes())
    selected_doi: str = st.session_state.get("_selected_doi", "")

    left, right = st.columns([1, 3], gap="small")

    with left:
        _render_paper_list(graph, visible, selected_doi)

    with right:
        st.caption(f"{len(visible)}/{n_nodes} papers · {n_edges} citation edges")

        node_cat: dict[str, str] = {}
        for cat_name in ("citing", "cited", "own"):
            for node in classification[cat_name]:
                node_cat[node] = cat_name

        filter_arg = None if visible == all_nodes else visible
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
                if doi != selected_doi:
                    st.session_state["_selected_doi"] = doi
                    st.rerun()
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
            col_text, col_link = st.columns([10, 1])
            with col_text:
                st.markdown(_format_citation(doi, attrs, style))
            with col_link:
                if doi:
                    st.link_button("↗", f"https://doi.org/{doi}")
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
        if "doi" in df.columns:
            df["doi_url"] = df["doi"].apply(
                lambda d: f"https://doi.org/{d}" if d else ""
            )
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config=(
                {
                    "doi_url": st.column_config.LinkColumn(
                        "Link", display_text="↗ open", width="small"
                    )
                }
                if "doi_url" in df.columns
                else None
            ),
        )

    if report.isolated_nodes:
        st.markdown("#### Isolated Nodes (no citation edges)")
        _render_doi_list(report.isolated_nodes)

    if report.no_abstract:
        st.markdown("#### Papers Missing Abstract")
        _render_doi_list(report.no_abstract)


def _tab_analytics() -> None:
    st.subheader("Centrality Leaderboard")
    cg, _ = _load_graph_builder()
    import pandas as pd

    rows = []
    for node, attrs in cg.graph.nodes(data=True):  # type: ignore[union-attr]
        rows.append(
            {
                "doi_url": f"https://doi.org/{node}" if node else "",
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
    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True,
        column_config={
            "doi_url": st.column_config.LinkColumn(
                "DOI", display_text="↗ open", width="small"
            )
        },
    )


def _tab_missing_papers() -> None:
    st.subheader("Papers Missing PDF")

    manifest_path = LIBRARY_DIR / "missing_pdfs.json"
    if not manifest_path.exists():
        st.info("No `missing_pdfs.json` found — run `run_expansion.py` to generate it.")
        return

    try:
        raw: list[dict[str, Any]] = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read manifest: {exc}")
        return

    # Join citation counts from in-memory graph
    cg, _ = _load_graph_builder()
    node_cc: dict[str, int] = {
        n: d.get("citation_count", 0)
        for n, d in cg.graph.nodes(data=True)  # type: ignore[union-attr]
    }
    for entry in raw:
        entry["citations"] = node_cc.get(entry.get("doi", ""), 0)

    raw.sort(key=lambda e: e.get("citations", 0), reverse=True)

    import pandas as pd

    rows_out = []
    for e in raw:
        authors: list[str] = e.get("authors") or []
        author_str = authors[0] if authors else ""
        if len(authors) > 1:
            author_str += " et al."
        rows_out.append(
            {
                "citations": e.get("citations", 0),
                "year": e.get("year"),
                "title": (e.get("title") or "")[:90],
                "first_author": author_str,
                "journal": (e.get("journal") or "")[:40],
                "doi_url": e.get("doi_url") or "",
            }
        )

    df = pd.DataFrame(rows_out)

    c1, c2 = st.columns([1, 3])
    c1.metric("Missing PDFs", len(df))
    c2.info(
        "Download papers from the DOI links, then run:\n\n"
        "```\npython organize_downloads.py --apply\n```\n"
        "to automatically match and move files from `~/Downloads` into the library."
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "doi_url": st.column_config.LinkColumn(
                "DOI", display_text="↗ open", width="small"
            ),
            "citations": st.column_config.NumberColumn("Citations", format="%d"),
            "year": st.column_config.NumberColumn("Year", format="%d"),
        },
    )


def _build_semantic_figure(
    graph: nx.DiGraph,
    positions: dict[str, tuple[float, float]],
    cluster_labels: dict[str, int],
    top_terms: dict[int, list[str]],
    visible_nodes: set[str] | None,
) -> Any:
    """Build a 2-D Plotly scatter figure coloured by semantic cluster."""
    import plotly.express as px
    import plotly.graph_objects as go

    from src.viz_engine import node_size as _ns

    nodes = [n for n in graph.nodes() if n in positions]
    if visible_nodes is not None:
        nodes = [n for n in nodes if n in visible_nodes]

    if not nodes:
        return go.Figure(
            layout=go.Layout(
                paper_bgcolor="#0a0a0a",
                plot_bgcolor="#111111",
                font={"color": "white"},
            )
        )

    palette = px.colors.qualitative.Plotly
    cluster_ids_present = sorted(set(cluster_labels.get(n, -1) for n in nodes))

    traces: list[go.Scatter] = []
    for cid in cluster_ids_present:
        cat_nodes = [n for n in nodes if cluster_labels.get(n, -1) == cid]
        if not cat_nodes:
            continue

        if cid == -1:
            color = "#444444"
            name = "No abstract"
        else:
            color = palette[cid % len(palette)]
            terms = top_terms.get(cid, [])
            name = f"Cluster {cid}: {', '.join(terms)}" if terms else f"Cluster {cid}"

        xs = [positions[n][0] for n in cat_nodes]
        ys = [positions[n][1] for n in cat_nodes]
        sizes = [_ns(graph.nodes[n].get("citation_count", 0)) * 1.2 for n in cat_nodes]

        hovers = []
        for n in cat_nodes:
            attrs = graph.nodes[n]
            au: list[str] = attrs.get("authors") or []
            first_a = au[0].split()[-1] if au else "?"
            hovers.append(
                f"<b>{_html.escape(str(attrs.get('title', n)))}</b><br>"
                f"{_html.escape(first_a)} · {attrs.get('year', '?')}<br>"
                f"{_html.escape(str(attrs.get('journal', '')))}<br>"
                f"Citations: {attrs.get('citation_count', 0)}<br>"
                f"<i>{_html.escape(name)}</i>"
            )

        traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker={
                    "color": color,
                    "size": sizes,
                    "opacity": 0.82,
                    "line": {"width": 0.5, "color": "rgba(255,255,255,0.15)"},
                },
                hovertext=hovers,
                hoverinfo="text",
                customdata=cat_nodes,
                name=name,
                showlegend=True,
            )
        )

    return go.Figure(
        data=traces,
        layout=go.Layout(
            paper_bgcolor="#0a0a0a",
            plot_bgcolor="#111111",
            font={"color": "white"},
            margin={"l": 20, "r": 20, "b": 20, "t": 20},
            xaxis={"visible": False, "showgrid": False},
            yaxis={"visible": False, "showgrid": False},
            showlegend=True,
            legend={
                "x": 1.01,
                "y": 1.0,
                "xanchor": "left",
                "bgcolor": "rgba(10,10,10,0.8)",
                "bordercolor": "#333",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            uirevision="constant",
        ),
    )


def _tab_semantic_map() -> None:
    st.subheader("Semantic Map")
    cg, _ = _load_graph_builder()
    graph: nx.DiGraph = cg.graph  # type: ignore[assignment]

    if graph.number_of_nodes() == 0:
        st.warning("Graph is empty — run `run_expansion.py` first.")
        return

    classification = _classify_nodes(graph)
    n_own = len(classification["own"])
    n_cited = len(classification["cited"])
    n_citing = len(classification["citing"])

    c1, c2, c3, _, c5 = st.columns([2, 2, 2, 1, 4])
    show_own = c1.checkbox(f"Own ({n_own})", value=True, key="sem_own")
    show_cited = c2.checkbox(f"Cited ({n_cited})", value=True, key="sem_cited")
    show_citing = c3.checkbox(f"Citing ({n_citing})", value=True, key="sem_citing")
    k = int(c5.slider("Clusters (k)", min_value=2, max_value=20, value=6, key="sem_k"))

    visible: set[str] = set()
    if show_own:
        visible |= classification["own"]
    if show_cited:
        visible |= classification["cited"]
    if show_citing:
        visible |= classification["citing"]

    mtime = GRAPH_PATH.stat().st_mtime if GRAPH_PATH.exists() else 0.0

    with st.spinner("Computing semantic layout (t-SNE)…"):
        try:
            positions, cluster_labels, top_terms = _compute_tsne_layout(mtime, k)
        except ImportError:
            st.error(
                "scikit-learn is required for the Semantic Map. "
                "Install it with: `uv pip install scikit-learn>=1.0`"
            )
            return

    n_with_text = sum(1 for doi in graph.nodes() if cluster_labels.get(doi, -1) != -1)
    eff_k = min(k, n_with_text)
    st.caption(
        f"{len(visible)}/{graph.number_of_nodes()} papers visible · "
        f"{n_with_text} have abstracts · k={eff_k} effective clusters"
    )

    if n_with_text == 0:
        st.warning(
            "No papers have abstracts yet. "
            "Run `run_expansion.py --enrich` to fetch them."
        )
        return

    all_nodes: set[str] = set(graph.nodes())
    filter_arg = None if visible == all_nodes else (visible or None)
    fig = _build_semantic_figure(
        graph, positions, cluster_labels, top_terms, filter_arg
    )

    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        key="semantic_map",
    )

    points = getattr(getattr(event, "selection", None), "points", [])
    if points:
        doi = str(points[0].get("customdata", ""))
        if doi:
            if doi != st.session_state.get("_sem_last_doi_opened"):
                st.session_state["_sem_last_doi_opened"] = doi
                import streamlit.components.v1 as components

                components.html(
                    f'<script>window.open("https://doi.org/{doi}", "_blank");</script>',
                    height=0,
                )
            _paper_card(doi)


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
    _inject_css()
    st.title("AugmentedScholar — Network Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Citation Map",
            "Papers",
            "Gap Analysis",
            "Analytics",
            "Missing Papers",
            "Semantic Map",
        ]
    )
    with tab1:
        _tab_citation_map()
    with tab2:
        _tab_papers()
    with tab3:
        _tab_gap_analysis()
    with tab4:
        _tab_analytics()
    with tab5:
        _tab_missing_papers()
    with tab6:
        _tab_semantic_map()


if __name__ == "__main__" or "streamlit" in sys.modules:
    main()
