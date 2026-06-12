"""3D Plotly visualisation utilities for the AugmentedScholar citation graph.

Build a Plotly Scatter3d figure from a NetworkX DiGraph.  Nodes are coloured
on a blue → red gradient by publication year; size scales with citation count.
Edge traces are rendered as semi-transparent lines between node positions.

Typical usage::

    from src.viz_engine import build_plotly_3d, compute_layout_3d
    positions = compute_layout_3d(graph)
    fig = build_plotly_3d(graph, positions=positions)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import plotly.graph_objects as go

# Colour gradient endpoints (kept for temporal_colormap utility function)
_COLOUR_OLD = (0, 0, 220)
_COLOUR_NEW = (220, 0, 0)

_NODE_MIN_SIZE = 6.0
_NODE_MAX_SIZE = 28.0

# Plotly colorscale used in build_plotly_3d (blue → red)
_COLORSCALE = [[0.0, "#0000dc"], [1.0, "#dc0000"]]
# Fixed lower year bound: papers from ≤ 2010 all appear as deep blue
_YEAR_MIN = 2010

# Colorbar rendered on the left of the figure
_COLORBAR: dict[str, Any] = {
    "x": 0.0,
    "xanchor": "left",
    "y": 0.5,
    "yanchor": "middle",
    "title": {
        "text": "Year",
        "side": "right",
        "font": {"color": "white", "size": 11},
    },
    "thickness": 12,
    "len": 0.55,
    "tickformat": "d",
    "tickfont": {"color": "white", "size": 10},
    "outlinewidth": 0,
}

# Per-category marker styles — rendered back-to-front so own papers appear on top
_CATEGORY_STYLE: dict[str, dict[str, Any]] = {
    "other": {
        "symbol": "circle",
        "size_scale": 0.9,
        "opacity": 0.60,
        "line_width": 0.3,
        "line_color": "rgba(255,255,255,0.2)",
        "name": "Other",
    },
    "citing": {
        "symbol": "cross",
        "size_scale": 1.0,
        "opacity": 0.72,
        "line_width": 0.4,
        "line_color": "rgba(255,255,255,0.3)",
        "name": "Citing (citing author)",
    },
    "cited": {
        "symbol": "circle",
        "size_scale": 1.0,
        "opacity": 0.72,
        "line_width": 0.4,
        "line_color": "rgba(255,255,255,0.3)",
        "name": "Cited (references)",
    },
    "own": {
        "symbol": "diamond",
        "size_scale": 1.8,
        "opacity": 1.0,
        "line_width": 1.5,
        "line_color": "white",
        "name": "Own papers",
    },
}
# Render order: other → citing → cited → own (own drawn last = on top)
_CAT_ORDER = ["other", "citing", "cited", "own"]


def temporal_colormap(years: list[int]) -> list[str]:
    """Return one hex colour per year, interpolated blue → red by publication age.

    Args:
        years: Publication years in graph-iteration order.

    Returns:
        List of ``'#rrggbb'`` strings, same length as *years*.

    Examples:
        >>> temporal_colormap([2000, 2010, 2020])
        ['#0000dc', '#6e006e', '#dc0000']
    """
    if not years:
        return []
    min_y, max_y = min(years), max(years)
    results: list[str] = []
    for y in years:
        t = 0.5 if max_y == min_y else (y - min_y) / (max_y - min_y)
        t = max(0.0, min(1.0, t))
        r = int(_COLOUR_OLD[0] + t * (_COLOUR_NEW[0] - _COLOUR_OLD[0]))
        g = int(_COLOUR_OLD[1] + t * (_COLOUR_NEW[1] - _COLOUR_OLD[1]))
        b = int(_COLOUR_OLD[2] + t * (_COLOUR_NEW[2] - _COLOUR_OLD[2]))
        results.append(f"#{r:02x}{g:02x}{b:02x}")
    return results


def compute_layout_3d(
    graph: nx.DiGraph,
    *,
    seed: int = 42,
    max_nodes: int = 1500,
) -> dict[str, tuple[float, float, float]]:
    """Compute 3D spring-layout positions for all nodes in *graph*.

    For graphs larger than *max_nodes* the layout is computed on a subgraph
    of the highest-degree nodes, keeping the visualisation tractable.

    Args:
        graph: Citation DiGraph.
        seed: Random seed for reproducibility.
        max_nodes: Cap on nodes passed to :func:`networkx.spring_layout`.

    Returns:
        Mapping of ``node_id → (x, y, z)``.
    """
    if graph.number_of_nodes() == 0:
        return {}

    if graph.number_of_nodes() > max_nodes:
        top = sorted(graph.degree(), key=lambda kv: kv[1], reverse=True)[:max_nodes]
        sub: nx.DiGraph = graph.subgraph([n for n, _ in top])
    else:
        sub = graph

    raw: dict[str, Any] = nx.spring_layout(sub, dim=3, seed=seed)
    return {node: (float(v[0]), float(v[1]), float(v[2])) for node, v in raw.items()}


def node_size(citation_count: int) -> float:
    """Log-scaled node size clamped to [_NODE_MIN_SIZE, _NODE_MAX_SIZE]."""
    raw = _NODE_MIN_SIZE + 6 * math.log10(citation_count + 1)
    return max(_NODE_MIN_SIZE, min(_NODE_MAX_SIZE, raw))


def build_plotly_3d(
    graph: nx.DiGraph,
    *,
    positions: dict[str, tuple[float, float, float]] | None = None,
    node_categories: dict[str, str] | None = None,
) -> go.Figure:
    """Assemble an interactive Plotly 3D scatter figure for *graph*.

    Nodes are coloured by publication year on a blue → red scale anchored at
    :data:`_YEAR_MIN` (2010) with a colorbar on the left.  Size scales with
    citation count (log scale).  Each marker carries its DOI as
    ``customdata`` for Streamlit click-event callbacks.

    When *node_categories* is provided, nodes are split into labelled traces
    with distinct marker symbols: ``"own"`` (diamond, large, fully opaque),
    ``"cited"`` (circle), ``"citing"`` (cross), ``"other"`` (circle, dim).

    Args:
        graph: Directed citation graph with node attributes *year*, *title*,
            *journal*, *citation_count*, *authors*.
        positions: Precomputed 3D positions from :func:`compute_layout_3d`.
            Computed on the fly when ``None``.
        node_categories: Mapping of ``node_id → category`` where category is
            one of ``"own"``, ``"cited"``, ``"citing"``, ``"other"``.
            When ``None``, all nodes are rendered as a single trace.

    Returns:
        :class:`plotly.graph_objects.Figure` with edge trace and one or more
        node traces.
    """
    if positions is None:
        positions = compute_layout_3d(graph)

    nodes = [n for n in graph.nodes() if n in positions]
    all_years = [graph.nodes[n].get("year", _YEAR_MIN) for n in nodes]
    cmax = max(all_years) if all_years else _YEAR_MIN + 14

    # Edge trace — grey semi-transparent lines
    ex: list[float | None] = []
    ey: list[float | None] = []
    ez: list[float | None] = []
    for src, dst in graph.edges():
        if src in positions and dst in positions:
            sx, sy, sz = positions[src]
            dx, dy, dz = positions[dst]
            ex += [sx, dx, None]
            ey += [sy, dy, None]
            ez += [sz, dz, None]

    traces: list[go.Scatter3d] = [
        go.Scatter3d(
            x=ex,
            y=ey,
            z=ez,
            mode="lines",
            line={"width": 0.5, "color": "rgba(150,150,150,0.25)"},
            hoverinfo="none",
            showlegend=False,
        )
    ]

    # Node traces rendered back-to-front; own papers appear on top
    colorbar_placed = False
    for cat in _CAT_ORDER:
        if node_categories is not None:
            cat_nodes = [n for n in nodes if node_categories.get(n, "other") == cat]
        elif cat == "other":
            cat_nodes = nodes  # single trace when no category mapping
        else:
            continue
        if not cat_nodes:
            continue

        style = _CATEGORY_STYLE[cat]
        years = [graph.nodes[n].get("year", _YEAR_MIN) for n in cat_nodes]
        sizes = [
            node_size(graph.nodes[n].get("citation_count", 0)) * style["size_scale"]
            for n in cat_nodes
        ]
        labels: list[str] = []
        hovers: list[str] = []
        for n in cat_nodes:
            attrs = graph.nodes[n]
            author_list: list[str] = attrs.get("authors", [])
            first = author_list[0].split()[-1] if author_list else "?"
            labels.append(f"{first} ({attrs.get('year', '?')})")
            hovers.append(
                f"<b>{attrs.get('title', n)}</b><br>"
                f"{attrs.get('journal', '')}<br>"
                f"Citations: {attrs.get('citation_count', 0)}"
            )

        marker: dict[str, Any] = {
            "symbol": style["symbol"],
            "size": sizes,
            "color": years,
            "colorscale": _COLORSCALE,
            "cmin": _YEAR_MIN,
            "cmax": cmax,
            "opacity": style["opacity"],
            "line": {"width": style["line_width"], "color": style["line_color"]},
            "showscale": not colorbar_placed,
        }
        if not colorbar_placed:
            marker["colorbar"] = _COLORBAR
            colorbar_placed = True

        traces.append(
            go.Scatter3d(
                x=[positions[n][0] for n in cat_nodes],
                y=[positions[n][1] for n in cat_nodes],
                z=[positions[n][2] for n in cat_nodes],
                mode="markers+text",
                marker=marker,
                text=labels,
                textposition="top center",
                textfont={"size": 8, "color": "white"},
                hovertext=hovers,
                hoverinfo="text",
                customdata=cat_nodes,
                name=style["name"],
                showlegend=node_categories is not None,
            )
        )

    show_legend = node_categories is not None
    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            paper_bgcolor="#0a0a0a",
            font={"color": "white"},
            scene={
                "domain": {"x": [0.08, 1.0], "y": [0, 1]},
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
                "bgcolor": "#0a0a0a",
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
            showlegend=show_legend,
            legend={
                "x": 0.09,
                "y": 0.99,
                "xanchor": "left",
                "yanchor": "top",
                "bgcolor": "rgba(10,10,10,0.7)",
                "font": {"size": 11},
            },
            uirevision="constant",
        ),
    )
    return fig


def load_paper_metadata(doi: str, library_dir: Path) -> dict[str, Any] | None:
    """Load the JSON sidecar for *doi* from *library_dir*.

    Scans all ``*.json`` files in *library_dir* for a ``doi`` field matching
    *doi*.  Returns the first match, or ``None`` if not found.

    Args:
        doi: DOI string to look up.
        library_dir: Path to the library root directory.

    Returns:
        Parsed sidecar dict, or ``None`` if not found.
    """
    for json_path in library_dir.glob("*.json"):
        try:
            data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("doi") == doi:
            return data
    return None
