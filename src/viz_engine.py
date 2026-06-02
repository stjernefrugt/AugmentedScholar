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

# Colour gradient: blue (oldest) → red (most recent)
_COLOUR_OLD = (0, 0, 220)
_COLOUR_NEW = (220, 0, 0)

_NODE_MIN_SIZE = 6.0
_NODE_MAX_SIZE = 28.0


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


def _node_size(citation_count: int) -> float:
    """Log-scaled node size clamped to [_NODE_MIN_SIZE, _NODE_MAX_SIZE]."""
    raw = _NODE_MIN_SIZE + 6 * math.log10(citation_count + 1)
    return max(_NODE_MIN_SIZE, min(_NODE_MAX_SIZE, raw))


def build_plotly_3d(
    graph: nx.DiGraph,
    *,
    positions: dict[str, tuple[float, float, float]] | None = None,
) -> go.Figure:
    """Assemble an interactive Plotly 3D scatter figure for *graph*.

    Nodes are coloured by publication year (blue=old, red=recent) and sized
    proportionally to their citation count (log scale).  Each node marker
    carries its DOI as ``customdata`` so that Streamlit's ``on_select``
    callback can identify which paper was clicked.

    Args:
        graph: Directed citation graph with node attributes *year*, *title*,
            *journal*, *citation_count*, *authors*.
        positions: Precomputed 3D positions from :func:`compute_layout_3d`.
            Computed on the fly when ``None``.

    Returns:
        :class:`plotly.graph_objects.Figure` containing an edge trace and a
        node trace.
    """
    if positions is None:
        positions = compute_layout_3d(graph)

    nodes = [n for n in graph.nodes() if n in positions]
    years = [graph.nodes[n].get("year", 2000) for n in nodes]
    colours = temporal_colormap(years)

    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    sizes: list[float] = []
    labels: list[str] = []
    hovers: list[str] = []
    custom: list[str] = []

    for node in nodes:
        x, y, z = positions[node]
        attrs = graph.nodes[node]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        sizes.append(_node_size(attrs.get("citation_count", 0)))
        author_list: list[str] = attrs.get("authors", [])
        first = author_list[0].split()[-1] if author_list else "?"
        labels.append(f"{first} ({attrs.get('year', '?')})")
        hovers.append(
            f"<b>{attrs.get('title', node)}</b><br>"
            f"{attrs.get('journal', '')}<br>"
            f"Citations: {attrs.get('citation_count', 0)}"
        )
        custom.append(node)

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

    edge_trace = go.Scatter3d(
        x=ex,
        y=ey,
        z=ez,
        mode="lines",
        line={"width": 0.5, "color": "rgba(150,150,150,0.25)"},
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers+text",
        marker={
            "size": sizes,
            "color": colours,
            "opacity": 0.85,
            "line": {"width": 0.5, "color": "white"},
        },
        text=labels,
        textposition="top center",
        textfont={"size": 8, "color": "white"},
        hovertext=hovers,
        hoverinfo="text",
        customdata=custom,
        name="papers",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#0a0a0a",
            font={"color": "white"},
            scene={
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "zaxis": {"visible": False},
                "bgcolor": "#0a0a0a",
            },
            margin={"l": 0, "r": 0, "b": 0, "t": 0},
            showlegend=False,
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
