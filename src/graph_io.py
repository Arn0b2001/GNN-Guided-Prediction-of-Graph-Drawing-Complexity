"""Load GraphML files into NetworkX graphs."""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def read_graphml(path: str | Path) -> nx.Graph:
    """Read an undirected simple graph from GraphML."""
    g = nx.read_graphml(path)
    # Normalize node labels to integers where possible.
    mapping = {}
    for node in g.nodes():
        if isinstance(node, str) and node.startswith("n") and node[1:].isdigit():
            mapping[node] = int(node[1:])
        elif isinstance(node, str) and node.isdigit():
            mapping[node] = int(node)
    if mapping:
        g = nx.relabel_nodes(g, mapping)
    g = nx.Graph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def graph_stats(g: nx.Graph) -> dict:
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "density": float(nx.density(g)) if g.number_of_nodes() > 1 else 0.0,
        "is_connected": nx.is_connected(g),
        "n_components": nx.number_connected_components(g),
    }
