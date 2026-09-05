"""Block-cut tree and basic structural features."""

from __future__ import annotations

import json

import networkx as nx


def block_cut_features(g: nx.Graph) -> dict:
    """Extract block-cut tree statistics."""
    if g.number_of_nodes() == 0:
        return {
            "n_blocks": 0,
            "n_cut_vertices": 0,
            "max_block_size": 0,
            "bc_tree_n_nodes": 0,
            "bc_tree_n_edges": 0,
        }

    blocks = list(nx.biconnected_components(g))
    cut_vertices = list(nx.articulation_points(g))
    block_sizes = [len(b) for b in blocks]

    # Build block-cut tree (bipartite: block nodes + cut vertex nodes)
    bc = nx.Graph()
    for i, block in enumerate(blocks):
        bc.add_node(f"B{i}", kind="block", size=len(block))
        for v in block:
            if v in cut_vertices:
                bc.add_node(f"C{v}", kind="cut", vertex=v)
                bc.add_edge(f"B{i}", f"C{v}")

    return {
        "n_blocks": len(blocks),
        "n_cut_vertices": len(cut_vertices),
        "max_block_size": max(block_sizes) if block_sizes else 0,
        "bc_tree_n_nodes": bc.number_of_nodes(),
        "bc_tree_n_edges": bc.number_of_edges(),
    }


def structural_features(g: nx.Graph, peel_layers: dict | None = None) -> dict:
    feats = block_cut_features(g)
    feats["avg_degree"] = (
        sum(dict(g.degree()).values()) / g.number_of_nodes() if g.number_of_nodes() else 0.0
    )
    feats["max_degree"] = max((d for _, d in g.degree()), default=0)
    if peel_layers:
        feats["max_peel_layer"] = max(peel_layers.values()) if peel_layers else 0
        feats["avg_peel_layer"] = sum(peel_layers.values()) / len(peel_layers)
    else:
        feats["max_peel_layer"] = None
        feats["avg_peel_layer"] = None
    return feats


def block_cut_tree_json(g: nx.Graph) -> str:
    """Serialize block-cut tree for storage."""
    if g.number_of_nodes() == 0:
        return json.dumps({"nodes": [], "edges": []})

    blocks = list(nx.biconnected_components(g))
    cut_vertices = set(nx.articulation_points(g))
    nodes = []
    edges = []
    for i, block in enumerate(blocks):
        nodes.append({"id": f"B{i}", "kind": "block", "vertices": sorted(block, key=str)})
        for v in block:
            if v in cut_vertices:
                cid = f"C{v}"
                if not any(n["id"] == cid for n in nodes):
                    nodes.append({"id": cid, "kind": "cut", "vertex": v})
                edges.append({"source": f"B{i}", "target": cid})
    return json.dumps({"nodes": nodes, "edges": edges})
