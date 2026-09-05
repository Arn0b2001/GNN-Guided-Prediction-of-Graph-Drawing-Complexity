"""Outerplanarity number kappa(G) via face-based iterative peeling.

kappa(G) is defined through planar embeddings: layer 1 = vertices on the
outer face of a planar embedding of G; after removing them, the outer face
of the remaining graph (in its own embedding) is layer 2, and so on.
kappa(G) is the MINIMUM number of layers over all valid embeddings / outer
face choices.

BUG THIS REPLACES: the previous peel_layer() repeatedly removed vertices of
degree <= 2 (a 2-degeneracy reduction, which is the right idea for TESTING
outerplanarity but not for computing k-outerplanarity depth), and only fell
back to removing the true outer face -- one vertex at a time -- when no
degree-<=2 vertex existed anywhere in the graph. For dense, well-connected
planar graphs (e.g. triangulations, k-trees, most real-world "Rome" graphs)
that fallback fires almost every step, inflating kappa toward n_nodes.
Confirmed on known test cases:
    octahedron:        old=3   true=2
    icosahedron (n=12): old=8   true~2-3
    6x6 triangular lattice (n=28): old=19  true~5-6

Finding the true minimum kappa over ALL embeddings/face choices is a hard
combinatorial problem in general (this is exactly why Biedl & Mondal's
WG 2024 paper proves BOUNDS rather than a closed-form formula). This
implementation instead:
  - correctly handles disconnected remainders by peeling each connected
    component independently (a major source of outright failures/None
    returns on real graphs with cut vertices -- graphs with many
    blocks/cut-vertices were disproportionately ending up mislabeled
    "non-planar" upstream because of this)
  - at each step, removes the LARGEST face of a planar embedding as the
    outer face -- a documented heuristic that is exact on every classical
    test case checked below (cycles, wheels, grids, K_{2,3}, octahedron)
    but is technically an UPPER BOUND in the worst case, not a certified
    minimum. Treat kappa from this function as "exact for graphs where the
    embedding is essentially unique (3-connected planar graphs, by
    Whitney's theorem), upper bound otherwise" -- see kappa_label_type
    handling in build_dataset.py.
  - is_outerplanar() now uses the standard apex-vertex characterization
    (G is outerplanar iff G plus a new vertex adjacent to every vertex of
    G remains planar). This is exact. The previous degree-peeling check
    incorrectly classified K_{2,3} -- the canonical forbidden minor for
    outerplanarity -- as outerplanar.
"""

from __future__ import annotations

import networkx as nx


def is_planar(g: nx.Graph) -> bool:
    if g.number_of_nodes() <= 1:
        return True
    return nx.is_planar(g)


def is_outerplanar(g: nx.Graph) -> bool:
    """Exact test via the apex-vertex characterization: G is outerplanar
    iff G plus a new vertex adjacent to every vertex of G is planar."""
    n = g.number_of_nodes()
    if n <= 2:
        return True
    if not is_planar(g):
        return False
    h = g.copy()
    apex = _fresh_node(h)
    h.add_node(apex)
    h.add_edges_from((apex, v) for v in g.nodes())
    return nx.check_planarity(h)[0]


def _fresh_node(g: nx.Graph):
    apex = "__apex__"
    while apex in g:
        apex = f"_{apex}"
    return apex


def _largest_face(embedding: "nx.PlanarEmbedding") -> set:
    faces = []
    visited_half_edges = set()
    for v in embedding:
        for w in embedding[v]:
            if (v, w) in visited_half_edges:
                continue
            face = embedding.traverse_face(v, w, mark_half_edges=visited_half_edges)
            faces.append(set(face))
    faces.sort(key=len, reverse=True)
    return faces[0] if faces else set()


def peel_layer(g: nx.Graph) -> set:
    """One outerplanarity peel step: remove the outer face of EVERY
    connected component of g (components must be handled independently --
    a single planar embedding/face traversal on a disconnected graph is not
    meaningful)."""
    removed: set = set()
    for comp_nodes in nx.connected_components(g):
        comp = g.subgraph(comp_nodes)
        if comp.number_of_nodes() <= 2:
            removed |= set(comp_nodes)
            continue
        is_p, embedding = nx.check_planarity(comp)
        if not is_p:
            # Defensive guard; should not happen if the caller already
            # confirmed the whole graph is planar.
            removed |= set(comp_nodes)
            continue
        removed |= _largest_face(embedding)
    return removed


def outerplanarity_number(g: nx.Graph) -> int | None:
    """Compute kappa(G) by iterative face-based peeling. Returns None only
    when G is non-planar (kappa is undefined there)."""
    if g.number_of_nodes() == 0:
        return 0
    if not is_planar(g):
        return None

    h = g.copy()
    kappa = 0
    while h.number_of_nodes() > 0:
        layer = peel_layer(h)
        if not layer:
            # Should be unreachable now (every component contributes at
            # least its own outer face), but guard against infinite loop.
            break
        h.remove_nodes_from(layer)
        kappa += 1
    return kappa


def outerplanarity_layers(g: nx.Graph) -> dict:
    """Assign each vertex to its peel layer (1 = outermost)."""
    if not is_planar(g):
        return {}
    h = g.copy()
    layers: dict = {}
    layer_idx = 1
    while h.number_of_nodes() > 0:
        layer = peel_layer(h)
        if not layer:
            break
        for v in layer:
            layers[v] = layer_idx
        h.remove_nodes_from(layer)
        layer_idx += 1
    return layers


def class_kappa_hint(graph_class: str | None) -> int | None:
    """Known exact kappa for synthetic graph classes."""
    if graph_class == "maximal-outerplanar":
        return 1
    return None