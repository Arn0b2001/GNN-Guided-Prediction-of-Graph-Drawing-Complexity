"""Bend complexity estimates and bounds -- v2.

Replaces the earlier non-planar fallback (which just returned n_edges,
flagged "trivial_bound") with a real, standard graph-drawing technique:
greedy planarization.

METHOD (for non-planar graphs):
  1. Repeatedly remove the edge between the two highest-degree endpoints
     until the remaining graph is planar. The number of edges removed is
     an UPPER BOUND on the graph's "skewness" -- the minimum number of
     edges whose deletion makes a graph planar. Skewness is a standard,
     well-studied graph parameter (it appears in Garey & Johnson's NP-hard
     problem list; computing it exactly is NP-hard, same as the drawing
     measures already in this codebase, which is why an upper-bound
     heuristic is used here rather than an exact algorithm).
  2. Apply the existing excess-edge bend bound to the remaining planar
     "core" graph.
  3. Add one bend per removed edge, on the standard convention that a
     removed/crossing edge must be rerouted around the obstruction with at
     least one extra bend to avoid the crossing in a polyline drawing.

This is a heuristic UPPER bound, not a proven minimum -- label it as such
("planarization_bound") in any results you report, and say so explicitly
in your methodology section: "bend complexity for non-planar graphs is
approximated via greedy planarization (edge removal to planarity) plus a
polyline-rerouting penalty of one bend per removed edge; skewness itself
is NP-hard to compute exactly, so this is an upper bound, not a proven
minimum." That sentence is honest and defensible on its own -- it does
not require inventing new theory, just naming the technique for what it
is.
"""

from __future__ import annotations

import networkx as nx


def planarize_upper_bound(g: nx.Graph) -> tuple[int, nx.Graph]:
    """Greedy edge removal toward planarity.
    Returns (num_edges_removed, planar_core_graph)."""
    h = g.copy()
    removed = 0
    while h.number_of_edges() > 0 and not nx.check_planarity(h)[0]:
        edge = max(h.edges(), key=lambda e: h.degree(e[0]) + h.degree(e[1]))
        h.remove_edge(*edge)
        removed += 1
    return removed, h


def bend_upper_bound(g: nx.Graph) -> tuple[int, str]:
    """Conservative upper bound on bends in a polyline drawing.
    Returns (bound, label_type)."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    if n <= 1:
        return 0, "exact"

    from .outerplanarity import is_outerplanar

    if is_outerplanar(g):
        return 0, "exact"

    if nx.is_planar(g) and n >= 3:
        excess = max(0, m - (2 * n - 3))
        return excess, "bound"

    # Non-planar: planarize, bound the planar core, add rerouting penalty.
    skew_ub, core = planarize_upper_bound(g)
    cn, cm = core.number_of_nodes(), core.number_of_edges()
    core_excess = max(0, cm - (2 * cn - 3)) if cn >= 3 else 0
    return core_excess + skew_ub, "planarization_bound"


def bend_lower_bound(g: nx.Graph) -> int:
    if g.number_of_edges() == 0:
        return 0
    return 0


def bend_complexity(g: nx.Graph, kappa: int | None = None) -> tuple[int, str]:
    """Return (bend_estimate, label_type)."""
    from .outerplanarity import is_outerplanar

    if is_outerplanar(g):
        return 0, "exact"

    upper, upper_type = bend_upper_bound(g)

    if kappa is not None and kappa > 0:
        est = max(0, (kappa - 1) * g.number_of_edges() // max(g.number_of_nodes(), 1))
        return min(est, upper), upper_type

    return upper, upper_type
