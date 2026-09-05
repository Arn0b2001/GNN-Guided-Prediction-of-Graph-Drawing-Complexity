"""Minimum layer count for upward layered drawings (small graphs: ILP)."""

from __future__ import annotations

import math

import networkx as nx

try:
    import pulp
except ImportError:  # pragma: no cover
    pulp = None


def layer_upper_bound(g: nx.Graph) -> int:
    """Trivial upper bound: one vertex per layer."""
    return max(g.number_of_nodes(), 1)


def layer_lower_bound(g: nx.Graph) -> int:
    """Simple lower bound from clique / path structure."""
    if g.number_of_nodes() == 0:
        return 0
    try:
        clique = nx.graph_clique_number(g)
    except Exception:
        clique = max((len(c) for c in nx.find_cliques(g)), default=1)
    return max(clique, 1)


def minimum_layers_ilp(g: nx.Graph, max_layers: int | None = None) -> int | None:
    """
    Minimum number of layers in a proper layered drawing (vertices on horizontal
    layers, edges go between adjacent layers after orienting edges upward).

    Uses ILP on small graphs. Returns None if pulp unavailable or infeasible.
    """
    if pulp is None:
        return None
    n = g.number_of_nodes()
    if n == 0:
        return 0
    if n == 1:
        return 1

    nodes = list(g.nodes())
    max_layers = max_layers or min(n, layer_upper_bound(g))
    lb = layer_lower_bound(g)

    prob = pulp.LpProblem("min_layers", pulp.LpMinimize)
    y = {L: pulp.LpVariable(f"y_{L}", cat="Binary") for L in range(1, max_layers + 1)}
    x = {
        (v, L): pulp.LpVariable(f"x_{v}_{L}", cat="Binary")
        for v in nodes
        for L in range(1, max_layers + 1)
    }

    prob += pulp.lpSum(y[L] for L in range(1, max_layers + 1))
    for v in nodes:
        prob += pulp.lpSum(x[(v, L)] for L in range(1, max_layers + 1)) == 1
    for L in range(1, max_layers + 1):
        for v in nodes:
            prob += x[(v, L)] <= y[L]
    for u, v in g.edges():
        prob += pulp.lpSum(
            x[(u, L)] + x[(v, L + 1)] + x[(v, L)] + x[(u, L + 1)]
            for L in range(1, max_layers)
        ) >= 1

    prob += pulp.lpSum(y[L] for L in range(1, max_layers + 1)) >= lb
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    return int(round(pulp.value(prob.objective)))


def minimum_layers(g: nx.Graph, exact_max_nodes: int = 80) -> tuple[int, str]:
    """
    Return (layer_count, label_type) where label_type is 'exact' or 'bound'.
    """
    n = g.number_of_nodes()
    if n == 0:
        return 0, "exact"
    if n <= exact_max_nodes:
        exact = minimum_layers_ilp(g)
        if exact is not None:
            return exact, "exact"

    # Bound: pathwidth + 1 is a safe upper bound for layer width; use a simpler proxy.
    if nx.is_tree(g):
        return int(math.ceil(math.log2(n + 1))) + 1, "bound"
    return min(layer_upper_bound(g), max(layer_lower_bound(g), int(math.sqrt(n)) + 1)), "bound"


def class_layer_hint(graph_class: str | None, n_nodes: int) -> int | None:
    if graph_class == "maximal-outerplanar":
        return min(n_nodes, 3)  # outerplanar graphs need few layers
    return None
