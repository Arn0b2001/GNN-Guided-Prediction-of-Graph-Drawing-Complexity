"""
Full-dataset version of identify_bound_violations.py -- scans the ENTIRE
labels.csv (not just one train/test split) for kappa values that exceed
the proven Biedl-Mondal bound, and re-verifies EVERY violation using the
multi-strategy witness search (works for large graphs, unlike plain
exhaustive search which is capped at ~18 nodes).

This exists because a separate "full scan" script variant only had the
exhaustive small-graph search wired in, so every violation (all of which
turned out to be large k-tree graphs) was skipped rather than checked --
producing a misleading "0 confirmed" result. This version includes both
paths, matching the per-split script that already confirmed most
violations are real overestimates.

Usage:
    python identify_bound_violations_full.py --labels_csv path/to/labels.csv
"""
from __future__ import annotations

import argparse
import csv

import networkx as nx

EXACT_SEARCH_MAX_NODES = 18


def biedl_mondal_bound(g: nx.Graph) -> float:
    n = g.number_of_nodes()
    if n <= 1:
        return 0.0
    if not nx.is_connected(g):
        return max(biedl_mondal_bound(g.subgraph(c)) for c in nx.connected_components(g))
    rad = nx.radius(g)
    return min(1 + rad, (n + 26) / 6)


def _all_faces(embedding: "nx.PlanarEmbedding") -> list:
    faces = []
    visited = set()
    for v in embedding:
        for w in embedding[v]:
            if (v, w) in visited:
                continue
            face = embedding.traverse_face(v, w, mark_half_edges=visited)
            faces.append(set(face))
    return faces


def _largest_face(embedding: "nx.PlanarEmbedding") -> set:
    faces = _all_faces(embedding)
    faces.sort(key=len, reverse=True)
    return faces[0] if faces else set()


def exact_kappa(g: nx.Graph) -> int | None:
    n = g.number_of_nodes()
    if n == 0:
        return 0
    if n <= 2:
        return 1
    comps = list(nx.connected_components(g))
    if len(comps) > 1:
        vals = [exact_kappa(g.subgraph(c).copy()) for c in comps]
        if any(v is None for v in vals):
            return None
        return max(vals)
    is_planar, embedding = nx.check_planarity(g)
    if not is_planar:
        return None
    faces = _all_faces(embedding)
    if not faces:
        return 1
    best = None
    for face in faces:
        remaining = g.copy()
        remaining.remove_nodes_from(face)
        sub = exact_kappa(remaining)
        if sub is None:
            continue
        candidate = 1 + sub
        if best is None or candidate < best:
            best = candidate
    return best


def _peel_from_face_then_heuristic(g: nx.Graph, first_face: set) -> int:
    h = g.copy()
    h.remove_nodes_from(first_face)
    kappa = 1
    while h.number_of_nodes() > 0:
        layer = set()
        for comp_nodes in nx.connected_components(h):
            comp = h.subgraph(comp_nodes)
            if comp.number_of_nodes() <= 2:
                layer |= set(comp_nodes)
                continue
            is_p, embedding = nx.check_planarity(comp)
            if not is_p:
                layer |= set(comp_nodes)
                continue
            layer |= _largest_face(embedding)
        if not layer:
            break
        h.remove_nodes_from(layer)
        kappa += 1
    return kappa


def reverify_large_graph_kappa(g: nx.Graph, n_random_restarts: int = 8) -> tuple:
    import random
    is_planar, embedding = nx.check_planarity(g)
    if not is_planar:
        return None, "non-planar"

    candidates = []
    if nx.is_connected(g):
        center_vertex = nx.center(g)[0]
        faces = _all_faces(embedding)
        center_faces = [f for f in faces if center_vertex in f]
        if center_faces:
            k = _peel_from_face_then_heuristic(g, center_faces[0])
            candidates.append((k, "radius-center face (Biedl-Mondal Obs.1 construction)"))

    all_faces = _all_faces(embedding)
    if all_faces:
        smallest = min(all_faces, key=len)
        k = _peel_from_face_then_heuristic(g, smallest)
        candidates.append((k, "smallest face"))

    random.seed(0)
    for i in range(min(n_random_restarts, len(all_faces))):
        face = random.choice(all_faces)
        k = _peel_from_face_then_heuristic(g, face)
        candidates.append((k, f"random face #{i}"))

    if not candidates:
        return None, "no candidate faces found"
    best_k, best_strategy = min(candidates, key=lambda x: x[0])
    return best_k, best_strategy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    args = ap.parse_args()

    print("Scanning entire dataset for bound violations...")
    print(f"Reading from: {args.labels_csv}")

    violations = []
    total = 0
    n_planar_kappa = 0
    n_read_errors = 0

    with open(args.labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            kappa_type = row.get("kappa_label_type", "")
            if kappa_type in ("not_applicable", "failed", ""):
                continue
            raw_kappa = row.get("kappa", "")
            if raw_kappa in ("", None):
                continue
            try:
                true_kappa = float(raw_kappa)
            except ValueError:
                continue
            n_planar_kappa += 1

            try:
                g = nx.read_graphml(row["path"])
                g = nx.Graph(g)
            except Exception:
                n_read_errors += 1
                continue

            bound = biedl_mondal_bound(g)
            gap = bound - true_kappa
            if gap < -1e-6:
                violations.append({
                    "graph_id": row.get("graph_id", "?"),
                    "n_nodes": g.number_of_nodes(),
                    "recorded_kappa": true_kappa,
                    "bound": bound,
                    "gap": gap,
                    "graph": g,
                })

    print(f"{'='*70}\nFULL DATASET SCAN RESULTS\n{'='*70}")
    print(f"Total graphs scanned: {total}")
    print(f"Planar graphs with kappa: {n_planar_kappa}")
    print(f"Graphs with read errors: {n_read_errors}")
    print(f"Bound violations found: {len(violations)}")
    print(f"Violation rate: {100*len(violations)/max(n_planar_kappa,1):.2f}%")

    print(f"\n{'='*70}\nDETAILED VIOLATION ANALYSIS (with multi-strategy re-verification)\n{'='*70}")
    print(f"{'graph_id':<20}{'n_nodes':<10}{'recorded_kappa':<16}{'bound':<10}{'gap':<10}{'best_found':<14}{'note'}")
    print("-" * 130)

    n_confirmed = 0
    n_loose_bound = 0
    n_skipped = 0
    n_failed = 0

    for v in violations:
        n = v["n_nodes"]
        if n <= EXACT_SEARCH_MAX_NODES:
            exact = exact_kappa(v["graph"])
            if exact is None:
                note = "exact search failed"
                n_failed += 1
                best_str = "None"
            elif exact < v["recorded_kappa"]:
                note = "CONFIRMED (exact search): recorded label was an overestimate"
                n_confirmed += 1
                best_str = str(exact)
            else:
                note = "matches exact search -- bound itself may be loose here"
                n_loose_bound += 1
                best_str = str(exact)
        else:
            best_k, strategy = reverify_large_graph_kappa(v["graph"])
            if best_k is None:
                note = "re-verification failed"
                n_failed += 1
                best_str = "None"
            elif best_k < v["recorded_kappa"]:
                note = f"CONFIRMED (multi-strategy witness via '{strategy}'): overestimate"
                n_confirmed += 1
                best_str = str(best_k)
            else:
                note = f"no strategy beat recorded label (best={best_k} via '{strategy}') -- inconclusive"
                n_loose_bound += 1
                best_str = str(best_k)

        print(f"{str(v['graph_id']):<20}{n:<10}{v['recorded_kappa']:<16.1f}{v['bound']:<10.2f}"
              f"{v['gap']:<10.2f}{best_str:<14}{note}")

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"Total violations: {len(violations)}")
    print(f"  CONFIRMED overestimates: {n_confirmed}")
    print(f"  Inconclusive / bound may be loose: {n_loose_bound}")
    print(f"  Re-verification failed: {n_failed}")
    if violations:
        pct_of_planar = 100 * n_confirmed / n_planar_kappa
        pct_of_violations = 100 * n_confirmed / len(violations)
        print(f"\nConfirmed overestimates as % of ALL planar kappa labels (n={n_planar_kappa}): {pct_of_planar:.2f}%")
        print(f"Confirmed overestimates as % of flagged violations: {pct_of_violations:.1f}%")


if __name__ == "__main__":
    main()
