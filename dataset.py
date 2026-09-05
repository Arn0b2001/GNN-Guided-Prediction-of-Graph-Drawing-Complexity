"""
Convert the graph drawing complexity benchmark (labels.csv + GraphML files)
into a PyTorch Geometric dataset.

Usage:
    from dataset import GraphDrawingComplexityDataset
    ds = GraphDrawingComplexityDataset(labels_csv="labels.csv", target="kappa")
    ds[0]  # -> torch_geometric.data.Data

Design notes:
  - Node features are structural, not learned embeddings: degree, clustering
    coefficient, and a one-hot "is cut vertex" flag. This is a deliberately
    simple baseline feature set -- Objective 2's structure-aware variants
    (SPQR-tree / block-cut-tree conditioning) should extend this, not
    replace it wholesale, so the baseline GCN/GIN stays a fair ablation.
  - `target` selects which label column becomes `data.y`. Only rows where
    that target is a real value are included -- e.g. selecting "kappa"
    automatically drops the 4,719 `not_applicable` (non-planar) rows,
    since kappa is undefined there. This must be a per-target filter, not
    a single global filter, since kappa/layers/bends have different
    applicability.
  - Rows with label_type == "trivial_bound" are EXCLUDED by default when
    target="bend_complexity", since those labels (currently == n_edges for
    non-planar graphs) carry no real drawing-theoretic signal, per the
    earlier dataset audit. Set include_trivial_bounds=True to override.
"""
from __future__ import annotations

import csv
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset

TARGET_COLUMNS = {
    "kappa": ("kappa", "kappa_label_type", {"not_applicable"}),
    "min_layers": ("min_layers", "layers_label_type", set()),
    "bend_complexity": ("bend_complexity", "bends_label_type", set()),
}


def _node_features(g: nx.Graph) -> torch.Tensor:
    """Simple structural node features: degree, clustering coeff, cut-vertex flag."""
    nodes = list(g.nodes())
    deg = dict(g.degree())
    try:
        clust = nx.clustering(g)
    except Exception:
        clust = {v: 0.0 for v in nodes}
    cut_vertices = set(nx.articulation_points(g)) if g.number_of_nodes() > 2 else set()

    feats = []
    for v in nodes:
        feats.append([
            float(deg.get(v, 0)),
            float(clust.get(v, 0.0)),
            1.0 if v in cut_vertices else 0.0,
        ])
    return torch.tensor(feats, dtype=torch.float)


def _to_pyg_data(g: nx.Graph, y_value: float) -> Data | None:
    if g.number_of_nodes() < 2:
        return None
    nodes = list(g.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # make undirected explicit for message passing
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    x = _node_features(g)
    y = torch.tensor([y_value], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)


class GraphDrawingComplexityDataset(InMemoryDataset):
    def __init__(
        self,
        labels_csv: str,
        target: str = "kappa",
        include_trivial_bounds: bool = False,
        root: str | None = None,
        transform=None,
    ):
        if target not in TARGET_COLUMNS:
            raise ValueError(f"target must be one of {list(TARGET_COLUMNS)}")
        self.labels_csv = labels_csv
        self.target = target
        self.include_trivial_bounds = include_trivial_bounds
        super().__init__(root or ".", transform)
        self.data, self.slices = self._build()

    def _build(self):
        value_col, type_col, exclude_types = TARGET_COLUMNS[self.target]
        exclude_types = set(exclude_types)
        if self.target == "bend_complexity" and not self.include_trivial_bounds:
            exclude_types.add("trivial_bound")

        data_list = []
        skipped_missing_value = 0
        skipped_excluded_type = 0
        skipped_unreadable = 0

        with open(self.labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label_type = row.get(type_col, "")
                if label_type in exclude_types:
                    skipped_excluded_type += 1
                    continue
                raw_val = row.get(value_col, "")
                if raw_val in ("", None):
                    skipped_missing_value += 1
                    continue
                try:
                    y_value = float(raw_val)
                except ValueError:
                    skipped_missing_value += 1
                    continue

                path = row["path"]
                try:
                    g = nx.read_graphml(path)
                    g = nx.Graph(g)  # drop direction/multi-edge info if present
                except Exception:
                    skipped_unreadable += 1
                    continue

                d = _to_pyg_data(g, y_value)
                if d is None:
                    skipped_unreadable += 1
                    continue
                d.graph_id = row.get("graph_id", "")
                data_list.append(d)

        print(
            f"[{self.target}] loaded {len(data_list)} graphs | "
            f"skipped: {skipped_excluded_type} excluded-label-type, "
            f"{skipped_missing_value} missing-value, "
            f"{skipped_unreadable} unreadable-graph"
        )
        return self.collate(data_list)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "test_data/labels.csv"
    for target in ["kappa", "min_layers", "bend_complexity"]:
        ds = GraphDrawingComplexityDataset(labels_csv=csv_path, target=target)
        print(f"  -> dataset size: {len(ds)}, example: {ds[0]}")
