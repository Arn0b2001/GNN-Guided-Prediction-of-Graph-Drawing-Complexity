"""
Richer node features for the "Layer-Guided GNN" from your proposal's
architecture table (Row 3: "Node embeddings initialised with
outerplanarity layer index" / inductive bias: outerplanar structure).

NOTE ON SCOPE: this implements the Layer-Guided variant, NOT the
"Structure-Aware GNN" row (Row 2: "message passing on SPQR tree + original
graph jointly"). That one needs an actual SPQR-tree construction and joint
message passing over two graphs -- networkx has no built-in SPQR-tree
implementation, so it's a separate, larger piece of work. This gives you
the tractable one first; say the word when you want the SPQR-tree version
and I'll scope that separately.

Extra node features beyond the baseline's (degree, clustering, cut-vertex
flag):
  - block_size_norm: size of the node's biconnected component / n_nodes
  - peel_layer: this node's outerplanarity peel-layer index (1 = outer
    face, 2 = next layer in, ...). 0 for non-planar graphs (undefined).
    Computed directly with the FIXED peeling algorithm from earlier in
    this conversation, not re-parsed from labels.csv's peel_layers column
    -- avoids any risk of node-id mismatches between that column and
    however read_graphml re-keys nodes on load.
"""
from __future__ import annotations

import csv

import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset

from dataset import TARGET_COLUMNS

MAX_PEEL_LAYER_EMBED = 50  # clamp ceiling for the embedding table size


def _is_planar(g: nx.Graph) -> bool:
    if g.number_of_nodes() <= 1:
        return True
    return nx.is_planar(g)


def _largest_face(embedding: "nx.PlanarEmbedding") -> set:
    faces = []
    visited = set()
    for v in embedding:
        for w in embedding[v]:
            if (v, w) in visited:
                continue
            face = embedding.traverse_face(v, w, mark_half_edges=visited)
            faces.append(set(face))
    faces.sort(key=len, reverse=True)
    return faces[0] if faces else set()


def _peel_layer(g: nx.Graph) -> set:
    removed: set = set()
    for comp_nodes in nx.connected_components(g):
        comp = g.subgraph(comp_nodes)
        if comp.number_of_nodes() <= 2:
            removed |= set(comp_nodes)
            continue
        is_p, embedding = nx.check_planarity(comp)
        if not is_p:
            removed |= set(comp_nodes)
            continue
        removed |= _largest_face(embedding)
    return removed


def outerplanarity_layers(g: nx.Graph) -> dict:
    """Same fixed face-based peeling validated earlier in this conversation."""
    if not _is_planar(g):
        return {}
    h = g.copy()
    layers: dict = {}
    layer_idx = 1
    while h.number_of_nodes() > 0:
        layer = _peel_layer(h)
        if not layer:
            break
        for v in layer:
            layers[v] = layer_idx
        h.remove_nodes_from(layer)
        layer_idx += 1
    return layers


def _block_sizes(g: nx.Graph) -> dict:
    """Map each node -> size of its LARGEST biconnected component
    (cut vertices belong to multiple blocks; take the largest for a single
    scalar feature)."""
    sizes: dict = {}
    if g.number_of_nodes() <= 1:
        return {v: 1 for v in g.nodes()}
    for comp in nx.biconnected_components(g):
        for v in comp:
            sizes[v] = max(sizes.get(v, 0), len(comp))
    for v in g.nodes():
        sizes.setdefault(v, 1)
    return sizes


def _structural_node_features(g: nx.Graph) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (continuous_feats [n,4], peel_layer_idx [n] long) for use
    with an nn.Embedding on the discrete layer index."""
    nodes = list(g.nodes())
    n = len(nodes)
    deg = dict(g.degree())
    try:
        clust = nx.clustering(g)
    except Exception:
        clust = {v: 0.0 for v in nodes}
    cut_vertices = set(nx.articulation_points(g)) if n > 2 else set()
    block_sizes = _block_sizes(g)
    peel_layers = outerplanarity_layers(g)

    cont_feats = []
    layer_idx = []
    for v in nodes:
        cont_feats.append([
            float(deg.get(v, 0)),
            float(clust.get(v, 0.0)),
            1.0 if v in cut_vertices else 0.0,
            float(block_sizes.get(v, 1)) / max(n, 1),
        ])
        L = peel_layers.get(v, 0)  # 0 = non-planar / undefined
        layer_idx.append(min(L, MAX_PEEL_LAYER_EMBED - 1))

    return (
        torch.tensor(cont_feats, dtype=torch.float),
        torch.tensor(layer_idx, dtype=torch.long),
    )


def _to_pyg_data(g: nx.Graph, y_value: float) -> Data | None:
    if g.number_of_nodes() < 2:
        return None
    nodes = list(g.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    if not edges:
        return None
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    x_cont, layer_idx = _structural_node_features(g)
    y = torch.tensor([y_value], dtype=torch.float)
    return Data(x=x_cont, layer_idx=layer_idx, edge_index=edge_index, y=y)


class LayerGuidedDataset(InMemoryDataset):
    """Same filtering/loading logic as GraphDrawingComplexityDataset, but
    with structural node features (block size, cut vertex, peel layer)
    instead of the baseline's plain degree/clustering."""

    def __init__(self, labels_csv: str, target: str = "kappa",
                 include_trivial_bounds: bool = False, root: str | None = None, transform=None):
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
        skipped = {"excluded_type": 0, "missing_value": 0, "unreadable": 0}

        with open(self.labels_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get(type_col, "") in exclude_types:
                    skipped["excluded_type"] += 1
                    continue
                raw_val = row.get(value_col, "")
                if raw_val in ("", None):
                    skipped["missing_value"] += 1
                    continue
                try:
                    y_value = float(raw_val)
                except ValueError:
                    skipped["missing_value"] += 1
                    continue

                try:
                    g = nx.read_graphml(row["path"])
                    g = nx.Graph(g)
                except Exception:
                    skipped["unreadable"] += 1
                    continue

                d = _to_pyg_data(g, y_value)
                if d is None:
                    skipped["unreadable"] += 1
                    continue
                d.graph_id = row.get("graph_id", "")
                data_list.append(d)

        print(f"[{self.target}] loaded {len(data_list)} graphs | skipped: "
              f"{skipped['excluded_type']} excluded-label-type, "
              f"{skipped['missing_value']} missing-value, "
              f"{skipped['unreadable']} unreadable-graph")
        return self.collate(data_list)


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "test_data/labels.csv"
    ds = LayerGuidedDataset(labels_csv=csv_path, target="kappa")
    print(f"dataset size: {len(ds)}")
    print(ds[0])
