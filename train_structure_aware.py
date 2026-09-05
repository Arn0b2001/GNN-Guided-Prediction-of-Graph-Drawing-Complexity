"""
Structure-Aware GNN, using the BLOCK-CUT TREE as the auxiliary decomposition
graph (Row 2 of your proposal's architecture table calls for the SPQR tree
specifically; see the note below on why this uses block-cut tree instead).

WHY BLOCK-CUT TREE, NOT SPQR TREE:
A real SPQR-tree needs a triconnectivity decomposition algorithm
(Hopcroft-Tarjan style), which is nontrivial to hand-roll correctly --
exactly the kind of "looks plausible, subtly wrong" risk this whole
project has been hunting down in the outerplanarity/bends code. The one
real tested library found (ogdf-python, official bindings to the OGDF C++
library) needs a heavy C++ toolchain that isn't practical to get running
reliably here. Your proposal text itself lists three valid structural
decompositions -- "SPQR trees, block-cut trees, outerplanarity layers" --
so this uses block-cut tree, which networkx supports natively and
correctly via biconnected_components/articulation_points. If you get
ogdf-python (or another verified SPQR library) working in your own
environment later, the same joint-message-passing pattern here can be
adapted to a real SPQR tree.

ARCHITECTURE:
  1. GIN message passing on the original graph.
  2. Separately build the block-cut tree (block nodes + cut-vertex nodes,
     connected by membership edges) and run GIN message passing on it.
  3. Each original node's embedding is fused with the embedding of the
     block-cut-tree node representing ITS block membership (a node
     belonging to multiple blocks, i.e. a cut vertex, uses its own
     cut-vertex node's embedding).
  4. Pool and predict.
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from dataset import TARGET_COLUMNS

RESULTS_CSV = "results_log.csv"


# ---------------------------------------------------------------- dataset --

def _block_cut_tree(g: nx.Graph):
    """Returns (bc_tree, node_to_bc_node) where node_to_bc_node maps each
    ORIGINAL node -> the block-cut-tree node representing it (its own
    cut-vertex node if it's a cut vertex, else its block's node)."""
    bc = nx.Graph()
    blocks = list(nx.biconnected_components(g))
    cut_vertices = set(nx.articulation_points(g)) if g.number_of_nodes() > 2 else set()

    block_ids = [f"B{i}" for i in range(len(blocks))]
    for bid in block_ids:
        bc.add_node(bid)
    for cv in cut_vertices:
        bc.add_node(cv)

    node_to_bc: dict = {}
    for bid, block in zip(block_ids, blocks):
        for v in block:
            if v in cut_vertices:
                bc.add_edge(cv := v, bid)
                node_to_bc.setdefault(v, v)  # cut vertex maps to itself
            else:
                node_to_bc[v] = bid

    for v in g.nodes():
        node_to_bc.setdefault(v, block_ids[0] if block_ids else v)
        if v not in bc:
            bc.add_node(v)

    if bc.number_of_nodes() == 0:
        bc.add_node("B0")

    return bc, node_to_bc


def _base_node_features(g: nx.Graph) -> torch.Tensor:
    deg = dict(g.degree())
    try:
        clust = nx.clustering(g)
    except Exception:
        clust = {v: 0.0 for v in g.nodes()}
    cut_vertices = set(nx.articulation_points(g)) if g.number_of_nodes() > 2 else set()
    feats = []
    for v in g.nodes():
        feats.append([
            float(deg.get(v, 0)),
            float(clust.get(v, 0.0)),
            1.0 if v in cut_vertices else 0.0,
        ])
    return torch.tensor(feats, dtype=torch.float)


def _bc_node_features(bc: nx.Graph, blocks_sizes: dict) -> torch.Tensor:
    """Simple structural features for block-cut-tree nodes: block size (0
    for cut-vertex nodes) and degree in the bc-tree (how many blocks a cut
    vertex touches, or trivially 1-ish for block nodes)."""
    feats = []
    for n in bc.nodes():
        size = blocks_sizes.get(n, 0)
        deg = bc.degree(n)
        is_block = 1.0 if str(n).startswith("B") else 0.0
        feats.append([float(size), float(deg), is_block])
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
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x = _base_node_features(g)

    bc, node_to_bc = _block_cut_tree(g)
    bc_nodes = list(bc.nodes())
    bc_idx = {b: i for i, b in enumerate(bc_nodes)}

    block_sizes: dict = {}
    for comp in nx.biconnected_components(g):
        bid = None
        # find which bc node id this component maps to via any member's mapping
        for v in comp:
            candidate = node_to_bc.get(v)
            if candidate is not None and str(candidate).startswith("B"):
                bid = candidate
                break
        if bid is not None:
            block_sizes[bid] = len(comp)

    bc_x = _bc_node_features(bc, block_sizes)
    bc_edges = [(bc_idx[u], bc_idx[v]) for u, v in bc.edges()]
    if bc_edges:
        bc_edge_index = torch.tensor(bc_edges, dtype=torch.long).t().contiguous()
        bc_edge_index = torch.cat([bc_edge_index, bc_edge_index.flip(0)], dim=1)
    else:
        bc_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # map each original node -> its bc-tree node index, for the fusion step
    node_to_bc_idx = torch.tensor(
        [bc_idx[node_to_bc[v]] for v in nodes], dtype=torch.long
    )

    y = torch.tensor([y_value], dtype=torch.float)
    d = Data(x=x, edge_index=edge_index, y=y)
    d.bc_x = bc_x
    d.bc_edge_index = bc_edge_index
    d.node_to_bc_idx = node_to_bc_idx
    d.num_bc_nodes = bc_x.shape[0]
    return d


class StructureAwareDataset(InMemoryDataset):
    def __init__(self, labels_csv: str, target: str = "kappa",
                 include_trivial_bounds: bool = False, root: str | None = None, transform=None):
        if target not in TARGET_COLUMNS:
            raise ValueError(f"target must be one of {list(TARGET_COLUMNS)}")
        self.labels_csv = labels_csv
        self.target = target
        self.include_trivial_bounds = include_trivial_bounds
        super().__init__(root or ".", transform)
        self.data_list = self._build()

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
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ------------------------------------------------------------------ model --

class StructureAwareGIN(torch.nn.Module):
    def __init__(self, in_dim: int, bc_in_dim: int, hidden: int = 64, layers: int = 3):
        super().__init__()

        def make_convs(d):
            convs = torch.nn.ModuleList()
            for _ in range(layers):
                mlp = torch.nn.Sequential(
                    torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
                )
                convs.append(GINConv(mlp))
                d = hidden
            return convs

        self.main_convs = make_convs(in_dim)
        self.bc_convs = make_convs(bc_in_dim)
        self.fuse = torch.nn.Linear(hidden * 2, hidden)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch, bc_x, bc_edge_index, bc_batch, node_to_bc_idx_global):
        h = x
        for conv in self.main_convs:
            h = F.relu(conv(h, edge_index))

        hb = bc_x
        for conv in self.bc_convs:
            hb = F.relu(conv(hb, bc_edge_index))

        # gather the bc-tree embedding for each original node's block/cut-vertex
        fused_bc = hb[node_to_bc_idx_global]
        h = F.relu(self.fuse(torch.cat([h, fused_bc], dim=-1)))

        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)


# ----------------------------------------------------------------- batching --

def collate_structure_aware(batch_list):
    """Manual batching -- these Data objects carry a second graph (the
    block-cut tree) that PyG's default collate doesn't know how to offset,
    so node indices into it are combined by hand here."""
    xs, edge_indices, batch_idx = [], [], []
    bc_xs, bc_edge_indices, bc_batch_idx = [], [], []
    node_to_bc_global = []
    ys, graph_ids = [], []

    node_offset = 0
    bc_offset = 0
    for i, d in enumerate(batch_list):
        n = d.x.shape[0]
        xs.append(d.x)
        edge_indices.append(d.edge_index + node_offset)
        batch_idx.append(torch.full((n,), i, dtype=torch.long))

        bc_n = d.bc_x.shape[0]
        bc_xs.append(d.bc_x)
        bc_edge_indices.append(d.bc_edge_index + bc_offset)
        bc_batch_idx.append(torch.full((bc_n,), i, dtype=torch.long))

        node_to_bc_global.append(d.node_to_bc_idx + bc_offset)
        ys.append(d.y)
        graph_ids.append(getattr(d, "graph_id", ""))

        node_offset += n
        bc_offset += bc_n

    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "batch": torch.cat(batch_idx, dim=0),
        "bc_x": torch.cat(bc_xs, dim=0),
        "bc_edge_index": torch.cat(bc_edge_indices, dim=1),
        "bc_batch": torch.cat(bc_batch_idx, dim=0),
        "node_to_bc_idx_global": torch.cat(node_to_bc_global, dim=0),
        "y": torch.cat(ys, dim=0),
    }


class SimpleLoader:
    """Thin batching loader (avoids torch_geometric.loader.DataLoader,
    which doesn't know how to combine the auxiliary block-cut-tree graph)."""

    def __init__(self, data_list, batch_size, shuffle=False):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        idx = list(range(len(self.data_list)))
        if self.shuffle:
            import random
            random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            chunk = [self.data_list[j] for j in idx[i:i + self.batch_size]]
            yield collate_structure_aware(chunk)

    def __len__(self):
        return (len(self.data_list) + self.batch_size - 1) // self.batch_size


# ------------------------------------------------------------------- train --

def split_dataset(data_list, train_frac=0.7, val_frac=0.15, seed=0):
    n = len(data_list)
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return ([data_list[i] for i in perm[:n_train]],
            [data_list[i] for i in perm[n_train:n_train + n_val]],
            [data_list[i] for i in perm[n_train + n_val:]])


def _graph_size_features(data_list) -> np.ndarray:
    return np.array([[d.x.shape[0], d.edge_index.shape[1] // 2] for d in data_list], dtype=float)


def trivial_baselines(train_set, test_set) -> dict:
    y_train = np.array([d.y.item() for d in train_set])
    y_test = np.array([d.y.item() for d in test_set])
    mean_mae = float(np.mean(np.abs(y_test - y_train.mean())))
    X_train = _graph_size_features(train_set)
    X_test = _graph_size_features(test_set)
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_mae = float(np.mean(np.abs(lr_pred - y_test)))
    lr_rho = spearmanr(lr_pred, y_test).correlation if len(y_test) > 1 else float("nan")
    return {"mean_mae": mean_mae, "linear_mae": lr_mae, "linear_spearman": lr_rho}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        out = model(
            batch["x"].to(device), batch["edge_index"].to(device), batch["batch"].to(device),
            batch["bc_x"].to(device), batch["bc_edge_index"].to(device), batch["bc_batch"].to(device),
            batch["node_to_bc_idx_global"].to(device),
        )
        preds.extend(out.cpu().tolist())
        trues.extend(batch["y"].tolist())
    mae = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
    rho = spearmanr(preds, trues).correlation if len(preds) > 1 else float("nan")
    return mae, rho


def log_result(row: dict):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--target", choices=["kappa", "min_layers", "bend_complexity"], default="kappa")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = StructureAwareDataset(labels_csv=args.labels_csv, target=args.target)
    if len(ds) < 10:
        raise SystemExit(f"Only {len(ds)} usable graphs for target={args.target}.")

    train_set, val_set, test_set = split_dataset(ds.data_list, seed=args.seed)
    baselines = trivial_baselines(train_set, test_set)
    print(f"\n[baselines | target={args.target}] mean_mae={baselines['mean_mae']:.4f}  "
          f"linear_mae={baselines['linear_mae']:.4f}  linear_spearman={baselines['linear_spearman']:.4f}\n")

    train_loader = SimpleLoader(train_set, args.batch_size, shuffle=True)
    val_loader = SimpleLoader(val_set, args.batch_size)
    test_loader = SimpleLoader(test_set, args.batch_size)

    in_dim = ds.data_list[0].x.shape[1]
    bc_in_dim = ds.data_list[0].bc_x.shape[1]
    model = StructureAwareGIN(in_dim, bc_in_dim, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_graphs = 0
        for batch in train_loader:
            opt.zero_grad()
            out = model(
                batch["x"].to(device), batch["edge_index"].to(device), batch["batch"].to(device),
                batch["bc_x"].to(device), batch["bc_edge_index"].to(device), batch["bc_batch"].to(device),
                batch["node_to_bc_idx_global"].to(device),
            )
            y = batch["y"].to(device)
            loss = F.mse_loss(out, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * y.shape[0]
            n_graphs += y.shape[0]

        if epoch % 10 == 0 or epoch == args.epochs:
            val_mae, val_rho = evaluate(model, val_loader, device)
            print(f"epoch {epoch:4d} | train_mse {total_loss/n_graphs:.4f} | "
                  f"val_mae {val_mae:.4f} | val_spearman {val_rho:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae, test_rho = evaluate(model, test_loader, device)

    beats_linear = test_mae < baselines["linear_mae"]
    print(f"\nFINAL [structure_aware_gin/{args.target}] test_mae={test_mae:.4f} test_spearman={test_rho:.4f}")
    print(f"  vs. linear-on-size baseline: mae={baselines['linear_mae']:.4f} "
          f"spearman={baselines['linear_spearman']:.4f}  "
          f"-> {'BEATS' if beats_linear else 'DOES NOT BEAT'} the size-only baseline on MAE")
    print(f"  (n_train={len(train_set)}, n_val={len(val_set)}, n_test={len(test_set)})")

    log_result({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "structure_aware_gin_blockcut",
        "target": args.target,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "layers": args.layers,
        "lr": args.lr,
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "mean_baseline_mae": baselines["mean_mae"],
        "linear_baseline_mae": baselines["linear_mae"],
        "linear_baseline_spearman": baselines["linear_spearman"],
        "test_mae": test_mae,
        "test_spearman": test_rho,
        "beats_linear_baseline_mae": beats_linear,
    })
    print(f"  logged -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
