"""
Baseline GCN / GIN training for graph drawing complexity regression.

This is the "Baseline GCN/GIN — None (ablation)" row from Objective 2's
architecture table: no structural inductive bias (no SPQR-tree, no
block-cut-tree, no outerplanarity-layer conditioning). Structure-aware
variants should be compared against this, not built as a replacement for it.

Usage:
    python train_baseline.py --labels_csv path/to/labels.csv --target kappa --model gin
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool

from dataset import GraphDrawingComplexityDataset


class GCNRegressor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(GCNConv(d, hidden))
            d = hidden
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


class GINRegressor(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
            )
            self.convs.append(GINConv(mlp))
            d = hidden
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


def split_dataset(ds, train_frac=0.7, val_frac=0.15, seed=0):
    n = len(ds)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return [ds[i] for i in train_idx], [ds[i] for i in val_idx], [ds[i] for i in test_idx], train_idx, val_idx, test_idx


def _graph_size_features(dataset):
    """Extract graph size features (n_nodes, n_edges) from dataset."""
    features = []
    for data in dataset:
        n_nodes = data.num_nodes
        n_edges = data.edge_index.shape[1] // 2  # undirected, so divide by 2
        features.append([n_nodes, n_edges])
    return np.array(features)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        preds.extend(out.cpu().tolist())
        trues.extend(batch.y.cpu().tolist())
    mae = sum(abs(p - t) for p, t in zip(preds, trues)) / max(len(preds), 1)
    rho = spearmanr(preds, trues).correlation if len(preds) > 1 else float("nan")
    return mae, rho


def get_graph_features(labels_csv, target):
    """Extract graph size features and target values from labels.csv."""
    value_col, type_col, exclude_types = {
        "kappa": ("kappa", "kappa_label_type", {"not_applicable"}),
        "min_layers": ("min_layers", "layers_label_type", set()),
        "bend_complexity": ("bend_complexity", "bends_label_type", set()),
    }[target]
    
    n_nodes, n_edges, targets = [], [], []
    
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_type = row.get(type_col, "")
            if label_type in exclude_types:
                continue
            raw_val = row.get(value_col, "")
            if raw_val in ("", None):
                continue
            try:
                y_value = float(raw_val)
            except ValueError:
                continue
            
            n_nodes.append(int(row["n_nodes"]))
            n_edges.append(int(row["n_edges"]))
            targets.append(y_value)
    
    return np.array(n_nodes), np.array(n_edges), np.array(targets)


def evaluate_trivial_baselines(labels_csv, target, train_idx, val_idx, test_idx):
    """Evaluate mean predictor and linear regression baselines."""
    n_nodes, n_edges, targets = get_graph_features(labels_csv, target)
    
    # Split according to the same indices
    train_nodes, train_targets = n_nodes[train_idx], targets[train_idx]
    val_nodes, val_targets = n_nodes[val_idx], targets[val_idx]
    test_nodes, test_targets = n_nodes[test_idx], targets[test_idx]
    
    train_edges, val_edges, test_edges = n_edges[train_idx], n_edges[val_idx], n_edges[test_idx]
    
    # Mean predictor baseline
    mean_pred = np.mean(train_targets)
    val_mae_mean = np.mean(np.abs(val_targets - mean_pred))
    test_mae_mean = np.mean(np.abs(test_targets - mean_pred))
    # Spearman is undefined for constant predictions
    val_rho_mean = 0.0  # No correlation with constant prediction
    test_rho_mean = 0.0
    
    # Linear regression on size (n_nodes + n_edges)
    X_train = np.column_stack([train_nodes, train_edges])
    X_val = np.column_stack([val_nodes, val_edges])
    X_test = np.column_stack([test_nodes, test_edges])
    
    lr = LinearRegression()
    lr.fit(X_train, train_targets)
    
    val_pred_lr = lr.predict(X_val)
    test_pred_lr = lr.predict(X_test)
    
    val_mae_lr = np.mean(np.abs(val_targets - val_pred_lr))
    test_mae_lr = np.mean(np.abs(test_targets - test_pred_lr))
    val_rho_lr = spearmanr(val_pred_lr, val_targets).correlation
    test_rho_lr = spearmanr(test_pred_lr, test_targets).correlation
    
    # Check correlation between target and graph size
    rho_nodes = spearmanr(n_nodes, targets).correlation
    rho_edges = spearmanr(n_edges, targets).correlation
    
    return {
        "mean": (val_mae_mean, val_rho_mean, test_mae_mean, test_rho_mean),
        "linear": (val_mae_lr, val_rho_lr, test_mae_lr, test_rho_lr),
        "size_correlation": (rho_nodes, rho_edges)
    }


def print_baseline_results(baseline_results, target):
    """Print baseline comparison results."""
    print(f"\nBASELINE COMPARISON for {target}:")
    print("-" * 70)
    
    mean_val_mae, mean_val_rho, mean_test_mae, mean_test_rho = baseline_results["mean"]
    lr_val_mae, lr_val_rho, lr_test_mae, lr_test_rho = baseline_results["linear"]
    rho_nodes, rho_edges = baseline_results["size_correlation"]
    
    print(f"Target vs graph size correlation:")
    print(f"  Spearman(target, n_nodes): {rho_nodes:.4f}")
    print(f"  Spearman(target, n_edges): {rho_edges:.4f}")
    
    print(f"\nMean predictor baseline:")
    print(f"  Val: MAE={mean_val_mae:.4f}, Spearman={mean_val_rho:.4f}")
    print(f"  Test: MAE={mean_test_mae:.4f}, Spearman={mean_test_rho:.4f}")
    
    print(f"\nLinear regression on (n_nodes, n_edges):")
    print(f"  Val: MAE={lr_val_mae:.4f}, Spearman={lr_val_rho:.4f}")
    print(f"  Test: MAE={lr_test_mae:.4f}, Spearman={lr_test_rho:.4f}")
    print("-" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--target", choices=["kappa", "min_layers", "bend_complexity"], default="kappa")
    ap.add_argument("--model", choices=["gcn", "gin"], default="gin")
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

    ds = GraphDrawingComplexityDataset(labels_csv=args.labels_csv, target=args.target)
    if len(ds) < 10:
        raise SystemExit(
            f"Only {len(ds)} usable graphs for target={args.target}. "
            "Need more data (or a less restrictive label filter) before training."
        )

    train_set, val_set, test_set, train_idx, val_idx, test_idx = split_dataset(ds)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    # Evaluate trivial baselines
    baseline_results = evaluate_trivial_baselines(args.labels_csv, args.target, train_idx, val_idx, test_idx)
    print_baseline_results(baseline_results, args.target)

    in_dim = ds[0].x.shape[1]
    model_cls = GCNRegressor if args.model == "gcn" else GINRegressor
    model = model_cls(in_dim, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 10 == 0 or epoch == args.epochs:
            val_mae, val_rho = evaluate(model, val_loader, device)
            train_mse = total_loss / len(train_set)
            print(f"epoch {epoch:4d} | train_mse {train_mse:.4f} | val_mae {val_mae:.4f} | val_spearman {val_rho:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae, test_rho = evaluate(model, test_loader, device)
    
    # Compare with baselines
    _, _, mean_test_mae, mean_test_rho = baseline_results["mean"]
    _, _, lr_test_mae, lr_test_rho = baseline_results["linear"]
    
    print(f"\nFINAL RESULTS [{args.model}/{args.target}]:")
    print(f"  GNN:          test_mae={test_mae:.4f}, test_spearman={test_rho:.4f}")
    print(f"  Mean baseline: test_mae={mean_test_mae:.4f}, test_spearman={mean_test_rho:.4f}")
    print(f"  Linear baseline: test_mae={lr_test_mae:.4f}, test_spearman={lr_test_rho:.4f}")
    print(f"  Improvement over mean: MAE={(mean_test_mae - test_mae):.4f}, Spearman={(test_rho - mean_test_rho):.4f}")
    print(f"  Improvement over linear: MAE={(lr_test_mae - test_mae):.4f}, Spearman={(test_rho - lr_test_rho):.4f}")
    print(f"  (n_train={len(train_set)}, n_val={len(val_set)}, n_test={len(test_set)})")


if __name__ == "__main__":
    main()
