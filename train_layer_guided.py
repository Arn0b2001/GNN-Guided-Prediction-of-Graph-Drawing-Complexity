"""
Layer-Guided GNN -- Row 3 of your proposal's architecture table.
Inductive bias: node embeddings initialised with outerplanarity layer index.

Compares directly against the baseline GIN/GCN and the linear-on-size
baseline, using the exact same split logic and evaluation metrics as
train_baseline.py, so results are apples-to-apples.

Usage:
    python train_layer_guided.py --labels_csv path/to/labels.csv --target kappa --epochs 100
"""
from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from dataset_structural import LayerGuidedDataset, MAX_PEEL_LAYER_EMBED

RESULTS_CSV = "results_log.csv"


class LayerGuidedGIN(torch.nn.Module):
    """Discrete peel-layer index goes through its own embedding table
    (this is the literal 'node embeddings initialised with outerplanarity
    layer index' bias from the proposal), concatenated with the continuous
    structural features (degree, clustering, cut-vertex flag, block size),
    then standard GIN message passing on top."""

    def __init__(self, in_dim_continuous: int, hidden: int = 64, layers: int = 3,
                 layer_embed_dim: int = 8):
        super().__init__()
        self.layer_embed = torch.nn.Embedding(MAX_PEEL_LAYER_EMBED, layer_embed_dim)
        in_dim = in_dim_continuous + layer_embed_dim

        self.convs = torch.nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden)
            )
            self.convs.append(GINConv(mlp))
            d = hidden
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, layer_idx, edge_index, batch):
        layer_emb = self.layer_embed(layer_idx)
        x = torch.cat([x, layer_emb], dim=-1)
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
    return ([ds[i] for i in perm[:n_train]],
            [ds[i] for i in perm[n_train:n_train + n_val]],
            [ds[i] for i in perm[n_train + n_val:]])


def _graph_size_features(data_list) -> np.ndarray:
    feats = []
    for d in data_list:
        n_nodes = d.num_nodes
        n_edges = d.edge_index.shape[1] // 2
        feats.append([n_nodes, n_edges])
    return np.array(feats, dtype=float)


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
        batch = batch.to(device)
        out = model(batch.x, batch.layer_idx, batch.edge_index, batch.batch)
        preds.extend(out.cpu().tolist())
        trues.extend(batch.y.cpu().tolist())
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
    ap.add_argument("--layer_embed_dim", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LayerGuidedDataset(labels_csv=args.labels_csv, target=args.target)
    if len(ds) < 10:
        raise SystemExit(f"Only {len(ds)} usable graphs for target={args.target}.")

    train_set, val_set, test_set = split_dataset(ds, seed=args.seed)
    baselines = trivial_baselines(train_set, test_set)
    print(f"\n[baselines | target={args.target}] mean_mae={baselines['mean_mae']:.4f}  "
          f"linear_mae={baselines['linear_mae']:.4f}  linear_spearman={baselines['linear_spearman']:.4f}\n")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    in_dim = ds[0].x.shape[1]
    model = LayerGuidedGIN(in_dim, hidden=args.hidden, layers=args.layers,
                            layer_embed_dim=args.layer_embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.layer_idx, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs

        if epoch % 10 == 0 or epoch == args.epochs:
            val_mae, val_rho = evaluate(model, val_loader, device)
            print(f"epoch {epoch:4d} | train_mse {total_loss/len(train_set):.4f} | "
                  f"val_mae {val_mae:.4f} | val_spearman {val_rho:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae, test_rho = evaluate(model, test_loader, device)

    beats_linear = test_mae < baselines["linear_mae"]
    print(f"\nFINAL [layer_guided_gin/{args.target}] test_mae={test_mae:.4f} test_spearman={test_rho:.4f}")
    print(f"  vs. linear-on-size baseline: mae={baselines['linear_mae']:.4f} "
          f"spearman={baselines['linear_spearman']:.4f}  "
          f"-> {'BEATS' if beats_linear else 'DOES NOT BEAT'} the size-only baseline on MAE")
    print(f"  (n_train={len(train_set)}, n_val={len(val_set)}, n_test={len(test_set)})")

    log_result({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "layer_guided_gin",
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
