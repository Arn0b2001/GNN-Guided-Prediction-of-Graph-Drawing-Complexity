"""
Stratified evaluation for bend_complexity.

The pooled MAE/Spearman for bend_complexity are misleading: 34.6% of true
values are exactly 0 (outerplanar graphs, kappa=1), and a model that
predicts "small" everywhere gets a great pooled MAE while saying nothing
about whether it can rank the harder non-zero tail. This script re-scores
predictions split by true_value == 0 vs > 0, for both the GNN and the
linear-on-size baseline, so the two regimes aren't blended into one
misleading number.

Usage:
    python eval_bend_complexity_stratified.py --labels_csv path/to/labels.csv --model gin
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from torch_geometric.loader import DataLoader

from dataset import GraphDrawingComplexityDataset
from train_baseline import (
    GCNRegressor,
    GINRegressor,
    _graph_size_features,
    split_dataset,
)


def stratified_scores(preds: np.ndarray, trues: np.ndarray, label: str) -> None:
    zero_mask = trues == 0
    nonzero_mask = ~zero_mask

    def _report(mask, name):
        n = mask.sum()
        if n == 0:
            print(f"    {name}: n=0 (no examples)")
            return
        mae = float(np.mean(np.abs(preds[mask] - trues[mask])))
        rho = spearmanr(preds[mask], trues[mask]).correlation if n > 1 and trues[mask].std() > 0 else float("nan")
        print(f"    {name}: n={n:5d}  mae={mae:8.4f}  spearman={rho if rho==rho else float('nan'):.4f}")

    print(f"  [{label}]")
    _report(np.ones_like(zero_mask, dtype=bool), "all       ")
    _report(zero_mask, "zero only ")
    _report(nonzero_mask, "nonzero   ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--model", choices=["gcn", "gin"], default="gin")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GraphDrawingComplexityDataset(labels_csv=args.labels_csv, target="bend_complexity")
    train_set, val_set, test_set, _, _, _ = split_dataset(ds)

    y_test = np.array([d.y.item() for d in test_set])
    frac_zero = float((y_test == 0).mean())
    print(f"\ntest set: n={len(y_test)}, frac_zero={frac_zero:.3f}, "
          f"median={np.median(y_test):.2f}, mean={y_test.mean():.2f}\n")

    # --- linear-on-size baseline, stratified ---
    y_train = np.array([d.y.item() for d in train_set])
    X_train = _graph_size_features(train_set)
    X_test = _graph_size_features(test_set)
    lr = LinearRegression().fit(X_train, y_train)
    lin_preds = lr.predict(X_test)
    stratified_scores(lin_preds, y_test, "linear-on-size baseline")

    # --- GNN, stratified ---
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    in_dim = ds[0].x.shape[1]
    model_cls = GCNRegressor if args.model == "gcn" else GINRegressor
    model = model_cls(in_dim, hidden=args.hidden, layers=args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mae, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            opt.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                v_preds, v_trues = [], []
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    v_preds.extend(out.cpu().tolist())
                    v_trues.extend(batch.y.cpu().tolist())
            val_mae = float(np.mean(np.abs(np.array(v_preds) - np.array(v_trues))))
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    gnn_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)
            gnn_preds.extend(out.cpu().tolist())
    gnn_preds = np.array(gnn_preds)

    print()
    stratified_scores(gnn_preds, y_test, f"{args.model.upper()} (GNN)")

    print(
        "\nRead this as: 'zero only' rows are the easy outerplanar cases (kappa=1 -> "
        "bend_complexity=0 by definition, no drawing-complexity signal needed). "
        "'nonzero' is the real test of whether the model learned anything beyond that. "
        "If GNN's nonzero MAE/Spearman aren't clearly better than the linear baseline's "
        "nonzero numbers, the pooled result from the earlier run was misleading."
    )


if __name__ == "__main__":
    main()
