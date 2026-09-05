"""
Objective 3, part 2: REAL bound-tightness analysis.

Uses Observation 1 from Biedl & Mondal, "Improved Outerplanarity Bounds for
Planar Graphs" (WG 2024 / arXiv:2407.04282) -- your proposed advisor's own
paper, and the one your proposal cites as providing ground-truth labels:

    outerplanarity(G) <= min(1 + rad(G), (n+26)/6)

This is a PROVEN upper bound, not a heuristic -- unlike the earlier
"trivial_bound"/"planarization_bound" placeholders used elsewhere in this
project for bend_complexity, this one is real, citable theory, directly
from the paper your proposal is built around. Sanity-checked against every
known-answer test graph used throughout this project (cycle, wheel, K_2,3,
octahedron, grid) -- the bound holds in every case, as it must (it's a
theorem), and is genuinely informative (e.g. exact on the wheel graph).

What this script checks:
  1. How tight is the published bound in practice, on your actual dataset?
     (average gap between the bound and the true kappa value)
  2. Does the GNN's prediction ever exceed the bound? Since the bound is a
     PROVEN ceiling on the true value, a model predicting above it on many
     graphs would be a meaningful diagnostic (the model isn't respecting a
     known mathematical constraint) -- report this honestly either way.
  3. Is the GNN's prediction closer to the true value than the published
     bound is? (This is the "is there room for tighter bounds" question
     your proposal's Objective 3 asks about directly.)

Usage:
    python bound_tightness_analysis.py --labels_csv path/to/labels.csv --epochs 100
"""
from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.loader import DataLoader

from dataset_structural import LayerGuidedDataset
from train_layer_guided import LayerGuidedGIN, split_dataset


def biedl_mondal_bound(g: nx.Graph) -> float:
    """Observation 1, Biedl & Mondal (WG 2024): outerplanarity(G) <=
    min(1 + rad(G), (n+26)/6) for planar G. Proven, not heuristic."""
    n = g.number_of_nodes()
    if n <= 1:
        return 0.0
    if not nx.is_connected(g):
        return max(biedl_mondal_bound(g.subgraph(c)) for c in nx.connected_components(g))
    rad = nx.radius(g)
    return min(1 + rad, (n + 26) / 6)


def train_model(ds, device, epochs, seed=0, hidden=64, layers=3, layer_embed_dim=8, lr=1e-3):
    torch.manual_seed(seed)
    train_set, val_set, test_set = split_dataset(ds, seed=seed)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    in_dim = ds[0].x.shape[1]
    model = LayerGuidedGIN(in_dim, hidden=hidden, layers=layers, layer_embed_dim=layer_embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_mae, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch.x, batch.layer_idx, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y)
            loss.backward()
            opt.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                preds, trues = [], []
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.layer_idx, batch.edge_index, batch.batch)
                    preds.extend(out.cpu().tolist())
                    trues.extend(batch.y.cpu().tolist())
            val_mae = float(np.mean(np.abs(np.array(preds) - np.array(trues))))
            print(f"  epoch {epoch:4d} | val_mae {val_mae:.4f}")
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, test_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LayerGuidedDataset(labels_csv=args.labels_csv, target="kappa")
    if len(ds) < 20:
        raise SystemExit(f"Only {len(ds)} usable graphs -- need more for a meaningful analysis.")

    print(f"Training layer-guided GIN on kappa (seed={args.seed})...")
    model, test_set = train_model(ds, device, args.epochs, seed=args.seed)

    print(f"\n{'='*70}\nBOUND-TIGHTNESS ANALYSIS (Biedl & Mondal, WG 2024, Observation 1)\n{'='*70}")

    model.eval()
    records = []
    n_graphml_errors = 0
    with torch.no_grad():
        for d in test_set:
            x = d.x.to(device)
            edge_index = d.edge_index.to(device)
            layer_idx = d.layer_idx.to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            pred = model(x, layer_idx, edge_index, batch).item()
            true_kappa = d.y.item()

            n = x.shape[0]
            # reconstruct a plain graph from edge_index to compute the bound
            # (edge_index is symmetrized; take each edge once)
            edges = edge_index.t().cpu().numpy()
            gg = nx.Graph()
            gg.add_nodes_from(range(n))
            gg.add_edges_from([(int(u), int(v)) for u, v in edges])

            try:
                bound = biedl_mondal_bound(gg)
            except Exception:
                n_graphml_errors += 1
                continue

            records.append({
                "true_kappa": true_kappa,
                "prediction": pred,
                "bound": bound,
                "n_nodes": n,
            })

    true_k = np.array([r["true_kappa"] for r in records])
    preds = np.array([r["prediction"] for r in records])
    bounds = np.array([r["bound"] for r in records])

    bound_gap = bounds - true_k
    pred_error = np.abs(preds - true_k)
    bound_error = np.abs(bounds - true_k)  # treating the bound itself as a "prediction"

    n_violations = int((preds > bounds + 1e-6).sum())

    print(f"n_test={len(records)} (skipped {n_graphml_errors} due to reconstruction errors)")
    print(f"\n1. Bound tightness on this dataset:")
    print(f"   avg(bound - true_kappa) = {bound_gap.mean():.4f}  (0 = bound is exact; theorem guarantees >= 0)")
    print(f"   min gap = {bound_gap.min():.4f}   max gap = {bound_gap.max():.4f}")

    print(f"\n2. Does the GNN respect the proven bound?")
    print(f"   predictions exceeding the bound: {n_violations}/{len(records)} "
          f"({100*n_violations/len(records):.1f}%)")
    print(
        "   (Some violations are expected and NOT a correctness bug -- the model is a "
        "regressor with no hard constraint forcing predictions beneath the bound. A high "
        "violation rate would suggest the model isn't implicitly respecting known "
        "structure; a low rate suggests its predictions are naturally bound-consistent.)"
    )

    print(f"\n3. Is the GNN's prediction closer to truth than the bound is?")
    print(f"   GNN mean abs error:   {pred_error.mean():.4f}")
    print(f"   Bound's mean abs error (treating bound as the estimate): {bound_error.mean():.4f}")
    tighter_frac = float((pred_error < bound_error).mean())
    print(f"   GNN closer to truth than the bound: {100*tighter_frac:.1f}% of graphs")
    print(
        "\nInterpretation: this directly addresses Objective 3's 'bound tightness' question -- "
        "if the GNN is more accurate than the proven bound on most graphs, that's evidence the "
        "model has learned something the closed-form bound doesn't capture (useful signal, not "
        "a replacement for the theorem). If the bound is already tighter, that says the closed-form "
        "result is hard to beat empirically on this dataset -- also a legitimate, reportable finding."
    )


if __name__ == "__main__":
    main()
