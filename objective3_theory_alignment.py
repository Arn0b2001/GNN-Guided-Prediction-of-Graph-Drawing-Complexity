"""
Objective 3: Theory-Learning Interface.

Checks whether the layer-guided GNN's learned predictions align with real
outerplanarity theory, in two ways:

1. FEATURE ATTRIBUTION (GNNExplainer): for each test graph, get a per-node
   importance score from the trained model, then correlate it against the
   TRUE peel-layer index of each node (1 = outer face, computed with the
   same validated peeling algorithm used throughout this project). If the
   model has genuinely learned outerplanarity structure, importance should
   correlate with peel depth in some consistent way across graphs -- this
   script reports the correlation per graph and in aggregate, without
   assuming in advance which direction it "should" point.

2. ERROR VS STRUCTURAL COMPLEXITY: rather than compare against a
   theoretical bound formula (which would need to be pulled from the WG
   2024 paper's actual results, not guessed at), this checks whether
   prediction error grows with structural complexity (n_blocks,
   n_cut_vertices) -- a defensible, implementable proxy for "does the
   model's understanding hold up on harder graphs," using only quantities
   this codebase can already compute correctly.

Usage:
    python objective3_theory_alignment.py --labels_csv path/to/labels.csv --epochs 100
"""
from __future__ import annotations

import argparse

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
from torch_geometric.loader import DataLoader

from dataset_structural import (
    LayerGuidedDataset,
    MAX_PEEL_LAYER_EMBED,
    outerplanarity_layers,
)
from train_layer_guided import LayerGuidedGIN, split_dataset


class _WrappedModel(torch.nn.Module):
    """GNNExplainer expects forward(x, edge_index, batch=...) with x being
    the ONLY node feature tensor it perturbs. Our model needs a second,
    non-perturbable input (layer_idx), so this wrapper freezes layer_idx
    per-call and exposes the signature GNNExplainer expects."""

    def __init__(self, model: LayerGuidedGIN, layer_idx: torch.Tensor):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        return self.model(x, self.layer_idx, edge_index, batch)


def train_model(ds, device, epochs, seed=0, hidden=64, layers=3, layer_embed_dim=8, lr=1e-3):
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
    return model, train_set, val_set, test_set


def feature_attribution_analysis(model, test_set, device, n_graphs=30):
    print(f"\n{'='*70}\nFEATURE ATTRIBUTION: does node importance track peel-layer depth?\n{'='*70}")
    model.eval()
    per_graph_corrs = []
    skipped_no_variance = 0

    for i, d in enumerate(test_set[:n_graphs]):
        x = d.x.to(device)
        edge_index = d.edge_index.to(device)
        layer_idx = d.layer_idx.to(device)
        n = x.shape[0]

        if n < 4 or edge_index.shape[1] == 0:
            continue

        wrapped = _WrappedModel(model, layer_idx).to(device)
        explainer = Explainer(
            model=wrapped,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,
            model_config=ModelConfig(mode="regression", task_level="graph", return_type="raw"),
        )
        batch = torch.zeros(n, dtype=torch.long, device=device)
        explanation = explainer(x=x, edge_index=edge_index, batch=batch)
        node_importance = explanation.node_mask.sum(dim=-1).detach().cpu().numpy()

        true_layers = layer_idx.cpu().numpy()
        if true_layers.max() == true_layers.min():
            skipped_no_variance += 1
            continue  # all nodes same layer (e.g. kappa=1 graph) -- no ranking possible

        rho = spearmanr(node_importance, true_layers).correlation
        if rho == rho:  # not nan
            per_graph_corrs.append(rho)

    if not per_graph_corrs:
        print("No graphs with enough layer variance to analyze. Try more test graphs or check kappa>1 subset.")
        return

    arr = np.array(per_graph_corrs)
    print(f"Analyzed {len(arr)} graphs (skipped {skipped_no_variance} with uniform peel-layer, e.g. kappa=1)")
    print(f"Correlation(node importance, true peel-layer index) across graphs:")
    print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}  positive_fraction={float((arr>0).mean()):.2f}")
    print(
        "\nInterpretation: a consistently NEGATIVE correlation means the model weights "
        "OUTER-layer nodes (layer=1) more heavily -- plausible, since the outer face is "
        "what directly determines whether a graph peels down in few steps. A consistently "
        "POSITIVE correlation would mean it weights INNER/core nodes more heavily. Near-zero "
        "or high-variance results mean no consistent alignment was found -- report honestly "
        "either way, this is a real empirical question, not one with a 'correct' answer to steer toward."
    )


def structural_complexity_error_analysis(model, test_set, device):
    print(f"\n{'='*70}\nERROR vs STRUCTURAL COMPLEXITY\n{'='*70}")
    model.eval()
    records = []
    with torch.no_grad():
        for d in test_set:
            x = d.x.to(device)
            edge_index = d.edge_index.to(device)
            layer_idx = d.layer_idx.to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            pred = model(x, layer_idx, edge_index, batch).item()
            true = d.y.item()
            # x columns: [degree, clustering, cut_vertex_flag, block_size_norm]
            n_cut_vertices = float(x[:, 2].sum().item())
            n_nodes = x.shape[0]
            records.append({
                "abs_error": abs(pred - true),
                "n_nodes": n_nodes,
                "n_cut_vertices": n_cut_vertices,
                "cut_vertex_density": n_cut_vertices / max(n_nodes, 1),
            })

    errors = np.array([r["abs_error"] for r in records])
    cvd = np.array([r["cut_vertex_density"] for r in records])
    n_nodes_arr = np.array([r["n_nodes"] for r in records])

    rho_cv = spearmanr(errors, cvd).correlation if len(errors) > 1 else float("nan")
    rho_size = spearmanr(errors, n_nodes_arr).correlation if len(errors) > 1 else float("nan")

    print(f"n_test={len(records)}")
    print(f"Spearman(abs_error, cut_vertex_density): {rho_cv:.4f}")
    print(f"Spearman(abs_error, n_nodes):             {rho_size:.4f}")
    print(
        "\nInterpretation: a positive correlation with cut-vertex density means the model "
        "struggles more on structurally complex (highly block-decomposed) graphs -- exactly "
        "the regime where the peeling algorithm itself has more room for subtlety. A "
        "near-zero or negative result means error is NOT explained by structural complexity, "
        "which is itself a meaningful (and reportable) finding."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--n_explain", type=int, default=30,
                     help="number of test graphs to run GNNExplainer on (slow -- keep modest)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = LayerGuidedDataset(labels_csv=args.labels_csv, target="kappa")
    if len(ds) < 20:
        raise SystemExit(f"Only {len(ds)} usable graphs -- need more for a meaningful analysis.")

    print(f"Training layer-guided GIN on kappa with seed={args.seed}...")
    model, train_set, val_set, test_set = train_model(ds, device, args.epochs, seed=args.seed)

    feature_attribution_analysis(model, test_set, device, n_graphs=args.n_explain)
    structural_complexity_error_analysis(model, test_set, device)


if __name__ == "__main__":
    main()
