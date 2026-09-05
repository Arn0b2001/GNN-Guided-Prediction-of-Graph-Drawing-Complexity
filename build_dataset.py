#!/usr/bin/env python3
"""
Build the graph drawing complexity benchmark dataset.

Phase 1 of the research proposal:
  - Unify Rome + synthetic corpora
  - Compute kappa(G), layer count, bend complexity
  - Export structural features (block-cut tree stats, peel layers)

Usage:
    pip install -r requirements.txt
    python build_dataset.py                  # full pipeline
    python build_dataset.py --limit 50       # smoke test
    python build_dataset.py --index-only     # only rebuild graph index
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import networkx as nx

from src.features.structural import block_cut_tree_json, structural_features
from src.graph_io import read_graphml
from src.index_graphs import GraphRecord, build_index, load_config, write_index
from src.labels.bends import bend_complexity
from src.labels.layers import class_layer_hint, minimum_layers
from src.labels.outerplanarity import (
    class_kappa_hint,
    outerplanarity_layers,
    outerplanarity_number,
)


def complexity_class(value: int | None, low: int = 2, high: int = 5) -> str | None:
    if value is None:
        return None
    if value <= low:
        return "low"
    if value <= high:
        return "medium"
    return "high"


def label_record(rec: GraphRecord, cfg: dict) -> dict:
    g = read_graphml(rec.path)
    labeling = cfg["labeling"]

    # Planarity must come from an actual planarity test, not from whether
    # kappa computation happened to succeed. Previously `is_planar = kappa
    # is not None`, which silently mislabels any planar graph as
    # "non-planar" whenever outerplanarity_number() fails/gives up on it
    # (e.g. graphs that disconnect into multiple components mid-peel).
    is_planar = nx.check_planarity(g)[0]

    kappa_hint = class_kappa_hint(rec.graph_class)
    if kappa_hint is not None:
        kappa = kappa_hint
        kappa_type = "exact"
    elif is_planar:
        kappa = outerplanarity_number(g)
        kappa_type = "exact" if kappa is not None else "unknown"
        if kappa is None:
            # Genuinely planar but the peeling algorithm failed to produce
            # a value -- this should NOT be silently swallowed. Surface it
            # so failures are visible/countable instead of masquerading as
            # "this graph is non-planar".
            kappa_type = "failed"
    else:
        # Outerplanarity number is undefined for non-planar graphs.
        kappa = None
        kappa_type = "not_applicable"

    layer_hint = class_layer_hint(rec.graph_class, rec.n_nodes)
    if layer_hint is not None:
        layers, layer_type = layer_hint, "exact"
    else:
        layers, layer_type = minimum_layers(
            g, exact_max_nodes=labeling["exact_layer_max_nodes"]
        )

    bends, bend_type = bend_complexity(g, kappa=kappa)
    peel = outerplanarity_layers(g) if kappa is not None else {}
    feats = structural_features(g, peel_layers=peel or None)

    return {
        "graph_id": rec.graph_id,
        "source": rec.source,
        "path": rec.path,
        "graph_class": rec.graph_class,
        "n_nodes": rec.n_nodes,
        "n_edges": rec.n_edges,
        "is_planar": is_planar,
        "kappa": kappa,
        "kappa_label_type": kappa_type,
        "min_layers": layers,
        "layers_label_type": layer_type,
        "bend_complexity": bends,
        "bends_label_type": bend_type,
        "kappa_class": complexity_class(kappa),
        "layers_class": complexity_class(layers, low=3, high=8),
        "bends_class": complexity_class(bends, low=0, high=10),
        **feats,
        "block_cut_tree": block_cut_tree_json(g),
        "peel_layers": json.dumps({str(k): v for k, v in peel.items()}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph drawing complexity dataset")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--limit", type=int, default=None, help="Process only first N graphs")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    index_path = ROOT / cfg["output"]["index_file"]
    print("Building graph index...")
    records = build_index(args.config)
    write_index(records, index_path)
    print(f"Indexed {len(records)} graphs -> {index_path}")

    if args.index_only:
        return

    if args.limit:
        records = records[: args.limit]

    rows = []
    errors = []
    for rec in tqdm(records, desc="Labeling graphs"):
        try:
            rows.append(label_record(rec, cfg))
        except Exception as exc:
            errors.append({"graph_id": rec.graph_id, "error": str(exc)})

    labels_path = ROOT / cfg["output"]["labels_file"]
    features_path = ROOT / cfg["output"]["features_file"]

    df = pd.DataFrame(rows)
    df.to_csv(labels_path, index=False)

    feature_cols = [
        c
        for c in df.columns
        if c
        not in {
            "path",
            "block_cut_tree",
            "peel_layers",
            "graph_class",
            "source",
            "graph_id",
        }
    ]
    df[feature_cols].to_csv(features_path, index=False)

    summary = {
        "total_graphs": len(records),
        "labeled": len(rows),
        "errors": len(errors),
        "by_source": df.groupby("source").size().to_dict() if len(df) else {},
        "by_class": df.groupby("graph_class").size().to_dict() if len(df) else {},
        "kappa_stats": df["kappa"].describe().to_dict() if "kappa" in df else {},
        "layers_stats": df["min_layers"].describe().to_dict() if "min_layers" in df else {},
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if errors:
        with open(out_dir / "errors.json", "w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

    print(f"\nDone. Labels: {labels_path}")
    print(f"Features: {features_path}")
    print(f"Summary: {summary_path}")
    if errors:
        print(f"Warnings: {len(errors)} graphs failed (see {out_dir / 'errors.json'})")


if __name__ == "__main__":
    main()