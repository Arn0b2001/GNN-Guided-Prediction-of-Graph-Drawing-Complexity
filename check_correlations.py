#!/usr/bin/env python3
"""
Check correlations between complexity measures and graph size.
This helps identify if high Spearman scores are artifacts of graph size correlation.
"""

import csv
import numpy as np
from scipy.stats import spearmanr
import pandas as pd

def check_correlations(labels_csv):
    """Check correlations between all complexity measures and graph size."""
    
    # Load data
    df = pd.read_csv(labels_csv)
    
    print("=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    
    # Define targets to check
    targets = {
        "kappa": "kappa",
        "min_layers": "min_layers", 
        "bend_complexity": "bend_complexity"
    }
    
    print("\nCorrelation with graph size:")
    print("-" * 70)
    print(f"{'Target':<20} {'vs n_nodes':<15} {'vs n_edges':<15} {'vs density':<15}")
    print("-" * 70)
    
    for target_name, target_col in targets.items():
        # Filter valid values
        valid_mask = df[target_col].notna()
        valid_df = df[valid_mask]
        
        if len(valid_df) == 0:
            print(f"{target_name:<20} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
            continue
        
        target_vals = valid_df[target_col].values
        n_nodes = valid_df['n_nodes'].values
        n_edges = valid_df['n_edges'].values
        density = valid_df['n_edges'] / (valid_df['n_nodes'] * (valid_df['n_nodes'] - 1) / 2)
        
        rho_nodes = spearmanr(n_nodes, target_vals).correlation
        rho_edges = spearmanr(n_edges, target_vals).correlation
        rho_density = spearmanr(density, target_vals).correlation
        
        print(f"{target_name:<20} {rho_nodes:>13.4f} {rho_edges:>13.4f} {rho_density:>13.4f}")
    
    print("\nCorrelation between complexity measures:")
    print("-" * 70)
    
    # Check kappa vs bend_complexity (both available for planar graphs)
    planar_mask = df['kappa'].notna() & df['bend_complexity'].notna()
    if planar_mask.sum() > 0:
        kappa_vals = df[planar_mask]['kappa'].values
        bend_vals = df[planar_mask]['bend_complexity'].values
        rho_kappa_bend = spearmanr(kappa_vals, bend_vals).correlation
        print(f"kappa vs bend_complexity: {rho_kappa_bend:.4f} (n={planar_mask.sum()})")
    
    # Check kappa vs min_layers
    kappa_layer_mask = df['kappa'].notna() & df['min_layers'].notna()
    if kappa_layer_mask.sum() > 0:
        kappa_vals = df[kappa_layer_mask]['kappa'].values
        layer_vals = df[kappa_layer_mask]['min_layers'].values
        rho_kappa_layer = spearmanr(kappa_vals, layer_vals).correlation
        print(f"kappa vs min_layers: {rho_kappa_layer:.4f} (n={kappa_layer_mask.sum()})")
    
    # Check bend_complexity vs min_layers
    bend_layer_mask = df['bend_complexity'].notna() & df['min_layers'].notna()
    if bend_layer_mask.sum() > 0:
        bend_vals = df[bend_layer_mask]['bend_complexity'].values
        layer_vals = df[bend_layer_mask]['min_layers'].values
        rho_bend_layer = spearmanr(bend_vals, layer_vals).correlation
        print(f"bend_complexity vs min_layers: {rho_bend_layer:.4f} (n={bend_layer_mask.sum()})")
    
    print("\nInterpretation:")
    print("-" * 70)
    print("High correlation (>0.7) with graph size suggests results may be size artifacts.")
    print("High correlation between measures suggests they may capture similar information.")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/benchmark/labels.csv"
    check_correlations(csv_path)