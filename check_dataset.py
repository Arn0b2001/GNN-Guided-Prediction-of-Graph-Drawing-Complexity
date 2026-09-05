#!/usr/bin/env python3
"""Check dataset completeness and quality."""

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.graph_io import read_graphml

def main():
    labels_path = ROOT / "output/benchmark/labels.csv"
    df = pd.read_csv(labels_path)
    
    print("=" * 60)
    print("DATASET COMPLETENESS CHECK")
    print("=" * 60)
    
    # Basic stats
    print(f"\nTotal graphs: {len(df)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"Graph classes: {df['graph_class'].value_counts().to_dict()}")
    
    # Check Rome edge counts
    print("\n" + "=" * 60)
    print("CHECKING ROME GRAPH EDGE COUNTS")
    print("=" * 60)
    rome_df = df[df['source'] == 'rome'].head(5)
    for idx, row in rome_df.iterrows():
        try:
            g = read_graphml(row['path'])
            actual_edges = g.number_of_edges()
            recorded_edges = row['n_edges']
            print(f"{row['graph_id']}: actual={actual_edges}, recorded={recorded_edges}, match={actual_edges == recorded_edges}")
        except Exception as e:
            print(f"{row['graph_id']}: ERROR - {e}")
    
    # Check label completeness
    print("\n" + "=" * 60)
    print("LABEL COMPLETENESS")
    print("=" * 60)
    
    # Critical labels
    critical_labels = ['kappa', 'min_layers', 'bend_complexity', 'kappa_class', 'layers_class', 'bends_class']
    for label in critical_labels:
        missing = df[label].isnull().sum()
        present = len(df) - missing
        print(f"{label}: {present}/{len(df)} present ({missing} missing)")
    
    # Check structural features
    print("\n" + "=" * 60)
    print("STRUCTURAL FEATURES")
    print("=" * 60)
    struct_features = ['n_blocks', 'n_cut_vertices', 'max_block_size', 'bc_tree_n_nodes', 'bc_tree_n_edges', 'avg_degree', 'max_degree']
    for feat in struct_features:
        missing = df[feat].isnull().sum()
        present = len(df) - missing
        print(f"{feat}: {present}/{len(df)} present ({missing} missing)")
    
    # Check planar vs non-planar
    print("\n" + "=" * 60)
    print("PLANARITY DISTRIBUTION")
    print("=" * 60)
    planar_count = df['is_planar'].sum()
    non_planar_count = len(df) - planar_count
    print(f"Planar: {planar_count} ({planar_count/len(df)*100:.1f}%)")
    print(f"Non-planar: {non_planar_count} ({non_planar_count/len(df)*100:.1f}%)")
    
    # Node distribution
    print("\n" + "=" * 60)
    print("NODE DISTRIBUTION")
    print("=" * 60)
    print(df['n_nodes'].describe())
    
    # Layer distribution
    print("\n" + "=" * 60)
    print("LAYER DISTRIBUTION")
    print("=" * 60)
    print(df['min_layers'].describe())
    
    # Check for any errors in output
    print("\n" + "=" * 60)
    print("ERROR FILES")
    print("=" * 60)
    error_file = ROOT / "output/benchmark/errors.json"
    if error_file.exists():
        print(f"ERROR: {error_file} exists - some graphs failed processing")
    else:
        print("OK: No error file - all graphs processed successfully")
    
    print("\n" + "=" * 60)
    print("RESEARCH READINESS ASSESSMENT")
    print("=" * 60)
    
    issues = []
    
    # Check 1: Rome edge counts not populated
    if (df[df['source'] == 'rome']['n_edges'] == 0).all():
        issues.append("Rome graph edge counts are 0 (not populated from actual graph files)")
    
    # Check 2: Missing kappa for non-planar graphs (expected)
    planar_df = df[df['is_planar']]
    if planar_df['kappa'].isnull().any():
        issues.append("Some planar graphs missing kappa values")
    
    # Check 3: Missing structural features
    if df['n_blocks'].isnull().any():
        issues.append("Some graphs missing structural features")
    
    if issues:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("OK: All critical checks passed")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()