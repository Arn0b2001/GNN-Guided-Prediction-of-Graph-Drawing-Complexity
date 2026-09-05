#!/usr/bin/env python3
"""Validate dataset against research requirements for GNN-Guided Prediction of Graph Drawing Complexity Measures."""

import pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent

def main():
    labels_path = ROOT / "output/benchmark/labels.csv"
    summary_path = ROOT / "output/benchmark/summary.json"
    
    df = pd.read_csv(labels_path)
    with open(summary_path) as f:
        summary = json.load(f)
    
    print("=" * 70)
    print("RESEARCH READINESS VALIDATION REPORT")
    print("Research: GNN-Guided Prediction for Graph Drawing Complexity Measures")
    print("=" * 70)
    
    # Requirement 1: Dataset size and node range
    print("\n1. DATASET SIZE & NODE RANGE")
    print("-" * 70)
    target_min_nodes, target_max_nodes = 10, 500
    target_total = 8000
    
    actual_min = df['n_nodes'].min()
    actual_max = df['n_nodes'].max()
    actual_total = len(df)
    
    print(f"Target: {target_total} graphs, {target_min_nodes}-{target_max_nodes} nodes")
    print(f"Actual: {actual_total} graphs, {actual_min}-{actual_max} nodes")
    
    size_ok = (actual_min >= target_min_nodes and actual_max <= target_max_nodes)
    total_ok = abs(actual_total - target_total) / target_total < 0.1  # Within 10%
    
    print(f"Node range: {'PASS' if size_ok else 'FAIL'}")
    print(f"Total count: {'PASS' if total_ok else 'FAIL'}")
    
    # Requirement 2: Label completeness
    print("\n2. LABEL COMPLETENESS")
    print("-" * 70)
    
    # Core labels
    core_labels = {
        'kappa': 'Outerplanarity number',
        'min_layers': 'Minimum layer count', 
        'bend_complexity': 'Bend complexity'
    }
    
    for col, desc in core_labels.items():
        # For kappa, only planar graphs should have values
        if col == 'kappa':
            planar_df = df[df['is_planar']]
            present = planar_df[col].notna().sum()
            total_planar = len(planar_df)
            coverage = present / total_planar * 100 if total_planar > 0 else 0
            print(f"{desc} (planar only): {present}/{total_planar} ({coverage:.1f}%)")
        else:
            present = df[col].notna().sum()
            coverage = present / len(df) * 100
            print(f"{desc}: {present}/{len(df)} ({coverage:.1f}%)")
    
    # Complexity classes
    class_labels = ['kappa_class', 'layers_class', 'bends_class']
    print("\nComplexity classes:")
    for col in class_labels:
        present = df[col].notna().sum()
        coverage = present / len(df) * 100
        print(f"  {col}: {present}/{len(df)} ({coverage:.1f}%)")
    
    # Requirement 3: Structural features
    print("\n3. STRUCTURAL FEATURES")
    print("-" * 70)
    
    struct_features = [
        'n_blocks', 'n_cut_vertices', 'max_block_size',
        'bc_tree_n_nodes', 'bc_tree_n_edges', 'avg_degree', 'max_degree'
    ]
    
    for feat in struct_features:
        present = df[feat].notna().sum()
        coverage = present / len(df) * 100
        print(f"{feat}: {present}/{len(df)} ({coverage:.1f}%)")
    
    # Additional features
    peel_present = df['peel_layers'].notna().sum()
    bc_present = df['block_cut_tree'].notna().sum()
    print(f"peel_layers: {peel_present}/{len(df)} ({peel_present/len(df)*100:.1f}%)")
    print(f"block_cut_tree: {bc_present}/{len(df)} ({bc_present/len(df)*100:.1f}%)")
    
    # Requirement 4: Graph class distribution
    print("\n4. GRAPH CLASS DISTRIBUTION")
    print("-" * 70)
    
    class_dist = df['graph_class'].value_counts()
    print("Synthetic graph classes:")
    for cls, count in class_dist.items():
        print(f"  {cls}: {count}")
    
    rome_count = (df['source'] == 'rome').sum()
    synthetic_count = (df['source'] == 'synthetic').sum()
    print(f"\nSource distribution:")
    print(f"  Rome: {rome_count}")
    print(f"  Synthetic: {synthetic_count}")
    
    # Requirement 5: Label type distribution
    print("\n5. LABEL TYPE DISTRIBUTION")
    print("-" * 70)
    
    print("Kappa label types:")
    print(df['kappa_label_type'].value_counts())
    
    print("\nLayer label types:")
    print(df['layers_label_type'].value_counts())
    
    print("\nBend label types:")
    print(df['bends_label_type'].value_counts())
    
    # Requirement 6: Data quality checks
    print("\n6. DATA QUALITY")
    print("-" * 70)
    
    # Check for any missing critical data
    missing_critical = df[['n_nodes', 'n_edges', 'min_layers', 'bend_complexity']].isnull().any().any()
    print(f"Missing critical data (nodes, edges, layers, bends): {'FAIL' if missing_critical else 'PASS'}")
    
    # Check edge counts are reasonable
    invalid_edges = (df['n_edges'] <= 0).sum()
    print(f"Graphs with invalid edge counts (<=0): {invalid_edges} {'FAIL' if invalid_edges > 0 else 'PASS'}")
    
    # Check for duplicate graph IDs
    duplicates = df['graph_id'].duplicated().sum()
    print(f"Duplicate graph IDs: {duplicates} {'FAIL' if duplicates > 0 else 'PASS'}")
    
    # Final assessment
    print("\n" + "=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    
    all_checks_pass = all([
        size_ok, total_ok,
        df['min_layers'].notna().all(),
        df['bend_complexity'].notna().all(),
        df[struct_features].notna().all().all(),
        not missing_critical,
        invalid_edges == 0,
        duplicates == 0
    ])
    
    if all_checks_pass:
        print("STATUS: READY FOR RESEARCH")
        print("\nThe dataset meets all critical requirements for:")
        print("- GNN training on graph structural features")
        print("- Prediction of graph drawing complexity measures")
        print("- Analysis across multiple graph classes and complexity levels")
    else:
        print("STATUS: NEEDS ATTENTION")
        print("Some critical checks failed - review details above")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()