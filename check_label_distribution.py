#!/usr/bin/env python3
"""
Check label distribution skew for bend_complexity and other targets.
This helps identify if MAE metrics are inflated by majority-easy cases.
"""

import csv
import numpy as np
import pandas as pd

def check_distribution(labels_csv, target):
    """Check distribution of target values."""
    
    # Load data
    df = pd.read_csv(labels_csv)
    
    # Filter for target
    if target == "kappa":
        df = df[df['kappa'].notna()]
        values = df['kappa'].values
    elif target == "min_layers":
        values = df['min_layers'].values
    elif target == "bend_complexity":
        df = df[df['bend_complexity'].notna()]
        values = df['bend_complexity'].values
    else:
        raise ValueError(f"Unknown target: {target}")
    
    print(f"\n{'='*70}")
    print(f"LABEL DISTRIBUTION ANALYSIS: {target}")
    print(f"{'='*70}")
    print(f"Total samples: {len(values)}")
    print(f"Min: {values.min():.2f}")
    print(f"Max: {values.max():.2f}")
    print(f"Mean: {values.mean():.2f}")
    print(f"Median: {np.median(values):.2f}")
    print(f"Std: {values.std():.2f}")
    
    # Check for zeros
    frac_zero = (values == 0).mean()
    print(f"Fraction exactly zero: {frac_zero:.1%}")
    
    # Check for small values
    frac_small = (values <= 1).mean()
    print(f"Fraction <= 1: {frac_small:.1%}")
    
    # Check distribution tails
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(values, p):.2f}")
    
    # Check if heavy skew
    skew_warning = ""
    if frac_zero > 0.3:
        skew_warning = "WARNING: >30% of values are zero - MAE may be inflated by easy majority"
    elif values.mean() / np.median(values) > 3:
        skew_warning = "WARNING: Mean > 3x median - heavy right skew suspected"
    
    if skew_warning:
        print(f"\n{skew_warning}")
        print("Recommendation: Report stratified MAE (e.g., MAE for values > 0)")
    
    print(f"{'='*70}")

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "output/benchmark/labels.csv"
    
    # Check all targets
    for target in ["kappa", "min_layers", "bend_complexity"]:
        check_distribution(csv_path, target)