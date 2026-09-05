#!/usr/bin/env python3
"""
Run multiple training runs with different seeds to assess variance.
Runs each model with 5 different seeds and reports variance statistics.
"""

import subprocess
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def run_training(script_name, labels_csv, target, model_type, epochs, seeds):
    """Run training with multiple seeds and collect results."""
    results = []
    
    for seed in seeds:
        print(f"\nRunning {script_name} with seed={seed}, target={target}, model={model_type}")
        
        cmd = [
            "python", script_name,
            "--labels_csv", labels_csv,
            "--target", target,
            "--epochs", str(epochs),
            "--seed", str(seed)
        ]
        
        if model_type and script_name == "train_baseline.py":
            cmd.extend(["--model", model_type])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            output = result.stdout + result.stderr
            
            # Parse output for key metrics
            mae = None
            spearman = None
            
            for line in output.split('\n'):
                if 'test_mae=' in line:
                    # Extract test_mae value
                    parts = line.split('test_mae=')[1].split()[0]
                    try:
                        mae = float(parts)
                    except:
                        pass
                if 'test_spearman=' in line:
                    # Extract test_spearman value
                    parts = line.split('test_spearman=')[1].split()[0]
                    try:
                        spearman = float(parts)
                    except:
                        pass
            
            if mae is not None and spearman is not None:
                results.append({
                    'seed': seed,
                    'mae': mae,
                    'spearman': spearman
                })
                print(f"  Seed {seed}: MAE={mae:.4f}, Spearman={spearman:.4f}")
            else:
                print(f"  Seed {seed}: Failed to parse results")
                print(f"  Output: {output[-500:]}")  # Last 500 chars
                
        except subprocess.TimeoutExpired:
            print(f"  Seed {seed}: Timeout")
        except Exception as e:
            print(f"  Seed {seed}: Error - {e}")
    
    return results

def summarize_results(results, model_name, target):
    """Calculate and print variance statistics."""
    if not results:
        print(f"\nNo results for {model_name} on {target}")
        return None
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY: {model_name} on {target}")
    print(f"{'='*70}")
    
    mae_stats = {
        'mean': df['mae'].mean(),
        'std': df['mae'].std(),
        'min': df['mae'].min(),
        'max': df['mae'].max(),
        'median': df['mae'].median()
    }
    
    spearman_stats = {
        'mean': df['spearman'].mean(),
        'std': df['spearman'].std(),
        'min': df['spearman'].min(),
        'max': df['spearman'].max(),
        'median': df['spearman'].median()
    }
    
    print(f"\nMAE Statistics:")
    print(f"  Mean: {mae_stats['mean']:.4f}")
    print(f"  Std:  {mae_stats['std']:.4f}")
    print(f"  Min:  {mae_stats['min']:.4f}")
    print(f"  Max:  {mae_stats['max']:.4f}")
    print(f"  Median: {mae_stats['median']:.4f}")
    
    print(f"\nSpearman Statistics:")
    print(f"  Mean: {spearman_stats['mean']:.4f}")
    print(f"  Std:  {spearman_stats['std']:.4f}")
    print(f"  Min:  {spearman_stats['min']:.4f}")
    print(f"  Max:  {spearman_stats['max']:.4f}")
    print(f"  Median: {spearman_stats['median']:.4f}")
    
    print(f"\nIndividual runs:")
    for _, row in df.iterrows():
        print(f"  Seed {row['seed']}: MAE={row['mae']:.4f}, Spearman={row['spearman']:.4f}")
    
    print(f"{'='*70}")
    
    return {
        'model': model_name,
        'target': target,
        'mae_stats': mae_stats,
        'spearman_stats': spearman_stats,
        'individual_results': results
    }

def main():
    labels_csv = "output/benchmark/labels.csv"
    epochs = 100
    seeds = [0, 1, 2, 3, 4]  # 5 different seeds
    
    targets = ["kappa", "min_layers", "bend_complexity"]
    
    all_results = []
    
    # Define models to run (only missing ones)
    experiments = [
        ("train_baseline.py", "gin", "Baseline GIN"),
        ("train_baseline.py", "gcn", "Baseline GCN"),
    ]
    
    for target in targets:
        print(f"\n{'#'*70}")
        print(f"# RUNNING EXPERIMENTS FOR TARGET: {target}")
        print(f"{'#'*70}")
        
        for script, model, name in experiments:
            results = run_training(script, labels_csv, target, model, epochs, seeds)
            summary = summarize_results(results, name, target)
            if summary:
                all_results.append(summary)
    
    # Final summary table
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY TABLE")
    print(f"{'='*70}")
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            'Model': result['model'],
            'Target': result['target'],
            'MAE_Mean': result['mae_stats']['mean'],
            'MAE_Std': result['mae_stats']['std'],
            'Spearman_Mean': result['spearman_stats']['mean'],
            'Spearman_Std': result['spearman_stats']['std']
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results
    summary_df.to_csv("multi_seed_results.csv", index=False)
    print(f"\nResults saved to multi_seed_results.csv")

if __name__ == "__main__":
    main()