#!/usr/bin/env python3
"""Fix Rome graph edge counts by loading actual graph files."""

import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.graph_io import read_graphml

def main():
    labels_path = ROOT / "output/benchmark/labels.csv"
    df = pd.read_csv(labels_path)
    
    print("Fixing Rome graph edge counts...")
    
    # Find Rome graphs with 0 edge counts
    rome_mask = (df['source'] == 'rome') & (df['n_edges'] == 0)
    rome_to_fix = df[rome_mask]
    
    print(f"Found {len(rome_to_fix)} Rome graphs with edge count = 0")
    
    # Update edge counts by loading actual graph files
    for idx, row in tqdm(rome_to_fix.iterrows(), total=len(rome_to_fix), desc="Loading graphs"):
        try:
            g = read_graphml(row['path'])
            df.at[idx, 'n_edges'] = g.number_of_edges()
        except Exception as e:
            print(f"Error loading {row['graph_id']}: {e}")
    
    # Save updated dataset
    df.to_csv(labels_path, index=False)
    print(f"Updated edge counts saved to {labels_path}")
    
    # Verify the fix
    rome_after = df[df['source'] == 'rome']
    print(f"\nRome edge count statistics after fix:")
    print(rome_after['n_edges'].describe())
    
    zero_edges = (rome_after['n_edges'] == 0).sum()
    if zero_edges > 0:
        print(f"WARNING: {zero_edges} Rome graphs still have edge count = 0")
    else:
        print("All Rome graphs now have correct edge counts")

if __name__ == "__main__":
    main()