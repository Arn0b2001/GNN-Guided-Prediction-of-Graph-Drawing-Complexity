#!/usr/bin/env python3
"""
End-to-end pipeline test for GNN training on graph drawing complexity dataset.
Tests the complete workflow from dataset loading to model training.
"""

import torch
from dataset import GraphDrawingComplexityDataset
from train_baseline import GCNRegressor, GINRegressor, split_dataset
from torch_geometric.loader import DataLoader

def test_pipeline():
    print("=" * 70)
    print("END-TO-END PIPELINE TEST")
    print("=" * 70)
    
    labels_csv = "output/benchmark/labels.csv"
    
    # Test 1: Dataset loading for all targets
    print("\n1. Testing dataset loading for all targets...")
    targets = ["kappa", "min_layers", "bend_complexity"]
    datasets = {}
    
    for target in targets:
        try:
            ds = GraphDrawingComplexityDataset(labels_csv=labels_csv, target=target)
            datasets[target] = ds
            print(f"  {target}: {len(ds)} graphs loaded successfully")
            
            # Test data format
            sample = ds[0]
            print(f"    Sample features: {sample.x.shape}")
            print(f"    Sample edges: {sample.edge_index.shape}")
            print(f"    Sample label: {sample.y.item()}")
        except Exception as e:
            print(f"  {target}: FAILED - {e}")
            return False
    
    # Test 2: Train/val/test split
    print("\n2. Testing train/val/test split...")
    for target, ds in datasets.items():
        try:
            train, val, test = split_dataset(ds)
            print(f"  {target}: train={len(train)}, val={len(val)}, test={len(test)}")
        except Exception as e:
            print(f"  {target}: FAILED - {e}")
            return False
    
    # Test 3: Model instantiation
    print("\n3. Testing model instantiation...")
    for target, ds in datasets.items():
        in_dim = ds[0].x.shape[1]
        try:
            gcn = GCNRegressor(in_dim)
            gin = GINRegressor(in_dim)
            print(f"  {target}: GCN and GIN models created successfully")
        except Exception as e:
            print(f"  {target}: FAILED - {e}")
            return False
    
    # Test 4: Data loading and forward pass
    print("\n4. Testing data loading and forward pass...")
    device = torch.device("cpu")
    
    for target, ds in datasets.items():
        try:
            train, val, test = split_dataset(ds)
            train_loader = DataLoader(train, batch_size=16, shuffle=True)
            
            in_dim = ds[0].x.shape[1]
            model = GINRegressor(in_dim).to(device)
            
            for batch in train_loader:
                batch = batch.to(device)
                output = model(batch.x, batch.edge_index, batch.batch)
                print(f"  {target}: Forward pass successful, output shape: {output.shape}")
                break
        except Exception as e:
            print(f"  {target}: FAILED - {e}")
            return False
    
    # Test 5: Quick training iteration
    print("\n5. Testing quick training iteration...")
    for target, ds in datasets.items():
        try:
            train, val, test = split_dataset(ds)
            train_loader = DataLoader(train[:100], batch_size=16, shuffle=True)  # Small subset
            
            in_dim = ds[0].x.shape[1]
            model = GINRegressor(in_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.functional.mse_loss(output, batch.y)
                loss.backward()
                optimizer.step()
                print(f"  {target}: Training iteration successful, loss: {loss.item():.4f}")
                break
        except Exception as e:
            print(f"  {target}: FAILED - {e}")
            return False
    
    print("\n" + "=" * 70)
    print("PIPELINE TEST: ALL CHECKS PASSED")
    print("=" * 70)
    print("\nThe end-to-end pipeline is working correctly:")
    print("- Dataset loading for all targets [OK]")
    print("- Train/val/test splitting [OK]")
    print("- Model instantiation (GCN/GIN) [OK]")
    print("- Data loading and forward pass [OK]")
    print("- Training iteration [OK]")
    print("\nReady for full-scale training experiments!")
    
    return True

if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)