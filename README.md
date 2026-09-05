# Graph Drawing Complexity Benchmark (GNN Dataset)

Benchmark dataset for **GNN-Guided Prediction for Graph Drawing Complexity Measures** - a research project on predicting graph drawing complexity measures using Graph Neural Networks.

## Overview

This project implements a complete pipeline for:
1. **Dataset Construction**: Building a unified graph drawing complexity benchmark from Rome and synthetic graph corpora
2. **Complexity Labeling**: Computing outerplanarity number (κ), minimum layer count, and bend complexity
3. **GNN Training**: Baseline and structure-aware GNN models (GCN, GIN) for regression tasks
4. **Theory Alignment**: Validating model predictions against proven theoretical bounds

## Dataset

| Source | Count | Node range | Graph classes |
|--------|-------|------------|---------------|
| Rome graphs | 6,000 (selected) | 10–100 | Real-world graphs |
| Synthetic | 1,981 | 3–500 | k-tree, maximal-outerplanar, random-planar, series-parallel |
| **Total** | **7,981** | **10–500** | **Mixed** |

## Labels and Features

**Complexity Measures:**
- `kappa` — Outerplanarity number (peeling algorithm)
- `min_layers` — Minimum layer count (ILP exact for n≤80, bounds otherwise)  
- `bend_complexity` — Bend complexity (planarization-based bound for non-planar graphs)

**Structural Features:**
- Block-cut tree statistics
- Peel-layer assignments
- Node degree, clustering coefficient, cut-vertex flags

## Installation

```bash
# Basic dataset construction
pip install -r requirements.txt

# GNN training
pip install -r requirements_gnn.txt
```

## Quick Start

```bash
# Build the dataset
python build_dataset.py

# Train baseline GNN
python train_baseline.py --labels_csv output/benchmark/labels.csv --target kappa --model gin --epochs 100

# Train layer-guided GNN
python train_layer_guided.py --labels_csv output/benchmark/labels.csv --target kappa --epochs 100

# Train structure-aware GNN (block-cut tree)
python train_structure_aware.py --labels_csv output/benchmark/labels.csv --target kappa --epochs 100
```

## Project Structure

```
graph-drawing-gnn-dataset/
├── src/                      # Core labeling algorithms
│   ├── labels/              # Complexity measure computation
│   ├── features/            # Structural feature extraction
│   └── graph_io.py          # Graph I/O utilities
├── output/benchmark/         # Generated dataset
├── dataset.py               # PyG dataset loader (baseline)
├── dataset_structural.py     # PyG dataset loader (structural features)
├── train_baseline.py        # Baseline GCN/GIN training
├── train_layer_guided.py    # Layer-guided GNN training
├── train_structure_aware.py # Structure-aware GNN training
├── objective3_theory_alignment.py     # Theory alignment analysis
├── bound_tightness_analysis.py         # Theoretical bound validation
├── identify_bound_violations_full.py  # Data quality verification
└── config.yaml              # Configuration file
```

## Key Scripts

- `build_dataset.py` - Main dataset construction pipeline
- `train_baseline.py` - Baseline GCN/GIN models with trivial baselines
- `train_layer_guided.py` - Layer-guided GNN with outerplanarity inductive bias
- `train_structure_aware.py` - Block-cut tree structure-aware GNN
- `objective3_theory_alignment.py` - GNNExplainer feature attribution analysis
- `bound_tightness_analysis.py` - Validation against Biedl-Mondal theoretical bounds
- `validate_research_ready.py` - Dataset quality validation

## Results

Current performance (layer-guided GIN):
- **κ prediction**: MAE ≈ 0.18, Spearman ≈ 0.96
- **min_layers prediction**: MAE ≈ 1.0, Spearman ≈ 0.78
- **bend_complexity**: MAE ≈ 10.9, Spearman ≈ 0.91

## Research Contributions

1. **Novel Dataset**: First comprehensive benchmark for graph drawing complexity prediction
2. **Structure-Aware Models**: Layer-guided and block-cut-tree conditioned GNNs
3. **Theory Alignment**: Empirical validation against proven theoretical bounds
4. **Data Quality**: Rigorous validation of labeling pipeline with known limitations

## Citation

If you use this dataset or code, please cite:

```bibtex
@software{graph_drawing_complexity,
  title={Graph Drawing Complexity Benchmark for GNN Research},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/graph-drawing-gnn-dataset}
}
```


## Acknowledgments

- Rome graphs: [Rome Graphs Dataset]
- Synthetic graphs: Generated using NetworkX
- Theoretical bounds: Based on Biedl & Mondal (WG 2024)
