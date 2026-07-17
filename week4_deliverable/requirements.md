# Week4 Advanced Spatio-Temporal Models — Requirements

## Core Dependencies (inherited from Week3)
torch>=2.0.0
torch_geometric>=2.3.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0        # for sparse matrix ops in AGFormer static adj

## Notes
- All three models (STGCN, AGFormer, Spacetimeformer) use the SAME data pipeline,
  trainer skeleton, and evaluation framework as the Week3 baseline.
- No additional dependencies required beyond Week3 setup.
- To install torch_geometric on a fresh machine:
    pip install torch_geometric
    # For GCNConv/GATConv backends (torch_sparse recommended):
    pip install torch_sparse torch_scatter torch_cluster torch_spline_conv -f https://pytorch-geometric.com/whl/torch-2.0.0+.html
