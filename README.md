# MPNN — Message Passing Neural Networks for Molecular Property Prediction

A learning project: build a Message Passing Neural Network from scratch in PyTorch to predict molecular properties, then compare with the PyTorch Geometric implementation.

## Roadmap

1. **Setup** — repo, environment, dependencies
2. **Featurization** — SMILES → graph tensors via RDKit
3. **Message function** — per-edge MLP
4. **Aggregation** — sum/mean messages per node (`scatter_add`)
5. **Update function** — GRU cell + T-step message passing
6. **Readout** — graph-level pooling + prediction head
7. **Training** — QM9 or ESOL, batching variable-sized graphs
8. **PyTorch Geometric rewrite** — compare to the reference implementation
9. **Extensions** — GCN / GAT / GIN / D-MPNN

## Status

In progress — step 1 (setup).
