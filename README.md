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

## Environment setup

```bash
conda create -n mpnn python=3.11 -y
conda activate mpnn
pip install -r requirements.txt
```

### GPU (optional)

For training on NVIDIA GPUs, replace the CPU-only PyTorch wheel with a CUDA
build (CUDA 12.1 is a safe choice for driver versions supporting CUDA 12.x):

```bash
pip install --upgrade --force-reinstall torch \
  --index-url https://download.pytorch.org/whl/cu121
```

Training scripts auto-detect CUDA via `torch.cuda.is_available()`; no code
changes needed.

## Status

In progress — step 1 (setup).
