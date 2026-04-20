# MPNN — Message Passing Neural Networks for Molecular Property Prediction

A learning project that builds a Message Passing Neural Network for molecular
property prediction **twice** — once from scratch in plain PyTorch, once on
PyTorch Geometric — then benchmarks it against GCN / GAT / GIN and investigates
how random vs scaffold splits and sum-pool vs Set2Set readout affect results.

Targets ESOL (Delaney 2004, ~1128 molecules, water solubility regression) as a
small, CPU-friendly working dataset.

## What you'll find here

- Complete from-scratch MPNN, one file per equation from Gilmer et al. 2017
- Equivalent PyTorch Geometric reimplementation (~1/3 the LOC)
- GCN, GAT, GIN baselines for a four-way comparison on ESOL
- Bemis-Murcko scaffold splitter showing how much random splits overstate performance
- Nine runnable demo / training / benchmark scripts

## Repo structure

```
mpnn/
  featurize.py         SMILES -> (x, edge_index, edge_attr) tensors via RDKit
  message.py           M: per-edge MLP message function            (scratch)
  aggregate.py         sum / mean / max via index_add / scatter    (scratch)
  update.py            U: GRUCell-based node update                (scratch)
  encoder.py           atom embed + T-step unrolled message passing (scratch)
  readout.py           R: permutation-invariant global pooling     (scratch)
  model.py             full MPNN = encoder + readout + MLP head    (scratch)
  pyg_model.py         same model on torch_geometric.MessagePassing
  pyg_baselines.py     GCN / GAT / GIN (all PyG)
  data.py              ESOL loader + supergraph collate for DataLoader
  splits.py            Bemis-Murcko scaffold split
  train.py             train_epoch / evaluate helpers

scripts/
  01_featurize_demo.py        show MolGraph shapes on aspirin, caffeine, ...
  02_message_demo.py          run EdgeMessage on aspirin
  03_aggregate_demo.py        hand-built graph + aspirin, verify scatter math
  04_encoder_demo.py          T-step encoder + tied vs untied parameter counts
  05_model_demo.py            permutation invariance + batching correctness
  06_train_esol.py            train scratch MPNN on ESOL (CPU or GPU)
  07_train_esol_pyg.py        same training with the PyG model
  08_benchmark_baselines.py   MPNN vs GCN vs GAT vs GIN shootout
  09_scaffold_attention.py    2x2 random/scaffold x sum_mean/set2set
```

The `mpnn/*.py` from-scratch modules are preserved unchanged as a study
reference. The PyG version (`pyg_model.py`) sits alongside; each file has a
one-line cross-reference comment pointing at its counterpart.

## Setup

```bash
conda create -n mpnn python=3.11 -y
conda activate mpnn
pip install -r requirements.txt
```

### GPU (optional)

For NVIDIA GPUs, replace CPU-only PyTorch with a CUDA build (cu121 works with
any driver supporting CUDA >=12.1):

```bash
pip install --upgrade --force-reinstall torch \
  --index-url https://download.pytorch.org/whl/cu121
```

Training scripts auto-detect CUDA; no code changes needed.

## Reproducing results

Run scripts in order. Each `scripts/0X_*.py` is standalone; run from the repo
root with the `mpnn` env active:

```bash
python -m scripts.01_featurize_demo
python -m scripts.02_message_demo
# ...
python -m scripts.06_train_esol            # full training, ~25s on GPU
python -m scripts.08_benchmark_baselines   # 4-model comparison, ~2 min on GPU
python -m scripts.09_scaffold_attention    # 2x2 comparison, ~2 min on GPU
```

Plots are saved to `results/` (gitignored). The ESOL CSV downloads to
`data/esol.csv` on first use.

## The MPNN in code: Gilmer et al. 2017 equations

| Paper equation | Intent | File |
|---|---|---|
| `m_{vw} = M_t(h_v, h_w, e_{vw})` | Per-edge message | `mpnn/message.py` |
| `m_v = sum_{w in N(v)} m_{vw}` | Aggregate over neighbors | `mpnn/aggregate.py` |
| `h_v' = U_t(h_v, m_v)` | Update node state | `mpnn/update.py` |
| T-step unroll (shared or untied M, U) | Receptive field grows to T hops | `mpnn/encoder.py` |
| `y = R({h_v})` | Graph-level readout | `mpnn/readout.py` |

The PyG version (`mpnn/pyg_model.py`) collapses these five files into one
`MPNNLayer(MessagePassing)` class by inheriting `propagate()`, which wraps
`message()` -> scatter aggregation -> user-defined `update()`.

## Key results (ESOL, single seed, RTX 3060)

### Step 9A — Architecture shootout, random split

```
Model               Params   Val RMSE   Test RMSE   Test MAE
MPNN (edge-aware)   47,489    0.5537      0.7609     0.5112
GCN                 22,209    0.6633      0.7525     0.5642
GAT (4 heads)       22,593    0.5728      0.7185     0.5047   <- best test
GIN                 34,689    0.6332      0.7379     0.5319
```

### Step 9D — Scaffold split + attention readout

```
Config                     Params   Val RMSE   Test RMSE
random + sum_mean          47,489    0.5532      0.7668
random + set2set           97,153    0.7966      0.7477
scaffold + sum_mean        47,489    0.7929      1.1874   <- +0.42 vs random
scaffold + set2set         97,153    1.3120      1.3019
```

Single seed means rankings within a narrow band (~0.03 RMSE) are noisy;
published benchmarks average over 5-10 seeds for a reason.

## Lessons learned

1. **Random splits overstate performance.** Scaffold split adds ~0.42 log
   units of test RMSE on ESOL and has 0 scaffold overlap between train and
   test (random split had 20). The scaffold number is the honest one.
2. **Edge-awareness is not decisive on small datasets.** MPNN wins on val
   but loses on test to GAT, which ignores edge features. Solubility leans
   more on atom-level descriptors than specific bond types here.
3. **More expressive != better on small data.** Set2Set (2x params, what
   Gilmer used) underperformed sum-mean on the hard split. GIN's theoretical
   expressiveness did not translate to wins either.
4. **Build it from scratch once.** The PyG version makes sense precisely
   because you already know what `propagate()` is doing under the hood.

## References

- Gilmer, Schoenholz, Riley, Vinyals, Dahl (2017). *Neural Message Passing for Quantum Chemistry.* ICML.
- Delaney (2004). *ESOL: Estimating Aqueous Solubility Directly from Molecular Structure.* J. Chem. Inf. Comput. Sci.
- Bemis & Murcko (1996). *The Properties of Known Drugs. 1. Molecular Frameworks.* J. Med. Chem.
- Vinyals, Bengio, Kudlur (2016). *Order Matters: Sequence to Sequence for Sets.* ICLR. (Set2Set)
- Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.
- Velickovic et al. (2018). *Graph Attention Networks.* ICLR.
- Xu, Hu, Leskovec, Jegelka (2019). *How Powerful are Graph Neural Networks?* ICLR.

## License

MIT. Learning project; code prioritizes clarity over performance or polish.
