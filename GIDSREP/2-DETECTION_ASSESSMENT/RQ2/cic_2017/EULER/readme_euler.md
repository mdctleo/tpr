# EULER on CIC-IDS 2017 — Pipeline Documentation

> **Model**: EULER (Embeddings from Uncertain Links in Evolving Relationships)  
> **Dataset**: CIC-IDS 2017 (Canadian Institute for Cybersecurity)  
> **Framework**: PyTorch + PyTorch Geometric + Distributed RPC  
> **Task**: Temporal link-prediction / link-detection for network anomaly detection

---

## ⚠️ Current Status (as of 2026-04-22)

### CIC-IDS 2017 (main benchmark)
- **Not yet executed.** No CIC-IDS 2017 result files exist (`cic2017_*.txt` / `predictions_latest.pkl`).
- The job script (`myjobs/projects/default/gidsrep_cic2017/main_job.sh`) is ready and configured for all three configs (baseline, KDE-decode, reduced), but the SLURM job has not been submitted.
- Data is in place: `cic2017/` → `euler/` (4 slice files + nmap.pkl ✅) and `cic2017_red/` → `euler_red/` (4 slice files + nmap.pkl ✅).
- `kde_vectors_euler.pkl` exists (~1 MB, DPGMM 20-dim vectors, precomputed from training period). ✅

### HyperVision attacks (43 attacks × 3 configs = 129 runs)
- **All 129 runs failed** — zero successful completions.
- Every log in `results/hv_*.log` ends with the same `ZeroDivisionError`:
  ```
  File "spinup.py", line 578, in run_all
      max_workers = int((tr_end-tr_start) // delta)
  ZeroDivisionError: integer division or modulo by zero
  ```
- **Root cause**: the HV branch in `run.py` converts `args.delta = int(args.delta)`.
  The default `--delta 0.5` (seconds) gets truncated to `0` via `int(0.5) == 0`.
  For CIC-IDS 2017 this is fine because that branch uses `int(delta * 60)` (e.g. `0.5 × 60 = 30`),
  but HV uses direct microsecond integers so the fix is to multiply by the correct scale factor
  (1,000,000 for microseconds) or require `--delta` to be specified as an integer number of
  microseconds when `--attack` / `--dataset HV_*` is used.

### `loaders/load_hv.py` — source file missing
- The source `loaders/load_hv.py` has been **deleted**; only the compiled bytecode
  `loaders/__pycache__/load_hv.cpython-38.pyc` remains.
- HV runs currently import from `__pycache__` (Python falls back to `.pyc` when `.py` is absent),
  so the import works in the existing environment but the code cannot be read, modified, or
  re-used in a different Python version without reconstructing the file.

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Two Implementations](#2-two-implementations)  
3. [Distributed Architecture](#3-distributed-architecture)  
4. [Dataset & Data Files](#4-dataset--data-files)  
5. [Data Loading Pipeline](#5-data-loading-pipeline)  
6. [Graph Construction & Snapshots](#6-graph-construction--snapshots)  
7. [Model Architecture](#7-model-architecture)  
8. [Training Loop](#8-training-loop)  
9. [Cutoff Calibration](#9-cutoff-calibration)  
10. [Testing & Evaluation](#10-testing--evaluation)  
11. [Hyperparameters & CLI Arguments](#11-hyperparameters--cli-arguments)  
12. [Baseline Results on CIC-IDS 2017](#12-baseline-results-on-cic-ids-2017)  
13. [File Reference](#13-file-reference)  
14. [Data Field Reference](#14-data-field-reference)  
15. [Key Differences from ARGUS](#15-key-differences-from-argus)  
16. [KDE Edge Feature Integration Points](#16-kde-edge-feature-integration-points)  
17. [KDE & Reduced Graph Implementation](#17-kde--reduced-graph-implementation)

---

## 1. Overview

EULER frames network intrusion detection as a **temporal link-prediction / link-detection** problem on dynamic graphs. Like ARGUS, it builds temporal graph snapshots, encodes them with GNNs, and tracks temporal dynamics with an RNN — but with a fundamentally different execution model:

1. **Build temporal graph snapshots** — partition network flows into fixed-width time windows (deltas), where nodes are network entities and aggregated flows are edges.
2. **Distributed encoding** — multiple worker processes (via PyTorch Distributed RPC) each hold a subset of snapshots and encode them independently using a shared-parameter GCN/GAT/SAGE.
3. **Temporal modeling** — a master process runs a GRU/LSTM on the sequence of embeddings streamed back from workers.
4. **Edge scoring** — dot-product decode: `σ(z_src · z_dst)`. High score = expected (benign); low score = anomalous.
5. **Threshold-based detection** — optimal cutoff learned on validation data.

**Loss function**: Standard BCE (Binary Cross Entropy) with negative sampling, optimized with Adam via `DistributedOptimizer`.

**Key architectural feature**: EULER uses **PyTorch Distributed RPC** with multiple worker processes. Each worker holds a GCN wrapped in DDP (DistributedDataParallel), allowing parallel processing of graph snapshots across workers.

---

## 2. Two Implementations

EULER supports two paradigms selected by the `--impl` argument:

| Impl | `--impl` | Class | Edge Scoring Strategy | Description |
|------|----------|-------|-----------------------|-------------|
| **Detector** | `DETECT` / `D` | `DetectorRecurrent` | Score `E_t` using `z_t` | Static link detection — reconstruct current snapshot |
| **Predictor** | `PREDICT` / `P` / `PRED` | `PredictorRecurrent` | Score `E_t` using `z_{t-1}` | Dynamic link prediction — predict next snapshot |

### Detector (Default)
- For snapshot `t`, uses `z_t` (the encoding of snapshot `t` itself) to score edges in `E_t`.
- Tests if the model can **reconstruct** the current graph.
- Anomalous edges are those that deviate from the learned reconstruction.

### Predictor
- For snapshot `t`, uses `z_{t-1}` (the encoding of the *previous* snapshot) to score edges in `E_t`.
- Tests if the model can **predict** the next graph from the current one.
- Worker 0 is the "head" and skips `E_0` (no prior embedding exists for it).
- A dummy `z_{-1} = zeros` is prepended to align embeddings with edge lists.

### Three Encoder Options

| Encoder | `--encoder` | Conv Layer | Edge Weight Support | Description |
|---------|-------------|------------|---------------------|-------------|
| **GCN** | `GCN` | `GCNConv` | ✅ Yes | 2-layer GCN with DropEdge(0.8), ReLU+Dropout |
| **GAT** | `GAT` | `GATConv(heads=3)` | ❌ No | 2-layer GAT, no edge weights, DropEdge on ei only |
| **SAGE** | `SAGE` | `PoolSAGEConv` | ❌ No | Custom maxpool GraphSAGE implementation |

All three encoders share the same 2-layer architecture and inherit from the `GCN` base class.

---

## 3. Distributed Architecture

EULER uses a **master-worker** pattern with PyTorch Distributed RPC:

```
                    ┌─────────────────────────────┐
                    │      Master (rank=N)         │
                    │  ┌─────────────────────┐     │
                    │  │   GRU / LSTM         │     │
                    │  │   (temporal model)    │     │
                    │  └────────┬────────────┘     │
                    │           │                   │
                    │    Receives z_t from workers  │
                    │    Runs RNN sequentially      │
                    │    Computes loss / scores     │
                    └─────┬─────┬─────┬────────────┘
                          │     │     │
                  ┌───────┘     │     └───────┐
                  ▼             ▼             ▼
          ┌──────────┐  ┌──────────┐  ┌──────────┐
          │ Worker 0  │  │ Worker 1  │  │ Worker 2  │
          │ (rank=0)  │  │ (rank=1)  │  │ (rank=2)  │
          │           │  │           │  │           │
          │ GCN + DDP │  │ GCN + DDP │  │ GCN + DDP │
          │ Snapshots │  │ Snapshots │  │ Snapshots │
          │  0..k-1   │  │  k..2k-1  │  │ 2k..3k-1  │
          └──────────┘  └──────────┘  └──────────┘
```

- **Workers** (rank 0..N-1): Each loads a partition of the snapshots and holds a GCN encoder wrapped in DDP. Workers share parameters via DDP's gradient synchronization.
- **Master** (rank N): Holds the RNN and orchestrates training. Uses `DistributedOptimizer` for gradient updates.
- **Communication**: Via `rpc.rpc_sync` / `rpc.rpc_async` for method calls, DDP for parameter sync.
- **Default**: 4 workers + 1 master = 5 processes (configurable via `-w`).

---

## 4. Dataset & Data Files

### CIC-IDS 2017 (Preprocessed — euler/ format)

| Metric | Value |
|--------|-------|
| Total flows | 2,830,742 |
| Unique nodes | 19,129 |
| Benign flows | 2,273,096 (80.3%) |
| Attack flows | 557,646 (19.7%) |
| Timestamp range | 0 – 374,762 seconds |
| Train/test split | `DATE_OF_EVIL_LANL = 29,136` |
| Validation | Last 5% of training time |

> EULER uses the `euler/` data format where nodes are **IP addresses** (19,129 unique). The `euler/` format differs from `argus_flow/` only in the flow-feature columns (Duration, Bytes, Packets are raw per-flow values rather than the aggregated totals used by ARGUS).

### File Layout

```
cic2017/                          (local copy of euler/)
├── 0.txt          (749,002 lines)
├── 100000.txt     (894,966 lines)
├── 200000.txt     (483,529 lines)
├── 300000.txt     (703,245 lines)
└── nmap.pkl       (list of 19,129 IP-address strings)
```

Files split at `FILE_DELTA = 100,000` timestamp boundaries. The `cic2017/` directory is a local copy of `GIDSREP/1-DATA_PROCESSING/cic_2017/euler/`.

---

## 5. Data Loading Pipeline

### Line Format

`ts,src,dst,field3,field4,field5,label` (7 comma-separated fields)

The loader only parses 4 of the 7 fields:
```python
fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[-1][:-1]))
```
→ `(timestamp, source_node, destination_node, label)` — **fields at indices 3, 4, 5 are IGNORED**.

### Edge Construction Per Snapshot

```python
def add_edge(et, is_anom=0):
    if et in edges_t:
        val = edges_t[et]
        edges_t[et] = (max(is_anom, val[0]), val[1]+1)
    else:
        edges_t[et] = (is_anom, 1)
```

For each directed `(src, dst)` pair within a delta window:
- **Deduplicates** to one edge per unique `(src, dst)`
- Tracks: `(max_label, count)` — label = max of all labels for that edge, count = number of flows
- Edge **weight** = count of duplicate flows (later standardized → sigmoid)
- **No edge attributes** — only edge index (`ei`) and edge weight (`ew`)
- Self-loops are filtered out

### Parallel Loading

Unlike ARGUS (which loads sequentially), EULER distributes loading across multiple jobs using `joblib.Parallel`:
```python
datas = Parallel(n_jobs=workers, prefer='processes')(
    delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers)
)
```
Each job loads a time-range partition, then results are joined via `data_reduce`.

### Data Object: `TData`

The loader produces a `TData` object (extends PyG `Data`) containing:

| Field | Type | Description |
|-------|------|-------------|
| `eis` | `list[Tensor[2, E_t]]` | Edge indices per timestep `t` |
| `xs` | `Tensor[N, N]` | Node features = identity matrix (one-hot, `N = 19,130`) |
| `ews` | `list[Tensor[E_t]]` | Edge weights per timestep (standardized flow counts) |
| `ys` | `list[Tensor[E_t]]` or `None` | Edge labels per timestep (only during testing) |
| `masks` | `list[Tensor[E_t]]` | Train/val split masks (95%/5% random split per timestep) |
| `cnt` | `list[Tensor[E_t]]` | Raw edge counts (pre-standardization) |
| `T` | `int` | Number of timesteps |

> **Key difference from ARGUS**: EULER's TData has **no `eas` (edge attributes)** — only `eis`, `ews`, `ys`, `masks`, `cnt`. There is no flow feature processing whatsoever.

---

## 6. Graph Construction & Snapshots

```
Time axis:  0 ─────────────── 29,136 ────────────────── 374,762
            │    TRAINING      │    TESTING              │
```

**Snapshot creation** (`delta` parameter):
- Default `--delta 0.5` → `delta = 0.5 × 60 = 30 seconds`
- CIC experiment: `--delta 10` → `delta = 10 × 60 = 600 seconds`
- All flows within a delta window are aggregated into a single graph snapshot
- Self-loops filtered out

**Train/Val split**: Last 5% of training time (same as ARGUS):
```python
val = max((tr_end - tr_start) // 20, delta*2)
val_start = tr_end - val
```

**Worker partitioning**: Snapshots are divided across workers:
- Each worker gets `ceil(total_snapshots / num_workers)` snapshots
- Workers load their partition independently, in parallel

---

## 7. Model Architecture

### 7a. GCN Encoder (Default)

```
Input: x ∈ ℝ^(N × N)  (identity matrix, one-hot node features, N=19,130)
       ei ∈ ℤ^(2 × E)  (edge index)
       ew ∈ ℝ^E          (edge weights)

DropEdge(0.8):  randomly drop 80% of edges during training
Layer 1:  GCNConv(N, h_dim=32)  + self-loops, edge_weight=ew
          → ReLU → Dropout(0.25)
Layer 2:  GCNConv(32, z_dim=16) + self-loops, edge_weight=ew
          → Tanh

Output: z ∈ ℝ^(N × z_dim)  (node embeddings per snapshot)
```

- **2 layers** (vs ARGUS's 4 layers for Argus_OPTC or 3+NNConv for Argus_LANL)
- **DropEdge(0.8)** — drops 80% of edges (more aggressive than ARGUS's 0.5)
- **No edge attributes** — only uses edge index and edge weight
- Each worker holds an identical-architecture GCN, synchronized via DDP

### 7b. GAT Encoder

```
Input: x ∈ ℝ^(N × N), ei ∈ ℤ^(2 × E)

DropEdge(0.8):  drop edges (edge index only, no weights)
Layer 1:  GATConv(N, 32, heads=3, concat=False)
          → ReLU → Dropout(0.25)
Layer 2:  GATConv(32, 16, heads=3, concat=False)
          → Tanh

Output: z ∈ ℝ^(N × 16)
```

- Does **NOT** use edge weights (GAT computes its own attention weights)
- 3 attention heads, averaged (concat=False)

### 7c. SAGE Encoder (Custom PoolSAGEConv)

```
Input: x ∈ ℝ^(N × N), ei ∈ ℤ^(2 × E)

Layer 1:  PoolSAGEConv(N, 32)   — maxpool aggregation
Layer 2:  PoolSAGEConv(32, 16)  — maxpool aggregation
          → Tanh

Output: z ∈ ℝ^(N × 16)
```

`PoolSAGEConv` is a custom implementation:
```python
class PoolSAGEConv(MessagePassing):
    aggr = 'max'
    x_e = Linear(in, out) → ReLU → propagate → Linear(out, out)
    x_r = Linear(in, out)
    output = L2_normalize(x_r + x_e)
```

### 7d. Temporal Model (GRU/LSTM)

Per-snapshot GCN embeddings stream through a GRU:

```
For each snapshot t (from each worker sequentially):
    z_t = GCN(snapshot_t)           # ℝ^(N × 32) or ℝ^(N × 16)
    h_t = GRU(z_t, h_{t-1})        # ℝ^(N × 32)
    z_final_t = Linear(h_t)         # ℝ^(N × 16)
```

```
GRU: input_dim=32 → hidden_dim=32 → Linear(32→16)
     Dropout(0.25) on input
```

### 7e. Edge Scoring (Decode)

For an edge `(src, dst)` at time `t`:

$$\text{score}(src, dst, t) = \sigma\left(\sum_{d=1}^{16} z_t[src]_d \cdot z_t[dst]_d\right)$$

- **High score** → model "expected" this edge → benign
- **Low score** → model did NOT expect this edge → potentially anomalous

---

## 8. Training Loop

### Loss: Binary Cross Entropy (BCE) with Negative Sampling

```python
def bce(self, t_scores, f_scores):
    EPS = 1e-6
    pos_loss = -torch.log(t_scores + EPS).mean()
    neg_loss = -torch.log(1 - f_scores + EPS).mean()
    return (pos_loss + neg_loss) * 0.5
```

- **Optimizer**: `DistributedOptimizer(Adam, lr=0.01)` — wraps Adam for distributed RPC
- **Gradient propagation**: via `dist_autograd.backward()` — automatically handles gradients across RPC boundaries
- **Negative sampling ratio**: `nratio=10` (default) — 10× more negative edges than positive per snapshot
- Loss is computed per-snapshot on each worker, then averaged across workers

### Training Flow

```
for epoch in range(epochs):
    with dist_autograd.context():
        zs = model.forward(TRAIN)                    # Workers encode → master RNN
        loss = model.loss_fn(zs, TRAIN, nratio=10)   # Workers compute BCE
        dist_autograd.backward(loss)                  # Distributed backward pass
        dist_optimizer.step()                         # Distributed parameter update

    # Validation (no_grad)
    zs = model.forward(TRAIN, no_grad=True)
    p, n = model.score_edges(zs, VAL)
    auc, ap = get_score(p, n)
    # Early stopping with patience
```

---

## 9. Cutoff Calibration

After training, the optimal decision threshold is found on validation data:

```python
def get_cutoff(model, h0, times, kwargs, lambda_param):
    # Load validation data onto worker 0
    Encoder.load_new_data(model.gcns[0], LOAD_FN, val_kwargs)
    # Forward pass through worker 0 → RNN
    zs = Encoder.forward(model.gcns[0], TData.ALL, True)
    zs, h0 = model.rnn(zs, h0, include_h=True)
    # Score edges
    p, n = Encoder.score_edges(model.gcns[0], zs, TData.ALL, val_nratio)
    # Optimal cutoff
    model.cutoff = get_optimal_cutoff(p, n, fw=lambda_param)
```

The `fpweight` parameter (default 0.6) controls TPR vs FPR trade-off:
- Higher → penalizes FP more → fewer alerts, lower TPR
- Lower → penalizes FN more → more alerts, higher TPR

---

## 10. Testing & Evaluation

```python
# 1. Distribute test data across all workers
# 2. Forward pass: Workers encode → Master RNN
# 3. Score ALL edges: decode(ei, z) for every snapshot on each worker
# 4. Classify: score ≤ cutoff → anomalous (1), else → benign (0)
# 5. Compute metrics
```

**Score inversion**: `scores = 1 - scores` before computing AUC/AP (same as ARGUS).

### Metrics Reported

| Metric | Description |
|--------|-------------|
| AUC | Area Under ROC Curve |
| AP | Average Precision (AUPRC) |
| F1 | F1 Score at learned cutoff |
| TPR | True Positive Rate (Recall) |
| FPR | False Positive Rate |
| TP/FP | True/False Positive counts |
| FwdTime | Forward pass inference time (seconds) |
| TPE | Time Per Epoch during training (seconds) |

---

## 11. Hyperparameters & CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `LANL` | Dataset. `cic_2017` or `CIC_2017` for CIC-IDS. Must start with `c` |
| `--delta` / `-d` | `0.5` | Snapshot width in minutes (×60 → seconds) |
| `--encoder` / `-e` | `GCN` | Encoder: `GCN`, `GAT`, or `SAGE` |
| `--rnn` / `-r` | `GRU` | Temporal model: `GRU`, `LSTM`, or `NONE` |
| `--impl` / `-i` | `DETECT` | Implementation: `DETECT` or `PREDICT` |
| `--hidden` / `-H` | `32` | GCN hidden dimension |
| `--zdim` / `-z` | `16` | Final embedding dimension |
| `--lr` | `0.01` | Learning rate for Adam optimizer |
| `--fpweight` | `0.6` | FP weight for cutoff optimization (0–1) |
| `--patience` | `10` | Early stopping patience (epochs without improvement) |
| `--workers` / `-w` | `4` | Number of distributed worker processes |
| `--threads` / `-T` | `1` | Threads per worker |
| `--tests` / `-t` | `1` | Number of independent runs (for averaging) |
| `--ngrus` / `-n` | `1` | Number of stacked GRU layers |
| `--load` / `-l` | `False` | Load pre-trained model |
| `--nowrite` | `False` | Don't write results to file |

### Internal Training Defaults (hardcoded in `DEFAULT_TR`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | `100` | Maximum training epochs |
| `nratio` | `10` | Negative sampling ratio (training) |
| `val_nratio` | `1` | Negative sampling ratio (validation) |
| `min` | `1` | Minimum epochs before early stopping kicks in |

### Baseline Command (from `cic.sh`)

```bash
python run.py -t 5 -d 10 -e GCN -w 6 --lr 0.01 --fpweight 0.48 --dataset cic_2017
```

---

## 12. Baseline Results on CIC-IDS 2017

> **Status: No results yet.** The CIC-IDS 2017 benchmark job (`main_job.sh`) has not been submitted.
> The table below is the target from earlier exploratory runs (pre-GIDSREP integration).
> It will be replaced with actual results once the SLURM job completes.

### GCN + GRU (Detector, delta=10min, 5 runs) — **target / reference only**

| Metric | Mean | Stderr |
|--------|------|--------|
| AUC | 0.6953 | ±0.0116 |
| AP | 0.3708 | ±0.0034 |
| F1 | 0.4078 | ±0.0086 |
| TPR | 0.6016 | ±0.0035 |
| FPR | 0.2689 | ±0.0132 |
| TP | 167,966 | ±973 |
| FP | 378,186 | ±18,504 |
| TPE | 92.8s | ±0.52 |
| FwdTime | 528.2s | ±6.01 |

### HyperVision attacks — **all failed, ZeroDivisionError**

All 129 runs (43 attacks × baseline / kde / red) produced the same crash in
`spinup.py` because `int(0.5) == 0` for the HV delta (see [⚠️ Current Status](#️-current-status-as-of-2026-04-22)).
No metric data is available.

---

## 13. File Reference

| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `run.py` | 407 | ✅ | Entry point, CLI args, dataset routing, process spawning. Extended with `--kde`, `--kde_decode`, `--red`, `--attack`, `--kde_dim`, `--data_dir`. |
| `spinup.py` | 629 | ✅ | Distributed process management, train/val/test orchestration, `build_raw_predictions()`, `_save_predictions()`. |
| `models/embedders.py` | 291 | ✅ | GCN (with NNConv when `ea_dim>0`), GAT, SAGE encoders + DropEdge + PoolSAGEConv. |
| `models/euler_interface.py` | 431 | ✅ | Abstract classes: `Euler_Embed_Unit`, `Euler_Encoder` (DDP), `Euler_Recurrent`. |
| `models/euler_detector.py` | 308 | ✅ | `DetectorEncoder` (KDE-decode MLP + `_get_kde_vecs`) + `DetectorRecurrent`. `decode_all_with_edges()` added for per-edge tracking. |
| `models/euler_predictor.py` | ~270 | ✅ | `PredictorEncoder` + `PredictorRecurrent` (future link prediction, mirrors detector). |
| `models/recurrent.py` | 123 | ✅ | GRU, LSTM, EmptyModel (no-op temporal model). |
| `models/serial_models.py` | 617 | ❌ not used | VGAE, GraphGRU, EvolveGCN baselines (not used in CIC experiments). |
| `models/utils.py` | 42 | ✅ | RPC utility methods (`_remote_method`, `_param_rrefs`). |
| `loaders/load_cic.py` | 321 | ✅ | Data loader for `cic2017/` (euler/ format). Supports `kde_file`, `use_flows`, `kde_decode` flags. Builds `eas` (NNConv mode) or `kde_dict` (decode mode). |
| `loaders/tdata.py` | 173 | ✅ | `TData` class — `ei_masked`, `ew_masked`, `ea_masked`, `get_kde_tensor()`. |
| `loaders/load_utils.py` | ~90 | ✅ | Edge weight standardization, train/val split. |
| `loaders/load_hv.py` | — | ⚠️ **SOURCE MISSING** | HyperVision data loader and `configure_euler_for_hv()`. Source `.py` deleted; only compiled bytecode `__pycache__/load_hv.cpython-38.pyc` remains. Works in current env only. |
| `hv_attacks_config.py` | 481 | ✅ | Auto-generated per-attack constants for all 43 HyperVision attacks (`date_of_evil`, `time_range`, `unique_ips`, `pikachu_snapshot_delta`). |
| `compute_kde_features_euler.py` | 234 | ✅ | Pre-computes DPGMM density vectors from per-edge timestamp differences. Output: `kde_vectors_euler.pkl`. |
| `generate_reduced_euler.py` | 166 | ✅ | Two-pass rewriter: replaces `(dur, bytes, pkts)` columns with `(first_ts, last_ts, count)` for every directed edge. Output: `euler_red/`. |
| `utils.py` | ~80 | ✅ | Scoring utilities (AUC, AP, optimal cutoff, F1). |
| `cic.sh` | 1 | ✅ | One-liner baseline run command (`run.py -t 5 -d 10 -e GCN -w 6 ...`). |
| `cic2017/` | — | ✅ symlink | → `1-DATA_PROCESSING/cic_2017/euler/` (4 slice files + `nmap.pkl`). |
| `cic2017_red/` | — | ✅ symlink | → `1-DATA_PROCESSING/cic_2017/euler_red/` (4 slice files + `nmap.pkl`). |
| `kde_vectors_euler.pkl` | — | ✅ ~1 MB | Pre-computed DPGMM KDE edge features (20-dim, training period only). |
| `results/hv_*.log` | — | ❌ all failed | 129 log files from HV attack runs, all crashing with `ZeroDivisionError`. |

---

## 14. Data Field Reference

### cic2017/ (copy of euler/)

**Sample line**: `0,0,1,1.0,12.0,2.0,0`

| Index | Field | Type | Used by Loader | Description |
|-------|-------|------|----------------|-------------|
| 0 | `ts` | int | ✅ Yes | Timestamp (seconds from start) |
| 1 | `src` | int | ✅ Yes | Source node ID (mapped from IP address) |
| 2 | `dst` | int | ✅ Yes | Destination node ID (mapped from IP address) |
| 3 | `dur` | float | ❌ Ignored | Flow Duration |
| 4 | `bytes` | float | ❌ Ignored | Total Length of Fwd Packets |
| 5 | `pkts` | float | ❌ Ignored | Total Fwd Packets |
| 6 | `label` | int | ✅ Yes | 0 = benign, 1 = attack |

### nmap.pkl

- Type: `list` of 19,129 strings (IP addresses)
- Example: `['8.254.250.126', '192.168.10.5', '8.253.185.121', ...]`
- Used to map integer node IDs back to original IP addresses
- Identical node mapping as `argus_flow/nmap.pkl`

---

## 15. Key Differences from ARGUS

| Feature | EULER | ARGUS |
|---------|-------|-------|
| **Execution model** | Distributed RPC (multi-process workers) | Single-process, sequential |
| **Loss function** | BCE with negative sampling | APLoss (libauc) + SOAP optimizer |
| **Encoder variants** | GCN, GAT, SAGE (2-layer each) | Argus_OPTC (4×GCN), Argus_LANL (3×GCN + NNConv) |
| **Edge attributes** | ❌ None — only edge weights | ✅ 6-dim flow features (L_cic_flow config) |
| **DropEdge** | 0.8 (drops 80%) | 0.5 (drops 50%) |
| **Negative sampling ratio** | 10 (training) | 1 |
| **Data format** | euler/ (IP nodes) | euler/ or argus_flow/ (IP nodes) |
| **Node count** | 19,129 (IPs) | 19,129 (O_cic and L_cic_flow) |
| **Parallel data loading** | ✅ joblib Parallel | ❌ Sequential |
| **Multiple test runs** | ✅ `-t 5` runs and averages | ❌ Single run |
| **Predictor mode** | ✅ `--impl PREDICT` | ❌ Detector only |
| **Data directory** | Local `cic2017/` | Reads from `--data_dir` |

---

## 16. KDE Edge Feature Integration Points

EULER currently uses **no edge attributes at all** — only edge index and edge weight (flow count). This means KDE vectors must be introduced as a new signal. Below are the viable integration points, from least to most invasive:

### Option A: Add NNConv Layer to GCN Encoder (Recommended)

Replace the final `GCNConv` layer with an `NNConv` layer that accepts KDE vectors (and optionally reduced-graph features) as `edge_attr`:

**Current GCN architecture** (2 layers, both GCNConv):
```python
# Layer 1: GCNConv (unchanged)
x = self.c1(x, ei, edge_weight=ew)   # GCNConv(N, 32)
x = self.relu(x)
x = self.drop(x)

# Layer 2: GCNConv → replace with NNConv
x = self.c2(x, ei, edge_weight=ew)   # GCNConv(32, 16)
x = self.tanh(x)
```

**Modified** (Layer 2 becomes NNConv):
```python
# Layer 1: GCNConv (unchanged)
x = self.c1(x, ei, edge_weight=ew)
x = self.relu(x)
x = self.drop(x)

# Layer 2: NNConv with KDE edge attributes
x = self.c2(x, ei, edge_attr=ea)     # NNConv(32, 16, nn_kde)
x = self.tanh(x)
```

Where `nn_kde` is:
```python
ea_dim = data.ea_dim   # K (KDE only) or 6+K (flow + KDE)
nn_kde = nn.Sequential(
    nn.Linear(ea_dim, max(ea_dim, 8)), nn.ReLU(),
    nn.Linear(max(ea_dim, 8), h_dim * z_dim)
)
self.c2 = NNConv(h_dim, z_dim, nn_kde, aggr='mean')
```

**Changes required**:
1. **`loaders/load_cic.py`**: Add `kde_file` kwarg, load KDE pickle, look up KDE vector per edge at snapshot boundary, build `eas` list (like `load_cic_flow.py` does in ARGUS)
2. **`loaders/tdata.py`**: Add `eas` field and `ea_dim`, add `ea_masked()` method
3. **`models/embedders.py`**: In GCN constructor, conditionally create NNConv for layer 2 when `data.ea_dim > 0`. In `forward_once()`, pass `ea` to NNConv
4. **`run.py`**: Add `--kde`, `--kde_file`, `--kde_dim`, `--red` CLI args
5. **`spinup.py`**: Thread `kde_file` through all `kwargs`/`ld_args` dicts

**This is the same approach as ARGUS Option A** — the GCN first 1 layer encodes topology, then NNConv incorporates edge-level features.

### Option B: Edge Decode Scoring with KDE

Modify the decode function to incorporate KDE vectors into the edge scoring:

```python
# Current: pure dot-product
score = σ(z_src · z_dst)

# Modified: MLP on [z_src ⊙ z_dst || kde_vector]
score = MLP([z_src * z_dst, kde_vec])
```

**Changes**: `euler_interface.py` → `decode()`, plus a `kde_dict` lookup at decode time.

**Invasiveness**: Medium — changes the scoring function for all modes.

### Option C: 3-Layer GCN + NNConv (Additive)

Keep both GCNConv layers unchanged and **add a 3rd layer** (NNConv) on top:

```python
x = self.c1(x, ei, edge_weight=ew)   # GCNConv(N, 32) — topology
x = relu(x); x = drop(x)
x = self.c2(x, ei, edge_weight=ew)   # GCNConv(32, 32) — topology
x = relu(x); x = drop(x)
x = self.c3(x, ei, edge_attr=ea)     # NNConv(32, 16) — KDE features
x = tanh(x)
```

**Advantage**: Preserves the original GCN behavior exactly, adds KDE as a refinement.  
**Disadvantage**: Adds depth and parameters; may slow training.

### Option D: Node Feature Augmentation

Aggregate KDE stats of incident edges per node → replace/augment identity matrix `xs`.

**Invasiveness**: High — fundamentally changes input representation.

### Recommendation

**Start with Option A** (NNConv replacing layer 2) — it's the most direct analog of what was done for ARGUS, requires the fewest conceptual changes, and keeps the baseline intact when `--kde` is not set (NNConv with zero-dim edge attrs is equivalent to skipping it; or conditionally use GCNConv vs NNConv based on a flag).

---

## Architecture Diagram

```
  cic2017/*.txt               Distributed Workers              Master
 ┌────────────┐     ┌───────────────────────────────┐    ┌──────────────┐
 │ts,src,dst, │     │  Worker 0    Worker 1    ...  │    │              │
 │...,label   │────▶│  ┌───────┐  ┌───────┐        │    │   GRU/LSTM   │
 └────────────┘     │  │ GCN   │  │ GCN   │        │───▶│   (temporal)  │
                    │  │(DDP)  │  │(DDP)  │        │    │              │
   fmt_line →       │  │snap   │  │snap   │        │    │  z → decode  │
   (ts,src,dst,lbl) │  │0..k-1 │  │k..2k-1│        │    │  → score    │
                    │  └───────┘  └───────┘        │    │  → classify  │
                    └───────────────────────────────┘    └──────────────┘
                           │                                    │
                    Shared params                        loss / metrics
                    via DDP sync                         via dist_autograd
```

---

## 17. KDE & Reduced Graph Implementation

This section documents the changes made to integrate **KDE timestamp-diff edge features** (Option A from Section 16) and a **reduced-graph configuration** into the EULER pipeline, mirroring the approach used in ARGUS.

### 17.1 New Scripts

#### `compute_kde_features_euler.py`

Pre-computes DPGMM (BayesianGaussianMixture) density vectors from per-edge timestamp differences observed during the training period (`ts < 29136`).

- **Pass 1**: Scans all `euler/*.txt` files, collects per-edge `(src, dst)` timestamp lists (training period only).
- **Pass 2**: For each edge with > 10 timestamps, computes inter-arrival time differences, fits a `BayesianGaussianMixture` (n_components = min(10, #unique diffs)), evaluates on a K-point grid spanning min–max diff, L2-normalizes the result.
- **Output**: `kde_vectors_euler.pkl` — a pickle dict mapping `(src_int, dst_int) → np.ndarray(K,)`. Edges with ≤ 10 timestamps are skipped; the loader falls back to a zero vector for unseen edges.

```bash
python compute_kde_features_euler.py \
    --data_dir  /path/to/euler \
    --output    kde_vectors_euler.pkl \
    --kde_dim   20 \
    --tr_end    29136
```

#### `generate_reduced_euler.py`

Creates `euler_red/` — a reduced-feature version of `euler/` with the same row count (no deduplication) but per-flow features (dur, bytes, pkts) replaced by global edge statistics:

| Column  | Original (euler/) | Reduced (euler_red/) |
|---------|-------------------|----------------------|
| col 0   | ts                | ts (unchanged)       |
| col 1   | src               | src (unchanged)      |
| col 2   | dst               | dst (unchanged)      |
| col 3   | dur               | **first_ts**         |
| col 4   | bytes             | **last_ts**          |
| col 5   | pkts              | **count**            |
| col 6   | label             | label (unchanged)    |

Two-pass approach:
1. Scan all input files → build per-edge `(first_ts, last_ts, count)` across the entire dataset.
2. Re-read input, substitute columns 3–5, write to output directory. Also copies `nmap.pkl`.

```bash
python generate_reduced_euler.py \
    --input_dir  /path/to/euler \
    --output_dir /path/to/euler_red
```

### 17.2 Modified Files

#### `loaders/tdata.py`

| Change | Description |
|--------|-------------|
| `__init__` signature | Added `eas=None` parameter |
| `ea_dim` property | Computed as `eas[0].size(0)` if `eas` is not None, else `0`. Shape convention: `(feat_dim, num_edges)` per snapshot. |
| `ea_masked(enum, t)` method | New method. Returns masked edge attributes analogous to `ei_masked` / `ew_masked`. For TRAIN: `eas[t][:, masks[t]]`, for VAL: `eas[t][:, ~masks[t]]`, for TEST: `eas[t]` (full). Returns `None` when `eas` is None. |

#### `loaders/load_cic.py`

| Change | Description |
|--------|-------------|
| `import numpy` | Added `import numpy as np` |
| `load_lanl_dist()` | Added `kde_file=None, use_flows=False` params. Threads them into per-worker kwargs. After workers finish, collects `eas` via `data_reduce('eas')` and passes to `TData`. |
| `make_data_obj()` | Added `eas=None` param, forwards to `TData` constructor. |
| `load_partial_lanl()` | Major rewrite. (1) Loads KDE pickle if `kde_file` is provided. (2) Adds `fmt_flow` lambda to parse columns 3,4,5. (3) Per snapshot, when `use_flows` or `kde_dict` is present, builds per-edge feature vectors: 6-dim flow stats (mean/std of dur, bytes, pkts) + K-dim KDE vector. (4) Stores as `eas` list of tensors shaped `(feat_dim, num_edges)`. |

**Flow feature computation** (per snapshot, per edge):

```
feat = []
if use_flows:
    feat = [mean(dur), std(dur), mean(bytes), std(bytes), mean(pkts), std(pkts)]
if kde:
    feat = feat + kde_dict.get((src, dst), zeros(K))
```

#### `models/embedders.py`

| Change | Description |
|--------|-------------|
| Import | Added `NNConv` to `torch_geometric.nn` imports |
| `GCN.__init__` | After loading data, checks `self.data.ea_dim`. If > 0: creates `NNConv(h_dim, z_dim, nn_ea, aggr='mean')` where `nn_ea` is `Linear(ea_dim, max(ea_dim,8)) → ReLU → Linear(max(ea_dim,8), h_dim*z_dim)`. Sets `self.use_ea = True`. If 0: uses `GCNConv(h_dim, z_dim)` as before. Sets `self.use_ea = False`. |
| `GCN.forward_once` | (1) Calls `self.data.ea_masked(mask_enum, i)` to get edge attrs. (2) **DropEdge refactored**: instead of using the `DropEdge` module (which only handles ei+ew), applies the same boolean mask to `ei`, `ew`, and `ea` simultaneously. (3) Layer 2: if `use_ea and ea is not None`, calls `self.c2(x, ei, edge_attr=ea.t())` (NNConv expects `(E, F)`); else calls `self.c2(x, ei, edge_weight=ew)`. |

**Architecture when `ea_dim > 0`**:

```
Layer 1:  GCNConv(x_dim, 32)  →  ReLU  →  Dropout(0.25)
Layer 2:  NNConv(32, 16, nn_ea)  →  Tanh
                  ↑
              ea.t()  shape (E, ea_dim)
```

**Architecture when `ea_dim == 0`** (baseline, unchanged):

```
Layer 1:  GCNConv(x_dim, 32)  →  ReLU  →  Dropout(0.25)
Layer 2:  GCNConv(32, 16)     →  Tanh
```

> **Note**: GAT and SAGE encoders inherit from GCN but override `forward_once()` — they do not use edge attributes and were not modified. KDE/reduced features apply only to the GCN encoder.

#### `run.py`

| Change | Description |
|--------|-------------|
| New CLI args | `--data_dir` (str): override data directory. `--kde` (flag): enable KDE. `--kde_file` (str, default `kde_vectors_euler.pkl`): path to KDE pickle. `--kde_dim` (int, default 20): KDE vector size. `--red` (flag): enable reduced graph config. |
| Dataset routing | If `--data_dir` is set → override `LANL_FOLDER`. Elif `--red` → set `LANL_FOLDER = './cic2017_red/'`. `kde_file` is only passed when `--kde` is set. `use_flows` is set to `True` when `--red` is active. |
| `run_all()` call | Now passes `kde_file=args.kde_file, use_flows=args.use_flows` |

#### `spinup.py`

`kde_file` and `use_flows` are threaded through the entire distributed call chain:

```
run_all(kde_file, use_flows)
  → times dict: {'kde_file': ..., 'use_flows': ...}
  → mp.spawn(init_procs, args=(..., kde_file, use_flows))
    → init_workers(kde_file, use_flows)
      → get_work_units(kde_file, use_flows)
        → each worker kwargs: {'kde_file': ..., 'use_flows': ...}
    → init_empty_workers(kde_file, use_flows)
      → get_work_units(kde_file, use_flows)
  → get_cutoff()  — validation loader kwargs include kde_file/use_flows
  → test()        — test loader kwargs include kde_file/use_flows
```

All 8+ function signatures were updated to accept and forward these parameters.

### 17.3 Three Run Configurations

All runs use the same hyperparameters: GCN encoder, 6 workers, δ=10 min, lr=0.01, fpweight=0.48, 5 test repeats.

#### Pre-computation (run once before submitting jobs)

```bash
# 1. Compute KDE vectors (takes ~5–15 min depending on data size)
cd /path/to/EULER
python compute_kde_features_euler.py \
    --data_dir /path/to/euler \
    --output   kde_vectors_euler.pkl \
    --kde_dim  20 \
    --tr_end   29136

# 2. Generate reduced-graph data (takes ~2–5 min)
python generate_reduced_euler.py \
    --input_dir  /path/to/euler \
    --output_dir /path/to/euler_red
```

#### Run 1: Baseline (no edge features)

Standard EULER — 2-layer GCN, identity-matrix node features, no edge attributes.

```bash
python run.py \
    -t 5 -d 10 -e GCN -w 6 \
    --lr 0.01 --fpweight 0.48 \
    --dataset cic_2017
```

#### Run 2: KDE timestamp-diff edge features

GCN layer 2 replaced by NNConv. 20-dim DPGMM density vectors as edge attributes.

```bash
python run.py \
    -t 5 -d 10 -e GCN -w 6 \
    --lr 0.01 --fpweight 0.48 \
    --dataset cic_2017 \
    --kde \
    --kde_file kde_vectors_euler.pkl \
    --kde_dim 20
```

#### Run 3: Reduced graph (first_ts, last_ts, count)

GCN layer 2 replaced by NNConv. 6-dim flow features (mean/std of first_ts, last_ts, count per snapshot).

```bash
python run.py \
    -t 5 -d 10 -e GCN -w 6 \
    --lr 0.01 --fpweight 0.48 \
    --dataset cic_2017 \
    --red
```

### 17.4 Edge Attribute Dimensions Summary

| Configuration | `use_flows` | `kde_dict` | `ea_dim` | Feature composition |
|---------------|-------------|------------|----------|---------------------|
| Baseline      | False       | empty      | 0        | — (no edge attrs, GCNConv layer 2) |
| KDE           | False       | loaded     | 20       | `[kde_0 .. kde_19]` |
| Reduced       | True        | empty      | 6        | `[mean_dur, std_dur, mean_bytes, std_bytes, mean_pkts, std_pkts]` |
| KDE + Reduced | True        | loaded     | 26       | `[flow_0..5, kde_0..19]` (possible but not benchmarked) |

---

## 18. KDE at Decode Step — Implementation

### 18.2 Current Decode Architecture

EULER has a two-level decode chain:

```
Euler_Embed_Unit.decode(src, dst, z):         # Inner — lives on each worker
    return sigmoid( z[src] · z[dst] )

Euler_Encoder.decode(e, z):                   # Outer — DDP wrapper
    src, dst = e
    return self.module.decode(src, dst, z)
```

**Every scoring path** (training loss, validation cutoff, test evaluation) goes through these two methods. The decode sees only `z` embeddings — no edge features, no KDE.

**Call sites**:
1. `DetectorEncoder.calc_loss()` / `PredictorEncoder.calc_loss()` — training BCE loss
2. `DetectorEncoder.score_edges()` / `PredictorEncoder.score_edges()` — validation/test scoring
3. `DetectorRecurrent.get_cutoff()` — calibration on validation data
4. `DetectorRecurrent.test()` — final evaluation

### 18.3 Proposed Design: Option B (Additive KDE at Decode)

**Principle**: Keep the baseline dot-product path unchanged. Add a small MLP that sees
**only the KDE vector** and produces a scalar adjustment to the logit.

```
Baseline:   score = sigmoid( z_src · z_dst )

Option B:   score = sigmoid( z_src · z_dst  +  KDE_MLP(kde_vec) )
                                               └── Linear(K, K//2) → ReLU → Linear(K//2, 1)
```

**Why additive**:
- Dot-product discrimination is preserved — KDE cannot override structural signal
- MLP sees only KDE (not z) → cannot memorise link-prediction task
- Edges without KDE get `MLP(zeros)` = constant bias → same ranking as baseline
- ~100–200 parameters (vs thousands for NNConv)

### 18.4 Implementation Plan

#### Step 1: Modify `Euler_Embed_Unit` (or subclass) to hold KDE_MLP

**File**: `models/embedders.py` (or a new `models/euler_detector.py`)

```python
# In DetectorEncoder.__init__:
kde_dim = getattr(self.module.data, 'kde_dim', 0)
self._kde_dim = kde_dim
if kde_dim > 0:
    mlp_h = max(kde_dim // 2, 8)
    self.kde_mlp = nn.Sequential(
        nn.Linear(kde_dim, mlp_h),
        nn.ReLU(),
        nn.Linear(mlp_h, 1)
    )
```

#### Step 2: Modify `decode()` to accept and use KDE vectors

**File**: `models/embedders.py` — `Euler_Embed_Unit.decode`

```python
def decode(self, src, dst, z, kde_vecs=None):
    base_score = (z[src] * z[dst]).sum(dim=1)
    if kde_vecs is not None and self._kde_dim > 0:
        kde_adj = self.kde_mlp(kde_vecs).squeeze(1)
        return torch.sigmoid(base_score + kde_adj)
    return torch.sigmoid(base_score)
```

#### Step 3: Propagate KDE tensors through the scoring call chain

**Files affected**: `euler_detector.py`, `euler_predictor.py`

At each call site that invokes `decode()`, look up KDE vectors from `TData`:

```python
# In score_edges / calc_loss:
ei = data.ei_masked(enum, t)
src, dst = ei[0], ei[1]
kde_vecs = data.get_kde_tensor(src, dst)  # (E, K) or None
if kde_vecs is not None:
    kde_vecs = kde_vecs.to(z.device)
score = self.decode(src, dst, z, kde_vecs=kde_vecs)
```

For **negative edges** (used in training loss):
- Negative edges are randomly sampled → unlikely to have KDE vectors
- Pass `kde_vecs=None` or zero-vectors for negatives
- This is correct: negatives should score low regardless

#### Step 4: Remove KDE from encoder path

**File**: `models/embedders.py` — `GCN.__init__` and `forward_once`

- When using decode-step KDE, layer 2 should revert to plain `GCNConv` (no NNConv)
- `ea_dim` becomes 0 for the encoder; KDE is accessed only at decode time
- Edge attributes (`eas`) are no longer needed in the encoder forward pass

#### Step 5: Ensure `TData.get_kde_tensor()` is accessible at decode time

**File**: `loaders/tdata.py`

The existing `get_kde_tensor(src_t, dst_t)` method returns `(E, kde_dim)` float tensor.
This is already suitable for decode-time use. No changes needed if it exists; if not,
add the same implementation as ARGUS:

```python
def get_kde_tensor(self, src_t, dst_t):
    if self.kde_dim == 0 or self.kde_dict is None:
        return None
    zero = [0.0] * self.kde_dim
    vecs = [
        list(self.kde_dict.get((int(s), int(d)), zero))
        for s, d in zip(src_t.tolist(), dst_t.tolist())
    ]
    return torch.tensor(vecs, dtype=torch.float32)
```

### 18.5 Parameter Budget

| Component | Parameters | Notes |
|-----------|-----------|-------|
| KDE_MLP (K=20) | `20×10 + 10 + 10×1 + 1 = 221` | Linear(20,10) → ReLU → Linear(10,1) |
| KDE_MLP (K=10) | `10×8 + 8 + 8×1 + 1 = 97` | Linear(10,8) → ReLU → Linear(8,1) |
| Baseline GCN (no KDE) | ~614K | GCNConv layers + GRU + Linear |
| NNConv encoder KDE (current) | +2,000–14,000 | Depends on `ea_dim`; causes instability |

The decode-step MLP adds **< 0.04%** overhead to the total parameter count.

### 18.7 Recommended Approach

**Primary**: Option **B** (additive decode MLP) — proven viable in ARGUS (AUC improved 0.585 → 0.616, TN recovered from 0 to 185K).

**If B is insufficient**: Try Option **C** (gated decode) — the gate learns to suppress KDE contribution for edges where the zero-vector fallback would be misleading.

**Avoid**: Option **E** (concatenated MLP) — failed catastrophically in ARGUS (TN=0, all scores saturated to 1.0) because the MLP learned "KDE present → real edge → 1.0".

### 18.8 Key Design Decisions

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Fallback for unseen edges | Hard zeros (not learned) | Learned fallback caused saturation in ARGUS — the parameter optimised to push all unseen-edge scores high |
| KDE for negative edges | `None` (skip KDE adjustment) | Negatives are random; KDE lookup is meaningless and would add a constant bias |
| Encoder with KDE at decode | Revert to plain GCNConv (no NNConv) | Avoids parameter explosion and mostly-zero confusion in NNConv |
| KDE dimension | 20 (current) or 10 (reduced) | 10 reduces zero-padding impact; 20 has more information; test both |
| MLP depth | 2 layers (Linear → ReLU → Linear) | Sufficient for scalar adjustment; deeper MLPs risk overfitting with ~5K KDE edges |

### 18.9 Files to Modify (Summary)

| File | Change | For which option |
|------|--------|-----------------|
| `models/embedders.py` | Revert layer 2 to GCNConv when decode-KDE is active; add `_kde_dim` field | B, C, D |
| `models/euler_detector.py` | Add `kde_mlp` to `DetectorEncoder.__init__`; modify `decode()`, `calc_loss()`, `score_edges()` | B, C, D, E |
| `models/euler_predictor.py` | Same changes as detector | B, C, D, E |
| `loaders/tdata.py` | Ensure `get_kde_tensor()` exists (already present) | All |
| `loaders/load_cic.py` | Load KDE pickle but don't inject into `eas`; store on TData only | B, C, D, E |
| `spinup.py` | Pass `kde_decode=True` flag to distinguish encoder-KDE from decode-KDE | All |
| `main.py` / `run.py` | Add `--kde_decode` CLI flag | All |
---

## 18. Prediction Saving & Per-Attack-Type Analysis

### 18.1 Overview

Starting with the 6-config benchmark run, EULER now saves per-edge predictions
to a pickle file after each `test()` call. This enables post-hoc analysis of
which CIC-IDS 2017 attack types are detected (TP) vs missed (FN) by each config.

### 18.2 What Changed

| File | Change |
|------|--------|
| `models/euler_detector.py` | `decode_all()` returns 4-tuple `(preds, ys, cnts, eis_out)` — appends `ei.detach().cpu()` per snapshot. `DetectorRecurrent.score_all()` unpacks and propagates the edge-index list. |
| `spinup.py` | Added `PREDICTIONS_FILE = 'predictions_latest.pkl'` constant. Added `_save_predictions(scores, labels, weights, eis, cutoff)` helper that converts tensors → numpy and pickles them. `test()` now calls `scores, labels, weights, eis = model.score_all(zs)` then `_save_predictions(...)`. |

### 18.3 Predictions Pickle Format

```python
{
    'scores':  [np.array, ...],   # one per test snapshot
    'labels':  [np.array, ...],   # one per test snapshot
    'weights': [np.array, ...],   # one per test snapshot
    'eis':     [np.array(2,E), ...],  # edge indices per snapshot
    'cutoff':  float              # optimal cutoff from calibration
}
```

### 18.4 Three Run Configurations

| Config | CLI flags | Data dir | Description |
|--------|-----------|----------|-------------|
| **Baseline** | (none) | `cic2017/` | Standard EULER GCN+GRU |
| **KDE Decode** | `--kde_decode --kde_file kde_vectors_euler.pkl --kde_dim 20` | `cic2017/` | KDE features added at decode step via MLP |
| **Reduced** | `--red` | `cic2017_red/` | Reduced graph (fewer nodes/edges) |

All three use: `-t 5 -d 10 -e GCN -w 6 --lr 0.01 --fpweight 0.48 --dataset cic_2017`

### 18.5 Output Structure

After `main_job.sh` runs, results are saved to:

```
results/<timestamp>/
  EULER/
    baseline/
      cic2017_.txt          # mean±stderr metrics from 5 runs
      predictions.pkl        # per-edge predictions from last run
    kde_decode/
      cic2017_.txt
      predictions.pkl
    reduced/
      cic2017_.txt
      predictions.pkl
```

### 18.6 Post-Hoc Attack-Type Mapping

`common_analysis/map_attacks_to_cic_types.py --results_dir <path>` reads each
`predictions.pkl`, maps edge indices → IP addresses via `nmap.pkl`, joins against
the CIC-IDS 2017 master CSV to determine attack types, and computes per-type
TP/FP/TN/FN using the model's cutoff.

### 18.7 Current File Structure

```
EULER/
├── run.py                  # CLI entry point (--kde_decode, --kde, --red flags)
├── spinup.py               # Master process: train/test/predict + _save_predictions()
├── models/
│   ├── euler_detector.py   # DetectorEncoder (decode_all returns eis), DetectorRecurrent
│   ├── euler_predictor.py  # Predictor variant (mirror of detector)
│   └── recurrent.py        # GRU wrapper
├── loaders/
│   ├── load_cic.py         # CIC-IDS 2017 data loading (supports KDE + reduced)
│   ├── load_lanl_dist.py   # Distributed loading for LANL/CIC
│   └── tdata.py            # TData container with get_kde_tensor()
├── cic2017/                # Full graph data (0.txt ... 300000.txt + nmap.pkl)
├── cic2017_red/            # Reduced graph data
└── kde_vectors_euler.pkl   # Precomputed KDE edge features
```