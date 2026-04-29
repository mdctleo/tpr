# ARGUS on CIC-IDS 2017 — Pipeline Documentation

> **Model**: ARGUS (Adaptive Recurrent Graph-based Unsupervised System)  
> **Dataset**: CIC-IDS 2017 (Canadian Institute for Cybersecurity)  
> **Framework**: PyTorch + PyTorch Geometric + libauc  
> **Task**: Temporal link-prediction for network anomaly/intrusion detection

---

## Table of Contents

1. [Overview](#1-overview)  
2. [Two Configurations](#2-two-configurations)  
3. [Dataset & Data Files](#3-dataset--data-files)  
4. [Data Loading Pipeline](#4-data-loading-pipeline)  
5. [Graph Construction & Snapshots](#5-graph-construction--snapshots)  
6. [Model Architecture](#6-model-architecture)  
7. [Training Loop](#7-training-loop)  
8. [Cutoff Calibration](#8-cutoff-calibration)  
9. [Testing & Evaluation](#9-testing--evaluation)  
10. [Hyperparameters & CLI Arguments](#10-hyperparameters--cli-arguments)  
11. [Baseline Results on CIC-IDS 2017](#11-baseline-results-on-cic-ids-2017)  
12. [File Reference](#12-file-reference)  
13. [KDE Edge Feature Integration Points](#13-kde-edge-feature-integration-points)  
14. [Data Field Reference (euler/ vs argus_flow/)](#14-data-field-reference)

---

## 1. Overview

ARGUS frames network intrusion detection as a **temporal link-prediction** problem on dynamic graphs. The core idea:

1. **Build temporal graph snapshots** — partition network flows into fixed-width time windows (deltas), where IPs are nodes and aggregated flows are edges.
2. **Encode each snapshot** — a Graph Convolutional Network (GCN) learns node embeddings for each snapshot independently.
3. **Model temporal dynamics** — a Gated Recurrent Unit (GRU) processes the sequence of per-snapshot embeddings, capturing how the graph evolves over time.
4. **Score edges via link prediction** — for each edge `(src, dst)`, compute `σ(z_src · z_dst)` (dot-product decode). Edges that "should exist" (benign) score high; unexpected (anomalous) edges score low.
5. **Threshold-based detection** — a cutoff learned on validation data separates normal from anomalous edges.

**Loss function**: ARGUS uses **APLoss** (Average Precision Loss) from the `libauc` library, optimized with the **SOAP** (Stochastic Optimization of AP) optimizer — a specialized optimizer designed to directly maximize Average Precision, which is better suited for imbalanced classification than standard BCE.

---

## 2. Two Configurations

ARGUS runs on CIC-IDS 2017 in **two configurations**, selected by the `--dataset` argument:

| Config | `--dataset` | Data Directory | Loader | Encoder | Edge Attributes | Description |
|--------|-------------|---------------|--------|---------|-----------------|-------------|
| **Without Flow** | `O_cic` | `euler/` | `load_cic.py` | `Argus_OPTC` | None (edge weight only) | GCN-only, 4-layer, no flow features |
| **With Flow** | `L_cic_flow` | `argus_flow/` | `load_cic_flow.py` | `Argus_LANL` | 6-dim flow stats | GCN + NNConv with flow-derived edge attributes |

### How the dataset flag routes execution (`main.py`):
- `--dataset O_cic` → starts with `'O'` → uses `load_cic.py`, `Argus_OPTC` encoder, reads from `euler/`
- `--dataset L_cic_flow` → starts with `'L'` → uses `load_cic_flow.py`, `Argus_LANL` encoder, reads from `argus_flow/`

---

## 3. Dataset & Data Files

### CIC-IDS 2017 (Preprocessed)

| Metric | Value |
|--------|-------|
| Total flows | 2,830,742 |
| Unique nodes (IPs) | 19,129 |
| Benign flows | 2,273,096 (80.3%) |
| Attack flows | 557,646 (19.7%) |
| Timestamp range | 0 – 374,762 seconds |
| Train/test split | `DATE_OF_EVIL_LANL = 29,136` |
| Validation | Last 5% of training time |

### File Layout

Both `euler/` and `argus_flow/` contain the same file structure:

```
euler/                          argus_flow/
├── 0.txt          (749,002)    ├── 0.txt          (749,002)
├── 100000.txt     (894,966)    ├── 100000.txt     (894,966)
├── 200000.txt     (483,529)    ├── 200000.txt     (483,529)
├── 300000.txt     (703,245)    ├── 300000.txt     (703,245)
└── nmap.pkl                    └── nmap.pkl
```

Files are split at timestamp boundaries (`FILE_DELTA = 100,000`). Each `.txt` file contains one flow per line. `nmap.pkl` is a node-ID mapping dictionary.

---

## 4. Data Loading Pipeline

### 4a. Without Flow (`load_cic.py` → `euler/`)

**Line format**: `ts,src,dst,field3,field4,field5,label` (7 comma-separated fields)

> **Important**: Although `euler/*.txt` files contain 7 fields, `load_cic.py` only parses 4:
> ```python
> fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[-1][:-1]))
> ```
> → `(timestamp, source_node, destination_node, label)` — **fields at indices 3, 4, 5 are IGNORED**.

**Edge construction per snapshot**:
- `add_edge(et, is_anom)` deduplicates edges per (src, dst) pair
- Tracks: `(max_label, count)` — label = max(all labels for that edge), count = number of flows
- Edge **weight** = count of duplicate flows (later standardized → sigmoid)
- Edge **attributes** = None

### 4b. With Flow (`load_cic_flow.py` → `argus_flow/`)

**Line format**: `ts,src,dst,duration,bytes,packets,label` (7 fields, all used)

```python
fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]), int(x[5]), int(x[6][:-1]))
```

**Edge construction per snapshot**:
- Same deduplication: `add_edge(et, is_anom)` → `(max_label, count)`
- Additionally: `add_flow(et, dur, bytes, pkts)` → accumulates per-edge flow statistics
- At snapshot boundary, computes **6-dimensional edge attributes** per edge:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `mean_duration` | Mean of flow durations for this edge in the snapshot |
| 1 | `std_duration` | Std deviation of flow durations |
| 2 | `mean_pkt_cnt` | Mean of packet counts |
| 3 | `std_pkt_cnt` | Std deviation of packet counts |
| 4 | `mean_byte_cnt` | Mean of byte counts |
| 5 | `std_byte_cnt` | Std deviation of byte counts |

- Edges without flow data receive `[0, 0, 0, 0, 0, 0]` as fallback
- Edge attributes are standardized via `std_edge_a()` (standardize → sigmoid)

### 4c. Data Object: `TData`

Both loaders produce a `TData` object (extends PyG `Data`) containing:

| Field | Type | Description |
|-------|------|-------------|
| `eis` | `list[Tensor[2, E_t]]` | Edge indices per timestep `t` |
| `xs` | `Tensor[N, N]` | Node features = identity matrix (one-hot, `N = 19,129`) |
| `ews` | `list[Tensor[E_t]]` | Edge weights per timestep (standardized flow counts) |
| `eas` | `list[Tensor[6, E_t]]` or `None` | Edge attributes per timestep (only with flow) |
| `ys` | `list[Tensor[E_t]]` or `None` | Edge labels per timestep (only during testing) |
| `masks` | `list[Tensor[E_t]]` | Train/val split masks (95%/5% random split per timestep) |
| `cnt` | `list[Tensor[E_t]]` | Raw edge counts (pre-standardization) |
| `T` | `int` | Number of timesteps |

---

## 5. Graph Construction & Snapshots

```
Time axis:  0 ─────────────── 29,136 ────────────────── 374,762
            │    TRAINING      │    TESTING              │
            │ (benign only)    │ (benign + attacks)      │
            │                  │                         │
            └─ val = last 5% ─┘                         │
              of training time                          │
```

**Snapshot creation** (`delta` parameter):
- Default `--delta 15` (O_cic) → `delta = 15 × 60 = 900 seconds`
- Default `--delta 10` (L_cic_flow) → `delta = 10 × 60 = 600 seconds`
- All flows within a delta window are aggregated into a single graph snapshot
- Self-loops are filtered out

**Train/Val split**: The validation set is the last 5% of training time:
```python
val = max((tr_end - tr_start) // 20, delta*2)
val_start = tr_end - val
```

---

## 6. Model Architecture

### 6a. Argus_OPTC (Without Flow — O_cic)

```
Input: x ∈ ℝ^(N × N)  (identity matrix, one-hot node features)
       ei ∈ ℤ^(2 × E)  (edge index)
       ew ∈ ℝ^E          (edge weights)

Layer 1:  GCNConv(N, 32)     + self-loops
Layer 2:  GCNConv(32, 32)    + self-loops
          → ReLU → Dropout(0.1)
Layer 3:  GCNConv(32, 16)    + self-loops
          → ReLU → Dropout(0.1)
Layer 4:  GCNConv(32, 16)    + self-loops
          → Tanh

Output: z ∈ ℝ^(N × 16)  (node embeddings per snapshot)
```

- Uses **DropEdge(0.5)** on edge index/weights during training (randomly drops 50% of edges)
- Only uses edge index (`ei`) and edge weight (`ew`) — **no edge attributes**

### 6b. Argus_LANL (With Flow — L_cic_flow)

```
Input: x ∈ ℝ^(N × N)     (identity matrix)
       ei ∈ ℤ^(2 × E)     (edge index)
       ew ∈ ℝ^E            (edge weights)
       ea ∈ ℝ^(E × 6)     (6-dim flow edge attributes)

Layer 1:  GCNConv(N, 32)     + self-loops, edge_weight=ew
Layer 2:  GCNConv(32, 32)    + self-loops, edge_weight=ew
          → ReLU → Dropout(0.1)
Layer 3:  GCNConv(32, 16)    + self-loops, edge_weight=ew
          → ReLU → Dropout(0.1)
Layer 4:  NNConv(32, 16, nn4, aggr='mean')    ← uses ea (edge attributes!)
          where nn4 = Linear(6→8) → ReLU → Linear(8→32×16=512)
          → Tanh

Output: z ∈ ℝ^(N × 16)  (node embeddings per snapshot)
```

**Key difference**: The final layer is **NNConv** (Neural Network Convolution / Neural Message Passing) instead of GCNConv. NNConv uses a small neural network (`nn4`) to transform edge attributes into message-passing weights:

```python
nn4 = nn.Sequential(
    nn.Linear(6, 8),        # 6-dim flow features → 8
    nn.ReLU(),
    nn.Linear(8, h_dim * z_dim)  # 8 → 32×16 = 512 (weight matrix per edge)
)
self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')
```

This allows the model to learn **edge-attribute-conditioned message passing**: each edge gets a unique transformation matrix based on its flow statistics.

### 6c. Temporal Model (GRU)

The per-snapshot GCN embeddings are fed sequentially through a GRU:

```
For each snapshot t:
    z_t = GCN(snapshot_t)                     # ℝ^(N × 32) or ℝ^(N × 16)
    h_t, hidden = GRU(z_t.unsqueeze(0), h_{t-1})  # temporal encoding
    z_final_t = Linear(h_t)                    # ℝ^(N × 16)
```

```
GRU: input_dim=32 → hidden_dim=32 → Linear(32→16) → z_dim=16
     Dropout(0.25) on input
```

### 6d. Edge Scoring (Decode)

For an edge `(src, dst)` at time `t`:

$$\text{score}(src, dst, t) = \sigma\left(\sum_{d=1}^{16} z_t[src]_d \cdot z_t[dst]_d\right)$$

- **High score** → model "expected" this edge → benign
- **Low score** → model did NOT expect this edge → potentially anomalous

---

## 7. Training Loop

```
for epoch in range(epochs):
    # Forward pass through all training snapshots
    zs = model.forward(TRAIN)           # GCN → GRU for each snapshot
    
    # Compute APLoss (positive edges vs. negative samples)
    loss = model.loss_fn(zs, TRAIN)     # APLoss + SOAP optimizer
    loss.backward()
    optimizer.step()
    
    # Validation (no gradient)
    zs = model.forward(TRAIN, no_grad=True)
    p_scores, n_scores = model.score_edges(zs, VAL)
    auc, ap = get_score(p_scores, n_scores)
    
    # Early stopping: if AUC+AP doesn't improve for `patience` epochs, stop
    if auc + ap > best:
        save_model()
    elif epoch >= 1:
        no_progress += 1
        if no_progress == patience: break
```

### Loss Function: APLoss

ARGUS uses `APLoss` from `libauc` instead of standard BCE:

```python
ap_loss = APLoss(data_len=2830742, margin=0.8, gamma=0.1, surr_loss='squared_hinge')
```

- **margin=0.8**: minimum desired gap between positive and negative scores
- **surr_loss='squared_hinge'**: differentiable surrogate for 0-1 loss
- **Optimizer**: `SOAP(lr=..., mode='adam', weight_decay=0.0)` — stochastic optimization targeting AP directly

### Negative Sampling

For each positive edge in a snapshot, `nratio=1` negative edge is sampled randomly (ensuring it doesn't exist in the current edge list). The Kipf-Welling fast negative sampling method is used.

---

## 8. Cutoff Calibration

After training, the optimal decision threshold is found on validation data:

```python
def get_cutoff(model, h0, times, args, lambda_param, use_flows, device):
    # Load validation time window
    # Forward pass → score all validation edges
    p_scores, n_scores = score_edges(zs, ALL)
    
    # Find threshold that minimizes |tw*TPR - fw*(1-FPR)|
    # where fw = fpweight (default 0.45-0.46)
    cutoff = get_optimal_cutoff(p_scores, n_scores, fw=lambda_param)
```

The `fpweight` parameter controls the trade-off between TPR and FPR:
- Higher `fpweight` → penalizes false positives more → fewer alerts, lower TPR
- Lower `fpweight` → penalizes false negatives more → more alerts, higher TPR

---

## 9. Testing & Evaluation

```python
# 1. Load test data (timestamps 29,136 → 374,762)
# 2. Forward pass: GCN → GRU → per-snapshot embeddings
# 3. Score ALL edges: decode(src, dst, z) for every edge in every test snapshot
# 4. Classify: score ≤ cutoff → anomalous (1), else → benign (0)
# 5. Compute metrics
```

**Note on score inversion**: Scores are inverted (`scores = 1 - scores`) before computing AUC/AP because the decode gives **high scores to expected/benign edges** but `roc_auc_score` expects high scores for the positive (anomalous) class.

### Metrics Reported

| Metric | Description |
|--------|-------------|
| AUC | Area Under ROC Curve |
| AP | Average Precision (AUPRC) |
| F1 | F1 Score at learned cutoff |
| TPR | True Positive Rate (Recall) |
| FPR | False Positive Rate |
| TP/FP/TN/FN | Confusion matrix components |
| FwdTime | Forward pass inference time (seconds) |
| TPE | Time Per Epoch during training (seconds) |

---

## 10. Hyperparameters & CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `LANL` | `O_cic` (without flow) or `L_cic_flow` (with flow) |
| `--delta` | `1` | Snapshot width in minutes (×60 → seconds) |
| `--encoder_name` | `ARGUS` | Encoder type (only ARGUS supported for CIC) |
| `--rnn` | `GRU` | Temporal model: `GRU`, `LSTM`, or `NONE` |
| `--hidden` | `32` | GCN hidden dimension |
| `--zdim` | `16` | Final embedding dimension |
| `--lr` | `0.01` | Learning rate for SOAP optimizer |
| `--fpweight` | `0.6` | FP weight for cutoff optimization (0–1) |
| `--patience` | `3` | Early stopping patience (epochs without improvement) |
| `--epochs` | `200` | Maximum training epochs |
| `--nratio` | `1` | Negative sampling ratio |
| `--flows` | `True` | Edge attributes flag (`--flows` disables them, `store_false`) |
| `--load` | `False` | Load pre-trained model instead of training |
| `--gpu` | `False` | Use GPU if available |
| `--data_dir` | `...cic_2017` | Base directory for euler/ and argus_flow/ |

### Recommended Commands (from `cic.sh`)

```bash
# Without flow (O_cic)
python main.py --dataset O_cic --delta 15 --lr 0.01 --hidden 32 -z 16 \
    --fpweight 0.46 --epoch 100 --patience 3

# With flow (L_cic_flow)
python main.py --dataset L_cic_flow --delta 10 --lr 0.05 --hidden 32 -z 16 \
    --fpweight 0.45 --epoch 100 --patience 3
```

---

## 11. Baseline Results on CIC-IDS 2017

### O_cic (Without Flow)

| Metric | Value |
|--------|-------|
| AUC | 0.6985 |
| AP | 0.2967 |
| F1 | 0.3690 |
| TPR | 0.4738 |
| FPR | 0.2183 |
| TP | 127,587 |
| FP | 294,591 |
| TN | 1,054,972 |
| FN | 141,699 |
| TPE | 156.4s |
| FwdTime | 488.8s |

### L_cic_flow (With Flow)

| Metric | Value |
|--------|-------|
| AUC | 0.5853 |
| AP | 0.0009 |
| F1 | 0.0020 |
| TPR | 0.3886 |
| FPR | 0.2698 |
| TP | 68 |
| FP | 69,203 |
| TN | 187,302 |
| FN | 107 |
| TPE | 11.7s |
| FwdTime | 61.6s |

> **Note**: The L_cic_flow config has very few test positives (175 total = 68 TP + 107 FN) compared to O_cic (269,286 total), suggesting the delta/snapshot parameters produce very different edge-level label aggregations. The O_cic config with delta=15min creates broader snapshots that capture more attack-labeled edges, while L_cic_flow with delta=10min creates narrower snapshots.

---

## 12. File Reference

| File | Lines | Description |
|------|-------|-------------|
| `main.py` | 116 | Entry point, CLI args, dataset routing, device setup |
| `classification.py` | 271 | Train/val/test orchestration, loss, scoring, metrics |
| `models/argus.py` | 411 | GCN encoders (Argus_OPTC, Argus_LANL), DetectorEncoder, Argus wrapper |
| `models/recurrent.py` | 48 | GRU, LSTM, EmptyModel |
| `loaders/load_cic.py` | 312 | Data loader for euler/ (without flow) |
| `loaders/load_cic_flow.py` | ~280 | Data loader for argus_flow/ (with flow) |
| `loaders/tdata.py` | ~200 | TData class (temporal graph data object) |
| `loaders/load_utils.py` | 89 | Edge weight/attribute standardization, train/val split |
| `utils.py` | 60 | Scoring utilities (AUC, AP, optimal cutoff) |
| `cic.sh` | 4 | Run commands for both configs |
| `results_cic.txt` | ~20 | Saved baseline results |
| `Exps/` | — | Saved model checkpoints (.pkl) |

---

## 13. KDE Edge Feature Integration Points

### Where KDE timestamp-difference density vectors can be incorporated:

#### Option A: L_cic_flow — Concatenate to existing flow edge attributes

The `Argus_LANL` encoder already processes 6-dim edge attributes via **NNConv**:

```python
# Current: 6-dim flow features
nn4 = nn.Sequential(
    nn.Linear(6, 8), nn.ReLU(),
    nn.Linear(8, h_dim * z_dim)
)
self.c4 = NNConv(h_dim, z_dim, nn4, aggr='mean')
```

KDE vectors (K-dim, e.g., 20-dim) can be **concatenated** to the 6 flow features → `6+K = 26` input dims:

```python
# Modified: 6 flow + 20 KDE = 26-dim edge features
nn4 = nn.Sequential(
    nn.Linear(26, 32), nn.ReLU(),
    nn.Linear(32, h_dim * z_dim)
)
```

**Integration point**: In `load_cic_flow.py`, after computing the 6 flow stats per snapshot edge, look up the pre-computed KDE vector for that `(src, dst)` pair and append it. Edges without a KDE vector get `zeros(K)` as fallback.

#### Option B: O_cic — Add NNConv layer with KDE-only edge attributes

Currently `Argus_OPTC` uses **only GCNConv** (no edge attributes). To add KDE features:

1. Replace the final `GCNConv` layer (c4) with an **NNConv** layer that takes KDE vectors as `edge_attr`
2. Or add a 5th NNConv layer after the existing 4 GCNConv layers

```python
# Add NNConv with KDE-only features (K-dim)
nn_kde = nn.Sequential(
    nn.Linear(K, 16), nn.ReLU(),
    nn.Linear(16, h_dim * z_dim)
)
self.c_kde = NNConv(h_dim, z_dim, nn_kde, aggr='mean')
```

**Integration point**: In `load_cic.py`, add KDE vector lookup per snapshot edge → populate `eas` list in `TData`. The `Argus_OPTC.forward_once()` would need to call `ea_masked()` and pass to NNConv.

#### Where KDE features are computed

KDE features should be **pre-computed on training data only** (timestamps 0–29,136) before ARGUS training starts:

1. For each directed edge `(src, dst)` in training data, collect all timestamps
2. Compute timestamp differences: `Δt_i = t_{i+1} - t_i`
3. Fit a DPGMM (BayesianGaussianMixture) on the timestamp differences
4. Evaluate on a fixed grid → produce a K-dim density vector
5. L2-normalize the vector
6. Save to a pickle file: `{(src, dst): array(K)}` 
7. At load time, look up per-edge KDE vectors for each snapshot

---

## 14. Data Field Reference

### euler/ (ARGUS without flow)

**Sample line**: `0,0,1,0.0,12.0,0.0,0`

| Index | Field | Type | Used by Loader | Description |
|-------|-------|------|----------------|-------------|
| 0 | `ts` | int | ✅ Yes | Timestamp (seconds from start) |
| 1 | `src` | int | ✅ Yes | Source node ID |
| 2 | `dst` | int | ✅ Yes | Destination node ID |
| 3 | field_3 | float | ❌ **Ignored** | Unknown (possibly duration, always 0.0 in sample) |
| 4 | field_4 | float | ❌ **Ignored** | Unknown (possibly byte count, e.g., 12.0) |
| 5 | field_5 | float | ❌ **Ignored** | Unknown (possibly packet count, always 0.0 in sample) |
| 6 | `label` | int | ✅ Yes | 0 = benign, 1 = attack |

> The loader uses: `fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[-1][:-1]))` — only fields 0, 1, 2, and the last field (6) are parsed.

### argus_flow/ (ARGUS with flow)

**Sample line**: `0,0,1,1.0,12,2,0`

| Index | Field | Type | Used by Loader | Description |
|-------|-------|------|----------------|-------------|
| 0 | `ts` | int | ✅ Yes | Timestamp (seconds from start) |
| 1 | `src` | int | ✅ Yes | Source node ID |
| 2 | `dst` | int | ✅ Yes | Destination node ID |
| 3 | `duration` | float | ✅ Yes | Flow duration (seconds) |
| 4 | `bytes` | float/int | ✅ Yes | Byte count of the flow |
| 5 | `packets` | int | ✅ Yes | Packet count of the flow |
| 6 | `label` | int | ✅ Yes | 0 = benign, 1 = attack |

> The loader uses: `fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), float(x[3]), float(x[4]), int(x[5]), int(x[6][:-1]))` — all 7 fields are parsed.

### Common to Both

- **Total lines**: 2,830,742 across all 4 files
- **File split**: At `FILE_DELTA = 100,000` timestamp boundaries (0.txt, 100000.txt, 200000.txt, 300000.txt)
- **Node map**: `nmap.pkl` — dictionary mapping original IPs to integer node IDs (19,129 nodes)
- **Label distribution**: 2,273,096 benign (0) / 557,646 attack (1)

---

## Architecture Diagram

```
                    ┌──────────────────────────────────────────┐
                    │            ARGUS Pipeline                 │
                    └──────────────────────────────────────────┘

   Raw Data Files          Graph Snapshots           GCN Encoder
  ┌─────────────┐        ┌───────────────┐        ┌─────────────────┐
  │ euler/*.txt  │──┐     │ Snapshot t=0  │───────▶│ Argus_OPTC      │
  │ or           │  │     │ Snapshot t=1  │───────▶│ (4× GCNConv)    │
  │ argus_flow/* │──┤     │ Snapshot t=2  │───────▶│ or              │
  └─────────────┘  │     │ ...           │───────▶│ Argus_LANL      │
                   │     └───────────────┘        │ (3× GCNConv +   │
                   │            ▲                  │  1× NNConv)     │
                   │            │                  └────────┬────────┘
                   │   delta-width time                     │
                   │   window aggregation           z_t ∈ ℝ^(N×32)
                   │   + edge deduplication                 │
                   │   + flow stat computation              ▼
                   │                              ┌─────────────────┐
                   │                              │   GRU/LSTM      │
                   │                              │ (temporal model) │
                   │                              └────────┬────────┘
                   │                                       │
                   │                               z_t ∈ ℝ^(N×16)
                   │                                       │
                   │                                       ▼
                   │                              ┌─────────────────┐
                   │                              │  Dot-Product     │
                   │                              │  Decode + σ      │
                   │                              │                  │
                   │                              │ score(s,d) =     │
                   │                              │  σ(z_s · z_d)    │
                   │                              └────────┬────────┘
                   │                                       │
                   │                                 edge scores
                   │                                       │
                   │                                       ▼
                   │                              ┌─────────────────┐
                   │                              │ APLoss + SOAP   │
                   │                              │ (training)      │
                   │                              │ or              │
                   │                              │ Threshold       │
                   │                              │ (inference)     │
                   └──────────────────────────────└─────────────────┘
```

---

## 15. KDE Integration — Detailed Implementation Plan (Option A: NNConv Edge Attributes)

### 15.1 Overview

**Goal**: Pre-compute a K-dim DPGMM density vector for each directed edge `(src, dst)` from its timestamp differences in training data, then **concatenate** this vector to the existing 6-dim flow features so NNConv in `Argus_LANL` operates on `(6+K)`-dim edge attributes.

**Key decisions** (aligned with PIKACHU KDE):

| Decision | Choice |
|----------|--------|
| Density estimator | `BayesianGaussianMixture` (sklearn DPGMM) |
| Feature dimension (K) | 20 (default) |
| Edge direction | Directed — `(A→B)` and `(B→A)` get separate KDE vectors |
| Fallback | `zeros(K)` for edges not seen in training or with < 2 timestamps |
| Normalization | L2-normalize each vector |
| Grid | `np.linspace(global_min_diff, global_95th_pct_diff, K)` |
| Training data only | `ts < 29,136` (DATE_OF_EVIL_LANL) |

### 15.2 Pre-computation Script: `compute_kde_features_argus.py`

**Location**: `ARGUS/compute_kde_features_argus.py`

**Input**: `argus_flow/*.txt` files (reads `ts, src, dst` from each line)

**Output**: Pickle file `kde_vectors_argus.pkl` containing:
```python
{
    (src_int, dst_int): np.ndarray(K,),   # L2-normalized density vector
    ...
}
```

**Algorithm**:
```
1. Scan all lines in argus_flow/*.txt where ts < TR_END (29136)
2. For each directed (src, dst), collect sorted list of timestamps
3. Compute global timestamp-difference statistics:
   - all_diffs = concat of all per-edge diffs
   - grid = linspace(min(all_diffs), percentile(all_diffs, 95), K)
4. For each (src, dst) with >= 2 timestamps:
   a. diffs = t[i+1] - t[i]
   b. Fit BayesianGaussianMixture(n_components=10, ...) on diffs
   c. Evaluate log-density on grid → exp → K-dim vector
   d. L2-normalize (fallback to zeros if norm == 0)
5. Save dict to pickle
```

**CLI**:
```bash
python compute_kde_features_argus.py \
    --data_dir /path/to/cic_2017/argus_flow \
    --output kde_vectors_argus.pkl \
    --kde_dim 20 \
    --tr_end 29136

python compute_kde_features_argus.py    --data_dir /path/to/cic_2017/argus_flow --output kde_vectors_argus.pkl    --kde_dim 20   --tr_end 29136
```

### 15.3 Loader Changes: `loaders/load_cic_flow.py`

**What changes** (Option B — KDE at decode, not in edge attributes):

- `load_lanl_dist()` and `load_partial_lanl()` accept a new `kde_file=None` kwarg
- When `kde_file` is set, load the pickle at the start of `load_partial_lanl()`
- `eas` stays **6-dim only** — the KDE dict is **not** appended to flow features
- Instead, `kde_dict` is stored on the `TData` object and looked up at decode time
- `make_data_obj()` passes `kde_dict=kde_dict` to `TData`

### 15.4 Data Object Changes: `loaders/tdata.py`

**Bug fix**: `ea_dim` was hardcoded to 5, but actual CIC-IDS flow features are 6-dim.

**Fix**: Dynamically compute `ea_dim` from actual data:

```python
if not isinstance(eas, None.__class__) and len(eas) > 0:
    self.ea_dim = eas[0].size(0)   # always 6 (flow only — KDE at decode)
elif use_flows:
    self.ea_dim = 5                # legacy fallback
else:
    self.ea_dim = 0
```

**New fields** (Option B):
- `kde_dict` — stored in `__dict__` directly to bypass PyG GlobalStorage; maps `(src_int, dst_int) → array(K,)`; `None` for baseline/reduced configs
- `kde_dim` — `K` when dict is loaded, `0` otherwise
- `get_kde_tensor(src_t, dst_t)` — looks up each edge in `kde_dict`, returns `FloatTensor(E, K)` with zero-vector fallback for unseen edges

### 15.5 Model Changes: `models/argus.py` (`Argus_LANL`)

**NNConv unchanged** — `nn4` stays fixed at 6-dim input:
```python
nn4 = nn.Sequential(nn.Linear(6, 8), nn.ReLU(),
                    nn.Linear(8, h_dim * z_dim))
```

**New: `KDE_MLP`** added to `Argus_LANL` for decode-time scoring (Option B):
```python
# Built only when kde_dim > 0
mlp_in = z_dim + kde_dim          # e.g. 16 + 20 = 36
self.kde_mlp = nn.Sequential(
    nn.Linear(mlp_in, max(mlp_in, 16)),
    nn.ReLU(),
    nn.Linear(max(mlp_in, 16), 1)
)
```

**`decode()` override**:
```python
def decode(self, src, dst, z):
    if self._kde_dim > 0:
        hadamard = z[src] * z[dst]                         # (E, z_dim)
        kde_vecs = self.data.get_kde_tensor(src, dst).to(z.device)  # (E, K)
        feat = torch.cat([hadamard, kde_vecs], dim=1)      # (E, z_dim+K)
        return torch.sigmoid(self.kde_mlp(feat).squeeze(1))
    return torch.sigmoid((z[src] * z[dst]).sum(dim=1))     # baseline fallback
```

~912 params total (vs 14,526 under Option A). Baseline and reduced configs (`kde_dim=0`) follow the original dot-product path unchanged.

### 15.6 Entry Point Changes: `main.py`

Add three new CLI arguments:
```python
ap.add_argument('--kde', action='store_true',
                help='Enable KDE timestamp-diff edge features')
ap.add_argument('--kde_file', type=str, default='kde_vectors_argus.pkl',
                help='Path to pre-computed KDE pickle')
ap.add_argument('--kde_dim', type=int, default=20,
                help='Dimensionality of KDE density vectors')
ap.add_argument('--red', action='store_true',
                help='Use reduced graph from argus_flow_red/')
```

**Routing logic**:
- When `--kde` is set and dataset starts with `'L'`: pass `kde_file` into loader kwargs
- When `--red` is set and dataset starts with `'L'`: change directory from `argus_flow/` to `argus_flow_red/`
- Both flags can be combined (KDE + reduced graph)

### 15.7 Data Flow Diagram

```
                        ┌─────────────────────────────────┐
                        │  compute_kde_features_argus.py   │
                        │  (offline, run once)              │
                        └────────────┬────────────────────┘
                                     │
                             kde_vectors_argus.pkl
                         {(src,dst): array(20)}
                                     │
  argus_flow/*.txt                   │
  ┌──────────────┐                   ▼
  │ts,s,d,dur,   │──▶ load_cic_flow.py ──▶ TData
  │bytes,pkts,lbl│    ┌────────────────┐    ├── eis: edge indices
  └──────────────┘    │Per snapshot:   │    ├── ews: edge weights (counts)
                      │ 6 flow stats   │    ├── eas: 6-dim flow only (unchanged)
                      │ (KDE stored    │    ├── kde_dict: {(s,d)→vec(K)} ◀── NEW
                      │  on TData,     │    └── ys:  labels
                      │  not in eas)   │
                      └────────────────┘
                                             │
                                             ▼
                                     ┌───────────────────────┐
                                     │ Argus_LANL             │
                                     │ GCNConv×3 + NNConv(6)  │  (unchanged)
                                     │ GRU → z embeddings     │
                                     └──────────┬────────────┘
                                                │
                                     ┌──────────▼────────────┐
                                     │ decode(src, dst, z)    │
                                     │ hadamard = z[s]⊙z[d]   │
                                     │ kde = kde_dict[s,d]    │◀── KDE injected here
                                     │ MLP([hadamard ∥ kde])  │
                                     └───────────────────────┘
```

### 15.8 Reduced Graph: `generate_reduced_argus_flow.py`

**Location**: `ARGUS/generate_reduced_argus_flow.py`

**Goal**: Produce a version of `argus_flow/` that retains **every row** (same line count, same file splits, no deduplication) but replaces the per-flow features (duration, bytes, packets) with **global temporal summary statistics** for each directed `(src, dst)` pair:

| Original flow features | Reduced flow features |
|------------------------|-----------------------|
| `duration` (per-flow)  | `first_ts` (earliest timestamp of this edge, global) |
| `bytes` (per-flow)     | `last_ts` (latest timestamp of this edge, global) |
| `packets` (per-flow)   | `count` (total number of flows for this edge, global) |

These three values are **identical across all rows** sharing the same `(src, dst)` pair — they capture the edge's global lifetime and frequency rather than individual flow characteristics.

**Output line format** (same 7-field CSV, compatible with `load_cic_flow.py`):
```
ts, src, dst, first_ts, last_ts, count, label
```
Note: the original per-row `ts` and `label` are preserved; only fields 3–5 are replaced.

**Output directory**: `GIDSREP/1-DATA_PROCESSING/cic_2017/argus_flow_red/`
- **Same file structure** as `argus_flow/`: `0.txt`, `100000.txt`, `200000.txt`, `300000.txt`
- **Same number of lines** per file (no deduplication)
- Copies `nmap.pkl` from `argus_flow/`

**Two-pass algorithm**:
```
Pass 1 — scan all argus_flow/*.txt files, compute per-edge global stats:
    For each unique (src, dst):
        first_ts  = min(all timestamps)
        last_ts   = max(all timestamps)
        count     = total number of flows

Pass 2 — re-read all input files, write output with substituted features:
    For each row  ts, src, dst, dur, bytes, pkts, label:
        Write     ts, src, dst, first_ts, last_ts, count, label
        (using the global stats for this (src, dst) pair)
```

**Note on loader behavior**: Within any delta-window snapshot, all rows for the same `(src, dst)` pair share the same `(first_ts, last_ts, count)` values. When the loader aggregates duplicate edges per snapshot, it computes mean/std of these features:
- `mean_duration = first_ts`, `std_duration = 0` (constant across duplicates)
- `mean_pkt_cnt = last_ts`, `std_pkt_cnt = 0`
- `mean_byte_cnt = count`, `std_byte_cnt = 0`
- The resulting 6-dim edge attributes are `[first_ts, 0, last_ts, 0, count, 0]`
- NNConv learns from this temporally-informed representation

### 15.9 KDE Integration Options Considered

| # | Integration Point | Where in Pipeline | Description | Status |
|---|-------------------|-------------------|-------------|--------|
| **A** | NNConv edge_attr | GCN encoding | Concatenate KDE to flow features → NNConv learns KDE-conditioned message passing | **Abandoned** — TN=0, AUC≈0.5 (parameter explosion + mostly-zero fallback vectors; see §17.1) |
| **B** | **Edge decode scoring** | **After GRU** | **Replace dot-product decode with `MLP([z_src ⊙ z_dst ∥ kde])` → KDE directly informs anomaly score** | **Implemented (current)** |
| **C** | Argus_OPTC (O_cic) | GCN encoding | Add NNConv layer with KDE-only features to the currently flow-free encoder | Not tried |
| **D** | Node feature augmentation | Input | For each node, aggregate KDE stats of incident edges → replace/augment identity matrix `xs` | Not tried |

---

## 16. Run Commands

All commands are run from the `ARGUS/` directory:

```bash
cd /scratch/asawan15/GIDSREP/2-DETECTION_ASSESSMENT/RQ2/cic_2017/ARGUS
```

### 16.1 Preprocessing (run once)

**Step 1 — Generate KDE timestamp-diff vectors** (from training data only):

```bash
python compute_kde_features_argus.py \
    --data_dir /scratch/asawan15/GIDSREP/1-DATA_PROCESSING/cic_2017/argus_flow \
    --output   kde_vectors_argus.pkl \
    --kde_dim  20 \
    --tr_end   29136
```

**Step 2 — Generate reduced graph data** (same rows, temporal summary features):

```bash
python generate_reduced_argus_flow.py \
    --input_dir  /scratch/asawan15/GIDSREP/1-DATA_PROCESSING/cic_2017/argus_flow \
    --output_dir /scratch/asawan15/GIDSREP/1-DATA_PROCESSING/cic_2017/argus_flow_red
```

### 16.2 Experiment Runs

**Run 1 — Baseline: ARGUS with flow**

```bash
python main.py --dataset L_cic_flow --delta 10 --lr 0.05 --hidden 32 -z 16 \
    --fpweight 0.45 --epochs 100 --patience 3 --gpu
```

**Run 2 — ARGUS with flow + KDE timestamp-diff edge features**

```bash
python main.py --dataset L_cic_flow --delta 10 --lr 0.05 --hidden 32 -z 16 --fpweight 0.45 --epochs 100 --patience 3 --gpu --kde --kde_file kde_vectors_argus.pkl --kde_dim 20
```

**Run 3 — ARGUS with flow on reduced graphs**

```bash
python main.py --dataset L_cic_flow --delta 10 --lr 0.05 --hidden 32 -z 16 \
    --fpweight 0.45 --epochs 100 --patience 3 --gpu \
    --red
```

> **Note**: `--kde` and `--red` can be combined to run KDE features on reduced graphs:
> ```bash
> python main.py --dataset L_cic_flow --delta 10 --lr 0.05 --hidden 32 -z 16 \
>     --fpweight 0.45 --epochs 100 --patience 3 --gpu \
>     --kde --kde_file kde_vectors_argus.pkl --kde_dim 20 --red
> ```
---

## 17. KDE at Decode Step (Option B) — Replacing Option A

### 17.1 Why Option A Failed

Option A injected KDE as additional dimensions in the NNConv edge attributes:

```
eas = [mean_dur, std_dur, mean_pkt, std_pkt, mean_byte, std_byte,  ← 6 flow
       kde_0, ..., kde_19]                                          ← 20 KDE
```

Three compounding problems made this degenerate (TN=0, AUC≈0.5):

| Problem | Root Cause | Effect |
|---------|-----------|--------|
| Parameter explosion | NNConv `nn4` grows from `Linear(6,8)→…` (4,800 params) to `Linear(26,26)→…` (14,526 params) | Unstable gradients, z embeddings collapse |
| Mostly-zero KDE | Only ~5K edges have real KDE vectors; majority get `[0,…,0]` as fallback | 20 zero dimensions confuse NNConv weight generation |
| Indirect signal path | KDE → NNConv weight matrix → z → dot-product score (3 hops) | Optimization signal too diluted to attribute benefit to KDE |

### 17.2 Option B Design (Additive)

**Principle**: Keep the baseline scoring function (dot-product) completely unchanged.
The KDE MLP provides a small *additive scalar adjustment* to the dot-product logit.
Critically, the MLP sees **only the KDE vector** — never the z embeddings — so it
cannot memorise the link-prediction task.

```
Baseline score:  sigmoid( z_src · z_dst )

Option B score:  sigmoid( z_src · z_dst  +  KDE_MLP(kde_vec) )
                                           └── Linear(20,10) → ReLU → Linear(10,1)  (~221 params)
```

Why **replacement** MLP (prior attempt) failed:
- MLP saw `[z_src ⊙ z_dst ∥ kde_vec]` → could learn "KDE present → real edge → score 1.0"
- Training validation (real edges vs *random negatives*) achieved AUC=0.98
  but all real edges saturated to 1.0 at test time → TN=0, same as Option A

Why **additive** design works:
- **Dot-product always contributes**: structural z information is never discarded
- **MLP cannot overfit link-prediction**: it only sees KDE, not z embeddings
- **Small adjustment**: ~221 params (vs 912 replacement, vs 14,526 Option A)
- **Graceful fallback**: edges with `kde=zeros` → `MLP(zeros)` = constant bias →
  score = `sigmoid(dot_product + constant)`, same ranking as baseline

### 17.3 Data Flow

```
  argus_flow/*.txt
  ┌─────────────┐
  │ts,s,d,dur,  │──▶ load_cic_flow.py ──▶ TData
  │bytes,pkts,  │    eas: 6-dim flow          ├── eis, ews, eas (6-dim only)
  │label        │    kde_dict: stored here ──▶└── kde_dict: {(src,dst)→vec(20,)}
  └─────────────┘
                                                     │
                       GCN layers (unchanged)        │
                       c1 GCNConv(x_dim → 32)        │
                       c2 GCNConv(32 → 32)            │
                       c3 GCNConv(32 → 16)            │
                       c4 NNConv(16 → 16, nn4(6-dim)) │  ← NNConv reverted to 6-dim
                                ↓                    │
                       z per node (16-dim)           │
                                ↓                    │
              ┌──────────────────────────────────────┘
              │    At decode(src, dst, z):
              │    hadamard = z[src] ⊙ z[dst]        (E, 16)
              │    kde      = lookup(src, dst)        (E, 20)
              │    feat     = concat(hadamard, kde)   (E, 36)
              │    score    = sigmoid(KDE_MLP(feat))  (E,)
              └─────────────────────────────────────▶ score
```

### 17.4 Files Changed

#### `loaders/tdata.py`
- Remove KDE from `eas` path — `eas` is back to 6-dim flow features only
- Add `kde_dict` field (stored in `__dict__` to bypass PyG storage) — maps `(src_int, dst_int) → np.ndarray(K,)`
- Add `kde_dim` field: `K` when dict is loaded, `0` otherwise
- Add `get_kde_tensor(src_t, dst_t)` method: given 1D tensors of node IDs, returns `torch.FloatTensor(E, K)` via dict lookup (zeros fallback for unseen edges)

#### `loaders/load_cic_flow.py`
- Revert: KDE is no longer appended to `eas` — the `if kde_dict:` block that did `fs[eij] = fs[eij] + list(kde_dict.get(...))` is removed
- `load_partial_lanl()`: still loads KDE pickle when `kde_file` is set, but now stores it for `make_data_obj()` to pass to `TData` as `kde_dict=`, not into `eas`
- `make_data_obj()`: new `kde_dict=None` param → forwarded to `TData`
- `load_partial_lanl_job()` / `load_lanl_dist()`: thread `kde_dict` back through the parallel reduce

#### `models/argus.py`
- `Argus_LANL` is **unchanged** from baseline — `nn4` is 6-dim, no `decode()` override, no `kde_mlp`
- `KDE_MLP` lives in `DetectorEncoder` — **additive** design, MLP sees only KDE:
  ```python
  kde_dim = getattr(module.data, 'kde_dim', 0)
  if kde_dim > 0:
      mlp_h = max(kde_dim // 2, 8)          # 20 → 10
      self.kde_mlp = nn.Sequential(
          nn.Linear(kde_dim, mlp_h),
          nn.ReLU(),
          nn.Linear(mlp_h, 1)
      )                                      # ~221 params
  ```
- `DetectorEncoder.decode(e, z, no_grad)` override:
  - Computes `base_score = (z[src] * z[dst]).sum(dim=1)` (baseline dot-product, always)
  - If `_kde_dim > 0`: `kde_adj = kde_mlp(kde_vec)` → `sigmoid(base_score + kde_adj)`
  - Else: `sigmoid(base_score)` (identical to baseline)

#### `main.py`
- No changes needed — `args.kde_file_path` is already set correctly and passed through `classification.py` → `get_cutoff()` → `test()` → loader `kde_file=` kwarg

### 17.5 Training Behavior

Because `DetectorEncoder.decode()` is called by both `calc_loss_argus()` (training) and `decode_all()` (testing), the `KDE_MLP` is trained end-to-end without any changes to `classification.py`.

The MLP learns: *given the structural similarity score between two nodes (Hadamard product of GRU embeddings), does the temporal access pattern (captured by KDE) indicate anomaly?*

### 17.6 Edge Attribute Dimensions After This Change

| Configuration | `eas` dim | `kde_dim` | Score function |
|---------------|-----------|-----------|----------------|
| Baseline flow | 6 | 0 | `sigmoid(z_src · z_dst)` |
| KDE decode (Option B) | 6 | 10 | `sigmoid(z_src · z_dst + KDE_MLP(kde))` |
| Reduced graph | 6 | 0 | `sigmoid(z_src · z_dst)` |
| Reduced + KDE | 6 | 10 | `sigmoid(z_src · z_dst + KDE_MLP(kde))` |

---

## 18. KDE Integration — Full Pipeline Analysis & Improvement Directions

### 18.1 All Points in the Pipeline Where KDE Vectors Could Be Integrated

The ARGUS `L_cic_flow` pipeline has **five distinct integration points** for KDE vectors. Only one is currently active (Option B, decode-time additive scalar).

```
argus_flow/*.txt
       │
       ▼
[1] load_partial_lanl()          ← POINT 1: Edge attribute (eas) — concatenate to 6-dim flow features
       │                            (Option A, abandoned)
       │  edges_t, temp_flows
       ▼
[2] make_data_obj() / TData      ← POINT 2: Node-level aggregation — assign KDE as node feature
       │                            (new idea)
       │  TData.eis, eas, ews
       ▼
[3] Argus_LANL.forward_once()    ← POINT 3: GCN message-passing weight — use KDE to gate
       │  c1→c2→c3 (GCNConv)         NNConv edge-attr directly (Option A)
       │  c4 (NNConv, ea=6-dim)  ← POINT 4: Concatenate KDE into NNConv edge attrs (Option A variant)
       │
       ▼
[5] GRU (recurrent.py)           (no natural KDE injection point here — temporal dynamics,
       │                           KDE is static)
       ▼
[6] DetectorEncoder.decode()     ← POINT 5: Additive scalar at decode (Option B — ACTIVE)
       │   sigmoid(z·z + MLP(kde))
       ▼
  edge score ∈ (0, 1)
```

#### Point 1 — Edge Attributes (Option A, concatenated to `eas`)
Concatenate the K-dim KDE vector to the 6-dim flow features → `eas` becomes `(6+K)`-dim. `NNConv`'s internal MLP (`nn4`) processes all edge attributes jointly to produce per-edge message-passing weights.

**Status**: Implemented and tested. Produced degenerate results (TN=0, all edges flagged).

#### Point 2 — Node Features
For each node, aggregate KDE vectors across all its *outgoing* (or incoming) edges at training time — e.g., mean KDE vector across all edges where node appears as source. Append this to or replace the one-hot node feature matrix `xs`. The GCN then sees temporal rhythm information at the node level from the first layer.

**Status**: Not implemented. Novel integration point.

#### Point 3 — GCN Edge Weight Gating
Use a scalar derived from the KDE vector (e.g., `sigmoid(KDE_MLP(kde_vec).squeeze())`) as a per-edge multiplier applied to `ew` (edge weight) before passing to GCNConv layers 1–3. This modulates how much each edge contributes to neighbourhood aggregation based on its temporal regularity.

**Status**: Not implemented. Would require storing KDE scalars per edge per snapshot.

#### Point 4 — NNConv Edge Attribute Augmentation (Option A variant, fixed)
Same as Point 1 but with a **separate projection head** that first normalizes KDE vectors into a comparable range with the 6-dim flow stats, then concatenates. Alternatively, inject KDE as a *residual addition* to `ea` after projecting to 6-dim rather than expanding `nn4`'s input dimension.

**Status**: Not implemented.

#### Point 5 — Decode-time Additive Scalar (Option B — ACTIVE)
`score = sigmoid(z_src · z_dst + KDE_MLP(kde_vec))`. The MLP sees only the K-dim KDE vector and outputs a scalar bias. Structurally separates temporal rhythm signal from embedding-based structural signal.

**Status**: Implemented. Produces marginal AUC improvement (+0.030) but slightly lower TP than baseline.

---

### 18.2 Why Option A Failed — Analysis and Counters

**Stated hypothesis**: The 20-dim (now 10-dim) KDE vectors are zero for non-KDE edges (edges with < 2 training timestamps), and these zero-padded vectors dominate the 6-dim flow features in the NNConv MLP, producing a collapsed threshold where almost everything is flagged anomalous (TN = 0).

#### Is This Correct?

**Partially correct.** The zero-vector domination argument is sound but incomplete. A fuller diagnosis:

| Root Cause | Validity | Explanation |
|------------|----------|-------------|
| Zero-vector domination | ✅ Correct | 5,215 of ~39,402 training edges have KDE vectors. The remaining ~87% have `zeros(K)`. After concatenation, these zero-padded rows form a structurally identical block in `nn4`'s input space. NNConv's MLP maps them all to the same message-passing weight — meaning all non-KDE edges become indistinguishable by the edge-feature path. |
| Threshold collapse | ✅ Correct | With ~87% of edges mapped identically by `nn4`, the model cannot discriminate — it either flags all or flags none. Results (TP=175, TN=0) show the cutoff collapsed below all scores, treating everything as anomalous. |
| Parameter explosion | ✅ Correct | Expanding `nn4` input from 6 to 26 dims adds ~1,150 additional parameters to `nn4` alone. With sparse KDE coverage, these extra parameters are trained on identical zero vectors for most edges → gradient signal is noisy and conflicting. |
| 3-hop indirect signal | ✅ Correct | KDE vectors affect `c4` (NNConv), whose output feeds GRU, whose output feeds decode. The gradient path from a KDE-related detection back to `nn4` is long (3 hops), causing vanishing/diluted gradients for KDE-specific parameters. |

**Counters / nuances**:

1. **Zero-padding is a design choice, not a fundamental flaw.** A learned embedding for "missing KDE" (a trainable fallback vector instead of hard zeros) would allow `nn4` to assign a *different* weight to non-KDE edges rather than treating them identically. This is a straightforward fix for Option A.

2. **The AUC of 0.4998 (< 0.5) in Option A is diagnostic.** An AUC below 0.5 means the model is *anti-discriminative* — it consistently ranks anomalies higher than benign edges, but the cutoff inversion means everything gets flagged. This suggests the model actually *did* learn something, but the calibration step (`get_optimal_cutoff`) chose a threshold that inverts the ranking. A fixed threshold (e.g., 0.5) applied to Option A scores might actually produce better metrics than the calibrated one.

3. **The 6-dim flow features are not zero-padded.** The argument applies only to the KDE portion. Flow features are present for all edges (with a 0-vector fallback for edges with no flows). The interaction between 6-dim real data and K-dim zeros inside `nn4`'s linear layers is the actual problem — `nn4` learns to use *either* flow features *or* KDE features but not both, because zero-masking creates a degenerate gradient landscape.

4. **Option A might work with a different architecture.** If KDE were passed through a *separate* MLP branch and the output *added* to (not concatenated into) the NNConv message, the zero-masking problem disappears. This is effectively what Option B does, but at decode time instead of message-passing time.

---

### 18.3 Option B Results Analysis

| Metric | Baseline | KDE Option B | Reduced |
|--------|----------|--------------|---------|
| TP | 68 | 63 | 19 |
| FP | 69,203 | 71,538 | 77,463 |
| FN | 107 | 112 | 156 |
| TN | 187,302 | 184,967 | 179,042 |
| Precision | 0.00098 | 0.00088 | 0.00025 |
| Recall / TPR | 0.389 | 0.360 | 0.109 |
| AUC | 0.5853 | 0.6155 | 0.3994 |
| F1 | 0.00196 | 0.00176 | 0.00049 |
| Time (s) | 64.9 | 78.0 | 64.6 |

**Key observations**:
- AUC improves (+0.030), indicating the *ranking* of edges is better — the model is more correct about *relative* anomaly ordering.
- TP *decreases* slightly (68 → 63). This is because the additive KDE bias shifts scores slightly but the cutoff recalibrates. Some attack edges happen to share the same `(src, dst)` as frequently-seen training edges, so their KDE score is *high* (regular), which suppresses their anomaly score below the cutoff.
- The KDE ceiling is low: vectors are static per-edge and identical between normal and attack flows on the same edge. The MLP cannot distinguish a *normal* traversal of edge `(A, B)` from an *attack* traversal — both get the same KDE vector.

---

### 18.4 Ways to Improve TP in the Option B Pipeline

These are ordered from low to high implementation complexity.

#### Improvement 1 — Lower the FPWeight (Threshold Tuning)
**Mechanism**: The `fpweight` parameter controls the cutoff calibration trade-off. Lower `fpweight` → lower threshold → more edges flagged → higher TP at cost of more FP.
```bash
python main.py ... --fpweight 0.30  # default is 0.45
```
**Expected effect**: Directly increases TP. Trade-off: FP also increases. AUC unchanged (it's threshold-independent).
**Complexity**: Zero code changes. One parameter.

#### Improvement 2 — Increase `nratio` (More Negative Sampling)
**Mechanism**: `--nratio 2` (or higher) samples 2× as many negative edges per positive during training. This makes the training task harder, pushing the model to place benign edges clearly above the decision boundary and anomalous edges clearly below.
```bash
python main.py ... --nratio 2
```
**Expected effect**: Potentially improves AUC and pushes attack-edge scores lower relative to benign, which may improve TP at a given threshold.
**Complexity**: Zero code changes.

#### Improvement 3 — Learned Fallback for Unseen KDE Edges
**Mechanism**: Replace the hard zero-vector fallback in `get_kde_tensor()` with a **trainable fallback embedding** — a single learnable vector of size K, initialized to zeros, that the model can tune to represent "edge not seen in training". This allows the KDE_MLP to treat unseen edges differently from edges with a genuine zero-density KDE.

In `DetectorEncoder.__init__`:
```python
if kde_dim > 0:
    self.kde_fallback = nn.Parameter(torch.zeros(kde_dim))
```
In `DetectorEncoder.decode()`:
```python
kde_vecs = self.module.data.get_kde_tensor(src, dst, fallback=self.kde_fallback)
```
In `TData.get_kde_tensor()`:
```python
def get_kde_tensor(self, src_t, dst_t, fallback=None):
    zero = fallback if fallback is not None else torch.zeros(self.kde_dim)
    vecs = [self.kde_dict.get((int(s), int(d))) for s, d in zip(src_t.tolist(), dst_t.tolist())]
    result = torch.stack([
        torch.tensor(v, dtype=torch.float32) if v is not None else zero
        for v in vecs
    ])
    return result
```
**Expected effect**: Unseen edges at test time (likely attack edges on novel `(src,dst)` pairs) get a distinctive learned representation rather than a zero vector that looks identical to all other unseen edges. This gives the MLP a signal that the edge is *new* rather than just *zero-density*.
**Complexity**: ~15 lines of code across two files.

#### Improvement 4 — KDE as Node Feature Aggregation (Integration Point 2)
**Mechanism**: For each node, compute the **mean KDE vector** across all its training edges. Append this K-dim vector to the one-hot node features, giving nodes a "temporal signature" based on their historical connectivity patterns. Nodes that appear only in attacks (new IPs) will have zero mean KDE (not seen in training), which the GCN can propagate differently.

In `TData.__init__`, precompute:
```python
if kde_dict is not None:
    node_kde = torch.zeros(num_nodes, kde_dim)
    node_kde_cnt = torch.zeros(num_nodes)
    for (src, dst), vec in kde_dict.items():
        node_kde[src] += torch.tensor(vec)
        node_kde_cnt[src] += 1
    mask = node_kde_cnt > 0
    node_kde[mask] /= node_kde_cnt[mask].unsqueeze(1)
    self.xs = torch.cat([self.xs, node_kde], dim=1)  # (N, N+K)
```
This requires updating `Argus_LANL.c1` input size to `x_dim + K`.
**Expected effect**: Attack edges from nodes not in the training graph get a distinctive zero-KDE node embedding that the GCN propagates. This is a *structural* signal rather than a per-instance signal.
**Complexity**: ~20 lines; requires adjusting `c1` input dimension in `Argus_LANL`.

#### Improvement 5 — Separate KDE Branch with Residual Injection into NNConv (Option A fixed)
**Mechanism**: Project KDE vectors to 6-dim via a small linear layer *before* concatenating into `eas`, so the NNConv `nn4` MLP always sees 6+6=12 consistently-scaled features. Non-KDE edges get a projected zero, but the projection is learnable and can specialise.

Alternatively: run a **separate NNConv layer** `c4_kde` using only KDE vectors as edge attributes, and *add* its output to `c4`'s output:
```python
z_flow = self.c4(x, ei, edge_attr=ea_flow)      # uses 6-dim flow ea
z_kde  = self.c4_kde(x, ei, edge_attr=ea_kde)   # uses K-dim KDE ea
z = self.ac(z_flow + z_kde)
```
Non-KDE edges have zero KDE edge attr → `c4_kde` output is zero → pure flow path. KDE edges have both paths active → richer representation.
**Expected effect**: Cleanly separates the two feature types, avoiding the zero-domination problem of Option A while keeping KDE in the message-passing (not just decode) step.
**Complexity**: ~30 lines; requires storing KDE in `eas` separately from flow features, or using `get_kde_tensor()` inside `forward_once()`.

#### Improvement 6 — Dynamic KDE: Per-Snapshot KDE Update
**Mechanism**: The fundamental limitation of static KDE is that all occurrences of edge `(A, B)` — whether normal or attack — get the same precomputed vector. A dynamic approach would update the KDE estimate as the test window advances:
1. Start with training KDE vectors
2. After each test snapshot, update the KDE for edges seen in that snapshot using a running DPGMM
3. Use the *updated* KDE vector for the *next* snapshot's decode

This turns KDE from a static per-edge feature into a **time-evolving signal**. Attack bursts would produce timestamp differences very different from the training distribution → the KDE vector diverges from the training estimate → anomaly signal increases over time.
**Expected effect**: Highest potential improvement. Removes the ceiling of static KDE.
**Complexity**: High. Requires online DPGMM updates in `classification.py`'s test loop.

---

### 18.5 Summary Table

| Improvement | Changes Required | Expected TP Gain | Complexity |
|-------------|-----------------|------------------|------------|
| 1. Lower fpweight | CLI arg only | Direct, tunable | Trivial |
| 2. Higher nratio | CLI arg only | Indirect (better separation) | Trivial |
| 3. Learned fallback vector | ~15 lines | Small-moderate | Low |
| 4. KDE as node feature | ~20 lines + c1 resize | Moderate | Medium |
| 5. Separate KDE NNConv branch | ~30 lines | Moderate-high | Medium |
| 6. Dynamic per-snapshot KDE | ~100+ lines | High (removes ceiling) | High |

---

## 19. Prediction Saving & Per-Attack-Type Analysis

### 19.1 Overview

Starting with the 6-config benchmark run, ARGUS now saves per-edge predictions
to a pickle file after each `test()` call. This enables post-hoc analysis of
which CIC-IDS 2017 attack types are detected (TP) vs missed (FN) by each config.

### 19.2 What Changed

| File | Change |
|------|--------|
| `models/argus.py` | `DetectorEncoder.decode_all()` returns 4-tuple `(preds, ys, cnts, eis_out)` — appends `self.module.data.eis[i].detach().cpu().numpy()` per snapshot. `Argus.score_all()` unpacks and propagates the edge-index list. |
| `classification.py` | Added `PREDICTIONS_FILE = 'predictions_latest.pkl'` constant. Added `_save_predictions(scores, labels, weights, eis, cutoff)` helper. `test()` now calls `scores, labels, weights, eis = model.score_all(zs)` then `_save_predictions(...)`. |

### 19.3 Predictions Pickle Format

```python
{
    'scores':  [np.array, ...],   # one per test snapshot
    'labels':  [np.array, ...],   # one per test snapshot
    'weights': [np.array, ...],   # one per test snapshot
    'eis':     [np.array(2,E), ...],  # edge indices per snapshot
    'cutoff':  float              # optimal cutoff from calibration
}
```

### 19.4 Three Run Configurations (L_cic_flow only)

| Config | CLI flags | Data dir | Description |
|--------|-----------|----------|-------------|
| **Baseline** | (none) | `argus_flow/` | Standard ARGUS GCN+GRU with flow features |
| **KDE** | `--kde --kde_file kde_vectors_argus.pkl --kde_dim 10` | `argus_flow/` | KDE features added at decode step via MLP |
| **Reduced** | `--red` | `argus_flow_red/` | Reduced graph (fewer nodes/edges) |

All three use: `--dataset L_cic_flow --data_dir <path> --delta 10 --lr 0.05 --hidden 32 -z 16 --fpweight 0.45 --epochs 100 --patience 3`

### 19.5 Output Structure

```
results/<timestamp>/
  ARGUS/
    baseline/
      results_cic.txt        # metrics from single run
      predictions.pkl         # per-edge predictions
    kde/
      results_cic.txt
      predictions.pkl
    reduced/
      results_cic.txt
      predictions.pkl
```

### 19.6 Post-Hoc Attack-Type Mapping

`common_analysis/map_attacks_to_cic_types.py --results_dir <path>` reads each
`predictions.pkl`, maps edge indices → IP addresses via `nmap.pkl`, joins against
the CIC-IDS 2017 master CSV to determine attack types, and computes per-type
TP/FP/TN/FN using the model's cutoff.

### 19.7 Current File Structure

```
ARGUS/
├── main.py                 # CLI entry point (--kde, --red, --kde_file, --kde_dim flags)
├── classification.py       # Train/test logic + _save_predictions()
├── utils.py                # Scoring utilities
├── models/
│   └── argus.py            # DetectorEncoder (decode_all returns eis), Argus wrapper
├── loaders/
│   ├── load_cic_flow.py    # L_cic_flow data loading (supports KDE + reduced)
│   ├── load_cic.py         # O_cic data loading
│   └── tdata.py            # TData container with get_kde_tensor()
├── libauc/                 # APLoss + SOAP optimizer
├── generate_reduced_argus_flow.py  # Utility to create argus_flow_red/
└── kde_vectors_argus.pkl   # Precomputed KDE edge features
```

Data directories (under `1-DATA_PROCESSING/cic_2017/`):
```
argus_flow/                 # Full graph data with flow features
argus_flow_red/             # Reduced graph data with flow features
```