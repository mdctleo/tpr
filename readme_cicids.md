# CIC-IDS-2017 Integration — Pipeline Deep-Dive

> **Branch:** `kde_ts_CIS_IDS`  
> **HEAD commit:** `0f3eb4b` — *CIC IDS 2017 integration*  
> **Prior commit:** `12266a8` — *Added config for KDE of timestamp diffs*

---

## Table of Contents

1. [What Changed (Commit Diff Summary)](#1-what-changed-commit-diff-summary)
2. [Dataset Overview: CIC-IDS-2017](#2-dataset-overview-cic-ids-2017)
3. [Benchmarked Methods](#3-benchmarked-methods)
4. [Data Ingestion & Ground-Truth Generation](#4-data-ingestion--ground-truth-generation)
5. [Preprocessing: KDE Computation (`kde_computation.py`)](#5-preprocessing-kde-computation)
6. [Preprocessing: Graph Reduction (`reduce_graphs_kde.py`)](#6-preprocessing-graph-reduction)
7. [The Full Pipeline — Stage by Stage](#7-the-full-pipeline--stage-by-stage)
   - [Stage 1: Construction](#stage-1-construction)
   - [Stage 2: Transformation](#stage-2-transformation)
   - [Stage 3: Featurization](#stage-3-featurization)
   - [Stage 4: Feat Inference](#stage-4-feat-inference)
   - [Stage 5: Batching](#stage-5-batching)
   - [Stage 6: Training](#stage-6-training)
   - [Stage 7: Evaluation](#stage-7-evaluation)
   - [Stage 8: Triage](#stage-8-triage)
8. [Per-Attack Evaluation Mechanism](#8-per-attack-evaluation-mechanism)
9. [KDE Patch (Runtime Monkey-Patching)](#9-kde-patch-runtime-monkey-patching)
10. [Configuration Reference](#10-configuration-reference)
11. [Reproduction Commands](#11-reproduction-commands)

---

## 1. What Changed (Commit Diff Summary)

Between `12266a8` (KDE diff config for DARPA) and `0f3eb4b` (CIC-IDS integration), **56 files** were touched (+7 526 / −684 lines). The key changes:

| Area | Files | Summary |
|------|-------|---------|
| **Dataset config** | `config.py` (+168 lines) | Added `CIC_IDS_2017` and `CIC_IDS_2017_PER_ATTACK` dataset definitions with all 16 attack ground-truth entries, per-attack test file mapping, and 1 node type / 3 edge types |
| **YAML configs** | 9 new `config/*_cicids*.yml` | Baseline, KDE-ts, KDE-diff, and RED configs for both Orthrus and Kairos |
| **Ground truth** | 16 new CSV files in `Ground_Truth/CIC-IDS-2017/` | Per-attack node-level ground truth (attacker + victim IPs) |
| **DB ingestion** | `create_database_cic_ids_2017.py` (312 lines) | Reads 5 raw CSV files → PostgreSQL; maps IPs as netflow nodes, flows as events |
| **GT generation** | `generate_ground_truth.py` (311 lines) | Defines 16 attack chains with attacker/victim IPs; writes PIDSMaker-format CSVs |
| **KDE computation** | `kde_computation.py` (major rewrite) | GPU-accelerated DPGMM (`BayesianGaussianMixtureGPU`), CIC-IDS noon anchor, summary-stat mode, parallel file loading, configurable `artifacts_dir` |
| **Graph reduction** | `reduce_graphs_kde.py` (expanded) | `--all_hashes`, `--symlink_stages`, per-dataset dimension configs, `done.txt` markers |
| **Pipeline core** | `pipeline.py`, `factory.py`, `main.py` | `ground_truth_version: none`, flexible `get_darpa_tc_node_feats_from_cfg`, day-deduplication, KDE-patch detection via `use_precomputed` |
| **Graph building** | `build_magic_graphs.py` | Per-attack test graph generation: `build_per_attack_test_graphs()`, `filter_events_for_attack()` |
| **Evaluation** | `evaluation.py`, `evaluation_utils.py`, `node_evaluation.py` | Per-attack evaluation loop: isolated metrics per attack, cross-attack averaging |
| **Labelling** | `labelling.py` | `get_GP_of_each_attack()`, flexible node-table queries for CIC-IDS (only netflow) |
| **Dataset utils** | `dataset_utils.py` | CIC-IDS edge types `{TCP, UDP, Other}`, node types `{netflow}`, triplet features |

---

## 2. Dataset Overview: CIC-IDS-2017

CIC-IDS-2017 is a **network intrusion detection** dataset from the Canadian Institute for Cybersecurity. It contains 5 days of traffic (Monday–Friday, July 3–7, 2017) captured as NetFlow records.

| Property | Value |
|----------|-------|
| **Node types** | 1 (`netflow` — IP addresses) |
| **Edge types** | 3 (`TCP`, `UDP`, `Other`) |
| **Days** | 5 (Mon=day 3, Tue=day 4, Wed=day 5, Thu=day 6, Fri=day 7) |
| **Train split** | Monday (day 3) — benign only |
| **Val split** | Tuesday (day 4) — benign + FTP/SSH Patator |
| **Test split** | Wed + Thu + Fri (days 5–7) — benign + 14 distinct attacks |
| **Timestamp unit** | nanoseconds |

### Attack Inventory (14 test attacks)

| Day | Attack | Time Window (EDT) |
|-----|--------|-------------------|
| Wednesday | DoS Slowloris | 09:47–10:10 |
| Wednesday | DoS Slowhttptest | 10:14–10:35 |
| Wednesday | DoS Hulk | 10:43–11:00 |
| Wednesday | DoS GoldenEye | 11:10–11:23 |
| Wednesday | Heartbleed | 15:12–15:32 |
| Thursday | Web Brute Force | 09:20–10:00 |
| Thursday | Web XSS | 10:15–10:35 |
| Thursday | Web SQL Injection | 10:40–10:42 |
| Thursday | Infiltration Step 1 | 14:19–14:35 |
| Thursday | Infiltration CoolDisk | 14:53–15:00 |
| Thursday | Infiltration Step 2 | 15:04–15:45 |
| Friday | Botnet ARES | 10:02–11:02 |
| Friday | Port Scan | 13:55–15:29 |
| Friday | DDoS LOIT | 15:56–16:16 |

---

## 3. Benchmarked Methods

Four time-encoding variants are benchmarked, each on **two base models** (Orthrus and Kairos):

| Method | Config suffix | Time Encoding | Edge Reduction | RKHS dim |
|--------|--------------|---------------|----------------|----------|
| **Baseline** | `_cicids` / `_cicids_og` | Standard TGN cosine | None | N/A |
| **KDE of raw timestamps** | `_cicids_kde_ts` | GPU DPGMM on raw ts | KDE-eligible edges collapsed | 20 |
| **KDE of timestamp diffs** | `_cicids_kde_diff` | GPU DPGMM on inter-arrival times | KDE-eligible edges collapsed | 20 |
| **Summary stats (RED)** | `_cicids_red` | (first_ts, last_ts, count) | All edges collapsed to 1 representative | 3 |

---

## 4. Data Ingestion & Ground-Truth Generation

### 4a. Database Creation (`create_database_cic_ids_2017.py`)

Reads the 5 raw CSV files (`Monday-WorkingHours.csv` … `Friday-WorkingHours.csv`) and loads them into PostgreSQL under the `cic_ids_2017` database.

**Three-phase ingestion:**

1. **Phase 1 — IP Discovery:** Scans all 5 CSVs to collect every unique IP address that appears as source or destination.

2. **Phase 2 — Node Table Population:** Inserts IPs into `netflow_node_table` in sorted order. Each IP gets a deterministic integer `index_id`. The `node_uuid` is the raw IP string itself. Tables `subject_node_table` and `file_node_table` remain **empty** (CIC-IDS is purely network-level).

3. **Phase 3 — Event Ingestion:** For each CSV, a subprocess worker:
   - Maps each flow to `(src_ip, src_index_id, protocol, dst_ip, dst_index_id, event_uuid, timestamp_ns)`.
   - Writes to a temp TSV file.
   - Bulk-loads via `COPY FROM` for performance.
   - Protocols are mapped: `6 → TCP`, `17 → UDP`, everything else → `Other`.
   - CSV day mapping: `Monday → day 3`, `Tuesday → day 4`, … `Friday → day 7`.

### 4b. Ground-Truth Generation (`generate_ground_truth.py`)

Generates 16 per-attack ground-truth CSV files in `Ground_Truth/CIC-IDS-2017/`.

Each CSV has three columns: `node_uuid` (IP address), `label` (`attacker` / `victim`), `placeholder`.

**Key attacker IPs across all attacks:**

| IP | Role |
|----|------|
| `172.16.0.1` | Firewall/NAT — always marked attacker |
| `205.174.165.73` | Kali — external attacker |
| `192.168.10.8` | Win Vista — becomes attacker in Infiltration Step 2 |
| `205.174.165.69/70/71` | External Win machines (DDoS LOIT) |

---

## 5. Preprocessing: KDE Computation

**Script:** `kde_computation.py`

This is the **offline preprocessing** step that pre-computes edge-level temporal feature vectors. It runs *before* the main pipeline and produces `.pt` files consumed at training time.

### 5a. High-Level Flow

```
.TemporalData.simple files
         │
    ┌────▼────────────────────┐
    │  extract_edge_timestamps │  ← parallel file loading (ThreadPoolExecutor)
    │  Group events by          │    vectorised grouping (np.lexsort + boundary detection)
    │  (src, dst, edge_type)   │
    └────┬────────────────────┘
         │ Dict[(src,dst,et)] → List[float]
         │
    ┌────▼────────────────────┐
    │  filter_merged_edges     │  ← keep edges with ≥ min_occurrences (default 10)
    │                          │    for kde_diff: compute |diff(sorted timestamps)|
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  [CIC-IDS only, kde_ts]  │  ← _anchor_cicids_noon(): shift mean → 12:30 PM
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  scale_merged_edges      │  ← MaxAbsScaler per edge → data in [-1, 1]
    └────┬────────────────────┘
         │
    ┌────▼────────────────────┐
    │  preprocess_long_edges   │  ← subsample edges > 50K timestamps
    └────┬────────────────────┘
         │
    ┌────▼────────────────────────────────┐
    │  fit_batched_gpu_dpgmm (GPU path)    │
    │  ┌─────────────────────────────┐     │
    │  │ BayesianGaussianMixtureGPU  │     │  ← Variational Bayesian GMM
    │  │  • Stick-breaking DP prior  │     │    (CAVI with NIG conjugacy)
    │  │  • K-means++ init           │     │
    │  │  • Batched on GPU (256/chunk)│    │
    │  └──────────┬──────────────────┘     │
    │             │                        │
    │  ┌──────────▼──────────────────┐     │
    │  │ score_samples_grid          │     │  ← evaluate density on rkhs_dim-point
    │  │ → L1-normalised density     │     │    uniform grid per edge
    │  └─────────────────────────────┘     │
    └────┬────────────────────────────────┘
         │
    ┌────▼────────────────────┐
    │  save_results            │  → {dataset}_kde_vectors.pt
    │                          │  → {dataset}_kde_stats.json
    └──────────────────────────┘
```

### 5b. Three Computation Modes

#### Mode 1: `kde_ts` — KDE on Raw Timestamps

Each edge's raw timestamp sequence is used directly:

1. **CIC-IDS noon anchor** (CIC-IDS only): Each edge's timestamps are shifted so the mean falls at 12:30 PM (45,000 seconds past midnight). This normalises across the 5 weekdays.
2. **MaxAbsScaler**: Divide by max absolute value → data in [-1, 1].
3. **GPU DPGMM**: Fit a Bayesian Gaussian Mixture (K=100 components, γ=5.0, stick-breaking DP prior) via CAVI. Evaluate the fitted density on a 20-point uniform grid → 20-D vector.

#### Mode 2: `kde_diff` — KDE on Timestamp Differences

Captures the *tempo* (inter-arrival pattern) rather than absolute timing:

1. Sort timestamps, compute `|diff(sorted timestamps)|` → inter-arrival times.
2. **MaxAbsScaler** → data in [-1, 1].
3. **GPU DPGMM** → 20-D density vector (same as kde_ts).

#### Mode 3: `red` (Summary Stats)

A simple, non-parametric baseline:

1. For **every** edge (no min_occurrences filter):
   - `first_ts = min(timestamps)`
   - `last_ts = max(timestamps)`
   - `count = len(timestamps)`
2. Output: 3-D vector `(first_ts, last_ts, count)`.

### 5c. The GPU DPGMM Model (`BayesianGaussianMixtureGPU`)

A fully GPU-accelerated variational Bayesian Gaussian mixture for batched 1-D data.

**Conjugate model:**
- $v_k \sim \text{Beta}(1, \gamma)$ — stick-breaking (K-truncated)
- $\mu_k \sim \mathcal{N}(\mu_0, 1/\lambda_0)$
- $\sigma^2_k \sim \text{InvGamma}(\alpha_0, \beta_0)$
- $z_n \sim \text{Categorical}(\pi(v))$
- $x_n | z_n = k \sim \mathcal{N}(\mu_k, \sigma^2_k)$

**CAVI updates** (per iteration):
- **E-step:** Compute log responsibilities $\log r_{nk}$ from expected log-likelihoods and stick-breaking weights.
- **M-step:** Closed-form Normal-Inverse-Gamma conjugate updates for $(\mu_k, \lambda_k, \alpha_k, \beta_k)$ and Beta updates for stick-breaking parameters.
- **Convergence:** ELBO-based, with early-stop patience.

**Density evaluation:** After fitting, the mixture density is evaluated on a uniform grid of `rkhs_dim` (default 20) points spanning each edge's `[min, max]`. The resulting density is L1-normalised.

### 5d. CLI Usage

```bash
# Raw timestamps (kde_ts)
python kde_computation.py kairos_cicids_kde_ts CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors --artifacts_dir artifacts_cicids

# Timestamp diffs (kde_diff)
python kde_computation.py kairos_cicids_kde_diff CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors_diff --artifacts_dir artifacts_cicids --use_timestamp_diffs

# Summary stats (RED)
python kde_computation.py kairos_cicids_red CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors_red --artifacts_dir artifacts_cicids --use_summary_stats

# Specific feat_inference hash
python kde_computation.py kairos_cicids_kde_ts CIC_IDS_2017_PER_ATTACK \
    --feat_hash fe201634 --artifacts_dir artifacts_cicids
```

---

## 6. Preprocessing: Graph Reduction

**Script:** `scripts/reduce_graphs_kde.py`

After KDE vectors are computed, this script **reduces the graph artifacts** by collapsing multi-occurrence edges into single representatives. This reduces training graph size without losing temporal information (which is now captured by the precomputed KDE/summary vectors).

### 6a. How Reduction Works

For each `.TemporalData.simple` file:

1. **Load** the edge data (src, dst, msg, t, y).
2. **Extract edge types** from the msg tensor (one-hot encoded at position `node_type_dim + emb_dim`).
3. **Classify edges:**
   - **KDE-eligible** (key exists in the precomputed vectors): All occurrences of `(src, dst, edge_type)` are collapsed to the **first occurrence** (sorted by timestamp). The temporal history is captured by the precomputed KDE vector.
   - **Non-KDE** (< `min_occurrences` timestamps across all files): All occurrences are **kept as-is** to preserve their individual timestamps.
4. **Save** the reduced graph to the output directory.

### 6b. Output Structure

```
artifacts_cicids_kde_ts_reduced/
├── construction/  → symlink to artifacts_cicids/construction/
├── transformation/ → symlink to artifacts_cicids/transformation/
├── featurization/ → symlink to artifacts_cicids/featurization/
└── feat_inference/
    └── CIC_IDS_2017_PER_ATTACK/
        └── feat_inference/
            └── <hash>/
                ├── done.txt
                └── edge_embeds/
                    ├── train/ ← reduced
                    ├── val/   ← reduced
                    └── test/  ← copied unchanged
```

### 6c. CLI Usage

```bash
# Reduce all hashes + symlink prior stages
python scripts/reduce_graphs_kde.py CIC_IDS_2017_PER_ATTACK \
    --artifacts_dir artifacts_cicids \
    --kde_vectors_dir kde_vectors \
    --output_suffix _kde_ts_reduced \
    --all_hashes --symlink_stages
```

> **Note for CIC-IDS:** Per-file reduction gives only ~3.5% because the same (src, dst, type) edge rarely repeats within a single 1-minute time window. The benefit comes from the KDE temporal encoding itself, not the graph size reduction.

---

## 7. The Full Pipeline — Stage by Stage

The main pipeline is orchestrated by `pidsmaker/main.py` and executes **8 sequential stages**:

```
construction → transformation → featurization → feat_inference → batching → training → evaluation → triage
```

Each stage checks a `done.txt` marker in its output directory. If the marker exists, the stage is skipped (restart logic). The `--force_restart <stage>` flag forces re-execution from that stage onward.

---

### Stage 1: Construction

**Module:** `pidsmaker/tasks/construction.py`  
**Builder:** `build_magic_graphs.py`

**Purpose:** Query PostgreSQL and build time-windowed NetworkX directed graphs.

**How it works for CIC-IDS:**

1. **Node loading:** Queries `netflow_node_table` (only table with data for CIC-IDS). Builds `nid → node_uuid` and `node_uuid → nid` mappings. For CIC-IDS, `node_uuid` = raw IP address.

2. **Day-by-day graph construction:** For each day in `start_end_day_range` (3–7):
   - Queries `event_table` ordered by `timestamp_rec`.
   - Filters events by valid edge types (`rel2id` = `{TCP: 1, UDP: 2, Other: 3}`).
   - Creates time-windowed graphs: each window is `time_window_size` minutes (default 1). Within each window, all edges are added to a single NetworkX DiGraph.
   - Saves as `graph_{day}/graph_{window_idx}.pt`.

3. **Per-attack test graph generation** (when `per_attack_test_graphs: True`):
   - For each test file (e.g., `graph_5_dos_slowloris`):
     - Parses the day number (5 → Wednesday).
     - Loads the full day's events from PostgreSQL.
     - Calls `filter_events_for_attack()`: keeps only **benign traffic + the target attack**. Events from *other* attacks on the same day are removed.
     - Creates time-windowed graphs and saves to `graph_5_dos_slowloris/graph_{idx}.pt`.
   - This ensures each test graph contains exactly one attack against a benign background, enabling isolated evaluation.

**Input:** PostgreSQL database (`cic_ids_2017`)  
**Output:** `{artifacts_dir}/construction/{dataset}/construction/<hash>/`

---

### Stage 2: Transformation

**Module:** `pidsmaker/tasks/transformation.py`

**Purpose:** Apply graph-level transformations (e.g., undirected, DAG conversion).

**CIC-IDS:** Uses `"none"` transformation — the construction output is simply copied to the transformation directory.

**Input:** Construction output graphs  
**Output:** `{artifacts_dir}/transformation/{dataset}/transformation/<hash>/`

---

### Stage 3: Featurization

**Module:** `pidsmaker/tasks/featurization.py`

**Purpose:** Train or build node embedding models (Word2Vec, Doc2Vec, etc.).

**CIC-IDS:** Uses `"only_type"` or `"magic"` featurization method. Both are **no-ops** at this stage — embeddings are computed directly in feat_inference using one-hot type vectors.

**Input:** Transformation output  
**Output:** `{artifacts_dir}/featurization/{dataset}/featurization/<hash>/` (mostly empty for CIC-IDS)

---

### Stage 4: Feat Inference

**Module:** `pidsmaker/tasks/feat_inference.py`

**Purpose:** Convert NetworkX graphs into `TemporalData` objects with edge-level message vectors (the `msg` tensor).

**How the msg tensor is structured:**

```
msg = [src_node_type_onehot | src_embedding | edge_type_onehot | dst_node_type_onehot | dst_embedding]
```

For CIC-IDS (1 node type, 3 edge types, no node embeddings with `only_type`):
- `src_node_type_onehot`: 1-D (just `[1]` for netflow)
- `edge_type_onehot`: 3-D (`TCP`, `UDP`, `Other`)
- `dst_node_type_onehot`: 1-D
- `src_embedding` / `dst_embedding`: may be zero-length or small depending on the featurization method

**Output per graph file:** A `.TemporalData.simple` file containing:
- `src`: source node indices (int tensor)
- `dst`: destination node indices (int tensor)
- `t`: timestamps (float tensor, nanoseconds)
- `msg`: message vectors (float tensor, shape `[num_edges, msg_dim]`)
- `y`: labels (zeros for benign, ones for malicious — though labelling happens later)

**Per-attack handling:** Per-attack test files (e.g., `graph_5_dos_slowloris`) are processed into separate directories. Filenames use `__` separators to keep per-attack files distinct.

**Input:** Transformation output + featurization models  
**Output:** `{artifacts_dir}/feat_inference/{dataset}/feat_inference/<hash>/edge_embeds/{train,val,test}/`

---

### Stage 5: Batching

**Module:** `pidsmaker/tasks/batching.py`

**Purpose:** Pre-process and optionally save all datasets to disk for faster training iteration.

If `presave_batching` is enabled, all `TemporalData` objects are collated and saved as a single `torch_graphs.pkl` per split. Otherwise, this stage is a no-op and data is loaded on-the-fly during training.

**Input:** Feat inference output  
**Output:** `{artifacts_dir}/batching/{dataset}/batching/<hash>/`

---

### Stage 6: Training

**Module:** `pidsmaker/tasks/training.py` → `training_loop.py`

**Purpose:** Train the temporal graph neural network (TGN-based encoder + decoder).

**How training works:**

1. **Load batched temporal data** from the batching stage output.
2. **For each epoch:**
   - Iterate over time windows in the training split.
   - For each batch, the TGN encoder processes the `msg` tensor, updates node memory, and computes embeddings via temporal attention.
   - The decoder (link prediction / reconstruction) computes edge-level losses.
   - Backpropagation + optimizer step.
3. **Per-epoch edge losses** are saved to disk for the evaluation stage.
4. **Validation:** After each epoch, compute losses on the validation split. The best validation score is returned to the main loop.

**KDE-patched training:** When a KDE config is active (`use_precomputed: true`), the time encoder is monkey-patched to use precomputed RKHS vectors instead of standard cosine time encoding (see [Section 9](#9-kde-patch-runtime-monkey-patching)).

**Input:** Batching output  
**Output:** `{artifacts_dir}/training/{dataset}/training/<hash>/` (model checkpoints + edge losses)

---

### Stage 7: Evaluation

**Module:** `pidsmaker/tasks/evaluation.py`

**Purpose:** Evaluate the trained model's anomaly detection performance.

**Standard evaluation flow:**

1. **Compute time-window labels** (`compute_tw_labels`): Map ground-truth attack time windows to time-window indices in the test data.
2. **For each training epoch's saved edge losses:**
   - Load edge losses for test and validation splits.
   - Aggregate edge losses to node-level scores.
   - Compute precision, recall, F-score, AP, AUC, accuracy.
   - Select the best epoch by ADP score or discrimination.
3. **Log results** to Weights & Biases.

**Per-attack evaluation** (CIC-IDS-2017 specific — see [Section 8](#8-per-attack-evaluation-mechanism) for full details).

**Input:** Training edge losses + ground truth  
**Output:** `{artifacts_dir}/evaluation/{dataset}/evaluation/<hash>/results/` + W&B logs

---

### Stage 8: Triage

**Module:** `pidsmaker/tasks/triage.py`

**Purpose:** Post-hoc analysis of flagged nodes, neighbourhood visualization, and optional attack graph tracing.

For CIC-IDS, this stage is typically not the primary focus — the per-attack metrics from evaluation are the main deliverable.

---

## 8. Per-Attack Evaluation Mechanism

The per-attack evaluation is the core innovation for CIC-IDS benchmarking. It enables **isolated evaluation** of each attack type.

### How It Works End-to-End

**1. Config setup** (`CIC_IDS_2017_PER_ATTACK` in `config.py`):
- `per_attack_test_graphs: True` enables per-attack graph generation.
- `test_files` lists 14 entries like `graph_5_dos_slowloris`, `graph_7_ddos_loit`, etc.
- `test_file_to_attack_idx` maps each test file to an attack index (0–13).
- `attack_to_time_window` defines the time boundaries of each attack.

**2. Construction** (`build_magic_graphs.py`):
- Test days (Wed/Thu/Fri) are skipped in the main loop.
- `build_per_attack_test_graphs()` creates separate graph directories per attack.
- `filter_events_for_attack()` keeps only benign traffic + the single target attack for that graph.

**3. Feat inference:**
- Per-attack graph directories (e.g., `graph_5_dos_slowloris/`) are processed individually.
- Output files are named with `__` separators to prevent collisions.

**4. Evaluation** (`evaluation.py`):
```
For each training epoch:
    For each test_file in cfg.dataset.test_files:
        attack_idx = test_file_to_attack_idx[test_file]
        selected_tw_indices = attack_to_tw_indices[attack_idx]
        attack_gp = attack_to_GPs_all[attack_idx]
        
        stats = evaluation_fn(
            val_path, test_path, epoch, cfg,
            selected_tw_indices=selected_tw_indices,
            ground_truth_nids=attack_gp["nids"],
        )
        per_attack_stats.append((test_file, stats))
    
    # Average metrics across all 14 attacks
    global_stats = {key: mean([s[key] for s in per_attack_stats])}
    
    # Also log per-attack stats with prefixed keys
    # e.g., "graph_5_dos_slowloris/precision", "graph_7_ddos_loit/recall"
```

**Key functions:**
- `build_attack_to_tw_indices(cfg)` → maps attack index → list of time-window indices that belong to that attack in the test data.
- `get_GP_of_each_attack(cfg)` → loads per-attack ground-truth node IDs.
- `compute_tw_labels(cfg)` → maps time windows to malicious node sets.

---

## 9. KDE Patch (Runtime Monkey-Patching)

**Module:** `pidsmaker/kde_patch.py`

When a KDE config is active (`use_precomputed: true` in `kde_params`), the main pipeline applies a monkey-patch that replaces the TGN time encoder.

### What Gets Patched

1. **`tgn_module.TimeEncoder`** is replaced with `KDEPatchedTimeEncoder`:
   - Contains a `PrecomputedKDETimeEncoder` that loads RKHS vectors from `kde_vectors/` or `kde_vectors_diff/` or `kde_vectors_red/`.
   - For edges **with** precomputed vectors: `output = Linear(rkhs_vector)` — a trainable projection from RKHS dim → time encoding dim.
   - For edges **without** vectors (< min_occurrences): `output = Linear(t_diff).cos()` — standard cosine fallback.

2. **`TGNEncoder.forward`** is patched to pass `(src, dst, edge_type)` to the time encoder for KDE vector lookup.

### Lookup Mechanism

The `RKHSVectorLoader` maintains a dict mapping `(src, dst, edge_type)` → `Tensor[rkhs_dim]`. At training time, batch lookups are performed:
- Extract `src`, `dst`, `edge_type` from the current batch.
- Look up each edge's RKHS vector.
- Return a `(batch_size, rkhs_dim)` matrix + a boolean mask indicating which edges had vectors.

### Activation Logic

In `main.py`:
```python
if KDE_PATCH_AVAILABLE and hasattr(cfg, 'kde_params') and getattr(cfg.kde_params, 'use_precomputed', False):
    patch_for_kde_time_encoding(cfg)
```

This covers all KDE variants (kde_ts, kde_diff, and RED) — not just configs with "kde" in the name.

---

## 10. Configuration Reference

### Baseline Configs (no KDE)

| Config | Base | Description |
|--------|------|-------------|
| `kairos_cicids.yml` | `kairos` | Kairos baseline on CIC-IDS |
| `orthrus_cicids.yml` | `orthrus_non_snooped_edge_ts` | Orthrus baseline on CIC-IDS |

### KDE Timestamp Configs

| Config | KDE Mode | RKHS dim | min_occ | Vectors Dir |
|--------|----------|----------|---------|-------------|
| `kairos_cicids_kde_ts.yml` | raw timestamps | 20 | 10 | `kde_vectors` |
| `orthrus_cicids_kde_ts.yml` | raw timestamps | 20 | 10 | `kde_vectors` |

### KDE Diff Configs

| Config | KDE Mode | RKHS dim | min_occ | Vectors Dir |
|--------|----------|----------|---------|-------------|
| `kairos_cicids_kde_diff.yml` | timestamp diffs | 20 | 10 | `kde_vectors_diff` |
| `orthrus_cicids_kde_diff.yml` | timestamp diffs | 20 | 10 | `kde_vectors_diff` |

### Summary Stats (RED) Configs

| Config | KDE Mode | RKHS dim | min_occ | Vectors Dir |
|--------|----------|----------|---------|-------------|
| `kairos_cicids_red.yml` | summary stats | 3 | 1 | `kde_vectors_red` |
| `orthrus_cicids_red.yml` | summary stats | 3 | 1 | `kde_vectors_red_orthrus` |

### Common KDE Parameters (all kde_ts/kde_diff configs)

```yaml
kde_params:
  n_components: 100        # DP truncation level K
  gamma: 5.0               # DP concentration
  max_iter: 300             # Max CAVI iterations
  tol: 1.0e-4              # Convergence tolerance
  patience: 20             # Early-stop patience
  n_init: 1                # Random restarts
  init_method: 'kmeans'    # K-means++ initialization
  truncate_threshold: 0.99 # Component pruning cutoff
  max_n_per_edge: 50000    # Subsample long edges
  chunk_size: 256          # GPU batch size
  feat_inference_dir: artifacts_cicids  # Artifacts base dir
  use_precomputed: true    # Use offline vectors
```

---

## 11. Reproduction Commands

### Prerequisites

```bash
# Start the PostgreSQL + Apptainer environment
cd /scratch/asawan15/PIDSMaker/scripts/apptainer
make up
cd ../../

# Ingest CIC-IDS-2017 into PostgreSQL (one-time)
python dataset_preprocessing/cic_ids_2017/create_database_cic_ids_2017.py \
    --input /path/to/CIC-IDS-2017/ --db-name cic_ids_2017

# Generate ground truth (one-time)
python dataset_preprocessing/cic_ids_2017/generate_ground_truth.py \
    --input /path/to/CIC-IDS-2017/
```

### Step 1: Run Baseline (builds construction → evaluation)

```bash
# Kairos baseline
python -m pidsmaker.main kairos_cicids CIC_IDS_2017_PER_ATTACK \
    --training.encoder.dropout=0.3 --training.lr=0.0001 \
    --training.node_hid_dim=256 --training.node_out_dim=256 \
    --training.num_epochs=12 --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --artifact_dir ./artifacts_cicids/ --database_host localhost \
    --force_restart evaluation --wandb

# Orthrus baseline
python -m pidsmaker.main orthrus_cicids CIC_IDS_2017_PER_ATTACK \
    --training.encoder.dropout=0.3 --training.lr=0.0001 \
    --training.node_hid_dim=64 --training.node_out_dim=64 \
    --training.num_epochs=12 --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --artifact_dir ./artifacts_cicids/ --database_host localhost \
    --force_restart evaluation --wandb
```

### Step 2: Compute KDE Vectors (offline)

```bash
# KDE of raw timestamps
python kde_computation.py kairos_cicids_kde_ts CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors --artifacts_dir artifacts_cicids

# KDE of timestamp diffs
python kde_computation.py kairos_cicids_kde_diff CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors_diff --artifacts_dir artifacts_cicids --use_timestamp_diffs

# Summary stats (RED) for Orthrus
python kde_computation.py orthrus_cicids_red CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors_red_orthrus --artifacts_dir artifacts_cicids \
    --use_summary_stats --feat_hash 4c773cf1

# Summary stats (RED) for Kairos
python kde_computation.py kairos_cicids_red CIC_IDS_2017_PER_ATTACK \
    --output_dir kde_vectors_red --artifacts_dir artifacts_cicids \
    --use_summary_stats --feat_hash fe201634
```

### Step 3: Reduce Graphs (for KDE configs only, not RED)

```bash
# Reduce for kde_ts
python scripts/reduce_graphs_kde.py CIC_IDS_2017_PER_ATTACK \
    --artifacts_dir artifacts_cicids --kde_vectors_dir kde_vectors \
    --output_suffix _kde_ts_reduced --all_hashes --symlink_stages

# Reduce for kde_diff
python scripts/reduce_graphs_kde.py CIC_IDS_2017_PER_ATTACK \
    --artifacts_dir artifacts_cicids --kde_vectors_dir kde_vectors_diff \
    --output_suffix _kde_ts_diff_reduced --all_hashes --symlink_stages
```

### Step 4: Train KDE / RED Variants

```bash
# Kairos KDE-ts (uses reduced artifacts)
python -m pidsmaker.main kairos_cicids_kde_ts CIC_IDS_2017_PER_ATTACK \
    --training.encoder.dropout=0.3 --training.lr=0.0001 \
    --training.node_hid_dim=256 --training.node_out_dim=256 \
    --training.num_epochs=12 --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --artifact_dir ./artifacts_cicids_kde_ts_reduced/ --database_host localhost \
    --force_restart batching --wandb

# Kairos KDE-diff (uses diff-reduced artifacts)
python -m pidsmaker.main kairos_cicids_kde_diff CIC_IDS_2017_PER_ATTACK \
    --training.encoder.dropout=0.3 --training.lr=0.0001 \
    --training.node_hid_dim=256 --training.node_out_dim=256 \
    --training.num_epochs=12 --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --artifact_dir ./artifacts_cicids_kde_ts_diff_reduced/ --database_host localhost \
    --force_restart batching --wandb

# Kairos RED (uses original artifacts — no graph reduction needed)
python -m pidsmaker.main kairos_cicids_red CIC_IDS_2017_PER_ATTACK \
    --training.encoder.dropout=0.3 --training.lr=0.0001 \
    --training.node_hid_dim=256 --training.node_out_dim=256 \
    --training.num_epochs=12 --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --artifact_dir ./artifacts_cicids/ --database_host localhost \
    --force_restart batching --wandb
```

> **Note on `--force_restart`:** Use `evaluation` to re-run only eval (reuses trained model). Use `batching` when the graph artifacts changed (reduced graphs, new KDE vectors). Never use `--restart_from_scratch` for KDE configs — it would wipe construction/featurization that the reduced artifacts depend on.

---

## Appendix: Directory Layout

```
PIDSMaker/
├── config/
│   ├── kairos_cicids.yml                # Baseline Kairos
│   ├── kairos_cicids_kde_ts.yml         # KDE raw timestamps
│   ├── kairos_cicids_kde_diff.yml       # KDE timestamp diffs
│   ├── kairos_cicids_red.yml            # Summary stats
│   ├── orthrus_cicids.yml               # Baseline Orthrus
│   ├── orthrus_cicids_kde_ts.yml
│   ├── orthrus_cicids_kde_diff.yml
│   └── orthrus_cicids_red.yml
├── Ground_Truth/CIC-IDS-2017/
│   ├── node_ftp_patator_tue.csv         # 16 per-attack ground truth files
│   ├── node_ssh_patator_tue.csv
│   ├── node_dos_slowloris_wed.csv
│   └── ...
├── kde_vectors/                          # Precomputed KDE (raw ts) vectors
│   └── CIC_IDS_2017_PER_ATTACK_kde_vectors.pt
├── kde_vectors_diff/                     # Precomputed KDE (diffs) vectors
├── kde_vectors_red/                      # Summary stat vectors (Kairos)
├── kde_vectors_red_orthrus/              # Summary stat vectors (Orthrus)
├── artifacts_cicids/                     # Base artifacts from baseline runs
│   ├── construction/
│   ├── transformation/
│   ├── featurization/
│   └── feat_inference/
├── artifacts_cicids_kde_ts_reduced/      # Reduced artifacts for KDE-ts
│   ├── construction/ → symlink
│   ├── transformation/ → symlink
│   ├── featurization/ → symlink
│   └── feat_inference/ (reduced)
├── artifacts_cicids_kde_ts_diff_reduced/ # Reduced artifacts for KDE-diff
├── kde_computation.py                    # Offline KDE computation
├── scripts/reduce_graphs_kde.py          # Graph reduction
└── dataset_preprocessing/cic_ids_2017/
    ├── create_database_cic_ids_2017.py   # DB ingestion
    └── generate_ground_truth.py          # Ground truth generation
```
