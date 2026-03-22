# KDE-Enhanced Pipeline Commands

This document describes the full pipeline for running KDE-enhanced temporal graph processing.

## Overview

The KDE (Kernel Density Estimation) pipeline consists of three stages:
1. **KDE Computation**: Compute RKHS vectors for edges with sufficient temporal observations
2. **Graph Reduction**: Reduce graphs by collapsing KDE-eligible edges
3. **Training**: Train models using either normal or KDE-reduced artifacts

## Prerequisites

```bash
# Activate the environment
cd /scratch/asawan15/PIDSMaker/scripts/apptainer
source activate pids
make up
cd ../../
```

---

## Stage 1: KDE Computation

Compute KDE vectors for a dataset. This extracts edges with `>= min_occurrences` timestamps and computes RKHS vectors.

**Edge keys are 3-tuples: `(src, dst, edge_type)`** to differentiate edges by type.

### Technical Note: Edge Type Extraction

The raw `.TemporalData.simple` files store edge types embedded within the `msg` tensor, not as a separate attribute. The msg structure is:

```
msg = [src_type | src_emb | edge_type | dst_type | dst_emb]
```

For DARPA E3 datasets:
- **CADETS/CLEARSCOPE**: `msg_dim=272`, `emb_dim=120`, edge_type at `[128:144]`
- **THEIA**: `msg_dim=528`, `emb_dim=248`, edge_type at `[256:272]`

The `kde_computation.py` and `reduce_graphs_kde.py` scripts extract edge types using:
```python
emb_dim = (msg_dim - 2 * node_type_dim - edge_type_dim) // 2
edge_type_start = node_type_dim + emb_dim
edge_types = msg[:, edge_type_start:edge_type_start + edge_type_dim].argmax(dim=1)
```

### Commands

```bash
# For CLEARSCOPE_E3
python kde_computation.py orthrus_edge_kde_ts CLEARSCOPE_E3 --output_dir kde_vectors

# For CADETS_E3
python kde_computation.py orthrus_edge_kde_ts CADETS_E3 --output_dir kde_vectors

# For THEIA_E3
python kde_computation.py orthrus_edge_kde_ts THEIA_E3 --output_dir kde_vectors
```

### Output
- `kde_vectors/{DATASET}_kde_vectors.pt` - RKHS vectors for KDE-eligible edges
- `kde_vectors/{DATASET}_kde_stats.json` - Statistics and metadata including `edge_occurrence_counts`

### Parameters (from config)
- `min_occurrences: 10` - Minimum timestamps for KDE eligibility
- `rkhs_dim: 20` - RKHS vector dimension
- `bandwidth: 'scott'` - KDE bandwidth method

---

## Stage 2: Graph Reduction

Reduce graphs by collapsing KDE-eligible edges to single representatives.

### Commands

```bash
# For CLEARSCOPE_E3
python scripts/reduce_graphs_kde.py CLEARSCOPE_E3 --kde_vectors_dir kde_vectors

# For CADETS_E3
python scripts/reduce_graphs_kde.py CADETS_E3 --kde_vectors_dir kde_vectors

# For THEIA_E3
python scripts/reduce_graphs_kde.py THEIA_E3 --kde_vectors_dir kde_vectors
```

### Output
- Creates reduced artifacts in `artifacts_reduced/feat_inference/{DATASET}/...`
- Each KDE-eligible edge is collapsed to its first occurrence
- Non-KDE edges are preserved as-is

---

## Stage 3: Training

### Normal Config (Baseline - uses `artifacts/`)

```bash
# CLEARSCOPE_E3 - Normal
python -m pidsmaker.main orthrus_non_snooped_edge_ts CLEARSCOPE_E3 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.0001 \
    --training.node_hid_dim=64 \
    --training.node_out_dim=64 \
    --training.num_epochs=12 \
    --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --force_restart=batching \
    --artifact_dir ./artifacts/ \
    --database_host localhost \
    --wandb

# CADETS_E3 - Normal
python -m pidsmaker.main orthrus_non_snooped_edge_ts CADETS_E3 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.001 \
    --training.node_hid_dim=256 \
    --training.node_out_dim=256 \
    --training.num_epochs=12 \
    --featurization.emb_dim=256 \
    --force_restart=batching \
    --artifact_dir ./artifacts/ \
    --database_host localhost \
    --wandb

# THEIA_E3 - Normal
python -m pidsmaker.main orthrus_non_snooped_edge_ts THEIA_E3 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.001 \
    --training.node_hid_dim=128 \
    --training.node_out_dim=128 \
    --training.num_epochs=12 \
    --featurization.emb_dim=256 \
    --construction.time_window_size=5 \
    --force_restart=batching \
    --artifact_dir ./artifacts/ \
    --database_host localhost \
    --wandb
```

### KDE Config (uses `artifacts_reduced/`)

```bash
# CLEARSCOPE_E3 - KDE Enhanced
python -m pidsmaker.main orthrus_edge_kde_ts CLEARSCOPE_E3 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.0001 \
    --training.node_hid_dim=64 \
    --training.node_out_dim=64 \
    --training.num_epochs=12 \
    --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --force_restart=batching \
    --artifact_dir ./artifacts_reduced/ \
    --database_host localhost \
    --wandb

# CADETS_E3 - KDE Enhanced
python -m pidsmaker.main orthrus_edge_kde_ts CADETS_E3 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.001 \
    --training.node_hid_dim=256 \
    --training.node_out_dim=256 \
    --training.num_epochs=12 \
    --featurization.emb_dim=256 \
    --force_restart=batching \
    --artifact_dir ./artifacts_reduced/ \
    --database_host localhost \
    --wandb

# THEIA_E3 - KDE Enhanced
python -m pidsmaker.main orthrus_edge_kde_ts THEIA_E3 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.001 \
    --training.node_hid_dim=128 \
    --training.node_out_dim=128 \
    --training.num_epochs=12 \
    --featurization.emb_dim=256 \
    --construction.time_window_size=5 \
    --force_restart=batching \
    --artifact_dir ./artifacts_reduced/ \
    --database_host localhost \
    --wandb
```

---

## Full Pipeline Script (SLURM)

Create a job script `main_job.sh`:

```bash
#!/bin/bash 
#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 40
#SBATCH --mem=400G
#SBATCH --gres=gpu:a100:1
#SBATCH -t 1-00:00:00
#SBATCH -q public
#SBATCH -A grp_moozmen
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asawan15@asu.edu
#SBATCH --export=NONE

# Load environment
module load mamba
cd /scratch/asawan15/PIDSMaker/scripts/apptainer
source activate pids
make up
cd ../../

DATASET="CLEARSCOPE_E3"  # Change as needed: CLEARSCOPE_E3, CADETS_E3, THEIA_E3

echo "=========================================="
echo "Stage 1: Computing KDE vectors for $DATASET"
echo "=========================================="
python kde_computation.py orthrus_edge_kde_ts $DATASET --output_dir kde_vectors

echo "=========================================="
echo "Stage 2: Reducing graphs for $DATASET"
echo "=========================================="
python scripts/reduce_graphs_kde.py $DATASET --kde_vectors_dir kde_vectors

echo "=========================================="
echo "Stage 3a: Training NORMAL config for $DATASET"
echo "=========================================="
python -m pidsmaker.main orthrus_non_snooped_edge_ts $DATASET \
    --training.encoder.dropout=0.3 \
    --training.lr=0.0001 \
    --training.node_hid_dim=64 \
    --training.node_out_dim=64 \
    --training.num_epochs=12 \
    --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --force_restart=batching \
    --artifact_dir ./artifacts/ \
    --database_host localhost \
    --wandb

echo "=========================================="
echo "Stage 3b: Training KDE config for $DATASET"
echo "=========================================="
python -m pidsmaker.main orthrus_edge_kde_ts $DATASET \
    --training.encoder.dropout=0.3 \
    --training.lr=0.0001 \
    --training.node_hid_dim=64 \
    --training.node_out_dim=64 \
    --training.num_epochs=12 \
    --featurization.emb_dim=128 \
    --construction.time_window_size=1 \
    --force_restart=batching \
    --artifact_dir ./artifacts_reduced/ \
    --database_host localhost \
    --wandb

echo "=========================================="
echo "Pipeline complete for $DATASET!"
echo "=========================================="
```

Submit with:
```bash
sbatch main_job.sh
```

---

## Batch Timing Analysis

Both configs will log batch timing information. Results are saved to `timing_results/` directory.

Key metrics logged per batch:
- `total_edges` - Total edges in batch
- `kde_eligible_edges` - Edges with >= 10 timestamps
- `edges_that_could_be_reduced` - KDE-eligible edges with count > 1
- `total_timestamps` - Total timestamps (= total edges)
- `kde_eligible_timestamps` - Timestamps in KDE-eligible edges
- `forward_time_ms` - Forward pass timing

---

## Edge Key Format

All edge identification now uses 3-tuples: `(src, dst, edge_type)`

This ensures edges are differentiated by:
- Source node ID
- Destination node ID  
- Edge type (from one-hot encoded `edge_type` attribute)

The `edge_occurrence_counts` are stored in the KDE vectors metadata for accurate tracking.
