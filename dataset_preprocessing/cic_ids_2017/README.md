# CIC-IDS-2017 Integration for PIDSMaker

## Overview

This integrates the CIC-IDS-2017 network intrusion detection dataset into PIDSMaker.
Unlike DARPA TC (system-level provenance with processes, files, and netflows),
CIC-IDS-2017 is **network-level flow data** — all nodes are IP addresses, and
edges are TCP/UDP/Other packet flows.

## Prerequisites

- PostgreSQL running (via Docker or directly)
- CIC-IDS-2017 CSV files at: `/path/to/CIC-IDS-2017/`
  - `Monday-WorkingHours.csv`
  - `Tuesday-WorkingHours.csv`
  - `Wednesday-WorkingHours.csv`
  - `Thursday-WorkingHours.csv`
  - `Friday-WorkingHours.csv`

## Step-by-Step Instructions

### Step 1: Create and Populate the Database

```bash
cd gnns-latest/dataset_preprocessing/cic_ids_2017/

python3 create_database_cic_ids_2017.py \
    --input ~/provenance_graph_construction/CIC-IDS-2017 \
    --db-name cic_ids_2017 \
    --host localhost --port 5432 \
    --user postgres --password postgres \
    --create-db
```

This will:
1. Create the `cic_ids_2017` database
2. Create PIDSMaker-compatible tables (event_table, netflow_node_table, etc.)
3. Scan all CSVs for unique IPs → insert into `netflow_node_table`
4. Insert all ~46M events into `event_table` (timestamps converted to nanoseconds)

**Note:** The subject_node_table and file_node_table will be empty (this is expected).

### Step 2: Generate Ground Truth Files

```bash
python3 generate_ground_truth.py \
    --input ~/provenance_graph_construction/CIC-IDS-2017
```

This creates CSV files in `Ground_Truth/CIC-IDS-2017/` with malicious node IDs.
Both `orthrus_cicids` and `kairos_cicids` pipelines use `ground_truth_version: none`,
so they resolve `_ground_truth_dir` directly to `Ground_Truth/` — no subdirectory,
no duplication between pipelines.

**Ground truth policy (attacker-side only):**
- `172.16.0.1` (firewall/NAT) = external attacker Kali after NAT
- `192.168.10.8` (Win Vista) = becomes attacker during Thursday Infiltration step 2
- Victim nodes are NOT marked malicious (unless they pass on the attack)

### Step 3: Run PIDSMaker

**Orthrus-style (Word2Vec featurization — based on `orthrus_non_snooped_edge_ts`):**
```bash
python pidsmaker/main.py orthrus_cicids CIC_IDS_2017 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.001 \
    --training.node_hid_dim=256 \
    --training.node_out_dim=256 \
    --training.num_epochs=12 \
    --featurization.emb_dim=256
```

**Kairos-style (Hierarchical Feature Hashing — based on `kairos`):**
```bash
python pidsmaker/main.py kairos_cicids CIC_IDS_2017 \
    --training.encoder.dropout=0.3 \
    --training.lr=0.001 \
    --training.node_hid_dim=256 \
    --training.node_out_dim=256 \
    --training.num_epochs=12 \
    --featurization.emb_dim=256
```

## Key Differences from DARPA TC Datasets

| Aspect | DARPA TC (CADETS_E3) | CIC-IDS-2017 |
|---|---|---|
| Node types | 3 (subject, file, netflow) | 1 (netflow/IP only) |
| Edge types | 10 syscalls | 3 (TCP, UDP, Other) |
| Node labels | paths, cmdlines, IPs | IP addresses only |
| Timestamps | nanoseconds | seconds (converted to ns at ingestion) |
| Attacker visibility | Malicious processes/files on host | Attacker IP behind NAT (172.16.0.1) |

## Dataset Schedule

| Day | Date | Content |
|---|---|---|
| Monday (graph_3) | Jul 3 | **Benign** → training |
| Tuesday (graph_4) | Jul 4 | FTP-Patator, SSH-Patator → val+test |
| Wednesday (graph_5) | Jul 5 | DoS (Slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed → test |
| Thursday (graph_6) | Jul 6 | Web attacks (BF, XSS, SQLi), Infiltration → test |
| Friday (graph_7) | Jul 7 | Botnet, PortScan, DDoS LOIT → test |

## Files Created/Modified

### New files:
- `dataset_preprocessing/cic_ids_2017/create_database_cic_ids_2017.py` — DB ingestion
- `dataset_preprocessing/cic_ids_2017/generate_ground_truth.py` — Ground truth generation
- `config/orthrus_cicids.yml` — Orthrus-style config (inherits `orthrus_non_snooped_edge_ts`)
- `config/kairos_cicids.yml` — Kairos-style config (inherits `kairos`)

### Modified files:
- `pidsmaker/config/config.py` — Added CIC_IDS_2017 dataset config
- `pidsmaker/utils/dataset_utils.py` — Added rel2id_cic_ids, ntype2id_cic_ids, CIC_IDS_DATASETS
- `pidsmaker/config/pipeline.py` — Made get_darpa_tc_node_feats_from_cfg handle missing node types
- `pidsmaker/tasks/feat_inference.py` — Pass cfg to get_node_map()
- `pidsmaker/factory.py` — Pass cfg to get_node_map() (2 locations)
- `pidsmaker/utils/data_utils.py` — Pass cfg to get_node_map()
- `pidsmaker/preprocessing/build_graph_methods/build_magic_graphs.py` — Pass cfg to get_node_map()
- `postgres/init-create-databases.sh` — Added cic_ids_2017 to database list
