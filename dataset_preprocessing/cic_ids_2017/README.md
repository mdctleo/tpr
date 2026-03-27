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
    --input /scratch/asawan15/cic-ids \
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
    --input /scratch/asawan15/cic-ids
```

This creates 16 CSV files in `Ground_Truth/CIC-IDS-2017/` marking all malicious nodes.
Both `orthrus_cicids` and `kairos_cicids` pipelines use `ground_truth_version: none`,
so they resolve `_ground_truth_dir` directly to `Ground_Truth/` — no subdirectory,
no duplication between pipelines.

> **This is the only step you need to re-run** if you change the ground truth policy.
> The database (`cic_ids_2017`) is independent of ground truth and does not need to
> be rebuilt.

**Ground truth policy (full attack chain — attackers AND victims):**
- `172.16.0.1` (firewall/NAT) — always marked malicious; all external attacks pass through it
- `205.174.165.73` (Kali) — external attacker, included in all externally-sourced attacks
- `192.168.10.8` (Win Vista) — victim in Infiltration step 1; becomes attacker in step 2
- `205.174.165.69/70/71` (external Win) — DDoS LOIT attackers, NAT'd through 172.16.0.1;
  these IPs are absent from the dataset CSVs (only `172.16.0.1` is written for that attack)
- Victim IPs are marked malicious per official CIC-IDS-2017 documentation; where the CSV
  time window is available, victims were confirmed by querying actual flow data

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

## Dataset Schedule and Ground Truth Counts

| Day | Date | Attacks | Malicious nodes | Benign nodes |
|---|---|---|---|---|
| Monday (graph_3) | Jul 3 | **Benign** (training) | 0 | 16,558 |
| Tuesday (graph_4) | Jul 4 | FTP-Patator, SSH-Patator | **3** | 16,555 |
| Wednesday (graph_5) | Jul 5 | DoS (Slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed | **4** | 16,554 |
| Thursday (graph_6) | Jul 6 | Web attacks (BF, XSS, SQLi), Infiltration (step1, Cool Disk, step2) | **15** | 16,543 |
| Friday (graph_7) | Jul 7 | Botnet ARES, PortScan, DDoS LOIT | **8** | 16,550 |

**15 unique malicious IPs across all days** (total dataset size: 16,558 unique IPs).

### Per-day malicious IP breakdown

**Tuesday:** `172.16.0.1` (attacker), `205.174.165.73` (attacker), `192.168.10.50` (victim — WebServer)

**Wednesday:** `172.16.0.1`, `205.174.165.73`, `192.168.10.50` (victim — WebServer), `192.168.10.51` (victim — Ubuntu12, Heartbleed only)

**Thursday:** `172.16.0.1`, `205.174.165.73`, plus all internal clients targeted by Infiltration step2 portscan:
`192.168.10.3` (DNS), `192.168.10.5` (Win8), `192.168.10.8` (Vista — victim→attacker),
`192.168.10.9` (Win7), `192.168.10.12` (Ubuntu16 64B), `192.168.10.14` (Win10 32B),
`192.168.10.15` (Win10 64B), `192.168.10.16` (Ubuntu16 32B), `192.168.10.17` (Ubuntu14 64B),
`192.168.10.19` (Ubuntu14 32B), `192.168.10.25` (MAC), `192.168.10.50` (WebServer), `192.168.10.51` (Ubuntu12)

**Friday:** `172.16.0.1`, `205.174.165.73`, `192.168.10.5`, `192.168.10.8`, `192.168.10.9`,
`192.168.10.14`, `192.168.10.15` (botnet victims), `192.168.10.50` (WebServer — PortScan/DDoS)

## Network Topology

**Firewall / NAT:** `205.174.165.80` (external), `172.16.0.1` (internal)

**Attacker network (outsiders):**
- Kali: `205.174.165.73`
- Windows (DDoS): `205.174.165.69`, `205.174.165.70`, `205.174.165.71`

**Victim network (insiders):**
| IP | Machine |
|---|---|
| `192.168.10.3` | DNS + DC Server |
| `192.168.10.5` | Win 8.1 64B |
| `192.168.10.8` | Win Vista 64B |
| `192.168.10.9` | Win 7 Pro 64B |
| `192.168.10.12` | Ubuntu 16.4 64B |
| `192.168.10.14` | Win 10 pro 32B |
| `192.168.10.15` | Win 10 64B |
| `192.168.10.16` | Ubuntu 16.4 32B |
| `192.168.10.17` | Ubuntu 14.4 64B |
| `192.168.10.19` | Ubuntu 14.4 32B |
| `192.168.10.25` | MAC |
| `192.168.10.50` | Web Server 16 (public: `205.174.165.68`) |
| `192.168.10.51` | Ubuntu Server 12 (public: `205.174.165.66`) |

**NAT path for most attacks:**
```
205.174.165.73 → 205.174.165.80 (firewall) → 172.16.0.1 → <victim>
```
The Kali IP (`205.174.165.73`) appears directly in the dataset for Infiltration step 1
and Botnet ARES (where NAT is bypassed). For all other attacks it is hidden behind
`172.16.0.1` in the flow data, but both are marked malicious in ground truth.

## CSV Coverage Notes

The CIC-IDS-2017 CSVs used here are preprocessed into `(src_ip, src_type, dst_ip,
dst_type, protocol, unix_timestamp)` format. Their actual time coverage in EDT is:

| File | Coverage (EDT) | Attacks outside window |
|---|---|---|
| `Tuesday-WorkingHours.csv` | 07:53 – 13:32 | SSH-Patator (14:00–15:00) |
| `Wednesday-WorkingHours.csv` | 07:42 – **10:18** | DoS Hulk, GoldenEye, Heartbleed |
| `Thursday-WorkingHours.csv` | 07:59 – 15:21 | *(all windows covered)* |
| `Friday-WorkingHours.csv` | 07:59 – 15:01 | DDoS LOIT (15:56–16:16) |

For attacks outside the available window, victim IPs are taken from the official
CIC-IDS-2017 dataset documentation rather than from live CSV flow analysis.

## Files Created/Modified

### New files:
- `dataset_preprocessing/cic_ids_2017/create_database_cic_ids_2017.py` — DB ingestion
- `dataset_preprocessing/cic_ids_2017/generate_ground_truth.py` — Ground truth generation
- `config/orthrus_cicids.yml` — Orthrus-style config (inherits `orthrus_non_snooped_edge_ts`)
- `config/kairos_cicids.yml` — Kairos-style config (inherits `kairos`)

### Modified files:
- `pidsmaker/config/config.py` — Added CIC_IDS_2017 dataset config (16 attacks incl. Cool Disk)
- `pidsmaker/utils/dataset_utils.py` — Added rel2id_cic_ids, ntype2id_cic_ids, CIC_IDS_DATASETS
- `pidsmaker/config/pipeline.py` — Made get_darpa_tc_node_feats_from_cfg handle missing node types
- `pidsmaker/tasks/feat_inference.py` — Pass cfg to get_node_map()
- `pidsmaker/factory.py` — Pass cfg to get_node_map() (2 locations)
- `pidsmaker/utils/data_utils.py` — Pass cfg to get_node_map()
- `pidsmaker/preprocessing/build_graph_methods/build_magic_graphs.py` — Pass cfg to get_node_map()
- `pidsmaker/detection/evaluation_methods/evaluation_utils.py` — Added color entry for 16th attack
- `postgres/init-create-databases.sh` — Added cic_ids_2017 to database list
