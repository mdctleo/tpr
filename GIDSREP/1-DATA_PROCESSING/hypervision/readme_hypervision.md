# HyperVision Dataset → EULER / ARGUS / PIKACHU Integration Plan

> **Source**: HyperVision packet-level TSV files (17 eligible attack scenarios)  
> **Target**: Generate input formats for EULER, ARGUS, and PIKACHU  
> **Configs per model**: Baseline, KDE-enhanced, Reduced graph  

---

## Table of Contents

1. [Source Dataset Overview](#1-source-dataset-overview)
2. [Dataset Analysis Findings](#2-dataset-analysis-findings)
3. [Key Differences from CIC-IDS 2017](#3-key-differences-from-cic-ids-2017)
4. [Design Decisions](#4-design-decisions)
5. [Train / Test Split Strategy](#5-train--test-split-strategy)
6. [Step-by-Step Integration Plan](#6-step-by-step-integration-plan)
7. [Output Format Specifications](#7-output-format-specifications)
8. [KDE Feature Generation](#8-kde-feature-generation)
9. [Reduced Graph Generation](#9-reduced-graph-generation)
10. [File Structure](#10-file-structure)

---

## 1. Source Dataset Overview

### Data Format

Each of the 43 TSV files represents one attack scenario injected into the same
shared benign background traffic (~12.8M packets). **Packet-level**, not flow-level.

| Column | Type | Description |
|--------|------|-------------|
| `src_ip` | str | Source IP (anonymized, e.g. `3.0.27.133`) |
| `src_port` | int | Source port (0 for ICMP) |
| `dst_ip` | str | Destination IP |
| `dst_port` | int | Destination port |
| `edge_type` | str | Packet type: `TCP_SYN`, `TCP_ACK`, `UDP`, `ICMP`, etc. (9 types) |
| `timestamp_us` | int | Timestamp in **microseconds** (relative, starts near 0) |
| `pkt_length` | int | Packet length in bytes |
| `label` | int | 0 = benign, 1 = attack |

### Attack Categories (43 files total, 17 eligible)

| Category | Subcategory | Total Files | Eligible (atk > 5s) | Examples |
|----------|-------------|-------------|----------------------|----------|
| **Encrypted Flooding** | Link flooding | 6 | 5 | crossfirela, lrtcpdos02 |
| | Password cracking | 6 | 4 | sshpwdla, telnetpwdla |
| | SSH inject | 3 | 2 | ackport, ipidport |
| **Traditional Brute Force** | Amplification | 7 | 4 | dnsrdos, charrdos, riprdos, ssdprdos |
| | Brute scanning | 7 | 1 | sqlscan |
| | Probing vulnerable apps | 10 | 0 | — |
| | Source spoof | 4 | 1 | synsdos |

### 17 Eligible Files (attacks starting ≥ 5 seconds)

| File | Subcategory | First Attack (s) | Attack Duration (s) | Unique IPs | Attack IPs |
|------|-------------|-------------------|---------------------|-----------|-----------|
| charrdos | amplification | 5.64 | 36.22 | 138,774 | 214 |
| dnsrdos | amplification | 5.95 | 34.82 | 138,774 | 214 |
| ssdprdos | amplification | 6.20 | 33.72 | 139,872 | 1,314 |
| riprdos | amplification | 9.05 | 33.27 | 139,073 | 514 |
| crossfirela | link_flooding | 9.30 | 28.40 | 178,683 | 143,494 |
| synsdos | source_spoof | 9.35 | 32.87 | 204,094 | 166,290 |
| lrtcpdos05 | link_flooding | 14.01 | 37.67 | 42,765 | 80 |
| lrtcpdos10 | link_flooding | 17.79 | 31.91 | 42,758 | 80 |
| ackport | ssh_inject | 17.86 | 32.39 | 42,762 | 23 |
| lrtcpdos02 | link_flooding | 18.76 | 35.61 | 42,731 | 50 |
| sqlscan | brute_scanning | 23.39 | 34.09 | 120,385 | 77,693 |
| ipidport | ssh_inject | 25.45 | 33.59 | 138,584 | 100,763 |
| crossfiresm | link_flooding | 29.07 | 16.08 | 178,581 | 326 |
| sshpwdla | password_cracking | 34.34 | 7.12 | 118,624 | 201 |
| telnetpwdmd | password_cracking | 35.14 | 11.99 | 118,470 | 51 |
| telnetpwdla | password_cracking | 35.58 | 12.19 | 118,520 | 101 |
| sshpwdsm | password_cracking | 56.25 | 7.86 | 118,482 | 51 |

---

## 2. Dataset Analysis Findings

### 2.1 Per-File Statistics (All 43 Files)

| Metric | Range |
|--------|-------|
| Packets per file | 12.8M – 15.4M |
| Attack packets per file | 10K – 2.4M (0.08% – 15.4%) |
| Unique IPs per file | 42K – 398K |
| Attack IPs per file | 23 – 352,811 |
| Benign IPs per file | 34K – 178K |
| Timestamp range | ~27s – 128s per file |
| Attack duration | 7s – 122s per file |

### 2.2 IP Disjointness Across Attacks

**Attack IPs are almost entirely disjoint across attack files:**

- **0 IPs** are common to ALL 43 files' attack traffic (src or dst).
- Union of all attack SRC IPs across 43 files: **70,725**
- Union of all attack DST IPs across 43 files: **1,287,301**
- Union of all attack IPs (either role): **1,302,366**
- Within subcategories, attack SRC IP sharing is moderate (e.g., amplification_attack:
  164 common src IPs out of 1,304 union) but across subcategories it's near-zero.

### 2.3 Benign vs Attack IP Overlap (Within Each File)

For each file, the vast majority of attack IPs (~90%+) **never appear in benign
traffic**. Typical examples:

| File | Attack IPs | Also in Benign | Attack-Only |
|------|-----------|---------------|-------------|
| crossfirela | 143,494 | 8,679 (6%) | 134,815 (94%) |
| sshscan | 352,811 | 9,500 (3%) | 343,311 (97%) |
| dnsrdos | 214 | 7 (3%) | 207 (97%) |
| lrtcpdos02 | 50 | 49 (98%) | 1 (2%) |

**Exception**: Small targeted attacks (lrtcpdos*) use IPs that are predominantly
also benign, since they target existing infrastructure.

### 2.4 Temporal Distribution: t<5s vs t>5s

**26 out of 43 files** have attacks starting **before** t=5 seconds. These are
**excluded** from our eligible set to ensure a clean benign training prefix.

For the t<5s vs t>5s IP analysis:

- **Benign IPs**: ~3K–30K IPs appear before 5s, ~90K–170K appear only after 5s.
  Only ~5K–10K are shared across both windows.
- **Attack IPs**: Almost completely disjoint between t<5s and t>5s. Even in files where
  attacks span both windows, overlap is typically **0–15 IPs** out of tens of thousands.

**Key insight**: Attack IPs are ephemeral — different IPs are used at different times,
even within the same attack scenario.

### 2.5 Benign-Only Time Windows

For the 17 eligible files, the benign prefix (before first attack) ranges from
**5.64s** (charrdos) to **56.25s** (sshpwdsm). This provides a guaranteed
benign-only training window for each file.

**No common benign prefix across ALL 43 files** (9 files have attacks from t=0).
This is why we restrict to the 17 files with attacks starting ≥ 5s.

---

## 3. Key Differences from CIC-IDS 2017

| Aspect | CIC-IDS 2017 | HyperVision |
|--------|-------------|-------------|
| **Granularity** | Flow-level | **Packet-level** |
| **Flow features** | Duration, FwdPkts, BwdPkts, FwdBytes, BwdBytes | **None** — only `pkt_length` and `edge_type` |
| **Timestamps** | Seconds (epoch), 5-day span | **Microseconds** (relative), ~150 sec per file |
| **Time span** | 5 working days | ~30–130 seconds per scenario |
| **Nodes** | 19,129 unique IPs | 42K–204K per eligible file |
| **Attack structure** | All attacks in one merged file | 17 separate scenario files |
| **Train/test split** | Day 1 (Monday, benign) = train | Benign prefix (before first attack) = train |
| **Background traffic** | Unique per day | Same benign traffic reused across all files |

### Critical Implications

1. **No native flow features** → We **synthesize pseudo-flow features** by aggregating
   packets within each `(src_ip, dst_ip)` group per snapshot: `pkt_count`, `total_bytes`,
   `duration`, forward/backward splits. This enables ARGUS `L_cic_flow` with NNConv.

2. **Large node count per file** (42K–204K) → EULER can handle this (GCN scales with
   sparse adjacency). PIKACHU's identity matrix approach is infeasible beyond ~20K
   nodes — requires either node capping or skipping PIKACHU.

3. **Short time span** (~30–130 sec) → Use much smaller snapshot deltas than CIC-IDS.

4. **17 independent scenarios** → Each processed independently with its own node map,
   temporal train/test split, and model run.

---

## 4. Design Decisions

### ✅ Decision 1: Which attack scenarios to use?

**Answer: The 17 files where attacks start ≥ 5 seconds into the trace.**

These are the only files that provide a clean benign-only temporal prefix for
training without data leakage. The 26 files with earlier attacks are excluded.

The 17 files cover 5 of the 7 subcategories: amplification (4), link_flooding (5),
password_cracking (4), ssh_inject (2), brute_scanning (1), source_spoof (1).
Missing: probing_vulnerable_application (all 10 files have attacks from t≈0).

**❓ NEEDS CONFIRMATION** — Use all 17, or start with a smaller subset?

---

### ✅ Decision 2: How to handle the node count?

**Answer: (B) Per-file node map.**

Each scenario file gets its own `nmap.pkl` containing only IPs present in that file.
Node counts per eligible file range from 42K to 204K.

**❓ Sub-question for PIKACHU**: Even 42K nodes → a 42K × 42K identity matrix (7 GB).
Options:
- **(A)** Cap at top-20K most active nodes (drop packets involving others)
- **(B)** Replace identity features with learnable `nn.Embedding` (code change)
- **(C)** Aggregate IPs to /24 subnets for PIKACHU only
- **(D)** Skip PIKACHU for HyperVision entirely

**❓ NEEDS CONFIRMATION**

---

### ✅ Decision 3: Train/test split strategy?

**Answer: Temporal split at first attack timestamp.**

For each of the 17 eligible files:
- **Training set**: All packets from `t=0` to `t=first_attack_ts` (guaranteed 100% benign)
- **Test set**: All packets from `t=first_attack_ts` onward (mixed benign + attack)

This is directly analogous to CIC-IDS 2017's Monday (benign) → Tue–Fri (mixed) split.
No data leakage: training and test occupy strictly non-overlapping time windows.

Benign training window ranges from 5.64s (charrdos) to 56.25s (sshpwdsm).

---

### ✅ Decision 4: Snapshot delta (time window width)?

**Options:**
- **(A)** delta = 1 second → ~30–130 snapshots per file
- **(B)** delta = 5 seconds → ~6–26 snapshots per file
- **(C)** delta = 0.5 seconds → ~60–260 snapshots
- **(D)** Different deltas per model

**❓ NEEDS CONFIRMATION**

---

### ✅ Decision 5: Packet → Edge aggregation strategy?

**Answer: (C) Synthesize flow features.**

Group packets by `(src_ip, dst_ip)` within each snapshot, compute:
- `pkt_count` — number of packets
- `total_bytes` — sum of `pkt_length`
- `duration_us` — max timestamp − min timestamp in group
- `fwd_pkt_count` / `bwd_pkt_count` — using port heuristic (lower port = server/dst)
- `fwd_bytes` / `bwd_bytes`
- `label` — max label in group (1 if ANY packet is attack)

This enables:
- EULER: 7-col format `(ts, src, dst, duration, total_bytes, pkt_count, label)`
- ARGUS: with NNConv edge attributes `(bytes, duration, pkt_count, ...)`

---

### ✅ Decision 6: PIKACHU node count limitation?

**Options:**
- **(A)** Cap at top-N most active nodes (N ≈ 20K). Drop all packets involving
  other nodes. Preserves PIKACHU code as-is.
- **(B)** Replace identity features with learnable `nn.Embedding` — requires code changes
- **(C)** Aggregate IPs to /24 subnets for PIKACHU only
- **(D)** Skip PIKACHU for HyperVision entirely (focus on EULER + ARGUS)

**❓ NEEDS CONFIRMATION**

---

## 5. Train / Test Split Strategy

### 5.1 Approach: Temporal Split at First Attack

For each of the 17 eligible files, the split is purely temporal:

```
|<---- TRAINING (benign only) ---->|<---- TEST (benign + attack) ---->|
t=0                          first_attack_ts                     t=end
```

- **Training**: `timestamp_us < first_attack_us` — guaranteed 100% benign packets
- **Test**: `timestamp_us >= first_attack_us` — mixed benign + attack packets
- **No overlap** between train and test time windows → **no data leakage**

### 5.2 Per-File Training Windows

| File | Train Window | Test Window | Train Duration |
|------|-------------|------------|----------------|
| charrdos | 0 – 5.64s | 5.64s – 42s | 5.64s |
| dnsrdos | 0 – 5.95s | 5.95s – 41s | 5.95s |
| ssdprdos | 0 – 6.20s | 6.20s – 40s | 6.20s |
| riprdos | 0 – 9.05s | 9.05s – 42s | 9.05s |
| crossfirela | 0 – 9.30s | 9.30s – 38s | 9.30s |
| synsdos | 0 – 9.35s | 9.35s – 42s | 9.35s |
| lrtcpdos05 | 0 – 14.01s | 14.01s – 52s | 14.01s |
| lrtcpdos10 | 0 – 17.79s | 17.79s – 50s | 17.79s |
| ackport | 0 – 17.86s | 17.86s – 50s | 17.86s |
| lrtcpdos02 | 0 – 18.76s | 18.76s – 54s | 18.76s |
| sqlscan | 0 – 23.39s | 23.39s – 57s | 23.39s |
| ipidport | 0 – 25.45s | 25.45s – 59s | 25.45s |
| crossfiresm | 0 – 29.07s | 29.07s – 45s | 29.07s |
| sshpwdla | 0 – 34.34s | 34.34s – 41s | 34.34s |
| telnetpwdmd | 0 – 35.14s | 35.14s – 47s | 35.14s |
| telnetpwdla | 0 – 35.58s | 35.58s – 48s | 35.58s |
| sshpwdsm | 0 – 56.25s | 56.25s – 64s | 56.25s |

### 5.3 How Models Use the Split

| Model | Training | Testing |
|-------|----------|---------|
| **EULER** | GCN+GRU learns benign graph snapshots from [0, first_atk). InnerProductDecoder learns to reconstruct benign edges. | Scores edges in [first_atk, end). Attack edges (unseen topology) get low scores → anomaly. |
| **ARGUS** | Same + NNConv learns normal synthesized flow features. | Attack edges have anomalous flow patterns + anomalous topology. |
| **PIKACHU** | CTDNE learns benign temporal embeddings. GRU-AE learns benign snapshot sequences. Edge prob `w` learns normal connectivity. | Attack snapshots have new nodes/edges → high reconstruction error + low edge probability. |

### 5.4 `time_set.pkl` Definition

For each file:
```python
time_set = [first_attack_timestamp_in_model_units, last_timestamp_in_model_units]
```
- `time_set[0]` = temporal boundary between training and test
- Everything before `time_set[0]` = train, everything from `time_set[0]` onward = test

---

## 6. Step-by-Step Integration Plan

### Step 0: Analyze per-file statistics (DONE)
- Script: `analyze_all_stats.py` → `dataset_stats.json`
- Additional scripts: `analyze_benign_gaps.py`, `analyze_ip_overlap_5s.py`,
  `analyze_attack_ip_in_benign.py`

### Step 1: `process_hypervision.py` — Master processing script

#### Step 1a: Load TSV and parse
```python
df = pd.read_csv(tsv_path, sep='\t')
# Columns: src_ip, src_port, dst_ip, dst_port, edge_type, timestamp_us, pkt_length, label
# Timestamps kept as-is in microseconds
```

#### Step 1b: Build node map (from ALL IPs — benign + attack)
```python
all_ips = set(df['src_ip'].unique()) | set(df['dst_ip'].unique())
nmap = {ip: idx for idx, ip in enumerate(sorted(all_ips))}
```

#### Step 1c: Temporal split
```python
first_attack_us = df[df['label'] == 1]['timestamp_us'].min()
df_train = df[df['timestamp_us'] < first_attack_us]   # benign only
df_test  = df[df['timestamp_us'] >= first_attack_us]   # mixed
```

#### Step 1d: Aggregate packets → edges (per snapshot)
```python
group = df_window.groupby(['src_id', 'dst_id']).agg(
    pkt_count=('pkt_length', 'count'),
    total_bytes=('pkt_length', 'sum'),
    duration=('timestamp_us', lambda x: x.max() - x.min()),
    label=('label', 'max'),
)
```

### Step 2: Generate EULER format
- `euler/` directory with time-sliced `.txt` files + `nmap.pkl`
- Training snapshots: from `df_train`
- Test snapshots: from `df_test`
- `time_set.pkl` for split boundary

### Step 3: Generate ARGUS format
- `argus_flow/` directory with synthesized flow features + `nmap.pkl` + `time_set.pkl`

### Step 4: Generate PIKACHU format
- `pikachu_hv.csv` with train + test snapshots
- Node-capped version if Option A for Decision 6

---

## 7. Output Format Specifications

### EULER format (`euler/`)

| Field | Description | Source |
|-------|-------------|--------|
| `timestamp_us` | Timestamp in microseconds | As-is from TSV |
| `src_id` | Source node integer ID | `nmap[src_ip]` |
| `dst_id` | Destination node integer ID | `nmap[dst_ip]` |
| `duration` | Microsecond span within aggregation group | `max(ts) - min(ts)` |
| `total_bytes` | Sum of pkt_length in group | Aggregated |
| `pkt_count` | Number of packets in group | Aggregated |
| `label` | 0=benign, 1=attack | `max(label)` |

### ARGUS flow format (`argus_flow/`)

All EULER fields plus:

| Extra Field | Description |
|-------------|-------------|
| `fwd_pkts` | Forward packet count (src_port > dst_port heuristic) |
| `bwd_pkts` | Backward packet count |
| `fwd_bytes` | Forward total bytes |
| `bwd_bytes` | Backward total bytes |

### PIKACHU format (`pikachu_hv.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | int | `timestamp_us` as-is |
| `src_computer` | str | Source IP string |
| `dst_computer` | str | Destination IP string |
| `label` | int | 0 or 1 |
| `snapshot` | int | `timestamp_us // delta_us` |

---

## 8. KDE Feature Generation

### Approach

Same as CIC-IDS 2017 — for each unique `(src, dst)` pair in **training data**,
fit a DPGMM to the inter-arrival time distribution (consecutive `timestamp_us`
differences for that pair), producing a fixed-dimension density vector.

**Consideration**: HyperVision has microsecond resolution, so inter-arrival times
may be very small (μs scale). May need log-transform or normalization.

### Scripts to create
- `compute_kde_euler.py`
- `compute_kde_argus.py`
- `compute_kde_pikachu.py`

---

## 9. Reduced Graph Generation

Same as CIC-IDS 2017:
1. Sort all edges by timestamp
2. `drop_duplicates(subset=['src_id', 'dst_id'], keep='first')`
3. Re-sort by timestamp

Applied to both train and test sets independently.

---

## 10. File Structure (Planned)

```
GIDSREP/1-DATA_PROCESSING/hypervision/
├── readme_hypervision.md              (this file)
├── process_hypervision.py             (master processing script)
├── compute_kde_euler.py
├── compute_kde_argus.py
├── compute_kde_pikachu.py
├── generate_reduced_euler.py
├── generate_reduced_argus.py
├── generate_reduced_pikachu.py
├── dataset_stats.json                 (per-file statistics, from Step 0)
│
├── <scenario_name>/                   (one per eligible attack scenario)
│   ├── euler/                         (EULER format: .txt files + nmap.pkl)
│   ├── euler_red/                     (reduced EULER)
│   ├── argus_flow/                    (ARGUS flow format + nmap.pkl + time_set.pkl)
│   ├── argus_flow_red/                (reduced ARGUS flow)
│   ├── pikachu_hv.csv                 (PIKACHU format)
│   ├── pikachu_hv_red.csv             (reduced PIKACHU)
│   ├── kde_vectors_euler.pkl
│   ├── kde_vectors_argus.pkl
│   ├── kde_vectors_pikachu.pkl
│   └── stats.json
```

---

## 11. Summary of Decisions

| # | Decision | Answer | Status |
|---|----------|--------|--------|
| 1 | Which scenarios? | 17 files with attacks starting ≥ 5s | ❓ All 17 or subset? |
| 2 | Node count | Per-file subset (42K–204K nodes) | ✅ Decided |
| 3 | Train/test split | Temporal split at first_attack_ts (no data leakage) | ✅ Decided |
| 4 | Snapshot delta | TBD | ❓ Needs confirmation |
| 5 | Packet→edge agg | Synthesize flow features | ✅ Decided |
| 6 | PIKACHU nodes | Top-20K / skip / other | ❓ Needs confirmation |

