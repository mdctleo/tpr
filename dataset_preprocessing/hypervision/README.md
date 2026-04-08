# HyperVision Dataset Preprocessing for PIDSMaker

This directory contains scripts to ingest the **full** HyperVision dataset
(43 attacks across 7 subclasses) into PostgreSQL for use with PIDSMaker's
temporal graph pipeline.

## Dataset Overview

The HyperVision dataset consists of packet-level network traffic captures.
Each of the 43 TSV files contains ~12.8 M benign background packets mixed with
a specific attack type.  All 43 files share the same benign background; only the
attack traffic differs.

### 7 Subclasses (43 attacks total)

| Subclass | # Attacks | Example Files |
|----------|-----------|---------------|
| crossfire | 2 | crossfiresm.tsv, crossfirela.tsv |
| brute_password | 6 | sshpwdsm.tsv, sshpwdla.tsv, ftppwdsm.tsv, … |
| brute_scanning | 7 | dnsscan.tsv, httpscan.tsv, sqlscan.tsv, … |
| ddos | 5 | charrdos.tsv, udpflood.tsv, synflood.tsv, … |
| port_scanning | 7 | sqlscan.tsv, sshscan.tsv, icmpscan.tsv, … |
| probing_scan | 7 | ssh_lrscan.tsv, dns_lrscan.tsv, … |
| sdos | 9 | icmpsdos.tsv, tcpsdos.tsv, udpsdos.tsv, … |

### Data Format

TSV with columns: `src_ip`, `src_port`, `dst_ip`, `dst_port`, `edge_type`,
`timestamp_us`, `pkt_length`, `label`

- **Timestamps**: Microseconds (relative, starting near 0)
- **Edge types** (9): ICMP, TCP_ACK, TCP_ACK+FIN, TCP_ACK+RST, TCP_RST, TCP_SYN, TCP_SYN+ACK, UDP, UNKNOWN
- **Labels**: 0 = benign, 1 = attack

## Day Mapping (44 synthetic days)

Timestamps are artificially offset to day boundaries so PIDSMaker's day-based
pipeline processes each attack independently.  Base epoch:
`2024-01-01 00:00:00 US/Eastern` (1 704 085 200 000 000 000 ns).

| Days | Content | Split | Details |
|------|---------|-------|---------|
| 1 | Deduplicated benign from all 43 files | Train | MD5-hashed dedup on (src_ip, dst_ip, edge_type, ts, pkt_len) |
| 2–8 | 1 attack per subclass (best representative) + benign | Validation | crossfiresm, sshpwdla, ackport, charrdos, sqlscan, ssh_lrscan, icmpsdos |
| 9–44 | Remaining 36 attacks, one per day | Test | Each day = full benign + single attack |

### Validation representatives (picked by max attack windows, alphabetical tiebreak)

| Day | Subclass | Attack |
|-----|----------|--------|
| 2 | crossfire | crossfiresm |
| 3 | brute_password | sshpwdla |
| 4 | brute_scanning | ackport |
| 5 | ddos | charrdos |
| 6 | port_scanning | sqlscan |
| 7 | probing_scan | ssh_lrscan |
| 8 | sdos | icmpsdos |

## Usage

### Step 1: Generate Ground Truth

```bash
python3 generate_ground_truth_hypervision.py \
    --input /path/to/hypervision_dataset/
```

This produces 43 CSV files under `Ground_Truth/HyperVision/` (e.g.
`node_crossfiresm.csv`, `node_sshpwdla.csv`, …).

### Step 2: Create Database

```bash
python3 create_database_hypervision.py \
    --input /path/to/hypervision_dataset/ \
    --db-name hypervision \
    --host localhost --port 5432 \
    --user postgres --password postgres \
    --create-db
```

This ingests all 44 days (Day 1 benign-only, Days 2–44 each with
benign + attack) into a PostgreSQL `event_table`.

### Step 3: Run Pipeline

```bash
# Kairos — per-attack evaluation
python -m pidsmaker.main --model kairos_hypervision --dataset HYPERVISION_PER_ATTACK

# Orthrus — per-attack evaluation
python -m pidsmaker.main --model orthrus_hypervision --dataset HYPERVISION_PER_ATTACK

# Without per-attack isolation (aggregate test)
python -m pidsmaker.main --model kairos_hypervision --dataset HYPERVISION
```

## PIDSMaker Configuration

- **Dataset names**: `HYPERVISION`, `HYPERVISION_PER_ATTACK`
- **Config files**: `config/kairos_hypervision.yml`, `config/orthrus_hypervision.yml`
- **Database**: `hypervision`
- **year_month**: `2024-01`
- **Day range**: 1–44 (config value `start_end_day_range = (1, 45)`)
- **Node types**: 1 (`netflow` — IP addresses)
- **Edge types**: 9 (packet types)
- **Ground truth**: `Ground_Truth/HyperVision/node_<attack>.csv` (43 files)

## Files in This Directory

| File | Purpose |
|------|---------|
| `create_database_hypervision.py` | Ingest all 44 days into PostgreSQL |
| `generate_ground_truth_hypervision.py` | Extract attacker/victim IPs from label column |
| `README.md` | This file |

## Technical Notes

- **Day > 31 handling**: `build_magic_graphs.py` uses `datetime + timedelta(days=day-1)`
  via `_day_to_dates()` to correctly produce February dates for days 32–44.
- **Benign deduplication**: Day 1 collects benign rows from all 43 files and
  deduplicates via MD5 hash of `(src_ip, dst_ip, edge_type, timestamp_us, pkt_len)`.
- **Timestamp formula**: `day_base_ns + int(timestamp_us) * 1000` where
  `day_base_ns = datetime_to_ns(2024-01-01 + timedelta(days=day-1))`.
