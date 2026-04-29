# HyperVision
![Licence](https://img.shields.io/github/license/fuchuanpu/HyperVision)
![Last](https://img.shields.io/github/last-commit/fuchuanpu/HyperVision)
![Language](https://img.shields.io/github/languages/count/fuchuanpu/HyperVision)

A demo of the flow interaction graph based attack traffic detection system, i.e., HyperVision:

___Detecting Unknown Encrypted Malicious Traffic in Real Time via Flow Interaction Graph Analysis___  
In the $30^{th}$ Network and Distributed System Security Symposium ([NDSS'23](https://www.ndss-symposium.org/ndss-paper/detecting-unknown-encrypted-malicious-traffic-in-real-time-via-flow-interaction-graph-analysis/)).  
[Chuanpu Fu](https://www.fuchuanpu.cn), [Qi Li](https://sites.google.com/site/qili2012), and [Ke Xu](http://www.thucsnet.org/xuke.html).  


> The new CDN for the dataset has been successfully established. Please feel free to explore and utilize it! 🍺


## __0x00__ Hardware
- AWS EC2 c4.4xlarge, 100GB SSD, canonical `Ubuntu` 22.04 LTS (amd64, 3/3/2023).
- Tencent Cloud CVM, _with similar OS and hardware configurations_.

## __0x01__ Software
The demo can be built from a clean `Ubuntu` env.

```bash
# Establish env.
git clone https://github.com/fuchuanpu/HyperVision.git
cd HyperVision
sudo ./env/install_all.sh

# Download dataset.
wget https://www.hypervision.fuchuanpu.xyz/hypervision-dataset.tar.gz
tar -xxf hypervision-dataset.tar.gz
rm $_

# Build and run HyperVision.
./script/rebuild.sh
./script/expand.sh
cd build && ../script/run_all_brute.sh && cd ..

# Analyze the results.
cd ./result_analyze
./batch_analyzer.py -g brute
cat ./log/brute/*.log | grep AU_ROC
cd -
```

## __0x02__ Detailed Workflow & Architecture

### Pipeline Overview

```
Raw PCAP ──► Packet Parsing ──► Flow Construction ──► Edge Construction ──► Graph Construction
                                                                                  │
              Per-Packet Scores ◄── Score Propagation ◄── Clustering & Scoring ◄──┘
```

The system builds a **flow interaction graph** from raw network traffic, then applies
unsupervised graph-structure and feature-based anomaly detection to identify malicious
traffic — all without relying on payload decryption or prior attack signatures.

---

### Stage 1 — Packet Parsing (`packet_parse/`)

The `pcap_parser` class reads raw PCAP files via PcapPlusPlus and extracts per-packet
features into lightweight structs.

| Per-Packet Feature | Type | Description |
|---|---|---|
| `time_stamp` | `double` | Epoch timestamp in seconds |
| `type_code` | `uint16_t` bitmask | Protocol flags: `TCP_SYN`, `TCP_ACK`, `TCP_FIN`, `TCP_RST`, `TCP_PSH`, `TCP_URG`, `UDP`, `ICMP`, `OTHER`, etc. Multiple flags can be OR'd together |
| `ip_total_len` | `uint16_t` | IP-layer total length in bytes |
| `src_ip` / `dst_ip` | `uint32_t` (v4) or `uint128_t` (v6) | Source and destination IP addresses |
| `src_port` / `dst_port` | `uint16_t` | Transport-layer ports |

A derived **stack code** collapses TCP subtypes into a single `TCP` code, retaining
`UDP`, `ICMP`, and `OTHER` as distinct values.

---

### Stage 2 — Flow Construction (`flow_construct/`)

Handled by `explicit_constructor`. Flows are identified by **5-tuple**:
`(src_ip, dst_ip, src_port, dst_port, stack_code)`.

**Algorithm:**
1. Packets are partitioned into chunks processed by parallel threads
2. Each thread maintains a hash table keyed by 5-tuple
3. Packets are appended to matching flows
4. Every `eviction_check_interval` (default **5.0 s**), flows idle for longer than
   `flow_timeout` (default **10.0 s**) are evicted
5. A merge pass combines flows from different threads that share the same 5-tuple
   (if inter-arrival < `flow_timeout`)

**Per-flow data stored** (`flow_define.hpp`):

| Member | Description |
|---|---|
| `start_ts` / `end_ts` | First and last packet timestamp |
| `pkt_code` | Bitwise OR of all packet `type_code` values |
| `pkt_list` | Vector of pointers to constituent packets |
| `origin_index` | Maps each packet back to its index in the global packet array (used for final score propagation) |

**Config:** `flow_timeout` (10.0 s), `eviction_check_interval` (5.0 s).

---

### Stage 3 — Edge Construction (`graph_analyze/edge_constructor`)

Flows are partitioned into **long** vs **short** based on a packet-count threshold
`edge_long_line` (default **15 packets**).

#### 3a. Long Edges (> 15 packets)

Each long flow becomes one `long_edge`. **Per-long-edge features** are distribution
summaries computed from the packet sequence:

| Feature | Type | Description |
|---|---|---|
| `len_dist` | Histogram | Packet lengths binned at `length_bin_size` (default 10 bytes) |
| `type_dist` | Histogram | Packet type-code frequencies |
| `interval_dist` | Histogram | Inter-packet intervals binned at 1 ms |

**Derived flags on long edges:**
- `is_elephant`: > 8 000 packets or > 5 MB total
- `is_periodic`: avg rate > 50 pkt/s or < 2 unique lengths
- `is_scan`: > 10 SYN, FIN, or RST packets
- `flow_completion_time`: total_time / total_packets

#### 3b. Short Edges (≤ 15 packets)

Short flows are **aggregated** into `short_edge` objects by shared attributes to form
statistically meaningful groups:

1. Group by `(src_ip, dst_ip, pkt_code)` — aggregate if group size > `edge_agg_line` (default **20**)
2. Remaining → try `(src_ip, pkt_code)`
3. Remaining → try `(dst_ip, pkt_code)`
4. Remaining → no aggregation (singleton)

Each `short_edge` stores:
- `flow_list` — the collection of aggregated flows
- `agg_code` — bitmask indicating which dimensions are shared (`SRC_IP`, `DST_IP`,
  `SRC_PORT`, `DST_PORT`, `PKT_CODE`)

**Config:** `length_bin_size` (10), `edge_long_line` (15), `edge_agg_line` (20).

---

### Stage 4 — Graph Construction & Connected Components

The **flow interaction graph** has:
- **Vertices:** IP addresses
- **Edges:** `long_edge` and `short_edge` objects

Adjacency maps track outbound/inbound edges per vertex, separately for long edges,
non-aggregated short edges, and aggregated short edges.

**Connected components** are found via DFS on the undirected projection. Only the
top `component_select_rate` (default **1%**) most anomalous components proceed to
detailed analysis.

**Component-level features** (6-dimensional):
1. Vertex count
2. Number of long edges
3. Number of short edges
4. Number of aggregated short edges
5. Total bytes in long edges
6. Total bytes in short edges

These are min-max normalised and clustered with **DBSCAN** (`eps`, `min_pts`).
Anomaly score = minimum Euclidean distance to any cluster centroid.

---

### Stage 5 — Per-Component Anomaly Scoring

For each selected component, long and short edges are scored separately via a
**two-phase clustering** approach.

#### Long Edge Scoring

**Phase 1 — DBSCAN clustering (4–5 dims):**
1. Source out-degree
2. Source in-degree
3. Destination out-degree
4. Destination in-degree
5. _(optional)_ packet type code

After clustering, a **Z3-based minimum vertex cover** identifies critical IP
addresses; edges are re-grouped by critical address.

**Phase 2 — KMeans (K=`k_long`, default 10) on refined features (11 dims):**
1–4. Degree features (same as Phase 1)
5. Most frequent packet type code
6. Count of most frequent type
7. Most frequent packet length bin
8. Count of most frequent length
9. Total packet count
10. Flow completion time
11. Average packet rate

**Long edge anomaly score:**

$$S_l = a_l \cdot d_{\text{centroid}} + b_l \cdot \log_2(\text{cluster\_size} + 1) - c_l \cdot \Delta t_{\text{cluster}}$$

Defaults: $a_l = 0.1$, $b_l = 1.0$, $c_l = 0.5$.

#### Short Edge Scoring

**Phase 1 — DBSCAN (8–9 dims):**
1–4. Binary flags for shared `src_ip`, `dst_ip`, `src_port`, `dst_port`
5–8. Degree features
9. _(optional)_ protocol code of first flow

**Phase 2 — KMeans (K=`k_short`) on refined features (12 dims):**
1–8. Same as Phase 1 (without optional proto)
9. Number of aggregated flows
10. Packet count of first flow
11. Aggregation type code
12. Average timestamp of first flow's packets

**Short edge anomaly score:**

$$S_s = a_s \cdot d_{\text{centroid}} + b_s \cdot \log_2(\text{agg\_size} \times \text{cluster\_size} + 1) - c_s \cdot \Delta t_{\text{cluster}}$$

---

### Stage 6 — Score Propagation

1. Every packet starts with score **−1** (benign)
2. For each long edge, all constituent packets receive $S_l$
3. For each short edge, all constituent packets receive $S_s$
4. If a packet appears in multiple edges, it receives the **maximum** score
5. Output: per-packet `(ground_truth_label, anomaly_score)` pairs

---

### Stage 7 — Result Analysis (`result_analyze/`)

`batch_analyzer.py` reads per-packet `(label, score)` files and computes:
- **ROC metrics:** AUC-ROC, TPR@FPR=0.1, FPR@TPR=0.9, EER
- **PRC metrics:** F1, F2, Precision, Recall, AUC-PRC, Accuracy, FP/FN counts
- A `water_line` threshold from `configure.json` for binary classification

---

### Dataset & Labelling (`dataset_construct/`)

Two input modes:
1. **From PCAP:** packets parsed directly; labels assigned by matching attacker IPs
   after a configurable `attack_start_time`
2. **From pre-exported data:** `.data` text file
   (`4/6 srcIP dstIP srcPort dstPort timestamp typeCode length`) + `.label` binary
   string of `0`/`1`

**Config:** `train_ratio` (fraction treated as benign context), `attacker_ips`,
`attack_start_time`.

---

### Configuration Reference

Each JSON config (in `configuration/`) contains five sections:

| Section | Key Parameters (defaults) |
|---|---|
| `dataset_construct` | `data_path`, `label_path`, `train_ratio`, `attacker_ips`, `attack_start_time` |
| `flow_construct` | `flow_timeout` (10.0), `eviction_check_interval` (5.0) |
| `edge_construct` | `length_bin_size` (10), `edge_long_line` (15), `edge_agg_line` (20) |
| `graph_analyze` | `component_select_rate` (0.01), `score_weights_{long,short}` ($a$, $b$, $c$), `dbscan_{component,long,short}` ($\varepsilon$, min_pts), `k_long`, `k_short` (KMeans K) |
| `result_save` | `output_path`, `log_path` |

**External dependencies:** PcapPlusPlus, Boost (hash), mlpack (KMeans, DBSCAN,
MinMaxScaler), Z3 (SMT solver), Armadillo.

---

## __0x03__ KDE Density Feature Integration Plan

### Motivation

HyperVision's long-edge features include a coarse **histogram of inter-packet
intervals** (`interval_dist`). A KDE (kernel density estimation) vector — specifically
a BayesianGaussianMixture (DPGMM) density evaluated on a fixed grid — provides a
**smoother, normalised, fixed-dimensional** representation of the inter-arrival time
distribution. This captures subtle timing patterns (e.g., periodic beaconing,
bursty exfiltration) that fixed-width histograms may miss.

### Where to Integrate

| Aspect | Detail |
|---|---|
| **Target edge type** | **Long edges** (≥ 15 packets → enough inter-arrival samples for reliable density estimation). Short edges could use aggregate-level KDE but benefit is marginal. |
| **Insertion point (C++)** | In `edge_constructor.cpp`, inside `construct_long_edges()` where `len_dist`, `type_dist`, and `interval_dist` are already computed from the packet sequence |
| **New member** | `long_edge::kde_density_vec` — `arma::fvec` of dimension `kde_dim` (e.g., 20 or 64) |
| **Feature pipeline** | Append `kde_density_vec` to the Phase 2 refined feature vector in `graph_analysis_long.cpp` (currently 11 dims → 11 + `kde_dim`) |

### Implementation Steps

#### Step 1 — Add KDE computation to C++ edge construction

```
// In graph_analyze/edge_define.hpp — add to long_edge class:
arma::fvec kde_density_vec;   // Fixed-dim KDE feature (default: 20-d)

// In graph_analyze/edge_constructor.cpp — inside construct_long_edges():
//   After computing interval_dist from the packet list:
//   1. Extract sorted inter-arrival times → diffs[]
//   2. MaxAbsScale to [-1, 1]
//   3. Fit BayesianGaussianMixture (via mlpack's GMM or an embedded Python call)
//   4. Evaluate density on uniform grid of kde_dim points
//   5. L2-normalize → store in kde_density_vec
```

> **Note:** mlpack provides `mlpack::gmm::GMM` which can approximate the
> BayesianGaussianMixture behaviour. Alternatively, pre-compute KDE vectors in
> Python (`compute_kde_features.py`) and load them at runtime via a lookup table
> keyed by flow 5-tuple.

#### Step 2 — Integrate into Phase 2 feature vector

```
// In graph_analyze/graph_analysis_long.cpp — get_refined_feature():
//   Current: 11-dim vector [degrees, type, length, count, time, rate]
//   New:     (11 + kde_dim)-dim vector — append kde_density_vec elements
```

#### Step 3 — Configuration

Add to the `edge_construct` section of JSON configs:

```json
{
  "kde_dim": 10,
  "kde_n_components": 10,
  "kde_concentration_prior": 0.1,
  "kde_max_iter": 200,
  "kde_min_packets": 10
}
```

#### Step 4 — Hybrid approach (recommended for rapid prototyping)

Instead of implementing DPGMM in C++, use a **two-pass hybrid**:

1. **Pass 1 (Python):** Run `compute_kde_features.py` on exported `.data` files to
   produce a pickle mapping `(src_ip, dst_ip) → kde_vec[20]`
2. **Pass 2 (C++):** Load the pickle (or a converted CSV/binary) at startup in
   `edge_constructor`; look up each long edge's `(src_ip, dst_ip)` to populate
   `kde_density_vec`
3. Fall back to zero-vector for unseen edges (same as the Python script's fallback)

This avoids reimplementing DPGMM in C++ while preserving the real-time detection
pipeline for deployment.

#### Step 5 — Validation

- Compare AUC-ROC with and without KDE features across all attack categories
  (brute force, flooding, scanning, etc.)
- Ablation: KDE-only vs histogram-only vs combined
- Dimensionality sweep: `kde_dim` ∈ {10, 20, 32, 64}

### Expected Impact

| Attack Type | Why KDE Helps |
|---|---|
| **Beaconing / C2** | Periodic inter-arrival patterns create sharp KDE peaks detectable as outliers |
| **Slow exfiltration** | Distinctive long-tail density vs normal browsing |
| **Flooding / DDoS** | Near-zero inter-arrival creates a delta-like KDE easily separable |
| **Scanning** | Uniform rapid probing produces flat KDE distinct from normal traffic |

---

## __0x04__ Running the 43 Attacks (Baseline vs KDE-Enhanced)

The 43 attacks span two dataset categories:

| Category | Subcategory | Attacks | Config Dir |
|---|---|---|---|
| **encrypted_flooding_traffic** | link_flooding | crossfire{la,md,sm}, lrtcpdos{02,05,10} | misc |
| | password_cracking | sshpwd{la,md,sm}, telnetpwd{la,md,sm} | misc |
| | ssh_inject | ackport, ipidaddr, ipidport | misc |
| **traditional_brute_force_attack** | amplification_attack | charrdos, cldaprdos, dnsrdos, memcachedrdos, ntprdos, riprdos, ssdprdos | bruteforce |
| | brute_scanning | dnsscan, httpscan, httpsscan, icmpscan, ntpscan, sqlscan, sshscan | bruteforce |
| | probing_vulnerable_application | {dns,http,icmp,netbios,rdp,smtp,snmp,ssh,telnet,vlc}_lrscan | lrscan |
| | source_spoof | icmpsdos, rstsdos, synsdos, udpsdos | bruteforce |

### A. Build HyperVision

```bash
cd HyperVision
./script/rebuild.sh
./script/expand.sh
mkdir -p kde_features temp_kde cache_kde
```

### B. Run Baseline (all 43 attacks, no KDE)

```bash
cd build && ../script/run_all_43_baseline.sh && cd ..

# Analyze results
cd result_analyze
./batch_analyzer.py -g brute
./batch_analyzer.py -g lrscan
./batch_analyzer.py -g misc
cat ./log/brute/*.log ./log/lrscan/*.log ./log/misc/*.log | grep AU_ROC
cd -
```

### C. Precompute KDE Features (Python, one-time)

```bash
pip install scikit-learn numpy   # if not installed

cd script
python compute_kde_features.py \
    --all-attacks \
    --data-dir ../data \
    --output-dir ../kde_features \
    --kde-dim 20 \
    --min-timestamps 10
cd -
```

This produces `kde_features/{attack}_kde.csv` for each of the 43 attacks.

### D. Generate KDE-Enhanced Configs (one-time)

```bash
python script/generate_kde_configs.py
```

Creates `configuration/{bruteforce,lrscan,misc}_kde/{attack}.json` — identical to
baseline configs but with `"kde_features_path": "../kde_features/{attack}_kde.csv"`
in the `graph_analyze` section. Results are saved to `temp_kde/`.

### E. Run KDE-Enhanced (all 43 attacks)

```bash
cd build && ../script/run_all_43_kde.sh && cd ..

# Analyze KDE results (point analyzer at temp_kde)
cd result_analyze
./batch_analyzer.py -g brute --result-dir ../temp_kde
./batch_analyzer.py -g lrscan --result-dir ../temp_kde
./batch_analyzer.py -g misc --result-dir ../temp_kde
cat ./log/brute_kde/*.log ./log/lrscan_kde/*.log ./log/misc_kde/*.log | grep AU_ROC
cd -
```

### F. Compare Baseline vs KDE

```bash
# Side-by-side AUC-ROC comparison
echo "=== BASELINE ===" && cat cache/*.log | grep AU_ROC
echo "=== KDE ===" && cat cache_kde/*.log | grep AU_ROC
```

### G. Reading & Visualising Results

#### Output Format

Each HyperVision run produces a `.txt` file in `temp/` (baseline) or `temp_kde/`
(KDE-enhanced). Each line is one packet:

```
<ground_truth_label>  <anomaly_score>
```

- `ground_truth_label`: `0` = benign, `1` = malicious
- `anomaly_score`: floating point; higher = more anomalous. Benign packets
  that were never assigned to any edge retain their default score (typically
  negative, e.g., `−1.0` or `−0.78`).

Example (`temp/charrdos.txt`):
```
0 -0.7837
0 -0.7837
0 -1
1 5.038
1 5.038
```

#### Running the Built-In Analyzer

The `result_analyze/` directory contains the official evaluation tools:

```bash
cd result_analyze

# Analyze one group at a time (creates log/{group}/ and figure/{group}/)
python batch_analyzer.py -g brute     # 18 bruteforce attacks
python batch_analyzer.py -g lrscan    # 10 lrscan attacks
python batch_analyzer.py -g misc      # 15 misc attacks

# Quick AUC-ROC summary
grep "AU_ROC" log/brute/*.log log/lrscan/*.log log/misc/*.log
```

**What `batch_analyzer.py` does:**
1. Reads `configure.json` which maps group → {attack: water_line}
2. Spawns one process per attack, calling `basic_analyzer.py -t <attack>`
3. Each `basic_analyzer.py` reads `../temp/<attack>.txt` and computes:

| Metric | Description |
|---|---|
| `AU_ROC` | Area under ROC curve (primary metric) |
| `TPR@FPR=0.1` | True positive rate when FPR is closest to 0.1 |
| `FPR@TPR=0.9` | False positive rate when TPR is closest to 0.9 |
| `EER` | Equal error rate (FPR where FPR ≈ FNR) |
| `F1` / `F2` | F-beta scores at the `water_line` threshold |
| `Precision` / `Recall` | At the `water_line` threshold |
| `AU_PRC` | Area under precision-recall curve |
| `Accuracy` | Binary accuracy at `water_line` |
| `FN` / `FP` | False negative / false positive counts |

4. Saves ROC and PRC curve plots to `figure/{group}/{attack}_ROC.png` and
   `figure/{group}/{attack}_PRC.png`
5. Saves full log to `log/{group}/{attack}.log`

**The `water_line` threshold** (default `11.0` for all attacks in `configure.json`)
is the binary classification cutoff: packets with `score > water_line` are predicted
malicious. The ROC/AUC metrics are threshold-independent; F1/Precision/Recall use
this threshold.

#### Summary Script

A convenience script `summarize_results.py` parses all log files and produces
formatted tables:

```bash
cd result_analyze

# Per-group summary + full table
python summarize_results.py --log-dir ./log --groups brute lrscan misc

# Export to CSV
python summarize_results.py --log-dir ./log --groups brute lrscan misc --csv baseline_results.csv

# Compare baseline vs KDE (after running KDE experiments)
python summarize_results.py --log-dir ./log --groups brute lrscan misc \
    --compare-dir ./log_kde --compare-label "KDE-Enhanced"
```

#### Baseline Results (43 attacks)

> spam1/spam50/spam100 are listed in `configure.json` but not in our dataset.

**Per-Group Average AU_ROC:**

| Group | # Attacks | Mean AU_ROC | Min AU_ROC | Hardest Attack |
|---|---:|---:|---:|---|
| brute | 18 | 0.999530 | 0.997246 | httpscan |
| lrscan | 10 | 0.995222 | 0.953616 | smtp_lrscan |
| misc | 15 | 0.986759 | 0.917002 | ipidaddr |
| **Overall** | **43** | **0.993800** | **0.917002** | **ipidaddr** |

**Full Results — Bruteforce (18 attacks):**

| Attack | AU_ROC | TPR@FPR=0.1 | EER | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| charrdos | 0.999928 | 0.999914 | 0.000027 | 0.999978 | 0.999999 | 0.999957 |
| cldaprdos | 0.999975 | 0.999968 | 0.000027 | 0.999992 | 0.999999 | 0.999984 |
| dnsrdos | 0.999979 | 0.999974 | 0.000027 | 0.999993 | 0.999999 | 0.999987 |
| dnsscan | 0.999922 | 1.000000 | 0.000078 | 0.996054 | 0.992208 | 0.999961 |
| httpscan | 0.997246 | 0.997195 | 0.003146 | 0.987300 | 0.976700 | 0.998476 |
| httpsscan | 0.998667 | 0.998429 | 0.000542 | 0.997643 | 0.996120 | 0.999175 |
| icmpscan | 0.998864 | 0.998689 | 0.001312 | 0.970500 | 0.945300 | 0.998689 |
| icmpsdos | 0.999923 | 0.999912 | 0.000026 | 0.999978 | 0.999999 | 0.999956 |
| memcachedrdos | 0.999984 | 0.999980 | 0.000000 | 0.999995 | 0.999999 | 0.999990 |
| ntprdos | 0.999990 | 0.999988 | 0.000000 | 0.999997 | 0.999999 | 0.999994 |
| ntpscan | 0.999922 | 0.999985 | 0.000078 | 0.997067 | 0.994197 | 0.999985 |
| riprdos | 0.999900 | 0.999884 | 0.000000 | 0.999971 | 0.999999 | 0.999942 |
| rstsdos | 0.999955 | 0.999949 | 0.000000 | 0.999987 | 0.999999 | 0.999974 |
| sqlscan | 0.997997 | 0.997689 | 0.003211 | 0.988900 | 0.980200 | 0.997956 |
| ssdprdos | 0.999722 | 0.999718 | 0.000000 | 0.999248 | 0.999993 | 0.998506 |
| sshscan | 0.999555 | 0.999492 | 0.001022 | 0.990394 | 0.982100 | 0.998988 |
| synsdos | 0.999966 | 0.999960 | 0.000000 | 0.999990 | 0.999999 | 0.999980 |
| udpsdos | 0.999932 | 0.999923 | 0.000000 | 0.999980 | 0.999999 | 0.999962 |

**Full Results — LR Scan (10 attacks):**

| Attack | AU_ROC | TPR@FPR=0.1 | EER | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| dns_lrscan | 0.999826 | 0.999956 | 0.000473 | 0.993759 | 0.988300 | 0.999312 |
| http_lrscan | 0.999957 | 0.999958 | 0.000105 | 0.992536 | 0.986600 | 0.998619 |
| icmp_lrscan | 0.999989 | 0.999985 | 0.000000 | 0.970600 | 0.951100 | 0.992667 |
| netbios_lrscan | 0.999831 | 0.999957 | 0.000473 | 0.993961 | 0.988600 | 0.999431 |
| rdp_lrscan | 0.999978 | 0.999980 | 0.000548 | 0.962300 | 0.936300 | 0.992664 |
| smtp_lrscan | 0.953616 | 0.953358 | 0.086880 | 0.982600 | 0.989200 | 0.976100 |
| snmp_lrscan | 0.999691 | 0.999986 | 0.000473 | 0.987600 | 0.976300 | 0.999445 |
| ssh_lrscan | 0.999559 | 0.999455 | 0.000614 | 0.984000 | 0.970500 | 0.998221 |
| telnet_lrscan | 0.999970 | 0.999979 | 0.000467 | 0.960400 | 0.933100 | 0.992663 |
| vlc_lrscan | 0.999799 | 0.999949 | 0.000473 | 0.992875 | 0.986500 | 0.999427 |

**Full Results — Misc (15 attacks):**

| Attack | AU_ROC | TPR@FPR=0.1 | EER | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| ackport | 0.992209 | 0.977026 | 0.002878 | 0.975600 | 0.986100 | 0.965500 |
| crossfirela | 0.999829 | 0.999078 | 0.002728 | 0.989800 | 0.982200 | 0.997718 |
| crossfiremd | 0.986722 | 0.958618 | 0.045621 | 0.976200 | 0.975200 | 0.977200 |
| crossfiresm | 0.999846 | 0.999756 | 0.002766 | 0.994737 | 0.997942 | 0.991575 |
| ipidaddr | 0.917002 | 0.899982 | 0.042869 | 0.973700 | 0.999958 | 0.950000 |
| ipidport | 0.941197 | 0.726826 | 0.027742 | 0.918600 | 0.995328 | 0.863400 |
| lrtcpdos02 | 0.996532 | 0.996275 | 0.000282 | 0.997958 | 0.997817 | 0.998098 |
| lrtcpdos05 | 0.991551 | 0.989816 | 0.000295 | 0.994664 | 0.994459 | 0.994869 |
| lrtcpdos10 | 0.978479 | 0.977374 | 0.002877 | 0.988200 | 0.987700 | 0.988600 |
| sshpwdla | 1.000000 | 1.000000 | 0.000000 | 0.913500 | 0.852900 | 0.999460 |
| sshpwdmd | 0.999519 | 1.000000 | 0.000556 | 0.981800 | 0.965400 | 0.999410 |
| sshpwdsm | 0.999520 | 1.000000 | 0.000548 | 0.998253 | 0.997055 | 0.999460 |
| telnetpwdla | 0.999533 | 1.000000 | 0.000467 | 0.977100 | 0.956500 | 0.999726 |
| telnetpwdmd | 1.000000 | 1.000000 | 0.000000 | 0.946800 | 0.904100 | 0.999726 |
| telnetpwdsm | 0.999450 | 1.000000 | 0.000556 | 0.981900 | 0.965700 | 0.999452 |

#### Interpreting the Results

- **AU_ROC ≥ 0.999:** Near-perfect detection. Most amplification attacks (charrdos,
  cldaprdos, dnsrdos, etc.) and password cracking (sshpwd*, telnetpwd*) fall here.

- **AU_ROC 0.95–0.999:** Strong detection with some misses. smtp_lrscan (0.954) is
  the weakest lrscan — SMTP scanning traffic closely resembles legitimate mail flows.

- **AU_ROC < 0.95:** Hardest attacks. ipidaddr (0.917) and ipidport (0.941) are SSH
  side-channel injection attacks that generate very little distinctive traffic,
  making them inherently challenging for flow-level detectors.

- **EER (Equal Error Rate):** Lower is better. Most attacks have EER < 0.003.
  crossfiremd (0.046) and smtp_lrscan (0.087) are outliers.

- **F1 vs AU_ROC discrepancy:** Some attacks (e.g., sshpwdla: AU_ROC=1.000, F1=0.914)
  have perfect ranking but suboptimal F1. This means the `water_line` threshold
  (11.0) is not well-calibrated for that attack — the detector correctly ranks
  malicious packets above benign ones, but the fixed threshold cuts too aggressively.

#### Generated Figures

ROC and PRC curves are saved to:
```
result_analyze/figure/{group}/{attack}_ROC.png
result_analyze/figure/{group}/{attack}_PRC.png
```

For example:
- `figure/brute/charrdos_ROC.png` — ROC curve with AUC annotation
- `figure/brute/charrdos_PRC.png` — Precision-Recall curve

---

## __0x05__ GPU & Performance Acceleration Plan

### Overview

This section outlines two acceleration strategies:
- **Plan A — KDE-only speedup (no dependency changes):** Accelerate `compute_kde_features.py` using GPU-backed libraries that are drop-in replacements for scikit-learn.
- **Plan B — Full pipeline GPU acceleration (baseline + KDE):** Port the C++ HyperVision core to leverage GPU parallelism. This requires new dependencies (CUDA, cuML, etc.).

---

### Plan A — Accelerate KDE Feature Precomputation Only (No Dependency Breakage)

The bottleneck is `BayesianGaussianMixture.fit()` called ~100k–300k times per attack
(once per unique `(src_ip, dst_ip)` pair). Each fit is independent → embarrassingly parallel.

#### A1. CPU Parallelism (already implemented)

The script now uses `joblib.Parallel` to distribute BGMM fits across all CPU cores.
For a 30-core node this gives ~20–25× speedup over serial execution.

```bash
# Use all cores (default)
python compute_kde_features.py --all-attacks --data-dir ../data --output-dir ../kde_features

# Explicitly limit to N cores
python compute_kde_features.py --all-attacks -j 16 --data-dir ../data --output-dir ../kde_features
```

#### A2. GPU-Accelerated Batched CAVI DPGMM (Recommended — Proven Approach)

> **Reference implementation:** `PIDSMaker/kde_computation.py` — the
> `BayesianGaussianMixtureGPU` class implements exactly this approach and is
> production-tested on CIC-IDS-2017 and HyperVision datasets.

The key insight is that scikit-learn's `BayesianGaussianMixture` is a **single-edge,
CPU-only** implementation. Instead, we implement the same Variational Bayesian GMM
via **Coordinate Ascent Variational Inference (CAVI)** using batched PyTorch tensor
operations on GPU — fitting **hundreds of edges simultaneously** in one kernel launch.

##### Mathematical Foundation: CAVI for Dirichlet Process GMM

The generative model uses a stick-breaking Dirichlet Process prior truncated at $K$
components:

$$v_k \sim \text{Beta}(1, \gamma) \qquad \text{(stick-breaking)}$$
$$\mu_k \sim \mathcal{N}(\mu_0, 1/\lambda_0) \qquad \text{(component means)}$$
$$\sigma^2_k \sim \text{InvGamma}(\alpha_0, \beta_0) \qquad \text{(component variances)}$$
$$z_n \sim \text{Cat}(\pi(v)) \qquad \text{(assignments)}$$
$$x_n | z_n = k \sim \mathcal{N}(\mu_k, \sigma^2_k) \qquad \text{(observations)}$$

The variational family is fully factorised:

$$q(v_k) = \text{Beta}(\tilde{a}_k, \tilde{b}_k), \quad q(\mu_k) = \mathcal{N}(m_k, 1/\tilde{\lambda}_k)$$
$$q(\sigma^2_k) = \text{InvGamma}(\tilde{\alpha}_k, \tilde{\beta}_k), \quad q(z_n) = \text{Cat}(r_n)$$

Because all conjugate pairs are in the exponential family, each CAVI update has a
**closed-form** solution — no gradient computation needed — making it ideal for GPU
execution.

##### CAVI Iteration (all operations batched over B edges)

**E-step** — Compute log-responsibilities for all $B \times N \times K$ triples:

$$\log r_{n,k} = \mathbb{E}[\log \pi_k] - \tfrac{1}{2}\mathbb{E}[\log \sigma^2_k] - \tfrac{1}{2}\mathbb{E}[\sigma^{-2}_k] \cdot (x_n - m_k)^2 - \tfrac{1}{2} \tfrac{\mathbb{E}[\sigma^{-2}_k]}{\tilde{\lambda}_k} + \text{const}$$

where $\mathbb{E}[\log \pi_k]$ comes from the stick-breaking via digamma functions:

$$\mathbb{E}[\log \pi_k] = \psi(\tilde{a}_k) - \psi(\tilde{a}_k + \tilde{b}_k) + \sum_{j=1}^{k-1} \big[\psi(\tilde{b}_j) - \psi(\tilde{a}_j + \tilde{b}_j)\big]$$

**M-step** — Closed-form Normal-Inverse-Gamma conjugate updates:

$$\tilde{\lambda}_k = \lambda_0 + N_k, \qquad m_k = \frac{\lambda_0 \mu_0 + \sum_n r_{nk} x_n}{\tilde{\lambda}_k}$$

$$\tilde{\alpha}_k = \alpha_0 + N_k/2, \qquad \tilde{\beta}_k = \beta_0 + \tfrac{1}{2} S_k + \tfrac{\lambda_0 N_k (\bar{x}_k - \mu_0)^2}{2\tilde{\lambda}_k}$$

$$\tilde{a}_k = 1 + N_k, \qquad \tilde{b}_k = \gamma + \sum_{j>k} N_j$$

**ELBO** (for convergence detection):

$$\mathcal{L} = \underbrace{\sum_n \log \sum_k \exp(\mathbb{E}[\log \pi_k] + \mathbb{E}[\log p(x_n|k)])}_{\text{data + assignment}} - \text{KL}[q(v) \| p(v)] - \text{KL}[q(\mu) \| p(\mu)] - \text{KL}[q(\sigma^2) \| p(\sigma^2)]$$

##### GPU Implementation Architecture

```
                     ┌──────────────────────────────────────┐
                     │ CPU: Data Preparation                │
                     │  1. Read .data files                 │
                     │  2. Group by (src_ip, dst_ip)        │
                     │  3. Compute inter-arrival diffs       │
                     │  4. MaxAbsScaler per edge → [-1, 1]  │
                     │  5. Sort by length, chunk into batches│
                     └──────────────┬───────────────────────┘
                                    │
                     ┌──────────────▼───────────────────────┐
                     │ GPU: Batched CAVI per chunk           │
                     │  Chunk = B edges, padded to N_max     │
                     │                                       │
                     │  for chunk in chunks:                 │
                     │    X, mask = pad_to_matrix(chunk)     │
                     │    # X: (B, N_max), mask: (B, N_max)  │
                     │                                       │
                     │    m_k = kmeans_init(X, mask, K)      │
                     │    for iter in range(max_iter):        │
                     │      log_r = E_step(X, m_k, ...)      │
                     │      # (B, N_max, K) — ONE kernel     │
                     │      R = softmax(log_r)               │
                     │      N_k, x_bar, S_k = M_step(X, R)  │
                     │      # batched bmm — ONE kernel       │
                     │      elbo = compute_elbo(...)         │
                     │      if converged: break              │
                     │                                       │
                     │    density = score_on_grid(grid_size)  │
                     │    # (B, grid_size) — ONE kernel      │
                     └──────────────┬───────────────────────┘
                                    │
                     ┌──────────────▼───────────────────────┐
                     │ CPU: Write CSV                        │
                     │  src_ip, dst_ip, kde_0, ..., kde_D    │
                     └──────────────────────────────────────┘
```

##### Key Implementation Details

1. **Batched k-means++ initialisation:** Instead of random init, use deterministic
   k-means++ on GPU — first centre = per-edge mean, subsequent centres = point with
   max min-distance to existing centres. This dramatically reduces CAVI iterations
   (typically converges in 30–80 iterations vs 200+ with random init).

2. **Padding & masking:** Edges have different lengths. Pad to a `(B, N_max)` matrix
   with a boolean mask. All arithmetic uses `mask_f = mask.float()` to zero out
   padding positions. Sort edges by length before chunking to minimise padding waste.

3. **Chunk size:** Process 256 edges per GPU batch. On an A100 (80 GB), this fits
   comfortably even with `N_max = 50,000`. Adjust based on available VRAM:

   | GPU | VRAM | Recommended chunk_size |
   |---|---|---|
   | RTX 3090 | 24 GB | 64–128 |
   | A100 40 GB | 40 GB | 128–256 |
   | A100 80 GB | 80 GB | 256–512 |
   | H100 | 80 GB | 256–512 |

4. **Subsampling:** Edges with >50,000 data points are randomly subsampled before
   fitting. DPGMM converges well before seeing all data.

5. **Memory management:** Call `torch.cuda.empty_cache()` between chunks. Each chunk
   allocates `O(B × N_max × K)` for responsibilities.

6. **Convergence:** Track ELBO normalised per data point. Stop after `patience` (20)
   non-improving iterations. The tolerance `tol = 1e-4` works well for density
   evaluation (not parameter recovery).

7. **Density evaluation on grid:** After fitting, evaluate the mixture density on a
   uniform grid of `kde_dim` points spanning `[min, max]` of each edge's scaled data:

   $$p(g_d) = \sum_k w_k \cdot \mathcal{N}(g_d \mid m_k, \sigma^2_k), \qquad d = 1, \dots, D$$

   L1-normalise to get the final RKHS feature vector. This is a single `(B, D, K)`
   tensor operation — no loop over edges.

##### Performance Comparison

| Method | 284k pairs (crossfirela) | All 43 attacks | Hardware |
|---|---|---|---|
| sklearn BGMM (serial) | ~8 hours | ~200+ hours | 1 CPU core |
| sklearn BGMM (joblib, 128 cores) | ~5–10 min | ~4–8 hours | 128 CPU cores |
| **Batched CAVI GPU** | **~30–90 sec** | **~10–30 min** | **1× A100** |

The GPU version achieves **100–500× speedup** over serial CPU because:
- All B edges in a chunk share the same kernel launch (no Python loop)
- E-step is a single `(B, N, K)` tensor broadcast
- M-step is a single `torch.bmm` call
- No gradient computation needed (closed-form updates)

##### Adaptation for HyperVision `compute_kde_features.py`

The changes are minimal — replace the `compute_bgmm_density()` function and the
per-edge loop with the batched GPU pipeline:

```python
# In compute_kde_features.py — replace the joblib parallel loop with:

import torch
from pidsmaker_kde import BayesianGaussianMixtureGPU  # or copy class inline

def process_data_file_gpu(data_path, output_path, kde_dim=20, min_timestamps=10,
                          n_components=100, gamma=5.0, chunk_size=256, device="cuda"):
    edge_timestamps = read_data_file(data_path)  # existing parsing code

    # 1. Filter & compute inter-arrival diffs
    edge_keys, data_arrays = [], []
    for (src, dst), ts_list in edge_timestamps.items():
        ts = np.sort(ts_list)
        diffs = np.abs(np.diff(ts))
        if len(diffs) >= min_timestamps:
            edge_keys.append((src, dst))
            data_arrays.append(diffs)

    # 2. MaxAbsScale per edge
    scaled = [arr / max(np.abs(arr).max(), 1e-15) for arr in data_arrays]

    # 3. Sort by length, chunk, pad to matrix, fit on GPU
    order = sorted(range(len(scaled)), key=lambda i: len(scaled[i]))
    results = {}

    for c0 in range(0, len(order), chunk_size):
        chunk_idx = order[c0 : c0 + chunk_size]
        chunk_arrays = [scaled[i] for i in chunk_idx]

        # Pad to (B, N_max) matrix
        B = len(chunk_arrays)
        N_max = max(len(a) for a in chunk_arrays)
        X = torch.zeros(B, N_max, device=device)
        mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)
        for i, arr in enumerate(chunk_arrays):
            X[i, :len(arr)] = torch.from_numpy(arr.astype(np.float32))
            mask[i, :len(arr)] = True

        # Fit batched CAVI DPGMM
        model = BayesianGaussianMixtureGPU(
            n_components=n_components, gamma=gamma, device=device)
        model.fit_batch(X, mask)

        # Evaluate density → (B, kde_dim)
        density = model.score_samples_grid(X, mask, grid_size=kde_dim)

        for local_i, orig_idx in enumerate(chunk_idx):
            results[edge_keys[orig_idx]] = density[local_i].cpu().numpy()

        del X, mask, model, density
        torch.cuda.empty_cache()

    write_csv(results, output_path)  # existing CSV writer
```

**Dependencies added:** `torch` only (PyTorch). Does **not** break any existing
C++ dependencies or the HyperVision binary. The GPU code runs entirely in the
Python preprocessing step.

##### Default Hyperparameters (from PIDSMaker reference)

| Parameter | Default | Notes |
|---|---|---|
| `n_components` (K) | 100 | Truncation level; DP prior auto-prunes unused components |
| `gamma` | 5.0 | DP concentration; larger → more components retained |
| `max_iter` | 300 | Max CAVI iterations; typically converges in 30–80 |
| `tol` | 1e-4 | Per-point normalised ELBO change threshold |
| `patience` | 20 | Early-stop after this many non-improving iterations |
| `n_init` | 1 | Number of random restarts (k-means init is deterministic) |
| `init_method` | "kmeans" | k-means++ on GPU; alternative: "linear" (evenly spaced) |
| `max_n_per_edge` | 50,000 | Subsample larger edges to bound VRAM |
| `chunk_size` | 256 | Edges per GPU batch |
| `truncate_threshold` | 0.99 | Cumulative weight cutoff for pruning components |

##### Priors (for MaxAbsScaled data in [-1, 1])

| Prior | Value | Interpretation |
|---|---|---|
| $\mu_0$ | 0 | Mean prior centred at origin |
| $\lambda_0$ | 1 | Unit precision on mean prior |
| $\alpha_0$ | 3.0 | InvGamma shape → prior mode at $\beta_0 / (\alpha_0 + 1) = 0.125$ |
| $\beta_0$ | 0.5 | InvGamma scale → moderate variance prior |

#### A3. GPU-Accelerated via PyTorch GMM (Simpler Alternative)

If the full CAVI DPGMM is not needed, a simpler EM-based GMM can be implemented
in ~50 lines of PyTorch. This lacks the automatic component pruning of the DP prior
but is faster to implement:

```python
# Simplified batched EM (no DP prior, fixed K)
for _it in range(max_iter):
    # E-step: (B, N, K) responsibilities
    log_lik = -0.5 * ((X.unsqueeze(2) - mu.unsqueeze(1)) / sigma.unsqueeze(1))**2
    log_r = log_lik + torch.log(weights).unsqueeze(1)
    R = torch.softmax(log_r, dim=2) * mask.unsqueeze(2)
    # M-step: weighted statistics
    N_k = R.sum(dim=1)
    mu = (X.unsqueeze(2) * R).sum(dim=1) / N_k.clamp(min=1e-10)
    # ... etc
```

#### A4. Reduce Work via Subsampling

For pairs with >10,000 timestamps, subsample to 5,000 before fitting. This is
mathematically justified since DPGMM converges well before seeing all data:

```python
if len(timestamps) > 5000:
    timestamps = np.random.choice(timestamps, 5000, replace=False)
```

**Expected speedup:** 2–5× for heavy-tailed attacks (crossfire, lrtcpdos) with
negligible accuracy loss.

---

### Plan B — Full Pipeline GPU Acceleration (Baseline + KDE)

> ⚠️ **This requires significant code changes and new CUDA dependencies.** No code
> changes are made here — this is a planning document only.

#### B1. Architecture Overview

```
                         CPU (current)                    GPU (proposed)
                    ┌─────────────────────┐          ┌─────────────────────────┐
  Stage 1: Parse    │ PcapPlusPlus         │    →     │ Same (I/O bound)        │
  Stage 2: Flows    │ Hash table + threads │    →     │ cuDF groupby or custom  │
  Stage 3: Edges    │ Sequential loop      │    →     │ CUDA kernel per edge    │
  Stage 4: Graph    │ DFS + adjacency map  │    →     │ cuGraph components      │
  Stage 5: Cluster  │ mlpack DBSCAN/KMeans │    →     │ cuML DBSCAN/KMeans      │
  Stage 6: Scores   │ Sequential assign    │    →     │ Thrust parallel scatter │
                    └─────────────────────┘          └─────────────────────────┘
```

#### B2. Stage-by-Stage GPU Porting Plan

##### Stage 1 — Packet Parsing (Low priority)

- **Current:** PcapPlusPlus reads packets sequentially from PCAP files
- **GPU opportunity:** Minimal. PCAP parsing is I/O-bound, not compute-bound
- **Recommendation:** Keep on CPU. Use memory-mapped I/O (`mmap`) for faster reads
- **Estimated speedup:** 1.2–1.5× (I/O limited)

##### Stage 2 — Flow Construction (Medium priority)

- **Current:** Multi-threaded hash table with periodic eviction
- **GPU approach:**
  1. Upload all packet 5-tuples to GPU as a `cuDF` DataFrame
  2. `groupby([src_ip, dst_ip, src_port, dst_port, proto])` → one group per flow
  3. Sort by timestamp within each group
  4. Apply timeout splits via a custom CUDA kernel (flag gaps > 10s)
- **Dependencies:** `cuDF` (RAPIDS) or custom CUDA kernels
- **Estimated speedup:** 5–10× for large PCAPs (>10M packets)
- **Complexity:** Medium — need to handle flow timeout splits in parallel

##### Stage 3 — Edge Construction & Feature Extraction (High priority ⭐)

This is the most compute-intensive stage and benefits most from GPU parallelism.

- **Current:** Sequential loop over flows; histogram computation per flow
- **GPU approach:**
  1. Copy all per-flow packet arrays to GPU (contiguous memory, CSR format)
  2. **Histogram kernels:** One CUDA block per flow → parallel binning of lengths,
     types, and inter-arrival times using `atomicAdd` to shared memory bins
  3. **KDE computation:** Batch GMM fit on GPU (see Plan A2 above)
  4. **Flag computation:** `is_elephant`, `is_periodic`, `is_scan` → trivial
     parallel reduction kernels
- **Implementation:**
  ```
  // Pseudocode for GPU histogram kernel
  __global__ void compute_edge_features(
      const Packet* packets,    // all packets, contiguous
      const int* flow_offsets,  // CSR row pointers
      float* len_hist,          // output: (n_flows, n_bins)
      float* type_hist,
      float* interval_hist,
      int n_flows, int n_len_bins, int n_type_bins, int n_int_bins)
  {
      int flow_id = blockIdx.x;
      int start = flow_offsets[flow_id];
      int end = flow_offsets[flow_id + 1];
      // Thread-parallel histogram accumulation...
  }
  ```
- **Dependencies:** CUDA toolkit, `thrust` (header-only, ships with CUDA)
- **Estimated speedup:** 20–50× for histogram computation; 100×+ for KDE
- **Complexity:** High — requires restructuring data layout to be GPU-friendly

##### Stage 4 — Graph Construction & Connected Components (Medium priority)

- **Current:** DFS-based connected components on CPU
- **GPU approach:**
  1. Build CSR adjacency from edge list → `thrust::sort` + `thrust::unique`
  2. Connected components via `cuGraph` (`cugraph::components::connected_components`)
  3. Component feature extraction → parallel reduction per component
- **Dependencies:** [cuGraph](https://github.com/rapidsai/cugraph) (RAPIDS)
- **Estimated speedup:** 10–50× for graphs with >100k vertices
- **Complexity:** Medium — cuGraph API closely mirrors NetworkX

##### Stage 5 — Clustering & Anomaly Scoring (High priority ⭐)

- **Current:** mlpack DBSCAN + KMeans (CPU, single-threaded)
- **GPU approach:**
  1. Replace `mlpack::dbscan` → `cuml::DBSCAN` (RAPIDS cuML)
  2. Replace `mlpack::kmeans` → `cuml::KMeans`
  3. Z3 vertex cover → keep on CPU (SMT solving is inherently sequential)
  4. Score computation → `thrust::transform` parallel map
- **cuML DBSCAN API (C++):**
  ```cpp
  #include <cuml/cluster/dbscan.hpp>
  cuml::dbscan(handle, input_data, n_samples, n_features, eps, min_pts, labels);
  ```
- **Dependencies:** cuML C++ library, RAFT (RAPIDS)
- **Estimated speedup:** 10–100× for DBSCAN on large components (>10k edges)
- **Complexity:** Medium — cuML C++ API is well-documented

##### Stage 6 — Score Propagation (Low priority)

- **Current:** Sequential loop assigning scores to packets
- **GPU approach:** `thrust::scatter` with max-reduction for overlapping assignments
- **Estimated speedup:** 5–10×
- **Complexity:** Low

#### B3. Dependency Changes for Full GPU Port

| Component | CPU (current) | GPU (proposed) | Notes |
|---|---|---|---|
| PCAP parsing | PcapPlusPlus | PcapPlusPlus (unchanged) | I/O bound |
| Flow grouping | Custom C++ hash table | cuDF or custom CUDA | Optional |
| Histograms | Manual loops | Custom CUDA kernels | Most impactful |
| KDE/GMM | scikit-learn (Python) | cuML GaussianMixture | Python or C++ |
| DBSCAN | mlpack 3.4.2 | cuML DBSCAN | C++ API available |
| KMeans | mlpack 3.4.2 | cuML KMeans | C++ API available |
| SMT solver | Z3 | Z3 (unchanged) | Sequential |
| Graph CC | Custom DFS | cuGraph | Optional |
| Linear algebra | Armadillo | cuBLAS / RAFT | If needed |

**New dependencies:** CUDA Toolkit ≥ 11.8, RAPIDS (cuML, cuGraph, cuDF) ≥ 24.10,
Thrust (ships with CUDA).

#### B4. Recommended Implementation Order

| Priority | Stage | Effort | Speedup | Breaks deps? |
|---|---|---|---|---|
| 1 | **KDE precompute (Plan A2/A3)** | 1 day | 50–200× | ❌ No |
| 2 | **Stage 5 — cuML DBSCAN/KMeans** | 3–5 days | 10–100× | ✅ Replaces mlpack |
| 3 | **Stage 3 — CUDA histogram kernels** | 3–5 days | 20–50× | ❌ Additive |
| 4 | **Stage 4 — cuGraph components** | 2–3 days | 10–50× | ❌ Additive |
| 5 | **Stage 2 — GPU flow construction** | 5–7 days | 5–10× | ✅ Major refactor |
| 6 | **Stage 6 — Thrust scatter** | 1 day | 5–10× | ❌ Additive |

**Total estimated effort:** 2–3 weeks for full GPU port.

#### B5. Quick Win: KDE-Only GPU Without Breaking Dependencies

The fastest path to GPU acceleration **without touching C++ code or existing
dependencies** is Plan A (above). Summary:

1. **Install cuML** in a separate conda env (does not affect C++ build)
2. **Modify only `compute_kde_features.py`** to use `cuml.GaussianMixture`
3. **Everything else unchanged** — same configs, same C++ binary, same run scripts
4. **Expected end-to-end time reduction:** KDE precompute drops from hours → minutes;
   C++ runtime unchanged

This is the **recommended approach** for immediate speedup with zero risk to the
existing pipeline.

---

## __0x06__ Reference
``` bibtex
@inproceedings{NDSS23-HyperVision,
  author    = {Chuanpu Fu and
               others},
  title     = {Detecting Unknown Encrypted Malicious Traffic in Real Time via Flow 
               Interaction Graph Analysis},
  booktitle = {NDSS},
  publisher = {ISOC},
  year      = {2023}
}
```
