# PIDSMaker CIC-IDS-2017 — Node Classification & Timing Analysis Notes

**Generated:** 2026-04-03 21:27:09

## Table of Contents

1. [Overview](#overview)
2. [Important Caveats](#important-caveats)
3. [Configuration Details](#configuration-details)
4. [Best Epoch Selection](#best-epoch-selection)
5. [Per-Attack Malicious Node Classification Diffs](#per-attack-malicious-node-classification-diffs)
6. [Full Node Classification Diffs (.pth)](#full-node-classification-diffs-pth)
7. [Training Timing Analysis](#training-timing-analysis)
8. [Inference Timing Analysis](#inference-timing-analysis)
9. [Detailed Per-Config Per-Epoch Per-Attack Node Lists](#detailed-per-config-per-epoch-per-attack-node-lists)

## Overview

This document provides a detailed analysis comparing node classifications and timing
between baseline PIDSMaker configurations and their KDE-enhanced / RED-enhanced variants
on the CIC-IDS-2017 dataset (14 per-attack evaluations across 3 graphs).

**Configs analyzed:**

| Config | Model | Type | Artifact Dir | Hash (first 8) |
|--------|-------|------|-------------|---------------|
| kairos_cicids | kairos | Baseline | artifacts_cicids | d894a741... |
| kairos_cicids_kde_diff | kairos | Enhanced (baseline: kairos_cicids) | artifacts_cicids_kde_ts_diff_reduced | d894a741... |
| kairos_cicids_kde_ts | kairos | Enhanced (baseline: kairos_cicids) | artifacts_cicids_kde_ts_reduced | d894a741... |
| kairos_cicids_red | kairos | Enhanced (baseline: kairos_cicids) | artifacts_cicids_red_reduced | d894a741... |
| orthrus_cicids | orthrus | Baseline | artifacts_cicids | aa3d7440... |
| orthrus_cicids_kde_diff | orthrus | Enhanced (baseline: orthrus_cicids) | artifacts_cicids_kde_ts_diff_reduced | aa3d7440... |
| orthrus_cicids_kde_ts | orthrus | Enhanced (baseline: orthrus_cicids) | artifacts_cicids_kde_ts_reduced | aa3d7440... |
| orthrus_cicids_red | orthrus | Enhanced (baseline: orthrus_cicids) | artifacts_cicids_red_reduced | aa3d7440... |

**Comparison pairs:**

- **kairos_cicids_kde_ts** vs **kairos_cicids**
- **orthrus_cicids_kde_ts** vs **orthrus_cicids**
- **kairos_cicids_kde_diff** vs **kairos_cicids**
- **orthrus_cicids_kde_diff** vs **orthrus_cicids**
- **kairos_cicids_red** vs **kairos_cicids**
- **orthrus_cicids_red** vs **orthrus_cicids**

## Important Caveats

### 1. result_model_epoch_X.pth Only Contains Last-Evaluated Attack

The `.pth` files in `precision_recall_dir/` are **overwritten during each attack's evaluation**.
This means the file only reflects node classifications for the **last evaluated attack**,
which is **graph_7_ddos_loit** (attack order is fixed across all configs).

For the .pth-based diffs, the y_true labels and y_hat predictions only correspond to
the `graph_7_ddos_loit` attack's evaluation context. The 2 malicious nodes in
these files are the ones relevant to that specific attack.

### 2. Per-Attack Malicious Node Classification From Logs

For per-attack malicious node TP/FN classification, we parse the wandb `output.log` files.
These logs contain lines like:
```
-> Malicious node 4761  : loss=6.1844 | is TP: ✅ 192.168.10.50 -> ...
-> Malicious node 3580  : loss=5.9732 | is TP: ❌ 192.168.10.51 -> ...
```

This gives us **malicious node** classification per attack. For **benign node**
classification changes per attack, we can only report the FP count delta (not individual
node IDs), since the .pth file is overwritten.

### 3. Orthrus Baseline Missing Inference Timing

The `orthrus_cicids` baseline configuration does **not** have `inference_tainted_batches_*.json`
files in its `batch_timing/` directory. This means we cannot compute inference timing
for orthrus baseline. Training timing IS available for all configs including orthrus baseline.

### 4. Best Epoch May Differ Between Enhanced and Baseline

When comparing best epochs, note that the best epoch for an enhanced config may differ
from its baseline's best epoch. This is expected — the edge reduction changes training
dynamics and may shift which epoch performs best.

## Configuration Details

| Config | wandb Run | Best Epoch | Artifact Path |
|--------|-----------|------------|---------------|
| kairos_cicids | run-20260403_022135-k1f4y9oo | 1 | artifacts_cicids/evaluation/evaluation/d894a741f4e1947819404f0420d422e88748a58298764bf4d56f98dfd13b38a2/CIC_IDS_2017_PER_ATTACK |
| kairos_cicids_kde_diff | run-20260403_081130-gefm583r | 0 | artifacts_cicids_kde_ts_diff_reduced/evaluation/evaluation/d894a741f4e1947819404f0420d422e88748a58298764bf4d56f98dfd13b38a2/CIC_IDS_2017_PER_ATTACK |
| kairos_cicids_kde_ts | run-20260403_062042-o693lwyf | 0 | artifacts_cicids_kde_ts_reduced/evaluation/evaluation/d894a741f4e1947819404f0420d422e88748a58298764bf4d56f98dfd13b38a2/CIC_IDS_2017_PER_ATTACK |
| kairos_cicids_red | run-20260403_173823-l88hmsi6 | 0 | artifacts_cicids_red_reduced/evaluation/evaluation/d894a741f4e1947819404f0420d422e88748a58298764bf4d56f98dfd13b38a2/CIC_IDS_2017_PER_ATTACK |
| orthrus_cicids | run-20260403_014359-w7n1uqtr | 0 | artifacts_cicids/evaluation/evaluation/aa3d7440e620ea1d6a3b92e5faffac7c154bc866affaaba6eaa6b14e7d6f8144/CIC_IDS_2017_PER_ATTACK |
| orthrus_cicids_kde_diff | run-20260403_071729-94zpg3qc | 5 | artifacts_cicids_kde_ts_diff_reduced/evaluation/evaluation/aa3d7440e620ea1d6a3b92e5faffac7c154bc866affaaba6eaa6b14e7d6f8144/CIC_IDS_2017_PER_ATTACK |
| orthrus_cicids_kde_ts | run-20260403_052853-xn9gc9ym | 9 | artifacts_cicids_kde_ts_reduced/evaluation/evaluation/aa3d7440e620ea1d6a3b92e5faffac7c154bc866affaaba6eaa6b14e7d6f8144/CIC_IDS_2017_PER_ATTACK |
| orthrus_cicids_red | run-20260403_172408-arq94flm | 0 | artifacts_cicids_red_reduced/evaluation/evaluation/aa3d7440e620ea1d6a3b92e5faffac7c154bc866affaaba6eaa6b14e7d6f8144/CIC_IDS_2017_PER_ATTACK |

## Best Epoch Selection

Criteria (in order of priority):
1. Highest number of attacks detected (attacks with TP > 0)
2. Highest number of unique TP malicious node IDs across all attacks
3. Highest total TP count across all attacks

| Config | Best Epoch | Attacks Detected | Unique TP Nodes | Total TP |
|--------|------------|-----------------|-----------------|----------|
| kairos_cicids | 1 | 2/14 | 3 | 4 |
| kairos_cicids_kde_diff | 0 | 2/14 | 2 | 3 |
| kairos_cicids_kde_ts | 0 | 2/14 | 4 | 5 |
| kairos_cicids_red | 0 | 0/14 | 0 | 0 |
| orthrus_cicids | 0 | 0/14 | 0 | 0 |
| orthrus_cicids_kde_diff | 5 | 9/14 | 3 | 10 |
| orthrus_cicids_kde_ts | 9 | 1/14 | 2 | 2 |
| orthrus_cicids_red | 0 | 0/14 | 0 | 0 |

## Per-Attack Malicious Node Classification Diffs

These tables show, for each comparison pair and each attack, how many malicious nodes
changed classification between the enhanced and baseline configs.

- **TP Gained**: Was FN (missed) in baseline, now TP (detected) in enhanced
- **TP Lost**: Was TP (detected) in baseline, now FN (missed) in enhanced
- **FP Delta**: Change in false positives (positive = more FPs in enhanced)

### Best Epoch

#### kairos_cicids_kde_ts (epoch 0) vs kairos_cicids (epoch 1)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 3 | 4 | +1 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 3 | 5 | +2 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 3 | 5 | +2 |
| 6/infiltration_step1 | 3 | 1 | 1 | 0 | 0 | 4 | 4 | 0 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 3 | 3 | 0 |
| 6/infiltration_step2 | 13 | 3 | 4 | **2** | **1** | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 4 | 5 | **2** | **1** | 16 | 21 | +5 |

**Node-level changes:**

- graph_6_infiltration_step2: TP gained = [4755, 4756]
- graph_6_infiltration_step2: TP lost = [4754]

#### orthrus_cicids_kde_ts (epoch 9) vs orthrus_cicids (epoch 0)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step2 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 2 | **2** | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 0 | 2 | **2** | **0** | 0 | 0 | 0 |

**Node-level changes:**

- graph_7_portscan: TP gained = [3580, 4761]

#### kairos_cicids_kde_diff (epoch 0) vs kairos_cicids (epoch 1)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 3 | 2 | -1 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 3 | 2 | -1 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 3 | 2 | -1 |
| 6/infiltration_step1 | 3 | 1 | 1 | 0 | 0 | 4 | 2 | -2 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 3 | 2 | -1 |
| 6/infiltration_step2 | 13 | 3 | 2 | 0 | **1** | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 4 | 3 | **0** | **1** | 16 | 10 | -6 |

**Node-level changes:**

- graph_6_infiltration_step2: TP lost = [4754]

#### orthrus_cicids_kde_diff (epoch 5) vs orthrus_cicids (epoch 0)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 1 | **1** | 0 | 0 | 3 | +3 |
| 6/web_xss | 3 | 0 | 1 | **1** | 0 | 0 | 1 | +1 |
| 6/web_sqli | 3 | 0 | 1 | **1** | 0 | 0 | 1 | +1 |
| 6/infiltration_step1 | 3 | 0 | 2 | **2** | 0 | 0 | 1 | +1 |
| 6/infiltration_cooldisk | 3 | 0 | 1 | **1** | 0 | 0 | 3 | +3 |
| 6/infiltration_step2 | 13 | 0 | 1 | **1** | 0 | 0 | 1 | +1 |
| 7/botnet | 7 | 0 | 1 | **1** | 0 | 0 | 4 | +4 |
| 7/portscan | 3 | 0 | 1 | **1** | 0 | 0 | 4 | +4 |
| 7/ddos_loit | 2 | 0 | 1 | **1** | 0 | 0 | 3 | +3 |
| **TOTAL** | 50 | 0 | 10 | **10** | **0** | 0 | 21 | +21 |

**Node-level changes:**

- graph_6_web_bruteforce: TP gained = [3580]
- graph_6_web_xss: TP gained = [3580]
- graph_6_web_sqli: TP gained = [3580]
- graph_6_infiltration_step1: TP gained = [3580, 4763]
- graph_6_infiltration_cooldisk: TP gained = [3580]
- graph_6_infiltration_step2: TP gained = [4762]
- graph_7_botnet: TP gained = [3580]
- graph_7_portscan: TP gained = [3580]
- graph_7_ddos_loit: TP gained = [3580]

#### kairos_cicids_red (epoch 0) vs kairos_cicids (epoch 1)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 3 | 0 | -3 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 3 | 0 | -3 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 3 | 0 | -3 |
| 6/infiltration_step1 | 3 | 1 | 0 | 0 | **1** | 4 | 0 | -4 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 3 | 0 | -3 |
| 6/infiltration_step2 | 13 | 3 | 0 | 0 | **3** | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 4 | 0 | **0** | **4** | 16 | 0 | -16 |

**Node-level changes:**

- graph_6_infiltration_step1: TP lost = [4763]
- graph_6_infiltration_step2: TP lost = [4754, 4760, 4763]

#### orthrus_cicids_red (epoch 0) vs orthrus_cicids (epoch 0)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step2 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 0 | 0 | **0** | **0** | 0 | 0 | 0 |

*No malicious node classification changes between enhanced and baseline.*

### Last Epoch (11)

#### kairos_cicids_kde_ts (epoch 11) vs kairos_cicids (epoch 11)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 2 | +2 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 2 | 2 | 0 |
| 6/infiltration_step1 | 3 | 1 | 0 | 0 | **1** | 2 | 0 | -2 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 2 | 0 | -2 |
| 6/infiltration_step2 | 13 | 2 | 1 | 0 | **1** | 0 | 1 | +1 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 3 | 1 | **0** | **2** | 10 | 9 | -1 |

**Node-level changes:**

- graph_6_infiltration_step1: TP lost = [4763]
- graph_6_infiltration_step2: TP lost = [4760]

#### orthrus_cicids_kde_ts (epoch 11) vs orthrus_cicids (epoch 11)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step2 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 2 | **2** | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 0 | 2 | **2** | **0** | 0 | 0 | 0 |

**Node-level changes:**

- graph_7_portscan: TP gained = [3580, 4761]

#### kairos_cicids_kde_diff (epoch 11) vs kairos_cicids (epoch 11)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 2 | +2 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 2 | +2 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 2 | +2 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 2 | +2 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 2 | +2 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 2 | 3 | +1 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 2 | 3 | +1 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 2 | 3 | +1 |
| 6/infiltration_step1 | 3 | 1 | 1 | 0 | 0 | 2 | 4 | +2 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 2 | 3 | +1 |
| 6/infiltration_step2 | 13 | 2 | 1 | 0 | **1** | 0 | 2 | +2 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 3 | 2 | **0** | **1** | 10 | 28 | +18 |

**Node-level changes:**

- graph_6_infiltration_step2: TP lost = [4760]

#### orthrus_cicids_kde_diff (epoch 11) vs orthrus_cicids (epoch 11)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step2 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 1 | **1** | 0 | 0 | 2 | +2 |
| 7/portscan | 3 | 0 | 1 | **1** | 0 | 0 | 2 | +2 |
| 7/ddos_loit | 2 | 0 | 1 | **1** | 0 | 0 | 1 | +1 |
| **TOTAL** | 50 | 0 | 3 | **3** | **0** | 0 | 5 | +5 |

**Node-level changes:**

- graph_7_botnet: TP gained = [3580]
- graph_7_portscan: TP gained = [3580]
- graph_7_ddos_loit: TP gained = [3580]

#### kairos_cicids_red (epoch 11) vs kairos_cicids (epoch 11)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 2 | 0 | -2 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 2 | 0 | -2 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 2 | 0 | -2 |
| 6/infiltration_step1 | 3 | 1 | 0 | 0 | **1** | 2 | 0 | -2 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 2 | 0 | -2 |
| 6/infiltration_step2 | 13 | 2 | 0 | 0 | **2** | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 3 | 0 | **0** | **3** | 10 | 0 | -10 |

**Node-level changes:**

- graph_6_infiltration_step1: TP lost = [4763]
- graph_6_infiltration_step2: TP lost = [4760, 4763]

#### orthrus_cicids_red (epoch 11) vs orthrus_cicids (epoch 11)

| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |
|--------|-----------|---------|--------|-----------|---------|---------|--------|------|
| 5/dos_slowloris | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_slowhttptest | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_hulk | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/dos_goldeneye | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 5/heartbleed | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_bruteforce | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_xss | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/web_sqli | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step1 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_cooldisk | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 6/infiltration_step2 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/botnet | 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/portscan | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7/ddos_loit | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **TOTAL** | 50 | 0 | 0 | **0** | **0** | 0 | 0 | 0 |

*No malicious node classification changes between enhanced and baseline.*

## Full Node Classification Diffs (.pth)

These diffs are based on `result_model_epoch_X.pth` files and reflect the classification
state after evaluating **graph_7_ddos_loit** (the last attack in the evaluation order).
All nodes (benign + malicious) are included.

| Enhanced Config | Baseline Config | Epoch Type | Enh Epoch | Base Epoch | Total Nodes | Flipped | Ben→Mal | Mal→Ben | Incorr→Corr | Corr→Incorr |
|----------------|-----------------|------------|-----------|------------|-------------|---------|---------|---------|-------------|-------------|
| kairos_cicids_kde_ts | kairos_cicids | best | 0 | 1 | 6143 | 0 | 0 | 0 | 0 | 0 |
| orthrus_cicids_kde_ts | orthrus_cicids | best | 9 | 0 | 6143 | 0 | 0 | 0 | 0 | 0 |
| kairos_cicids_kde_diff | kairos_cicids | best | 0 | 1 | 6143 | 0 | 0 | 0 | 0 | 0 |
| orthrus_cicids_kde_diff | orthrus_cicids | best | 5 | 0 | 6143 | 4 | 4 | 0 | 1 | 3 |
| kairos_cicids_red | kairos_cicids | best | 0 | 1 | 6143 | 0 | 0 | 0 | 0 | 0 |
| orthrus_cicids_red | orthrus_cicids | best | 0 | 0 | 6143 | 3 | 3 | 0 | 0 | 3 |
| kairos_cicids_kde_ts | kairos_cicids | last | 11 | 11 | 6143 | 0 | 0 | 0 | 0 | 0 |
| orthrus_cicids_kde_ts | orthrus_cicids | last | 11 | 11 | 6143 | 0 | 0 | 0 | 0 | 0 |
| kairos_cicids_kde_diff | kairos_cicids | last | 11 | 11 | 6143 | 0 | 0 | 0 | 0 | 0 |
| orthrus_cicids_kde_diff | orthrus_cicids | last | 11 | 11 | 6143 | 2 | 2 | 0 | 1 | 1 |
| kairos_cicids_red | kairos_cicids | last | 11 | 11 | 6143 | 0 | 0 | 0 | 0 | 0 |
| orthrus_cicids_red | orthrus_cicids | last | 11 | 11 | 6143 | 2 | 2 | 0 | 0 | 2 |

### Flipped Node Details

**orthrus_cicids_kde_diff vs orthrus_cicids (best epoch, enhanced=5, baseline=0):**

| Node ID | Baseline ŷ | Enhanced ŷ | y_true | Baseline Score | Enhanced Score | Change |
|---------|-----------|-----------|--------|---------------|---------------|--------|
| 3580 | 0 (ben) | 1 (mal) | 1 | 6.6739 | 15.1656 | ✅ improved |
| 4750 | 0 (ben) | 1 (mal) | 0 | 5.6040 | 12.4547 | ❌ worsened |
| 4759 | 0 (ben) | 1 (mal) | 0 | 7.8170 | 13.2218 | ❌ worsened |
| 4762 | 0 (ben) | 1 (mal) | 0 | 6.3617 | 15.1656 | ❌ worsened |

**orthrus_cicids_red vs orthrus_cicids (best epoch, enhanced=0, baseline=0):**

| Node ID | Baseline ŷ | Enhanced ŷ | y_true | Baseline Score | Enhanced Score | Change |
|---------|-----------|-----------|--------|---------------|---------------|--------|
| 4750 | 0 (ben) | 1 (mal) | 0 | 5.6040 | 14.1298 | ❌ worsened |
| 4759 | 0 (ben) | 1 (mal) | 0 | 7.8170 | 13.5621 | ❌ worsened |
| 4762 | 0 (ben) | 1 (mal) | 0 | 6.3617 | 14.1298 | ❌ worsened |

**orthrus_cicids_kde_diff vs orthrus_cicids (last epoch, enhanced=11, baseline=11):**

| Node ID | Baseline ŷ | Enhanced ŷ | y_true | Baseline Score | Enhanced Score | Change |
|---------|-----------|-----------|--------|---------------|---------------|--------|
| 3580 | 0 (ben) | 1 (mal) | 1 | 7.9913 | 14.7248 | ✅ improved |
| 4762 | 0 (ben) | 1 (mal) | 0 | 6.8787 | 14.7248 | ❌ worsened |

**orthrus_cicids_red vs orthrus_cicids (last epoch, enhanced=11, baseline=11):**

| Node ID | Baseline ŷ | Enhanced ŷ | y_true | Baseline Score | Enhanced Score | Change |
|---------|-----------|-----------|--------|---------------|---------------|--------|
| 4750 | 0 (ben) | 1 (mal) | 0 | 5.8510 | 14.6745 | ❌ worsened |
| 4759 | 0 (ben) | 1 (mal) | 0 | 6.5376 | 14.6745 | ❌ worsened |

## Training Timing Analysis

Average time per KDE tainted batch during training. Only non-baseline configs have
KDE tainted batches (baselines don't use KDE edge reduction).

### Baseline Training Timing (reference)

| Config | Tainted Batches/Epoch | Total Epochs | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) |
|--------|----------------------|-------------|---------------|-----------------|-------------------|
| kairos_cicids | 287 | 12 | 101.12 | 93.07 | 8.05 |

*Note: Baseline configs also have tainted batches because the tainted batch tracking
mechanism records batches containing tainted/attack-related edges regardless of whether
KDE reduction is applied.*

### Enhanced Config Training Timing (Grand Average Across All Epochs)

| Config | Tainted Batches (total) | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | Avg KDE Eligible | Avg Edges Reduced |
|--------|------------------------|---------------|-----------------|-------------------|----------------|-----------------|-------------------|
| kairos_cicids_kde_ts | 3432 | 85.35 | 77.72 | 7.63 | 0.1570 | 71.6 | 71.6 |
| orthrus_cicids_kde_ts | 3432 | 75.66 | 69.52 | 6.14 | 0.5730 | 288.8 | 288.8 |
| kairos_cicids_kde_diff | 3432 | 84.57 | 76.68 | 7.89 | 0.1570 | 71.6 | 71.6 |
| orthrus_cicids_kde_diff | 3432 | 74.94 | 68.20 | 6.74 | 0.5730 | 288.8 | 288.8 |
| kairos_cicids_red | 3432 | 100.37 | 92.94 | 7.42 | 0.1862 | 85.4 | 83.4 |
| orthrus_cicids_red | 3432 | 99.01 | 92.31 | 6.70 | 1.0000 | 526.7 | 502.8 |

### Per-Epoch Training Timing Detail

#### kairos_cicids_kde_ts

| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|
| 0 | 286 | 80.18 | 72.37 | 7.80 | 0.1570 | 71.6 | 71.6 |
| 1 | 286 | 87.26 | 79.63 | 7.63 | 0.1570 | 71.6 | 71.6 |
| 2 | 286 | 83.82 | 75.47 | 8.35 | 0.1570 | 71.6 | 71.6 |
| 3 | 286 | 83.63 | 75.27 | 8.37 | 0.1570 | 71.6 | 71.6 |
| 4 | 286 | 91.36 | 83.71 | 7.65 | 0.1570 | 71.6 | 71.6 |
| 5 | 286 | 82.52 | 75.44 | 7.08 | 0.1570 | 71.6 | 71.6 |
| 6 | 286 | 87.56 | 80.08 | 7.49 | 0.1570 | 71.6 | 71.6 |
| 7 | 286 | 87.75 | 80.42 | 7.33 | 0.1570 | 71.6 | 71.6 |
| 8 | 286 | 83.95 | 76.56 | 7.38 | 0.1570 | 71.6 | 71.6 |
| 9 | 286 | 81.29 | 74.30 | 6.99 | 0.1570 | 71.6 | 71.6 |
| 10 | 286 | 87.96 | 80.19 | 7.77 | 0.1570 | 71.6 | 71.6 |
| 11 | 286 | 86.91 | 79.19 | 7.72 | 0.1570 | 71.6 | 71.6 |

#### orthrus_cicids_kde_ts

| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|
| 0 | 286 | 98.67 | 91.95 | 6.72 | 0.5730 | 288.8 | 288.8 |
| 1 | 286 | 72.41 | 65.79 | 6.62 | 0.5730 | 288.8 | 288.8 |
| 2 | 286 | 71.03 | 64.60 | 6.43 | 0.5730 | 288.8 | 288.8 |
| 3 | 286 | 69.98 | 63.85 | 6.13 | 0.5730 | 288.8 | 288.8 |
| 4 | 286 | 69.73 | 64.16 | 5.57 | 0.5730 | 288.8 | 288.8 |
| 5 | 286 | 69.43 | 63.90 | 5.53 | 0.5730 | 288.8 | 288.8 |
| 6 | 286 | 80.76 | 74.10 | 6.66 | 0.5730 | 288.8 | 288.8 |
| 7 | 286 | 77.93 | 71.45 | 6.48 | 0.5730 | 288.8 | 288.8 |
| 8 | 286 | 69.79 | 64.20 | 5.59 | 0.5730 | 288.8 | 288.8 |
| 9 | 286 | 70.00 | 64.42 | 5.58 | 0.5730 | 288.8 | 288.8 |
| 10 | 286 | 84.16 | 77.52 | 6.63 | 0.5730 | 288.8 | 288.8 |
| 11 | 286 | 74.10 | 68.34 | 5.76 | 0.5730 | 288.8 | 288.8 |

#### kairos_cicids_kde_diff

| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|
| 0 | 286 | 80.08 | 72.61 | 7.48 | 0.1570 | 71.6 | 71.6 |
| 1 | 286 | 80.47 | 72.54 | 7.93 | 0.1570 | 71.6 | 71.6 |
| 2 | 286 | 81.17 | 74.05 | 7.12 | 0.1570 | 71.6 | 71.6 |
| 3 | 286 | 80.40 | 73.30 | 7.10 | 0.1570 | 71.6 | 71.6 |
| 4 | 286 | 94.17 | 85.85 | 8.32 | 0.1570 | 71.6 | 71.6 |
| 5 | 286 | 93.23 | 84.90 | 8.33 | 0.1570 | 71.6 | 71.6 |
| 6 | 286 | 87.27 | 78.91 | 8.35 | 0.1570 | 71.6 | 71.6 |
| 7 | 286 | 84.73 | 76.56 | 8.17 | 0.1570 | 71.6 | 71.6 |
| 8 | 286 | 81.07 | 73.19 | 7.88 | 0.1570 | 71.6 | 71.6 |
| 9 | 286 | 79.38 | 71.46 | 7.92 | 0.1570 | 71.6 | 71.6 |
| 10 | 286 | 87.66 | 79.62 | 8.05 | 0.1570 | 71.6 | 71.6 |
| 11 | 286 | 85.17 | 77.14 | 8.03 | 0.1570 | 71.6 | 71.6 |

#### orthrus_cicids_kde_diff

| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|
| 0 | 286 | 79.89 | 73.18 | 6.70 | 0.5730 | 288.8 | 288.8 |
| 1 | 286 | 76.22 | 69.46 | 6.76 | 0.5730 | 288.8 | 288.8 |
| 2 | 286 | 73.20 | 66.49 | 6.71 | 0.5730 | 288.8 | 288.8 |
| 3 | 286 | 73.77 | 66.98 | 6.79 | 0.5730 | 288.8 | 288.8 |
| 4 | 286 | 73.79 | 67.12 | 6.67 | 0.5730 | 288.8 | 288.8 |
| 5 | 286 | 73.45 | 66.79 | 6.66 | 0.5730 | 288.8 | 288.8 |
| 6 | 286 | 77.80 | 70.77 | 7.03 | 0.5730 | 288.8 | 288.8 |
| 7 | 286 | 77.38 | 70.44 | 6.94 | 0.5730 | 288.8 | 288.8 |
| 8 | 286 | 73.18 | 66.61 | 6.57 | 0.5730 | 288.8 | 288.8 |
| 9 | 286 | 73.34 | 66.95 | 6.40 | 0.5730 | 288.8 | 288.8 |
| 10 | 286 | 71.28 | 64.83 | 6.46 | 0.5730 | 288.8 | 288.8 |
| 11 | 286 | 76.04 | 68.77 | 7.27 | 0.5730 | 288.8 | 288.8 |

#### kairos_cicids_red

| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|
| 0 | 286 | 103.28 | 95.19 | 8.10 | 0.1862 | 85.4 | 83.4 |
| 1 | 286 | 101.16 | 93.39 | 7.77 | 0.1862 | 85.4 | 83.4 |
| 2 | 286 | 101.77 | 93.96 | 7.81 | 0.1862 | 85.4 | 83.4 |
| 3 | 286 | 102.16 | 94.32 | 7.84 | 0.1862 | 85.4 | 83.4 |
| 4 | 286 | 101.63 | 93.85 | 7.78 | 0.1862 | 85.4 | 83.4 |
| 5 | 286 | 102.14 | 94.36 | 7.78 | 0.1862 | 85.4 | 83.4 |
| 6 | 286 | 100.63 | 93.34 | 7.28 | 0.1862 | 85.4 | 83.4 |
| 7 | 286 | 98.99 | 92.03 | 6.96 | 0.1862 | 85.4 | 83.4 |
| 8 | 286 | 97.73 | 90.79 | 6.94 | 0.1862 | 85.4 | 83.4 |
| 9 | 286 | 98.41 | 91.44 | 6.98 | 0.1862 | 85.4 | 83.4 |
| 10 | 286 | 98.05 | 91.13 | 6.92 | 0.1862 | 85.4 | 83.4 |
| 11 | 286 | 98.45 | 91.51 | 6.94 | 0.1862 | 85.4 | 83.4 |

#### orthrus_cicids_red

| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|
| 0 | 286 | 126.11 | 119.01 | 7.10 | 1.0000 | 526.7 | 502.8 |
| 1 | 286 | 95.85 | 89.14 | 6.71 | 1.0000 | 526.7 | 502.8 |
| 2 | 286 | 95.88 | 89.23 | 6.64 | 1.0000 | 526.7 | 502.8 |
| 3 | 286 | 97.17 | 90.50 | 6.67 | 1.0000 | 526.7 | 502.8 |
| 4 | 286 | 97.31 | 90.60 | 6.70 | 1.0000 | 526.7 | 502.8 |
| 5 | 286 | 96.74 | 90.10 | 6.64 | 1.0000 | 526.7 | 502.8 |
| 6 | 286 | 96.29 | 89.61 | 6.69 | 1.0000 | 526.7 | 502.8 |
| 7 | 286 | 96.23 | 89.55 | 6.68 | 1.0000 | 526.7 | 502.8 |
| 8 | 286 | 98.85 | 92.21 | 6.64 | 1.0000 | 526.7 | 502.8 |
| 9 | 286 | 96.01 | 89.34 | 6.67 | 1.0000 | 526.7 | 502.8 |
| 10 | 286 | 95.72 | 89.07 | 6.66 | 1.0000 | 526.7 | 502.8 |
| 11 | 286 | 96.01 | 89.38 | 6.62 | 1.0000 | 526.7 | 502.8 |

## Inference Timing Analysis

Average time per KDE tainted batch during inference (test split only).

**Note:** `orthrus_cicids` baseline does NOT have inference tainted batch timing files.

### Baseline Inference Timing (reference)

| Config | Epoch | Epoch Type | Tainted Batches | Avg Total (ms) | Avg Forward (ms) |
|--------|-------|------------|----------------|---------------|-----------------|
| kairos_cicids | 1 | best | 3422 | 77.06 | 77.06 |
| kairos_cicids | 11 | last | 3422 | 82.21 | 82.21 |

### Enhanced Config Inference Timing

| Config | Epoch | Epoch Type | Tainted Batches | Avg Total (ms) | Avg Forward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |
|--------|-------|------------|----------------|---------------|-----------------|----------------|-------------|---------------|
| kairos_cicids_kde_ts | 0 | best | 3416 | 62.88 | 62.88 | 0.1556 | 55.5 | 55.5 |
| orthrus_cicids_kde_ts | 9 | best | 3422 | 58.27 | 58.27 | 0.4058 | 163.9 | 163.9 |
| kairos_cicids_kde_diff | 0 | best | 3416 | 60.70 | 60.70 | 0.1556 | 55.5 | 55.5 |
| orthrus_cicids_kde_diff | 5 | best | 3422 | 57.35 | 57.35 | 0.4058 | 163.9 | 163.9 |
| kairos_cicids_red | 0 | best | 3422 | 81.04 | 81.04 | 0.1871 | 67.7 | 64.9 |
| orthrus_cicids_red | 0 | best | 3422 | 76.99 | 76.99 | 0.6149 | 261.1 | 250.2 |
| kairos_cicids_kde_ts | 11 | last | 3416 | 63.60 | 63.60 | 0.1556 | 55.5 | 55.5 |
| orthrus_cicids_kde_ts | 11 | last | 3422 | 57.69 | 57.69 | 0.4058 | 163.9 | 163.9 |
| kairos_cicids_kde_diff | 11 | last | 3416 | 63.50 | 63.50 | 0.1556 | 55.5 | 55.5 |
| orthrus_cicids_kde_diff | 11 | last | 3422 | 60.29 | 60.29 | 0.4058 | 163.9 | 163.9 |
| kairos_cicids_red | 11 | last | 3422 | 81.79 | 81.79 | 0.1871 | 67.7 | 64.9 |
| orthrus_cicids_red | 11 | last | 3422 | 74.01 | 74.01 | 0.6149 | 261.1 | 250.2 |

## Detailed Per-Config Per-Epoch Per-Attack Node Lists

This section lists every malicious node and its classification (TP ✅ / FN ❌)
for each config, each evaluated epoch, and each attack.

### kairos_cicids

#### Epoch 0

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.7800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.7800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7790 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.1030 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=7.5360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1250 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=7.5380 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1250 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=7.5380 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1250 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=3, FN=2, TN=7213

- Node 3580: loss=7.5360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.4620 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.1250 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=7.5350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.7730 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.1480 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=2, FP=0, FN=11, TN=7190

- Node 4751: loss=8.3200 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.8940 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.9720 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.3600 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.2930 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.3370 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.7730 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.8740 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.4360 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.8450 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.7310 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.4360 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.0610 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.9900 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.7680 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.7050 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.3060 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.6100 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.4750 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.1580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.9960 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.4560 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1590 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.5970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.4560 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.9110 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9600 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.9110 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9550 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.9480 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=3, FN=3, TN=7461

- Node 3580: loss=8.6130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=3, FN=3, TN=7122

- Node 3580: loss=8.6130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=3, FN=3, TN=7021

- Node 3580: loss=8.6130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=4, FN=2, TN=7212

- Node 3580: loss=8.6120 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.7370 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.1610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=3, FN=3, TN=7057

- Node 3580: loss=8.6110 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.8850 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.1610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=3, FP=0, FN=10, TN=7190

- Node 4751: loss=9.5700 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.0090 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.1850 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.6510 | ✅ TP | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.5700 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.6080 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.8860 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.0530 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.7160 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=9.0300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8270 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.7160 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.2750 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=6.7530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.9220 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.0440 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.5880 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.6980 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2290 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.7540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.1100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.9260 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.9260 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.9260 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.9260 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.9260 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.7770 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=3, FN=3, TN=7461

- Node 3580: loss=9.5720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.2370 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2400 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=3, FN=3, TN=7122

- Node 3580: loss=9.5710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.2370 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2400 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=3, FN=3, TN=7021

- Node 3580: loss=9.5720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.2370 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2400 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=3, FN=2, TN=7213

- Node 3580: loss=9.5710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=11.2050 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2400 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=3, FN=3, TN=7057

- Node 3580: loss=9.5710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=10.1000 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2400 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=3, FP=0, FN=10, TN=7190

- Node 4751: loss=10.9640 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=10.2080 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=10.4890 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=11.0790 | ✅ TP | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=10.9720 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=11.0250 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=10.1000 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=10.2820 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=11.1640 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=10.2370 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.9480 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=11.1640 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=10.5400 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.0970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=10.0800 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.5480 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.9470 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.8600 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.5080 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3100 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.0970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3100 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.1180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.7010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.7010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=9.3670 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=3, FN=3, TN=7461

- Node 3580: loss=10.1080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.9490 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=3, FN=3, TN=7122

- Node 3580: loss=10.1080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.9490 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=3, FN=3, TN=7021

- Node 3580: loss=10.1080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.9490 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=4, FN=2, TN=7212

- Node 3580: loss=10.1080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=12.0610 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=3, FN=3, TN=7057

- Node 3580: loss=10.1080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=10.7310 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=3, FP=0, FN=10, TN=7190

- Node 4751: loss=11.7710 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=10.8630 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=11.1180 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=11.9230 | ✅ TP | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=11.7360 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=11.8200 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=10.7310 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=10.9450 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=12.0150 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=10.9490 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=10.3680 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=12.0150 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=11.2390 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.1040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=10.7170 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.6130 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.9690 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.5000 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.3430 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.1040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8440 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.5360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8440 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.8750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.8750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.8750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.8750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.8750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=9.4830 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=10.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=10.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=10.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=2, FN=2, TN=7214

- Node 3580: loss=10.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=12.6270 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=10.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=11.0030 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=2, FP=0, FN=11, TN=7190

- Node 4751: loss=12.3020 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=11.2440 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=11.6220 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=12.4380 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=12.2470 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=12.3500 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=11.0030 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=11.3410 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=12.5510 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=11.3460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=10.7580 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=12.5510 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=11.6920 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.0810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=11.0370 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.7540 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.7250 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.7370 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.1990 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3200 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.0810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.0210 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3200 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.1680 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.0210 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5500 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5500 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5500 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5500 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=9.5500 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=10.7350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.5250 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=10.7350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.5250 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=10.7350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.5250 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=2, FN=2, TN=7214

- Node 3580: loss=10.7350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=13.0250 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=10.7360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=11.2790 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=2, FP=0, FN=11, TN=7190

- Node 4751: loss=12.6430 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=11.4370 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=11.8450 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=12.8260 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=12.6230 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=12.7150 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=11.2790 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=11.6680 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=12.9570 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=11.5250 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.0340 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=12.9570 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=12.0000 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.2120 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=11.3540 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.7160 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.9970 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=11.0280 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2490 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3860 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.2120 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3860 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.3370 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.5400 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4020 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.5400 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4020 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.5400 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4020 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.5400 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4020 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.5400 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=9.6160 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=10.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.6360 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2840 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=10.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.6360 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2840 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=10.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.6360 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2840 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=2, FN=2, TN=7214

- Node 3580: loss=10.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=13.2690 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2840 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=10.8700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=11.3810 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2840 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=2, FP=0, FN=11, TN=7190

- Node 4751: loss=12.8630 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=11.5750 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=11.9840 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=13.0470 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=12.8360 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=12.9200 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=11.3810 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=11.8400 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=13.1910 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=11.6360 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.1010 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=13.1910 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=12.1770 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.2360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=11.4700 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.5690 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.8980 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=11.1950 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2220 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3820 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.2360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2760 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3820 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.1540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2760 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### kairos_cicids_kde_diff

#### Epoch 0 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.6990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.6990 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.7010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.7010 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.7080 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.7130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.7130 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.6990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.6010 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=6.1170 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7810 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1680 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=6.1210 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7890 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1680 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=6.1220 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7860 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1680 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=2, FN=2, TN=7214

- Node 3580: loss=6.1170 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=7.2370 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.1680 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=6.0950 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.5560 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2770 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=2, FP=0, FN=11, TN=7190

- Node 4751: loss=7.1000 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.7100 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.7140 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.0390 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.0330 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.0700 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.5570 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.7460 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.1770 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=5.7780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.1600 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=7.1770 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.9950 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.0130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=6.5850 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.7110 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.1490 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=6.6650 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.2000 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2420 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.7450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.1620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2420 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.1540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1540 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.8460 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.9260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.8460 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.9260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.8410 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.9240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.8410 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=4.9230 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.8470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.9150 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=6.5410 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=6.5420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=6.5420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2730 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=1, FN=2, TN=7215

- Node 3580: loss=6.5360 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=7.8780 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.1890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=6.5280 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.0650 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.3070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=7.6520 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.2900 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.2490 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.5630 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.6590 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.6770 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.0670 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.2720 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.8020 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.5400 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=7.8020 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.5420 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.2300 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.1870 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.3240 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.4330 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.2410 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.5500 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2250 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.0720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.4380 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2250 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.3750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3750 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.0780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.0780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0770 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0770 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.6430 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=6.7410 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=6.7420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=6.7430 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=6.7420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.3370 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.1290 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=6.7390 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.5540 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2710 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.1240 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.8320 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.5930 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.9980 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.0620 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.1620 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.5540 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.3620 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.3270 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2670 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.7430 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.3270 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.9840 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.3310 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.6060 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.5310 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.8480 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.8120 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.7850 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.1390 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.6290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1390 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.7340 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7340 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.1720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1720 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.1720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1720 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.1810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1810 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.1800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1800 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.1720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.4580 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=6.4380 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.5180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=6.4380 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.5180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=6.4390 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.5180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=3, FN=2, TN=7213

- Node 3580: loss=6.4460 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.5750 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=6.4490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.9390 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2320 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=8.0910 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.7810 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.5300 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.9120 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.9280 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.0540 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.1430 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.0530 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.2100 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.5180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.4460 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.7290 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.7970 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.4370 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.9540 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.1110 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.0000 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.2160 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.9750 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0990 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.0210 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1270 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0990 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.4820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4820 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.3050 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3050 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.3050 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3050 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.3130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3130 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.3130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3130 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.3050 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.8700 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=6.4910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0500 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=6.4950 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0500 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=6.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0500 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=2, FN=3, TN=7214

- Node 3580: loss=6.4980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.6420 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0500 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=6.5090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.6820 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2080 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=8.2270 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.7260 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.5940 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.8070 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.9400 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.8720 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.3980 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.7170 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.9300 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.4950 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.9450 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.7790 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.4650 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.0570 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.4010 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.4070 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.5900 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.1100 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.1030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.6990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3090 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.5270 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5270 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=2, FN=2, TN=4774

- Node 3580: loss=5.5190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5190 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=2, FN=2, TN=4318

- Node 3580: loss=5.5190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5190 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=2, FN=2, TN=4251

- Node 3580: loss=5.5150 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5150 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.5150 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5150 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=2, FN=2, TN=4251

- Node 3580: loss=5.5190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.3890 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=6.5440 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0910 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=6.5440 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0910 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=6.5420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0910 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=6.5440 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.7490 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0910 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=6.5550 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.9280 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=8.2280 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.9030 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1110 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.7360 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.0420 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.7840 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.4730 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.7400 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.7720 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.6620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.5430 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.1330 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.8830 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=1, FN=6, TN=6926

- Node 3580: loss=5.3870 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.4840 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.2880 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.3840 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.9130 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.1810 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0970 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.8990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0970 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.4290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=2, FN=2, TN=4774

- Node 3580: loss=5.3760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3760 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=2, FN=2, TN=4318

- Node 3580: loss=5.3760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3760 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=2, FN=2, TN=4251

- Node 3580: loss=5.3780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=2, FN=2, TN=4251

- Node 3580: loss=5.3780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=2, FN=2, TN=4251

- Node 3580: loss=5.3760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.1900 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=3, FN=3, TN=7461

- Node 3580: loss=6.5660 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=3, FN=3, TN=7122

- Node 3580: loss=6.5750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=3, FN=3, TN=7021

- Node 3580: loss=6.5730 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=4, FN=2, TN=7212

- Node 3580: loss=6.5760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.1120 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=3, FN=3, TN=7057

- Node 3580: loss=6.5810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.0540 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.1950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=2, FN=12, TN=7188

- Node 4751: loss=8.6400 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.7220 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2310 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.7510 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.8530 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.0050 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.4570 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.7910 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.9630 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.9290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.5700 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.2770 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.1970 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.5140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.1860 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.6540 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.6270 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.7970 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.4110 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.1130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.2050 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.3820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3820 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### kairos_cicids_kde_ts

#### Epoch 0 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.1510 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0370 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.1530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0360 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.1560 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0320 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.1600 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0070 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.1510 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.6690 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=4, FN=3, TN=7460

- Node 3580: loss=6.3180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.5540 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=5, FN=3, TN=7120

- Node 3580: loss=6.3240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.5540 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=5, FN=3, TN=7019

- Node 3580: loss=6.3230 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.5540 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=4, FN=2, TN=7212

- Node 3580: loss=6.3110 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=7.5930 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.5540 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=3, FN=3, TN=7057

- Node 3580: loss=6.3090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.7620 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.5690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=4, FP=0, FN=9, TN=7190

- Node 4751: loss=7.4210 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.0660 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.0110 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.2690 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.4610 | ✅ TP | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.4380 | ✅ TP | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.7640 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.4290 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.5880 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.3070 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.5310 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=7.5880 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.3260 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=4.8870 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=6.6710 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.8670 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.8310 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=6.8560 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.5380 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.4890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.0700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.8710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=4.4040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0330 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.5870 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5330 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.5890 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5310 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.5880 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.5880 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5160 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.5860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.3400 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=6.7890 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.6020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=6.7920 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9450 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.6020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=6.7920 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9450 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.6020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=2, FN=2, TN=7214

- Node 3580: loss=6.7930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.3390 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.6020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=6.7920 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.4230 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.5660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=2, FP=0, FN=11, TN=7190

- Node 4751: loss=8.1540 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.7070 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.7330 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.9700 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.1660 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.1450 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.4250 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.0470 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.3160 | ✅ TP | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.9430 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.2140 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.3160 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.0430 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.1470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.5150 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.4660 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.1030 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.6230 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.8650 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.4850 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.7750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3420 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4850 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.0740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5410 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.4120 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.6370 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.4130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.6380 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.4020 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.6380 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.4010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.6380 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.4130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.7120 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=7.1850 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=7.1950 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=7.1940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=7.1920 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.8280 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=7.1800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.9680 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.2430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.5840 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.2600 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.0880 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.3410 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.6000 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.6210 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.9710 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.3390 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.8150 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.3780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.7310 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.8150 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.4310 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.7300 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.9230 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.8590 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.0980 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.0580 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.9550 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.5440 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8800 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.3820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.0030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.0030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0050 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0050 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0040 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.9570 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=7.2140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4410 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3370 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=7.2140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4410 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3370 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=7.2150 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4410 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3370 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=7.2110 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.9020 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.3370 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=7.2050 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.0580 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.3170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.7140 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.2720 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.0870 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.4380 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.6980 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.6950 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.0600 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.9900 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.9050 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.4410 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.6980 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.9550 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.5720 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.8070 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.1920 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.8570 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.9460 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.3410 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.8480 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.4650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.7740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0540 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4640 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.0370 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5600 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.4530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4530 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.4530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4530 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.4570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.4570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.4530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.3640 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=7.5020 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9340 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3880 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=7.4990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9340 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3880 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=7.4980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9340 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3880 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=7.4980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.0610 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.3880 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=7.5030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.2610 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.3860 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.8920 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.4130 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2200 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.4350 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.6040 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.6570 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.2600 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.7510 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.8280 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.9340 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.7050 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.3510 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.5240 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.8540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.6730 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.3690 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.9680 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.9120 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.9280 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3090 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.0340 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3090 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.2990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.3470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.7420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7420 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.7420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7420 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.7450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7450 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.7450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7450 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.7420 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.4710 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=7.5240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9550 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4630 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=7.5220 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9550 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4630 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=7.5210 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9550 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4630 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=7.5240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.1630 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.4630 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=7.5260 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.2060 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.4510 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.6930 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.3550 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1980 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.2460 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.4760 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.4490 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.6750 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.5480 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.6220 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.9550 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.6730 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.4190 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.5290 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.8850 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.7520 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.3170 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.9100 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.8660 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.0010 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2860 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.0970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6210 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2860 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.5970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.7700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7700 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.7700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7700 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.7590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=2, FN=2, TN=4251

- Node 3580: loss=5.7590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.7700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.3570 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=7.5460 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=7.5450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=7.5460 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=7.5400 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.2980 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.4690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=7.5390 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.3720 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.4590 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=9.1030 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.2760 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.3200 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.2400 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.4960 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.3250 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.6870 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.3810 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.4770 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.3180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=7.7630 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.6930 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.3900 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=6.0450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.8420 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.7570 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.8510 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.9830 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.9640 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3220 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.3410 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3220 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.6080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.6080 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### kairos_cicids_red

#### Epoch 0 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.2080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.2080 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.2070 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.2030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0140 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.2090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=5.9690 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=5.7820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0060 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1640 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=5.8010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.9960 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=5.7980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.9970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1640 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=5.7950 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=6.5040 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.1650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=5.7800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.5790 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.3070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=6.2990 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=5.7590 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.2510 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=6.2970 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=6.8750 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=5.7080 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.2140 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.1990 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=6.4340 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=5.9960 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=5.8500 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=6.5050 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.2130 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.9730 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=6.3760 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.5610 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.2600 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=6.4350 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.0210 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2870 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.2270 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2270 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2860 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.9730 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.1840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=5.0820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.3120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=5.0810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2430 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2420 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0790 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2400 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=5.0820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.5720 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=5.4940 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2530 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4080 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=5.5130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=5.5100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4080 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=5.5060 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=7.0460 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.4080 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=5.4960 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.6730 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.4200 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=6.9390 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.1270 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.6720 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=6.6840 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.1070 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=6.1040 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.3910 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=6.6730 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=6.6700 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.0990 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=7.0510 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.7630 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=6.1910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=6.5120 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.8200 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.4550 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.0790 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.1570 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.5650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.5320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6090 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.5650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.1910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6090 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=3.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=3.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=3.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=3.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=3.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.2260 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=4.9660 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7470 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=4.9860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7470 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=4.9850 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.0970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7470 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=4.9840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=7.7980 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.7470 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=4.9790 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.5000 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.7470 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=7.6560 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.3960 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.1770 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.2390 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=6.7920 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=6.7340 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.5050 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.3510 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.0550 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.0960 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.6810 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=7.7990 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.3950 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=4.5030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.2050 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.9350 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8190 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.6130 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.2110 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=1.0920 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.2640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.7570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=1.0930 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=4.4140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.5480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.1290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.1290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.1290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.1290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.2840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.1290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.4390 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=4.9130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.1120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=1.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=4.9140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.1120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=1.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=4.9130 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.1120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=1.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=4.9140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.0120 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=1.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=4.9120 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.4270 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=1.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=7.8510 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.5520 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.3080 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.4740 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.0640 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.0640 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.4160 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.1770 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.0770 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.1110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.7550 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.0120 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.5960 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.2140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.4380 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.1400 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8350 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=7.8590 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.3820 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=1.0690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.4070 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.9780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=1.0690 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.2140 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=3.9550 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0230 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=3.9550 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0230 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=3.9550 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0230 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=3.9550 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0230 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=3.9550 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.6960 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=5.1490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2000 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=5.1490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2000 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=5.1500 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2000 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=5.1580 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.2660 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.7950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=5.1600 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.4140 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.7950 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.0690 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.6340 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.3750 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.6970 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.3140 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.2850 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.3790 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.1060 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.1260 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2000 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.8600 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.2660 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.8340 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.1720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.6920 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.1840 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8640 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.0820 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.4420 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.8670 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.4700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.8300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.8670 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.1720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.4780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.2320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0870 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.2320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0870 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.2320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0870 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.2320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.0870 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.2320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.8370 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=5.4190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=5.4190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=5.4190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=5.4200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.3890 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.7660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=5.4210 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.6550 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.7660 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.2170 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.6880 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.4710 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.8650 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.5310 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.5290 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.6550 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.4350 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.3600 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.8940 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.3890 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.9540 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.5640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.8220 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.3250 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8820 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.2070 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.6810 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.7700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=5.7800 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.9640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7700 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.5640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.9300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=4.2570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=4.2570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.2570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.2570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=5.1100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=4.2570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.9370 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=5.5580 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.8360 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=5.5630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.8360 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=5.5630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.8360 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=5.5680 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=8.4460 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.8360 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=5.5670 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.6750 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.8360 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.2260 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=6.8010 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.4240 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=7.9270 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=7.5720 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=7.6110 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.6750 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=5.6840 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=7.4760 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=6.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=6.8290 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=8.4450 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=7.9870 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=5.8540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.8740 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.2860 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8260 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.2620 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=6.6240 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.7610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.0440 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.7610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=5.8540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.2180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### orthrus_cicids

#### Epoch 0 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.6550 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.8590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.8600 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.8590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.8600 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.9330 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2910 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.8710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.2510 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.3210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=10.2800 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5020 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.6410 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=10.4390 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.8840 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.3880 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.2510 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.0880 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.7730 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8560 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.7730 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.2840 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=6.6740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.0650 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.6790 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.7740 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.0310 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=2.9830 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3930 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.7840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.7840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3930 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.6740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.2320 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.3180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.3210 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.3200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.3200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.0480 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.3320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.3040 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.7550 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.7890 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.7390 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.7410 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.4800 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.9060 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.3020 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.7370 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.9260 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.3190 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.2110 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.2610 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.2760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.3880 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.5100 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.7580 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.1610 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.0950 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3740 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.5520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3740 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.2760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2760 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.4420 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.0040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.0000 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.9990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.0020 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.5030 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.0200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.3130 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0340 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1860 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5350 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2140 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.1910 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.0360 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.2420 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.3070 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.9110 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.3810 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.0000 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.3810 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.7680 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.5970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.6470 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.7730 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.2890 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.5110 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.6810 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.1040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.8950 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8950 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.5970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.6540 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.9000 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.9000 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.8980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.8990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.5880 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.9040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.9750 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1320 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.3660 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1380 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.3040 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.0970 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.4020 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.9700 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.9460 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.5300 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8950 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.5300 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6390 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.4250 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.8980 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.8340 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.3140 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.1280 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.7830 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.6180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.4250 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4250 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.9750 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.2650 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.2620 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=9.2630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.2630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.8760 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.2750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.1180 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.4450 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5240 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1970 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.7140 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.4650 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.6780 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.1180 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.0000 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.9120 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.2610 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.9120 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.7870 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.5810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.3210 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.5920 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.0810 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.4150 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2580 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.6640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.5810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5810 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.2840 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.8490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.8510 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.8520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.8490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.6880 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.8530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.1260 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1570 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.3410 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.0980 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.5020 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.2520 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.4690 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.1250 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.0750 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.6710 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8500 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.6710 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6310 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.7470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.8610 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.1290 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.3230 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.5890 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2600 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2320 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.7470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.7470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2320 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.7470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.7470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.5010 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.0610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.0660 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=9.0650 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.0450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.8500 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.0350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.4150 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.2400 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.4420 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1360 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.6920 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.3670 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.6970 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.4150 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.3650 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.8910 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.0530 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.8910 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6410 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.9910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.5890 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.4720 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.0630 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.6600 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.7870 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3570 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.9910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3570 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.9910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### orthrus_cicids_kde_diff

#### Epoch 0

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.5270 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1330 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.5270 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.5220 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.5220 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.5270 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.4070 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.6190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0720 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.6190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0720 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=9.6190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0720 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.6190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.2210 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0720 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.6190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.6590 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.1020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.9670 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.2350 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.9750 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.6250 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.5930 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.0940 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.6630 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.6110 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.2070 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.1520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.6190 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.2260 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.6050 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=3, FN=6, TN=6924

- Node 3580: loss=11.1920 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.2630 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.7050 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.9600 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.4110 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.6910 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.4140 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=4, FN=2, TN=7026

- Node 3580: loss=11.1920 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.4140 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=3, FN=1, TN=6138

- Node 3580: loss=10.8510 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1750 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.8590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.8590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.3190 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=1, FP=1, FN=2, TN=7463

- Node 3580: loss=11.6100 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2200 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=1, FP=1, FN=2, TN=7124

- Node 3580: loss=11.6100 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2200 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=1, FP=1, FN=2, TN=7023

- Node 3580: loss=11.6100 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2200 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=1, FN=2, TN=7215

- Node 3580: loss=11.6100 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.1620 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=1, FP=1, FN=2, TN=7059

- Node 3580: loss=11.6100 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.1950 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0530 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=9.1490 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.8080 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.4620 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.8820 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.9610 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.3680 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.2010 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.8510 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.3150 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.2200 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.6100 | ✅ TP | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.2260 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.2580 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=4, FN=6, TN=6923

- Node 3580: loss=12.8440 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.5770 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.0230 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=7.1250 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.1090 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.1070 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=4, FN=2, TN=7026

- Node 3580: loss=12.8490 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0890 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=3, FN=1, TN=6138

- Node 3580: loss=12.3570 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.3900 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3900 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.3900 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3900 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.3960 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3960 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.3960 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3960 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.3900 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.2790 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=1, FP=1, FN=2, TN=7463

- Node 3580: loss=12.0410 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.7720 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0150 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=1, FP=1, FN=2, TN=7124

- Node 3580: loss=12.0410 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.7720 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0150 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=1, FP=1, FN=2, TN=7023

- Node 3580: loss=12.0410 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.7720 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0150 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=1, FP=1, FN=2, TN=7215

- Node 3580: loss=12.0410 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.8540 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0150 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=1, FP=1, FN=2, TN=7059

- Node 3580: loss=12.0410 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.9720 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0240 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=8.7750 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5930 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2590 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.4260 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.7030 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.8020 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.9660 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.4460 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.7920 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.7720 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=12.1340 | ✅ TP | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.0820 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.9780 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=4, FN=6, TN=6923

- Node 3580: loss=16.3410 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.8620 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.2870 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.7290 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.6470 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.3410 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0570 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=4, FN=2, TN=7026

- Node 3580: loss=16.3440 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.6930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0570 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=3, FN=1, TN=6138

- Node 3580: loss=15.4940 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.7010 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.6880 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.6880 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.6930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.7420 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=1, FP=3, FN=2, TN=7461

- Node 3580: loss=11.9040 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.9170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=1, FP=1, FN=2, TN=7124

- Node 3580: loss=11.9040 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.9170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=1, FP=1, FN=2, TN=7023

- Node 3580: loss=11.9040 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.9170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=2, FP=1, FN=1, TN=7215

- Node 3580: loss=11.9040 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=11.1570 | ✅ TP | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0070 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=1, FP=3, FN=2, TN=7057

- Node 3580: loss=11.9040 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.8800 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0230 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=8.8570 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.3910 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.5870 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.3320 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.4850 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.7940 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.8760 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.6080 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.7560 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.9170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.9420 | ✅ TP | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.4700 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.8960 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=4, FN=6, TN=6923

- Node 3580: loss=15.6810 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.6510 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.3530 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.7700 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.2230 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.6890 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0760 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=4, FN=2, TN=7026

- Node 3580: loss=15.6770 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0760 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=3, FN=1, TN=6138

- Node 3580: loss=15.1660 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.8860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.8860 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.8860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.8860 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.8830 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.8830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.8830 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.8830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.8860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.6590 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=12.7240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=12.7240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=12.7240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=12.7240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.5680 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=12.7240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.4790 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=9.4030 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.0440 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2280 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.2170 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.3560 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.6140 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.4810 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.4020 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.5740 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.6300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=13.1500 | ✅ TP | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.5760 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.9020 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=4, FN=6, TN=6923

- Node 3580: loss=17.3900 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.9970 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.9530 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.6210 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.0850 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.6360 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=4, FN=2, TN=7026

- Node 3580: loss=17.4040 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.4620 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=3, FN=1, TN=6138

- Node 3580: loss=16.9130 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.3990 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=9.1570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=9.1570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=9.1570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=9.1570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=9.1570 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=9.1840 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=10.2980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=10.2980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=10.2980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=10.2980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.4060 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=10.2980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.4570 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0140 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.3340 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.8460 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1600 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.3560 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.2830 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.6580 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.4560 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.6370 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.5960 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.4610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=10.4820 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.4080 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6370 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=3, FN=6, TN=6924

- Node 3580: loss=14.7120 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.5530 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.9480 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8640 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.1760 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.7330 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=3, FN=2, TN=7027

- Node 3580: loss=14.7130 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.7460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=2, FN=1, TN=6139

- Node 3580: loss=14.5080 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.7460 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=9.0180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=9.0180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=9.0180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=9.0180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.0180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=9.0180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.9200 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=11.2470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6010 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=11.2470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6010 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=11.2470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6010 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=11.2470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.8190 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=11.2470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.3860 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0100 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1010 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.9030 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.5020 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.5750 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.7180 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.9920 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.3860 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.0120 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.0060 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.6010 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.2470 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.8240 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.0120 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=1, FP=2, FN=6, TN=6925

- Node 3580: loss=14.7250 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.8070 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=6.5530 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.4390 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.7670 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.4990 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0440 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=1, FP=2, FN=2, TN=7028

- Node 3580: loss=14.7250 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0440 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=1, FP=1, FN=1, TN=6140

- Node 3580: loss=14.7250 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.2680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### orthrus_cicids_kde_ts

#### Epoch 0

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.5630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1700 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.5630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6920 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.5590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6920 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.5590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.6920 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.5630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.9730 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=10.2840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1280 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=10.2880 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1280 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=10.2870 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1280 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=2, FN=3, TN=7214

- Node 3580: loss=10.2790 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=11.1840 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0580 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=10.2930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=9.2930 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=10.5900 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.8420 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.7020 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=10.1100 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=10.2400 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.7030 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=9.2830 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=12.8640 | ✅ TP | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.7250 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=9.1280 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=10.2850 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=11.3060 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=10.5090 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=8.0180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=9.0210 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.6380 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.9090 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.2840 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.4110 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.5370 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=9.1380 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.8750 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.5370 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.5350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.8750 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.8340 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.3680 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.8340 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.0290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.8290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.0290 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.8290 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.0260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.8340 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=9.0370 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=2, FN=3, TN=7462

- Node 3580: loss=9.6510 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=2, FN=3, TN=7123

- Node 3580: loss=9.6540 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=2, FN=3, TN=7022

- Node 3580: loss=9.6520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=2, FN=3, TN=7214

- Node 3580: loss=9.6590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.3610 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0430 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=2, FN=3, TN=7058

- Node 3580: loss=9.6780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.8320 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0450 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=1, FP=1, FN=12, TN=7189

- Node 4751: loss=9.9890 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.3760 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.0660 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.3420 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.5490 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.9490 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.8300 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=13.2400 | ✅ TP | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.0190 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=9.1110 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.6540 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.3380 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.8540 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=8.0610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.7850 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2690 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=7.0250 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.9290 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.9140 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2010 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=9.7560 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.8830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2010 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=8.0610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.8830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.1820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1820 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.1820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1820 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.1710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.1710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1710 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.1820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.6830 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=10.5730 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=10.5780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=10.5760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=10.5710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.4020 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0160 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=10.5840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.5030 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0280 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=10.0820 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.7090 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.3860 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.3260 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.7630 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.9470 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.4950 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=10.3420 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.8520 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.5740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=10.5720 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.5740 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.9130 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=8.6780 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=9.0650 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1350 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=6.8660 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.1420 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.8210 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0300 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=8.7710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.3000 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0300 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=8.2700 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.3000 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.2590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.2590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.9240 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=11.0580 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=11.0740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=11.0720 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=11.0640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.7130 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0060 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=11.0840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.6450 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0170 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=10.1350 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=10.0010 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.3740 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.7180 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.9850 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.3100 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.6620 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.4240 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.2250 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.4120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.0660 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.7200 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=10.3990 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=9.5280 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=9.4560 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.9650 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=7.0840 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.2850 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.3550 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0280 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=10.0170 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.0170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0280 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=9.1790 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.1790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.2190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2190 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.2190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2190 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2170 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2170 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2190 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.6080 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=10.9830 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4850 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=10.9860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4850 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=10.9840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4850 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=10.9610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.8850 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=10.9640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.6740 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0090 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.9210 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.9570 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.0650 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.3980 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.7390 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.0600 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.6740 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.2120 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.0130 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.4850 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=10.9760 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.9010 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=10.3100 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=10.2820 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=9.3160 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.8550 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=7.1180 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.7720 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.3800 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0230 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=11.8790 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=11.8790 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0230 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=9.9170 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.5120 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.5610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.5610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.5610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.5610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.5610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.5610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=8.3300 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=11.1010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=11.1030 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=11.1010 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.4260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=11.0860 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=11.1470 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=11.0830 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.6060 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=10.2340 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=10.0410 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.2940 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.5520 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.9360 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.2890 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.5990 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.5140 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.3200 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.4260 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.1000 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=11.1410 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=11.0150 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=10.7930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=9.8380 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.5780 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=7.3500 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.4760 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.3680 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0340 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=2, FP=0, FN=1, TN=7030

- Node 3580: loss=12.5940 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=12.5940 | ✅ TP | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0340 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=10.7930 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=9.9950 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=8.2390 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2390 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=8.2390 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2390 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2300 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2300 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2300 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=8.2390 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.2870 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=11.2090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=11.2090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=11.2090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.2030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=11.2090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=11.4220 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0020 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=11.2090 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.3000 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0030 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=10.2230 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=9.8520 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=9.1250 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.1470 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.7410 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.0790 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.2990 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=9.2230 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.1470 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.2030 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=11.2090 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=11.4330 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=10.7770 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=10.9900 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=9.3980 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.2450 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=7.1140 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=10.4960 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=5.4050 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.0240 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=2, FP=0, FN=1, TN=7030

- Node 3580: loss=13.0450 | ✅ TP | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=13.0450 | ✅ TP | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0240 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=10.9900 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=10.2390 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

### orthrus_cicids_red

#### Epoch 0 ⭐ BEST

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.9100 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.9100 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.6550 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.8590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.8600 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.8590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2650 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.8600 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=10.9330 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2910 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.8710 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=8.2510 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.3210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=10.2800 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5020 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.6410 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=10.4390 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.8840 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=10.3880 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=8.2510 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.0880 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=10.7730 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=8.1930 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8560 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=10.7730 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=9.2840 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=6.6740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=8.0650 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.6790 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=5.7740 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=8.0310 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=2.9830 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3930 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=6.7840 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.7840 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3930 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=6.6740 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.6740 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 1

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8240 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8240 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.2320 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.3180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.3210 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.3200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.3200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.0480 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.2270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.3320 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.3040 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0980 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=8.7550 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=7.7890 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=7.7390 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=8.7410 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=8.4800 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=8.9060 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.3020 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.7370 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=8.9260 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.1880 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.3190 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.2110 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.2610 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.2760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.3880 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.5100 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.7580 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.1610 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.0950 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3740 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.5520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3740 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.2760 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2760 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 3

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0590 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0590 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.4420 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.0040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.0000 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.9990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.0020 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.5030 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0270 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.0200 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.3130 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0340 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1860 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5350 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.2140 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.1910 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.0360 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.2420 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.3070 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.9110 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.3810 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.6570 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.0000 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.3810 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.7680 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.5970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.6470 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.7730 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.2890 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.5110 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.6810 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.1040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.8950 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.8950 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.1040 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.5970 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5970 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 5

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3520 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.3520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.6540 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.9000 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.9000 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.8980 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.8990 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.5880 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.9040 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=6.9750 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0260 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1320 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.3660 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1380 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.3040 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.0970 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.4020 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=6.9700 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=7.9460 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.5300 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.3830 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8950 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.5300 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6390 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.4250 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.8980 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.8340 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.3140 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.1280 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.7830 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.6180 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6180 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.4250 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.4250 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 7

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=6.8690 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=6.8690 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.9750 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.2650 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.2620 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=9.2630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.2630 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.8760 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.2750 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.1180 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0190 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.4450 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.5240 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1970 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.7140 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.4650 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.6780 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.1180 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.0000 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.9120 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.2780 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.2610 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.9120 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.7870 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.5810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.3210 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.5920 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.0810 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.4150 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2580 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.6640 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.6640 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2610 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.5810 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.5810 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 9

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.0470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.0470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=6.2840 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=8.8490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=8.8510 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=8.8520 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=8.8490 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.6880 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=8.8530 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.1260 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0210 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.1570 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.3410 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.0980 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.5020 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.2520 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.4690 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.1250 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.0750 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.6710 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.2480 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=8.8500 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.6710 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6310 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.7470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.8610 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.1290 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.3230 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.5890 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=4.2600 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.2320 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.7470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.7470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.2320 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.7470 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.7470 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

#### Epoch 11

**5/dos_slowloris** — TP=0, FP=0, FN=2, TN=4776

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_slowhttptest** — TP=0, FP=0, FN=2, TN=4320

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_hulk** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/dos_goldeneye** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.1610 | ❌ FN | 192.168.10.50:0->192.168.10.50:0

**5/heartbleed** — TP=0, FP=0, FN=2, TN=4253

- Node 3580: loss=7.1610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4762: loss=7.5010 | ❌ FN | 192.168.10.51:0->192.168.10.51:0

**6/web_bruteforce** — TP=0, FP=0, FN=3, TN=7464

- Node 3580: loss=9.0610 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_xss** — TP=0, FP=0, FN=3, TN=7125

- Node 3580: loss=9.0660 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/web_sqli** — TP=0, FP=0, FN=3, TN=7024

- Node 3580: loss=9.0650 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step1** — TP=0, FP=0, FN=3, TN=7216

- Node 3580: loss=9.0450 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4763: loss=9.8500 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_cooldisk** — TP=0, FP=0, FN=3, TN=7060

- Node 3580: loss=9.0350 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4757: loss=7.4150 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 5568: loss=0.0130 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**6/infiltration_step2** — TP=0, FP=0, FN=13, TN=7190

- Node 4751: loss=9.2400 | ❌ FN | 192.168.10.12:0->192.168.10.12:0
- Node 4752: loss=8.4420 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=8.1360 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4754: loss=9.6920 | ❌ FN | 192.168.10.16:0->192.168.10.16:0
- Node 4755: loss=9.3670 | ❌ FN | 192.168.10.17:0->192.168.10.17:0
- Node 4756: loss=9.6970 | ❌ FN | 192.168.10.19:0->192.168.10.19:0
- Node 4757: loss=7.4150 | ❌ FN | 192.168.10.25:0->192.168.10.25:0
- Node 4759: loss=8.3650 | ❌ FN | 192.168.10.3:0->192.168.10.3:0
- Node 4760: loss=9.8910 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4761: loss=7.3170 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 4762: loss=9.0530 | ❌ FN | 192.168.10.51:0->192.168.10.51:0
- Node 4763: loss=9.8910 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=8.6410 | ❌ FN | 192.168.10.9:0->192.168.10.9:0

**7/botnet** — TP=0, FP=0, FN=7, TN=6927

- Node 3580: loss=7.9910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4752: loss=7.5890 | ❌ FN | 192.168.10.14:0->192.168.10.14:0
- Node 4753: loss=5.4720 | ❌ FN | 192.168.10.15:0->192.168.10.15:0
- Node 4760: loss=4.0630 | ❌ FN | 192.168.10.5:0->192.168.10.5:0
- Node 4763: loss=9.6600 | ❌ FN | 192.168.10.8:0->192.168.10.8:0
- Node 4764: loss=3.7870 | ❌ FN | 192.168.10.9:0->192.168.10.9:0
- Node 5568: loss=0.3570 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/portscan** — TP=0, FP=0, FN=3, TN=7030

- Node 3580: loss=7.9910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
- Node 5568: loss=0.3570 | ❌ FN | 205.174.165.73:0->205.174.165.73:0

**7/ddos_loit** — TP=0, FP=0, FN=2, TN=6141

- Node 3580: loss=7.9910 | ❌ FN | 172.16.0.1:0->172.16.0.1:0
- Node 4761: loss=7.9910 | ❌ FN | 192.168.10.50:0->192.168.10.50:0
