#!/usr/bin/env python3
"""
Map TP/FP/TN/FN from ARGUS, EULER, PIKACHU back to CIC-IDS 2017 attack types.

Two modes:
  1. --results_dir <path>  : Read predictions.pkl files saved by EULER/ARGUS
     runs, compute per-attack-type TP/FP/TN/FN using the model's actual
     classifications.  Requires nmap.pkl for node ID → IP mapping.

  2. (no --results_dir)    : Legacy mode — replay data aggregation to show the
     *ceiling* of detectable attacks per type (no model predictions needed).

Output:
    - Per-config tables: how many attack edges of each CIC-IDS 2017 type were
      detected (TP) vs missed (FN), plus false positive counts.
    - Saved to map_attacks_to_cic_types_results.json

Requires:  pandas, numpy, pickle (stdlib), tqdm, argparse
"""

import os
import sys
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sklearn import metrics as sk_metrics
    from sklearn.metrics import average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ============================================================================
# Paths (adjust if your layout differs)
# ============================================================================
GIDSREP = "/scratch/asawan15/GIDSREP"
DATA_PROC = os.path.join(GIDSREP, "1-DATA_PROCESSING", "cic_2017")
MODELS    = os.path.join(GIDSREP, "2-DETECTION_ASSESSMENT", "RQ2", "cic_2017")

# Original CIC-IDS 2017 merged CSV (has ' Label' and 'label' columns)
CIC_MASTER_CSV = os.path.join(DATA_PROC, "cic_2017_1.csv")

# EULER data dirs
EULER_DIR     = os.path.join(MODELS, "EULER", "cic2017")
EULER_RED_DIR = os.path.join(MODELS, "EULER", "cic2017_red")
EULER_NMAP    = os.path.join(EULER_DIR, "nmap.pkl")

# PIKACHU data
PIKACHU_CSV     = os.path.join(MODELS, "PIKACHU", "dataset", "cic_20.csv")
PIKACHU_RED_CSV = os.path.join(MODELS, "PIKACHU", "dataset", "cic_20_red.csv")

# ARGUS uses argus_flow/ for L_cic_flow
ARGUS_DIR = os.path.join(DATA_PROC, "argus_flow")
ARGUS_NMAP = os.path.join(ARGUS_DIR, "nmap.pkl") if os.path.exists(
    os.path.join(DATA_PROC, "argus_flow", "nmap.pkl")
) else EULER_NMAP

# Test split thresholds
EULER_ARGUS_TR_END = 29136   # ts >= this is test
PIKACHU_TRAIN_SNAP = 24      # snapshots 0-24 are Monday (train)
FILE_DELTA = 100_000

# EULER/ARGUS delta (seconds)
EULER_DELTA = 600   # --delta 10
ARGUS_DELTA_FLOW = 600
ARGUS_DELTA_NOFLOW = 900


# ============================================================================
# Step 1: Load master CIC-IDS 2017 with attack-type labels
# ============================================================================
def load_master_cic():
    """Load cic_2017_1.csv with (time_int, Source IP, Destination IP, Label, label)."""
    print("Loading master CIC-IDS 2017 CSV (this may take a moment)...")
    header = pd.read_csv(CIC_MASTER_CSV, nrows=0, index_col=0, encoding='latin-1')
    all_cols = list(header.columns)
    need = ['time_int', ' Source IP', ' Destination IP', ' Label', 'label']
    use_positions = [0] + [all_cols.index(c) + 1 for c in need]
    df = pd.read_csv(CIC_MASTER_CSV, index_col=0, usecols=use_positions,
                      encoding='latin-1', low_memory=False)
    cols = [c for c in need if c in df.columns]
    df = df[cols].copy()
    df.rename(columns={' Source IP': 'src_ip', ' Destination IP': 'dst_ip',
                       ' Label': 'attack_type'}, inplace=True)
    df['attack_type'] = df['attack_type'].str.strip()
    print(f"  Loaded {len(df)} flows, {df['label'].sum()} attacks, "
          f"{df['attack_type'].nunique()} attack types")
    return df


# ============================================================================
# Step 2: Build reverse node map  (integer ID → IP string)
# ============================================================================
def load_nmap(path):
    """nmap.pkl is a list where index = node ID, value = IP string."""
    nmap = pickle.load(open(path, 'rb'))
    if isinstance(nmap, list):
        return nmap
    elif isinstance(nmap, dict):
        inv = [''] * (max(nmap.values()) + 1)
        for ip, idx in nmap.items():
            inv[idx] = ip
        return inv
    raise TypeError(f"Unexpected nmap type: {type(nmap)}")


# ============================================================================
# Step 3: Replay EULER/ARGUS aggregation for test edges (legacy mode)
# ============================================================================
def replay_euler_argus_aggregation(data_dir, nmap, delta, tr_end, total_end=374762):
    """
    Replays the load_partial_lanl aggregation logic to reconstruct which
    (src_ip, dst_ip) edges exist in the test set and their labels.
    """
    all_lines = []
    for fstart in range(0, total_end + FILE_DELTA, FILE_DELTA):
        fpath = os.path.join(data_dir, f"{fstart}.txt")
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                ts = int(parts[0])
                if ts < tr_end:
                    continue
                src = int(parts[1])
                dst = int(parts[2])
                lbl = int(parts[-1])
                all_lines.append((ts, src, dst, lbl))

    print(f"  Read {len(all_lines)} test flows from {data_dir}")

    edges_out = []
    all_lines.sort(key=lambda x: x[0])

    snap_start = tr_end
    snap_edges = {}

    for ts, src, dst, lbl in tqdm(all_lines, desc="  Aggregating"):
        while ts >= snap_start + delta:
            if snap_edges:
                for (s, d), info in snap_edges.items():
                    edges_out.append({
                        'src_id': s, 'dst_id': d,
                        'src_ip': nmap[s], 'dst_ip': nmap[d],
                        'snapshot_start': snap_start,
                        'label': info['label'],
                        'flow_count': info['count'],
                        'flow_timestamps': info['timestamps'],
                    })
            snap_edges = {}
            snap_start += delta

        if src == dst:
            continue

        et = (src, dst)
        if et in snap_edges:
            snap_edges[et]['label'] = max(snap_edges[et]['label'], lbl)
            snap_edges[et]['count'] += 1
            snap_edges[et]['timestamps'].append(ts)
        else:
            snap_edges[et] = {'label': lbl, 'count': 1, 'timestamps': [ts]}

    if snap_edges:
        for (s, d), info in snap_edges.items():
            edges_out.append({
                'src_id': s, 'dst_id': d,
                'src_ip': nmap[s], 'dst_ip': nmap[d],
                'snapshot_start': snap_start,
                'label': info['label'],
                'flow_count': info['count'],
                'flow_timestamps': info['timestamps'],
            })

    return edges_out


# ============================================================================
# Step 4: Join aggregated edges to CIC-IDS attack types
# ============================================================================
def join_edges_to_attack_types(edges, master_df):
    """
    For each aggregated edge with label=1, look up which attack types
    the constituent flows belong to.
    """
    print("  Building attack flow lookup...")
    atk = master_df[master_df['label'] == 1]
    atk_lookup = defaultdict(list)
    for _, row in tqdm(atk.iterrows(), total=len(atk), desc="  Indexing attacks"):
        key = (row['src_ip'], row['dst_ip'])
        atk_lookup[key].append((row['time_int'], row['attack_type']))

    edge_types = []
    for e in tqdm(edges, desc="  Joining"):
        if e['label'] == 0:
            edge_types.append({**e, 'attack_types': {}, 'primary_type': 'BENIGN'})
            continue

        key = (e['src_ip'], e['dst_ip'])
        ts_set = set(e['flow_timestamps'])
        snap_start = e['snapshot_start']

        types_count = defaultdict(int)
        if key in atk_lookup:
            for t, atype in atk_lookup[key]:
                if snap_start <= t < snap_start + EULER_DELTA:
                    types_count[atype] += 1

        if not types_count and key in atk_lookup:
            for t, atype in atk_lookup[key]:
                if t in ts_set:
                    types_count[atype] += 1

        primary = max(types_count, key=types_count.get) if types_count else 'UNKNOWN'
        edge_types.append({
            'src_ip': e['src_ip'], 'dst_ip': e['dst_ip'],
            'snapshot_start': e['snapshot_start'],
            'label': e['label'],
            'flow_count': e['flow_count'],
            'attack_types': dict(types_count),
            'primary_type': primary,
        })

    return edge_types


# ============================================================================
# Step 5: PIKACHU flow-level join
# ============================================================================
def pikachu_flow_attack_types(csv_path, master_df, train_snap=PIKACHU_TRAIN_SNAP):
    print(f"  Loading PIKACHU dataset: {csv_path}")
    pik = pd.read_csv(csv_path, index_col=0)
    pik_test = pik[pik['snapshot'] > train_snap].copy()
    print(f"  Test flows: {len(pik_test)}, attacks: {pik_test['label'].sum()}")

    atk_master = master_df[master_df['label'] == 1][['time_int', 'src_ip', 'dst_ip', 'attack_type']]
    atk_lookup = {}
    for _, row in tqdm(atk_master.iterrows(), total=len(atk_master), desc="  Indexing"):
        key = (row['time_int'], row['src_ip'], row['dst_ip'])
        atk_lookup[key] = row['attack_type']

    type_counts = defaultdict(int)
    total_attacks = 0
    for _, row in tqdm(pik_test.iterrows(), total=len(pik_test), desc="  Joining"):
        if row['label'] == 0:
            continue
        total_attacks += 1
        key = (row['timestamp'], row['src_computer'], row['dst_computer'])
        atype = atk_lookup.get(key, None)
        if atype is None:
            key2 = (row['timestamp'], row['dst_computer'], row['src_computer'])
            atype = atk_lookup.get(key2, 'UNKNOWN')
        type_counts[atype] += 1

    return dict(type_counts), total_attacks, len(pik_test)


# ============================================================================
# Step 6: Summary tables
# ============================================================================
def summarize_euler_argus(edge_types, model_name, config_name):
    attack_edges = [e for e in edge_types if e['label'] == 1]
    benign_edges = [e for e in edge_types if e['label'] == 0]

    type_counts = defaultdict(int)
    type_flow_counts = defaultdict(int)
    for e in attack_edges:
        type_counts[e['primary_type']] += 1
        type_flow_counts[e['primary_type']] += e['flow_count']

    print(f"\n{'='*70}")
    print(f"  {model_name} — {config_name}")
    print(f"{'='*70}")
    print(f"  Total test edges:   {len(edge_types)}")
    print(f"  Attack edges:       {len(attack_edges)} (aggregated)")
    print(f"  Benign edges:       {len(benign_edges)} (aggregated)")
    print(f"\n  Attack edges by CIC-IDS 2017 type:")
    print(f"  {'Attack Type':<35} {'Agg Edges':>10} {'Raw Flows':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    for atype in sorted(type_counts, key=type_counts.get, reverse=True):
        print(f"  {atype:<35} {type_counts[atype]:>10} {type_flow_counts[atype]:>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    print(f"  {'TOTAL':<35} {len(attack_edges):>10} {sum(type_flow_counts.values()):>10}")

    return {
        'model': model_name,
        'config': config_name,
        'total_test_edges': len(edge_types),
        'attack_edges': len(attack_edges),
        'benign_edges': len(benign_edges),
        'by_type': dict(type_counts),
        'by_type_flows': dict(type_flow_counts),
    }


def summarize_pikachu(type_counts, total_attacks, total_test, config_name):
    print(f"\n{'='*70}")
    print(f"  PIKACHU — {config_name}")
    print(f"{'='*70}")
    print(f"  Total test flows:   {total_test}")
    print(f"  Attack flows:       {total_attacks}")
    print(f"  Benign flows:       {total_test - total_attacks}")
    print(f"\n  Attack flows by CIC-IDS 2017 type:")
    print(f"  {'Attack Type':<35} {'Flows':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for atype in sorted(type_counts, key=type_counts.get, reverse=True):
        print(f"  {atype:<35} {type_counts[atype]:>10}")
    print(f"  {'-'*35} {'-'*10}")
    print(f"  {'TOTAL':<35} {sum(type_counts.values()):>10}")

    return {
        'model': 'PIKACHU',
        'config': config_name,
        'total_test_flows': total_test,
        'attack_flows': total_attacks,
        'benign_flows': total_test - total_attacks,
        'by_type': dict(type_counts),
    }


# ============================================================================
# NEW: Process predictions.pkl to get per-attack-type TP/FP/TN/FN
# ============================================================================
def build_attack_type_lookup(master_df):
    """Build (src_ip, dst_ip) → set of attack types for attack flows."""
    atk = master_df[master_df['label'] == 1]
    # (src_ip, dst_ip) → set(attack_types)
    lookup = defaultdict(set)
    for _, row in tqdm(atk.iterrows(), total=len(atk), desc="  Building attack-type lookup"):
        lookup[(row['src_ip'], row['dst_ip'])].add(row['attack_type'])
    return lookup


def analyze_predictions(pred_path, nmap, atk_type_lookup, model_name, config_name):
    """
    Read predictions.pkl, classify edges using the saved cutoff, and
    compute per-attack-type TP/FP/TN/FN.

    predictions.pkl contains:
        scores:  list of arrays (one per snapshot)
        labels:  list of arrays (one per snapshot)
        eis:     list of (2, E_i) arrays (one per snapshot)
        cutoff:  float
    """
    print(f"\n  Loading predictions: {pred_path}")
    pred = pickle.load(open(pred_path, 'rb'))

    scores_all = np.concatenate(pred['scores'], axis=0)
    labels_all = np.concatenate(pred['labels'], axis=0).clip(max=1)
    eis_all = np.concatenate(pred['eis'], axis=1)  # (2, total_edges)
    cutoff = pred['cutoff']

    total_edges = len(scores_all)
    classified = np.zeros(total_edges)
    classified[scores_all <= cutoff] = 1  # low score → anomalous

    # Map edge indices to IPs and then to attack types
    src_ids = eis_all[0]
    dst_ids = eis_all[1]

    # Per-attack-type counters
    type_tp = defaultdict(int)  # attack edge, correctly detected
    type_fn = defaultdict(int)  # attack edge, missed
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0

    for i in range(total_edges):
        true_label = int(labels_all[i])
        pred_label = int(classified[i])

        if true_label == 1:
            src_ip = nmap[int(src_ids[i])]
            dst_ip = nmap[int(dst_ids[i])]
            types = atk_type_lookup.get((src_ip, dst_ip), {'UNKNOWN'})

            if pred_label == 1:
                total_tp += 1
                for t in types:
                    type_tp[t] += 1
            else:
                total_fn += 1
                for t in types:
                    type_fn[t] += 1
        else:
            if pred_label == 1:
                total_fp += 1
            else:
                total_tn += 1

    # Compute comprehensive metrics
    tpr = total_tp / max(total_tp + total_fn, 1)
    fpr = total_fp / max(total_fp + total_tn, 1)
    precision = total_tp / max(total_tp + total_fp, 1)
    accuracy = (total_tp + total_tn) / max(total_edges, 1)

    auc_val = None
    ap_val = None
    if HAS_SKLEARN and len(np.unique(labels_all)) > 1:
        try:
            auc_val = float(sk_metrics.roc_auc_score(labels_all, 1 - scores_all))
        except Exception:
            auc_val = None
        try:
            ap_val = float(average_precision_score(labels_all, 1 - scores_all))
        except Exception:
            ap_val = None

    # Print summary
    print(f"\n{'='*70}")
    print(f"  {model_name} — {config_name} (from predictions.pkl)")
    print(f"{'='*70}")
    print(f"  Total test edges: {total_edges}")
    print(f"  Cutoff: {cutoff:.4f}")
    print(f"  TP={total_tp}  FP={total_fp}  TN={total_tn}  FN={total_fn}")
    print(f"  TPR={tpr:.4f}  FPR={fpr:.4f}  Precision={precision:.4f}  Accuracy={accuracy:.4f}")
    if auc_val is not None:
        print(f"  AUC={auc_val:.4f}  AP={ap_val:.4f}")

    all_types = sorted(set(list(type_tp.keys()) + list(type_fn.keys())))
    print(f"\n  {'Attack Type':<35} {'TP':>6} {'FN':>6} {'Total':>6} {'Recall':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for atype in all_types:
        tp = type_tp.get(atype, 0)
        fn = type_fn.get(atype, 0)
        tot = tp + fn
        recall = tp / max(tot, 1)
        print(f"  {atype:<35} {tp:>6} {fn:>6} {tot:>6} {recall:>8.4f}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    print(f"  {'TOTAL':<35} {total_tp:>6} {total_fn:>6} {total_tp+total_fn:>6} {tpr:>8.4f}")

    return {
        'model': model_name,
        'config': config_name,
        'total_edges': total_edges,
        'cutoff': cutoff,
        'TP': total_tp, 'FP': total_fp, 'TN': total_tn, 'FN': total_fn,
        'TPR': tpr, 'FPR': fpr, 'Precision': precision, 'Accuracy': accuracy,
        'AUC': auc_val, 'AP': ap_val,
        'attacks_detected': total_tp,
        'tp_by_type': dict(type_tp),
        'fn_by_type': dict(type_fn),
    }


def analyze_pikachu_predictions(pred_path, atk_type_lookup_ip, model_name, config_name):
    """
    Read PIKACHU predictions.pkl which has flow-level data with IPs directly.

    PIKACHU pickle format:
        src_ips:   list of IP strings
        dst_ips:   list of IP strings
        scores:    np.array of anomaly scores
        labels:    np.array of true labels (0/1)
        preds:     np.array of predicted labels (0/1)
        cutoff:    float
        TP, FP, TN, FN, AUC, AP, TPR, FPR, Precision, Accuracy: precomputed
    """
    print(f"\n  Loading PIKACHU predictions: {pred_path}")
    pred = pickle.load(open(pred_path, 'rb'))

    src_ips = pred['src_ips']
    dst_ips = pred['dst_ips']
    labels = np.array(pred['labels'], dtype=int)
    preds = np.array(pred['preds'], dtype=int)
    scores = np.array(pred['scores'])
    cutoff = pred['cutoff']

    total_flows = len(labels)

    # Use precomputed metrics if available, else compute
    total_tp = int(pred.get('TP', 0))
    total_fp = int(pred.get('FP', 0))
    total_tn = int(pred.get('TN', 0))
    total_fn = int(pred.get('FN', 0))
    auc_val = pred.get('AUC', None)
    ap_val = pred.get('AP', None)
    tpr = pred.get('TPR', total_tp / max(total_tp + total_fn, 1))
    fpr_val = pred.get('FPR', total_fp / max(total_fp + total_tn, 1))
    precision = pred.get('Precision', total_tp / max(total_tp + total_fp, 1))
    accuracy = pred.get('Accuracy', (total_tp + total_tn) / max(total_flows, 1))

    # Per-attack-type breakdown
    type_tp = defaultdict(int)
    type_fn = defaultdict(int)

    for i in range(total_flows):
        if int(labels[i]) == 1:
            key = (src_ips[i], dst_ips[i])
            types = atk_type_lookup_ip.get(key, None)
            if types is None:
                types = atk_type_lookup_ip.get((dst_ips[i], src_ips[i]), {'UNKNOWN'})
            if int(preds[i]) == 1:
                for t in types:
                    type_tp[t] += 1
            else:
                for t in types:
                    type_fn[t] += 1

    # Print summary
    print(f"\n{'='*70}")
    print(f"  {model_name} — {config_name} (from predictions.pkl)")
    print(f"{'='*70}")
    print(f"  Total test flows: {total_flows}")
    print(f"  Cutoff: {cutoff:.6f}")
    print(f"  TP={total_tp}  FP={total_fp}  TN={total_tn}  FN={total_fn}")
    print(f"  TPR={tpr:.4f}  FPR={fpr_val:.4f}  Precision={precision:.4f}  Accuracy={accuracy:.4f}")
    if auc_val is not None:
        print(f"  AUC={auc_val:.4f}  AP={ap_val:.4f}")

    all_types = sorted(set(list(type_tp.keys()) + list(type_fn.keys())))
    print(f"\n  {'Attack Type':<35} {'TP':>6} {'FN':>6} {'Total':>6} {'Recall':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for atype in all_types:
        tp = type_tp.get(atype, 0)
        fn = type_fn.get(atype, 0)
        tot = tp + fn
        recall = tp / max(tot, 1)
        print(f"  {atype:<35} {tp:>6} {fn:>6} {tot:>6} {recall:>8.4f}")
    print(f"  {'-'*35} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    print(f"  {'TOTAL':<35} {sum(type_tp.values()):>6} {sum(type_fn.values()):>6} "
          f"{sum(type_tp.values())+sum(type_fn.values()):>6} {tpr:>8.4f}")

    return {
        'model': model_name,
        'config': config_name,
        'total_flows': total_flows,
        'cutoff': cutoff,
        'TP': total_tp, 'FP': total_fp, 'TN': total_tn, 'FN': total_fn,
        'TPR': float(tpr), 'FPR': float(fpr_val),
        'Precision': float(precision), 'Accuracy': float(accuracy),
        'AUC': float(auc_val) if auc_val is not None else None,
        'AP': float(ap_val) if ap_val is not None else None,
        'attacks_detected': total_tp,
        'tp_by_type': dict(type_tp),
        'fn_by_type': dict(type_fn),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Map attacks to CIC-IDS 2017 types')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Path to results dir with predictions.pkl files '
                             '(e.g., GIDSREP/results/20260416_123456/). '
                             'If not set, runs legacy data-ceiling mode.')
    args = parser.parse_args()

    results = {}

    # --- Load master CSV ---
    master = load_master_cic()
    print(f"\nAttack types in CIC-IDS 2017:")
    for atype, cnt in master[master['label']==1]['attack_type'].value_counts().items():
        print(f"  {atype}: {cnt}")

    # ==================================================================
    # MODE 1: Analyze predictions from model runs
    # ==================================================================
    if args.results_dir:
        results_dir = args.results_dir
        print(f"\n{'#'*70}")
        print(f"# Analyzing predictions from: {results_dir}")
        print(f"{'#'*70}")

        # Build attack-type lookup (src_ip, dst_ip) → set of types
        atk_type_lookup = build_attack_type_lookup(master)

        # --- Load nmap ---
        nmap = load_nmap(EULER_NMAP)
        print(f"Node map: {len(nmap)} IPs")

        # EULER/ARGUS configs (edge-index based predictions)
        ea_configs = [
            ('EULER',  'baseline',   'EULER/baseline/predictions.pkl'),
            ('EULER',  'kde_decode', 'EULER/kde_decode/predictions.pkl'),
            ('EULER',  'reduced',    'EULER/reduced/predictions.pkl'),
            ('ARGUS',  'baseline',   'ARGUS/baseline/predictions.pkl'),
            ('ARGUS',  'kde',        'ARGUS/kde/predictions.pkl'),
            ('ARGUS',  'reduced',    'ARGUS/reduced/predictions.pkl'),
        ]

        for model_name, config_name, rel_path in ea_configs:
            pred_path = os.path.join(results_dir, rel_path)
            if not os.path.exists(pred_path):
                print(f"\n  SKIP: {pred_path} not found")
                continue
            results[f"{model_name}_{config_name}"] = analyze_predictions(
                pred_path, nmap, atk_type_lookup, model_name, config_name
            )

        # PIKACHU configs (flow-level, IPs directly in pickle)
        pik_configs = [
            ('PIKACHU', 'kde',     'PIKACHU/kde/predictions.pkl'),
            ('PIKACHU', 'reduced', 'PIKACHU/reduced/predictions.pkl'),
        ]

        for model_name, config_name, rel_path in pik_configs:
            pred_path = os.path.join(results_dir, rel_path)
            if not os.path.exists(pred_path):
                print(f"\n  SKIP: {pred_path} not found")
                continue
            results[f"{model_name}_{config_name}"] = analyze_pikachu_predictions(
                pred_path, atk_type_lookup, model_name, config_name
            )

        # ============================================================
        # COMPREHENSIVE SUMMARY TABLE: all 8 configs
        # ============================================================
        cfg_names = list(results.keys())
        if cfg_names:
            print(f"\n\n{'='*120}")
            print("COMPREHENSIVE SUMMARY — ALL CONFIGS")
            print(f"{'='*120}")
            print(f"  {'Config':<25} {'TP':>6} {'FP':>8} {'TN':>8} {'FN':>6} "
                  f"{'AUC':>8} {'AP':>8} {'TPR':>8} {'FPR':>8} "
                  f"{'Prec':>8} {'Acc':>8} {'#Atk Det':>9}")
            print(f"  {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*6} "
                  f"{'-'*8} {'-'*8} {'-'*8} {'-'*8} "
                  f"{'-'*8} {'-'*8} {'-'*9}")
            for c in cfg_names:
                r = results[c]
                auc_s = f"{r['AUC']:.4f}" if r.get('AUC') is not None else '   N/A'
                ap_s  = f"{r['AP']:.4f}" if r.get('AP') is not None else '   N/A'
                print(f"  {c:<25} {r['TP']:>6} {r['FP']:>8} {r['TN']:>8} {r['FN']:>6} "
                      f"{auc_s:>8} {ap_s:>8} {r['TPR']:>8.4f} {r['FPR']:>8.4f} "
                      f"{r['Precision']:>8.4f} {r['Accuracy']:>8.4f} {r.get('attacks_detected', r['TP']):>9}")

            # Per-attack-type TP cross-config table
            all_types = set()
            for r in results.values():
                all_types.update(r.get('tp_by_type', {}).keys())
                all_types.update(r.get('fn_by_type', {}).keys())
            all_types.discard('UNKNOWN')
            all_types = sorted(all_types)

            print(f"\n{'='*120}")
            print("PER-ATTACK-TYPE TP / TOTAL (across configs)")
            print(f"{'='*120}")
            header = f"{'Attack Type':<30}"
            for c in cfg_names:
                header += f" {c:>16}"
            print(header)
            print("-" * len(header))

            for atype in all_types:
                row = f"{atype:<30}"
                for c in cfg_names:
                    tp = results[c].get('tp_by_type', {}).get(atype, 0)
                    fn = results[c].get('fn_by_type', {}).get(atype, 0)
                    tot = tp + fn
                    row += f" {tp:>6}/{tot:<8}"
                print(row)

            print("-" * len(header))
            row = f"{'TOTAL':<30}"
            for c in cfg_names:
                row += f" {results[c]['TP']:>6}/{results[c]['TP']+results[c]['FN']:<8}"
            print(row)

    # ==================================================================
    # MODE 2: Legacy data-ceiling mode (no predictions)
    # ==================================================================
    else:
        print("\nRunning in legacy mode (data ceiling, no model predictions)")

        nmap = load_nmap(EULER_NMAP)
        print(f"\nNode map: {len(nmap)} IPs")

        # EULER Baseline & KDE
        print("\n" + "="*70)
        print("REPLAYING EULER AGGREGATION (euler/, delta=600s)")
        print("="*70)
        euler_edges = replay_euler_argus_aggregation(
            EULER_DIR, nmap, EULER_DELTA, EULER_ARGUS_TR_END
        )
        euler_typed = join_edges_to_attack_types(euler_edges, master)
        results['euler_baseline'] = summarize_euler_argus(
            euler_typed, 'EULER', 'Baseline & KDE (euler/, delta=600s)'
        )

        # EULER Reduced
        print("\n" + "="*70)
        print("REPLAYING EULER REDUCED AGGREGATION (euler_red/, delta=600s)")
        print("="*70)
        euler_red_edges = replay_euler_argus_aggregation(
            EULER_RED_DIR, nmap, EULER_DELTA, EULER_ARGUS_TR_END
        )
        euler_red_typed = join_edges_to_attack_types(euler_red_edges, master)
        results['euler_reduced'] = summarize_euler_argus(
            euler_red_typed, 'EULER', 'Reduced (euler_red/, delta=600s)'
        )

        # ARGUS L_cic_flow
        argus_data = ARGUS_DIR if os.path.exists(os.path.join(ARGUS_DIR, '0.txt')) else EULER_DIR
        argus_nmap_path = os.path.join(argus_data, 'nmap.pkl')
        argus_nmap = load_nmap(argus_nmap_path) if os.path.exists(argus_nmap_path) else nmap

        print("\n" + "="*70)
        print(f"REPLAYING ARGUS L_cic_flow AGGREGATION ({argus_data}, delta={ARGUS_DELTA_FLOW}s)")
        print("="*70)
        argus_edges = replay_euler_argus_aggregation(
            argus_data, argus_nmap, ARGUS_DELTA_FLOW, EULER_ARGUS_TR_END
        )
        argus_typed = join_edges_to_attack_types(argus_edges, master)
        results['argus_flow'] = summarize_euler_argus(
            argus_typed, 'ARGUS', 'L_cic_flow (delta=600s)'
        )

        # ARGUS O_cic
        print("\n" + "="*70)
        print(f"REPLAYING ARGUS O_cic AGGREGATION (euler/, delta={ARGUS_DELTA_NOFLOW}s)")
        print("="*70)
        argus_ocic_edges = replay_euler_argus_aggregation(
            EULER_DIR, nmap, ARGUS_DELTA_NOFLOW, EULER_ARGUS_TR_END
        )
        argus_ocic_typed = join_edges_to_attack_types(argus_ocic_edges, master)
        results['argus_ocic'] = summarize_euler_argus(
            argus_ocic_typed, 'ARGUS', 'O_cic (euler/, delta=900s)'
        )

        # PIKACHU Baseline
        print("\n" + "="*70)
        print("PIKACHU BASELINE & KDE (cic_20.csv)")
        print("="*70)
        pik_types, pik_atk, pik_total = pikachu_flow_attack_types(
            PIKACHU_CSV, master
        )
        results['pikachu_baseline'] = summarize_pikachu(
            pik_types, pik_atk, pik_total, 'Baseline & KDE (cic_20.csv)'
        )

        # PIKACHU Reduced
        if os.path.exists(PIKACHU_RED_CSV):
            print("\n" + "="*70)
            print("PIKACHU REDUCED (cic_20_red.csv)")
            print("="*70)
            pik_red_types, pik_red_atk, pik_red_total = pikachu_flow_attack_types(
                PIKACHU_RED_CSV, master
            )
            results['pikachu_reduced'] = summarize_pikachu(
                pik_red_types, pik_red_atk, pik_red_total, 'Reduced (cic_20_red.csv)'
            )

        # Cross-model summary
        print("\n" + "="*70)
        print("CROSS-MODEL SUMMARY: Test Set Attack Composition")
        print("="*70)

        all_types = set()
        for r in results.values():
            all_types.update(r.get('by_type', {}).keys())
        all_types.discard('BENIGN')
        all_types.discard('UNKNOWN')
        all_types = sorted(all_types)

        configs = list(results.keys())
        header = f"{'Attack Type':<30}"
        for c in configs:
            header += f" {c:>18}"
        print(header)
        print("-" * len(header))

        for atype in all_types:
            row = f"{atype:<30}"
            for c in configs:
                val = results[c].get('by_type', {}).get(atype, 0)
                row += f" {val:>18}"
            print(row)

        row = f"{'TOTAL ATTACK':<30}"
        for c in configs:
            if 'attack_edges' in results[c]:
                val = results[c]['attack_edges']
            else:
                val = results[c]['attack_flows']
            row += f" {val:>18}"
        print("-" * len(header))
        print(row)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'map_attacks_to_cic_types_results.json')
    json.dump(results, open(out_path, 'w'), indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == '__main__':
    main()
