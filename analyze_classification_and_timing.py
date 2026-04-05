#!/usr/bin/env python3
"""
Unified analysis script for PIDSMaker CIC-IDS-2017 experiments.

Produces:
1. Node classification diffs between enhanced configs and baselines
   - From .pth files: full per-node y_hat diffs (last-evaluated attack only)
   - From logs: per-attack malicious node TP/FN diffs
2. Average training and inference timing per KDE tainted batch
3. Detailed notes.md documenting all findings

IMPORTANT CAVEATS:
- result_model_epoch_X.pth files are overwritten per-attack during evaluation.
  They only contain node classifications for the LAST evaluated attack (graph_7_ddos_loit).
  For per-attack malicious node classification, we rely on the log's TP/FN lines.
- orthrus_cicids baseline has NO inference timing JSON files.
"""

import csv
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import torch

# ============================================================
# Configuration
# ============================================================

PIDSMAKER_DIR = os.path.dirname(os.path.abspath(__file__))
WANDB_DIR = os.path.join(PIDSMAKER_DIR, "wandb")

# The 14 attacks in evaluation order
ATTACKS = [
    "graph_5_dos_slowloris",
    "graph_5_dos_slowhttptest",
    "graph_5_dos_hulk",
    "graph_5_dos_goldeneye",
    "graph_5_heartbleed",
    "graph_6_web_bruteforce",
    "graph_6_web_xss",
    "graph_6_web_sqli",
    "graph_6_infiltration_step1",
    "graph_6_infiltration_cooldisk",
    "graph_6_infiltration_step2",
    "graph_7_botnet",
    "graph_7_portscan",
    "graph_7_ddos_loit",
]

# Last evaluated attack (the one whose results are in .pth files)
LAST_EVALUATED_ATTACK = "graph_7_ddos_loit"

# Evaluated epochs
EVAL_EPOCHS = [0, 1, 3, 5, 7, 9, 11]
LAST_EPOCH = 11

# wandb run mapping
RUNS = [
    ("run-20260403_173823-l88hmsi6", "kairos_cicids_red"),
    ("run-20260403_172408-arq94flm", "orthrus_cicids_red"),
    ("run-20260403_081130-gefm583r", "kairos_cicids_kde_diff"),
    ("run-20260403_071729-94zpg3qc", "orthrus_cicids_kde_diff"),
    ("run-20260403_062042-o693lwyf", "kairos_cicids_kde_ts"),
    ("run-20260403_052853-xn9gc9ym", "orthrus_cicids_kde_ts"),
    ("run-20260403_022135-k1f4y9oo", "kairos_cicids"),
    ("run-20260403_014359-w7n1uqtr", "orthrus_cicids"),
]

# Config hashes and artifact directory mapping
KAIROS_HASH = "d894a741f4e1947819404f0420d422e88748a58298764bf4d56f98dfd13b38a2"
ORTHRUS_HASH = "aa3d7440e620ea1d6a3b92e5faffac7c154bc866affaaba6eaa6b14e7d6f8144"

CONFIG_META = {
    "kairos_cicids": {
        "hash": KAIROS_HASH,
        "artifact_dir": "artifacts_cicids",
        "model_type": "kairos",
        "is_baseline": True,
        "baseline_of": None,
    },
    "orthrus_cicids": {
        "hash": ORTHRUS_HASH,
        "artifact_dir": "artifacts_cicids",
        "model_type": "orthrus",
        "is_baseline": True,
        "baseline_of": None,
    },
    "kairos_cicids_kde_ts": {
        "hash": KAIROS_HASH,
        "artifact_dir": "artifacts_cicids_kde_ts_reduced",
        "model_type": "kairos",
        "is_baseline": False,
        "baseline_of": "kairos_cicids",
    },
    "orthrus_cicids_kde_ts": {
        "hash": ORTHRUS_HASH,
        "artifact_dir": "artifacts_cicids_kde_ts_reduced",
        "model_type": "orthrus",
        "is_baseline": False,
        "baseline_of": "orthrus_cicids",
    },
    "kairos_cicids_kde_diff": {
        "hash": KAIROS_HASH,
        "artifact_dir": "artifacts_cicids_kde_ts_diff_reduced",
        "model_type": "kairos",
        "is_baseline": False,
        "baseline_of": "kairos_cicids",
    },
    "orthrus_cicids_kde_diff": {
        "hash": ORTHRUS_HASH,
        "artifact_dir": "artifacts_cicids_kde_ts_diff_reduced",
        "model_type": "orthrus",
        "is_baseline": False,
        "baseline_of": "orthrus_cicids",
    },
    "kairos_cicids_red": {
        "hash": KAIROS_HASH,
        "artifact_dir": "artifacts_cicids_red_reduced",
        "model_type": "kairos",
        "is_baseline": False,
        "baseline_of": "kairos_cicids",
    },
    "orthrus_cicids_red": {
        "hash": ORTHRUS_HASH,
        "artifact_dir": "artifacts_cicids_red_reduced",
        "model_type": "orthrus",
        "is_baseline": False,
        "baseline_of": "orthrus_cicids",
    },
}

# Comparison pairs: (enhanced_config, baseline_config)
COMPARISON_PAIRS = [
    ("kairos_cicids_kde_ts", "kairos_cicids"),
    ("orthrus_cicids_kde_ts", "orthrus_cicids"),
    ("kairos_cicids_kde_diff", "kairos_cicids"),
    ("orthrus_cicids_kde_diff", "orthrus_cicids"),
    ("kairos_cicids_red", "kairos_cicids"),
    ("orthrus_cicids_red", "orthrus_cicids"),
]

# Best epoch per config (from extract_best_epoch_results.py output)
# Will be computed dynamically
BEST_EPOCHS = {}


def get_artifact_base(config_name):
    """Get the base path for a config's evaluation artifacts."""
    meta = CONFIG_META[config_name]
    return os.path.join(
        PIDSMAKER_DIR,
        meta["artifact_dir"],
        "evaluation", "evaluation",
        meta["hash"],
        "CIC_IDS_2017_PER_ATTACK",
    )


def get_result_pth_path(config_name, epoch):
    """Get the path to result_model_epoch_X.pth for a config."""
    base = get_artifact_base(config_name)
    return os.path.join(base, "precision_recall_dir", f"result_model_epoch_{epoch}.pth")


def get_training_tainted_json_path(config_name):
    """Get the path to training tainted_batches JSON."""
    base = get_artifact_base(config_name)
    return os.path.join(base, "batch_timing", "tainted_batches_CIC_IDS_2017_PER_ATTACK.json")


def get_inference_tainted_json_path(config_name, epoch):
    """Get the path to inference tainted_batches JSON for a given epoch."""
    base = get_artifact_base(config_name)
    return os.path.join(base, "batch_timing",
                        f"inference_tainted_batches_CIC_IDS_2017_PER_ATTACK_epoch{epoch}.json")


# ============================================================
# Part 1: Parse logs for per-attack malicious node classification
# ============================================================

def parse_log_malicious_nodes(log_path):
    """
    Parse wandb output.log to extract per-attack per-epoch malicious node classifications.

    Returns:
        dict: {epoch: {attack: {node_id: {"is_tp": bool, "loss": float, "info": str}}}}
    """
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = defaultdict(lambda: defaultdict(dict))

    eval_header_re = re.compile(
        r"\[@model_epoch_(\d+)\] - Test Evaluation \((graph_\w+), TWs=(\d+)\)"
    )
    # Match both TP ✅ and FN ❌
    mal_node_re = re.compile(
        r"-> Malicious node (\d+)\s+: loss=([\d.]+) \| is TP: (✅|❌)\s*(.*)"
    )

    current_epoch = None
    current_attack = None

    for line in lines:
        m = eval_header_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            current_attack = m.group(2)
            continue

        if current_epoch is not None and current_attack is not None:
            m2 = mal_node_re.search(line)
            if m2:
                node_id = int(m2.group(1).strip())
                loss = float(m2.group(2))
                is_tp = m2.group(3) == "✅"
                info = m2.group(4).strip()
                result[current_epoch][current_attack][node_id] = {
                    "is_tp": is_tp,
                    "loss": loss,
                    "info": info,
                }

    return dict(result)


def parse_log_stats(log_path):
    """
    Parse wandb output.log to extract per-attack per-epoch stats (TP/FP/FN/TN).

    Returns:
        dict: {epoch: {attack: {metric: value}}}
    """
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result = defaultdict(lambda: defaultdict(dict))

    epoch_re = re.compile(r"\[@model_epoch_(\d+)\] - Stats")
    metric_re = re.compile(r"- (graph_\w+)/(tp|fp|tn|fn|precision|recall|accuracy):\s+([\d.eE+-]+)")

    current_epoch = None

    for line in lines:
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            continue

        if current_epoch is not None:
            m2 = metric_re.search(line)
            if m2:
                attack = m2.group(1)
                metric = m2.group(2)
                value = float(m2.group(3))
                if metric in ("tp", "fp", "tn", "fn"):
                    value = int(value)
                result[current_epoch][attack][metric] = value

    return dict(result)


def select_best_epoch(stats_data, mal_nodes_data):
    """Select best epoch using same criteria as extract_best_epoch_results.py."""
    epoch_scores = []
    for epoch_num in stats_data:
        stats = stats_data[epoch_num]
        attacks_detected = sum(1 for a in ATTACKS if stats.get(a, {}).get("tp", 0) > 0)

        # Unique malicious node IDs with TP
        unique_tp_ids = set()
        if epoch_num in mal_nodes_data:
            for a in ATTACKS:
                for nid, ndata in mal_nodes_data.get(epoch_num, {}).get(a, {}).items():
                    if ndata["is_tp"]:
                        unique_tp_ids.add(nid)
        unique_tp_count = len(unique_tp_ids)

        total_tp = sum(stats.get(a, {}).get("tp", 0) for a in ATTACKS)
        epoch_scores.append((attacks_detected, unique_tp_count, total_tp, epoch_num))

    epoch_scores.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))
    return epoch_scores[0][3]


# ============================================================
# Part 2: Load .pth node classifications
# ============================================================

def load_pth_classifications(config_name, epoch):
    """
    Load result_model_epoch_X.pth and return per-node classification dict.

    CAVEAT: This only reflects the LAST evaluated attack (graph_7_ddos_loit).

    Returns:
        dict: {node_id: {"y_true": int, "y_hat": int, "score": float, "is_seen": int}}
    """
    path = get_result_pth_path(config_name, epoch)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found")
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    result = {}
    for node_id, node_data in data.items():
        result[node_id] = {
            "y_true": node_data.get("y_true", 0),
            "y_hat": node_data.get("y_hat", 0),
            "score": node_data.get("score", 0.0),
            "is_seen": node_data.get("is_seen", 0),
        }
    return result


def compute_pth_diff(enhanced_data, baseline_data):
    """
    Compare node classifications from .pth files between enhanced and baseline.

    Returns dict with detailed diff information.
    """
    all_nodes = set(enhanced_data.keys()) | set(baseline_data.keys())
    common_nodes = set(enhanced_data.keys()) & set(baseline_data.keys())

    diff = {
        "total_nodes_enhanced": len(enhanced_data),
        "total_nodes_baseline": len(baseline_data),
        "common_nodes": len(common_nodes),
        "flipped_nodes": [],  # list of (node_id, baseline_yhat, enhanced_yhat, y_true)
        "enhanced_only_nodes": sorted(set(enhanced_data.keys()) - set(baseline_data.keys())),
        "baseline_only_nodes": sorted(set(baseline_data.keys()) - set(enhanced_data.keys())),
        # Confusion matrix counts
        "baseline_tp": 0, "baseline_fp": 0, "baseline_tn": 0, "baseline_fn": 0,
        "enhanced_tp": 0, "enhanced_fp": 0, "enhanced_tn": 0, "enhanced_fn": 0,
    }

    for nid in common_nodes:
        b = baseline_data[nid]
        e = enhanced_data[nid]

        # Baseline confusion
        if b["y_true"] == 1 and b["y_hat"] == 1: diff["baseline_tp"] += 1
        elif b["y_true"] == 0 and b["y_hat"] == 1: diff["baseline_fp"] += 1
        elif b["y_true"] == 0 and b["y_hat"] == 0: diff["baseline_tn"] += 1
        elif b["y_true"] == 1 and b["y_hat"] == 0: diff["baseline_fn"] += 1

        # Enhanced confusion
        if e["y_true"] == 1 and e["y_hat"] == 1: diff["enhanced_tp"] += 1
        elif e["y_true"] == 0 and e["y_hat"] == 1: diff["enhanced_fp"] += 1
        elif e["y_true"] == 0 and e["y_hat"] == 0: diff["enhanced_tn"] += 1
        elif e["y_true"] == 1 and e["y_hat"] == 0: diff["enhanced_fn"] += 1

        if b["y_hat"] != e["y_hat"]:
            diff["flipped_nodes"].append({
                "node_id": nid,
                "baseline_yhat": b["y_hat"],
                "enhanced_yhat": e["y_hat"],
                "y_true": b["y_true"],
                "baseline_score": b["score"],
                "enhanced_score": e["score"],
            })

    diff["flipped_nodes"].sort(key=lambda x: x["node_id"])
    diff["total_flipped"] = len(diff["flipped_nodes"])

    # Categorize flips
    diff["flipped_benign_to_mal"] = sum(
        1 for f in diff["flipped_nodes"] if f["baseline_yhat"] == 0 and f["enhanced_yhat"] == 1
    )
    diff["flipped_mal_to_benign"] = sum(
        1 for f in diff["flipped_nodes"] if f["baseline_yhat"] == 1 and f["enhanced_yhat"] == 0
    )
    # Among flips, how many are correct flips?
    diff["flipped_correct_to_incorrect"] = sum(
        1 for f in diff["flipped_nodes"]
        if f["baseline_yhat"] == f["y_true"] and f["enhanced_yhat"] != f["y_true"]
    )
    diff["flipped_incorrect_to_correct"] = sum(
        1 for f in diff["flipped_nodes"]
        if f["baseline_yhat"] != f["y_true"] and f["enhanced_yhat"] == f["y_true"]
    )
    diff["flipped_both_wrong"] = sum(
        1 for f in diff["flipped_nodes"]
        if f["baseline_yhat"] != f["y_true"] and f["enhanced_yhat"] != f["y_true"]
    )

    return diff


# ============================================================
# Part 3: Per-attack malicious node diffs from logs
# ============================================================

def compute_malicious_node_diff(enhanced_mal, baseline_mal, enhanced_stats, baseline_stats):
    """
    Compare per-attack malicious node classifications between enhanced and baseline.

    Args:
        enhanced_mal: {attack: {node_id: {"is_tp": bool, ...}}}
        baseline_mal: {attack: {node_id: {"is_tp": bool, ...}}}
        enhanced_stats: {attack: {"tp":..., "fp":..., "fn":..., "tn":...}}
        baseline_stats: same

    Returns list of per-attack diff rows.
    """
    rows = []
    for attack in ATTACKS:
        e_nodes = enhanced_mal.get(attack, {})
        b_nodes = baseline_mal.get(attack, {})
        e_stats = enhanced_stats.get(attack, {})
        b_stats = baseline_stats.get(attack, {})

        all_mal_nodes = set(e_nodes.keys()) | set(b_nodes.keys())

        # Malicious node classification changes
        tp_gained = []  # was FN in baseline, now TP in enhanced
        tp_lost = []    # was TP in baseline, now FN in enhanced
        unchanged_tp = []
        unchanged_fn = []
        enhanced_only = []
        baseline_only = []

        for nid in all_mal_nodes:
            if nid in e_nodes and nid in b_nodes:
                if b_nodes[nid]["is_tp"] and e_nodes[nid]["is_tp"]:
                    unchanged_tp.append(nid)
                elif not b_nodes[nid]["is_tp"] and not e_nodes[nid]["is_tp"]:
                    unchanged_fn.append(nid)
                elif not b_nodes[nid]["is_tp"] and e_nodes[nid]["is_tp"]:
                    tp_gained.append(nid)
                elif b_nodes[nid]["is_tp"] and not e_nodes[nid]["is_tp"]:
                    tp_lost.append(nid)
            elif nid in e_nodes and nid not in b_nodes:
                enhanced_only.append(nid)
            elif nid not in e_nodes and nid in b_nodes:
                baseline_only.append(nid)

        # FP delta (benign nodes misclassified)
        e_fp = e_stats.get("fp", 0)
        b_fp = b_stats.get("fp", 0)
        fp_delta = e_fp - b_fp

        rows.append({
            "attack": attack,
            "baseline_tp": b_stats.get("tp", 0),
            "enhanced_tp": e_stats.get("tp", 0),
            "baseline_fp": b_stats.get("fp", 0),
            "enhanced_fp": e_stats.get("fp", 0),
            "baseline_fn": b_stats.get("fn", 0),
            "enhanced_fn": e_stats.get("fn", 0),
            "baseline_tn": b_stats.get("tn", 0),
            "enhanced_tn": e_stats.get("tn", 0),
            "tp_gained": len(tp_gained),
            "tp_lost": len(tp_lost),
            "unchanged_tp": len(unchanged_tp),
            "unchanged_fn": len(unchanged_fn),
            "fp_delta": fp_delta,
            "tp_gained_nodes": sorted(tp_gained),
            "tp_lost_nodes": sorted(tp_lost),
            "total_malicious_nodes": len(all_mal_nodes),
        })

    return rows


# ============================================================
# Part 4: Timing analysis
# ============================================================

def parse_training_timing(config_name):
    """
    Parse training tainted_batches JSON and compute avg time per tainted batch per epoch.

    Returns:
        dict: {epoch: {"avg_total_ms": float, "avg_forward_ms": float, "avg_backward_ms": float,
                        "num_batches": int, "avg_taint_ratio": float,
                        "avg_kde_eligible": float, "avg_edges_reduced": float}}
    """
    path = get_training_tainted_json_path(config_name)
    if not os.path.exists(path):
        print(f"  WARNING: Training timing not found: {path}")
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    batches = data.get("batches", [])
    epoch_data = defaultdict(list)
    for b in batches:
        if b.get("phase") == "train":
            epoch_data[b["epoch"]].append(b)

    result = {}
    for epoch, batch_list in sorted(epoch_data.items()):
        n = len(batch_list)
        if n == 0:
            continue
        result[epoch] = {
            "num_batches": n,
            "avg_total_ms": sum(b["total_time_ms"] for b in batch_list) / n,
            "avg_forward_ms": sum(b["forward_time_ms"] for b in batch_list) / n,
            "avg_backward_ms": sum(b["backward_time_ms"] for b in batch_list) / n,
            "avg_taint_ratio": sum(b["taint_ratio"] for b in batch_list) / n,
            "avg_kde_eligible": sum(b.get("kde_eligible_edges", 0) for b in batch_list) / n,
            "avg_edges_reduced": sum(b.get("edges_reduced", 0) for b in batch_list) / n,
            "total_kde_eligible": sum(b.get("kde_eligible_edges", 0) for b in batch_list),
            "total_edges_reduced": sum(b.get("edges_reduced", 0) for b in batch_list),
        }

    return result


def parse_inference_timing(config_name, epoch):
    """
    Parse inference tainted_batches JSON for a given epoch.
    Only considers 'inference_test' phase batches.

    Returns:
        dict with avg timing stats, or None if file not found.
    """
    path = get_inference_tainted_json_path(config_name, epoch)
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    batches = [b for b in data.get("batches", []) if b.get("phase") == "inference_test"]
    if not batches:
        return None

    n = len(batches)
    return {
        "num_batches": n,
        "avg_total_ms": sum(b["total_time_ms"] for b in batches) / n,
        "avg_forward_ms": sum(b["forward_time_ms"] for b in batches) / n,
        "avg_taint_ratio": sum(b["taint_ratio"] for b in batches) / n,
        "avg_kde_eligible": sum(b.get("kde_eligible_edges", 0) for b in batches) / n,
        "avg_edges_reduced": sum(b.get("edges_reduced", 0) for b in batches) / n,
        "total_kde_eligible": sum(b.get("kde_eligible_edges", 0) for b in batches),
        "total_edges_reduced": sum(b.get("edges_reduced", 0) for b in batches),
    }


# ============================================================
# Main analysis
# ============================================================

def main():
    print("=" * 90)
    print("PIDSMaker CIC-IDS-2017 — Node Classification & Timing Analysis")
    print("=" * 90)

    run_map = {config: run_dir for run_dir, config in RUNS}

    # ----------------------------------------------------------
    # Step 1: Parse all logs
    # ----------------------------------------------------------
    print("\n[Step 1] Parsing wandb logs for all configs...")
    all_mal_nodes = {}   # config -> {epoch -> {attack -> {node_id -> info}}}
    all_stats = {}       # config -> {epoch -> {attack -> {metric -> value}}}

    for run_dir, config_name in RUNS:
        log_path = os.path.join(WANDB_DIR, run_dir, "files", "output.log")
        if not os.path.exists(log_path):
            print(f"  WARNING: {log_path} not found, skipping {config_name}")
            continue
        all_mal_nodes[config_name] = parse_log_malicious_nodes(log_path)
        all_stats[config_name] = parse_log_stats(log_path)

        # Compute best epoch
        best_ep = select_best_epoch(all_stats[config_name], all_mal_nodes[config_name])
        BEST_EPOCHS[config_name] = best_ep
        print(f"  {config_name}: parsed OK, best_epoch={best_ep}")

    # ----------------------------------------------------------
    # Step 2: Per-attack malicious node classification diffs
    # ----------------------------------------------------------
    print("\n[Step 2] Computing per-attack malicious node classification diffs...")

    all_mal_diff_rows_best = []
    all_mal_diff_rows_last = []

    for enhanced, baseline in COMPARISON_PAIRS:
        if enhanced not in all_stats or baseline not in all_stats:
            print(f"  WARNING: Missing data for {enhanced} vs {baseline}, skipping")
            continue

        for epoch_label, epoch_getter in [("best", lambda c: BEST_EPOCHS.get(c)),
                                           ("last", lambda c: LAST_EPOCH)]:
            e_epoch = epoch_getter(enhanced)
            b_epoch = epoch_getter(baseline)

            if e_epoch is None or b_epoch is None:
                continue

            e_mal = all_mal_nodes.get(enhanced, {}).get(e_epoch, {})
            b_mal = all_mal_nodes.get(baseline, {}).get(b_epoch, {})
            e_stats = all_stats.get(enhanced, {}).get(e_epoch, {})
            b_stats = all_stats.get(baseline, {}).get(b_epoch, {})

            diff_rows = compute_malicious_node_diff(e_mal, b_mal, e_stats, b_stats)

            for row in diff_rows:
                row["enhanced_config"] = enhanced
                row["baseline_config"] = baseline
                row["enhanced_epoch"] = e_epoch
                row["baseline_epoch"] = b_epoch

            if epoch_label == "best":
                all_mal_diff_rows_best.extend(diff_rows)
            else:
                all_mal_diff_rows_last.extend(diff_rows)

    # ----------------------------------------------------------
    # Step 3: .pth-based full node classification diffs
    # ----------------------------------------------------------
    print("\n[Step 3] Computing .pth-based node classification diffs (last-evaluated attack only)...")

    pth_diff_rows_best = []
    pth_diff_rows_last = []

    for enhanced, baseline in COMPARISON_PAIRS:
        for epoch_label, epoch_getter in [("best", lambda c: BEST_EPOCHS.get(c)),
                                           ("last", lambda c: LAST_EPOCH)]:
            e_epoch = epoch_getter(enhanced)
            b_epoch = epoch_getter(baseline)

            if e_epoch is None or b_epoch is None:
                continue

            print(f"  {enhanced} (epoch {e_epoch}) vs {baseline} (epoch {b_epoch}) [{epoch_label}]")
            e_data = load_pth_classifications(enhanced, e_epoch)
            b_data = load_pth_classifications(baseline, b_epoch)

            if not e_data or not b_data:
                print(f"    WARNING: Could not load .pth data, skipping")
                continue

            diff = compute_pth_diff(e_data, b_data)

            row = {
                "enhanced_config": enhanced,
                "baseline_config": baseline,
                "enhanced_epoch": e_epoch,
                "baseline_epoch": b_epoch,
                "epoch_type": epoch_label,
                "last_evaluated_attack": LAST_EVALUATED_ATTACK,
                "total_nodes": diff["common_nodes"],
                "total_flipped": diff["total_flipped"],
                "flipped_benign_to_mal": diff["flipped_benign_to_mal"],
                "flipped_mal_to_benign": diff["flipped_mal_to_benign"],
                "flipped_incorrect_to_correct": diff["flipped_incorrect_to_correct"],
                "flipped_correct_to_incorrect": diff["flipped_correct_to_incorrect"],
                "baseline_tp": diff["baseline_tp"],
                "baseline_fp": diff["baseline_fp"],
                "baseline_fn": diff["baseline_fn"],
                "baseline_tn": diff["baseline_tn"],
                "enhanced_tp": diff["enhanced_tp"],
                "enhanced_fp": diff["enhanced_fp"],
                "enhanced_fn": diff["enhanced_fn"],
                "enhanced_tn": diff["enhanced_tn"],
                "flipped_node_ids": "; ".join(
                    str(f["node_id"]) for f in diff["flipped_nodes"]
                ),
                "flipped_node_details": "; ".join(
                    f"node_{f['node_id']}(base_yhat={f['baseline_yhat']},enh_yhat={f['enhanced_yhat']},y_true={f['y_true']},base_score={f['baseline_score']:.4f},enh_score={f['enhanced_score']:.4f})"
                    for f in diff["flipped_nodes"]
                ),
            }

            if epoch_label == "best":
                pth_diff_rows_best.append(row)
            else:
                pth_diff_rows_last.append(row)

    # ----------------------------------------------------------
    # Step 4: Timing analysis
    # ----------------------------------------------------------
    print("\n[Step 4] Computing timing statistics...")

    training_timing_rows = []
    inference_timing_rows_best = []
    inference_timing_rows_last = []

    # Training timing: average across ALL epochs for each config (including baselines)
    all_configs_ordered = [
        c for _, c in RUNS  # preserve RUNS order
    ]

    for config_name in all_configs_ordered:
        print(f"  Training timing: {config_name}")
        timing = parse_training_timing(config_name)
        if not timing:
            continue

        # Grand average across all epochs
        all_epochs_total_ms = []
        all_epochs_forward_ms = []
        all_epochs_backward_ms = []
        all_epochs_taint_ratio = []
        all_epochs_kde_eligible = []
        all_epochs_edges_reduced = []
        total_batches = 0

        for ep, ep_data in sorted(timing.items()):
            n = ep_data["num_batches"]
            total_batches += n
            all_epochs_total_ms.extend([ep_data["avg_total_ms"]] * n)
            all_epochs_forward_ms.extend([ep_data["avg_forward_ms"]] * n)
            all_epochs_backward_ms.extend([ep_data["avg_backward_ms"]] * n)
            all_epochs_taint_ratio.extend([ep_data["avg_taint_ratio"]] * n)
            all_epochs_kde_eligible.extend([ep_data["avg_kde_eligible"]] * n)
            all_epochs_edges_reduced.extend([ep_data["avg_edges_reduced"]] * n)

        # Per-epoch rows
        for ep in sorted(timing.keys()):
            ep_data = timing[ep]
            training_timing_rows.append({
                "config": config_name,
                "epoch": ep,
                "num_tainted_batches": ep_data["num_batches"],
                "avg_total_time_ms": round(ep_data["avg_total_ms"], 4),
                "avg_forward_time_ms": round(ep_data["avg_forward_ms"], 4),
                "avg_backward_time_ms": round(ep_data["avg_backward_ms"], 4),
                "avg_taint_ratio": round(ep_data["avg_taint_ratio"], 6),
                "avg_kde_eligible_edges": round(ep_data["avg_kde_eligible"], 2),
                "avg_edges_reduced": round(ep_data["avg_edges_reduced"], 2),
                "total_kde_eligible_edges": ep_data["total_kde_eligible"],
                "total_edges_reduced": ep_data["total_edges_reduced"],
            })

        # Grand average row
        if total_batches > 0:
            training_timing_rows.append({
                "config": config_name,
                "epoch": "ALL",
                "num_tainted_batches": total_batches,
                "avg_total_time_ms": round(sum(all_epochs_total_ms) / len(all_epochs_total_ms), 4),
                "avg_forward_time_ms": round(sum(all_epochs_forward_ms) / len(all_epochs_forward_ms), 4),
                "avg_backward_time_ms": round(sum(all_epochs_backward_ms) / len(all_epochs_backward_ms), 4),
                "avg_taint_ratio": round(sum(all_epochs_taint_ratio) / len(all_epochs_taint_ratio), 6),
                "avg_kde_eligible_edges": round(sum(all_epochs_kde_eligible) / len(all_epochs_kde_eligible), 2),
                "avg_edges_reduced": round(sum(all_epochs_edges_reduced) / len(all_epochs_edges_reduced), 2),
                "total_kde_eligible_edges": sum(t["total_kde_eligible"] for t in timing.values()),
                "total_edges_reduced": sum(t["total_edges_reduced"] for t in timing.values()),
            })

    # Inference timing: for best and last epoch (all configs)
    for config_name in all_configs_ordered:
        for epoch_label, epoch_val in [("best", BEST_EPOCHS.get(config_name)),
                                        ("last", LAST_EPOCH)]:
            if epoch_val is None:
                continue

            inf_timing = parse_inference_timing(config_name, epoch_val)
            if inf_timing is None:
                print(f"  Inference timing: {config_name} epoch {epoch_val} ({epoch_label}) — NOT FOUND")
                continue

            row = {
                "config": config_name,
                "epoch": epoch_val,
                "epoch_type": epoch_label,
                "num_tainted_batches": inf_timing["num_batches"],
                "avg_total_time_ms": round(inf_timing["avg_total_ms"], 4),
                "avg_forward_time_ms": round(inf_timing["avg_forward_ms"], 4),
                "avg_taint_ratio": round(inf_timing["avg_taint_ratio"], 6),
                "avg_kde_eligible_edges": round(inf_timing["avg_kde_eligible"], 2),
                "avg_edges_reduced": round(inf_timing["avg_edges_reduced"], 2),
                "total_kde_eligible_edges": inf_timing["total_kde_eligible"],
                "total_edges_reduced": inf_timing["total_edges_reduced"],
            }

            if epoch_label == "best":
                inference_timing_rows_best.append(row)
            else:
                inference_timing_rows_last.append(row)

        print(f"  Inference timing: {config_name} — done")

    # ----------------------------------------------------------
    # Step 5: Write output CSVs
    # ----------------------------------------------------------
    print("\n[Step 5] Writing output CSVs...")
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # --- CSV 1: Per-attack malicious node diff (best epoch) ---
    path1 = os.path.join(out_dir, "node_classification_diff_best_epoch.csv")
    fields1 = [
        "enhanced_config", "baseline_config", "enhanced_epoch", "baseline_epoch",
        "attack", "total_malicious_nodes",
        "baseline_tp", "enhanced_tp", "tp_gained", "tp_lost",
        "unchanged_tp", "unchanged_fn",
        "baseline_fp", "enhanced_fp", "fp_delta",
        "baseline_fn", "enhanced_fn",
        "baseline_tn", "enhanced_tn",
        "tp_gained_nodes", "tp_lost_nodes",
    ]
    with open(path1, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields1, extrasaction="ignore")
        writer.writeheader()
        for row in all_mal_diff_rows_best:
            row_copy = dict(row)
            row_copy["tp_gained_nodes"] = "; ".join(str(n) for n in row["tp_gained_nodes"])
            row_copy["tp_lost_nodes"] = "; ".join(str(n) for n in row["tp_lost_nodes"])
            writer.writerow(row_copy)
    print(f"  Written: {path1} ({len(all_mal_diff_rows_best)} rows)")

    # --- CSV 2: Per-attack malicious node diff (last epoch) ---
    path2 = os.path.join(out_dir, "node_classification_diff_last_epoch.csv")
    with open(path2, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields1, extrasaction="ignore")
        writer.writeheader()
        for row in all_mal_diff_rows_last:
            row_copy = dict(row)
            row_copy["tp_gained_nodes"] = "; ".join(str(n) for n in row["tp_gained_nodes"])
            row_copy["tp_lost_nodes"] = "; ".join(str(n) for n in row["tp_lost_nodes"])
            writer.writerow(row_copy)
    print(f"  Written: {path2} ({len(all_mal_diff_rows_last)} rows)")

    # --- CSV 3: .pth full node diff (best + last combined) ---
    path3 = os.path.join(out_dir, "pth_node_classification_diff.csv")
    fields3 = [
        "enhanced_config", "baseline_config", "enhanced_epoch", "baseline_epoch",
        "epoch_type", "last_evaluated_attack", "total_nodes",
        "total_flipped", "flipped_benign_to_mal", "flipped_mal_to_benign",
        "flipped_incorrect_to_correct", "flipped_correct_to_incorrect",
        "baseline_tp", "baseline_fp", "baseline_fn", "baseline_tn",
        "enhanced_tp", "enhanced_fp", "enhanced_fn", "enhanced_tn",
        "flipped_node_ids", "flipped_node_details",
    ]
    with open(path3, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields3)
        writer.writeheader()
        writer.writerows(pth_diff_rows_best)
        writer.writerows(pth_diff_rows_last)
    print(f"  Written: {path3} ({len(pth_diff_rows_best) + len(pth_diff_rows_last)} rows)")

    # --- CSV 4: Training timing ---
    path4 = os.path.join(out_dir, "training_timing_per_epoch.csv")
    fields4 = [
        "config", "epoch", "num_tainted_batches",
        "avg_total_time_ms", "avg_forward_time_ms", "avg_backward_time_ms",
        "avg_taint_ratio", "avg_kde_eligible_edges", "avg_edges_reduced",
        "total_kde_eligible_edges", "total_edges_reduced",
    ]
    with open(path4, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields4)
        writer.writeheader()
        writer.writerows(training_timing_rows)
    print(f"  Written: {path4} ({len(training_timing_rows)} rows)")

    # --- CSV 5: Inference timing ---
    path5 = os.path.join(out_dir, "inference_timing.csv")
    fields5 = [
        "config", "epoch", "epoch_type", "num_tainted_batches",
        "avg_total_time_ms", "avg_forward_time_ms",
        "avg_taint_ratio", "avg_kde_eligible_edges", "avg_edges_reduced",
        "total_kde_eligible_edges", "total_edges_reduced",
    ]
    with open(path5, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields5)
        writer.writeheader()
        writer.writerows(inference_timing_rows_best)
        writer.writerows(inference_timing_rows_last)
    print(f"  Written: {path5} ({len(inference_timing_rows_best) + len(inference_timing_rows_last)} rows)")

    # --- CSV 6: Summary classification diff table ---
    path6 = os.path.join(out_dir, "classification_diff_summary.csv")
    fields6 = [
        "enhanced_config", "baseline_config", "epoch_type",
        "enhanced_epoch", "baseline_epoch",
        "attacks_with_tp_gained", "attacks_with_tp_lost",
        "total_tp_gained_across_attacks", "total_tp_lost_across_attacks",
        "total_fp_delta_across_attacks",
        "pth_total_flipped_nodes", "pth_flipped_benign_to_mal", "pth_flipped_mal_to_benign",
        "pth_flipped_incorrect_to_correct", "pth_flipped_correct_to_incorrect",
    ]
    summary_rows = []
    for epoch_label, mal_rows, pth_rows in [
        ("best", all_mal_diff_rows_best, pth_diff_rows_best),
        ("last", all_mal_diff_rows_last, pth_diff_rows_last),
    ]:
        for enhanced, baseline in COMPARISON_PAIRS:
            pair_mal = [r for r in mal_rows
                        if r["enhanced_config"] == enhanced and r["baseline_config"] == baseline]
            pair_pth = [r for r in pth_rows
                        if r["enhanced_config"] == enhanced and r["baseline_config"] == baseline]

            if not pair_mal:
                continue

            e_epoch = pair_mal[0]["enhanced_epoch"]
            b_epoch = pair_mal[0]["baseline_epoch"]

            attacks_tp_gained = sum(1 for r in pair_mal if r["tp_gained"] > 0)
            attacks_tp_lost = sum(1 for r in pair_mal if r["tp_lost"] > 0)
            total_tp_gained = sum(r["tp_gained"] for r in pair_mal)
            total_tp_lost = sum(r["tp_lost"] for r in pair_mal)
            total_fp_delta = sum(r["fp_delta"] for r in pair_mal)

            pth_info = pair_pth[0] if pair_pth else {}

            summary_rows.append({
                "enhanced_config": enhanced,
                "baseline_config": baseline,
                "epoch_type": epoch_label,
                "enhanced_epoch": e_epoch,
                "baseline_epoch": b_epoch,
                "attacks_with_tp_gained": attacks_tp_gained,
                "attacks_with_tp_lost": attacks_tp_lost,
                "total_tp_gained_across_attacks": total_tp_gained,
                "total_tp_lost_across_attacks": total_tp_lost,
                "total_fp_delta_across_attacks": total_fp_delta,
                "pth_total_flipped_nodes": pth_info.get("total_flipped", ""),
                "pth_flipped_benign_to_mal": pth_info.get("flipped_benign_to_mal", ""),
                "pth_flipped_mal_to_benign": pth_info.get("flipped_mal_to_benign", ""),
                "pth_flipped_incorrect_to_correct": pth_info.get("flipped_incorrect_to_correct", ""),
                "pth_flipped_correct_to_incorrect": pth_info.get("flipped_correct_to_incorrect", ""),
            })

    with open(path6, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields6)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Written: {path6} ({len(summary_rows)} rows)")

    # ----------------------------------------------------------
    # Step 6: Write notes.md
    # ----------------------------------------------------------
    print("\n[Step 6] Writing notes.md...")
    write_notes_md(
        all_mal_nodes, all_stats, all_mal_diff_rows_best, all_mal_diff_rows_last,
        pth_diff_rows_best, pth_diff_rows_last,
        training_timing_rows, inference_timing_rows_best, inference_timing_rows_last,
    )

    print("\n" + "=" * 90)
    print("Analysis complete!")
    print("=" * 90)


# ============================================================
# Notes.md generation
# ============================================================

def write_notes_md(
    all_mal_nodes, all_stats,
    mal_diff_best, mal_diff_last,
    pth_diff_best, pth_diff_last,
    training_timing_rows,
    inference_timing_best, inference_timing_last,
):
    """Write detailed notes.md with all classification and timing findings."""

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notes.md")

    lines = []
    w = lines.append

    w("# PIDSMaker CIC-IDS-2017 — Node Classification & Timing Analysis Notes")
    w("")
    w(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("")
    w("## Table of Contents")
    w("")
    w("1. [Overview](#overview)")
    w("2. [Important Caveats](#important-caveats)")
    w("3. [Configuration Details](#configuration-details)")
    w("4. [Best Epoch Selection](#best-epoch-selection)")
    w("5. [Per-Attack Malicious Node Classification Diffs](#per-attack-malicious-node-classification-diffs)")
    w("6. [Full Node Classification Diffs (.pth)](#full-node-classification-diffs-pth)")
    w("7. [Training Timing Analysis](#training-timing-analysis)")
    w("8. [Inference Timing Analysis](#inference-timing-analysis)")
    w("9. [Detailed Per-Config Per-Epoch Per-Attack Node Lists](#detailed-per-config-per-epoch-per-attack-node-lists)")
    w("")

    # --- Overview ---
    w("## Overview")
    w("")
    w("This document provides a detailed analysis comparing node classifications and timing")
    w("between baseline PIDSMaker configurations and their KDE-enhanced / RED-enhanced variants")
    w("on the CIC-IDS-2017 dataset (14 per-attack evaluations across 3 graphs).")
    w("")
    w("**Configs analyzed:**")
    w("")
    w("| Config | Model | Type | Artifact Dir | Hash (first 8) |")
    w("|--------|-------|------|-------------|---------------|")
    for config_name, meta in sorted(CONFIG_META.items()):
        ctype = "Baseline" if meta["is_baseline"] else f"Enhanced (baseline: {meta['baseline_of']})"
        w(f"| {config_name} | {meta['model_type']} | {ctype} | {meta['artifact_dir']} | {meta['hash'][:8]}... |")
    w("")

    w("**Comparison pairs:**")
    w("")
    for enhanced, baseline in COMPARISON_PAIRS:
        w(f"- **{enhanced}** vs **{baseline}**")
    w("")

    # --- Caveats ---
    w("## Important Caveats")
    w("")
    w("### 1. result_model_epoch_X.pth Only Contains Last-Evaluated Attack")
    w("")
    w("The `.pth` files in `precision_recall_dir/` are **overwritten during each attack's evaluation**.")
    w("This means the file only reflects node classifications for the **last evaluated attack**,")
    w(f"which is **{LAST_EVALUATED_ATTACK}** (attack order is fixed across all configs).")
    w("")
    w("For the .pth-based diffs, the y_true labels and y_hat predictions only correspond to")
    w(f"the `{LAST_EVALUATED_ATTACK}` attack's evaluation context. The 2 malicious nodes in")
    w("these files are the ones relevant to that specific attack.")
    w("")
    w("### 2. Per-Attack Malicious Node Classification From Logs")
    w("")
    w("For per-attack malicious node TP/FN classification, we parse the wandb `output.log` files.")
    w("These logs contain lines like:")
    w("```")
    w("-> Malicious node 4761  : loss=6.1844 | is TP: ✅ 192.168.10.50 -> ...")
    w("-> Malicious node 3580  : loss=5.9732 | is TP: ❌ 192.168.10.51 -> ...")
    w("```")
    w("")
    w("This gives us **malicious node** classification per attack. For **benign node**")
    w("classification changes per attack, we can only report the FP count delta (not individual")
    w("node IDs), since the .pth file is overwritten.")
    w("")
    w("### 3. Orthrus Baseline Missing Inference Timing")
    w("")
    w("The `orthrus_cicids` baseline configuration does **not** have `inference_tainted_batches_*.json`")
    w("files in its `batch_timing/` directory. This means we cannot compute inference timing")
    w("for orthrus baseline. Training timing IS available for all configs including orthrus baseline.")
    w("")
    w("### 4. Best Epoch May Differ Between Enhanced and Baseline")
    w("")
    w("When comparing best epochs, note that the best epoch for an enhanced config may differ")
    w("from its baseline's best epoch. This is expected — the edge reduction changes training")
    w("dynamics and may shift which epoch performs best.")
    w("")

    # --- Configuration Details ---
    w("## Configuration Details")
    w("")
    w("| Config | wandb Run | Best Epoch | Artifact Path |")
    w("|--------|-----------|------------|---------------|")
    run_map = {config: run_dir for run_dir, config in RUNS}
    for config_name in sorted(CONFIG_META.keys()):
        be = BEST_EPOCHS.get(config_name, "N/A")
        run_dir = run_map.get(config_name, "N/A")
        art_base = get_artifact_base(config_name)
        # Shorten path
        short_art = art_base.replace(PIDSMAKER_DIR + "/", "")
        w(f"| {config_name} | {run_dir} | {be} | {short_art} |")
    w("")

    # --- Best Epoch Selection ---
    w("## Best Epoch Selection")
    w("")
    w("Criteria (in order of priority):")
    w("1. Highest number of attacks detected (attacks with TP > 0)")
    w("2. Highest number of unique TP malicious node IDs across all attacks")
    w("3. Highest total TP count across all attacks")
    w("")
    w("| Config | Best Epoch | Attacks Detected | Unique TP Nodes | Total TP |")
    w("|--------|------------|-----------------|-----------------|----------|")
    for config_name in sorted(CONFIG_META.keys()):
        be = BEST_EPOCHS.get(config_name)
        if be is None:
            continue
        stats = all_stats.get(config_name, {}).get(be, {})
        mal = all_mal_nodes.get(config_name, {}).get(be, {})
        atk_det = sum(1 for a in ATTACKS if stats.get(a, {}).get("tp", 0) > 0)
        total_tp = sum(stats.get(a, {}).get("tp", 0) for a in ATTACKS)
        unique_ids = set()
        for a in ATTACKS:
            for nid, ndata in mal.get(a, {}).items():
                if ndata["is_tp"]:
                    unique_ids.add(nid)
        w(f"| {config_name} | {be} | {atk_det}/14 | {len(unique_ids)} | {total_tp} |")
    w("")

    # --- Per-Attack Malicious Node Diffs ---
    w("## Per-Attack Malicious Node Classification Diffs")
    w("")
    w("These tables show, for each comparison pair and each attack, how many malicious nodes")
    w("changed classification between the enhanced and baseline configs.")
    w("")
    w("- **TP Gained**: Was FN (missed) in baseline, now TP (detected) in enhanced")
    w("- **TP Lost**: Was TP (detected) in baseline, now FN (missed) in enhanced")
    w("- **FP Delta**: Change in false positives (positive = more FPs in enhanced)")
    w("")

    for epoch_label, diff_rows in [("Best Epoch", mal_diff_best), ("Last Epoch (11)", mal_diff_last)]:
        w(f"### {epoch_label}")
        w("")

        for enhanced, baseline in COMPARISON_PAIRS:
            pair_rows = [r for r in diff_rows
                         if r["enhanced_config"] == enhanced and r["baseline_config"] == baseline]
            if not pair_rows:
                continue

            e_epoch = pair_rows[0]["enhanced_epoch"]
            b_epoch = pair_rows[0]["baseline_epoch"]

            w(f"#### {enhanced} (epoch {e_epoch}) vs {baseline} (epoch {b_epoch})")
            w("")
            w("| Attack | Mal Nodes | Base TP | Enh TP | TP Gained | TP Lost | Base FP | Enh FP | FP Δ |")
            w("|--------|-----------|---------|--------|-----------|---------|---------|--------|------|")

            totals = {"mal": 0, "b_tp": 0, "e_tp": 0, "gained": 0, "lost": 0, "b_fp": 0, "e_fp": 0, "fp_d": 0}
            for r in pair_rows:
                attack_short = r["attack"].replace("graph_5_", "5/").replace("graph_6_", "6/").replace("graph_7_", "7/")
                gained_str = f"**{r['tp_gained']}**" if r["tp_gained"] > 0 else str(r["tp_gained"])
                lost_str = f"**{r['tp_lost']}**" if r["tp_lost"] > 0 else str(r["tp_lost"])
                fp_d = r["fp_delta"]
                fp_str = f"+{fp_d}" if fp_d > 0 else str(fp_d)
                w(f"| {attack_short} | {r['total_malicious_nodes']} | {r['baseline_tp']} | {r['enhanced_tp']} | {gained_str} | {lost_str} | {r['baseline_fp']} | {r['enhanced_fp']} | {fp_str} |")
                totals["mal"] += r["total_malicious_nodes"]
                totals["b_tp"] += r["baseline_tp"]
                totals["e_tp"] += r["enhanced_tp"]
                totals["gained"] += r["tp_gained"]
                totals["lost"] += r["tp_lost"]
                totals["b_fp"] += r["baseline_fp"]
                totals["e_fp"] += r["enhanced_fp"]
                totals["fp_d"] += r["fp_delta"]
            fp_total_str = f"+{totals['fp_d']}" if totals["fp_d"] > 0 else str(totals["fp_d"])
            w(f"| **TOTAL** | {totals['mal']} | {totals['b_tp']} | {totals['e_tp']} | **{totals['gained']}** | **{totals['lost']}** | {totals['b_fp']} | {totals['e_fp']} | {fp_total_str} |")
            w("")

            # List the actual gained/lost node IDs
            any_changes = False
            for r in pair_rows:
                if r["tp_gained"] > 0 or r["tp_lost"] > 0:
                    if not any_changes:
                        w("**Node-level changes:**")
                        w("")
                        any_changes = True
                    attack_short = r["attack"]
                    if r["tp_gained"] > 0:
                        nodes_str = ", ".join(str(n) for n in r["tp_gained_nodes"])
                        w(f"- {attack_short}: TP gained = [{nodes_str}]")
                    if r["tp_lost"] > 0:
                        nodes_str = ", ".join(str(n) for n in r["tp_lost_nodes"])
                        w(f"- {attack_short}: TP lost = [{nodes_str}]")

            if not any_changes:
                w("*No malicious node classification changes between enhanced and baseline.*")
            w("")

    # --- .pth Full Node Diffs ---
    w("## Full Node Classification Diffs (.pth)")
    w("")
    w(f"These diffs are based on `result_model_epoch_X.pth` files and reflect the classification")
    w(f"state after evaluating **{LAST_EVALUATED_ATTACK}** (the last attack in the evaluation order).")
    w(f"All nodes (benign + malicious) are included.")
    w("")

    all_pth = pth_diff_best + pth_diff_last
    if all_pth:
        w("| Enhanced Config | Baseline Config | Epoch Type | Enh Epoch | Base Epoch | Total Nodes | Flipped | Ben→Mal | Mal→Ben | Incorr→Corr | Corr→Incorr |")
        w("|----------------|-----------------|------------|-----------|------------|-------------|---------|---------|---------|-------------|-------------|")
        for r in all_pth:
            w(f"| {r['enhanced_config']} | {r['baseline_config']} | {r['epoch_type']} | {r['enhanced_epoch']} | {r['baseline_epoch']} | {r['total_nodes']} | {r['total_flipped']} | {r['flipped_benign_to_mal']} | {r['flipped_mal_to_benign']} | {r['flipped_incorrect_to_correct']} | {r['flipped_correct_to_incorrect']} |")
        w("")

        # Detail the flipped nodes
        w("### Flipped Node Details")
        w("")
        for r in all_pth:
            if r["total_flipped"] > 0:
                w(f"**{r['enhanced_config']} vs {r['baseline_config']} ({r['epoch_type']} epoch, enhanced={r['enhanced_epoch']}, baseline={r['baseline_epoch']}):**")
                w("")
                w("| Node ID | Baseline ŷ | Enhanced ŷ | y_true | Baseline Score | Enhanced Score | Change |")
                w("|---------|-----------|-----------|--------|---------------|---------------|--------|")
                details = r["flipped_node_details"].split("; ")
                for d in details:
                    # Parse the detail string
                    import re as _re
                    m = _re.match(
                        r"node_(\d+)\(base_yhat=(\d+),enh_yhat=(\d+),y_true=(\d+),base_score=([\d.]+),enh_score=([\d.]+)\)",
                        d
                    )
                    if m:
                        nid = m.group(1)
                        b_yhat = int(m.group(2))
                        e_yhat = int(m.group(3))
                        y_true = int(m.group(4))
                        b_score = float(m.group(5))
                        e_score = float(m.group(6))
                        b_label = "mal" if b_yhat == 1 else "ben"
                        e_label = "mal" if e_yhat == 1 else "ben"
                        correct = "✅ improved" if e_yhat == y_true else "❌ worsened"
                        w(f"| {nid} | {b_yhat} ({b_label}) | {e_yhat} ({e_label}) | {y_true} | {b_score:.4f} | {e_score:.4f} | {correct} |")
                w("")
    else:
        w("*No .pth diff data available.*")
        w("")

    # --- Training Timing ---
    w("## Training Timing Analysis")
    w("")
    w("Average time per tainted batch during training.")
    w("")
    w("*Note: `orthrus_cicids` baseline has 0 tainted batches (no timing data).")
    w("Baseline tainted batch tracking records batches containing tainted/attack-related edges")
    w("regardless of whether KDE reduction is applied.*")
    w("")

    # Grand average table for all configs
    w("### Grand Average Across All Epochs")
    w("")
    w("| Config | Tainted Batches (total) | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | Avg KDE Eligible | Avg Edges Reduced |")
    w("|--------|------------------------|---------------|-----------------|-------------------|----------------|-----------------|-------------------|")
    for row in training_timing_rows:
        if row["epoch"] == "ALL":
            w(f"| {row['config']} | {row['num_tainted_batches']} | {row['avg_total_time_ms']:.2f} | {row['avg_forward_time_ms']:.2f} | {row['avg_backward_time_ms']:.2f} | {row['avg_taint_ratio']:.4f} | {row['avg_kde_eligible_edges']:.1f} | {row['avg_edges_reduced']:.1f} |")
    w("")

    w("### Per-Epoch Training Timing Detail")
    w("")
    # Group by config
    configs_in_timing = []
    seen = set()
    for row in training_timing_rows:
        if row["config"] not in seen and row["epoch"] != "ALL":
            configs_in_timing.append(row["config"])
            seen.add(row["config"])

    for config_name in configs_in_timing:
        config_rows = [r for r in training_timing_rows if r["config"] == config_name and r["epoch"] != "ALL"]
        w(f"#### {config_name}")
        w("")
        w("| Epoch | Batches | Avg Total (ms) | Avg Forward (ms) | Avg Backward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |")
        w("|-------|---------|---------------|-----------------|-------------------|----------------|-------------|---------------|")
        for r in config_rows:
            w(f"| {r['epoch']} | {r['num_tainted_batches']} | {r['avg_total_time_ms']:.2f} | {r['avg_forward_time_ms']:.2f} | {r['avg_backward_time_ms']:.2f} | {r['avg_taint_ratio']:.4f} | {r['avg_kde_eligible_edges']:.1f} | {r['avg_edges_reduced']:.1f} |")
        w("")

    # --- Inference Timing ---
    w("## Inference Timing Analysis")
    w("")
    w("Average time per KDE tainted batch during inference (test split only).")
    w("")
    w("**Note:** `orthrus_cicids` baseline does NOT have inference tainted batch timing files.")
    w("")

    all_inf = inference_timing_best + inference_timing_last
    if all_inf:
        w("### All Configs Inference Timing")
        w("")
        w("| Config | Epoch | Epoch Type | Tainted Batches | Avg Total (ms) | Avg Forward (ms) | Avg Taint Ratio | KDE Eligible | Edges Reduced |")
        w("|--------|-------|------------|----------------|---------------|-----------------|----------------|-------------|---------------|")
        for r in all_inf:
            w(f"| {r['config']} | {r['epoch']} | {r['epoch_type']} | {r['num_tainted_batches']} | {r['avg_total_time_ms']:.2f} | {r['avg_forward_time_ms']:.2f} | {r['avg_taint_ratio']:.4f} | {r['avg_kde_eligible_edges']:.1f} | {r['avg_edges_reduced']:.1f} |")
        w("")
    else:
        w("*No inference timing data available for enhanced configs.*")
        w("")

    # --- Detailed Node Lists ---
    w("## Detailed Per-Config Per-Epoch Per-Attack Node Lists")
    w("")
    w("This section lists every malicious node and its classification (TP ✅ / FN ❌)")
    w("for each config, each evaluated epoch, and each attack.")
    w("")

    for config_name in sorted(all_mal_nodes.keys()):
        w(f"### {config_name}")
        w("")
        epochs_data = all_mal_nodes[config_name]
        stats_data = all_stats.get(config_name, {})

        for epoch in sorted(epochs_data.keys()):
            best_marker = " ⭐ BEST" if epoch == BEST_EPOCHS.get(config_name) else ""
            w(f"#### Epoch {epoch}{best_marker}")
            w("")

            epoch_mal = epochs_data[epoch]
            epoch_stats = stats_data.get(epoch, {})

            has_any = False
            for attack in ATTACKS:
                nodes = epoch_mal.get(attack, {})
                stats = epoch_stats.get(attack, {})
                if not nodes and not stats:
                    continue

                has_any = True
                tp = stats.get("tp", 0)
                fp = stats.get("fp", 0)
                fn = stats.get("fn", 0)
                tn = stats.get("tn", 0)
                attack_short = attack.replace("graph_5_", "5/").replace("graph_6_", "6/").replace("graph_7_", "7/")

                w(f"**{attack_short}** — TP={tp}, FP={fp}, FN={fn}, TN={tn}")
                w("")
                if nodes:
                    for nid in sorted(nodes.keys()):
                        ndata = nodes[nid]
                        marker = "✅ TP" if ndata["is_tp"] else "❌ FN"
                        w(f"- Node {nid}: loss={ndata['loss']:.4f} | {marker} | {ndata['info']}")
                else:
                    w("- *(no malicious nodes listed in log)*")
                w("")

            if not has_any:
                w("*(no evaluation data for this epoch)*")
                w("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Written: {out_path}")


if __name__ == "__main__":
    main()
