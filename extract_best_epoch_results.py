#!/usr/bin/env python3
"""
Extract best epoch evaluation results from PIDSMaker wandb runs.

Best epoch selection criteria (in priority order):
1. Highest number of attacks detected (attacks with TP > 0)
2. Highest number of total unique malicious nodes detected (TP) across attacks
3. Highest total malicious nodes detected across attacks (sum of TP counts)
4. Random tiebreaker

For the best epoch of each run, outputs per-attack:
  TP, FP, FN, TN, precision, recall, accuracy, TP_node_IDs
"""

import re
import csv
import os
from collections import defaultdict

# ============================================================
# Configuration: 7 completed runs (latest = kairos_cicids_red is incomplete, skipped)
# ============================================================
WANDB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wandb")

RUNS = [
    # (run_dir, config_name)
    ("run-20260403_173823-l88hmsi6", "kairos_cicids_red"),
    ("run-20260403_172408-arq94flm", "orthrus_cicids_red"),
    ("run-20260403_081130-gefm583r", "kairos_cicids_kde_diff"),
    ("run-20260403_071729-94zpg3qc", "orthrus_cicids_kde_diff"),
    ("run-20260403_062042-o693lwyf", "kairos_cicids_kde_ts"),
    ("run-20260403_052853-xn9gc9ym", "orthrus_cicids_kde_ts"),
    ("run-20260403_022135-k1f4y9oo", "kairos_cicids"),
    ("run-20260403_014359-w7n1uqtr", "orthrus_cicids"),
]

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


def parse_log(filepath):
    """
    Parse a wandb output.log and return structured data per epoch.

    Returns:
        dict: epoch_num -> {
            'stats': { attack_name: {tp, fp, tn, fn, precision, recall, accuracy} },
            'tp_node_ids': { attack_name: [ (node_id, ip_info), ... ] }
        }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    epochs = {}

    # --- Phase 1: Parse per-attack stats from the Stats blocks ---
    # Pattern: [@model_epoch_X] - Stats  followed by per-attack metrics
    stats_re = re.compile(r"\[@model_epoch_(\d+)\] - Stats")
    metric_re = re.compile(r"- (graph_\w+)/(tp|fp|tn|fn|precision|recall|accuracy):\s+([\d.eE+-]+)")

    current_epoch = None
    for line in lines:
        m = stats_re.search(line)
        if m:
            current_epoch = int(m.group(1))
            if current_epoch not in epochs:
                epochs[current_epoch] = {"stats": defaultdict(dict), "tp_node_ids": defaultdict(list)}
            continue

        if current_epoch is not None:
            m2 = metric_re.search(line)
            if m2:
                attack = m2.group(1)
                metric = m2.group(2)
                value = float(m2.group(3))
                if metric in ("tp", "fp", "tn", "fn"):
                    value = int(value)
                epochs[current_epoch]["stats"][attack][metric] = value

    # --- Phase 2: Parse TP node IDs per attack per epoch ---
    # The log structure is:
    #   [@model_epoch_X] - Test Evaluation (ATTACK_NAME, TWs=...)
    #   ... malicious node lines ...
    #   "TPs per attack:"
    #   [@model_epoch_X] - Test Evaluation (NEXT_ATTACK, ...) or [@model_epoch_X] - Stats
    #
    # We track current epoch+attack context and collect TP nodes.

    eval_header_re = re.compile(
        r"\[@model_epoch_(\d+)\] - Test Evaluation \((graph_\w+), TWs=(\d+)\)"
    )
    tp_node_re = re.compile(
        r"-> Malicious node (\d+)\s+: loss=[\d.]+ \| is TP: ✅ (.+)"
    )

    current_epoch_ctx = None
    current_attack_ctx = None

    for line in lines:
        m = eval_header_re.search(line)
        if m:
            current_epoch_ctx = int(m.group(1))
            current_attack_ctx = m.group(2)
            continue

        if current_epoch_ctx is not None and current_attack_ctx is not None:
            m2 = tp_node_re.search(line)
            if m2:
                node_id = m2.group(1).strip()
                ip_info = m2.group(2).strip()
                epochs[current_epoch_ctx]["tp_node_ids"][current_attack_ctx].append(
                    (node_id, ip_info)
                )

    return dict(epochs)


def select_best_epoch(epochs_data):
    """
    Select best epoch according to the criteria:
    1. Highest number of attacks detected (attacks with TP > 0)
    2. Highest number of total unique malicious node IDs detected across attacks
    3. Highest total TP count across attacks
    4. Random (pick first from tied set)
    """
    epoch_scores = []
    for epoch_num, data in epochs_data.items():
        stats = data["stats"]
        tp_nodes = data["tp_node_ids"]

        # Criterion 1: number of attacks with TP > 0
        attacks_detected = sum(1 for a in ATTACKS if stats.get(a, {}).get("tp", 0) > 0)

        # Criterion 2: total unique malicious node IDs across all attacks
        unique_tp_node_ids = set()
        for a in ATTACKS:
            for node_id, _ in tp_nodes.get(a, []):
                unique_tp_node_ids.add(node_id)
        unique_tp_count = len(unique_tp_node_ids)

        # Criterion 3: total TP count across all attacks
        total_tp = sum(stats.get(a, {}).get("tp", 0) for a in ATTACKS)

        epoch_scores.append((attacks_detected, unique_tp_count, total_tp, epoch_num))

    # Sort descending by criteria 1, 2, 3; then by epoch_num ascending for determinism
    epoch_scores.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))

    best = epoch_scores[0]
    return best[3]  # epoch_num


def compute_summary(epochs_data, epoch_num):
    """Return (attacks_detected, unique_tp_count, total_tp) for a given epoch."""
    data = epochs_data[epoch_num]
    stats = data["stats"]
    tp_nodes = data["tp_node_ids"]
    attacks_detected = sum(1 for a in ATTACKS if stats.get(a, {}).get("tp", 0) > 0)
    total_tp = sum(stats.get(a, {}).get("tp", 0) for a in ATTACKS)
    unique_ids = set()
    for a in ATTACKS:
        for nid, _ in tp_nodes.get(a, []):
            unique_ids.add(nid)
    return attacks_detected, len(unique_ids), total_tp


def main():
    output_rows = []
    best_summary_rows = []   # for the compact summary CSV
    last_summary_rows = []   # for the compact summary CSV

    print("=" * 80)
    print("Best Epoch Selection Results")
    print("=" * 80)

    for run_dir, config_name in RUNS:
        log_path = os.path.join(WANDB_DIR, run_dir, "files", "output.log")
        if not os.path.exists(log_path):
            print(f"WARNING: {log_path} not found, skipping.")
            continue

        epochs_data = parse_log(log_path)
        best_epoch = select_best_epoch(epochs_data)
        best_data = epochs_data[best_epoch]

        # Print summary
        stats = best_data["stats"]
        tp_nodes = best_data["tp_node_ids"]
        attacks_detected, unique_tp_count, total_tp = compute_summary(epochs_data, best_epoch)

        best_summary_rows.append({
            "config": config_name,
            "best_epoch": best_epoch,
            "attacks_detected": f"{attacks_detected}/{len(ATTACKS)}",
            "unique_tp_nodes": unique_tp_count,
            "total_tp": total_tp,
        })

        print(f"\n{config_name} ({run_dir})")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Attacks detected: {attacks_detected}/{len(ATTACKS)}")
        print(f"  Total TP (sum): {total_tp}")
        print(f"  Unique TP node IDs: {unique_tp_count}")

        # Print per-epoch summary for context
        print(f"  All epochs summary:")
        for ep in sorted(epochs_data.keys()):
            ep_stats = epochs_data[ep]["stats"]
            ep_tp_nodes = epochs_data[ep]["tp_node_ids"]
            ep_atk = sum(1 for a in ATTACKS if ep_stats.get(a, {}).get("tp", 0) > 0)
            ep_total_tp = sum(ep_stats.get(a, {}).get("tp", 0) for a in ATTACKS)
            ep_unique = set()
            for a in ATTACKS:
                for nid, _ in ep_tp_nodes.get(a, []):
                    ep_unique.add(nid)
            marker = " <-- BEST" if ep == best_epoch else ""
            print(f"    Epoch {ep:2d}: attacks_detected={ep_atk}, unique_tp_nodes={len(ep_unique)}, total_tp={ep_total_tp}{marker}")

        # Build CSV rows for this run
        for attack in ATTACKS:
            s = stats.get(attack, {})
            tp_val = s.get("tp", 0)
            fp_val = s.get("fp", 0)
            fn_val = s.get("fn", 0)
            tn_val = s.get("tn", 0)
            prec_val = s.get("precision", 0.0)
            rec_val = s.get("recall", 0.0)
            acc_val = s.get("accuracy", 0.0)

            # TP node IDs for this attack
            tp_node_list = tp_nodes.get(attack, [])
            tp_node_ids_str = "; ".join(
                f"{nid} ({ip})" for nid, ip in tp_node_list
            ) if tp_node_list else ""

            output_rows.append({
                "config": config_name,
                "run_dir": run_dir,
                "best_epoch": best_epoch,
                "attack": attack,
                "TP": tp_val,
                "FP": fp_val,
                "FN": fn_val,
                "TN": tn_val,
                "precision": prec_val,
                "recall": rec_val,
                "accuracy": acc_val,
                "TP_node_IDs": tp_node_ids_str,
            })

    # Write CSV for best epoch
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_epoch_results.csv")
    fieldnames = [
        "config", "run_dir", "best_epoch", "attack",
        "TP", "FP", "FN", "TN",
        "precision", "recall", "accuracy",
        "TP_node_IDs",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\n{'=' * 80}")
    print(f"CSV written to: {csv_path}")
    print(f"Total rows: {len(output_rows)} ({len(RUNS)} runs x {len(ATTACKS)} attacks)")
    print(f"{'=' * 80}")

    # ============================================================
    # Generate last epoch (epoch 11) results table
    # ============================================================
    print(f"\n{'=' * 80}")
    print("Last Epoch (Epoch 11) Results")
    print("=" * 80)

    last_epoch_rows = []
    LAST_EPOCH = 11

    for run_dir, config_name in RUNS:
        log_path = os.path.join(WANDB_DIR, run_dir, "files", "output.log")
        if not os.path.exists(log_path):
            continue

        epochs_data = parse_log(log_path)
        if LAST_EPOCH not in epochs_data:
            print(f"WARNING: Epoch {LAST_EPOCH} not found in {config_name}, skipping.")
            continue

        last_data = epochs_data[LAST_EPOCH]
        stats = last_data["stats"]
        tp_nodes = last_data["tp_node_ids"]

        # Print summary
        attacks_detected, unique_tp_count, total_tp = compute_summary(epochs_data, LAST_EPOCH)

        last_summary_rows.append({
            "config": config_name,
            "epoch": LAST_EPOCH,
            "attacks_detected": f"{attacks_detected}/{len(ATTACKS)}",
            "unique_tp_nodes": unique_tp_count,
            "total_tp": total_tp,
        })

        print(f"\n{config_name} ({run_dir})")
        print(f"  Epoch: {LAST_EPOCH}")
        print(f"  Attacks detected: {attacks_detected}/{len(ATTACKS)}")
        print(f"  Total TP (sum): {total_tp}")
        print(f"  Unique TP node IDs: {unique_tp_count}")

        # Build CSV rows for this run
        for attack in ATTACKS:
            s = stats.get(attack, {})
            tp_val = s.get("tp", 0)
            fp_val = s.get("fp", 0)
            fn_val = s.get("fn", 0)
            tn_val = s.get("tn", 0)
            prec_val = s.get("precision", 0.0)
            rec_val = s.get("recall", 0.0)
            acc_val = s.get("accuracy", 0.0)

            # TP node IDs for this attack
            tp_node_list = tp_nodes.get(attack, [])
            tp_node_ids_str = "; ".join(
                f"{nid} ({ip})" for nid, ip in tp_node_list
            ) if tp_node_list else ""

            last_epoch_rows.append({
                "config": config_name,
                "run_dir": run_dir,
                "epoch": LAST_EPOCH,
                "attack": attack,
                "TP": tp_val,
                "FP": fp_val,
                "FN": fn_val,
                "TN": tn_val,
                "precision": prec_val,
                "recall": rec_val,
                "accuracy": acc_val,
                "TP_node_IDs": tp_node_ids_str,
            })

    # Write CSV for last epoch
    last_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_epoch_results.csv")
    last_fieldnames = [
        "config", "run_dir", "epoch", "attack",
        "TP", "FP", "FN", "TN",
        "precision", "recall", "accuracy",
        "TP_node_IDs",
    ]
    with open(last_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=last_fieldnames)
        writer.writeheader()
        writer.writerows(last_epoch_rows)

    print(f"\n{'=' * 80}")
    print(f"CSV written to: {last_csv_path}")
    print(f"Total rows: {len(last_epoch_rows)} ({len(RUNS)} runs x {len(ATTACKS)} attacks)")
    print(f"{'=' * 80}")

    # ============================================================
    # Write compact summary CSVs
    # ============================================================
    summary_fieldnames = ["config", "best_epoch", "attacks_detected", "unique_tp_nodes", "total_tp"]
    best_summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_epoch_summary.csv")
    with open(best_summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(best_summary_rows)
    print(f"\nSummary CSV (best epoch) written to: {best_summary_path}")

    last_summary_fieldnames = ["config", "epoch", "attacks_detected", "unique_tp_nodes", "total_tp"]
    last_summary_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "last_epoch_summary.csv")
    with open(last_summary_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=last_summary_fieldnames)
        writer.writeheader()
        writer.writerows(last_summary_rows)
    print(f"Summary CSV (last epoch) written to: {last_summary_path}")


if __name__ == "__main__":
    main()
