#!/usr/bin/env python3
"""
Extract node classifications (TP, FP, TN, FN) from evaluation result files.

For each config/dataset/epoch combination, loads the result .pth file and
classifies every node based on:
  - y_hat=1 → predicted malicious (above threshold)
  - y_hat=0 → predicted benign (below threshold)
  - y_true=1 → actually malicious
  - y_true=0 → actually benign

Categories:
  TP (True Positive):  y_true=1, y_hat=1  (malicious, correctly detected)
  FP (False Positive): y_true=0, y_hat=1  (benign, incorrectly flagged)
  TN (True Negative):  y_true=0, y_hat=0  (benign, correctly ignored)
  FN (False Negative): y_true=1, y_hat=0  (malicious, missed)
"""

import torch
import json
import os
import csv

BASE_DIR = "/scratch/asawan15/PIDSMaker"
ARTIFACTS_BASE = os.path.join(BASE_DIR, "artifacts_base_and_kde_ts")
OUTPUT_DIR = os.path.join(BASE_DIR, "node_classifications")

# ============================================================
# Config mappings using the artifacts_base_and_kde_ts folder
# Path structure: evaluation/evaluation/{hash}/{dataset}/precision_recall_dir/
# ============================================================

# artifacts_base_and_kde_ts/artifacts/: orthrus_non_snooped_edge_ts and kairos
# artifacts_base_and_kde_ts/artifacts_reduced/: orthrus_edge_kde_ts and kairos_kde_ts

# Epoch mappings (from user specification):
# orthrus_non_snooped_edge_ts: clearscope=0, cadets=5, theia=1
# orthrus_edge_kde_ts: clearscope=0, cadets=7, theia=3
# kairos_kde_ts: clearscope=1, cadets=5, theia=1
# kairos: clearscope=3, cadets=11, theia=0

CONFIGS = [
    # --- orthrus_non_snooped_edge_ts (artifacts/) ---
    {
        "config": "orthrus_non_snooped_edge_ts",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 0,
        "artifact_dir": "artifacts",
        "hash": "293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e",
    },
    {
        "config": "orthrus_non_snooped_edge_ts",
        "dataset": "CADETS_E3",
        "epoch": 5,
        "artifact_dir": "artifacts",
        "hash": "970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727",
    },
    {
        "config": "orthrus_non_snooped_edge_ts",
        "dataset": "THEIA_E3",
        "epoch": 1,
        "artifact_dir": "artifacts",
        "hash": "9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457",
    },
    # --- orthrus_edge_kde_ts (artifacts_reduced/) ---
    {
        "config": "orthrus_edge_kde_ts",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 0,
        "artifact_dir": "artifacts_reduced",
        "hash": "2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7",
    },
    {
        "config": "orthrus_edge_kde_ts",
        "dataset": "CADETS_E3",
        "epoch": 7,
        "artifact_dir": "artifacts_reduced",
        "hash": "970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727",
    },
    {
        "config": "orthrus_edge_kde_ts",
        "dataset": "THEIA_E3",
        "epoch": 3,
        "artifact_dir": "artifacts_reduced",
        "hash": "9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457",
    },
    # --- kairos_kde_ts (artifacts_reduced/) ---
    {
        "config": "kairos_kde_ts",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 1,
        "artifact_dir": "artifacts_reduced",
        "hash": "293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e",
    },
    {
        "config": "kairos_kde_ts",
        "dataset": "CADETS_E3",
        "epoch": 5,
        "artifact_dir": "artifacts_reduced",
        "hash": "133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269",
    },
    {
        "config": "kairos_kde_ts",
        "dataset": "THEIA_E3",
        "epoch": 1,
        "artifact_dir": "artifacts_reduced",
        "hash": "e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5",
    },
    # --- kairos (artifacts/) ---
    {
        "config": "kairos",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 3,
        "artifact_dir": "artifacts",
        "hash": "2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7",
    },
    {
        "config": "kairos",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifact_dir": "artifacts",
        "hash": "133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269",
    },
    {
        "config": "kairos",
        "dataset": "THEIA_E3",
        "epoch": 0,
        "artifact_dir": "artifacts",
        "hash": "e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5",
    },
    # --- orthrus_kde_diff (artifacts_reduced_diff/) ---
    {
        "config": "orthrus_kde_diff",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 0,
        "artifact_dir": "artifacts_reduced_diff",
        "hash": "2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7",
    },
    {
        "config": "orthrus_kde_diff",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifact_dir": "artifacts_reduced_diff",
        "hash": "970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727",
    },
    {
        "config": "orthrus_kde_diff",
        "dataset": "THEIA_E3",
        "epoch": 0,
        "artifact_dir": "artifacts_reduced_diff",
        "hash": "9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457",
    },
    # --- kairos_kde_diff (artifacts_reduced_diff/) ---
    {
        "config": "kairos_kde_diff",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 3,
        "artifact_dir": "artifacts_reduced_diff",
        "hash": "293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e",
    },
    {
        "config": "kairos_kde_diff",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifact_dir": "artifacts_reduced_diff",
        "hash": "133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269",
    },
    {
        "config": "kairos_kde_diff",
        "dataset": "THEIA_E3",
        "epoch": 1,
        "artifact_dir": "artifacts_reduced_diff",
        "hash": "e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5",
    },
]


def extract_node_classifications(result_path):
    """Load a result .pth file and classify all nodes into TP/FP/TN/FN."""
    result = torch.load(result_path, map_location="cpu")

    tp_nodes = []  # y_true=1, y_hat=1
    fp_nodes = []  # y_true=0, y_hat=1
    tn_nodes = []  # y_true=0, y_hat=0
    fn_nodes = []  # y_true=1, y_hat=0

    for node_id, info in result.items():
        y_true = info["y_true"]
        y_hat = info["y_hat"]
        score = info["score"]

        node_entry = {"node_id": node_id, "score": score}

        if y_true == 1 and y_hat == 1:
            tp_nodes.append(node_entry)
        elif y_true == 0 and y_hat == 1:
            fp_nodes.append(node_entry)
        elif y_true == 0 and y_hat == 0:
            tn_nodes.append(node_entry)
        elif y_true == 1 and y_hat == 0:
            fn_nodes.append(node_entry)

    # Sort each list by score descending
    tp_nodes.sort(key=lambda x: x["score"], reverse=True)
    fp_nodes.sort(key=lambda x: x["score"], reverse=True)
    fn_nodes.sort(key=lambda x: x["score"], reverse=True)
    tn_nodes.sort(key=lambda x: x["score"], reverse=True)

    return tp_nodes, fp_nodes, tn_nodes, fn_nodes


def write_csv(filepath, nodes, category_label):
    """Write node list to CSV."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "score", "category"])
        for n in nodes:
            writer.writerow([n["node_id"], f"{n['score']:.6f}", category_label])


def main():
    overall_summary = {}

    for cfg in CONFIGS:
        config_name = cfg["config"]
        dataset = cfg["dataset"]
        epoch = cfg["epoch"]
        artifact_dir = cfg["artifact_dir"]
        hash_dir = cfg["hash"]

        result_path = os.path.join(
            ARTIFACTS_BASE,
            artifact_dir,
            "evaluation",
            "evaluation",
            hash_dir,
            dataset,
            "precision_recall_dir",
            f"result_model_epoch_{epoch}.pth",
        )

        # Also load the stats file for threshold info
        stats_path = os.path.join(
            ARTIFACTS_BASE,
            artifact_dir,
            "evaluation",
            "evaluation",
            hash_dir,
            dataset,
            "precision_recall_dir",
            f"stats_model_epoch_{epoch}.pth",
        )

        print(f"\n{'='*80}")
        print(f"Config: {config_name} | Dataset: {dataset} | Epoch: {epoch}")
        print(f"Artifact dir: {artifact_dir}/evaluation/evaluation/{hash_dir[:12]}...")
        print(f"Result file: {result_path}")

        if not os.path.exists(result_path):
            print(f"  ERROR: Result file not found!")
            continue

        tp_nodes, fp_nodes, tn_nodes, fn_nodes = extract_node_classifications(
            result_path
        )

        # Load stats for threshold
        threshold = None
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
            # The threshold is implicitly embedded in the y_hat predictions
            # Print key stats
            print(f"  Stats: TP={stats.get('tp',0)}, FP={stats.get('fp',0)}, "
                  f"TN={stats.get('tn',0)}, FN={stats.get('fn',0)}")
            print(f"  Precision={stats.get('precision',0):.5f}, "
                  f"Recall={stats.get('recall',0):.5f}, "
                  f"F1={stats.get('fscore',0):.5f}")

        # Verify counts match stats
        print(f"  Extracted: TP={len(tp_nodes)}, FP={len(fp_nodes)}, "
              f"TN={len(tn_nodes)}, FN={len(fn_nodes)}")
        print(f"  Total malicious (above threshold): {len(tp_nodes) + len(fp_nodes)}")
        print(f"  Total benign (below threshold): {len(tn_nodes) + len(fn_nodes)}")
        print(f"  Total nodes: {len(tp_nodes) + len(fp_nodes) + len(tn_nodes) + len(fn_nodes)}")

        # Create output directory
        out_dir = os.path.join(OUTPUT_DIR, config_name, dataset)
        os.makedirs(out_dir, exist_ok=True)

        # Write individual CSV files for each category
        write_csv(os.path.join(out_dir, "true_positives.csv"), tp_nodes, "TP")
        write_csv(os.path.join(out_dir, "false_positives.csv"), fp_nodes, "FP")
        write_csv(os.path.join(out_dir, "true_negatives.csv"), tn_nodes, "TN")
        write_csv(os.path.join(out_dir, "false_negatives.csv"), fn_nodes, "FN")

        # Write combined malicious (above threshold) = TP + FP
        malicious_nodes = [
            {**n, "category": "TP"} for n in tp_nodes
        ] + [
            {**n, "category": "FP"} for n in fp_nodes
        ]
        malicious_nodes.sort(key=lambda x: x["score"], reverse=True)
        with open(os.path.join(out_dir, "malicious_above_threshold.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "score", "category"])
            for n in malicious_nodes:
                writer.writerow([n["node_id"], f"{n['score']:.6f}", n["category"]])

        # Write combined benign (below threshold) = TN + FN
        benign_nodes = [
            {**n, "category": "FN"} for n in fn_nodes
        ] + [
            {**n, "category": "TN"} for n in tn_nodes
        ]
        benign_nodes.sort(key=lambda x: x["score"], reverse=True)
        with open(os.path.join(out_dir, "benign_below_threshold.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["node_id", "score", "category"])
            for n in benign_nodes:
                writer.writerow([n["node_id"], f"{n['score']:.6f}", n["category"]])

        # Write JSON with just node ID lists
        node_id_lists = {
            "config": config_name,
            "dataset": dataset,
            "epoch": epoch,
            "artifact_dir": artifact_dir,
            "hash": hash_dir,
            "TP_node_ids": sorted([n["node_id"] for n in tp_nodes]),
            "FP_node_ids": sorted([n["node_id"] for n in fp_nodes]),
            "TN_node_ids_count": len(tn_nodes),  # Too many to list individually
            "FN_node_ids": sorted([n["node_id"] for n in fn_nodes]),
            "malicious_node_ids": sorted([n["node_id"] for n in tp_nodes + fp_nodes]),
            "counts": {
                "TP": len(tp_nodes),
                "FP": len(fp_nodes),
                "TN": len(tn_nodes),
                "FN": len(fn_nodes),
                "total_malicious_above_threshold": len(tp_nodes) + len(fp_nodes),
                "total_benign_below_threshold": len(tn_nodes) + len(fn_nodes),
                "total_nodes": len(tp_nodes) + len(fp_nodes) + len(tn_nodes) + len(fn_nodes),
            },
        }
        with open(os.path.join(out_dir, "node_id_lists.json"), "w") as f:
            json.dump(node_id_lists, f, indent=2)

        # Write a human-readable summary
        with open(os.path.join(out_dir, "summary.txt"), "w") as f:
            f.write(f"Node Classification Summary\n")
            f.write(f"{'='*60}\n")
            f.write(f"Config:       {config_name}\n")
            f.write(f"Dataset:      {dataset}\n")
            f.write(f"Epoch:        {epoch}\n")
            f.write(f"Artifact Dir: {artifact_dir}/evaluation/evaluation/{hash_dir[:12]}...\n")
            f.write(f"\n")
            f.write(f"MALICIOUS (Above Threshold) - y_hat=1:\n")
            f.write(f"  True Positives (TP):  {len(tp_nodes):>8d} nodes\n")
            f.write(f"  False Positives (FP): {len(fp_nodes):>8d} nodes\n")
            f.write(f"  Total:                {len(tp_nodes)+len(fp_nodes):>8d} nodes\n")
            f.write(f"\n")
            f.write(f"BENIGN (Below Threshold) - y_hat=0:\n")
            f.write(f"  True Negatives (TN):  {len(tn_nodes):>8d} nodes\n")
            f.write(f"  False Negatives (FN): {len(fn_nodes):>8d} nodes\n")
            f.write(f"  Total:                {len(tn_nodes)+len(fn_nodes):>8d} nodes\n")
            f.write(f"\n")
            f.write(f"TOTAL NODES: {len(tp_nodes)+len(fp_nodes)+len(tn_nodes)+len(fn_nodes)}\n")
            f.write(f"\n")

            # List TP node IDs (usually small)
            f.write(f"--- True Positive Node IDs (score) ---\n")
            for n in tp_nodes:
                f.write(f"  {n['node_id']}: {n['score']:.6f}\n")
            f.write(f"\n")

            # List FP node IDs
            f.write(f"--- False Positive Node IDs (score) [first 200] ---\n")
            for n in fp_nodes[:200]:
                f.write(f"  {n['node_id']}: {n['score']:.6f}\n")
            if len(fp_nodes) > 200:
                f.write(f"  ... and {len(fp_nodes)-200} more (see CSV file)\n")
            f.write(f"\n")

            # List FN node IDs (usually small-ish)
            f.write(f"--- False Negative Node IDs (score) ---\n")
            for n in fn_nodes:
                f.write(f"  {n['node_id']}: {n['score']:.6f}\n")
            f.write(f"\n")

            f.write(f"--- True Negative Node IDs ---\n")
            f.write(f"  {len(tn_nodes)} nodes (see true_negatives.csv for full list)\n")

        # Track for overall summary
        if config_name not in overall_summary:
            overall_summary[config_name] = {}
        overall_summary[config_name][dataset] = {
            "epoch": epoch,
            "TP": len(tp_nodes),
            "FP": len(fp_nodes),
            "TN": len(tn_nodes),
            "FN": len(fn_nodes),
            "total_malicious": len(tp_nodes) + len(fp_nodes),
            "total_benign": len(tn_nodes) + len(fn_nodes),
            "total_nodes": len(tp_nodes) + len(fp_nodes) + len(tn_nodes) + len(fn_nodes),
        }

        print(f"  Output written to: {out_dir}")

    # Write overall summary JSON
    with open(os.path.join(OUTPUT_DIR, "overall_summary.json"), "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY")
    print(f"{'='*80}")
    for config_name, datasets in overall_summary.items():
        for dataset, counts in datasets.items():
            print(f"  {config_name:35s} {dataset:15s} epoch={counts['epoch']:2d}  "
                  f"TP={counts['TP']:5d}  FP={counts['FP']:6d}  "
                  f"TN={counts['TN']:7d}  FN={counts['FN']:4d}  "
                  f"Total={counts['total_nodes']}")

    print(f"\nAll results written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
