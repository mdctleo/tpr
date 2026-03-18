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
import glob

BASE_DIR = "/scratch/asawan15/PIDSMaker"
OUTPUT_DIR = os.path.join(BASE_DIR, "node_classifications")

# ============================================================
# Artifacts directories for different config types
# ============================================================
ARTIFACTS_DIRS = {
    # Original KDE configs (artifacts_base_and_kde_ts)
    "artifacts_base_and_kde_ts": os.path.join(BASE_DIR, "artifacts_base_and_kde_ts"),
    # KDE diff configs (timestamp differences)
    "artifacts_reduced_diff": os.path.join(BASE_DIR, "artifacts_reduced_diff"),
    # Reduced graphs (temporal summary features)
    "artifacts_reduced_all": os.path.join(BASE_DIR, "artifacts_reduced_all"),
}

# ============================================================
# Static config mappings (known hash values)
# For new runs, use auto-discovery mode with discover_runs()
# ============================================================

STATIC_CONFIGS = [
    # --- orthrus_non_snooped_edge_ts (artifacts_base_and_kde_ts/artifacts/) ---
    {
        "config": "orthrus_non_snooped_edge_ts",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 0,
        "artifacts_base": "artifacts_base_and_kde_ts",
        "artifact_subdir": "artifacts",
        "hash": "293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e",
    },
    {
        "config": "orthrus_non_snooped_edge_ts",
        "dataset": "CADETS_E3",
        "epoch": 5,
        "artifacts_base": "artifacts_base_and_kde_ts",
        "artifact_subdir": "artifacts",
        "hash": "970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727",
    },
    {
        "config": "orthrus_non_snooped_edge_ts",
        "dataset": "THEIA_E3",
        "epoch": 1,
        "artifacts_base": "artifacts_base_and_kde_ts",
        "artifact_subdir": "artifacts",
        "hash": "9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457",
    },
    # --- kairos (artifacts_base_and_kde_ts/artifacts/) ---
    {
        "config": "kairos",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 3,
        "artifacts_base": "artifacts_base_and_kde_ts",
        "artifact_subdir": "artifacts",
        "hash": "2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7",
    },
    {
        "config": "kairos",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifacts_base": "artifacts_base_and_kde_ts",
        "artifact_subdir": "artifacts",
        "hash": "133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269",
    },
    {
        "config": "kairos",
        "dataset": "THEIA_E3",
        "epoch": 0,
        "artifacts_base": "artifacts_base_and_kde_ts",
        "artifact_subdir": "artifacts",
        "hash": "e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5",
    },
    # --- orthrus_kde_diff (artifacts_reduced_diff/) ---
    {
        "config": "orthrus_kde_diff",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 0,
        "artifacts_base": "artifacts_reduced_diff",
        "artifact_subdir": "",
        "hash": "2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7",
    },
    {
        "config": "orthrus_kde_diff",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifacts_base": "artifacts_reduced_diff",
        "artifact_subdir": "",
        "hash": "970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727",
    },
    {
        "config": "orthrus_kde_diff",
        "dataset": "THEIA_E3",
        "epoch": 0,
        "artifacts_base": "artifacts_reduced_diff",
        "artifact_subdir": "",
        "hash": "9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457",
    },
    # --- kairos_kde_diff (artifacts_reduced_diff/) ---
    {
        "config": "kairos_kde_diff",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 3,
        "artifacts_base": "artifacts_reduced_diff",
        "artifact_subdir": "",
        "hash": "293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e",
    },
    {
        "config": "kairos_kde_diff",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifacts_base": "artifacts_reduced_diff",
        "artifact_subdir": "",
        "hash": "133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269",
    },
    {
        "config": "kairos_kde_diff",
        "dataset": "THEIA_E3",
        "epoch": 1,
        "artifacts_base": "artifacts_reduced_diff",
        "artifact_subdir": "",
        "hash": "e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5",
    },
    # --- orthrus_red (artifacts_reduced_all/) ---
    {
        "config": "orthrus_red",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 5,
        "artifacts_base": "artifacts_reduced_all",
        "artifact_subdir": "",
        "hash": "293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e",
    },
    {
        "config": "orthrus_red",
        "dataset": "CADETS_E3",
        "epoch": 11,
        "artifacts_base": "artifacts_reduced_all",
        "artifact_subdir": "",
        "hash": "970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727",
    },
    {
        "config": "orthrus_red",
        "dataset": "THEIA_E3",
        "epoch": 3,
        "artifacts_base": "artifacts_reduced_all",
        "artifact_subdir": "",
        "hash": "9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457",
    },
    # --- kairos_red (artifacts_reduced_all/) ---
    {
        "config": "kairos_red",
        "dataset": "CLEARSCOPE_E3",
        "epoch": 7,
        "artifacts_base": "artifacts_reduced_all",
        "artifact_subdir": "",
        "hash": "2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7",
    },
    {
        "config": "kairos_red",
        "dataset": "CADETS_E3",
        "epoch": 1,
        "artifacts_base": "artifacts_reduced_all",
        "artifact_subdir": "",
        "hash": "133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269",
    },
    {
        "config": "kairos_red",
        "dataset": "THEIA_E3",
        "epoch": 3,
        "artifacts_base": "artifacts_reduced_all",
        "artifact_subdir": "",
        "hash": "e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5",
    },
]

# ============================================================
# Auto-discovery for reduced graph runs (orthrus_red, kairos_red)
# ============================================================

def discover_reduced_runs(artifacts_base_dir, config_patterns=None):
    """
    Auto-discover evaluation runs in an artifacts directory.
    
    Looks for evaluation results at:
    {artifacts_base_dir}/evaluation/evaluation/{hash}/{dataset}/precision_recall_dir/
    
    Args:
        artifacts_base_dir: Base directory to search
        config_patterns: List of config name patterns to match (optional)
        
    Returns:
        List of config dicts with discovered runs
    """
    configs = []
    eval_base = os.path.join(artifacts_base_dir, "evaluation", "evaluation")
    
    if not os.path.exists(eval_base):
        print(f"  No evaluation directory found: {eval_base}")
        return configs
    
    # List hash directories
    for hash_dir in os.listdir(eval_base):
        hash_path = os.path.join(eval_base, hash_dir)
        if not os.path.isdir(hash_path):
            continue
        
        # List dataset directories within hash
        for dataset in os.listdir(hash_path):
            dataset_path = os.path.join(hash_path, dataset)
            pr_dir = os.path.join(dataset_path, "precision_recall_dir")
            
            if not os.path.exists(pr_dir):
                continue
            
            # Find result files and get best epoch
            result_files = glob.glob(os.path.join(pr_dir, "result_model_epoch_*.pth"))
            if not result_files:
                continue
            
            # Get best epoch from stats files (look for best F1 score)
            best_epoch = find_best_epoch(pr_dir)
            
            # Try to determine config name from run metadata
            config_name = determine_config_name(hash_path, dataset)
            
            configs.append({
                "config": config_name,
                "dataset": dataset,
                "epoch": best_epoch,
                "artifacts_base": os.path.basename(artifacts_base_dir),
                "artifact_subdir": "",
                "hash": hash_dir,
                "auto_discovered": True,
            })
    
    return configs


def find_best_epoch(pr_dir):
    """Find the epoch with best F1 score in the precision_recall_dir."""
    best_epoch = 0
    best_f1 = -1
    
    stats_files = glob.glob(os.path.join(pr_dir, "stats_model_epoch_*.pth"))
    
    for stats_file in stats_files:
        try:
            epoch = int(stats_file.split("epoch_")[1].split(".pth")[0])
            stats = torch.load(stats_file, map_location="cpu")
            f1 = stats.get('fscore', 0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
        except Exception:
            continue
    
    return best_epoch


def determine_config_name(hash_path, dataset):
    """Try to determine config name from run metadata or path."""
    # Try to read from a config.json or similar metadata file
    for metadata_file in ["config.json", "run_config.json", "metadata.json"]:
        meta_path = os.path.join(hash_path, dataset, metadata_file)
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                if "config" in meta:
                    return meta["config"]
            except Exception:
                pass
    
    # Fallback: try to infer from parent directory structure
    parent = os.path.dirname(os.path.dirname(os.path.dirname(hash_path)))
    parent_name = os.path.basename(parent)
    
    if "reduced_all" in parent_name:
        return "reduced_graph"  # Will need manual mapping later
    
    return "unknown"


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
    """
    Main function to extract node classifications.
    
    Processes both static configs and auto-discovers runs from artifacts_reduced_all.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract node classifications from evaluation results")
    parser.add_argument("--discover", action="store_true", 
                        help="Auto-discover runs from artifacts_reduced_all")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Config name to assign to discovered runs (e.g., 'orthrus_red', 'kairos_red')")
    parser.add_argument("--artifacts-dir", type=str, default=None,
                        help="Artifacts directory to search for runs")
    args = parser.parse_args()
    
    overall_summary = {}
    
    # Determine which configs to process
    configs_to_process = list(STATIC_CONFIGS)
    
    # Auto-discover runs from artifacts_reduced_all if requested
    if args.discover or args.artifacts_dir:
        artifacts_dir = args.artifacts_dir or os.path.join(BASE_DIR, "artifacts_reduced_all")
        print(f"\n{'='*80}")
        print(f"Auto-discovering runs from: {artifacts_dir}")
        print(f"{'='*80}")
        
        discovered = discover_reduced_runs(artifacts_dir)
        
        # Assign config name if provided
        if args.config_name:
            for cfg in discovered:
                cfg["config"] = args.config_name
        
        print(f"  Discovered {len(discovered)} runs")
        for cfg in discovered:
            print(f"    - {cfg['config']}/{cfg['dataset']} (epoch={cfg['epoch']}, hash={cfg['hash'][:12]}...)")
        
        configs_to_process.extend(discovered)

    for cfg in configs_to_process:
        config_name = cfg["config"]
        dataset = cfg["dataset"]
        epoch = cfg["epoch"]
        hash_dir = cfg["hash"]
        
        # Build result path based on config structure
        artifacts_base = cfg.get("artifacts_base", "artifacts_reduced_all")
        artifact_subdir = cfg.get("artifact_subdir", "")
        
        # Determine base path
        if artifacts_base in ARTIFACTS_DIRS:
            base_path = ARTIFACTS_DIRS[artifacts_base]
        else:
            base_path = os.path.join(BASE_DIR, artifacts_base)
        
        if artifact_subdir:
            result_path = os.path.join(
                base_path,
                artifact_subdir,
                "evaluation",
                "evaluation",
                hash_dir,
                dataset,
                "precision_recall_dir",
                f"result_model_epoch_{epoch}.pth",
            )
            stats_path = os.path.join(
                base_path,
                artifact_subdir,
                "evaluation",
                "evaluation",
                hash_dir,
                dataset,
                "precision_recall_dir",
                f"stats_model_epoch_{epoch}.pth",
            )
            artifact_display = f"{artifacts_base}/{artifact_subdir}"
        else:
            result_path = os.path.join(
                base_path,
                "evaluation",
                "evaluation",
                hash_dir,
                dataset,
                "precision_recall_dir",
                f"result_model_epoch_{epoch}.pth",
            )
            stats_path = os.path.join(
                base_path,
                "evaluation",
                "evaluation",
                hash_dir,
                dataset,
                "precision_recall_dir",
                f"stats_model_epoch_{epoch}.pth",
            )
            artifact_display = artifacts_base

        print(f"\n{'='*80}")
        print(f"Config: {config_name} | Dataset: {dataset} | Epoch: {epoch}")
        print(f"Artifact dir: {artifact_display}/evaluation/evaluation/{hash_dir[:12]}...")
        print(f"Result file: {result_path}")

        if not os.path.exists(result_path):
            print(f"  ERROR: Result file not found!")
            continue

        tp_nodes, fp_nodes, tn_nodes, fn_nodes = extract_node_classifications(
            result_path
        )

        # Load stats for threshold
        if os.path.exists(stats_path):
            stats = torch.load(stats_path, map_location="cpu")
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
            "artifacts_base": artifacts_base,
            "artifact_subdir": artifact_subdir,
            "hash": hash_dir,
            "TP_node_ids": sorted([n["node_id"] for n in tp_nodes]),
            "FP_node_ids": sorted([n["node_id"] for n in fp_nodes]),
            "TN_node_ids_count": len(tn_nodes),
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
            f.write(f"Artifact Dir: {artifact_display}/evaluation/evaluation/{hash_dir[:12]}...\n")
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
