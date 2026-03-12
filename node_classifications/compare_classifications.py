#!/usr/bin/env python3
"""
Compare node classifications between two configs for each dataset.

For each dataset, finds nodes classified differently:
  - Malicious in A but Benign in B
  - Benign in A but Malicious in B

Uses malicious_above_threshold.csv and benign_below_threshold.csv.
"""

import csv
import json
import os

BASE_DIR = "/scratch/asawan15/PIDSMaker/node_classifications"

SCENARIOS = [
    {
        "name": "orthrus_edge_kde_ts_vs_orthrus_non_snooped_edge_ts",
        "A": "orthrus_edge_kde_ts",
        "B": "orthrus_non_snooped_edge_ts",
    },
    {
        "name": "kairos_kde_ts_vs_kairos",
        "A": "kairos_kde_ts",
        "B": "kairos",
    },
]

DATASETS = ["CLEARSCOPE_E3", "CADETS_E3", "THEIA_E3"]


def load_node_set_from_csv(csv_path):
    """Load node IDs from a CSV file into a dict {node_id: (score, category)}."""
    nodes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes[int(row["node_id"])] = {
                "score": float(row["score"]),
                "category": row["category"],
            }
    return nodes


def compare_configs(config_a, config_b, dataset):
    """Compare classifications between config A and config B for a dataset."""
    mal_a_path = os.path.join(BASE_DIR, config_a, dataset, "malicious_above_threshold.csv")
    ben_a_path = os.path.join(BASE_DIR, config_a, dataset, "benign_below_threshold.csv")
    mal_b_path = os.path.join(BASE_DIR, config_b, dataset, "malicious_above_threshold.csv")
    ben_b_path = os.path.join(BASE_DIR, config_b, dataset, "benign_below_threshold.csv")

    mal_a = load_node_set_from_csv(mal_a_path)
    ben_a = load_node_set_from_csv(ben_a_path)
    mal_b = load_node_set_from_csv(mal_b_path)
    ben_b = load_node_set_from_csv(ben_b_path)

    mal_a_ids = set(mal_a.keys())
    ben_a_ids = set(ben_a.keys())
    mal_b_ids = set(mal_b.keys())
    ben_b_ids = set(ben_b.keys())

    # Nodes malicious in A but benign in B
    mal_in_a_ben_in_b = mal_a_ids & ben_b_ids
    # Nodes benign in A but malicious in B
    ben_in_a_mal_in_b = ben_a_ids & mal_b_ids
    # Total differently classified
    differently_classified = mal_in_a_ben_in_b | ben_in_a_mal_in_b

    return {
        "mal_a": mal_a,
        "ben_a": ben_a,
        "mal_b": mal_b,
        "ben_b": ben_b,
        "mal_in_a_ben_in_b": mal_in_a_ben_in_b,
        "ben_in_a_mal_in_b": ben_in_a_mal_in_b,
        "differently_classified": differently_classified,
    }


def write_diff_csv(filepath, node_ids, info_a, info_b, config_a, config_b):
    """Write a CSV with the differently classified nodes and their details."""
    rows = []
    for nid in sorted(node_ids):
        a_info = info_a.get(nid, {})
        b_info = info_b.get(nid, {})
        rows.append({
            "node_id": nid,
            f"{config_a}_classification": "malicious" if nid in info_a else "benign",
            f"{config_a}_score": a_info.get("score", ""),
            f"{config_a}_category": a_info.get("category", ""),
            f"{config_b}_classification": "malicious" if nid in info_b else "benign",
            f"{config_b}_score": b_info.get("score", ""),
            f"{config_b}_category": b_info.get("category", ""),
        })

    with open(filepath, "w", newline="") as f:
        fieldnames = [
            "node_id",
            f"{config_a}_classification", f"{config_a}_score", f"{config_a}_category",
            f"{config_b}_classification", f"{config_b}_score", f"{config_b}_category",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    overall = {}

    for scenario in SCENARIOS:
        config_a = scenario["A"]
        config_b = scenario["B"]
        scenario_name = scenario["name"]
        out_dir = os.path.join(BASE_DIR, scenario_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Scenario: A={config_a}  vs  B={config_b}")
        print(f"{'='*80}")

        scenario_summary = {}

        for dataset in DATASETS:
            print(f"\n  Dataset: {dataset}")

            result = compare_configs(config_a, config_b, dataset)

            mal_in_a_ben_in_b = result["mal_in_a_ben_in_b"]
            ben_in_a_mal_in_b = result["ben_in_a_mal_in_b"]
            diff = result["differently_classified"]

            print(f"    Malicious in A, Benign in B:  {len(mal_in_a_ben_in_b)}")
            print(f"    Benign in A, Malicious in B:  {len(ben_in_a_mal_in_b)}")
            print(f"    Total differently classified: {len(diff)}")

            ds_dir = os.path.join(out_dir, dataset)
            os.makedirs(ds_dir, exist_ok=True)

            # All malicious-in-A-but-benign-in-B: use mal_a for A info, ben_b for B info
            info_a_mal = {nid: result["mal_a"][nid] for nid in mal_in_a_ben_in_b}
            info_b_ben = {nid: result["ben_b"][nid] for nid in mal_in_a_ben_in_b}
            write_diff_csv(
                os.path.join(ds_dir, "malicious_in_A_benign_in_B.csv"),
                mal_in_a_ben_in_b, info_a_mal, info_b_ben, config_a, config_b,
            )

            # All benign-in-A-but-malicious-in-B: use ben_a for A info, mal_b for B info
            info_a_ben = {nid: result["ben_a"][nid] for nid in ben_in_a_mal_in_b}
            info_b_mal = {nid: result["mal_b"][nid] for nid in ben_in_a_mal_in_b}
            write_diff_csv(
                os.path.join(ds_dir, "benign_in_A_malicious_in_B.csv"),
                ben_in_a_mal_in_b, info_a_ben, info_b_mal, config_a, config_b,
            )

            # Combined differently classified
            all_info_a = {}
            all_info_b = {}
            for nid in diff:
                if nid in result["mal_a"]:
                    all_info_a[nid] = result["mal_a"][nid]
                else:
                    all_info_a[nid] = result["ben_a"][nid]
                if nid in result["mal_b"]:
                    all_info_b[nid] = result["mal_b"][nid]
                else:
                    all_info_b[nid] = result["ben_b"][nid]

            # Write combined CSV with explicit classification labels
            combined_path = os.path.join(ds_dir, "all_differently_classified.csv")
            with open(combined_path, "w", newline="") as f:
                fieldnames = [
                    "node_id",
                    f"{config_a}_classification", f"{config_a}_score", f"{config_a}_category",
                    f"{config_b}_classification", f"{config_b}_score", f"{config_b}_category",
                    "direction",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for nid in sorted(diff):
                    a_class = "malicious" if nid in result["mal_a"] else "benign"
                    b_class = "malicious" if nid in result["mal_b"] else "benign"
                    direction = f"{a_class}_in_A→{b_class}_in_B"
                    writer.writerow({
                        "node_id": nid,
                        f"{config_a}_classification": a_class,
                        f"{config_a}_score": all_info_a[nid]["score"],
                        f"{config_a}_category": all_info_a[nid]["category"],
                        f"{config_b}_classification": b_class,
                        f"{config_b}_score": all_info_b[nid]["score"],
                        f"{config_b}_category": all_info_b[nid]["category"],
                        "direction": direction,
                    })

            # Write summary
            with open(os.path.join(ds_dir, "summary.txt"), "w") as f:
                f.write(f"Classification Comparison: A={config_a} vs B={config_b}\n")
                f.write(f"Dataset: {dataset}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"A ({config_a}):\n")
                f.write(f"  Total malicious: {len(result['mal_a'])}\n")
                f.write(f"  Total benign:    {len(result['ben_a'])}\n\n")
                f.write(f"B ({config_b}):\n")
                f.write(f"  Total malicious: {len(result['mal_b'])}\n")
                f.write(f"  Total benign:    {len(result['ben_b'])}\n\n")
                f.write(f"Differences:\n")
                f.write(f"  Malicious in A, Benign in B:  {len(mal_in_a_ben_in_b)}\n")
                f.write(f"  Benign in A, Malicious in B:  {len(ben_in_a_mal_in_b)}\n")
                f.write(f"  Total differently classified: {len(diff)}\n")

            scenario_summary[dataset] = {
                "A_total_malicious": len(result["mal_a"]),
                "A_total_benign": len(result["ben_a"]),
                "B_total_malicious": len(result["mal_b"]),
                "B_total_benign": len(result["ben_b"]),
                "malicious_in_A_benign_in_B": len(mal_in_a_ben_in_b),
                "benign_in_A_malicious_in_B": len(ben_in_a_mal_in_b),
                "total_differently_classified": len(diff),
            }

        overall[scenario_name] = {
            "A": config_a,
            "B": config_b,
            "datasets": scenario_summary,
        }

        # Write scenario-level summary JSON
        with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
            json.dump({"A": config_a, "B": config_b, "datasets": scenario_summary}, f, indent=2)

    # Write overall comparison summary
    with open(os.path.join(BASE_DIR, "comparison_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    # Print final table
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    for sname, sdata in overall.items():
        print(f"\n  {sdata['A']} (A) vs {sdata['B']} (B):")
        for ds, counts in sdata["datasets"].items():
            print(f"    {ds:15s}  "
                  f"mal_in_A_ben_in_B={counts['malicious_in_A_benign_in_B']:6d}  "
                  f"ben_in_A_mal_in_B={counts['benign_in_A_malicious_in_B']:6d}  "
                  f"total_diff={counts['total_differently_classified']:6d}")


if __name__ == "__main__":
    main()
