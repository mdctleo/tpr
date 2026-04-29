#!/usr/bin/env python3
"""
summarize_results.py — Parse HyperVision log files and produce a summary table + comparison.

Usage:
    # Summarize baseline results
    python summarize_results.py --log-dir ./log --groups brute lrscan misc

    # Compare baseline vs KDE results
    python summarize_results.py --log-dir ./log --groups brute lrscan misc \
        --compare-dir ./log_kde --compare-label "KDE-Enhanced"

    # Output as CSV
    python summarize_results.py --log-dir ./log --groups brute lrscan misc --csv results.csv
"""

import argparse
import os
import re
import sys
from collections import OrderedDict


def parse_log(log_path):
    """Extract all metrics from a single basic_analyzer.py log file."""
    metrics = {}
    if not os.path.isfile(log_path):
        return None
    with open(log_path, 'r') as f:
        text = f.read()

    patterns = {
        'AU_ROC':    r'AU_ROC=([0-9.]+)',
        'TPR@FPR0.1': r'TPR=([0-9.]+) \(FPR=0\.1\)',
        'FPR@TPR0.9': r'FPR=([0-9.]+) \(TPR=0\.9\)',
        'EER':       r'EER=([0-9.]+)',
        'F1':        r'F1-score=([0-9.]+)',
        'F2':        r'F2-score=([0-9.]+)',
        'Precision':  r'Precision=([0-9.]+)',
        'Recall':     r'Recall=([0-9.]+)',
        'AU_PRC':     r'AU_PRC=([0-9.]+)',
        'Accuracy':   r'Accuracy=([0-9.]+)',
        'FN':         r'FN=(\d+)',
        'FP':         r'FP=(\d+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            metrics[key] = float(val) if '.' in val else int(val)
    return metrics if metrics else None


def collect_group(log_dir, group):
    """Collect metrics for all attacks in a group."""
    group_dir = os.path.join(log_dir, group)
    if not os.path.isdir(group_dir):
        return {}
    results = OrderedDict()
    for fname in sorted(os.listdir(group_dir)):
        if fname.endswith('.log'):
            attack = fname[:-4]
            m = parse_log(os.path.join(group_dir, fname))
            if m:
                results[attack] = m
    return results


def print_table(all_results, metric_cols=None, title="Baseline"):
    """Print a formatted markdown table."""
    if metric_cols is None:
        metric_cols = ['AU_ROC', 'TPR@FPR0.1', 'FPR@TPR0.9', 'EER', 'F1', 'Precision', 'Recall', 'Accuracy']

    print(f"\n### {title}\n")
    header = "| Group | Attack | " + " | ".join(metric_cols) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(metric_cols)) + "|"
    print(header)
    print(sep)

    for group, attacks in all_results.items():
        for attack, metrics in attacks.items():
            vals = []
            for col in metric_cols:
                v = metrics.get(col)
                if v is None:
                    vals.append("—")
                elif isinstance(v, int):
                    vals.append(str(v))
                elif v >= 0.99:
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(f"{v:.4f}")
            print(f"| {group} | {attack} | " + " | ".join(vals) + " |")


def print_comparison(baseline, kde, metric='AU_ROC'):
    """Print side-by-side comparison table for a single metric."""
    print(f"\n### Comparison: {metric}\n")
    print("| Group | Attack | Baseline | KDE-Enhanced | Δ |")
    print("|---|---|---:|---:|---:|")

    for group in baseline:
        for attack in baseline[group]:
            b_val = baseline[group][attack].get(metric)
            k_val = kde.get(group, {}).get(attack, {}).get(metric)
            b_str = f"{b_val:.6f}" if b_val is not None else "—"
            k_str = f"{k_val:.6f}" if k_val is not None else "—"
            if b_val is not None and k_val is not None:
                delta = k_val - b_val
                d_str = f"{delta:+.6f}"
            else:
                d_str = "—"
            print(f"| {group} | {attack} | {b_str} | {k_str} | {d_str} |")


def print_group_summary(all_results, title=""):
    """Print per-group average AU_ROC."""
    print(f"\n### Per-Group Average AU_ROC {title}\n")
    print("| Group | # Attacks | Mean AU_ROC | Min AU_ROC | Min Attack |")
    print("|---|---:|---:|---:|---|")
    total_attacks = 0
    total_auroc = 0
    for group, attacks in all_results.items():
        aurocs = [(a, m.get('AU_ROC', 0)) for a, m in attacks.items() if m.get('AU_ROC') is not None]
        if not aurocs:
            continue
        n = len(aurocs)
        mean_auroc = sum(v for _, v in aurocs) / n
        min_attack, min_auroc = min(aurocs, key=lambda x: x[1])
        total_attacks += n
        total_auroc += sum(v for _, v in aurocs)
        print(f"| {group} | {n} | {mean_auroc:.6f} | {min_auroc:.6f} | {min_attack} |")
    if total_attacks > 0:
        print(f"| **Overall** | **{total_attacks}** | **{total_auroc/total_attacks:.6f}** | | |")


def write_csv(all_results, csv_path, metric_cols=None):
    """Write results to CSV."""
    if metric_cols is None:
        metric_cols = ['AU_ROC', 'TPR@FPR0.1', 'FPR@TPR0.9', 'EER', 'F1', 'F2',
                       'Precision', 'Recall', 'AU_PRC', 'Accuracy', 'FN', 'FP']
    with open(csv_path, 'w') as f:
        f.write("group,attack," + ",".join(metric_cols) + "\n")
        for group, attacks in all_results.items():
            for attack, metrics in attacks.items():
                vals = [str(metrics.get(col, '')) for col in metric_cols]
                f.write(f"{group},{attack}," + ",".join(vals) + "\n")
    print(f"CSV written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Summarize HyperVision results')
    parser.add_argument('--log-dir', default='./log', help='Directory containing group log subdirs')
    parser.add_argument('--groups', nargs='+', default=['brute', 'lrscan', 'misc'])
    parser.add_argument('--compare-dir', default=None, help='Second log dir for comparison')
    parser.add_argument('--compare-label', default='KDE-Enhanced')
    parser.add_argument('--csv', default=None, help='Output CSV path')
    parser.add_argument('--metric', default='AU_ROC', help='Metric for comparison')
    args = parser.parse_args()

    all_results = OrderedDict()
    for g in args.groups:
        r = collect_group(args.log_dir, g)
        if r:
            all_results[g] = r

    if not all_results:
        print("No results found!", file=sys.stderr)
        sys.exit(1)

    print_group_summary(all_results, "(Baseline)")
    print_table(all_results, title="Baseline — Full Results")

    if args.csv:
        write_csv(all_results, args.csv)

    if args.compare_dir:
        kde_results = OrderedDict()
        for g in args.groups:
            r = collect_group(args.compare_dir, g)
            if r:
                kde_results[g] = r
        if kde_results:
            print_group_summary(kde_results, f"({args.compare_label})")
            print_comparison(all_results, kde_results, args.metric)


if __name__ == '__main__':
    main()
