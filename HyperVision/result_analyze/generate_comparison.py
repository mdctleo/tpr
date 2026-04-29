#!/usr/bin/env python3
"""
Generate results_comparison.csv with separate tables for every metric,
comparing baseline vs KDE-enhanced for all 43 attacks.
"""

import os
import re
import csv
import sys

GROUPS = {
    "brute": [
        "charrdos", "cldaprdos", "dnsrdos", "dnsscan", "httpscan", "httpsscan",
        "icmpscan", "icmpsdos", "memcachedrdos", "ntprdos", "ntpscan", "riprdos",
        "rstsdos", "sqlscan", "ssdprdos", "sshscan", "synsdos", "udpsdos",
    ],
    "lrscan": [
        "dns_lrscan", "http_lrscan", "icmp_lrscan", "netbios_lrscan", "rdp_lrscan",
        "smtp_lrscan", "snmp_lrscan", "ssh_lrscan", "telnet_lrscan", "vlc_lrscan",
    ],
    "misc": [
        "ackport", "crossfirela", "crossfiremd", "crossfiresm", "ipidaddr",
        "ipidport", "lrtcpdos02", "lrtcpdos05", "lrtcpdos10", "sshpwdla",
        "sshpwdmd", "sshpwdsm", "telnetpwdla", "telnetpwdmd", "telnetpwdsm",
    ],
}

METRICS = [
    ("AU_ROC",      r"AU_ROC=([\d.]+)"),
    ("AU_PRC",      r"AU_PRC=([\d.]+)"),
    ("TPR@FPR=0.1", r"TPR=([\d.]+) \(FPR=0\.1\)"),
    ("FPR@TPR=0.9", r"FPR=([\d.]+) \(TPR=0\.9\)"),
    ("EER",         r"EER=([\d.]+)"),
    ("F1",          r"F1-score=([\d.]+)"),
    ("F2",          r"F2-score=([\d.]+)"),
    ("Precision",   r"Precision=([\d.]+)"),
    ("Recall",      r"Recall=([\d.]+)"),
    ("Accuracy",    r"Accuracy=([\d.]+)"),
    ("FN",          r"FN=(\d+)"),
    ("FP",          r"FP=(\d+)"),
    ("Time_read_s",    r"get_resulf_from_file\(\) Finished\] cost time:\s+([\d.]+)"),
    ("Time_analyze_s", r"analyze_result\(\) Finished\] cost time:\s+([\d.]+)"),
]


def parse_log(path):
    """Extract all metrics from a log file."""
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    result = {}
    for name, pattern in METRICS:
        m = re.search(pattern, text)
        if m:
            val = m.group(1)
            result[name] = int(val) if name in ("FN", "FP") else float(val)
        else:
            result[name] = None
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", default=os.path.join(os.path.dirname(__file__), "log"))
    parser.add_argument("--kde-dir", default=os.path.join(os.path.dirname(__file__), "log_kde"))
    parser.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "results_comparison.csv"))
    args = parser.parse_args()
    baseline_dir = args.baseline_dir
    kde_dir = args.kde_dir
    out_path = args.output

    # Collect all data
    rows = []  # (group, attack, baseline_metrics, kde_metrics)
    for group, attacks in GROUPS.items():
        for attack in attacks:
            b = parse_log(os.path.join(baseline_dir, group, f"{attack}.log"))
            k = parse_log(os.path.join(kde_dir, group, f"{attack}.log"))
            rows.append((group, attack, b, k))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)

        # ── One table per metric ──
        metric_names = [m[0] for m in METRICS]
        for metric in metric_names:
            w.writerow([])
            w.writerow([f"=== {metric} ==="])
            w.writerow(["Group", "Attack", "Baseline", "KDE-Enhanced", "Delta", "Delta%"])

            group_base_vals = {}
            group_kde_vals = {}
            all_base = []
            all_kde = []

            for group, attack, bm, km in rows:
                bv = bm.get(metric) if bm else None
                kv = km.get(metric) if km else None

                if bv is not None and kv is not None:
                    delta = kv - bv
                    if bv != 0:
                        delta_pct = delta / abs(bv) * 100
                    else:
                        delta_pct = 0.0 if delta == 0 else float('inf')
                    w.writerow([group, attack,
                                f"{bv:.6f}" if isinstance(bv, float) else str(bv),
                                f"{kv:.6f}" if isinstance(kv, float) else str(kv),
                                f"{delta:+.6f}" if isinstance(delta, float) else f"{delta:+d}",
                                f"{delta_pct:+.4f}%"])
                    group_base_vals.setdefault(group, []).append(bv)
                    group_kde_vals.setdefault(group, []).append(kv)
                    all_base.append(bv)
                    all_kde.append(kv)
                elif bv is not None:
                    w.writerow([group, attack,
                                f"{bv:.6f}" if isinstance(bv, float) else str(bv),
                                "N/A", "N/A", "N/A"])
                elif kv is not None:
                    w.writerow([group, attack, "N/A",
                                f"{kv:.6f}" if isinstance(kv, float) else str(kv),
                                "N/A", "N/A"])

            # Group averages
            w.writerow([])
            w.writerow([f"--- {metric} Group Averages ---"])
            w.writerow(["Group", "# Attacks", "Baseline Mean", "KDE Mean", "Delta Mean"])
            for group in GROUPS:
                bvals = group_base_vals.get(group, [])
                kvals = group_kde_vals.get(group, [])
                if bvals and kvals:
                    bm_ = sum(bvals) / len(bvals)
                    km_ = sum(kvals) / len(kvals)
                    w.writerow([group, len(bvals),
                                f"{bm_:.6f}" if isinstance(bm_, float) else str(bm_),
                                f"{km_:.6f}" if isinstance(km_, float) else str(km_),
                                f"{km_ - bm_:+.6f}"])

            if all_base and all_kde:
                bm_all = sum(all_base) / len(all_base)
                km_all = sum(all_kde) / len(all_kde)
                w.writerow(["OVERALL", len(all_base),
                            f"{bm_all:.6f}" if isinstance(bm_all, float) else str(bm_all),
                            f"{km_all:.6f}" if isinstance(km_all, float) else str(km_all),
                            f"{km_all - bm_all:+.6f}"])

        # ── Grand summary table ──
        w.writerow([])
        w.writerow(["=== GRAND SUMMARY (all metrics, all attacks) ==="])
        header = ["Group", "Attack"] + \
                 [f"{m}_Baseline" for m in metric_names] + \
                 [f"{m}_KDE" for m in metric_names] + \
                 [f"{m}_Delta" for m in metric_names]
        w.writerow(header)
        for group, attack, bm, km in rows:
            row = [group, attack]
            for m in metric_names:
                bv = bm.get(m) if bm else None
                row.append(f"{bv:.6f}" if isinstance(bv, float) else (str(bv) if bv is not None else "N/A"))
            for m in metric_names:
                kv = km.get(m) if km else None
                row.append(f"{kv:.6f}" if isinstance(kv, float) else (str(kv) if kv is not None else "N/A"))
            for m in metric_names:
                bv = bm.get(m) if bm else None
                kv = km.get(m) if km else None
                if bv is not None and kv is not None:
                    d = kv - bv
                    row.append(f"{d:+.6f}" if isinstance(d, float) else f"{d:+d}")
                else:
                    row.append("N/A")
            w.writerow(row)

    print(f"Written: {out_path}")
    print(f"  {len(rows)} attacks, {len(metric_names)} metrics")


if __name__ == "__main__":
    main()
