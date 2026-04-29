#!/usr/bin/env python3
"""
generate_reduced_euler.py
=========================
Generate a "reduced-feature" version of euler/ that keeps the
**exact same rows** (same number of lines, same file splits, no
deduplication) but replaces the per-flow features (duration, bytes,
packets) with global temporal summary statistics for the corresponding
directed (src, dst) pair:

    first_ts  — earliest timestamp of this (src, dst) across ALL data
    last_ts   — latest  timestamp of this (src, dst) across ALL data
    count     — total number of flows for this (src, dst) across ALL data

Output line format (same 7-field CSV, compatible with load_cic.py):

    ts, src, dst, first_ts, last_ts, count, label

Two-pass approach:
    Pass 1  — scan all input files, compute per-edge (src,dst) stats
    Pass 2  — re-read input files, substitute features, write output

Usage
-----
    python generate_reduced_euler.py \
        --input_dir  /path/to/cic_2017/euler \
        --output_dir /path/to/cic_2017/euler_red

The script also copies nmap.pkl from input_dir to output_dir.
"""

import argparse
import os
import shutil
import time
from collections import defaultdict


FILE_DELTA = 100000


# ──────────────────────────────────────────────────────────────────
# Pass 1: collect global per-edge statistics
# ──────────────────────────────────────────────────────────────────

def collect_edge_stats(input_dir, file_delta=FILE_DELTA):
    """Scan all slice files and accumulate per-edge statistics."""
    stats = defaultdict(lambda: [float("inf"), float("-inf"), 0])

    cur_slice = 0
    total_lines = 0
    while True:
        fname = os.path.join(input_dir, f"{cur_slice}.txt")
        if not os.path.exists(fname):
            break
        with open(fname, "r") as fh:
            for line in fh:
                parts = line.split(",")
                ts  = int(parts[0])
                src = int(parts[1])
                dst = int(parts[2])
                total_lines += 1

                rec = stats[(src, dst)]
                if ts < rec[0]:
                    rec[0] = ts
                if ts > rec[1]:
                    rec[1] = ts
                rec[2] += 1

        cur_slice += file_delta

    return dict(stats), total_lines


# ──────────────────────────────────────────────────────────────────
# Pass 2: re-read input, substitute features, write output
# ──────────────────────────────────────────────────────────────────

def rewrite_with_stats(input_dir, output_dir, edge_stats, file_delta=FILE_DELTA):
    """Re-read every input file and write an identically-structured output
    where fields [3,4,5] (duration, bytes, packets) are replaced with
    (first_ts, last_ts, count) for that edge."""
    cur_slice = 0
    written = 0

    while True:
        in_fname = os.path.join(input_dir, f"{cur_slice}.txt")
        if not os.path.exists(in_fname):
            break

        out_fname = os.path.join(output_dir, f"{cur_slice}.txt")
        with open(in_fname, "r") as fin, open(out_fname, "w") as fout:
            for line in fin:
                parts = line.strip().split(",")
                src = int(parts[1])
                dst = int(parts[2])
                label = parts[-1]

                fts, lts, cnt = edge_stats[(src, dst)]

                fout.write(f"{parts[0]},{parts[1]},{parts[2]},"
                           f"{fts},{lts},{cnt},{label}\n")
                written += 1

        cur_slice += file_delta

    return written


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate euler_red/: same rows as euler/ "
                    "but with (first_ts, last_ts, count) as flow features"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to euler/ directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output directory (euler_red/)"
    )
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 1
    print(f"[1/3] Pass 1 — collecting per-edge stats from {args.input_dir} ...")
    edge_stats, total_lines = collect_edge_stats(args.input_dir)
    n_edges = len(edge_stats)
    print(f"       {n_edges} unique directed edges, {total_lines} total lines")

    # Pass 2
    print(f"[2/3] Pass 2 — rewriting with reduced features to {args.output_dir} ...")
    written = rewrite_with_stats(args.input_dir, args.output_dir, edge_stats)
    assert written == total_lines, f"Line count mismatch: {written} vs {total_lines}"
    print(f"       {written} lines written")

    # Copy nmap.pkl
    print("[3/3] Copying nmap.pkl ...")
    nmap_src = os.path.join(args.input_dir, "nmap.pkl")
    if os.path.exists(nmap_src):
        shutil.copy2(nmap_src, os.path.join(args.output_dir, "nmap.pkl"))
        print("       Done.")
    else:
        print("       WARNING: nmap.pkl not found in input_dir")

    elapsed = time.time() - t0
    n_files = len([f for f in os.listdir(args.output_dir) if f.endswith('.txt')])
    print(f"\nComplete in {elapsed:.1f}s")
    print(f"  Input        : {args.input_dir}")
    print(f"  Output       : {args.output_dir}")
    print(f"  Files        : {n_files} .txt + nmap.pkl")
    print(f"  Unique edges : {n_edges}")
    print(f"  Total lines  : {written}")


if __name__ == '__main__':
    main()
