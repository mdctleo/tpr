#!/usr/bin/env python3
"""
generate_reduced_argus_flow.py
==============================
Generate a "reduced-feature" version of argus_flow/ that keeps the
**exact same rows** (same number of lines, same file splits, no
deduplication) but replaces the per-flow features (duration, bytes,
packets) with global temporal summary statistics for the corresponding
directed (src, dst) pair:

    first_ts  — earliest timestamp of this (src, dst) across ALL data
    last_ts   — latest  timestamp of this (src, dst) across ALL data
    count     — total number of flows for this (src, dst) across ALL data

These three values are **identical** for every row that shares the same
(src, dst) pair — they capture the edge's lifetime and frequency rather
than individual flow characteristics.

Output line format (same 7-field CSV, compatible with load_cic_flow.py):

    ts, src, dst, first_ts, last_ts, count, label

Two-pass approach:
    Pass 1  — scan all input files, compute per-edge (src,dst) stats
    Pass 2  — re-read input files, substitute features, write output

Usage
-----
    python generate_reduced_argus_flow.py \\
        --input_dir  /path/to/cic_2017/argus_flow \\
        --output_dir /path/to/cic_2017/argus_flow_red

The script also copies nmap.pkl from input_dir to output_dir.
"""

import argparse
import os
import shutil
import time
from collections import defaultdict


FILE_DELTA = 100000      # same slice boundaries as original argus_flow/


# ──────────────────────────────────────────────────────────────────
# Pass 1: collect global per-edge statistics
# ──────────────────────────────────────────────────────────────────

def collect_edge_stats(input_dir, file_delta=FILE_DELTA):
    """Scan all slice files and accumulate per-edge statistics.

    Returns
    -------
    edge_stats : dict[ (int,int), tuple(int, int, int) ]
        Keys = (src, dst).  Values = (first_ts, last_ts, count).
    """
    stats = defaultdict(lambda: [float("inf"), float("-inf"), 0])
    # stats[(src,dst)] = [first_ts, last_ts, count]

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
    the precomputed (first_ts, last_ts, count) for that edge.

    Returns the total number of lines written.
    """
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
                label = parts[-1]              # keep original label per row

                fts, lts, cnt = edge_stats[(src, dst)]

                # ts, src, dst, first_ts, last_ts, count, label
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
        description="Generate argus_flow_red/: same rows as argus_flow/ "
                    "but with (first_ts, last_ts, count) as flow features"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to argus_flow/ directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output directory (argus_flow_red/)"
    )
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    # Pass 1: collect edge statistics
    print(f"[1/3] Pass 1 — collecting per-edge stats from {args.input_dir} ...")
    edge_stats, total_lines = collect_edge_stats(args.input_dir)
    n_edges = len(edge_stats)
    print(f"       {total_lines:,} input lines, {n_edges:,} unique directed (src,dst) pairs")

    # Pass 2: rewrite with substituted features
    print(f"[2/3] Pass 2 — rewriting to {args.output_dir} ...")
    written = rewrite_with_stats(args.input_dir, args.output_dir, edge_stats)
    print(f"       Wrote {written:,} rows (should equal {total_lines:,} input rows)")

    assert written == total_lines, (
        f"Line count mismatch: wrote {written:,} but input had {total_lines:,}"
    )

    # Copy nmap.pkl
    nmap_src = os.path.join(args.input_dir, "nmap.pkl")
    nmap_dst = os.path.join(args.output_dir, "nmap.pkl")
    if os.path.exists(nmap_src):
        shutil.copy2(nmap_src, nmap_dst)
        print(f"[3/3] Copied nmap.pkl")
    else:
        print(f"[3/3] WARNING: nmap.pkl not found in {args.input_dir}")

    # Summary
    out_files = sorted(f for f in os.listdir(args.output_dir) if f.endswith(".txt"))
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output: {args.output_dir}/")
    for f in out_files:
        fpath = os.path.join(args.output_dir, f)
        n_lines = sum(1 for _ in open(fpath))
        print(f"  {f:>15s}  {n_lines:>8,} lines")


if __name__ == "__main__":
    main()
