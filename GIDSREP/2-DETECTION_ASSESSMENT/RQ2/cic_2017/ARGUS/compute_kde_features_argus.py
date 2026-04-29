#!/usr/bin/env python3
"""
compute_kde_features_argus.py
=============================
Pre-compute DPGMM (BayesianGaussianMixture) density vectors from
timestamp differences for every directed edge (src, dst) observed
in ARGUS training data (argus_flow/).

The resulting pickle maps  (src_int, dst_int) → np.ndarray(K,)
where K is the KDE dimensionality (default 20).

Usage
-----
    python compute_kde_features_argus.py \
        --data_dir /path/to/cic_2017/argus_flow \
        --output   kde_vectors_argus.pkl \
        --kde_dim  20 \
        --tr_end   29136

The output pickle is later consumed by the modified load_cic_flow.py
when the --kde flag is passed to main.py.
"""

import argparse
import os
import pickle
import time
import warnings
import numpy as np
from collections import defaultdict
from sklearn.mixture import BayesianGaussianMixture

# Suppress sklearn convergence warnings (duplicate diffs cause KMeans init warnings)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*Number of distinct clusters.*')


# ──────────────────────────────────────────────────────────────────
# 1. Collect per-edge timestamps from training data
# ──────────────────────────────────────────────────────────────────

def collect_timestamps(data_dir, tr_end, file_delta=100000):
    """Read argus_flow/*.txt files and collect timestamps per directed edge.

    Only lines with ts < tr_end are included (training period).
    Returns
    -------
    edge_ts : dict[ (int,int), list[int] ]
        Mapping from directed edge to sorted list of timestamps.
    """
    edge_ts = defaultdict(list)

    # Determine which slice files to read
    cur_slice = 0
    while True:
        fname = os.path.join(data_dir, f"{cur_slice}.txt")
        if not os.path.exists(fname):
            break

        with open(fname, "r") as fh:
            for line in fh:
                parts = line.strip().split(",")
                ts = int(parts[0])
                if ts >= tr_end:
                    # Files are sorted by ts within each slice; once we
                    # exceed tr_end we can skip the rest of this file
                    # (but not necessarily all files — timestamps can reset
                    #  across slice boundaries, so we keep iterating slices).
                    break
                src = int(parts[1])
                dst = int(parts[2])
                if src == dst:          # skip self-loops
                    continue
                edge_ts[(src, dst)].append(ts)

        cur_slice += file_delta

    # Sort each edge's timestamps
    for key in edge_ts:
        edge_ts[key].sort()

    return dict(edge_ts)


# ──────────────────────────────────────────────────────────────────
# 2. Compute the evaluation grid from global diff distribution
# ──────────────────────────────────────────────────────────────────

def compute_grid(edge_ts, kde_dim, upper_pct=95):
    """Build a fixed grid of `kde_dim` points spanning the timestamp-diff
    distribution across all training edges.

    Parameters
    ----------
    edge_ts   : dict  — from collect_timestamps()
    kde_dim   : int   — number of grid points
    upper_pct : float — upper percentile for grid range (avoids outliers)

    Returns
    -------
    grid : np.ndarray of shape (kde_dim,)
    """
    all_diffs = []
    for ts_list in edge_ts.values():
        if len(ts_list) <= 10:
            continue
        arr = np.array(ts_list)
        diffs = np.diff(arr).astype(np.float64)
        all_diffs.append(diffs)

    if len(all_diffs) == 0:
        # Degenerate case — return a dummy grid
        return np.linspace(0, 1, kde_dim)

    all_diffs = np.concatenate(all_diffs)
    lo = float(all_diffs.min())
    hi = float(np.percentile(all_diffs, upper_pct))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, kde_dim)


# ──────────────────────────────────────────────────────────────────
# 3. Fit DPGMM and evaluate density for one edge
# ──────────────────────────────────────────────────────────────────

def fit_edge_kde(diffs, grid, n_components=10):
    """Fit a BayesianGaussianMixture on `diffs` and evaluate on `grid`.

    Parameters
    ----------
    diffs : np.ndarray of shape (n_diffs,)
    grid  : np.ndarray of shape (K,)

    Returns
    -------
    density : np.ndarray of shape (K,), L2-normalised
    """
    X = diffs.reshape(-1, 1)
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=200,
        random_state=42,
        n_init=1,
    )
    bgm.fit(X)
    log_prob = bgm.score_samples(grid.reshape(-1, 1))   # (K,)
    density = np.exp(log_prob)

    norm = np.linalg.norm(density)
    if norm > 0:
        density /= norm
    else:
        density = np.zeros_like(density)
    return density


# ──────────────────────────────────────────────────────────────────
# 4. Main driver
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute DPGMM KDE density vectors for ARGUS edge features"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to argus_flow/ directory containing *.txt slice files"
    )
    parser.add_argument(
        "--output", type=str, default="kde_vectors_argus.pkl",
        help="Output pickle file path"
    )
    parser.add_argument(
        "--kde_dim", type=int, default=10,
        help="Number of grid points for density evaluation (K)"
    )
    parser.add_argument(
        "--tr_end", type=int, default=29136,
        help="End of training period (exclusive). Default=29136 (DATE_OF_EVIL_LANL)"
    )
    parser.add_argument(
        "--n_components", type=int, default=10,
        help="Max mixture components for DPGMM"
    )
    parser.add_argument(
        "--upper_pct", type=float, default=95,
        help="Upper percentile for grid range"
    )
    args = parser.parse_args()

    t0 = time.time()

    # Step 1: collect timestamps
    print(f"[1/4] Collecting training timestamps from {args.data_dir}  (ts < {args.tr_end}) ...")
    edge_ts = collect_timestamps(args.data_dir, args.tr_end)
    n_edges = len(edge_ts)
    n_ts_total = sum(len(v) for v in edge_ts.values())
    print(f"       {n_edges:,} directed edges, {n_ts_total:,} total timestamps")

    # Step 2: build grid
    print(f"[2/4] Computing evaluation grid (K={args.kde_dim}, upper_pct={args.upper_pct}) ...")
    grid = compute_grid(edge_ts, args.kde_dim, upper_pct=args.upper_pct)
    print(f"       Grid range: [{grid[0]:.2f}, {grid[-1]:.2f}]")

    # Step 3: fit DPGMM per edge
    print(f"[3/4] Fitting DPGMM for each edge (n_components<={args.n_components}) ...")
    kde_dict = {}
    skipped = 0
    for idx, ((src, dst), ts_list) in enumerate(edge_ts.items()):
        if len(ts_list) <= 10:
            # Not enough timestamps → skip (loader falls back to zeros)
            skipped += 1
            continue

        diffs = np.diff(np.array(ts_list)).astype(np.float64)
        n_unique = len(np.unique(diffs))
        n_comp = min(args.n_components, n_unique)
        if n_comp < 1:
            skipped += 1
            continue
        kde_dict[(src, dst)] = fit_edge_kde(diffs, grid, n_components=n_comp)

        if (idx + 1) % 2000 == 0:
            print(f"       ... processed {idx+1:,}/{n_edges:,} edges")

    fitted = n_edges - skipped
    print(f"       Fitted: {fitted:,}  |  Skipped (≤ 10 timestamps): {skipped:,}")

    # Step 4: save
    print(f"[4/4] Saving to {args.output} ...")
    with open(args.output, "wb") as f:
        pickle.dump({
            "kde_dict": kde_dict,
            "grid": grid,
            "kde_dim": args.kde_dim,
            "tr_end": args.tr_end,
            "n_edges": n_edges,
            "n_fitted": fitted,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Output: {args.output}  ({os.path.getsize(args.output)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
