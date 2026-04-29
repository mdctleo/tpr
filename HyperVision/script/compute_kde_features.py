#!/usr/bin/env python3
"""
compute_kde_features.py — Precompute KDE density vectors for HyperVision .data files.

Reads a HyperVision .data file (format: ip_ver src_ip dst_ip src_port dst_port timestamp type_code length),
computes BayesianGaussianMixture (DPGMM) density vectors per directed (src_ip, dst_ip) pair
from inter-arrival times, and writes a CSV file loadable by the C++ KDE loader.

Usage:
    python compute_kde_features.py --data ../data/charrdos.data --output ../kde_features/charrdos.csv
    python compute_kde_features.py --all-attacks --data-dir ../data --output-dir ../kde_features

Output CSV format (no header):
    src_ip_int,dst_ip_int,kde_0,kde_1,...,kde_{dim-1}
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict
import multiprocessing as mp

try:
    from sklearn.mixture import BayesianGaussianMixture
    from sklearn.preprocessing import MaxAbsScaler
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
except ImportError:
    pass

# All 43 attacks from encrypted_flooding_traffic + traditional_brute_force_attack
ALL_43_ATTACKS = [
    # encrypted_flooding_traffic / link_flooding (6)
    "crossfirela", "crossfiremd", "crossfiresm", "lrtcpdos02", "lrtcpdos05", "lrtcpdos10",
    # encrypted_flooding_traffic / password_cracking (6)
    "sshpwdla", "sshpwdmd", "sshpwdsm", "telnetpwdla", "telnetpwdmd", "telnetpwdsm",
    # encrypted_flooding_traffic / ssh_inject (3)
    "ackport", "ipidaddr", "ipidport",
    # traditional_brute_force_attack / amplification_attack (7)
    "charrdos", "cldaprdos", "dnsrdos", "memcachedrdos", "ntprdos", "riprdos", "ssdprdos",
    # traditional_brute_force_attack / brute_scanning (7)
    "dnsscan", "httpscan", "httpsscan", "icmpscan", "ntpscan", "sqlscan", "sshscan",
    # traditional_brute_force_attack / probing_vulnerable_application (10)
    "dns_lrscan", "http_lrscan", "icmp_lrscan", "netbios_lrscan", "rdp_lrscan",
    "smtp_lrscan", "snmp_lrscan", "ssh_lrscan", "telnet_lrscan", "vlc_lrscan",
    # traditional_brute_force_attack / source_spoof (4)
    "icmpsdos", "rstsdos", "synsdos", "udpsdos",
]


def compute_bgmm_density(timestamps, kde_dim=20, n_components=10,
                          weight_concentration_prior=0.1, max_iter=200):
    """Compute L2-normalised DPGMM density vector from sorted timestamps."""
    timestamps = np.sort(timestamps)
    diffs = np.abs(np.diff(timestamps))
    if len(diffs) == 0:
        return None

    scaler = MaxAbsScaler()
    scaled = scaler.fit_transform(diffs.reshape(-1, 1)).flatten()

    if np.std(scaled) < 1e-10:
        vec = np.zeros(kde_dim, dtype=np.float64)
        vec[kde_dim // 2] = 1.0
        return vec

    try:
        bgmm = BayesianGaussianMixture(
            n_components=min(n_components, len(scaled)),
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            init_params='k-means++',
            random_state=42,
            tol=1e-3,
            reg_covar=1e-6,
        )
        bgmm.fit(scaled.reshape(-1, 1))
        gmin, gmax = scaled.min(), scaled.max()
        margin = 0.1 * (gmax - gmin) if gmax > gmin else 0.1
        grid = np.linspace(gmin - margin, gmax + margin, kde_dim)
        log_density = bgmm.score_samples(grid.reshape(-1, 1))
        density = np.exp(log_density)
        norm = np.linalg.norm(density)
        if norm > 1e-8:
            density /= norm
        return density
    except Exception:
        return None


def process_data_file(data_path, output_path, kde_dim=20, min_timestamps=10,
                      n_components=10, wcp=0.1, max_iter=200, n_jobs=0):
    """Process one .data file → one .csv KDE feature file."""
    print(f"  Reading {data_path} ...")
    edge_timestamps = defaultdict(list)

    with open(data_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            # Skip IPv6 entries (protocol 6) — only IPv4 (protocol 4) fits uint32
            if parts[0] != '4':
                continue
            # ip_ver src_ip dst_ip src_port dst_port timestamp type_code length
            src_ip = int(parts[1])
            dst_ip = int(parts[2])
            timestamp = float(parts[5])
            edge_timestamps[(src_ip, dst_ip)].append(timestamp)

    print(f"    Unique (src,dst) pairs: {len(edge_timestamps)}")

    zero_vec = np.zeros(kde_dim, dtype=np.float64)

    # Separate pairs that need fitting from those that don't
    keys = list(edge_timestamps.keys())
    to_fit_keys = []
    to_fit_ts = []
    fallback_results = []
    for k in keys:
        ts = np.array(edge_timestamps[k])
        if len(ts) <= min_timestamps:
            fallback_results.append((k, zero_vec))
        else:
            to_fit_keys.append(k)
            to_fit_ts.append(ts)

    _n_jobs = (mp.cpu_count() if n_jobs <= 0 else n_jobs)
    _n_jobs = min(_n_jobs, max(1, len(to_fit_keys)))
    print(f"    Fitting {len(to_fit_keys)} pairs on {_n_jobs} cores, {len(fallback_results)} fallback")

    if HAS_JOBLIB and len(to_fit_keys) > 0:
        fitted_vecs = Parallel(n_jobs=_n_jobs, backend='loky', verbose=0)(
            delayed(compute_bgmm_density)(ts, kde_dim, n_components, wcp, max_iter)
            for ts in to_fit_ts
        )
    elif len(to_fit_keys) > 0:
        fitted_vecs = [compute_bgmm_density(ts, kde_dim, n_components, wcp, max_iter) for ts in to_fit_ts]
    else:
        fitted_vecs = []

    kde_count = 0
    fallback_count = len(fallback_results)
    results = list(fallback_results)  # (key, vec) tuples
    for k, vec in zip(to_fit_keys, fitted_vecs):
        if vec is None:
            results.append((k, zero_vec))
            fallback_count += 1
        else:
            results.append((k, vec))
            kde_count += 1

    print(f"    DPGMM fitted: {kde_count}, fallback: {fallback_count}")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for (src, dst), vec in results:
            vec_str = ','.join(f'{v:.8f}' for v in vec)
            f.write(f'{src},{dst},{vec_str}\n')

    print(f"    Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute KDE features for HyperVision .data files')
    parser.add_argument('--data', type=str, help='Single .data file path')
    parser.add_argument('--output', type=str, help='Single output CSV path')
    parser.add_argument('--all-attacks', action='store_true',
                        help='Process all 43 attacks')
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing .data files (for --all-attacks)')
    parser.add_argument('--output-dir', type=str, default='../kde_features',
                        help='Output directory for KDE CSVs (for --all-attacks)')
    parser.add_argument('--kde-dim', type=int, default=20)
    parser.add_argument('--min-timestamps', type=int, default=10)
    parser.add_argument('--n-components', type=int, default=10)
    parser.add_argument('--concentration-prior', type=float, default=0.1)
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--jobs', '-j', type=int, default=0,
                        help='Number of parallel jobs (0 = all cores)')
    args = parser.parse_args()

    if args.all_attacks:
        print(f"Processing all {len(ALL_43_ATTACKS)} attacks from {args.data_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        for attack in ALL_43_ATTACKS:
            data_path = os.path.join(args.data_dir, f'{attack}.data')
            out_path = os.path.join(args.output_dir, f'{attack}_kde.csv')
            if not os.path.exists(data_path):
                print(f"  SKIP: {data_path} not found")
                continue
            process_data_file(data_path, out_path, args.kde_dim, args.min_timestamps,
                              args.n_components, args.concentration_prior, args.max_iter,
                              args.jobs)
        print("\nDone! All KDE features computed.")
    elif args.data and args.output:
        process_data_file(args.data, args.output, args.kde_dim, args.min_timestamps,
                          args.n_components, args.concentration_prior, args.max_iter,
                          args.jobs)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
