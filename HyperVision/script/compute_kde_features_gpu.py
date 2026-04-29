#!/usr/bin/env python3
"""
compute_kde_features_gpu.py — GPU-accelerated KDE density vector computation
for HyperVision .data files using batched CAVI DPGMM (PyTorch on CUDA).

Implements Coordinate Ascent Variational Inference (CAVI) with closed-form
Normal-Inverse-Gamma conjugate updates plus a stick-breaking Dirichlet Process
weight prior.  All heavy arithmetic is expressed as batched PyTorch tensor ops
that run on a single GPU without any per-edge Python loop.

Reference: PIDSMaker/kde_computation.py (BayesianGaussianMixtureGPU class)

Usage:
    # All 43 attacks on GPU
    python compute_kde_features_gpu.py --all-attacks --data-dir ../data --output-dir ../kde_features

    # Single attack
    python compute_kde_features_gpu.py --data ../data/charrdos.data --output ../kde_features/charrdos_kde.csv

    # CPU fallback (no CUDA)
    python compute_kde_features_gpu.py --all-attacks --device cpu --data-dir ../data --output-dir ../kde_features

Output CSV format (no header) — identical to compute_kde_features.py:
    src_ip_int,dst_ip_int,kde_0,kde_1,...,kde_{dim-1}
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch", file=sys.stderr)
    sys.exit(1)


# =====================================================================
#  All 43 attacks
# =====================================================================
ALL_43_ATTACKS = [
    "crossfirela", "crossfiremd", "crossfiresm", "lrtcpdos02", "lrtcpdos05", "lrtcpdos10",
    "sshpwdla", "sshpwdmd", "sshpwdsm", "telnetpwdla", "telnetpwdmd", "telnetpwdsm",
    "ackport", "ipidaddr", "ipidport",
    "charrdos", "cldaprdos", "dnsrdos", "memcachedrdos", "ntprdos", "riprdos", "ssdprdos",
    "dnsscan", "httpscan", "httpsscan", "icmpscan", "ntpscan", "sqlscan", "sshscan",
    "dns_lrscan", "http_lrscan", "icmp_lrscan", "netbios_lrscan", "rdp_lrscan",
    "smtp_lrscan", "snmp_lrscan", "ssh_lrscan", "telnet_lrscan", "vlc_lrscan",
    "icmpsdos", "rstsdos", "synsdos", "udpsdos",
]


# =====================================================================
#  BayesianGaussianMixtureGPU — Batched CAVI DPGMM
# =====================================================================

class BayesianGaussianMixtureGPU:
    """
    GPU-accelerated variational Bayesian Gaussian mixture for batched 1-D data.

    Implements CAVI with closed-form Normal-Inverse-Gamma conjugate updates
    plus a stick-breaking Dirichlet Process weight prior.

    Conjugate model:
        v_k  ~ Beta(1, gamma)               (stick-breaking)
        mu_k ~ N(mu_0, 1/lambda_0)          (component means)
        sigma^2_k ~ InvGamma(alpha_0, beta_0) (component variances)
        z_n  ~ Categorical(pi(v))            (assignments)
        x_n | z_n=k ~ N(mu_k, sigma^2_k)    (observations)
    """

    LOG_SQRT2PI = 0.9189385332046727

    def __init__(
        self,
        n_components: int = 100,
        gamma: float = 5.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        patience: int = 20,
        n_init: int = 1,
        init_method: str = "kmeans",
        device: str = "cuda",
    ):
        self.n_components = n_components
        self.K = n_components
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.patience = patience
        self.n_init = n_init
        self.init_method = init_method
        self.device = device

        # Fitted state
        self.weights_: Optional[torch.Tensor] = None
        self.means_: Optional[torch.Tensor] = None
        self.stds_: Optional[torch.Tensor] = None

    def fit_batch(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> "BayesianGaussianMixtureGPU":
        """
        Fit the model on a batch of 1-D sequences.

        Parameters
        ----------
        X : (B, N_max) padded data matrix
        mask : (B, N_max) bool — True for valid positions
        """
        if X.ndim == 1:
            X = X.unsqueeze(0)
        B, N_max = X.shape
        X = X.to(self.device)

        if mask is None:
            mask = ~torch.isnan(X)
        else:
            mask = mask.to(self.device, dtype=torch.bool)
        mask_f = mask.float()
        N_eff = mask_f.sum(dim=1)

        X = X.clone()
        X[~mask] = 0.0

        K = min(self.K, int(N_eff.max().item()))
        K = max(K, 1)

        best_elbo = torch.full((B,), float("-inf"), device=self.device)
        best_state = None

        for _ in range(self.n_init):
            state, elbo = self._fit_once(X, mask, mask_f, N_eff, B, N_max, K)
            improved = elbo > best_elbo
            if best_state is None:
                best_state = state
                best_elbo = elbo
            else:
                for key in state:
                    best_state[key] = torch.where(
                        improved.unsqueeze(-1) if state[key].ndim == 2 else improved,
                        state[key], best_state[key],
                    )
                best_elbo = torch.where(improved, elbo, best_elbo)

        self._compute_mixture_params(best_state, B, K)
        return self

    def _fit_once(self, X, mask, mask_f, N_eff, B, N_max, K):
        dev = self.device
        gamma = self.gamma

        # Priors (unit-scale, for MaxAbsScaled data in [-1, 1])
        mu0 = torch.zeros(B, 1, device=dev)
        lam0 = torch.ones(B, 1, device=dev)
        a0 = torch.full((B, 1), 3.0, device=dev)
        b0 = torch.full((B, 1), 0.5, device=dev)

        log_beta_prior = -math.log(max(gamma, 1e-30))

        # Initialise variational parameters
        m_k = self._init_means(X, mask, mask_f, N_eff, B, K)
        lam_k = (lam0 + 1.0).expand(B, K).clone()
        a_k = a0.expand(B, K).clone()
        b_k = b0.expand(B, K).clone()
        alpha_q = torch.ones(B, K, device=dev)
        beta_q = torch.full((B, K), gamma, device=dev)

        prev_elbo = torch.full((B,), float("-inf"), device=dev)
        no_improve = torch.zeros(B, dtype=torch.long, device=dev)

        for _it in range(self.max_iter):
            # ---- E-step: log responsibilities ----
            dig_ab = torch.digamma(alpha_q + beta_q)
            e_log_v = torch.digamma(alpha_q) - dig_ab
            e_log_1mv = torch.digamma(beta_q) - dig_ab
            e_log_pi = e_log_v + torch.cat([
                torch.zeros(B, 1, device=dev),
                torch.cumsum(e_log_1mv, dim=1)[:, :-1],
            ], dim=1)

            e_prec = a_k / b_k
            e_log_var = torch.log(b_k) - torch.digamma(a_k)

            X3 = X.unsqueeze(2)       # (B, N, 1)
            m3 = m_k.unsqueeze(1)     # (B, 1, K)
            p3 = e_prec.unsqueeze(1)  # (B, 1, K)
            lv3 = e_log_var.unsqueeze(1)
            lk3 = lam_k.unsqueeze(1)

            log_lik = (
                -0.5 * lv3
                - 0.5 * p3 * (X3 - m3).pow(2)
                - 0.5 * p3 / lk3
                - self.LOG_SQRT2PI
            )

            log_r_unnorm = e_log_pi.unsqueeze(1) + log_lik
            log_r_unnorm = log_r_unnorm.masked_fill(~mask.unsqueeze(2), -1e30)
            log_normalizer = torch.logsumexp(log_r_unnorm, dim=2)
            log_r = log_r_unnorm - log_normalizer.unsqueeze(2)
            R = log_r.exp() * mask_f.unsqueeze(2)

            # ---- M-step: closed-form updates ----
            N_k = R.sum(dim=1)
            x_bar_num = torch.bmm(X.unsqueeze(1), R).squeeze(1)
            x_bar = x_bar_num / N_k.clamp(min=1e-10)
            x2_w = torch.bmm(X.pow(2).unsqueeze(1), R).squeeze(1)
            S_k = (x2_w - N_k * x_bar.pow(2)).clamp(min=0)

            alpha_q = 1.0 + N_k
            N_sfx = torch.flip(torch.cumsum(torch.flip(N_k, [1]), 1), [1])
            beta_q = (gamma + torch.cat([
                N_sfx[:, 1:], torch.zeros(B, 1, device=dev),
            ], dim=1)).clamp(min=1e-10)

            lam_k = lam0 + N_k
            m_k = (lam0 * mu0 + x_bar_num) / lam_k

            cross = lam0 * N_k * (x_bar - mu0).pow(2) / (2.0 * lam_k)
            a_k = a0 + N_k / 2.0
            b_k = (b0 + S_k / 2.0 + cross).clamp(min=1e-8)

            # ---- ELBO ----
            ell_assign = (log_normalizer * mask_f).sum(dim=1)

            dig_aq = torch.digamma(alpha_q)
            dig_bq = torch.digamma(beta_q)
            dig_abq = torch.digamma(alpha_q + beta_q)
            kl_v = (
                log_beta_prior
                - torch.lgamma(alpha_q) - torch.lgamma(beta_q)
                + torch.lgamma(alpha_q + beta_q)
                + (alpha_q - 1.0) * dig_aq
                + (beta_q - gamma) * dig_bq
                + (1.0 + gamma - alpha_q - beta_q) * dig_abq
            ).sum(dim=1)

            kl_mu = 0.5 * (
                torch.log(lam_k / lam0)
                + lam0 / lam_k
                + lam0 * (m_k - mu0).pow(2)
                - 1.0
            ).sum(dim=1)

            kl_sigma = (
                a0 * (torch.log(b_k) - torch.log(b0))
                + torch.lgamma(a0) - torch.lgamma(a_k)
                + (a_k - a0) * torch.digamma(a_k)
                + b0 * a_k / b_k
                - a_k
            ).sum(dim=1)

            elbo = (ell_assign - kl_v - kl_mu - kl_sigma) / N_eff.clamp(min=1)

            delta = (elbo - prev_elbo).abs()
            no_improve = torch.where(
                delta < self.tol,
                no_improve + 1,
                torch.zeros_like(no_improve),
            )
            prev_elbo = elbo
            if (no_improve >= self.patience).all():
                break

        state = {
            "m_k": m_k, "lam_k": lam_k, "a_k": a_k, "b_k": b_k,
            "alpha_q": alpha_q, "beta_q": beta_q,
            "converged": no_improve >= self.patience,
        }
        return state, elbo

    def _init_means(self, X, mask, mask_f, N_eff, B, K):
        dev = self.device
        if self.init_method == "kmeans" and K >= 2:
            return self._kmeans_init(X, mask, mask_f, N_eff, B, K)
        else:
            x_min = X.masked_fill(~mask, 1e30).min(dim=1).values
            x_max = X.masked_fill(~mask, -1e30).max(dim=1).values
            t_lin = torch.linspace(0.0, 1.0, K, device=dev)
            span = (x_max - x_min).clamp(min=1e-10)
            return x_min.unsqueeze(1) + t_lin.unsqueeze(0) * span.unsqueeze(1)

    def _kmeans_init(self, X, mask, mask_f, N_eff, B, K, n_iter=20):
        dev = self.device
        x_sum = (X * mask_f).sum(dim=1)
        x_mean = x_sum / N_eff.clamp(min=1)
        centres = x_mean.unsqueeze(1)

        for _ in range(1, K):
            dists_sq = (X.unsqueeze(2) - centres.unsqueeze(1)).pow(2)
            min_dist_sq = dists_sq.min(dim=2).values * mask_f
            new_idx = min_dist_sq.argmax(dim=1)
            new_centre = X[torch.arange(B, device=dev), new_idx].unsqueeze(1)
            centres = torch.cat([centres, new_centre], dim=1)

        for _ in range(n_iter):
            dists_sq = (X.unsqueeze(2) - centres.unsqueeze(1)).pow(2)
            dists_sq = dists_sq.masked_fill(~mask.unsqueeze(2), 1e30)
            assigns = dists_sq.argmin(dim=2)
            one_hot = torch.zeros(B, X.shape[1], K, device=dev)
            one_hot.scatter_(2, assigns.unsqueeze(2), 1.0)
            one_hot = one_hot * mask_f.unsqueeze(2)
            counts = one_hot.sum(dim=1).clamp(min=1e-10)
            sums = torch.bmm(X.unsqueeze(1), one_hot).squeeze(1)
            new_centres = sums / counts
            empty = counts < 0.5
            centres = torch.where(empty, centres, new_centres)

        return centres

    def _compute_mixture_params(self, state, B, K):
        alpha_q = state["alpha_q"]
        beta_q = state["beta_q"]
        e_v = alpha_q / (alpha_q + beta_q)
        remain = torch.cat([
            torch.ones(B, 1, device=self.device),
            torch.cumprod(
                (1.0 - e_v[:, :-1]).clamp(min=0, max=1.0 - 1e-10), dim=1
            ),
        ], dim=1)
        weights = (e_v * remain).clamp(min=0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-10)

        means = state["m_k"]
        stds = (state["b_k"] / (state["a_k"] - 1.0).clamp(min=1e-6)).sqrt().clamp(min=1e-6)

        self.weights_ = weights
        self.means_ = means
        self.stds_ = stds

    def score_samples_grid(self, X, mask, grid_size):
        """Evaluate fitted mixture density on a per-edge uniform grid.

        Returns (B, grid_size) L1-normalised density tensor.
        """
        B = X.shape[0]
        dev = self.device

        x_min = X.masked_fill(~mask, 1e30).min(dim=1).values
        x_max = X.masked_fill(~mask, -1e30).max(dim=1).values
        span = (x_max - x_min).clamp(min=1e-10)

        t_lin = torch.linspace(0.0, 1.0, grid_size, device=dev)
        grid = x_min.unsqueeze(1) + t_lin * span.unsqueeze(1)

        g3 = grid.unsqueeze(2)
        m3 = self.means_.unsqueeze(1)
        s3 = self.stds_.unsqueeze(1)
        w3 = self.weights_.unsqueeze(1)

        log_gauss = (
            -0.5 * ((g3 - m3) / s3).pow(2) - s3.log() - self.LOG_SQRT2PI
        )
        log_w = w3.log().masked_fill(w3 < 1e-30, -1e30)
        density = torch.logsumexp(log_w + log_gauss, dim=2).exp()

        density = density.clamp(min=0)
        density = density / density.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return density


# =====================================================================
#  Pipeline helpers
# =====================================================================

def read_data_file(data_path: str) -> Dict[Tuple[int, int], List[float]]:
    """Read a .data file and group timestamps by (src_ip, dst_ip)."""
    edge_timestamps: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    with open(data_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            # Skip IPv6 entries (protocol 6) — only IPv4 (protocol 4) fits uint32
            if parts[0] != '4':
                continue
            src_ip = int(parts[1])
            dst_ip = int(parts[2])
            timestamp = float(parts[5])
            edge_timestamps[(src_ip, dst_ip)].append(timestamp)
    return edge_timestamps


def prepare_edge_data(
    edge_timestamps: Dict[Tuple[int, int], List[float]],
    min_timestamps: int = 10,
    max_n_per_edge: int = 50_000,
) -> Tuple[List[Tuple[int, int]], List[np.ndarray], List[Tuple[int, int]]]:
    """
    Filter, compute inter-arrival diffs, MaxAbsScale.

    Returns:
        fit_keys: edges with enough data for DPGMM
        scaled_arrays: MaxAbsScaled diff arrays (float32)
        fallback_keys: edges below min_timestamps
    """
    fit_keys: List[Tuple[int, int]] = []
    scaled_arrays: List[np.ndarray] = []
    fallback_keys: List[Tuple[int, int]] = []

    rng = np.random.default_rng(42)

    for key, ts_list in edge_timestamps.items():
        ts = np.array(ts_list, dtype=np.float64)
        if len(ts) <= min_timestamps:
            fallback_keys.append(key)
            continue

        ts.sort()
        diffs = np.abs(np.diff(ts))
        if len(diffs) < 2:
            fallback_keys.append(key)
            continue

        # Subsample if too long
        if len(diffs) > max_n_per_edge:
            idx = rng.choice(len(diffs), max_n_per_edge, replace=False)
            diffs = diffs[np.sort(idx)]

        # MaxAbsScale to [-1, 1]
        max_abs = np.abs(diffs).max()
        if max_abs < 1e-15:
            max_abs = 1.0
        scaled = (diffs / max_abs).astype(np.float32)

        # Check for degenerate (constant) data
        if np.std(scaled) < 1e-10:
            fallback_keys.append(key)
            continue

        fit_keys.append(key)
        scaled_arrays.append(scaled)

    return fit_keys, scaled_arrays, fallback_keys


def pad_to_matrix(
    arrays: List[np.ndarray], device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad list of 1-D arrays into (B, N_max) matrix + mask."""
    B = len(arrays)
    lengths = [len(a) for a in arrays]
    N_max = max(lengths)

    X = torch.zeros(B, N_max, dtype=torch.float32, device=device)
    mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for i, (arr, n) in enumerate(zip(arrays, lengths)):
        X[i, :n] = torch.from_numpy(arr).to(device)
        mask[i, :n] = True

    return X, mask


# =====================================================================
#  Main GPU pipeline
# =====================================================================

def process_data_file_gpu(
    data_path: str,
    output_path: str,
    kde_dim: int = 20,
    min_timestamps: int = 10,
    n_components: int = 100,
    gamma: float = 5.0,
    max_iter: int = 300,
    tol: float = 1e-4,
    patience: int = 20,
    chunk_size: int = 256,
    max_n_per_edge: int = 50_000,
    device: str = "cuda",
):
    """Process one .data file → one .csv KDE feature file using GPU CAVI DPGMM."""
    t0 = time.time()
    print(f"  Reading {data_path} ...")
    edge_timestamps = read_data_file(data_path)
    print(f"    Unique (src,dst) pairs: {len(edge_timestamps)}")

    # Prepare data
    fit_keys, scaled_arrays, fallback_keys = prepare_edge_data(
        edge_timestamps, min_timestamps, max_n_per_edge
    )
    print(f"    To fit: {len(fit_keys)}, fallback: {len(fallback_keys)}")

    zero_vec = np.zeros(kde_dim, dtype=np.float32)
    results: Dict[Tuple[int, int], np.ndarray] = {}

    # Fallback edges get zero vector
    for key in fallback_keys:
        results[key] = zero_vec

    if len(fit_keys) > 0:
        # Sort by length to minimise padding waste
        order = sorted(range(len(scaled_arrays)), key=lambda i: len(scaled_arrays[i]))

        n_chunks = (len(order) + chunk_size - 1) // chunk_size
        print(f"    Processing {n_chunks} GPU chunks (chunk_size={chunk_size}) on {device} ...")

        for c_idx, c0 in enumerate(range(0, len(order), chunk_size)):
            chunk_indices = order[c0: c0 + chunk_size]
            chunk_arrays = [scaled_arrays[i] for i in chunk_indices]

            X_batch, mask_batch = pad_to_matrix(chunk_arrays, device=device)

            model = BayesianGaussianMixtureGPU(
                n_components=n_components,
                gamma=gamma,
                max_iter=max_iter,
                tol=tol,
                patience=patience,
                n_init=1,
                init_method="kmeans",
                device=device,
            )
            model.fit_batch(X_batch, mask_batch)

            density = model.score_samples_grid(X_batch, mask_batch, grid_size=kde_dim)
            density_np = density.cpu().numpy()

            for local_i, orig_idx in enumerate(chunk_indices):
                results[fit_keys[orig_idx]] = density_np[local_i]

            # Free GPU memory between chunks
            del X_batch, mask_batch, model, density
            if device == "cuda":
                torch.cuda.empty_cache()

            if (c_idx + 1) % 10 == 0 or c_idx == n_chunks - 1:
                print(f"      Chunk {c_idx + 1}/{n_chunks} done")

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for (src, dst), vec in results.items():
            vec_str = ','.join(f'{v:.8f}' for v in vec)
            f.write(f'{src},{dst},{vec_str}\n')

    elapsed = time.time() - t0
    print(f"    Saved: {output_path}  ({len(results)} edges, {elapsed:.1f}s)")


# =====================================================================
#  CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated KDE features for HyperVision (batched CAVI DPGMM)')
    parser.add_argument('--data', type=str, help='Single .data file path')
    parser.add_argument('--output', type=str, help='Single output CSV path')
    parser.add_argument('--all-attacks', action='store_true',
                        help='Process all 43 attacks')
    parser.add_argument('--data-dir', type=str, default='../data')
    parser.add_argument('--output-dir', type=str, default='../kde_features')
    parser.add_argument('--kde-dim', type=int, default=20,
                        help='Dimension of KDE density vector (grid points)')
    parser.add_argument('--min-timestamps', type=int, default=10)
    parser.add_argument('--n-components', type=int, default=100,
                        help='Truncation level K for stick-breaking DP (default: 100)')
    parser.add_argument('--gamma', type=float, default=5.0,
                        help='DP concentration parameter')
    parser.add_argument('--max-iter', type=int, default=300)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--chunk-size', type=int, default=256,
                        help='Edges per GPU batch')
    parser.add_argument('--max-n-per-edge', type=int, default=50_000,
                        help='Subsample edges longer than this')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU", file=sys.stderr)
        args.device = 'cpu'

    if args.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem_in_bytes / (1024**3) if hasattr(torch.cuda.get_device_properties(0), 'total_mem_in_bytes') else torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Using GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    else:
        print("Using CPU (slower)")

    common_kwargs = dict(
        kde_dim=args.kde_dim,
        min_timestamps=args.min_timestamps,
        n_components=args.n_components,
        gamma=args.gamma,
        max_iter=args.max_iter,
        tol=args.tol,
        patience=args.patience,
        chunk_size=args.chunk_size,
        max_n_per_edge=args.max_n_per_edge,
        device=args.device,
    )

    if args.all_attacks:
        t_total = time.time()
        print(f"Processing all {len(ALL_43_ATTACKS)} attacks from {args.data_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        for attack in ALL_43_ATTACKS:
            data_path = os.path.join(args.data_dir, f'{attack}.data')
            out_path = os.path.join(args.output_dir, f'{attack}_kde.csv')
            if not os.path.exists(data_path):
                print(f"  SKIP: {data_path} not found")
                continue
            process_data_file_gpu(data_path, out_path, **common_kwargs)
        elapsed = time.time() - t_total
        print(f"\nDone! All KDE features computed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    elif args.data and args.output:
        process_data_file_gpu(args.data, args.output, **common_kwargs)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
