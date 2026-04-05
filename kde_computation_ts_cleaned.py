#!/usr/bin/env python3
"""
Offline KDE Computation Script for KAIROS-KDE

This script preprocesses temporal graph datasets to compute RKHS vectors for edges
with sufficient temporal observations (≥ min_occurrences timestamps).

Fits a truncated Dirichlet Process Gaussian Mixture Model (DPGMM) to approximate
the density of edge raw timestamps (z-scored), then evaluates the learned density
on a uniform grid to produce a fixed-size feature vector.

GPU path uses batched Coordinate Ascent Variational Inference (CAVI) with
closed-form conjugate updates.  CPU fallback uses per-edge Pyro SVI workers.

Usage:
    python kde_computation.py kairos_kde_ts CLEARSCOPE_E3 --n_workers 32
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import math

import numpy as np
import torch
import yaml
from tqdm import tqdm

# DPGMM dependencies (CPU fallback path — Pyro SVI per-edge workers)
try:
    import pyro
    import pyro.distributions as pyro_dist
    from torch.distributions import constraints as torch_constraints
    from pyro.optim import Adam as PyroAdam, ClippedAdam
    from pyro.infer import SVI, TraceEnum_ELBO
    from sklearn.cluster import KMeans as SKLearnKMeans
    from sklearn.metrics import pairwise_distances
    from sklearn.mixture import GaussianMixture
    DPGMM_AVAILABLE = True
except ImportError as _dpgmm_import_err:
    DPGMM_AVAILABLE = False

# Add pidsmaker to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pidsmaker.utils.data_utils import load_data_set, collate_temporal_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Suppress Pyro's per-edge "Guessed max_plate_nesting" INFO messages.
logging.getLogger("pyro").setLevel(logging.WARNING)


def _dpgmm_raw_ts_worker(packed_args: tuple):
    """
    Module-level worker for multiprocessing.Pool (spawn context).

    Fits a DPGMM on z-scored raw timestamps (not diffs).
    Each subprocess has its own Pyro param store (no conflicts).  GPU is
    explicitly hidden so the worker never tries to initialise CUDA.
    """
    import os as _os
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
    edge, timestamps_list, rkhs_dim = packed_args
    logging.getLogger("pyro").setLevel(logging.WARNING)
    import warnings
    warnings.filterwarnings("ignore")

    try:
        timestamps_array = np.array(timestamps_list, dtype=np.float64).astype(np.float32)

        if len(timestamps_array) < 2:
            return edge, None

        # Z-score standardise raw timestamps
        ts_mean = float(np.mean(timestamps_array))
        ts_std  = float(np.std(timestamps_array))
        if ts_std < 1e-12:
            ts_std = 1.0
        z_ts = (timestamps_array - ts_mean) / ts_std

        dpgmm = _PyroDPGMM(device="cpu")
        means, stds, weights = dpgmm.fit(z_ts)

        gmm = _PyroDPGMM.build_gmm(means, stds, weights)
        if gmm is None:
            return edge, None

        grid_min, grid_max = float(z_ts.min()), float(z_ts.max())
        if grid_max - grid_min < 1e-12:
            return edge, np.ones(rkhs_dim, dtype=np.float32)

        grid = np.linspace(grid_min, grid_max, rkhs_dim)
        densities = np.exp(gmm.score_samples(grid.reshape(-1, 1))).astype(np.float32)
        return edge, densities

    except Exception:
        return edge, None


def _load_file_worker(args: tuple):
    """
    Module-level worker for ThreadPoolExecutor.

    Loads one .TemporalData.simple file and returns four aligned numpy arrays:
    (src, dst, edge_types, t).  Using threads (not processes) is correct here
    because torch.load releases the GIL during disk I/O, allowing true
    concurrent file loading without CUDA-fork issues.
    """
    file_path, node_type_dim, edge_type_dim = args
    try:
        data = torch.load(file_path, map_location='cpu')
        src = data.src.cpu().numpy().astype(np.int64)
        dst = data.dst.cpu().numpy().astype(np.int64)
        t   = data.t.cpu().numpy()
        if hasattr(data, 'msg') and data.msg is not None:
            edge_types = extract_edge_type_from_msg(
                data.msg, node_type_dim, edge_type_dim
            ).astype(np.int64)
        elif hasattr(data, 'edge_type') and data.edge_type is not None:
            edge_types = data.edge_type.max(dim=1).indices.cpu().numpy().astype(np.int64)
        else:
            edge_types = np.zeros(len(src), dtype=np.int64)
        return src, dst, edge_types, t
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# GPU-accelerated batch operations
# ---------------------------------------------------------------------------

def _gpu_batched_dpgmm_cavi(
    z_data_list: list,
    edge_keys: list,
    rkhs_dim: int,
    K: int = 20,
    gamma: float = 5.0,
    n_iter: int = 150,
    tol: float = 1e-4,
    patience: int = 15,
    chunk_size: int = 256,
    max_n_per_edge: int = 50_000,
    device: str = "cuda",
) -> dict:
    """
    Fully GPU-native batched DP-GMM via Coordinate Ascent Variational Inference
    (CAVI), designed for a single high-VRAM GPU (e.g. NVIDIA A100).

    Replaces the old three-phase Pyro pipeline:
        Phase B (GPU K-Means init)
        + Phase C (CPU SVI workers via Pyro)
        + Phase D (GPU GMM density eval)
    with a single all-GPU pass that processes all edges simultaneously in chunks.

    Why CAVI instead of Pyro SVI?
    ──────────────────────────────
    • The conjugate DP-GMM (Normal-InverseGamma prior + stick-breaking) admits
      closed-form E-step and M-step updates — no auto-diff or optimizer needed.
    • All updates are pure PyTorch tensor ops → naturally vectorised over a
      batch dimension B (edges) with no per-edge Python loops.
    • No Pyro global param store → no process isolation required → no
      multiprocessing.Pool, no spawn overhead, no IPC serialisation.
    • Convergence in 50–150 CAVI iterations vs. 500 SVI epochs → ~5-10× fewer
      arithmetic operations per edge.
    • cuBLAS batched GEMM (torch.bmm) used for sufficient statistics, keeping
      peak VRAM at O(B × N_max × K) ≈ 1–2 GB per chunk on A100.

    Conjugate model (data is z-scored so priors are unit-scale)
    ────────────────
        v_k  ~ Beta(1, γ)                 (stick-breaking, K-truncated)
        μ_k  ~ N(μ₀=0, 1/λ₀=1)
        σ_k² ~ InvGamma(α₀=3, β₀=0.5)   (E[σ²]=0.25)
        zₙ   ~ Categorical(π)
        xₙ|k ~ N(μ_k, σ_k²)

    Variational family:
        q(vₖ)  = Beta(ã_k, b̃_k)
        q(μₖ)  = N(mₖ, 1/λ̃_k)
        q(σₖ²) = InvGamma(ã_k, b̃_k)
        q(zₙ)  = Categorical(rₙ)

    CAVI closed-form updates
    ────────────────────────
    E-step:
        log r_nk ∝ E[log π_k] + E[log p(x_n | k)]
        E[log π_k]       = ψ(ã_k) - ψ(ã_k+b̃_k) + Σ_{j<k}[ψ(b̃_j)-ψ(ã_j+b̃_j)]
        E[log p(x_n|k)]  = -½E[log σ²_k] - ½E[σ⁻²_k](x_n-m_k)² - ½E[σ⁻²_k]/λ̃_k

    M-step (closed form from Normal-InverseGamma conjugacy):
        N_k  = Σ_n r_nk,   x̄_k = Σ_n r_nk x_n / N_k,   S_k = Σ_n r_nk (x_n-x̄_k)²
        ã_k  = 1 + N_k,                 b̃_k = γ + Σ_{j>k} N_j
        λ̃_k  = λ₀ + N_k,               m_k  = (λ₀μ₀ + Σ r_nk x_n) / λ̃_k
        ã_k  = α₀ + N_k/2,             b̃_k  = β₀ + S_k/2 + λ₀N_k(x̄_k-μ₀)²/(2λ̃_k)

    Args:
        z_data_list   : list of 1-D float32 np arrays (z-scored raw timestamps)
        edge_keys     : parallel list of edge identifier tuples (output dict keys)
        rkhs_dim      : number of uniform grid points for the density feature vector
        K             : DP truncation level (number of mixture components)
        gamma         : DP concentration parameter
        n_iter        : maximum CAVI iterations per chunk
        tol           : convergence tolerance on per-point normalised ELBO change
        patience      : early-stop after this many non-improving iterations
        chunk_size    : edges processed simultaneously (tune to GPU VRAM;
                        256 ≈ 1–2 GB per chunk for typical DARPA E3 edge lengths)
        max_n_per_edge: randomly subsample edges longer than this to bound VRAM
        device        : torch device string ('cuda' or 'cpu')

    Returns:
        dict mapping each edge_key → float32 np array of shape (rkhs_dim,)
    """
    LOG_SQRT2PI = 0.9189385332  # log(√(2π))
    results: dict = {}

    # Sort edges by ascending length so each chunk has similar N_max, minimising
    # padding waste (shorter edges are padded less when grouped together).
    order = sorted(range(len(z_data_list)), key=lambda i: len(z_data_list[i]))

    for c0 in tqdm(range(0, len(order), chunk_size), desc="DPGMM CAVI (GPU)"):
        chunk_idx = order[c0 : c0 + chunk_size]
        B = len(chunk_idx)

        # Optionally subsample very long edges to keep the R=(B,N,K) tensor
        # within VRAM budget.
        chunk_zd = []
        for idx in chunk_idx:
            z = z_data_list[idx]
            if len(z) > max_n_per_edge:
                sel = np.random.choice(len(z), max_n_per_edge, replace=False)
                z   = z[np.sort(sel)]
            chunk_zd.append(z)

        lengths = [len(z) for z in chunk_zd]
        N_max   = max(lengths)

        # ── (B, N_max) padded data tensor and boolean validity mask ──────
        X    = torch.zeros(B, N_max, device=device, dtype=torch.float32)
        mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)
        for i, (z, n) in enumerate(zip(chunk_zd, lengths)):
            X[i, :n]    = torch.from_numpy(z).to(device)
            mask[i, :n] = True
        mask_f = mask.float()           # (B, N_max)  — float version for arithmetic
        N_eff  = mask_f.sum(dim=1)      # (B,)

        # ── Priors (z-scored data → μ₀=0, λ₀=1, α₀=3, β₀=0.5) ──────────
        #   E[σ²] = β₀/(α₀-1) = 0.25  (appropriate for standardised data)
        mu0  = torch.zeros(B, 1, device=device)
        lam0 = torch.ones (B, 1, device=device)
        a0   = torch.full ((B, 1), 3.0, device=device)
        b0   = torch.full ((B, 1), 0.5, device=device)

        # ── Initialise variational parameters ─────────────────────────────
        # Means: evenly spaced in [-2, 2] (covers ~95% of z-scored mass)
        t_init  = torch.linspace(-2.0, 2.0, K, device=device)   # (K,)
        m_k     = t_init.unsqueeze(0).expand(B, K).clone()       # (B, K)
        lam_k   = (lam0 + 1.0).expand(B, K).clone()              # (B, K)
        a_k     = a0.expand(B, K).clone()                         # (B, K)
        b_k     = b0.expand(B, K).clone()                         # (B, K)
        # Stick-breaking Beta variational params
        alpha_q = torch.ones (B, K, device=device)
        beta_q  = torch.full ((B, K), gamma, device=device)

        # ── CAVI loop ──────────────────────────────────────────────────────
        prev_elbo  = torch.full((B,), float('-inf'), device=device)
        no_improve = torch.zeros(B, dtype=torch.long, device=device)

        for _ in range(n_iter):
            # ---- E-step: log responsibilities ----
            # E[log π_k] via stick-breaking expectations  (B, K)
            dig_ab    = torch.digamma(alpha_q + beta_q)
            e_log_v   = torch.digamma(alpha_q) - dig_ab
            e_log_1mv = torch.digamma(beta_q)  - dig_ab
            e_log_pi  = e_log_v + torch.cat([
                torch.zeros(B, 1, device=device),
                torch.cumsum(e_log_1mv, dim=1)[:, :-1],
            ], dim=1)                                              # (B, K)

            # E[σ⁻²_k] = ã_k/b̃_k  and  E[log σ²_k] = log b̃_k − ψ(ã_k)
            e_prec    = a_k / b_k                                 # (B, K)
            e_log_var = torch.log(b_k) - torch.digamma(a_k)      # (B, K)

            # E[log p(xₙ|k)] for all (n, k):
            #   −½E[log σ²] − ½E[σ⁻²](x−m)² − ½E[σ⁻²]/λ̃ − ½log(2π)
            # Broadcast: X (B,N,1), m_k/e_prec/lam_k/e_log_var → (B,1,K)
            X3  = X.unsqueeze(2)
            m3  = m_k.unsqueeze(1)
            p3  = e_prec.unsqueeze(1)
            lv3 = e_log_var.unsqueeze(1)
            lk3 = lam_k.unsqueeze(1)
            log_lik = (-0.5 * lv3
                       - 0.5 * p3 * (X3 - m3).pow(2)
                       - 0.5 * p3 / lk3
                       - LOG_SQRT2PI)                             # (B, N, K)

            # Normalise; zero padded rows
            log_r = e_log_pi.unsqueeze(1) + log_lik              # (B, N, K)
            log_r = log_r.masked_fill(~mask.unsqueeze(2), -1e30)
            log_r = log_r - torch.logsumexp(log_r, dim=2, keepdim=True)
            R     = log_r.exp() * mask_f.unsqueeze(2)            # (B, N, K)

            # ---- M-step: closed-form updates ----
            # Sufficient statistics via batched GEMM — avoids (B,N,K) temps
            N_k       = R.sum(dim=1)                              # (B, K)
            x_bar_num = torch.bmm(X.unsqueeze(1), R).squeeze(1)  # (B, K): Σ rₙₖ xₙ
            x_bar     = x_bar_num / N_k.clamp(min=1e-10)         # (B, K)
            x2_w      = torch.bmm(                               # (B, K): Σ rₙₖ xₙ²
                X.pow(2).unsqueeze(1), R).squeeze(1)
            S_k       = (x2_w - N_k * x_bar.pow(2)).clamp(min=0) # (B, K)

            # Stick-breaking: ã_k=1+N_k;  b̃_k=γ+Σ_{j>k}N_j
            alpha_q = 1.0 + N_k
            N_sfx   = torch.flip(torch.cumsum(torch.flip(N_k, [1]), 1), [1])
            beta_q  = (gamma + torch.cat([
                N_sfx[:, 1:], torch.zeros(B, 1, device=device)
            ], dim=1)).clamp(min=1e-10)

            # Normal: λ̃_k=λ₀+N_k;  m_k=(λ₀μ₀+Σrₙₖxₙ)/λ̃_k
            lam_k = lam0 + N_k
            m_k   = (lam0 * mu0 + x_bar_num) / lam_k

            # InvGamma: ã_k=α₀+N_k/2;  b̃_k=β₀+S_k/2+cross
            cross = lam0 * N_k * (x_bar - mu0).pow(2) / (2.0 * lam_k)
            a_k   = a0 + N_k / 2.0
            b_k   = (b0 + S_k / 2.0 + cross).clamp(min=1e-8)

            # Convergence: normalised data ELBO per point
            elbo       = (R * log_lik).sum(dim=[1, 2]) / N_eff.clamp(min=1)
            delta      = (elbo - prev_elbo).abs()
            no_improve = torch.where(
                delta < tol, no_improve + 1, torch.zeros_like(no_improve))
            prev_elbo  = elbo
            if (no_improve >= patience).all():
                break

        # ── Extract GMM weights via stick-breaking expectation ────────────
        e_v     = alpha_q / (alpha_q + beta_q)                   # (B, K)
        remain  = torch.cat([
            torch.ones(B, 1, device=device),
            torch.cumprod((1.0 - e_v[:, :-1]).clamp(min=0, max=1.0 - 1e-10), dim=1),
        ], dim=1)                                                  # (B, K)
        weights = (e_v * remain).clamp(min=0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-10)

        # E[σ²_k] = b̃_k/(ã_k−1);  std_k = √E[σ²_k]
        stds = (b_k / (a_k - 1.0).clamp(min=1e-6)).sqrt().clamp(min=1e-6)  # (B, K)

        # ── Evaluate density on a rkhs_dim-point grid (z-scored space) ───
        x_min_e = X.masked_fill(~mask,  1e30).min(dim=1).values   # (B,)
        x_max_e = X.masked_fill(~mask, -1e30).max(dim=1).values   # (B,)
        span    = (x_max_e - x_min_e).clamp(min=1e-10)
        t_lin   = torch.linspace(0.0, 1.0, rkhs_dim, device=device)
        grid    = x_min_e.unsqueeze(1) + t_lin * span.unsqueeze(1)  # (B, D)

        g3 = grid.unsqueeze(2)     # (B, D, 1)
        m3 = m_k.unsqueeze(1)      # (B, 1, K)
        s3 = stds.unsqueeze(1)     # (B, 1, K)
        w3 = weights.unsqueeze(1)  # (B, 1, K)
        log_gauss = -0.5 * ((g3 - m3) / s3).pow(2) - s3.log() - LOG_SQRT2PI
        log_w     = w3.log().masked_fill(w3 < 1e-30, -1e30)
        density   = torch.logsumexp(log_w + log_gauss, dim=2).exp()  # (B, D)

        # L1-normalise onto the probability simplex
        density = density.clamp(min=0)
        density = density / density.sum(dim=1, keepdim=True).clamp(min=1e-10)

        density_np = density.cpu().numpy()
        for local_i, orig_idx in enumerate(chunk_idx):
            results[edge_keys[orig_idx]] = density_np[local_i].astype(np.float32)

    return results


def load_config(model_name: str) -> Dict:
    """Load configuration from YAML file."""
    config_file = f"config/{model_name}.yml"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load included config if specified
    if '_include_yml' in cfg:
        base_config_file = f"config/{cfg['_include_yml']}.yml"
        with open(base_config_file, 'r') as f:
            base_cfg = yaml.safe_load(f)
        # Merge configs (cfg overrides base_cfg)
        base_cfg.update(cfg)
        cfg = base_cfg
    
    return cfg


class SimpleConfig:
    """Simple config object that mimics pidsmaker config structure."""
    def __init__(self, cfg_dict: Dict, dataset_name: str):
        self.dataset = type('obj', (object,), {
            'name': dataset_name,
            'train_files': cfg_dict.get('dataset', {}).get('train_files', []),
            'val_files': cfg_dict.get('dataset', {}).get('val_files', []),
            'test_files': cfg_dict.get('dataset', {}).get('test_files', [])
        })()
        
        # Add required attributes
        self.feat_inference = type('obj', (object,), {
            '_task_path': cfg_dict.get('feat_inference', {}).get('_task_path', f'artifacts/{dataset_name}/feat_inference')
        })()
        
        self.featurization = type('obj', (object,), cfg_dict.get('featurization', {}))()
        self.construction = type('obj', (object,), cfg_dict.get('construction', {}))()
        self.batching = type('obj', (object,), cfg_dict.get('batching', {}))()
        
        self._dict = cfg_dict
        
    def get(self, key, default=None):
        return self._dict.get(key, default)


# ---------------------------------------------------------------------------
# Embedded DPGMM (ported from tpr/pyro_dpgmm.py) — used by the CPU
# fallback path (_dpgmm_raw_ts_worker) and sequential n_workers=1 path.
# ---------------------------------------------------------------------------

def _expected_log_sticks(alpha, beta):
    """Compute E_q[log v_k] + running sum E_q[log(1-v_j)]."""
    dig_sum   = torch.digamma(alpha + beta)
    e_log_v   = torch.digamma(alpha) - dig_sum
    e_log_1mv = torch.digamma(beta)  - dig_sum
    prefix    = torch.cumsum(e_log_1mv, dim=0)
    prefix    = torch.cat([prefix.new_zeros(1), prefix[:-1]])
    return e_log_v + prefix


class _PyroDPGMM:
    """
    Truncated DP-GMM with stick-breaking prior.

    Adapted from tpr/pyro_dpgmm.py.  Key difference: priors are
    parameterised from the observed data range so that no external
    MinMax normalisation is needed.

    Hardcoded defaults (not exposed in YAML):
        K              = 20     (truncation level)
        gamma          = 5.0    (DP concentration)
        num_epochs     = 500    (max SVI epochs)
        patience       = 50     (early-stopping patience)
        batch_size     = 10000  (SVI mini-batch size)
        truncate_thres = 0.99   (cumulative weight threshold)
    """

    # ---- class-level defaults (tweak here, not in YAML) ----
    DEFAULT_K          = 20
    DEFAULT_GAMMA      = 5.0
    DEFAULT_NUM_EPOCHS = 500   # most edges converge well before this
    DEFAULT_PATIENCE   = 50    # stop early if EMA-ELBO stalls for 50 epochs
    DEFAULT_BATCH_SIZE = 10000
    DEFAULT_TRUNCATE   = 0.99

    def __init__(self, K=None, gamma=None, device=None):
        self.K     = K     or self.DEFAULT_K
        self.gamma = gamma or self.DEFAULT_GAMMA
        # Workers pass device="cpu" to avoid CUDA-fork issues; main process
        # auto-detects (will use GPU if available).
        self.device = device if device is not None else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------ #
    #  Initialisation helpers                                              #
    # ------------------------------------------------------------------ #
    def _set_data_scaled_priors(self, data: torch.Tensor):
        """Set prior hyper-parameters from the observed data range."""
        data_np = data.cpu().numpy()
        data_mean = float(np.mean(data_np))
        data_var  = float(np.var(data_np))
        if data_var < 1e-12:
            data_var = 1.0  # degenerate edge

        # mu0 = centre of data; lambda0 = 1/data_var (weak prior)
        self.mu0     = torch.tensor(data_mean, device=self.device, dtype=torch.float32)
        self.lambda0 = torch.tensor(1.0 / data_var, device=self.device, dtype=torch.float32)

        # InverseGamma prior: E[sigma2] ~ small fraction of data variance
        alpha0 = 3.0
        expected_cluster_var = data_var * 0.01  # 1 % of data spread
        beta0  = expected_cluster_var * (alpha0 - 1)
        self.alpha0 = torch.tensor(alpha0, device=self.device, dtype=torch.float32)
        self.beta0  = torch.tensor(max(beta0, 1e-8), device=self.device, dtype=torch.float32)

        # Store data range for the guide constraint
        data_min = float(np.min(data_np))
        data_max = float(np.max(data_np))
        margin   = (data_max - data_min) * 0.1 + 1e-8
        self._data_lo = data_min - margin
        self._data_hi = data_max + margin

    def _kmeans_init(self, data: torch.Tensor):
        """K-Means init with nearest-neighbour variance heuristic (from tpr)."""
        X_np = data.cpu().numpy().reshape(-1, 1)
        global_var = float(np.var(X_np))

        while True:
            kmeans = SKLearnKMeans(n_clusters=self.K, random_state=69, n_init=10).fit(X_np)
            labels = kmeans.predict(X_np)
            means  = kmeans.cluster_centers_.reshape(-1)

            if self.K > 1:
                dists = pairwise_distances(means.reshape(-1, 1))
                np.fill_diagonal(dists, np.inf)
                min_center_dists = np.min(dists, axis=1)
            else:
                min_center_dists = np.array([np.sqrt(global_var) * 3.0])

            variances = []
            for k in range(self.K):
                pts = X_np[labels == k]
                topo_var = (min_center_dists[k] / 3.0) ** 2
                if len(pts) > 1:
                    adj = float(np.var(pts)) + topo_var * 0.1
                else:
                    adj = topo_var
                variances.append(max(adj, 1e-6))

            if len(means) == self.K:
                break
            self.K = max(len(means), 1)

        self.init_means     = torch.tensor(means, device=self.device, dtype=torch.float32)
        self.init_variances = torch.tensor(variances, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------ #
    #  Pyro model / guide                                                  #
    # ------------------------------------------------------------------ #
    def _stick_breaking(self, v):
        eps = 1e-6
        remaining = torch.cumprod(1 - v + eps, dim=-1)
        remaining = torch.roll(remaining, shifts=1, dims=0)
        remaining[0] = 1.0
        pi = v * remaining
        return pi / pi.sum(-1, keepdim=True)

    def _model(self, data, weights=None):
        if weights is None:
            weights = torch.ones(data.shape[0], device=self.device)
        N, K = data.shape[0], self.K

        with pyro.plate("components", K):
            v = pyro.sample("v", pyro_dist.Beta(
                torch.ones(K, device=self.device) * self.gamma,
                torch.full((K,), self.gamma, device=self.device)))
            mu = pyro.sample("mu", pyro_dist.Normal(
                self.mu0, 1.0 / torch.sqrt(self.lambda0)))
            sigma2 = pyro.sample("sigma2",
                pyro_dist.InverseGamma(self.alpha0, self.beta0)).to(self.device)

        pi = self._stick_breaking(v)

        with pyro.plate("data", N, subsample_size=self._batch_size) as ind:
            x_batch = data[ind]
            w_batch = weights[ind]
            z = pyro.sample("z",
                pyro_dist.Categorical(pi).expand([x_batch.shape[0]]),
                infer={"enumerate": "parallel", "scale": w_batch})
            safe_sigma = torch.clamp(sigma2[z], min=1e-8)
            pyro.sample("obs", pyro_dist.Normal(mu[z], safe_sigma.sqrt()), obs=x_batch)

    def _guide(self, data, weights=None):
        K, N = self.K, data.shape[0]

        m_loc = pyro.param("m_loc", self.init_means.clone(),
                           constraint=torch_constraints.interval(self._data_lo, self._data_hi))
        alpha_q = pyro.param("alpha_q",
            torch.ones(K, device=self.device) * self.gamma,
            constraint=torch_constraints.greater_than(1e-3))
        beta_q = pyro.param("beta_q",
            torch.ones(K, device=self.device) * self.gamma,
            constraint=torch_constraints.greater_than(1e-3))
        m_scl = pyro.param("m_scl",
            self.init_variances.clone().sqrt().clamp(min=1e-5),
            constraint=torch_constraints.greater_than(1e-6))
        a_q = pyro.param("a_q",
            torch.full((K,), max(float(self.alpha0), 1.05), device=self.device),
            constraint=torch_constraints.greater_than(1.01))
        b_q = pyro.param("b_q",
            self.init_variances.clone().clamp(min=1e-6),
            constraint=torch_constraints.greater_than(1e-8))

        with pyro.plate("components", K):
            pyro.sample("v", pyro_dist.Beta(alpha_q, beta_q))
            pyro.sample("mu", pyro_dist.Normal(m_loc, m_scl))
            pyro.sample("sigma2", pyro_dist.InverseGamma(a_q, b_q)).to(self.device)

        with pyro.plate("data", N, subsample_size=self._batch_size):
            pass

    # ------------------------------------------------------------------ #
    #  Fit + parameter extraction                                          #
    # ------------------------------------------------------------------ #
    def fit(self, data_np: np.ndarray,
            num_epochs: int = None,
            batch_size: int = None,
            patience: int  = None):
        """
        Fit the DPGMM on 1-D numpy data.

        Returns
        -------
        means, stds, weights : np.ndarray
            Truncated (99 %) GMM parameters.
        """
        num_epochs = num_epochs or self.DEFAULT_NUM_EPOCHS
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        patience   = patience   or self.DEFAULT_PATIENCE

        data_t = torch.tensor(data_np, device=self.device, dtype=torch.float32).squeeze(-1)
        N = data_t.shape[0]
        self.K = min(self.K, N)
        self._batch_size = min(batch_size, N)

        self._set_data_scaled_priors(data_t)
        self._kmeans_init(data_t)
        self.K = self.init_means.shape[0]

        pyro.clear_param_store()
        # ClippedAdam clips gradients to clip_norm before each update,
        # preventing NaN explosions from large-scale data.
        optimizer = ClippedAdam({"lr": 0.0005, "clip_norm": 10.0})
        svi = SVI(self._model, self._guide, optimizer, loss=TraceEnum_ELBO())

        steps_per_epoch = math.ceil(N / self._batch_size)
        best_elbo, no_improve = float("-inf"), 0
        ema_elbo, best_params = None, None

        weights_t = torch.ones(N, device=self.device)

        for epoch in range(num_epochs):
            epoch_loss = sum(svi.step(data_t, weights_t) for _ in range(steps_per_epoch))

            # Skip NaN/Inf steps (can occur early in training; ClippedAdam
            # will self-correct on the next step).
            if not math.isfinite(epoch_loss):
                no_improve += 1
                if no_improve >= patience:
                    break
                continue

            raw_elbo = -epoch_loss / steps_per_epoch
            ema_elbo = raw_elbo if ema_elbo is None else 0.9 * ema_elbo + 0.1 * raw_elbo

            if ema_elbo > best_elbo + 1e-3:
                best_elbo  = ema_elbo
                no_improve = 0
                best_params = pyro.get_param_store().get_state()
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        if best_params is not None:
            pyro.get_param_store().set_state(best_params)

        return self._truncate()

    def _get_params(self):
        with torch.no_grad():
            log_pi = _expected_log_sticks(
                pyro.param("alpha_q"), pyro.param("beta_q"))
            pi = torch.exp(log_pi)
            weights = (pi / pi.sum()).cpu().numpy()
            means   = pyro.param("m_loc").cpu().numpy()
            variances = (pyro.param("b_q") / (pyro.param("a_q") - 1)).cpu().numpy()
        return means, variances, weights

    def _truncate(self, threshold=None):
        threshold = threshold or self.DEFAULT_TRUNCATE
        means, variances, weights = self._get_params()
        weights = weights / weights.sum()
        idx = np.argsort(weights)[::-1]
        w_s, m_s, std_s = weights[idx], means[idx], np.sqrt(variances[idx])
        cut = np.searchsorted(np.cumsum(w_s), threshold) + 1
        kept_w = w_s[:cut]; kept_w = kept_w / kept_w.sum()
        return m_s[:cut], std_s[:cut], kept_w

    @staticmethod
    def build_gmm(means, stds, weights):
        """Wrap truncated params into a sklearn GaussianMixture for scoring."""
        K = len(means)
        if K == 0:
            return None
        gmm = GaussianMixture(n_components=K, covariance_type="diag")
        gmm.weights_     = weights
        gmm.means_       = means.reshape(K, 1)
        covs             = (stds ** 2).reshape(K, 1)
        gmm.covariances_ = covs
        safe = np.maximum(covs, np.finfo(covs.dtype).eps)
        gmm.precisions_cholesky_ = 1.0 / np.sqrt(safe)
        gmm.n_features_in_ = 1
        gmm.n_components_  = K
        return gmm


class KDEVectorComputer:
    """
    Computes RKHS vectors from edge timestamps using DPGMM density estimation.

    Fits a truncated Dirichlet Process Gaussian Mixture Model on z-scored raw
    timestamp values and evaluates the learned density on a uniform grid of
    *rkhs_dim* points.  The GPU CAVI path handles the heavy lifting; this class
    mainly holds configuration and provides the sequential (n_workers=1) fallback.
    """

    def __init__(
        self,
        rkhs_dim: int = 20,
        min_occurrences: int = 10,
        **_kwargs,
    ):
        self.rkhs_dim = rkhs_dim
        self.min_occurrences = min_occurrences

        logger.info(f"Initialized KDEVectorComputer:")
        logger.info(f"  - RKHS dimension: {rkhs_dim}")
        logger.info(f"  - Min occurrences: {min_occurrences}")

    def timestamps_to_rkhs_vector_dpgmm(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Convert raw timestamps to RKHS vector using DPGMM density estimation.

        Pipeline:
        1. Z-score standardise the raw timestamps.
        2. Fit _PyroDPGMM (K=20, early-stopped SVI) on the standardised values.
        3. Wrap the truncated (99 %) GMM params into a sklearn GaussianMixture.
        4. Evaluate the density on a uniform grid spanning
           [min(z_ts), max(z_ts)] with *rkhs_dim* points.
        5. Return the density values as the feature vector.

        Args:
            timestamps: Array of raw timestamps for a single edge

        Returns:
            RKHS vector of dimension rkhs_dim (density evaluated on grid)
        """
        if not DPGMM_AVAILABLE:
            raise RuntimeError(
                "DPGMM dependencies not installed.  "
                "Run:  mamba install -c conda-forge pyro-ppl  "
                "(or pip install pyro-ppl)"
            )

        try:
            ts = np.array(timestamps, dtype=np.float64).astype(np.float32)
            if len(ts) < 2:
                logger.warning("Not enough timestamps, using zero vector")
                return np.zeros(self.rkhs_dim, dtype=np.float32)

            # Z-score standardise raw timestamps
            ts_mean = float(np.mean(ts))
            ts_std  = float(np.std(ts))
            if ts_std < 1e-12:
                ts_std = 1.0
            z_ts = (ts - ts_mean) / ts_std

            # Fit DPGMM on standardised timestamps
            dpgmm = _PyroDPGMM()
            means, stds, weights = dpgmm.fit(z_ts)

            # Build sklearn GMM
            gmm = _PyroDPGMM.build_gmm(means, stds, weights)
            if gmm is None:
                logger.warning("DPGMM produced no components, using zero vector")
                return np.zeros(self.rkhs_dim, dtype=np.float32)

            # Evaluate density on a uniform grid (standardised domain)
            grid_min = float(z_ts.min())
            grid_max = float(z_ts.max())
            if grid_max - grid_min < 1e-12:
                # Degenerate: all timestamps identical -> flat density
                return np.ones(self.rkhs_dim, dtype=np.float32)

            grid = np.linspace(grid_min, grid_max, self.rkhs_dim)
            log_densities = gmm.score_samples(grid.reshape(-1, 1))
            densities = np.exp(log_densities)

            return densities.astype(np.float32)

        except Exception as e:
            logger.warning(f"DPGMM computation on timestamps failed: {e}, using zero vector")
            return np.zeros(self.rkhs_dim, dtype=np.float32)


def get_latest_dataset_folder(dataset_name: str, feat_hash: str = None) -> str:
    """
    Get the dataset folder in the corresponding feat_inference directory.
    
    Args:
        dataset_name: Name of the dataset
        feat_hash: Optional hash prefix/name of the specific run directory.
                   If provided, uses that directory directly instead of
                   auto-selecting the latest one by modification time.
                   E.g. "a1b2c3d4" to select
                   artifacts/feat_inference/{dataset}/feat_inference/a1b2c3d4/
        
    Returns:
        Path to the selected folder
    """
    import glob
    import os
    
    base_path = f"/scratch/asawan15/PIDSMaker/artifacts/feat_inference/{dataset_name}/feat_inference"
    
    if feat_hash is not None:
        target = os.path.join(base_path, feat_hash)
        if not os.path.isdir(target):
            # Try glob in case user gave a prefix
            matches = glob.glob(os.path.join(base_path, f"{feat_hash}*"))
            matches = [m for m in matches if os.path.isdir(m)]
            if not matches:
                raise FileNotFoundError(
                    f"No directory matching '{feat_hash}' under {base_path}. "
                    f"Available: {[os.path.basename(d) for d in glob.glob(os.path.join(base_path, '*')) if os.path.isdir(d)]}"
                )
            target = matches[0]
        logger.info(f"Using specified feat_inference folder: {target}")
        return target
    
    # Find all subdirectories (versioned folders)
    subdirs = glob.glob(os.path.join(base_path, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    
    if not subdirs:
        raise FileNotFoundError(f"No dataset folders found at {base_path}")
    
    # Get the latest by modification time
    latest_dir = max(subdirs, key=lambda d: os.path.getmtime(d))
    logger.info(f"Found latest dataset folder: {latest_dir}")
    
    return latest_dir


def extract_edge_type_from_msg(msg: torch.Tensor, node_type_dim: int = 8, edge_type_dim: int = 16) -> np.ndarray:
    """
    Extract edge type indices from the msg tensor.
    
    The msg tensor structure is: [src_type, src_emb, edge_type, dst_type, dst_emb]
    where edge_type is one-hot encoded at position (node_type_dim + emb_dim).
    
    Args:
        msg: Message tensor of shape (N, msg_dim)
        node_type_dim: Number of node type dimensions (default 8 for DARPA E3)
        edge_type_dim: Number of edge type dimensions (default 16 for DARPA E3)
        
    Returns:
        Array of edge type indices (argmax of the one-hot edge type portion)
    """
    msg_dim = msg.shape[1]
    
    # Calculate emb_dim from msg structure:
    # msg_dim = node_type_dim + emb_dim + edge_type_dim + node_type_dim + emb_dim
    # msg_dim = 2 * node_type_dim + 2 * emb_dim + edge_type_dim
    emb_dim = (msg_dim - 2 * node_type_dim - edge_type_dim) // 2
    
    # Edge type starts at: node_type_dim + emb_dim
    edge_type_start = node_type_dim + emb_dim
    edge_type_end = edge_type_start + edge_type_dim
    
    # Extract edge type slice and get argmax
    edge_type_slice = msg[:, edge_type_start:edge_type_end]
    edge_types = edge_type_slice.argmax(dim=1).cpu().numpy()
    
    return edge_types


def extract_edge_timestamps(
    cfg: SimpleConfig,
    dataset_name: str,
    feat_hash: str = None,
    n_workers: int = 1,
) -> Dict[Tuple[int, int, int], List[float]]:
    """
    Extract all edges and their timestamps from the dataset.

    Speedups vs the original serial implementation:
    1. Files are loaded in parallel using a ThreadPoolExecutor (IO-bound;
       torch.load releases the GIL during disk reads).
    2. Per-edge grouping is done with numpy vectorised ops (lexsort +
       boundary detection) instead of a Python-level row-by-row loop,
       cutting the grouping step from O(N) Python iterations to a handful
       of C-level numpy calls.

    Args:
        cfg:          Configuration object.
        dataset_name: Name of the dataset.
        feat_hash:    Optional hash/prefix of the feat_inference run dir.
                      If None the latest directory (by mtime) is used.
        n_workers:    Number of threads for parallel file loading.

    Returns:
        Dictionary mapping (src, dst, edge_type) -> list of timestamps.
    """
    import glob

    logger.info(f"Loading dataset: {dataset_name}")
    base_path = get_latest_dataset_folder(dataset_name, feat_hash=feat_hash)

    node_type_dim = getattr(getattr(cfg, 'dataset', None), 'num_node_types', 8)
    edge_type_dim = getattr(getattr(cfg, 'dataset', None), 'num_edge_types', 16)
    logger.info(f"Using node_type_dim={node_type_dim}, edge_type_dim={edge_type_dim}")

    # ------------------------------------------------------------------ #
    #  1. Collect all file paths across splits                             #
    # ------------------------------------------------------------------ #
    all_files = []
    for split in ['train', 'val']:
        pattern = f"{base_path}/edge_embeds/{split}/*.TemporalData.simple"
        split_files = glob.glob(pattern)
        if not split_files:
            logger.warning(f"No files found for {split} split at {pattern}")
        else:
            logger.info(f"Found {len(split_files)} files for {split} split")
            all_files.extend(split_files)

    if not all_files:
        logger.warning("No files found for any split")
        return {}

    # ------------------------------------------------------------------ #
    #  2. Load files in parallel (IO-bound → threads)                     #
    # ------------------------------------------------------------------ #
    io_workers = min(n_workers, len(all_files))
    logger.info(f"Loading {len(all_files)} files with {io_workers} IO thread(s)...")

    file_args = [(f, node_type_dim, edge_type_dim) for f in all_files]
    chunks = []
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for result in tqdm(
            pool.map(_load_file_worker, file_args),
            total=len(file_args),
            desc="Loading files",
        ):
            if result is not None:
                chunks.append(result)

    if not chunks:
        return {}

    # ------------------------------------------------------------------ #
    #  3. Concatenate all per-file arrays                                 #
    # ------------------------------------------------------------------ #
    all_src = np.concatenate([c[0] for c in chunks])
    all_dst = np.concatenate([c[1] for c in chunks])
    all_et  = np.concatenate([c[2] for c in chunks])
    all_t   = np.concatenate([c[3] for c in chunks])
    del chunks

    # ------------------------------------------------------------------ #
    #  4. Vectorised grouping — replaces the O(N) Python per-row loop     #
    #                                                                      #
    #  np.lexsort last key = primary sort, so (all_src, all_dst, all_et)  #
    #  sorts primarily by src, then dst, then edge_type.                  #
    # ------------------------------------------------------------------ #
    logger.info(f"Grouping {len(all_src):,} events into edges (vectorised)...")
    sort_idx = np.lexsort((all_et, all_dst, all_src))
    s_src = all_src[sort_idx]
    s_dst = all_dst[sort_idx]
    s_et  = all_et[sort_idx]
    s_t   = all_t[sort_idx]
    del sort_idx, all_src, all_dst, all_et, all_t

    # Find where the (src, dst, et) triple changes between consecutive rows.
    changed = (
        (s_src[1:] != s_src[:-1]) |
        (s_dst[1:] != s_dst[:-1]) |
        (s_et[1:]  != s_et[:-1])
    )
    boundaries = np.where(changed)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends   = np.concatenate([boundaries, [len(s_t)]])

    # Build output dict — one entry per unique edge.
    edge_timestamps = {
        (int(s_src[i]), int(s_dst[i]), int(s_et[i])): s_t[i:j].tolist()
        for i, j in zip(starts, ends)
    }

    logger.info(f"Extracted {len(edge_timestamps):,} unique edges "
                f"from {len(s_t):,} events")
    return edge_timestamps


def compute_rkhs_vectors(
    edge_timestamps: Dict[Tuple[int, int, int], List[float]],
    kde_computer: KDEVectorComputer,
    batch_size: int = 1000,
    n_workers: int = 1,
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int, int], int]]:
    """
    Compute RKHS vectors for all edges with sufficient timestamps.

    Args:
        edge_timestamps: Dictionary of (src, dst, edge_type) -> timestamps
        kde_computer: KDEVectorComputer instance
        batch_size: Process edges in batches for progress tracking (sequential mode)
        n_workers: Worker count.  >1 with CUDA enables the all-GPU CAVI path
                   (no CPU workers needed).  >1 without CUDA uses a
                   multiprocessing.Pool of CPU-bound Pyro SVI workers.

    Returns:
        Tuple of:
            - Dictionary mapping (src, dst, edge_type) -> RKHS vector (torch.Tensor)
            - Dictionary mapping (src, dst, edge_type) -> occurrence count
    """
    edge_vectors = {}
    edge_occurrence_counts = {}  # Track occurrence counts for all edges
    frequent_edges = []
    rare_edges = []
    
    # Filter edges by occurrence count and track all counts
    logger.info(f"Filtering edges with >= {kde_computer.min_occurrences} timestamps...")
    for edge, timestamps in edge_timestamps.items():
        count = len(timestamps)
        edge_occurrence_counts[edge] = count  # Store count for all edges
        if count >= kde_computer.min_occurrences:
            frequent_edges.append((edge, timestamps))
        else:
            rare_edges.append(edge)
    
    logger.info(f"Frequent edges (>= {kde_computer.min_occurrences} timestamps): {len(frequent_edges)}")
    logger.info(f"Rare edges (< {kde_computer.min_occurrences} timestamps): {len(rare_edges)}")
    
    # Compute RKHS vectors for frequent edges
    logger.info(f"Computing RKHS vectors using raw timestamps DPGMM "
                f"(workers={n_workers})...")

    if n_workers > 1:
        # ----------------------------------------------------------------
        # Parallel path.  GPU → batched CAVI; no GPU → CPU pool workers.
        # ----------------------------------------------------------------
        cuda_avail = torch.cuda.is_available()
        gpu_dev    = "cuda" if cuda_avail else "cpu"

        # ---- Phase A: prepare z-scored raw timestamps (CPU, vectorised) ----
        logger.info("Phase A: z-scoring raw timestamps for all frequent edges...")
        edge_keys, z_data_list = [], []
        skipped = 0
        for edge, timestamps in frequent_edges:
            ts = np.sort(np.array(timestamps, dtype=np.float64))
            data = ts.astype(np.float32)
            if len(data) < 2:
                skipped += 1
                continue
            mu, sig = float(np.mean(data)), float(np.std(data))
            if sig < 1e-12:
                sig = 1.0
            edge_keys.append(edge)
            z_data_list.append((data - mu) / sig)
        logger.info(f"  {len(edge_keys):,} edges ready, {skipped} skipped")

        if cuda_avail:
            # ---- GPU-native batched CAVI ----
            logger.info(
                f"GPU CAVI: fitting DP-GMM on {len(edge_keys):,} edges "
                f"(K={_PyroDPGMM.DEFAULT_K}, γ={_PyroDPGMM.DEFAULT_GAMMA}, "
                f"device={gpu_dev})..."
            )
            density_map = _gpu_batched_dpgmm_cavi(
                z_data_list,
                edge_keys,
                rkhs_dim=kde_computer.rkhs_dim,
                K=_PyroDPGMM.DEFAULT_K,
                gamma=_PyroDPGMM.DEFAULT_GAMMA,
                device=gpu_dev,
            )
            for edge, vec in density_map.items():
                edge_vectors[edge] = torch.tensor(vec, dtype=torch.float32)

        else:
            # No GPU: fall back to per-edge CPU-only Pyro SVI workers
            logger.info("No CUDA detected — falling back to CPU-only parallel workers.")
            worker_args = [
                (edge, timestamps, kde_computer.rkhs_dim)
                for edge, timestamps in frequent_edges
            ]
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                cs = max(1, len(worker_args) // (n_workers * 4))
                for edge, vector in tqdm(
                    pool.imap_unordered(_dpgmm_raw_ts_worker, worker_args, chunksize=cs),
                    total=len(frequent_edges),
                    desc="DPGMM (CPU parallel)",
                ):
                    if vector is not None:
                        edge_vectors[edge] = torch.tensor(vector, dtype=torch.float32)
    else:
        # ----------------------------------------------------------------
        # Sequential path (n_workers=1 fallback).
        # ----------------------------------------------------------------
        for i in tqdm(range(0, len(frequent_edges), batch_size), desc="Processing batches"):
            batch = frequent_edges[i:i + batch_size]

            for edge, timestamps in batch:
                timestamps_array = np.array(timestamps, dtype=np.float64)
                rkhs_vector = kde_computer.timestamps_to_rkhs_vector_dpgmm(timestamps_array)
                edge_vectors[edge] = torch.tensor(rkhs_vector, dtype=torch.float32)

            if (i // batch_size) % 10 == 0:
                import gc
                gc.collect()

    logger.info(f"Computed {len(edge_vectors)} RKHS vectors")
    logger.info(f"Tracked occurrence counts for {len(edge_occurrence_counts)} total edges")
    return edge_vectors, edge_occurrence_counts


def save_results(
    edge_vectors: Dict[Tuple[int, int, int], torch.Tensor],
    edge_occurrence_counts: Dict[Tuple[int, int, int], int],
    dataset_name: str,
    kde_params: Dict,
    output_dir: str = "kde_vectors"
):
    """
    Save RKHS vectors and statistics to disk.
    
    Args:
        edge_vectors: Dictionary of (src, dst, edge_type) -> RKHS vector
        edge_occurrence_counts: Dictionary of (src, dst, edge_type) -> occurrence count
        dataset_name: Name of the dataset
        kde_params: KDE parameters
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert edge_occurrence_counts keys to strings for JSON serialization
    # Format: "src,dst,edge_type" -> count
    edge_counts_serializable = {
        f"{s},{d},{et}": count 
        for (s, d, et), count in edge_occurrence_counts.items()
    }
    
    # Prepare metadata
    metadata = {
        'dataset': dataset_name,
        'method': 'dpgmm_raw_timestamps',
        'min_occurrences': kde_params.get('min_occurrences', 10),
        'rkhs_dim': kde_params.get('rkhs_dim', 20),
        'num_edges': len(edge_vectors),
        'num_total_edges_tracked': len(edge_occurrence_counts),
        'edge_key_format': '(src, dst, edge_type)',
        'timestamp': datetime.now().isoformat(),
        'edge_occurrence_counts': edge_counts_serializable,
    }
    
    # Save vectors
    output_file = os.path.join(output_dir, f"{dataset_name}_kde_vectors.pt")
    logger.info(f"Saving RKHS vectors to {output_file}...")
    torch.save({
        'edge_vectors': edge_vectors,
        'metadata': metadata
    }, output_file)
    
    # Save statistics
    stats_file = os.path.join(output_dir, f"{dataset_name}_kde_stats.json")
    logger.info(f"Saving statistics to {stats_file}...")
    
    # Compute statistics
    if len(edge_vectors) > 0:
        vector_norms = [v.norm().item() for v in edge_vectors.values()]
        stats = {
            **metadata,
            'statistics': {
                'num_edges_with_vectors': len(edge_vectors),
                'vector_norm_mean': float(np.mean(vector_norms)),
                'vector_norm_std': float(np.std(vector_norms)),
                'vector_norm_min': float(np.min(vector_norms)),
                'vector_norm_max': float(np.max(vector_norms)),
                'total_size_mb': os.path.getsize(output_file) / (1024 * 1024)
            }
        }
    else:
        stats = {
            **metadata,
            'statistics': {
                'num_edges_with_vectors': 0,
                'vector_norm_mean': 0.0,
                'vector_norm_std': 0.0,
                'vector_norm_min': 0.0,
                'vector_norm_max': 0.0,
                'total_size_mb': os.path.getsize(output_file) / (1024 * 1024)
            }
        }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Results saved successfully!")
    logger.info(f"  - Vectors: {output_file} ({stats['statistics']['total_size_mb']:.2f} MB)")
    logger.info(f"  - Stats: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description="Compute DPGMM RKHS vectors for temporal graphs")
    parser.add_argument('model', type=str, help='Model name (e.g., kairos_kde_ts)')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CLEARSCOPE_E3)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: from config kde_vectors_dir)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for sequential processing')
    parser.add_argument('--feat_hash', type=str, default=None,
                        help='Hash (or prefix) of the feat_inference run directory to use. '
                             'E.g. "a1b2c3d4" selects artifacts/feat_inference/{DATASET}/feat_inference/a1b2c3d4/. '
                             'If omitted, the latest directory (by modification time) is auto-selected.')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of parallel workers.  >1 with GPU uses batched '
                             'CAVI; >1 without GPU spawns CPU Pyro SVI workers.  '
                             f'Max useful value ≈ CPU cores '
                             f'(this machine has {multiprocessing.cpu_count()}).')

    args = parser.parse_args()
    
    # Load configuration
    cfg_dict = load_config(args.model)
    cfg = SimpleConfig(cfg_dict, args.dataset)
    kde_params = cfg_dict.get('kde_params', {})
    
    # Determine output directory (CLI arg overrides config)
    output_dir = args.output_dir or kde_params.get('kde_vectors_dir', 'kde_vectors')
    
    logger.info("=" * 80)
    logger.info("DPGMM RKHS Vector Computation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Workers: {args.n_workers}")
    logger.info(f"Feat hash: {args.feat_hash or '(auto-latest)'}")
    logger.info("=" * 80)
    
    # Initialize KDE computer
    kde_computer = KDEVectorComputer(
        rkhs_dim=kde_params.get('rkhs_dim', 20),
        min_occurrences=kde_params.get('min_occurrences', 10),
    )
    
    # Extract edge timestamps
    edge_timestamps = extract_edge_timestamps(
        cfg, args.dataset, feat_hash=args.feat_hash, n_workers=args.n_workers)
    
    # Compute RKHS vectors (returns both vectors and occurrence counts)
    edge_vectors, edge_occurrence_counts = compute_rkhs_vectors(
        edge_timestamps, kde_computer, args.batch_size, n_workers=args.n_workers)
    
    # Save results (including edge_occurrence_counts in metadata)
    save_results(edge_vectors, edge_occurrence_counts, args.dataset, kde_params, output_dir)
    
    logger.info("=" * 80)
    logger.info("Computation complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
