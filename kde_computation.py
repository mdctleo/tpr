#!/usr/bin/env python3
"""
Offline KDE Computation Script for KAIROS-KDE / Orthrus-KDE

Computes RKHS feature vectors for temporal graph edges using a GPU-accelerated
Variational Bayesian Gaussian Mixture Model (BayesianGaussianMixtureGPU).

Supports two modes controlled by the ``--use_timestamp_diffs`` flag or the
``use_timestamp_diffs`` YAML config key:

1. **kde_ts** (raw timestamps):
   Each edge's raw timestamp sequence is scaled with MaxAbsScaler, then fitted
   by the batched GPU DPGMM.  For CIC-IDS-2017 the mean of each day's
   timestamps is artificially anchored to 12:30 PM so that all five weekday
   files share a common reference point.

2. **kde_diff** (timestamp differences / inter-arrival times):
   Sorted timestamps -> absolute consecutive differences, scaled with
   MaxAbsScaler, then fitted by the batched GPU DPGMM.

The fitted GMM density is evaluated on a uniform grid of ``rkhs_dim`` points
spanning [min, max] of each edge's (scaled) data.  The resulting density
vector is the RKHS feature.

Usage
-----
    # Raw timestamps mode
    python kde_computation.py kairos_cicids_kde_ts CICIDS_MONDAY

    # Timestamp differences mode
    python kde_computation.py kairos_cicids_kde_diff CICIDS_MONDAY --output_dir kde_vectors_diff

    # Specify feat_inference hash
    python kde_computation.py kairos_cicids_kde_ts CICIDS_MONDAY --feat_hash abc123

Output
------
    ``{output_dir}/{DATASET}_kde_vectors.pt`` containing::

        {'edge_vectors': {(src, dst, edge_type): Tensor, ...},
         'metadata':     {<params + stats>}}
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import multiprocessing
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import gaussian_kde
from tqdm import tqdm

# sklearn for building output GMMs
try:
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import MaxAbsScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pidsmaker.utils.data_utils import load_data_set, collate_temporal_data
from pidsmaker.config.config import DATASET_DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
#  BayesianGaussianMixtureGPU
#  GPU-accelerated variational Bayesian GMM for batched 1-D data
# =====================================================================

class BayesianGaussianMixtureGPU:
    r"""
    GPU-accelerated variational Bayesian Gaussian mixture for **batched 1-D
    data**.

    Implements Coordinate Ascent Variational Inference (CAVI) with
    closed-form Normal-Inverse-Gamma conjugate updates plus a
    stick-breaking Dirichlet Process weight prior.  All heavy arithmetic
    is expressed as batched PyTorch tensor ops that run on a single GPU
    without any per-edge Python loop.

    Conjugate model
    ---------------
        v_k  ~ Beta(1, gamma)               stick-breaking (K-truncated)
        mu_k ~ N(mu_0, 1/lambda_0)
        sigma^2_k ~ InvGamma(alpha_0, beta_0)
        z_n  ~ Categorical(pi(v))
        x_n | z_n=k ~ N(mu_k, sigma^2_k)

    Variational family (fully factorised)
    --------------------------------------
        q(v_k)        = Beta(a_tilde_k, b_tilde_k)
        q(mu_k)       = N(m_k, 1/lambda_tilde_k)
        q(sigma^2_k)  = InvGamma(a_tilde_k, b_tilde_k)
        q(z_n)        = Categorical(r_n)

    Parameters
    ----------
    n_components : int
        Truncation level K for stick-breaking.  Components with negligible
        weight are pruned after fitting via *truncate_threshold*.
    gamma : float
        DP concentration parameter.  Larger -> more components retained.
    max_iter : int
        Maximum CAVI iterations per batch.
    tol : float
        Convergence tolerance on per-point normalised ELBO change.
    patience : int
        Early-stop after this many non-improving iterations.
    n_init : int
        Number of independent restarts; best (by ELBO) is kept.
    init_method : str
        ``"kmeans"`` (default) or ``"linear"`` (evenly spaced in [-2, 2]).
    max_n_per_edge : int
        Subsample edges longer than this to bound GPU VRAM.
    device : str
        PyTorch device string (``"cuda"`` or ``"cpu"``).
    """

    LOG_SQRT2PI = 0.9189385332046727  # log(sqrt(2*pi))

    def __init__(
        self,
        n_components: int = 100,
        gamma: float = 5.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        patience: int = 20,
        n_init: int = 1,
        init_method: str = "kmeans",
        max_n_per_edge: int = 50_000,
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
        self.max_n_per_edge = max_n_per_edge
        self.device = device

        # Fitted state (populated by fit_batch)
        self.m_k_: Optional[torch.Tensor] = None       # (B, K) means
        self.lam_k_: Optional[torch.Tensor] = None     # (B, K)
        self.a_k_: Optional[torch.Tensor] = None       # (B, K)
        self.b_k_: Optional[torch.Tensor] = None       # (B, K)
        self.alpha_q_: Optional[torch.Tensor] = None    # (B, K) Beta param a
        self.beta_q_: Optional[torch.Tensor] = None     # (B, K) Beta param b
        self.weights_: Optional[torch.Tensor] = None    # (B, K)
        self.means_: Optional[torch.Tensor] = None      # (B, K)
        self.stds_: Optional[torch.Tensor] = None       # (B, K)
        self.converged_: Optional[torch.Tensor] = None  # (B,) bool

    # -- public interface ------------------------------------------------
    def fit_batch(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> "BayesianGaussianMixtureGPU":
        """
        Fit the model on a batch of 1-D sequences.

        Parameters
        ----------
        X : Tensor, shape (B, N_max)
            Padded data matrix.  Invalid (padded) positions must be masked
            via *mask* or set to NaN.
        mask : Tensor, shape (B, N_max), dtype bool, optional
            ``True`` for valid positions.  If *None*, NaN positions in *X*
            are treated as invalid.

        Returns
        -------
        self
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
        N_eff = mask_f.sum(dim=1)  # (B,)

        # Replace NaN/padding with 0 to avoid polluting arithmetic
        X = X.clone()
        X[~mask] = 0.0

        K = min(self.K, int(N_eff.max().item()))
        K = max(K, 1)

        best_elbo = torch.full((B,), float("-inf"), device=self.device)
        best_state = None

        for _init_run in range(self.n_init):
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

        # Unpack best state
        self.m_k_ = best_state["m_k"]
        self.lam_k_ = best_state["lam_k"]
        self.a_k_ = best_state["a_k"]
        self.b_k_ = best_state["b_k"]
        self.alpha_q_ = best_state["alpha_q"]
        self.beta_q_ = best_state["beta_q"]
        self.converged_ = best_state["converged"]

        # Derive user-friendly parameters
        self._compute_mixture_params(B, K)
        return self

    # -- private: single CAVI run ----------------------------------------
    def _fit_once(self, X, mask, mask_f, N_eff, B, N_max, K):
        """Run one full CAVI optimisation; return state dict + final ELBO."""
        dev = self.device
        gamma = self.gamma

        # -- Priors (unit-scale, suitable for MaxAbsScaled data in [-1, 1]) --
        mu0  = torch.zeros(B, 1, device=dev)
        lam0 = torch.ones(B, 1, device=dev)
        a0   = torch.full((B, 1), 3.0, device=dev)
        b0   = torch.full((B, 1), 0.5, device=dev)

        # Prior constant for Beta KL:  log B(1, gamma) = lgamma(1)+lgamma(gamma)-lgamma(1+gamma)
        # Since lgamma(1)=0 and Gamma(1+g)=g*Gamma(g), this simplifies to -log(gamma).
        log_beta_prior = -math.log(max(gamma, 1e-30))

        # -- Initialise variational parameters --
        m_k = self._init_means(X, mask, mask_f, N_eff, B, K)  # (B, K)
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

            X3 = X.unsqueeze(2)
            m3 = m_k.unsqueeze(1)
            p3 = e_prec.unsqueeze(1)
            lv3 = e_log_var.unsqueeze(1)
            lk3 = lam_k.unsqueeze(1)

            log_lik = (
                -0.5 * lv3
                - 0.5 * p3 * (X3 - m3).pow(2)
                - 0.5 * p3 / lk3
                - self.LOG_SQRT2PI
            )  # (B, N_max, K)

            log_r_unnorm = e_log_pi.unsqueeze(1) + log_lik      # (B, N, K)
            log_r_unnorm = log_r_unnorm.masked_fill(~mask.unsqueeze(2), -1e30)
            log_normalizer = torch.logsumexp(log_r_unnorm, dim=2) # (B, N)
            log_r = log_r_unnorm - log_normalizer.unsqueeze(2)
            R = log_r.exp() * mask_f.unsqueeze(2)

            # ---- M-step: closed-form updates ----
            N_k = R.sum(dim=1)  # (B, K)
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

            # ---- Full ELBO computation ----
            #
            # ELBO = E_q[log p(X,Z,v,mu,sigma^2)] - E_q[log q(Z,v,mu,sigma^2)]
            #
            # Decomposed into four terms:
            #
            # (a) Data likelihood + assignment (prior + entropy):
            #     E_q[log p(X|Z,mu,sigma^2)] + E_q[log p(Z|pi)] + H(q(Z))
            #   = sum_n logsumexp_k( E[log pi_k] + E[log p(x_n | k)] )
            #     (The log-normalizer of the responsibilities; see Bishop
            #      PRML Eq 10.71 or Blei et al. 2006 Appendix.)
            #
            # (b) KL[ q(v_k) || p(v_k) ] for stick-breaking Beta variables
            #     q(v_k) = Beta(alpha_q_k, beta_q_k),  p(v_k) = Beta(1, gamma)
            #
            # (c) KL[ q(mu_k) || p(mu_k) ] for Normal means
            #     q(mu_k) = N(m_k, 1/lam_k),  p(mu_k) = N(mu0, 1/lam0)
            #
            # (d) KL[ q(sigma^2_k) || p(sigma^2_k) ] for InvGamma variances
            #     q(sigma^2_k) = IG(a_k, b_k),  p(sigma^2_k) = IG(a0, b0)
            #
            # ELBO = (a) - (b) - (c) - (d)

            # (a) Data + assignment: sum_n logsumexp_k(E[log pi_k] + E[log p(x_n|k)])
            #     log_normalizer was captured during the E-step; masked positions
            #     are zeroed via mask_f so they contribute nothing.
            ell_assign = (log_normalizer * mask_f).sum(dim=1)          # (B,)

            # (b) KL[ Beta(alpha_q, beta_q) || Beta(1, gamma) ]  (summed over K)
            #     = log B(1,gamma) - log B(alpha_q,beta_q)
            #       + (alpha_q - 1) psi(alpha_q) + (beta_q - gamma) psi(beta_q)
            #       + (1 + gamma - alpha_q - beta_q) psi(alpha_q + beta_q)
            dig_aq  = torch.digamma(alpha_q)
            dig_bq  = torch.digamma(beta_q)
            dig_abq = torch.digamma(alpha_q + beta_q)
            kl_v = (
                log_beta_prior                                        # log B(1,γ)
                - torch.lgamma(alpha_q) - torch.lgamma(beta_q)
                + torch.lgamma(alpha_q + beta_q)                      # -log B(ã,b̃)
                + (alpha_q - 1.0) * dig_aq
                + (beta_q - gamma) * dig_bq
                + (1.0 + gamma - alpha_q - beta_q) * dig_abq
            ).sum(dim=1)                                               # (B,)

            # (c) KL[ N(m_k, 1/lam_k) || N(mu0, 1/lam0) ]  (summed over K)
            #     = 0.5 * [ log(lam_k/lam0) + lam0/lam_k
            #               + lam0*(m_k - mu0)^2 - 1 ]
            kl_mu = 0.5 * (
                torch.log(lam_k / lam0)
                + lam0 / lam_k
                + lam0 * (m_k - mu0).pow(2)
                - 1.0
            ).sum(dim=1)                                               # (B,)

            # (d) KL[ IG(a_k, b_k) || IG(a0, b0) ]  (summed over K)
            #     = a0*(log b_k - log b0) + lgamma(a0) - lgamma(a_k)
            #       + (a_k - a0)*psi(a_k) + b0*a_k/b_k - a_k
            kl_sigma = (
                a0 * (torch.log(b_k) - torch.log(b0))
                + torch.lgamma(a0) - torch.lgamma(a_k)
                + (a_k - a0) * torch.digamma(a_k)
                + b0 * a_k / b_k
                - a_k
            ).sum(dim=1)                                               # (B,)

            # Full ELBO, normalised per data point
            elbo = (ell_assign - kl_v - kl_mu - kl_sigma) / N_eff.clamp(min=1)

            # ---- Convergence check ----
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

    # -- initialisation helpers ------------------------------------------
    def _init_means(self, X, mask, mask_f, N_eff, B, K):
        """Initialise component means via k-means or linearly."""
        dev = self.device

        if self.init_method == "kmeans" and K >= 2:
            return self._kmeans_init(X, mask, mask_f, N_eff, B, K)
        else:
            # Linear: evenly spaced between per-edge min and max
            x_min = X.masked_fill(~mask, 1e30).min(dim=1).values   # (B,)
            x_max = X.masked_fill(~mask, -1e30).max(dim=1).values  # (B,)
            t_lin = torch.linspace(0.0, 1.0, K, device=dev)        # (K,)
            span = (x_max - x_min).clamp(min=1e-10)
            return x_min.unsqueeze(1) + t_lin.unsqueeze(0) * span.unsqueeze(1)

    def _kmeans_init(self, X, mask, mask_f, N_eff, B, K, n_iter: int = 20):
        """
        Batched mini-k-means on GPU to initialise component means.

        Uses k-means++ style seeding: first centre = per-edge mean,
        subsequent centres picked as the point with maximum min-distance
        from existing centres (deterministic k-means++).
        """
        dev = self.device

        # First centre: per-edge mean
        x_sum = (X * mask_f).sum(dim=1)   # (B,)
        x_mean = x_sum / N_eff.clamp(min=1)
        centres = x_mean.unsqueeze(1)  # (B, 1)

        for _c in range(1, K):
            # Squared distance from each point to nearest existing centre
            dists_sq = (X.unsqueeze(2) - centres.unsqueeze(1)).pow(2)
            min_dist_sq = dists_sq.min(dim=2).values  # (B, N)
            min_dist_sq = min_dist_sq * mask_f         # zero out padding
            # Pick the point with max min-distance
            new_idx = min_dist_sq.argmax(dim=1)        # (B,)
            new_centre = X[torch.arange(B, device=dev), new_idx].unsqueeze(1)
            centres = torch.cat([centres, new_centre], dim=1)

        # Refine with Lloyd iterations
        for _it in range(n_iter):
            dists_sq = (X.unsqueeze(2) - centres.unsqueeze(1)).pow(2)  # (B, N, K)
            dists_sq = dists_sq.masked_fill(~mask.unsqueeze(2), 1e30)
            assigns = dists_sq.argmin(dim=2)  # (B, N)

            one_hot = torch.zeros(B, X.shape[1], K, device=dev)
            one_hot.scatter_(2, assigns.unsqueeze(2), 1.0)
            one_hot = one_hot * mask_f.unsqueeze(2)

            counts = one_hot.sum(dim=1).clamp(min=1e-10)         # (B, K)
            sums = torch.bmm(X.unsqueeze(1), one_hot).squeeze(1)  # (B, K)
            new_centres = sums / counts

            # Keep old centre if cluster became empty
            empty = counts < 0.5
            centres = torch.where(empty, centres, new_centres)

        return centres  # (B, K)

    # -- derive mixture parameters from variational posteriors -----------
    def _compute_mixture_params(self, B: int, K: int):
        """Compute means, stds, weights from fitted variational parameters."""
        # Stick-breaking weights
        e_v = self.alpha_q_ / (self.alpha_q_ + self.beta_q_)
        remain = torch.cat([
            torch.ones(B, 1, device=self.device),
            torch.cumprod(
                (1.0 - e_v[:, :-1]).clamp(min=0, max=1.0 - 1e-10), dim=1
            ),
        ], dim=1)
        weights = (e_v * remain).clamp(min=0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-10)

        # Posterior mean and std
        means = self.m_k_
        stds = (self.b_k_ / (self.a_k_ - 1.0).clamp(min=1e-6)).sqrt().clamp(min=1e-6)

        self.weights_ = weights
        self.means_ = means
        self.stds_ = stds

    # -- density evaluation ----------------------------------------------
    def score_samples_grid(
        self,
        X: torch.Tensor,
        mask: Optional[torch.Tensor],
        grid_size: int,
    ) -> torch.Tensor:
        """
        Evaluate the fitted mixture density on a per-edge uniform grid.

        Parameters
        ----------
        X : (B, N_max) -- original data (used only to find min/max per edge)
        mask : (B, N_max) bool
        grid_size : int -- number of grid points (= rkhs_dim)

        Returns
        -------
        density : (B, grid_size) float32 tensor -- L1-normalised density values
        """
        B = X.shape[0]
        dev = self.device

        x_min = X.masked_fill(~mask, 1e30).min(dim=1).values
        x_max = X.masked_fill(~mask, -1e30).max(dim=1).values
        span = (x_max - x_min).clamp(min=1e-10)

        t_lin = torch.linspace(0.0, 1.0, grid_size, device=dev)
        grid = x_min.unsqueeze(1) + t_lin * span.unsqueeze(1)  # (B, D)

        g3 = grid.unsqueeze(2)          # (B, D, 1)
        m3 = self.means_.unsqueeze(1)   # (B, 1, K)
        s3 = self.stds_.unsqueeze(1)    # (B, 1, K)
        w3 = self.weights_.unsqueeze(1) # (B, 1, K)

        log_gauss = (
            -0.5 * ((g3 - m3) / s3).pow(2) - s3.log() - self.LOG_SQRT2PI
        )
        log_w = w3.log().masked_fill(w3 < 1e-30, -1e30)
        density = torch.logsumexp(log_w + log_gauss, dim=2).exp()  # (B, D)

        # L1-normalise
        density = density.clamp(min=0)
        density = density / density.sum(dim=1, keepdim=True).clamp(min=1e-10)
        return density

    # -- conversion to sklearn GMMs --------------------------------------
    def to_sklearn_gmms(
        self, truncate_threshold: float = 0.99
    ) -> List[Optional[GaussianMixture]]:
        """
        Convert each batch element's fitted mixture to a truncated
        sklearn ``GaussianMixture`` (for downstream compatibility).

        Parameters
        ----------
        truncate_threshold : float
            Cumulative weight threshold for pruning negligible components.

        Returns
        -------
        list of GaussianMixture or None (if a fit failed)
        """
        B = self.weights_.shape[0]
        weights_np = self.weights_.cpu().numpy()
        means_np = self.means_.cpu().numpy()
        stds_np = self.stds_.cpu().numpy()

        gmms: List[Optional[GaussianMixture]] = []
        for b in range(B):
            w = weights_np[b]
            m = means_np[b]
            s = stds_np[b]

            # Sort by descending weight and truncate
            idx = np.argsort(w)[::-1]
            w_sorted = w[idx]
            m_sorted = m[idx]
            s_sorted = s[idx]

            cum_w = np.cumsum(w_sorted)
            cut = int(np.searchsorted(cum_w, truncate_threshold)) + 1
            cut = min(cut, len(w_sorted))

            w_kept = w_sorted[:cut]
            w_kept = w_kept / w_kept.sum()
            m_kept = m_sorted[:cut]
            s_kept = s_sorted[:cut]

            if cut == 0 or np.any(~np.isfinite(m_kept)):
                gmms.append(None)
                continue

            gmm = GaussianMixture(n_components=cut, covariance_type="diag")
            gmm.weights_ = w_kept
            gmm.means_ = m_kept.reshape(cut, 1)
            covs = (s_kept ** 2).reshape(cut, 1)
            gmm.covariances_ = covs
            safe = np.maximum(covs, np.finfo(covs.dtype).eps)
            gmm.precisions_cholesky_ = 1.0 / np.sqrt(safe)
            gmm.n_features_in_ = 1
            gmm.n_components_ = cut
            gmms.append(gmm)

        return gmms


# =====================================================================
#  Pipeline helpers
# =====================================================================

def filter_merged_edges(
    edge_timestamps: Dict[Tuple[int, int, int], List[float]],
    min_occurrences: int,
    use_timestamp_diffs: bool,
) -> Tuple[
    List[Tuple[int, int, int]],
    List[np.ndarray],
    Dict[Tuple[int, int, int], int],
]:
    """
    Filter edges with enough observations and prepare data arrays.

    For ``kde_ts`` mode the raw timestamps are returned.
    For ``kde_diff`` mode the absolute inter-arrival times are returned.

    Returns
    -------
    edge_keys : list of edge tuples
    data_arrays : list of 1-D float64 numpy arrays (one per edge)
    edge_occurrence_counts : counts for *all* edges (not just frequent ones)
    """
    edge_keys: List[Tuple[int, int, int]] = []
    data_arrays: List[np.ndarray] = []
    edge_occurrence_counts: Dict[Tuple[int, int, int], int] = {}

    for edge, ts_list in edge_timestamps.items():
        count = len(ts_list)
        edge_occurrence_counts[edge] = count

        if count < min_occurrences:
            continue

        ts_arr = np.array(ts_list, dtype=np.float64)

        if use_timestamp_diffs:
            sorted_ts = np.sort(ts_arr)
            diffs = np.abs(np.diff(sorted_ts))
            if len(diffs) < 2:
                continue
            data_arrays.append(diffs)
        else:
            data_arrays.append(ts_arr)

        edge_keys.append(edge)

    return edge_keys, data_arrays, edge_occurrence_counts


def _anchor_cicids_noon(
    data_arrays: List[np.ndarray],
) -> List[np.ndarray]:
    """
    CIC-IDS-2017 timestamp anchor: shift each edge's timestamps so that
    the *mean* falls at 12:30 PM (45 000 seconds past midnight).

    The CIC-IDS dataset spans five weekdays (Monday-Friday) captured in
    separate CSVs.  Because the five files have different absolute
    timestamp ranges, we anchor each edge's mean to a common reference
    point (12 h 30 min = 45 000 s) so that the GMM sees consistent
    temporal patterns across days.

    This is applied **only in kde_ts mode** (raw timestamps) and **only
    for CIC-IDS datasets** (detected by the caller).
    """
    NOON_ANCHOR = 45_000.0  # 12 h 30 m in seconds

    anchored: List[np.ndarray] = []
    for arr in data_arrays:
        current_mean = float(arr.mean())
        shift = NOON_ANCHOR - current_mean
        anchored.append(arr + shift)
    return anchored


def scale_merged_edges(
    data_arrays: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Scale each edge's data independently with MaxAbsScaler.

    Each array is divided by its max absolute value so the result lies
    in [-1, 1].  The per-edge scale factors are returned so that the
    fitted GMM can be un-scaled later if needed.

    Returns
    -------
    scaled : list of 1-D float32 arrays
    scale_factors : list of per-edge max-abs values
    """
    scaled: List[np.ndarray] = []
    scale_factors: List[float] = []

    for arr in data_arrays:
        max_abs = float(np.abs(arr).max())
        if max_abs < 1e-15:
            max_abs = 1.0
        scale_factors.append(max_abs)
        scaled.append((arr / max_abs).astype(np.float32))

    return scaled, scale_factors


def preprocess_long_merged_edges(
    scaled_arrays: List[np.ndarray],
    max_n_per_edge: int = 50_000,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Subsample edges longer than *max_n_per_edge* to bound GPU VRAM.

    The subsample is random but reproducible if a *rng* is supplied.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    out: List[np.ndarray] = []
    for arr in scaled_arrays:
        if len(arr) > max_n_per_edge:
            idx = rng.choice(len(arr), max_n_per_edge, replace=False)
            arr = arr[np.sort(idx)]
        out.append(arr)
    return out


def merged_edges_to_matrix(
    arrays: List[np.ndarray],
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of 1-D arrays into a (B, N_max) matrix and a boolean mask.

    Returns
    -------
    X : (B, N_max) float32 tensor on *device*
    mask : (B, N_max) bool tensor on *device*
    """
    B = len(arrays)
    lengths = [len(a) for a in arrays]
    N_max = max(lengths)

    X = torch.zeros(B, N_max, dtype=torch.float32, device=device)
    mask = torch.zeros(B, N_max, dtype=torch.bool, device=device)

    for i, (arr, n) in enumerate(zip(arrays, lengths)):
        X[i, :n] = torch.from_numpy(arr).to(device)
        mask[i, :n] = True

    return X, mask


def gpu_batches_to_gmms(
    model: BayesianGaussianMixtureGPU,
    truncate_threshold: float = 0.99,
) -> List[Optional[GaussianMixture]]:
    """Wrapper around model.to_sklearn_gmms for readability."""
    return model.to_sklearn_gmms(truncate_threshold=truncate_threshold)


# =====================================================================
#  Batched GPU DPGMM driver
# =====================================================================

def fit_batched_gpu_dpgmm(
    edge_keys: List[Tuple[int, int, int]],
    data_arrays: List[np.ndarray],
    rkhs_dim: int = 20,
    n_components: int = 100,
    gamma: float = 5.0,
    max_iter: int = 300,
    tol: float = 1e-4,
    patience: int = 20,
    n_init: int = 1,
    init_method: str = "kmeans",
    truncate_threshold: float = 0.99,
    max_n_per_edge: int = 50_000,
    chunk_size: int = 256,
    is_cicids: bool = False,
    use_timestamp_diffs: bool = False,
    device: str = "cuda",
) -> Dict[Tuple[int, int, int], np.ndarray]:
    """
    Full batched GPU DPGMM pipeline:

    1. (kde_ts + CIC-IDS only) anchor timestamps to 12:30 PM
    2. MaxAbsScaler per edge
    3. Subsample long edges
    4. Chunk edges into GPU batches
    5. Fit BayesianGaussianMixtureGPU per chunk
    6. Evaluate density on rkhs_dim-point grid

    Returns
    -------
    dict mapping edge_key -> float32 numpy array of shape (rkhs_dim,)
    """
    logger.info(f"GPU DPGMM pipeline: {len(edge_keys):,} edges, "
                f"K={n_components}, gamma={gamma}, device={device}")

    # 1. CIC-IDS noon anchor (kde_ts mode only)
    if is_cicids and not use_timestamp_diffs:
        logger.info("Applying CIC-IDS 12:30 PM noon anchor to raw timestamps...")
        data_arrays = _anchor_cicids_noon(data_arrays)

    # 2. MaxAbsScaler
    logger.info("Scaling edges with MaxAbsScaler...")
    scaled_arrays, scale_factors = scale_merged_edges(data_arrays)

    # 3. Subsample long edges
    scaled_arrays = preprocess_long_merged_edges(
        scaled_arrays, max_n_per_edge=max_n_per_edge
    )

    # 4-6. Process in chunks
    results: Dict[Tuple[int, int, int], np.ndarray] = {}

    # Sort by length so chunks have similar N_max -> less padding waste
    order = sorted(range(len(scaled_arrays)), key=lambda i: len(scaled_arrays[i]))

    for c0 in tqdm(range(0, len(order), chunk_size), desc="DPGMM GPU batches"):
        chunk_idx = order[c0: c0 + chunk_size]
        chunk_arrays = [scaled_arrays[i] for i in chunk_idx]

        X_batch, mask_batch = merged_edges_to_matrix(chunk_arrays, device=device)

        model = BayesianGaussianMixtureGPU(
            n_components=n_components,
            gamma=gamma,
            max_iter=max_iter,
            tol=tol,
            patience=patience,
            n_init=n_init,
            init_method=init_method,
            max_n_per_edge=max_n_per_edge,
            device=device,
        )
        model.fit_batch(X_batch, mask_batch)

        # Evaluate density on grid
        density = model.score_samples_grid(X_batch, mask_batch, grid_size=rkhs_dim)
        density_np = density.cpu().numpy()

        for local_i, orig_idx in enumerate(chunk_idx):
            results[edge_keys[orig_idx]] = density_np[local_i].astype(np.float32)

        # Free GPU memory between chunks
        del X_batch, mask_batch, model, density
        if device == "cuda":
            torch.cuda.empty_cache()

    return results


# =====================================================================
#  Legacy: KDE raw-timestamp mode (scipy gaussian_kde + hand-crafted RKHS)
#  Used as CPU-only fallback when no CUDA is available.
# =====================================================================

class KDEVectorComputer:
    """
    Parameter holder and CPU fallback for RKHS vector computation.

    The main GPU path uses ``fit_batched_gpu_dpgmm``; this class provides
    the scipy-based ``kde_to_rkhs_vector`` fallback for raw-timestamp
    mode on CPU and holds shared parameters (rkhs_dim, min_occurrences,
    etc.).
    """

    def __init__(
        self,
        rkhs_dim: int = 20,
        min_occurrences: int = 10,
        bandwidth: str = "scott",
        n_quadrature_points: int = 10,
        use_timestamp_diffs: bool = False,
    ):
        self.rkhs_dim = rkhs_dim
        self.min_occurrences = min_occurrences
        self.bandwidth = bandwidth
        self.n_quadrature_points = n_quadrature_points
        self.use_timestamp_diffs = use_timestamp_diffs

        # Precompute Gauss-Hermite quadrature points and weights (scipy fallback)
        self.quad_points, self.quad_weights = self._compute_quadrature()

        logger.info("Initialized KDEVectorComputer:")
        logger.info(f"  rkhs_dim={rkhs_dim}  min_occ={min_occurrences}")
        logger.info(f"  bandwidth={bandwidth}  quad_pts={n_quadrature_points}")
        logger.info(f"  use_timestamp_diffs={use_timestamp_diffs}")

    def _compute_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        from numpy.polynomial.hermite import hermgauss
        pts, wts = hermgauss(self.n_quadrature_points)
        return pts, wts / np.sqrt(np.pi)

    # -- scipy-based fallback (kde_ts mode, CPU only) --------------------
    def kde_to_rkhs_vector(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Convert timestamps to RKHS vector via scipy gaussian_kde.

        Used only as a CPU fallback when CUDA is not available.
        """
        try:
            kde = gaussian_kde(timestamps, bw_method=self.bandwidth)
            mean = np.mean(timestamps)
            std = max(np.std(timestamps), 1e-6)
            scaled_pts = mean + std * np.sqrt(2) * self.quad_points
            kde_vals = kde(scaled_pts)
            weighted = kde_vals * self.quad_weights

            from scipy.stats import skew, kurtosis as kurt_fn
            moments = np.array([
                mean, std,
                float(skew(timestamps)),
                float(kurt_fn(timestamps)),
            ])
            freqs = np.array([1.0, 2.0, 3.0])
            fourier = []
            for f in freqs:
                fourier.append(np.mean(np.cos(2 * np.pi * f * timestamps / (mean + 1e-6))))
                fourier.append(np.mean(np.sin(2 * np.pi * f * timestamps / (mean + 1e-6))))
            quantiles = np.percentile(timestamps, [25, 50, 75])

            vec = np.concatenate([
                weighted[:5], moments,
                np.array(fourier), quantiles,
                np.array([len(timestamps), timestamps.max() - timestamps.min()]),
            ])
            if len(vec) < self.rkhs_dim:
                vec = np.pad(vec, (0, self.rkhs_dim - len(vec)))
            else:
                vec = vec[: self.rkhs_dim]
            return vec.astype(np.float32)
        except Exception as e:
            logger.warning(f"KDE failed: {e}, zero vector")
            return np.zeros(self.rkhs_dim, dtype=np.float32)


# =====================================================================
#  File I/O helpers
# =====================================================================

def _load_file_worker(args: tuple):
    """
    Thread-pool worker: load one ``.TemporalData.simple`` file.

    Returns (src, dst, edge_types, t) as numpy arrays, or None on error.
    """
    file_path, node_type_dim, edge_type_dim = args
    try:
        data = torch.load(file_path, map_location="cpu")
        src = data.src.cpu().numpy().astype(np.int64)
        dst = data.dst.cpu().numpy().astype(np.int64)
        t = data.t.cpu().numpy()
        if hasattr(data, "msg") and data.msg is not None:
            edge_types = extract_edge_type_from_msg(
                data.msg, node_type_dim, edge_type_dim
            ).astype(np.int64)
        elif hasattr(data, "edge_type") and data.edge_type is not None:
            edge_types = data.edge_type.max(dim=1).indices.cpu().numpy().astype(np.int64)
        else:
            edge_types = np.zeros(len(src), dtype=np.int64)
        return src, dst, edge_types, t
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def extract_edge_type_from_msg(
    msg: torch.Tensor, node_type_dim: int = 8, edge_type_dim: int = 16
) -> np.ndarray:
    """Extract edge type indices from one-hot encoded msg tensor."""
    msg_dim = msg.shape[1]
    emb_dim = (msg_dim - 2 * node_type_dim - edge_type_dim) // 2
    start = node_type_dim + emb_dim
    end = start + edge_type_dim
    return msg[:, start:end].argmax(dim=1).cpu().numpy()


# =====================================================================
#  Config loading
# =====================================================================

def load_config(model_name: str) -> Dict:
    """Load a YAML configuration file, resolving ``_include_yml``."""
    config_file = f"config/{model_name}.yml"
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)
    if "_include_yml" in cfg:
        base_path = f"config/{cfg['_include_yml']}.yml"
        with open(base_path, "r") as f:
            base = yaml.safe_load(f)
        base.update(cfg)
        cfg = base
    return cfg


class SimpleConfig:
    """Thin config wrapper mimicking the pidsmaker config structure."""

    def __init__(self, cfg_dict: Dict, dataset_name: str):
        _ds_reg = DATASET_DEFAULT_CONFIG.get(dataset_name, {})
        self.dataset = type("obj", (object,), {
            "name": dataset_name,
            "train_files": cfg_dict.get("dataset", {}).get("train_files", []),
            "val_files": cfg_dict.get("dataset", {}).get("val_files", []),
            "test_files": cfg_dict.get("dataset", {}).get("test_files", []),
            "num_node_types": _ds_reg.get("num_node_types", 8),
            "num_edge_types": _ds_reg.get("num_edge_types", 16),
        })()
        self.feat_inference = type("obj", (object,), {
            "_task_path": cfg_dict.get("feat_inference", {}).get(
                "_task_path", f"artifacts/{dataset_name}/feat_inference"
            ),
        })()
        self.featurization = type("obj", (object,), cfg_dict.get("featurization", {}))()
        self.construction = type("obj", (object,), cfg_dict.get("construction", {}))()
        self.batching = type("obj", (object,), cfg_dict.get("batching", {}))()
        self._dict = cfg_dict

    def get(self, key, default=None):
        return self._dict.get(key, default)


# =====================================================================
#  Dataset folder resolution
# =====================================================================

def get_latest_dataset_folder(dataset_name: str, feat_hash: str = None, artifacts_dir: str = "artifacts") -> str:
    """
    Resolve the feat_inference directory for a dataset.

    If *feat_hash* is given, look for an exact match (or prefix).
    Otherwise pick the most recently modified directory.

    Args:
        dataset_name: Dataset name (e.g. CIC_IDS_2017)
        feat_hash: Optional specific feat_inference hash to use
        artifacts_dir: Base artifacts directory (e.g. 'artifacts_cicids').
                       Relative paths are resolved against the project root.
    """
    import glob

    # Resolve relative paths against the project root (directory of this file)
    _project_root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(artifacts_dir):
        artifacts_dir = os.path.join(_project_root, artifacts_dir)

    base_path = os.path.join(artifacts_dir, "feat_inference", dataset_name, "feat_inference")

    if feat_hash is not None:
        target = os.path.join(base_path, feat_hash)
        if not os.path.isdir(target):
            matches = glob.glob(os.path.join(base_path, f"{feat_hash}*"))
            matches = [m for m in matches if os.path.isdir(m)]
            if not matches:
                all_dirs = [
                    os.path.basename(d)
                    for d in glob.glob(os.path.join(base_path, "*"))
                    if os.path.isdir(d)
                ]
                raise FileNotFoundError(
                    f"No dir matching '{feat_hash}' under {base_path}. "
                    f"Available: {all_dirs}"
                )
            target = matches[0]
        logger.info(f"Using feat_inference folder: {target}")
        return target

    subdirs = glob.glob(os.path.join(base_path, "*"))
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if not subdirs:
        raise FileNotFoundError(f"No dataset folders at {base_path}")

    latest = max(subdirs, key=lambda d: os.path.getmtime(d))
    logger.info(f"Latest dataset folder: {latest}")
    return latest


# =====================================================================
#  Edge timestamp extraction (parallel + vectorised)
# =====================================================================

def extract_edge_timestamps(
    cfg: SimpleConfig,
    dataset_name: str,
    feat_hash: str = None,
    n_workers: int = 1,
    artifacts_dir: str = "artifacts_cicids",
) -> Dict[Tuple[int, int, int], List[float]]:
    """
    Load ``.TemporalData.simple`` files and group events by edge.

    Files are loaded in parallel (ThreadPoolExecutor -- IO-bound).
    Grouping is vectorised via ``np.lexsort`` + boundary detection.
    """
    import glob

    logger.info(f"Loading dataset: {dataset_name}")
    base_path = get_latest_dataset_folder(dataset_name, feat_hash=feat_hash, artifacts_dir=artifacts_dir)

    node_type_dim = getattr(getattr(cfg, "dataset", None), "num_node_types", 8)
    edge_type_dim = getattr(getattr(cfg, "dataset", None), "num_edge_types", 16)
    logger.info(f"node_type_dim={node_type_dim}, edge_type_dim={edge_type_dim}")

    all_files: List[str] = []
    for split in ["train", "val"]:
        pattern = f"{base_path}/edge_embeds/{split}/*.TemporalData.simple"
        split_files = glob.glob(pattern)
        if not split_files:
            logger.warning(f"No files for {split} at {pattern}")
        else:
            logger.info(f"  {split}: {len(split_files)} files")
            all_files.extend(split_files)

    if not all_files:
        logger.warning("No files found for any split")
        return {}

    io_workers = min(n_workers, len(all_files))
    logger.info(f"Loading {len(all_files)} files ({io_workers} IO thread(s))...")

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

    all_src = np.concatenate([c[0] for c in chunks])
    all_dst = np.concatenate([c[1] for c in chunks])
    all_et = np.concatenate([c[2] for c in chunks])
    all_t = np.concatenate([c[3] for c in chunks])
    del chunks

    logger.info(f"Grouping {len(all_src):,} events into edges...")
    sort_idx = np.lexsort((all_et, all_dst, all_src))
    s_src = all_src[sort_idx]
    s_dst = all_dst[sort_idx]
    s_et = all_et[sort_idx]
    s_t = all_t[sort_idx]
    del sort_idx, all_src, all_dst, all_et, all_t

    changed = (
        (s_src[1:] != s_src[:-1])
        | (s_dst[1:] != s_dst[:-1])
        | (s_et[1:] != s_et[:-1])
    )
    boundaries = np.where(changed)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(s_t)]])

    edge_timestamps = {
        (int(s_src[i]), int(s_dst[i]), int(s_et[i])): s_t[i:j].tolist()
        for i, j in zip(starts, ends)
    }
    logger.info(
        f"Extracted {len(edge_timestamps):,} unique edges "
        f"from {len(s_t):,} events"
    )
    return edge_timestamps


# =====================================================================
#  Main computation entry point
# =====================================================================

def compute_rkhs_vectors(
    edge_timestamps: Dict[Tuple[int, int, int], List[float]],
    kde_computer: KDEVectorComputer,
    kde_params: Dict,
    batch_size: int = 1000,
    n_workers: int = 1,
    is_cicids: bool = False,
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int, int], int]]:
    """
    Compute RKHS vectors for all edges with >= min_occurrences timestamps.

    Uses the GPU-accelerated ``BayesianGaussianMixtureGPU`` when CUDA is
    available; falls back to the scipy-based ``KDEVectorComputer`` on CPU.

    Parameters
    ----------
    edge_timestamps : dict  --  (src, dst, et) -> list of timestamps
    kde_computer : KDEVectorComputer
    kde_params : dict  --  from YAML config (carries BGMM hyper-params)
    batch_size : int  --  chunk size for GPU batches (or sequential fallback)
    n_workers : int  --  IO threads (file loading)
    is_cicids : bool  --  enable CIC-IDS noon anchor

    Returns
    -------
    edge_vectors : dict  --  (src, dst, et) -> Tensor of shape (rkhs_dim,)
    edge_occurrence_counts : dict  --  (src, dst, et) -> int
    """
    use_diffs = kde_computer.use_timestamp_diffs
    rkhs_dim = kde_computer.rkhs_dim
    min_occ = kde_computer.min_occurrences

    # -- 1. filter + prepare data arrays --
    logger.info(f"Filtering edges with >= {min_occ} timestamps ...")
    edge_keys, data_arrays, edge_occurrence_counts = filter_merged_edges(
        edge_timestamps, min_occ, use_diffs,
    )
    n_frequent = len(edge_keys)
    n_total = len(edge_occurrence_counts)
    logger.info(f"Frequent edges: {n_frequent:,} / {n_total:,} total")

    if n_frequent == 0:
        return {}, edge_occurrence_counts

    # -- 2. GPU path --
    cuda_avail = torch.cuda.is_available()
    device = "cuda" if cuda_avail else "cpu"

    if cuda_avail:
        mode_str = "timestamp DIFFS" if use_diffs else "raw timestamps"
        logger.info(f"GPU path ({mode_str}): {n_frequent:,} edges on {device}")

        density_map = fit_batched_gpu_dpgmm(
            edge_keys=edge_keys,
            data_arrays=data_arrays,
            rkhs_dim=rkhs_dim,
            n_components=kde_params.get("n_components", 100),
            gamma=kde_params.get("gamma", 5.0),
            max_iter=kde_params.get("max_iter", 300),
            tol=kde_params.get("tol", 1e-4),
            patience=kde_params.get("patience", 20),
            n_init=kde_params.get("n_init", 1),
            init_method=kde_params.get("init_method", "kmeans"),
            truncate_threshold=kde_params.get("truncate_threshold", 0.99),
            max_n_per_edge=kde_params.get("max_n_per_edge", 50_000),
            chunk_size=kde_params.get("chunk_size", 256),
            is_cicids=is_cicids,
            use_timestamp_diffs=use_diffs,
            device=device,
        )

        edge_vectors = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in density_map.items()
        }
    else:
        # -- CPU fallback (scipy KDE for raw ts; scipy KDE on diffs) --
        logger.info(f"No CUDA -- CPU fallback ({n_frequent:,} edges)")
        edge_vectors = {}

        # For CPU, if raw ts mode, apply CIC-IDS anchor then MaxAbsScale then grid
        if is_cicids and not use_diffs:
            data_arrays = _anchor_cicids_noon(data_arrays)

        for idx in tqdm(range(n_frequent), desc="CPU RKHS"):
            edge = edge_keys[idx]
            arr = data_arrays[idx]

            if use_diffs:
                # Simple fallback: MaxAbsScale + uniform density grid
                max_abs = float(np.abs(arr).max())
                if max_abs < 1e-15:
                    max_abs = 1.0
                scaled = arr / max_abs
                grid = np.linspace(scaled.min(), scaled.max(), rkhs_dim)
                # Use scipy KDE on the scaled diffs
                try:
                    kde = gaussian_kde(scaled)
                    density = kde(grid).astype(np.float32)
                    density = density / max(density.sum(), 1e-10)
                except Exception:
                    density = np.zeros(rkhs_dim, dtype=np.float32)
                edge_vectors[edge] = torch.tensor(density, dtype=torch.float32)
            else:
                vec = kde_computer.kde_to_rkhs_vector(arr)
                edge_vectors[edge] = torch.tensor(vec, dtype=torch.float32)

    logger.info(f"Computed {len(edge_vectors):,} RKHS vectors")
    return edge_vectors, edge_occurrence_counts


# =====================================================================
#  Summary statistics mode: (first_ts, last_ts, count) per edge
# =====================================================================

def compute_summary_stat_vectors(
    edge_timestamps: Dict[Tuple[int, int, int], List[float]],
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int, int], int]]:
    """
    Compute a 3-D summary vector (first_ts, last_ts, count) for EVERY edge.

    Unlike the KDE/DPGMM path, there is no minimum-occurrences filter:
    every edge that appears at least once gets a vector.

    Parameters
    ----------
    edge_timestamps : dict  --  (src, dst, et) -> list of timestamps

    Returns
    -------
    edge_vectors : dict  --  (src, dst, et) -> Tensor of shape (3,)
    edge_occurrence_counts : dict  --  (src, dst, et) -> int
    """
    edge_vectors: Dict[Tuple[int, int, int], torch.Tensor] = {}
    edge_occurrence_counts: Dict[Tuple[int, int, int], int] = {}

    for edge, ts_list in edge_timestamps.items():
        count = len(ts_list)
        edge_occurrence_counts[edge] = count
        ts_arr = np.array(ts_list, dtype=np.float64)
        first_ts = float(ts_arr.min())
        last_ts = float(ts_arr.max())
        edge_vectors[edge] = torch.tensor(
            [first_ts, last_ts, float(count)], dtype=torch.float32
        )

    logger.info(f"Computed summary-stat vectors for {len(edge_vectors):,} edges "
                f"(all edges, no min_occurrences filter)")
    return edge_vectors, edge_occurrence_counts


# =====================================================================
#  Save results
# =====================================================================

def save_results(
    edge_vectors: Dict[Tuple[int, int, int], torch.Tensor],
    edge_occurrence_counts: Dict[Tuple[int, int, int], int],
    dataset_name: str,
    kde_params: Dict,
    output_dir: str = "kde_vectors",
):
    """
    Persist RKHS vectors and metadata.

    Output
    ------
    ``{output_dir}/{dataset_name}_kde_vectors.pt`` with::

        {'edge_vectors': {(src, dst, et): Tensor, ...},
         'metadata':     {...}}

    ``{output_dir}/{dataset_name}_kde_stats.json`` with summary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)

    edge_counts_ser = {
        f"{s},{d},{et}": c for (s, d, et), c in edge_occurrence_counts.items()
    }

    metadata = {
        "dataset": dataset_name,
        "min_occurrences": kde_params.get("min_occurrences", 10),
        "rkhs_dim": kde_params.get("rkhs_dim", 20),
        "n_quadrature_points": kde_params.get("n_quadrature_points", 10),
        "bandwidth": kde_params.get("bandwidth", "scott"),
        "n_components": kde_params.get("n_components", 100),
        "gamma": kde_params.get("gamma", 5.0),
        "scaler": "MaxAbsScaler",
        "init_method": kde_params.get("init_method", "kmeans"),
        "num_edges": len(edge_vectors),
        "num_total_edges_tracked": len(edge_occurrence_counts),
        "edge_key_format": "(src, dst, edge_type)",
        "timestamp": datetime.now().isoformat(),
        "edge_occurrence_counts": edge_counts_ser,
    }

    output_file = os.path.join(output_dir, f"{dataset_name}_kde_vectors.pt")
    logger.info(f"Saving RKHS vectors -> {output_file}")
    torch.save({"edge_vectors": edge_vectors, "metadata": metadata}, output_file)

    # Statistics
    stats_file = os.path.join(output_dir, f"{dataset_name}_kde_stats.json")
    if edge_vectors:
        norms = [v.norm().item() for v in edge_vectors.values()]
        stats = {
            **metadata,
            "statistics": {
                "num_edges_with_vectors": len(edge_vectors),
                "vector_norm_mean": float(np.mean(norms)),
                "vector_norm_std": float(np.std(norms)),
                "vector_norm_min": float(np.min(norms)),
                "vector_norm_max": float(np.max(norms)),
                "total_size_mb": os.path.getsize(output_file) / (1024 * 1024),
            },
        }
    else:
        stats = {
            **metadata,
            "statistics": {
                "num_edges_with_vectors": 0,
                "vector_norm_mean": 0.0,
                "vector_norm_std": 0.0,
                "vector_norm_min": 0.0,
                "vector_norm_max": 0.0,
                "total_size_mb": os.path.getsize(output_file) / (1024 * 1024),
            },
        }

    # Remove non-serialisable edge_occurrence_counts from JSON stats
    json_stats = {k: v for k, v in stats.items() if k != "edge_occurrence_counts"}
    with open(stats_file, "w") as f:
        json.dump(json_stats, f, indent=2)

    logger.info(f"Stats -> {stats_file}")
    logger.info(
        f"  vectors: {stats['statistics']['num_edges_with_vectors']}, "
        f"size: {stats['statistics']['total_size_mb']:.2f} MB"
    )


# =====================================================================
#  CLI
# =====================================================================

def _detect_cicids(dataset_name: str, model_name: str) -> bool:
    """Heuristic: is this a CIC-IDS dataset?"""
    combined = (dataset_name + model_name).lower()
    return "cicids" in combined or "cic_ids" in combined or "cic-ids" in combined


def main():
    parser = argparse.ArgumentParser(
        description="Compute RKHS vectors for temporal graph edges "
        "(GPU-accelerated BayesianGaussianMixture DPGMM)",
    )
    parser.add_argument(
        "model", type=str,
        help="Config name (e.g. kairos_cicids_kde_ts, kairos_cicids_kde_diff)",
    )
    parser.add_argument(
        "dataset", type=str,
        help="Dataset name (e.g. CICIDS_MONDAY, CLEARSCOPE_E3)",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Chunk size for GPU batches (default 256)")
    parser.add_argument(
        "--use_timestamp_diffs", action="store_true",
        help="Compute on inter-arrival times instead of raw timestamps",
    )
    parser.add_argument(
        "--use_summary_stats", action="store_true",
        help="Compute (first_ts, last_ts, count) summary vectors for ALL edges "
             "(no min_occurrences filter, no KDE/DPGMM). rkhs_dim is forced to 3.",
    )
    parser.add_argument("--feat_hash", type=str, default=None)
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="IO threads for file loading",
    )
    parser.add_argument(
        "--artifacts_dir", type=str, default=None,
        help="Base artifacts directory containing feat_inference/ "
             "(e.g. artifacts_cicids). Overrides config feat_inference_dir. "
             "Defaults to kde_params.feat_inference_dir in config, then 'artifacts'.",
    )

    args = parser.parse_args()

    # Load config
    cfg_dict = load_config(args.model)
    cfg = SimpleConfig(cfg_dict, args.dataset)
    kde_params = cfg_dict.get("kde_params", {})

    use_diffs = args.use_timestamp_diffs or kde_params.get("use_timestamp_diffs", False)
    use_summary_stats = args.use_summary_stats or kde_params.get("use_summary_stats", False)
    output_dir = args.output_dir or kde_params.get("kde_vectors_dir", "kde_vectors")
    # Resolve artifacts_dir: CLI > config > default 'artifacts'
    artifacts_dir = (
        args.artifacts_dir
        or kde_params.get("feat_inference_dir", None)
        or "artifacts"
    )
    is_cicids = _detect_cicids(args.dataset, args.model)

    if use_summary_stats:
        mode_str = "Summary stats (first_ts, last_ts, count)"
    elif use_diffs:
        mode_str = "Timestamp DIFFS"
    else:
        mode_str = "Raw timestamps"

    logger.info("=" * 80)
    logger.info("KDE RKHS Vector Computation  (BayesianGaussianMixtureGPU)")
    logger.info("=" * 80)
    logger.info(f"Model:      {args.model}")
    logger.info(f"Dataset:    {args.dataset}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Artifacts:  {artifacts_dir}")
    logger.info(f"Mode:       {mode_str}")
    logger.info(f"Scaler:     MaxAbsScaler")
    logger.info(f"Init:       {kde_params.get('init_method', 'kmeans')}")
    logger.info(f"CIC-IDS:    {is_cicids}")
    logger.info(f"Feat hash:  {args.feat_hash or '(auto-latest)'}")
    logger.info(f"GPU:        {torch.cuda.is_available()}")
    logger.info("=" * 80)

    # Extract edge timestamps
    edge_timestamps = extract_edge_timestamps(
        cfg, args.dataset, feat_hash=args.feat_hash, n_workers=args.n_workers,
        artifacts_dir=artifacts_dir,
    )

    if use_summary_stats:
        # Summary stats mode: (first_ts, last_ts, count) for EVERY edge
        edge_vectors, edge_occurrence_counts = compute_summary_stat_vectors(
            edge_timestamps,
        )
        kde_params_full = dict(kde_params)
        kde_params_full["use_summary_stats"] = True
        kde_params_full["rkhs_dim"] = 3
        kde_params_full["min_occurrences"] = 1
    else:
        # Standard KDE / DPGMM mode
        kde_computer = KDEVectorComputer(
            rkhs_dim=kde_params.get("rkhs_dim", 20),
            min_occurrences=kde_params.get("min_occurrences", 10),
            bandwidth=kde_params.get("bandwidth", "scott"),
            n_quadrature_points=kde_params.get("n_quadrature_points", 10),
            use_timestamp_diffs=use_diffs,
        )
        edge_vectors, edge_occurrence_counts = compute_rkhs_vectors(
            edge_timestamps,
            kde_computer,
            kde_params,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            is_cicids=is_cicids,
        )
        kde_params_full = dict(kde_params)
        kde_params_full["use_timestamp_diffs"] = use_diffs

    # Save
    save_results(
        edge_vectors, edge_occurrence_counts,
        args.dataset, kde_params_full, output_dir,
    )

    logger.info("=" * 80)
    logger.info("Computation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
