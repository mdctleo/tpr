from typing import Any, Dict, List, Sequence

import math

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from joblib import Parallel, delayed

import torch


def _logsumexp(x: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable logsumexp."""
    x = np.asarray(x)
    x_max = np.max(x, axis=axis, keepdims=True)
    safe_max = np.where(np.isfinite(x_max), x_max, 0.0)
    exp_sum = np.sum(np.exp(x - safe_max), axis=axis, keepdims=True)
    result = safe_max + np.log(exp_sum)
    result = np.where(np.isfinite(x_max), result, -np.inf)
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    return result



def _coerce_1d(values: Any, *, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 2:
        if arr.shape[1] != 1:
            raise ValueError(f"Expected shape (n_samples, 1), got {arr.shape}.")
        arr = arr[:, 0]
    elif arr.ndim != 1:
        raise ValueError(f"Expected 1D or 2D single-column values, got {arr.shape}.")
    return arr

def compute_batched_nll_cpu(
    batch_dict: Dict[str, np.ndarray],
    sample_size: int = 500000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute batch of negative log likelihoods on CPU using NumPy.
    Use compute_batched_nll() wrapper for automatic GPU fallback.

    Parameters:
    batch_dict: output from build_batched_gmm_inputs()
    sample_size: subsample timestamps if more than this (for speed)
    random_state: RNG seed

    Returns:
    [B] array of NLL values (one per row/edge)
    """
    timestamps = batch_dict["timestamps"]  # [B, N]
    timestamp_mask = batch_dict["timestamp_mask"]  # [B, N]
    means = batch_dict["means"]  # [B, K]
    covariances = batch_dict["covariances"]  # [B, K]
    weights = batch_dict["weights"]  # [B, K]
    weight_mask = batch_dict["weight_mask"]  # [B, K]

    B = timestamps.shape[0]
    
    # Create eval_mask and subsample before broadcasting to reduce memory
    eval_mask = timestamp_mask.copy()
    if sample_size and sample_size > 0:
        rng = np.random.default_rng(random_state)
        for b in range(B):
            valid_idx = np.flatnonzero(timestamp_mask[b])
            if len(valid_idx) > sample_size:
                keep_idx = rng.choice(valid_idx, size=sample_size, replace=False)
                eval_mask[b] = False
                eval_mask[b, keep_idx] = True
    
    # Extract only valid/subsampled timestamps to reduce N before broadcasting
    # Find max number of valid timestamps per row
    max_valid_per_row = eval_mask.sum(axis=1).max()
    
    # Build reduced timestamp array [B, N_reduced]
    timestamps_reduced = np.full((B, max_valid_per_row), np.nan, dtype=timestamps.dtype)
    for b in range(B):
        valid_idx = np.flatnonzero(eval_mask[b])
        n_valid = len(valid_idx)
        timestamps_reduced[b, :n_valid] = timestamps[b, valid_idx]
    
    # Compute log(w_k) with masking
    log_w = np.where(weight_mask[:, None], np.log(np.clip(weights, 1e-12, None))[:, None], -np.inf)  # [B, 1, K]

    # Now do [B, N_reduced, K] broadcasting instead of [B, N_full, K]
    diff = timestamps_reduced[:, :, None] - means[:, None, :]  # [B, N_reduced, K]
    log_gauss = -0.5 * (
        np.log(2.0 * math.pi * covariances[:, None, :])
        + (diff * diff) / covariances[:, None, :]
    )  # [B, N_reduced, K]

    # log p(x) = logsumexp_k(log w_k + log N(x | mu_k, var_k))
    weighted = log_gauss + log_w  # [B, N_reduced, K]
    log_px = _logsumexp(weighted, axis=-1)  # [B, N_reduced]

    # Average log likelihood over valid samples per row
    counts = np.maximum(eval_mask.sum(axis=1), 1)  # [B]
    summed = np.nansum(np.where(np.isfinite(log_px), log_px, 0.0), axis=1)  # [B]
    avg_log_likelihood = summed / counts  # [B]
    
    nll = -avg_log_likelihood  # [B]
    return nll


def compute_batched_nll_gpu(
    batch_dict: Dict[str, np.ndarray],
    sample_size: int = 500000,
    random_state: int = 42,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute batch of negative log likelihoods on GPU using PyTorch.
    Significantly faster than CPU version for large N and B.

    Parameters:
    batch_dict: output from build_batched_gmm_inputs()
    sample_size: subsample timestamps if more than this (for speed)
    random_state: RNG seed
    device: torch device ("cuda", "cuda:0", etc.)

    Returns:
    [B] array of NLL values (one per row/edge)
    """
    timestamps = batch_dict["timestamps"]  # [B, N]
    timestamp_mask = batch_dict["timestamp_mask"]  # [B, N]
    means = batch_dict["means"]  # [B, K]
    covariances = batch_dict["covariances"]  # [B, K]
    weights = batch_dict["weights"]  # [B, K]
    weight_mask = batch_dict["weight_mask"]  # [B, K]

    B = timestamps.shape[0]
    
    # Create eval_mask and subsample before broadcasting to reduce memory
    eval_mask = timestamp_mask.copy()
    if sample_size and sample_size > 0:
        rng = np.random.default_rng(random_state)
        for b in range(B):
            valid_idx = np.flatnonzero(timestamp_mask[b])
            if len(valid_idx) > sample_size:
                keep_idx = rng.choice(valid_idx, size=sample_size, replace=False)
                eval_mask[b] = False
                eval_mask[b, keep_idx] = True
    
    # Extract only valid/subsampled timestamps to reduce N before broadcasting
    max_valid_per_row = eval_mask.sum(axis=1).max()
    
    # Build reduced timestamp array [B, N_reduced]
    timestamps_reduced = np.full((B, max_valid_per_row), np.nan, dtype=np.float32)
    for b in range(B):
        valid_idx = np.flatnonzero(eval_mask[b])
        n_valid = len(valid_idx)
        timestamps_reduced[b, :n_valid] = timestamps[b, valid_idx]

    # print(f"GPU NLL: Reduced timestamps to shape {timestamps_reduced.shape} for efficient GPU computation", flush=True)
    
    # Convert to GPU tensors
    timestamps_gpu = torch.from_numpy(timestamps_reduced).to(device)  # [B, N_reduced]
    means_gpu = torch.from_numpy(means.astype(np.float32)).to(device)  # [B, K]
    covariances_gpu = torch.from_numpy(covariances.astype(np.float32)).to(device)  # [B, K]
    weights_gpu = torch.from_numpy(weights.astype(np.float32)).to(device)  # [B, K]
    weight_mask_gpu = torch.from_numpy(weight_mask).to(device)  # [B, K]
    
    # Compute log(w_k) with masking on GPU
    log_w = torch.where(
        weight_mask_gpu[:, None],
        torch.log(torch.clamp(weights_gpu, min=1e-12))[:, None],
        torch.tensor(-np.inf, dtype=torch.float32, device=device)
    )  # [B, 1, K]
    
    # GPU [B, N_reduced, K] broadcasting
    diff = timestamps_gpu[:, :, None] - means_gpu[:, None, :]  # [B, N_reduced, K]
    log_gauss = -0.5 * (
        torch.log(2.0 * math.pi * covariances_gpu[:, None, :])
        + (diff * diff) / covariances_gpu[:, None, :]
    )  # [B, N_reduced, K]
    
    # log p(x) = logsumexp_k(log w_k + log N(x | mu_k, var_k))
    weighted = log_gauss + log_w  # [B, N_reduced, K]
    log_px = torch.logsumexp(weighted, dim=-1)  # [B, N_reduced]
    
    # Average log likelihood over valid samples per row
    counts = torch.tensor(np.maximum(eval_mask.sum(axis=1), 1), dtype=torch.float32, device=device)  # [B]
    summed = torch.nansum(torch.where(torch.isfinite(log_px), log_px, torch.tensor(0.0, device=device)), dim=1)  # [B]
    avg_log_likelihood = summed / counts  # [B]
    
    nll = -avg_log_likelihood  # [B]
    return nll.cpu().numpy()


def build_batched_gmm_inputs(
    gmms: Sequence[GaussianMixture],
    raw_timestamps: Sequence[Any],
    batch_size: int = 256,
) -> List[Dict[str, np.ndarray]]:
    """
    Extract and pad matrices from sklearn GMMs and raw timestamps for batched evaluation.
    Returns results chunked by batch_size.

    Parameters:
    gmms: list of fitted sklearn GaussianMixture objects.
    raw_timestamps: list of timestamp arrays/lists (1D or 2D single-column).
    batch_size: number of elements per returned batch (default: 256)

    Returns:
    list of dicts, each with keys:
    - "timestamps": [B, N_max] padded timestamp matrix (NaN-filled)
    - "timestamp_mask": [B, N_max] bool mask of valid positions
    - "means": [B, K_max] padded component means
    - "covariances": [B, K_max] padded component variances
    - "weights": [B, K_max] padded normalized component weights
    - "weight_mask": [B, K_max] bool mask of valid components
    """
    if len(gmms) != len(raw_timestamps):
        raise ValueError("gmms and raw_timestamps must have the same length.")
    
    total_size = len(gmms)
    results = []
    
    # Process in chunks
    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        
        chunk_gmms = gmms[start_idx:end_idx]
        chunk_timestamps = raw_timestamps[start_idx:end_idx]
        chunk_batch_size = end_idx - start_idx
        
        # Extract and normalize raw timestamps into padded matrix
        ts_list: List[np.ndarray] = []
        for values in chunk_timestamps:
            ts = _coerce_1d(values, dtype=np.float64)
            ts = ts[np.isfinite(ts)]
            if ts.size == 0:
                raise ValueError("timestamp array has no finite values")
            ts_list.append(ts)
        
        max_ts = max(len(ts) for ts in ts_list)
        timestamps = np.full((chunk_batch_size, max_ts), np.nan, dtype=np.float64)
        timestamp_mask = np.zeros((chunk_batch_size, max_ts), dtype=bool)
        
        for i, ts in enumerate(ts_list):
            n = len(ts)
            timestamps[i, :n] = ts
            timestamp_mask[i, :n] = True
        
        # Extract and normalize GMM parameters into padded matrices
        means_list: List[np.ndarray] = []
        covariances_list: List[np.ndarray] = []
        weights_list: List[np.ndarray] = []
        
        for gmm in chunk_gmms:
            m = _coerce_1d(gmm.means_, dtype=np.float64)
            c = np.clip(_coerce_1d(gmm.covariances_, dtype=np.float64), 1e-12, None)
            w = _coerce_1d(gmm.weights_, dtype=np.float64)
            
            if not (m.shape[0] == c.shape[0] == w.shape[0]):
                raise ValueError("GMM parameter length mismatch")
            
            w = w / w.sum()  # renormalize
            
            means_list.append(m)
            covariances_list.append(c)
            weights_list.append(w)
        
        max_k = max(len(w) for w in weights_list)
        means = np.zeros((chunk_batch_size, max_k), dtype=np.float64)
        covariances = np.ones((chunk_batch_size, max_k), dtype=np.float64)
        weights = np.zeros((chunk_batch_size, max_k), dtype=np.float64)
        weight_mask = np.zeros((chunk_batch_size, max_k), dtype=bool)
        
        for i, (m, c, w) in enumerate(zip(means_list, covariances_list, weights_list)):
            k = len(w)
            means[i, :k] = m
            covariances[i, :k] = c
            weights[i, :k] = w
            weight_mask[i, :k] = True
        
        results.append({
            "timestamps": timestamps,
            "timestamp_mask": timestamp_mask,
            "means": means,
            "covariances": covariances,
            "weights": weights,
            "weight_mask": weight_mask,
        })
    
    return results


def _sheather_jones_bandwidth_1d(data):
    """
    Compute a Sheather-Jones-style bandwidth using adaptive quadrature.
    Always returns a valid bandwidth; never returns None.
    Falls back through multiple strategies if earlier methods fail.
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    # Remove non-finite values
    data = data[np.isfinite(data)]
    n = len(data)
    
    # If insufficient data, use default rule of thumb
    if n < 2:
        return 1.0
    
    # Early exit if all data is identical
    if np.std(data) == 0:
        data_range = np.ptp(data)
        if data_range == 0:
            return 1.0  # Constant data
        return data_range / 10.0  # Small bandwidth for narrow range
    
    std_dev = np.std(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    sigma = min(std_dev, iqr / 1.34)
    
    # Try Sheather-Jones
    if sigma > 0:
        try:
            def pilot_density_second_derivative(x):
                return np.mean(
                    norm.pdf((x - data) / sigma) * ((x - data) ** 2 - sigma**2) / sigma**5
                )

            def integrand(x):
                return pilot_density_second_derivative(x) ** 2

            integral, _ = quad(integrand, np.min(data) - 3 * sigma, np.max(data) + 3 * sigma)
            integral += 0.01 * np.var(data)

            if integral > 0 and np.isfinite(integral):
                bandwidth = (1.06 * sigma * n ** (-1 / 5)) / (integral ** (1 / 5))
                if np.isfinite(bandwidth) and bandwidth > 0:
                    return bandwidth
        except Exception:
            pass  # Fall through to fallback
    
    # Fallback 1: Scott's rule (1.06 * std * n^(-1/5))
    scott_bw = 1.06 * std_dev * n ** (-1 / 5)
    if np.isfinite(scott_bw) and scott_bw > 0:
        return scott_bw
    
    # Fallback 2: Silverman's rule (0.9 * std * n^(-1/5))
    silverman_bw = 0.9 * std_dev * n ** (-1 / 5)
    if np.isfinite(silverman_bw) and silverman_bw > 0:
        return silverman_bw
    
    # Fallback 3: Simple rule based on data range
    data_range = np.ptp(data)
    if data_range > 0:
        return data_range / (4 * n ** (1/5))
    
    # Fallback 4: Last resort
    return 1.0


def _compute_kl_single_row(
    b: int,
    x_data: np.ndarray,
    means_b: np.ndarray,
    covs_b: np.ndarray,
    weights_b: np.ndarray,
    mask_k: np.ndarray,
    grid_size: int,
    epsilon: float,
) -> float:
    """
    Compute KL divergence for a single row.
    Returns 0.0 as default for edge cases (no data, constant data, numerical issues).
    
    Parameters:
    b: row index (for debugging)
    x_data: valid timestamps for this row
    means_b, covs_b, weights_b, mask_k: GMM parameters
    grid_size: number of grid points
    epsilon: regularization
    
    Returns:
    KL divergence value (or 0.0 if computation fails)
    """
    if len(x_data) == 0:
        return 0.0  # No data: no divergence
    
    x_min, x_max = x_data.min(), x_data.max()
    
    if x_min == x_max:
        return 0.0  # Constant data: consider as perfect fit
    
    # Create evaluation grid
    grid = np.linspace(x_min, x_max, int(max(128, grid_size)), dtype=np.float64)
    
    # === TRUE DENSITY P: KDE from raw data ===
    bw = _sheather_jones_bandwidth_1d(x_data)
    kde = KernelDensity(kernel="gaussian", bandwidth=bw).fit(x_data.reshape(-1, 1))
    log_p_kde = kde.score_samples(grid.reshape(-1, 1))
    p = np.exp(log_p_kde)
    p_sum = p.sum()
    
    # Guard: invalid KDE density
    if not np.isfinite(p_sum) or p_sum <= 0:
        return 0.0
    
    p = p / (p_sum + epsilon)
    
    # === MODEL DENSITY Q: GMM ===
    # Vectorized Gaussian evaluation
    diff = grid[:, np.newaxis] - means_b[np.newaxis, :]  # [grid_size, K]
    q_components = (
        np.exp(-0.5 * (diff * diff) / covs_b[np.newaxis, :])
        / np.sqrt(2.0 * math.pi * covs_b[np.newaxis, :])
    )  # [grid_size, K]
    
    # Apply masks and weights
    q_components[:, ~mask_k] = 0.0
    q = np.sum(q_components * weights_b[np.newaxis, :], axis=1)  # [grid_size]
    q_sum = q.sum()
    
    # Guard: all components masked or invalid GMM
    if not np.isfinite(q_sum) or q_sum <= 0:
        return 0.0
    
    q = q / (q_sum + epsilon)
    
    # === KL(P || Q) ===
    p_safe = p + epsilon
    q_safe = q + epsilon
    log_p_safe = np.log(p_safe)
    log_q_safe = np.log(q_safe)
    
    # Guard: invalid log values
    if not np.all(np.isfinite(log_p_safe)) or not np.all(np.isfinite(log_q_safe)):
        return 0.0
    
    kl = np.sum(p_safe * (log_p_safe - log_q_safe))
    
    # Guard: final result invalid
    if not np.isfinite(kl):
        return 0.0
    
    return kl



def compute_batched_kl(
    batch_dict: Dict[str, np.ndarray],
    grid_size: int = 2048,
    epsilon: float = 1e-10,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute batch of KL divergences: KL(P_kde || Q_gmm).
    
    P is a Kernel Density Estimate from raw timestamps (Sheather-Jones bandwidth).
    Q is the fitted Gaussian Mixture Model.
    Both evaluated on a row-specific grid.
    
    Uses joblib parallelization across rows for CPU efficiency.

    Parameters:
    batch_dict: output from build_batched_gmm_inputs()
    grid_size: number of grid points per row
    epsilon: regularization to avoid log(0)
    n_jobs: number of parallel jobs (-1 uses all CPUs)

    Returns:
    [B] array of KL divergence values (one per row/edge)
    """
    timestamps = batch_dict["timestamps"]  # [B, N]
    timestamp_mask = batch_dict["timestamp_mask"]  # [B, N]
    means = batch_dict["means"]  # [B, K]
    covariances = batch_dict["covariances"]  # [B, K]
    weights = batch_dict["weights"]  # [B, K]
    weight_mask = batch_dict["weight_mask"]  # [B, K]

    B = timestamps.shape[0]

    # Extract valid data for each row (lazy, just indices)
    row_data = []
    for b in range(B):
        mask_b = timestamp_mask[b]
        if not mask_b.any():
            row_data.append(None)
        else:
            x_data = timestamps[b, mask_b]
            row_data.append((
                x_data,
                means[b],
                covariances[b],
                weights[b],
                weight_mask[b],
            ))

    # Parallelize KL computation across rows
    kl_divs = Parallel(n_jobs=n_jobs)(
        delayed(_compute_kl_single_row)(
            b,
            row_data[b][0] if row_data[b] else np.array([]),
            row_data[b][1] if row_data[b] else np.array([]),
            row_data[b][2] if row_data[b] else np.array([]),
            row_data[b][3] if row_data[b] else np.array([]),
            row_data[b][4] if row_data[b] else np.array([]),
            grid_size,
            epsilon,
        )
        for b in range(B)
    )

    return np.array(kl_divs, dtype=np.float64)
