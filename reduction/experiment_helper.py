from matplotlib import pyplot as plt

from create_graph_memory import fit_edges_memory, filter_edges_by_count, create_and_merge_graph, average_of_edges
from batched_metrics import build_batched_gmm_inputs, compute_batched_kl, compute_batched_nll_gpu
from tqdm import tqdm
from enhanced_edge import EnhancedEdge
from BayesianGaussianMixtureGPU import BayesianGaussianMixtureGPU
from pyro_dpgmm import truncate_clusters
from datetime import datetime
from dbstream import OnlineDBStreamClustering, tune_dbstream_params
from concurrent.futures import ThreadPoolExecutor, as_completed

import glob
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import pickle
import os

    
def load_merged_edges(csv_path, target_nodes, args):
    files = []
    if type(csv_path) is list:
        for path in csv_path:
            files.extend(glob.glob(path))
    else:
        files = glob.glob(csv_path)

    merged_edges, total_edge_count = create_and_merge_graph(files, target_nodes, args)

    print("Total edge count: ", total_edge_count, flush=True)
    print("Number of merged edges: ", len(merged_edges), flush=True)
    
    # Optionally compute timestamp differences right after loading
    use_timestamp_differences = getattr(args, "use_timestamp_differences", False)
    if use_timestamp_differences:
        print("Converting timestamps to differences (delta timestamps)...", flush=True)
        merged_edges = compute_timestamp_differences(merged_edges)
        print("Number of merged edges after differencing: ", len(merged_edges), flush=True)
    
    return merged_edges, total_edge_count


def filter_merged_edges(merged_edges, min_count=200):
    merged_edges = filter_edges_by_count(merged_edges, min_count=min_count)

    edge_counts = np.array([len(edge) for edge in merged_edges.values()])
    if edge_counts.size > 0:
        avg_edge_count_pre_filter = float(np.mean(edge_counts))
        median_edge_count_pre_filter = float(np.median(edge_counts))
        min_edge_count_pre_filter = int(np.min(edge_counts))
        max_edge_count_pre_filter = int(np.max(edge_counts))
    else:
        avg_edge_count_pre_filter = median_edge_count_pre_filter = 0.0
        min_edge_count_pre_filter = max_edge_count_pre_filter = 0

    print(
        f"Edge count statistics before filtering: avg={avg_edge_count_pre_filter}, "
        f"median={median_edge_count_pre_filter}, min={min_edge_count_pre_filter}, "
        f"max={max_edge_count_pre_filter}",
        flush=True,
    )
    print("Number of merged edges after filtering: ", len(merged_edges), flush=True)
    return merged_edges


def scale_merged_edges(merged_edges, scaler_type="minmax"):
    scaler_name = scaler_type.lower()
    scaler_map = {
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "maxabs": MaxAbsScaler,
        "none": None,
    }
    if scaler_name not in scaler_map:
        raise ValueError("scaler_type must be 'minmax', 'standard', 'maxabs', or 'none'.")

    scaler_cls = scaler_map[scaler_name]
    scalers = {}
    for key, timestamps in tqdm(
        merged_edges.items(),
        total=len(merged_edges),
        desc="scaling edges...",
    ):
        merged_edges[key] = np.array(sorted(timestamps), dtype=np.float64).reshape(-1, 1)

        if scaler_cls is None:
            scalers[key] = None
            continue

        scaler = scaler_cls()
        merged_edges[key] = scaler.fit_transform(merged_edges[key])
        scalers[key] = scaler

    return merged_edges, scalers


def compute_timestamp_differences(merged_edges):
    """
    Convert raw timestamps to timestamp differences (delta timestamps).
    
    For each edge, computes differences: diff[i] = timestamps[i] - timestamps[i-1]
    The first timestamp is dropped since it has no previous value.
    
    Parameters:
    merged_edges (dict): Mapping from edge key to timestamp sequences.
    
    Returns:
    dict: Mapping from edge key to timestamp differences.
    
    Example:
    >>> timestamps = [100, 105, 110, 115]
    >>> differences = [5, 5, 5]  # 105-100, 110-105, 115-110
    """
    diff_edges = {}
    
    for key, timestamps in tqdm(
        merged_edges.items(),
        total=len(merged_edges),
        desc="Computing timestamp differences...",
    ):
        # Convert to numpy array and sort
        ts_array = np.array(sorted(timestamps), dtype=np.float64)
        
        # Compute differences: each value minus the previous value
        if len(ts_array) > 1:
            differences = np.diff(ts_array)
            diff_edges[key] = differences
        else:
            # If only 1 timestamp, we can't compute differences, keep empty or skip
            diff_edges[key] = np.array([], dtype=np.float64)
    
    return diff_edges


def _coerce_timestamp_array(values, dtype=np.float64):
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError(f"Expected timestamps with shape (n_samples, 1), got {values.shape}.")
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError(f"Expected timestamps to be 1D or 2D with one column, got {values.shape}.")
    return values


def split_merged_edges_by_datapoint_count(merged_edges, update_batch_size):
    """
    Split a merged_edges dictionary into multiple batches, where each batch contains
    all edge keys with chunks of approximately `update_batch_size` data points per edge.
    
    Each edge's timestamps are independently split into chunks. Batches are created by 
    pairing chunks across all edges round-robin style, so each batch contains one chunk 
    per edge (if that edge still has remaining chunks).
    
    Parameters:
    merged_edges (dict): Mapping from edge key (node1_id, node2_id, syscall) 
        to list/array of timestamps.
    update_batch_size (int): Target number of data points per chunk per edge.
    
    Returns:
    list[dict]: List of batch dictionaries. Each batch contains all edge keys 
        that have remaining chunks, with each key mapped to one chunk of timestamps.
    
    Example:
    >>> merged_edges = {
    ...     ('a', 'b', 'sys1'): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ...     ('c', 'd', 'sys2'): list(range(1, 41)),  # 40 timestamps
    ... }
    >>> batches = split_merged_edges_by_datapoint_count(merged_edges, 5)
    >>> # Batch 0: {('a', 'b', 'sys1'): [1..5], ('c', 'd', 'sys2'): [1..5]}
    >>> # Batch 1: {('a', 'b', 'sys1'): [6..10], ('c', 'd', 'sys2'): [6..10]}
    >>> # Batch 2: {('c', 'd', 'sys2'): [11..15]}
    >>> # Batch 3: {('c', 'd', 'sys2'): [16..20]}
    >>> # ... (more batches for edge B only until all 40 timestamps are consumed)
    """
    if update_batch_size <= 0:
        raise ValueError("update_batch_size must be > 0.")
    
    # Step 1: Split each edge's timestamps into chunks
    edge_chunks = {}
    for edge_key, timestamps in merged_edges.items():
        # Convert to list if ndarray
        timestamps_list = timestamps.tolist() if isinstance(timestamps, np.ndarray) else list(timestamps)
        
        # Split into chunks of update_batch_size
        chunks = []
        for i in range(0, len(timestamps_list), update_batch_size):
            chunk = timestamps_list[i:i + update_batch_size]
            chunks.append(chunk)
        
        edge_chunks[edge_key] = chunks
    
    # Step 2: Create batches by pairing chunks round-robin across all edges
    batches = []
    max_chunks = max(len(chunks) for chunks in edge_chunks.values()) if edge_chunks else 0
    
    for batch_idx in range(max_chunks):
        batch = {}
        for edge_key, chunks in edge_chunks.items():
            if batch_idx < len(chunks):
                batch[edge_key] = chunks[batch_idx]
        
        if batch:  # Only add non-empty batches
            batches.append(batch)
    
    return batches


def _extract_gauss_hermite_points_from_gmm(gmm, n_points=10, total_weight=1.0):
    """
    Extract Gauss-Hermite quadrature points from a 1D Gaussian Mixture Model.
    
    For each component in the GMM, generate Gauss-Hermite quadrature points scaled to
    the component's mean and variance. This gives an optimal numerical approximation
    of the mixture distribution.
    
    Parameters:
    gmm: sklearn GaussianMixture object (assumed 1D with diag covariance)
    n_points (int): Number of Gauss-Hermite quadrature points per component.
    total_weight (float): Scale the total weight to this value.
    
    Returns:
    tuple: (points, weights) where
        - points: (n_components * n_points,) array of quadrature points
        - weights: (n_components * n_points,) array of normalized weights
    
    Notes:
    Gauss-Hermite quadrature is mathematically optimal for Gaussian integrals and
    naturally concentrates points in high-probability regions (typically within ~3σ).
    This is superior to uniform grid sampling because:
    - Points follow the distribution structure
    - No need for explicit padding
    - Requires fewer points for same fidelity
    """
    from numpy.polynomial.hermite import hermgauss
    
    # Get Gauss-Hermite quadrature nodes and weights
    gh_points, gh_weights = hermgauss(n_points)  # Standard Gaussian quadrature
    gh_points = np.asarray(gh_points, dtype=np.float64)
    gh_weights = np.asarray(gh_weights, dtype=np.float64)
    
    all_points = []
    all_weights = []
    
    n_components = len(gmm.weights_)
    means = gmm.means_.ravel()
    covariances = gmm.covariances_.ravel()
    component_weights = gmm.weights_
    
    for k in range(n_components):
        mu = float(means[k])
        sigma = np.sqrt(float(covariances[k]))
        component_weight = float(component_weights[k])
        
        # Transform Gauss-Hermite points to this component's space
        # For standard Gaussian: points = μ + σ√2 * z_i
        transformed_points = mu + sigma * np.sqrt(2.0) * gh_points
        
        # Quadrature weights are normalized by √π (for standard Hermite polynomial)
        # Multiply by component weight to reflect mixture proportion
        transformed_weights = (component_weight * gh_weights) / np.sqrt(np.pi)
        
        all_points.extend(transformed_points)
        all_weights.extend(transformed_weights)
    
    all_points = np.asarray(all_points, dtype=np.float64)
    all_weights = np.asarray(all_weights, dtype=np.float64)
    
    # Normalize weights to sum to total_weight
    weight_sum = all_weights.sum()
    if weight_sum > 0:
        all_weights = (total_weight / weight_sum) * all_weights
    else:
        all_weights = np.ones_like(all_weights) / len(all_weights) * total_weight
    
    return all_points, all_weights


def merge_gmm_grid_with_batch_edges(
    previous_gmms,
    batch_merged_edges,
    edge_data_counts=None,
    args=None,
    gh_points_per_component=10,
):
    """
    Merge weighted Gauss-Hermite samples from previous GMMs with new batch data points.
    
    Always uses Gauss-Hermite quadrature points, which are mathematically optimal for
    Gaussian integrals and naturally concentrate points in high-probability regions.
    
    Weighting: old data samples are weighted by the actual count of historical data points
    they represent, while new batch points each receive weight 1.0.
    
    Parameters:
    previous_gmms (dict): Mapping from edge key to sklearn GaussianMixture object.
        These are the previously fitted models from which to extract quadrature points.
    batch_merged_edges (dict): Mapping from edge key to list/array of new timestamps.
        These are the new batch data points to merge with the old Gauss-Hermite samples.
    edge_data_counts (dict, optional): Mapping from edge key to the count of historical
        data points each edge represents. If provided, old samples for each edge are 
        weighted by this count. If not provided or edge key is missing, defaults to 1.0.
    args (Namespace, optional): Experiment arguments. If provided, gh_points_per_component
        can be extracted from args if not overridden by the parameter.
    gh_points_per_component (int): Number of Gauss-Hermite quadrature points per 
        GMM component. Typically 5-15 points suffice.
    
    Returns:
    dict: Mapping from edge key to dict with keys "timestamps" and "sample_weight".
        - "timestamps": (n_samples + n_batch, 1) array with Gauss-Hermite samples first, 
          then batch data
        - "sample_weight": (n_samples + n_batch, 1) array where old samples have total weight
          equal to their historical data count, and batch points each get weight 1.0
    
    Example:
    >>> # Edge A: 100 old data points, 10 new batch points
    >>> # Edge B: 1000 old data points, 50 new batch points
    >>> previous_gmms = {('a', 'b', 'sys'): gmm_a, ('c', 'd', 'sys'): gmm_b}
    >>> batch_edges = {('a', 'b', 'sys'): [0.3, 0.5, ...], ('c', 'd', 'sys'): [0.1, 0.2, ...]}
    >>> edge_counts = {('a', 'b', 'sys'): 100, ('c', 'd', 'sys'): 1000}
    >>> result = merge_gmm_grid_with_batch_edges(
    ...     previous_gmms, 
    ...     batch_edges,
    ...     edge_data_counts=edge_counts
    ... )
    >>> # Result['a', 'b', 'sys']['sample_weight'] will have old samples weighted by 100,
    >>> # and new batch points each weighted by 1.0
    """
    # Extract gh_points_per_component from args if provided
    if args is not None:
        gh_points_per_component = getattr(args, 'gh_points_per_component', gh_points_per_component)
    
    if not batch_merged_edges:
        return {}
    
    # Initialize edge_data_counts if not provided
    if edge_data_counts is None:
        edge_data_counts = {}
    
    augmented_edges = {}
    
    for edge_key, batch_timestamps in batch_merged_edges.items():
        # Convert to numpy array if needed
        batch_ts = np.asarray(batch_timestamps, dtype=np.float64).ravel()
        n_batch_points = len(batch_ts)
        
        # Get the corresponding GMM (if it exists)
        if edge_key not in previous_gmms:
            # If edge is new (not in previous GMMs), just use batch data with uniform weights
            augmented_edges[edge_key] = {
                "timestamps": batch_ts.reshape(-1, 1),
                "sample_weight": np.ones(n_batch_points, dtype=np.float64).reshape(-1, 1),
            }
            continue
        
        gmm = previous_gmms[edge_key]
        
        # Get the historical data count for this edge (default to 1.0 if not provided)
        old_data_count = float(edge_data_counts.get(edge_key, 1.0))
        print("old data count: ", old_data_count, flush=True)
        
        # Extract Gauss-Hermite quadrature points weighted by the historical data count
        gh_pts, gh_wts = _extract_gauss_hermite_points_from_gmm(
            gmm, 
            n_points=gh_points_per_component,
            total_weight=old_data_count
        )
        
        # Samples first (Gauss-Hermite quadrature points), then batch data
        combined_timestamps = np.concatenate([gh_pts, batch_ts], dtype=np.float64)
        
        # Create sample weights: previous weights for samples, 1.0 for batch points
        combined_weights = np.concatenate([
            gh_wts,
            np.ones(len(batch_ts), dtype=np.float64)
        ], dtype=np.float64)
        
        augmented_edges[edge_key] = {
            "timestamps": combined_timestamps.reshape(-1, 1),
            "sample_weight": combined_weights.reshape(-1, 1),
        }
    
    return augmented_edges


def _sheather_jones_bandwidth_1d(data):
    """
    Compute a Sheather-Jones-style bandwidth using adaptive quadrature.
    """
    data = np.asarray(data, dtype=np.float64).ravel()

    if len(data) < 2:
        return None
    if not np.all(np.isfinite(data)):
        return None

    n = len(data)
    std_dev = np.std(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    sigma = min(std_dev, iqr / 1.34)

    if sigma == 0:
        return None

    def pilot_density_second_derivative(x):
        return np.mean(
            norm.pdf((x - data) / sigma) * ((x - data) ** 2 - sigma**2) / sigma**5
        )

    def integrand(x):
        return pilot_density_second_derivative(x) ** 2

    integral, _ = quad(integrand, np.min(data) - 3 * sigma, np.max(data) + 3 * sigma)
    integral += 0.01 * np.var(data)

    if integral <= 0 or not np.isfinite(integral):
        fallback = 1.06 * std_dev * n ** (-1 / 5)
        return fallback if np.isfinite(fallback) and fallback > 0 else None

    bandwidth = (1.06 * sigma * n ** (-1 / 5)) / (integral ** (1 / 5))
    return bandwidth if np.isfinite(bandwidth) and bandwidth > 0 else None


def _validate_long_preprocessing_args(
    *,
    length_threshold,
    sample_length,
    kde_grid_size,
    kde_total_weight,
):
    if length_threshold < 0:
        raise ValueError("length_threshold must be >= 0.")
    if sample_length < 2:
        raise ValueError("sample_length must be >= 2.")
    if kde_grid_size < 2:
        raise ValueError("kde_grid_size must be >= 2.")
    if kde_total_weight <= 0:
        raise ValueError("kde_total_weight must be > 0.")


def random_sample_long_merged_edges(
    merged_edges,
    *,
    length_threshold=50000,
    sample_length=50000,
    random_state=42,
):
    """
    Randomly downsample only the edges longer than `length_threshold`.
    Preserves sample_weight if present alongside timestamps.
    """
    print("Random sample preprocess", flush=True)
    _validate_long_preprocessing_args(
        length_threshold=length_threshold,
        sample_length=sample_length,
        kde_grid_size=2,
        kde_total_weight=1.0,
    )

    rng = np.random.default_rng(random_state)
    processed_edges = {}
    unchanged_count = 0
    sampled_count = 0

    for key, edge_data in tqdm(
        merged_edges.items(),
        total=len(merged_edges),
        desc="Random sampling edges",
    ):
        # Extract timestamps and sample_weight if present
        if isinstance(edge_data, dict) and "timestamps" in edge_data:
            values = _coerce_timestamp_array(edge_data["timestamps"], dtype=np.float64)
            sample_weight = edge_data.get("sample_weight")
            if sample_weight is not None:
                sample_weight = _coerce_timestamp_array(sample_weight, dtype=np.float64)
        else:
            values = _coerce_timestamp_array(edge_data, dtype=np.float64)
            sample_weight = None
        
        values = values[np.isfinite(values)]
        original_length = values.shape[0]

        if original_length == 0:
            raise ValueError(f"Edge {key} has no finite timestamps after preprocessing.")

        if original_length <= length_threshold:
            if sample_weight is not None:
                processed_edges[key] = {
                    "timestamps": values.reshape(-1, 1),
                    "sample_weight": sample_weight.reshape(-1, 1),
                }
            else:
                processed_edges[key] = values.reshape(-1, 1)
            unchanged_count += 1
            continue

        target_length = min(sample_length, original_length)
        sampled_idx = rng.choice(original_length, size=target_length, replace=False)
        sampled_idx = np.sort(sampled_idx)  # Sort indices to maintain order
        sampled_values = values[sampled_idx]
        
        if sample_weight is not None:
            sampled_weights = sample_weight[sampled_idx]
            processed_edges[key] = {
                "timestamps": sampled_values.reshape(-1, 1),
                "sample_weight": sampled_weights.reshape(-1, 1),
            }
        else:
            processed_edges[key] = sampled_values.reshape(-1, 1)
        sampled_count += 1

    print(
        "Long-series random sampling summary: "
        f"unchanged={unchanged_count}, random_sampled={sampled_count}",
        flush=True,
    )
    return processed_edges


def kde_preprocess_long_merged_edges(
    merged_edges,
    *,
    length_threshold=50000,
    sample_length=2000,
    kde_grid_size=400,
    kde_total_weight=None,
    kde_grid_padding=0.10,
    bandwidth_grid_size=1024,
):
    """
    Replace edges longer than `length_threshold` with weighted KDE grid points.
    Preserves sample_weight if present alongside timestamps.

    Set `length_threshold=0` to apply this preprocessing to every edge.
    """
    print("kde preprcoess", flush=True)
    kde_total_weight = float(sample_length if kde_total_weight is None else kde_total_weight)
    _validate_long_preprocessing_args(
        length_threshold=max(length_threshold, 0),
        sample_length=sample_length,
        kde_grid_size=kde_grid_size,
        kde_total_weight=kde_total_weight,
    )

    processed_edges = {}
    unchanged_count = 0
    kde_count = 0
    fallback_count = 0

    for key, edge_data in tqdm(
        merged_edges.items(),
        total=len(merged_edges),
        desc="KDE preprocessing edges",
    ):
        # Extract timestamps and sample_weight if present
        if isinstance(edge_data, dict) and "timestamps" in edge_data:
            values_2d = np.asarray(edge_data["timestamps"], dtype=np.float64).reshape(-1, 1)
            existing_sample_weight = edge_data.get("sample_weight")
        else:
            values_2d = np.asarray(edge_data, dtype=np.float64).reshape(-1, 1)
            existing_sample_weight = None
        
        original_length = values_2d.shape[0]

        if original_length <= length_threshold:
            if existing_sample_weight is not None:
                processed_edges[key] = {
                    "timestamps": values_2d,
                    "sample_weight": np.asarray(existing_sample_weight, dtype=np.float64).reshape(-1, 1),
                }
            else:
                processed_edges[key] = values_2d
            unchanged_count += 1
            continue

        values = values_2d[:, 0]
        x_min = float(values.min())
        x_max = float(values.max())
        bandwidth = _sheather_jones_bandwidth_1d(values, grid_size=bandwidth_grid_size)
        if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
            if existing_sample_weight is not None:
                processed_edges[key] = {
                    "timestamps": values_2d,
                    "sample_weight": np.asarray(existing_sample_weight, dtype=np.float64).reshape(-1, 1),
                }
            else:
                processed_edges[key] = values_2d
            fallback_count += 1
            continue

        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(values_2d)

        width = x_max - x_min
        pad = kde_grid_padding * max(width, 1e-8)
        grid_1d = np.linspace(x_min - pad, x_max + pad, kde_grid_size, dtype=np.float64)
        grid = grid_1d[:, None]

        density = np.exp(kde.score_samples(grid))
        density_sum = density.sum(dtype=np.float64)
        if not np.isfinite(density_sum) or density_sum <= 0:
            if existing_sample_weight is not None:
                processed_edges[key] = {
                    "timestamps": values_2d,
                    "sample_weight": np.asarray(existing_sample_weight, dtype=np.float64).reshape(-1, 1),
                }
            else:
                processed_edges[key] = values_2d
            fallback_count += 1
            continue

        # Normalize KDE weights
        kde_weights = (kde_total_weight / density_sum) * density
        
        # If there were existing sample weights, combine them with KDE weights
        if existing_sample_weight is not None:
            # When combining: kde_weights already sum to kde_total_weight
            # We multiply by existing weights to weight the KDE samples
            existing_weight_avg = np.mean(existing_sample_weight)
            combined_weights = kde_weights * existing_weight_avg
        else:
            combined_weights = kde_weights
        
        processed_edges[key] = {
            "timestamps": grid,
            "sample_weight": combined_weights[:, None],
        }
        kde_count += 1

    print(
        "KDE preprocessing summary: "
        f"unchanged={unchanged_count}, kde_weighted={kde_count}, fallbacks={fallback_count}",
        flush=True,
    )
    return processed_edges


def preprocess_long_merged_edges(
    merged_edges,
    *,
    mode="none",
    length_threshold=50000,
    sample_length=50000,
    random_state=42,
    kde_grid_size=400,
    kde_total_weight=None,
    kde_grid_padding=0.10,
    bandwidth_grid_size=1024,
):
    """
    Dispatch long-series preprocessing to the selected method.
    """
    print("Current mode: ", mode, flush=True)
    if mode == "none":
        processed = {}
        for key, edge_data in merged_edges.items():
            if isinstance(edge_data, dict) and "timestamps" in edge_data:
                # Preserve structure with both timestamps and sample_weight
                timestamps = _coerce_timestamp_array(edge_data["timestamps"], dtype=np.float64).reshape(-1, 1)
                sample_weight = edge_data.get("sample_weight")
                if sample_weight is not None:
                    sample_weight = _coerce_timestamp_array(sample_weight, dtype=np.float64).reshape(-1, 1)
                    processed[key] = {
                        "timestamps": timestamps,
                        "sample_weight": sample_weight,
                    }
                else:
                    processed[key] = timestamps
            else:
                # Simple array case
                processed[key] = _coerce_timestamp_array(edge_data, dtype=np.float64).reshape(-1, 1)
        return processed
    
    elif mode == "random":
        return random_sample_long_merged_edges(
            merged_edges,
            length_threshold=length_threshold,
            sample_length=sample_length,
            random_state=random_state,
        )
    elif mode == "kde":
        return kde_preprocess_long_merged_edges(
            merged_edges,
            length_threshold=length_threshold,
            sample_length=sample_length,
            kde_grid_size=kde_grid_size,
            kde_total_weight=kde_total_weight,
            kde_grid_padding=kde_grid_padding,
            bandwidth_grid_size=bandwidth_grid_size,
        )
    raise ValueError("mode must be 'none', 'random', or 'kde'.")

def prep_data(csv_path, target_nodes, args):
    merged_edges, _ = load_merged_edges(csv_path, target_nodes, args)
    merged_edges = filter_merged_edges(merged_edges, min_count=200)
    scaler_type = getattr(args, "scaler_type", "minmax")
    merged_edges, scalers = scale_merged_edges(merged_edges, scaler_type=scaler_type)
    return merged_edges, scalers

def merged_edges_to_matrix(merged_edges, batch_size=256, dtype=np.float64, pad_value=np.nan, sort_by_length=True):
    """
    Convert raw merged edge timestamps or weighted timestamp payloads into padded GPU batches.

    Parameters:
    merged_edges (dict): Mapping from edge key to timestamps shaped
        (n_samples, 1) / (n_samples,) or payload dicts with `timestamps`
        and optional `sample_weight`.
    batch_size (int): Maximum number of edges per batch.
    dtype: Output NumPy dtype for the matrices.
    pad_value: Fill value for padded positions.
    sort_by_length (bool): If True, sort edges by number of timestamps before
        batching so similarly sized edges are grouped together and padding is reduced.

    Returns:
    list[dict]: Batch dictionaries with keys `edge_keys`, `matrix`, `mask`,
        `sample_weight`, `lengths`, `batch_size`, `min_timestamps`, and `max_timestamps`.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    normalized_edges = []
    has_any_sample_weight = False

    for key, edge_data in merged_edges.items():
        if isinstance(edge_data, dict) and "timestamps" in edge_data:
            values = _coerce_timestamp_array(edge_data["timestamps"], dtype=dtype)
            sample_weight = edge_data.get("sample_weight")
            if sample_weight is not None:
                sample_weight = _coerce_timestamp_array(sample_weight, dtype=dtype)
                if sample_weight.shape[0] != values.shape[0]:
                    raise ValueError(f"sample_weight for edge {key} must match timestamps length.")
                has_any_sample_weight = True
        else:
            values = _coerce_timestamp_array(edge_data, dtype=dtype)
            sample_weight = None

        normalized_edges.append((key, values, sample_weight))

    if not normalized_edges:
        return []

    if sort_by_length:
        normalized_edges.sort(key=lambda item: (item[1].shape[0], str(item[0])))

    batches = []
    for start_idx in range(0, len(normalized_edges), batch_size):
        edge_group = normalized_edges[start_idx:start_idx + batch_size]
        edge_keys = [key for key, _, _ in edge_group]
        lengths = np.array([values.shape[0] for _, values, _ in edge_group], dtype=np.int64)
        max_samples = int(lengths.max())
        matrix = np.full((len(edge_group), max_samples), pad_value, dtype=dtype)
        mask = np.zeros((len(edge_group), max_samples), dtype=bool)
        sample_weight_matrix = None
        if has_any_sample_weight:
            sample_weight_matrix = np.zeros((len(edge_group), max_samples), dtype=dtype)

        for row_idx, (_, values, sample_weight) in enumerate(edge_group):
            length = values.shape[0]
            matrix[row_idx, :length] = values
            mask[row_idx, :length] = True
            if sample_weight_matrix is not None:
                if sample_weight is None:
                    sample_weight_matrix[row_idx, :length] = 1.0
                else:
                    sample_weight_matrix[row_idx, :length] = sample_weight

        batches.append(
            {
                "edge_keys": edge_keys,
                "matrix": matrix,
                "mask": mask,
                "sample_weight": sample_weight_matrix,
                "lengths": lengths,
                "batch_size": len(edge_group),
                "min_timestamps": int(lengths.min()),
                "max_timestamps": int(lengths.max()),
            }
        )

    return batches

def _gaussian_mixture_from_params(means, covariances, weights):
    means = np.asarray(means, dtype=np.float64).reshape(-1, 1)
    covariances = np.asarray(covariances, dtype=np.float64).reshape(-1, 1)
    covariances = np.clip(covariances, 1e-12, None)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights = weights / weights.sum()

    gmm = GaussianMixture(n_components=len(weights), covariance_type="diag")
    gmm.weights_ = weights
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.precisions_cholesky_ = 1.0 / np.sqrt(covariances)
    gmm.precisions_ = gmm.precisions_cholesky_ ** 2
    gmm.converged_ = True
    gmm.n_features_in_ = 1
    return gmm


def dpgmm_batches_to_gmms(batch_models, batches, truncate_threshold=0.99):
    """
    Convert batched BayesianGaussianMixtureGPU results into sklearn GaussianMixture objects.

    Parameters:
    batch_models (list): One fitted GPU model per batch returned by `merged_edges_to_matrix`.
    batches (list): Batch metadata from `merged_edges_to_matrix`.
    truncate_threshold (float): Weight mass kept when truncating tiny mixture components.

    Returns:
    dict: Mapping from edge key to sklearn GaussianMixture object.
    """
    if len(batch_models) != len(batches):
        raise ValueError("batch_models and batches must have the same length.")

    gmms = {}
    for batch_model, batch in zip(batch_models, batches):
        for row_idx, key in enumerate(batch["edge_keys"]):
            means = np.asarray(batch_model.means_[row_idx]).reshape(-1)
            covariances = np.asarray(batch_model.covariances_[row_idx]).reshape(-1)
            weights = np.asarray(batch_model.weights_[row_idx]).reshape(-1)
            means, covariances, weights = truncate_clusters(
                means=means,
                variances=covariances,
                weights=weights,
                threshold=truncate_threshold,
            )
            gmms[key] = _gaussian_mixture_from_params(means, covariances, weights)

    return gmms


def gmms_to_enhanced_edges(gmms, merged_edges, args):
    """
    Convert a dict of sklearn GaussianMixture objects back into EnhancedEdge objects.
    """
    enhanced_edges = {}
    for key, gmm in gmms.items():
        enhanced_edge = EnhancedEdge(u=key[0], v=key[1], syscall=key[2], args=args)
        enhanced_edge.timestamps = np.asarray(merged_edges[key]).reshape(-1, 1)
        enhanced_edge.count = len(enhanced_edge.timestamps)
        enhanced_edge.means = np.asarray(gmm.means_).reshape(-1)
        enhanced_edge.covariances = np.asarray(gmm.covariances_).reshape(-1)
        enhanced_edge.weights = np.asarray(gmm.weights_).reshape(-1)
        enhanced_edge.num_clusters = len(enhanced_edge.means)
        enhanced_edge.kde = gmm
        enhanced_edges[key] = enhanced_edge

    return enhanced_edges


def fit_batched_gpu_dpgmm(
    merged_edges,
    *,
    n_components=100,
    min_count=200,
    scaler_type="minmax",
    long_data_mode="none",
    long_data_threshold=50000,
    sample_length=50000,
    kde_grid_size=400,
    kde_total_weight=None,
    kde_grid_padding=0.10,
    batch_size=256,
    max_iter=100,
    random_state=42,
    device="cuda",
    dtype=None,
    truncate_threshold=0.99,
    sort_by_length=True,
    preprocessed=False
    ):
    """
    Fit batched 1D GPU-accelerated DPGMMs from a plain dict of edge timestamps.

    Parameters:
    merged_edges (dict): Mapping from edge key to timestamp sequences.
    n_components (int): Maximum number of mixture components per edge.
    min_count (int): Minimum number of timestamps required to keep an edge.
    scaler_type (str): One of 'minmax', 'standard', or 'none'.
    long_data_mode (str): One of 'none', 'random', or 'kde' for preprocessing before fitting.
    long_data_threshold (int): Length above which preprocessing is applied when
        `long_data_mode='random'` or `long_data_mode='kde'`. Set it to `0` to apply
        the chosen preprocessing mode to every edge.
    sample_length (int): Target size for random downsampling and default KDE weight mass.
    kde_grid_size (int): Number of KDE grid points when `long_data_mode='kde'`.
    kde_total_weight (float or None): Total sample-weight mass assigned to the KDE grid.
    kde_grid_padding (float): Relative padding added around the KDE support grid.
    batch_size (int): Maximum number of edges processed together per GPU batch.
    max_iter (int): Maximum VB iterations for each batch fit.
    random_state (int): Random seed used by the preprocessing and GPU mixture model.
    device (str): Torch device, typically 'cuda'.
    dtype: Optional torch dtype override.
    truncate_threshold (float): Weight mass kept after truncating tiny components.
    sort_by_length (bool): Sort edges by length before batching to reduce padding.

    Returns:
    dict: Mapping from edge key to sklearn GaussianMixture object.
    """

    if preprocessed:
        scaled_edges = merged_edges
    else:
        filtered_edges = filter_merged_edges(dict(merged_edges), min_count=min_count)
        scaled_edges, _ = scale_merged_edges(filtered_edges, scaler_type=scaler_type)
    
    
    preprocessed_edges = preprocess_long_merged_edges(
        scaled_edges,
        mode=long_data_mode,
        length_threshold=long_data_threshold,
        sample_length=sample_length,
        random_state=random_state,
        kde_grid_size=kde_grid_size,
        kde_total_weight=kde_total_weight,
        kde_grid_padding=kde_grid_padding,
    )
    batches = merged_edges_to_matrix(
        preprocessed_edges,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
    )

    batch_models = []
    model_kwargs = dict(
        n_components=n_components,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_process",
        max_iter=max_iter,
        random_state=random_state,
        device=device,
    )
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    for batch in tqdm(batches, total=len(batches), desc="Processing GPU DPGMM batches..."):
        model = BayesianGaussianMixtureGPU(**model_kwargs)
        model.fit(
            batch["matrix"],
            sample_weight=batch["sample_weight"],
            mask=batch["mask"],
        )
        batch_models.append(model)

    gmms = dpgmm_batches_to_gmms(
        batch_models,
        batches,
        truncate_threshold=truncate_threshold,
    )

    return gmms, scaled_edges

def gpu_batches_to_enhanced_edges(batch_models, batches, merged_edges, args, truncate_threshold=0.99):
    """
    Convert batched BayesianGaussianMixtureGPU results back into EnhancedEdge objects.

    Parameters:
    batch_models (list): One fitted GPU model per batch returned by `merged_edges_to_matrix`.
    batches (list): Batch metadata from `merged_edges_to_matrix`.
    merged_edges (dict): Original merged edge timestamps keyed by edge tuple.
    args: Experiment args used to construct EnhancedEdge objects.
    truncate_threshold (float): Weight mass kept when truncating tiny mixture components.

    Returns:
    dict: Mapping from edge key to populated EnhancedEdge object.
    """
    if len(batch_models) != len(batches):
        raise ValueError("batch_models and batches must have the same length.")

    enhanced_edges = {}
    for batch_model, batch in zip(batch_models, batches):
        for row_idx, key in enumerate(batch["edge_keys"]):
            means = np.asarray(batch_model.means_[row_idx]).reshape(-1)
            covariances = np.asarray(batch_model.covariances_[row_idx]).reshape(-1)
            weights = np.asarray(batch_model.weights_[row_idx]).reshape(-1)
            means, covariances, weights = truncate_clusters(
                means=means,
                variances=covariances,
                weights=weights,
                threshold=truncate_threshold,
            )

            enhanced_edge = EnhancedEdge(u=key[0], v=key[1], syscall=key[2], args=args)
            enhanced_edge.timestamps = np.asarray(merged_edges[key]).reshape(-1, 1)
            enhanced_edge.count = len(enhanced_edge.timestamps)
            enhanced_edge.means = means
            enhanced_edge.covariances = covariances
            enhanced_edge.weights = weights
            enhanced_edge.num_clusters = len(means)
            enhanced_edge.kde = _gaussian_mixture_from_params(means, covariances, weights)
            enhanced_edges[key] = enhanced_edge

    return enhanced_edges

def enhanced_edges_to_pickle_dict(name, enhanced_edges, scalers):
    edge_dict = {}
    for key, enhanced_edge in enhanced_edges.items():
        edge_dict[key] = {
            "kde": enhanced_edge.kde,
            "scaler": scalers[key],
            "means": enhanced_edge.means if enhanced_edge.means is not None else [],
            "covariances": enhanced_edge.covariances if enhanced_edge.covariances is not None else [],
            "weights": enhanced_edge.weights if enhanced_edge.weights is not None else [],
            "num_clusters": enhanced_edge.num_clusters if enhanced_edge.num_clusters is not None else 0
        }
    
    with open(f"{name}", "wb") as f:
        pickle.dump(edge_dict, f)

    return edge_dict


def evaluate_enhancement_storage(gmms, scaled_merged_edges):
    print("Evaluating enhancement...", flush=True)
    print(f"Number of edges: {len(gmms)}", flush=True)
    print(f"Type of gmms: {type(gmms)}", flush=True)
    print(f"Number of scaled edges: {len(scaled_merged_edges.keys())}", flush=True)
    
    raw_timestamps = []
    total_components = 0
    total_raw_timestamps = 0
    component_counts = []
    
    for key in gmms.keys():
        # Get timestamps for this edge
        edge_timestamps = scaled_merged_edges[key]
        raw_timestamps.append(edge_timestamps)
        
        # Count raw timestamps
        if isinstance(edge_timestamps, dict) and "timestamps" in edge_timestamps:
            num_timestamps = len(edge_timestamps["timestamps"])
        else:
            num_timestamps = len(edge_timestamps)
        
        total_raw_timestamps += num_timestamps
        
        # Count components from GMM
        gmm = gmms[key]
        num_components = len(gmm.weights_)
        total_components += num_components
        component_counts.append(num_components)
    
    print(f"Total number of raw timestamps: {total_raw_timestamps}", flush=True)
    print(f"Total number of DPGMM components: {total_components}", flush=True)
    print(f"Average components per edge: {np.mean(component_counts):.2f}", flush=True)
    print(f"Ratio of components to timestamps: {total_components / total_raw_timestamps:.4f}", flush=True)
    print(f"Min components per edge: {np.min(component_counts)}", flush=True)
    print(f"Max components per edge: {np.max(component_counts)}", flush=True)
    print(f"Median components per edge: {np.median(component_counts):.2f}", flush=True)

def evaluate_enhancement_fidelity(gmms, scaled_merged_edges, batch_size=256):
    nll_count = 0
    nll_sum = 0
    kl_count = 0
    kl_sum = 0
    raw_timestamps =[]
    for key in gmms.keys():
        edge_timestamps = scaled_merged_edges[key]
        raw_timestamps.append(edge_timestamps)
    
    batches = build_batched_gmm_inputs(list(gmms.values()), raw_timestamps, batch_size=batch_size)
    for batch in tqdm(batches, total=len(batches), desc="Evaluating enhanced edges..."):
        nll = compute_batched_nll_gpu(batch)
        actual_batch_size = len(nll)  # Use actual number of elements, not assumed batch_size
        nll_sum += nll.sum()
        nll_count += actual_batch_size

        kl = compute_batched_kl(batch)
        
        # # KL should always be finite (defaults to 0.0 on errors)
        assert np.all(np.isfinite(kl)), f"KL divergence contains unexpected values: {kl}"
        
        kl_sum += kl.sum()
        kl_count += actual_batch_size

    # Validate KL computation state (should never be NaN now)
    assert np.isfinite(kl_sum), f"KL sum is invalid: {kl_sum}"
    assert kl_count > 0, f"No edges processed: kl_count={kl_count}"

    # Calculate average scores after processing all batches
    average_nll = nll_sum / nll_count if nll_count > 0 else float('nan')
    average_kl = kl_sum / kl_count if kl_count > 0 else float('nan')

    print(f"Average Negative Log-Likelihood (NLL): {average_nll}", flush=True)
    print(f"Average KL Divergence: {average_kl}", flush=True)



def draw_gmm_distribution_over_data(
    gmms,
    scaled_merged_edges,
    scalers=None,
    edge_key=None,
    figsize=(12, 6),
    save_path=None,
    max_bins=1000,
):
    """
    Draw the distribution of a GMM model over the original data points as a histogram with GMM density overlay.
    
    Parameters:
    -----------
    gmms : dict
        Dictionary mapping edge keys to fitted Gaussian Mixture Models to visualize.
    scaled_merged_edges : dict
        Dictionary mapping edge keys to timestamp arrays. Can be either:
        - dict[key] = numpy array of shape (n_samples,) or (n_samples, 1)
        - dict[key] = {'timestamps': array, ...}
    scalers : dict, optional
        Dictionary mapping edge keys to sklearn scalers. If provided, scaled timestamps will be
        inverse transformed back to original time scale before computing inter-event times.
    edge_key : str or tuple, optional
        Specific edge key to visualize. If None, uses the first edge available.
    figsize : tuple, default=(12, 6)
        Figure size (width, height) in inches.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    max_bins : int, default=1000
        Maximum number of bins for the histogram (to prevent excessive binning).
    
    Example:
    --------
    >>> from sklearn.mixture import GaussianMixture
    >>> import numpy as np
    >>> data = np.random.randn(1000).reshape(-1, 1)
    >>> gmm = GaussianMixture(n_components=3).fit(data)
    >>> scaled_edges = {('a', 'b', 'sys'): data}

    >>> fig, ax = draw_gmm_distribution_over_data(gmms, scaled_edges, 
    ...                                             edge_key=('a', 'b', 'sys'))
    """
    # Determine which edge to visualize
    if edge_key is None:
        edge_key = list(scaled_merged_edges.keys())[0]
    
    # Extract data from edge
    edge_data = scaled_merged_edges[edge_key]
    if isinstance(edge_data, dict) and "timestamps" in edge_data:
        scaled_timestamps = np.asarray(edge_data["timestamps"], dtype=np.float64).reshape(-1, 1)
    else:
        scaled_timestamps = np.asarray(edge_data, dtype=np.float64).reshape(-1, 1)
    
    # Ensure timestamps are clean (remove any non-finite values)
    scaled_timestamps = scaled_timestamps[np.all(np.isfinite(scaled_timestamps), axis=1)]
    
    if len(scaled_timestamps) == 0:
        raise ValueError(f"Edge {edge_key} has no finite data points.")
    
    # Work in scaled space since GMM was fitted on scaled data
    data_scaled = scaled_timestamps.ravel()
    
    # Get scaler for optional x-axis label mapping
    scaler = None
    if scalers is not None and edge_key in scalers:
        scaler = scalers[edge_key]
    
    # Apply log transformation: log10(Δt + 1) for better visualization
    # This spreads out small values and compresses large outliers
    data_transformed = np.log10(data_scaled + 1.0)
    
    # Calculate number of bins using IQR-based method on transformed data
    iqr = np.subtract(*np.percentile(data_transformed, [75, 25]))
    bin_width = 2 * iqr / np.cbrt(len(data_transformed))
    
    # Handle case where bin_width is zero or invalid
    if bin_width <= 0 or not np.isfinite(bin_width):
        n_bins = min(int(np.ceil(np.log2(len(data_transformed)) + 1)), max_bins)
    else:
        n_bins = int(np.ceil((np.max(data_transformed) - np.min(data_transformed)) / bin_width))
        n_bins = min(n_bins, max_bins)  # Cap at max_bins to prevent excessive binning
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram with log-transformed data
    counts, bins, patches = ax.hist(
        data_transformed,
        bins=n_bins,
        density=True,
        alpha=0.6,
        color='lightblue',
        edgecolor='darkblue',
        label='Inter-arrival Time Frequency'
    )
    
    # Create evaluation grid in transformed space for proper density alignment
    x_min_transformed = data_transformed.min()
    x_max_transformed = data_transformed.max()
    x_range_transformed = x_max_transformed - x_min_transformed
    pad = 0.05 * x_range_transformed
    x_transformed_grid = np.linspace(x_min_transformed - pad, x_max_transformed + pad, 300)
    
    # Convert back to original scaled space for GMM evaluation
    x_scaled_grid = 10 ** x_transformed_grid - 1.0
    x_scaled_grid = x_scaled_grid.reshape(-1, 1)
    
    # Compute GMM density on the grid (in original scaled space)
    gmm = gmms[edge_key]
    density_original = np.exp(gmm.score_samples(x_scaled_grid))
    
    # Apply Jacobian correction for the log transformation
    # If y = log10(x + 1), then dy/dx = 1/((x+1)*ln(10))
    # So the density in transformed space is: p_y(y) = p_x(x) * |dx/dy|
    # where |dx/dy| = (x+1)*ln(10) = (10^y)*ln(10)
    jacobian = (10 ** x_transformed_grid) * np.log(10)
    density_transformed = density_original.ravel() * jacobian.ravel()
    
    ax.plot(x_transformed_grid, density_transformed, 'orange', linewidth=2.5, label='Timing Distribution Function')
    
    # Set x-axis label with the log transformation shown
    ax.set_xlabel(r'$\log_{10}(\Delta t + 1)$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        f'Timing Distribution Function over Inter-arrival Times\n',
        # f'Edge: {edge_key} | n_samples: {len(data)} | n_components: {gmm.n_components}',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save figure if path is provided
    if save_path is not None:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"Figure saved to {save_path}", flush=True)
    


def find_multimodal_edges(
    gmms,
    scaled_merged_edges,
    min_components=2,
    weight_threshold=0.05,
    top_n=None,
    sort_by="entropy",
):
    """
    Find and rank multimodal edges based on component distribution characteristics.
    
    An edge is considered multimodal if it has multiple components with significant weights.
    This helps identify edges with complex, multi-peaked distributions worth visualizing.
    Edges are prioritized by the amount of data they contain.
    
    Parameters:
    -----------
    gmms : dict
        Dictionary mapping edge keys to sklearn GaussianMixture objects.
    scaled_merged_edges : dict
        Dictionary mapping edge keys to timestamp arrays (for counting data points).
    min_components : int, default=2
        Minimum number of components for an edge to be considered multimodal.
    weight_threshold : float, default=0.05
        Minimum weight for a component to count as "significant" (0.0 to 1.0).
        Components below this threshold are ignored. Set to 0.0 to count all components.
    top_n : int, optional
        If provided, return only the top N edges. If None, return all multimodal edges.
    sort_by : str, default="n_timestamps"
        Sorting criterion. Options:
        - "n_timestamps": Sort by number of data points (more data = higher priority)
        - "entropy": Sort by entropy of component weights (higher entropy = more balanced)
        - "n_components": Sort by number of significant components
        - "variance": Sort by mean standard deviation of components
    
    Returns:
    --------
    list : List of tuples (edge_key, score_dict) sorted by the specified criterion.
           score_dict contains: n_components, entropy, max_weight, mean_std, n_timestamps, all_weights
    
    Example:
    --------
    >>> multimodal_edges = find_multimodal_edges(gmms, scaled_edges, min_components=2, top_n=10)
    >>> for edge_key, scores in multimodal_edges[:3]:
    ...     print(f"Edge {edge_key}: {scores['n_timestamps']} data points, {scores['n_components']} components")
    ...     fig, ax = draw_gmm_distribution_over_data(gmms[edge_key], scaled_edges, edge_key=edge_key)
    """
    if sort_by not in ["n_timestamps", "entropy", "n_components", "variance"]:
        raise ValueError("sort_by must be one of: 'n_timestamps', 'entropy', 'n_components', 'variance'")
    
    multimodal_edges = []
    
    for edge_key, gmm in gmms.items():
        # Get component weights and filter by threshold
        weights = np.asarray(gmm.weights_, dtype=np.float64)
        significant_weights = weights[weights >= weight_threshold]
        n_significant = len(significant_weights)
        
        # Check if edge is multimodal
        if n_significant < min_components:
            continue
        
        # Count number of timestamps for this edge
        edge_data = scaled_merged_edges[edge_key]
        if isinstance(edge_data, dict) and "timestamps" in edge_data:
            n_timestamps = len(edge_data["timestamps"])
        else:
            n_timestamps = len(edge_data)
        
        # Compute entropy of weights (measure of balance/uniformity)
        # Higher entropy = more balanced components (more multimodal)
        # Entropy = -sum(p * log(p))
        weight_sum = significant_weights.sum()
        if weight_sum > 0:
            normalized_weights = significant_weights / weight_sum
            entropy = -np.sum(normalized_weights * np.log(normalized_weights + 1e-10))
        else:
            entropy = 0.0
        
        # Compute other statistics
        max_weight = weights.max()
        
        # Compute mean standard deviation of components
        covariances = np.asarray(gmm.covariances_, dtype=np.float64).ravel()
        stds = np.sqrt(covariances)
        mean_std = np.mean(stds)
        
        score_dict = {
            "n_components": gmm.n_components,
            "n_significant": n_significant,
            "n_timestamps": n_timestamps,
            "entropy": float(entropy),
            "max_weight": float(max_weight),
            "mean_std": float(mean_std),
            "all_weights": weights,
        }
        
        multimodal_edges.append((edge_key, score_dict))
    
    # Sort by specified criterion
    if sort_by == "n_timestamps":
        # More data points = higher priority
        multimodal_edges.sort(key=lambda x: x[1]["n_timestamps"], reverse=True)
    elif sort_by == "entropy":
        # Higher entropy = more interesting (more balanced multimodal)
        multimodal_edges.sort(key=lambda x: x[1]["entropy"], reverse=True)
    elif sort_by == "n_components":
        # More components = more complex
        multimodal_edges.sort(key=lambda x: x[1]["n_components"], reverse=True)
    elif sort_by == "variance":
        # Larger variance = more spread out
        multimodal_edges.sort(key=lambda x: x[1]["mean_std"], reverse=True)
    
    # Limit to top N if requested
    if top_n is not None:
        multimodal_edges = multimodal_edges[:top_n]
    
    return multimodal_edges


# def evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args):

#     csv_name = csv_path.split("/")[-1].split(".")[0]
        
#     technique_name = ""
#     if args.use_dpgmm:
#         technique_name = "dpgmm"
#     elif args.use_kmeans:
#         technique_name = "kmeans"
#     elif args.use_dbstream:
#         technique_name = "dbstream"
#     else:
#         technique_name = "kde"

#     technique = {
#         technique_name: {
#             "neg_average_log_likelihood": [],
#             "integrated_square_error": [],
#             "mean_squared_error": [],
#             "kl_divergence": [],
#             "wasserstein_distance": [],
#             "timestamp_count": [],
#             "num_clusters": [],
#             "raw_timestamp_count": []
#         }
#     }

#     visualization_count = 0
    
#     for key, edge in tqdm(enhanced_edges.items(), total=len(enhanced_edges.values()), desc="evaluating..."):
#         all_timestamps = merged_edges[key]
#         # print("all timstamps shape: ", all_timestamps.shape, flush=True)
    
#         neg_log_likelihood = edge.compute_average_log_likelihood(all_timestamps)
#         kl_divergence = edge.compute_kl_divergence(all_timestamps)
#         # wasserstein_dist = edge.compute_wasserstein_distance(all_timestamps)
#         # integrated_square_error = edge.compute_integrated_square_error(all_timestamps)

#         # if neg_log_likelihood is None or kl_divergence is None:
#         #     continue

#         technique[technique_name]["neg_average_log_likelihood"].append(neg_log_likelihood)
#         technique[technique_name]["kl_divergence"].append(kl_divergence)
#         # technique[technique_name]["wasserstein_distance"].append(wasserstein_dist)
#         technique[technique_name]["raw_timestamp_count"].append(len(all_timestamps))
#         technique[technique_name]["num_clusters"].append(edge.num_clusters)
#         # technique[technique_name]["integrated_square_error"].append(integrated_square_error)
#         if args.visualize and visualization_count < 10:
#             visualization_count += 1
#             print("Got inside visualization for edge: ", flush=True)
#             visualization_folder = f"./kde_visualizations/{csv_name}_{technique_name}"
#             os.makedirs(f"{visualization_folder}", exist_ok=True)
#             edge.visualize_distribution(timestamps=all_timestamps, name=f"{visualization_folder}/{edge.u}_{edge.v}.png")

#         # technique[technique_name]["timestamp_count"].append(len(edge.timestamps))

#     average_likelihood = average_of_edges(technique[technique_name]["neg_average_log_likelihood"])
#     average_num_clusters = average_of_edges(technique[technique_name]["num_clusters"])
#     average_kl_divergence = average_of_edges(technique[technique_name]["kl_divergence"])
#     # average_wasserstein_distance = average_of_edges(technique[technique_name]["wasserstein_distance"])
#     # average_integrated_square_error = average_of_edges(technique[technique_name]["integrated_square_error"])

#     print(f"=== Results for {csv_name} using {technique_name} ===", flush=True)
#     print(f"{technique_name} average likelihood: ", average_likelihood, flush=True)
#     print(f"{technique_name} average kl divergence: ", average_kl_divergence, flush=True)
#     print(f"{technique_name} average num clusters: ", average_num_clusters, flush=True)
#     print(f"{technique_name} total num clusters: ", sum(technique[technique_name]["num_clusters"]), flush=True)
#     print(f"{technique_name} total raw timestamps: ", sum(technique[technique_name]["raw_timestamp_count"]), flush=True)
    # print(f"{technique_name} average wasserstein distance: ", average_wasserstein_distance, flush=True)
    # print(f"{technique_name} integrated square error: ", average_integrated_square_error, flush=True)



# def reduce(dataset_name, max_components=100, grid_points=400, total_count=2000, method="scan_gmm"):
        
#     dataset_path = f"./datasets/{dataset_name}/{method}.pkl"
#     if os.path.exists(dataset_path):
#         print(f"Loading preprocessed dataset from {dataset_path}", flush=True)
#         with open(dataset_path, "rb") as f:
#             enhanced_edges = pickle.load(f)
    
#         return enhanced_edges
    
#     dataset_path = f"./datasets/{dataset_name}/*.csv"

#     args = {}
#     args.sys_call = True
#     args.method = method
#     args.zero_center = False
#     args.detection = False
#     args.visualize = False
#     args.K = max_components
#     args.grid_points = grid_points
#     args.total_count = total_count

#     merged_edges, scalers = prep_data(dataset_path, None, args)
#     enhanced_edges = fit_data(merged_edges, args)

#     save_path = f"./datasets/{dataset_name}/{method}.pkl"
#     edges_dict = enhanced_edges_to_pickle_dict(save_path, enhanced_edges, scalers)
    
#     return edges_dict

# def split_data(merged_edges, k=20000):
#     """
#     Split the values of each dictionary item into chunks of approximately K data points,
#     and return a list of dictionaries, each containing chunks for the same keys.

#     Parameters:
#     merged_edges (dict): Dictionary with keys and array values to split.
#     k (int): Number of data points per chunk.

#     Returns:
#     list of dict: A list of dictionaries, each containing chunks of data for the same keys.
#     """
#     # Initialize a list to hold the split dictionaries
#     split_dicts = []

#     # Iterate over each key-value pair in the dictionary
#     for key, values in merged_edges.items():
#         # Convert values to a NumPy array if not already
#         values = np.array(values)

#         if len(values) < 2000:
#             k = len(values)  # If fewer than 2000 points, use all points as a single chunk
#         else:
#             # k = max(1, len(values) // 4)  # Ensure k is at least 1
#             k = 40000
        
#         # Split the values into chunks of size K
#         chunks = [values[i:i + k] for i in range(0, len(values), k)]
        
#         # Ensure there are enough dictionaries in the split_dicts list
#         while len(split_dicts) < len(chunks):
#             split_dicts.append({})

#         # Assign each chunk to the corresponding dictionary in split_dicts
#         for i, chunk in enumerate(chunks):
#             split_dicts[i][key] = chunk

#     # Print split statistics for debugging
#     for i, split in enumerate(split_dicts):
#         total_points = sum(len(values) for values in split.values())
#         print(f"Split {i + 1}: {len(split)} keys, {total_points} total points", flush=True)

#     return split_dicts

# def split_data_cic_ids(merged_edges, scaler, args):
#     time_ranges = [
#         (1499054400, 1499140799),  # Monday
#         (1499140800, 1499227199),  # Tuesday
#         (1499227200, 1499313599),  # Wednesday
#         (1499400000, 1499486399)   # Friday
#     ]

#     scaled_time_ranges = [
#         (scaler.transform([[start]])[0][0], scaler.transform([[end]])[0][0])
#         for start, end in time_ranges
#     ]

#     # Initialize a list to hold the split dictionaries
#     split_dicts = [{} for _ in range(len(scaled_time_ranges))]

#     # Iterate over each key-value pair in the dictionary
#     for key, timestamps in merged_edges.items():
#         for i, (start, end) in enumerate(scaled_time_ranges):
#             # Filter timestamps that fall within the current time range
#             split_dicts[i][key] = np.array(timestamps)[(start <= np.array(timestamps)) & (np.array(timestamps) <= end)]
#             split_dicts[i][key] = split_dicts[i][key].reshape(-1, 1)  # Reshape to 2D array 
#     return split_dicts


# def process_splits(splits, args):
#     enhanced_edges = {}

#     # Step 1: Fit each split individually
#     split_enhanced_edges = []
#     for index, split in tqdm(enumerate(splits), total=len(splits), desc="Fitting individual splits..."):
#         split_edges = {}
#         for (node1_id, node2_id, syscall), timestamps in tqdm(split.items(), total=len(split), desc=f"Fitting split {index}..."):
#             args.K = 20  # Adjust K for the initial fit
#             enhanced_edge = EnhancedEdge(u=node1_id, v=node2_id, syscall=syscall, args=args)
#             enhanced_edge.fit(timestamps=timestamps, args=args, data_weights=None)
#             split_edges[(node1_id, node2_id, syscall)] = enhanced_edge
#         split_enhanced_edges.append(split_edges)

#         evaluate_enhancement(f"split_{index}", split_edges, split, args)

#  # Step 2: Extract GH3 pseudo-points from all splits
#     combined_gh3_points = {}
#     combined_gh3_weights = {}
#     for split_edges in split_enhanced_edges:
#         for key, edge in split_edges.items():
#             if key not in combined_gh3_points:
#                 combined_gh3_points[key] = []
#                 combined_gh3_weights[key] = []
#             gh3_points, gh3_weights = edge.gh3_pseudo_points()
#             combined_gh3_points[key].append(gh3_points)
#             combined_gh3_weights[key].append(gh3_weights)


#     # Step 3: Combine GH3 pseudo-points and fit a unified distribution
#     for key in combined_gh3_points.keys():
#         # Combine GH3 pseudo-points and weights for this edge
#         all_points = np.concatenate(combined_gh3_points[key], axis=0)
#         all_weights = np.concatenate(combined_gh3_weights[key], axis=0)

#         # Fit a single unified distribution for this edge
#         args.K = 100
#         enhanced_edge = EnhancedEdge(u=key[0], v=key[1], syscall=key[2], args=args)
#         enhanced_edge.fit(timestamps=all_points, args=args, data_weights=all_weights)
#         enhanced_edges[key] = enhanced_edge


#     return enhanced_edges

# def set_adversarial_edge_timestamps(target_start_time, target_end_time, merged_edges, scalers):
#     attacker_victim_pair = ("172.16.0.1_same", "192.168.10.50_same", "same")
#     target_start_timestamp = int(datetime.strptime(target_start_time, '%Y-%m-%d %H:%M:%S').timestamp())
#     target_end_timestamp = int(datetime.strptime(target_end_time, '%Y-%m-%d %H:%M:%S').timestamp())
#     # Ensure scalers are in the same order as merged_edges
#     for (key, timestamps), scaler in zip(merged_edges.items(), scalers):
#         # Inverse transform the scaled timestamps to get the original timestamps
#         original_timestamps = scaler.inverse_transform(np.array(timestamps).reshape(-1, 1)).flatten()

#         if attacker_victim_pair == key:
#             print(f"Shifting timestamps for edge: {attacker_victim_pair}", flush=True)

#             # Replace the earliest and latest timestamps with the target values
#             shifted_timestamps = np.copy(original_timestamps)
#             shifted_timestamps[np.argmin(original_timestamps)] = target_start_timestamp
#             shifted_timestamps[np.argmax(original_timestamps)] = target_end_timestamp


#             # Transform the updated timestamps back to the scaled range
#             scaled_shifted_timestamps = scaler.transform(shifted_timestamps.reshape(-1, 1))

#             # Update the timestamps in merged_edges
#             merged_edges[attacker_victim_pair] = scaled_shifted_timestamps

#             # Convert the shifted timestamps to human-readable format for verification
#             start_time_human = datetime.fromtimestamp(min(shifted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
#             end_time_human = datetime.fromtimestamp(max(shifted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
#             print(f"Shifted Edge: {attacker_victim_pair}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)
#         else:
#             # For all other edges, just print the human-readable timestamps
#             start_time_human = datetime.fromtimestamp(min(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
#             end_time_human = datetime.fromtimestamp(max(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
#             print(f"Edge: {key}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)


def _fit_dbstream_edge(edge_info, args):
    """
    Worker function to fit DBSTREAM for a single edge.
    
    Parameters:
    edge_info (tuple): (edge_key, timestamps) where edge_key is (u, v, syscall)
    args: Experiment args containing DBSTREAM hyperparameters (unused but kept for compatibility)
    
    Returns:
    tuple: (edge_key, gmm) where gmm is a sklearn GaussianMixture object, or (edge_key, None) if fitting fails
    """
    edge_key, timestamps = edge_info
    
    try:
        # Extract timestamps as 1D array
        timestamps_1d = _coerce_timestamp_array(timestamps, dtype=np.float64)
        
        if len(timestamps_1d) < 2:
            print(f"Edge {edge_key}: Insufficient data (n={len(timestamps_1d)})", flush=True)
            return edge_key, None
        
        # Compute bandwidth for DBSTREAM parameter tuning
        bandwidth = _sheather_jones_bandwidth_1d(timestamps_1d)
        
        # Tune DBSTREAM parameters
        clustering_threshold, fading_factor, cleanup_interval, intersection_factor, minimum_weight, fallback_variances = tune_dbstream_params(
            timestamps_1d, 
            bandwidth=bandwidth
        )
        
        # Initialize DBSTREAM clustering
        dbstream = OnlineDBStreamClustering(
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight,
            fallback_variances=fallback_variances,
        )
        
        # Learn from timestamps
        for timestamp in timestamps_1d:
            dbstream.learn_data(timestamp)
        
        # Predict for final clustering
        for timestamp in timestamps_1d:
            dbstream.predict_data(timestamp)
        
        # Extract GMM parameters from DBSTREAM
        means, covs, weights, gmm = dbstream.get_gmm_params()
        
        if gmm is None:
            print(f"Edge {edge_key}: GMM is None after fitting DBSTREAM", flush=True)
            return edge_key, None
        
        print(f"Edge {edge_key}: Successfully fitted DBSTREAM with {len(means)} clusters", flush=True)
        return edge_key, gmm
        
    except Exception as e:
        print(f"Edge {edge_key}: Error during DBSTREAM fitting - {str(e)}", flush=True)
        return edge_key, None


def fit_batched_dbstream(
    merged_edges,
    *,
    min_count=200,
    scaler_type="minmax",
    long_data_mode="none",
    long_data_threshold=50000,
    sample_length=50000,
    kde_grid_size=400,
    kde_total_weight=None,
    kde_grid_padding=0.10,
    max_workers=4,
    args=None,
):
    """
    Fit DBSTREAM clustering for multiple edges in parallel.
    
    Parameters:
    merged_edges (dict): Mapping from edge key (u, v, syscall) to timestamp sequences.
    min_count (int): Minimum number of timestamps required to keep an edge.
    scaler_type (str): One of 'minmax', 'standard', or 'none'.
    long_data_mode (str): One of 'none', 'random', or 'kde' for preprocessing before fitting.
    long_data_threshold (int): Length above which preprocessing is applied.
    sample_length (int): Target size for random downsampling and default KDE weight mass.
    kde_grid_size (int): Number of KDE grid points when `long_data_mode='kde'`.
    kde_total_weight (float or None): Total sample-weight mass assigned to the KDE grid.
    kde_grid_padding (float): Relative padding added around the KDE support grid.
    max_workers (int): Maximum number of parallel workers for edge fitting.
    args: Experiment args (currently unused but kept for API compatibility).
    
    Returns:
    tuple: (gmms_dict, scaled_edges_dict) where gmms_dict maps edge keys to sklearn 
           GaussianMixture objects and scaled_edges_dict maps edge keys to scaled timestamps.
           Matches the return format of fit_batched_gpu_dpgmm.
    """
    
    # Filter edges by minimum count
    filtered_edges = filter_merged_edges(dict(merged_edges), min_count=min_count)
    
    # Scale edges
    scaled_edges, _ = scale_merged_edges(filtered_edges, scaler_type=scaler_type)
    
    # Preprocess long edges if needed
    preprocessed_edges = preprocess_long_merged_edges(
        scaled_edges,
        mode=long_data_mode,
        length_threshold=long_data_threshold,
        sample_length=sample_length,
        kde_grid_size=kde_grid_size,
        kde_total_weight=kde_total_weight,
        kde_grid_padding=kde_grid_padding,
    )
    
    # Extract timestamps for parallel processing
    edge_info_list = []
    for key, edge_data in preprocessed_edges.items():
        if isinstance(edge_data, dict) and "timestamps" in edge_data:
            timestamps = edge_data["timestamps"]
        else:
            timestamps = edge_data
        edge_info_list.append((key, timestamps))
    
    gmms = {}
    
    # Parallel fitting of edges using ThreadPoolExecutor
    print(f"Starting parallel DBSTREAM fitting with {max_workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_edge = {
            executor.submit(_fit_dbstream_edge, edge_info, args): edge_info[0]
            for edge_info in edge_info_list
        }
        
        # Process completed tasks as they finish
        for future in tqdm(as_completed(future_to_edge), total=len(future_to_edge), desc="Fitting DBSTREAM edges in parallel..."):
            edge_key = future_to_edge[future]
            try:
                edge_key_result, gmm = future.result()
                if gmm is not None:
                    gmms[edge_key_result] = gmm
            except Exception as e:
                print(f"Edge {edge_key}: Failed to fit DBSTREAM - {str(e)}", flush=True)
    
    print(f"Parallel DBSTREAM fitting completed. Fitted {len(gmms)} out of {len(edge_info_list)} edges.", flush=True)
    
    return gmms, scaled_edges

