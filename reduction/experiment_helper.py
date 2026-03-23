from create_graph_memory import fit_edges_memory, filter_edges_by_count, create_and_merge_graph, average_of_edges
from tqdm import tqdm
from enhanced_edge import EnhancedEdge
from BayesianGaussianMixtureGPU import BayesianGaussianMixtureGPU
from pyro_dpgmm import truncate_clusters
from datetime import datetime

import glob
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.stats import norm
from scipy.integrate import quad
import numpy as np
import pickle
import os

def reduce_helper(csv_path, target_nodes, args):
    
    technique_name = ""
    if args.use_dpgmm:
        technique_name = "dpgmm"
    elif args.use_kmeans:
        technique_name = "kmeans"
    elif args.use_dbstream:
        technique_name = "dbstream"
    else:
        technique_name = "kde"


    technique = {
        technique_name: {
            "neg_average_log_likelihood": [],
            "integrated_square_error": [],
            "mean_squared_error": [],
            "kl_divergence": [],
            "timestamp_count": [],
            "num_clusters": []
        }
    }

    min_edge_count = 200
    

    files = glob.glob(csv_path)

    merged_edges, total_edge_count = create_and_merge_graph(files, target_nodes, args)
    print("Total edge count: ", total_edge_count, flush=True)
    print("Num of merged edges: ", len(merged_edges), flush=True)        
    
    merged_edges = filter_edges_by_count(merged_edges, min_edge_count)
    print("Num of merged edges: ", len(merged_edges), flush=True)
    fit_edges_memory(merged_edges, args)

    for edge in merged_edges.values():
        neg_log_likelihood = edge.compute_average_log_likelihood()
        integrated_square_error = edge.compute_integrated_square_error()

        if neg_log_likelihood is None or integrated_square_error is None:
            continue

        technique[technique_name]["neg_average_log_likelihood"].append(edge.compute_average_log_likelihood())
        technique[technique_name]["integrated_square_error"].append(edge.compute_integrated_square_error())
        technique[technique_name]["kl_divergence"].append(edge.compute_kl_divergence())
        technique[technique_name]["num_clusters"].append(edge.num_clusters)
        technique[technique_name]["timestamp_count"].append(len(edge.timestamps))
        edge.visualize_distribution(name=f"{technique_name}_{csv_path.split('/')[-1].split('.')[0]}")

    average_likelihood = average_of_edges(technique[technique_name]["neg_average_log_likelihood"])
    average_integrated_square_error = average_of_edges(technique[technique_name]["integrated_square_error"])
    average_num_clusters = average_of_edges(technique[technique_name]["num_clusters"])
    average_kl_divergence = average_of_edges(technique[technique_name]["kl_divergence"])

    print(f"=== Results for {csv_path} using {technique_name} ===", flush=True)
    print(f"{technique_name} average likelihood: ", average_likelihood, flush=True)
    print(f"{technique_name} integrated square error: ", average_integrated_square_error, flush=True)
    print(f"{technique_name} average kl divergence: ", average_kl_divergence, flush=True)
    print(f"{technique_name} average num clusters: ", average_num_clusters, flush=True)

def reduce(dataset_name, max_components=100, grid_points=400, total_count=2000, method="scan_gmm"):
        
    dataset_path = f"./datasets/{dataset_name}/{method}.pkl"
    if os.path.exists(dataset_path):
        print(f"Loading preprocessed dataset from {dataset_path}", flush=True)
        with open(dataset_path, "rb") as f:
            enhanced_edges = pickle.load(f)
    
        return enhanced_edges
    
    dataset_path = f"./datasets/{dataset_name}/*.csv"

    args = {}
    args.sys_call = True
    args.method = method
    args.zero_center = False
    args.detection = False
    args.visualize = False
    args.K = max_components
    args.grid_points = grid_points
    args.total_count = total_count

    merged_edges, scalers = prep_data(dataset_path, None, args)
    enhanced_edges = fit_data(merged_edges, args)

    save_path = f"./datasets/{dataset_name}/{method}.pkl"
    edges_dict = enhanced_edges_to_pickle_dict(save_path, enhanced_edges, scalers)
    
    return edges_dict

    
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
        "none": None,
    }
    if scaler_name not in scaler_map:
        raise ValueError("scaler_type must be 'minmax', 'standard', or 'none'.")

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


def _coerce_timestamp_array(values, dtype=np.float64):
    values = np.asarray(values, dtype=dtype)
    if values.ndim == 2:
        if values.shape[1] != 1:
            raise ValueError(f"Expected timestamps with shape (n_samples, 1), got {values.shape}.")
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError(f"Expected timestamps to be 1D or 2D with one column, got {values.shape}.")
    return values


def _sheather_jones_bandwidth_1d(data, grid_size=1024):
    """
    Compute a Sheather-Jones-style bandwidth using adaptive quadrature.

    This is the older project implementation. `grid_size` is kept only for
    signature compatibility and is not used.
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


# def random_sample_long_merged_edges(
#     merged_edges,
#     *,
#     length_threshold=50000,
#     sample_length=50000,
#     random_state=42,
# ):
#     """
#     Randomly downsample only the edges longer than `length_threshold`.
#     """
#     print("Random sample preprocess", flush=True)
#     _validate_long_preprocessing_args(
#         length_threshold=length_threshold,
#         sample_length=sample_length,
#         kde_grid_size=2,
#         kde_total_weight=1.0,
#     )

#     rng = np.random.default_rng(random_state)
#     processed_edges = {}
#     unchanged_count = 0
#     sampled_count = 0

#     for key, timestamps in tqdm(
#         merged_edges.items(),
#         total=len(merged_edges),
#         desc="Random sampling edges",
#     ):
#         values = _coerce_timestamp_array(timestamps, dtype=np.float64)
#         values = values[np.isfinite(values)]
#         original_length = values.shape[0]

#         if original_length == 0:
#             raise ValueError(f"Edge {key} has no finite timestamps after preprocessing.")

#         if original_length <= length_threshold:
#             processed_edges[key] = values.reshape(-1, 1)
#             unchanged_count += 1
#             continue

#         target_length = min(sample_length, original_length)
#         sampled_idx = rng.choice(original_length, size=target_length, replace=False)
#         sampled_values = np.sort(values[sampled_idx])
#         processed_edges[key] = sampled_values.reshape(-1, 1)
#         sampled_count += 1

#     print(
#         "Long-series random sampling summary: "
#         f"unchanged={unchanged_count}, random_sampled={sampled_count}",
#         flush=True,
#     )
#     return processed_edges


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

    for key, timestamps in tqdm(
        merged_edges.items(),
        total=len(merged_edges),
        desc="KDE preprocessing edges",
    ):
        values_2d = np.asarray(timestamps, dtype=np.float64).reshape(-1, 1)
        original_length = values_2d.shape[0]

        if original_length <= length_threshold:
            processed_edges[key] = values_2d
            unchanged_count += 1
            continue

        values = values_2d[:, 0]
        x_min = float(values.min())
        x_max = float(values.max())
        bandwidth = _sheather_jones_bandwidth_1d(values, grid_size=bandwidth_grid_size)
        if bandwidth is None or not np.isfinite(bandwidth) or bandwidth <= 0:
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
            processed_edges[key] = values_2d
            fallback_count += 1
            continue

        sample_weight = (kde_total_weight / density_sum) * density
        processed_edges[key] = {
            "timestamps": grid,
            "sample_weight": sample_weight[:, None],
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
        return {
            key: _coerce_timestamp_array(timestamps, dtype=np.float64).reshape(-1, 1)
            for key, timestamps in merged_edges.items()
        }
    
    # elif mode == "random":
    #     return random_sample_long_merged_edges(
    #         merged_edges,
    #         length_threshold=length_threshold,
    #         sample_length=sample_length,
    #         random_state=random_state,
    #     )
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


def gpu_batches_to_gmms(batch_models, batches, truncate_threshold=0.99):
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
    max_iter=300,
    random_state=42,
    device="cuda",
    dtype=None,
    truncate_threshold=0.99,
    sort_by_length=True,
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

    gmms = gpu_batches_to_gmms(
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

def fit_data(merged_edges, args):
    """
    Fit the merged edges to the EnhancedEdge model.

    Parameters:
    merged_edges (dict): Dictionary with keys as tuples of node IDs and syscall, and values as lists of timestamps.
    args: Arguments containing configuration for the fitting process.

    Returns:
    dict: A dictionary of EnhancedEdge objects fitted with the provided timestamps.
    """
    enhanced_edges = {}
    
    for (node1_id, node2_id, syscall), timestamps in tqdm(merged_edges.items(), total=len(merged_edges), desc="Fitting enhanced edges..."):
        enhanced_edge = EnhancedEdge(u=node1_id, v=node2_id, syscall=syscall, args=args)
        enhanced_edge.fit(timestamps=timestamps, args=args, data_weights=None)
        enhanced_edges[(node1_id, node2_id, syscall)] = enhanced_edge

    print("Number of enhanced edges after fitting: ", len(enhanced_edges), flush=True)

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
        
def split_data(merged_edges, k=20000):
    """
    Split the values of each dictionary item into chunks of approximately K data points,
    and return a list of dictionaries, each containing chunks for the same keys.

    Parameters:
    merged_edges (dict): Dictionary with keys and array values to split.
    k (int): Number of data points per chunk.

    Returns:
    list of dict: A list of dictionaries, each containing chunks of data for the same keys.
    """
    # Initialize a list to hold the split dictionaries
    split_dicts = []

    # Iterate over each key-value pair in the dictionary
    for key, values in merged_edges.items():
        # Convert values to a NumPy array if not already
        values = np.array(values)

        if len(values) < 2000:
            k = len(values)  # If fewer than 2000 points, use all points as a single chunk
        else:
            # k = max(1, len(values) // 4)  # Ensure k is at least 1
            k = 40000
        
        # Split the values into chunks of size K
        chunks = [values[i:i + k] for i in range(0, len(values), k)]
        
        # Ensure there are enough dictionaries in the split_dicts list
        while len(split_dicts) < len(chunks):
            split_dicts.append({})

        # Assign each chunk to the corresponding dictionary in split_dicts
        for i, chunk in enumerate(chunks):
            split_dicts[i][key] = chunk

    # Print split statistics for debugging
    for i, split in enumerate(split_dicts):
        total_points = sum(len(values) for values in split.values())
        print(f"Split {i + 1}: {len(split)} keys, {total_points} total points", flush=True)

    return split_dicts

def split_data_cic_ids(merged_edges, scaler, args):
    time_ranges = [
        (1499054400, 1499140799),  # Monday
        (1499140800, 1499227199),  # Tuesday
        (1499227200, 1499313599),  # Wednesday
        (1499400000, 1499486399)   # Friday
    ]

    scaled_time_ranges = [
        (scaler.transform([[start]])[0][0], scaler.transform([[end]])[0][0])
        for start, end in time_ranges
    ]

    # Initialize a list to hold the split dictionaries
    split_dicts = [{} for _ in range(len(scaled_time_ranges))]

    # Iterate over each key-value pair in the dictionary
    for key, timestamps in merged_edges.items():
        for i, (start, end) in enumerate(scaled_time_ranges):
            # Filter timestamps that fall within the current time range
            split_dicts[i][key] = np.array(timestamps)[(start <= np.array(timestamps)) & (np.array(timestamps) <= end)]
            split_dicts[i][key] = split_dicts[i][key].reshape(-1, 1)  # Reshape to 2D array 
    return split_dicts


def process_splits(splits, args):
    enhanced_edges = {}

    # Step 1: Fit each split individually
    split_enhanced_edges = []
    for index, split in tqdm(enumerate(splits), total=len(splits), desc="Fitting individual splits..."):
        split_edges = {}
        for (node1_id, node2_id, syscall), timestamps in tqdm(split.items(), total=len(split), desc=f"Fitting split {index}..."):
            args.K = 20  # Adjust K for the initial fit
            enhanced_edge = EnhancedEdge(u=node1_id, v=node2_id, syscall=syscall, args=args)
            enhanced_edge.fit(timestamps=timestamps, args=args, data_weights=None)
            split_edges[(node1_id, node2_id, syscall)] = enhanced_edge
        split_enhanced_edges.append(split_edges)

        evaluate_enhancement(f"split_{index}", split_edges, split, args)

 # Step 2: Extract GH3 pseudo-points from all splits
    combined_gh3_points = {}
    combined_gh3_weights = {}
    for split_edges in split_enhanced_edges:
        for key, edge in split_edges.items():
            if key not in combined_gh3_points:
                combined_gh3_points[key] = []
                combined_gh3_weights[key] = []
            gh3_points, gh3_weights = edge.gh3_pseudo_points()
            combined_gh3_points[key].append(gh3_points)
            combined_gh3_weights[key].append(gh3_weights)


    # Step 3: Combine GH3 pseudo-points and fit a unified distribution
    for key in combined_gh3_points.keys():
        # Combine GH3 pseudo-points and weights for this edge
        all_points = np.concatenate(combined_gh3_points[key], axis=0)
        all_weights = np.concatenate(combined_gh3_weights[key], axis=0)

        # Fit a single unified distribution for this edge
        args.K = 100
        enhanced_edge = EnhancedEdge(u=key[0], v=key[1], syscall=key[2], args=args)
        enhanced_edge.fit(timestamps=all_points, args=args, data_weights=all_weights)
        enhanced_edges[key] = enhanced_edge


    return enhanced_edges

def evaluate_enhancement(csv_path, enhanced_edges, merged_edges, args):

    csv_name = csv_path.split("/")[-1].split(".")[0]
        
    technique_name = ""
    if args.use_dpgmm:
        technique_name = "dpgmm"
    elif args.use_kmeans:
        technique_name = "kmeans"
    elif args.use_dbstream:
        technique_name = "dbstream"
    else:
        technique_name = "kde"

    technique = {
        technique_name: {
            "neg_average_log_likelihood": [],
            "integrated_square_error": [],
            "mean_squared_error": [],
            "kl_divergence": [],
            "wasserstein_distance": [],
            "timestamp_count": [],
            "num_clusters": []
        }
    }

    visualization_count = 0
    
    for key, edge in enhanced_edges.items():
        all_timestamps = merged_edges[key]
        # print("all timstamps shape: ", all_timestamps.shape, flush=True)
    
        neg_log_likelihood = edge.compute_average_log_likelihood(all_timestamps)
        kl_divergence = edge.compute_kl_divergence(all_timestamps)
        wasserstein_dist = edge.compute_wasserstein_distance(all_timestamps)
        integrated_square_error = edge.compute_integrated_square_error(all_timestamps)

        if neg_log_likelihood is None or kl_divergence is None:
            continue

        technique[technique_name]["neg_average_log_likelihood"].append(neg_log_likelihood)
        technique[technique_name]["kl_divergence"].append(kl_divergence)
        technique[technique_name]["wasserstein_distance"].append(wasserstein_dist)
        technique[technique_name]["num_clusters"].append(edge.num_clusters)
        technique[technique_name]["integrated_square_error"].append(integrated_square_error)
        if args.visualize and visualization_count < 10:
            visualization_count += 1
            print("Got inside visualization for edge: ", flush=True)
            edge.visualize_distribution(timestamps=all_timestamps, name=f"./kde_visualizations/{csv_name}_{technique_name}_{edge.u}_{edge.v}.png")

        # technique[technique_name]["timestamp_count"].append(len(edge.timestamps))

    average_likelihood = average_of_edges(technique[technique_name]["neg_average_log_likelihood"])
    average_num_clusters = average_of_edges(technique[technique_name]["num_clusters"])
    average_kl_divergence = average_of_edges(technique[technique_name]["kl_divergence"])
    average_wasserstein_distance = average_of_edges(technique[technique_name]["wasserstein_distance"])
    average_integrated_square_error = average_of_edges(technique[technique_name]["integrated_square_error"])

    print(f"=== Results for {csv_name} using {technique_name} ===", flush=True)
    print(f"{technique_name} average likelihood: ", average_likelihood, flush=True)
    print(f"{technique_name} average kl divergence: ", average_kl_divergence, flush=True)
    print(f"{technique_name} average num clusters: ", average_num_clusters, flush=True)
    print(f"{technique_name} average wasserstein distance: ", average_wasserstein_distance, flush=True)
    print(f"{technique_name} integrated square error: ", average_integrated_square_error, flush=True)


def set_adversarial_edge_timestamps(target_start_time, target_end_time, merged_edges, scalers):
    attacker_victim_pair = ("172.16.0.1_same", "192.168.10.50_same", "same")
    target_start_timestamp = int(datetime.strptime(target_start_time, '%Y-%m-%d %H:%M:%S').timestamp())
    target_end_timestamp = int(datetime.strptime(target_end_time, '%Y-%m-%d %H:%M:%S').timestamp())
    # Ensure scalers are in the same order as merged_edges
    for (key, timestamps), scaler in zip(merged_edges.items(), scalers):
        # Inverse transform the scaled timestamps to get the original timestamps
        original_timestamps = scaler.inverse_transform(np.array(timestamps).reshape(-1, 1)).flatten()

        if attacker_victim_pair == key:
            print(f"Shifting timestamps for edge: {attacker_victim_pair}", flush=True)

            # Replace the earliest and latest timestamps with the target values
            shifted_timestamps = np.copy(original_timestamps)
            shifted_timestamps[np.argmin(original_timestamps)] = target_start_timestamp
            shifted_timestamps[np.argmax(original_timestamps)] = target_end_timestamp

            # Transform the updated timestamps back to the scaled range
            scaled_shifted_timestamps = scaler.transform(shifted_timestamps.reshape(-1, 1))

            # Update the timestamps in merged_edges
            merged_edges[attacker_victim_pair] = scaled_shifted_timestamps

            # Convert the shifted timestamps to human-readable format for verification
            start_time_human = datetime.fromtimestamp(min(shifted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            end_time_human = datetime.fromtimestamp(max(shifted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Shifted Edge: {attacker_victim_pair}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)
        else:
            # For all other edges, just print the human-readable timestamps
            start_time_human = datetime.fromtimestamp(min(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            end_time_human = datetime.fromtimestamp(max(original_timestamps)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"Edge: {key}, Start Time: {start_time_human}, End Time: {end_time_human}", flush=True)
