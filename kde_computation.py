#!/usr/bin/env python3
"""
Offline KDE Computation Script for KAIROS-KDE

This script preprocesses temporal graph datasets to compute RKHS vectors for edges
with sufficient temporal observations (≥ min_occurrences timestamps).

Supports two modes:
1. Raw timestamps (default): KDE computed on raw timestamp values
2. Timestamp differences (--use_timestamp_diffs): KDE computed on inter-arrival times
   (differences between consecutive timestamps). This captures temporal patterns.

Usage:
    # Raw timestamps mode (original)
    python kde_computation.py kairos_kde_ts CLEARSCOPE_E3
    
    # Timestamp differences mode (new)
    python kde_computation.py kairos_kde_diff CLEARSCOPE_E3 --output_dir kde_vectors_diff
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from scipy.stats import gaussian_kde
from scipy.special import hermite
from tqdm import tqdm

# Add pidsmaker to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pidsmaker.utils.data_utils import load_data_set, collate_temporal_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


class KDEVectorComputer:
    """
    Computes RKHS vectors from edge timestamps using Kernel Density Estimation.
    
    Supports two modes:
    1. Raw timestamps: KDE on absolute timestamp values
    2. Timestamp differences: KDE on inter-arrival times (t[i+1] - t[i])
    """
    
    def __init__(
        self,
        rkhs_dim: int = 20,
        min_occurrences: int = 10,
        bandwidth: str = 'scott',
        n_quadrature_points: int = 10,
        use_timestamp_diffs: bool = False
    ):
        self.rkhs_dim = rkhs_dim
        self.min_occurrences = min_occurrences
        self.bandwidth = bandwidth
        self.n_quadrature_points = n_quadrature_points
        self.use_timestamp_diffs = use_timestamp_diffs
        
        # Precompute Gauss-Hermite quadrature points and weights
        self.quad_points, self.quad_weights = self._compute_quadrature()
        
        logger.info(f"Initialized KDEVectorComputer:")
        logger.info(f"  - RKHS dimension: {rkhs_dim}")
        logger.info(f"  - Min occurrences: {min_occurrences}")
        logger.info(f"  - Bandwidth: {bandwidth}")
        logger.info(f"  - Quadrature points: {n_quadrature_points}")
        logger.info(f"  - Use timestamp diffs: {use_timestamp_diffs}")
    
    def _compute_quadrature(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Gauss-Hermite quadrature points and weights."""
        from numpy.polynomial.hermite import hermgauss
        points, weights = hermgauss(self.n_quadrature_points)
        # Normalize weights for probability integration
        weights = weights / np.sqrt(np.pi)
        return points, weights
    
    def kde_to_rkhs_vector(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Convert timestamps to RKHS vector via KDE and Gauss-Hermite quadrature.
        
        Args:
            timestamps: Array of timestamps for a single edge
            
        Returns:
            RKHS vector of dimension rkhs_dim
        """
        try:
            # Compute KDE
            kde = gaussian_kde(timestamps, bw_method=self.bandwidth)
            
            # Get data statistics for scaling
            mean = np.mean(timestamps)
            std = np.std(timestamps)
            if std < 1e-6:
                std = 1.0
            
            # Scale quadrature points to data range
            scaled_points = mean + std * np.sqrt(2) * self.quad_points
            
            # Evaluate KDE at quadrature points
            kde_values = kde(scaled_points)
            
            # Weighted KDE values (first component)
            weighted_values = kde_values * self.quad_weights
            
            # Statistical moments
            moments = np.array([
                np.mean(timestamps),
                np.std(timestamps),
                self._safe_skewness(timestamps),
                self._safe_kurtosis(timestamps)
            ])
            
            # Fourier features (capture periodicity)
            freqs = np.array([1.0, 2.0, 3.0])
            fourier_features = []
            for freq in freqs:
                fourier_features.append(np.mean(np.cos(2 * np.pi * freq * timestamps / (mean + 1e-6))))
                fourier_features.append(np.mean(np.sin(2 * np.pi * freq * timestamps / (mean + 1e-6))))
            fourier_features = np.array(fourier_features)
            
            # Quantiles
            quantiles = np.percentile(timestamps, [25, 50, 75])
            
            # Combine all features
            rkhs_features = np.concatenate([
                weighted_values[:5],      # 5 dims
                moments,                  # 4 dims
                fourier_features,         # 6 dims
                quantiles,                # 3 dims
                np.array([len(timestamps), timestamps.max() - timestamps.min()])  # 2 dims
            ])
            
            # Pad or truncate to exact rkhs_dim
            if len(rkhs_features) < self.rkhs_dim:
                rkhs_features = np.pad(rkhs_features, (0, self.rkhs_dim - len(rkhs_features)))
            else:
                rkhs_features = rkhs_features[:self.rkhs_dim]
            
            return rkhs_features.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"KDE computation failed: {e}, using zero vector")
            return np.zeros(self.rkhs_dim, dtype=np.float32)
    
    def _safe_skewness(self, data: np.ndarray) -> float:
        """Compute skewness safely."""
        try:
            from scipy.stats import skew
            return float(skew(data))
        except:
            return 0.0
    
    def _safe_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis safely."""
        try:
            from scipy.stats import kurtosis
            return float(kurtosis(data))
        except:
            return 0.0
    
    def compute_timestamp_diffs(self, timestamps: np.ndarray) -> np.ndarray:
        """
        Convert sorted timestamps to differences between consecutive timestamps.
        
        Args:
            timestamps: Array of timestamps (will be sorted)
            
        Returns:
            Array of inter-arrival times (length n-1)
        """
        sorted_ts = np.sort(timestamps)
        return np.diff(sorted_ts)
    
    def timestamp_diffs_to_rkhs_vector(self, timestamp_diffs: np.ndarray) -> np.ndarray:
        """
        Convert timestamp DIFFERENCES (inter-arrival times) to RKHS vector.
        
        This captures the temporal pattern of edge occurrences:
        - Mean inter-arrival time (avg time between events)
        - Std of inter-arrival times (regularity)
        - Skewness (bursty vs uniform arrivals)
        - Kurtosis (heavy tails in timing)
        
        Args:
            timestamp_diffs: Array of inter-arrival times for a single edge
            
        Returns:
            RKHS vector of dimension rkhs_dim
        """
        try:
            # Filter out zero or negative diffs (shouldn't happen but be safe)
            valid_diffs = timestamp_diffs[timestamp_diffs > 0]
            if len(valid_diffs) < 2:
                logger.warning("Not enough valid timestamp diffs, using zero vector")
                return np.zeros(self.rkhs_dim, dtype=np.float32)
            
            # Compute KDE on inter-arrival times
            kde = gaussian_kde(valid_diffs, bw_method=self.bandwidth)
            
            # Get statistics for scaling
            mean = np.mean(valid_diffs)
            std = np.std(valid_diffs)
            if std < 1e-6:
                std = 1.0
            
            # Scale quadrature points to data range
            scaled_points = mean + std * np.sqrt(2) * self.quad_points
            # Ensure points are positive (inter-arrival times are positive)
            scaled_points = np.maximum(scaled_points, 1e-6)
            
            # Evaluate KDE at quadrature points
            kde_values = kde(scaled_points)
            
            # Weighted KDE values
            weighted_values = kde_values * self.quad_weights
            
            # Statistical moments of inter-arrival times
            moments = np.array([
                np.mean(valid_diffs),       # Mean inter-arrival time
                np.std(valid_diffs),        # Std of inter-arrival times
                self._safe_skewness(valid_diffs),  # Skewness (burstiness)
                self._safe_kurtosis(valid_diffs)   # Kurtosis (heavy tails)
            ])
            
            # Fourier features on normalized diffs (capture periodicity in arrivals)
            normalized_diffs = valid_diffs / (mean + 1e-6)
            freqs = np.array([1.0, 2.0, 3.0])
            fourier_features = []
            for freq in freqs:
                fourier_features.append(np.mean(np.cos(2 * np.pi * freq * normalized_diffs)))
                fourier_features.append(np.mean(np.sin(2 * np.pi * freq * normalized_diffs)))
            fourier_features = np.array(fourier_features)
            
            # Quantiles of inter-arrival times
            quantiles = np.percentile(valid_diffs, [25, 50, 75])
            
            # Combine all features
            rkhs_features = np.concatenate([
                weighted_values[:5],      # 5 dims: KDE at quadrature points
                moments,                  # 4 dims: mean, std, skew, kurtosis
                fourier_features,         # 6 dims: periodicity features
                quantiles,                # 3 dims: 25th, 50th, 75th percentiles
                np.array([len(valid_diffs), valid_diffs.max() - valid_diffs.min()])  # 2 dims: count, range
            ])
            
            # Pad or truncate to exact rkhs_dim
            if len(rkhs_features) < self.rkhs_dim:
                rkhs_features = np.pad(rkhs_features, (0, self.rkhs_dim - len(rkhs_features)))
            else:
                rkhs_features = rkhs_features[:self.rkhs_dim]
            
            return rkhs_features.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"KDE computation on diffs failed: {e}, using zero vector")
            return np.zeros(self.rkhs_dim, dtype=np.float32)


def get_latest_dataset_folder(dataset_name: str) -> str:
    """
    Get the latest updated folder in the corresponding dataset directory.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to the latest updated folder
    """
    import glob
    import os
    
    base_path = f"/scratch/asawan15/PIDSMaker/artifacts/feat_inference/{dataset_name}/feat_inference"
    
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


def extract_edge_timestamps(cfg: SimpleConfig, dataset_name: str) -> Dict[Tuple[int, int, int], List[float]]:
    """
    Extract all edges and their timestamps from the dataset.
    
    Args:
        cfg: Configuration object
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary mapping (src, dst, edge_type) -> list of timestamps
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    edge_timestamps = defaultdict(list)
    
    # Find processed graph files directly
    import glob
    
    # Get the latest dataset folder instead of using a static path
    base_path = get_latest_dataset_folder(dataset_name)
    
    # Get dimension parameters from config or use defaults for DARPA E3
    node_type_dim = getattr(getattr(cfg, 'dataset', None), 'num_node_types', 8)
    edge_type_dim = getattr(getattr(cfg, 'dataset', None), 'num_edge_types', 16)
    logger.info(f"Using node_type_dim={node_type_dim}, edge_type_dim={edge_type_dim}")
    
    # Load all splits
    for split in ['train', 'val']:#, 'test']:
        logger.info(f"Processing {split} split...")
        
        # Find all .TemporalData.simple files for this split
        pattern = f"{base_path}/edge_embeds/{split}/*.TemporalData.simple"
        files = glob.glob(pattern)
        
        if not files:
            logger.warning(f"No files found for {split} split at {pattern}")
            continue
        
        logger.info(f"Found {len(files)} files for {split} split")
        
        # Process each file
        for file_path in tqdm(files, desc=f"Extracting {split} edges"):
            try:
                # Load the temporal data
                data = torch.load(file_path, map_location='cpu')
                
                # Extract edges and timestamps
                src = data.src.cpu().numpy()
                dst = data.dst.cpu().numpy()
                t = data.t.cpu().numpy()
                
                # Extract edge types from msg tensor
                # The edge_type is embedded in the msg tensor, not a separate attribute
                if hasattr(data, 'msg') and data.msg is not None:
                    edge_types = extract_edge_type_from_msg(
                        data.msg, node_type_dim, edge_type_dim
                    )
                elif hasattr(data, 'edge_type') and data.edge_type is not None:
                    # Fallback: edge_type is one-hot encoded as separate attribute
                    edge_types = data.edge_type.max(dim=1).indices.cpu().numpy()
                else:
                    # Default to edge type 0 if not available
                    logger.warning(f"No edge type info in {file_path}, defaulting to 0")
                    edge_types = np.zeros(len(src), dtype=np.int64)
                
                # Group by edge (src, dst, edge_type)
                for s, d, et, timestamp in zip(src, dst, edge_types, t):
                    edge_key = (int(s), int(d), int(et))
                    edge_timestamps[edge_key].append(float(timestamp))
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
    
    logger.info(f"Extracted {len(edge_timestamps)} unique edges")
    return edge_timestamps


def compute_rkhs_vectors(
    edge_timestamps: Dict[Tuple[int, int, int], List[float]],
    kde_computer: KDEVectorComputer,
    batch_size: int = 1000
) -> Tuple[Dict[Tuple[int, int, int], torch.Tensor], Dict[Tuple[int, int, int], int]]:
    """
    Compute RKHS vectors for all edges with sufficient timestamps.
    
    Args:
        edge_timestamps: Dictionary of (src, dst, edge_type) -> timestamps
        kde_computer: KDEVectorComputer instance
        batch_size: Process edges in batches for progress tracking
        
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
    mode_str = "timestamp DIFFERENCES" if kde_computer.use_timestamp_diffs else "raw timestamps"
    logger.info(f"Computing RKHS vectors using {mode_str}...")
    
    for i in tqdm(range(0, len(frequent_edges), batch_size), desc="Processing batches"):
        batch = frequent_edges[i:i + batch_size]
        
        for edge, timestamps in batch:
            timestamps_array = np.array(timestamps, dtype=np.float64)
            
            if kde_computer.use_timestamp_diffs:
                # Compute KDE on inter-arrival times (timestamp differences)
                timestamp_diffs = kde_computer.compute_timestamp_diffs(timestamps_array)
                if len(timestamp_diffs) < 2:
                    # Not enough diffs for KDE (should be rare with min_occurrences=10)
                    logger.warning(f"Edge {edge} has only {len(timestamp_diffs)} diffs, skipping")
                    continue
                rkhs_vector = kde_computer.timestamp_diffs_to_rkhs_vector(timestamp_diffs)
            else:
                # Original mode: KDE on raw timestamps
                rkhs_vector = kde_computer.kde_to_rkhs_vector(timestamps_array)
            
            edge_vectors[edge] = torch.tensor(rkhs_vector, dtype=torch.float32)
        
        # Clear memory periodically
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
        'min_occurrences': kde_params.get('min_occurrences', 10),
        'rkhs_dim': kde_params.get('rkhs_dim', 20),
        'n_quadrature_points': kde_params.get('n_quadrature_points', 10),
        'bandwidth': kde_params.get('bandwidth', 'scott'),
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
    parser = argparse.ArgumentParser(description="Compute KDE RKHS vectors for temporal graphs")
    parser.add_argument('model', type=str, help='Model name (e.g., kairos_kde_ts, kairos_kde_diff)')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CLEARSCOPE_E3)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (default: from config kde_vectors_dir)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    parser.add_argument('--use_timestamp_diffs', action='store_true', 
                        help='Compute KDE on timestamp differences instead of raw timestamps (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg_dict = load_config(args.model)
    cfg = SimpleConfig(cfg_dict, args.dataset)
    kde_params = cfg_dict.get('kde_params', {})
    
    # Determine whether to use timestamp diffs (CLI arg overrides config)
    use_timestamp_diffs = args.use_timestamp_diffs or kde_params.get('use_timestamp_diffs', False)
    
    # Determine output directory (CLI arg overrides config)
    output_dir = args.output_dir or kde_params.get('kde_vectors_dir', 'kde_vectors')
    
    logger.info("=" * 80)
    logger.info("KDE RKHS Vector Computation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Mode: {'Timestamp DIFFERENCES' if use_timestamp_diffs else 'Raw timestamps'}")
    logger.info("=" * 80)
    
    # Initialize KDE computer
    kde_computer = KDEVectorComputer(
        rkhs_dim=kde_params.get('rkhs_dim', 20),
        min_occurrences=kde_params.get('min_occurrences', 10),
        bandwidth=kde_params.get('bandwidth', 'scott'),
        n_quadrature_points=kde_params.get('n_quadrature_points', 10),
        use_timestamp_diffs=use_timestamp_diffs
    )
    
    # Extract edge timestamps
    edge_timestamps = extract_edge_timestamps(cfg, args.dataset)
    
    # Compute RKHS vectors (returns both vectors and occurrence counts)
    edge_vectors, edge_occurrence_counts = compute_rkhs_vectors(edge_timestamps, kde_computer, args.batch_size)
    
    # Add use_timestamp_diffs to kde_params for metadata
    kde_params_with_mode = dict(kde_params)
    kde_params_with_mode['use_timestamp_diffs'] = use_timestamp_diffs
    
    # Save results (including edge_occurrence_counts in metadata)
    save_results(edge_vectors, edge_occurrence_counts, args.dataset, kde_params_with_mode, output_dir)
    
    logger.info("=" * 80)
    logger.info("Computation complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
