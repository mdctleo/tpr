#!/usr/bin/env python3
"""
Offline KDE Computation Script for KAIROS-KDE

This script preprocesses temporal graph datasets to compute RKHS vectors for edges
with sufficient temporal observations (≥ min_occurrences timestamps).

Usage:
    docker compose exec pids bash -c "python kde_computation.py kairos_kde_ts CLEARSCOPE_E3"
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
    """
    
    def __init__(
        self,
        rkhs_dim: int = 20,
        min_occurrences: int = 10,
        bandwidth: str = 'scott',
        n_quadrature_points: int = 10
    ):
        self.rkhs_dim = rkhs_dim
        self.min_occurrences = min_occurrences
        self.bandwidth = bandwidth
        self.n_quadrature_points = n_quadrature_points
        
        # Precompute Gauss-Hermite quadrature points and weights
        self.quad_points, self.quad_weights = self._compute_quadrature()
        
        logger.info(f"Initialized KDEVectorComputer:")
        logger.info(f"  - RKHS dimension: {rkhs_dim}")
        logger.info(f"  - Min occurrences: {min_occurrences}")
        logger.info(f"  - Bandwidth: {bandwidth}")
        logger.info(f"  - Quadrature points: {n_quadrature_points}")
    
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


def extract_edge_timestamps(cfg: SimpleConfig, dataset_name: str) -> Dict[Tuple[int, int], List[float]]:
    """
    Extract all edges and their timestamps from the dataset.
    
    Args:
        cfg: Configuration object
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary mapping (src, dst) -> list of timestamps
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    edge_timestamps = defaultdict(list)
    
    # Find processed graph files directly
    import glob
    
    # Get the latest dataset folder instead of using a static path
    base_path = get_latest_dataset_folder(dataset_name)
    
    # Load all splits
    for split in ['train', 'val', 'test']:
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
                
                # Group by edge
                for s, d, timestamp in zip(src, dst, t):
                    edge_key = (int(s), int(d))
                    edge_timestamps[edge_key].append(float(timestamp))
                    
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                continue
    
    logger.info(f"Extracted {len(edge_timestamps)} unique edges")
    return edge_timestamps


def compute_rkhs_vectors(
    edge_timestamps: Dict[Tuple[int, int], List[float]],
    kde_computer: KDEVectorComputer,
    batch_size: int = 1000
) -> Dict[Tuple[int, int], torch.Tensor]:
    """
    Compute RKHS vectors for all edges with sufficient timestamps.
    
    Args:
        edge_timestamps: Dictionary of edge -> timestamps
        kde_computer: KDEVectorComputer instance
        batch_size: Process edges in batches for progress tracking
        
    Returns:
        Dictionary mapping edge -> RKHS vector (torch.Tensor)
    """
    edge_vectors = {}
    frequent_edges = []
    rare_edges = []
    
    # Filter edges by occurrence count
    logger.info(f"Filtering edges with >= {kde_computer.min_occurrences} timestamps...")
    for edge, timestamps in edge_timestamps.items():
        if len(timestamps) >= kde_computer.min_occurrences:
            frequent_edges.append((edge, timestamps))
        else:
            rare_edges.append(edge)
    
    logger.info(f"Frequent edges (>= {kde_computer.min_occurrences} timestamps): {len(frequent_edges)}")
    logger.info(f"Rare edges (< {kde_computer.min_occurrences} timestamps): {len(rare_edges)}")
    
    # Compute RKHS vectors for frequent edges
    logger.info("Computing RKHS vectors...")
    for i in tqdm(range(0, len(frequent_edges), batch_size), desc="Processing batches"):
        batch = frequent_edges[i:i + batch_size]
        
        for edge, timestamps in batch:
            timestamps_array = np.array(timestamps, dtype=np.float64)
            rkhs_vector = kde_computer.kde_to_rkhs_vector(timestamps_array)
            edge_vectors[edge] = torch.tensor(rkhs_vector, dtype=torch.float32)
        
        # Clear memory periodically
        if (i // batch_size) % 10 == 0:
            import gc
            gc.collect()
    
    logger.info(f"Computed {len(edge_vectors)} RKHS vectors")
    return edge_vectors


def save_results(
    edge_vectors: Dict[Tuple[int, int], torch.Tensor],
    dataset_name: str,
    kde_params: Dict,
    output_dir: str = "kde_vectors"
):
    """
    Save RKHS vectors and statistics to disk.
    
    Args:
        edge_vectors: Dictionary of edge -> RKHS vector
        dataset_name: Name of the dataset
        kde_params: KDE parameters
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'dataset': dataset_name,
        'min_occurrences': kde_params.get('min_occurrences', 10),
        'rkhs_dim': kde_params.get('rkhs_dim', 20),
        'n_quadrature_points': kde_params.get('n_quadrature_points', 10),
        'bandwidth': kde_params.get('bandwidth', 'scott'),
        'num_edges': len(edge_vectors),
        'timestamp': datetime.now().isoformat()
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
    parser.add_argument('model', type=str, help='Model name (e.g., kairos_kde_ts)')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CLEARSCOPE_E3)')
    parser.add_argument('--output_dir', type=str, default='kde_vectors', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for processing')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("KDE RKHS Vector Computation")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)
    
    # Load configuration
    cfg_dict = load_config(args.model)
    cfg = SimpleConfig(cfg_dict, args.dataset)
    kde_params = cfg_dict.get('kde_params', {})
    
    # Initialize KDE computer
    kde_computer = KDEVectorComputer(
        rkhs_dim=kde_params.get('rkhs_dim', 20),
        min_occurrences=kde_params.get('min_occurrences', 10),
        bandwidth=kde_params.get('bandwidth', 'scott'),
        n_quadrature_points=kde_params.get('n_quadrature_points', 10)
    )
    
    # Extract edge timestamps
    edge_timestamps = extract_edge_timestamps(cfg, args.dataset)
    
    # Compute RKHS vectors
    edge_vectors = compute_rkhs_vectors(edge_timestamps, kde_computer, args.batch_size)
    
    # Save results
    save_results(edge_vectors, args.dataset, kde_params, args.output_dir)
    
    logger.info("=" * 80)
    logger.info("Computation complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
