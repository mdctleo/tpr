#!/usr/bin/env python3
# ******************************************************************************
# compute_kde_features.py
#
# Precompute KDE-based timestamp difference features for each unique (src, dst)
# edge pair from the TRAINING snapshots only. Run this ONCE before main.py.
#
# Uses BayesianGaussianMixture (DPGMM) to fit a density model on inter-arrival
# times, then evaluates on a uniform grid to produce fixed-dim feature vectors.
#
# Usage:
#   python compute_kde_features.py --input dataset/cic/cic_20.csv --trainwin 25
#
# Output:
#   weights/kde_edge_features.pickle
#   A dict: { (src_ip, dst_ip): np.array(shape=(20,), dtype=float32), ... }
#
# Date      Name       Description
# ========  =========  ========================================================
# 2024      Asawan     KDE edge feature computation for PIKACHU enhancement
# ******************************************************************************

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm
import warnings

# Suppress convergence warnings from BayesianGaussianMixture
warnings.filterwarnings('ignore', category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute KDE edge features for PIKACHU')
    parser.add_argument('--input', type=str, default='dataset/cic/cic_20.csv',
                        help='Path to the CIC-IDS processed CSV file')
    parser.add_argument('--trainwin', type=int, default=25,
                        help='Number of training snapshots (default: 25 for Monday snapshots 0-24)')
    parser.add_argument('--kde_dim', type=int, default=20,
                        help='Dimension of KDE feature vector (number of grid points)')
    parser.add_argument('--min_timestamps', type=int, default=10,
                        help='Minimum timestamps required to fit a model (else use fallback)')
    parser.add_argument('--output', type=str, default='weights/kde_edge_features.pickle',
                        help='Output pickle file path')
    parser.add_argument('--dataset', type=str, default='cic_20',
                        help='Dataset name for logging')
    # BayesianGaussianMixture parameters
    parser.add_argument('--n_components', type=int, default=10,
                        help='Maximum number of mixture components for DPGMM')
    parser.add_argument('--weight_concentration_prior', type=float, default=0.1,
                        help='Dirichlet concentration prior (smaller = sparser mixture)')
    parser.add_argument('--max_iter', type=int, default=200,
                        help='Maximum EM iterations for DPGMM')
    return parser.parse_args()


def compute_bgmm_density_vector(timestamps: np.ndarray, kde_dim: int = 20,
                                 n_components: int = 10,
                                 weight_concentration_prior: float = 0.1,
                                 max_iter: int = 200) -> np.ndarray:
    """
    Compute a density feature vector using BayesianGaussianMixture (DPGMM).
    
    1. Compute consecutive absolute differences (inter-arrival times)
    2. Scale differences to [-1, 1] using MaxAbsScaler
    3. Fit BayesianGaussianMixture (variational Bayesian GMM with Dirichlet Process prior)
    4. Evaluate density on uniform grid of kde_dim points
    5. L2-normalize the result
    
    Args:
        timestamps: Sorted array of timestamps for this edge
        kde_dim: Number of grid points for density evaluation
        n_components: Maximum number of mixture components
        weight_concentration_prior: Dirichlet concentration (smaller = sparser)
        max_iter: Maximum EM iterations
    
    Returns:
        L2-normalized kde_dim-dimensional feature vector, or None if fitting fails
    """
    # Compute consecutive differences (inter-arrival times)
    timestamps = np.sort(timestamps)
    diffs = np.abs(np.diff(timestamps))
    
    if len(diffs) == 0:
        return None  # Will use fallback
    
    # Scale to [-1, 1] range
    scaler = MaxAbsScaler()
    scaled_diffs = scaler.fit_transform(diffs.reshape(-1, 1)).flatten()
    
    # Handle edge case where all diffs are the same (would cause singular covariance)
    if np.std(scaled_diffs) < 1e-10:
        # Constant inter-arrival time: return a delta-like vector
        # (peak in the middle, zeros elsewhere)
        vec = np.zeros(kde_dim, dtype=np.float32)
        vec[kde_dim // 2] = 1.0
        return vec
    
    try:
        # Fit BayesianGaussianMixture (DPGMM)
        # Uses variational inference with Dirichlet Process prior
        bgmm = BayesianGaussianMixture(
            n_components=min(n_components, len(scaled_diffs)),
            covariance_type='full',
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=weight_concentration_prior,
            max_iter=max_iter,
            init_params='k-means++',
            random_state=42,
            warm_start=False,
            verbose=0,
            tol=1e-3,
            reg_covar=1e-6,
        )
        
        # Fit the model (data must be 2D for sklearn)
        bgmm.fit(scaled_diffs.reshape(-1, 1))
        
        # Create evaluation grid
        grid_min, grid_max = scaled_diffs.min(), scaled_diffs.max()
        # Expand grid slightly to capture tails
        margin = 0.1 * (grid_max - grid_min) if grid_max > grid_min else 0.1
        grid = np.linspace(grid_min - margin, grid_max + margin, kde_dim)
        
        # Evaluate log-density on grid, then exponentiate
        # score_samples returns log-likelihood
        log_density = bgmm.score_samples(grid.reshape(-1, 1))
        density = np.exp(log_density).astype(np.float32)
        
        # L2 normalize
        norm = np.linalg.norm(density)
        if norm > 1e-8:
            density = density / norm
        
        return density
        
    except Exception as e:
        # Fitting can fail for degenerate cases
        return None


def main():
    args = parse_args()
    
    print("=" * 60)
    print("KDE Edge Feature Computation for PIKACHU")
    print("Using BayesianGaussianMixture (DPGMM)")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Training window: {args.trainwin} snapshots")
    print(f"KDE dimension: {args.kde_dim}")
    print(f"Min timestamps threshold: {args.min_timestamps}")
    print(f"DPGMM components: {args.n_components}")
    print(f"DPGMM concentration prior: {args.weight_concentration_prior}")
    print(f"Output: {args.output}")
    print()
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.input)
    print(f"Total records: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # Get unique snapshot IDs in order
    snapshot_ids = sorted(df['snapshot'].unique())
    print(f"Total unique snapshots: {len(snapshot_ids)}")
    print(f"Snapshot ID range: {snapshot_ids[0]} to {snapshot_ids[-1]}")
    
    # Filter to training snapshots only
    # Training snapshots are the first `trainwin` unique snapshot IDs
    # For CIC-IDS 2017: snapshots 0-24 (Monday) = first 25 entries
    train_snapshot_ids = snapshot_ids[:args.trainwin]
    print(f"Training snapshot IDs: {train_snapshot_ids[0]} to {train_snapshot_ids[-1]}")
    
    train_df = df[df['snapshot'].isin(train_snapshot_ids)]
    print(f"Training records: {len(train_df):,}")
    
    # Group by (src, dst) and collect timestamps
    print("\nGrouping edges by (src, dst)...")
    
    # Identify source and destination columns
    if 'src_computer' in train_df.columns:
        src_col, dst_col = 'src_computer', 'dst_computer'
    elif 'src' in train_df.columns:
        src_col, dst_col = 'src', 'dst'
    else:
        raise ValueError(f"Cannot find source/destination columns. Available: {list(train_df.columns)}")
    
    # Identify timestamp column
    if 'timestamp' in train_df.columns:
        ts_col = 'timestamp'
    elif 'time' in train_df.columns:
        ts_col = 'time'
    else:
        raise ValueError(f"Cannot find timestamp column. Available: {list(train_df.columns)}")
    
    print(f"Using columns: src={src_col}, dst={dst_col}, timestamp={ts_col}")
    
    # Group by directed edge (src → dst)
    edge_groups = train_df.groupby([src_col, dst_col])[ts_col].apply(list)
    print(f"Unique directed (src, dst) pairs in training: {len(edge_groups):,}")
    
    # Compute KDE features for each edge
    print(f"\nComputing DPGMM density features (min_timestamps > {args.min_timestamps})...")
    
    kde_features = {}
    fallback_count = 0
    kde_count = 0
    failed_count = 0
    
    fallback_vector = np.zeros(args.kde_dim, dtype=np.float32)
    
    for (src, dst), timestamps in tqdm(edge_groups.items(), desc="Computing DPGMM"):
        timestamps = np.array(timestamps)
        
        if len(timestamps) <= args.min_timestamps:
            # Use fallback: zeros vector
            kde_features[(src, dst)] = fallback_vector.copy()
            fallback_count += 1
        else:
            kde_vec = compute_bgmm_density_vector(
                timestamps, 
                args.kde_dim,
                n_components=args.n_components,
                weight_concentration_prior=args.weight_concentration_prior,
                max_iter=args.max_iter
            )
            if kde_vec is not None:
                kde_features[(src, dst)] = kde_vec
                kde_count += 1
            else:
                kde_features[(src, dst)] = fallback_vector.copy()
                failed_count += 1
    
    print(f"\nDPGMM computation complete:")
    print(f"  - DPGMM fitted successfully: {kde_count:,}")
    print(f"  - Fallback (≤{args.min_timestamps} timestamps): {fallback_count:,}")
    print(f"  - Failed DPGMM (exception): {failed_count:,}")
    print(f"  - Total edge pairs: {len(kde_features):,}")
    
    # Save to pickle
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(kde_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nSaved to: {args.output} ({file_size_mb:.2f} MB)")
    
    # Print some statistics about the density vectors
    if kde_count > 0:
        all_norms = [np.linalg.norm(v) for v in kde_features.values() if np.any(v != 0)]
        print(f"\nDensity vector statistics (non-zero vectors):")
        print(f"  - Count: {len(all_norms)}")
        print(f"  - Norm mean: {np.mean(all_norms):.4f}")
        print(f"  - Norm std: {np.std(all_norms):.4f}")
        print(f"  - Norm range: [{np.min(all_norms):.4f}, {np.max(all_norms):.4f}]")
    
    print("\n" + "=" * 60)
    print("Done! You can now run main.py with --kde flag to use these features.")
    print("=" * 60)


if __name__ == "__main__":
    main()
