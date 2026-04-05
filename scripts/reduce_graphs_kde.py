#!/usr/bin/env python3
"""
Graph Reducer Script for KDE-based Edge Reduction

This script creates reduced versions of the graph artifacts by collapsing
edges that have >= min_occurrences (default: 10) timestamps into single 
representative edges. Only KDE-eligible edges are reduced; other edges are 
preserved as-is.

The KDE vectors for reduced edges should already be computed by kde_computation.py.
This script uses that information to determine which edges to collapse.

For each KDE-eligible edge (src, dst, edge_type) with >= 10 timestamps:
- Keep only one representative occurrence (the first one)
- The temporal history is captured via the precomputed KDE vectors

Edges with < 10 timestamps are NOT collapsed (kept as separate edges).

Usage:
    python scripts/reduce_graphs_kde.py <dataset>
    python scripts/reduce_graphs_kde.py CADETS_E3 --artifacts_dir ./artifacts
    python scripts/reduce_graphs_kde.py CLEARSCOPE_E3 --kde_vectors_dir kde_vectors

Example:
    # First compute KDE vectors
    python kde_computation.py kairos_kde_ts CLEARSCOPE_E3
    
    # Then reduce graphs using the KDE information
    python scripts/reduce_graphs_kde.py CLEARSCOPE_E3
    
    # This will create reduced artifacts in:
    # artifacts/feat_inference_reduced/<dataset>/feat_inference/<hash>/edge_embeds/{train,val,test}/
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project root to path for imports
sys.path.insert(0, PROJECT_ROOT)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Dataset-specific dimension configurations
# Maps dataset names to (num_node_types, num_edge_types)
DATASET_DIMENSIONS = {
    # DARPA TC datasets (default)
    "CADETS_E3": (3, 10),
    "CADETS_E5": (3, 10),
    "CLEARSCOPE_E3": (3, 10),
    "THEIA_E3": (3, 10),
    "THEIA_E5": (3, 10),
    # OPTC datasets
    "optc_h051": (3, 10),
    "optc_h201": (3, 10),
    "optc_h501": (3, 10),
    # CIC-IDS-2017 (netflow only: 1 node type, 3 edge types: TCP, UDP, Other)
    "CIC_IDS_2017": (1, 3),
    "CIC_IDS_2017_PER_ATTACK": (1, 3),
}


def get_dataset_dimensions(dataset_name: str) -> Tuple[int, int]:
    """
    Get node_type_dim and edge_type_dim for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (num_node_types, num_edge_types)
    """
    if dataset_name in DATASET_DIMENSIONS:
        return DATASET_DIMENSIONS[dataset_name]
    else:
        # Default to DARPA E3 dimensions
        log(f"WARNING: Unknown dataset {dataset_name}, using default dimensions (3, 10)")
        return (3, 10)


def extract_edge_type_from_msg(msg: torch.Tensor, node_type_dim: int = 8, edge_type_dim: int = 16) -> torch.Tensor:
    """
    Extract edge type indices from the msg tensor.
    
    The msg tensor structure is: [src_type, src_emb, edge_type, dst_type, dst_emb]
    where edge_type is one-hot encoded at position (node_type_dim + emb_dim).
    
    Args:
        msg: Message tensor of shape (N, msg_dim)
        node_type_dim: Number of node type dimensions (default 8 for DARPA E3)
        edge_type_dim: Number of edge type dimensions (default 16 for DARPA E3)
        
    Returns:
        Tensor of edge type indices (argmax of the one-hot edge type portion)
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
    edge_types = edge_type_slice.argmax(dim=1)
    
    return edge_types


def load_kde_eligible_edges(kde_vectors_dir: str, dataset_name: str, device: Optional[torch.device] = None) -> Set[Tuple[int, int, int]]:
    """
    Load the set of KDE-eligible edges from precomputed vectors .pt file.
    
    Args:
        kde_vectors_dir: Directory containing KDE vectors files
        dataset_name: Name of the dataset
        device: Device to load tensors to (default: cuda if available)
        
    Returns:
        Set of (src, dst, edge_type) tuples that are KDE-eligible
    """
    # Use GPU if available
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    vectors_file = os.path.join(kde_vectors_dir, f"{dataset_name}_kde_vectors.pt")
    if os.path.exists(vectors_file):
        log(f"Loading KDE-eligible edges from {vectors_file} (device: {device})")
        data = torch.load(vectors_file, map_location=device)
        edge_vectors = data.get('edge_vectors', {})
        kde_edges = set(edge_vectors.keys())
        log(f"Loaded {len(kde_edges)} KDE-eligible edges")
        return kde_edges
    
    log(f"WARNING: KDE vectors file not found: {vectors_file}")
    return set()


def find_best_hash_dir(feat_inference_base: str) -> str:
    """Find the most recently modified hash directory (matches kde_computation.py's selection)."""
    if not os.path.isdir(feat_inference_base):
        raise FileNotFoundError(f"feat_inference base directory not found: {feat_inference_base}")

    subdirs = [
        os.path.join(feat_inference_base, d)
        for d in os.listdir(feat_inference_base)
        if os.path.isdir(os.path.join(feat_inference_base, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No hash directories found in {feat_inference_base}")

    best = max(subdirs, key=lambda d: os.path.getmtime(d))
    return best


def load_split_data(edge_embeds_dir: str, split: str):
    """Load all TemporalData.simple files for a split."""
    split_dir = os.path.join(edge_embeds_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    files = sorted(os.listdir(split_dir))
    if not files:
        raise FileNotFoundError(f"No files in {split_dir}")

    all_data = []
    for f in files:
        fpath = os.path.join(split_dir, f)
        data = torch.load(fpath, weights_only=False)
        all_data.append((f, data))

    log(f"  {split}: loaded {len(files)} files")
    return all_data


def reduce_graph_kde(
    data, 
    kde_eligible_edges: Set[Tuple[int, int, int]],
    node_type_dim: int = 3,
    edge_type_dim: int = 10,
) -> Tuple[object, Dict]:
    """
    Reduce a graph by collapsing ONLY KDE-eligible edges.
    
    For each edge (src, dst, edge_type):
    - If it's KDE-eligible (>= 10 timestamps in full dataset): collapse all occurrences to one
    - If it's NOT KDE-eligible: keep all occurrences as-is
    
    Args:
        data: TemporalData object with src, dst, t, msg, y, edge_type attributes
        kde_eligible_edges: Set of (src, dst, edge_type) tuples that have KDE vectors
        node_type_dim: Number of node type dimensions for the dataset
        edge_type_dim: Number of edge type dimensions for the dataset
        
    Returns:
        Tuple of (reduced_data, reduction_stats)
    """
    src = data.src.cpu()
    dst = data.dst.cpu()
    t = data.t.cpu()
    msg = data.msg.cpu()
    y = data.y.cpu() if hasattr(data, 'y') and data.y is not None else torch.zeros(len(src), dtype=torch.long)
    
    # Get edge types from msg tensor (edge_type is embedded in msg, not a separate attribute)
    if hasattr(data, 'msg') and data.msg is not None:
        edge_types = extract_edge_type_from_msg(msg, node_type_dim, edge_type_dim)
    elif hasattr(data, 'edge_type') and data.edge_type is not None:
        # Fallback: edge_type is a separate attribute
        edge_type = data.edge_type
        if edge_type.ndim == 2:
            # One-hot encoded, get indices
            edge_types = edge_type.max(dim=1).indices.cpu()
        else:
            edge_types = edge_type.cpu()
    else:
        # Default to edge type 0 if not available
        edge_types = torch.zeros(len(src), dtype=torch.long)
    
    num_edges = src.size(0)
    
    # Group KDE-eligible edges by (src, dst, edge_type) - these will be collapsed
    # Non-KDE edges are tracked separately - these are kept as-is
    kde_edge_groups = defaultdict(list)  # (src, dst, edge_type) -> [indices]
    non_kde_indices = []  # indices of non-KDE edges
    
    for i in range(num_edges):
        edge_key = (int(src[i]), int(dst[i]), int(edge_types[i]))
        if edge_key in kde_eligible_edges:
            kde_edge_groups[edge_key].append(i)
        else:
            non_kde_indices.append(i)
    
    # Statistics
    num_kde_edge_types = len(kde_edge_groups)
    num_kde_edge_occurrences = sum(len(indices) for indices in kde_edge_groups.values())
    num_non_kde_edges = len(non_kde_indices)
    
    # For KDE-eligible edges: keep only the first occurrence (sorted by timestamp)
    kde_keep_indices = []
    for edge_key, indices in kde_edge_groups.items():
        # Sort by timestamp and keep the first
        indices_sorted = sorted(indices, key=lambda i: t[i].item())
        kde_keep_indices.append(indices_sorted[0])
    
    # Combine: all non-KDE indices + first occurrence of each KDE edge
    keep_indices = sorted(non_kde_indices + kde_keep_indices)  # Maintain temporal order
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    
    # Create reduced data object
    from pidsmaker.utils.data_utils import CollatableTemporalData
    
    reduced_data = CollatableTemporalData(
        src=src[keep_tensor],
        dst=dst[keep_tensor],
        t=t[keep_tensor],
        msg=msg[keep_tensor],
        y=y[keep_tensor],
    )
    
    # Copy any additional attributes
    for attr in ['original_edge_index', 'original_n_id', 'n_id', 'edge_index']:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            val = getattr(data, attr)
            if isinstance(val, torch.Tensor) and val.size(-1) == num_edges:
                # Index into the last dimension
                if val.ndim == 1:
                    setattr(reduced_data, attr, val[keep_tensor])
                elif val.ndim == 2:
                    setattr(reduced_data, attr, val[:, keep_tensor])
            else:
                setattr(reduced_data, attr, val)
    
    # Compute reduction statistics
    edges_removed = num_edges - len(keep_indices)
    
    stats = {
        'original_edges': num_edges,
        'kde_edge_types': num_kde_edge_types,
        'kde_edge_occurrences': num_kde_edge_occurrences,
        'non_kde_edges': num_non_kde_edges,
        'reduced_edges': len(keep_indices),
        'edges_removed': edges_removed,
        'kde_edges_collapsed': num_kde_edge_occurrences - num_kde_edge_types,  # How many KDE duplicates removed
        'reduction_ratio': edges_removed / num_edges * 100 if num_edges > 0 else 0,
    }
    
    return reduced_data, stats


def reduce_split(
    edge_embeds_dir: str, 
    split: str, 
    output_dir: str, 
    kde_eligible_edges: Set[Tuple[int, int, int]],
    node_type_dim: int = 3,
    edge_type_dim: int = 10,
):
    """Reduce all graphs in a split."""
    split_files = load_split_data(edge_embeds_dir, split)
    
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    total_stats = defaultdict(int)
    
    for filename, data in split_files:
        reduced_data, stats = reduce_graph_kde(data, kde_eligible_edges, node_type_dim, edge_type_dim)
        
        # Save reduced graph
        output_path = os.path.join(output_split_dir, filename)
        torch.save(reduced_data, output_path)
        
        # Accumulate statistics
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                total_stats[k] += v
    
    # Compute averages
    n_files = len(split_files)
    avg_stats = {
        'num_files': n_files,
        'total_original_edges': total_stats['original_edges'],
        'total_reduced_edges': total_stats['reduced_edges'],
        'total_edges_removed': total_stats['edges_removed'],
        'total_kde_edges_collapsed': total_stats['kde_edges_collapsed'],
        'overall_reduction_ratio': (
            total_stats['edges_removed'] / total_stats['original_edges'] * 100 
            if total_stats['original_edges'] > 0 else 0
        ),
    }
    
    return avg_stats


def main():
    parser = argparse.ArgumentParser(description="Reduce graphs by collapsing KDE-eligible edges")
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CLEARSCOPE_E3)')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', 
                        help='Base artifacts directory')
    parser.add_argument('--kde_vectors_dir', type=str, default='kde_vectors',
                        help='Directory containing KDE vectors from kde_computation.py')
    parser.add_argument('--output_suffix', type=str, default='_reduced',
                        help='Suffix for output directory name')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val'],
                        help='Splits to process (default: train, val - test is kept unreduced)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    parser.add_argument('--feat_hash', type=str, default=None,
                        help='Specific feat_inference hash directory to process (default: latest)')
    parser.add_argument('--all_hashes', action='store_true',
                        help='Process ALL feat_inference hash directories (not just the latest)')
    parser.add_argument('--symlink_stages', action='store_true',
                        help='Symlink construction/transformation/featurization from source artifacts_dir '
                             'into the reduced output dir so the pipeline can find all prior stages')
    
    args = parser.parse_args()
    
    # Setup device
    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log("=" * 70)
    log(f"KDE-based Graph Reduction for {args.dataset}")
    log(f"Using device: {device}")
    log("=" * 70)
    
    # Get dataset-specific dimensions
    node_type_dim, edge_type_dim = get_dataset_dimensions(args.dataset)
    log(f"Dataset dimensions: node_type_dim={node_type_dim}, edge_type_dim={edge_type_dim}")
    
    # Load KDE-eligible edges
    kde_eligible_edges = load_kde_eligible_edges(args.kde_vectors_dir, args.dataset, device)
    
    if not kde_eligible_edges:
        log("ERROR: No KDE-eligible edges found. Please run kde_computation.py first.")
        log(f"  python kde_computation.py kairos_kde_ts {args.dataset}")
        sys.exit(1)
    
    log(f"KDE-eligible edges: {len(kde_eligible_edges)}")
    
    # Find input directory / directories
    feat_inference_base = os.path.join(args.artifacts_dir, "feat_inference", args.dataset, "feat_inference")
    
    if args.all_hashes:
        # Process ALL hash directories
        hash_dirs = [
            os.path.join(feat_inference_base, d)
            for d in os.listdir(feat_inference_base)
            if os.path.isdir(os.path.join(feat_inference_base, d))
        ]
        if not hash_dirs:
            log("ERROR: No hash directories found.")
            sys.exit(1)
        log(f"Processing ALL {len(hash_dirs)} hash directories")
    elif args.feat_hash:
        # Process a specific hash (exact or prefix match)
        specific = os.path.join(feat_inference_base, args.feat_hash)
        if os.path.isdir(specific):
            hash_dirs = [specific]
        else:
            # Try prefix match
            matches = [
                os.path.join(feat_inference_base, d)
                for d in os.listdir(feat_inference_base)
                if d.startswith(args.feat_hash) and os.path.isdir(os.path.join(feat_inference_base, d))
            ]
            if not matches:
                log(f"ERROR: No hash directory matching '{args.feat_hash}' found under {feat_inference_base}")
                sys.exit(1)
            if len(matches) > 1:
                log(f"ERROR: Ambiguous prefix '{args.feat_hash}' matches multiple dirs: {matches}")
                sys.exit(1)
            hash_dirs = matches
    else:
        # Process only the latest (default)
        hash_dirs = [find_best_hash_dir(feat_inference_base)]
    
    for hash_dir in hash_dirs:
        hash_name = os.path.basename(hash_dir)
        log(f"\n{'='*70}")
        log(f"Processing hash: {hash_name}")
        log(f"{'='*70}")
        edge_embeds_dir = os.path.join(hash_dir, "edge_embeds")
    
        log(f"Input directory: {edge_embeds_dir}")
    
        # Setup output directory (parallel to the original, with suffix)
        # e.g., artifacts/feat_inference_reduced/DATASET/feat_inference/HASH/edge_embeds/
        output_base = args.artifacts_dir.rstrip('/') + f"{args.output_suffix}"
        output_dir = edge_embeds_dir.replace(args.artifacts_dir, output_base)
    
        log(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
        # Process each split (reduce train/val, copy test unchanged)
        all_stats = {}
        for split in args.splits:
            log(f"\nProcessing {split} split...")
            try:
                stats = reduce_split(edge_embeds_dir, split, output_dir, kde_eligible_edges, node_type_dim, edge_type_dim)
                all_stats[split] = stats
            
                log(f"  {split} reduction complete:")
                log(f"    Original edges: {stats['total_original_edges']:,}")
                log(f"    Reduced edges:  {stats['total_reduced_edges']:,}")
                log(f"    Edges removed:  {stats['total_edges_removed']:,}")
                log(f"    KDE edges collapsed: {stats['total_kde_edges_collapsed']:,}")
                log(f"    Reduction ratio: {stats['overall_reduction_ratio']:.2f}%")
            except FileNotFoundError as e:
                log(f"  WARNING: Skipping {split} - {e}")
    
        # Copy test split unchanged (if not already processed)
        if 'test' not in args.splits:
            test_src = os.path.join(edge_embeds_dir, 'test')
            test_dst = os.path.join(output_dir, 'test')
            if os.path.isdir(test_src) and not os.path.exists(test_dst):
                log(f"\nCopying test split unchanged...")
                import shutil
                shutil.copytree(test_src, test_dst)
                log(f"  Test split copied to {test_dst}")
    
        # Save summary
        summary_file = os.path.join(output_dir, "reduction_summary.json")
        summary = {
            'dataset': args.dataset,
            'kde_eligible_edges': len(kde_eligible_edges),
            'input_dir': edge_embeds_dir,
            'output_dir': output_dir,
            'splits': all_stats,
        }
    
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
        log(f"\nSummary saved to {summary_file}")
    
        # Overall summary
        log("\n" + "=" * 70)
        log("REDUCTION SUMMARY")
        log("=" * 70)
    
        total_original = sum(s.get('total_original_edges', 0) for s in all_stats.values())
        total_reduced = sum(s.get('total_reduced_edges', 0) for s in all_stats.values())
        total_kde_collapsed = sum(s.get('total_kde_edges_collapsed', 0) for s in all_stats.values())
    
        log(f"Total original edges: {total_original:,}")
        log(f"Total reduced edges:  {total_reduced:,}")
        log(f"Total KDE edges collapsed: {total_kde_collapsed:,}")
        log(f"Overall reduction: {(total_original - total_reduced) / total_original * 100:.2f}%" if total_original > 0 else "N/A")
    
        log("\nNote: Only KDE-eligible edges (>= 10 timestamps) were collapsed.")
        log("Non-KDE edges are preserved as-is to maintain their individual timestamps.")

        # Write done.txt marker so the pipeline recognizes this stage as complete
        done_file = os.path.join(os.path.dirname(edge_embeds_dir.replace(args.artifacts_dir, output_base)), "done.txt")
        os.makedirs(os.path.dirname(done_file), exist_ok=True)
        with open(done_file, 'w') as f:
            f.write(f"Reduced from {args.artifacts_dir} with KDE vectors from {args.kde_vectors_dir}\n")
        log(f"Written done.txt marker: {done_file}")

    # Symlink prior pipeline stages from source artifacts_dir into the reduced output dir
    if args.symlink_stages:
        output_base = args.artifacts_dir.rstrip('/') + f"{args.output_suffix}"
        stages_to_link = ['construction', 'transformation', 'featurization']
        log(f"\n{'='*70}")
        log("Symlinking prior pipeline stages into reduced output dir")
        log(f"{'='*70}")
        for stage in stages_to_link:
            src = os.path.abspath(os.path.join(args.artifacts_dir, stage))
            dst = os.path.join(output_base, stage)
            if os.path.exists(dst) or os.path.islink(dst):
                log(f"  {stage}: already exists at {dst}, skipping")
            elif os.path.isdir(src):
                os.symlink(src, dst)
                log(f"  {stage}: symlinked {dst} -> {src}")
            else:
                log(f"  {stage}: source not found at {src}, skipping")
        log("Done. The reduced artifacts dir is now ready for --force_restart batching")


if __name__ == "__main__":
    main()
