#!/usr/bin/env python3
"""
Graph Reducer Script for Full Edge Reduction with Temporal Summary Features

This script creates reduced versions of the graph artifacts by collapsing
ALL edges with the same (src, dst, edge_type) into a single representative edge.

For each unique edge (src, dst, edge_type):
- Keep only one representative occurrence (the first one by timestamp)
- Store temporal summary features: (first_ts, last_ts, count) normalized
- These features replace the original time encoding in the model

Unlike reduce_graphs_kde.py, this script:
- Does NOT require precomputed KDE vectors
- Reduces ALL edges (not just those with >= min_occurrences)
- Processes ALL splits including test (for consistent evaluation)
- Stores edge_temporal_features directly in the reduced graph

Usage:
    python scripts/reduce_graphs_all.py <dataset>
    python scripts/reduce_graphs_all.py CLEARSCOPE_E3 --artifacts_dir ./artifacts
    python scripts/reduce_graphs_all.py CADETS_E3 --output_suffix _reduced_all

Example:
    # Reduce graphs for training with temporal summary features
    python scripts/reduce_graphs_all.py CLEARSCOPE_E3
    
    # This will create reduced artifacts in:
    # artifacts_reduced_all/feat_inference/<dataset>/feat_inference/<hash>/edge_embeds/{train,val,test}/
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add project root to path for imports
sys.path.insert(0, PROJECT_ROOT)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def find_best_hash_dir(feat_inference_base: str) -> str:
    """Find the hash directory with the largest disk usage."""
    if not os.path.isdir(feat_inference_base):
        raise FileNotFoundError(f"feat_inference base directory not found: {feat_inference_base}")

    subdirs = [
        os.path.join(feat_inference_base, d)
        for d in os.listdir(feat_inference_base)
        if os.path.isdir(os.path.join(feat_inference_base, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No hash directories found in {feat_inference_base}")

    result = subprocess.run(["du", "-s"] + subdirs, capture_output=True, text=True)
    lines = [l.split(maxsplit=1) for l in result.stdout.strip().split("\n") if l.strip()]
    best = max(lines, key=lambda x: int(x[0]))
    return best[1].strip()


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


def normalize_temporal_features(first_ts: torch.Tensor, last_ts: torch.Tensor, 
                                 count: torch.Tensor, graph_t_min: float, 
                                 graph_t_max: float) -> torch.Tensor:
    """
    Normalize temporal features using log-scale relative to graph time range.
    
    Args:
        first_ts: First timestamp per edge
        last_ts: Last timestamp per edge
        count: Number of occurrences per edge
        graph_t_min: Minimum timestamp in the graph
        graph_t_max: Maximum timestamp in the graph
        
    Returns:
        Tensor of shape (num_edges, 3) with normalized (first_ts, last_ts, count)
    """
    # Normalize timestamps to [0, 1] relative to graph time range
    time_range = graph_t_max - graph_t_min
    if time_range < 1e-6:
        time_range = 1.0  # Avoid division by zero
    
    first_ts_norm = (first_ts - graph_t_min) / time_range
    last_ts_norm = (last_ts - graph_t_min) / time_range
    
    # Log-scale for count (add 1 to handle count=1 case)
    count_norm = torch.log1p(count.float())
    # Normalize count to roughly [0, 1] range (log(1000) ≈ 6.9)
    count_norm = count_norm / 10.0  # Scaling factor
    
    # Stack into (num_edges, 3) tensor
    features = torch.stack([first_ts_norm, last_ts_norm, count_norm], dim=1)
    
    return features.float()


def reduce_graph_all_edges(data) -> Tuple[object, Dict]:
    """
    Reduce a graph by collapsing ALL edges with same (src, dst, edge_type).
    
    For each unique edge (src, dst, edge_type):
    - Keep only the first occurrence (by timestamp)
    - Store temporal summary: (first_ts, last_ts, count) normalized
    
    Args:
        data: TemporalData object with src, dst, t, msg, y attributes
        
    Returns:
        Tuple of (reduced_data, reduction_stats)
    """
    src = data.src.cpu()
    dst = data.dst.cpu()
    t = data.t.cpu()
    msg = data.msg.cpu()
    y = data.y.cpu() if hasattr(data, 'y') and data.y is not None else torch.zeros(len(src), dtype=torch.long)
    
    # Get edge types from msg tensor
    if hasattr(data, 'msg') and data.msg is not None:
        edge_types = extract_edge_type_from_msg(msg)
    elif hasattr(data, 'edge_type') and data.edge_type is not None:
        edge_type = data.edge_type
        if edge_type.ndim == 2:
            edge_types = edge_type.max(dim=1).indices.cpu()
        else:
            edge_types = edge_type.cpu()
    else:
        edge_types = torch.zeros(len(src), dtype=torch.long)
    
    num_edges = src.size(0)
    
    # Get graph time range for normalization
    graph_t_min = t.min().item()
    graph_t_max = t.max().item()
    
    # Group ALL edges by (src, dst, edge_type)
    edge_groups = defaultdict(list)  # (src, dst, edge_type) -> [indices]
    
    for i in range(num_edges):
        edge_key = (int(src[i]), int(dst[i]), int(edge_types[i]))
        edge_groups[edge_key].append(i)
    
    # For each edge group: compute temporal summary and keep first occurrence
    keep_indices = []
    first_ts_list = []
    last_ts_list = []
    count_list = []
    
    for edge_key, indices in edge_groups.items():
        # Sort by timestamp
        indices_sorted = sorted(indices, key=lambda i: t[i].item())
        
        # Keep the first occurrence
        keep_indices.append(indices_sorted[0])
        
        # Compute temporal summary
        timestamps = t[indices_sorted]
        first_ts_list.append(timestamps[0].item())
        last_ts_list.append(timestamps[-1].item())
        count_list.append(len(indices))
    
    # Sort keep_indices to maintain temporal order in output
    # (sort by the timestamp of the kept edge)
    sorted_order = sorted(range(len(keep_indices)), key=lambda i: t[keep_indices[i]].item())
    keep_indices = [keep_indices[i] for i in sorted_order]
    first_ts_list = [first_ts_list[i] for i in sorted_order]
    last_ts_list = [last_ts_list[i] for i in sorted_order]
    count_list = [count_list[i] for i in sorted_order]
    
    keep_tensor = torch.tensor(keep_indices, dtype=torch.long)
    
    # Create temporal features tensor
    first_ts = torch.tensor(first_ts_list, dtype=torch.float64)
    last_ts = torch.tensor(last_ts_list, dtype=torch.float64)
    count = torch.tensor(count_list, dtype=torch.long)
    
    # Normalize temporal features
    edge_temporal_features = normalize_temporal_features(
        first_ts, last_ts, count, graph_t_min, graph_t_max
    )
    
    # Create reduced data object
    from pidsmaker.utils.data_utils import CollatableTemporalData
    
    reduced_data = CollatableTemporalData(
        src=src[keep_tensor],
        dst=dst[keep_tensor],
        t=t[keep_tensor],  # Keep t for TGN memory (first_ts per edge)
        msg=msg[keep_tensor],
        y=y[keep_tensor],
    )
    
    # Add temporal features as new attribute
    reduced_data.edge_temporal_features = edge_temporal_features
    
    # Copy any additional attributes
    for attr in ['original_edge_index', 'original_n_id', 'n_id', 'edge_index']:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            val = getattr(data, attr)
            if isinstance(val, torch.Tensor) and val.size(-1) == num_edges:
                if val.ndim == 1:
                    setattr(reduced_data, attr, val[keep_tensor])
                elif val.ndim == 2:
                    setattr(reduced_data, attr, val[:, keep_tensor])
            else:
                setattr(reduced_data, attr, val)
    
    # Compute reduction statistics
    num_unique_edges = len(edge_groups)
    edges_removed = num_edges - num_unique_edges
    
    stats = {
        'original_edges': num_edges,
        'unique_edge_types': num_unique_edges,
        'reduced_edges': num_unique_edges,
        'edges_removed': edges_removed,
        'reduction_ratio': edges_removed / num_edges * 100 if num_edges > 0 else 0,
        'avg_count_per_edge': np.mean(count_list) if count_list else 0,
        'max_count_per_edge': max(count_list) if count_list else 0,
    }
    
    return reduced_data, stats


def reduce_split(
    edge_embeds_dir: str, 
    split: str, 
    output_dir: str,
):
    """Reduce all graphs in a split."""
    split_files = load_split_data(edge_embeds_dir, split)
    
    output_split_dir = os.path.join(output_dir, split)
    os.makedirs(output_split_dir, exist_ok=True)
    
    total_stats = defaultdict(int)
    all_counts = []
    
    for filename, data in split_files:
        reduced_data, stats = reduce_graph_all_edges(data)
        
        # Save reduced graph
        output_path = os.path.join(output_split_dir, filename)
        torch.save(reduced_data, output_path)
        
        # Accumulate statistics
        for k, v in stats.items():
            if isinstance(v, (int, float)) and k not in ['avg_count_per_edge', 'max_count_per_edge']:
                total_stats[k] += v
        all_counts.append(stats['avg_count_per_edge'])
    
    # Compute averages
    n_files = len(split_files)
    avg_stats = {
        'num_files': n_files,
        'total_original_edges': total_stats['original_edges'],
        'total_reduced_edges': total_stats['reduced_edges'],
        'total_edges_removed': total_stats['edges_removed'],
        'overall_reduction_ratio': (
            total_stats['edges_removed'] / total_stats['original_edges'] * 100 
            if total_stats['original_edges'] > 0 else 0
        ),
        'avg_count_per_edge': np.mean(all_counts) if all_counts else 0,
    }
    
    return avg_stats


def main():
    parser = argparse.ArgumentParser(description="Reduce graphs by collapsing all duplicate edges")
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., CLEARSCOPE_E3)')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', 
                        help='Base artifacts directory')
    parser.add_argument('--output_suffix', type=str, default='_reduced_all',
                        help='Suffix for output directory name')
    parser.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'],
                        help='Splits to process (default: train, val, test)')
    
    args = parser.parse_args()
    
    log("=" * 70)
    log(f"Full Edge Reduction for {args.dataset}")
    log("=" * 70)
    
    # Find input directory
    feat_inference_base = os.path.join(args.artifacts_dir, "feat_inference", args.dataset, "feat_inference")
    hash_dir = find_best_hash_dir(feat_inference_base)
    edge_embeds_dir = os.path.join(hash_dir, "edge_embeds")
    
    log(f"Input directory: {edge_embeds_dir}")
    
    # Setup output directory
    output_base = args.artifacts_dir.rstrip('/') + f"{args.output_suffix}"
    output_dir = edge_embeds_dir.replace(args.artifacts_dir, output_base)
    
    log(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each split
    all_stats = {}
    for split in args.splits:
        log(f"\nProcessing {split} split...")
        try:
            stats = reduce_split(edge_embeds_dir, split, output_dir)
            all_stats[split] = stats
            
            log(f"  {split} reduction complete:")
            log(f"    Original edges: {stats['total_original_edges']:,}")
            log(f"    Reduced edges:  {stats['total_reduced_edges']:,}")
            log(f"    Edges removed:  {stats['total_edges_removed']:,}")
            log(f"    Reduction ratio: {stats['overall_reduction_ratio']:.2f}%")
            log(f"    Avg occurrences per edge: {stats['avg_count_per_edge']:.2f}")
        except FileNotFoundError as e:
            log(f"  WARNING: Skipping {split} - {e}")
    
    # Save summary
    summary_file = os.path.join(output_dir, "reduction_summary.json")
    summary = {
        'dataset': args.dataset,
        'mode': 'all_edges',
        'input_dir': edge_embeds_dir,
        'output_dir': output_dir,
        'splits': all_stats,
        'normalization': {
            'timestamps': 'relative_to_graph_range_[0,1]',
            'count': 'log1p_scaled_by_10',
        },
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
    
    log(f"Total original edges: {total_original:,}")
    log(f"Total reduced edges:  {total_reduced:,}")
    log(f"Overall reduction: {(total_original - total_reduced) / total_original * 100:.2f}%" if total_original > 0 else "N/A")
    
    log("\nNote: ALL duplicate edges collapsed. Each unique (src, dst, edge_type) appears once.")
    log("Temporal info preserved in edge_temporal_features: (first_ts_norm, last_ts_norm, count_norm)")


if __name__ == "__main__":
    main()
