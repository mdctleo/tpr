#!/usr/bin/env python3
"""
GPU Timing Analysis Script

Analyzes training and inference timing:
1. KDE-enhanced runs (from batch_timing JSON files in artifacts_reduced)
2. KDE diff runs (from batch_timing JSON files in artifacts_reduced_diff)
3. Base runs (from batch_timing JSON files in artifacts, if available)
4. Reduced graph runs (from simple_batch_timing JSON files in artifacts_reduced_all)

The goal is to compare the average time per batch between different configs.

Results are saved to gpu_timing_info/
"""

import json
import os
import re
import glob
from datetime import datetime
from collections import defaultdict
import statistics

# Base path for artifacts
BASE_PATH = "/scratch/asawan15/PIDSMaker"

# Hash → Config mapping based on hash_to_run_mapping.txt
# The same hash appears in both artifacts/ and artifacts_reduced/ for different configs
# because the hash is derived from dataset-level parameters, not the config name.

# KDE-enhanced configs (in artifacts_reduced/)
KDE_CONFIGS = {
    # orthrus_edge_kde_ts
    ("2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7", "CLEARSCOPE_E3"): {
        "config": "orthrus_edge_kde_ts", "epoch": 0
    },
    ("970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727", "CADETS_E3"): {
        "config": "orthrus_edge_kde_ts", "epoch": 7
    },
    ("9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457", "THEIA_E3"): {
        "config": "orthrus_edge_kde_ts", "epoch": 3
    },
    # kairos_kde_ts
    ("293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e", "CLEARSCOPE_E3"): {
        "config": "kairos_kde_ts", "epoch": 1
    },
    ("133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269", "CADETS_E3"): {
        "config": "kairos_kde_ts", "epoch": 5
    },
    ("e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5", "THEIA_E3"): {
        "config": "kairos_kde_ts", "epoch": 1
    },
}

# KDE diff configs (in artifacts_reduced_diff/) - uses timestamp differences
KDE_DIFF_CONFIGS = {
    # orthrus_kde_diff - using epoch 0 for all
    ("2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7", "CLEARSCOPE_E3"): {
        "config": "orthrus_kde_diff", "epoch": 0
    },
    ("970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727", "CADETS_E3"): {
        "config": "orthrus_kde_diff", "epoch": 0
    },
    ("9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457", "THEIA_E3"): {
        "config": "orthrus_kde_diff", "epoch": 0
    },
    # kairos_kde_diff - using epoch 0 for all
    ("293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e", "CLEARSCOPE_E3"): {
        "config": "kairos_kde_diff", "epoch": 0
    },
    ("133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269", "CADETS_E3"): {
        "config": "kairos_kde_diff", "epoch": 0
    },
    ("e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5", "THEIA_E3"): {
        "config": "kairos_kde_diff", "epoch": 0
    },
}

# Base configs (in artifacts/)
BASE_CONFIGS = {
    # orthrus_non_snooped_edge_ts - using epoch 0 for all
    ("293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e", "CLEARSCOPE_E3"): {
        "config": "orthrus_non_snooped_edge_ts", "epoch": 0
    },
    ("970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727", "CADETS_E3"): {
        "config": "orthrus_non_snooped_edge_ts", "epoch": 0
    },
    ("9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457", "THEIA_E3"): {
        "config": "orthrus_non_snooped_edge_ts", "epoch": 0
    },
    # kairos - using epoch 0 for all
    ("2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7", "CLEARSCOPE_E3"): {
        "config": "kairos", "epoch": 0
    },
    ("133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269", "CADETS_E3"): {
        "config": "kairos", "epoch": 0
    },
    ("e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5", "THEIA_E3"): {
        "config": "kairos", "epoch": 0
    },
}

# Reduced graph configs (in artifacts_reduced_all/) - temporal summary features
REDUCED_CONFIGS = {
    # orthrus_red
    ("293471020f6a7101d4266ecc1efeaa9d64e8ec367dcaa7635659a7dd4af2302e", "CLEARSCOPE_E3"): {
        "config": "orthrus_red", "epoch": 5
    },
    ("970c13085c1d6feabe4790a3ec192e29b1b4742bcde5bf3199c192962c698727", "CADETS_E3"): {
        "config": "orthrus_red", "epoch": 11
    },
    ("9364ddb2b1b64ea9dcf4f3b818defba39706f6bbdaf4ef6e07ad9df66813e457", "THEIA_E3"): {
        "config": "orthrus_red", "epoch": 3
    },
    # kairos_red
    ("2551955c45d630a0e97731a9ff890bc791f9b8e1f92f6eb467a175847dc281a7", "CLEARSCOPE_E3"): {
        "config": "kairos_red", "epoch": 7
    },
    ("133dbd81e39cf6fd439cc60da9f2fbea820e60e2a7629a91b7a02719415c6269", "CADETS_E3"): {
        "config": "kairos_red", "epoch": 1
    },
    ("e9f5191a26589f5ad9ac4b4b5c7d717f1789d1281a50d41e38f9c516a10f08b5", "THEIA_E3"): {
        "config": "kairos_red", "epoch": 3
    },
}

# Mapping of KDE config to its base config counterpart
KDE_TO_BASE_CONFIG = {
    "orthrus_edge_kde_ts": "orthrus_non_snooped_edge_ts",
    "kairos_kde_ts": "kairos",
    # KDE diff configs (timestamp differences)
    "orthrus_kde_diff": "orthrus_non_snooped_edge_ts",
    "kairos_kde_diff": "kairos",
    # Reduced graph configs
    "orthrus_red": "orthrus_non_snooped_edge_ts",
    "kairos_red": "kairos",
}


def parse_simple_batch_timing(batch_timing_dir, dataset):
    """
    Parse simple batch timing data from batch_timing directory.
    Used for reduced graph configs (no KDE taint tracking).
    
    Files:
    - batch_timing_{dataset}.json: All batches with simple timing
    - simple_batch_timing.json: Alternative format
    
    Returns dict with:
    - summary: {train: {stats...}, eval: {stats...}}
    - detailed_results: [list of batch timing dicts]
    """
    # Try simple_batch_timing.json first (new format)
    simple_file = os.path.join(batch_timing_dir, "simple_batch_timing.json")
    if os.path.exists(simple_file):
        with open(simple_file, 'r') as f:
            data = json.load(f)
        return data
    
    # Try batch_timing_{dataset}.json (standard format)
    timing_file = os.path.join(batch_timing_dir, f"batch_timing_{dataset}.json")
    if os.path.exists(timing_file):
        with open(timing_file, 'r') as f:
            data = json.load(f)
        return data
    
    return None


def discover_reduced_graph_runs(artifacts_dir):
    """
    Auto-discover reduced graph runs from artifacts_reduced_all.
    
    Looks for batch_timing directories with simple_batch_timing.json or batch_timing_*.json
    
    Returns dict of {config_dataset: timing_data}
    """
    results = {}
    eval_base = os.path.join(artifacts_dir, "evaluation", "evaluation")
    
    if not os.path.exists(eval_base):
        return results
    
    # List hash directories
    for hash_dir in os.listdir(eval_base):
        hash_path = os.path.join(eval_base, hash_dir)
        if not os.path.isdir(hash_path):
            continue
        
        # List dataset directories
        for dataset in os.listdir(hash_path):
            dataset_path = os.path.join(hash_path, dataset)
            batch_timing_dir = os.path.join(dataset_path, "batch_timing")
            
            if not os.path.exists(batch_timing_dir):
                continue
            
            timing_data = parse_simple_batch_timing(batch_timing_dir, dataset)
            if timing_data:
                # Try to determine config name from metadata
                config_name = determine_config_from_timing(batch_timing_dir, hash_path, dataset)
                key = f"{config_name}_{dataset}"
                results[key] = {
                    'config': config_name,
                    'dataset': dataset,
                    'hash': hash_dir,
                    'timing_data': timing_data,
                }
    
    return results


def determine_config_from_timing(batch_timing_dir, hash_path, dataset):
    """Try to determine config name from timing data or metadata."""
    # Check if there's config info in the timing file
    for fname in ["simple_batch_timing.json", f"batch_timing_{dataset}.json"]:
        fpath = os.path.join(batch_timing_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r') as f:
                    data = json.load(f)
                if 'config' in data:
                    return data['config']
            except Exception:
                pass
    
    # Fallback: infer from parent directory
    parent = os.path.dirname(os.path.dirname(os.path.dirname(hash_path)))
    parent_name = os.path.basename(parent)
    
    if "reduced_all" in parent_name:
        return "reduced_graph"
    
    return "unknown"


def parse_tainted_batches(batch_timing_dir, dataset):
    """
    Parse tainted batch timing data from batch_timing directory.
    
    Files:
    - tainted_batches_{dataset}.json: Training tainted batches (all epochs)
    - inference_tainted_batches_{dataset}_epoch{N}.json: Inference tainted batches per epoch
    
    Returns dict with:
    - training: {epoch: [list of batch timing dicts]}
    - inference: {epoch: [list of batch timing dicts]}
    """
    training_batches = defaultdict(list)
    inference_batches = defaultdict(list)
    
    # Parse training tainted batches
    training_file = os.path.join(batch_timing_dir, f"tainted_batches_{dataset}.json")
    if os.path.exists(training_file):
        with open(training_file, 'r') as f:
            data = json.load(f)
        
        for batch in data.get('batches', []):
            epoch = batch.get('epoch', 0)
            training_batches[epoch].append(batch)
    
    # Parse inference tainted batches for each epoch
    for filename in os.listdir(batch_timing_dir):
        if filename.startswith(f'inference_tainted_batches_{dataset}_epoch') and filename.endswith('.json'):
            match = re.search(r'epoch(\d+)', filename)
            if match:
                epoch = int(match.group(1))
                filepath = os.path.join(batch_timing_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                for batch in data.get('batches', []):
                    inference_batches[epoch].append(batch)
    
    return {
        'training': dict(training_batches),
        'inference': dict(inference_batches),
    }


def parse_all_inference_batches(batch_timing_dir, dataset, epoch):
    """
    Parse ALL inference batch timing data (not just tainted) from inference_batch_timing files.
    
    Files:
    - inference_batch_timing_{dataset}_epoch{N}.json: All inference batches for epoch N
    
    Returns dict with:
    - batches: [list of all batch timing dicts]
    - summary: summary stats from the file
    """
    inference_file = os.path.join(batch_timing_dir, f"inference_batch_timing_{dataset}_epoch{epoch}.json")
    
    if not os.path.exists(inference_file):
        return None
    
    with open(inference_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: 'results' (old) and 'detailed_results' (new/reduced)
    batches = data.get('results', data.get('detailed_results', []))
    
    return {
        'batches': batches,
        'summary': data.get('summary', {}),
        'config': data.get('config', {}),
    }


def parse_all_training_batches(batch_timing_dir, dataset):
    """
    Parse ALL training batch timing data (not just tainted) from batch_timing files.
    
    Files:
    - batch_timing_{dataset}.json: All training batches across all epochs
    
    Returns dict with:
    - batches: [list of all batch timing dicts]
    - summary: summary stats from the file
    """
    training_file = os.path.join(batch_timing_dir, f"batch_timing_{dataset}.json")
    
    if not os.path.exists(training_file):
        return None
    
    with open(training_file, 'r') as f:
        data = json.load(f)
    
    # Handle both formats: 'results' (old) and 'detailed_results' (new/reduced)
    batches = data.get('results', data.get('detailed_results', []))
    
    return {
        'batches': batches,
        'summary': data.get('summary', {}),
        'config': data.get('config', {}),
    }


def get_base_inference_time_for_tainted_batches(base_batch_timing_dir, kde_batch_timing_dir, dataset, base_epoch, kde_epoch):
    """
    Get average inference time from base config for the same batch IDs that are tainted in KDE config.
    
    This allows fair comparison: we compare the time base took for the same batches that KDE
    identified as tainted (containing KDE-eligible edges).
    """
    # Parse KDE tainted batches to get their batch IDs
    kde_tainted_file = os.path.join(kde_batch_timing_dir, f"inference_tainted_batches_{dataset}_epoch{kde_epoch}.json")
    if not os.path.exists(kde_tainted_file):
        return None
    
    with open(kde_tainted_file, 'r') as f:
        kde_data = json.load(f)
    
    # Note: tainted batches file uses 'batch_number', not 'batch_id'
    tainted_batch_ids = set(batch.get('batch_number') for batch in kde_data.get('batches', []))
    
    if not tainted_batch_ids:
        return None
    
    # Parse base inference batches
    base_inference_file = os.path.join(base_batch_timing_dir, f"inference_batch_timing_{dataset}_epoch{base_epoch}.json")
    if not os.path.exists(base_inference_file):
        return None
    
    with open(base_inference_file, 'r') as f:
        base_data = json.load(f)
    
    # Filter to only the batches that correspond to tainted batch IDs
    # Note: inference_batch_timing file uses 'batch_id'
    matching_batches = [
        batch for batch in base_data.get('results', [])
        if batch.get('batch_id') in tainted_batch_ids
    ]
    
    if not matching_batches:
        return None
    
    # Compute stats for these matching batches
    return compute_all_inference_stats(matching_batches)


def compute_all_inference_stats(batch_list):
    """
    Compute statistics for ALL inference batches (not just tainted).
    
    Returns dict with:
    - count: number of batches
    - total_time_ms: sum of total_time_ms
    - avg_time_ms: average total_time_ms per batch
    - avg_forward_ms: average forward pass time
    - total_edges: total edges across all batches
    - avg_edges_per_batch: average edges per batch
    """
    if not batch_list:
        return None
    
    count = len(batch_list)
    
    # Handle both formats: 'total_time_ms' (KDE) and 'forward_time_ms + backward_time_ms' (reduced)
    total_time_ms = 0
    for b in batch_list:
        if 'total_time_ms' in b and b['total_time_ms']:
            total_time_ms += b['total_time_ms']
        else:
            # For reduced graphs, total = forward + backward
            fwd = b.get('forward_time_ms', 0) or 0
            bwd = b.get('backward_time_ms', 0) or 0
            total_time_ms += fwd + bwd
    
    forward_times = [b.get('forward_time_ms', 0) or 0 for b in batch_list]
    
    # Handle both 'total_edges' and 'num_edges' field names
    total_edges = sum(b.get('total_edges', b.get('num_edges', 0)) or 0 for b in batch_list)
    kde_eligible = sum(b.get('kde_eligible_edges', 0) or 0 for b in batch_list)
    
    stats = {
        'count': count,
        'total_time_ms': total_time_ms,
        'avg_time_ms': total_time_ms / count if count > 0 else 0,
        'avg_forward_ms': statistics.mean(forward_times) if forward_times else 0,
        'total_edges': total_edges,
        'avg_edges_per_batch': total_edges / count if count > 0 else 0,
        'kde_eligible_edges': kde_eligible,
    }
    
    return stats


def compute_batch_stats(batch_list):
    """
    Compute statistics for a list of batch timing dicts.
    
    Returns dict with:
    - count: number of batches
    - total_time_ms: sum of total_time_ms
    - avg_time_ms: average total_time_ms per batch
    - avg_forward_ms: average forward pass time
    - avg_backward_ms: average backward pass time (training only)
    - total_edges: total edges across all batches
    - avg_edges_per_batch: average edges per batch
    - kde_eligible_edges: total KDE-eligible edges
    """
    if not batch_list:
        return None
    
    count = len(batch_list)
    
    # Handle both formats: 'total_time_ms' (KDE) and 'forward_time_ms + backward_time_ms' (reduced)
    total_time_ms = 0
    for b in batch_list:
        if 'total_time_ms' in b and b['total_time_ms']:
            total_time_ms += b['total_time_ms']
        else:
            # For reduced graphs, total = forward + backward
            fwd = b.get('forward_time_ms', 0) or 0
            bwd = b.get('backward_time_ms', 0) or 0
            total_time_ms += fwd + bwd
    
    forward_times = [b.get('forward_time_ms', 0) or 0 for b in batch_list]
    backward_times = [b.get('backward_time_ms') for b in batch_list if b.get('backward_time_ms') is not None]
    
    # Handle both 'total_edges' and 'num_edges' field names
    total_edges = sum(b.get('total_edges', b.get('num_edges', 0)) or 0 for b in batch_list)
    kde_eligible = sum(b.get('kde_eligible_edges', 0) or 0 for b in batch_list)
    
    stats = {
        'count': count,
        'total_time_ms': total_time_ms,
        'avg_time_ms': total_time_ms / count if count > 0 else 0,
        'avg_forward_ms': statistics.mean(forward_times) if forward_times else 0,
        'total_edges': total_edges,
        'avg_edges_per_batch': total_edges / count if count > 0 else 0,
        'kde_eligible_edges': kde_eligible,
    }
    
    if backward_times:
        stats['avg_backward_ms'] = statistics.mean(backward_times)
    
    return stats


def main():
    output_dir = "/scratch/asawan15/PIDSMaker/gpu_timing_info"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GPU TIMING ANALYSIS - Tainted Batches")
    print("=" * 80)
    
    # Collect timing data for all configs
    kde_timing = {}  # KDE-enhanced configs (artifacts_reduced)
    kde_diff_timing = {}  # KDE diff configs (artifacts_reduced_diff)
    base_timing = {}  # Base configs (may not have batch_timing data)
    
    # Collect ALL inference batch timing data (not just tainted)
    kde_diff_all_inference = {}  # All inference batches for KDE diff
    base_all_inference = {}  # All inference batches for base
    
    # Collect ALL training batch timing data (not just tainted)
    kde_diff_all_training = {}  # All training batches for KDE diff
    base_all_training = {}  # All training batches for base
    
    # ========================================================================
    # Skip KDE-enhanced runs (artifacts_reduced) - focus on kde_diff vs normal
    # ========================================================================
    print("\n--- Skipping KDE-enhanced runs (artifacts_reduced) - focusing on kde_diff vs normal ---")
    
    # ========================================================================
    # Parse KDE diff runs (artifacts_reduced_diff)
    # ========================================================================
    print("\n--- Parsing KDE diff runs (artifacts_reduced_diff) ---")
    
    kde_diff_base_dir = os.path.join(BASE_PATH, "artifacts_reduced_diff", "evaluation", "evaluation")
    
    for (hash_val, dataset), info in KDE_DIFF_CONFIGS.items():
        config = info['config']
        epoch = info['epoch']
        
        batch_timing_dir = os.path.join(kde_diff_base_dir, hash_val, dataset, "batch_timing")
        
        if not os.path.exists(batch_timing_dir):
            print(f"  [MISSING] {config}/{dataset}: {batch_timing_dir}")
            continue
        
        timing = parse_tainted_batches(batch_timing_dir, dataset)
        
        # Compute stats for training (all epochs combined)
        all_training_batches = []
        for batches in timing['training'].values():
            all_training_batches.extend(batches)
        
        # Compute stats for inference (specific epoch)
        inference_batches = timing['inference'].get(epoch, [])
        
        # Parse ALL inference batches for this epoch (not just tainted)
        all_inference_data = parse_all_inference_batches(batch_timing_dir, dataset, epoch)
        
        # Parse ALL training batches (not just tainted)
        all_training_data = parse_all_training_batches(batch_timing_dir, dataset)
        
        key = f"{config}_{dataset}"
        kde_diff_timing[key] = {
            'config': config,
            'dataset': dataset,
            'epoch': epoch,
            'hash': hash_val,
            'training_all_epochs': compute_batch_stats(all_training_batches),
            'training_by_epoch': {e: compute_batch_stats(b) for e, b in timing['training'].items()},
            'inference_epoch': compute_batch_stats(inference_batches),
        }
        
        if all_inference_data:
            kde_diff_all_inference[key] = {
                'config': config,
                'dataset': dataset,
                'epoch': epoch,
                'hash': hash_val,
                'stats': compute_all_inference_stats(all_inference_data['batches']),
                'summary': all_inference_data['summary'],
            }
        
        if all_training_data:
            kde_diff_all_training[key] = {
                'config': config,
                'dataset': dataset,
                'hash': hash_val,
                'stats': compute_all_inference_stats(all_training_data['batches']),
                'summary': all_training_data['summary'],
            }
        
        train_count = len(all_training_batches)
        inf_count = len(inference_batches)
        all_inf_count = len(all_inference_data['batches']) if all_inference_data else 0
        all_train_count = len(all_training_data['batches']) if all_training_data else 0
        print(f"  [OK] {config}/{dataset}: {train_count} tainted training batches, {all_train_count} total training batches, {inf_count} tainted inference batches, {all_inf_count} total inference batches (epoch {epoch})")
    
    # ========================================================================
    # Parse Base runs (artifacts) - if they have batch_timing data
    # ========================================================================
    print("\n--- Parsing Base runs (artifacts) ---")
    
    base_base_dir = os.path.join(BASE_PATH, "artifacts", "evaluation", "evaluation")
    base_timing = {}
    base_all_inference = {}
    base_inference_for_kde_tainted = {}  # Base inference times for KDE tainted batches
    
    for (hash_val, dataset), info in BASE_CONFIGS.items():
        config = info['config']
        epoch = info['epoch']
        
        batch_timing_dir = os.path.join(base_base_dir, hash_val, dataset, "batch_timing")
        
        if not os.path.exists(batch_timing_dir):
            print(f"  [MISSING] {config}/{dataset}: batch_timing not available")
            continue
        
        timing = parse_tainted_batches(batch_timing_dir, dataset)
        
        # Compute stats
        all_training_batches = []
        for batches in timing['training'].values():
            all_training_batches.extend(batches)
        
        inference_batches = timing['inference'].get(epoch, [])
        
        # Parse ALL inference batches for this epoch (not just tainted)
        all_inference_data = parse_all_inference_batches(batch_timing_dir, dataset, epoch)
        
        # Parse ALL training batches (not just tainted)
        all_training_data = parse_all_training_batches(batch_timing_dir, dataset)
        
        key = f"{config}_{dataset}"
        base_timing[key] = {
            'config': config,
            'dataset': dataset,
            'epoch': epoch,
            'hash': hash_val,
            'training_all_epochs': compute_batch_stats(all_training_batches),
            'training_by_epoch': {e: compute_batch_stats(b) for e, b in timing['training'].items()},
            'inference_epoch': compute_batch_stats(inference_batches),
        }
        
        if all_inference_data:
            base_all_inference[key] = {
                'config': config,
                'dataset': dataset,
                'epoch': epoch,
                'hash': hash_val,
                'stats': compute_all_inference_stats(all_inference_data['batches']),
                'summary': all_inference_data['summary'],
            }
        
        if all_training_data:
            base_all_training[key] = {
                'config': config,
                'dataset': dataset,
                'hash': hash_val,
                'stats': compute_all_inference_stats(all_training_data['batches']),
                'summary': all_training_data['summary'],
            }
        
        # Get base inference times for KDE tainted batches
        # Find corresponding KDE diff config for this base config
        kde_diff_config_map = {
            'orthrus_non_snooped_edge_ts': 'orthrus_kde_diff',
            'kairos': 'kairos_kde_diff'
        }
        if config in kde_diff_config_map:
            kde_config = kde_diff_config_map[config]
            # Find KDE batch timing dir for this dataset
            kde_hash = None
            kde_epoch = None
            for (h, d), info in KDE_DIFF_CONFIGS.items():
                if info['config'] == kde_config and d == dataset:
                    kde_hash = h
                    kde_epoch = info['epoch']
                    break
            
            if kde_hash and kde_epoch:
                kde_batch_timing_dir = os.path.join(kde_diff_base_dir, kde_hash, dataset, "batch_timing")
                base_inference_for_tainted = get_base_inference_time_for_tainted_batches(
                    batch_timing_dir, kde_batch_timing_dir, dataset, epoch, kde_epoch
                )
                if base_inference_for_tainted:
                    key = f"{config}_{dataset}"
                    base_inference_for_kde_tainted[key] = {
                        'config': config,
                        'dataset': dataset,
                        'epoch': epoch,
                        'kde_config': kde_config,
                        'kde_epoch': kde_epoch,
                        'stats': base_inference_for_tainted,
                    }
        
        train_count = len(all_training_batches)
        inf_count = len(inference_batches)
        all_inf_count = len(all_inference_data['batches']) if all_inference_data else 0
        all_train_count = len(all_training_data['batches']) if all_training_data else 0
        print(f"  [OK] {config}/{dataset}: {train_count} tainted training batches, {all_train_count} total training batches, {inf_count} tainted inference batches, {all_inf_count} total inference batches (epoch {epoch})")
    
    # ========================================================================
    # Parse Reduced Graph runs (artifacts_reduced_all) - Simple timing only
    # ========================================================================
    print("\n--- Parsing Reduced Graph runs (artifacts_reduced_all) ---")
    
    reduced_timing = {}  # Simple timing for reduced graph configs
    reduced_all_training = {}  # ALL training batches for reduced configs
    reduced_all_inference = {}  # ALL inference batches for reduced configs
    reduced_artifacts_dir = os.path.join(BASE_PATH, "artifacts_reduced_all")
    
    # Auto-discover reduced graph runs
    #discovered_reduced = discover_reduced_graph_runs(reduced_artifacts_dir)
    
    for (hash_val, dataset), info in REDUCED_CONFIGS.items():
        config = info['config']
        epoch = info['epoch']
        
        batch_timing_dir = os.path.join(reduced_artifacts_dir, "evaluation", "evaluation", hash_val, dataset, "batch_timing")
        
        if not os.path.exists(batch_timing_dir):
            print(f"  [MISSING] {config}/{dataset}: {batch_timing_dir}")
            continue
        
        timing = parse_tainted_batches(batch_timing_dir, dataset)
        
        # Compute stats for training (all epochs combined)
        all_training_batches = []
        for batches in timing['training'].values():
            all_training_batches.extend(batches)
        
        # Compute stats for inference (specific epoch)
        inference_batches = timing['inference'].get(epoch, [])
        
        # Parse ALL inference batches for this epoch (not just tainted)
        all_inference_data = parse_all_inference_batches(batch_timing_dir, dataset, epoch)
        
        # Parse ALL training batches (not just tainted)
        all_training_data = parse_all_training_batches(batch_timing_dir, dataset)
        
        key = f"{config}_{dataset}"
        
        # For reduced graphs, use all training data (not tainted) since they don't use KDE
        training_stats = None
        if all_training_data and all_training_data['batches']:
            training_stats = compute_all_inference_stats(all_training_data['batches'])
        elif all_training_batches:
            training_stats = compute_batch_stats(all_training_batches)
        
        reduced_timing[key] = {
            'config': config,
            'dataset': dataset,
            'epoch': epoch,
            'hash': hash_val,
            'training_all_epochs': training_stats,
            'training_by_epoch': {e: compute_batch_stats(b) for e, b in timing['training'].items()},
            'inference_epoch': compute_batch_stats(inference_batches),
        }
        
        if all_inference_data:
            reduced_all_inference[key] = {
                'config': config,
                'dataset': dataset,
                'epoch': epoch,
                'hash': hash_val,
                'stats': compute_all_inference_stats(all_inference_data['batches']),
                'summary': all_inference_data['summary'],
            }
        
        if all_training_data:
            reduced_all_training[key] = {
                'config': config,
                'dataset': dataset,
                'hash': hash_val,
                'stats': compute_all_inference_stats(all_training_data['batches']),
                'summary': all_training_data['summary'],
            }
        
        train_count = len(all_training_batches)
        inf_count = len(inference_batches)
        all_inf_count = len(all_inference_data['batches']) if all_inference_data else 0
        all_train_count = len(all_training_data['batches']) if all_training_data else 0
        print(f"  [OK] {config}/{dataset}: {train_count} tainted training batches, {all_train_count} total training batches, {inf_count} tainted inference batches, {all_inf_count} total inference batches (epoch {epoch})")
   
    
    # ========================================================================
    # Generate Report
    # ========================================================================
    print("\n" + "=" * 80)
    print("TAINTED BATCH TIMING RESULTS")
    print("=" * 80)
    
    report_lines = []
    report_lines.append("GPU Timing Analysis - Tainted Batches")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Note: 'Tainted batches' are batches containing KDE-eligible edges")
    report_lines.append("      (edges with timestamps that could be processed using KDE).")
    report_lines.append("")
    
    # Summary table
    report_lines.append("=" * 100)
    report_lines.append("SUMMARY: Average Time per Tainted Batch")
    report_lines.append("=" * 100)
    report_lines.append("")
    report_lines.append(f"{'Config':<30} {'Dataset':<15} {'Phase':<12} {'Batches':<10} {'Avg ms/batch':<15} {'Total Time (s)':<15}")
    report_lines.append("-" * 100)
    
    for key in sorted(kde_timing.keys()):
        data = kde_timing[key]
        config = data['config']
        dataset = data['dataset']
        
        # Training
        if data['training_all_epochs']:
            stats = data['training_all_epochs']
            report_lines.append(
                f"{config:<30} {dataset:<15} {'Training':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
            )
        
        # Inference
        if data['inference_epoch']:
            stats = data['inference_epoch']
            report_lines.append(
                f"{config:<30} {dataset:<15} {'Inference':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
            )
    
    # KDE diff configs summary (orthrus_kde_diff, kairos_kde_diff - use timestamp DIFFERENCES)
    if kde_diff_timing:
        report_lines.append("")
        report_lines.append("--- KDE Diff Configs (artifacts_reduced_diff) ---")
        report_lines.append("    Note: orthrus_kde_diff / kairos_kde_diff use timestamp DIFFERENCES as KDE features")
        report_lines.append("")
        
        for key in sorted(kde_diff_timing.keys()):
            data = kde_diff_timing[key]
            config = data['config']
            dataset = data['dataset']
            
            # Training
            if data['training_all_epochs']:
                stats = data['training_all_epochs']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Training':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
            
            # Inference
            if data['inference_epoch']:
                stats = data['inference_epoch']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Inference':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
    
    # Reduced graph configs summary (orthrus_red, kairos_red - collapsed edges)
    if reduced_timing:
        report_lines.append("")
        report_lines.append("--- Reduced Graph Configs (artifacts_reduced_all) ---")
        report_lines.append("    Note: orthrus_red / kairos_red use COLLAPSED EDGES with temporal summary features")
        report_lines.append("          (first_ts_norm, last_ts_norm, count_norm) instead of KDE vectors")
        report_lines.append("")
        
        for key in sorted(reduced_timing.keys()):
            data = reduced_timing[key]
            config = data['config']
            dataset = data['dataset']
            
            # Training
            if data['training_all_epochs']:
                stats = data['training_all_epochs']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Training':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
            
            # Inference
            if data['inference_epoch']:
                stats = data['inference_epoch']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Inference':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
    
    report_lines.append("")
    
    # If we have base timing data, add comparison
    if base_timing:
        report_lines.append("")
        report_lines.append("=" * 100)
        report_lines.append("BASE CONFIGS (Full Graph, No KDE)")
        report_lines.append("=" * 100)
        report_lines.append("    Note: orthrus_non_snooped_edge_ts / kairos use the FULL provenance graph")
        report_lines.append("          with original edge timestamps (no KDE, no edge collapsing)")
        report_lines.append("")
        report_lines.append(f"{'Config':<30} {'Dataset':<15} {'Phase':<12} {'Batches':<10} {'Avg ms/batch':<15} {'Total Time (s)':<15}")
        report_lines.append("-" * 100)
        
        for key in sorted(base_timing.keys()):
            data = base_timing[key]
            config = data['config']
            dataset = data['dataset']
            
            if data['training_all_epochs']:
                stats = data['training_all_epochs']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Training':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
            
            if data['inference_epoch']:
                stats = data['inference_epoch']
                report_lines.append(
                    f"{config:<30} {dataset:<15} {'Inference':<12} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f}"
                )
    
    # Detailed per-config breakdown
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("DETAILED BREAKDOWN BY CONFIG")
    report_lines.append("=" * 80)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        report_lines.append("")
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"KDE Config: {kde_config}")
        report_lines.append(f"Base Config: {base_config}")
        report_lines.append("=" * 80)
        
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            report_lines.append(f"\n--- {dataset} ---")
            
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            if kde_key in kde_timing:
                data = kde_timing[kde_key]
                report_lines.append(f"\n  KDE ({kde_config}), Epoch {data['epoch']}:")
                
                if data['training_all_epochs']:
                    stats = data['training_all_epochs']
                    report_lines.append(f"    Training: {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                    report_lines.append(f"              {stats['total_edges']:,} total edges, {stats['kde_eligible_edges']:,} KDE-eligible")
                    report_lines.append(f"              Forward: {stats['avg_forward_ms']:.2f} ms avg")
                    if 'avg_backward_ms' in stats:
                        report_lines.append(f"              Backward: {stats['avg_backward_ms']:.2f} ms avg")
                
                if data['inference_epoch']:
                    stats = data['inference_epoch']
                    report_lines.append(f"    Inference (epoch {data['epoch']}): {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                    report_lines.append(f"              {stats['total_edges']:,} total edges")
                else:
                    report_lines.append(f"    Inference (epoch {data['epoch']}): No tainted batches")
            
            # Also check kde_diff_timing
            if kde_key in kde_diff_timing:
                data = kde_diff_timing[kde_key]
                report_lines.append(f"\n  KDE Diff ({kde_config}), Epoch {data['epoch']}:")
                
                if data['training_all_epochs']:
                    stats = data['training_all_epochs']
                    report_lines.append(f"    Training: {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                    report_lines.append(f"              {stats['total_edges']:,} total edges, {stats['kde_eligible_edges']:,} KDE-eligible")
                    report_lines.append(f"              Forward: {stats['avg_forward_ms']:.2f} ms avg")
                    if 'avg_backward_ms' in stats:
                        report_lines.append(f"              Backward: {stats['avg_backward_ms']:.2f} ms avg")
                
                if data['inference_epoch']:
                    stats = data['inference_epoch']
                    report_lines.append(f"    Inference (epoch {data['epoch']}): {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                    report_lines.append(f"              {stats['total_edges']:,} total edges")
                else:
                    report_lines.append(f"    Inference (epoch {data['epoch']}): No tainted batches")
            
            if base_key in base_timing:
                data = base_timing[base_key]
                report_lines.append(f"\n  Base ({base_config}), Epoch {data['epoch']}:")
                
                if data['training_all_epochs']:
                    stats = data['training_all_epochs']
                    report_lines.append(f"    Training: {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
                
                if data['inference_epoch']:
                    stats = data['inference_epoch']
                    report_lines.append(f"    Inference (epoch {data['epoch']}): {stats['count']:,} batches, {stats['avg_time_ms']:.2f} ms/batch avg")
    
    # Speedup comparison table - KDE Diff vs Base (only kde_diff configs)
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 140)
    report_lines.append("SPEEDUP COMPARISON: KDE Diff vs Base Config - TRAINING")
    report_lines.append("=" * 140)
    report_lines.append("")
    report_lines.append("Note: KDE Diff configs (orthrus_kde_diff, kairos_kde_diff) use timestamp differences as KDE features")
    report_lines.append("      Speedup > 1.0 means KDE Diff is faster; < 1.0 means Base is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Config':<30} {'KDE Diff Config':<25} {'Base ms/batch':<15} {'KDE Diff ms/batch':<17} {'Speedup':<10}")
    report_lines.append("-" * 130)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        # Only process kde_diff configs here
        if 'kde_diff' not in kde_config:
            continue
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            if kde_key in kde_diff_timing and base_key in base_timing:
                kde_data = kde_diff_timing[kde_key]
                base_data = base_timing[base_key]
                
                if kde_data['training_all_epochs'] and base_data['training_all_epochs']:
                    kde_avg = kde_data['training_all_epochs']['avg_time_ms']
                    base_avg = base_data['training_all_epochs']['avg_time_ms']
                    speedup = base_avg / kde_avg if kde_avg > 0 else 0
                    
                    model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_config:<30} {kde_config:<25} {base_avg:<15.2f} {kde_avg:<17.2f} {speedup:<10.2f}x"
                    )
    
    # ========================================================================
    # REDUCED GRAPH SPEEDUP COMPARISON
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 140)
    report_lines.append("SPEEDUP COMPARISON: Reduced Graph vs Base Config - TRAINING")
    report_lines.append("=" * 140)
    report_lines.append("")
    report_lines.append("Note: Reduced Graph configs (orthrus_red, kairos_red) use COLLAPSED EDGES with")
    report_lines.append("      temporal summary features (first_ts, last_ts, count) instead of KDE vectors")
    report_lines.append("      Speedup > 1.0 means Reduced is FASTER; < 1.0 means Base is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Config':<30} {'Reduced Config':<20} {'Base ms/batch':<15} {'Reduced ms/batch':<17} {'Speedup':<10}")
    report_lines.append("-" * 130)
    
    for red_config, base_config in [('orthrus_red', 'orthrus_non_snooped_edge_ts'), 
                                     ('kairos_red', 'kairos')]:
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            red_key = f"{red_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            if red_key in reduced_timing and base_key in base_timing:
                red_data = reduced_timing[red_key]
                base_data = base_timing[base_key]
                
                if red_data['training_all_epochs'] and base_data['training_all_epochs']:
                    red_avg = red_data['training_all_epochs']['avg_time_ms']
                    base_avg = base_data['training_all_epochs']['avg_time_ms']
                    speedup = base_avg / red_avg if red_avg > 0 else 0
                    
                    model = 'orthrus' if 'orthrus' in red_config else 'kairos'
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_config:<30} {red_config:<20} {base_avg:<15.2f} {red_avg:<17.2f} {speedup:<10.2f}x"
                    )

    # ========================================================================
    # REDUCED GRAPH INFERENCE SPEEDUP COMPARISON
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 140)
    report_lines.append("SPEEDUP COMPARISON: Reduced Graph vs Base Config - INFERENCE")
    report_lines.append("=" * 140)
    report_lines.append("")
    report_lines.append("Note: Reduced Graph configs (orthrus_red, kairos_red) use COLLAPSED EDGES with")
    report_lines.append("      temporal summary features (first_ts, last_ts, count) instead of KDE vectors")
    report_lines.append("      Speedup > 1.0 means Reduced is FASTER; < 1.0 means Base is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Config':<30} {'Reduced Config':<20} {'Base Epoch':<12} {'Red Epoch':<12} {'Base ms/batch':<15} {'Reduced ms/batch':<17} {'Speedup':<10}")
    report_lines.append("-" * 160)
    
    for red_config, base_config in [('orthrus_red', 'orthrus_non_snooped_edge_ts'), 
                                     ('kairos_red', 'kairos')]:
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            red_key = f"{red_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            if red_key in reduced_all_inference and base_key in base_all_inference:
                red_data = reduced_all_inference[red_key]
                base_data = base_all_inference[base_key]
                
                if red_data['stats'] and base_data['stats']:
                    red_avg = red_data['stats']['avg_time_ms']
                    base_avg = base_data['stats']['avg_time_ms']
                    speedup = base_avg / red_avg if red_avg > 0 else 0
                    
                    model = 'orthrus' if 'orthrus' in red_config else 'kairos'
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_config:<30} {red_config:<20} {base_data['epoch']:<12} {red_data['epoch']:<12} {base_avg:<15.2f} {red_avg:<17.2f} {speedup:<10.2f}x"
                    )

    # ========================================================================
    # TAINTED BATCH COMPARISON: TRAINING TIMES
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 180)
    report_lines.append("TAINTED BATCH COMPARISON - TRAINING: KDE Diff (Epoch 0) vs Base")
    report_lines.append("=" * 180)
    report_lines.append("")
    report_lines.append("Note: Tainted batches = batches containing KDE-eligible edges")
    report_lines.append("      Speedup > 1.0 means Base is SLOWER (KDE is faster)")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Epoch':<12} {'Base Batches':<13} {'Base ms/batch':<15} {'KDE Batches':<13} {'KDE ms/batch':<15} {'Speedup':<12}")
    report_lines.append("-" * 180)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        if 'kde_diff' not in kde_config:
            continue
            
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
            
            # Training comparison
            if kde_key in kde_diff_timing and base_key in base_timing:
                kde_data = kde_diff_timing[kde_key]
                base_data = base_timing[base_key]
                
                if kde_data['training_all_epochs'] and base_data['training_all_epochs']:
                    kde_train = kde_data['training_all_epochs']
                    base_train = base_data['training_all_epochs']
                    speedup = base_train['avg_time_ms'] / kde_train['avg_time_ms'] if kde_train['avg_time_ms'] > 0 else 0
                    
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_data['epoch']:<12} {base_train['count']:<13} {base_train['avg_time_ms']:<15.2f} {kde_train['count']:<13} {kde_train['avg_time_ms']:<15.2f} {speedup:<12.2f}x"
                    )
    
    # ========================================================================
    # TAINTED BATCH COMPARISON: INFERENCE TIMES
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 180)
    report_lines.append("TAINTED BATCH COMPARISON - INFERENCE: KDE Diff (Epoch 0) vs Base")
    report_lines.append("=" * 180)
    report_lines.append("")
    report_lines.append("Note: Tainted batches = batches containing KDE-eligible edges")
    report_lines.append("      Base batches shown are the same batch IDs that KDE identified as tainted")
    report_lines.append("      Speedup > 1.0 means Base is SLOWER (KDE is faster)")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Epoch':<12} {'Base Batches':<13} {'Base ms/batch':<15} {'KDE Batches':<13} {'KDE ms/batch':<15} {'Speedup':<12}")
    report_lines.append("-" * 180)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        if 'kde_diff' not in kde_config:
            continue
            
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
            
            # Inference comparison (tainted batches only)
            if kde_key in kde_diff_timing and base_key in base_inference_for_kde_tainted:
                kde_data = kde_diff_timing[kde_key]
                base_tainted_data = base_inference_for_kde_tainted[base_key]
                
                if kde_data['inference_epoch']:
                    kde_inf = kde_data['inference_epoch']
                    base_inf = base_tainted_data['stats']
                    
                    # Base inference for same batch IDs as KDE tainted
                    if base_inf and base_inf['count'] > 0:
                        speedup = base_inf['avg_forward_ms'] / kde_inf['avg_time_ms'] if kde_inf['avg_time_ms'] > 0 else 0
                        report_lines.append(
                            f"{model:<12} {dataset:<15} {base_tainted_data['epoch']:<12} {base_inf['count']:<13} {base_inf['avg_forward_ms']:<15.2f} {kde_inf['count']:<13} {kde_inf['avg_time_ms']:<15.2f} {speedup:<12.2f}x"
                        )
                    else:
                        # Could not find matching base inference data
                        report_lines.append(
                            f"{model:<12} {dataset:<15} {'N/A':<12} {'0':<13} {'N/A':<15} {kde_inf['count']:<13} {kde_inf['avg_time_ms']:<15.2f} {'N/A':<12}"
                        )
    
    # ========================================================================
    # ALL INFERENCE BATCHES COMPARISON (not just tainted)
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 160)
    report_lines.append("ALL INFERENCE BATCHES COMPARISON: KDE vs Base Config")
    report_lines.append("=" * 160)
    report_lines.append("")
    report_lines.append("Note: This compares ALL inference batches (not just tainted)")
    report_lines.append("      Speedup > 1.0 means Base is faster; < 1.0 means KDE is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Config Type':<20} {'Epoch':<8} {'Batches':<10} {'Avg ms/batch':<15} {'Total Time (s)':<15} {'Total Edges':<15}")
    report_lines.append("-" * 160)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
            
            # Only process kde_diff configs
            if 'kde_diff' not in kde_config:
                continue
            
            # KDE Diff data
            if kde_key in kde_diff_all_inference:
                kde_data = kde_diff_all_inference[kde_key]
                if kde_data['stats']:
                    stats = kde_data['stats']
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {'KDE Diff':<20} {kde_data['epoch']:<8} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f} {stats['total_edges']:<15}"
                    )
            
            # Base data
            if base_key in base_all_inference:
                base_data = base_all_inference[base_key]
                if base_data['stats']:
                    stats = base_data['stats']
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {'Base':<20} {base_data['epoch']:<8} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f} {stats['total_edges']:<15}"
                    )
    
    # Inference Speedup table (ALL batches)
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 160)
    report_lines.append("INFERENCE SPEEDUP: KDE Diff vs Base (ALL Batches)")
    report_lines.append("=" * 160)
    report_lines.append("")
    report_lines.append("Note: Speedup > 1.0 means Base is faster than KDE Diff")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Epoch':<12} {'KDE Epoch':<12} {'Base ms/batch':<15} {'KDE ms/batch':<15} {'Speedup':<12} {'Base Total(s)':<15} {'KDE Total(s)':<15}")
    report_lines.append("-" * 160)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        if 'kde_diff' not in kde_config:
            continue
            
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
            
            if kde_key in kde_diff_all_inference and base_key in base_all_inference:
                kde_data = kde_diff_all_inference[kde_key]
                base_data = base_all_inference[base_key]
                
                if kde_data['stats'] and base_data['stats']:
                    kde_stats = kde_data['stats']
                    base_stats = base_data['stats']
                    
                    kde_avg = kde_stats['avg_time_ms']
                    base_avg = base_stats['avg_time_ms']
                    speedup = kde_avg / base_avg if base_avg > 0 else 0
                    
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_data['epoch']:<12} {kde_data['epoch']:<12} {base_avg:<15.2f} {kde_avg:<15.2f} {speedup:<12.2f}x {base_stats['total_time_ms']/1000:<15.2f} {kde_stats['total_time_ms']/1000:<15.2f}"
                    )
    
    # ========================================================================
    # ALL TRAINING BATCHES COMPARISON (not just tainted)
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 160)
    report_lines.append("ALL TRAINING BATCHES COMPARISON: KDE vs Base Config")
    report_lines.append("=" * 160)
    report_lines.append("")
    report_lines.append("Note: This compares ALL training batches (not just tainted)")
    report_lines.append("      Speedup > 1.0 means Base is faster; < 1.0 means KDE is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Config Type':<20} {'Batches':<10} {'Avg ms/batch':<15} {'Total Time (s)':<15} {'Total Edges':<15}")
    report_lines.append("-" * 160)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
            
            # Only process kde_diff configs
            if 'kde_diff' not in kde_config:
                continue
            
            # KDE Diff data
            if kde_key in kde_diff_all_training:
                kde_data = kde_diff_all_training[kde_key]
                if kde_data['stats']:
                    stats = kde_data['stats']
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {'KDE Diff':<20} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f} {stats['total_edges']:<15}"
                    )
            
            # Base data
            if base_key in base_all_training:
                base_data = base_all_training[base_key]
                if base_data['stats']:
                    stats = base_data['stats']
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {'Base':<20} {stats['count']:<10} {stats['avg_time_ms']:<15.2f} {stats['total_time_ms']/1000:<15.2f} {stats['total_edges']:<15}"
                    )
    
    # Training Speedup table (ALL batches)
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 160)
    report_lines.append("TRAINING SPEEDUP: KDE Diff vs Base (ALL Batches)")
    report_lines.append("=" * 160)
    report_lines.append("")
    report_lines.append("Note: Speedup > 1.0 means Base is faster than KDE Diff")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base ms/batch':<15} {'KDE ms/batch':<15} {'Speedup':<12} {'Base Total(s)':<15} {'KDE Total(s)':<15}")
    report_lines.append("-" * 160)
    
    for kde_config, base_config in KDE_TO_BASE_CONFIG.items():
        if 'kde_diff' not in kde_config:
            continue
            
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            kde_key = f"{kde_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            
            model = 'orthrus' if 'orthrus' in kde_config else 'kairos'
            
            if kde_key in kde_diff_all_training and base_key in base_all_training:
                kde_data = kde_diff_all_training[kde_key]
                base_data = base_all_training[base_key]
                
                if kde_data['stats'] and base_data['stats']:
                    kde_stats = kde_data['stats']
                    base_stats = base_data['stats']
                    
                    kde_avg = kde_stats['avg_time_ms']
                    base_avg = base_stats['avg_time_ms']
                    speedup = kde_avg / base_avg if base_avg > 0 else 0
                    
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_avg:<15.2f} {kde_avg:<15.2f} {speedup:<12.2f}x {base_stats['total_time_ms']/1000:<15.2f} {kde_stats['total_time_ms']/1000:<15.2f}"
                    )
    
    # ========================================================================
    # INFERENCE SPEEDUP: Reduced Graph vs Base
    # ========================================================================
    report_lines.append("")
    report_lines.append("")
    report_lines.append("=" * 140)
    report_lines.append("INFERENCE SPEEDUP: Reduced Graph vs Base Config")
    report_lines.append("=" * 140)
    report_lines.append("")
    report_lines.append("Note: Reduced Graph configs (orthrus_red, kairos_red) use COLLAPSED EDGES with")
    report_lines.append("      temporal summary features (first_ts, last_ts, count) instead of KDE vectors")
    report_lines.append("      Speedup > 1.0 means Reduced is FASTER; < 1.0 means Base is faster")
    report_lines.append("")
    report_lines.append(f"{'Model':<12} {'Dataset':<15} {'Base Config':<30} {'Reduced Config':<20} {'Base ms/batch':<15} {'Reduced ms/batch':<17} {'Speedup':<10}")
    report_lines.append("-" * 130)
    for red_config, base_config in [('orthrus_red', 'orthrus_non_snooped_edge_ts'), 
                                     ('kairos_red', 'kairos')]:
        for dataset in ['CLEARSCOPE_E3', 'CADETS_E3', 'THEIA_E3']:
            red_key = f"{red_config}_{dataset}"
            base_key = f"{base_config}_{dataset}"
            model = 'orthrus' if 'orthrus' in red_config else 'kairos'
            # Use inference_epoch for both
            if red_key in reduced_timing and base_key in base_timing:
                red_data = reduced_timing[red_key]
                base_data = base_timing[base_key]
                if red_data['inference_epoch'] and base_data['inference_epoch']:
                    red_avg = red_data['inference_epoch']['avg_time_ms']
                    base_avg = base_data['inference_epoch']['avg_time_ms']
                    speedup = base_avg / red_avg if red_avg > 0 else 0
                    report_lines.append(
                        f"{model:<12} {dataset:<15} {base_config:<30} {red_config:<20} {base_avg:<15.2f} {red_avg:<17.2f} {speedup:<10.2f}x"
                    )
    
    # Print and save report
    report_text = '\n'.join(report_lines)
    print(report_text)
    
    with open(os.path.join(output_dir, "tainted_batch_timing_report.txt"), 'w') as f:
        f.write(report_text)
    
    # Save detailed JSON
    detailed_results = {
        'kde_timing': kde_timing,
        'kde_diff_timing': kde_diff_timing,
        'base_timing': base_timing,
        'kde_diff_all_inference': kde_diff_all_inference,
        'base_all_inference': base_all_inference,
        'kde_diff_all_training': kde_diff_all_training,
        'base_all_training': base_all_training,
        'reduced_timing': reduced_timing,  # Added reduced graph timing
    }
    
    with open(os.path.join(output_dir, "tainted_batch_timing_detailed.json"), 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n\nResults saved to {output_dir}/")
    print("  - tainted_batch_timing_report.txt")
    print("  - tainted_batch_timing_detailed.json")


if __name__ == "__main__":
    main()
